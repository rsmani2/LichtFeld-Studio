/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/python_console_panel.hpp"
#include "gui/editor/console_output.hpp"
#include "gui/editor/python_editor.hpp"
#include "gui/terminal/terminal_widget.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/utils/windows_utils.hpp"
#include "theme/theme.hpp"

#include <chrono>
#include <fstream>
#include <sstream>
#include <thread>
#include <imgui.h>

#ifdef LFS_BUILD_PYTHON_BINDINGS
#include <Python.h>
#include <filesystem>
#include <mutex>

#include "core/services.hpp"
#include "python/package_manager.hpp"
#include "python/runner.hpp"
#include "scene/scene_manager.hpp"

namespace {
    std::once_flag g_console_init_once;
    std::once_flag g_syspath_init_once;

    const char* getPythonExecutable() {
#ifdef LFS_PYTHON_EXECUTABLE
        return LFS_PYTHON_EXECUTABLE;
#else
        return "python3";
#endif
    }

    void setup_sys_path() {
        std::call_once(g_syspath_init_once, [] {
            const PyGILState_STATE gil = PyGILState_Ensure();

            const std::filesystem::path build_python_dir =
                std::filesystem::path(PROJECT_ROOT_PATH) / "build" / "src" / "python";

            PyObject* sys_path = PySys_GetObject("path");
            if (sys_path) {
                PyObject* py_path = PyUnicode_FromString(build_python_dir.string().c_str());
                if (py_path) {
                    PyList_Insert(sys_path, 0, py_path);
                    Py_DECREF(py_path);
                }
            }

            PyGILState_Release(gil);
        });
    }

    void setup_console_output_capture() {
        std::call_once(g_console_init_once, [] {
            lfs::python::set_output_callback([](const std::string& text, bool is_error) {
                auto& state = lfs::vis::gui::panels::PythonConsoleState::getInstance();
                auto* output = state.getOutputTerminal();
                if (output) {
                    // Use ANSI escape codes for error coloring
                    if (is_error) {
                        output->write("\033[31m"); // Red
                        output->write(text);
                        output->write("\033[0m"); // Reset
                    } else {
                        output->write(text);
                    }
                }
            });
        });
    }

    void execute_python_code(const std::string& code, lfs::vis::gui::panels::PythonConsoleState& state) {
        std::string cmd = code;

        // Trim trailing whitespace
        while (!cmd.empty() && (cmd.back() == '\n' || cmd.back() == '\r' || cmd.back() == ' ')) {
            cmd.pop_back();
        }

        if (cmd.empty()) {
            return;
        }

        state.addToHistory(cmd);

        auto* output = state.getOutputTerminal();
        if (!output)
            return;

        // Clear output on each new run
        output->clear();

        // Get scene for context injection
        lfs::vis::Scene* scene = nullptr;
        if (auto* sm = lfs::vis::services().sceneOrNull()) {
            scene = &sm->getScene();
        }

        // Execute in-process with GIL
        const PyGILState_STATE gil = PyGILState_Ensure();

        // Import lichtfeld and inject scene context
        if (scene) {
            PyObject* lf_module = PyImport_ImportModule("lichtfeld");
            if (lf_module) {
                PyObject* set_ctx = PyObject_GetAttrString(lf_module, "_set_scene_context");
                if (set_ctx && PyCallable_Check(set_ctx)) {
                    PyObject* capsule = PyCapsule_New(scene, nullptr, nullptr);
                    if (capsule) {
                        PyObject* args = PyTuple_Pack(1, capsule);
                        PyObject_Call(set_ctx, args, nullptr);
                        Py_DECREF(args);
                        Py_DECREF(capsule);
                    }
                }
                Py_XDECREF(set_ctx);
                Py_DECREF(lf_module);
            }
        }

        // Execute user code
        const int result = PyRun_SimpleString(cmd.c_str());
        if (result != 0) {
            PyErr_Print();
        }

        // Clear scene context
        if (scene) {
            PyObject* lf_module = PyImport_ImportModule("lichtfeld");
            if (lf_module) {
                PyObject* clear_ctx = PyObject_GetAttrString(lf_module, "_clear_scene_context");
                if (clear_ctx && PyCallable_Check(clear_ctx)) {
                    PyObject_CallNoArgs(clear_ctx);
                }
                Py_XDECREF(clear_ctx);
                Py_DECREF(lf_module);
            }
        }

        PyGILState_Release(gil);

        // Switch to Output tab
        state.setActiveTab(0);
    }

    void reset_python_state(lfs::vis::gui::panels::PythonConsoleState& state) {
        // Clear output terminal
        auto* output = state.getOutputTerminal();
        if (output) {
            output->clear();
        }
    }

    bool load_script(const std::filesystem::path& path, lfs::vis::gui::panels::PythonConsoleState& state) {
        std::ifstream file(path);
        if (!file.is_open()) {
            state.addError("Failed to open: " + path.string());
            return false;
        }

        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());

        if (auto* editor = state.getEditor()) {
            editor->setText(content);
        }

        state.setScriptPath(path);
        state.setModified(false);
        state.addInfo("Loaded: " + path.filename().string());
        return true;
    }

    bool save_script(const std::filesystem::path& path, lfs::vis::gui::panels::PythonConsoleState& state) {
        auto* editor = state.getEditor();
        if (!editor) {
            return false;
        }

        std::ofstream file(path);
        if (!file.is_open()) {
            state.addError("Failed to save: " + path.string());
            return false;
        }

        file << editor->getText();
        file.close();

        state.setScriptPath(path);
        state.setModified(false);
        state.addInfo("Saved: " + path.filename().string());
        return true;
    }

    void open_script_dialog(lfs::vis::gui::panels::PythonConsoleState& state) {
        const auto& current = state.getScriptPath();
        const auto start_dir = current.empty() ? std::filesystem::path{} : current.parent_path();
        const auto path = lfs::vis::gui::OpenPythonFileDialog(start_dir);
        if (!path.empty()) {
            load_script(path, state);
        }
    }

    void save_script_dialog(lfs::vis::gui::panels::PythonConsoleState& state) {
        const auto& current = state.getScriptPath();
        const std::string default_name = current.empty() ? "script" : current.stem().string();
        const auto path = lfs::vis::gui::SavePythonFileDialog(default_name);
        if (!path.empty()) {
            save_script(path, state);
        }
    }

    void save_current_script(lfs::vis::gui::panels::PythonConsoleState& state) {
        const auto& current = state.getScriptPath();
        if (current.empty()) {
            save_script_dialog(state);
        } else {
            save_script(current, state);
        }
    }

} // namespace
#endif

namespace lfs::vis::gui::panels {

    PythonConsoleState::PythonConsoleState()
        : terminal_(std::make_unique<terminal::TerminalWidget>(80, 24)),
          output_terminal_(std::make_unique<terminal::TerminalWidget>(80, 24)),
          editor_(std::make_unique<editor::PythonEditor>()),
          uv_console_(std::make_unique<editor::ConsoleOutput>()) {
    }

    PythonConsoleState::~PythonConsoleState() = default;

    PythonConsoleState& PythonConsoleState::getInstance() {
        static PythonConsoleState instance;
        return instance;
    }

    void PythonConsoleState::addOutput(const std::string& text, uint32_t /*color*/) {
        std::lock_guard lock(mutex_);
        if (output_terminal_) {
            output_terminal_->write(text);
            output_terminal_->write("\n");
        }
    }

    void PythonConsoleState::addError(const std::string& text) {
        std::lock_guard lock(mutex_);
        if (output_terminal_) {
            output_terminal_->write("\033[31m"); // Red
            output_terminal_->write(text);
            output_terminal_->write("\033[0m\n"); // Reset + newline
        }
    }

    void PythonConsoleState::addInput(const std::string& text) {
        std::lock_guard lock(mutex_);
        if (output_terminal_) {
            output_terminal_->write("\033[32m>>> "); // Green prompt
            output_terminal_->write(text);
            output_terminal_->write("\033[0m\n"); // Reset + newline
        }
    }

    void PythonConsoleState::addInfo(const std::string& text) {
        std::lock_guard lock(mutex_);
        if (output_terminal_) {
            output_terminal_->write("\033[36m"); // Cyan
            output_terminal_->write(text);
            output_terminal_->write("\033[0m\n"); // Reset + newline
        }
    }

    void PythonConsoleState::clear() {
        std::lock_guard lock(mutex_);
        if (output_terminal_) {
            output_terminal_->clear();
        }
    }

    void PythonConsoleState::addToHistory(const std::string& cmd) {
        std::lock_guard lock(mutex_);
        if (!cmd.empty() && (command_history_.empty() || command_history_.back() != cmd)) {
            command_history_.push_back(cmd);
        }
        history_index_ = -1;
        if (editor_) {
            editor_->addToHistory(cmd);
        }
    }

    void PythonConsoleState::historyUp() {
        std::lock_guard lock(mutex_);
        if (command_history_.empty())
            return;
        if (history_index_ < 0) {
            history_index_ = static_cast<int>(command_history_.size()) - 1;
        } else if (history_index_ > 0) {
            history_index_--;
        }
    }

    void PythonConsoleState::historyDown() {
        std::lock_guard lock(mutex_);
        if (history_index_ < 0)
            return;
        if (history_index_ < static_cast<int>(command_history_.size()) - 1) {
            history_index_++;
        } else {
            history_index_ = -1;
        }
    }

    terminal::TerminalWidget* PythonConsoleState::getTerminal() {
        return terminal_.get();
    }

    terminal::TerminalWidget* PythonConsoleState::getOutputTerminal() {
        return output_terminal_.get();
    }

    editor::PythonEditor* PythonConsoleState::getEditor() {
        return editor_.get();
    }

    editor::ConsoleOutput* PythonConsoleState::getUvConsole() {
        return uv_console_.get();
    }

    namespace {
        // Splitter state
        float g_splitter_ratio = 0.6f;
        constexpr float MIN_PANE_HEIGHT = 100.0f;
        constexpr float SPLITTER_THICKNESS = 6.0f;

        // Package install state
        char g_package_input[256] = "";

        void start_async_install(const std::string& package, editor::ConsoleOutput* console) {
            if (package.empty() || !console)
                return;

            auto& pm = python::PackageManager::instance();
            if (pm.has_running_operation()) {
                console->addError("Another operation is already running");
                return;
            }

            console->addInfo("Installing " + package + "...");

            pm.install_async(
                package,
                [console](const std::string& line, bool is_error, bool is_line_update) {
                    if (is_line_update) {
                        console->updateLastLine(line, is_error);
                    } else if (is_error) {
                        console->addError(line);
                    } else {
                        console->addOutput(line);
                    }
                },
                [console, package](bool success, int exit_code) {
                    if (success) {
                        console->addInfo("Successfully installed " + package);
                    } else {
                        console->addError("Failed to install " + package + " (exit code " +
                                          std::to_string(exit_code) + ")");
                    }
                });
        }

        void start_async_uninstall(const std::string& package, editor::ConsoleOutput* console) {
            if (package.empty() || !console)
                return;

            auto& pm = python::PackageManager::instance();
            if (pm.has_running_operation()) {
                console->addError("Another operation is already running");
                return;
            }

            console->addInfo("Uninstalling " + package + "...");

            pm.uninstall_async(
                package,
                [console](const std::string& line, bool is_error, bool is_line_update) {
                    if (is_line_update) {
                        console->updateLastLine(line, is_error);
                    } else if (is_error) {
                        console->addError(line);
                    } else {
                        console->addOutput(line);
                    }
                },
                [console, package](bool success, int exit_code) {
                    if (success) {
                        console->addInfo("Successfully uninstalled " + package);
                    } else {
                        console->addError("Failed to uninstall " + package + " (exit code " +
                                          std::to_string(exit_code) + ")");
                    }
                });
        }
    } // namespace

    void DrawPythonConsole(const UIContext& ctx, bool* open) {
        if (!open || !*open)
            return;

#ifndef LFS_BUILD_PYTHON_BINDINGS
        ImGui::SetNextWindowSize(ImVec2(500, 300), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Python Console", open)) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f),
                               "Python bindings not available. Rebuild with -DBUILD_PYTHON_BINDINGS=ON");
        }
        ImGui::End();
        return;
#else
        // Initialize Python and set up output capture
        lfs::python::ensure_initialized();
        lfs::python::install_output_redirect();
        setup_sys_path();
        setup_console_output_capture();

        auto& state = PythonConsoleState::getInstance();
        const auto& t = theme();

        // Build window title with script name and modified indicator
        std::string window_title = "Python Console";
        if (!state.getScriptPath().empty()) {
            window_title += " - " + state.getScriptPath().filename().string();
        }
        if (state.isModified()) {
            window_title += " *";
        }

        ImGui::SetNextWindowSize(ImVec2(700, 600), ImGuiCond_FirstUseEver);
        if (!ImGui::Begin(window_title.c_str(), open, ImGuiWindowFlags_MenuBar)) {
            ImGui::End();
            return;
        }

        // Menu bar
        if (ImGui::BeginMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("New Script", "Ctrl+N")) {
                    if (auto* editor = state.getEditor()) {
                        editor->clear();
                    }
                    state.setScriptPath({});
                    state.setModified(false);
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Open...", "Ctrl+O")) {
                    open_script_dialog(state);
                }
                if (ImGui::MenuItem("Save", "Ctrl+S")) {
                    save_current_script(state);
                }
                if (ImGui::MenuItem("Save As...")) {
                    save_script_dialog(state);
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Edit")) {
                if (ImGui::MenuItem("Clear Output", "Ctrl+L")) {
                    state.clear();
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Copy Selection")) {
                    if (auto* output = state.getOutputTerminal()) {
                        ImGui::SetClipboardText(output->getSelection().c_str());
                    }
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Run")) {
                if (ImGui::MenuItem("Run Script", "F5")) {
                    if (auto* editor = state.getEditor()) {
                        execute_python_code(editor->getText(), state);
                    }
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Reset Python State", "Ctrl+R")) {
                    reset_python_state(state);
                }
                ImGui::EndMenu();
            }
            if (ImGui::BeginMenu("Help")) {
                ImGui::MenuItem("Ctrl+Enter to execute", nullptr, false, false);
                ImGui::MenuItem("Ctrl+Space for autocomplete", nullptr, false, false);
                ImGui::MenuItem("F5 to run script", nullptr, false, false);
                ImGui::MenuItem("Ctrl+R to reset state", nullptr, false, false);
                ImGui::EndMenu();
            }
            ImGui::EndMenuBar();
        }

        // Toolbar
        {
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 4));
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 4));

            // Run button
            ImGui::PushStyleColor(ImGuiCol_Button, t.palette.success);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, lighten(t.palette.success, 0.1f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, darken(t.palette.success, 0.1f));
            if (ImGui::Button("Run") || ImGui::IsKeyPressed(ImGuiKey_F5, false)) {
                if (auto* editor = state.getEditor()) {
                    execute_python_code(editor->getText(), state);
                }
            }
            ImGui::PopStyleColor(3);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Run script (F5)");
            }

            ImGui::SameLine();

            // Stop button (for animations and running scripts)
            const bool has_animation = python::has_frame_callback();
            const bool has_running_script = state.getOutputTerminal() && state.getOutputTerminal()->is_running();
            const bool can_stop = has_animation || has_running_script;
            if (!can_stop) {
                ImGui::BeginDisabled();
            }
            ImGui::PushStyleColor(ImGuiCol_Button, t.palette.error);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, lighten(t.palette.error, 0.1f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, darken(t.palette.error, 0.1f));
            if (ImGui::Button("Stop")) {
                if (has_animation) {
                    python::clear_frame_callback();
                }
                if (auto* output = state.getOutputTerminal()) {
                    output->interrupt();
                }
            }
            ImGui::PopStyleColor(3);
            if (!can_stop) {
                ImGui::EndDisabled();
            }
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("Stop running script (Ctrl+C)");
            }

            ImGui::SameLine();

            // Reset button
            if (ImGui::Button("Reset")) {
                reset_python_state(state);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Reset Python state (Ctrl+R)");
            }

            ImGui::SameLine();

            // Clear button
            if (ImGui::Button("Clear")) {
                state.clear();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Clear console output (Ctrl+L)");
            }

            ImGui::SameLine();
            ImGui::Separator();
            ImGui::SameLine();

            // Status indicator
            ImGui::TextColored(t.palette.text_dim, "Python");

            ImGui::PopStyleVar(2);
        }

        ImGui::Spacing();
        ImGui::Separator();

        // Calculate pane sizes
        const ImVec2 content_avail = ImGui::GetContentRegionAvail();
        const float total_height = content_avail.y;

        float top_height = total_height * g_splitter_ratio - SPLITTER_THICKNESS / 2;
        float bottom_height = total_height * (1.0f - g_splitter_ratio) - SPLITTER_THICKNESS / 2;

        top_height = std::max(top_height, MIN_PANE_HEIGHT);
        bottom_height = std::max(bottom_height, MIN_PANE_HEIGHT);

        // Script Editor (top pane)
        ImGui::BeginChild("##script_editor_pane", ImVec2(content_avail.x, top_height), false);
        {
            ImGui::TextColored(t.palette.text_dim, "Script Editor");
            ImGui::Spacing();

            const ImVec2 editor_size(ImGui::GetContentRegionAvail().x,
                                     ImGui::GetContentRegionAvail().y);

            // Use monospace font for code editor
            if (ctx.fonts.monospace) {
                ImGui::PushFont(ctx.fonts.monospace);
            }

            // Check if terminal has focus - prevent editor from receiving input
            bool terminal_has_focus = false;
            if (auto* terminal = state.getTerminal()) {
                terminal_has_focus = terminal->isFocused();
            }

            if (auto* editor = state.getEditor()) {
                // When terminal has focus, make editor read-only to prevent input capture
                editor->setReadOnly(terminal_has_focus);

                if (editor->render(editor_size)) {
                    // Ctrl+Enter was pressed - execute
                    execute_python_code(editor->getText(), state);
                }
            }

            if (ctx.fonts.monospace) {
                ImGui::PopFont();
            }
        }
        ImGui::EndChild();

        // Horizontal splitter
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, t.palette.primary_dim);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, t.palette.primary);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0);

        ImGui::Button("##splitter", ImVec2(content_avail.x, SPLITTER_THICKNESS));

        if (ImGui::IsItemHovered()) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        }

        if (ImGui::IsItemActive()) {
            const float delta = ImGui::GetIO().MouseDelta.y;
            if (delta != 0.0f) {
                g_splitter_ratio += delta / total_height;
                g_splitter_ratio = std::clamp(g_splitter_ratio, 0.2f, 0.8f);
            }
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        }

        ImGui::PopStyleVar();
        ImGui::PopStyleColor(3);

        // Bottom pane with tabs
        ImGui::BeginChild("##bottom_pane", ImVec2(content_avail.x, bottom_height), false);
        {
            if (ImGui::BeginTabBar("##console_tabs")) {
                // Output tab (read-only terminal for script output)
                if (ImGui::BeginTabItem("Output")) {
                    state.setActiveTab(0);

                    if (auto* output = state.getOutputTerminal()) {
                        // Don't auto-spawn - spawned on demand when running code
                        output->setReadOnly(true);
                        output->render(ctx.fonts.monospace);
                    }

                    ImGui::EndTabItem();
                }

                // Terminal tab (interactive Python REPL)
                if (ImGui::BeginTabItem("Terminal")) {
                    state.setActiveTab(1);

                    if (auto* terminal = state.getTerminal()) {
                        // Auto-spawn Python on first use
                        if (!terminal->is_running()) {
                            terminal->spawn(getPythonExecutable());
                        }
                        terminal->render(ctx.fonts.monospace);
                    }

                    ImGui::EndTabItem();
                }

                // Packages tab (UV package manager output)
                if (ImGui::BeginTabItem("Packages")) {
                    state.setActiveTab(2);

                    if (auto* uv_console = state.getUvConsole()) {
                        ImVec2 avail = ImGui::GetContentRegionAvail();

                        auto& pm = python::PackageManager::instance();
                        const bool is_running = pm.has_running_operation();

                        // Package input row
                        ImGui::BeginDisabled(is_running);
                        ImGui::SetNextItemWidth(avail.x - 220);
                        bool enter_pressed = ImGui::InputTextWithHint(
                            "##pkg_input", "Package name (e.g., numpy)",
                            g_package_input, sizeof(g_package_input),
                            ImGuiInputTextFlags_EnterReturnsTrue);
                        ImGui::SameLine();
                        if (ImGui::Button("Install", ImVec2(70, 0)) || enter_pressed) {
                            start_async_install(g_package_input, uv_console);
                            g_package_input[0] = '\0';
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Uninstall", ImVec2(70, 0))) {
                            start_async_uninstall(g_package_input, uv_console);
                            g_package_input[0] = '\0';
                        }
                        ImGui::EndDisabled();

                        ImGui::SameLine();
                        if (ImGui::Button("Clear", ImVec2(50, 0))) {
                            uv_console->clear();
                        }

                        avail.y -= ImGui::GetTextLineHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y;

                        // Status line if running
                        if (is_running) {
                            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Running UV operation...");
                            ImGui::SameLine();
                            if (ImGui::SmallButton("Cancel")) {
                                pm.cancel_async();
                            }
                            avail.y -= ImGui::GetTextLineHeightWithSpacing();
                        }

                        uv_console->render(avail);
                    }

                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }
        }
        ImGui::EndChild();

        // Handle keyboard shortcuts
        if (ImGui::GetIO().KeyCtrl) {
            if (ImGui::IsKeyPressed(ImGuiKey_L, false)) {
                state.clear();
            }
            if (ImGui::IsKeyPressed(ImGuiKey_R, false)) {
                reset_python_state(state);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_N, false)) {
                if (auto* editor = state.getEditor()) {
                    editor->clear();
                }
                state.setScriptPath({});
                state.setModified(false);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_O, false)) {
                open_script_dialog(state);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_S, false)) {
                save_current_script(state);
            }
        }

        ImGui::End();
#endif // LFS_BUILD_PYTHON_BINDINGS
    }

    void DrawDockedPythonConsole(const UIContext& ctx, const ImVec2& pos, const ImVec2& size) {
#ifndef LFS_BUILD_PYTHON_BINDINGS
        ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(size, ImGuiCond_Always);
        if (ImGui::Begin("##DockedPythonConsole", nullptr,
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar)) {
            ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f),
                               "Python bindings not available");
        }
        ImGui::End();
        return;
#else
        lfs::python::ensure_initialized();
        lfs::python::install_output_redirect();
        setup_sys_path();
        setup_console_output_capture();

        auto& state = PythonConsoleState::getInstance();
        const auto& t = theme();

        ImGui::SetNextWindowPos(pos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(size, ImGuiCond_Always);

        ImGui::PushStyleColor(ImGuiCol_WindowBg, t.palette.background);

        constexpr ImGuiWindowFlags PANEL_FLAGS =
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDocking |
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoScrollWithMouse;

        if (!ImGui::Begin("##DockedPythonConsole", nullptr, PANEL_FLAGS)) {
            ImGui::End();
            ImGui::PopStyleColor();
            return;
        }

        // Toolbar
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8, 4));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 4));

        if (ImGui::Button("New")) {
            if (auto* editor = state.getEditor()) {
                editor->clear();
            }
            state.setScriptPath({});
            state.setModified(false);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Clear editor (Ctrl+N)");

        ImGui::SameLine();

        // Load button
        if (ImGui::Button("Load")) {
            open_script_dialog(state);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Load script (Ctrl+O)");

        ImGui::SameLine();

        // Save button
        if (ImGui::Button("Save")) {
            save_current_script(state);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Save script (Ctrl+S)");

        ImGui::SameLine();

        // Format button
        if (ImGui::Button("Format")) {
            if (auto* editor = state.getEditor()) {
                const std::string formatted = lfs::python::format_python_code(editor->getText());
                editor->setText(formatted);
                state.setModified(true);
            }
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Format code (Ctrl+Shift+F)");

        ImGui::SameLine();
        ImGui::TextColored(t.palette.text_dim, "|");
        ImGui::SameLine();

        // Run button
        ImGui::PushStyleColor(ImGuiCol_Button, t.palette.success);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, lighten(t.palette.success, 0.1f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, darken(t.palette.success, 0.1f));
        if (ImGui::Button("Run") || ImGui::IsKeyPressed(ImGuiKey_F5, false)) {
            if (auto* editor = state.getEditor()) {
                execute_python_code(editor->getText(), state);
            }
        }
        ImGui::PopStyleColor(3);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Run script (F5)");

        ImGui::SameLine();

        // Stop button
        {
            const bool has_animation = python::has_frame_callback();
            const bool has_running_script = state.getOutputTerminal() && state.getOutputTerminal()->is_running();
            const bool can_stop = has_animation || has_running_script;
            if (!can_stop) {
                ImGui::BeginDisabled();
            }
            ImGui::PushStyleColor(ImGuiCol_Button, t.palette.error);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, lighten(t.palette.error, 0.1f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, darken(t.palette.error, 0.1f));
            if (ImGui::Button("Stop")) {
                if (has_animation) {
                    python::clear_frame_callback();
                }
                if (auto* output = state.getOutputTerminal()) {
                    output->interrupt();
                }
            }
            ImGui::PopStyleColor(3);
            if (!can_stop) {
                ImGui::EndDisabled();
            }
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("Stop running script (Ctrl+C)");
        }

        ImGui::SameLine();

        // Reset button
        if (ImGui::Button("Reset")) {
            reset_python_state(state);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Reset Python state (Ctrl+R)");

        ImGui::SameLine();

        // Clear button
        if (ImGui::Button("Clear")) {
            state.clear();
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Clear console (Ctrl+L)");

        ImGui::PopStyleVar(2);

        ImGui::Spacing();
        ImGui::Separator();

        // Calculate pane sizes
        const ImVec2 content_avail = ImGui::GetContentRegionAvail();
        const float total_height = content_avail.y;

        float top_height = total_height * g_splitter_ratio - SPLITTER_THICKNESS / 2;
        float bottom_height = total_height * (1.0f - g_splitter_ratio) - SPLITTER_THICKNESS / 2;

        top_height = std::max(top_height, MIN_PANE_HEIGHT);
        bottom_height = std::max(bottom_height, MIN_PANE_HEIGHT);

        // Script Editor (top pane)
        ImGui::BeginChild("##docked_script_editor_pane", ImVec2(content_avail.x, top_height), false,
                          ImGuiWindowFlags_HorizontalScrollbar);
        {
            ImGui::SetWindowFontScale(state.getFontScale());

            const ImVec2 editor_size(ImGui::GetContentRegionAvail().x,
                                     ImGui::GetContentRegionAvail().y);

            if (ctx.fonts.monospace) {
                ImGui::PushFont(ctx.fonts.monospace);
            }

            // Check if terminal has focus - prevent editor from receiving input
            bool terminal_has_focus = false;
            if (auto* terminal = state.getTerminal()) {
                terminal_has_focus = terminal->isFocused();
            }

            if (auto* editor = state.getEditor()) {
                // When terminal has focus, make editor read-only to prevent input capture
                editor->setReadOnly(terminal_has_focus);

                if (editor->render(editor_size)) {
                    execute_python_code(editor->getText(), state);
                }
            }

            if (ctx.fonts.monospace) {
                ImGui::PopFont();
            }

            ImGui::SetWindowFontScale(1.0f);
        }
        ImGui::EndChild();

        // Horizontal splitter
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, t.palette.primary_dim);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, t.palette.primary);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0);

        ImGui::Button("##docked_splitter", ImVec2(content_avail.x, SPLITTER_THICKNESS));

        if (ImGui::IsItemHovered()) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        }

        if (ImGui::IsItemActive()) {
            const float delta = ImGui::GetIO().MouseDelta.y;
            if (delta != 0.0f) {
                g_splitter_ratio += delta / total_height;
                g_splitter_ratio = std::clamp(g_splitter_ratio, 0.2f, 0.8f);
            }
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        }

        ImGui::PopStyleVar();
        ImGui::PopStyleColor(3);

        // Bottom pane with tabs
        ImGui::BeginChild("##docked_bottom_pane", ImVec2(content_avail.x, bottom_height), false);
        {
            ImGui::SetWindowFontScale(state.getFontScale());

            if (ImGui::BeginTabBar("##docked_console_tabs")) {
                // Output tab (read-only terminal for script output)
                if (ImGui::BeginTabItem("Output")) {
                    state.setActiveTab(0);

                    if (auto* output = state.getOutputTerminal()) {
                        // Don't auto-spawn - spawned on demand when running code
                        output->setReadOnly(true);
                        output->render(ctx.fonts.monospace);
                    }

                    ImGui::EndTabItem();
                }

                // Terminal tab (interactive Python REPL)
                if (ImGui::BeginTabItem("Terminal")) {
                    state.setActiveTab(1);

                    if (auto* terminal = state.getTerminal()) {
                        // Auto-spawn Python on first use
                        if (!terminal->is_running()) {
                            terminal->spawn(getPythonExecutable());
                        }
                        terminal->render(ctx.fonts.monospace);
                    }

                    ImGui::EndTabItem();
                }

                // Packages tab (UV package manager output)
                if (ImGui::BeginTabItem("Packages")) {
                    state.setActiveTab(2);

                    if (auto* uv_console = state.getUvConsole()) {
                        ImVec2 avail = ImGui::GetContentRegionAvail();

                        auto& pm = python::PackageManager::instance();
                        const bool is_running = pm.has_running_operation();

                        // Package input row
                        ImGui::BeginDisabled(is_running);
                        ImGui::SetNextItemWidth(avail.x - 220);
                        bool enter_pressed = ImGui::InputTextWithHint(
                            "##docked_pkg_input", "Package name (e.g., numpy)",
                            g_package_input, sizeof(g_package_input),
                            ImGuiInputTextFlags_EnterReturnsTrue);
                        ImGui::SameLine();
                        if (ImGui::Button("Install##docked", ImVec2(70, 0)) || enter_pressed) {
                            start_async_install(g_package_input, uv_console);
                            g_package_input[0] = '\0';
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Uninstall##docked", ImVec2(70, 0))) {
                            start_async_uninstall(g_package_input, uv_console);
                            g_package_input[0] = '\0';
                        }
                        ImGui::EndDisabled();

                        ImGui::SameLine();
                        if (ImGui::Button("Clear##docked", ImVec2(50, 0))) {
                            uv_console->clear();
                        }

                        avail.y -= ImGui::GetTextLineHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y;

                        // Status line if running
                        if (is_running) {
                            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Running UV operation...");
                            ImGui::SameLine();
                            if (ImGui::SmallButton("Cancel##docked")) {
                                pm.cancel_async();
                            }
                            avail.y -= ImGui::GetTextLineHeightWithSpacing();
                        }

                        uv_console->render(avail);
                    }

                    ImGui::EndTabItem();
                }

                ImGui::EndTabBar();
            }

            ImGui::SetWindowFontScale(1.0f);
        }
        ImGui::EndChild();

        // Keyboard shortcuts
        if (ImGui::GetIO().KeyCtrl) {
            if (ImGui::IsKeyPressed(ImGuiKey_L, false)) {
                state.clear();
            }
            if (ImGui::IsKeyPressed(ImGuiKey_R, false)) {
                reset_python_state(state);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_O, false)) {
                open_script_dialog(state);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_S, false)) {
                save_current_script(state);
            }
            if (ImGui::GetIO().KeyShift && ImGui::IsKeyPressed(ImGuiKey_F, false)) {
                if (auto* editor = state.getEditor()) {
                    const std::string formatted = lfs::python::format_python_code(editor->getText());
                    editor->setText(formatted);
                    state.setModified(true);
                }
            }
            // Font scaling: Ctrl++ / Ctrl+= to increase, Ctrl+- to decrease, Ctrl+0 to reset
            if (ImGui::IsKeyPressed(ImGuiKey_Equal, false) ||
                ImGui::IsKeyPressed(ImGuiKey_KeypadAdd, false)) {
                state.increaseFontScale();
            }
            if (ImGui::IsKeyPressed(ImGuiKey_Minus, false) ||
                ImGui::IsKeyPressed(ImGuiKey_KeypadSubtract, false)) {
                state.decreaseFontScale();
            }
            if (ImGui::IsKeyPressed(ImGuiKey_0, false) ||
                ImGui::IsKeyPressed(ImGuiKey_Keypad0, false)) {
                state.resetFontScale();
            }
        }

        ImGui::End();
        ImGui::PopStyleColor();
#endif // LFS_BUILD_PYTHON_BINDINGS
    }

} // namespace lfs::vis::gui::panels
