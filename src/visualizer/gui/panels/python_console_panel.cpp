/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/python_console_panel.hpp"
#include "gui/editor/console_output.hpp"
#include "gui/editor/python_editor.hpp"
#include "gui/ui_widgets.hpp"
#include "gui/utils/windows_utils.hpp"
#include "theme/theme.hpp"

#include <fstream>
#include <sstream>
#include <imgui.h>

#ifdef LFS_BUILD_PYTHON_BINDINGS
#include <Python.h>
#include <filesystem>
#include <mutex>

#include "core/services.hpp"
#include "python/runner.hpp"
#include "scene/scene_manager.hpp"

namespace {
    std::once_flag g_console_init_once;
    std::once_flag g_syspath_init_once;
    std::once_flag g_scene_provider_once;

    void setup_scene_provider() {
        std::call_once(g_scene_provider_once, [] {
            lfs::python::set_scene_provider([]() -> lfs::vis::Scene* {
                auto* sm = lfs::vis::services().sceneOrNull();
                return sm ? &sm->getScene() : nullptr;
            });
        });
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
                auto* output = state.getConsoleOutput();
                if (output) {
                    output->appendOutput(text, is_error);
                }
            });
        });
    }

    // Remove common leading whitespace from all lines (like Python's textwrap.dedent)
    std::string dedent(const std::string& text) {
        std::vector<std::string> lines;
        std::istringstream stream(text);
        std::string line;
        while (std::getline(stream, line)) {
            lines.push_back(line);
        }

        if (lines.empty()) {
            return text;
        }

        // Find minimum indentation (ignoring empty lines)
        size_t min_indent = std::string::npos;
        for (const auto& l : lines) {
            if (l.empty() || l.find_first_not_of(" \t") == std::string::npos) {
                continue;
            }
            size_t indent = l.find_first_not_of(" \t");
            if (indent < min_indent) {
                min_indent = indent;
            }
        }

        if (min_indent == 0 || min_indent == std::string::npos) {
            return text;
        }

        // Remove common indentation
        std::ostringstream result;
        for (size_t i = 0; i < lines.size(); ++i) {
            if (i > 0) {
                result << '\n';
            }
            if (lines[i].size() > min_indent) {
                result << lines[i].substr(min_indent);
            } else if (!lines[i].empty() && lines[i].find_first_not_of(" \t") != std::string::npos) {
                result << lines[i];
            }
        }
        return result.str();
    }

    void execute_python_code(const std::string& code, lfs::vis::gui::panels::PythonConsoleState& state) {
        std::string cmd = code;

        // Trim trailing whitespace
        while (!cmd.empty() && (cmd.back() == '\n' || cmd.back() == '\r' || cmd.back() == ' ')) {
            cmd.pop_back();
        }

        // Remove common leading indentation
        cmd = dedent(cmd);

        if (cmd.empty()) {
            return;
        }

        state.addInput(cmd);
        state.addToHistory(cmd);

        const PyGILState_STATE gil = PyGILState_Ensure();

        // For multi-line code, use Py_file_input mode
        const int start_mode = (cmd.find('\n') != std::string::npos) ? Py_file_input : Py_single_input;

        PyObject* const main_module = PyImport_AddModule("__main__");
        PyObject* const globals = PyModule_GetDict(main_module);

        PyObject* result = PyRun_String(cmd.c_str(), start_mode, globals, globals);

        if (result) {
            if (result != Py_None && start_mode == Py_single_input) {
                PyObject* str = PyObject_Str(result);
                if (str) {
                    const char* output_str = PyUnicode_AsUTF8(str);
                    if (output_str && output_str[0] != '\0') {
                        state.addOutput(output_str);
                    }
                    Py_DECREF(str);
                }
            }
            Py_DECREF(result);
        } else {
            PyErr_Print();
        }

        PyGILState_Release(gil);
    }

    void reset_python_state(lfs::vis::gui::panels::PythonConsoleState& state) {
        const PyGILState_STATE gil = PyGILState_Ensure();

        PyObject* main_module = PyImport_AddModule("__main__");
        PyObject* globals = PyModule_GetDict(main_module);

        // Get list of keys to delete (excluding builtins and standard items)
        PyObject* keys = PyDict_Keys(globals);
        if (keys) {
            const Py_ssize_t len = PyList_Size(keys);
            for (Py_ssize_t i = 0; i < len; i++) {
                PyObject* key = PyList_GetItem(keys, i);
                const char* key_str = PyUnicode_AsUTF8(key);
                if (key_str) {
                    // Skip builtins and special names
                    if (key_str[0] == '_' && key_str[1] == '_') {
                        continue;
                    }
                    if (std::string(key_str) == "builtins") {
                        continue;
                    }
                    PyDict_DelItem(globals, key);
                }
            }
            Py_DECREF(keys);
        }

        PyGILState_Release(gil);

        state.addInfo("Python state reset");
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
        : console_output_(std::make_unique<editor::ConsoleOutput>()),
          editor_(std::make_unique<editor::PythonEditor>()) {
    }

    PythonConsoleState::~PythonConsoleState() = default;

    PythonConsoleState& PythonConsoleState::getInstance() {
        static PythonConsoleState instance;
        return instance;
    }

    void PythonConsoleState::addOutput(const std::string& text, uint32_t /*color*/) {
        std::lock_guard lock(mutex_);
        if (console_output_) {
            console_output_->addOutput(text);
        }
    }

    void PythonConsoleState::addError(const std::string& text) {
        std::lock_guard lock(mutex_);
        if (console_output_) {
            console_output_->addError(text);
        }
    }

    void PythonConsoleState::addInput(const std::string& text) {
        std::lock_guard lock(mutex_);
        if (console_output_) {
            console_output_->addInput(text);
        }
    }

    void PythonConsoleState::addInfo(const std::string& text) {
        std::lock_guard lock(mutex_);
        if (console_output_) {
            console_output_->addInfo(text);
        }
    }

    void PythonConsoleState::clear() {
        std::lock_guard lock(mutex_);
        if (console_output_) {
            console_output_->clear();
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

    editor::ConsoleOutput* PythonConsoleState::getConsoleOutput() {
        return console_output_.get();
    }

    editor::PythonEditor* PythonConsoleState::getEditor() {
        return editor_.get();
    }

    namespace {
        // Splitter state
        float g_splitter_ratio = 0.6f;
        constexpr float MIN_PANE_HEIGHT = 100.0f;
        constexpr float SPLITTER_THICKNESS = 6.0f;
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
        setup_scene_provider();

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
                if (ImGui::MenuItem("Clear Console", "Ctrl+L")) {
                    state.clear();
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Copy Console Output")) {
                    if (auto* output = state.getConsoleOutput()) {
                        ImGui::SetClipboardText(output->getAllText().c_str());
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

            if (auto* editor = state.getEditor()) {
                if (editor->render(editor_size)) {
                    // Ctrl+Enter was pressed - execute
                    execute_python_code(editor->getText(), state);
                    editor->clear();
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

        // Console Output (bottom pane)
        ImGui::BeginChild("##console_pane", ImVec2(content_avail.x, bottom_height), false);
        {
            ImGui::TextColored(t.palette.text_dim, "Console Output");
            ImGui::Spacing();

            const ImVec2 output_size(ImGui::GetContentRegionAvail().x,
                                     ImGui::GetContentRegionAvail().y);

            // Use monospace font for console output
            if (ctx.fonts.monospace) {
                ImGui::PushFont(ctx.fonts.monospace);
            }

            if (auto* output = state.getConsoleOutput()) {
                output->render(output_size);
            }

            if (ctx.fonts.monospace) {
                ImGui::PopFont();
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
        setup_scene_provider();

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
            const ImVec2 editor_size(ImGui::GetContentRegionAvail().x,
                                     ImGui::GetContentRegionAvail().y);

            if (ctx.fonts.monospace) {
                ImGui::PushFont(ctx.fonts.monospace);
            }

            if (auto* editor = state.getEditor()) {
                if (editor->render(editor_size)) {
                    execute_python_code(editor->getText(), state);
                    editor->clear();
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

        // Console Output (bottom pane)
        ImGui::BeginChild("##docked_console_pane", ImVec2(content_avail.x, bottom_height), false);
        {
            const ImVec2 output_size(ImGui::GetContentRegionAvail().x,
                                     ImGui::GetContentRegionAvail().y);

            if (ctx.fonts.monospace) {
                ImGui::PushFont(ctx.fonts.monospace);
            }

            if (auto* output = state.getConsoleOutput()) {
                output->render(output_size);
            }

            if (ctx.fonts.monospace) {
                ImGui::PopFont();
            }
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
        }

        ImGui::End();
        ImGui::PopStyleColor();
#endif // LFS_BUILD_PYTHON_BINDINGS
    }

} // namespace lfs::vis::gui::panels
