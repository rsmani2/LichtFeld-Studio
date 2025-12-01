/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/panels/menu_bar.hpp"
#include "config.h"
#include "core_new/logger.hpp"
#include "gui/utils/windows_utils.hpp"
#include <GLFW/glfw3.h>
#include <imgui.h>

#include <cstdlib> // for system()
#ifdef _WIN32
#include <windows.h> // for ShellExecuteA
#endif

namespace lfs::vis::gui {

    MenuBar::MenuBar() = default;
    MenuBar::~MenuBar() = default;

    void MenuBar::render() {
        // Modern color scheme
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(12.0f, 8.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(12.0f, 6.0f));
        ImGui::PushStyleColor(ImGuiCol_MenuBarBg, ImVec4(0.15f, 0.15f, 0.17f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0.12f, 0.12f, 0.14f, 1.0f)); // dark menus
        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.35f, 0.65f, 1.0f, 0.25f));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.45f, 0.75f, 1.0f, 0.5f));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.45f, 0.75f, 1.0f, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.95f, 0.95f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TextDisabled, ImVec4(0.6f, 0.6f, 0.6f, 1.0f));

        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Open Project") && on_open_project_) {
                    on_open_project_();
                }
                if (ImGui::MenuItem("Import Dataset") && on_import_dataset_) {
                    on_import_dataset_();
                }
                if (ImGui::MenuItem("Import Ply") && on_import_ply_) {
                    on_import_ply_();
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Save Project As...") && on_save_project_as_) {
                    on_save_project_as_();
                }
                if (ImGui::MenuItem("Save Project", nullptr, false, !is_project_temp_) && on_save_project_) {
                    on_save_project_();
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Exit") && on_exit_) {
                    on_exit_();
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Edit")) {
                if (ImGui::MenuItem("Input Settings...")) {
                    show_input_settings_ = true;
                }
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Help")) {
                if (ImGui::MenuItem("Getting Started")) {
                    show_getting_started_ = true;
                }
                if (ImGui::MenuItem("About LichtFeld Studio")) {
                    show_about_window_ = true;
                }
                ImGui::EndMenu();
            }

            ImGui::EndMainMenuBar();
        }

        ImGui::PopStyleColor(7);
        ImGui::PopStyleVar(2);

        renderGettingStartedWindow();
        renderAboutWindow();
        renderInputSettingsWindow();
    }

    void MenuBar::openURL(const char* url) {
#ifdef _WIN32
        ShellExecuteA(nullptr, "open", url, nullptr, nullptr, SW_SHOWNORMAL);
#else
        std::string cmd = "xdg-open " + std::string(url);
        system(cmd.c_str());
#endif
    }

    void MenuBar::renderGettingStartedWindow() {
        if (!show_getting_started_) {
            return;
        }

        constexpr ImGuiWindowFlags WINDOW_FLAGS = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize;
        ImGui::SetNextWindowSize(ImVec2(700, 0), ImGuiCond_Once);

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20.0f, 20.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 12.0f));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.11f, 0.11f, 0.13f, 0.98f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.95f, 0.95f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.15f, 0.15f, 0.18f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.2f, 0.2f, 0.24f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.26f, 0.59f, 0.98f, 0.3f));

        if (ImGui::Begin("Getting Started", &show_getting_started_, WINDOW_FLAGS)) {
            ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
            ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "QUICK START GUIDE");
            ImGui::PopFont();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::TextWrapped("Learn how to prepare datasets and get started with LichtFeld Studio:");
            ImGui::Spacing();
            ImGui::Spacing();

            // Reality Scan video with modern styling
            const char* reality_scan_url = "http://www.youtube.com/watch?v=JWmkhTlbDvg";
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.24f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.26f, 0.26f, 0.3f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.3f, 0.3f, 0.35f, 1.0f));

            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "*");
            ImGui::SameLine();
            ImGui::TextWrapped("Using Reality Scan to create a dataset");

            ImGui::Indent(25.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
            ImGui::TextWrapped("%s", reality_scan_url);
            ImGui::PopStyleColor();

            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                if (ImGui::IsItemClicked()) {
                    openURL(reality_scan_url);
                }
            }
            ImGui::Unindent(25.0f);
            ImGui::PopStyleColor(3);

            ImGui::Spacing();

            // Colmap tutorial video
            const char* colmap_tutorial_url = "https://www.youtube.com/watch?v=-3TBbukYN00";
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.24f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.26f, 0.26f, 0.3f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.3f, 0.3f, 0.35f, 1.0f));

            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "*");
            ImGui::SameLine();
            ImGui::TextWrapped("Beginner Tutorial - Using COLMAP to create a dataset");

            ImGui::Indent(25.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
            ImGui::TextWrapped("%s", colmap_tutorial_url);
            ImGui::PopStyleColor();

            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            }
            if (ImGui::IsItemClicked()) {
                openURL(colmap_tutorial_url);
            }

            ImGui::Unindent(25.0f);
            ImGui::PopStyleColor(3);

            ImGui::Spacing();
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            // FAQ link
            ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "FREQUENTLY ASKED QUESTIONS");
            ImGui::Spacing();

            const char* faq_url = "https://github.com/MrNeRF/LichtFeld-Studio/blob/master/docs/docs/faq.md";
            ImGui::Indent(25.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
            ImGui::TextWrapped("%s", faq_url);
            ImGui::PopStyleColor();

            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            }
            if (ImGui::IsItemClicked()) {
                openURL(faq_url);
            }

            ImGui::Unindent(25.0f);
        }
        ImGui::End();

        ImGui::PopStyleColor(5);
        ImGui::PopStyleVar(3);
    }

    void MenuBar::renderAboutWindow() {
        if (!show_about_window_) {
            return;
        }

        constexpr ImGuiWindowFlags WINDOW_FLAGS = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_AlwaysAutoResize;
        ImGui::SetNextWindowSize(ImVec2(750, 0), ImGuiCond_Once);

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20.0f, 20.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 10.0f));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.11f, 0.11f, 0.13f, 0.98f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.95f, 0.95f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.15f, 0.15f, 0.18f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.2f, 0.2f, 0.24f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.26f, 0.59f, 0.98f, 0.3f));
        ImGui::PushStyleColor(ImGuiCol_TableHeaderBg, ImVec4(0.2f, 0.2f, 0.24f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TableBorderStrong, ImVec4(0.3f, 0.3f, 0.35f, 1.0f));

        static constexpr const char* REPO_URL = "https://github.com/MrNeRF/LichtFeld-Studio";
        static constexpr const char* WEBSITE_URL = "https://lichtfeld.io";

        if (ImGui::Begin("About LichtFeld Studio", &show_about_window_, WINDOW_FLAGS)) {
            ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "LICHTFELD STUDIO");
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            ImGui::TextWrapped(
                "A high-performance C++ and CUDA implementation of 3D Gaussian Splatting for "
                "real-time neural rendering, training, and visualization.");

            ImGui::Spacing();
            ImGui::Spacing();

            ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "BUILD INFORMATION");
            ImGui::Spacing();

            constexpr ImGuiTableFlags TABLE_FLAGS = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp;
            if (ImGui::BeginTable("build_info_table", 2, TABLE_FLAGS)) {
                ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                static constexpr ImVec4 LABEL_COLOR{0.7f, 0.7f, 0.7f, 1.0f};

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(LABEL_COLOR, "Version");
                ImGui::TableNextColumn();
                ImGui::TextWrapped("%s", GIT_TAGGED_VERSION);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                }
                if (ImGui::IsItemClicked()) {
                    ImGui::SetClipboardText(GIT_TAGGED_VERSION);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(LABEL_COLOR, "Commit");
                ImGui::TableNextColumn();
                ImGui::Text("%s", GIT_COMMIT_HASH_SHORT);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                }
                if (ImGui::IsItemClicked()) {
                    ImGui::SetClipboardText(GIT_COMMIT_HASH_SHORT);
                }

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(LABEL_COLOR, "Build Type");
                ImGui::TableNextColumn();
#ifdef DEBUG_BUILD
                ImGui::Text("Debug");
#else
                ImGui::Text("Release");
#endif

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(LABEL_COLOR, "Platform");
                ImGui::TableNextColumn();
#ifdef PLATFORM_WINDOWS
                ImGui::Text("Windows");
#elif defined(PLATFORM_LINUX)
                ImGui::Text("Linux");
#else
                ImGui::Text("Unknown");
#endif

                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextColored(LABEL_COLOR, "CUDA-GL Interop");
                ImGui::TableNextColumn();
#ifdef CUDA_GL_INTEROP_ENABLED
                ImGui::Text("Enabled");
#else
                ImGui::Text("Disabled");
#endif

                ImGui::EndTable();
            }

            ImGui::Spacing();
            ImGui::Spacing();

            ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "LINKS");
            ImGui::Spacing();

            static constexpr ImVec4 LINK_COLOR{0.4f, 0.7f, 1.0f, 1.0f};

            ImGui::Text("Repository:");
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, LINK_COLOR);
            ImGui::Text("%s", REPO_URL);
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            }
            if (ImGui::IsItemClicked()) {
                openURL(REPO_URL);
            }

            ImGui::Text("Website:");
            ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Text, LINK_COLOR);
            ImGui::Text("%s", WEBSITE_URL);
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
            }
            if (ImGui::IsItemClicked()) {
                openURL(WEBSITE_URL);
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "LichtFeld Studio Authors");
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), " | ");
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Licensed under GPLv3");
        }
        ImGui::End();

        ImGui::PopStyleColor(7);
        ImGui::PopStyleVar(3);
    }

    void MenuBar::setOnImportDataset(std::function<void()> callback) {
        on_import_dataset_ = std::move(callback);
    }

    void MenuBar::setOnOpenProject(std::function<void()> callback) {
        on_open_project_ = std::move(callback);
    }

    void MenuBar::setOnImportPLY(std::function<void()> callback) {
        on_import_ply_ = std::move(callback);
    }

    void MenuBar::setOnSaveProjectAs(std::function<void()> callback) {
        on_save_project_as_ = std::move(callback);
    }

    void MenuBar::setOnSaveProject(std::function<void()> callback) {
        on_save_project_ = std::move(callback);
    }

    void MenuBar::setOnExit(std::function<void()> callback) {
        on_exit_ = std::move(callback);
    }

    void MenuBar::renderBindingRow(const input::Action action) {
        static constexpr ImVec4 COLOR_ACTION{0.9f, 0.9f, 0.9f, 1.0f};
        static constexpr ImVec4 COLOR_BINDING{0.4f, 0.7f, 1.0f, 1.0f};
        static constexpr ImVec4 COLOR_WAITING{1.0f, 0.8f, 0.2f, 1.0f};
        static constexpr ImVec4 COLOR_REBIND{0.2f, 0.4f, 0.6f, 1.0f};
        static constexpr ImVec4 COLOR_REBIND_HOVER{0.3f, 0.5f, 0.7f, 1.0f};
        static constexpr ImVec4 COLOR_REBIND_ACTIVE{0.4f, 0.6f, 0.8f, 1.0f};
        static constexpr ImVec4 COLOR_CANCEL{0.6f, 0.2f, 0.2f, 1.0f};
        static constexpr ImVec4 COLOR_CANCEL_HOVER{0.7f, 0.3f, 0.3f, 1.0f};
        static constexpr ImVec4 COLOR_CANCEL_ACTIVE{0.8f, 0.4f, 0.4f, 1.0f};

        const bool is_rebinding = rebinding_action_.has_value() && *rebinding_action_ == action;

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextColored(COLOR_ACTION, "%s", input::getActionName(action).c_str());

        ImGui::TableNextColumn();
        if (is_rebinding) {
            ImGui::TextColored(COLOR_WAITING, "Press key or mouse...");
        } else {
            const std::string desc = input_bindings_->getTriggerDescription(action);
            ImGui::TextColored(COLOR_BINDING, "%s", desc.c_str());
        }

        ImGui::TableNextColumn();
        const int action_id = static_cast<int>(action);
        if (is_rebinding) {
            ImGui::PushStyleColor(ImGuiCol_Button, COLOR_CANCEL);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, COLOR_CANCEL_HOVER);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, COLOR_CANCEL_ACTIVE);
            char label[32];
            snprintf(label, sizeof(label), "Cancel##%d", action_id);
            if (ImGui::Button(label, ImVec2(-1, 0))) {
                cancelCapture();
            }
            ImGui::PopStyleColor(3);
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, COLOR_REBIND);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, COLOR_REBIND_HOVER);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, COLOR_REBIND_ACTIVE);
            char label[32];
            snprintf(label, sizeof(label), "Rebind##%d", action_id);
            if (ImGui::Button(label, ImVec2(-1, 0))) {
                rebinding_action_ = action;
            }
            ImGui::PopStyleColor(3);
        }
    }

    void MenuBar::captureKey(const int key, const int mods) {
        if (!rebinding_action_.has_value() || !input_bindings_) {
            return;
        }

        if (key == GLFW_KEY_ESCAPE) {
            cancelCapture();
            return;
        }

        // Ignore modifier-only keys
        if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT ||
            key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL ||
            key == GLFW_KEY_LEFT_ALT || key == GLFW_KEY_RIGHT_ALT ||
            key == GLFW_KEY_LEFT_SUPER || key == GLFW_KEY_RIGHT_SUPER) {
            return;
        }

        const input::KeyTrigger trigger{key, mods, false};
        input_bindings_->setBinding(*rebinding_action_, trigger);
        rebinding_action_.reset();
    }

    void MenuBar::captureMouseButton(const int button, const int mods) {
        if (!rebinding_action_.has_value() || !input_bindings_) {
            return;
        }

        const auto mouse_btn = static_cast<input::MouseButton>(button);
        const input::MouseDragTrigger trigger{mouse_btn, mods};
        input_bindings_->setBinding(*rebinding_action_, trigger);
        rebinding_action_.reset();
    }

    void MenuBar::cancelCapture() {
        rebinding_action_.reset();
    }

    void MenuBar::renderInputSettingsWindow() {
        if (!show_input_settings_) {
            cancelCapture();
            return;
        }

        constexpr ImGuiWindowFlags WINDOW_FLAGS = ImGuiWindowFlags_NoDocking;
        ImGui::SetNextWindowSize(ImVec2(600, 600), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSizeConstraints(ImVec2(400, 300), ImVec2(FLT_MAX, FLT_MAX));

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20.0f, 20.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 10.0f));
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.11f, 0.11f, 0.13f, 0.98f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.95f, 0.95f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.15f, 0.15f, 0.18f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.2f, 0.2f, 0.24f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.26f, 0.59f, 0.98f, 0.3f));
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.16f, 0.16f, 0.18f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.2f, 0.2f, 0.24f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.25f, 0.25f, 0.28f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0.12f, 0.12f, 0.14f, 0.98f));
        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.26f, 0.59f, 0.98f, 0.3f));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.26f, 0.59f, 0.98f, 0.5f));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.26f, 0.59f, 0.98f, 0.7f));
        ImGui::PushStyleColor(ImGuiCol_TableHeaderBg, ImVec4(0.2f, 0.2f, 0.24f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_TableBorderStrong, ImVec4(0.3f, 0.3f, 0.35f, 1.0f));

        if (ImGui::Begin("Input Settings", &show_input_settings_, WINDOW_FLAGS)) {
            ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "INPUT PROFILE");
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (input_bindings_) {
                ImGui::Text("Active Profile:");
                ImGui::SameLine();

                const auto profiles = input_bindings_->getAvailableProfiles();
                const auto& current = input_bindings_->getCurrentProfileName();
                const bool is_rebinding = rebinding_action_.has_value();

                if (is_rebinding) {
                    ImGui::BeginDisabled();
                }

                if (ImGui::BeginCombo("##profile", current.c_str())) {
                    for (const auto& profile : profiles) {
                        const bool is_selected = (profile == current);
                        if (ImGui::Selectable(profile.c_str(), is_selected)) {
                            input_bindings_->loadProfile(profile);
                        }
                        if (is_selected) {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }

                if (is_rebinding) {
                    ImGui::EndDisabled();
                }

                ImGui::Spacing();
                ImGui::Spacing();

                ImGui::TextColored(ImVec4(0.26f, 0.59f, 0.98f, 1.0f), "CURRENT BINDINGS");
                ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Click Rebind, then press key or mouse button");
                ImGui::Spacing();

                constexpr ImGuiTableFlags TABLE_FLAGS = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY;
                const float available_height = ImGui::GetContentRegionAvail().y - 100.0f;
                const float table_height = std::max(200.0f, available_height);

                if (ImGui::BeginTable("bindings_table", 3, TABLE_FLAGS, ImVec2(0, table_height))) {
                    ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 180.0f);
                    ImGui::TableSetupColumn("Binding", ImGuiTableColumnFlags_WidthStretch);
                    ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                    ImGui::TableHeadersRow();

                    static constexpr ImU32 SECTION_BG_COLOR = IM_COL32(40, 60, 80, 255);
                    static constexpr ImVec4 SECTION_TEXT_COLOR{0.5f, 0.8f, 1.0f, 1.0f};

                    const auto renderSectionHeader = [](const char* title) {
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, SECTION_BG_COLOR);
                        ImGui::TextColored(SECTION_TEXT_COLOR, "%s", title);
                        ImGui::TableNextColumn();
                        ImGui::TableNextColumn();
                    };

                    renderSectionHeader("NAVIGATION");
                    renderBindingRow(input::Action::CAMERA_ORBIT);
                    renderBindingRow(input::Action::CAMERA_PAN);
                    renderBindingRow(input::Action::CAMERA_ZOOM);
                    renderBindingRow(input::Action::CAMERA_MOVE_FORWARD);
                    renderBindingRow(input::Action::CAMERA_MOVE_BACKWARD);
                    renderBindingRow(input::Action::CAMERA_MOVE_LEFT);
                    renderBindingRow(input::Action::CAMERA_MOVE_RIGHT);
                    renderBindingRow(input::Action::CAMERA_MOVE_UP);
                    renderBindingRow(input::Action::CAMERA_MOVE_DOWN);
                    renderBindingRow(input::Action::CAMERA_RESET_HOME);
                    renderBindingRow(input::Action::CAMERA_NEXT_VIEW);
                    renderBindingRow(input::Action::CAMERA_PREV_VIEW);
                    renderBindingRow(input::Action::CAMERA_SPEED_UP);
                    renderBindingRow(input::Action::CAMERA_SPEED_DOWN);
                    renderBindingRow(input::Action::ZOOM_SPEED_UP);
                    renderBindingRow(input::Action::ZOOM_SPEED_DOWN);

                    renderSectionHeader("SELECTION");
                    renderBindingRow(input::Action::SELECTION_ADD);
                    renderBindingRow(input::Action::SELECTION_REMOVE);
                    renderBindingRow(input::Action::SELECT_MODE_CENTERS);
                    renderBindingRow(input::Action::SELECT_MODE_RECTANGLE);
                    renderBindingRow(input::Action::SELECT_MODE_POLYGON);
                    renderBindingRow(input::Action::SELECT_MODE_LASSO);
                    renderBindingRow(input::Action::SELECT_MODE_RINGS);
                    renderBindingRow(input::Action::TOGGLE_DEPTH_MODE);
                    renderBindingRow(input::Action::DEPTH_ADJUST_FAR);
                    renderBindingRow(input::Action::DEPTH_ADJUST_SIDE);

                    renderSectionHeader("EDITING");
                    renderBindingRow(input::Action::DELETE_SELECTED);
                    renderBindingRow(input::Action::UNDO);
                    renderBindingRow(input::Action::REDO);
                    renderBindingRow(input::Action::INVERT_SELECTION);
                    renderBindingRow(input::Action::DESELECT_ALL);
                    renderBindingRow(input::Action::APPLY_CROP_BOX);
                    renderBindingRow(input::Action::CANCEL_POLYGON);
                    renderBindingRow(input::Action::CYCLE_BRUSH_MODE);

                    renderSectionHeader("VIEW");
                    renderBindingRow(input::Action::TOGGLE_SPLIT_VIEW);
                    renderBindingRow(input::Action::TOGGLE_GT_COMPARISON);
                    renderBindingRow(input::Action::CYCLE_PLY);
                    renderBindingRow(input::Action::CYCLE_SELECTION_VIS);

                    ImGui::EndTable();
                }
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "Input bindings not available");
            }

            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();

            if (input_bindings_) {
                static constexpr ImVec4 BTN_SAVE{0.2f, 0.5f, 0.2f, 1.0f};
                static constexpr ImVec4 BTN_SAVE_HOVER{0.3f, 0.6f, 0.3f, 1.0f};
                static constexpr ImVec4 BTN_SAVE_ACTIVE{0.25f, 0.55f, 0.25f, 1.0f};
                static constexpr ImVec4 BTN_RESET{0.5f, 0.2f, 0.2f, 1.0f};
                static constexpr ImVec4 BTN_RESET_HOVER{0.6f, 0.3f, 0.3f, 1.0f};
                static constexpr ImVec4 BTN_RESET_ACTIVE{0.55f, 0.25f, 0.25f, 1.0f};

                ImGui::PushStyleColor(ImGuiCol_Button, BTN_SAVE);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, BTN_SAVE_HOVER);
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, BTN_SAVE_ACTIVE);
                if (ImGui::Button("Save Current Profile")) {
                    input_bindings_->saveProfile(input_bindings_->getCurrentProfileName());
                }
                ImGui::PopStyleColor(3);

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Button, BTN_RESET);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, BTN_RESET_HOVER);
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, BTN_RESET_ACTIVE);
                if (ImGui::Button("Reset to Default")) {
                    const auto config_dir = input::InputBindings::getConfigDir();
                    const auto saved_path = config_dir / "Default.json";
                    if (std::filesystem::exists(saved_path)) {
                        std::filesystem::remove(saved_path);
                    }
                    input_bindings_->loadProfile("Default");
                    // Save hardcoded defaults to disk
                    input_bindings_->saveProfile("Default");
                }
                ImGui::PopStyleColor(3);

                ImGui::Spacing();

                static constexpr ImVec4 BTN_IO{0.3f, 0.3f, 0.5f, 1.0f};
                static constexpr ImVec4 BTN_IO_HOVER{0.4f, 0.4f, 0.6f, 1.0f};
                static constexpr ImVec4 BTN_IO_ACTIVE{0.35f, 0.35f, 0.55f, 1.0f};

                ImGui::PushStyleColor(ImGuiCol_Button, BTN_IO);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, BTN_IO_HOVER);
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, BTN_IO_ACTIVE);
                if (ImGui::Button("Export...")) {
                    const auto path = SaveJsonFileDialog("input_bindings");
                    if (!path.empty()) {
                        input_bindings_->saveProfileToFile(path);
                    }
                }
                ImGui::PopStyleColor(3);

                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_Button, BTN_IO);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, BTN_IO_HOVER);
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, BTN_IO_ACTIVE);
                if (ImGui::Button("Import...")) {
                    if (const auto path = OpenJsonFileDialog(); !path.empty() && std::filesystem::exists(path)) {
                        input_bindings_->loadProfileFromFile(path);
                    }
                }
                ImGui::PopStyleColor(3);

                ImGui::Spacing();
            }

            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "Save to persist custom bindings");
        }
        ImGui::End();

        ImGui::PopStyleColor(14);
        ImGui::PopStyleVar(3);
    }

} // namespace lfs::vis::gui
