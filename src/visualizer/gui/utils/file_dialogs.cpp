/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "file_dialogs.hpp"
#include "core/logger.hpp"
#include <filesystem>
#include <imgui.h>
#include <cstring>

#ifdef WIN32
#include "windows_utils.hpp"
#include <ShlObj.h>
#include <windows.h>
#endif

namespace gs::gui {

#ifndef WIN32
    // Static variables for export dialog on Linux (import uses FileBrowser)
    static bool show_export_dialog = false;
    static char export_path_buffer[1024] = "";
    static std::function<void(const std::string&)> export_callback;

    // Callback storage for file browser integration (Linux)
    static std::function<void(const std::string&)> import_callback_;
    static std::function<void(const std::string&)> open_project_callback_;
    static std::function<void(const std::string&)> open_ply_callback_;
    static std::function<void(const std::string&)> open_dataset_callback_;

    // Render export config dialog (Linux only)
    void RenderExportConfigDialog() {
        if (!show_export_dialog) return;

        ImGui::SetNextWindowSize(ImVec2(600, 200), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Export Configuration", &show_export_dialog, ImGuiWindowFlags_NoDocking)) {
            ImGui::Text("Save configuration to:");
            ImGui::InputText("##exportpath", export_path_buffer, sizeof(export_path_buffer));

            ImGui::Separator();
            ImGui::TextWrapped("Enter the full path including filename (e.g., /home/user/config.json)");
            ImGui::Text("Current directory: %s", std::filesystem::current_path().string().c_str());

            ImGui::Separator();

            if (ImGui::Button("Save", ImVec2(120, 0))) {
                if (strlen(export_path_buffer) > 0) {
                    std::filesystem::path path(export_path_buffer);

                    // Add .json extension if missing
                    if (path.extension().empty()) {
                        path += ".json";
                    }

                    if (export_callback) {
                        export_callback(path.string());
                    }

                    show_export_dialog = false;
                    export_path_buffer[0] = '\0';
                    export_callback = nullptr;
                }
            }

            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                show_export_dialog = false;
                export_path_buffer[0] = '\0';
                export_callback = nullptr;
            }
        }
        ImGui::End();
    }

    // Get stored callbacks for file browser integration
    std::function<void(const std::string&)> GetImportConfigCallback() { return import_callback_; }
    std::function<void(const std::string&)> GetOpenProjectCallback() { return open_project_callback_; }
    std::function<void(const std::string&)> GetOpenPlyCallback() { return open_ply_callback_; }
    std::function<void(const std::string&)> GetOpenDatasetCallback() { return open_dataset_callback_; }

    void ClearPendingCallbacks() {
        import_callback_ = nullptr;
        open_project_callback_ = nullptr;
        open_ply_callback_ = nullptr;
        open_dataset_callback_ = nullptr;
    }
#endif

    void ExportConfigFileDialog(std::function<void(const std::string&)> callback) {
#ifdef WIN32
        // Use native Windows dialog
        PWSTR filePath = nullptr;
        COMDLG_FILTERSPEC rgSpec[] = {
            {L"JSON Configuration", L"*.json"},
            {L"All Files", L"*.*"}
        };

        // Get absolute path to parameter directory
        auto param_dir = std::filesystem::absolute("parameter");
        std::wstring param_dir_wstr = param_dir.wstring();

        if (SUCCEEDED(gs::gui::utils::saveFileNative(filePath, rgSpec, 2, L"optimization_config.json", param_dir_wstr.c_str()))) {
            std::filesystem::path config_path(filePath);
            callback(config_path.string());
        }
#else
        // Use export dialog for Linux
        export_callback = callback;
        show_export_dialog = true;
        auto param_dir = std::filesystem::absolute("parameter");
        std::string default_name = (param_dir / "optimization_config.json").string();
        strncpy(export_path_buffer, default_name.c_str(), sizeof(export_path_buffer) - 1);
        export_path_buffer[sizeof(export_path_buffer) - 1] = '\0';
#endif
    }

    void ImportConfigFileDialog(std::function<void(const std::string&)> callback) {
#ifdef WIN32
        // Use native Windows dialog
        PWSTR filePath = nullptr;
        COMDLG_FILTERSPEC rgSpec[] = {
            {L"JSON Configuration", L"*.json"},
            {L"All Files", L"*.*"}
        };

        // Get absolute path to parameter directory
        auto param_dir = std::filesystem::absolute("parameter");
        std::wstring param_dir_wstr = param_dir.wstring();

        if (SUCCEEDED(gs::gui::utils::selectFileNative(filePath, rgSpec, 2, false, param_dir_wstr.c_str()))) {
            std::filesystem::path config_path(filePath);
            callback(config_path.string());
        }
#else
        // Store callback for FileBrowser to use (Linux)
        import_callback_ = callback;
#endif
    }

    void OpenProjectFileDialog(std::function<void(const std::string&)> callback) {
#ifdef WIN32
        PWSTR filePath = nullptr;
        COMDLG_FILTERSPEC rgSpec[] = {
            {L"LichtFeldStudio Project File", L"*.lfs;*.ls"},
        };

        if (SUCCEEDED(gs::gui::utils::selectFileNative(filePath, rgSpec, 1, false))) {
            std::filesystem::path project_path(filePath);
            callback(project_path.string());
        }
#else
        // Store callback for FileBrowser to use (Linux)
        open_project_callback_ = callback;
#endif
    }

    void OpenPlyFileDialog(std::function<void(const std::string&)> callback) {
#ifdef WIN32
        PWSTR filePath = nullptr;
        COMDLG_FILTERSPEC rgSpec[] = {
            {L"Point Cloud", L"*.ply"},
        };

        if (SUCCEEDED(gs::gui::utils::selectFileNative(filePath, rgSpec, 1, false))) {
            std::filesystem::path ply_path(filePath);
            callback(ply_path.string());
        }
#else
        // Store callback for FileBrowser to use (Linux)
        open_ply_callback_ = callback;
#endif
    }

    void OpenDatasetFolderDialog(std::function<void(const std::string&)> callback) {
#ifdef WIN32
        PWSTR filePath = nullptr;
        if (SUCCEEDED(gs::gui::utils::selectFileNative(filePath, nullptr, 0, true))) {
            std::filesystem::path dataset_path(filePath);
            if (std::filesystem::is_directory(dataset_path)) {
                callback(dataset_path.string());
            }
        }
#else
        // Store callback for FileBrowser to use (Linux)
        open_dataset_callback_ = callback;
#endif
    }

    void SaveProjectFileDialog(bool* p_open, std::function<void(const std::string&)> callback) {
#ifdef WIN32
        PWSTR filePath = nullptr;
        if (SUCCEEDED(gs::gui::utils::selectFileNative(filePath, nullptr, 0, true))) {
            std::filesystem::path project_path(filePath);
            callback(project_path.string());
            if (p_open) *p_open = false;
        }
#else
        // This is handled by SaveProjectBrowser on Linux
        // Not used for config export/import
        (void)p_open;
        (void)callback;
#endif
    }

} // namespace gs::gui
