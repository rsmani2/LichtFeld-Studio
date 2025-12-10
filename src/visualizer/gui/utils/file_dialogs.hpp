/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <functional>
#include <string>

namespace gs::gui {

    // Cross-platform file dialog functions
    // Windows: Uses native dialogs
    // Linux: Uses existing FileBrowser for import, simple dialog for export

    /**
     * Open a file dialog to export configuration
     * Windows: Native save dialog
     * Linux: Simple save dialog
     */
    void ExportConfigFileDialog(std::function<void(const std::string&)> callback);

    /**
     * Open a file dialog to import configuration
     * Windows: Native open dialog
     * Linux: Triggers FileBrowser (callback stored internally)
     */
    void ImportConfigFileDialog(std::function<void(const std::string&)> callback);

    /**
     * Open a project file dialog
     */
    void OpenProjectFileDialog(std::function<void(const std::string&)> callback);

    /**
     * Open a PLY file dialog
     */
    void OpenPlyFileDialog(std::function<void(const std::string&)> callback);

    /**
     * Open a dataset folder dialog
     */
    void OpenDatasetFolderDialog(std::function<void(const std::string&)> callback);

    /**
     * Save project dialog
     */
    void SaveProjectFileDialog(bool* p_open, std::function<void(const std::string&)> callback);

#ifndef WIN32
    // Linux-specific functions for FileBrowser integration

    /**
     * Render export config dialog (Linux only)
     * Call this in the main render loop
     */
    void RenderExportConfigDialog();

    /**
     * Get stored callbacks for FileBrowser to use
     */
    std::function<void(const std::string&)> GetImportConfigCallback();
    std::function<void(const std::string&)> GetOpenProjectCallback();
    std::function<void(const std::string&)> GetOpenPlyCallback();
    std::function<void(const std::string&)> GetOpenDatasetCallback();

    /**
     * Clear all pending callbacks
     */
    void ClearPendingCallbacks();
#endif

} // namespace gs::gui
