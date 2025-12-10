/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#ifdef WIN32
#include <Shobjidl.h>
#include <Windows.h>
#endif

namespace gs::gui {

#ifdef WIN32

    namespace utils {
        /**
         * Opens a native Windows file/folder selection dialog
         * @param strDirectory Output path selected by the user
         * @param rgSpec File type filters (can be nullptr)
         * @param cFileTypes Number of file type filters
         * @param blnDirectory True to select folders, false for files
         * @param defaultFolder Default folder path to open (can be nullptr)
         * @return HRESULT indicating success or failure
         */
        HRESULT selectFileNative(PWSTR& strDirectory,
                                 COMDLG_FILTERSPEC rgSpec[] = nullptr,
                                 UINT cFileTypes = 0,
                                 bool blnDirectory = false,
                                 LPCWSTR defaultFolder = nullptr);

        /**
         * Opens a native Windows save file dialog
         * @param strDirectory Output path selected by the user
         * @param rgSpec File type filters (can be nullptr)
         * @param cFileTypes Number of file type filters
         * @param defaultFileName Default file name to suggest
         * @param defaultFolder Default folder path to open (can be nullptr)
         * @return HRESULT indicating success or failure
         */
        HRESULT saveFileNative(PWSTR& strDirectory,
                               COMDLG_FILTERSPEC rgSpec[] = nullptr,
                               UINT cFileTypes = 0,
                               LPCWSTR defaultFileName = nullptr,
                               LPCWSTR defaultFolder = nullptr);
    } // namespace utils

    // in windows- open file browser that search for lfs project
    void OpenProjectFileDialog();
    // in windows- open file browser that search for ply files
    void OpenPlyFileDialog();
    // in windows- open file browser that search directories
    void OpenDatasetFolderDialog();
    // in windows- open file browser for exporting config JSON
    void ExportConfigFileDialog();
    // in windows- open file browser for importing config JSON
    void ImportConfigFileDialog();
#endif
} // namespace gs::gui