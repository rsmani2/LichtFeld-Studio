/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <functional>
#include <future>
#include <mutex>
#include <string>
#include <vector>

namespace lfs::python {

    struct PackageInfo {
        std::string name;
        std::string version;
    };

    struct InstallResult {
        bool success = false;
        std::string output;
        std::string error;
    };

    class PackageManager {
    public:
        static PackageManager& instance();

        // UV binary location
        std::filesystem::path uv_path() const;
        bool is_uv_available() const;

        // Package operations (blocking)
        InstallResult install(const std::string& package);
        InstallResult uninstall(const std::string& package);

        // List installed packages
        std::vector<PackageInfo> list_installed() const;

        // Check if package is installed
        bool is_installed(const std::string& package) const;

        // Get site-packages directory
        std::filesystem::path site_packages_dir() const;

    private:
        PackageManager();

        // Execute command and capture output
        InstallResult execute_uv(const std::vector<std::string>& args) const;

        std::filesystem::path m_site_packages;
        mutable std::mutex m_mutex;
    };

} // namespace lfs::python
