/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
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

        std::filesystem::path uv_path() const;
        bool is_uv_available() const;

        std::filesystem::path venv_dir() const;
        std::filesystem::path venv_python() const;
        std::filesystem::path site_packages_dir() const;
        bool ensure_venv();
        bool is_venv_ready() const;

        InstallResult install(const std::string& package);
        InstallResult uninstall(const std::string& package);
        InstallResult install_torch(const std::string& cuda_version = "auto",
                                    const std::string& torch_version = "");

        std::vector<PackageInfo> list_installed() const;
        bool is_installed(const std::string& package) const;

    private:
        PackageManager();
        InstallResult execute_uv(const std::vector<std::string>& args) const;

        std::filesystem::path m_venv_dir;
        mutable std::mutex m_mutex;
        mutable bool m_venv_ready = false;
    };

} // namespace lfs::python
