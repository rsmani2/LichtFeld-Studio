/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "package_manager.hpp"
#include "runner.hpp"

#include <core/logger.hpp>

#include <array>
#include <cstdio>
#include <fstream>
#include <regex>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace lfs::python {

    namespace {

        std::filesystem::path get_executable_dir() {
#ifdef _WIN32
            wchar_t path[MAX_PATH];
            GetModuleFileNameW(nullptr, path, MAX_PATH);
            return std::filesystem::path(path).parent_path();
#else
            char path[4096];
            ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
            if (len != -1) {
                path[len] = '\0';
                return std::filesystem::path(path).parent_path();
            }
            return std::filesystem::current_path();
#endif
        }

#ifdef _WIN32
        constexpr const char* UV_BINARY_NAME = "uv.exe";
#else
        constexpr const char* UV_BINARY_NAME = "uv";
#endif

        // Execute command and capture stdout/stderr
        std::pair<int, std::string> execute_command(const std::string& cmd) {
            std::string output;
            int exit_code = -1;

#ifdef _WIN32
            SECURITY_ATTRIBUTES sa;
            sa.nLength = sizeof(SECURITY_ATTRIBUTES);
            sa.bInheritHandle = TRUE;
            sa.lpSecurityDescriptor = nullptr;

            HANDLE hReadPipe, hWritePipe;
            if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0)) {
                return {-1, "Failed to create pipe"};
            }

            SetHandleInformation(hReadPipe, HANDLE_FLAG_INHERIT, 0);

            STARTUPINFOW si = {};
            si.cb = sizeof(si);
            si.hStdOutput = hWritePipe;
            si.hStdError = hWritePipe;
            si.dwFlags |= STARTF_USESTDHANDLES;

            PROCESS_INFORMATION pi = {};

            std::wstring wcmd(cmd.begin(), cmd.end());
            if (!CreateProcessW(nullptr, wcmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
                CloseHandle(hReadPipe);
                CloseHandle(hWritePipe);
                return {-1, "Failed to create process"};
            }

            CloseHandle(hWritePipe);

            char buffer[4096];
            DWORD bytesRead;
            while (ReadFile(hReadPipe, buffer, sizeof(buffer) - 1, &bytesRead, nullptr) && bytesRead > 0) {
                buffer[bytesRead] = '\0';
                output += buffer;
            }

            WaitForSingleObject(pi.hProcess, INFINITE);

            DWORD exitCodeDword;
            GetExitCodeProcess(pi.hProcess, &exitCodeDword);
            exit_code = static_cast<int>(exitCodeDword);

            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
            CloseHandle(hReadPipe);
#else
            FILE* pipe = popen((cmd + " 2>&1").c_str(), "r");
            if (!pipe) {
                return {-1, "Failed to execute command"};
            }

            char buffer[4096];
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                output += buffer;
            }

            exit_code = pclose(pipe);
            if (WIFEXITED(exit_code)) {
                exit_code = WEXITSTATUS(exit_code);
            }
#endif

            return {exit_code, output};
        }

    } // namespace

    PackageManager::PackageManager() : m_site_packages(get_user_packages_dir()) {}

    PackageManager& PackageManager::instance() {
        static PackageManager instance;
        return instance;
    }

    std::filesystem::path PackageManager::uv_path() const {
        // Check relative to executable (portable deployment)
        auto exe_dir = get_executable_dir();
        auto relative_uv = exe_dir / "bin" / UV_BINARY_NAME;
        if (std::filesystem::exists(relative_uv)) {
            return relative_uv;
        }

        // Check in exe directory directly
        auto direct_uv = exe_dir / UV_BINARY_NAME;
        if (std::filesystem::exists(direct_uv)) {
            return direct_uv;
        }

        // Check PATH (for development)
#ifdef _WIN32
        auto [exit_code, result] = execute_command("where uv");
#else
        auto [exit_code, result] = execute_command("which uv");
#endif

        if (exit_code == 0 && !result.empty()) {
            // Remove trailing newline
            while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
                result.pop_back();
            }
            std::filesystem::path found_path(result);
            if (std::filesystem::exists(found_path)) {
                return found_path;
            }
        }

        return {};
    }

    bool PackageManager::is_uv_available() const {
        return !uv_path().empty();
    }

    std::filesystem::path PackageManager::site_packages_dir() const {
        return m_site_packages;
    }

    InstallResult PackageManager::execute_uv(const std::vector<std::string>& args) const {
        InstallResult result;

        auto uv = uv_path();
        if (uv.empty()) {
            result.error = "uv package manager not found";
            return result;
        }

        // Build command string
        std::ostringstream cmd;
        cmd << "\"" << uv.string() << "\"";
        for (const auto& arg : args) {
            cmd << " " << arg;
        }

        LOG_INFO("Executing: {}", cmd.str());

        auto [exit_code, output] = execute_command(cmd.str());

        result.output = output;
        result.success = (exit_code == 0);
        if (!result.success) {
            result.error = output.empty() ? "Command failed with exit code " + std::to_string(exit_code) : output;
        }

        return result;
    }

    InstallResult PackageManager::install(const std::string& package) {
        std::lock_guard lock(m_mutex);

        if (!std::filesystem::exists(m_site_packages)) {
            std::error_code ec;
            std::filesystem::create_directories(m_site_packages, ec);
            if (ec) {
                InstallResult result;
                result.error = "Failed to create site-packages directory: " + ec.message();
                return result;
            }
        }

        LOG_INFO("Installing package: {} to {}", package, m_site_packages.string());

        return execute_uv({"pip", "install", package, "--target", m_site_packages.string(), "--quiet"});
    }

    InstallResult PackageManager::uninstall(const std::string& package) {
        std::lock_guard lock(m_mutex);

        LOG_INFO("Uninstalling package: {}", package);

        // uv pip uninstall doesn't support --target, so we need to set PYTHONUSERBASE
        // or just delete the package directory directly

        // Find package directory
        std::filesystem::path pkg_dir = m_site_packages / package;
        std::filesystem::path pkg_info = m_site_packages / (package + ".dist-info");

        // Also check for normalized names (e.g., "my_package" vs "my-package")
        std::string normalized = package;
        std::replace(normalized.begin(), normalized.end(), '-', '_');

        if (!std::filesystem::exists(pkg_dir)) {
            pkg_dir = m_site_packages / normalized;
        }
        if (!std::filesystem::exists(pkg_info)) {
            // Search for any matching dist-info
            for (const auto& entry : std::filesystem::directory_iterator(m_site_packages)) {
                if (entry.is_directory()) {
                    std::string name = entry.path().filename().string();
                    if (name.find(".dist-info") != std::string::npos) {
                        // Extract package name from dist-info
                        std::string info_pkg = name.substr(0, name.find('-'));
                        std::string info_pkg_normalized = info_pkg;
                        std::replace(info_pkg_normalized.begin(), info_pkg_normalized.end(), '_', '-');

                        if (info_pkg == package || info_pkg_normalized == package || info_pkg == normalized) {
                            pkg_info = entry.path();
                            break;
                        }
                    }
                }
            }
        }

        InstallResult result;
        std::error_code ec;

        // Remove package directory
        if (std::filesystem::exists(pkg_dir)) {
            std::filesystem::remove_all(pkg_dir, ec);
            if (ec) {
                result.error = "Failed to remove package directory: " + ec.message();
                return result;
            }
        }

        // Remove dist-info
        if (std::filesystem::exists(pkg_info)) {
            std::filesystem::remove_all(pkg_info, ec);
            if (ec) {
                result.error = "Failed to remove dist-info: " + ec.message();
                return result;
            }
        }

        result.success = true;
        result.output = "Uninstalled " + package;
        return result;
    }

    std::vector<PackageInfo> PackageManager::list_installed() const {
        std::lock_guard lock(m_mutex);
        std::vector<PackageInfo> packages;

        if (!std::filesystem::exists(m_site_packages)) {
            return packages;
        }

        // Look for *.dist-info directories
        for (const auto& entry : std::filesystem::directory_iterator(m_site_packages)) {
            if (!entry.is_directory()) {
                continue;
            }

            std::string name = entry.path().filename().string();
            if (name.find(".dist-info") == std::string::npos) {
                continue;
            }

            // Parse name-version.dist-info
            std::regex pattern(R"((.+)-(.+)\.dist-info)");
            std::smatch match;
            if (std::regex_match(name, match, pattern)) {
                PackageInfo info;
                info.name = match[1].str();
                info.version = match[2].str();
                packages.push_back(info);
            }
        }

        return packages;
    }

    bool PackageManager::is_installed(const std::string& package) const {
        auto packages = list_installed();
        std::string normalized = package;
        std::replace(normalized.begin(), normalized.end(), '-', '_');

        for (const auto& pkg : packages) {
            std::string pkg_normalized = pkg.name;
            std::replace(pkg_normalized.begin(), pkg_normalized.end(), '-', '_');

            if (pkg.name == package || pkg_normalized == normalized) {
                return true;
            }
        }
        return false;
    }

} // namespace lfs::python
