/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_packages.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "../package_manager.hpp"

namespace lfs::python {

    void register_packages(nb::module_& m) {
        auto pkg = m.def_submodule("packages", "Package management via uv");

        // PackageInfo struct
        nb::class_<PackageInfo>(pkg, "PackageInfo")
            .def_ro("name", &PackageInfo::name, "Package name")
            .def_ro("version", &PackageInfo::version, "Package version")
            .def("__repr__", [](const PackageInfo& p) { return p.name + "==" + p.version; });

        // install()
        pkg.def(
            "install",
            [](const std::string& package) {
                auto& pm = PackageManager::instance();
                if (!pm.is_uv_available()) {
                    throw std::runtime_error("uv package manager not found");
                }
                auto result = pm.install(package);
                if (!result.success) {
                    throw std::runtime_error(result.error);
                }
                return result.output;
            },
            nb::arg("package"),
            R"doc(Install a package from PyPI.

Args:
    package: Package name with optional version specifier (e.g., "numpy", "scipy>=1.10")

Returns:
    Installation output

Raises:
    RuntimeError: If uv is not available or installation fails

Example:
    >>> lf.packages.install("numpy")
    >>> lf.packages.install("scipy>=1.10")
)doc");

        // uninstall()
        pkg.def(
            "uninstall",
            [](const std::string& package) {
                auto& pm = PackageManager::instance();
                auto result = pm.uninstall(package);
                if (!result.success) {
                    throw std::runtime_error(result.error);
                }
                return result.output;
            },
            nb::arg("package"),
            R"doc(Uninstall a package.

Args:
    package: Package name to uninstall

Returns:
    Uninstallation output

Raises:
    RuntimeError: If uninstallation fails
)doc");

        // list()
        pkg.def(
            "list",
            []() {
                auto& pm = PackageManager::instance();
                return pm.list_installed();
            },
            R"doc(List installed packages.

Returns:
    List of PackageInfo objects with name and version

Example:
    >>> for pkg in lf.packages.list():
    ...     print(f"{pkg.name} {pkg.version}")
)doc");

        // is_installed()
        pkg.def(
            "is_installed",
            [](const std::string& package) {
                auto& pm = PackageManager::instance();
                return pm.is_installed(package);
            },
            nb::arg("package"),
            R"doc(Check if a package is installed.

Args:
    package: Package name to check

Returns:
    True if package is installed, False otherwise
)doc");

        // is_uv_available()
        pkg.def(
            "is_uv_available",
            []() {
                auto& pm = PackageManager::instance();
                return pm.is_uv_available();
            },
            R"doc(Check if uv package manager is available.

Returns:
    True if uv is found and can be executed
)doc");

        // site_packages_dir()
        pkg.def(
            "site_packages_dir",
            []() {
                auto& pm = PackageManager::instance();
                return pm.site_packages_dir().string();
            },
            R"doc(Get the user site-packages directory.

Returns:
    Path to ~/.lichtfeld/site-packages where packages are installed
)doc");
    }

} // namespace lfs::python
