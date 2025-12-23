/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "runner.hpp"

#include <filesystem>
#include <format>
#include <string>

#include <core/logger.hpp>

#ifdef LFS_BUILD_PYTHON_BINDINGS
#include <Python.h>
#endif

namespace lfs::python {

    std::expected<void, std::string> run_scripts(const std::vector<std::filesystem::path>& scripts) {
        if (scripts.empty()) {
            return {};
        }

#ifndef LFS_BUILD_PYTHON_BINDINGS
        return std::unexpected("Python bindings disabled; rebuild with -DBUILD_PYTHON_BINDINGS=ON");
#else
        if (!Py_IsInitialized()) {
            Py_Initialize();
            PyEval_InitThreads();
        }

        PyGILState_STATE gil_state = PyGILState_Ensure();

        // Add build directory (where lichtfeld.so lives) to sys.path
        {
            const std::filesystem::path build_python_dir =
                std::filesystem::path(PROJECT_ROOT_PATH) / "build" / "src" / "python";

            PyObject* sys_path = PySys_GetObject("path"); // borrowed
            PyObject* py_path = PyUnicode_FromString(build_python_dir.string().c_str());
            PyList_Append(sys_path, py_path);
            Py_DECREF(py_path);
        }


        for (const auto& script : scripts) {
            if (!std::filesystem::exists(script)) {
                PyGILState_Release(gil_state);
                return std::unexpected(std::format("Python script not found: {}", script.string()));
            }

            // Ensure script directory is on sys.path
            const auto parent = script.parent_path().string();
            if (!parent.empty()) {
                PyObject* sys_path = PySys_GetObject("path"); // borrowed ref
                PyObject* py_parent = PyUnicode_FromString(parent.c_str());
                if (sys_path && py_parent) {
                    PyList_Append(sys_path, py_parent);
                }
                Py_XDECREF(py_parent);
            }

            FILE* fp = fopen(script.string().c_str(), "r");
            if (!fp) {
                PyGILState_Release(gil_state);
                return std::unexpected(std::format("Failed to open Python script: {}", script.string()));
            }

            LOG_INFO("Executing Python script: {}", script.string());
            const int rc = PyRun_SimpleFileEx(fp, script.string().c_str(), /*closeit=*/1);
            if (rc != 0) {
                PyGILState_Release(gil_state);
                return std::unexpected(std::format("Python script failed: {} (rc={})", script.string(), rc));
            }
        }

        PyGILState_Release(gil_state);
        return {};
#endif
    }

} // namespace lfs::python
