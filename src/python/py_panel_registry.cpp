/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_panel_registry.hpp"
#include "core/logger.hpp"
#include <Python.h>

namespace lfs::python {

    namespace {
        DrawPanelsCallback g_draw_callback;
        HasPanelsCallback g_has_callback;
        CleanupCallback g_cleanup_callback;

        void set_scene_context_internal(lfs::vis::Scene* scene) {
            if (!scene)
                return;
            if (!Py_IsInitialized())
                return;

            PyGILState_STATE gil = PyGILState_Ensure();
            PyObject* lf_module = PyImport_ImportModule("lichtfeld");
            if (lf_module) {
                PyObject* set_ctx = PyObject_GetAttrString(lf_module, "_set_scene_context");
                if (set_ctx && PyCallable_Check(set_ctx)) {
                    PyObject* capsule = PyCapsule_New(scene, nullptr, nullptr);
                    if (capsule) {
                        PyObject* args = PyTuple_Pack(1, capsule);
                        PyObject_Call(set_ctx, args, nullptr);
                        Py_DECREF(args);
                        Py_DECREF(capsule);
                    }
                }
                Py_XDECREF(set_ctx);
                Py_DECREF(lf_module);
            }
            PyGILState_Release(gil);
        }

        void clear_scene_context_internal() {
            if (!Py_IsInitialized())
                return;

            PyGILState_STATE gil = PyGILState_Ensure();
            PyObject* lf_module = PyImport_ImportModule("lichtfeld");
            if (lf_module) {
                PyObject* clear_ctx = PyObject_GetAttrString(lf_module, "_clear_scene_context");
                if (clear_ctx && PyCallable_Check(clear_ctx)) {
                    PyObject_CallNoArgs(clear_ctx);
                }
                Py_XDECREF(clear_ctx);
                Py_DECREF(lf_module);
            }
            PyGILState_Release(gil);
        }
    } // namespace

    void set_panel_draw_callback(DrawPanelsCallback cb) {
        g_draw_callback = std::move(cb);
        LOG_INFO("Python panel draw callback registered");
    }

    void set_panel_has_callback(HasPanelsCallback cb) {
        g_has_callback = std::move(cb);
    }

    void set_python_cleanup_callback(CleanupCallback cb) {
        g_cleanup_callback = std::move(cb);
    }

    void clear_panel_callbacks() {
        g_draw_callback = nullptr;
        g_has_callback = nullptr;
        g_cleanup_callback = nullptr;
    }

    void invoke_python_cleanup() {
        if (g_cleanup_callback) {
            g_cleanup_callback();
        }
    }

    void draw_python_panels(PanelSpace space, lfs::vis::Scene* scene) {
        if (g_draw_callback) {
            if (scene) {
                set_scene_context_internal(scene);
            }
            g_draw_callback(space);
            if (scene) {
                clear_scene_context_internal();
            }
        } else {
            // Callback not set - Python module not loaded yet
            static bool warned = false;
            if (!warned) {
                LOG_DEBUG("Python panel draw callback not set (Python module not loaded?)");
                warned = true;
            }
        }
    }

    bool has_python_panels(PanelSpace space) {
        if (g_has_callback) {
            return g_has_callback(space);
        }
        return false;
    }

} // namespace lfs::python
