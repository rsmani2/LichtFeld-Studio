/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <functional>

namespace lfs::vis {
    class Scene;
}

namespace lfs::python {

    // Panel space types
    enum class PanelSpace {
        SidePanel,
        Floating,
        ViewportOverlay
    };

    // Callback types for the Python panel system
    using DrawPanelsCallback = std::function<void(PanelSpace)>;
    using HasPanelsCallback = std::function<bool(PanelSpace)>;
    using CleanupCallback = std::function<void()>;

    // Register callbacks from the Python module
    void set_panel_draw_callback(DrawPanelsCallback cb);
    void set_panel_has_callback(HasPanelsCallback cb);
    void set_python_cleanup_callback(CleanupCallback cb);
    void clear_panel_callbacks();

    // C++ interface for the visualizer
    void draw_python_panels(PanelSpace space, lfs::vis::Scene* scene = nullptr);
    bool has_python_panels(PanelSpace space);
    void invoke_python_cleanup();

} // namespace lfs::python
