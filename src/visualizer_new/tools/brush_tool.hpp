/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "tool_base.hpp"
#include "core_new/tensor.hpp"
#include <glm/glm.hpp>
#include <vector>

namespace lfs::vis::tools {

    class BrushTool : public ToolBase {
    public:
        enum class BrushMode { Select, Deselect, Delete };

        BrushTool();
        ~BrushTool() override = default;

        std::string_view getName() const override { return "Brush Tool"; }
        std::string_view getDescription() const override { return "Paint to select Gaussians"; }

        bool initialize(const ToolContext& ctx) override;
        void shutdown() override;
        void update(const ToolContext& ctx) override;
        void renderUI(const lfs::vis::gui::UIContext& ui_ctx, bool* p_open) override;

        bool isActive() const { return is_painting_; }
        bool handleMouseButton(int button, int action, double x, double y, const ToolContext& ctx);
        bool handleMouseMove(double x, double y, const ToolContext& ctx);
        bool handleScroll(double x_offset, double y_offset, const ToolContext& ctx);

        float getBrushRadius() const { return brush_radius_; }
        void setBrushRadius(float radius) { brush_radius_ = std::clamp(radius, 1.0f, 500.0f); }
        BrushMode getMode() const { return mode_; }
        void setMode(BrushMode mode) { mode_ = mode; }

    protected:
        void onEnabledChanged(bool enabled) override;

    private:
        BrushMode mode_ = BrushMode::Select;
        float brush_radius_ = 20.0f;
        bool is_painting_ = false;
        std::vector<glm::vec2> stroke_points_;
        glm::vec2 last_mouse_pos_{0.0f};
        const ToolContext* tool_context_ = nullptr;

        // Cumulative selection tensor - accumulates selections during a stroke
        lfs::core::Tensor cumulative_selection_;

        void beginStroke(double x, double y, const ToolContext& ctx);
        void continueStroke(double x, double y);
        void endStroke();
        void updateSelectionAtPoint(double x, double y, const ToolContext& ctx);
    };

} // namespace lfs::vis::tools
