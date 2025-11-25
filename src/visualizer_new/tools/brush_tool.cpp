/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tools/brush_tool.hpp"
#include "internal/viewport.hpp"
#include "scene/scene_manager.hpp"
#include "core_new/tensor.hpp"
#include <imgui.h>

namespace lfs::vis::tools {

    BrushTool::BrushTool() = default;

    bool BrushTool::initialize(const ToolContext& ctx) {
        tool_context_ = &ctx;
        return true;
    }

    void BrushTool::shutdown() {
        tool_context_ = nullptr;
        stroke_points_.clear();
        is_painting_ = false;
    }

    void BrushTool::update([[maybe_unused]] const ToolContext& ctx) {
        if (isEnabled()) {
            double mx, my;
            glfwGetCursorPos(ctx.getWindow(), &mx, &my);
            last_mouse_pos_ = glm::vec2(static_cast<float>(mx), static_cast<float>(my));
        }
    }

    void BrushTool::renderUI([[maybe_unused]] const lfs::vis::gui::UIContext& ui_ctx,
                              [[maybe_unused]] bool* p_open) {
        if (!isEnabled()) return;

        // Hide brush cursor when hovering over ImGui panels
        if (ImGui::GetIO().WantCaptureMouse) return;

        ImDrawList* const draw_list = ImGui::GetForegroundDrawList();
        // Use stored mouse position from GLFW (same source as selection coordinates)
        // to ensure the brush circle matches exactly where selection happens
        const ImVec2 mouse_pos(last_mouse_pos_.x, last_mouse_pos_.y);

        ImU32 brush_color = IM_COL32(0, 200, 255, 180);
        switch (mode_) {
            case BrushMode::Select:   brush_color = IM_COL32(0, 200, 255, 180); break;
            case BrushMode::Deselect: brush_color = IM_COL32(255, 200, 0, 180); break;
            case BrushMode::Delete:   brush_color = IM_COL32(255, 50, 50, 180); break;
        }

        draw_list->AddCircle(mouse_pos, brush_radius_, brush_color, 32, 2.0f);
        draw_list->AddCircleFilled(mouse_pos, 3.0f, brush_color);

        const char* mode_text = nullptr;
        switch (mode_) {
            case BrushMode::Select:   mode_text = "Select"; break;
            case BrushMode::Deselect: mode_text = "Deselect"; break;
            case BrushMode::Delete:   mode_text = "Delete"; break;
        }
        if (mode_text) {
            draw_list->AddText(ImVec2(mouse_pos.x + brush_radius_ + 5, mouse_pos.y - 10),
                              brush_color, mode_text);
        }
    }

    bool BrushTool::handleMouseButton(int button, int action, double x, double y,
                                       [[maybe_unused]] const ToolContext& ctx) {
        if (!isEnabled()) return false;

        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            if (action == GLFW_PRESS) {
                beginStroke(x, y, ctx);
                updateSelectionAtPoint(x, y, ctx);
                return true;
            } else if (action == GLFW_RELEASE && is_painting_) {
                endStroke();
                return true;
            }
        }
        return false;
    }

    bool BrushTool::handleMouseMove(double x, double y, [[maybe_unused]] const ToolContext& ctx) {
        if (!isEnabled()) return false;

        last_mouse_pos_ = glm::vec2(static_cast<float>(x), static_cast<float>(y));

        if (is_painting_) {
            continueStroke(x, y);
            updateSelectionAtPoint(x, y, ctx);
            ctx.requestRender();
            return true;
        }
        return false;
    }

    bool BrushTool::handleScroll([[maybe_unused]] double x_offset, double y_offset,
                                  [[maybe_unused]] const ToolContext& ctx) {
        if (!isEnabled()) return false;

        bool should_resize = is_painting_ ||
                             glfwGetKey(ctx.getWindow(), GLFW_KEY_LEFT_ALT) == GLFW_PRESS ||
                             glfwGetKey(ctx.getWindow(), GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;

        if (should_resize) {
            float scale_factor = (y_offset > 0) ? 1.1f : 0.9f;
            brush_radius_ = std::clamp(brush_radius_ * scale_factor, 1.0f, 500.0f);
            return true;
        }
        return false;
    }

    void BrushTool::onEnabledChanged(bool enabled) {
        if (!enabled) {
            is_painting_ = false;
            stroke_points_.clear();
        }

        // Enable/disable screen positions output in the rasterizer
        if (tool_context_) {
            auto* rendering_manager = tool_context_->getRenderingManager();
            if (rendering_manager) {
                rendering_manager->setOutputScreenPositions(enabled);
                if (enabled) {
                    rendering_manager->markDirty();  // Request re-render to get screen positions
                }
            }
        }
    }

    void BrushTool::beginStroke(double x, double y, const ToolContext& ctx) {
        is_painting_ = true;
        stroke_points_.clear();
        stroke_points_.push_back(glm::vec2(static_cast<float>(x), static_cast<float>(y)));

        auto* scene_manager = ctx.getSceneManager();
        if (scene_manager) {
            // Clear scene selection at start of stroke
            scene_manager->getScene().clearSelection();

            // Initialize selection tensor - zeros, will accumulate during stroke
            size_t num_gaussians = scene_manager->getScene().getTotalGaussianCount();
            if (num_gaussians > 0) {
                cumulative_selection_ = lfs::core::Tensor::zeros(
                    {num_gaussians}, lfs::core::Device::CUDA, lfs::core::DataType::Bool);
            }
        }
    }

    void BrushTool::continueStroke(double x, double y) {
        if (!is_painting_) return;

        glm::vec2 new_point(static_cast<float>(x), static_cast<float>(y));
        if (stroke_points_.empty() ||
            glm::length(new_point - stroke_points_.back()) > brush_radius_ * 0.1f) {
            stroke_points_.push_back(new_point);
        }
    }

    void BrushTool::endStroke() {
        is_painting_ = false;
        stroke_points_.clear();

        // Clear brush state on rendering manager
        if (tool_context_) {
            auto* rendering_manager = tool_context_->getRenderingManager();
            if (rendering_manager) {
                rendering_manager->clearBrushState();
            }
        }

        // Clear selection tensor on mouse release
        cumulative_selection_ = lfs::core::Tensor();
    }

    void BrushTool::updateSelectionAtPoint(double x, double y, const ToolContext& ctx) {
        auto* rendering_manager = ctx.getRenderingManager();
        if (!rendering_manager) return;

        // Convert mouse position from window coords to image coords
        const auto& bounds = ctx.getViewportBounds();
        const auto& viewport = ctx.getViewport();

        // Get actual render size from cached result
        const auto& cached = rendering_manager->getCachedResult();
        int render_w = cached.image ? static_cast<int>(cached.image->size(2)) : viewport.windowSize.x;
        int render_h = cached.image ? static_cast<int>(cached.image->size(1)) : viewport.windowSize.y;

        // Mouse relative to bounds
        float rel_x = static_cast<float>(x) - bounds.x;
        float rel_y = static_cast<float>(y) - bounds.y;

        // Scale from display coords to actual render coords
        float scale_x = static_cast<float>(render_w) / bounds.width;
        float scale_y = static_cast<float>(render_h) / bounds.height;

        float image_x = rel_x * scale_x;
        float image_y = rel_y * scale_y;
        float scaled_radius = brush_radius_ * scale_x;

        // Set brush state on rendering manager with our cumulative selection tensor
        // The kernel will accumulate (OR) selections into this tensor during preprocess
        rendering_manager->setBrushState(true, image_x, image_y, scaled_radius, &cumulative_selection_);
    }

} // namespace lfs::vis::tools
