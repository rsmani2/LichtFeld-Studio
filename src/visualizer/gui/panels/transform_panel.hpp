/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/panels/gizmo_toolbar.hpp"
#include "gui/ui_context.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <string>

namespace lfs::vis::gui::panels {

    struct TransformPanelState {
        bool editing_active = false;
        std::string editing_node_name;
        glm::mat4 transform_before_edit{1.0f};

        glm::vec3 initial_translation{0.0f};
        glm::vec3 initial_scale{1.0f};

        // Euler display state - avoids gimbal lock from constant decomposition
        glm::vec3 euler_display{0.0f};
        std::string euler_display_node;
        glm::quat euler_display_rotation{1.0f, 0.0f, 0.0f, 0.0f};
    };

    void DrawTransformControls(const UIContext& ctx, ToolType current_tool,
                               TransformSpace transform_space, TransformPanelState& state);

} // namespace lfs::vis::gui::panels
