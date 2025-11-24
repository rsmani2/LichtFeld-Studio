/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gl_resources.hpp"
#include "rendering_new/rendering.hpp"
#include "shader_manager.hpp"
#include <glm/glm.hpp>

namespace lfs::rendering {
    class RenderPivotPoint {
    public:
        RenderPivotPoint() = default;
        ~RenderPivotPoint() = default;

        Result<void> init();
        [[nodiscard]] bool isInitialized() const { return initialized_; }

        void setPosition(const glm::vec3& position) { pivot_position_ = position; }
        void setSize(float size) { sphere_size_ = size; }

        Result<void> render(const glm::mat4& view, const glm::mat4& projection);

    private:
        GLuint shader_program_{0};
        VAO vao_;
        VBO vbo_;

        glm::vec3 pivot_position_{0.0f, 0.0f, 0.0f};
        float sphere_size_{0.15f};
        int vertex_count_{0};
        bool initialized_{false};
    };
} // namespace lfs::rendering
