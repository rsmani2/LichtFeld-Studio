/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "pivot_renderer.hpp"
#include "core_new/logger.hpp"
#include "gl_state_guard.hpp"
#include "shader_manager.hpp"
#include <glad/glad.h>

namespace lfs::rendering {

    Result<void> RenderPivotPoint::init() {
        if (initialized_) {
            return {};
        }

        // Create simple flat red shader inline
        const char* vert_src = R"(
            #version 330 core
            layout (location = 0) in vec3 position;
            uniform mat4 u_mvp;
            void main() {
                gl_Position = u_mvp * vec4(position, 1.0);
            }
        )";

        const char* frag_src = R"(
            #version 330 core
            out vec4 FragColor;
            void main() {
                FragColor = vec4(1.0, 0.0, 0.0, 1.0);
            }
        )";

        GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vert_shader, 1, &vert_src, nullptr);
        glCompileShader(vert_shader);

        GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(frag_shader, 1, &frag_src, nullptr);
        glCompileShader(frag_shader);

        GLuint program = glCreateProgram();
        glAttachShader(program, vert_shader);
        glAttachShader(program, frag_shader);
        glLinkProgram(program);

        glDeleteShader(vert_shader);
        glDeleteShader(frag_shader);

        shader_program_ = program;

        // Create VAO/VBO for single point
        auto vao_result = create_vao();
        if (!vao_result) {
            return std::unexpected(vao_result.error());
        }

        auto vbo_result = create_vbo();
        if (!vbo_result) {
            return std::unexpected(vbo_result.error());
        }
        vbo_ = std::move(*vbo_result);

        VAOBuilder builder(std::move(*vao_result));
        builder.attachVBO(vbo_)
            .setAttribute({.index = 0, .size = 3, .type = GL_FLOAT,
                           .normalized = GL_FALSE, .stride = 0, .offset = nullptr, .divisor = 0});
        vao_ = builder.build();

        initialized_ = true;
        LOG_DEBUG("Pivot renderer initialized");
        return {};
    }

    Result<void> RenderPivotPoint::render(const glm::mat4& view, const glm::mat4& projection) {
        if (!initialized_ || shader_program_ == 0) {
            return std::unexpected("Pivot renderer not initialized");
        }

        // Upload pivot position
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3), &pivot_position_, GL_DYNAMIC_DRAW);

        // Use shader and set uniform
        glUseProgram(shader_program_);
        glm::mat4 mvp = projection * view;
        GLint mvp_loc = glGetUniformLocation(shader_program_, "u_mvp");
        glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, &mvp[0][0]);

        // Draw big red point
        glPointSize(30.0f);  // Big point
        glEnable(GL_POINT_SMOOTH);  // Smooth round point
        VAOBinder vao_bind(vao_);
        glDrawArrays(GL_POINTS, 0, 1);

        return {};
    }

} // namespace lfs::rendering
