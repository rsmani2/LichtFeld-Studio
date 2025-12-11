/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/ui_widgets.hpp"
#include "scene/scene_manager.hpp"
#include "theme/theme.hpp"
#include "training/training_manager.hpp"
#include "visualizer_impl.hpp"
#include <cstdarg>
#include <imgui.h>

namespace lfs::vis::gui::widgets {

    using namespace lfs::core::events;

    bool SliderWithReset(const char* label, float* v, float min, float max, float reset_value) {
        bool changed = ImGui::SliderFloat(label, v, min, max);

        ImGui::SameLine();
        ImGui::PushID(label);
        if (ImGui::Button("Reset")) {
            *v = reset_value;
            changed = true;
        }
        ImGui::PopID();

        return changed;
    }

    bool DragFloat3WithReset(const char* label, float* v, float speed, float reset_value) {
        bool changed = ImGui::DragFloat3(label, v, speed);

        ImGui::SameLine();
        ImGui::PushID(label);
        if (ImGui::Button("Reset")) {
            v[0] = v[1] = v[2] = reset_value;
            changed = true;
        }
        ImGui::PopID();

        return changed;
    }

    void HelpMarker(const char* desc) {
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
            ImGui::TextUnformatted(desc);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }

    void TableRow(const char* label, const char* format, ...) {
        ImGui::Text("%s:", label);
        ImGui::SameLine(120); // Align values at column 120

        va_list args;
        va_start(args, format);
        ImGui::TextV(format, args);
        va_end(args);
    }

    void DrawProgressBar(float fraction, const char* overlay_text) {
        ImGui::ProgressBar(fraction, ImVec2(-1, 0), overlay_text);
    }

    void DrawLossPlot(const float* values, int count, float min_val, float max_val, const char* label) {
        if (count <= 0)
            return;

        // Ensure we have a valid, non-empty label
        const char* plot_label = (label && strlen(label) > 0) ? label : "Plot##default";

        // Simple line plot using ImGui
        ImGui::PlotLines(
            "Plot##default",
            values,
            count,
            0,
            plot_label,
            min_val,
            max_val,
            ImVec2(ImGui::GetContentRegionAvail().x, 80));
    }

    void DrawModeStatus(const UIContext& ctx) {
        auto* scene_manager = ctx.viewer->getSceneManager();
        if (!scene_manager) {
            ImGui::Text("Mode: Unknown");
            return;
        }

        const auto& t = theme();
        const char* mode_str = "Unknown";
        ImVec4 mode_color = t.palette.text_dim;

        // Content determines base mode
        SceneManager::ContentType content = scene_manager->getContentType();

        switch (content) {
        case SceneManager::ContentType::Empty:
            mode_str = "Empty";
            mode_color = t.palette.text_dim;
            break;

        case SceneManager::ContentType::SplatFiles:
            mode_str = "Splat Viewer";
            mode_color = t.palette.info;
            break;

        case SceneManager::ContentType::Dataset: {
            // For dataset, check training state from TrainerManager
            auto* trainer_manager = scene_manager->getTrainerManager();
            if (!trainer_manager || !trainer_manager->hasTrainer()) {
                mode_str = "Dataset (No Trainer)";
                mode_color = t.palette.text_dim;
            } else {
                // Use trainer state for specific mode
                auto state = trainer_manager->getState();
                switch (state) {
                case TrainerManager::State::Ready:
                    mode_str = "Dataset (Ready)";
                    mode_color = t.palette.success;
                    break;
                case TrainerManager::State::Running:
                    mode_str = "Training";
                    mode_color = t.palette.warning;
                    break;
                case TrainerManager::State::Paused:
                    mode_str = "Training (Paused)";
                    mode_color = lighten(t.palette.warning, -0.3f);
                    break;
                case TrainerManager::State::Completed:
                    mode_str = "Training Complete";
                    mode_color = t.palette.success;
                    break;
                case TrainerManager::State::Error:
                    mode_str = "Training Error";
                    mode_color = t.palette.error;
                    break;
                case TrainerManager::State::Stopping:
                    mode_str = "Stopping...";
                    mode_color = darken(t.palette.error, 0.3f);
                    break;
                default:
                    mode_str = "Dataset";
                    mode_color = t.palette.text_dim;
                }
            }
            break;
        }
        }

        ImGui::TextColored(mode_color, "Mode: %s", mode_str);

        // Display scene info
        auto info = scene_manager->getSceneInfo();
        if (info.num_gaussians > 0) {
            ImGui::Text("Gaussians: %zu", info.num_gaussians);
        }

        if (info.source_type == "PLY" && info.num_nodes > 0) {
            ImGui::Text("PLY Models: %zu", info.num_nodes);
        }

        // Display training iteration if actively training
        if (content == SceneManager::ContentType::Dataset) {
            auto* trainer_manager = scene_manager->getTrainerManager();
            if (trainer_manager && trainer_manager->isRunning()) {
                int iteration = trainer_manager->getCurrentIteration();
                if (iteration > 0) {
                    ImGui::Text("Iteration: %d", iteration);
                }
            }
        }
    }

    void DrawModeStatusWithContentSwitch(const UIContext& ctx) {
        DrawModeStatus(ctx);
    }

    void DrawWindowShadow(const ImVec2& pos, const ImVec2& size, const float rounding) {
        const auto& t = theme();
        if (!t.shadows.enabled) return;

        constexpr int LAYER_COUNT = 16;
        constexpr float FALLOFF_SCALE = 0.12f;
        constexpr float ROUNDING_SCALE = 0.3f;

        auto* const draw_list = ImGui::GetBackgroundDrawList();
        const ImVec2& off = t.shadows.offset;
        const float blur = t.shadows.blur;
        const float base_alpha = t.shadows.alpha * 255.0f;

        for (int i = 0; i < LAYER_COUNT; ++i) {
            const float t_val = static_cast<float>(i) / (LAYER_COUNT - 1);
            const float inv_t = 1.0f - t_val;
            const float falloff = inv_t * inv_t * inv_t;
            const int alpha = static_cast<int>(base_alpha * falloff * FALLOFF_SCALE);
            if (alpha < 1) continue;

            const float expand = blur * t_val;
            const ImVec2 p1 = {pos.x + off.x - expand, pos.y + off.y - expand};
            const ImVec2 p2 = {pos.x + size.x + off.x + expand, pos.y + size.y + off.y + expand};
            draw_list->AddRectFilled(p1, p2, IM_COL32(0, 0, 0, alpha), rounding + expand * ROUNDING_SCALE);
        }
    }

    void DrawViewportVignette(const ImVec2& pos, const ImVec2& size) {
        const auto& t = theme();
        if (!t.vignette.enabled) return;

        constexpr float EDGE_SCALE = 0.5f;
        constexpr ImU32 TRANSPARENT = IM_COL32(0, 0, 0, 0);

        auto* const draw_list = ImGui::GetForegroundDrawList();
        const float edge_mult = (1.0f - t.vignette.radius) * EDGE_SCALE * (1.0f + t.vignette.softness);
        const float edge_w = size.x * edge_mult;
        const float edge_h = size.y * edge_mult;
        const ImU32 dark = IM_COL32(0, 0, 0, static_cast<int>(t.vignette.intensity * 255.0f));

        const float x1 = pos.x, y1 = pos.y;
        const float x2 = pos.x + size.x, y2 = pos.y + size.y;

        draw_list->AddRectFilledMultiColor({x1, y1}, {x1 + edge_w, y2}, dark, TRANSPARENT, TRANSPARENT, dark);
        draw_list->AddRectFilledMultiColor({x2 - edge_w, y1}, {x2, y2}, TRANSPARENT, dark, dark, TRANSPARENT);
        draw_list->AddRectFilledMultiColor({x1, y1}, {x2, y1 + edge_h}, dark, dark, TRANSPARENT, TRANSPARENT);
        draw_list->AddRectFilledMultiColor({x1, y2 - edge_h}, {x2, y2}, TRANSPARENT, TRANSPARENT, dark, dark);
    }

} // namespace lfs::vis::gui::widgets