/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tools/translation_gizmo_tool.hpp"
#include "core_new/events.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <print>

namespace lfs::vis::tools {

    using namespace lfs::core::events;

    TranslationGizmoTool::TranslationGizmoTool() {
        // Initialize with identity transform
        current_transform_ = lfs::geometry::EuclideanTransform();
    }

    bool TranslationGizmoTool::initialize(const ToolContext& ctx) {
        // Store context for later use
        tool_context_ = &ctx;

        auto* render_manager = ctx.getRenderingManager();
        if (!render_manager) {
            return false;
        }

        auto* engine = render_manager->getRenderingEngine();
        if (!engine) {
            return false;
        }

        // Get the gizmo interaction interface
        gizmo_interaction_ = engine->getGizmoInteraction();
        if (!gizmo_interaction_) {
            std::println("Failed to get gizmo interaction interface");
            return false;
        }

        // Initialize transform based on mode
        if (apply_to_world_) {
            auto settings = render_manager->getSettings();
            current_transform_ = settings.world_transform;
        } else {
            // Sync from selected node
            syncFromSelectedNode(ctx);
        }

        // Note: Node transform gizmo is now handled by ImGuizmo in gui_manager.cpp
        // This old custom gizmo tool is deprecated for node transforms.
        // The event listener below is disabled to prevent interference.
        //
        // ui::NodeSelected::when([this](const auto& event) {
        //     if (event.type == "PLY" && !apply_to_world_ && tool_context_) {
        //         if (!isEnabled()) {
        //             setEnabled(true);
        //         }
        //         syncFromSelectedNode(*tool_context_);
        //     }
        // });

        std::println("Translation Gizmo Tool initialized (mode: {})",
                     apply_to_world_ ? "world" : "node");
        return true;
    }

    void TranslationGizmoTool::shutdown() {
        gizmo_interaction_.reset();
        is_dragging_ = false;
        selected_element_ = lfs::rendering::GizmoElement::None;
        hovered_element_ = lfs::rendering::GizmoElement::None;
        tool_context_ = nullptr;
    }

    void TranslationGizmoTool::onEnabledChanged(bool enabled) {
        if (!enabled && is_dragging_) {
            // Cancel any ongoing drag
            is_dragging_ = false;
            selected_element_ = lfs::rendering::GizmoElement::None;
            if (gizmo_interaction_) {
                gizmo_interaction_->endDrag();
            }
        }
        // Gizmo rendering is now handled by ImGuizmo in gui_manager.cpp
    }

    void TranslationGizmoTool::update(const ToolContext& ctx) {
        if (!isEnabled() || !gizmo_interaction_) {
            return;
        }

        // In node mode, sync with selected node when not dragging
        if (!is_dragging_ && !apply_to_world_) {
            syncFromSelectedNode(ctx);
        }
    }

    bool TranslationGizmoTool::handleMouseButton(int button, int action, double x, double y,
                                                 const ToolContext& ctx) {
        if (!isEnabled() || !gizmo_interaction_) {
            return false;
        }

        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            if (action == GLFW_PRESS) {
                // Get matrices
                glm::mat4 view = getViewMatrix(ctx);
                glm::mat4 projection = getProjectionMatrix(ctx);
                glm::vec3 position = current_transform_.getTranslation();

                // Check for gizmo hit
                auto hit_element = gizmo_interaction_->pick(
                    glm::vec2(x, y), view, projection, position);

                if (hit_element != lfs::rendering::GizmoElement::None) {
                    // Start dragging
                    selected_element_ = hit_element;
                    is_dragging_ = true;
                    drag_start_transform_ = current_transform_;
                    drag_start_gizmo_pos_ = position;

                    // Start the drag operation
                    drag_start_position_ = gizmo_interaction_->startDrag(
                        hit_element, glm::vec2(x, y), view, projection, position);

                    std::println("Started dragging gizmo element: {}", static_cast<int>(hit_element));
                    return true; // Consume the event
                }
            } else if (action == GLFW_RELEASE && is_dragging_) {
                // End dragging
                is_dragging_ = false;
                selected_element_ = lfs::rendering::GizmoElement::None;
                gizmo_interaction_->endDrag();

                // Apply the final transform
                if (apply_to_world_) {
                    updateWorldTransform(ctx);
                } else {
                    updateNodeTransform(ctx);
                }

                std::println("Ended gizmo drag. Final position: ({:.2f}, {:.2f}, {:.2f})",
                             current_transform_.getTranslation().x,
                             current_transform_.getTranslation().y,
                             current_transform_.getTranslation().z);
                return true;
            }
        }

        return false;
    }

    bool TranslationGizmoTool::handleMouseMove(double x, double y, const ToolContext& ctx) {
        if (!isEnabled() || !gizmo_interaction_) {
            return false;
        }

        if (is_dragging_) {
            // Update drag
            glm::mat4 view = getViewMatrix(ctx);
            glm::mat4 projection = getProjectionMatrix(ctx);

            glm::vec3 new_position = gizmo_interaction_->updateDrag(
                glm::vec2(x, y), view, projection);

            // Update transform with new position
            current_transform_ = lfs::geometry::EuclideanTransform(
                current_transform_.getRotationMat(),
                new_position);

            // Update transform in real-time
            if (apply_to_world_) {
                updateWorldTransform(ctx);
            } else {
                updateNodeTransform(ctx);
            }

            return true; // Consume the event
        } else {
            // Update hover state
            glm::mat4 view = getViewMatrix(ctx);
            glm::mat4 projection = getProjectionMatrix(ctx);
            glm::vec3 position = current_transform_.getTranslation();

            auto new_hover = gizmo_interaction_->pick(
                glm::vec2(x, y), view, projection, position);

            if (new_hover != hovered_element_) {
                hovered_element_ = new_hover;
                gizmo_interaction_->setHovered(new_hover);
            }
        }

        return false;
    }

    void TranslationGizmoTool::renderUI([[maybe_unused]] const lfs::vis::gui::UIContext& ui_ctx, bool* p_open) {
        if (ImGui::Begin("Translation Gizmo", p_open)) {
            ImGui::Text("Gizmo Settings");
            ImGui::Separator();

            // Enable/disable
            bool enabled = isEnabled();
            if (ImGui::Checkbox("Enable Gizmo", &enabled)) {
                setEnabled(enabled);
            }

            if (enabled) {
                // Mode selector
                ImGui::Separator();
                ImGui::Text("Mode:");
                if (ImGui::RadioButton("World Transform", apply_to_world_)) {
                    apply_to_world_ = true;
                    if (tool_context_) {
                        auto* rm = tool_context_->getRenderingManager();
                        if (rm) {
                            auto settings = rm->getSettings();
                            current_transform_ = settings.world_transform;
                        }
                    }
                }
                ImGui::SameLine();
                if (ImGui::RadioButton("Node Transform", !apply_to_world_)) {
                    apply_to_world_ = false;
                    if (tool_context_) {
                        syncFromSelectedNode(*tool_context_);
                    }
                }

                // Show selected node info in node mode
                if (!apply_to_world_) {
                    auto* sm = tool_context_ ? tool_context_->getSceneManager() : nullptr;
                    if (sm && sm->hasSelectedNode()) {
                        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Selected: %s",
                                          sm->getSelectedNodeName().c_str());
                    } else {
                        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "No node selected");
                        ImGui::TextWrapped("Select a node in the Scene panel to transform it.");
                    }
                }

                ImGui::Separator();
                ImGui::Text("Transform");

                // Display current position
                glm::vec3 position = current_transform_.getTranslation();
                if (ImGui::DragFloat3("Position", &position.x, 0.01f)) {
                    current_transform_ = lfs::geometry::EuclideanTransform(
                        current_transform_.getRotationMat(),
                        position);
                    if (tool_context_) {
                        if (apply_to_world_) {
                            updateWorldTransform(*tool_context_);
                        } else {
                            updateNodeTransform(*tool_context_);
                        }
                    }
                }

                // Reset button
                if (ImGui::Button("Reset Transform")) {
                    current_transform_ = lfs::geometry::EuclideanTransform();
                    if (tool_context_) {
                        if (apply_to_world_) {
                            updateWorldTransform(*tool_context_);
                        } else {
                            updateNodeTransform(*tool_context_);
                        }
                    }
                }

                // Status
                ImGui::Separator();
                if (is_dragging_) {
                    ImGui::TextColored(ImVec4(0, 1, 0, 1), "Dragging: %s",
                                       selected_element_ == lfs::rendering::GizmoElement::XAxis ? "X Axis" : selected_element_ == lfs::rendering::GizmoElement::YAxis ? "Y Axis"
                                                                                                         : selected_element_ == lfs::rendering::GizmoElement::ZAxis   ? "Z Axis"
                                                                                                         : selected_element_ == lfs::rendering::GizmoElement::XYPlane ? "XY Plane"
                                                                                                         : selected_element_ == lfs::rendering::GizmoElement::XZPlane ? "XZ Plane"
                                                                                                         : selected_element_ == lfs::rendering::GizmoElement::YZPlane ? "YZ Plane"
                                                                                                                                                                      : "Unknown");
                } else if (hovered_element_ != lfs::rendering::GizmoElement::None) {
                    ImGui::Text("Hovering: %s",
                                hovered_element_ == lfs::rendering::GizmoElement::XAxis ? "X Axis" : hovered_element_ == lfs::rendering::GizmoElement::YAxis ? "Y Axis"
                                                                                                 : hovered_element_ == lfs::rendering::GizmoElement::ZAxis   ? "Z Axis"
                                                                                                 : hovered_element_ == lfs::rendering::GizmoElement::XYPlane ? "XY Plane"
                                                                                                 : hovered_element_ == lfs::rendering::GizmoElement::XZPlane ? "XZ Plane"
                                                                                                 : hovered_element_ == lfs::rendering::GizmoElement::YZPlane ? "YZ Plane"
                                                                                                                                                             : "Unknown");
                } else {
                    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1), "Ready");
                }
            }
        }
        ImGui::End();
    }

    glm::mat4 TranslationGizmoTool::getViewMatrix(const ToolContext& ctx) const {
        const auto& viewport = ctx.getViewport();
        return viewport.getViewMatrix();
    }

    glm::mat4 TranslationGizmoTool::getProjectionMatrix(const ToolContext& ctx) const {
        const auto& viewport = ctx.getViewport();
        auto* render_manager = ctx.getRenderingManager();
        float fov = render_manager ? render_manager->getSettings().fov : 60.0f;
        return viewport.getProjectionMatrix(fov);
    }

    void TranslationGizmoTool::updateWorldTransform(const ToolContext& ctx) {
        auto* render_manager = const_cast<RenderingManager*>(ctx.getRenderingManager());
        if (!render_manager) {
            return;
        }

        // Update the world transform in rendering settings
        auto settings = render_manager->getSettings();
        settings.world_transform = current_transform_;
        render_manager->updateSettings(settings);

        // Also emit event to update the world transform panel if it exists
        ui::RenderSettingsChanged{}.emit();
    }

    void TranslationGizmoTool::updateNodeTransform(const ToolContext& ctx) {
        auto* scene_manager = ctx.getSceneManager();
        if (!scene_manager) {
            return;
        }

        if (!scene_manager->hasSelectedNode()) {
            std::println("No node selected for transform");
            return;
        }

        // Update the selected node's translation
        glm::vec3 translation = current_transform_.getTranslation();
        scene_manager->setSelectedNodeTranslation(translation);
        // Gizmo rendering is now handled by ImGuizmo in gui_manager.cpp
    }

    void TranslationGizmoTool::syncFromSelectedNode(const ToolContext& ctx) {
        auto* scene_manager = ctx.getSceneManager();

        if (!scene_manager || !scene_manager->hasSelectedNode()) {
            // No node selected - reset to identity
            current_transform_ = lfs::geometry::EuclideanTransform();
        } else {
            // Get the current translation offset of the selected node
            glm::vec3 translation = scene_manager->getSelectedNodeTranslation();
            current_transform_ = lfs::geometry::EuclideanTransform(
                glm::mat3(1.0f),
                translation);
        }
        // Gizmo rendering is now handled by ImGuizmo in gui_manager.cpp
    }

} // namespace lfs::vis::tools