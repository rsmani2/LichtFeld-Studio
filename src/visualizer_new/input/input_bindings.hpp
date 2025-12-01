/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <GLFW/glfw3.h>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace lfs::vis::input {

    // Actions that can be bound to inputs
    enum class Action {
        // Camera navigation
        CAMERA_ORBIT,
        CAMERA_PAN,
        CAMERA_ZOOM,
        CAMERA_ROLL,
        CAMERA_MOVE_FORWARD,
        CAMERA_MOVE_BACKWARD,
        CAMERA_MOVE_LEFT,
        CAMERA_MOVE_RIGHT,
        CAMERA_MOVE_UP,
        CAMERA_MOVE_DOWN,
        CAMERA_RESET_HOME,
        CAMERA_SET_PIVOT,
        CAMERA_NEXT_VIEW,
        CAMERA_PREV_VIEW,
        CAMERA_SPEED_UP,
        CAMERA_SPEED_DOWN,
        ZOOM_SPEED_UP,
        ZOOM_SPEED_DOWN,

        // View controls
        TOGGLE_SPLIT_VIEW,
        TOGGLE_GT_COMPARISON,
        TOGGLE_DEPTH_MODE,
        CYCLE_PLY,

        // Selection/editing
        DELETE_SELECTED,
        UNDO,
        REDO,
        INVERT_SELECTION,
        DESELECT_ALL,

        // Depth filter (in selection tool)
        DEPTH_ADJUST_FAR,
        DEPTH_ADJUST_SIDE,

        // Tool-specific
        BRUSH_RESIZE,
        CYCLE_BRUSH_MODE,
        CONFIRM_POLYGON,
        CANCEL_POLYGON,
        UNDO_POLYGON_VERTEX,
        CYCLE_SELECTION_VIS,

        // Selection actions (mouse + modifier)
        SELECTION_ADD,
        SELECTION_REMOVE,

        // Selection mode shortcuts
        SELECT_MODE_CENTERS,
        SELECT_MODE_RECTANGLE,
        SELECT_MODE_POLYGON,
        SELECT_MODE_LASSO,
        SELECT_MODE_RINGS,

        // Misc
        APPLY_CROP_BOX,
    };

    // Modifier flags (can be combined)
    // Note: Using MODIFIER_ prefix to avoid Windows macro conflicts (MOD_NONE, MOD_SHIFT etc.)
    enum Modifier : int {
        MODIFIER_NONE = 0,
        MODIFIER_SHIFT = GLFW_MOD_SHIFT,
        MODIFIER_CTRL = GLFW_MOD_CONTROL,
        MODIFIER_ALT = GLFW_MOD_ALT,
        MODIFIER_SUPER = GLFW_MOD_SUPER,
    };

    // Mouse button identifiers
    enum class MouseButton {
        LEFT = GLFW_MOUSE_BUTTON_LEFT,
        RIGHT = GLFW_MOUSE_BUTTON_RIGHT,
        MIDDLE = GLFW_MOUSE_BUTTON_MIDDLE,
    };

    // Input trigger types
    struct KeyTrigger {
        int key;                           // GLFW key code
        int modifiers = MODIFIER_NONE;     // Required modifiers
        bool on_repeat = false;            // Allow repeat
    };

    struct MouseButtonTrigger {
        MouseButton button;
        int modifiers = MODIFIER_NONE;
    };

    struct MouseScrollTrigger {
        int modifiers = MODIFIER_NONE;     // Modifiers to activate
    };

    struct MouseDragTrigger {
        MouseButton button;
        int modifiers = MODIFIER_NONE;
    };

    using InputTrigger = std::variant<KeyTrigger, MouseButtonTrigger, MouseScrollTrigger, MouseDragTrigger>;

    // A single binding: trigger -> action
    struct Binding {
        InputTrigger trigger;
        Action action;
        std::string description;     // Human-readable description
    };

    // A profile contains a named set of bindings
    struct Profile {
        std::string name;
        std::string description;
        std::vector<Binding> bindings;
    };

    // Input binding manager
    class InputBindings {
    public:
        InputBindings();

        // Profile management
        void loadProfile(const std::string& name);
        void saveProfile(const std::string& name) const;
        bool loadProfileFromFile(const std::filesystem::path& path);
        bool saveProfileToFile(const std::filesystem::path& path) const;
        std::vector<std::string> getAvailableProfiles() const;
        const std::string& getCurrentProfileName() const { return current_profile_name_; }

        // Get config directory for profile storage
        static std::filesystem::path getConfigDir();

        // Query bindings
        std::optional<Action> getActionForKey(int key, int modifiers) const;
        std::optional<Action> getActionForMouseButton(MouseButton button, int modifiers) const;
        std::optional<Action> getActionForScroll(int modifiers) const;
        std::optional<Action> getActionForDrag(MouseButton button, int modifiers) const;

        // Get trigger for an action (for UI display)
        std::optional<InputTrigger> getTriggerForAction(Action action) const;
        std::string getTriggerDescription(Action action) const;

        // Modify bindings
        void setBinding(Action action, const InputTrigger& trigger);
        void clearBinding(Action action);

        // Built-in profiles
        static Profile createDefaultProfile();

    private:
        std::string current_profile_name_;
        std::vector<Binding> bindings_;

        // Lookup maps for fast queries
        std::map<std::pair<int, int>, Action> key_map_;           // (key, mods) -> action
        std::map<std::pair<MouseButton, int>, Action> mouse_button_map_;
        std::map<int, Action> scroll_map_;                        // mods -> action
        std::map<std::pair<MouseButton, int>, Action> drag_map_;

        void rebuildLookupMaps();
        static bool modifiersMatch(int required, int actual);
    };

    // Helper to get human-readable names
    std::string getActionName(Action action);
    std::string getKeyName(int key);
    std::string getMouseButtonName(MouseButton button);
    std::string getModifierString(int modifiers);

} // namespace lfs::vis::input
