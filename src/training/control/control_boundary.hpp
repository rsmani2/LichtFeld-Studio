/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lfs::training {

    class Trainer; // forward declaration (non-owning in hooks)

    enum class ControlHook {
        TrainingStart,
        IterationStart,
        PreOptimizerStep,
        PostStep,
        TrainingEnd
    };

    struct ControlHookHash {
        std::size_t operator()(ControlHook hook) const noexcept {
            return static_cast<std::size_t>(hook);
        }
    };

    struct HookContext {
        int iteration = 0;
        float loss = 0.0f;
        std::size_t num_gaussians = 0;
        bool is_refining = false;
        Trainer* trainer = nullptr; // non-owning
    };

    /**
     * @brief Central boundary for Python-driven control. Holds registered callbacks and executes
     *        them at deterministic hook points. Thread-safe, singleton.
     */
    class ControlBoundary {
    public:
        using Callback = std::function<void(const HookContext&)>;

        static ControlBoundary& instance();

        // Register a callback for a specific hook. Returns an opaque handle for later removal.
        std::size_t register_callback(ControlHook hook, Callback cb);

        // Unregister a previously registered callback. No-op if not found.
        void unregister_callback(ControlHook hook, std::size_t handle);

        // Notify all callbacks for a hook. Safe to call from the training thread.
        void notify(ControlHook hook, const HookContext& ctx);

        // Remove all callbacks (used by sessions for cleanup).
        void clear_all();

    private:
        ControlBoundary() = default;
        ControlBoundary(const ControlBoundary&) = delete;
        ControlBoundary& operator=(const ControlBoundary&) = delete;

        struct Registration {
            std::size_t id;
            Callback cb;
        };

        std::mutex mutex_;
        std::unordered_map<ControlHook, std::vector<Registration>, ControlHookHash> callbacks_;
        std::atomic<std::size_t> next_id_{1};
    };

} // namespace lfs::training
