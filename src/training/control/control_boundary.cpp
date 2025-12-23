/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "control/control_boundary.hpp"

#include <algorithm>

namespace lfs::training {

    ControlBoundary& ControlBoundary::instance() {
        static ControlBoundary boundary;
        return boundary;
    }

    std::size_t ControlBoundary::register_callback(ControlHook hook, Callback cb) {
        if (!cb) {
            return 0;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        const auto id = next_id_.fetch_add(1, std::memory_order_relaxed);
        callbacks_[hook].push_back(Registration{.id = id, .cb = std::move(cb)});
        return id;
    }

    void ControlBoundary::unregister_callback(ControlHook hook, std::size_t handle) {
        if (handle == 0) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        auto it = callbacks_.find(hook);
        if (it == callbacks_.end()) {
            return;
        }

        auto& vec = it->second;
        vec.erase(std::remove_if(vec.begin(), vec.end(), [&](const Registration& r) { return r.id == handle; }), vec.end());

        if (vec.empty()) {
            callbacks_.erase(it);
        }
    }

    void ControlBoundary::notify(ControlHook hook, const HookContext& ctx) {
        // Copy callbacks under lock, then invoke outside to avoid deadlocks.
        std::vector<Registration> local;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = callbacks_.find(hook);
            if (it != callbacks_.end()) {
                local = it->second;
            }
        }

        for (const auto& reg : local) {
            if (reg.cb) {
                reg.cb(ctx);
            }
        }
    }

    void ControlBoundary::clear_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        callbacks_.clear();
    }

} // namespace lfs::training
