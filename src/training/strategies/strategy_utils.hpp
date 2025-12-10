/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "istrategy.hpp"
#include "optimizers/scheduler.hpp"
#include <memory>
#include <torch/torch.h>

namespace gs::training {
    void initialize_gaussians(gs::SplatData& splat_data);

    std::unique_ptr<torch::optim::Optimizer> create_optimizer(
        gs::SplatData& splat_data,
        const gs::param::OptimizationParameters& params);

    std::unique_ptr<ExponentialLR> create_scheduler(
        const gs::param::OptimizationParameters& params,
        torch::optim::Optimizer* optimizer,
        int param_group_index = -1);

    // Unified scheduler creation with optional warmup
    std::unique_ptr<WarmupExponentialLR> create_warmup_scheduler(
        const gs::param::OptimizationParameters& params,
        torch::optim::Optimizer* optimizer,
        int param_group_index = -1,
        int warmup_steps = 0,
        float warmup_start_factor = 1.0f);

    // Helper to compute decay gamma
    inline double compute_lr_decay_gamma(float final_lr_fraction, size_t iterations) {
        return std::pow(final_lr_fraction, 1.0 / iterations);
    }

    // Use explicit type alias to help MSVC
    using ParamUpdateFn = std::function<torch::Tensor(const int, const torch::Tensor)>;
    using OptimizerUpdateFn = std::function<std::unique_ptr<torch::optim::OptimizerParamState>(
        torch::optim::OptimizerParamState&, const torch::Tensor)>;

    void update_param_with_optimizer(
        const ParamUpdateFn& param_fn,
        const OptimizerUpdateFn& optimizer_fn,
        std::unique_ptr<torch::optim::Optimizer>& optimizer,
        gs::SplatData& splat_data,
        std::vector<size_t> param_idxs = {0, 1, 2, 3, 4, 5});
} // namespace gs::training
