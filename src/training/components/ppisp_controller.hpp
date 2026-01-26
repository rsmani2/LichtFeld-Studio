/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core/tensor.hpp"
#include <cmath>
#include <istream>
#include <ostream>

namespace lfs::training {

    struct PPISPControllerConfig {
        double lr = 2e-3;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 1e-15;
        int warmup_steps = 500;
        double warmup_start_factor = 0.01;
        double final_lr_factor = 0.01;
    };

    class PPISPController {
    public:
        using Config = PPISPControllerConfig;

        PPISPController(int total_iterations, Config config = {});

        lfs::core::Tensor predict(const lfs::core::Tensor& rendered_rgb, float exposure_prior = 1.0f);
        void backward(const lfs::core::Tensor& grad_output);
        lfs::core::Tensor distillation_loss(const lfs::core::Tensor& pred, const lfs::core::Tensor& target);
        void optimizer_step();
        void zero_grad();
        void scheduler_step();

        double get_lr() const { return current_lr_; }
        int64_t get_step() const { return step_; }

        // Serialization (full state for checkpoints)
        void serialize(std::ostream& os) const;
        void deserialize(std::istream& is);

        // Inference-only serialization (weights only, no Adam state)
        void serialize_inference(std::ostream& os) const;
        void deserialize_inference(std::istream& is);

    private:
        void compute_bias_corrections(float& bc1_rcp, float& bc2_sqrt_rcp) const {
            const double bc1 = 1.0 - std::pow(config_.beta1, step_ + 1);
            const double bc2 = 1.0 - std::pow(config_.beta2, step_ + 1);
            bc1_rcp = static_cast<float>(1.0 / bc1);
            bc2_sqrt_rcp = static_cast<float>(1.0 / std::sqrt(bc2));
        }

        void adam_update(lfs::core::Tensor& param, lfs::core::Tensor& exp_avg,
                         lfs::core::Tensor& exp_avg_sq, const lfs::core::Tensor& grad);

        void ensure_buffers_allocated(size_t H, size_t W);

        // Conv layers (fixed random features, not trained)
        lfs::core::Tensor conv1_w_, conv1_b_;
        lfs::core::Tensor conv2_w_, conv2_b_;
        lfs::core::Tensor conv3_w_, conv3_b_;

        // FC layers (trained)
        lfs::core::Tensor fc1_w_, fc1_b_;
        lfs::core::Tensor fc2_w_, fc2_b_;
        lfs::core::Tensor fc3_w_, fc3_b_;

        // FC gradients
        lfs::core::Tensor fc1_w_grad_, fc1_b_grad_;
        lfs::core::Tensor fc2_w_grad_, fc2_b_grad_;
        lfs::core::Tensor fc3_w_grad_, fc3_b_grad_;

        // FC Adam state
        lfs::core::Tensor fc1_w_m_, fc1_w_v_, fc1_b_m_, fc1_b_v_;
        lfs::core::Tensor fc2_w_m_, fc2_w_v_, fc2_b_m_, fc2_b_v_;
        lfs::core::Tensor fc3_w_m_, fc3_w_v_, fc3_b_m_, fc3_b_v_;

        // Pre-allocated buffers
        size_t buf_h_ = 0, buf_w_ = 0;
        lfs::core::Tensor buf_conv1_;
        lfs::core::Tensor buf_pool_;
        lfs::core::Tensor buf_conv2_;
        lfs::core::Tensor buf_conv3_;
        lfs::core::Tensor buf_pool2_;
        lfs::core::Tensor buf_fc1_;
        lfs::core::Tensor buf_fc2_;
        lfs::core::Tensor buf_output_;
        lfs::core::Tensor fc_input_buffer_;
        lfs::core::Tensor cached_flat_;

        Config config_;
        int64_t step_ = 0;
        double current_lr_;
        double initial_lr_;
        int total_iterations_;
    };

} // namespace lfs::training
