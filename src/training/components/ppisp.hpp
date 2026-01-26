/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core/tensor.hpp"
#include "lfs/kernels/ppisp.cuh"
#include <cmath>
#include <istream>
#include <ostream>

namespace lfs::training {

    struct PPISPConfig {
        double lr = 1e-3;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 1e-15;
        int warmup_steps = 1000;
        double warmup_start_factor = 0.01;
        double final_lr_factor = 0.01;
        float reg_weight = 0.001f;
    };

    /// Physically-Plausible Image Signal Processing for per-camera/per-frame appearance modeling
    class PPISP {
    public:
        using Config = PPISPConfig;

        PPISP(int num_cameras, int num_frames, int total_iterations, Config config = {});

        /// Forward pass: apply ISP pipeline (exposure, vignetting, color correction, CRF)
        lfs::core::Tensor apply(const lfs::core::Tensor& rgb, int camera_idx, int frame_idx);

        /// Apply ISP with controller-predicted params (for novel view synthesis)
        /// @param rgb input image [C,H,W]
        /// @param controller_params [1,9] = [exposure, color_params[0:8]]
        /// @param camera_idx camera index for vignetting/CRF (use 0 or averaged if unknown)
        lfs::core::Tensor apply_with_controller_params(const lfs::core::Tensor& rgb,
                                                       const lfs::core::Tensor& controller_params,
                                                       int camera_idx = 0);

        /// Backward pass: accumulate gradients (call optimizer_step after all backward calls)
        lfs::core::Tensor backward(const lfs::core::Tensor& rgb, const lfs::core::Tensor& grad_output, int camera_idx,
                                   int frame_idx);

        /// Compute regularization loss (returns GPU tensor for async accumulation)
        lfs::core::Tensor reg_loss_gpu();

        /// Accumulate regularization gradients
        void reg_backward();

        /// Apply Adam with all accumulated gradients
        void optimizer_step();

        /// Clear gradients for next iteration
        void zero_grad();

        /// Update learning rate schedule
        void scheduler_step();

        // Accessors
        int num_cameras() const { return num_cameras_; }
        int num_frames() const { return num_frames_; }
        double get_lr() const { return current_lr_; }
        int64_t get_step() const { return step_; }
        float get_reg_weight() const { return config_.reg_weight; }

        /// Get learned parameters for a specific frame as [1,9] tensor
        /// Returns: [exposure, color_params[0:8]] for distillation target
        lfs::core::Tensor get_params_for_frame(int frame_idx) const;

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

        // Parameter tensors (all on GPU)
        // Exposure: [num_frames] - per-frame exposure in log-space
        lfs::core::Tensor exposure_params_;
        lfs::core::Tensor exposure_exp_avg_;
        lfs::core::Tensor exposure_exp_avg_sq_;
        lfs::core::Tensor exposure_grad_;

        // Vignetting: [num_cameras, 3, 5] - per-camera per-channel vignetting
        // 5 params: cx, cy, alpha0, alpha1, alpha2
        lfs::core::Tensor vignetting_params_;
        lfs::core::Tensor vignetting_exp_avg_;
        lfs::core::Tensor vignetting_exp_avg_sq_;
        lfs::core::Tensor vignetting_grad_;

        // Color: [num_frames, 8] - per-frame color latent offsets
        // 8 params: 4 color points x 2D offsets
        lfs::core::Tensor color_params_;
        lfs::core::Tensor color_exp_avg_;
        lfs::core::Tensor color_exp_avg_sq_;
        lfs::core::Tensor color_grad_;

        // CRF: [num_cameras, 3, 4] - per-camera per-channel CRF
        // 4 params: toe, shoulder, gamma, center
        lfs::core::Tensor crf_params_;
        lfs::core::Tensor crf_exp_avg_;
        lfs::core::Tensor crf_exp_avg_sq_;
        lfs::core::Tensor crf_grad_;

        Config config_;
        int64_t step_ = 0;
        double current_lr_;
        double initial_lr_;
        int total_iterations_;
        int num_cameras_;
        int num_frames_;
    };

} // namespace lfs::training
