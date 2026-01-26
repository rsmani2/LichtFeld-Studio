/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "ppisp_controller.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>

namespace lfs::training {

    namespace {
        constexpr uint32_t CHECKPOINT_MAGIC = 0x4C465043;
        constexpr uint32_t CHECKPOINT_VERSION = 1;

        lfs::core::Tensor kaiming_uniform(size_t fan_in, size_t fan_out) {
            float bound = std::sqrt(6.0f / static_cast<float>(fan_in));
            return lfs::core::Tensor::uniform({fan_out, fan_in}, -bound, bound, lfs::core::Device::CUDA);
        }

        lfs::core::Tensor zeros_bias(size_t size) {
            return lfs::core::Tensor::zeros({size}, lfs::core::Device::CUDA);
        }

        lfs::core::Tensor zeros_like(const lfs::core::Tensor& t) {
            return lfs::core::Tensor::zeros(t.shape(), t.device());
        }
    } // namespace

    PPISPController::PPISPController(int total_iterations, Config config)
        : config_(config),
          current_lr_(config.lr),
          initial_lr_(config.lr),
          total_iterations_(total_iterations) {

        conv1_w_ = kaiming_uniform(3, 16);
        conv1_b_ = zeros_bias(16);
        conv2_w_ = kaiming_uniform(16, 32);
        conv2_b_ = zeros_bias(32);
        conv3_w_ = kaiming_uniform(32, 64);
        conv3_b_ = zeros_bias(64);

        fc1_w_ = kaiming_uniform(1601, 128);
        fc1_b_ = zeros_bias(128);
        fc2_w_ = kaiming_uniform(128, 128);
        fc2_b_ = zeros_bias(128);
        fc3_w_ = kaiming_uniform(128, 9);
        fc3_b_ = zeros_bias(9);

        fc1_w_grad_ = zeros_like(fc1_w_);
        fc1_b_grad_ = zeros_like(fc1_b_);
        fc2_w_grad_ = zeros_like(fc2_w_);
        fc2_b_grad_ = zeros_like(fc2_b_);
        fc3_w_grad_ = zeros_like(fc3_w_);
        fc3_b_grad_ = zeros_like(fc3_b_);

        fc1_w_m_ = zeros_like(fc1_w_);
        fc1_w_v_ = zeros_like(fc1_w_);
        fc1_b_m_ = zeros_like(fc1_b_);
        fc1_b_v_ = zeros_like(fc1_b_);
        fc2_w_m_ = zeros_like(fc2_w_);
        fc2_w_v_ = zeros_like(fc2_w_);
        fc2_b_m_ = zeros_like(fc2_b_);
        fc2_b_v_ = zeros_like(fc2_b_);
        fc3_w_m_ = zeros_like(fc3_w_);
        fc3_w_v_ = zeros_like(fc3_w_);
        fc3_b_m_ = zeros_like(fc3_b_);
        fc3_b_v_ = zeros_like(fc3_b_);

        buf_pool2_ = lfs::core::Tensor::empty({1, 64, 5, 5}, lfs::core::Device::CUDA);
        buf_fc1_ = lfs::core::Tensor::empty({1, 128}, lfs::core::Device::CUDA);
        buf_fc2_ = lfs::core::Tensor::empty({1, 128}, lfs::core::Device::CUDA);
        buf_output_ = lfs::core::Tensor::empty({1, 9}, lfs::core::Device::CUDA);
        fc_input_buffer_ = lfs::core::Tensor::zeros({1, 1601}, lfs::core::Device::CUDA);
        float default_prior = 1.0f;
        cudaMemcpy(fc_input_buffer_.ptr<float>() + 1600, &default_prior, sizeof(float), cudaMemcpyHostToDevice);

        LOG_DEBUG("PPISPController: total_iterations={}, lr={:.2e}", total_iterations, config.lr);
    }

    void PPISPController::ensure_buffers_allocated(size_t H, size_t W) {
        if (buf_h_ == H && buf_w_ == W)
            return;

        const size_t pool_h = H / 3;
        const size_t pool_w = W / 3;

        buf_conv1_ = lfs::core::Tensor::empty({1, 16, H, W}, lfs::core::Device::CUDA);
        buf_pool_ = lfs::core::Tensor::empty({1, 16, pool_h, pool_w}, lfs::core::Device::CUDA);
        buf_conv2_ = lfs::core::Tensor::empty({1, 32, pool_h, pool_w}, lfs::core::Device::CUDA);
        buf_conv3_ = lfs::core::Tensor::empty({1, 64, pool_h, pool_w}, lfs::core::Device::CUDA);

        buf_h_ = H;
        buf_w_ = W;
    }

    lfs::core::Tensor PPISPController::predict(const lfs::core::Tensor& rendered_rgb, float exposure_prior) {
        assert(rendered_rgb.shape().rank() == 4);
        assert(rendered_rgb.shape()[0] == 1);
        assert(rendered_rgb.shape()[1] == 3);

        const size_t H = rendered_rgb.shape()[2];
        const size_t W = rendered_rgb.shape()[3];
        ensure_buffers_allocated(H, W);

        rendered_rgb.conv1x1_bias_relu_out(conv1_w_, conv1_b_, buf_conv1_);
        buf_conv1_.max_pool2d_out(3, 3, 0, buf_pool_);
        buf_pool_.conv1x1_bias_relu_out(conv2_w_, conv2_b_, buf_conv2_);
        buf_conv2_.conv1x1_bias_relu_out(conv3_w_, conv3_b_, buf_conv3_);
        buf_conv3_.adaptive_avg_pool2d_out(5, 5, buf_pool2_);

        auto flat = buf_pool2_.flatten(1);
        cudaMemcpyAsync(fc_input_buffer_.ptr<float>(), flat.ptr<float>(),
                        1600 * sizeof(float), cudaMemcpyDeviceToDevice, nullptr);
        if (exposure_prior != 1.0f) {
            cudaMemcpyAsync(fc_input_buffer_.ptr<float>() + 1600, &exposure_prior,
                            sizeof(float), cudaMemcpyHostToDevice, nullptr);
        }
        cached_flat_ = fc_input_buffer_;

        cached_flat_.linear_bias_relu_out(fc1_w_, fc1_b_, buf_fc1_);
        buf_fc1_.linear_bias_relu_out(fc2_w_, fc2_b_, buf_fc2_);
        buf_fc2_.linear_out(fc3_w_, fc3_b_, buf_output_);

        return buf_output_;
    }

    void PPISPController::backward(const lfs::core::Tensor& grad_output) {
        assert(grad_output.shape().rank() == 2);
        assert(grad_output.shape()[0] == 1 && grad_output.shape()[1] == 9);

        auto grad_fc3 = grad_output;
        fc3_w_grad_.add_(buf_fc2_.t().mm(grad_fc3).t());
        fc3_b_grad_.add_(grad_fc3.sum(0));
        auto grad_fc2_out = grad_fc3.mm(fc3_w_);

        auto grad_fc2_pre = grad_fc2_out.mul(buf_fc2_.gt(0.0f).to(lfs::core::DataType::Float32));
        fc2_w_grad_.add_(buf_fc1_.t().mm(grad_fc2_pre).t());
        fc2_b_grad_.add_(grad_fc2_pre.sum(0));
        auto grad_fc1_out = grad_fc2_pre.mm(fc2_w_);

        auto grad_fc1_pre = grad_fc1_out.mul(buf_fc1_.gt(0.0f).to(lfs::core::DataType::Float32));
        fc1_w_grad_.add_(cached_flat_.t().mm(grad_fc1_pre).t());
        fc1_b_grad_.add_(grad_fc1_pre.sum(0));
    }

    lfs::core::Tensor PPISPController::distillation_loss(const lfs::core::Tensor& pred,
                                                         const lfs::core::Tensor& target) {
        assert(pred.shape() == target.shape());
        return pred.sub(target).square().mean();
    }

    void PPISPController::adam_update(lfs::core::Tensor& param, lfs::core::Tensor& exp_avg,
                                      lfs::core::Tensor& exp_avg_sq, const lfs::core::Tensor& grad) {
        float bc1_rcp, bc2_sqrt_rcp;
        compute_bias_corrections(bc1_rcp, bc2_sqrt_rcp);

        const float lr = static_cast<float>(current_lr_);
        const float beta1 = static_cast<float>(config_.beta1);
        const float beta2 = static_cast<float>(config_.beta2);
        const float eps = static_cast<float>(config_.eps);

        exp_avg.mul_(beta1).add_(grad.mul(1.0f - beta1));
        exp_avg_sq.mul_(beta2).add_(grad.square().mul(1.0f - beta2));

        auto m_hat = exp_avg.mul(bc1_rcp);
        auto v_hat = exp_avg_sq.mul(bc2_sqrt_rcp * bc2_sqrt_rcp);
        param.sub_(m_hat.div(v_hat.sqrt().add(eps)).mul(lr));
    }

    void PPISPController::optimizer_step() {
        adam_update(fc1_w_, fc1_w_m_, fc1_w_v_, fc1_w_grad_);
        adam_update(fc1_b_, fc1_b_m_, fc1_b_v_, fc1_b_grad_);
        adam_update(fc2_w_, fc2_w_m_, fc2_w_v_, fc2_w_grad_);
        adam_update(fc2_b_, fc2_b_m_, fc2_b_v_, fc2_b_grad_);
        adam_update(fc3_w_, fc3_w_m_, fc3_w_v_, fc3_w_grad_);
        adam_update(fc3_b_, fc3_b_m_, fc3_b_v_, fc3_b_grad_);
    }

    void PPISPController::zero_grad() {
        fc1_w_grad_.zero_();
        fc1_b_grad_.zero_();
        fc2_w_grad_.zero_();
        fc2_b_grad_.zero_();
        fc3_w_grad_.zero_();
        fc3_b_grad_.zero_();
    }

    void PPISPController::scheduler_step() {
        ++step_;

        if (step_ <= config_.warmup_steps) {
            const double progress = static_cast<double>(step_) / config_.warmup_steps;
            const double scale = config_.warmup_start_factor + (1.0 - config_.warmup_start_factor) * progress;
            current_lr_ = initial_lr_ * scale;
        } else {
            const double gamma =
                std::pow(config_.final_lr_factor, 1.0 / (total_iterations_ - config_.warmup_steps));
            current_lr_ = initial_lr_ * std::pow(gamma, step_ - config_.warmup_steps);
        }
    }

    void PPISPController::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&CHECKPOINT_MAGIC), sizeof(CHECKPOINT_MAGIC));
        os.write(reinterpret_cast<const char*>(&CHECKPOINT_VERSION), sizeof(CHECKPOINT_VERSION));

        os.write(reinterpret_cast<const char*>(&config_), sizeof(config_));
        os.write(reinterpret_cast<const char*>(&step_), sizeof(step_));
        os.write(reinterpret_cast<const char*>(&current_lr_), sizeof(current_lr_));
        os.write(reinterpret_cast<const char*>(&initial_lr_), sizeof(initial_lr_));
        os.write(reinterpret_cast<const char*>(&total_iterations_), sizeof(total_iterations_));

        os << conv1_w_ << conv1_b_;
        os << conv2_w_ << conv2_b_;
        os << conv3_w_ << conv3_b_;

        os << fc1_w_ << fc1_b_ << fc1_w_m_ << fc1_w_v_ << fc1_b_m_ << fc1_b_v_;
        os << fc2_w_ << fc2_b_ << fc2_w_m_ << fc2_w_v_ << fc2_b_m_ << fc2_b_v_;
        os << fc3_w_ << fc3_b_ << fc3_w_m_ << fc3_w_v_ << fc3_b_m_ << fc3_b_v_;
    }

    void PPISPController::deserialize(std::istream& is) {
        uint32_t magic, version;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != CHECKPOINT_MAGIC)
            throw std::runtime_error("Invalid PPISPController checkpoint");
        if (version != CHECKPOINT_VERSION)
            throw std::runtime_error("Unsupported PPISPController checkpoint version");

        is.read(reinterpret_cast<char*>(&config_), sizeof(config_));
        is.read(reinterpret_cast<char*>(&step_), sizeof(step_));
        is.read(reinterpret_cast<char*>(&current_lr_), sizeof(current_lr_));
        is.read(reinterpret_cast<char*>(&initial_lr_), sizeof(initial_lr_));
        is.read(reinterpret_cast<char*>(&total_iterations_), sizeof(total_iterations_));

        is >> conv1_w_ >> conv1_b_;
        is >> conv2_w_ >> conv2_b_;
        is >> conv3_w_ >> conv3_b_;

        is >> fc1_w_ >> fc1_b_ >> fc1_w_m_ >> fc1_w_v_ >> fc1_b_m_ >> fc1_b_v_;
        is >> fc2_w_ >> fc2_b_ >> fc2_w_m_ >> fc2_w_v_ >> fc2_b_m_ >> fc2_b_v_;
        is >> fc3_w_ >> fc3_b_ >> fc3_w_m_ >> fc3_w_v_ >> fc3_b_m_ >> fc3_b_v_;

        conv1_w_ = conv1_w_.cuda();
        conv1_b_ = conv1_b_.cuda();
        conv2_w_ = conv2_w_.cuda();
        conv2_b_ = conv2_b_.cuda();
        conv3_w_ = conv3_w_.cuda();
        conv3_b_ = conv3_b_.cuda();

        fc1_w_ = fc1_w_.cuda();
        fc1_b_ = fc1_b_.cuda();
        fc1_w_m_ = fc1_w_m_.cuda();
        fc1_w_v_ = fc1_w_v_.cuda();
        fc1_b_m_ = fc1_b_m_.cuda();
        fc1_b_v_ = fc1_b_v_.cuda();

        fc2_w_ = fc2_w_.cuda();
        fc2_b_ = fc2_b_.cuda();
        fc2_w_m_ = fc2_w_m_.cuda();
        fc2_w_v_ = fc2_w_v_.cuda();
        fc2_b_m_ = fc2_b_m_.cuda();
        fc2_b_v_ = fc2_b_v_.cuda();

        fc3_w_ = fc3_w_.cuda();
        fc3_b_ = fc3_b_.cuda();
        fc3_w_m_ = fc3_w_m_.cuda();
        fc3_w_v_ = fc3_w_v_.cuda();
        fc3_b_m_ = fc3_b_m_.cuda();
        fc3_b_v_ = fc3_b_v_.cuda();

        fc1_w_grad_ = zeros_like(fc1_w_);
        fc1_b_grad_ = zeros_like(fc1_b_);
        fc2_w_grad_ = zeros_like(fc2_w_);
        fc2_b_grad_ = zeros_like(fc2_b_);
        fc3_w_grad_ = zeros_like(fc3_w_);
        fc3_b_grad_ = zeros_like(fc3_b_);
    }

    void PPISPController::serialize_inference(std::ostream& os) const {
        constexpr uint32_t INFERENCE_MAGIC = 0x4C464349; // "LFCI" - Controller Inference
        constexpr uint32_t INFERENCE_VERSION = 1;

        os.write(reinterpret_cast<const char*>(&INFERENCE_MAGIC), sizeof(INFERENCE_MAGIC));
        os.write(reinterpret_cast<const char*>(&INFERENCE_VERSION), sizeof(INFERENCE_VERSION));

        os << conv1_w_ << conv1_b_;
        os << conv2_w_ << conv2_b_;
        os << conv3_w_ << conv3_b_;

        os << fc1_w_ << fc1_b_;
        os << fc2_w_ << fc2_b_;
        os << fc3_w_ << fc3_b_;
    }

    void PPISPController::deserialize_inference(std::istream& is) {
        constexpr uint32_t INFERENCE_MAGIC = 0x4C464349;
        constexpr uint32_t INFERENCE_VERSION = 1;

        uint32_t magic, version;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != INFERENCE_MAGIC) {
            throw std::runtime_error("Invalid PPISPController inference file");
        }
        if (version != INFERENCE_VERSION) {
            throw std::runtime_error("Unsupported PPISPController inference version");
        }

        is >> conv1_w_ >> conv1_b_;
        is >> conv2_w_ >> conv2_b_;
        is >> conv3_w_ >> conv3_b_;

        is >> fc1_w_ >> fc1_b_;
        is >> fc2_w_ >> fc2_b_;
        is >> fc3_w_ >> fc3_b_;

        conv1_w_ = conv1_w_.cuda();
        conv1_b_ = conv1_b_.cuda();
        conv2_w_ = conv2_w_.cuda();
        conv2_b_ = conv2_b_.cuda();
        conv3_w_ = conv3_w_.cuda();
        conv3_b_ = conv3_b_.cuda();

        fc1_w_ = fc1_w_.cuda();
        fc1_b_ = fc1_b_.cuda();
        fc2_w_ = fc2_w_.cuda();
        fc2_b_ = fc2_b_.cuda();
        fc3_w_ = fc3_w_.cuda();
        fc3_b_ = fc3_b_.cuda();
    }

} // namespace lfs::training
