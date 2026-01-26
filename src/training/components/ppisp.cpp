/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "ppisp.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace lfs::training {

    namespace {
        constexpr uint32_t CHECKPOINT_MAGIC = 0x4C465050; // "LFPP"
        constexpr uint32_t CHECKPOINT_VERSION = 1;
    } // namespace

    PPISP::PPISP(int num_cameras, int num_frames, int total_iterations, Config config)
        : config_(config),
          current_lr_(config.lr),
          initial_lr_(config.lr),
          total_iterations_(total_iterations),
          num_cameras_(num_cameras),
          num_frames_(num_frames) {

        assert(num_cameras > 0 && "num_cameras must be positive");
        assert(num_frames > 0 && "num_frames must be positive");

        // Allocate exposure params [num_frames]
        exposure_params_ = lfs::core::Tensor::zeros({static_cast<size_t>(num_frames)}, lfs::core::Device::CUDA);
        exposure_exp_avg_ = lfs::core::Tensor::zeros({static_cast<size_t>(num_frames)}, lfs::core::Device::CUDA);
        exposure_exp_avg_sq_ = lfs::core::Tensor::zeros({static_cast<size_t>(num_frames)}, lfs::core::Device::CUDA);
        exposure_grad_ = lfs::core::Tensor::zeros({static_cast<size_t>(num_frames)}, lfs::core::Device::CUDA);

        // Allocate vignetting params [num_cameras * 3 * 5]
        size_t vig_size = static_cast<size_t>(num_cameras) * 3 * 5;
        vignetting_params_ = lfs::core::Tensor::zeros({vig_size}, lfs::core::Device::CUDA);
        vignetting_exp_avg_ = lfs::core::Tensor::zeros({vig_size}, lfs::core::Device::CUDA);
        vignetting_exp_avg_sq_ = lfs::core::Tensor::zeros({vig_size}, lfs::core::Device::CUDA);
        vignetting_grad_ = lfs::core::Tensor::zeros({vig_size}, lfs::core::Device::CUDA);

        // Allocate color params [num_frames * 8]
        size_t color_size = static_cast<size_t>(num_frames) * 8;
        color_params_ = lfs::core::Tensor::zeros({color_size}, lfs::core::Device::CUDA);
        color_exp_avg_ = lfs::core::Tensor::zeros({color_size}, lfs::core::Device::CUDA);
        color_exp_avg_sq_ = lfs::core::Tensor::zeros({color_size}, lfs::core::Device::CUDA);
        color_grad_ = lfs::core::Tensor::zeros({color_size}, lfs::core::Device::CUDA);

        // Allocate CRF params [num_cameras * 3 * 4]
        size_t crf_size = static_cast<size_t>(num_cameras) * 3 * 4;
        crf_params_ = lfs::core::Tensor::zeros({crf_size}, lfs::core::Device::CUDA);
        crf_exp_avg_ = lfs::core::Tensor::zeros({crf_size}, lfs::core::Device::CUDA);
        crf_exp_avg_sq_ = lfs::core::Tensor::zeros({crf_size}, lfs::core::Device::CUDA);
        crf_grad_ = lfs::core::Tensor::zeros({crf_size}, lfs::core::Device::CUDA);

        // Initialize to identity (already zeros, which is identity for these params)
        kernels::launch_ppisp_init_identity(exposure_params_.ptr<float>(), vignetting_params_.ptr<float>(),
                                            color_params_.ptr<float>(), crf_params_.ptr<float>(), num_cameras,
                                            num_frames, nullptr);

        LOG_DEBUG("PPISP: {} cameras, {} frames, lr={:.2e}, reg_weight={:.4f}", num_cameras, num_frames, config.lr,
                  config.reg_weight);
    }

    lfs::core::Tensor PPISP::apply(const lfs::core::Tensor& rgb, int camera_idx, int frame_idx) {
        assert(camera_idx >= -1 && camera_idx < num_cameras_ && "camera_idx out of range");
        assert(frame_idx >= -1 && frame_idx < num_frames_ && "frame_idx out of range");

        const auto& shape = rgb.shape();
        assert(shape.rank() == 3 && shape[0] == 3 && "Expected CHW layout with 3 channels");

        const int h = static_cast<int>(shape[1]);
        const int w = static_cast<int>(shape[2]);

        auto output = lfs::core::Tensor::empty({3, shape[1], shape[2]}, lfs::core::Device::CUDA);

        kernels::launch_ppisp_forward_chw(exposure_params_.ptr<float>(), vignetting_params_.ptr<float>(),
                                          color_params_.ptr<float>(), crf_params_.ptr<float>(), rgb.ptr<float>(),
                                          output.ptr<float>(), h, w, num_cameras_, num_frames_, camera_idx, frame_idx,
                                          nullptr);

        return output;
    }

    lfs::core::Tensor PPISP::apply_with_controller_params(const lfs::core::Tensor& rgb,
                                                          const lfs::core::Tensor& controller_params,
                                                          int camera_idx) {
        assert(controller_params.shape().rank() == 2 && "Expected [1,9]");
        assert(controller_params.shape()[0] == 1 && controller_params.shape()[1] == 9);
        assert(camera_idx >= 0 && camera_idx < num_cameras_ && "camera_idx out of range");

        const auto& shape = rgb.shape();
        assert(shape.rank() == 3 && shape[0] == 3 && "Expected CHW layout with 3 channels");

        const int h = static_cast<int>(shape[1]);
        const int w = static_cast<int>(shape[2]);

        // Extract exposure (index 0) and color params (indices 1-8) from controller output
        auto exposure_temp = controller_params.slice(1, 0, 1).reshape({1});
        auto color_temp = controller_params.slice(1, 1, 9).reshape({8});

        auto output = lfs::core::Tensor::empty({3, shape[1], shape[2]}, lfs::core::Device::CUDA);

        // Use controller-predicted exposure and color, but existing vignetting and CRF from camera
        kernels::launch_ppisp_forward_chw(exposure_temp.ptr<float>(), vignetting_params_.ptr<float>(),
                                          color_temp.ptr<float>(), crf_params_.ptr<float>(), rgb.ptr<float>(),
                                          output.ptr<float>(), h, w, num_cameras_, 1, camera_idx, 0, nullptr);

        return output;
    }

    lfs::core::Tensor PPISP::backward(const lfs::core::Tensor& rgb, const lfs::core::Tensor& grad_output,
                                      int camera_idx, int frame_idx) {
        assert(camera_idx >= -1 && camera_idx < num_cameras_ && "camera_idx out of range");
        assert(frame_idx >= -1 && frame_idx < num_frames_ && "frame_idx out of range");

        const auto& shape = rgb.shape();
        assert(shape.rank() == 3 && shape[0] == 3 && "Expected CHW layout with 3 channels");

        const int h = static_cast<int>(shape[1]);
        const int w = static_cast<int>(shape[2]);

        auto grad_rgb = lfs::core::Tensor::empty({3, shape[1], shape[2]}, lfs::core::Device::CUDA);

        kernels::launch_ppisp_backward_chw(
            exposure_params_.ptr<float>(), vignetting_params_.ptr<float>(), color_params_.ptr<float>(),
            crf_params_.ptr<float>(), rgb.ptr<float>(), grad_output.ptr<float>(), exposure_grad_.ptr<float>(),
            vignetting_grad_.ptr<float>(), color_grad_.ptr<float>(), crf_grad_.ptr<float>(), grad_rgb.ptr<float>(), h,
            w, num_cameras_, num_frames_, camera_idx, frame_idx, nullptr);

        return grad_rgb;
    }

    lfs::core::Tensor PPISP::reg_loss_gpu() {
        auto loss = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);

        // Add L2 regularization for all parameters
        kernels::launch_ppisp_reg_loss(exposure_params_.ptr<float>(), loss.ptr<float>(),
                                       static_cast<int>(exposure_params_.numel()), nullptr);
        kernels::launch_ppisp_reg_loss(vignetting_params_.ptr<float>(), loss.ptr<float>(),
                                       static_cast<int>(vignetting_params_.numel()), nullptr);
        kernels::launch_ppisp_reg_loss(color_params_.ptr<float>(), loss.ptr<float>(),
                                       static_cast<int>(color_params_.numel()), nullptr);
        kernels::launch_ppisp_reg_loss(crf_params_.ptr<float>(), loss.ptr<float>(),
                                       static_cast<int>(crf_params_.numel()), nullptr);

        return loss;
    }

    void PPISP::reg_backward() {
        const float weight = config_.reg_weight;

        kernels::launch_ppisp_reg_backward(exposure_params_.ptr<float>(), exposure_grad_.ptr<float>(), weight,
                                           static_cast<int>(exposure_params_.numel()), nullptr);
        kernels::launch_ppisp_reg_backward(vignetting_params_.ptr<float>(), vignetting_grad_.ptr<float>(), weight,
                                           static_cast<int>(vignetting_params_.numel()), nullptr);
        kernels::launch_ppisp_reg_backward(color_params_.ptr<float>(), color_grad_.ptr<float>(), weight,
                                           static_cast<int>(color_params_.numel()), nullptr);
        kernels::launch_ppisp_reg_backward(crf_params_.ptr<float>(), crf_grad_.ptr<float>(), weight,
                                           static_cast<int>(crf_params_.numel()), nullptr);
    }

    void PPISP::optimizer_step() {
        float bc1_rcp, bc2_sqrt_rcp;
        compute_bias_corrections(bc1_rcp, bc2_sqrt_rcp);

        const float lr = static_cast<float>(current_lr_);
        const float beta1 = static_cast<float>(config_.beta1);
        const float beta2 = static_cast<float>(config_.beta2);
        const float eps = static_cast<float>(config_.eps);

        // Update exposure
        kernels::launch_ppisp_adam_update(exposure_params_.ptr<float>(), exposure_exp_avg_.ptr<float>(),
                                          exposure_exp_avg_sq_.ptr<float>(), exposure_grad_.ptr<float>(),
                                          static_cast<int>(exposure_params_.numel()), lr, beta1, beta2, bc1_rcp,
                                          bc2_sqrt_rcp, eps, nullptr);

        // Update vignetting
        kernels::launch_ppisp_adam_update(vignetting_params_.ptr<float>(), vignetting_exp_avg_.ptr<float>(),
                                          vignetting_exp_avg_sq_.ptr<float>(), vignetting_grad_.ptr<float>(),
                                          static_cast<int>(vignetting_params_.numel()), lr, beta1, beta2, bc1_rcp,
                                          bc2_sqrt_rcp, eps, nullptr);

        // Update color
        kernels::launch_ppisp_adam_update(color_params_.ptr<float>(), color_exp_avg_.ptr<float>(),
                                          color_exp_avg_sq_.ptr<float>(), color_grad_.ptr<float>(),
                                          static_cast<int>(color_params_.numel()), lr, beta1, beta2, bc1_rcp,
                                          bc2_sqrt_rcp, eps, nullptr);

        // Update CRF
        kernels::launch_ppisp_adam_update(crf_params_.ptr<float>(), crf_exp_avg_.ptr<float>(),
                                          crf_exp_avg_sq_.ptr<float>(), crf_grad_.ptr<float>(),
                                          static_cast<int>(crf_params_.numel()), lr, beta1, beta2, bc1_rcp,
                                          bc2_sqrt_rcp, eps, nullptr);
    }

    void PPISP::zero_grad() {
        cudaMemsetAsync(exposure_grad_.ptr<float>(), 0, exposure_grad_.numel() * sizeof(float), nullptr);
        cudaMemsetAsync(vignetting_grad_.ptr<float>(), 0, vignetting_grad_.numel() * sizeof(float), nullptr);
        cudaMemsetAsync(color_grad_.ptr<float>(), 0, color_grad_.numel() * sizeof(float), nullptr);
        cudaMemsetAsync(crf_grad_.ptr<float>(), 0, crf_grad_.numel() * sizeof(float), nullptr);
    }

    void PPISP::scheduler_step() {
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

    lfs::core::Tensor PPISP::get_params_for_frame(int frame_idx) const {
        assert(frame_idx >= 0 && frame_idx < num_frames_ && "frame_idx out of range");

        // Get exposure param for this frame: exposure_params_[frame_idx]
        auto exposure = exposure_params_.slice(0, frame_idx, frame_idx + 1);

        // Get color params for this frame: color_params_ is flat [num_frames * 8]
        // Extract [frame_idx * 8 : (frame_idx + 1) * 8]
        size_t color_start = static_cast<size_t>(frame_idx) * 8;
        auto color = color_params_.slice(0, color_start, color_start + 8);

        // Concatenate: [1] + [8] = [9], then reshape to [1, 9]
        auto params = lfs::core::Tensor::cat({exposure, color}, 0);
        return params.reshape({1, 9});
    }

    void PPISP::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&CHECKPOINT_MAGIC), sizeof(CHECKPOINT_MAGIC));
        os.write(reinterpret_cast<const char*>(&CHECKPOINT_VERSION), sizeof(CHECKPOINT_VERSION));

        os.write(reinterpret_cast<const char*>(&num_cameras_), sizeof(num_cameras_));
        os.write(reinterpret_cast<const char*>(&num_frames_), sizeof(num_frames_));
        os.write(reinterpret_cast<const char*>(&config_), sizeof(config_));
        os.write(reinterpret_cast<const char*>(&step_), sizeof(step_));
        os.write(reinterpret_cast<const char*>(&current_lr_), sizeof(current_lr_));
        os.write(reinterpret_cast<const char*>(&initial_lr_), sizeof(initial_lr_));
        os.write(reinterpret_cast<const char*>(&total_iterations_), sizeof(total_iterations_));

        os << exposure_params_ << exposure_exp_avg_ << exposure_exp_avg_sq_;
        os << vignetting_params_ << vignetting_exp_avg_ << vignetting_exp_avg_sq_;
        os << color_params_ << color_exp_avg_ << color_exp_avg_sq_;
        os << crf_params_ << crf_exp_avg_ << crf_exp_avg_sq_;
    }

    void PPISP::deserialize(std::istream& is) {
        uint32_t magic, version;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != CHECKPOINT_MAGIC) {
            throw std::runtime_error("Invalid PPISP checkpoint");
        }
        if (version != CHECKPOINT_VERSION) {
            throw std::runtime_error("Unsupported PPISP checkpoint version");
        }

        is.read(reinterpret_cast<char*>(&num_cameras_), sizeof(num_cameras_));
        is.read(reinterpret_cast<char*>(&num_frames_), sizeof(num_frames_));
        is.read(reinterpret_cast<char*>(&config_), sizeof(config_));
        is.read(reinterpret_cast<char*>(&step_), sizeof(step_));
        is.read(reinterpret_cast<char*>(&current_lr_), sizeof(current_lr_));
        is.read(reinterpret_cast<char*>(&initial_lr_), sizeof(initial_lr_));
        is.read(reinterpret_cast<char*>(&total_iterations_), sizeof(total_iterations_));

        is >> exposure_params_ >> exposure_exp_avg_ >> exposure_exp_avg_sq_;
        is >> vignetting_params_ >> vignetting_exp_avg_ >> vignetting_exp_avg_sq_;
        is >> color_params_ >> color_exp_avg_ >> color_exp_avg_sq_;
        is >> crf_params_ >> crf_exp_avg_ >> crf_exp_avg_sq_;

        // Move to CUDA
        exposure_params_ = exposure_params_.cuda();
        exposure_exp_avg_ = exposure_exp_avg_.cuda();
        exposure_exp_avg_sq_ = exposure_exp_avg_sq_.cuda();
        vignetting_params_ = vignetting_params_.cuda();
        vignetting_exp_avg_ = vignetting_exp_avg_.cuda();
        vignetting_exp_avg_sq_ = vignetting_exp_avg_sq_.cuda();
        color_params_ = color_params_.cuda();
        color_exp_avg_ = color_exp_avg_.cuda();
        color_exp_avg_sq_ = color_exp_avg_sq_.cuda();
        crf_params_ = crf_params_.cuda();
        crf_exp_avg_ = crf_exp_avg_.cuda();
        crf_exp_avg_sq_ = crf_exp_avg_sq_.cuda();

        // Recreate gradient buffers
        exposure_grad_ = lfs::core::Tensor::zeros({static_cast<size_t>(num_frames_)}, lfs::core::Device::CUDA);
        vignetting_grad_ =
            lfs::core::Tensor::zeros({static_cast<size_t>(num_cameras_) * 3 * 5}, lfs::core::Device::CUDA);
        color_grad_ = lfs::core::Tensor::zeros({static_cast<size_t>(num_frames_) * 8}, lfs::core::Device::CUDA);
        crf_grad_ = lfs::core::Tensor::zeros({static_cast<size_t>(num_cameras_) * 3 * 4}, lfs::core::Device::CUDA);
    }

    void PPISP::serialize_inference(std::ostream& os) const {
        constexpr uint32_t INFERENCE_MAGIC = 0x4C465049; // "LFPI" - PPISP Inference
        constexpr uint32_t INFERENCE_VERSION = 1;

        os.write(reinterpret_cast<const char*>(&INFERENCE_MAGIC), sizeof(INFERENCE_MAGIC));
        os.write(reinterpret_cast<const char*>(&INFERENCE_VERSION), sizeof(INFERENCE_VERSION));

        os.write(reinterpret_cast<const char*>(&num_cameras_), sizeof(num_cameras_));
        os.write(reinterpret_cast<const char*>(&num_frames_), sizeof(num_frames_));

        os << exposure_params_;
        os << vignetting_params_;
        os << color_params_;
        os << crf_params_;
    }

    void PPISP::deserialize_inference(std::istream& is) {
        constexpr uint32_t INFERENCE_MAGIC = 0x4C465049;
        constexpr uint32_t INFERENCE_VERSION = 1;

        uint32_t magic, version;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != INFERENCE_MAGIC) {
            throw std::runtime_error("Invalid PPISP inference file");
        }
        if (version != INFERENCE_VERSION) {
            throw std::runtime_error("Unsupported PPISP inference version");
        }

        is.read(reinterpret_cast<char*>(&num_cameras_), sizeof(num_cameras_));
        is.read(reinterpret_cast<char*>(&num_frames_), sizeof(num_frames_));

        is >> exposure_params_;
        is >> vignetting_params_;
        is >> color_params_;
        is >> crf_params_;

        exposure_params_ = exposure_params_.cuda();
        vignetting_params_ = vignetting_params_.cuda();
        color_params_ = color_params_.cuda();
        crf_params_ = crf_params_.cuda();
    }

} // namespace lfs::training
