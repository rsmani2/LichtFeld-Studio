/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "forward.h"
#include "rasterization_api_tensor.h"
#include "rasterization_config.h"
#include "core_new/cuda/memory_arena.hpp"
#include <functional>
#include <stdexcept>
#include <tuple>

namespace lfs::rendering {

    inline std::function<char*(size_t)> resize_function_wrapper_tensor(Tensor& t) {
        return [&t](size_t N) -> char* {
            if (N == 0) {
                t = Tensor::empty({0}, lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
                return nullptr;
            }
            t = Tensor::empty({N}, lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
            return reinterpret_cast<char*>(t.ptr<uint8_t>());
        };
    }

    inline void check_tensor_input(bool debug, const Tensor& tensor, const char* name) {
        if (debug) {
            if (!tensor.is_valid() || tensor.device() != lfs::core::Device::CUDA ||
                tensor.dtype() != lfs::core::DataType::Float32 || !tensor.is_contiguous()) {
                throw std::runtime_error("Input tensor '" + std::string(name) +
                                         "' must be a contiguous CUDA float tensor.");
            }
        }
    }

    std::tuple<Tensor, Tensor>
    forward_wrapper_tensor(
        const Tensor& means,
        const Tensor& scales_raw,
        const Tensor& rotations_raw,
        const Tensor& opacities_raw,
        const Tensor& sh_coefficients_0,
        const Tensor& sh_coefficients_rest,
        const Tensor& w2c,
        const Tensor& cam_position,
        const int active_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane) {

        check_tensor_input(config::debug, means, "means");
        check_tensor_input(config::debug, scales_raw, "scales_raw");
        check_tensor_input(config::debug, rotations_raw, "rotations_raw");
        check_tensor_input(config::debug, opacities_raw, "opacities_raw");
        check_tensor_input(config::debug, sh_coefficients_0, "sh_coefficients_0");
        check_tensor_input(config::debug, sh_coefficients_rest, "sh_coefficients_rest");

        const int n_primitives = static_cast<int>(means.size(0));
        const int total_bases_sh_rest = static_cast<int>(sh_coefficients_rest.size(1));

        Tensor image = Tensor::empty({3, static_cast<size_t>(height), static_cast<size_t>(width)},
                                     lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        Tensor alpha = Tensor::empty({1, static_cast<size_t>(height), static_cast<size_t>(width)},
                                     lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        // Coordinate with training: wait for training, use shared arena
        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
        arena.set_rendering_active(true);
        arena.wait_for_training();
        uint64_t frame_id = arena.begin_frame(true);  // true = from_rendering
        auto arena_allocator = arena.get_allocator(frame_id);

        const std::function<char*(size_t)> per_primitive_buffers_func = arena_allocator;
        const std::function<char*(size_t)> per_tile_buffers_func = arena_allocator;
        const std::function<char*(size_t)> per_instance_buffers_func = arena_allocator;

        Tensor w2c_contig = w2c.is_contiguous() ? w2c : w2c.contiguous();
        Tensor cam_pos_contig = cam_position.is_contiguous() ? cam_position : cam_position.contiguous();

        forward(
            per_primitive_buffers_func,
            per_tile_buffers_func,
            per_instance_buffers_func,
            reinterpret_cast<const float3*>(means.ptr<float>()),
            reinterpret_cast<const float3*>(scales_raw.ptr<float>()),
            reinterpret_cast<const float4*>(rotations_raw.ptr<float>()),
            opacities_raw.ptr<float>(),
            reinterpret_cast<const float3*>(sh_coefficients_0.ptr<float>()),
            reinterpret_cast<const float3*>(sh_coefficients_rest.ptr<float>()),
            reinterpret_cast<const float4*>(w2c_contig.ptr<float>()),
            reinterpret_cast<const float3*>(cam_pos_contig.ptr<float>()),
            image.ptr<float>(),
            alpha.ptr<float>(),
            n_primitives,
            active_sh_bases,
            total_bases_sh_rest,
            width,
            height,
            focal_x,
            focal_y,
            center_x,
            center_y,
            near_plane,
            far_plane);

        arena.end_frame(frame_id, true);  // true = from_rendering
        arena.set_rendering_active(false);

        return {std::move(image), std::move(alpha)};
    }

} // namespace lfs::rendering
