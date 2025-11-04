/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "fast_rasterizer.hpp"
#include "core_new/logger.hpp"

namespace lfs::training {
    std::pair<RenderOutput, FastRasterizeContext> fast_rasterize_forward(
        lfs::core::Camera& viewpoint_camera,
        lfs::core::SplatData& gaussian_model,
        lfs::core::Tensor& bg_color) {
        // Get camera parameters
        const int width = static_cast<int>(viewpoint_camera.image_width());
        const int height = static_cast<int>(viewpoint_camera.image_height());
        auto [fx, fy, cx, cy] = viewpoint_camera.get_intrinsics();

        // Get Gaussian parameters
        auto means = gaussian_model.means();
        auto raw_opacities = gaussian_model.opacity_raw();
        auto raw_scales = gaussian_model.scaling_raw();
        auto raw_rotations = gaussian_model.rotation_raw();
        auto sh0 = gaussian_model.sh0();
        auto shN = gaussian_model.shN();

        const int sh_degree = gaussian_model.get_active_sh_degree();
        const int active_sh_bases = (sh_degree + 1) * (sh_degree + 1);

        constexpr float near_plane = 0.01f;
        constexpr float far_plane = 1e10f;

        auto w2c = viewpoint_camera.world_view_transform();
        auto cam_position = viewpoint_camera.cam_position();

        // Cache contiguous pointers to avoid repeated .contiguous() calls
        // Camera data is tiny (12-64 bytes) and already on GPU - cache the pointer!
        const float* w2c_ptr = w2c.contiguous().ptr<float>();
        const float* cam_position_ptr = cam_position.contiguous().ptr<float>();

        const int n_primitives = static_cast<int>(means.shape()[0]);
        const int total_bases_sh_rest = static_cast<int>(shN.shape()[1]);

        // Allocate output tensors
        lfs::core::Tensor image = lfs::core::Tensor::empty({3, static_cast<size_t>(height), static_cast<size_t>(width)});
        lfs::core::Tensor alpha = lfs::core::Tensor::empty({1, static_cast<size_t>(height), static_cast<size_t>(width)});

        // Call forward_raw with raw pointers (no PyTorch wrappers)
        auto forward_ctx = fast_lfs::rasterization::forward_raw(
            means.ptr<float>(),
            raw_scales.ptr<float>(),
            raw_rotations.ptr<float>(),
            raw_opacities.ptr<float>(),
            sh0.ptr<float>(),
            shN.ptr<float>(),
            w2c_ptr,
            cam_position_ptr,
            image.ptr<float>(),
            alpha.ptr<float>(),
            n_primitives,
            active_sh_bases,
            total_bases_sh_rest,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            near_plane,
            far_plane);

        // Prepare render output
        RenderOutput render_output;
        // output = image + (1 - alpha) * bg_color
        auto alpha_complement = (alpha * -1.0f) + 1.0f;  // 1 - alpha
        auto bg_contribution = alpha_complement * bg_color.unsqueeze(-1).unsqueeze(-1);
        render_output.image = image + bg_contribution;
        render_output.alpha = alpha;
        render_output.width = width;
        render_output.height = height;

        // Prepare context for backward
        FastRasterizeContext ctx;
        ctx.image = image;
        ctx.alpha = alpha;
        ctx.bg_color = bg_color;  // Save bg_color for alpha gradient

        // Save parameters (avoid re-fetching in backward)
        ctx.means = means;
        ctx.raw_scales = raw_scales;
        ctx.raw_rotations = raw_rotations;
        ctx.shN = shN;
        ctx.w2c = w2c;
        ctx.cam_position = cam_position;

        // Cache GPU pointers to avoid repeated contiguous() calls in backward
        ctx.w2c_ptr = w2c_ptr;
        ctx.cam_position_ptr = cam_position_ptr;

        // Store forward context (contains buffer pointers, frame_id, etc.)
        ctx.forward_ctx = forward_ctx;

        ctx.active_sh_bases = active_sh_bases;
        ctx.total_bases_sh_rest = total_bases_sh_rest;
        ctx.width = width;
        ctx.height = height;
        ctx.focal_x = fx;
        ctx.focal_y = fy;
        ctx.center_x = cx;
        ctx.center_y = cy;
        ctx.near_plane = near_plane;
        ctx.far_plane = far_plane;

        return {render_output, ctx};
    }

    void fast_rasterize_backward(
        const FastRasterizeContext& ctx,
        const lfs::core::Tensor& grad_image,
        lfs::core::SplatData& gaussian_model) {

        // Compute gradient w.r.t. alpha from background blending
        // Forward: output_image = image + (1 - alpha) * bg_color
        // where bg_color is [3], alpha is [1, H, W], output_image is [3, H, W]
        //
        // Backward:
        // ∂L/∂image_raw = ∂L/∂output_image (grad_image)
        // ∂L/∂alpha = -sum_over_channels(∂L/∂output_image * bg_color)
        //
        // grad_image shape: [3, H, W] or [H, W, 3]
        // bg_color shape: [3]
        // alpha shape: [1, H, W]

        lfs::core::Tensor grad_alpha;

        // Determine the layout of grad_image
        if (grad_image.shape()[0] == 3) {
            // Layout: [3, H, W]
            // ∂L/∂alpha[h,w] = -sum_c(grad_image[c,h,w] * bg_color[c])
            auto bg_expanded = ctx.bg_color.reshape({3, 1, 1});  // [3, 1, 1]
            grad_alpha = (grad_image * bg_expanded).sum({0}, false) * -1.0f;  // [H, W]
        } else if (grad_image.shape()[2] == 3) {
            // Layout: [H, W, 3]
            // ∂L/∂alpha[h,w] = -sum_c(grad_image[h,w,c] * bg_color[c])
            auto bg_expanded = ctx.bg_color.reshape({1, 1, 3});  // [1, 1, 3]
            grad_alpha = (grad_image * bg_expanded).sum({2}, false) * -1.0f;  // [H, W]
        } else {
            throw std::runtime_error("Unexpected grad_image shape in fast_rasterize_backward");
        }

        const int n_primitives = static_cast<int>(ctx.means.shape()[0]);

        // Call backward_raw with raw pointers to SplatData gradient buffers
        const bool update_densification_info = gaussian_model._densification_info.ndim() > 0 &&
                                                gaussian_model._densification_info.shape()[0] > 0;
        auto backward_result = fast_lfs::rasterization::backward_raw(
            update_densification_info ? gaussian_model._densification_info.ptr<float>() : nullptr,
            grad_image.ptr<float>(),
            grad_alpha.ptr<float>(),
            ctx.image.ptr<float>(),
            ctx.alpha.ptr<float>(),
            ctx.means.ptr<float>(),
            ctx.raw_scales.ptr<float>(),
            ctx.raw_rotations.ptr<float>(),
            ctx.shN.ptr<float>(),
            ctx.w2c_ptr,
            ctx.cam_position_ptr,
            ctx.forward_ctx,
            gaussian_model.means_grad().ptr<float>(),        // Direct access to SplatData gradients
            gaussian_model.scaling_grad().ptr<float>(),
            gaussian_model.rotation_grad().ptr<float>(),
            gaussian_model.opacity_grad().ptr<float>(),
            gaussian_model.sh0_grad().ptr<float>(),
            gaussian_model.shN_grad().ptr<float>(),
            nullptr,  // grad_w2c not needed for now
            n_primitives,
            ctx.active_sh_bases,
            ctx.total_bases_sh_rest,
            ctx.width,
            ctx.height,
            ctx.focal_x,
            ctx.focal_y,
            ctx.center_x,
            ctx.center_y);

        if (!backward_result.success) {
            throw std::runtime_error(std::string("Backward failed: ") + backward_result.error_message);
        }

        // No need to accumulate - gradients were written directly to SplatData buffers
        // The trainer calls zero_gradients() at the start of each iteration
    }
} // namespace lfs::training
