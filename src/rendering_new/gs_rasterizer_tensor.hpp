/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core_new/camera.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/tensor.hpp"

namespace lfs::rendering {

    // Import Tensor from lfs::core
    using lfs::core::Tensor;

    /**
     * @brief Rasterize Gaussians using tensor-based backend (libtorch-free)
     *
     * @param viewpoint_camera Camera parameters
     * @param gaussian_model Gaussian splat data
     * @param bg_color Background color [3]
     * @param show_rings Enable ring mode visualization
     * @param ring_width Width of the ring band
     * @return Rendered image [3, H, W]
     */
    Tensor rasterize_tensor(
        const lfs::core::Camera& viewpoint_camera,
        const lfs::core::SplatData& gaussian_model,
        const Tensor& bg_color,
        bool show_rings = false,
        float ring_width = 0.002f);

} // namespace lfs::rendering
