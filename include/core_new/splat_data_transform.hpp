/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <glm/glm.hpp>

namespace lfs::geometry {
    class BoundingBox;
}

namespace lfs::core {

    // Forward declaration
    class SplatData;

    /**
     * @brief Apply a transformation matrix to SplatData
     * @param splat_data The splat data to transform (modified in-place)
     * @param transform_matrix 4x4 transformation matrix
     * @return Reference to the modified splat_data
     */
    SplatData& transform(SplatData& splat_data, const glm::mat4& transform_matrix);

    /**
     * @brief Crop SplatData by a bounding box
     * @param splat_data The splat data to crop
     * @param bounding_box The bounding box to crop by
     * @return New SplatData containing only points inside the bounding box
     */
    SplatData crop_by_cropbox(const SplatData& splat_data,
                              const lfs::geometry::BoundingBox& bounding_box);

    /**
     * @brief Randomly select a subset of splats
     * @param splat_data The splat data to modify (modified in-place)
     * @param num_required_splat Number of splats to keep
     * @param seed Random seed for reproducibility (default: 0)
     */
    void random_choose(SplatData& splat_data, int num_required_splat, int seed = 0);

} // namespace lfs::core
