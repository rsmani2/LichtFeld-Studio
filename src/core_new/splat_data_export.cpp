/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/splat_data_export.hpp"
#include "core_new/splat_data.hpp"
#include "core_new/logger.hpp"
#include "core_new/sogs.hpp"
#include "external/tinyply.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>

namespace {

    // Async save state - managed per-process
    std::mutex g_save_mutex;
    std::vector<std::future<void>> g_save_futures;

    void cleanup_finished_saves() {
        std::lock_guard<std::mutex> lock(g_save_mutex);

        g_save_futures.erase(
            std::remove_if(g_save_futures.begin(), g_save_futures.end(),
                           [](const std::future<void>& f) {
                               return !f.valid() ||
                                      f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
                           }),
            g_save_futures.end());

        if (g_save_futures.size() > 5) {
            LOG_WARN("Multiple saves pending: {} operations in queue", g_save_futures.size());
        }
    }

    /**
     * @brief Write PLY file implementation
     */
    void write_ply_impl(const lfs::core::PointCloud& pc,
                        const std::filesystem::path& root,
                        int iteration,
                        const std::string& stem) {
        namespace fs = std::filesystem;
        fs::create_directories(root);

        std::vector<lfs::core::Tensor> tensors;
        tensors.push_back(pc.means.cpu().contiguous());

        if (pc.normals.is_valid()) {
            tensors.push_back(pc.normals.cpu().contiguous());
        }

        if (pc.sh0.is_valid()) {
            LOG_INFO("write_ply_impl: pc.sh0 shape: ndim={}, shape=[{}{}{}]",
                     pc.sh0.ndim(),
                     pc.sh0.shape()[0],
                     pc.sh0.ndim() >= 2 ? fmt::format(", {}", pc.sh0.shape()[1]) : "",
                     pc.sh0.ndim() >= 3 ? fmt::format(", {}", pc.sh0.shape()[2]) : "");

            if (pc.sh0.ndim() == 3) {
                LOG_INFO("write_ply_impl: sh0 is 3D, transposing and flattening");
                auto sh0_transposed = pc.sh0.transpose(1, 2).contiguous();
                tensors.push_back(sh0_transposed.flatten(1).cpu().contiguous());
            } else if (pc.sh0.ndim() == 2) {
                LOG_INFO("write_ply_impl: sh0 is 2D, using as-is");
                tensors.push_back(pc.sh0.cpu().contiguous());
            } else {
                LOG_ERROR("write_ply_impl: Unexpected sh0 ndim: {}", pc.sh0.ndim());
                tensors.push_back(pc.sh0.cpu().contiguous());
            }
        }

        if (pc.shN.is_valid()) {
            LOG_INFO("write_ply_impl: pc.shN shape: ndim={}, shape=[{}{}{}]",
                     pc.shN.ndim(),
                     pc.shN.shape()[0],
                     pc.shN.ndim() >= 2 ? fmt::format(", {}", pc.shN.shape()[1]) : "",
                     pc.shN.ndim() >= 3 ? fmt::format(", {}", pc.shN.shape()[2]) : "");

            if (pc.shN.ndim() == 3) {
                LOG_INFO("write_ply_impl: shN is 3D, transposing and flattening");
                auto shN_transposed = pc.shN.transpose(1, 2).contiguous();
                tensors.push_back(shN_transposed.flatten(1).cpu().contiguous());
            } else if (pc.shN.ndim() == 2) {
                LOG_INFO("write_ply_impl: shN is 2D, using as-is");
                tensors.push_back(pc.shN.cpu().contiguous());
            } else {
                LOG_ERROR("write_ply_impl: Unexpected shN ndim: {}", pc.shN.ndim());
                tensors.push_back(pc.shN.cpu().contiguous());
            }
        }

        if (pc.opacity.is_valid()) {
            tensors.push_back(pc.opacity.cpu().contiguous());
        }

        if (pc.scaling.is_valid()) {
            tensors.push_back(pc.scaling.cpu().contiguous());
        }

        if (pc.rotation.is_valid()) {
            tensors.push_back(pc.rotation.cpu().contiguous());
        }

        auto write_output_ply = [](const fs::path& file_path,
                                   const std::vector<lfs::core::Tensor>& data,
                                   const std::vector<std::string>& attr_names) {
            tinyply::PlyFile ply;
            size_t attr_off = 0;

            for (const auto& tensor : data) {
                const size_t cols = tensor.size(1);
                std::vector<std::string> attrs(attr_names.begin() + attr_off,
                                               attr_names.begin() + attr_off + cols);

                ply.add_properties_to_element(
                    "vertex",
                    attrs,
                    tinyply::Type::FLOAT32,
                    tensor.size(0),
                    reinterpret_cast<uint8_t*>(const_cast<float*>(tensor.ptr<float>())),
                    tinyply::Type::INVALID, 0);

                attr_off += cols;
            }

            std::filebuf fb;
            fb.open(file_path, std::ios::out | std::ios::binary);
            std::ostream out_stream(&fb);
            ply.write(out_stream, /*binary=*/true);
        };

        if (stem.empty()) {
            write_output_ply(
                root / ("splat_" + std::to_string(iteration) + ".ply"),
                tensors,
                pc.attribute_names);
        } else {
            write_output_ply(
                root / std::string(stem + ".ply"),
                tensors,
                pc.attribute_names);
        }
    }

    /**
     * @brief Write SOG format implementation
     */
    std::filesystem::path write_sog_impl(const lfs::core::SplatData& splat_data,
                                         const std::filesystem::path& root,
                                         int iteration,
                                         int kmeans_iterations) {
        namespace fs = std::filesystem;

        fs::path sog_dir = root / "sog";
        fs::create_directories(sog_dir);

        std::filesystem::path sog_out_path = sog_dir /
                                             ("splat_" + std::to_string(iteration) + "_sog.sog");

        lfs::core::SogWriteOptions options{
            .iterations = kmeans_iterations,
            .output_path = sog_out_path};

        auto result = lfs::core::write_sog(splat_data, options);
        if (!result) {
            LOG_ERROR("Failed to write SOG format: {}", result.error());
        } else {
            LOG_DEBUG("Successfully wrote SOG format for iteration {}", iteration);
        }

        return sog_out_path;
    }

} // anonymous namespace

namespace lfs::core {

    void save_ply(const SplatData& splat_data,
                  const std::filesystem::path& root,
                  int iteration,
                  bool join_threads,
                  std::string stem) {
        auto pc = to_point_cloud(splat_data);

        if (join_threads) {
            write_ply_impl(pc, root, iteration, stem);
        } else {
            cleanup_finished_saves();

            std::lock_guard<std::mutex> lock(g_save_mutex);
            g_save_futures.emplace_back(
                std::async(std::launch::async,
                           [pc = std::move(pc), root, iteration, stem]() {
                               try {
                                   write_ply_impl(pc, root, iteration, stem);
                               } catch (const std::exception& e) {
                                   LOG_ERROR("Failed to save PLY for iteration {}: {}",
                                             iteration, e.what());
                               }
                           }));
        }
    }

    std::filesystem::path save_sog(const SplatData& splat_data,
                                   const std::filesystem::path& root,
                                   int iteration,
                                   int kmeans_iterations,
                                   bool /* join_threads */) {
        // SOG is always synchronous - k-means clustering is too heavy for async
        return write_sog_impl(splat_data, root, iteration, kmeans_iterations);
    }

    PointCloud to_point_cloud(const SplatData& splat_data) {
        PointCloud pc;

        pc.means = splat_data._means.cpu().contiguous();
        pc.normals = Tensor::zeros_like(pc.means);

        if (splat_data._sh0.is_valid()) {
            LOG_INFO("to_point_cloud: _sh0 shape before cpu/contiguous: ndim={}, shape=[{}{}{}]",
                     splat_data._sh0.ndim(),
                     splat_data._sh0.shape()[0],
                     splat_data._sh0.ndim() >= 2 ? fmt::format(", {}", splat_data._sh0.shape()[1]) : "",
                     splat_data._sh0.ndim() >= 3 ? fmt::format(", {}", splat_data._sh0.shape()[2]) : "");

            auto sh0_cpu = splat_data._sh0.cpu().contiguous();

            LOG_INFO("to_point_cloud: sh0_cpu shape after cpu/contiguous: ndim={}, shape=[{}{}{}]",
                     sh0_cpu.ndim(),
                     sh0_cpu.shape()[0],
                     sh0_cpu.ndim() >= 2 ? fmt::format(", {}", sh0_cpu.shape()[1]) : "",
                     sh0_cpu.ndim() >= 3 ? fmt::format(", {}", sh0_cpu.shape()[2]) : "");

            if (sh0_cpu.ndim() == 3) {
                LOG_INFO("to_point_cloud: sh0 is 3D, will transpose and flatten");
                auto sh0_transposed = sh0_cpu.transpose(1, 2);
                size_t N = sh0_transposed.shape()[0];
                size_t flat_dim = sh0_transposed.shape()[1] * sh0_transposed.shape()[2];
                pc.sh0 = sh0_transposed.reshape({static_cast<int>(N), static_cast<int>(flat_dim)});
                LOG_INFO("to_point_cloud: sh0 after processing: shape=[{}, {}]", N, flat_dim);
            } else if (sh0_cpu.ndim() == 2) {
                LOG_INFO("to_point_cloud: sh0 is 2D, using as-is with shape=[{}, {}]",
                         sh0_cpu.shape()[0], sh0_cpu.shape()[1]);
                pc.sh0 = sh0_cpu;
            } else {
                LOG_ERROR("Unexpected sh0 dimensions: {}, shape: [{}]", sh0_cpu.ndim(),
                         sh0_cpu.ndim() >= 1 ? sh0_cpu.shape()[0] : 0);
                pc.sh0 = sh0_cpu;
            }
        }

        if (splat_data._shN.is_valid()) {
            LOG_INFO("to_point_cloud: _shN shape before cpu/contiguous: ndim={}, shape=[{}{}{}]",
                     splat_data._shN.ndim(),
                     splat_data._shN.shape()[0],
                     splat_data._shN.ndim() >= 2 ? fmt::format(", {}", splat_data._shN.shape()[1]) : "",
                     splat_data._shN.ndim() >= 3 ? fmt::format(", {}", splat_data._shN.shape()[2]) : "");

            auto shN_cpu = splat_data._shN.cpu().contiguous();

            LOG_INFO("to_point_cloud: shN_cpu shape after cpu/contiguous: ndim={}, shape=[{}{}{}]",
                     shN_cpu.ndim(),
                     shN_cpu.shape()[0],
                     shN_cpu.ndim() >= 2 ? fmt::format(", {}", shN_cpu.shape()[1]) : "",
                     shN_cpu.ndim() >= 3 ? fmt::format(", {}", shN_cpu.shape()[2]) : "");

            if (shN_cpu.ndim() == 3) {
                LOG_INFO("to_point_cloud: shN is 3D, will transpose and flatten");
                auto shN_transposed = shN_cpu.transpose(1, 2);
                size_t N = shN_transposed.shape()[0];
                size_t flat_dim = shN_transposed.shape()[1] * shN_transposed.shape()[2];
                pc.shN = shN_transposed.reshape({static_cast<int>(N), static_cast<int>(flat_dim)});
                LOG_INFO("to_point_cloud: shN after processing: shape=[{}, {}]", N, flat_dim);
            } else if (shN_cpu.ndim() == 2) {
                LOG_INFO("to_point_cloud: shN is 2D, using as-is with shape=[{}, {}]",
                         shN_cpu.shape()[0], shN_cpu.shape()[1]);
                pc.shN = shN_cpu;
            } else {
                LOG_ERROR("Unexpected shN dimensions: {}, shape: [{}]", shN_cpu.ndim(),
                         shN_cpu.ndim() >= 1 ? shN_cpu.shape()[0] : 0);
                pc.shN = shN_cpu;
            }
        }

        if (splat_data._opacity.is_valid()) {
            pc.opacity = splat_data._opacity.cpu().contiguous();
        }

        if (splat_data._scaling.is_valid()) {
            pc.scaling = splat_data._scaling.cpu().contiguous();
        }

        if (splat_data._rotation.is_valid()) {
            auto normalized_rotation = splat_data.get_rotation();
            pc.rotation = normalized_rotation.cpu().contiguous();
        }

        pc.attribute_names = get_attribute_names(splat_data);

        return pc;
    }

    std::vector<std::string> get_attribute_names(const SplatData& splat_data) {
        std::vector<std::string> a{"x", "y", "z", "nx", "ny", "nz"};

        if (splat_data._sh0.is_valid()) {
            size_t sh0_features;
            if (splat_data._sh0.ndim() == 3) {
                sh0_features = splat_data._sh0.shape()[1] * splat_data._sh0.shape()[2];
            } else if (splat_data._sh0.ndim() == 2) {
                sh0_features = splat_data._sh0.shape()[1];
            } else {
                LOG_ERROR("Unexpected sh0 ndim in get_attribute_names: {}", splat_data._sh0.ndim());
                sh0_features = 3;
            }
            for (size_t i = 0; i < sh0_features; ++i) {
                a.emplace_back("f_dc_" + std::to_string(i));
            }
        }

        if (splat_data._shN.is_valid()) {
            size_t shN_features;
            if (splat_data._shN.ndim() == 3) {
                shN_features = splat_data._shN.shape()[1] * splat_data._shN.shape()[2];
            } else if (splat_data._shN.ndim() == 2) {
                shN_features = splat_data._shN.shape()[1];
            } else {
                LOG_ERROR("Unexpected shN ndim in get_attribute_names: {}", splat_data._shN.ndim());
                shN_features = 45;
            }
            for (size_t i = 0; i < shN_features; ++i) {
                a.emplace_back("f_rest_" + std::to_string(i));
            }
        }

        a.emplace_back("opacity");

        if (splat_data._scaling.is_valid()) {
            for (size_t i = 0; i < splat_data._scaling.shape()[1]; ++i) {
                a.emplace_back("scale_" + std::to_string(i));
            }
        }

        if (splat_data._rotation.is_valid()) {
            for (size_t i = 0; i < splat_data._rotation.shape()[1]; ++i) {
                a.emplace_back("rot_" + std::to_string(i));
            }
        }

        return a;
    }

} // namespace lfs::core
