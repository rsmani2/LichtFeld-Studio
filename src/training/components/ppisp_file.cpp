/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "ppisp_file.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "ppisp.hpp"
#include "ppisp_controller.hpp"
#include <fstream>

namespace lfs::training {

    std::expected<void, std::string> save_ppisp_file(
        const std::filesystem::path& path,
        const PPISP& ppisp,
        const std::vector<std::unique_ptr<PPISPController>>* controllers) {

        try {
            std::ofstream file;
            if (!lfs::core::open_file_for_write(path, std::ios::binary, file)) {
                return std::unexpected("Failed to open file for writing: " + lfs::core::path_to_utf8(path));
            }

            PPISPFileHeader header{};
            header.num_cameras = static_cast<uint32_t>(ppisp.num_cameras());
            header.num_frames = static_cast<uint32_t>(ppisp.num_frames());
            header.flags = 0;
            if (controllers && !controllers->empty()) {
                header.flags |= static_cast<uint32_t>(PPISPFileFlags::HAS_CONTROLLER);
            }

            file.write(reinterpret_cast<const char*>(&header), sizeof(header));

            ppisp.serialize_inference(file);

            // Save per-camera controllers
            if (controllers && !controllers->empty()) {
                const uint32_t num_controllers = static_cast<uint32_t>(controllers->size());
                file.write(reinterpret_cast<const char*>(&num_controllers), sizeof(num_controllers));
                for (const auto& controller : *controllers) {
                    controller->serialize_inference(file);
                }
            }

            LOG_INFO("PPISP file saved: {} ({} cameras, {} frames{})",
                     lfs::core::path_to_utf8(path),
                     header.num_cameras,
                     header.num_frames,
                     (controllers && !controllers->empty())
                         ? ", +controllers(" + std::to_string(controllers->size()) + ")"
                         : "");

            return {};

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Failed to save PPISP file: ") + e.what());
        }
    }

    std::expected<void, std::string> load_ppisp_file(
        const std::filesystem::path& path,
        PPISP& ppisp,
        std::vector<std::unique_ptr<PPISPController>>* controllers) {

        try {
            std::ifstream file;
            if (!lfs::core::open_file_for_read(path, std::ios::binary, file)) {
                return std::unexpected("Failed to open file for reading: " + lfs::core::path_to_utf8(path));
            }

            PPISPFileHeader header{};
            file.read(reinterpret_cast<char*>(&header), sizeof(header));

            if (header.magic != PPISP_FILE_MAGIC) {
                return std::unexpected("Invalid PPISP file: wrong magic number");
            }

            if (header.version > PPISP_FILE_VERSION) {
                return std::unexpected("Unsupported PPISP file version: " + std::to_string(header.version));
            }

            if (static_cast<int>(header.num_cameras) != ppisp.num_cameras() ||
                static_cast<int>(header.num_frames) != ppisp.num_frames()) {
                return std::unexpected(
                    "PPISP dimension mismatch: file has " +
                    std::to_string(header.num_cameras) + " cameras, " +
                    std::to_string(header.num_frames) + " frames; expected " +
                    std::to_string(ppisp.num_cameras()) + " cameras, " +
                    std::to_string(ppisp.num_frames()) + " frames");
            }

            ppisp.deserialize_inference(file);

            if (has_flag(header.flags, PPISPFileFlags::HAS_CONTROLLER)) {
                uint32_t num_controllers = 0;
                file.read(reinterpret_cast<char*>(&num_controllers), sizeof(num_controllers));

                if (controllers && !controllers->empty()) {
                    if (num_controllers != controllers->size()) {
                        LOG_WARN("Controller count mismatch: file has {}, provided {} - loading min",
                                 num_controllers, controllers->size());
                    }
                    const size_t load_count = std::min(static_cast<size_t>(num_controllers), controllers->size());
                    for (size_t i = 0; i < load_count; ++i) {
                        (*controllers)[i]->deserialize_inference(file);
                    }
                    // Skip remaining controllers if file has more
                    for (size_t i = load_count; i < num_controllers; ++i) {
                        PPISPController temp(1);
                        temp.deserialize_inference(file);
                    }
                    LOG_INFO("PPISP file loaded: {} ({} cameras, {} frames, +controllers({}))",
                             lfs::core::path_to_utf8(path), header.num_cameras, header.num_frames, load_count);
                } else {
                    LOG_WARN("PPISP file has {} controllers but none provided - skipping", num_controllers);
                    // Skip all controller data
                    for (size_t i = 0; i < num_controllers; ++i) {
                        PPISPController temp(1);
                        temp.deserialize_inference(file);
                    }
                    LOG_INFO("PPISP file loaded: {} ({} cameras, {} frames)",
                             lfs::core::path_to_utf8(path), header.num_cameras, header.num_frames);
                }
            } else {
                if (controllers && !controllers->empty()) {
                    LOG_WARN("Controllers requested but not present in PPISP file");
                }
                LOG_INFO("PPISP file loaded: {} ({} cameras, {} frames)",
                         lfs::core::path_to_utf8(path), header.num_cameras, header.num_frames);
            }

            return {};

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Failed to load PPISP file: ") + e.what());
        }
    }

    std::filesystem::path find_ppisp_companion(const std::filesystem::path& splat_path) {
        auto companion = get_ppisp_companion_path(splat_path);
        if (std::filesystem::exists(companion)) {
            return companion;
        }
        return {};
    }

    std::filesystem::path get_ppisp_companion_path(const std::filesystem::path& splat_path) {
        auto path = splat_path;
        path.replace_extension(".ppisp");
        return path;
    }

} // namespace lfs::training
