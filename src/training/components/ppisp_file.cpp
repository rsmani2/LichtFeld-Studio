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
        const PPISPController* controller) {

        try {
            std::ofstream file;
            if (!lfs::core::open_file_for_write(path, std::ios::binary, file)) {
                return std::unexpected("Failed to open file for writing: " + lfs::core::path_to_utf8(path));
            }

            PPISPFileHeader header{};
            header.num_cameras = static_cast<uint32_t>(ppisp.num_cameras());
            header.num_frames = static_cast<uint32_t>(ppisp.num_frames());
            header.flags = 0;
            if (controller) {
                header.flags |= static_cast<uint32_t>(PPISPFileFlags::HAS_CONTROLLER);
            }

            file.write(reinterpret_cast<const char*>(&header), sizeof(header));

            ppisp.serialize_inference(file);

            if (controller) {
                controller->serialize_inference(file);
            }

            LOG_INFO("PPISP file saved: {} ({} cameras, {} frames{})",
                     lfs::core::path_to_utf8(path),
                     header.num_cameras,
                     header.num_frames,
                     controller ? ", +controller" : "");

            return {};

        } catch (const std::exception& e) {
            return std::unexpected(std::string("Failed to save PPISP file: ") + e.what());
        }
    }

    std::expected<void, std::string> load_ppisp_file(
        const std::filesystem::path& path,
        PPISP& ppisp,
        PPISPController* controller) {

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
                if (controller) {
                    controller->deserialize_inference(file);
                    LOG_INFO("PPISP file loaded: {} ({} cameras, {} frames, +controller)",
                             lfs::core::path_to_utf8(path), header.num_cameras, header.num_frames);
                } else {
                    LOG_WARN("PPISP file has controller but none provided - skipping controller data");
                    LOG_INFO("PPISP file loaded: {} ({} cameras, {} frames)",
                             lfs::core::path_to_utf8(path), header.num_cameras, header.num_frames);
                }
            } else {
                if (controller) {
                    LOG_WARN("Controller requested but not present in PPISP file");
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
