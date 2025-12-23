/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <expected>
#include <filesystem>
#include <vector>

namespace lfs::python {

    /**
     * @brief Execute a list of Python script files. Each script is expected to import `lichtfeld`
     *        and register its callbacks (e.g., with register_opacity_scaler or Session hooks).
     *
     * @return std::expected<void, std::string> error on failure (file missing, execution error,
     *         or interpreter unavailable when bindings are disabled).
     */
    std::expected<void, std::string> run_scripts(const std::vector<std::filesystem::path>& scripts);

} // namespace lfs::python
