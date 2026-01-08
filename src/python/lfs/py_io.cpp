/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_io.hpp"
#include "py_splat_data.hpp"
#include "py_tensor.hpp"

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include "io/exporter.hpp"
#include "io/loader.hpp"

#include <filesystem>
#include <format>

namespace lfs::python {

    namespace {

        // Progress callback wrapper for Python
        struct PyProgressCallback {
            nb::object callback;

            void operator()(float progress, const std::string& message) const {
                if (!callback)
                    return;
                nb::gil_scoped_acquire gil;
                try {
                    callback(progress, message);
                } catch (const std::exception& e) {
                    LOG_ERROR("Python progress callback error: {}", e.what());
                }
            }
        };

        // Export progress callback wrapper (returns bool for cancellation)
        struct PyExportProgressCallback {
            nb::object callback;

            bool operator()(float progress, const std::string& stage) const {
                if (!callback)
                    return true; // Continue
                nb::gil_scoped_acquire gil;
                try {
                    nb::object result = callback(progress, stage);
                    if (nb::isinstance<nb::bool_>(result)) {
                        return nb::cast<bool>(result);
                    }
                    return true;
                } catch (const std::exception& e) {
                    LOG_ERROR("Python export progress callback error: {}", e.what());
                    return false; // Cancel on error
                }
            }
        };

        // Python wrapper for LoadResult
        struct PyLoadResult {
            std::shared_ptr<core::SplatData> splat_data;
            std::shared_ptr<io::LoadedScene> scene;
            PyTensor scene_center;
            std::string loader_used;
            int64_t load_time_ms;
            std::vector<std::string> warnings;
            bool is_dataset;

            std::optional<PySplatData> get_splat_data() const {
                if (splat_data)
                    return PySplatData(splat_data.get());
                return std::nullopt;
            }
        };

    } // namespace

    void register_io(nb::module_& m) {
        // LoadResult class
        nb::class_<PyLoadResult>(m, "LoadResult")
            .def_prop_ro("splat_data", &PyLoadResult::get_splat_data, "Get loaded splat data (None if scene)")
            .def_prop_ro(
                "scene_center", [](const PyLoadResult& r) { return r.scene_center; }, "Scene center tensor")
            .def_prop_ro(
                "loader_used", [](const PyLoadResult& r) { return r.loader_used; }, "Name of loader that was used")
            .def_prop_ro(
                "load_time_ms", [](const PyLoadResult& r) { return r.load_time_ms; }, "Load time in milliseconds")
            .def_prop_ro(
                "warnings", [](const PyLoadResult& r) { return r.warnings; }, "Warnings from loading")
            .def_prop_ro(
                "is_dataset", [](const PyLoadResult& r) { return r.is_dataset; }, "True if loaded a dataset vs file");

        // load() - Load any supported file format
        m.def(
            "load",
            [](const std::filesystem::path& path, std::optional<std::string> format,
               std::optional<int> resize_factor, std::optional<int> max_width,
               std::optional<std::string> images_folder, nb::object progress) -> PyLoadResult {
                auto loader = io::Loader::create();

                io::LoadOptions options;
                if (resize_factor)
                    options.resize_factor = *resize_factor;
                if (max_width)
                    options.max_width = *max_width;
                if (images_folder)
                    options.images_folder = *images_folder;

                if (progress && !progress.is_none()) {
                    PyProgressCallback py_progress{nb::cast<nb::object>(progress)};
                    options.progress = [py_progress](float p, const std::string& msg) {
                        py_progress(p, msg);
                    };
                }

                auto result = loader->load(path, options);
                if (!result) {
                    throw std::runtime_error(
                        std::format("Failed to load '{}': {}", path.string(), result.error().format()));
                }

                PyLoadResult py_result;
                py_result.loader_used = result->loader_used;
                py_result.load_time_ms = result->load_time.count();
                py_result.warnings = result->warnings;
                py_result.scene_center = PyTensor(result->scene_center, false);

                if (std::holds_alternative<std::shared_ptr<core::SplatData>>(result->data)) {
                    py_result.splat_data = std::get<std::shared_ptr<core::SplatData>>(result->data);
                    py_result.is_dataset = false;
                } else {
                    auto scene = std::get<io::LoadedScene>(result->data);
                    py_result.scene = std::make_shared<io::LoadedScene>(std::move(scene));
                    py_result.is_dataset = true;
                }

                return py_result;
            },
            nb::arg("path"), nb::arg("format") = nb::none(), nb::arg("resize_factor") = nb::none(),
            nb::arg("max_width") = nb::none(), nb::arg("images_folder") = nb::none(),
            nb::arg("progress") = nb::none(),
            R"doc(Load a file or dataset.

Args:
    path: Path to file (.ply, .spz, .splat) or dataset directory (COLMAP/transforms.json)
    format: Optional format hint (auto-detected if not specified)
    resize_factor: Downscale images by this factor (datasets only)
    max_width: Maximum image width (datasets only)
    images_folder: Name of images folder in dataset (default: "images")
    progress: Optional callback(progress: float, message: str)

Returns:
    LoadResult with splat_data or scene data
)doc");

        // save_ply() - Save to PLY format
        m.def(
            "save_ply",
            [](const PySplatData& data, const std::filesystem::path& path, bool binary, nb::object progress) {
                io::PlySaveOptions options;
                options.output_path = path;
                options.binary = binary;

                if (progress && !progress.is_none()) {
                    PyExportProgressCallback py_progress{nb::cast<nb::object>(progress)};
                    options.progress_callback = [py_progress](float p, const std::string& stage) -> bool {
                        return py_progress(p, stage);
                    };
                }

                auto result = io::save_ply(*data.data(), options);
                if (!result) {
                    throw std::runtime_error(std::format("Failed to save PLY: {}", result.error().format()));
                }
            },
            nb::arg("data"), nb::arg("path"), nb::arg("binary") = true, nb::arg("progress") = nb::none(),
            R"doc(Save splat data to PLY file.

Args:
    data: SplatData to save
    path: Output file path
    binary: Use binary format (default: True, faster and smaller)
    progress: Optional callback(progress: float, stage: str) -> bool, return False to cancel
)doc");

        // save_sog() - Save to SuperSplat format
        m.def(
            "save_sog",
            [](const PySplatData& data, const std::filesystem::path& path, int kmeans_iterations, bool use_gpu,
               nb::object progress) {
                io::SogSaveOptions options;
                options.output_path = path;
                options.kmeans_iterations = kmeans_iterations;
                options.use_gpu = use_gpu;

                if (progress && !progress.is_none()) {
                    PyExportProgressCallback py_progress{nb::cast<nb::object>(progress)};
                    options.progress_callback = [py_progress](float p, const std::string& stage) -> bool {
                        return py_progress(p, stage);
                    };
                }

                auto result = io::save_sog(*data.data(), options);
                if (!result) {
                    throw std::runtime_error(std::format("Failed to save SOG: {}", result.error().format()));
                }
            },
            nb::arg("data"), nb::arg("path"), nb::arg("kmeans_iterations") = 10, nb::arg("use_gpu") = true,
            nb::arg("progress") = nb::none(),
            R"doc(Save splat data to SOG (SuperSplat) format.

Args:
    data: SplatData to save
    path: Output file path
    kmeans_iterations: K-means iterations for color quantization (default: 10)
    use_gpu: Use GPU for compression (default: True)
    progress: Optional callback(progress: float, stage: str) -> bool, return False to cancel
)doc");

        // save_spz() - Save to Niantic SPZ format
        m.def(
            "save_spz",
            [](const PySplatData& data, const std::filesystem::path& path) {
                io::SpzSaveOptions options;
                options.output_path = path;

                auto result = io::save_spz(*data.data(), options);
                if (!result) {
                    throw std::runtime_error(std::format("Failed to save SPZ: {}", result.error().format()));
                }
            },
            nb::arg("data"), nb::arg("path"),
            R"doc(Save splat data to SPZ (Niantic compressed) format.

Args:
    data: SplatData to save
    path: Output file path
)doc");

        // export_html() - Export standalone HTML viewer
        m.def(
            "export_html",
            [](const PySplatData& data, const std::filesystem::path& path, int kmeans_iterations, nb::object progress) {
                io::HtmlExportOptions options;
                options.output_path = path;
                options.kmeans_iterations = kmeans_iterations;

                if (progress && !progress.is_none()) {
                    nb::object py_cb = nb::cast<nb::object>(progress);
                    options.progress_callback = [py_cb](float p, const std::string& stage) {
                        nb::gil_scoped_acquire gil;
                        try {
                            py_cb(p, stage);
                        } catch (const std::exception& e) {
                            LOG_ERROR("HTML export progress callback error: {}", e.what());
                        }
                    };
                }

                auto result = io::export_html(*data.data(), options);
                if (!result) {
                    throw std::runtime_error(std::format("Failed to export HTML: {}", result.error().format()));
                }
            },
            nb::arg("data"), nb::arg("path"), nb::arg("kmeans_iterations") = 10, nb::arg("progress") = nb::none(),
            R"doc(Export splat data as standalone HTML viewer.

Args:
    data: SplatData to export
    path: Output HTML file path
    kmeans_iterations: K-means iterations for compression (default: 10)
    progress: Optional callback(progress: float, stage: str)
)doc");

        // is_dataset_path() - Check if path is a dataset
        m.def(
            "is_dataset_path",
            [](const std::filesystem::path& path) -> bool { return io::Loader::isDatasetPath(path); }, nb::arg("path"),
            "Check if path contains a dataset (COLMAP/transforms.json) vs single file");

        // get_supported_formats() - List supported formats
        m.def(
            "get_supported_formats",
            []() -> std::vector<std::string> {
                auto loader = io::Loader::create();
                return loader->getSupportedFormats();
            },
            "Get list of supported file formats");

        // get_supported_extensions() - List supported extensions
        m.def(
            "get_supported_extensions",
            []() -> std::vector<std::string> {
                auto loader = io::Loader::create();
                return loader->getSupportedExtensions();
            },
            "Get list of supported file extensions");
    }

} // namespace lfs::python
