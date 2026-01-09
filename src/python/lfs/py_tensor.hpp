/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace lfs::python {

    class PyTensor {
    public:
        PyTensor() = default;
        explicit PyTensor(core::Tensor tensor, bool owns_data = true);

        // Properties
        nb::tuple shape() const;
        size_t ndim() const;
        size_t numel() const;
        std::string device() const;
        std::string dtype() const;
        bool is_contiguous() const;
        bool is_cuda() const;
        size_t size(int dim) const;

        // Memory operations
        PyTensor clone() const;
        PyTensor cpu() const;
        PyTensor cuda() const;
        PyTensor contiguous() const;
        void sync() const;

        // Scalar extraction
        float item() const;
        float item_float() const { return item(); }
        int64_t item_int() const;
        bool item_bool() const;

        // NumPy conversion
        nb::object numpy(bool copy = true) const;

        // Static factory: create from NumPy
        static PyTensor from_numpy(nb::ndarray<> arr, bool copy = true);

        // Slicing (Phase 3)
        PyTensor getitem(const nb::object& key) const;
        void setitem(const nb::object& key, const nb::object& value);

        // Arithmetic operators (Phase 4)
        PyTensor add(const PyTensor& other) const;
        PyTensor add_scalar(float scalar) const;
        PyTensor sub(const PyTensor& other) const;
        PyTensor sub_scalar(float scalar) const;
        PyTensor rsub_scalar(float scalar) const;
        PyTensor mul(const PyTensor& other) const;
        PyTensor mul_scalar(float scalar) const;
        PyTensor div(const PyTensor& other) const;
        PyTensor div_scalar(float scalar) const;
        PyTensor rdiv_scalar(float scalar) const;
        PyTensor neg() const;
        PyTensor abs() const;
        PyTensor sigmoid() const;
        PyTensor exp() const;
        PyTensor log() const;
        PyTensor sqrt() const;
        PyTensor relu() const;

        // In-place arithmetic
        PyTensor& iadd(const PyTensor& other);
        PyTensor& iadd_scalar(float scalar);
        PyTensor& isub(const PyTensor& other);
        PyTensor& isub_scalar(float scalar);
        PyTensor& imul(const PyTensor& other);
        PyTensor& imul_scalar(float scalar);
        PyTensor& idiv(const PyTensor& other);
        PyTensor& idiv_scalar(float scalar);

        // Comparison operators (return Bool tensor)
        PyTensor eq(const PyTensor& other) const;
        PyTensor eq_scalar(float scalar) const;
        PyTensor ne(const PyTensor& other) const;
        PyTensor ne_scalar(float scalar) const;
        PyTensor lt(const PyTensor& other) const;
        PyTensor lt_scalar(float scalar) const;
        PyTensor le(const PyTensor& other) const;
        PyTensor le_scalar(float scalar) const;
        PyTensor gt(const PyTensor& other) const;
        PyTensor gt_scalar(float scalar) const;
        PyTensor ge(const PyTensor& other) const;
        PyTensor ge_scalar(float scalar) const;

        // Logical operators (Bool tensors)
        PyTensor logical_and(const PyTensor& other) const;
        PyTensor logical_or(const PyTensor& other) const;
        PyTensor logical_not() const;

        // Reduction operations
        PyTensor sum(std::optional<int> dim = std::nullopt, bool keepdim = false) const;
        PyTensor mean(std::optional<int> dim = std::nullopt, bool keepdim = false) const;
        PyTensor max(std::optional<int> dim = std::nullopt, bool keepdim = false) const;
        PyTensor min(std::optional<int> dim = std::nullopt, bool keepdim = false) const;
        float sum_scalar() const;
        float mean_scalar() const;
        float max_scalar() const;
        float min_scalar() const;

        // Shape operations
        PyTensor reshape(const std::vector<int64_t>& new_shape) const;
        PyTensor view(const std::vector<int64_t>& new_shape) const;
        PyTensor squeeze(std::optional<int> dim = std::nullopt) const;
        PyTensor unsqueeze(int dim) const;
        PyTensor transpose(int dim0, int dim1) const;
        PyTensor permute(const std::vector<int>& dims) const;
        PyTensor flatten(int start_dim = 0, int end_dim = -1) const;

        // String representation
        std::string repr() const;

        // DLPack protocol for zero-copy tensor exchange
        nb::tuple dlpack_device() const;
        nb::capsule dlpack(nb::object stream = nb::none()) const;
        static PyTensor from_dlpack(nb::object obj);

        // Access underlying tensor (for internal use)
        const core::Tensor& tensor() const { return tensor_; }
        core::Tensor& tensor() { return tensor_; }

    private:
        core::Tensor tensor_;
        bool owns_data_ = true;
        std::optional<nb::capsule> dlpack_capsule_;

        // Helper to parse Python slice
        struct SliceInfo {
            int64_t start;
            int64_t stop;
            int64_t step;
        };
        SliceInfo parse_slice(const nb::slice& sl, size_t dim_size) const;
    };

    // Register PyTensor with nanobind
    void register_tensor(nb::module_& m);

} // namespace lfs::python
