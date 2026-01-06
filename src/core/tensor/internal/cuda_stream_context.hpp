/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

namespace lfs::core {

// Thread-local CUDA stream management (PyTorch-style).
// Uses thread_local for O(1) access without locking.
class CUDAStreamContext {
public:
    static CUDAStreamContext& instance() {
        static CUDAStreamContext inst;
        return inst;
    }

    cudaStream_t getCurrentStream() const { return current_stream_; }
    void setCurrentStream(cudaStream_t stream) { current_stream_ = stream; }
    void resetToDefault() { current_stream_ = nullptr; }

private:
    CUDAStreamContext() = default;
    ~CUDAStreamContext() = default;
    CUDAStreamContext(const CUDAStreamContext&) = delete;
    CUDAStreamContext& operator=(const CUDAStreamContext&) = delete;

    static inline thread_local cudaStream_t current_stream_ = nullptr;
};

// RAII guard for scoped stream changes. Restores previous stream on destruction.
class CUDAStreamGuard {
public:
    explicit CUDAStreamGuard(cudaStream_t stream)
        : prev_stream_(CUDAStreamContext::instance().getCurrentStream()) {
        CUDAStreamContext::instance().setCurrentStream(stream);
    }

    ~CUDAStreamGuard() {
        CUDAStreamContext::instance().setCurrentStream(prev_stream_);
    }

    cudaStream_t originalStream() const { return prev_stream_; }
    cudaStream_t currentStream() const { return CUDAStreamContext::instance().getCurrentStream(); }
    void resetStream(cudaStream_t stream) { CUDAStreamContext::instance().setCurrentStream(stream); }

    CUDAStreamGuard(const CUDAStreamGuard&) = delete;
    CUDAStreamGuard& operator=(const CUDAStreamGuard&) = delete;
    CUDAStreamGuard(CUDAStreamGuard&&) = delete;
    CUDAStreamGuard& operator=(CUDAStreamGuard&&) = delete;

private:
    const cudaStream_t prev_stream_;
};

inline cudaStream_t getCurrentCUDAStream() {
    return CUDAStreamContext::instance().getCurrentStream();
}

inline void setCurrentCUDAStream(cudaStream_t stream) {
    CUDAStreamContext::instance().setCurrentStream(stream);
}

}  // namespace lfs::core
