/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/logger.hpp"
#include <array>
#include <atomic>
#include <cuda_runtime.h>
#include <mutex>

namespace lfs::core {

// CUDA stream pool with round-robin assignment per priority level.
// Follows PyTorch's c10/cuda/CUDAStream.cpp design.
class CUDAStreamPool {
public:
    static constexpr int STREAMS_PER_PRIORITY = 32;
    static constexpr int STREAM_INDEX_MASK = STREAMS_PER_PRIORITY - 1;

    static CUDAStreamPool& instance() {
        static CUDAStreamPool pool;
        return pool;
    }

    cudaStream_t getStreamFromPool(const bool high_priority = false) {
        ensureInitialized();
        if (high_priority) {
            const uint32_t idx = high_priority_idx_.fetch_add(1, std::memory_order_relaxed);
            return high_priority_streams_[idx & STREAM_INDEX_MASK];
        }
        const uint32_t idx = low_priority_idx_.fetch_add(1, std::memory_order_relaxed);
        return low_priority_streams_[idx & STREAM_INDEX_MASK];
    }

    static cudaStream_t getDefaultStream() { return nullptr; }
    size_t poolSize() const { return STREAMS_PER_PRIORITY * 2; }
    bool isInitialized() const { return initialized_.load(std::memory_order_acquire); }

    CUDAStreamPool(const CUDAStreamPool&) = delete;
    CUDAStreamPool& operator=(const CUDAStreamPool&) = delete;
    CUDAStreamPool(CUDAStreamPool&&) = delete;
    CUDAStreamPool& operator=(CUDAStreamPool&&) = delete;

private:
    CUDAStreamPool() = default;

    ~CUDAStreamPool() {
        if (!initialized_.load(std::memory_order_acquire)) return;
        for (auto& stream : low_priority_streams_) {
            if (stream) cudaStreamDestroy(stream);
        }
        for (auto& stream : high_priority_streams_) {
            if (stream) cudaStreamDestroy(stream);
        }
    }

    void ensureInitialized() {
        if (initialized_.load(std::memory_order_acquire)) return;
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (initialized_.load(std::memory_order_relaxed)) return;
        initializeStreams();
        initialized_.store(true, std::memory_order_release);
    }

    void initializeStreams() {
        int least_priority = 0, greatest_priority = 0;
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);

        for (int i = 0; i < STREAMS_PER_PRIORITY; ++i) {
            if (cudaStreamCreateWithPriority(&low_priority_streams_[i],
                    cudaStreamNonBlocking, least_priority) != cudaSuccess) {
                cudaStreamCreateWithFlags(&low_priority_streams_[i], cudaStreamNonBlocking);
            }
        }
        for (int i = 0; i < STREAMS_PER_PRIORITY; ++i) {
            if (cudaStreamCreateWithPriority(&high_priority_streams_[i],
                    cudaStreamNonBlocking, greatest_priority) != cudaSuccess) {
                cudaStreamCreateWithFlags(&high_priority_streams_[i], cudaStreamNonBlocking);
            }
        }
        LOG_DEBUG("CUDAStreamPool: {} streams initialized", poolSize());
    }

    std::array<cudaStream_t, STREAMS_PER_PRIORITY> low_priority_streams_{};
    std::array<cudaStream_t, STREAMS_PER_PRIORITY> high_priority_streams_{};

    std::atomic<uint32_t> low_priority_idx_{0};
    std::atomic<uint32_t> high_priority_idx_{0};

    std::atomic<bool> initialized_{false};
    std::mutex init_mutex_;
};

}  // namespace lfs::core
