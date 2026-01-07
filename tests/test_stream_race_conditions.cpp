/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/internal/size_bucketed_pool.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using namespace lfs::core;

class StreamRaceConditionTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceSynchronize();
        Tensor::manual_seed(42);
    }

    void TearDown() override {
        cudaDeviceSynchronize();
    }
};

// Test 1: Memory pool reuse race condition
// This test exposes the bug where memory freed with cudaMemsetAsync still in-flight
// could be reused before the async operation completed.
// Without the fix (rejecting non-default streams in cache_free), this test would
// produce corrupted data due to race conditions.
TEST_F(StreamRaceConditionTest, MemoryPoolReuseWithAsyncOps) {
    constexpr int NUM_ITERATIONS = 100;
    constexpr size_t TENSOR_SIZE = 1024 * 1024; // 1M elements

    cudaStream_t test_stream;
    ASSERT_EQ(cudaStreamCreate(&test_stream), cudaSuccess);

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // Allocate tensor and fill with known pattern
        auto t1 = Tensor::full({TENSOR_SIZE}, 42.0f, Device::CUDA);
        float* ptr1 = t1.ptr<float>();

        // Launch async memset on a different stream (simulates cudaMemsetAsync usage)
        cudaMemsetAsync(ptr1, 0, TENSOR_SIZE * sizeof(float), test_stream);

        // Destroy tensor - memory goes back to pool
        // Bug: Without proper sync, the memset might still be in-flight
        t1 = Tensor();

        // Immediately allocate new tensor - might get same memory from pool
        auto t2 = Tensor::full({TENSOR_SIZE}, 123.0f, Device::CUDA);

        // Sync and verify - if race occurred, values could be corrupted
        cudaDeviceSynchronize();

        auto cpu = t2.cpu();
        auto vec = cpu.to_vector();

        // Check a sample of values
        for (size_t i = 0; i < 100; ++i) {
            size_t idx = (i * TENSOR_SIZE / 100);
            ASSERT_FLOAT_EQ(vec[idx], 123.0f)
                << "Race condition detected at iteration " << iter << ", index " << idx
                << ": expected 123.0, got " << vec[idx];
        }
    }

    cudaStreamDestroy(test_stream);
}

// Test 2: Stress test rapid allocation/deallocation with async operations
// Increases probability of exposing races by creating many tensors rapidly
TEST_F(StreamRaceConditionTest, StressTestRapidAllocDealloc) {
    constexpr int NUM_TENSORS = 50;
    constexpr int NUM_ROUNDS = 20;
    constexpr size_t BASE_SIZE = 256 * 1024; // 256K elements per tensor

    cudaStream_t streams[4];
    for (int i = 0; i < 4; ++i) {
        ASSERT_EQ(cudaStreamCreate(&streams[i]), cudaSuccess);
    }

    for (int round = 0; round < NUM_ROUNDS; ++round) {
        std::vector<Tensor> tensors;
        tensors.reserve(NUM_TENSORS);

        // Create many tensors with async operations on different streams
        for (int i = 0; i < NUM_TENSORS; ++i) {
            size_t size = BASE_SIZE + (i * 1024); // Vary sizes
            auto t = Tensor::zeros({size}, Device::CUDA);

            // Simulate async work on various streams
            cudaStream_t stream = streams[i % 4];
            cudaMemsetAsync(t.ptr<float>(), 0xFF, size * sizeof(float), stream);

            tensors.push_back(std::move(t));
        }

        // Destroy half of them while async ops might still be running
        for (int i = 0; i < NUM_TENSORS / 2; ++i) {
            tensors[i] = Tensor();
        }

        // Allocate new tensors - might reuse memory
        std::vector<Tensor> new_tensors;
        for (int i = 0; i < NUM_TENSORS / 2; ++i) {
            size_t size = BASE_SIZE + (i * 1024);
            auto t = Tensor::full({size}, static_cast<float>(i), Device::CUDA);
            new_tensors.push_back(std::move(t));
        }

        // Sync all streams
        for (int i = 0; i < 4; ++i) {
            cudaStreamSynchronize(streams[i]);
        }

        // Verify new tensors have correct values
        for (int i = 0; i < NUM_TENSORS / 2; ++i) {
            auto cpu = new_tensors[i].cpu();
            auto vec = cpu.to_vector();
            float expected = static_cast<float>(i);

            // Check samples
            for (size_t j = 0; j < std::min(vec.size(), size_t(10)); ++j) {
                ASSERT_FLOAT_EQ(vec[j], expected)
                    << "Data corruption at round " << round << ", tensor " << i << ", index " << j;
            }
        }
    }

    for (int i = 0; i < 4; ++i) {
        cudaStreamDestroy(streams[i]);
    }
}

// Test 3: Stream destruction before tensor destruction (shutdown crash scenario)
// This test simulates the crash that occurred during shutdown when streams
// were destroyed while tensors still held references to them.
TEST_F(StreamRaceConditionTest, StreamDestroyedBeforeTensor) {
    constexpr size_t TENSOR_SIZE = 512 * 1024;

    // Create a stream
    cudaStream_t test_stream;
    ASSERT_EQ(cudaStreamCreate(&test_stream), cudaSuccess);

    // Allocate tensor and do async work on the stream
    auto tensor = Tensor::zeros({TENSOR_SIZE}, Device::CUDA);
    cudaMemsetAsync(tensor.ptr<float>(), 0, TENSOR_SIZE * sizeof(float), test_stream);

    // Destroy the stream BEFORE destroying the tensor
    // This simulates what happens during shutdown when stream pool is destroyed
    // before all tensors are cleaned up
    cudaStreamDestroy(test_stream);

    // Now destroy the tensor - this should NOT crash
    // Bug: Without the fix, cache_free would try to cudaStreamSynchronize
    // on the destroyed stream, causing a segfault
    EXPECT_NO_FATAL_FAILURE({
        tensor = Tensor();
        cudaDeviceSynchronize();
    });
}

// Test 4: Multiple threads allocating/deallocating with streams
// Tests thread safety of memory pool with stream operations
TEST_F(StreamRaceConditionTest, MultiThreadedStreamOperations) {
    constexpr int NUM_THREADS = 4;
    constexpr int OPS_PER_THREAD = 50;
    constexpr size_t TENSOR_SIZE = 128 * 1024;

    std::atomic<int> errors{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&errors, t]() {
            cudaStream_t stream;
            if (cudaStreamCreate(&stream) != cudaSuccess) {
                errors++;
                return;
            }

            for (int i = 0; i < OPS_PER_THREAD; ++i) {
                try {
                    // Create tensor with specific value
                    float expected = static_cast<float>(t * 1000 + i);
                    auto tensor = Tensor::full({TENSOR_SIZE}, expected, Device::CUDA);

                    // Do async work
                    cudaMemsetAsync(tensor.ptr<float>(), 0, 1024, stream);

                    // Sync and verify (first 256 elements should be 0, rest should be expected)
                    cudaStreamSynchronize(stream);

                    auto cpu = tensor.cpu();
                    auto vec = cpu.to_vector();

                    // Check that non-zeroed part is correct
                    for (size_t j = 256; j < 512; ++j) {
                        if (vec[j] != expected) {
                            errors++;
                            break;
                        }
                    }
                } catch (...) {
                    errors++;
                }
            }

            cudaStreamDestroy(stream);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(errors.load(), 0) << "Thread safety errors detected";
}

// Test 5: Size-bucketed pool specific test
// Tests the SizeBucketedPool directly to verify stream handling
TEST_F(StreamRaceConditionTest, SizeBucketedPoolStreamHandling) {
    auto& pool = SizeBucketedPool::instance();

    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    // Allocate memory from pool
    size_t size = 512 * 1024; // 512KB - falls into bucketed range
    void* ptr = pool.allocate(size, nullptr);
    ASSERT_NE(ptr, nullptr);

    // Do async operation
    cudaMemsetAsync(ptr, 0xAB, size, stream);

    // Try to cache_free with non-default stream
    // With the fix, this should return false (not cached) to avoid sync issues
    bool cached = pool.cache_free(ptr, size, stream);

    if (cached) {
        // If it was cached, the pool must have synced properly
        // Verify by allocating again and checking the memory
        void* ptr2 = pool.allocate(size, nullptr);
        if (ptr2 == ptr) {
            // Got same memory back - verify it's properly zeroed
            std::vector<uint8_t> host(size);
            cudaMemcpy(host.data(), ptr2, size, cudaMemcpyDeviceToHost);

            // All bytes should be 0xAB from the memset
            for (size_t i = 0; i < 100; ++i) {
                EXPECT_EQ(host[i], 0xAB) << "Memory not properly synced before reuse";
            }
        }
        pool.deallocate(ptr2, size, nullptr);
    } else {
        // Memory was not cached (expected with the fix for non-default streams)
        // It should have been freed directly - allocate new memory
        void* ptr2 = pool.allocate(size, nullptr);
        EXPECT_NE(ptr2, nullptr);
        pool.deallocate(ptr2, size, nullptr);
    }

    cudaStreamDestroy(stream);
}

// Test 6: Rapid create/destroy cycle simulating training loop
// This pattern mimics what happens during training iterations
TEST_F(StreamRaceConditionTest, TrainingLoopPattern) {
    constexpr int NUM_ITERATIONS = 100;
    constexpr size_t IMAGE_SIZE = 800 * 600 * 3; // Typical image tensor size

    cudaStream_t compute_stream;
    ASSERT_EQ(cudaStreamCreate(&compute_stream), cudaSuccess);

    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        // Simulate forward pass - create intermediate tensors
        auto image = Tensor::randn({IMAGE_SIZE}, Device::CUDA);
        auto gradients = Tensor::zeros({IMAGE_SIZE}, Device::CUDA);

        // Simulate async gradient computation
        cudaMemsetAsync(gradients.ptr<float>(), 0, IMAGE_SIZE * sizeof(float), compute_stream);

        // Simulate backward pass creating more temporaries
        auto grad_image = Tensor::zeros({IMAGE_SIZE}, Device::CUDA);

        // End of iteration - tensors go out of scope
        // Without proper sync, the next iteration might reuse memory
        // while async ops are still running
    }

    cudaStreamSynchronize(compute_stream);
    cudaStreamDestroy(compute_stream);

    // If we got here without hanging or crashing, the test passes
    SUCCEED();
}
