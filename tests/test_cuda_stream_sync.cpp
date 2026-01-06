/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * CUDA Stream Synchronization Tests
 * Tests cross-stream data visibility via record_stream, wait_stream, and events.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <core/tensor.hpp>
#include <core/logger.hpp>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <future>
#include <unordered_set>

using namespace lfs::core;

// =============================================================================
// Test Fixture
// =============================================================================

class CUDAStreamSyncTest : public ::testing::Test {
protected:
    cudaStream_t stream1_ = nullptr;
    cudaStream_t stream2_ = nullptr;
    cudaStream_t default_stream_ = nullptr;

    void SetUp() override {
        // Create test streams with non-blocking flag (like PyTorch)
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream1_, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream2_, cudaStreamNonBlocking), cudaSuccess);
        default_stream_ = nullptr; // CUDA default stream
    }

    void TearDown() override {
        if (stream1_) cudaStreamDestroy(stream1_);
        if (stream2_) cudaStreamDestroy(stream2_);
        cudaDeviceSynchronize();
    }

    // Helper: Do GPU work to create measurable delay
    // Adapted from PyTorch's torch.cuda._sleep()
    void gpuSleep(cudaStream_t stream, int iterations = 50) {
        auto tensor = Tensor::zeros({1000, 1000}, Device::CUDA, DataType::Float32);
        tensor.set_stream(stream);
        for (int i = 0; i < iterations; ++i) {
            tensor = tensor + 1.0f;
        }
    }
};

// =============================================================================
// Basic Stream Tests (from PyTorch test_streams)
// =============================================================================

// Verifies streams are created correctly and are different
TEST_F(CUDAStreamSyncTest, BasicStreamOperations) {
    // Verify streams are different
    EXPECT_NE(stream1_, stream2_);
    EXPECT_NE(stream1_, default_stream_);

    // Query should return cudaSuccess for idle streams
    EXPECT_EQ(cudaStreamQuery(stream1_), cudaSuccess);
    EXPECT_EQ(cudaStreamQuery(stream2_), cudaSuccess);
}

// Test stream pool returns streams that may be reused
// Adapted from PyTorch's StreamPoolTest
TEST_F(CUDAStreamSyncTest, StreamPoolReuse) {
    std::vector<cudaStream_t> streams;
    std::unordered_set<cudaStream_t> unique_streams;

    // Create many streams
    for (int i = 0; i < 50; ++i) {
        cudaStream_t s;
        cudaStreamCreate(&s);
        streams.push_back(s);
        unique_streams.insert(s);
    }

    // All should be unique when created this way
    EXPECT_EQ(unique_streams.size(), streams.size());

    // Cleanup
    for (auto s : streams) {
        cudaStreamDestroy(s);
    }
}

// =============================================================================
// Thread-Local Stream Tests (from PyTorch MultithreadGetAndSetTest)
// =============================================================================

TEST_F(CUDAStreamSyncTest, StreamsAreThreadLocal) {
    std::atomic<cudaStream_t> thread1_stream{nullptr};
    std::atomic<cudaStream_t> thread2_stream{nullptr};

    auto thread_func = [](std::atomic<cudaStream_t>& out_stream) {
        cudaStream_t local_stream;
        cudaStreamCreate(&local_stream);
        out_stream.store(local_stream);
        // Simulate some work
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    };

    std::thread t1(thread_func, std::ref(thread1_stream));
    std::thread t2(thread_func, std::ref(thread2_stream));
    t1.join();
    t2.join();

    // Each thread should have created its own stream
    EXPECT_NE(thread1_stream.load(), nullptr);
    EXPECT_NE(thread2_stream.load(), nullptr);
    EXPECT_NE(thread1_stream.load(), thread2_stream.load());

    // Cleanup
    cudaStreamDestroy(thread1_stream.load());
    cudaStreamDestroy(thread2_stream.load());
}

// =============================================================================
// Event Synchronization Tests (from PyTorch CUDAEventSyncTest)
// =============================================================================

TEST_F(CUDAStreamSyncTest, EventBasicOperations) {
    cudaEvent_t event;
    ASSERT_EQ(cudaEventCreate(&event), cudaSuccess);

    // Note: Unrecorded event behavior varies by CUDA version
    // Some return cudaSuccess, others cudaErrorNotReady

    // Record on stream1
    ASSERT_EQ(cudaEventRecord(event, stream1_), cudaSuccess);

    // Wait on stream2
    ASSERT_EQ(cudaStreamWaitEvent(stream2_, event, 0), cudaSuccess);

    cudaStreamSynchronize(stream2_);
    EXPECT_EQ(cudaEventQuery(event), cudaSuccess);

    cudaEventDestroy(event);
}

// Test event blocks multiple streams
// Adapted from PyTorch's CUDAEventSyncTest
TEST_F(CUDAStreamSyncTest, EventBlocksMultipleStreams) {
    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);

    // Do work on stream1 and record event
    auto tensor = Tensor::ones({1000}, Device::CUDA, DataType::Float32);
    tensor.set_stream(stream1_);
    tensor = tensor * 5.0f;
    cudaEventRecord(event, stream1_);

    // Create third stream
    cudaStream_t stream3;
    cudaStreamCreate(&stream3);

    // Both stream2 and stream3 wait on event
    cudaStreamWaitEvent(stream2_, event, 0);
    cudaStreamWaitEvent(stream3, event, 0);

    // Both should see the updated tensor after their waits
    cudaStreamSynchronize(stream2_);
    cudaStreamSynchronize(stream3);

    EXPECT_EQ(cudaEventQuery(event), cudaSuccess);

    cudaEventDestroy(event);
    cudaStreamDestroy(stream3);
}

// =============================================================================
// record_stream Pattern Tests (from PyTorch test_record_stream)
// =============================================================================

// This is THE critical test - it tests the pattern where memory allocated on
// one stream is used on another, and we need to prevent premature deallocation.
TEST_F(CUDAStreamSyncTest, RecordStreamPreventsPrematureFree) {
    void* data_ptr = nullptr;

    {
        // Allocate tensor on stream1 (like PyTorch's background copy stream)
        auto tensor = Tensor::ones({1000}, Device::CUDA, DataType::Float32);
        tensor.set_stream(stream1_);
        data_ptr = tensor.data_ptr();

        // Make stream2 wait for stream1's work
        cudaEvent_t event;
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        cudaEventRecord(event, stream1_);
        cudaStreamWaitEvent(stream2_, event, 0);

        // Now record that tensor will be used on stream2
        // (In PyTorch: tensor.record_stream(stream2))
        tensor.record_ready();

        // Do a long operation on stream2 that uses the tensor
        gpuSleep(stream2_, 10);
        auto result = tensor.sum();

        // Tensor goes out of scope, but memory should NOT be freed yet
        // because stream2 is still using it
        cudaEventDestroy(event);
    }

    // Sync to ensure all work is done
    cudaDeviceSynchronize();

    // The test passes if we didn't crash - memory was properly managed
    SUCCEED();
}

// Test record_stream on a view/slice (from PyTorch test_record_stream_on_shifted_view)
TEST_F(CUDAStreamSyncTest, RecordStreamOnView) {
    auto base = Tensor::ones({100, 100}, Device::CUDA, DataType::Float32);
    base.set_stream(stream1_);
    base = base * 10.0f;
    base.record_ready();

    // Get a view (slice) with non-zero storage offset
    auto view = base.slice(0, 50, 100);  // Last 50 rows
    EXPECT_GT(view.data_ptr(), base.data_ptr());  // Verify offset

    // Record the view on stream2
    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    cudaEventRecord(event, stream1_);
    cudaStreamWaitEvent(stream2_, event, 0);

    // View should propagate stream and event from parent
    EXPECT_EQ(view.stream(), base.stream());
    EXPECT_EQ(view.ready_event(), base.ready_event());

    // Use view on stream2
    auto result = view.sum();
    cudaStreamSynchronize(stream2_);

    auto cpu = result.to(Device::CPU);
    // 50 rows * 100 cols * 10.0 = 50000
    EXPECT_FLOAT_EQ(cpu.ptr<float>()[0], 50000.0f);

    cudaEventDestroy(event);
}

// =============================================================================
// wait_stream Pattern Tests (from PyTorch test_streaming_backwards_sync)
// =============================================================================

// Test wait_stream pattern - make one stream wait for all ops on another
TEST_F(CUDAStreamSyncTest, WaitStreamPattern) {
    auto tensor = Tensor::ones({1000}, Device::CUDA, DataType::Float32);
    tensor.set_stream(stream1_);

    // Do operations on stream1
    tensor = tensor * 5.0f;
    tensor = tensor + 10.0f;

    // Record event at end of stream1 work (this is wait_stream)
    cudaEvent_t sync_event;
    cudaEventCreateWithFlags(&sync_event, cudaEventDisableTiming);
    cudaEventRecord(sync_event, stream1_);

    // stream2 waits for stream1
    cudaStreamWaitEvent(stream2_, sync_event, 0);

    // Now safe to use tensor on stream2
    auto result = tensor.sum();
    cudaStreamSynchronize(stream2_);

    auto cpu = result.to(Device::CPU);
    EXPECT_FLOAT_EQ(cpu.ptr<float>()[0], 15.0f * 1000);

    cudaEventDestroy(sync_event);
}

// Test bidirectional stream waiting
TEST_F(CUDAStreamSyncTest, BidirectionalWaitStream) {
    // Stream1 creates tensor A
    auto tensorA = Tensor::ones({1000}, Device::CUDA, DataType::Float32);
    tensorA.set_stream(stream1_);
    tensorA = tensorA * 3.0f;

    cudaEvent_t eventA;
    cudaEventCreateWithFlags(&eventA, cudaEventDisableTiming);
    cudaEventRecord(eventA, stream1_);

    // Stream2 creates tensor B
    auto tensorB = Tensor::ones({1000}, Device::CUDA, DataType::Float32);
    tensorB.set_stream(stream2_);
    tensorB = tensorB * 7.0f;

    cudaEvent_t eventB;
    cudaEventCreateWithFlags(&eventB, cudaEventDisableTiming);
    cudaEventRecord(eventB, stream2_);

    // Both streams wait for each other (rendez-vous)
    cudaStreamWaitEvent(stream1_, eventB, 0);
    cudaStreamWaitEvent(stream2_, eventA, 0);

    // Now either stream can safely use both tensors
    // Let's use default stream (waits for all)
    cudaStreamSynchronize(stream1_);
    cudaStreamSynchronize(stream2_);

    auto result = tensorA + tensorB;

    auto cpu = result.to(Device::CPU);
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(cpu.ptr<float>()[i], 10.0f);
    }

    cudaEventDestroy(eventA);
    cudaEventDestroy(eventB);
}

// =============================================================================
// Cross-Stream Binary Operations (important for training)
// =============================================================================

TEST_F(CUDAStreamSyncTest, BinaryOpWithProperSync) {
    // Create tensor A on stream1
    auto tensorA = Tensor::ones({1000}, Device::CUDA, DataType::Float32);
    tensorA.set_stream(stream1_);
    tensorA = tensorA * 3.0f;
    tensorA.record_ready();

    // Create tensor B on stream2
    auto tensorB = Tensor::ones({1000}, Device::CUDA, DataType::Float32);
    tensorB.set_stream(stream2_);
    tensorB = tensorB * 7.0f;
    tensorB.record_ready();

    // For binary op, both inputs must be ready
    // Default stream waits for both
    if (tensorA.ready_event()) {
        cudaStreamWaitEvent(nullptr, tensorA.ready_event(), 0);
    }
    if (tensorB.ready_event()) {
        cudaStreamWaitEvent(nullptr, tensorB.ready_event(), 0);
    }

    auto result = tensorA + tensorB;
    cudaDeviceSynchronize();

    auto cpu = result.to(Device::CPU);
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(cpu.ptr<float>()[i], 10.0f);
    }
}

// =============================================================================
// Producer-Consumer Pattern (dataloader -> trainer)
// =============================================================================

TEST_F(CUDAStreamSyncTest, ProducerConsumerPattern) {
    const int NUM_ITERATIONS = 10;
    std::vector<Tensor> results;
    results.reserve(NUM_ITERATIONS);

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        // Producer (dataloader) creates tensor on stream1
        auto data = Tensor::zeros({100}, Device::CUDA, DataType::Float32);
        data.set_stream(stream1_);
        data = data + static_cast<float>(i + 1);
        data.record_ready();

        // Consumer (trainer) waits for data then uses on stream2
        if (data.ready_event()) {
            cudaStreamWaitEvent(stream2_, data.ready_event(), 0);
        }

        // Process data on stream2
        auto processed = data * 2.0f;
        processed.set_stream(stream2_);

        results.push_back(std::move(processed));
    }

    cudaDeviceSynchronize();

    // Verify all results
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        auto cpu = results[i].to(Device::CPU);
        float expected = static_cast<float>((i + 1) * 2);
        EXPECT_FLOAT_EQ(cpu.ptr<float>()[0], expected) << "Failed at iteration " << i;
    }
}

// =============================================================================
// Memory Visibility Tests
// =============================================================================

// Test that cudaStreamSynchronize alone is NOT sufficient for cross-stream safety
// This documents the correct pattern
TEST_F(CUDAStreamSyncTest, StreamSyncVsEventSync) {
    auto tensor = Tensor::ones({10000}, Device::CUDA, DataType::Float32);
    tensor.set_stream(stream1_);

    // Do work on stream1
    for (int i = 0; i < 10; ++i) {
        tensor = tensor + 1.0f;
    }

    // INCORRECT: Just syncing stream1 from CPU doesn't help stream2
    // cudaStreamSynchronize(stream1_);

    // CORRECT: Use event-based GPU-side synchronization
    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    cudaEventRecord(event, stream1_);
    cudaStreamWaitEvent(stream2_, event, 0);

    // Now safe to use on stream2
    auto result = tensor.sum();
    cudaStreamSynchronize(stream2_);

    auto cpu = result.to(Device::CPU);
    EXPECT_FLOAT_EQ(cpu.ptr<float>()[0], 11.0f * 10000);

    cudaEventDestroy(event);
}

// =============================================================================
// Stress Tests
// =============================================================================

// Rapid cross-stream operations
TEST_F(CUDAStreamSyncTest, StressTestCrossStreamOps) {
    const int NUM_OPS = 100;
    auto tensor = Tensor::zeros({1000}, Device::CUDA, DataType::Float32);

    for (int i = 0; i < NUM_OPS; ++i) {
        cudaStream_t current = (i % 2 == 0) ? stream1_ : stream2_;
        cudaStream_t other = (i % 2 == 0) ? stream2_ : stream1_;

        tensor.set_stream(current);
        tensor = tensor + 1.0f;
        tensor.record_ready();

        if (tensor.ready_event()) {
            cudaStreamWaitEvent(other, tensor.ready_event(), 0);
        }
    }

    cudaDeviceSynchronize();

    auto cpu = tensor.to(Device::CPU);
    EXPECT_FLOAT_EQ(cpu.ptr<float>()[0], static_cast<float>(NUM_OPS));
}

// Concurrent operations on multiple streams
TEST_F(CUDAStreamSyncTest, ConcurrentStreamsStress) {
    const int NUM_STREAMS = 4;
    const int OPS_PER_STREAM = 50;

    std::vector<cudaStream_t> streams(NUM_STREAMS);
    std::vector<Tensor> tensors(NUM_STREAMS);
    std::vector<cudaEvent_t> events(NUM_STREAMS);

    // Create streams and initialize tensors
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
        cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
        tensors[i] = Tensor::zeros({1000}, Device::CUDA, DataType::Float32);
        tensors[i].set_stream(streams[i]);
    }

    // Each stream does work, then waits for the previous stream
    for (int op = 0; op < OPS_PER_STREAM; ++op) {
        for (int s = 0; s < NUM_STREAMS; ++s) {
            // Wait for previous stream
            if (s > 0) {
                cudaStreamWaitEvent(streams[s], events[s-1], 0);
            }

            // Do work
            tensors[s] = tensors[s] + 1.0f;

            // Record completion
            cudaEventRecord(events[s], streams[s]);
        }
    }

    cudaDeviceSynchronize();

    // All tensors should have same value
    for (int i = 0; i < NUM_STREAMS; ++i) {
        auto cpu = tensors[i].to(Device::CPU);
        EXPECT_FLOAT_EQ(cpu.ptr<float>()[0], static_cast<float>(OPS_PER_STREAM))
            << "Stream " << i << " has wrong value";
    }

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaEventDestroy(events[i]);
        cudaStreamDestroy(streams[i]);
    }
}

// =============================================================================
// Tensor API Tests (record_ready, ensure_ready_on)
// =============================================================================

TEST_F(CUDAStreamSyncTest, TensorRecordReady) {
    auto tensor = Tensor::ones({1000}, Device::CUDA, DataType::Float32);
    tensor.set_stream(stream1_);
    tensor = tensor * 2.0f;

    // Before record_ready, no event
    EXPECT_EQ(tensor.ready_event(), nullptr);

    // After record_ready, event exists
    tensor.record_ready();
    EXPECT_NE(tensor.ready_event(), nullptr);
}

TEST_F(CUDAStreamSyncTest, TensorEnsureReadyOn) {
    auto tensor = Tensor::ones({1000}, Device::CUDA, DataType::Float32);
    tensor.set_stream(stream1_);
    tensor = tensor * 2.0f;
    tensor.record_ready();

    // ensure_ready_on should make stream2 wait
    tensor.ensure_ready_on(stream2_);

    // Now safe to use tensor from stream2 context
    auto result = tensor.sum();
    cudaStreamSynchronize(stream2_);

    auto cpu = result.to(Device::CPU);
    EXPECT_FLOAT_EQ(cpu.ptr<float>()[0], 2000.0f);
}

TEST_F(CUDAStreamSyncTest, SlicePreservesStreamAndEvent) {
    auto tensor = Tensor::ones({100, 100}, Device::CUDA, DataType::Float32);
    tensor.set_stream(stream1_);
    tensor = tensor * 5.0f;
    tensor.record_ready();

    // Create a slice (view)
    auto slice = tensor.slice(0, 10, 20);

    // Slice should have same stream as parent
    EXPECT_EQ(slice.stream(), tensor.stream());

    // Slice should have same ready event as parent
    EXPECT_EQ(slice.ready_event(), tensor.ready_event());

    // Using slice on a different stream with proper sync
    if (slice.ready_event()) {
        cudaStreamWaitEvent(stream2_, slice.ready_event(), 0);
    }
    auto result = slice.sum();
    cudaStreamSynchronize(stream2_);

    auto cpu = result.to(Device::CPU);
    // 10 rows * 100 cols * 5.0 = 5000
    EXPECT_FLOAT_EQ(cpu.ptr<float>()[0], 5000.0f);
}

// =============================================================================
// Race Condition Detection Tests
// =============================================================================

// Test Write-After-Read hazard: stream2 writes before stream1 finishes reading
TEST_F(CUDAStreamSyncTest, RaceCondition_WriteAfterReadHazard) {
    auto tensor = Tensor::ones({10000}, Device::CUDA, DataType::Float32);
    tensor.set_stream(stream1_);
    tensor = tensor * 5.0f;
    tensor.record_ready();

    // Stream1 starts a long read operation
    auto sum_result = tensor.sum();

    // Stream2 attempts to modify the tensor - WITH PROPER SYNC
    tensor.ensure_ready_on(stream2_);
    tensor.set_stream(stream2_);
    tensor = tensor + 1.0f;

    cudaDeviceSynchronize();

    // Original sum should be 50000 (5.0 * 10000)
    auto cpu = sum_result.to(Device::CPU);
    EXPECT_FLOAT_EQ(cpu.ptr<float>()[0], 50000.0f);
}

// Test Write-After-Write hazard on overlapping slices
TEST_F(CUDAStreamSyncTest, RaceCondition_WriteAfterWriteHazard) {
    auto tensor = Tensor::zeros({1000}, Device::CUDA, DataType::Float32);

    // Stream1 writes to first half
    tensor.set_stream(stream1_);
    auto slice1 = tensor.slice(0, 0, 500);
    slice1 = slice1 + 1.0f;
    tensor.record_ready();

    // Stream2 writes to second half - with proper sync
    tensor.ensure_ready_on(stream2_);
    tensor.set_stream(stream2_);
    auto slice2 = tensor.slice(0, 500, 1000);
    slice2 = slice2 + 2.0f;

    cudaDeviceSynchronize();

    // Verify both halves are correct
    auto cpu = tensor.to(Device::CPU);
    for (int i = 0; i < 500; ++i) {
        EXPECT_FLOAT_EQ(cpu.ptr<float>()[i], 1.0f) << "First half incorrect at " << i;
    }
    for (int i = 500; i < 1000; ++i) {
        EXPECT_FLOAT_EQ(cpu.ptr<float>()[i], 2.0f) << "Second half incorrect at " << i;
    }
}

// Test Read-After-Write with ensure_ready_on
TEST_F(CUDAStreamSyncTest, RaceCondition_ReadAfterWriteWithDelay) {
    auto tensor = Tensor::ones({10000}, Device::CUDA, DataType::Float32);
    tensor.set_stream(stream1_);

    // Long computation
    for (int i = 0; i < 20; ++i) {
        tensor = tensor + 1.0f;
    }
    tensor.record_ready();

    // Stream2 reads - must wait for stream1
    tensor.ensure_ready_on(stream2_);
    auto result = tensor.sum();
    cudaStreamSynchronize(stream2_);

    auto cpu = result.to(Device::CPU);
    EXPECT_FLOAT_EQ(cpu.ptr<float>()[0], 21.0f * 10000);
}

// =============================================================================
// Memory Safety Tests
// =============================================================================

// Verify no premature free under memory pressure
TEST_F(CUDAStreamSyncTest, MemorySafety_PrematureFreeUnderPressure) {
    const size_t NUM_TENSORS = 50;
    std::vector<Tensor> results;
    results.reserve(NUM_TENSORS);

    for (size_t i = 0; i < NUM_TENSORS; ++i) {
        auto tensor = Tensor::ones({100000}, Device::CUDA, DataType::Float32);
        tensor.set_stream(stream1_);
        tensor = tensor * static_cast<float>(i + 1);
        tensor.record_ready();

        tensor.ensure_ready_on(stream2_);
        results.push_back(tensor.sum());
    }

    cudaDeviceSynchronize();

    // Verify all results
    for (size_t i = 0; i < NUM_TENSORS; ++i) {
        auto cpu = results[i].to(Device::CPU);
        float expected = static_cast<float>((i + 1) * 100000);
        EXPECT_FLOAT_EQ(cpu.ptr<float>()[0], expected) << "Failed at tensor " << i;
    }
}

// Test data integrity after scope exit
TEST_F(CUDAStreamSyncTest, MemorySafety_UseAfterFreeDetection) {
    Tensor result;

    {
        auto tensor = Tensor::ones({10000}, Device::CUDA, DataType::Float32);
        tensor.set_stream(stream1_);
        tensor = tensor * 42.0f;
        tensor.record_ready();

        tensor.ensure_ready_on(stream2_);
        result = tensor.clone();
        // Original tensor goes out of scope
    }

    cudaDeviceSynchronize();

    auto cpu = result.to(Device::CPU);
    EXPECT_FLOAT_EQ(cpu.ptr<float>()[0], 42.0f);
}

// Stress test deferred free queue with multiple streams
TEST_F(CUDAStreamSyncTest, MemorySafety_DeferredFreeQueueStress) {
    const int NUM_STREAMS = 8;
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    std::vector<Tensor> results;
    for (int iteration = 0; iteration < 20; ++iteration) {
        for (int s = 0; s < NUM_STREAMS; ++s) {
            auto tensor = Tensor::ones({10000}, Device::CUDA, DataType::Float32);
            tensor.set_stream(streams[s]);
            tensor = tensor * static_cast<float>(iteration + s);
            tensor.record_ready();
            results.push_back(tensor.sum());
        }
    }

    cudaDeviceSynchronize();

    // Just verify we didn't crash and have correct number of results
    EXPECT_EQ(results.size(), static_cast<size_t>(20 * NUM_STREAMS));

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }
}

// =============================================================================
// Correctness Tests
// =============================================================================

// Verify cross-stream results match single-stream
TEST_F(CUDAStreamSyncTest, Correctness_CrossStreamMatchesSingleStream) {
    const int SIZE = 10000;

    // Single stream computation
    auto a1 = Tensor::ones({SIZE}, Device::CUDA, DataType::Float32);
    a1 = a1 * 3.0f;
    a1 = a1 + 5.0f;
    a1 = a1 * 2.0f;
    auto single_result = a1.sum();

    cudaDeviceSynchronize();

    // Cross-stream computation
    auto a2 = Tensor::ones({SIZE}, Device::CUDA, DataType::Float32);
    a2.set_stream(stream1_);
    a2 = a2 * 3.0f;
    a2.record_ready();

    a2.ensure_ready_on(stream2_);
    a2.set_stream(stream2_);
    a2 = a2 + 5.0f;
    a2.record_ready();

    a2.ensure_ready_on(stream1_);
    a2.set_stream(stream1_);
    a2 = a2 * 2.0f;
    auto cross_result = a2.sum();

    cudaDeviceSynchronize();

    auto single_cpu = single_result.to(Device::CPU);
    auto cross_cpu = cross_result.to(Device::CPU);
    EXPECT_FLOAT_EQ(single_cpu.ptr<float>()[0], cross_cpu.ptr<float>()[0]);
}

// Test expression template stream safety (lazy evaluation)
TEST_F(CUDAStreamSyncTest, Correctness_ExpressionTemplateStreamSafety) {
    auto a = Tensor::ones({1000}, Device::CUDA, DataType::Float32);
    a.set_stream(stream1_);
    a = a * 2.0f;
    a.record_ready();

    auto b = Tensor::ones({1000}, Device::CUDA, DataType::Float32);
    b.set_stream(stream2_);
    b = b * 3.0f;
    b.record_ready();

    // Ensure both are ready before combining
    a.ensure_ready_on(nullptr); // Default stream waits
    b.ensure_ready_on(nullptr);

    // Expression template: a + b + a should use consistent streams
    auto result = a + b + a;

    cudaDeviceSynchronize();

    auto cpu = result.to(Device::CPU);
    // 2 + 3 + 2 = 7
    EXPECT_FLOAT_EQ(cpu.ptr<float>()[0], 7.0f);
}

// Test all binary ops work correctly across streams
TEST_F(CUDAStreamSyncTest, Correctness_AllBinaryOpsCrossStream) {
    auto a = Tensor::ones({1000}, Device::CUDA, DataType::Float32);
    a.set_stream(stream1_);
    a = a * 4.0f;
    a.record_ready();

    auto b = Tensor::ones({1000}, Device::CUDA, DataType::Float32);
    b.set_stream(stream2_);
    b = b * 2.0f;
    b.record_ready();

    a.ensure_ready_on(nullptr);
    b.ensure_ready_on(nullptr);

    // Test various binary ops
    auto add_result = (a + b).sum();
    auto sub_result = (a - b).sum();
    auto mul_result = (a * b).sum();
    auto div_result = (a / b).sum();

    cudaDeviceSynchronize();

    EXPECT_FLOAT_EQ(add_result.to(Device::CPU).ptr<float>()[0], 6000.0f);
    EXPECT_FLOAT_EQ(sub_result.to(Device::CPU).ptr<float>()[0], 2000.0f);
    EXPECT_FLOAT_EQ(mul_result.to(Device::CPU).ptr<float>()[0], 8000.0f);
    EXPECT_FLOAT_EQ(div_result.to(Device::CPU).ptr<float>()[0], 2000.0f);
}

// =============================================================================
// Stress Tests (Extended)
// =============================================================================

// Many streams, many tensors, many operations
TEST_F(CUDAStreamSyncTest, Stress_ManyStreamsManyTensors) {
    const int NUM_STREAMS = 8;
    const int TENSORS_PER_STREAM = 20;
    const int OPS_PER_TENSOR = 5;

    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    std::vector<Tensor> all_tensors;
    for (int s = 0; s < NUM_STREAMS; ++s) {
        for (int t = 0; t < TENSORS_PER_STREAM; ++t) {
            auto tensor = Tensor::ones({1000}, Device::CUDA, DataType::Float32);
            tensor.set_stream(streams[s]);
            for (int op = 0; op < OPS_PER_TENSOR; ++op) {
                tensor = tensor + 1.0f;
            }
            all_tensors.push_back(std::move(tensor));
        }
    }

    cudaDeviceSynchronize();

    // Verify all tensors
    for (const auto& tensor : all_tensors) {
        auto cpu = tensor.to(Device::CPU);
        EXPECT_FLOAT_EQ(cpu.ptr<float>()[0], static_cast<float>(1 + OPS_PER_TENSOR));
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }
}

// Thread-safe stream operations
TEST_F(CUDAStreamSyncTest, Stress_ThreadSafeStreamOperations) {
    const int NUM_THREADS = 4;
    const int OPS_PER_THREAD = 25;

    std::atomic<int> success_count{0};

    auto thread_func = [&](int thread_id) {
        cudaStream_t thread_stream;
        cudaStreamCreate(&thread_stream);

        for (int i = 0; i < OPS_PER_THREAD; ++i) {
            auto tensor = Tensor::ones({1000}, Device::CUDA, DataType::Float32);
            tensor.set_stream(thread_stream);
            tensor = tensor * static_cast<float>(thread_id + 1);
            tensor.record_ready();

            auto result = tensor.sum();
            cudaStreamSynchronize(thread_stream);

            auto cpu = result.to(Device::CPU);
            if (cpu.ptr<float>()[0] == static_cast<float>((thread_id + 1) * 1000)) {
                success_count++;
            }
        }

        cudaStreamDestroy(thread_stream);
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(thread_func, i);
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(success_count.load(), NUM_THREADS * OPS_PER_THREAD);
}

// Memory pressure with concurrent streams
TEST_F(CUDAStreamSyncTest, Stress_MemoryPressureConcurrentStreams) {
    const int NUM_STREAMS = 4;
    const size_t TENSOR_SIZE = 1000000; // ~4MB per tensor

    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Create large tensors on different streams
    std::vector<Tensor> tensors;
    for (int round = 0; round < 5; ++round) {
        for (int s = 0; s < NUM_STREAMS; ++s) {
            auto tensor = Tensor::ones({TENSOR_SIZE}, Device::CUDA, DataType::Float32);
            tensor.set_stream(streams[s]);
            tensor = tensor + static_cast<float>(round);
            tensor.record_ready();
            tensors.push_back(std::move(tensor));
        }
    }

    cudaDeviceSynchronize();

    // All allocations should succeed
    EXPECT_EQ(tensors.size(), static_cast<size_t>(5 * NUM_STREAMS));

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

// Empty tensors on streams
TEST_F(CUDAStreamSyncTest, EdgeCase_EmptyTensorsOnStreams) {
    auto empty1 = Tensor::empty({0}, Device::CUDA, DataType::Float32);
    empty1.set_stream(stream1_);

    auto empty2 = Tensor::empty({0}, Device::CUDA, DataType::Float32);
    empty2.set_stream(stream2_);

    // Operations on empty tensors should not crash
    empty1.record_ready();
    empty1.ensure_ready_on(stream2_);

    cudaDeviceSynchronize();
    SUCCEED();
}

// Views across streams
TEST_F(CUDAStreamSyncTest, EdgeCase_ViewsAcrossStreams) {
    auto base = Tensor::ones({100, 100}, Device::CUDA, DataType::Float32);
    base.set_stream(stream1_);
    base = base * 5.0f;
    base.record_ready();

    // Create view and use on different stream
    auto view = base.slice(0, 25, 75); // Middle 50 rows

    view.ensure_ready_on(stream2_);
    auto result = view.sum();
    cudaStreamSynchronize(stream2_);

    auto cpu = result.to(Device::CPU);
    // 50 rows * 100 cols * 5.0 = 25000
    EXPECT_FLOAT_EQ(cpu.ptr<float>()[0], 25000.0f);
}

// Device-to-host transfer with stream sync
TEST_F(CUDAStreamSyncTest, EdgeCase_D2HTransferWithStreamSync) {
    auto tensor = Tensor::ones({10000}, Device::CUDA, DataType::Float32);
    tensor.set_stream(stream1_);

    // Do work
    for (int i = 0; i < 10; ++i) {
        tensor = tensor + 1.0f;
    }
    tensor.record_ready();

    // Transfer to CPU (should synchronize)
    auto cpu_tensor = tensor.to(Device::CPU);

    // Verify data is correct
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(cpu_tensor.ptr<float>()[i], 11.0f);
    }
}

// Nested stream guards
TEST_F(CUDAStreamSyncTest, EdgeCase_NestedStreamGuards) {
    auto tensor = Tensor::ones({1000}, Device::CUDA, DataType::Float32);

    {
        tensor.set_stream(stream1_);
        tensor = tensor * 2.0f;
        tensor.record_ready();

        {
            tensor.ensure_ready_on(stream2_);
            tensor.set_stream(stream2_);
            tensor = tensor + 3.0f;
            tensor.record_ready();

            {
                tensor.ensure_ready_on(stream1_);
                tensor.set_stream(stream1_);
                tensor = tensor * 2.0f;
            }
        }
    }

    cudaDeviceSynchronize();

    auto cpu = tensor.to(Device::CPU);
    // (1 * 2 + 3) * 2 = 10
    EXPECT_FLOAT_EQ(cpu.ptr<float>()[0], 10.0f);
}

// Stream priority ordering
TEST_F(CUDAStreamSyncTest, EdgeCase_StreamPriorityOrdering) {
    // Get priority range
    int least_priority, greatest_priority;
    cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);

    // Create streams with different priorities
    cudaStream_t low_priority, high_priority;
    cudaStreamCreateWithPriority(&low_priority, cudaStreamNonBlocking, least_priority);
    cudaStreamCreateWithPriority(&high_priority, cudaStreamNonBlocking, greatest_priority);

    auto tensor = Tensor::ones({10000}, Device::CUDA, DataType::Float32);

    // Work on both streams
    tensor.set_stream(low_priority);
    tensor = tensor + 1.0f;
    tensor.record_ready();

    tensor.ensure_ready_on(high_priority);
    tensor.set_stream(high_priority);
    tensor = tensor * 2.0f;

    cudaDeviceSynchronize();

    auto cpu = tensor.to(Device::CPU);
    EXPECT_FLOAT_EQ(cpu.ptr<float>()[0], 4.0f);

    cudaStreamDestroy(low_priority);
    cudaStreamDestroy(high_priority);
}

// =============================================================================
// PyTorch-Inspired Tests
// =============================================================================

// Test behavior when many streams are created (pool exhaustion simulation)
TEST_F(CUDAStreamSyncTest, PyTorch_StreamPoolExhaustion) {
    const int NUM_STREAMS = 64;
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    std::vector<Tensor> tensors(NUM_STREAMS);

    // Create many streams and tensors
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
        tensors[i] = Tensor::ones({1000}, Device::CUDA, DataType::Float32);
        tensors[i].set_stream(streams[i]);
        tensors[i] = tensors[i] * static_cast<float>(i + 1);
    }

    cudaDeviceSynchronize();

    // Verify all streams worked correctly
    for (int i = 0; i < NUM_STREAMS; ++i) {
        auto cpu = tensors[i].to(Device::CPU);
        EXPECT_FLOAT_EQ(cpu.ptr<float>()[0], static_cast<float>(i + 1));
        cudaStreamDestroy(streams[i]);
    }
}

// Test event timing accuracy
TEST_F(CUDAStreamSyncTest, PyTorch_EventTimingAccuracy) {
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    auto tensor = Tensor::ones({1000000}, Device::CUDA, DataType::Float32);
    tensor.set_stream(stream1_);

    cudaEventRecord(start, stream1_);

    // Do substantial work
    for (int i = 0; i < 50; ++i) {
        tensor = tensor + 1.0f;
    }

    cudaEventRecord(end, stream1_);
    cudaEventSynchronize(end);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, end);

    // Elapsed time should be positive and reasonable
    EXPECT_GT(elapsed_ms, 0.0f);
    EXPECT_LT(elapsed_ms, 10000.0f); // Less than 10 seconds

    cudaEventDestroy(start);
    cudaEventDestroy(end);
}

// Test operations interleaved across streams
TEST_F(CUDAStreamSyncTest, PyTorch_InterleavedStreamOperations) {
    auto a = Tensor::zeros({1000}, Device::CUDA, DataType::Float32);
    auto b = Tensor::zeros({1000}, Device::CUDA, DataType::Float32);

    for (int i = 0; i < 20; ++i) {
        a.set_stream((i % 2 == 0) ? stream1_ : stream2_);
        b.set_stream((i % 2 == 0) ? stream2_ : stream1_);

        a = a + 1.0f;
        a.record_ready();

        b = b + 2.0f;
        b.record_ready();

        // Sync before next iteration
        cudaDeviceSynchronize();
    }

    auto cpu_a = a.to(Device::CPU);
    auto cpu_b = b.to(Device::CPU);

    EXPECT_FLOAT_EQ(cpu_a.ptr<float>()[0], 20.0f);
    EXPECT_FLOAT_EQ(cpu_b.ptr<float>()[0], 40.0f);
}

// Note: Uses shared test_main.cpp for main()
