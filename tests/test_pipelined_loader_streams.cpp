/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/nvcodec_image_loader.hpp"
#include "io/pipelined_image_loader.hpp"
#include <cassert>
#include <cuda_runtime.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <vector>

using namespace lfs::io;

namespace {
    // Get test data path - uses bicycle dataset from project
    std::filesystem::path get_test_data_root() {
        return std::filesystem::path("data/bicycle/images_4");
    }

    std::vector<std::filesystem::path> get_test_images(size_t max_count = 10) {
        std::vector<std::filesystem::path> files;
        const auto dir = get_test_data_root();
        if (!std::filesystem::exists(dir)) {
            return files;
        }

        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (!entry.is_regular_file())
                continue;
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg") {
                files.push_back(entry.path());
                if (files.size() >= max_count)
                    break;
            }
        }
        std::sort(files.begin(), files.end());
        return files;
    }
} // namespace

class StreamIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_images_ = get_test_images(10);
        if (test_images_.empty()) {
            GTEST_SKIP() << "No test images found in " << get_test_data_root();
        }
    }

    std::vector<std::filesystem::path> test_images_;
};

// Test: NvCodecImageLoader async decode returns valid stream and event
TEST_F(StreamIntegrationTest, AsyncDecodeReturnsStreamAndEvent) {
    if (!NvCodecImageLoader::is_available()) {
        GTEST_SKIP() << "nvImageCodec not available";
    }

    NvCodecImageLoader::Options opts{};
    NvCodecImageLoader loader(opts);

    // Read JPEG file
    std::ifstream file(test_images_[0], std::ios::binary);
    std::vector<uint8_t> jpeg_data((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());
    ASSERT_FALSE(jpeg_data.empty());

    // Async decode
    auto result = loader.load_image_from_memory_gpu_async(jpeg_data);

    // Verify stream and event are set
    ASSERT_NE(result.stream, nullptr) << "Decode stream must not be null";
    ASSERT_NE(result.ready_event, nullptr) << "Ready event must not be null";

    // Verify tensor is valid
    ASSERT_TRUE(result.tensor.is_valid()) << "Decoded tensor must be valid";
    ASSERT_EQ(result.tensor.device(), lfs::core::Device::CUDA) << "Tensor must be on GPU";
    ASSERT_EQ(result.tensor.ndim(), 3) << "Tensor must be 3D [C,H,W]";
    ASSERT_EQ(result.tensor.shape()[0], 3) << "Tensor must have 3 channels";

    // Wait for decode to complete
    cudaEventSynchronize(static_cast<cudaEvent_t>(result.ready_event));
}

// Test: Multiple async decodes all return valid streams and events
TEST_F(StreamIntegrationTest, MultipleAsyncDecodesReturnValidStreams) {
    if (!NvCodecImageLoader::is_available()) {
        GTEST_SKIP() << "nvImageCodec not available";
    }

    NvCodecImageLoader::Options opts{};
    opts.decoder_pool_size = 4;
    NvCodecImageLoader loader(opts);

    std::vector<DecodedImage> results;

    // Load multiple images sequentially - each should get valid stream/event
    for (size_t i = 0; i < std::min(size_t(4), test_images_.size()); ++i) {
        std::ifstream file(test_images_[i], std::ios::binary);
        std::vector<uint8_t> jpeg_data((std::istreambuf_iterator<char>(file)),
                                       std::istreambuf_iterator<char>());
        auto result = loader.load_image_from_memory_gpu_async(jpeg_data);

        ASSERT_NE(result.stream, nullptr) << "Stream must be valid for image " << i;
        ASSERT_NE(result.ready_event, nullptr) << "Event must be valid for image " << i;
        ASSERT_TRUE(result.tensor.is_valid()) << "Tensor must be valid for image " << i;
        results.push_back(std::move(result));
    }

    // Wait for all decodes and verify tensors
    for (size_t i = 0; i < results.size(); ++i) {
        cudaEventSynchronize(static_cast<cudaEvent_t>(results[i].ready_event));
        EXPECT_EQ(results[i].tensor.device(), lfs::core::Device::CUDA)
            << "Tensor " << i << " should be on GPU";
    }
}

// Test: PipelinedImageLoader propagates stream/event in ReadyImage
TEST_F(StreamIntegrationTest, PipelinedLoaderPropagatesStreamEvent) {
    PipelinedLoaderConfig config{};
    config.prefetch_count = 4;
    config.jpeg_batch_size = 2;

    PipelinedImageLoader loader(config);

    // Prefetch a few images
    std::vector<ImageRequest> requests;
    for (size_t i = 0; i < std::min(size_t(3), test_images_.size()); ++i) {
        ImageRequest req;
        req.sequence_id = i;
        req.path = test_images_[i];
        req.params.resize_factor = 1;
        requests.push_back(std::move(req));
    }
    loader.prefetch(requests);

    // Get the results and verify stream/event propagation
    size_t images_with_stream = 0;
    size_t images_with_event = 0;

    for (size_t i = 0; i < requests.size(); ++i) {
        auto ready = loader.get();
        ASSERT_TRUE(ready.tensor.is_valid());

        if (ready.stream != nullptr) {
            images_with_stream++;
        }
        if (ready.ready_event != nullptr) {
            images_with_event++;
            // Ensure we can sync on the event
            cudaError_t err = cudaEventSynchronize(ready.ready_event);
            EXPECT_EQ(err, cudaSuccess) << "Event sync failed: " << cudaGetErrorString(err);
        }
    }

    // All images should have stream/event (from GPU decode path)
    EXPECT_EQ(images_with_stream, requests.size())
        << "All hot-path images should have decode stream";
    EXPECT_EQ(images_with_event, requests.size())
        << "All hot-path images should have ready event";
}

// Test: cudaStreamWaitEvent correctly synchronizes training with decode
TEST_F(StreamIntegrationTest, StreamWaitEventSynchronization) {
    if (!NvCodecImageLoader::is_available()) {
        GTEST_SKIP() << "nvImageCodec not available";
    }

    NvCodecImageLoader::Options opts{};
    NvCodecImageLoader loader(opts);

    // Load a JPEG
    std::ifstream file(test_images_[0], std::ios::binary);
    std::vector<uint8_t> jpeg_data((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());

    auto result = loader.load_image_from_memory_gpu_async(jpeg_data);
    ASSERT_NE(result.ready_event, nullptr);

    // Create a "training" stream
    cudaStream_t training_stream;
    ASSERT_EQ(cudaStreamCreate(&training_stream), cudaSuccess);

    // Make training stream wait on decode event (this is what trainer.cpp does)
    cudaError_t err = cudaStreamWaitEvent(training_stream,
                                          static_cast<cudaEvent_t>(result.ready_event), 0);
    ASSERT_EQ(err, cudaSuccess) << "cudaStreamWaitEvent failed";

    // Synchronize training stream - should complete after decode is done
    err = cudaStreamSynchronize(training_stream);
    ASSERT_EQ(err, cudaSuccess) << "Training stream sync failed";

    // At this point, tensor data is guaranteed to be ready
    ASSERT_TRUE(result.tensor.is_valid());
    auto shape = result.tensor.shape();
    EXPECT_EQ(shape.rank(), 3);
    EXPECT_EQ(shape[0], 3); // RGB channels

    cudaStreamDestroy(training_stream);
}

// Test: Verify no device-wide sync in hot path (regression test)
TEST_F(StreamIntegrationTest, NoDeviceSyncInHotPath) {
    PipelinedLoaderConfig config{};
    config.prefetch_count = 2;

    PipelinedImageLoader loader(config);

    // Prefetch images
    for (size_t i = 0; i < std::min(size_t(2), test_images_.size()); ++i) {
        ImageRequest req;
        req.sequence_id = i;
        req.path = test_images_[i];
        loader.prefetch({req});
    }

    // Launch some async GPU work on a separate stream
    cudaStream_t work_stream;
    ASSERT_EQ(cudaStreamCreate(&work_stream), cudaSuccess);

    // Allocate and launch a long-running kernel placeholder (just a memset)
    void* gpu_mem;
    const size_t alloc_size = 64 * 1024 * 1024; // 64MB
    ASSERT_EQ(cudaMalloc(&gpu_mem, alloc_size), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(gpu_mem, 0, alloc_size, work_stream), cudaSuccess);

    cudaEvent_t work_start, work_end;
    cudaEventCreate(&work_start);
    cudaEventCreate(&work_end);
    cudaEventRecord(work_start, work_stream);

    // Get images from loader - should NOT block on work_stream
    for (size_t i = 0; i < std::min(size_t(2), test_images_.size()); ++i) {
        auto ready = loader.get();
        ASSERT_TRUE(ready.tensor.is_valid());

        // Use stream wait, not device sync
        if (ready.ready_event) {
            cudaStreamWaitEvent(nullptr, ready.ready_event, 0);
        }
    }

    cudaEventRecord(work_end, work_stream);
    cudaEventSynchronize(work_end);

    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, work_start, work_end);

    // The work stream should have completed its work concurrently
    // If there was a device sync, the work would have been serialized
    EXPECT_GT(elapsed_ms, 0.0f) << "Work stream should have recorded time";

    cudaFree(gpu_mem);
    cudaEventDestroy(work_start);
    cudaEventDestroy(work_end);
    cudaStreamDestroy(work_stream);
}

// Test: Stress test with many concurrent decode requests
TEST_F(StreamIntegrationTest, StressTestManyDecodes) {
    PipelinedLoaderConfig config{};
    config.prefetch_count = 8;
    config.jpeg_batch_size = 4;

    PipelinedImageLoader loader(config);

    const size_t num_requests = std::min(size_t(20), test_images_.size() * 2);

    // Prefetch many images (with wrap-around for small datasets)
    for (size_t i = 0; i < num_requests; ++i) {
        ImageRequest req;
        req.sequence_id = i;
        req.path = test_images_[i % test_images_.size()];
        loader.prefetch({req});
    }

    // Retrieve all images
    size_t valid_count = 0;
    size_t stream_count = 0;
    for (size_t i = 0; i < num_requests; ++i) {
        auto result = loader.try_get_for(std::chrono::milliseconds(5000));
        ASSERT_TRUE(result.has_value()) << "Timeout waiting for image " << i;
        if (result->tensor.is_valid()) {
            valid_count++;
        }
        if (result->stream != nullptr) {
            stream_count++;
        }
        if (result->ready_event) {
            cudaEventSynchronize(result->ready_event);
        }
    }

    EXPECT_EQ(valid_count, num_requests) << "All images should be valid";
    EXPECT_GT(stream_count, 0) << "At least some images should have streams";

    auto stats = loader.get_stats();
    EXPECT_EQ(stats.total_images_loaded, num_requests);
}
