/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * Stress tests for NvCodecImageLoader to reproduce and verify fix for
 * the black images bug that occurs after ~6700 iterations.
 */

#include "io/nvcodec_image_loader.hpp"
#include "core/logger.hpp"
#include <atomic>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using namespace lfs::io;
using namespace lfs::core;

namespace {
    std::filesystem::path get_test_data_root() {
        return std::filesystem::path("data/bicycle/images_4");
    }

    std::vector<std::filesystem::path> get_test_images(size_t max_count = 100) {
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

    std::vector<uint8_t> read_file(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open: " + path.string());
        }
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<uint8_t> buffer(size);
        file.read(reinterpret_cast<char*>(buffer.data()), size);
        return buffer;
    }
} // namespace

class NvCodecStressTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!NvCodecImageLoader::is_available()) {
            GTEST_SKIP() << "nvImageCodec not available";
        }

        test_images_ = get_test_images(50);
        if (test_images_.empty()) {
            GTEST_SKIP() << "No test images found in " << get_test_data_root();
        }

        // Pre-load JPEG data for faster testing
        for (const auto& path : test_images_) {
            jpeg_data_.push_back(read_file(path));
        }
    }

    std::vector<std::filesystem::path> test_images_;
    std::vector<std::vector<uint8_t>> jpeg_data_;
};

/**
 * Test 1: Sequential decodes - decode same image many times
 * This tests if the decoder state gets corrupted over many iterations.
 */
TEST_F(NvCodecStressTest, SequentialDecodesNoBlackImages) {
    NvCodecImageLoader::Options opts{};
    opts.decoder_pool_size = 8;
    NvCodecImageLoader loader(opts);

    constexpr size_t NUM_ITERATIONS = 10000;
    size_t black_count = 0;
    size_t first_black_iter = 0;

    LOG_INFO("[StressTest] Starting {} sequential decodes", NUM_ITERATIONS);

    for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
        const auto& jpeg = jpeg_data_[i % jpeg_data_.size()];

        auto result = loader.load_image_from_memory_gpu_async(jpeg);

        // Wait for decode to complete
        cudaEventSynchronize(static_cast<cudaEvent_t>(result.ready_event));

        // Check for black image
        if (result.tensor.is_valid()) {
            float max_val = result.tensor.max().item<float>();
            if (max_val == 0.0f) {
                if (black_count == 0) {
                    first_black_iter = i;
                    LOG_ERROR("[StressTest] FIRST black image at iteration {}", i);
                }
                black_count++;
            }
        } else {
            LOG_ERROR("[StressTest] Invalid tensor at iteration {}", i);
            black_count++;
        }

        // Cleanup stream and event (async creates new stream per call)
        cudaStreamDestroy(static_cast<cudaStream_t>(result.stream));
        cudaEventDestroy(static_cast<cudaEvent_t>(result.ready_event));

        // Progress logging every 1000 iterations
        if ((i + 1) % 1000 == 0) {
            LOG_INFO("[StressTest] Progress: {}/{}, black images so far: {}",
                     i + 1, NUM_ITERATIONS, black_count);
        }
    }

    LOG_INFO("[StressTest] Completed {} decodes, {} black images",
             NUM_ITERATIONS, black_count);

    if (black_count > 0) {
        LOG_ERROR("[StressTest] First black image at iteration {}", first_black_iter);
    }

    EXPECT_EQ(black_count, 0) << "Found " << black_count << " black images, first at iteration " << first_black_iter;
}

/**
 * Test 2: Parallel decodes from multiple threads
 * This tests for race conditions in the decoder pool.
 */
TEST_F(NvCodecStressTest, ParallelDecodesNoBlackImages) {
    NvCodecImageLoader::Options opts{};
    opts.decoder_pool_size = 8;
    NvCodecImageLoader loader(opts);

    constexpr size_t NUM_THREADS = 8;
    constexpr size_t ITERATIONS_PER_THREAD = 1000;

    std::atomic<size_t> total_black_count{0};
    std::atomic<size_t> total_decodes{0};
    std::vector<std::thread> threads;

    LOG_INFO("[StressTest] Starting {} threads x {} iterations = {} parallel decodes",
             NUM_THREADS, ITERATIONS_PER_THREAD, NUM_THREADS * ITERATIONS_PER_THREAD);

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t] {
            size_t local_black = 0;
            for (size_t i = 0; i < ITERATIONS_PER_THREAD; ++i) {
                const auto& jpeg = jpeg_data_[(t * ITERATIONS_PER_THREAD + i) % jpeg_data_.size()];

                try {
                    auto result = loader.load_image_from_memory_gpu_async(jpeg);
                    cudaEventSynchronize(static_cast<cudaEvent_t>(result.ready_event));

                    if (result.tensor.is_valid()) {
                        float max_val = result.tensor.max().item<float>();
                        if (max_val == 0.0f) {
                            local_black++;
                            LOG_ERROR("[StressTest] Thread {} iter {}: BLACK IMAGE", t, i);
                        }
                    }

                    cudaStreamDestroy(static_cast<cudaStream_t>(result.stream));
                    cudaEventDestroy(static_cast<cudaEvent_t>(result.ready_event));
                    total_decodes++;
                } catch (const std::exception& e) {
                    LOG_ERROR("[StressTest] Thread {} iter {}: Exception: {}", t, i, e.what());
                    local_black++;
                }
            }
            total_black_count += local_black;
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    LOG_INFO("[StressTest] Parallel test complete: {} total decodes, {} black images",
             total_decodes.load(), total_black_count.load());

    EXPECT_EQ(total_black_count.load(), 0) << "Found " << total_black_count.load() << " black images in parallel test";
}

/**
 * Test 3: Rapid fire - many requests without waiting
 * This tests if the decoder pool handles backpressure correctly.
 */
TEST_F(NvCodecStressTest, RapidFireNoBlackImages) {
    NvCodecImageLoader::Options opts{};
    opts.decoder_pool_size = 4;  // Small pool to stress it
    NvCodecImageLoader loader(opts);

    constexpr size_t BATCH_SIZE = 20;
    constexpr size_t NUM_BATCHES = 100;

    size_t black_count = 0;

    LOG_INFO("[StressTest] Starting rapid fire: {} batches x {} images", NUM_BATCHES, BATCH_SIZE);

    for (size_t batch = 0; batch < NUM_BATCHES; ++batch) {
        std::vector<DecodedImage> pending;

        // Fire all requests
        for (size_t i = 0; i < BATCH_SIZE; ++i) {
            const auto& jpeg = jpeg_data_[(batch * BATCH_SIZE + i) % jpeg_data_.size()];
            pending.push_back(loader.load_image_from_memory_gpu_async(jpeg));
        }

        // Wait for all and check
        for (size_t i = 0; i < pending.size(); ++i) {
            cudaEventSynchronize(static_cast<cudaEvent_t>(pending[i].ready_event));

            if (pending[i].tensor.is_valid()) {
                float max_val = pending[i].tensor.max().item<float>();
                if (max_val == 0.0f) {
                    black_count++;
                    LOG_ERROR("[StressTest] Batch {} item {}: BLACK IMAGE", batch, i);
                }
            }

            cudaStreamDestroy(static_cast<cudaStream_t>(pending[i].stream));
            cudaEventDestroy(static_cast<cudaEvent_t>(pending[i].ready_event));
        }

        if ((batch + 1) % 20 == 0) {
            LOG_INFO("[StressTest] Progress: {}/{} batches, black images: {}",
                     batch + 1, NUM_BATCHES, black_count);
        }
    }

    LOG_INFO("[StressTest] Rapid fire complete: {} total decodes, {} black images",
             NUM_BATCHES * BATCH_SIZE, black_count);

    EXPECT_EQ(black_count, 0) << "Found " << black_count << " black images in rapid fire test";
}

/**
 * Test 4: Async without sync - similar to training loop usage
 * Fire async requests and sync later, simulating real training pipeline.
 */
TEST_F(NvCodecStressTest, AsyncPipelineNoBlackImages) {
    NvCodecImageLoader::Options opts{};
    opts.decoder_pool_size = 8;
    NvCodecImageLoader loader(opts);

    constexpr size_t NUM_ITERATIONS = 8000;
    constexpr size_t PIPELINE_DEPTH = 4;  // Number of in-flight decodes

    std::vector<DecodedImage> pipeline;
    size_t black_count = 0;
    size_t first_black_iter = 0;

    LOG_INFO("[StressTest] Starting async pipeline: {} iterations, depth {}",
             NUM_ITERATIONS, PIPELINE_DEPTH);

    // Fill initial pipeline
    for (size_t i = 0; i < PIPELINE_DEPTH && i < NUM_ITERATIONS; ++i) {
        const auto& jpeg = jpeg_data_[i % jpeg_data_.size()];
        pipeline.push_back(loader.load_image_from_memory_gpu_async(jpeg));
    }

    for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
        // Get the oldest result
        auto& result = pipeline[i % PIPELINE_DEPTH];

        // Wait for it
        cudaEventSynchronize(static_cast<cudaEvent_t>(result.ready_event));

        // Check for black
        if (result.tensor.is_valid()) {
            float max_val = result.tensor.max().item<float>();
            if (max_val == 0.0f) {
                if (black_count == 0) {
                    first_black_iter = i;
                    LOG_ERROR("[StressTest] FIRST black image at iteration {}", i);
                }
                black_count++;
            }
        }

        // Cleanup
        cudaStreamDestroy(static_cast<cudaStream_t>(result.stream));
        cudaEventDestroy(static_cast<cudaEvent_t>(result.ready_event));

        // Start next decode if there are more
        size_t next_idx = i + PIPELINE_DEPTH;
        if (next_idx < NUM_ITERATIONS) {
            const auto& jpeg = jpeg_data_[next_idx % jpeg_data_.size()];
            result = loader.load_image_from_memory_gpu_async(jpeg);
        }

        if ((i + 1) % 1000 == 0) {
            LOG_INFO("[StressTest] Progress: {}/{}, black images: {}",
                     i + 1, NUM_ITERATIONS, black_count);
        }
    }

    LOG_INFO("[StressTest] Async pipeline complete: {} black images", black_count);

    if (black_count > 0) {
        LOG_ERROR("[StressTest] First black image at iteration {}", first_black_iter);
    }

    EXPECT_EQ(black_count, 0) << "Found " << black_count << " black images, first at " << first_black_iter;
}

/**
 * Test 5: Synchronous decode stress test
 * Use the sync API to check if the bug is async-specific.
 */
TEST_F(NvCodecStressTest, SyncDecodesNoBlackImages) {
    NvCodecImageLoader::Options opts{};
    opts.decoder_pool_size = 8;
    NvCodecImageLoader loader(opts);

    constexpr size_t NUM_ITERATIONS = 10000;
    size_t black_count = 0;

    LOG_INFO("[StressTest] Starting {} synchronous decodes", NUM_ITERATIONS);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
        const auto& jpeg = jpeg_data_[i % jpeg_data_.size()];

        // Use synchronous API
        auto tensor = loader.load_image_from_memory_gpu(jpeg, 1, -1, stream);

        // Already synchronized by the sync API
        if (tensor.is_valid()) {
            float max_val = tensor.max().item<float>();
            if (max_val == 0.0f) {
                black_count++;
                LOG_ERROR("[StressTest] BLACK at iteration {}", i);
            }
        }

        if ((i + 1) % 1000 == 0) {
            LOG_INFO("[StressTest] Progress: {}/{}, black: {}", i + 1, NUM_ITERATIONS, black_count);
        }
    }

    cudaStreamDestroy(stream);

    LOG_INFO("[StressTest] Sync test complete: {} black images", black_count);
    EXPECT_EQ(black_count, 0);
}

/**
 * Test 6: Simple decode test - verify a few decodes work correctly.
 * Use this to debug the raw decode check issue.
 */
TEST_F(NvCodecStressTest, SimpleDecodeDebug) {
    NvCodecImageLoader::Options opts{};
    opts.decoder_pool_size = 1;  // Single decoder to simplify
    NvCodecImageLoader loader(opts);

    // Just decode 10 images and print detailed info
    for (size_t i = 0; i < 10; ++i) {
        const auto& jpeg = jpeg_data_[i];
        LOG_INFO("[Debug] Decoding image {} ({} bytes)", i, jpeg.size());

        auto result = loader.load_image_from_memory_gpu_async(jpeg);
        cudaEventSynchronize(static_cast<cudaEvent_t>(result.ready_event));

        if (result.tensor.is_valid()) {
            float min_val = result.tensor.min().item<float>();
            float max_val = result.tensor.max().item<float>();
            LOG_INFO("[Debug] Image {}: shape=[{},{},{}], min={:.4f}, max={:.4f}",
                     i, result.tensor.shape()[0], result.tensor.shape()[1],
                     result.tensor.shape()[2], min_val, max_val);

            if (max_val == 0.0f) {
                FAIL() << "Black image at iteration " << i;
            }
        } else {
            FAIL() << "Invalid tensor at iteration " << i;
        }

        cudaStreamDestroy(static_cast<cudaStream_t>(result.stream));
        cudaEventDestroy(static_cast<cudaEvent_t>(result.ready_event));
    }
}

/**
 * Test 7: Verify tensor content is actually valid (not just non-zero)
 * Check that pixel values are in expected range.
 */
TEST_F(NvCodecStressTest, ValidImageContent) {
    NvCodecImageLoader::Options opts{};
    NvCodecImageLoader loader(opts);

    constexpr size_t NUM_ITERATIONS = 1000;
    size_t invalid_count = 0;

    for (size_t i = 0; i < NUM_ITERATIONS; ++i) {
        const auto& jpeg = jpeg_data_[i % jpeg_data_.size()];
        auto result = loader.load_image_from_memory_gpu_async(jpeg);
        cudaEventSynchronize(static_cast<cudaEvent_t>(result.ready_event));

        if (result.tensor.is_valid()) {
            float min_val = result.tensor.min().item<float>();
            float max_val = result.tensor.max().item<float>();

            // Valid images should have values in [0, 1] range (normalized)
            // and have some variation (not all same value)
            if (max_val == 0.0f) {
                invalid_count++;
                LOG_ERROR("[StressTest] Black image at {}", i);
            } else if (max_val < 0.0f || max_val > 1.0f) {
                invalid_count++;
                LOG_ERROR("[StressTest] Invalid range at {}: min={}, max={}", i, min_val, max_val);
            } else if (max_val == min_val) {
                invalid_count++;
                LOG_ERROR("[StressTest] Constant image at {}: val={}", i, max_val);
            }
        } else {
            invalid_count++;
            LOG_ERROR("[StressTest] Invalid tensor at {}", i);
        }

        cudaStreamDestroy(static_cast<cudaStream_t>(result.stream));
        cudaEventDestroy(static_cast<cudaEvent_t>(result.ready_event));
    }

    EXPECT_EQ(invalid_count, 0) << "Found " << invalid_count << " invalid images";
}
