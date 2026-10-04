/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "portable/Pipeline.hpp"

#include <sycl/sycl.hpp>

#include <array>
#include <cstddef>
#include <iostream>
#include <map>
#include <mutex>
#include <vector>

namespace cctag {
namespace portable {

namespace {

/// The device images of one pyramid level, tightly packed
struct Level {
    std::uint32_t width;
    std::uint32_t height;
    std::uint8_t* source;
    std::int16_t* dx;
    std::int16_t* dy;
};

/// A 9-tap filter, centred on the 5th tap
using Taps = std::array<float, 9>;

/// The two halves of h_gauss_filter in cuda/frame_02_gaussian.cu, whose outer product is the 9x9
/// kernel of cvRecodedCanny
constexpr Taps gauss{
    0.000053390535453f,
    0.001768051711852f,
    0.021539279301849f,
    0.096532352630054f,
    0.159154943091895f,
    0.096532352630054f,
    0.021539279301849f,
    0.001768051711852f,
    0.000053390535453f
};
constexpr Taps derivative{
    -0.002683701023220f,
    -0.066653979229454f,
    -0.541341132946452f,
    -1.213061319425269f,
    0.0f,
    1.213061319425269f,
    0.541341132946452f,
    0.066653979229454f,
    0.002683701023220f
};

} // namespace

struct Pipeline {
    sycl::queue queue{sycl::property::queue::in_order{}};
    std::vector<Level> levels;
    /// Scratch image between the two passes of a separable filter, the size of level 0
    float* intermediate = nullptr;
};

namespace {

void release(Pipeline& pipeline) {
    pipeline.queue.wait();
    for (const Level& level : pipeline.levels) {
        sycl::free(level.source, pipeline.queue);
        sycl::free(level.dx, pipeline.queue);
        sycl::free(level.dy, pipeline.queue);
    }
    pipeline.levels.clear();
    sycl::free(pipeline.intermediate, pipeline.queue);
    pipeline.intermediate = nullptr;
}

void allocate(Pipeline& pipeline, std::uint32_t width, std::uint32_t height, std::uint32_t count) {
    const std::vector<Level>& levels = pipeline.levels;
    if (levels.size() == count && count > 0 && levels[0].width == width
        && levels[0].height == height) {
        return;
    }

    release(pipeline);
    pipeline.intermediate = sycl::malloc_device<float>(std::size_t(width) * height, pipeline.queue);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::size_t size = std::size_t(width) * height;
        pipeline.levels.push_back(
            {width,
             height,
             sycl::malloc_device<std::uint8_t>(size, pipeline.queue),
             sycl::malloc_device<std::int16_t>(size, pipeline.queue),
             sycl::malloc_device<std::int16_t>(size, pipeline.queue)}
        );
        width /= 2;
        height /= 2;
    }
}

/// Halves an image by averaging every 2x2 block, as cv::resize does when it halves
void downsample(sycl::queue& queue, const Level& src, const Level& dst) {
    const std::uint8_t* in = src.source;
    std::uint8_t* out = dst.source;
    const std::size_t inWidth = src.width;
    const std::size_t outWidth = dst.width;

    queue.parallel_for(sycl::range<2>(dst.height, dst.width), [=](sycl::id<2> id) {
        const std::size_t y = id[0];
        const std::size_t x = id[1];
        const std::uint8_t* top = in + 2 * y * inWidth + 2 * x;
        const std::uint8_t* bottom = top + inWidth;
        const std::uint32_t sum = top[0] + top[1] + bottom[0] + bottom[1];
        out[y * outWidth + x] = static_cast<std::uint8_t>((sum + 2) / 4);
    });
}

/// Filters every column of the source image, replicating the border rows
void filterColumns(sycl::queue& queue, const Level& level, const Taps taps, float* out) {
    const std::uint8_t* in = level.source;
    const int width = level.width;
    const int height = level.height;

    queue.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> id) {
        const int y = id[0];
        const int x = id[1];
        float sum = 0.0f;
        for (int k = 0; k < 9; ++k) {
            const int row = sycl::clamp(y + k - 4, 0, height - 1);
            sum += taps[k] * in[row * width + x];
        }
        out[y * width + x] = sum;
    });
}

/// Filters every row of an intermediate image, replicating the border columns and rounding to
/// nearest as cv::saturate_cast does
void filterRows(
    sycl::queue& queue,
    const Level& level,
    const float* in,
    const Taps taps,
    std::int16_t* out
) {
    const int width = level.width;
    const int height = level.height;

    queue.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> id) {
        const int y = id[0];
        const int x = id[1];
        float sum = 0.0f;
        for (int k = 0; k < 9; ++k) {
            const int column = sycl::clamp(x + k - 4, 0, width - 1);
            sum += taps[k] * in[y * width + column];
        }
        out[y * width + x] = static_cast<std::int16_t>(sycl::rint(sum));
    });
}

/// Computes dx and dy of a level as cvRecodedCanny does, with separable filters
void gradient(Pipeline& pipeline, const Level& level) {
    filterColumns(pipeline.queue, level, gauss, pipeline.intermediate);
    filterRows(pipeline.queue, level, pipeline.intermediate, derivative, level.dx);
    filterColumns(pipeline.queue, level, derivative, pipeline.intermediate);
    filterRows(pipeline.queue, level, pipeline.intermediate, gauss, level.dy);
}

} // namespace

Pipeline& pipeline(int pipeId) {
    // Never destroyed: AdaptiveCpp cannot free device memory from a static destructor
    static std::map<int, Pipeline>& pipelines = *new std::map<int, Pipeline>;
    static std::mutex mutex;

    const std::lock_guard<std::mutex> lock(mutex);
    const auto inserted = pipelines.try_emplace(pipeId);
    Pipeline& pipeline = inserted.first->second;
    if (inserted.second) {
        std::cerr << "SYCL device: "
                  << pipeline.queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    }
    return pipeline;
}

void load(
    Pipeline& pipeline,
    const std::uint8_t* image,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t levels
) {
    allocate(pipeline, width, height, levels);

    const std::vector<Level>& pyramid = pipeline.levels;
    pipeline.queue.memcpy(pyramid[0].source, image, std::size_t(width) * height);
    for (std::size_t i = 1; i < pyramid.size(); ++i) {
        downsample(pipeline.queue, pyramid[i - 1], pyramid[i]);
    }
    for (const Level& level : pyramid) {
        gradient(pipeline, level);
    }
}

void downloadSource(Pipeline& pipeline, std::uint32_t level, std::uint8_t* dst) {
    const Level& src = pipeline.levels[level];
    pipeline.queue.memcpy(dst, src.source, std::size_t(src.width) * src.height).wait();
}

void downloadGradient(Pipeline& pipeline, std::uint32_t level, std::int16_t* dx, std::int16_t* dy) {
    const Level& src = pipeline.levels[level];
    const std::size_t size = std::size_t(src.width) * src.height;
    pipeline.queue.memcpy(dx, src.dx, size * sizeof(std::int16_t));
    pipeline.queue.memcpy(dy, src.dy, size * sizeof(std::int16_t)).wait();
}

} // namespace portable
} // namespace cctag
