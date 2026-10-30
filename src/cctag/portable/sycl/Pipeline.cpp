/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/portable/Pipeline.hpp>
#include <cctag/portable/sycl/Stages.hpp>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <iostream>
#include <map>
#include <mutex>
#include <vector>

namespace cctag {
namespace portable {

struct Pipeline {
    sycl::queue queue{sycl::property::queue::in_order{}};
    std::vector<Level> levels;
    /// Temporary image the size of level 0, can be reused between passes
    float* intermediate = nullptr;
};

namespace {

void release(Pipeline& pipeline) {
    pipeline.queue.wait();
    for (const Level& level : pipeline.levels) {
        sycl::free(level.source, pipeline.queue);
        sycl::free(level.dx, pipeline.queue);
        sycl::free(level.dy, pipeline.queue);
        sycl::free(level.map, pipeline.queue);
    }
    pipeline.levels.clear();
    sycl::free(pipeline.intermediate, pipeline.queue);
    pipeline.intermediate = nullptr;
}

void allocate(Pipeline& pipeline, std::uint32_t width, std::uint32_t height, std::uint32_t count) {
    const std::vector<Level>& levels = pipeline.levels;
    if (levels.size() == count
        && count > 0
        && levels[0].width == width
        && levels[0].height == height) {
        return;
    }

    release(pipeline);
    pipeline.intermediate = sycl::malloc_device<float>(std::size_t(width) * height, pipeline.queue);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::size_t size = std::size_t(width) * height;
        // TODO: Struct initializer?
        pipeline.levels.push_back(
            {width,
             height,
             sycl::malloc_device<std::uint8_t>(size, pipeline.queue),
             sycl::malloc_device<std::int16_t>(size, pipeline.queue),
             sycl::malloc_device<std::int16_t>(size, pipeline.queue),
             sycl::malloc_device<Mark>(size, pipeline.queue)}
        );
        width /= 2;
        height /= 2;
    }
}

} // namespace

Pipeline& pipeline(int pipe_id) {
    // Never destroyed: AdaptiveCpp cannot free device memory from a static destructor
    static std::map<int, Pipeline>& pipelines = *new std::map<int, Pipeline>;
    static std::mutex mutex;

    const std::lock_guard<std::mutex> lock(mutex);
    const auto inserted = pipelines.try_emplace(pipe_id);
    Pipeline& pipeline = inserted.first->second;
    if (inserted.second) {
        std::cerr
            << "SYCL device: "
            << pipeline.queue.get_device().get_info<sycl::info::device::name>()
            << std::endl;
    }
    return pipeline;
}

void load(
    Pipeline& pipeline,
    const std::uint8_t* image,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t levels,
    float canny_thr_low,
    float canny_thr_high
) {
    allocate(pipeline, width, height, levels);
    sycl::queue& queue = pipeline.queue;

    const std::vector<Level>& pyramid = pipeline.levels;
    queue.memcpy(pyramid[0].source, image, std::size_t(width) * height);
    for (std::size_t i = 1; i < pyramid.size(); ++i) {
        downsample(queue, pyramid[i - 1], pyramid[i]);
    }
    // cvRecodedCanny scales the thresholds by 256 to compare with the magnitude
    const float low = canny_thr_low * 256;
    const float high = canny_thr_high * 256;
    for (const Level& level : pyramid) {
        gradient(queue, level, pipeline.intermediate);
        suppress(queue, level, low, high);
    }
}

void download_source(Pipeline& pipeline, std::uint32_t level, std::uint8_t* dst) {
    const Level& src = pipeline.levels[level];
    pipeline.queue.memcpy(dst, src.source, std::size_t(src.width) * src.height).wait();
}

void download_gradient(
    Pipeline& pipeline,
    std::uint32_t level,
    std::int16_t* dx,
    std::int16_t* dy
) {
    const Level& src = pipeline.levels[level];
    const std::size_t size = std::size_t(src.width) * src.height;
    pipeline.queue.memcpy(dx, src.dx, size * sizeof(std::int16_t));
    pipeline.queue.memcpy(dy, src.dy, size * sizeof(std::int16_t)).wait();
}

void download_map(Pipeline& pipeline, std::uint32_t level, Mark* dst) {
    const Level& src = pipeline.levels[level];
    pipeline.queue.memcpy(dst, src.map, std::size_t(src.width) * src.height).wait();
}

} // namespace portable
} // namespace cctag
