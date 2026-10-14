/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "portable/Pipeline.hpp"

#include "portable/Stages.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace cctag {
namespace portable {

struct Pipeline {
    sycl::queue queue{sycl::property::queue::in_order{}};
    std::vector<Level> levels;
    /// Temporary image the size of level 0, can be reused between passes
    float* intermediate = nullptr;
    /// Temporary value per row of level 0
    std::uint32_t* rows = nullptr;
    /// The buffers of the center search, grown as needed
    CenterBuffers centers{};
    /// The number of uploaded cuts and their samples
    std::uint32_t cut_count = 0;
    std::uint32_t samples = 0;
};

namespace {

void release_centers(Pipeline& pipeline) {
    const CenterBuffers& centers = pipeline.centers;
    sycl::free(centers.homographies, pipeline.queue);
    sycl::free(centers.cut, pipeline.queue);
    sycl::free(centers.signals, pipeline.queue);
    sycl::free(centers.out_of_bounds, pipeline.queue);
    sycl::free(centers.residuals, pipeline.queue);
    pipeline.centers = {};
}

void release(Pipeline& pipeline) {
    pipeline.queue.wait();
    for (const Level& level : pipeline.levels) {
        sycl::free(level.source, pipeline.queue);
        sycl::free(level.dx, pipeline.queue);
        sycl::free(level.dy, pipeline.queue);
        sycl::free(level.map, pipeline.queue);
        sycl::free(level.edgels, pipeline.queue);
        sycl::free(level.edgel_count, pipeline.queue);
    }
    pipeline.levels.clear();
    sycl::free(pipeline.intermediate, pipeline.queue);
    pipeline.intermediate = nullptr;
    sycl::free(pipeline.rows, pipeline.queue);
    pipeline.rows = nullptr;
    release_centers(pipeline);
}

void allocate(Pipeline& pipeline, std::uint32_t width, std::uint32_t height, std::uint32_t count) {
    const std::vector<Level>& levels = pipeline.levels;
    if (levels.size() == count && count > 0 && levels[0].width == width
        && levels[0].height == height) {
        return;
    }

    release(pipeline);
    pipeline.intermediate = sycl::malloc_device<float>(std::size_t(width) * height, pipeline.queue);
    pipeline.rows = sycl::malloc_device<std::uint32_t>(height, pipeline.queue);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::size_t size = std::size_t(width) * height;
        const std::uint32_t capacity = std::min<std::size_t>(size, max_edgels);
        pipeline.levels.push_back(
            {width,
             height,
             sycl::malloc_device<std::uint8_t>(size, pipeline.queue),
             sycl::malloc_device<std::int16_t>(size, pipeline.queue),
             sycl::malloc_device<std::int16_t>(size, pipeline.queue),
             sycl::malloc_device<Mark>(size, pipeline.queue),
             sycl::malloc_device<Edgel>(capacity, pipeline.queue),
             capacity,
             sycl::malloc_device<std::uint32_t>(1, pipeline.queue)}
        );
        width /= 2;
        height /= 2;
    }
}

/// Grows the center search buffers to fit, never shrinking them
void allocate_centers(
    Pipeline& pipeline,
    std::uint32_t points,
    std::uint32_t cuts,
    std::uint32_t samples
) {
    const CenterBuffers& old = pipeline.centers;
    if (points <= old.points && cuts <= old.cuts && samples <= old.samples) {
        return;
    }

    points = std::max(points, old.points);
    cuts = std::max(cuts, old.cuts);
    samples = std::max(samples, old.samples);
    pipeline.queue.wait();
    release_centers(pipeline);
    sycl::queue& queue = pipeline.queue;
    const std::size_t signals = std::size_t(points) * cuts;
    pipeline.centers = {
        points,
        cuts,
        samples,
        sycl::malloc_device<Homography>(points, queue),
        sycl::malloc_device<Cut>(cuts, queue),
        sycl::malloc_device<float>(signals * samples, queue),
        sycl::malloc_device<std::uint8_t>(signals, queue),
        sycl::malloc_device<float>(points, queue)
    };
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
    std::uint32_t levels,
    float canny_thr_low,
    float canny_thr_high,
    std::uint32_t dist_search,
    std::size_t n_crowns,
    float ratio_voting
) {
    if (n_crowns > max_crowns) {
        throw std::invalid_argument(
            "load: too many crowns to vote (nCrowns: " + std::to_string(n_crowns)
            + ", max: " + std::to_string(max_crowns) + ")"
        );
    }
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
        hysteresis(queue, level);
        thin(queue, level, reinterpret_cast<Mark*>(pipeline.intermediate));
        link(
            queue,
            level,
            dist_search,
            reinterpret_cast<std::int32_t*>(pipeline.intermediate),
            pipeline.rows
        );
        vote(queue, level, std::uint32_t(n_crowns), ratio_voting);
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

void download_edges(Pipeline& pipeline, std::uint32_t level, std::uint8_t* dst) {
    const Level& src = pipeline.levels[level];
    pipeline.queue.memcpy(dst, src.map, std::size_t(src.width) * src.height).wait();
}

void download_edgels(Pipeline& pipeline, std::uint32_t level, std::vector<Edgel>& dst) {
    const Level& src = pipeline.levels[level];
    std::uint32_t count = 0;
    pipeline.queue.memcpy(&count, src.edgel_count, sizeof(count)).wait();
    if (count > src.capacity) {
        throw std::length_error(
            "download_edgels: too many edge points (nb points: " + std::to_string(count)
            + ", max: " + std::to_string(src.capacity) + ")"
        );
    }
    dst.resize(count);
    if (count > 0) {
        pipeline.queue.memcpy(dst.data(), src.edgels, count * sizeof(Edgel)).wait();
    }
}

void upload_cuts(
    Pipeline& pipeline,
    const std::vector<Cut>& cuts,
    std::uint32_t samples,
    std::uint32_t points
) {
    allocate_centers(pipeline, points, std::uint32_t(cuts.size()), samples);
    pipeline.cut_count = std::uint32_t(cuts.size());
    pipeline.samples = samples;
    if (!cuts.empty()) {
        pipeline.queue.memcpy(pipeline.centers.cut, cuts.data(), cuts.size() * sizeof(Cut)).wait();
    }
}

void center_costs(
    Pipeline& pipeline,
    const std::vector<Homography>& homographies,
    std::vector<float>& residuals
) {
    const CenterBuffers& centers = pipeline.centers;
    const std::size_t points = homographies.size();
    if (points > centers.points) {
        throw std::length_error(
            "center_costs: too many homographies (nb homographies: " + std::to_string(points)
            + ", max: " + std::to_string(centers.points) + ")"
        );
    }
    residuals.resize(points);
    if (points == 0) {
        return;
    }

    sycl::queue& queue = pipeline.queue;
    queue.memcpy(centers.homographies, homographies.data(), points * sizeof(Homography));
    center_costs(
        queue,
        pipeline.levels[0],
        centers,
        std::uint32_t(points),
        pipeline.cut_count,
        pipeline.samples
    );
    queue.memcpy(residuals.data(), centers.residuals, points * sizeof(float)).wait();
}

} // namespace portable
} // namespace cctag
