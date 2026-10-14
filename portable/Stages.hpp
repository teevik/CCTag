/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_STAGES_HPP
#define CCTAG_PORTABLE_STAGES_HPP

#include "portable/Pipeline.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace cctag {
namespace portable {

/// A pixel of the edge map: weak or strong until hysteresis, then edge or none
enum class Mark : std::uint8_t {
    none = 0,
    weak = 1,
    strong = 2,
    edge = 255
};

/// A device-side view of one tightly packed image plane, cheap to capture into a kernel
template <typename T>
struct Plane {
    T* data;
    std::uint32_t width;
    std::uint32_t height;

    /// The same plane, read only
    operator Plane<const T>() const {
        return {data, width, height};
    }

    /// The pixel in row y, column x
    T& at(int y, int x) const {
        return data[std::size_t(y) * width + x];
    }

    /// The pixel at index i in scan order
    T& operator[](std::size_t i) const {
        return data[i];
    }

    /// The number of pixels
    std::size_t size() const {
        return std::size_t(width) * height;
    }
};

/// The device images of one pyramid level, tightly packed
struct Level {
    std::uint32_t width;
    std::uint32_t height;
    std::uint8_t* source;
    std::int16_t* dx;
    std::int16_t* dy;
    Mark* map;
    /// The edge points in scan order, room for capacity of them
    Edgel* edgels;
    std::uint32_t capacity;
    /// The number of edge pixels, which can exceed capacity
    std::uint32_t* edgel_count;

    Plane<std::uint8_t> source_plane() const {
        return {source, width, height};
    }
    Plane<std::int16_t> dx_plane() const {
        return {dx, width, height};
    }
    Plane<std::int16_t> dy_plane() const {
        return {dy, width, height};
    }
    Plane<Mark> map_plane() const {
        return {map, width, height};
    }
};

/// The device buffers of the center search
struct CenterBuffers {
    std::uint32_t points;
    std::uint32_t cuts;
    std::uint32_t samples;
    Homography* homographies;
    Cut* cut;
    /// The rectified signals, [point][cut][sample]
    float* signals;
    /// Whether a cut leaves the image, [point][cut]
    std::uint8_t* out_of_bounds;
    /// The residual of every point
    float* residuals;
};

/// 01: Halves an image by averaging every 2x2 block
void downsample(sycl::queue& queue, const Level& src, const Level& dst);

/// 02: Computes dx and dy of a level
void gradient(sycl::queue& queue, const Level& level, float* scratch);

/// 03: Canny non-maximum suppression and double thresholding, marks local maxima strong above high
/// and weak above low
void suppress(sycl::queue& queue, const Level& level, float low, float high);

/// 04: Canny hysteresis, makes every weak pixel connected to a strong one strong
void hysteresis(sycl::queue& queue, const Level& level);

/// 05: Thins the strong pixels of the map into edge pixels
void thin(sycl::queue& queue, const Level& level, Mark* scratch);

/// 06: Lists the edge pixels as edge points and links each to the first edge points its gradient
/// descent reaches within dist_search steps both ways. index is scratch for one int per pixel, rows
/// for one per row.
void link(
    sycl::queue& queue,
    const Level& level,
    std::uint32_t dist_search,
    std::int32_t* index,
    std::uint32_t* rows
);

/// 07: Every edge point follows its links across n_crowns crowns and votes for the edge point it
/// ends on
void vote(sycl::queue& queue, const Level& level, std::uint32_t n_crowns, float ratio_voting);

/// 08: Computes the residual of costFunctionGlob for every homography, or -1 where no pair of cuts
/// lies in the image
void center_costs(
    sycl::queue& queue,
    const Level& level,
    const CenterBuffers& buffers,
    std::uint32_t points,
    std::uint32_t cuts,
    std::uint32_t samples
);

} // namespace portable
} // namespace cctag

#endif
