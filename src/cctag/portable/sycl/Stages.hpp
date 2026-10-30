/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_SYCL_STAGES_HPP
#define CCTAG_PORTABLE_SYCL_STAGES_HPP

#include <cctag/portable/Pipeline.hpp>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace cctag {
namespace portable {

/// A view of image with given width and height in contiguous buffer
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

/// The images of one pyramid level
struct Level {
    std::uint32_t width;
    std::uint32_t height;
    std::uint8_t* source;
    std::int16_t* dx;
    std::int16_t* dy;
    Mark* map;

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

/// 01: Halves an image by averaging every 2x2 block
void downsample(sycl::queue& queue, const Level& src, const Level& dst);

/// 02: Computes dx and dy of a level
void gradient(sycl::queue& queue, const Level& level, float* scratch);

/// 03: Canny non-maximum suppression and double thresholding
void suppress(sycl::queue& queue, const Level& level, float low, float high);

} // namespace portable
} // namespace cctag

#endif
