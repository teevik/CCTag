/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_STAGES_HPP
#define CCTAG_PORTABLE_STAGES_HPP

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
    operator Plane<const T>() const { return {data, width, height}; }

    /// The pixel in row y, column x
    T& at(int y, int x) const { return data[std::size_t(y) * width + x]; }

    /// The pixel at index i in scan order
    T& operator[](std::size_t i) const { return data[i]; }

    /// The number of pixels
    std::size_t size() const { return std::size_t(width) * height; }
};

/// The device images of one pyramid level, tightly packed
struct Level {
    std::uint32_t width;
    std::uint32_t height;
    std::uint8_t* source;
    std::int16_t* dx;
    std::int16_t* dy;
    Mark* map;

    Plane<std::uint8_t> sourcePlane() const { return {source, width, height}; }
    Plane<std::int16_t> dxPlane() const { return {dx, width, height}; }
    Plane<std::int16_t> dyPlane() const { return {dy, width, height}; }
    Plane<Mark> mapPlane() const { return {map, width, height}; }
};

/// 01: halves an image by averaging every 2x2 block
void downsample(sycl::queue& queue, const Level& src, const Level& dst);

/// 02: computes dx and dy of a level
void gradient(sycl::queue& queue, const Level& level, float* scratch);

/// 03: non-maximum suppression: marks the pixels whose gradient magnitude peaks along the gradient
/// direction strong above high and weak above low
void suppress(sycl::queue& queue, const Level& level, float low, float high);

/// 04: makes every weak pixel of the map connected to a strong one strong, so strong marks exactly
/// the edges
void hysteresis(sycl::queue& queue, const Level& level);

/// 05: thins the strong pixels of the map in place into the edge image, marked edge or none
void thin(sycl::queue& queue, const Level& level, Mark* scratch);

} // namespace portable
} // namespace cctag

#endif
