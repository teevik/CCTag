/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_KERNELS_PLANE_HPP
#define CCTAG_PORTABLE_KERNELS_PLANE_HPP

#include <cstddef>
#include <cstdint>

namespace cctag::portable::kernels {

/// A two-dimensional array of one element type: the shape of every image-sized stage buffer.
/// `stride` is in elements, so a pitched device plane fits; the probe's `stride_bytes` is
/// `stride * sizeof(T)`.
template <class T>
struct Plane {
    T* data = nullptr;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::size_t stride = 0;

    T* row(std::uint32_t y) const {
        return data + static_cast<std::size_t>(y) * stride;
    }
};

} // namespace cctag::portable::kernels

#endif
