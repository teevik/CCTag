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

/// A non-owning view of a two-dimensional array
template <class T>
struct Plane {
    T* data = nullptr;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    /// Distance between row starts, in elements
    std::size_t stride = 0;

    T* row(std::uint32_t y) const {
        return data + static_cast<std::size_t>(y) * stride;
    }

    /// Returns a read-only view of the same plane
    Plane<const T> as_const() const {
        return {data, width, height, stride};
    }
};

} // namespace cctag::portable::kernels

#endif
