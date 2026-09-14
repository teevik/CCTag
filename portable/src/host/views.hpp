/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_HOST_VIEWS_HPP
#define CCTAG_PORTABLE_HOST_VIEWS_HPP

#include "kernels/plane.hpp"

#include <cctag/Probe.hpp>

#include <cstdint>
#include <span>

namespace cctag::portable {

/// Read-only host view of one pyramid level's source image
struct PyramidHost {
    kernels::Plane<const std::uint8_t> src;
};

/// Read-only host views of one pyramid level's horizontal and vertical gradients
struct GradientHost {
    kernels::Plane<const std::int16_t> dx;
    kernels::Plane<const std::int16_t> dy;
};

/// Read-only host view of one pyramid level's thinned edges
struct EdgesHost {
    kernels::Plane<const std::uint8_t> edges;
};

/// Read-only host view of one pyramid level's edge-point collection in canonical order
struct EdgePointsHost {
    std::uint32_t n;
    std::span<const std::int32_t> xy;
    std::span<const float> gradients;
};

/// Read-only host view of one pyramid level's vote graph
struct VoteHost {
    std::span<const std::int32_t> links;
    std::span<const std::int32_t> voters_offsets;
    std::span<const std::int32_t> voters_values;
    std::span<const std::int32_t> is_max;
    std::span<const float> flow_length;
    std::span<const std::int32_t> seeds;
    std::span<const std::int32_t> seed_order;
};

/// Converts a plane to the probe's format, with the row stride in bytes
template <class T>
inline cctag::Plane probe_plane(kernels::Plane<const T> plane) {
    return cctag::Plane{plane.width, plane.height, plane.stride * sizeof(T), plane.data};
}

} // namespace cctag::portable

#endif
