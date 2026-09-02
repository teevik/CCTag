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

namespace cctag::portable {

/// The host views of one level's stage buffers, materialised by a backend on request. Typed
/// planes are the internal vocabulary; `Probe.hpp`'s `const void*` planes are built from them by
/// `probe_plane` at the probe boundary only.
struct HostViews {
    kernels::Plane<const std::uint8_t> src;
    kernels::Plane<const std::int16_t> dx;
    kernels::Plane<const std::int16_t> dy;
};

template <class T>
inline cctag::Plane probe_plane(kernels::Plane<const T> plane) {
    return cctag::Plane{plane.width, plane.height, plane.stride * sizeof(T), plane.data};
}

} // namespace cctag::portable

#endif
