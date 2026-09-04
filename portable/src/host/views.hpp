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

// The host views (ADR 0001), one per per-level stage: read-only bundles of typed planes over a
// backend's stage buffers, materialised by `Backend::host_<stage>(Buffers&)` on request. Typed
// planes are the internal vocabulary; `Probe.hpp`'s `const void*` planes are built from them by
// `probe_plane` at the probe boundary only. Each stage the build lands adds its view here.

/// `src` of one level after `pyramid`.
struct PyramidHost {
    kernels::Plane<const std::uint8_t> src;
};

/// `dx`, `dy` of one level after `gradient`.
struct GradientHost {
    kernels::Plane<const std::int16_t> dx;
    kernels::Plane<const std::int16_t> dy;
};

/// The pointer-copy adapter to `Probe.hpp`'s plane; `stride_bytes` is the element stride scaled.
template <class T>
inline cctag::Plane probe_plane(kernels::Plane<const T> plane) {
    return cctag::Plane{plane.width, plane.height, plane.stride * sizeof(T), plane.data};
}

} // namespace cctag::portable

#endif
