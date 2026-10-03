/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PROBE_HPP
#define CCTAG_PROBE_HPP

#include <cstddef>
#include <cstdint>

namespace cctag {

/// One image of a pyramid level
struct Plane
{
    std::uint32_t width;
    std::uint32_t height;
    std::size_t stride_bytes;
    /// uint8_t for source and edge images, int16_t for gradients
    const void* data;
};

/// Observes the intermediate results of a detection
class Probe
{
  public:
    virtual ~Probe() = default;

    virtual void pyramid(std::uint32_t level, const Plane& source) {}
    virtual void gradient(std::uint32_t level, const Plane& dx, const Plane& dy) {}
    virtual void edges(std::uint32_t level, const Plane& edges) {}
};

} // namespace cctag

#endif
