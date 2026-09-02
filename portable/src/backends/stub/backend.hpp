/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_BACKENDS_STUB_BACKEND_HPP
#define CCTAG_PORTABLE_BACKENDS_STUB_BACKEND_HPP

#include "backends/cpu/backend.hpp"
#include "host/views.hpp"
#include "kernels/plane.hpp"

#include <cstdint>
#include <vector>

namespace cctag::portable::stub {

/// A stand-in for a device backend: its stage buffers are pitched planes in "device" memory that
/// the host sequence never sees, plus a CPU staging copy that a host view materialises into.
/// Delegating a stage to the baseline is a download, the baseline call, and an upload — the
/// generic host-view path, made visible.
struct Buffers {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::size_t pitch = 0;
    std::vector<std::uint8_t> device_src;
    std::vector<std::int16_t> device_dx;
    std::vector<std::int16_t> device_dy;
    mutable cpu::Buffers staging;

    void ensure(
        std::uint32_t level_width,
        std::uint32_t level_height,
        std::uint32_t from_width,
        std::uint32_t from_height
    );
};

/// The stub execution backend. `load` and `pyramid` delegate to the stage baseline through host
/// views; `gradient` is specialised the long way round: its own loop, in its own order, over the
/// same element function, straight into device memory.
struct Backend {
    using Buffers = stub::Buffers;

    static void load(Buffers& level0, kernels::Plane<const std::uint8_t> input);
    static void pyramid(Buffers& coarser, const Buffers& finer);
    static void gradient(Buffers& level);
    static HostViews host(const Buffers& level);
};

} // namespace cctag::portable::stub

#endif
