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
#include "host/context.hpp"
#include "host/views.hpp"
#include "kernels/plane.hpp"

#include <cstdint>
#include <vector>

namespace cctag::portable::stub {

/// A stand-in for a device backend: its stage buffers are pitched planes in "device" memory that
/// the host sequence never sees, plus staging buffers of the baseline's type. Delegating a stage
/// to the baseline fills staging from device memory, calls the baseline and drains the result
/// back; a host view fills staging. The cost of not specialising a stage, made visible.
struct Buffers {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::size_t pitch = 0;
    std::vector<std::uint8_t> device_src;
    std::vector<std::int16_t> device_dx;
    std::vector<std::int16_t> device_dy;
    /// `mutable` because delegating `pyramid` refreshes the finer level's staging copy through
    /// the contract's `const` finer level.
    mutable cpu::Buffers staging;

    void ensure(std::uint32_t level_width, std::uint32_t level_height);
};

/// The stub execution backend: a `CCTAG_PORTABLE_BACKEND` value and a test fixture, never a nix
/// leaf. `load` and `pyramid` delegate to the stage baseline through staging buffers; `gradient`
/// is specialised the long way round: its own loop, in its own order, over the same element
/// function, straight into device memory. Every stage the build lands is delegated here first.
struct Backend {
    using Buffers = stub::Buffers;

    static void load(Buffers& level0, kernels::Plane<const std::uint8_t> input);
    static void pyramid(Buffers& coarser, const Buffers& finer);
    static void gradient(Buffers& level);

    static PyramidHost host_pyramid(Buffers& level);
    static GradientHost host_gradient(Buffers& level);

    /// The "device" is synchronous: nothing to wait for.
    static void wait(Context<Backend>&) {}
};

} // namespace cctag::portable::stub

#endif
