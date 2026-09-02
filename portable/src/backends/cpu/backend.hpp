/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_BACKENDS_CPU_BACKEND_HPP
#define CCTAG_PORTABLE_BACKENDS_CPU_BACKEND_HPP

#include "host/views.hpp"
#include "kernels/plane.hpp"

#include <cstdint>
#include <vector>

namespace cctag::portable::cpu {

/// The stage buffers of one pyramid level, owned by the CPU backend: dense planes plus the
/// resize coefficient tables this level needs when its dimensions are not an exact halving of the
/// finer level's. Sized once by `ensure`; every frame rewrites the planes in full.
struct Buffers {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> src;
    std::vector<std::int16_t> dx;
    std::vector<std::int16_t> dy;

    // `pyramid` glue: which resize path this level takes from the finer level, and the generic
    // path's coefficient tables (empty on the fast path).
    bool exact_halving = false;
    std::uint32_t finer_height = 0;
    std::vector<std::int32_t> xofs;
    std::vector<std::int16_t> ialpha;
    std::uint32_t xmax = 0;
    std::vector<std::int32_t> yofs;
    std::vector<std::int16_t> ibeta;
    std::uint32_t ymax = 0;

    void ensure(
        std::uint32_t level_width,
        std::uint32_t level_height,
        std::uint32_t from_width,
        std::uint32_t from_height
    );

    kernels::Plane<std::uint8_t> src_plane() {
        return {src.data(), width, height, width};
    }
    kernels::Plane<std::int16_t> dx_plane() {
        return {dx.data(), width, height, width};
    }
    kernels::Plane<std::int16_t> dy_plane() {
        return {dy.data(), width, height, width};
    }
};

/// The CPU execution backend: the stage baseline. One static stage function per snapshot stage
/// (two so far), each a visible OpenMP loop over an element function in `kernels/`.
struct Backend {
    using Buffers = cpu::Buffers;

    /// The level-0 copy: the one place the pipeline touches foreign memory.
    static void load(Buffers& level0, kernels::Plane<const std::uint8_t> input);
    /// `src` of `coarser` from `src` of `finer`; no host view.
    static void pyramid(Buffers& coarser, const Buffers& finer);
    /// `src` -> `dx`, `dy`.
    static void gradient(Buffers& level);
    /// Zero-copy on the CPU.
    static HostViews host(const Buffers& level);
};

} // namespace cctag::portable::cpu

#endif
