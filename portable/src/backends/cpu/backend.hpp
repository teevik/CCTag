/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_BACKENDS_CPU_BACKEND_HPP
#define CCTAG_PORTABLE_BACKENDS_CPU_BACKEND_HPP

#include "host/context.hpp"
#include "host/views.hpp"
#include "kernels/plane.hpp"

#include <opencv2/core.hpp>

#include <cstdint>

namespace cctag::portable::cpu {

/// The stage buffers of one pyramid level, owned by the CPU backend (ADR 0002): dense OpenCV
/// planes, sized once by `ensure` and rewritten in full every frame. Kernels see them through
/// `kernels::Plane`. Each stage the build lands adds its storage here.
struct Buffers {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    cv::Mat1b src;
    cv::Mat1s dx;
    cv::Mat1s dy;

    void ensure(std::uint32_t level_width, std::uint32_t level_height);

    kernels::Plane<std::uint8_t> src_plane() {
        return {src[0], width, height, src.step1()};
    }
    kernels::Plane<std::int16_t> dx_plane() {
        return {dx[0], width, height, dx.step1()};
    }
    kernels::Plane<std::int16_t> dy_plane() {
        return {dy[0], width, height, dy.step1()};
    }
};

/// The CPU execution backend: the stage baseline (ADR 0001), one source file per stage. Where the
/// legacy pipeline called an OpenCV routine the baseline calls the same one (ADR 0006); a stage
/// is `Exact` against the reference snapshot under the same `OPENCV_CPU_DISABLE` mask the
/// reference producer uses. Host views are zero-copy and `wait` is a no-op: every stage call is
/// synchronous.
struct Backend {
    using Buffers = cpu::Buffers;

    /// The level-0 copy: the one place the pipeline touches foreign memory.
    static void load(Buffers& level0, kernels::Plane<const std::uint8_t> input);
    /// `src` of `coarser` from `src` of `finer`; no host view.
    static void pyramid(Buffers& coarser, const Buffers& finer);
    /// `src` -> `dx`, `dy`.
    static void gradient(Buffers& level);

    static PyramidHost host_pyramid(Buffers& level);
    static GradientHost host_gradient(Buffers& level);

    static void wait(Context<Backend>&) {}
};

} // namespace cctag::portable::cpu

#endif
