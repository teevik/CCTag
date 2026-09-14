/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_BACKENDS_CPU_BACKEND_HPP
#define CCTAG_PORTABLE_BACKENDS_CPU_BACKEND_HPP

#include "host/backend.hpp"
#include "host/views.hpp"
#include "kernels/plane.hpp"

#include <opencv2/core.hpp>

#include <cstdint>

namespace cctag::portable::cpu {

/// Holds one pyramid level's buffers
struct Buffers {
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    /// Source grayscale image
    cv::Mat1b src;
    /// Horizontal gradients
    cv::Mat1s dx;
    /// Vertical gradients
    cv::Mat1s dy;

    Buffers() = default;
    Buffers(Buffers&&) noexcept = default;
    Buffers& operator=(Buffers&&) noexcept = default;
    Buffers(const Buffers&) = delete;
    Buffers& operator=(const Buffers&) = delete;

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

/// The CPU execution backend, implementing ExecutionBackend
struct Backend {
    using Buffers = cpu::Buffers;

    /// Copies the input grayscale image into `level0.src`
    static void load(Buffers& level0, kernels::Plane<const std::uint8_t> input);
    /// Downsamples `finer.src` into `coarser.src` to build the next pyramid level
    static void pyramid(Buffers& coarser, const Buffers& finer);
    /// Computes horizontal (`dx`) and vertical (`dy`) gradients from `src`
    static void gradient(Buffers& level);

    static PyramidHost host_pyramid(Buffers& level);
    static GradientHost host_gradient(Buffers& level);

    static void wait(Context<Backend>&) {}
};

static_assert(ExecutionBackend<Backend>);

} // namespace cctag::portable::cpu

#endif
