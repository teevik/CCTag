/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_HOST_BACKEND_HPP
#define CCTAG_PORTABLE_HOST_BACKEND_HPP

#include "host/views.hpp"
#include "kernels/plane.hpp"

#include <cctag/Params.hpp>

#include <concepts>
#include <cstdint>

namespace cctag::portable {

template <class Backend>
struct Context;

/// Interface for a CCTag execution backend
template <class B>
concept ExecutionBackend = requires(
    typename B::Buffers& level,
    const typename B::Buffers& finer,
    Context<B>& context,
    const Parameters& params,
    kernels::Plane<const std::uint8_t> input,
    std::uint32_t w,
    std::uint32_t h
) {
    // Sizes this level's buffers to `w` by `h`, reusing storage if unchanged
    { level.ensure(w, h) };
    // Copies the input grayscale image into `level.src` at pyramid level 0
    { B::load(level, input) };
    // Downsamples `finer.src` into `level.src` to build the next pyramid level
    { B::pyramid(level, finer) };
    // Computes horizontal (`dx`) and vertical (`dy`) gradients from `src`
    { B::gradient(level) };
    // Finds and thins edges from `dx` and `dy`
    { B::edges(level, params) };
    // Returns a read-only host view of `src`, produced by `load` or `pyramid`
    { B::host_pyramid(level) } -> std::same_as<PyramidHost>;
    // Returns read-only host views of `dx` and `dy`, produced by `gradient`
    { B::host_gradient(level) } -> std::same_as<GradientHost>;
    // Returns a read-only host view of the thinned edges
    { B::host_edges(level) } -> std::same_as<EdgesHost>;
    // Waits for all previously submitted stages to finish
    { B::wait(context) };
};

} // namespace cctag::portable

#endif
