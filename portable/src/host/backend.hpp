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
    // Sizes this level's buffers, also retaining the input image's bounds for descent
    { level.ensure(w, h, input.width, input.height) };
    // Copies the input grayscale image into `level.src` at pyramid level 0
    { B::load(level, input) };
    // Downsamples `finer.src` into `level.src` to build the next pyramid level
    { B::pyramid(level, finer) };
    // Computes horizontal (`dx`) and vertical (`dy`) gradients from `src`
    { B::gradient(level) };
    // Finds and thins edges from `dx` and `dy`
    { B::edges(level, params) };
    // Compacts edges into the edge-point collection and rewrites the edge map
    { B::edge_points(level) };
    // Links edge points and gathers their votes into the vote graph
    { B::vote(level, params) };
    // Walks seeds, resolves ownership and gathers segments and their children
    { B::linking(level, params) };
    // Returns a read-only host view of `src`, produced by `load` or `pyramid`
    { B::host_pyramid(level) } -> std::same_as<PyramidHost>;
    // Returns read-only host views of `dx` and `dy`, produced by `gradient`
    { B::host_gradient(level) } -> std::same_as<GradientHost>;
    // Returns a read-only host view of the thinned edges
    { B::host_edges(level) } -> std::same_as<EdgesHost>;
    // Returns a read-only host view of the edge-point collection in canonical order
    { B::host_edge_points(level) } -> std::same_as<EdgePointsHost>;
    // Returns a read-only host view of the vote graph and ownership-resolution order
    { B::host_vote(level) } -> std::same_as<VoteHost>;
    // Returns a read-only host view of segments in ascending seed order
    { B::host_linking(level) } -> std::same_as<LinkingHost>;
    // Waits for all previously submitted stages to finish
    { B::wait(context) };
};

} // namespace cctag::portable

#endif
