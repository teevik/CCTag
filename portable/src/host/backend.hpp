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
    const typename B::Buffers& level0,
    Context<B>& context,
    const Parameters& params,
    kernels::Plane<const std::uint8_t> input,
    std::uint32_t w,
    std::uint32_t h
) {
    // Sizes this level's buffers
    { level.ensure(w, h) };
    // Copies the input grayscale image into `level.src` at pyramid level 0
    { B::load(level, input) };
    // Resamples `level0.src` into `level.src` at this level's resolution
    { B::pyramid(level, level0) };
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
    // Fits candidate markers across levels, then refits them against level-zero edges
    { B::candidates(context, params) };
    // Identifies candidate markers, deduplicates them and stably sorts them by id
    { B::markers(context, params) };
    // Materialises all per-level stage outputs for observation after linking
    // Views remain valid until the level is mutated. May wait and update staging buffers.
    { B::snapshot_views(level) } -> std::same_as<SnapshotViews>;
    // Waits for all previously submitted stages to finish
    { B::wait(context) };
};

} // namespace cctag::portable

#endif
