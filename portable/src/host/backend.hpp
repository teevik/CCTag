/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_HOST_BACKEND_HPP
#define CCTAG_PORTABLE_HOST_BACKEND_HPP

#include "host/context.hpp"
#include "host/views.hpp"
#include "kernels/plane.hpp"

#include <concepts>
#include <cstdint>

namespace cctag::portable {

/// Interface for an CCTag execution backend.
template <class B>
concept ExecutionBackend = requires(
    typename B::Buffers& level,
    const typename B::Buffers& finer,
    Context<B>& context,
    kernels::Plane<const std::uint8_t> input,
    std::uint32_t w,
    std::uint32_t h
) {
    // Sizes one level's buffers; a no-op when the size is unchanged.
    { level.ensure(w, h) };
    // Copies the input image into level 0: the only place the pipeline reads caller memory.
    { B::load(level, input) };
    // `src` of `level` from `src` of the next finer level.
    { B::pyramid(level, finer) };
    // `src` -> `dx`, `dy`.
    { B::gradient(level) };
    // Host views of the stages' outputs; `pyramid` and `load` share one.
    { B::host_pyramid(level) } -> std::same_as<PyramidHost>;
    { B::host_gradient(level) } -> std::same_as<GradientHost>;
    // Blocks until every stage call so far has completed; a no-op on the CPU.
    { B::wait(context) };
};

} // namespace cctag::portable

#endif
