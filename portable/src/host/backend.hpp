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

/// The execution-backend contract (ADR 0001): a struct of static stage functions, one per
/// snapshot stage, over an opaque per-level `Buffers` type the backend owns between stages; one
/// `host_<stage>(Buffers&)` per per-level stage that materialises that stage's host view
/// (non-const: materialising mutates staging on a device); and `wait(Context&)`, which forces
/// completion of everything enqueued and is called by the host sequence before a probe's
/// `leave`. Selection is at compile time: the host sequence is templated on the backend and
/// `ICCTag.cpp` instantiates the one `CCTAG_PORTABLE_BACKEND` selects; nothing else names one.
///
/// Every backend implements every stage. A stage it does not specialise is delegated to the stage
/// baseline (`cpu::<stage>`) through staging buffers of the baseline's type, as a visible line in
/// the backend's own source: there is no override trait, no runtime "unavailable" status, no
/// virtual dispatch. Stage functions run on the host, are synchronous on the CPU and may be
/// asynchronous elsewhere (only a host view or `wait` synchronises); they may throw, element
/// functions never do.
///
/// Each stage the build lands adds its stage function and its host view below; the requirements
/// are the stages that exist so far.
template <class B>
concept ExecutionBackend = requires(
    typename B::Buffers& level,
    const typename B::Buffers& finer,
    Context<B>& context,
    kernels::Plane<const std::uint8_t> input,
    std::uint32_t w,
    std::uint32_t h
) {
    // Sizes one level.
    { level.ensure(w, h) };
    // The level-0 copy: the one touch of foreign memory.
    { B::load(level, input) };
    // `src` of `level` from `src` of `finer`; no host view.
    { B::pyramid(level, finer) };
    // `src` -> `dx`, `dy`.
    { B::gradient(level) };
    { B::host_pyramid(level) } -> std::same_as<PyramidHost>;
    { B::host_gradient(level) } -> std::same_as<GradientHost>;
    // Force completion; a no-op on the CPU.
    { B::wait(context) };
};

} // namespace cctag::portable

#endif
