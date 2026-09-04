/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_HOST_DETECT_HPP
#define CCTAG_PORTABLE_HOST_DETECT_HPP

#include "host/backend.hpp"
#include "host/context.hpp"
#include "host/views.hpp"
#include "kernels/plane.hpp"

#include <cctag/Params.hpp>
#include <cctag/Probe.hpp>

#include <cstdint>

namespace cctag::portable {

/// Brackets one stage's span with the probe's timing callbacks. With a probe attached the
/// destructor forces completion (`Backend::wait`) before `leave`, so the span covers the stage's
/// work on a device backend too; on the CPU every stage call is synchronous and `wait` is a
/// no-op. Without a probe nothing is called: timing mode serialises phase A and is not the
/// overlapped schedule (ADR 0003, ADR 0004-timing).
template <ExecutionBackend Backend>
class StageTiming {
  public:
    StageTiming(Context<Backend>& context, Probe* probe, const char* stage) :
        context_(context),
        probe_(probe),
        stage_(stage) {
        if (probe_) probe_->enter(stage_);
    }
    ~StageTiming() {
        if (probe_) {
            Backend::wait(context_);
            probe_->leave(stage_);
        }
    }
    StageTiming(const StageTiming&) = delete;
    StageTiming& operator=(const StageTiming&) = delete;

  private:
    Context<Backend>& context_;
    Probe* probe_;
    const char* stage_;
};

/// The host sequence (ADR 0003): the only place an execution backend is named. Phase A calls
/// every level's stage functions without requesting a single host view; phase B materialises
/// host views in `Probe.hpp`'s callback order. The build stops after `gradient` so far; each
/// further stage adds one call per phase here.
template <ExecutionBackend Backend>
void detect(
    Context<Backend>& context,
    kernels::Plane<const std::uint8_t> input,
    const Parameters& params,
    Probe* probe
) {
    context.ensure(input.width, input.height, params);
    auto& levels = context.levels;
    const std::uint32_t count = static_cast<std::uint32_t>(levels.size());

    // Phase A: enqueue. The pyramid is chained level by level and stays backend-resident.
    {
        StageTiming<Backend> timing(context, probe, "pyramid");
        Backend::load(levels[0], input);
        for (std::uint32_t level = 1; level < count; ++level)
            Backend::pyramid(levels[level], levels[level - 1]);
    }
    {
        StageTiming<Backend> timing(context, probe, "gradient");
        for (std::uint32_t level = 0; level < count; ++level)
            Backend::gradient(levels[level]);
    }

    // Phase B: observe. Unobserved probes cost nothing; a host view is the only synchronisation.
    // The per-level image callbacks (pyramid, gradient) precede every processed-level callback
    // that later stages add, as `Probe.hpp` requires.
    if (probe) {
        for (std::uint32_t level = 0; level < count; ++level) {
            const PyramidHost pyramid = Backend::host_pyramid(levels[level]);
            probe->pyramid(level, probe_plane(pyramid.src));
            const GradientHost gradient = Backend::host_gradient(levels[level]);
            probe->gradient(level, probe_plane(gradient.dx), probe_plane(gradient.dy));
        }
    }
}

} // namespace cctag::portable

#endif
