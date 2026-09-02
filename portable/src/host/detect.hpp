/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_HOST_DETECT_HPP
#define CCTAG_PORTABLE_HOST_DETECT_HPP

#include "host/context.hpp"
#include "host/views.hpp"
#include "kernels/plane.hpp"

#include <cctag/Params.hpp>
#include <cctag/Probe.hpp>

#include <cstdint>

namespace cctag::portable {

/// Brackets a stage with the probe's timing callbacks. `leave` is where a device backend would
/// have to force completion (the per-stage timing ticket's constraint); on the CPU every stage
/// call is synchronous.
class StageTiming {
public:
    StageTiming(Probe* probe, const char* stage) : probe_(probe), stage_(stage) {
        if (probe_) {
            probe_->enter(stage_);
        }
    }
    ~StageTiming() {
        if (probe_) {
            probe_->leave(stage_);
        }
    }
    StageTiming(const StageTiming&) = delete;
    StageTiming& operator=(const StageTiming&) = delete;

private:
    Probe* probe_;
    const char* stage_;
};

/// The host sequence (ADR 0003): the only place an execution backend is named. Phase A calls every
/// level's stage functions without requesting a host view; phase B materialises host views in
/// `Probe.hpp`'s callback order. The prototype stops after `gradient`; each further stage of the
/// build adds one call per phase here.
template <class Backend>
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
        StageTiming timing(probe, "pyramid");
        Backend::load(levels[0], input);
        for (std::uint32_t level = 1; level < count; ++level) {
            Backend::pyramid(levels[level], levels[level - 1]);
        }
    }
    {
        StageTiming timing(probe, "gradient");
        for (std::uint32_t level = 0; level < count; ++level) {
            Backend::gradient(levels[level]);
        }
    }

    // Phase B: observe. Unobserved probes cost nothing; a host view is the only synchronisation.
    if (probe) {
        for (std::uint32_t level = 0; level < count; ++level) {
            const HostViews views = Backend::host(levels[level]);
            probe->pyramid(level, probe_plane(views.src));
            probe->gradient(level, probe_plane(views.dx), probe_plane(views.dy));
        }
    }
}

} // namespace cctag::portable

#endif
