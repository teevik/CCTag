/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_HOST_VIEWS_HPP
#define CCTAG_PORTABLE_HOST_VIEWS_HPP

#include "kernels/plane.hpp"

#include <cctag/Probe.hpp>

#include <cstdint>
#include <span>

namespace cctag::portable {

/// Read-only host view of one pyramid level's source image
struct PyramidHost {
    kernels::Plane<const std::uint8_t> src;
};

/// Read-only host views of one pyramid level's horizontal and vertical gradients
struct GradientHost {
    kernels::Plane<const std::int16_t> dx;
    kernels::Plane<const std::int16_t> dy;
};

/// Read-only host view of one pyramid level's thinned edges
struct EdgesHost {
    kernels::Plane<const std::uint8_t> edges;
};

/// Read-only host view of one pyramid level's edge-point collection in canonical order
struct EdgePointsHost {
    std::uint32_t n;
    std::span<const std::int32_t> xy;
    std::span<const float> gradients;
};

/// Read-only host view of one pyramid level's vote graph
struct VoteHost {
    std::span<const std::int32_t> links;
    std::span<const std::int32_t> voters_offsets;
    std::span<const std::int32_t> voters_values;
    std::span<const std::int32_t> is_max;
    std::span<const float> flow_length;
    std::span<const std::int32_t> seeds;
    std::span<const std::int32_t> seed_order;
};

/// Read-only host view of one pyramid level's segments in ascending seed order
struct LinkingHost {
    std::uint32_t c;
    std::span<const std::int32_t> seeds;
    std::span<const std::int32_t> segment_offsets;
    std::span<const std::int32_t> segment_values;
    std::span<const std::int32_t> child_counts;
    std::span<const float> avg_vote;
};

/// All per-level stage outputs for observation after linking, before candidates
/// Borrows execution-backend storage; consume before mutating or reusing that level
struct SnapshotViews {
    PyramidHost pyramid;
    GradientHost gradient;
    EdgesHost edges;
    EdgePointsHost edge_points;
    VoteHost vote;
    LinkingHost linking;
};

/// Read-only host view of the raw candidate markers across all pyramid levels
struct CandidatesHost {
    std::uint32_t n;
    std::span<const float> ellipses;
    std::span<const std::int32_t> levels;
    std::span<const float> quality;
};

/// Read-only host view of the final detection candidates
struct MarkersHost {
    std::uint32_t n;
    std::span<const float> xy;
    std::span<const std::int32_t> ids;
    std::span<const std::int32_t> statuses;
};

/// Converts a plane to the probe's format, with the row stride in bytes
template <class T>
inline cctag::Plane probe_plane(kernels::Plane<const T> plane) {
    return cctag::Plane{plane.width, plane.height, plane.stride * sizeof(T), plane.data};
}

/// Reports one level's stage outputs in probe order without copying their data
inline void observe(Probe& probe, std::uint32_t level, const SnapshotViews& views) {
    probe.pyramid(level, probe_plane(views.pyramid.src));
    probe.gradient(level, probe_plane(views.gradient.dx), probe_plane(views.gradient.dy));
    probe.edges(level, probe_plane(views.edges.edges));
    const auto& points = views.edge_points;
    probe.edge_points(level, {points.n, points.xy.data(), points.gradients.data()});
    const auto& vote = views.vote;
    probe.vote(
        level,
        {vote.links.data(),
         vote.voters_offsets.data(),
         vote.voters_values.data(),
         vote.is_max.data(),
         vote.flow_length.data(),
         static_cast<std::uint32_t>(vote.seed_order.size()),
         vote.seed_order.data()}
    );
    const auto& linking = views.linking;
    probe.linking(
        level,
        {linking.c,
         linking.seeds.data(),
         linking.segment_offsets.data(),
         linking.segment_values.data(),
         linking.child_counts.data(),
         linking.avg_vote.data()}
    );
}

} // namespace cctag::portable

#endif
