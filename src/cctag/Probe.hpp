/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PROBE_HPP
#define CCTAG_PROBE_HPP

#include <cstddef>
#include <cstdint>

namespace cctag {

struct Plane
{
    std::uint32_t width;
    std::uint32_t height;
    std::size_t stride_bytes;
    const void* data; // U8 for source/edges, I16 for dx/dy.
};

struct EdgePointsView
{
    std::uint32_t n;        // Number of edge points.
    const std::int32_t* xy; // [n, 2], pipeline enumeration order.
    const float* grad;      // [n, 2], (dx, dy).
};

struct VoteView
{
    const std::int32_t* links;          // [n, 2], -1 when absent.
    const std::int32_t* voters_offsets; // [n + 1].
    const std::int32_t* voters_values;
    const std::int32_t* is_max; // [n], received-vote count.
    const float* flow_length;   // [n].
    std::uint32_t n_seeds;
    const std::int32_t* seeds; // [n_seeds], ownership-resolution order.
};

struct LinkingView
{
    std::uint32_t c;                     // Number of candidate-marker segments.
    const std::int32_t* seeds;           // [c].
    const std::int32_t* segment_offsets; // [c + 1].
    const std::int32_t* segment_values;  // Segment walk order.
    const std::int32_t* child_counts;    // [c].
    const float* avg_vote;               // [c].
};

struct CandidatesView
{
    std::uint32_t n;           // Number of candidate markers.
    const float* ellipse;      // [n, 5], (cx, cy, a, b, angle), level-0 space.
    const std::int32_t* level; // [n].
    const float* quality;      // [n].
};

struct MarkersView
{
    std::uint32_t n;            // Number of detection candidates.
    const float* xy;            // [n, 2].
    const std::int32_t* id;     // [n], -1 when undefined.
    const std::int32_t* status; // [n].
};

/**
 * Pipeline-neutral observation point for detection stages.
 *
 * All view pointers are valid only for the duration of their callback and
 * must be copied by an observing probe. Indices are pipeline-local positions
 * in the xy array supplied to edge_points for the same pyramid level. They
 * are deliberately not canonical indices: an observing probe canonicalises
 * them while copying the transient views.
 *
 * Callbacks are made only from sequential code and at most once for each
 * (stage, level) in one detection. All pyramid callbacks precede the processed
 * level callbacks. Within a processed level, edge_points precedes vote, which
 * precedes linking. candidates (candidate markers) follows the level loop, and
 * markers (detection candidates) follows the final marker sort. A stage is
 * present exactly when its callback was made, so a pipeline may stop at any
 * stage boundary. The legacy CUDA path does not invoke a probe in v1.
 */
class Probe
{
  public:
    virtual ~Probe() = default;

    virtual void pyramid(std::uint32_t, const Plane&) {}
    virtual void gradient(std::uint32_t, const Plane&, const Plane&) {}
    virtual void edges(std::uint32_t, const Plane&) {}
    virtual void edge_points(std::uint32_t, const EdgePointsView&) {}
    virtual void vote(std::uint32_t, const VoteView&) {}
    virtual void linking(std::uint32_t, const LinkingView&) {}
    virtual void candidates(const CandidatesView&) {}
    virtual void markers(const MarkersView&) {}

    // Timing is recorded in memory by interested probes, never in snapshots.
    virtual void enter(const char*) {}
    virtual void leave(const char*) {}
};

} // namespace cctag

#endif
