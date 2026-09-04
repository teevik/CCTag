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

// Pixel data for one pyramid level. Rows may include padding.
struct Plane
{
    std::uint32_t width;
    std::uint32_t height;
    std::size_t stride_bytes;
    // uint8_t for source/edge images and int16_t for dx/dy gradients.
    const void* data;
};

struct EdgePointsView
{
    // Number of edge points.
    std::uint32_t n;
    // n (x, y) pairs, in the pipeline's edge-point order.
    const std::int32_t* xy;
    // n (dx, dy) pairs, in the same order as xy.
    const float* grad;
};

// Voting results for the n points reported by edge_points at this pyramid level.
struct VoteView
{
    const std::int32_t* links; // n pairs of (before, after) point indices; -1 means no link.

    // Voters for point i are stored in voters_values, from voters_offsets[i]
    // up to (but excluding) voters_offsets[i + 1]. There are n + 1 offsets.
    const std::int32_t* voters_offsets;
    const std::int32_t* voters_values;

    const std::int32_t* is_max; // Number of votes received by each point.
    const float* flow_length;   // One flow length per point.
    std::uint32_t n_seeds;
    const std::int32_t* seeds; // n_seeds point indices, in the order linking processes them.
};

struct LinkingView
{
    std::uint32_t c;           // Number of candidate-marker segments.
    const std::int32_t* seeds; // One seed point index per segment.

    // Points in segment i are stored in segment_values, from segment_offsets[i]
    // up to (but excluding) segment_offsets[i + 1], in the order of the segment walk.
    const std::int32_t* segment_offsets; // c + 1 offsets.
    const std::int32_t* segment_values;
    const std::int32_t* child_counts; // Number of child edge points per segment.
    const float* avg_vote;            // Voting score for each segment.
};

struct CandidatesView
{
    std::uint32_t n;           // Number of candidate markers.
    const float* ellipse;      // n (cx, cy, a, b, angle) groups, scaled to the original image.
    const std::int32_t* level; // Pyramid level where each candidate marker was found.
    const float* quality;      // One quality value per candidate marker.
};

struct MarkersView
{
    std::uint32_t n;            // Number of detection candidates.
    const float* xy;            // n (x, y) pairs.
    const std::int32_t* id;     // One marker ID per detection candidate; -1 if unidentified.
    const std::int32_t* status; // One identification status per detection candidate.
};

/**
 * Receives stage results and timing events from pipeline, must be overridden to
 * actually observe the data, noop by default.
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

    // enter/leave mark the start/end of a named stage's work
    virtual void enter(const char*) {}
    virtual void leave(const char*) {}
};

} // namespace cctag

#endif
