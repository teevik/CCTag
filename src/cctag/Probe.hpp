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

/// Pixel data for one pyramid level
struct Plane
{
    std::uint32_t width;
    std::uint32_t height;
    std::size_t stride_bytes;
    /// uint8_t for source/edge images and int16_t for dx/dy gradients
    const void* data;
};

struct EdgePointsView
{
    std::uint32_t n;
    /// `n` (x, y) pairs
    const std::int32_t* xy;
    /// n (dx, dy) pairs
    const float* gradients;
};

/// Voting results for the `n` points reported by `edge_points` at this pyramid level
struct VoteView
{
    /// `n` pairs of point indices, -1 means no link
    const std::int32_t* links;
    const std::int32_t* voters_offsets;
    const std::int32_t* voters_values;

    /// Number of votes received by each point
    const std::int32_t* is_max;
    const float* flow_length;
    std::uint32_t n_seeds;
    const std::int32_t* seeds;
};

struct LinkingView
{
    /// Number of candidate-marker segments
    std::uint32_t c;
    /// One seed point index per segment
    const std::int32_t* seeds;

    /// `c + 1` offsets delimiting segments in `segment_values`
    const std::int32_t* segment_offsets;
    /// Point indices in segment-walk order
    const std::int32_t* segment_values;
    /// Number of child edge points per segment
    const std::int32_t* child_counts;
    /// Voting score for each segment
    const float* avg_vote;
};

struct CandidatesView
{
    std::uint32_t n;
    /// `n` (cx, cy, a, b, angle) groups, scaled to the original image
    const float* ellipse;
    /// Pyramid level where each candidate marker was found
    const std::int32_t* level;
    const float* quality;
};

/// Detection candidates and their identification results
struct MarkersView
{
    std::uint32_t n;
    /// `n` (x, y) pairs
    const float* xy;
    /// Marker ID, -1 if unidentified
    const std::int32_t* id;
    const std::int32_t* status;
};

/// Receives pipeline stage results and timing events
/// Override callbacks to observe data; default implementations do nothing
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

    /// `enter`/`leave` mark the start/end of a named stage's work
    virtual void enter(const char*) {}
    virtual void leave(const char*) {}
};

} // namespace cctag

#endif
