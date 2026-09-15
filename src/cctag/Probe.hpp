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
    std::uint32_t point_count;
    /// `point_count` (x, y) pairs
    const std::int32_t* positions_xy;
    /// `point_count` (dx, dy) pairs
    const float* gradients;
};

/// Voting results for the points reported by `edge_points` at this pyramid level
struct VoteView
{
    /// (before, after) point indices for each edge point, -1 means no link
    const std::int32_t* linked_point_indices;
    const std::int32_t* voter_offsets;
    const std::int32_t* voter_point_indices;

    /// Vote count for each edge point, -1 if it did not qualify as a seed
    const std::int32_t* seed_vote_counts;
    const float* mean_flow_lengths;
    std::uint32_t seed_count;
    const std::int32_t* seed_point_indices;
};

struct LinkingView
{
    /// Number of candidate-marker segments
    std::uint32_t segment_count;
    /// One seed point index per segment
    const std::int32_t* seed_point_indices;

    /// `segment_count + 1` offsets delimiting segments in `segment_point_indices`
    const std::int32_t* segment_offsets;
    /// Point indices in segment-walk order
    const std::int32_t* segment_point_indices;
    /// Number of child edge points per segment
    const std::int32_t* child_point_counts;
    /// Voting score for each segment
    const float* vote_scores;
};

struct CandidatesView
{
    std::uint32_t candidate_count;
    /// `candidate_count` (cx, cy, a, b, angle) groups, scaled to the original image
    const float* outer_ellipse_parameters;
    /// Pyramid level where each candidate marker was found
    const std::int32_t* pyramid_levels;
    const float* quality_scores;
};

/// Detection candidates and their identification results
struct MarkersView
{
    std::uint32_t candidate_count;
    /// `candidate_count` (x, y) pairs
    const float* positions_xy;
    /// Marker IDs, -1 if unidentified
    const std::int32_t* marker_ids;
    const std::int32_t* identification_statuses;
};

/// Receives pipeline stage results and timing events
/// Override callbacks to observe data, by default no-op
class Probe
{
  public:
    virtual ~Probe() = default;

    /// False for timing-only probes, so the pipeline can skip constructing stage views
    virtual bool observes_stages() const { return true; }

    virtual void pyramid(std::uint32_t pyramid_level, const Plane& source) {}
    virtual void gradient(std::uint32_t pyramid_level, const Plane& gradient_x, const Plane& gradient_y) {}
    virtual void edges(std::uint32_t pyramid_level, const Plane& edges) {}
    virtual void edge_points(std::uint32_t pyramid_level, const EdgePointsView& edge_points) {}
    virtual void vote(std::uint32_t pyramid_level, const VoteView& votes) {}
    virtual void linking(std::uint32_t pyramid_level, const LinkingView& linking) {}
    virtual void candidates(const CandidatesView& candidates) {}
    virtual void markers(const MarkersView& markers) {}

    /// `enter`/`leave` mark the start/end of a named stage's work
    virtual void enter(const char* stage_name) {}
    virtual void leave(const char* stage_name) {}
};

} // namespace cctag

#endif
