/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_HOST_CANDIDATES_HPP
#define CCTAG_PORTABLE_HOST_CANDIDATES_HPP

#include "host/ellipse.hpp"
#include "kernels/pcg32.hpp"

#include <cstdint>
#include <vector>

namespace cctag::portable {

struct DirectedPoint {
    float x;
    float y;
    float dx;
    float dy;
};

/// A fitted marker with its outer points owned in level-zero coordinates
struct CandidateMarker {
    std::int32_t level = 0;
    float scale = 1;
    float quality = 0;
    Eigen::Vector2f center = Eigen::Vector2f::Zero();
    Ellipse outer_ellipse;
    Ellipse rescaled_outer_ellipse;
    std::vector<DirectedPoint> outer_points;
};

/// One candidate's loop-two result and private scratch for growing, assembly and refitting
struct CandidateSlot {
    std::int32_t seed = -1;
    std::int32_t label = -1;
    std::size_t score = 0;
    bool accepted = false;
    bool has_marker = false;
    kernels::Pcg32 random;
    Ellipse ellipse;
    std::vector<std::int32_t> filtered_children;
    std::vector<std::int32_t> outer_points;
    /// Run marks and auxiliary ring-walk marks, cleared through touched indices
    std::vector<std::uint8_t> processed;
    std::vector<std::int32_t> touched;
    /// Iterative depth-first traversal retains the legacy's recursive neighbour order
    struct Visit {
        std::int32_t point;
        int neighbour;
    };
    std::vector<Visit> stack;
    std::vector<std::int32_t> best_points;
    std::vector<std::int32_t> merged_points;
    std::vector<std::int32_t> hull_points;
    std::vector<Eigen::Vector2f> fit_points;
    std::vector<float> distances;
    EllipseFitScratch fit;
    std::vector<std::vector<DirectedPoint>> rings;
    CandidateMarker marker;
};

/// Per-level candidate slots and segment labels retained across frames
struct CandidateLevel {
    std::vector<CandidateSlot> slots;
    std::vector<std::int32_t> segment_label;
    std::vector<std::int32_t> labelled_points;
};

} // namespace cctag::portable

#endif
