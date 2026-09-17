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
#include <span>
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
    Ellipse rescaled_outer_ellipse;
    std::vector<DirectedPoint> outer_points;
};

/// Active candidate markers backed by storage retained across frames, including outer points
class CandidateMarkers {
  public:
    void clear() {
        count = 0;
    }

    void push_back(const CandidateMarker& marker) {
        if (count == storage.size()) {
            storage.emplace_back();
        }
        storage[count] = marker;
        ++count;
    }

    std::span<CandidateMarker> view() {
        return {storage.data(), count};
    }

    std::span<const CandidateMarker> view() const {
        return {storage.data(), count};
    }

  private:
    std::vector<CandidateMarker> storage;
    std::size_t count = 0;
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
    /// Outer-ring points accumulated while validating and assembling flow components
    std::vector<DirectedPoint> flow_outer_points;
    CandidateMarker marker;

    /// Starts the candidate's result state and random stream without releasing scratch storage
    void reset(int level, std::int32_t seed_index, std::size_t point_count) {
        // Touched indices refer to the previous frame's plane, so clear before resizing it.
        clear_marks();
        processed.resize(point_count, 0);
        seed = seed_index;
        accepted = false;
        has_marker = false;
        label = -1;
        filtered_children.clear();
        outer_points.clear();
        kernels::pcg32_seed(
            random,
            271828,
            (std::uint64_t{static_cast<std::uint32_t>(level)} << 32)
                | static_cast<std::uint32_t>(seed)
        );
    }

    void clear_marks() {
        for (const auto index : touched) {
            processed[index] = 0;
        }
        touched.clear();
    }

    void mark(std::int32_t index) {
        if (!processed[index]) {
            processed[index] = 1;
            touched.push_back(index);
        }
    }
};

/// Per-level candidate slots and segment labels retained across frames
struct CandidateLevel {
    std::vector<CandidateSlot> slots;
    std::vector<std::int32_t> segment_label;
    std::vector<std::int32_t> labelled_points;
};

} // namespace cctag::portable

#endif
