/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_HOST_MARKERS_HPP
#define CCTAG_PORTABLE_HOST_MARKERS_HPP

#include "host/candidates.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace cctag {
class CCTagMarkersBank;
}

namespace cctag::portable {

/// Identification results, retained even when the id is unreliable
struct Marker {
    std::int32_t id = -1;
    std::int32_t status = -1;
    Eigen::Vector2f center = Eigen::Vector2f::Zero();
    Ellipse outer_ellipse;
    Eigen::Matrix3f homography = Eigen::Matrix3f::Identity();
    float quality = 0;
};

/// Flat radius ratios in bank id order, with one offset per id and a final sentinel
struct MarkerBank {
    std::size_t crowns = 0;
    bool custom = false;
    std::vector<float> ratios;
    std::vector<std::size_t> offsets;

    void ensure(std::size_t count);
    void set(const CCTagMarkersBank& bank);
};

/// One candidate marker's cuts and scratch, reused across frames
struct IdentificationScratch {
    struct Cut {
        DirectedPoint stop;
        std::vector<float> signal;
        float variance = 0;
        bool out_of_bounds = false;
    };
    std::vector<std::pair<float, std::size_t>> angles;
    std::vector<Cut> cuts;
    std::vector<std::size_t> selected;
    std::vector<std::size_t> eligible;
    std::vector<float> refinement;
    std::vector<float> barcode;
    std::vector<float> sample;
    struct Score {
        std::size_t count = 0;
        float sum = 0;
    };
    std::vector<Score> scores;
};

} // namespace cctag::portable

#endif
