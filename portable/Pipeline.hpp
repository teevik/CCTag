/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_PIPELINE_HPP
#define CCTAG_PORTABLE_PIPELINE_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cctag {
namespace portable {

/// An edge point of a level, with its links and its vote
struct Edgel {
    std::int16_t x;
    std::int16_t y;
    std::int16_t dx;
    std::int16_t dy;
    /// The edge points reached against and along the gradient, or -1
    std::int32_t before;
    std::int32_t after;
    /// The edge point this one votes for, or -1
    std::int32_t chosen = -1;
    std::uint32_t votes = 0;
    /// The sum of the field line lengths of the voters
    float flow = 0;
};

/// The most edge points a level holds, EdgePointCollection::MAX_POINTS
constexpr std::uint32_t max_edgels = std::uint32_t(1) << 24;

/// A cut of the center search, as ImageCut: the image point it stops at and the part of it,
/// from 0 to 1, its signal covers
struct Cut {
    float x;
    float y;
    float begin;
    float end;
};

/// A homography of the center search, from the marker plane to the image, and its inverse, both
/// row-major
struct Homography {
    float h[9];
    float inverse[9];
};

/// The most crowns the vote follows, as RESERVE_MEM_MAX_CROWNS of CUDA
constexpr std::uint32_t max_crowns = 5;

/// The SYCL queue and device images of one pipe id
struct Pipeline;

/// The pipeline of a pipe id, created on first use. Detections on different pipelines can run at
/// the same time.
Pipeline& pipeline(int pipe_id);

/// Uploads a tightly packed gray image, builds the pyramid and runs every stage on every level.
/// Throws std::invalid_argument if n_crowns is more than max_crowns
void load(
    Pipeline& pipeline,
    const std::uint8_t* image,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t levels,
    float canny_thr_low,
    float canny_thr_high,
    std::uint32_t dist_search,
    std::size_t n_crowns,
    float ratio_voting
);

/// Copies the source image of a level into a packed host image
void download_source(Pipeline& pipeline, std::uint32_t level, std::uint8_t* dst);

/// Copies the gradient of a level into packed host images
void download_gradient(Pipeline& pipeline, std::uint32_t level, std::int16_t* dx, std::int16_t* dy);

/// Copies the thinned edge image of a level, 0 or 255, into a packed host image
void download_edges(Pipeline& pipeline, std::uint32_t level, std::uint8_t* dst);

/// Copies the edge points of a level. Throws std::length_error if the level has more than
/// max_edgels, as EdgePointCollection::add_point would
void download_edgels(Pipeline& pipeline, std::uint32_t level, std::vector<Edgel>& dst);

/// Uploads the cuts the next center_costs calls read and makes room for up to points homographies
/// per call
void upload_cuts(
    Pipeline& pipeline,
    const std::vector<Cut>& cuts,
    std::uint32_t samples,
    std::uint32_t points
);

/// Computes the residual of costFunctionGlob over the uploaded cuts for every homography, or -1
/// where no pair of cuts lies in the image. Throws std::length_error if there are more
/// homographies than upload_cuts made room for
void center_costs(
    Pipeline& pipeline,
    const std::vector<Homography>& homographies,
    std::vector<float>& residuals
);

} // namespace portable
} // namespace cctag

#endif
