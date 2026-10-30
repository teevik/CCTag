/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_PIPELINE_HPP
#define CCTAG_PORTABLE_PIPELINE_HPP

#include <cstdint>

namespace cctag {
namespace portable {

/// A pixel of the edge map after suppress
enum class Mark : std::uint8_t {
    none = 0,
    weak = 1,
    strong = 2
};

/// The SYCL queue and device images of one pipe id
struct Pipeline;

/// The pipeline of a pipe id, created on first use. Detections on different pipelines can run at
/// the same time.
Pipeline& pipeline(int pipe_id);

/// Uploads a tightly packed gray image, builds the pyramid and runs every stage on every level
void load(
    Pipeline& pipeline,
    const std::uint8_t* image,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t levels,
    float canny_thr_low,
    float canny_thr_high
);

/// Copies the source image of a level into a packed host image
void download_source(Pipeline& pipeline, std::uint32_t level, std::uint8_t* dst);

/// Copies the gradient of a level into packed host images
void download_gradient(Pipeline& pipeline, std::uint32_t level, std::int16_t* dx, std::int16_t* dy);

/// Copies the edge map of a level, one Mark per pixel, into a packed host image
void download_map(Pipeline& pipeline, std::uint32_t level, Mark* dst);

} // namespace portable
} // namespace cctag

#endif
