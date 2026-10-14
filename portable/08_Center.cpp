/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "portable/Stages.hpp"

namespace cctag {
namespace portable {

namespace {

/// The point (x, y) mapped by the row-major homography m, as applyHomography
void apply(const float* m, float x, float y, float& x_res, float& y_res) {
    const float u = m[0] * x + m[1] * y + m[2];
    const float v = m[3] * x + m[4] * y + m[5];
    const float w = m[6] * x + m[7] * y + m[8];
    x_res = u / w;
    y_res = v / w;
}

/// The image interpolated at (x, y), as the CPU getPixelBilinear, halved like it. (x, y) must
/// lie within the image, short of the last row and column.
float bilinear(Plane<const std::uint8_t> image, float x, float y) {
    const int px = int(x);
    const int py = int(y);
    const float fx = x - px;
    const float fy = y - py;
    const float fx1 = 1.0f - fx;
    const float fy1 = 1.0f - fy;
    return (image.at(py, px) * (fx1 * fy1) + image.at(py, px + 1) * (fx * fy1)
            + image.at(py + 1, px) * (fx1 * fy) + image.at(py + 1, px + 1) * (fx * fy))
        / 2;
}

/// Samples the rectified signal of a cut through a homography into signal, as
/// extractSignalUsingHomography. Returns whether a sample left the image, whose value is then not
/// written.
bool extract(
    Plane<const std::uint8_t> image,
    const Homography& homography,
    const Cut& cut,
    std::uint32_t samples,
    float* signal
) {
    float stop_x;
    float stop_y;
    apply(homography.inverse, cut.x, cut.y, stop_x, stop_y);

    const float x_start = cut.begin != 0.f ? stop_x * cut.begin : 0.f;
    const float y_start = cut.begin != 0.f ? stop_y * cut.begin : 0.f;
    const float x_stop = cut.end != 1.f ? stop_x * cut.end : stop_x;
    const float y_stop = cut.end != 1.f ? stop_y * cut.end : stop_y;
    const float step_x = (x_stop - x_start) / (samples - 1.f);
    const float step_y = (y_stop - y_start) / (samples - 1.f);

    bool out_of_bounds = false;
    float x = x_start;
    float y = y_start;
    for (std::uint32_t i = 0; i < samples; ++i) {
        float x_res;
        float y_res;
        apply(homography.h, x, y, x_res, y_res);
        if (x_res >= 0.f && x_res < image.width - 1.f && y_res >= 0.f
            && y_res < image.height - 1.f) {
            signal[i] = bilinear(image, x_res, y_res);
        } else {
            out_of_bounds = true;
        }
        x += step_x;
        y += step_y;
    }
    return out_of_bounds;
}

} // namespace

void center_costs(
    sycl::queue& queue,
    const Level& level,
    const CenterBuffers& buffers,
    std::uint32_t points,
    std::uint32_t cuts,
    std::uint32_t samples
) {
    const Plane<const std::uint8_t> image = level.source_plane();
    const Homography* homographies = buffers.homographies;
    const Cut* cut = buffers.cut;
    float* signals = buffers.signals;
    std::uint8_t* out_of_bounds = buffers.out_of_bounds;
    float* residuals = buffers.residuals;

    queue.parallel_for(sycl::range<1>(std::size_t(points) * cuts), [=](sycl::id<1> id) {
        const std::size_t i = id[0];
        out_of_bounds[i] =
            extract(image, homographies[i / cuts], cut[i % cuts], samples, signals + i * samples);
    });

    // The mean squared difference over the pairs of cuts in the image, as costFunctionGlob
    queue.parallel_for(sycl::range<1>(points), [=](sycl::id<1> id) {
        const std::size_t p = id[0];
        const float* signal = signals + p * cuts * samples;
        const std::uint8_t* outside = out_of_bounds + p * cuts;
        float res = 0;
        std::uint32_t pairs = 0;
        for (std::uint32_t i = 0; i + 1 < cuts; ++i) {
            if (outside[i]) {
                continue;
            }
            for (std::uint32_t j = i + 1; j < cuts; ++j) {
                if (outside[j]) {
                    continue;
                }
                for (std::uint32_t k = 0; k < samples; ++k) {
                    const float d = signal[i * samples + k] - signal[j * samples + k];
                    res += d * d;
                }
                ++pairs;
            }
        }
        residuals[p] = pairs > 0 ? res / pairs : -1.f;
    });
}

} // namespace portable
} // namespace cctag
