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

/// The work-items per homography, more than the 231 pairs of the default 22 cuts
constexpr std::uint32_t group_size = 256;

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
    return (image.at(py, px) * (fx1 * fy1)
            + image.at(py, px + 1) * (fx * fy1)
            + image.at(py + 1, px) * (fx1 * fy)
            + image.at(py + 1, px + 1) * (fx * fy))
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
        if (x_res >= 0.f
            && x_res < image.width - 1.f
            && y_res >= 0.f
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
    float* residuals = buffers.residuals;

    // One work-group per homography: each work-item samples a cut into local memory, then sums the
    // squared differences of a pair of cuts, as costFunctionGlob
    queue.submit([&](sycl::handler& handler) {
        const sycl::local_accessor<float, 1> signals(std::size_t(cuts) * samples, handler);
        const sycl::local_accessor<std::uint8_t, 1> out_of_bounds(cuts, handler);
        const auto kernel = [=](sycl::nd_item<1> item) {
            const std::uint32_t point = item.get_group(0);
            const std::uint32_t local = item.get_local_id(0);
            for (std::uint32_t c = local; c < cuts; c += group_size) {
                out_of_bounds[c] =
                    extract(image, homographies[point], cut[c], samples, &signals[c * samples]);
            }
            sycl::group_barrier(item.get_group());

            float sum = 0;
            std::uint32_t pairs = 0;
            std::uint32_t pair = 0;
            for (std::uint32_t i = 0; i + 1 < cuts; ++i) {
                for (std::uint32_t j = i + 1; j < cuts; ++j, ++pair) {
                    if (pair % group_size != local || out_of_bounds[i] || out_of_bounds[j]) {
                        continue;
                    }
                    for (std::uint32_t k = 0; k < samples; ++k) {
                        const float d = signals[i * samples + k] - signals[j * samples + k];
                        sum += d * d;
                    }
                    ++pairs;
                }
            }
            sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>());
            pairs = sycl::reduce_over_group(item.get_group(), pairs, sycl::plus<std::uint32_t>());
            if (local == 0) {
                residuals[point] = pairs > 0 ? sum / pairs : -1.f;
            }
        };
        handler.parallel_for(
            sycl::nd_range<1>(std::size_t(points) * group_size, group_size),
            kernel
        );
    });
}

} // namespace portable
} // namespace cctag
