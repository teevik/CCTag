/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "portable/Stages.hpp"

#include <array>

namespace cctag {
namespace portable {

namespace {

/// A 9-tap filter, centred on the 5th tap
using Taps = std::array<float, 9>;

/// The two halves of h_gauss_filter in cuda/frame_02_gaussian.cu, whose outer product is the 9x9
/// kernel of cvRecodedCanny
constexpr Taps gauss{
    0.000053390535453f,
    0.001768051711852f,
    0.021539279301849f,
    0.096532352630054f,
    0.159154943091895f,
    0.096532352630054f,
    0.021539279301849f,
    0.001768051711852f,
    0.000053390535453f
};
constexpr Taps derivative{
    -0.002683701023220f,
    -0.066653979229454f,
    -0.541341132946452f,
    -1.213061319425269f,
    0.0f,
    1.213061319425269f,
    0.541341132946452f,
    0.066653979229454f,
    0.002683701023220f
};

/// Filters every column of the source image, replicating the border rows
void filter_columns(
    sycl::queue& queue,
    const Plane<const std::uint8_t> in,
    const Taps taps,
    const Plane<float> out
) {
    const int width = in.width;
    const int height = in.height;

    queue.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> id) {
        const int y = id[0];
        const int x = id[1];
        float sum = 0.0f;
        for (int k = 0; k < 9; ++k) {
            const int row = sycl::clamp(y + k - 4, 0, height - 1);
            sum += taps[k] * in.at(row, x);
        }
        out.at(y, x) = sum;
    });
}

/// Filters every row of an intermediate image, replicating the border columns and rounding to
/// nearest as cv::saturate_cast does
void filter_rows(
    sycl::queue& queue,
    const Plane<const float> in,
    const Taps taps,
    const Plane<std::int16_t> out
) {
    const int width = in.width;
    const int height = in.height;

    queue.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> id) {
        const int y = id[0];
        const int x = id[1];
        float sum = 0.0f;
        for (int k = 0; k < 9; ++k) {
            const int column = sycl::clamp(x + k - 4, 0, width - 1);
            sum += taps[k] * in.at(y, column);
        }
        out.at(y, x) = static_cast<std::int16_t>(sycl::rint(sum));
    });
}

} // namespace

void gradient(sycl::queue& queue, const Level& level, float* scratch) {
    const Plane<float> intermediate{scratch, level.width, level.height};
    filter_columns(queue, level.source_plane(), gauss, intermediate);
    filter_rows(queue, intermediate, derivative, level.dx_plane());
    filter_columns(queue, level.source_plane(), derivative, intermediate);
    filter_rows(queue, intermediate, gauss, level.dy_plane());
}

} // namespace portable
} // namespace cctag
