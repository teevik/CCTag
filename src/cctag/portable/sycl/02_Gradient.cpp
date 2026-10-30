/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/portable/sycl/Stages.hpp>

#include <array>

namespace cctag {
namespace portable {

namespace {

/// How far a filter reaches either side of its centre
constexpr int radius = 4;

/// The weights of a filter, from offset -radius to radius
struct Weights {
    std::array<float, 2 * radius + 1> values;

    /// The weight at offset from the centre
    constexpr float operator()(int offset) const {
        return values[offset + radius];
    }
};

// The 9x9 kernel of cvRecodedCanny, split into two filters of 9 weights. Copied from
// cuda/frame_02_gaussian.cu
constexpr Weights gauss{
    {0.000053390535453f,
     0.001768051711852f,
     0.021539279301849f,
     0.096532352630054f,
     0.159154943091895f,
     0.096532352630054f,
     0.021539279301849f,
     0.001768051711852f,
     0.000053390535453f}
};
constexpr Weights derivative{
    {-0.002683701023220f,
     -0.066653979229454f,
     -0.541341132946452f,
     -1.213061319425269f,
     0.0f,
     1.213061319425269f,
     0.541341132946452f,
     0.066653979229454f,
     0.002683701023220f}
};

/// Filters down every column of the source image
void filter_columns(
    sycl::queue& queue,
    const Plane<const std::uint8_t> in,
    const Weights weights,
    const Plane<float> out
) {
    const int width = in.width;
    const int height = in.height;

    queue.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> id) {
        const int y = id[0];
        const int x = id[1];

        float sum = 0.0f;
        for (int offset = -radius; offset <= radius; ++offset) {
            const int row = sycl::clamp(y + offset, 0, height - 1);
            sum += weights(offset) * in.at(row, x);
        }

        out.at(y, x) = sum;
    });
}

/// Filters along every row of the intermediate image
void filter_rows(
    sycl::queue& queue,
    const Plane<const float> in,
    const Weights weights,
    const Plane<std::int16_t> out
) {
    const int width = in.width;
    const int height = in.height;

    queue.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> id) {
        const int y = id[0];
        const int x = id[1];

        float sum = 0.0f;
        for (int offset = -radius; offset <= radius; ++offset) {
            const int column = sycl::clamp(x + offset, 0, width - 1);
            sum += weights(offset) * in.at(y, column);
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
