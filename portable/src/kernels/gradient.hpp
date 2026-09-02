/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_KERNELS_GRADIENT_HPP
#define CCTAG_PORTABLE_KERNELS_GRADIENT_HPP

#include "resize.hpp" // round_to_int16

#include <cstddef>
#include <cstdint>

// Element function of the `gradient` stage: `cv::filter2D(src, CV_16SC1, kernel, BORDER_REPLICATE)`
// with the fork's 9x9 derivative kernel, as traced in docs/research/exact-stage-arithmetic.md §2.
// The 72 non-zero taps are visited in row-major order of the kernel, each a float32 multiply then
// a float32 add — never fused (the translation unit must be compiled with -ffp-contract=off) —
// then nearest-even rounding and an int16 clamp.

namespace cctag::portable::kernels {

/// `kerneldX` of `filter/cvRecode.cpp:74-84`, verbatim. `kerneldY` is its transpose.
inline constexpr float kDerivativeKernel[9][9] = {
    {-0.000000143284235f, -0.000003558691641f, -0.000028902492951f, -0.000064765993382f, 0.f, 0.000064765993382f, 0.000028902492951f, 0.000003558691641f, 0.000000143284235f},
    {-0.000004744922188f, -0.000117847682078f, -0.000957119116802f, -0.002144755142391f, 0.f, 0.002144755142391f, 0.000957119116802f, 0.000117847682078f, 0.000004744922188f},
    {-0.000057804985902f, -0.001435678675203f, -0.011660097860113f, -0.026128466569370f, 0.f, 0.026128466569370f, 0.011660097860113f, 0.001435678675203f, 0.000057804985902f},
    {-0.000259063973527f, -0.006434265427174f, -0.052256933138740f, -0.117099663048638f, 0.f, 0.117099663048638f, 0.052256933138740f, 0.006434265427174f, 0.000259063973527f},
    {-0.000427124283626f, -0.010608310271112f, -0.086157117207395f, -0.193064705260108f, 0.f, 0.193064705260108f, 0.086157117207395f, 0.010608310271112f, 0.000427124283626f},
    {-0.000259063973527f, -0.006434265427174f, -0.052256933138740f, -0.117099663048638f, 0.f, 0.117099663048638f, 0.052256933138740f, 0.006434265427174f, 0.000259063973527f},
    {-0.000057804985902f, -0.001435678675203f, -0.011660097860113f, -0.026128466569370f, 0.f, 0.026128466569370f, 0.011660097860113f, 0.001435678675203f, 0.000057804985902f},
    {-0.000004744922188f, -0.000117847682078f, -0.000957119116802f, -0.002144755142391f, 0.f, 0.002144755142391f, 0.000957119116802f, 0.000117847682078f, 0.000004744922188f},
    {-0.000000143284235f, -0.000003558691641f, -0.000028902492951f, -0.000064765993382f, 0.f, 0.000064765993382f, 0.000028902492951f, 0.000003558691641f, 0.000000143284235f},
};

/// The non-zero taps of one derivative kernel in the order `preprocess2DKernel` keeps them:
/// row-major, zero coefficients dropped. Offsets are relative to the anchor (4, 4).
struct GradientTaps {
    static constexpr int count = 72;
    float coefficient[count];
    std::int8_t dx[count];
    std::int8_t dy[count];
};

/// `transposed = false` builds the `dx` taps (column 4 is zero), `true` the `dy` taps (row 4).
constexpr GradientTaps make_gradient_taps(bool transposed) {
    GradientTaps taps{};
    int k = 0;
    for (int ky = 0; ky < 9; ++ky) {
        for (int kx = 0; kx < 9; ++kx) {
            const float c = transposed ? kDerivativeKernel[kx][ky] : kDerivativeKernel[ky][kx];
            if (c == 0.f) {
                continue;
            }
            taps.coefficient[k] = c;
            taps.dx[k] = static_cast<std::int8_t>(kx - 4);
            taps.dy[k] = static_cast<std::int8_t>(ky - 4);
            ++k;
        }
    }
    return taps;
}

inline constexpr GradientTaps kDxTaps = make_gradient_taps(false);
inline constexpr GradientTaps kDyTaps = make_gradient_taps(true);

/// One output pixel of `filter2D` with replicate borders: 72 separately rounded multiply-adds in
/// tap order, then `saturate_cast<short>`.
inline std::int16_t gradient_at(
    const std::uint8_t* src,
    std::size_t stride,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t x,
    std::uint32_t y,
    const GradientTaps& taps
) {
    const int last_x = static_cast<int>(width) - 1;
    const int last_y = static_cast<int>(height) - 1;
    float sum = 0.f;
    for (int k = 0; k < GradientTaps::count; ++k) {
        int sx = static_cast<int>(x) + taps.dx[k];
        int sy = static_cast<int>(y) + taps.dy[k];
        sx = sx < 0 ? 0 : (sx > last_x ? last_x : sx);
        sy = sy < 0 ? 0 : (sy > last_y ? last_y : sy);
        const float pixel = static_cast<float>(src[static_cast<std::size_t>(sy) * stride + sx]);
        const float product = taps.coefficient[k] * pixel;
        sum = sum + product;
    }
    return round_to_int16(sum);
}

} // namespace cctag::portable::kernels

#endif
