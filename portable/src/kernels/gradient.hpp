/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_KERNELS_GRADIENT_HPP
#define CCTAG_PORTABLE_KERNELS_GRADIENT_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace cctag::portable::kernels {

/// Rounds using the current rounding mode and clamps to the int16 range
inline std::int16_t round_to_int16(float value) {
    const float rounded = std::nearbyint(value);
    if (rounded >= 32767.f) {
        return 32767;
    }
    if (rounded <= -32768.f) {
        return -32768;
    }
    return static_cast<std::int16_t>(rounded);
}

/// Horizontal derivative kernel (`kerneldX`) from src/cctag/filter/cvRecode.cpp
/// The vertical derivative kernel is its transpose
inline constexpr float kDerivativeKernel[9][9] = {
    {-0.000000143284235f,
     -0.000003558691641f,
     -0.000028902492951f,
     -0.000064765993382f,
     0.f,
     0.000064765993382f,
     0.000028902492951f,
     0.000003558691641f,
     0.000000143284235f},
    {-0.000004744922188f,
     -0.000117847682078f,
     -0.000957119116802f,
     -0.002144755142391f,
     0.f,
     0.002144755142391f,
     0.000957119116802f,
     0.000117847682078f,
     0.000004744922188f},
    {-0.000057804985902f,
     -0.001435678675203f,
     -0.011660097860113f,
     -0.026128466569370f,
     0.f,
     0.026128466569370f,
     0.011660097860113f,
     0.001435678675203f,
     0.000057804985902f},
    {-0.000259063973527f,
     -0.006434265427174f,
     -0.052256933138740f,
     -0.117099663048638f,
     0.f,
     0.117099663048638f,
     0.052256933138740f,
     0.006434265427174f,
     0.000259063973527f},
    {-0.000427124283626f,
     -0.010608310271112f,
     -0.086157117207395f,
     -0.193064705260108f,
     0.f,
     0.193064705260108f,
     0.086157117207395f,
     0.010608310271112f,
     0.000427124283626f},
    {-0.000259063973527f,
     -0.006434265427174f,
     -0.052256933138740f,
     -0.117099663048638f,
     0.f,
     0.117099663048638f,
     0.052256933138740f,
     0.006434265427174f,
     0.000259063973527f},
    {-0.000057804985902f,
     -0.001435678675203f,
     -0.011660097860113f,
     -0.026128466569370f,
     0.f,
     0.026128466569370f,
     0.011660097860113f,
     0.001435678675203f,
     0.000057804985902f},
    {-0.000004744922188f,
     -0.000117847682078f,
     -0.000957119116802f,
     -0.002144755142391f,
     0.f,
     0.002144755142391f,
     0.000957119116802f,
     0.000117847682078f,
     0.000004744922188f},
    {-0.000000143284235f,
     -0.000003558691641f,
     -0.000028902492951f,
     -0.000064765993382f,
     0.f,
     0.000064765993382f,
     0.000028902492951f,
     0.000003558691641f,
     0.000000143284235f},
};

/// Nonzero coefficients and pixel offsets for one derivative kernel
/// Offsets are relative to the kernel's centre at (4, 4)
struct GradientTaps {
    static constexpr int count = 72;
    float coefficient[count];
    std::int8_t dx[count];
    std::int8_t dy[count];
};

/// Builds horizontal (`dx`) taps, or vertical (`dy`) taps when `transposed` is true
/// Skips the zero coefficients in the middle column or row
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

/// Computes one gradient pixel, repeating edge pixels outside the image
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

#ifdef CCTAG_TEST
#include <boost/ut.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>

namespace cctag::portable::kernels::tests {

using namespace boost::ut;

/// Creates a repeatable test image with rings and noise to vary the pixel values
inline cv::Mat1b test_image(int width, int height) {
    cv::Mat1b image(height, width);
    std::uint32_t state = 12345u;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            state = state * 1664525u + 1013904223u;
            const int ring = ((x * x + y * y) / 37) % 2 ? 200 : 40;
            image(y, x) = static_cast<std::uint8_t>(ring + (state >> 27));
        }
    }
    return image;
}

/// Checks both gradients against OpenCV's `filter2D` at every pixel
inline void expect_gradient_at_reproduces_filter2d(int width, int height) {
    const cv::Mat1b image = test_image(width, height);
    cv::Mat1f kernel_dx(9, 9);
    std::copy_n(&kernels::kDerivativeKernel[0][0], kernel_dx.total(), kernel_dx.begin());
    const cv::Mat1f kernel_dy = kernel_dx.t();
    cv::Mat1s dx, dy;
    cv::filter2D(image, dx, CV_16SC1, kernel_dx, cv::Point{-1, -1}, 0.0, cv::BORDER_REPLICATE);
    cv::filter2D(image, dy, CV_16SC1, kernel_dy, cv::Point{-1, -1}, 0.0, cv::BORDER_REPLICATE);

    const std::size_t stride = image.step1();
    const auto w = static_cast<std::uint32_t>(width);
    const auto h = static_cast<std::uint32_t>(height);
    for (std::uint32_t y = 0; y < h; ++y) {
        for (std::uint32_t x = 0; x < w; ++x) {
            expect(
                eq(kernels::gradient_at(image[0], stride, w, h, x, y, kernels::kDxTaps),
                   dx(static_cast<int>(y), static_cast<int>(x)))
            ) << "dx"
              << width << "x" << height << "pixel" << x << y;
            expect(
                eq(kernels::gradient_at(image[0], stride, w, h, x, y, kernels::kDyTaps),
                   dy(static_cast<int>(y), static_cast<int>(x)))
            ) << "dy"
              << width << "x" << height << "pixel" << x << y;
        }
    }
}

inline suite<"gradient_element"> gradient_element_suite = [] {
    "gradient at reproduces filter2d at every pixel"_test = [] {
        expect_gradient_at_reproduces_filter2d(37, 23);
    };

    // Check dimensions from 1 to 12, covering images smaller and larger than the 9x9 kernel
    "gradient at reproduces filter2d on images smaller than the kernel"_test = [] {
        for (int height = 1; height <= 12; ++height) {
            for (int width = 1; width <= 12; ++width) {
                expect_gradient_at_reproduces_filter2d(width, height);
            }
        }
    };
};

} // namespace cctag::portable::kernels::tests
#endif // CCTAG_TEST

#endif
