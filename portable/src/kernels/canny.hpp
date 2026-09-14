/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_KERNELS_CANNY_HPP
#define CCTAG_PORTABLE_KERNELS_CANNY_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace cctag::portable::kernels {

/// Computes the rounded Euclidean length of one gradient pixel
inline int magnitude_at(std::int16_t dx, std::int16_t dy) {
    const float x = static_cast<float>(dx);
    const float y = static_cast<float>(dy);
    return static_cast<int>(std::rint(std::sqrt(x * x + y * y)));
}

/// Classifies one pixel as suppressed (0), weak (1) or strong (2)
/// Magnitudes are the 3x3 neighbourhood in row-major order, zero outside the image
inline std::uint8_t nms_class_at(int dx, int dy, std::array<int, 9> magnitude, int low, int high) {
    const int m = magnitude[4];
    if (m <= low) {
        return 0;
    }

    // Select the diagonal before taking absolute values of the signed gradients
    const int s = (dx ^ dy) < 0 ? -1 : 1;
    const std::int64_t x = std::abs(dx);
    const std::int64_t y = static_cast<std::int64_t>(std::abs(dy)) << 15;
    constexpr int kTan22 = 13573;
    const std::int64_t tg22x = x * kTan22;
    const std::int64_t tg67x = tg22x + ((x + x) << 15);

    bool survives;
    if (y < tg22x) {
        survives = m > magnitude[3] && m >= magnitude[5];
    } else if (y > tg67x) {
        survives = m > magnitude[1] && m >= magnitude[7];
    } else {
        survives = m > magnitude[1 - s] && m > magnitude[7 + s];
    }
    return survives ? (m > high ? 2 : 1) : 0;
}

} // namespace cctag::portable::kernels

#ifdef CCTAG_TEST
#include <boost/ut.hpp>

namespace cctag::portable::tests::canny_element {

using namespace boost::ut;

inline suite<"canny_element"> canny_element_suite = [] {
    "magnitude rounds the euclidean gradient length to the nearest integer"_test = [] {
        expect(eq(kernels::magnitude_at(3, -4), 5));
        expect(eq(kernels::magnitude_at(1, 1), 1));
        expect(eq(kernels::magnitude_at(2, 2), 3));
        expect(eq(kernels::magnitude_at(-32768, 32767), 46340));
    };

    "nms compares the diagonal selected by the signed gradients"_test = [] {
        // Equal magnitudes suppress the northwest/southeast diagonal only
        const std::array<int, 9> magnitude = {14, 0, 0, 0, 14, 0, 0, 0, 14};
        expect(eq(kernels::nms_class_at(10, 10, magnitude, 2, 10), 0));
        expect(eq(kernels::nms_class_at(-10, -10, magnitude, 2, 10), 0));
        expect(eq(kernels::nms_class_at(-10, 10, magnitude, 2, 10), 2));
        expect(eq(kernels::nms_class_at(10, -10, magnitude, 2, 10), 2));
    };

    "nms keeps threshold and neighbour ties on the specified sides"_test = [] {
        std::array<int, 9> magnitude = {0, 0, 0, 0, 10, 0, 0, 0, 0};
        expect(eq(kernels::nms_class_at(10, 0, magnitude, 10, 20), 0));
        expect(eq(kernels::nms_class_at(10, 0, magnitude, 2, 10), 1));
        magnitude[5] = 10;
        expect(eq(kernels::nms_class_at(10, 0, magnitude, 2, 9), 2));
        magnitude[3] = 10;
        expect(eq(kernels::nms_class_at(10, 0, magnitude, 2, 9), 0));
        magnitude = {0, 0, 0, 0, 10, 0, 0, 10, 0};
        expect(eq(kernels::nms_class_at(0, -10, magnitude, 2, 9), 2));
        magnitude[1] = 10;
        expect(eq(kernels::nms_class_at(0, -10, magnitude, 2, 9), 0));

        // The angle comparison must also fit the full int16 gradient range
        magnitude = {0, 0, 0, 46340, 46340, 46340, 0, 0, 0};
        expect(eq(kernels::nms_class_at(-32768, 32767, magnitude, 2, 10), 2));
    };
};

} // namespace cctag::portable::tests::canny_element
#endif // CCTAG_TEST

#endif
