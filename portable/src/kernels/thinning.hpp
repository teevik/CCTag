/*
 * Copyright 2016, 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_KERNELS_THINNING_HPP
#define CCTAG_PORTABLE_KERNELS_THINNING_HPP

#include "kernels/plane.hpp"

#include <cstdint>

namespace cctag::portable::kernels {

/// Thinning pass 1 table from src/cctag/filter/thinning.cpp
inline constexpr std::uint8_t kThinning1[512] = {
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255,
    255, 255, 255, 255, 0,   255, 255, 0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 0,   0,   255, 255, 0,   0,   255,
    255, 0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   255, 255, 0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   255, 255, 0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    255, 255, 0,   0,   0,   255, 0,   0,   255, 255, 0,   0,   255, 255, 0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 0,   0,   0,
    255, 0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255
};

/// Thinning pass 2 table from src/cctag/filter/thinning.cpp
inline constexpr std::uint8_t kThinning2[512] = {
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 0,   255, 255, 255, 255, 0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   255, 0,   255, 255, 255,
    255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255,
    0,   0,   255, 0,   255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   255, 255, 255, 0,   255, 255, 255, 0,   0,   255, 255, 0,   255,
    255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255,
    255, 255, 255, 255, 255, 255, 255, 0,   0,   255, 0,   255, 255, 255, 255, 0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 0,   255, 255,
    255, 0,   0,   255, 255, 0,   255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    255, 255, 255, 255, 0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   255, 255, 255, 255, 255, 255, 255, 255, 0,   0,   255, 0,   255, 255, 255, 255, 0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 0,
    255, 255, 255, 0,   0,   255, 255, 0,   255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   255, 255, 255, 255, 255, 255, 255, 0,   0,   255,
    0,   255, 255, 255, 255, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0,   0,   0,   255, 255, 255, 0,   255, 255, 255, 0,   0,   255, 255, 0,   255, 255, 255
};

/// Computes one interior pixel of a thinning pass using its 3x3 neighbourhood
/// The caller leaves the output border untouched and keeps the passes in separate planes
inline std::uint8_t thinning_at(
    Plane<const std::uint8_t> input,
    std::uint32_t x,
    std::uint32_t y,
    const std::uint8_t* lut
) {
    const auto* up = input.row(y - 1);
    const auto* middle = input.row(y);
    const auto* down = input.row(y + 1);
    // The legacy tables number the neighbourhood in column-major order
    const int index = (up[x - 1] == 255) + (middle[x - 1] == 255) * 2 + (down[x - 1] == 255) * 4
        + (up[x] == 255) * 8 + (middle[x] == 255) * 16 + (down[x] == 255) * 32
        + (up[x + 1] == 255) * 64 + (middle[x + 1] == 255) * 128 + (down[x + 1] == 255) * 256;
    return lut[index];
}

} // namespace cctag::portable::kernels

#ifdef CCTAG_TEST
#include <boost/ut.hpp>

#include <array>

namespace cctag::portable::tests::thinning_element {

using namespace boost::ut;

inline suite<"thinning_element"> thinning_element_suite = [] {
    "thinning preserves an isolated pixel with a zero border"_test = [] {
        std::array<std::uint8_t, 9> pixels = {0, 0, 0, 0, 255, 0, 0, 0, 0};
        const kernels::Plane<const std::uint8_t> input{pixels.data(), 3, 3, 3};
        expect(eq(kernels::thinning_at(input, 1, 1, kernels::kThinning1), 255));
        expect(eq(kernels::thinning_at(input, 1, 1, kernels::kThinning2), 255));
        pixels[4] = 0;
        expect(eq(kernels::thinning_at(input, 1, 1, kernels::kThinning1), 0));
        expect(eq(kernels::thinning_at(input, 1, 1, kernels::kThinning2), 0));
    };

    "thinning reads neighbours in the legacy table order"_test = [] {
        // Centre and left column: table entry 23 removes the centre only in pass one
        const std::array<std::uint8_t, 9> pixels = {255, 0, 0, 255, 255, 0, 255, 0, 0};
        const kernels::Plane<const std::uint8_t> input{pixels.data(), 3, 3, 3};
        expect(eq(kernels::thinning_at(input, 1, 1, kernels::kThinning1), 0));
        expect(eq(kernels::thinning_at(input, 1, 1, kernels::kThinning2), 255));
    };
};

} // namespace cctag::portable::tests::thinning_element
#endif // CCTAG_TEST

#endif
