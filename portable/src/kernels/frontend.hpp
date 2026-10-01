/*
 * Copyright 2016, 2026, Simula Research Laboratory
 * SPDX-License-Identifier: MPL-2.0
 * Element functions of the pyramid, gradient and edge stages, derived from the CUDA
 * pipeline (src/cctag/cuda/frame_01_tex.cu through frame_05_thin.cu).
 */
#ifndef CCTAG_PORTABLE_KERNELS_FRONTEND_HPP
#define CCTAG_PORTABLE_KERNELS_FRONTEND_HPP

#include "kernels/plane.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace cctag::portable::kernels {

/// Taps of the separable gradient filter at offsets x = -4..4: `gaussian_kernel` is
/// exp(-x^2 / 2) / (2 pi) and `derivative_kernel` is 2x exp(-x^2 / 2). Each gradient
/// axis is smoothed across with the first and differentiated along with the second.
inline constexpr std::array<float, 9> gaussian_kernel = {
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
inline constexpr std::array<float, 9> derivative_kernel = {
    -0.002683701023220f,
    -0.066653979229454f,
    -0.541341132946452f,
    -1.213061319425269f,
    0.f,
    1.213061319425269f,
    0.541341132946452f,
    0.066653979229454f,
    0.002683701023220f
};

/// Samples level zero for pixel (x, y) of a `width` x `height` pyramid level, as CUDA's
/// clamped bilinear texture fetch at normalized coordinate (x / width, y / height) does.
/// The fetch quantizes the coordinate to 21 fraction bits and every interpolation
/// weight, including the cross term, to 8 bits, and returns a UNORM16 value.
inline std::uint8_t
pyramid_at(Plane<const std::uint8_t> source, int x, int y, int width, int height) {
    const auto coordinate = [](int value, int extent, std::uint32_t source_extent) {
        // Normalized coordinate with 21 fraction bits
        const auto normalized = static_cast<std::uint64_t>((float(value) / extent) * 2097152.f);
        // Texel coordinate minus half a texel, rounded to 1/256 texel
        const auto fixed =
            static_cast<std::int64_t>((normalized * source_extent + 4096) / 8192) - 128;
        // Split into the floored texel index and its 8-bit weight
        const auto pixel = fixed >= 0 ? fixed / 256 : -((-fixed + 255) / 256);
        return std::array<int, 2>{static_cast<int>(pixel), static_cast<int>(fixed - pixel * 256)};
    };
    const auto [ix, fx] = coordinate(x, width, source.width);
    const auto [iy, fy] = coordinate(y, height, source.height);
    const auto sample = [&](int px, int py) {
        return int(source.row(
            std::clamp(py, 0, int(source.height) - 1)
        )[std::clamp(px, 0, int(source.width) - 1)]);
    };
    const int a = sample(ix, iy), b = sample(ix + 1, iy);
    const int c = sample(ix, iy + 1), d = sample(ix + 1, iy + 1);
    const int cross = (fx * fy + 128) / 256;
    // Interpolate in 1/256 units, then widen the 8-bit scale to UNORM16 (255 -> 65535)
    const int weighted = a * 256 + (b - a) * fx + (c - a) * fy + (a - b - c + d) * cross;
    const int normalized = (weighted * 257 + 128) / 256;
    // Convert back to 8 bits, rounding the division and the multiplication separately
    return static_cast<std::uint8_t>((float(normalized) / 65535.f) * 255.f);
}

/// One pass of a separable 9-tap filter centred on (x, y): along y when `vertical`,
/// otherwise along x. Taps outside the plane replicate the border, and the products
/// accumulate with fused multiply-adds in tap order.
template <class T>
inline float filter_at(
    Plane<const T> source,
    int x,
    int y,
    bool vertical,
    const std::array<float, 9>& coefficients
) {
    float value = 0;
    for (int i = 0; i < 9; ++i) {
        const int px = std::clamp(x + (vertical ? 0 : i - 4), 0, int(source.width) - 1);
        const int py = std::clamp(y + (vertical ? i - 4 : 0), 0, int(source.height) - 1);
        value = std::fma(static_cast<float>(source.row(py)[px]), coefficients[i], value);
    }
    return value;
}

/// Truncated Euclidean gradient magnitude
inline int magnitude_at(std::int16_t dx, std::int16_t dy) {
    const std::int64_t x = dx, y = dy;
    return static_cast<int>(std::sqrt(static_cast<float>(x * x + y * y)));
}

/// Non-maximum suppression and double threshold at the centre of the row-major 3x3
/// magnitude neighbourhood `m`. Returns 0 when suppressed or at most `low`, 2 when above
/// `high`, and 1 otherwise. Along the gradient, the centre must exceed the neighbour that
/// comes first in `m` and be at least the other. The gradient sector is chosen in fixed
/// point: 13573 / 2^15 is tan(22.5 deg), and tan(67.5 deg) = tan(22.5 deg) + 2.
inline std::uint8_t nms_at(int dx, int dy, const std::array<int, 9>& m, float low, float high) {
    if (m[4] <= low) {
        return 0;
    }
    const int sign = (dx ^ dy) < 0 ? -1 : 1;
    const std::int64_t x = std::abs(dx), y = std::int64_t(std::abs(dy)) << 15;
    const auto tan22 = x * 13573, tan67 = tan22 + (x << 16);
    const int before = y < tan22 ? 3 : y > tan67 ? 1 : 1 - sign;
    const int after = y < tan22 ? 5 : y > tan67 ? 7 : 7 + sign;
    return m[4] > m[before] && m[4] >= m[after] ? 1 + (m[4] > high) : 0;
}

/// Lookup tables of the two thinning passes, indexed by the 8-neighbour mask: bit i is set
/// when neighbour i, clockwise from north, is an edge. An entry of 1 keeps the centre.
inline constexpr std::uint8_t thinning_first_pass[256] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1,
    1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1,
    1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1,
    1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1,
    0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1,
};

inline constexpr std::uint8_t thinning_second_pass[256] = {
    1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
};

/// Applies one thinning pass at (x, y) to a plane whose edge pixels are 2. The first pass
/// keeps an edge as 2 so that its output feeds the second pass, and the second keeps it as 1.
/// Pixels on the image border are cleared.
inline std::uint8_t thinning_at(Plane<const std::uint8_t> input, int x, int y, bool first) {
    if (x < 1 || y < 1 || x >= int(input.width) - 1 || y >= int(input.height) - 1
        || input.row(y)[x] != 2) {
        return 0;
    }
    constexpr int ox[] = {0, 1, 1, 1, 0, -1, -1, -1};
    constexpr int oy[] = {-1, -1, 0, 1, 1, 1, 0, -1};
    unsigned mask = 0;
    for (int i = 0; i < 8; ++i) {
        mask |= (input.row(y + oy[i])[x + ox[i]] == 2) << i;
    }
    return first ? 2 * thinning_first_pass[mask] : thinning_second_pass[mask];
}

} // namespace cctag::portable::kernels

#ifdef CCTAG_TEST
#include <boost/ut.hpp>

namespace cctag::portable::tests::frontend {

using namespace boost::ut;

inline suite<"frontend"> frontend_suite = [] {
    "texture sampling quantizes the cross weight before its UNORM16 result"_test = [] {
        const std::array<std::uint8_t, 4> corners{0, 0, 0, 255};
        const kernels::Plane<const std::uint8_t> plane{corners.data(), 2, 2, 2};
        // Expected value measured on CUDA's texture unit with a 2x2 image and derived by hand:
        // (100/257, 100/259) gives weights X = 71 and Y = 70 in 1/256 units, the cross weight
        // round(X * Y / 256) = 19, q = round(255 * 19 * 257 / 256) = 4864 and
        // trunc(q / 65535 * 255) = 18. Unquantized bilinear interpolation would give 19.
        expect(eq(kernels::pyramid_at(plane, 100, 100, 257, 259), 18));
    };

    "gradient magnitude truncates and does not overflow at the 16-bit extremes"_test = [] {
        expect(eq(kernels::magnitude_at(3, 4), 5));
        // sqrt(2 * 185^2) = 261.63
        expect(eq(kernels::magnitude_at(185, 185), 261));
        // sqrt(2 * 32767^2) = 46339.4 and sqrt(2 * 32768^2) = 46340.95
        expect(eq(kernels::magnitude_at(32767, 32767), 46339));
        expect(eq(kernels::magnitude_at(-32768, -32768), 46340));
    };

    "non-maximum suppression keeps a tie with the second neighbour along the gradient"_test = [] {
        // A diagonal gradient compares the centre with its top-left and bottom-right
        // neighbours. The centre must exceed the first and may equal the second.
        constexpr int dx = 10, dy = 10;
        constexpr float low = 2, high = 10;
        constexpr int strong = 2;
        // clang-format off
        const std::array<int, 9> tie_after{
            0,  0,  0,
            0, 14,  0,
            0,  0, 14,
        };
        const std::array<int, 9> tie_before{
            14,  0,  0,
             0, 14,  0,
             0,  0,  0,
        };
        // clang-format on
        expect(eq(kernels::nms_at(dx, dy, tie_after, low, high), strong));
        expect(eq(kernels::nms_at(dx, dy, tie_before, low, high), 0));
    };
};

} // namespace cctag::portable::tests::frontend
#endif // CCTAG_TEST

#endif
