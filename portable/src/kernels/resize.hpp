/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_KERNELS_RESIZE_HPP
#define CCTAG_PORTABLE_KERNELS_RESIZE_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>

// Element functions of the `pyramid` stage: the arithmetic of `cv::resize` at `Level.cpp:67` as
// traced in docs/research/exact-stage-arithmetic.md §1. Two paths, chosen per level by the caller:
//  - both dimensions exactly halved -> the INTER_AREA fast path, `(a + b + c + d + 2) >> 2`;
//  - otherwise -> the generic fixed-point bilinear path with OpenCV's coefficient tables.

namespace cctag::portable::kernels {

/// One pixel of a 2:1 halving. `src` is the finer level, `x`/`y` the coarser-level pixel.
inline std::uint8_t halve_at(const std::uint8_t* src, std::size_t stride, std::uint32_t x, std::uint32_t y) {
    const std::uint8_t* r0 = src + static_cast<std::size_t>(2 * y) * stride + 2 * x;
    const std::uint8_t* r1 = r0 + stride;
    const unsigned sum = static_cast<unsigned>(r0[0]) + r0[1] + r1[0] + r1[1];
    return static_cast<std::uint8_t>((sum + 2) >> 2);
}

/// `saturate_cast<short>(float)`: `cvRound` (nearest-even under the default rounding mode) then a
/// clamp to the int16 range.
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

/// The coefficient tables of one axis of OpenCV's generic INTER_LINEAR path for 8-bit input
/// (`resize.cpp`, `fixpt = true`, `INTER_RESIZE_COEF_SCALE = 2048`). Host-computed sequential glue:
/// `ofs[d]` is the first source index of destination `d`, `alpha[2d]`/`alpha[2d + 1]` its two
/// weights, `max` the first destination index that reads only `ofs[d]`.
struct LinearAxis {
    const std::int32_t* ofs;
    const std::int16_t* alpha;
    std::uint32_t max;
};

/// Fills `ofs` (`dst_len` entries) and `alpha` (`2 * dst_len` entries); returns `max`.
inline std::uint32_t linear_axis_tables(
    std::uint32_t src_len,
    std::uint32_t dst_len,
    std::int32_t* ofs,
    std::int16_t* alpha
) {
    const double inv_scale = static_cast<double>(dst_len) / src_len;
    const double scale = 1. / inv_scale;
    std::uint32_t max = dst_len;
    for (std::uint32_t d = 0; d < dst_len; ++d) {
        float f = static_cast<float>((d + 0.5) * scale - 0.5);
        std::int32_t s = static_cast<std::int32_t>(std::floor(f));
        f -= static_cast<float>(s);
        if (s < 0) {
            f = 0.f;
            s = 0;
        }
        if (s + 1 >= static_cast<std::int32_t>(src_len)) {
            max = max < d ? max : d;
            if (s >= static_cast<std::int32_t>(src_len) - 1) {
                f = 0.f;
                s = static_cast<std::int32_t>(src_len) - 1;
            }
        }
        ofs[d] = s;
        alpha[2 * d] = round_to_int16((1.f - f) * 2048.f);
        alpha[2 * d + 1] = round_to_int16(f * 2048.f);
    }
    return max;
}

/// One pixel of the generic fixed-point bilinear path. Horizontal pass in int32 on the two source
/// rows, vertical pass with OpenCV's double truncation (`>> 4` then `>> 16`), then `(v + 2) >> 2`.
inline std::uint8_t bilinear_at(
    const std::uint8_t* src,
    std::size_t stride,
    std::uint32_t src_height,
    std::uint32_t x,
    std::uint32_t y,
    LinearAxis xs,
    LinearAxis ys
) {
    const std::int32_t sx = xs.ofs[x];
    const std::int32_t a0 = xs.alpha[2 * x];
    const std::int32_t a1 = xs.alpha[2 * x + 1];
    const std::int32_t sy = ys.ofs[y];
    const std::int32_t last = static_cast<std::int32_t>(src_height) - 1;
    const std::int32_t r0 = sy < 0 ? 0 : (sy > last ? last : sy);
    const std::int32_t r1 = sy + 1 < 0 ? 0 : (sy + 1 > last ? last : sy + 1);
    const std::uint8_t* row0 = src + static_cast<std::size_t>(r0) * stride;
    const std::uint8_t* row1 = src + static_cast<std::size_t>(r1) * stride;
    std::int32_t s0;
    std::int32_t s1;
    if (x < xs.max) {
        s0 = row0[sx] * a0 + row0[sx + 1] * a1;
        s1 = row1[sx] * a0 + row1[sx + 1] * a1;
    } else {
        s0 = row0[sx] * 2048;
        s1 = row1[sx] * 2048;
    }
    const std::int32_t b0 = ys.alpha[2 * y];
    const std::int32_t b1 = ys.alpha[2 * y + 1];
    const std::int32_t v = ((b0 * (s0 >> 4)) >> 16) + ((b1 * (s1 >> 4)) >> 16);
    return static_cast<std::uint8_t>((v + 2) >> 2);
}

} // namespace cctag::portable::kernels

#endif
