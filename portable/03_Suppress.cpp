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

// tan(22.5°) and tan(67.5°) are used to determine the gradient direction
constexpr float tan22_5 = 0.41421356f;
constexpr float tan67_5 = 2.41421356f;

/// The neighbours of a pixel along its gradient, at `i - step` and `i + step`
struct Direction {
    int step;
    bool diagonal;
};

} // namespace

// Runs canny edge supression based on `level.dx` and `level.dy`, storing the result in `level.map`
// Conditions:
// - `none` if:
//   - `magnitude <= low`
//   - Pixel is on the border
//   - Isn't a local maximum
// - `weak` if local maximum and `low < magnitude <= high`
// - `strong` if local maximum and `magnitude > high`
void suppress(sycl::queue& queue, const Level& level, float low, float high) {
    const int width = level.width;
    const int height = level.height;
    const auto size = std::size_t(width) * std::size_t(height);

    auto dx_src = std::vector<std::int16_t>(size);
    auto dy_src = std::vector<std::int16_t>(size);
    queue.memcpy(dx_src.data(), level.dx, size * sizeof(std::int16_t));
    queue.memcpy(dy_src.data(), level.dy, size * sizeof(std::int16_t));
    queue.wait();

    const auto dx = Plane<std::int16_t>{dx_src.data(), level.width, level.height};
    const auto dy = Plane<std::int16_t>{dy_src.data(), level.width, level.height};

    const auto magnitude_at = [=](int i) -> int {
        const int gx = dx[i];
        const int gy = dy[i];
        return std::lrint(std::sqrt(float(gx * gx + gy * gy)));
    };

    // The gradient direction at index i, rounded to the nearest of the four axes through it
    const auto direction_at = [=](int i) -> Direction {
        const int gx = dx[i];
        const int gy = dy[i];
        const float ax = std::abs(float(gx));
        const float ay = std::abs(float(gy));

        if (ay < tan22_5 * ax) {
            return {1, false};
        }
        if (ay > tan67_5 * ax) {
            return {width, false};
        }

        const bool same_sign = (gx < 0) == (gy < 0);
        return {same_sign ? width + 1 : width - 1, true};
    };

    // Whether the pixel at index i is a local maximum along its gradient. A tie with the pixel
    // before suppresses it. A tie with the pixel after keeps it.
    const auto is_maximum = [=](int i, int m) -> bool {
        const Direction direction = direction_at(i);
        const int magnitude_before = magnitude_at(i - direction.step);
        const int magnitude_after = magnitude_at(i + direction.step);

        return m > magnitude_before
            && (direction.diagonal ? m > magnitude_after : m >= magnitude_after);
    };

    std::vector<Mark> map(size, Mark::none);

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            const int i = y * width + x;
            const int magnitude = magnitude_at(i);

            if (magnitude <= low || !is_maximum(i, magnitude)) {
                continue;
            }

            map[i] = magnitude > high ? Mark::strong : Mark::weak;
        }
    }

    // Wait, as map is freed on return
    queue.memcpy(level.map, map.data(), size * sizeof(Mark)).wait();
}

} // namespace portable
} // namespace cctag
