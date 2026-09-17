/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_KERNELS_VOTE_HPP
#define CCTAG_PORTABLE_KERNELS_VOTE_HPP

#include "kernels/plane.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

namespace cctag::portable::kernels {

inline void
descent_step(float dx, float dy, int& x, int& y, float& error, int& step_x, int& step_y) {
    const float slope = std::abs(dy / dx);
    step_x = (dx > 0) - (dx < 0);
    step_y = (dy > 0) - (dy < 0);
    error += slope;
    x += step_x;
    if (error >= 0.5f) {
        y += step_y;
        error -= 1;
    }
}

/// Follows the gradient from one edge point, skipping the first pixel
inline std::int32_t descent_from_gradient_at(
    int x,
    int y,
    int direction,
    Plane<const std::int32_t> edge_map,
    float point_dx,
    float point_dy,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::size_t max_steps,
    int gradient_threshold
) {
    float error = 0.f;
    float dx = direction * point_dx;
    float dy = direction * point_dy;
    const float reference_dx = dx;
    const float reference_dy = dy;
    const bool vertical = std::abs(dy) > std::abs(dx);
    int step_x = 0;
    int step_y = 0;
    const auto step = [&] {
        if (vertical) {
            descent_step(dy, dx, y, x, error, step_y, step_x);
        } else {
            descent_step(dx, dy, x, y, error, step_x, step_y);
        }
    };
    const auto inside = [&](int px, int py) {
        return px >= 0 && px < static_cast<int>(input_width) && py >= 0
            && py < static_cast<int>(input_height);
    };
    const auto point_at = [&](int px, int py) -> std::int32_t {
        if (static_cast<std::uint32_t>(px) >= edge_map.width
            || static_cast<std::uint32_t>(py) >= edge_map.height) {
            return -1;
        }
        return edge_map.row(py)[px];
    };

    step();
    if (dx * dx + dy * dy > gradient_threshold) {
        const float dx2 = point_dx;
        const float dy2 = point_dy;
        const float dot = dx2 * reference_dx + dy2 * reference_dy;
        direction = (dot > 0) - (dot < 0);
        dx = direction * dx2;
        dy = direction * dy2;
    }
    step();
    if (!inside(x, y)) {
        return -1;
    }
    if (const auto point = point_at(x, y); point != -1) {
        return point;
    }

    for (std::size_t n = 2; n <= max_steps; ++n) {
        step();
        if (!inside(x, y)) {
            return -1;
        }
        if (const auto point = point_at(x, y); point != -1) {
            return point;
        }
        const int adjacent_x = vertical ? x : x - step_x;
        const int adjacent_y = vertical ? y - step_y : y;
        if (!inside(adjacent_x, adjacent_y)) {
            return -1;
        }
        if (const auto point = point_at(adjacent_x, adjacent_y); point != -1) {
            return point;
        }
    }
    return -1;
}

// Original plane interface retained as the experiment's unchanged voting control.
inline std::int32_t descent_at(int x, int y, int direction,
    Plane<const std::int32_t> map, Plane<const std::int16_t> dx,
    Plane<const std::int16_t> dy, std::uint32_t iw, std::uint32_t ih,
    std::size_t steps, int threshold) {
    return descent_from_gradient_at(x,y,direction,map,dx.row(y)[x],dy.row(y)[x],iw,ih,steps,threshold);
}

/// One edge point's vote, or -1 when its field-line walk found no seed
struct CastVote {
    std::int32_t point = -1;
    float distance = 0.f;
};

/// Walks alternating before/after links, storing sub-segment lengths in this point's scratch
inline CastVote cast_vote_at(
    int point,
    std::span<const std::int32_t> xy,
    std::span<const float> gradients,
    std::span<const std::int32_t> links,
    std::size_t crowns,
    float ratio,
    std::span<float> distances
) {
    const auto opposed = [&](int a, int b) {
        const float dot =
            gradients[2 * a] * gradients[2 * b] + gradients[2 * a + 1] * gradients[2 * b + 1];
        return -dot >= 0.f;
    };
    const auto distance = [&](int a, int b) {
        const double dx = xy[2 * b] - xy[2 * a];
        const double dy = xy[2 * b + 1] - xy[2 * a + 1];
        return std::sqrt(static_cast<float>(dx * dx) + static_cast<float>(dy * dy));
    };
    std::size_t count = 0;
    const auto ratios_match = [&] {
        int valid = 1;
        for (std::size_t i = 0; i < count; ++i) {
            for (std::size_t j = i + 1; j < count; ++j) {
                valid = (distances[i] <= distances[j] * ratio)
                    && (distances[j] <= distances[i] * ratio) && valid;
            }
        }
        return valid != 0;
    };

    CastVote vote;
    int current = links[2 * point];
    if (current == -1 || !opposed(point, current)) {
        return vote;
    }
    distances[count++] = distance(point, current);
    vote.distance += distances[count - 1];
    for (std::size_t crown = 1; crown < crowns; ++crown) {
        vote.point = -1;
        int target = links[2 * current + 1];
        if (target == -1 || !opposed(target, current)) {
            break;
        }
        distances[count++] = distance(target, current);
        vote.distance += distances[count - 1];
        if (!ratios_match()) {
            break;
        }
        current = target;
        target = links[2 * current];
        if (target == -1 || !opposed(target, current)) {
            break;
        }
        distances[count++] = distance(target, current);
        vote.distance += distances[count - 1];
        if (!ratios_match()) {
            break;
        }
        current = target;
        vote.point = current;
    }
    return vote;
}

/// Computes a point's flow length as a running mean in canonical voter order
inline float
gather_flow_length_at(std::span<const std::int32_t> voters, std::span<const float> vote_distance) {
    float flow_length = 0.f;
    for (std::size_t k = 0; k < voters.size(); ++k) {
        flow_length = (flow_length * k + vote_distance[voters[k]]) / (k + 1);
    }
    return flow_length;
}

} // namespace cctag::portable::kernels

#endif
