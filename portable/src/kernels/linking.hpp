/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_KERNELS_LINKING_HPP
#define CCTAG_PORTABLE_KERNELS_LINKING_HPP

#include "kernels/plane.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>

namespace cctag::portable::kernels {

inline constexpr int kMaxLinkLength = 100;
inline constexpr int kSegmentSlotSize = 2 * kMaxLinkLength + 1;

enum class LinkStop {
    average_vote,
    edge_not_found,
    convexity_lost,
    max_length,
};

/// One seed's walk grows from the centre, leaving room for both directions
struct SegmentSlot {
    std::array<std::int32_t, kSegmentSlotSize> points;
    int begin = kMaxLinkLength;
    int end = kMaxLinkLength + 1;
    LinkStop stop_before = LinkStop::average_vote;
    LinkStop stop_after = LinkStop::average_vote;
};

/// Walks one direction, using only this seed's slot to test visited points
inline LinkStop link_direction(
    std::int32_t point,
    int dir,
    Plane<const std::int32_t> edge_map,
    std::span<const std::int32_t> xy,
    std::span<const float> gradients,
    std::span<const std::int32_t> voters_offsets,
    std::size_t window_size,
    float average_vote_min,
    SegmentSlot& slot
) {
    // A direction takes at most 100 steps, even when the requested window is larger
    std::array<float, kMaxLinkLength> angles;
    std::size_t first = 0;
    std::size_t count = 0;
    const std::size_t capacity = window_size < angles.size() ? window_size : angles.size();
    float average_vote = voters_offsets[point + 1] - voters_offsets[point];
    int length = 0;
    LinkStop stop = LinkStop::average_vote;
    while (length < kMaxLinkLength && average_vote >= average_vote_min) {
        constexpr float pi = std::numbers::pi_v<float>;
        const float angle = std::fmod(
            float(std::atan2(gradients[2 * point + 1], gradients[2 * point])) + 2.0f * pi,
            2.0f * pi
        );
        if (count == capacity) {
            first = (first + 1) % capacity;
            --count;
        }
        angles[(first + count) % capacity] = angle;
        ++count;

        const int shifting =
            static_cast<int>(std::round(((angle + pi / 4.0f) / (2.0f * pi)) * 8.0f)) - 1;
        constexpr int xoff[] = {1, 1, 0, -1, -1, -1, 0, 1};
        constexpr int yoff[] = {0, -1, -1, -1, 0, 1, 1, 1};
        stop = LinkStop::edge_not_found;
        bool found = false;
        for (int j = 0; j < 8; ++j) {
            const int offset = (dir == 1 ? 8 - shifting + j : shifting + j) % 8;
            const int x = xy[2 * point] + xoff[offset];
            const int y = xy[2 * point + 1] + dir * yoff[offset];
            if (x < 0 || y < 0 || x >= static_cast<int>(edge_map.width)
                || y >= static_cast<int>(edge_map.height)) {
                continue;
            }
            const auto next = edge_map.row(y)[x];
            if (next == -1) {
                continue;
            }
            bool visited = false;
            for (int i = slot.begin; i < slot.end; ++i) {
                if (slot.points[i] == next) {
                    visited = true;
                    break;
                }
            }
            if (visited) {
                continue;
            }

            const float s = dir * std::sin(angle - angles[first]);
            bool concave = s < 0.0f;
            if (count != window_size) {
                const float c = std::cos(angle - angles[first]);
                concave = ((s < -0.707f) && (c > 0.f)) || ((s < 0.f) && (c < 0.f));
            }
            if (concave) {
                stop = LinkStop::convexity_lost;
                break;
            }
            point = next;
            if (dir > 0) {
                slot.points[slot.end++] = point;
            } else {
                slot.points[--slot.begin] = point;
            }
            // The legacy uses the segment's size after appending the new point
            const auto size = static_cast<std::size_t>(slot.end - slot.begin);
            average_vote =
                (average_vote * size + (voters_offsets[point + 1] - voters_offsets[point]))
                / (size + 1.f);
            stop = LinkStop::average_vote;
            found = true;
            break;
        }
        ++length;
        if (!found) {
            break;
        }
    }
    return length == kMaxLinkLength ? LinkStop::max_length : stop;
}

/// Links one seed in both directions without reading or writing ownership marks
/// The angle window must be positive
inline void link_seed_at(
    std::int32_t seed,
    Plane<const std::int32_t> edge_map,
    std::span<const std::int32_t> xy,
    std::span<const float> gradients,
    std::span<const std::int32_t> voters_offsets,
    std::size_t window_size,
    float average_vote_min,
    SegmentSlot& slot
) {
    slot.begin = kMaxLinkLength;
    slot.end = kMaxLinkLength + 1;
    slot.points[kMaxLinkLength] = seed;
    slot.stop_after = link_direction(
        seed,
        1,
        edge_map,
        xy,
        gradients,
        voters_offsets,
        window_size,
        average_vote_min,
        slot
    );
    slot.stop_before = link_direction(
        seed,
        -1,
        edge_map,
        xy,
        gradients,
        voters_offsets,
        window_size,
        average_vote_min,
        slot
    );
}

/// Minimum received-vote count for a segment point's voters to become children
inline std::int32_t child_vote_min(
    std::span<const std::int32_t> segment,
    std::span<const std::int32_t> voters_offsets
) {
    std::int32_t vote_max = 1;
    for (const auto point : segment) {
        const auto count = voters_offsets[point + 1] - voters_offsets[point];
        if (count > vote_max) {
            vote_max = count;
        }
    }
    return vote_max / 14;
}

/// Gathers children in segment-walk order, preserving each point's voter order
inline void gather_children_at(
    std::span<const std::int32_t> segment,
    std::span<const std::int32_t> voters_offsets,
    std::span<const std::int32_t> voters_values,
    std::span<std::int32_t> children
) {
    const auto minimum = child_vote_min(segment, voters_offsets);
    std::size_t cursor = 0;
    for (const auto point : segment) {
        const auto begin = voters_offsets[point];
        const auto end = voters_offsets[point + 1];
        if (end - begin >= minimum) {
            for (auto i = begin; i < end; ++i) {
                children[cursor++] = voters_values[i];
            }
        }
    }
}

} // namespace cctag::portable::kernels

#endif
