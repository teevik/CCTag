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

/// Whether the gradients of a and b point at most a right angle apart in opposite directions,
/// the test cosDiffTheta >= _angleVoting of vote() in Vote.cpp with _angleVoting 0. The gradients
/// of stage 02 stay below 187, so the float product of the CPU is exact and has the same sign.
bool opposed(const Edgel& a, const Edgel& b) {
    return std::int64_t(a.dx) * b.dx + std::int64_t(a.dy) * b.dy <= 0;
}

/// The distance between the pixels of a and b, as distancePoints2D
float distance(const Edgel& a, const Edgel& b) {
    const int x = b.x - a.x;
    const int y = b.y - a.y;
    return sycl::sqrt(float(x * x) + float(y * y));
}

/// The edge point the edge point p votes for, or -1, and the length of the field line to it, as
/// the voting loop of vote() in Vote.cpp: from p one step before, then for every further crown
/// one step after and one step before, every step between opposed gradients and all step lengths
/// within a ratio of each other. Any failed step means no vote.
std::int32_t walk(
    const Edgel* edgels,
    std::uint32_t p,
    std::uint32_t n_crowns,
    float ratio_voting,
    float& total
) {
    float lengths[2 * max_crowns - 1];
    int n = 0;
    total = 0;
    const auto step = [&](std::int32_t current, std::int32_t target) {
        if (target < 0 || !opposed(edgels[target], edgels[current])) {
            return false;
        }
        const float length = distance(edgels[target], edgels[current]);
        // The pairs of earlier lengths passed already, so only the new pairs can fail
        for (int i = 0; i < n; ++i) {
            if (!(lengths[i] <= length * ratio_voting && length <= lengths[i] * ratio_voting)) {
                return false;
            }
        }
        lengths[n++] = length;
        total += length;
        return true;
    };

    std::int32_t current = edgels[p].before;
    if (!step(p, current)) {
        return -1;
    }
    for (std::uint32_t i = 1; i < n_crowns; ++i) {
        const std::int32_t after = edgels[current].after;
        if (!step(current, after)) {
            return -1;
        }
        current = after;
        const std::int32_t before = edgels[current].before;
        if (!step(current, before)) {
            return -1;
        }
        current = before;
    }
    // With a single crown the loop never chooses a point
    return n_crowns > 1 ? current : -1;
}

} // namespace

void vote(sycl::queue& queue, const Level& level, std::uint32_t n_crowns, float ratio_voting) {
    // The count is only known on the device, as in link
    Edgel* edgels = level.edgels;
    const std::uint32_t* count = level.edgel_count;
    queue.parallel_for(sycl::range<1>(level.capacity), [=](sycl::id<1> id) {
        const std::uint32_t i = id[0];
        if (i >= *count) {
            return;
        }
        float total;
        const std::int32_t chosen = walk(edgels, i, n_crowns, ratio_voting, total);
        edgels[i].chosen = chosen;
        if (chosen < 0) {
            return;
        }
        // Only the vote fields of other edge points are written, and only atomically
        sycl::atomic_ref<
            std::uint32_t,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>
            votes(edgels[chosen].votes);
        sycl::atomic_ref<
            float,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space>
            flow(edgels[chosen].flow);
        votes.fetch_add(1);
        flow.fetch_add(total);
    });
}

} // namespace portable
} // namespace cctag
