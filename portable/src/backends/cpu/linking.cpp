/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "kernels/linking.hpp"

#include "backends/cpu/backend.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace cctag::portable::cpu {

namespace {

/// Applies a direction's marks to the segment as it existed when that direction stopped
void mark_segment(
    Buffers& level,
    const kernels::SegmentSlot& slot,
    int begin,
    kernels::LinkStop stop,
    std::size_t window_size
) {
    int end = slot.end;
    if (stop == kernels::LinkStop::max_length || stop == kernels::LinkStop::convexity_lost) {
        if (static_cast<std::size_t>(end - begin) <= window_size) {
            return;
        }
        end -= static_cast<int>(window_size);
    } else if (stop != kernels::LinkStop::edge_not_found) {
        return;
    }
    for (int i = begin; i < end; ++i) {
        level.processed_in[slot.points[i]] = 1;
    }
}

} // namespace

void Backend::linking(Buffers& level, const Parameters& params) {
    if (params._windowSizeOnInnerEllipticSegment == 0) {
        throw std::invalid_argument("linking: the angle window must be positive");
    }
    const auto maximum = std::max(std::size_t{level.height / 2}, params._maximumNbSeeds);
    const int count = static_cast<int>(std::min(level.seed_order.size(), maximum));
    level.arena.resize(count);
    level.accepted_slots.clear();
    level.accepted_slots.reserve(count);
    level.processed_in.assign(level.n, 0);
    level.link_seeds.clear();
    level.segment_offsets.assign(1, 0);
    level.segment_values.clear();

    const kernels::Plane<const std::int32_t> edge_map =
        {level.edge_map[0], level.width, level.height, level.edge_map.step1()};
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < count; ++i) {
        kernels::link_seed_at(
            level.seed_order[i],
            edge_map,
            level.xy,
            level.gradients,
            level.voters_offsets,
            params._windowSizeOnInnerEllipticSegment,
            params._averageVoteMin,
            level.arena[i]
        );
    }

    // Resolve ownership in seed_order, including walks too short for loop two
    for (int i = 0; i < count; ++i) {
        const auto seed = level.seed_order[i];
        if (level.processed_in[seed]) {
            continue;
        }
        const auto& slot = level.arena[i];
        level.processed_in[seed] = 1;
        mark_segment(
            level,
            slot,
            kernels::kMaxLinkLength,
            slot.stop_after,
            params._windowSizeOnInnerEllipticSegment
        );
        // Both directions leave the last window at the end of the combined segment unmarked
        mark_segment(
            level,
            slot,
            slot.begin,
            slot.stop_before,
            params._windowSizeOnInnerEllipticSegment
        );
        level.accepted_slots.push_back(i);
    }

    // The slot index retains acceptance order while snapshot rows use ascending seeds
    std::sort(level.accepted_slots.begin(), level.accepted_slots.end(), [&](auto a, auto b) {
        return level.seed_order[a] < level.seed_order[b];
    });
    for (const auto index : level.accepted_slots) {
        const auto& slot = level.arena[index];
        if (level.segment_values.size() > static_cast<std::size_t>(
                std::numeric_limits<std::int32_t>::max() - (slot.end - slot.begin)
            )) {
            throw std::length_error("linking: too many segment points");
        }
        level.link_seeds.push_back(level.seed_order[index]);
        level.segment_values.insert(
            level.segment_values.end(),
            slot.points.begin() + slot.begin,
            slot.points.begin() + slot.end
        );
        level.segment_offsets.push_back(static_cast<std::int32_t>(level.segment_values.size()));
    }

    const int candidates = static_cast<int>(level.link_seeds.size());
    level.child_counts.resize(candidates);
    level.avg_vote.resize(candidates);
    level.children_offsets.assign(candidates + 1, 0);
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < candidates; ++i) {
        const auto segment = std::span<const std::int32_t>(level.segment_values)
                                 .subspan(
                                     level.segment_offsets[i],
                                     level.segment_offsets[i + 1] - level.segment_offsets[i]
                                 );
        const auto minimum = kernels::child_vote_min(segment, level.voters_offsets);
        std::int64_t received = 0;
        int voted = 0;
        std::int32_t children = 0;
        for (const auto point : segment) {
            const auto votes = level.voters_offsets[point + 1] - level.voters_offsets[point];
            received += votes;
            if (votes > 0) {
                ++voted;
            }
            if (votes >= minimum) {
                children += votes;
            }
        }
        // Widen the square so large voter sets cannot overflow a signed int
        level.avg_vote[i] = static_cast<float>(received * received) / static_cast<float>(voted);
        level.child_counts[i] = children;
    }
    for (int i = 0; i < candidates; ++i) {
        if (level.child_counts[i]
            > std::numeric_limits<std::int32_t>::max() - level.children_offsets[i]) {
            throw std::length_error("linking: too many children");
        }
        level.children_offsets[i + 1] = level.children_offsets[i] + level.child_counts[i];
    }
    level.children_values.resize(level.children_offsets.back());
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < candidates; ++i) {
        kernels::gather_children_at(
            std::span<const std::int32_t>(level.segment_values)
                .subspan(
                    level.segment_offsets[i],
                    level.segment_offsets[i + 1] - level.segment_offsets[i]
                ),
            level.voters_offsets,
            level.voters_values,
            std::span<std::int32_t>(level.children_values)
                .subspan(level.children_offsets[i], level.child_counts[i])
        );
    }

    level.loop_one_order.resize(candidates);
    std::iota(level.loop_one_order.begin(), level.loop_one_order.end(), 0);
    std::sort(level.loop_one_order.begin(), level.loop_one_order.end(), [&](auto a, auto b) {
        return level.avg_vote[a] > level.avg_vote[b]
            || (level.avg_vote[a] == level.avg_vote[b]
                && level.accepted_slots[a] > level.accepted_slots[b]);
    });
}

LinkingHost Backend::host_linking(Buffers& level) {
    return LinkingHost{
        static_cast<std::uint32_t>(level.link_seeds.size()),
        level.link_seeds,
        level.segment_offsets,
        level.segment_values,
        level.child_counts,
        level.avg_vote
    };
}

} // namespace cctag::portable::cpu

#ifdef CCTAG_TEST
#include <boost/ut.hpp>

#include <array>

namespace cctag::portable::tests::linking_stage {

using namespace boost::ut;

inline suite<"linking_stage"> linking_stage_suite = [] {
    "loop one orders vote scores descending and ties by reverse acceptance order"_test = [] {
        cpu::Buffers level;
        level.ensure(19, 1);
        level.edges.setTo(0);
        level.dx.setTo(1);
        level.dy.setTo(0);
        for (int x = 0; x < 19; x += 2) {
            level.edges(0, x) = 255;
        }
        cpu::Backend::edge_points(level);
        // Isolated seeds receive 2, 1, 2, 3 and 1 votes; the other points are only voters
        level.voters_offsets = {0, 2, 3, 5, 8, 9, 9, 9, 9, 9, 9};
        level.voters_values = {1, 2, 0, 3, 4, 5, 6, 7, 8};
        level.seed_order = {3, 0, 2, 1, 4};
        const cctag::Parameters params(3);
        cpu::Backend::linking(level, params);
        const LinkingHost linking = cpu::Backend::host_linking(level);
        expect(std::ranges::equal(linking.seeds, std::array{0, 1, 2, 3, 4}));
        expect(std::ranges::equal(linking.segment_values, std::array{0, 1, 2, 3, 4}));
        expect(std::ranges::equal(linking.avg_vote, std::array{4.f, 1.f, 4.f, 9.f, 1.f}));
        expect(level.loop_one_order == std::vector<std::int32_t>{3, 2, 0, 4, 1});
        expect(level.children_offsets == std::vector<std::int32_t>{0, 2, 3, 5, 8, 9});
        expect(level.children_values == level.voters_values);

        // A reused level must expose empty CSRs after a frame with no seeds
        level.seed_order.clear();
        cpu::Backend::linking(level, params);
        const LinkingHost empty = cpu::Backend::host_linking(level);
        expect(eq(empty.c, 0u));
        expect(empty.seeds.empty());
        expect(std::ranges::equal(empty.segment_offsets, std::array{0}));
        expect(empty.segment_values.empty());
        expect(empty.child_counts.empty());
        expect(empty.avg_vote.empty());
        expect(level.children_offsets == std::vector<std::int32_t>{0});
        expect(level.children_values.empty());
        expect(level.loop_one_order.empty());
    };

    "linking leaves the last angle window available when a walk reaches its length limit"_test =
        [] {
        cpu::Buffers level;
        level.ensure(301, 1);
        level.edges.setTo(255);
        level.dx.setTo(0);
        level.dy.setTo(1);
        cpu::Backend::edge_points(level);
        level.voters_offsets.resize(302);
        level.voters_values.resize(301);
        std::iota(level.voters_offsets.begin(), level.voters_offsets.end(), 0);
        std::iota(level.voters_values.begin(), level.voters_values.end(), 0);
        level.seed_order = {150, 230, 240};
        cctag::Parameters params(3);
        cpu::Backend::linking(level, params);
        const LinkingHost linking = cpu::Backend::host_linking(level);
        // The first walk is [50, 250], with [231, 250] left unmarked
        expect(std::ranges::equal(linking.seeds, std::array{150, 240}));
        expect(std::ranges::equal(linking.segment_offsets, std::array{0, 201, 362}));
        std::vector<std::int32_t> expected(201);
        std::iota(expected.begin(), expected.end(), 50);
        for (int i = 140; i <= 300; ++i) {
            expected.push_back(i);
        }
        expect(std::ranges::equal(linking.segment_values, expected));

        // Windows beyond a direction's length use the partial-window convexity test
        params._windowSizeOnInnerEllipticSegment = 101;
        cpu::Backend::linking(level, params);
        expect(std::ranges::equal(cpu::Backend::host_linking(level).seeds, std::array{150, 230}));
        params._windowSizeOnInnerEllipticSegment = 0;
        expect(throws<std::invalid_argument>([&] { cpu::Backend::linking(level, params); }));
    };

    "linking gathers children at the vote maximum divided by fourteen"_test = [] {
        cpu::Buffers level;
        level.ensure(32, 1);
        level.edges.setTo(255);
        level.dx.setTo(0);
        level.dy.setTo(1);
        cpu::Backend::edge_points(level);
        level.voters_offsets.assign(33, 31);
        level.voters_offsets[0] = 0;
        level.voters_offsets[1] = 28;
        level.voters_offsets[2] = 29;
        level.voters_values.resize(31);
        std::iota(level.voters_values.begin(), level.voters_values.end(), 1);
        level.seed_order = {0, 2, 1};
        cctag::Parameters params(3);
        cpu::Backend::linking(level, params);
        const LinkingHost linking = cpu::Backend::host_linking(level);
        expect(std::ranges::equal(linking.seeds, std::array{0}));
        expect(std::ranges::equal(linking.child_counts, std::array{30}));
        expect(std::ranges::equal(linking.avg_vote, std::array{961.f / 3.f}));
        // Keep the row with two votes, exclude the row with only one
        std::vector<std::int32_t> children(28);
        std::iota(children.begin(), children.end(), 1);
        children.push_back(30);
        children.push_back(31);
        expect(level.children_values == children);

        // A vote threshold stops growth without claiming the other seeds
        params._averageVoteMin = 29.f;
        cpu::Backend::linking(level, params);
        expect(std::ranges::equal(cpu::Backend::host_linking(level).seeds, std::array{0, 1, 2}));
        expect(
            std::ranges::equal(
                cpu::Backend::host_linking(level).segment_values,
                std::array{0, 1, 2}
            )
        );
    };
};

} // namespace cctag::portable::tests::linking_stage
#endif
