/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "kernels/vote.hpp"

#include "backends/cpu/backend.hpp"

#include <algorithm>
#include <stdexcept>

namespace cctag::portable::cpu {

void Backend::vote(Buffers& level, const Parameters& params) {
    if (params._angleVoting != 0) {
        throw std::domain_error(
            "thrVotingAngle must be equal to 0 or edge points gradients have to be normalized."
        );
    }
    const int n = static_cast<int>(level.n);
    const auto max_points =
        std::min(static_cast<std::size_t>(level.width) * level.height, std::size_t{kMaxEdgePoints});
    // Each point owns up to two sub-segments per crown after the first one
    const auto max_segments = level.vote_segments.max_size() / std::max(max_points, std::size_t{1});
    if (params._nCrowns > max_segments / 2) {
        throw std::length_error("vote: too many crown sub-segments");
    }
    const std::size_t segments = params._nCrowns > 0 ? 2 * params._nCrowns - 1 : 1;
    level.vote_segments.reserve(max_points * segments);
    level.vote_segments.resize(static_cast<std::size_t>(n) * segments);
    level.links.resize(2 * n);
    level.voted_for.resize(n);
    level.vote_distance.resize(n);
    level.voters_offsets.assign(n + 1, 0);
    level.voter_cursors.resize(n);
    level.is_max.resize(n);
    level.flow_length.resize(n);
    level.seeds.clear();
    level.seed_order.clear();

    const kernels::Plane<const std::int32_t> edge_map =
        {level.edge_map[0], level.width, level.height, level.edge_map.step1()};
    const auto dx = level.dx_plane().as_const();
    const auto dy = level.dy_plane().as_const();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int side = 0; side < 2; ++side) {
            if (level.prototype_compact_vote) {
                level.links[2 * i + side] = kernels::descent_from_gradient_at(
                    level.xy[2*i], level.xy[2*i+1], 2*side-1, edge_map,
                    level.gradients[2*i], level.gradients[2*i+1],
                    level.input_width, level.input_height, params._distSearch,
                    params._thrGradientMagInVote);
                continue;
            }
            level.links[2 * i + side] = kernels::descent_at(
                level.xy[2 * i],
                level.xy[2 * i + 1],
                2 * side - 1,
                edge_map,
                dx,
                dy,
                level.input_width,
                level.input_height,
                params._distSearch,
                params._thrGradientMagInVote
            );
        }
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const auto vote = kernels::cast_vote_at(
            i,
            level.xy,
            level.gradients,
            level.links,
            params._nCrowns,
            params._ratioVoting,
            std::span<float>(level.vote_segments)
                .subspan(static_cast<std::size_t>(i) * segments, segments)
        );
        level.voted_for[i] = vote.point;
        level.vote_distance[i] = vote.distance;
    }

    // Count, scan and scatter in canonical index order so each CSR row is ascending
    for (const auto point : level.voted_for) {
        if (point != -1) {
            ++level.voters_offsets[point + 1];
        }
    }
    for (int i = 0; i < n; ++i) {
        level.voters_offsets[i + 1] += level.voters_offsets[i];
        level.voter_cursors[i] = level.voters_offsets[i];
    }
    level.voters_values.resize(level.voters_offsets[n]);
    for (int i = 0; i < n; ++i) {
        const auto point = level.voted_for[i];
        if (point != -1) {
            level.voters_values[level.voter_cursors[point]++] = i;
        }
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const auto begin = level.voters_offsets[i];
        const auto count = level.voters_offsets[i + 1] - begin;
        level.flow_length[i] = kernels::gather_flow_length_at(
            std::span<const std::int32_t>(level.voters_values).subspan(begin, count),
            level.vote_distance
        );
        level.is_max[i] =
            count > 0 && static_cast<std::size_t>(count) >= params._minVotesToSelectCandidate
            ? count
            : -1;
    }
    for (int i = 0; i < n; ++i) {
        if (level.is_max[i] != -1) {
            level.seeds.push_back(i);
        }
    }
    level.seed_order.assign(level.seeds.begin(), level.seeds.end());
    std::sort(level.seed_order.begin(), level.seed_order.end(), [&](auto a, auto b) {
        return level.is_max[a] > level.is_max[b] || (level.is_max[a] == level.is_max[b] && a < b);
    });
}

VoteHost Backend::host_vote(Buffers& level) {
    return VoteHost{
        level.links,
        level.voters_offsets,
        level.voters_values,
        level.is_max,
        level.flow_length,
        level.seeds,
        level.seed_order
    };
}

} // namespace cctag::portable::cpu

#ifdef CCTAG_TEST
#include <boost/ut.hpp>

#include <array>

namespace cctag::portable::tests::vote_stage {

using namespace boost::ut;

inline suite<"vote_stage"> vote_stage_suite = [] {
    "vote gathers canonical voters and orders seeds by count then canonical index"_test = [] {
        cpu::Buffers level;
        level.ensure(15, 6);
        level.edges.setTo(0);
        level.dx.setTo(0);
        level.dy.setTo(0);
        // Six equally spaced crossings, with alternating gradients along a diameter
        for (int i = 0; i < 6; ++i) {
            level.edges(0, 2 + 2 * i) = 255;
            level.dx(0, 2 + 2 * i) = i % 2 == 0 ? -1 : 1;
        }
        // A second voter reaches (4, 0) along a 3-4-5 triangle, then follows the diameter
        level.edges(4, 1) = 255;
        level.dx(4, 1) = -3;
        level.dy(4, 1) = 4;
        cpu::Backend::edge_points(level);
        cctag::Parameters params(3);
        params._minVotesToSelectCandidate = 1;
        params._ratioVoting = 3.f;
        cpu::Backend::vote(level, params);
        const VoteHost vote = cpu::Backend::host_vote(level);
        expect(
            std::ranges::equal(vote.links, std::array{1, -1, 0, 2, 3, 1, 2, 4, 5, 3, 4, -1, 1, -1})
        );
        expect(std::ranges::equal(vote.voters_offsets, std::array{0, 1, 1, 1, 1, 1, 3, 3}));
        expect(std::ranges::equal(vote.voters_values, std::array{5, 0, 6}));
        expect(std::ranges::equal(vote.is_max, std::array{1, -1, -1, -1, -1, 2, -1}));
        // One flow has length 10; the other seed averages lengths 10 and 13
        expect(
            std::ranges::equal(vote.flow_length, std::array{10.f, 0.f, 0.f, 0.f, 0.f, 11.5f, 0.f})
        );
        expect(std::ranges::equal(vote.seeds, std::array{0, 5}));
        expect(std::ranges::equal(vote.seed_order, std::array{5, 0}));
        const auto* links_storage = vote.links.data();
        const auto* voters_storage = vote.voters_values.data();

        params._minVotesToSelectCandidate = 2;
        cpu::Backend::vote(level, params);
        expect(std::ranges::equal(cpu::Backend::host_vote(level).seeds, std::array{5}));

        // The longer sub-segment fails the ratio check, leaving a tie between the two seeds
        params._minVotesToSelectCandidate = 0;
        params._ratioVoting = 2.f;
        cpu::Backend::vote(level, params);
        expect(std::ranges::equal(cpu::Backend::host_vote(level).seed_order, std::array{0, 5}));
        expect(std::ranges::equal(cpu::Backend::host_vote(level).voters_values, std::array{5, 0}));

        level.edges.setTo(0);
        cpu::Backend::edge_points(level);
        cpu::Backend::vote(level, params);
        const VoteHost empty = cpu::Backend::host_vote(level);
        expect(empty.links.empty());
        expect(std::ranges::equal(empty.voters_offsets, std::array{0}));
        expect(empty.voters_values.empty());
        expect(empty.is_max.empty());
        expect(empty.flow_length.empty());
        expect(empty.seeds.empty());
        expect(empty.seed_order.empty());
        expect(eq(empty.links.data(), links_storage));
        expect(eq(empty.voters_values.data(), voters_storage));

        params._angleVoting = 1.f;
        expect(throws<std::domain_error>([&] { cpu::Backend::vote(level, params); }));
    };

    "vote requires a complete four crown walk"_test = [] {
        cpu::Buffers level;
        level.ensure(19, 1);
        level.edges.setTo(0);
        level.dx.setTo(0);
        level.dy.setTo(0);
        for (int i = 0; i < 8; ++i) {
            level.edges(0, 2 + 2 * i) = 255;
            level.dx(0, 2 + 2 * i) = i % 2 == 0 ? -1 : 1;
        }
        cpu::Backend::edge_points(level);
        cctag::Parameters params(4);
        params._minVotesToSelectCandidate = 1;
        cpu::Backend::vote(level, params);
        const VoteHost vote = cpu::Backend::host_vote(level);
        // Only the two extremities cross all seven sub-segments; inner walks stop early
        expect(std::ranges::equal(vote.seeds, std::array{0, 7}));
        expect(std::ranges::equal(vote.voters_values, std::array{7, 0}));
        expect(
            std::ranges::equal(
                vote.flow_length,
                std::array{14.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 14.f}
            )
        );
    };

    "vote descent checks the previous pixel past a coarse level border"_test = [] {
        for (const bool vertical : {false, true}) {
            cpu::Buffers level;
            level.ensure(vertical ? 4 : 5, vertical ? 5 : 4);
            level.dx.setTo(0);
            level.dy.setTo(0);
            level.edges.setTo(0);
            const int x = vertical ? 1 : 2;
            const int y = vertical ? 2 : 1;
            level.edges(y, x) = 255;
            level.edges(level.height - 1, level.width - 1) = 255;
            level.dx(y, x) = vertical ? 1 : 2;
            level.dy(y, x) = vertical ? 2 : 1;
            cpu::Backend::edge_points(level);
            const cctag::Parameters params(3);
            cpu::Backend::vote(level, params);
            expect(eq(cpu::Backend::host_vote(level).links[1], -1));

            // The third step passes the border, then looks back to the corner edge
            level.input_width = 2 * level.width;
            level.input_height = 2 * level.height;
            cpu::Backend::vote(level, params);
            expect(eq(cpu::Backend::host_vote(level).links[1], 1));
        }
    };
};

} // namespace cctag::portable::tests::vote_stage
#endif // CCTAG_TEST
