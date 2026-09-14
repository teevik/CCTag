/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "backends/cpu/backend.hpp"

#include <stdexcept>

namespace cctag::portable::cpu {

void Backend::edge_points(Buffers& level) {
    const int width = static_cast<int>(level.width);
    const int height = static_cast<int>(level.height);

#pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const auto* edges = level.edges[y];
        std::uint32_t count = 0;
        for (int x = 0; x < width; ++x) {
            count += edges[x] == 255;
        }
        level.row_offsets[y] = count;
    }

    // Scan in row order so each row owns a fixed range of canonical indices
    std::uint32_t n = 0;
    for (auto& offset : level.row_offsets) {
        const auto count = offset;
        if (count > kMaxEdgePoints - n) {
            throw std::length_error("edge_points: too many edge points in one pyramid level");
        }
        offset = n;
        n += count;
    }
    level.xy.resize(2 * n);
    level.gradients.resize(2 * n);
    level.n = n;

#pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const auto* edges = level.edges[y];
        const auto* dx = level.dx[y];
        const auto* dy = level.dy[y];
        auto* edge_map = level.edge_map[y];
        auto index = level.row_offsets[y];
        for (int x = 0; x < width; ++x) {
            if (edges[x] == 255) {
                level.xy[2 * index] = x;
                level.xy[2 * index + 1] = y;
                level.gradients[2 * index] = static_cast<float>(dx[x]);
                level.gradients[2 * index + 1] = static_cast<float>(dy[x]);
                edge_map[x] = static_cast<std::int32_t>(index++);
            } else {
                // Write every cell, including borders and edges removed since the last frame
                edge_map[x] = -1;
            }
        }
    }
}

EdgePointsHost Backend::host_edge_points(Buffers& level) {
    // Return read-only views of the interleaved coordinates and gradients
    return EdgePointsHost{level.n, level.xy, level.gradients};
}

} // namespace cctag::portable::cpu

#ifdef CCTAG_TEST
#include <boost/ut.hpp>

#include <array>

namespace cctag::portable::tests::edge_points_stage {

using namespace boost::ut;

inline suite<"edge_points_stage"> edge_points_stage_suite = [] {
    "edge points compact in canonical order and rewrite every edge map cell"_test = [] {
        cpu::Buffers level;
        level.ensure(5, 4);
        // Pitched planes, an empty row, border edges and a non-edge value other than zero
        const cv::Rect region(2, 0, 5, 4);
        level.edges = cv::Mat1b(4, 9, std::uint8_t{0})(region);
        level.dx = cv::Mat1s(4, 9, std::int16_t{0})(region);
        level.dy = cv::Mat1s(4, 9, std::int16_t{0})(region);
        cv::Mat1i edge_map(4, 9, -7);
        level.edge_map = edge_map(region);
        level.edges(0, 0) = 255;
        level.edges(0, 4) = 255;
        level.edges(2, 1) = 255;
        level.edges(2, 3) = 1;
        level.edges(3, 4) = 255;
        level.dx(0, 0) = -32768;
        level.dy(0, 0) = 32767;
        level.dx(0, 4) = 17;
        level.dy(0, 4) = -9;
        level.dx(2, 1) = -3;
        level.dy(2, 1) = 5;
        const std::array<std::int32_t, 8> xy = {0, 0, 4, 0, 1, 2, 4, 3};
        const std::array<float, 8> gradients = {-32768.f, 32767.f, 17.f, -9.f, -3.f, 5.f, 0.f, 0.f};
        const cv::Mat1i expected =
            (cv::Mat1i(4, 5) << 0,
             -1,
             -1,
             -1,
             1,
             -1,
             -1,
             -1,
             -1,
             -1,
             -1,
             2,
             -1,
             -1,
             -1,
             -1,
             -1,
             -1,
             -1,
             3);
        cpu::Backend::edge_points(level);
        const EdgePointsHost points = cpu::Backend::host_edge_points(level);
        expect(eq(points.n, 4u));
        expect(eq(points.xy.size(), xy.size())) << fatal;
        expect(eq(points.gradients.size(), gradients.size())) << fatal;
        for (std::size_t i = 0; i < xy.size(); ++i) {
            expect(eq(points.xy[i], xy[i])) << "coordinate" << i;
            expect(eq(points.gradients[i], gradients[i])) << "gradient" << i;
        }
        expect(eq(cv::countNonZero(level.edge_map != expected), 0));
        expect(eq(cv::countNonZero(edge_map.colRange(0, 2) != -7), 0));
        expect(eq(cv::countNonZero(edge_map.colRange(7, 9) != -7), 0));

        // Reusing the collection must remove old points, including those on the border
        const auto* xy_storage = points.xy.data();
        const auto* gradient_storage = points.gradients.data();
        level.edges.setTo(0);
        cpu::Backend::edge_points(level);
        const EdgePointsHost empty = cpu::Backend::host_edge_points(level);
        expect(eq(empty.n, 0u));
        expect(empty.xy.empty());
        expect(empty.gradients.empty());
        expect(eq(cv::countNonZero(level.edge_map != -1), 0));

        // A denser frame must also reuse the storage reserved for this level
        level.edges.setTo(255);
        cpu::Backend::edge_points(level);
        const EdgePointsHost dense = cpu::Backend::host_edge_points(level);
        expect(eq(dense.n, 20u));
        expect(eq(dense.xy.data(), xy_storage));
        expect(eq(dense.gradients.data(), gradient_storage));
        expect(eq(level.edge_map(3, 4), 19));

        level.ensure(1, 1);
        level.edges(0, 0) = 255;
        level.dx(0, 0) = -1;
        level.dy(0, 0) = 2;
        cpu::Backend::edge_points(level);
        expect(eq(level.n, 1u));
        expect(level.xy == std::vector<std::int32_t>{0, 0});
        expect(level.gradients == std::vector<float>{-1.f, 2.f});
        expect(eq(level.edge_map(0, 0), 0));
    };

    "edge points reject more than two to the twenty fourth points"_test = [] {
        cpu::Buffers level;
        level.ensure(4097, 4096);
        level.edges.setTo(255);
        expect(throws<std::length_error>([&] { cpu::Backend::edge_points(level); }));

        // Exactly 2^24 points still fit, including the final canonical index
        level.edges.col(4096).setTo(0);
        level.dx.setTo(0);
        level.dy.setTo(0);
        cpu::Backend::edge_points(level);
        expect(eq(level.n, 16777216u));
        expect(eq(level.edge_map(4095, 4095), 16777215));
        expect(eq(level.edge_map(4095, 4096), -1));

        // The limit counts points, not pixels; a sparse frame at the same size is valid
        level.edges.setTo(0);
        level.edges(4095, 4096) = 255;
        cpu::Backend::edge_points(level);
        expect(eq(level.n, 1u));
        expect(level.xy == std::vector<std::int32_t>{4096, 4095});
        expect(eq(level.edge_map(4095, 4096), 0));
        expect(eq(level.edge_map(0, 0), -1));
    };
};

} // namespace cctag::portable::tests::edge_points_stage
#endif // CCTAG_TEST
