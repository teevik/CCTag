/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "backends/cpu/backend.hpp"
#include "kernels/frontend.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace cctag::portable::cpu {

void Backend::edges(Buffers& level, const Parameters& params) {
    const int width = static_cast<int>(level.width);
    const int height = static_cast<int>(level.height);

#pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const auto* dx = level.dx[y];
        const auto* dy = level.dy[y];
        auto* magnitude = level.magnitude[y + 1] + 1;
        for (int x = 0; x < width; ++x) {
            magnitude[x] = kernels::magnitude_at(dx[x], dy[x]);
        }
    }

    // Classify every pixel independently before starting the flood. Neighbourhoods
    // replicate the magnitudes on the image border.
#pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const auto* dx = level.dx[y];
        const auto* dy = level.dy[y];
        auto* classes = level.nms_class[y + 1] + 1;
        for (int x = 0; x < width; ++x) {
            std::array<int, 9> neighbours{};
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    neighbours[(j + 1) * 3 + i + 1] = level.magnitude(
                        std::clamp(y + j, 0, height - 1) + 1,
                        std::clamp(x + i, 0, width - 1) + 1
                    );
                }
            }
            classes[x] = kernels::nms_at(
                dx[x],
                dy[x],
                neighbours,
                params._cannyThrLow * 256.f,
                params._cannyThrHigh * 256.f
            );
        }
    }

    auto& stack = level.hysteresis_stack;
    stack.clear();
    for (int y = 0; y < height; ++y) {
        auto* classes = level.nms_class[y + 1] + 1;
        for (int x = 0; x < width; ++x) {
            if (classes[x] == 2) {
                stack.push_back(classes + x);
            }
        }
    }

    // Keep every 8-connected component of NMS survivors containing a strong pixel.
    // The resulting set is order-independent, so a device may propagate to a fixed point instead.
    const auto stride = static_cast<std::ptrdiff_t>(level.nms_class.step1());
    const std::array<std::ptrdiff_t, 8> neighbours =
        {-stride - 1, -stride, -stride + 1, -1, 1, stride - 1, stride, stride + 1};
    while (!stack.empty()) {
        auto* pixel = stack.back();
        stack.pop_back();
        for (const auto offset : neighbours) {
            auto* neighbour = pixel + offset;
            if (*neighbour == 1) {
                *neighbour = 2;
                stack.push_back(neighbour);
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const auto* classes = level.nms_class[y + 1] + 1;
        auto* edges = level.edges[y];
        for (int x = 0; x < width; ++x) {
            edges[x] = classes[x] == 2 ? 2 : 0;
        }
    }

    // Hysteresis survivors are marked 2, the input value thinning expects
    const auto edges = level.edges_plane();
    const kernels::Plane<std::uint8_t>
        thinning{level.thinning[0], level.width, level.height, level.thinning.step1()};
#pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            thinning.row(y)[x] = kernels::thinning_at(edges.as_const(), x, y, true);
        }
    }
#pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            edges.row(y)[x] = kernels::thinning_at(thinning.as_const(), x, y, false);
        }
    }
}

} // namespace cctag::portable::cpu

#ifdef CCTAG_TEST
#include <boost/ut.hpp>

namespace cctag::portable::tests::edges_stage {

using namespace boost::ut;

inline suite<"edges_stage"> edges_stage_suite = [] {
    "diagonal weak edges connect in every direction regardless of column alignment"_test = [] {
        cpu::Buffers level;
        level.ensure(13, 9);
        for (int x = 4; x < 8; ++x) {
            for (int ox : {-1, 1}) {
                for (int oy : {-1, 1}) {
                    level.dx.setTo(0); level.dy.setTo(0);
                    level.dx(4, x) = 3;
                    level.dx(4 + oy, x + ox) = 11;
                    cpu::Backend::edges(level, cctag::Parameters(3));
                    expect(eq(int(level.edges(4, x)), 1));
                    expect(eq(int(level.edges(4 + oy, x + ox)), 1));
                    expect(eq(cv::countNonZero(level.edges), 2));
                }
            }
        }
    };
    "trailing columns propagate strength before thinning clears the border"_test = [] {
        cpu::Buffers level;
        for (int width = 5; width < 9; ++width) {
            level.ensure(width, 5);
            level.dx.setTo(0); level.dy.setTo(0);
            level.dy(2, width - 1) = 11;
            level.dx(1, width - 2) = 3;
            cpu::Backend::edges(level, cctag::Parameters(3));
            expect(eq(int(level.edges(1, width - 2)), 1));
            expect(eq(cv::countNonZero(level.edges), 1));
        }
    };
    "edges retain weak pixels connected to a strong pixel and clear reused scratch"_test = [] {
        cpu::Buffers level;
        level.ensure(4, 7);
        level.dx.setTo(0);
        level.dy.setTo(0);
        // A vertical line has horizontal gradients and zero neighbours on both sides.
        // Magnitude 2 is below the low threshold, and the disconnected lower weak pair is removed.
        // Thinning clears the image border, including the strong pixel at row zero.
        const std::array<std::int16_t, 7> gradient{11, 3, 10, 2, 3, 3, 0};
        for (int y = 0; y < 7; ++y) {
            level.dx(y, 1) = gradient[y];
        }
        cpu::Backend::edges(level, cctag::Parameters(3));
        const std::array<std::uint8_t, 7> expected{0, 1, 1, 0, 0, 0, 0};
        for (int y = 0; y < 7; ++y) {
            expect(eq(level.edges(y, 1), expected[y]));
        }
        level.dx.setTo(0);
        cpu::Backend::edges(level, cctag::Parameters(3));
        expect(eq(cv::countNonZero(level.edges), 0));
    };
};

} // namespace cctag::portable::tests::edges_stage
#endif // CCTAG_TEST
