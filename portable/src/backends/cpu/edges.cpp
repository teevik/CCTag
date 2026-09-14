/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "backends/cpu/backend.hpp"
#include "kernels/canny.hpp"
#include "kernels/thinning.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace cctag::portable::cpu {

void Backend::edges(Buffers& level, const Parameters& params) {
    const int width = static_cast<int>(level.width);
    const int height = static_cast<int>(level.height);
    const auto [low_threshold, high_threshold] =
        std::minmax(params._cannyThrLow, params._cannyThrHigh);
    const int low = cvFloor(low_threshold * 256.f);
    const int high = cvFloor(high_threshold * 256.f);

    // Keep the zero border around the magnitude plane for NMS at the image edges
#pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const auto* dx = level.dx[y];
        const auto* dy = level.dy[y];
        auto* magnitude = level.magnitude[y + 1] + 1;
        for (int x = 0; x < width; ++x) {
            magnitude[x] = kernels::magnitude_at(dx[x], dy[x]);
        }
    }

    // Classify every pixel independently before starting the flood
#pragma omp parallel for schedule(static)
    for (int y = 0; y < height; ++y) {
        const auto* dx = level.dx[y];
        const auto* dy = level.dy[y];
        const auto* up = level.magnitude[y];
        const auto* middle = level.magnitude[y + 1];
        const auto* down = level.magnitude[y + 2];
        auto* classes = level.nms_class[y + 1] + 1;
        for (int x = 0; x < width; ++x) {
            classes[x] = kernels::nms_class_at(
                dx[x],
                dy[x],
                {up[x],
                 up[x + 1],
                 up[x + 2],
                 middle[x],
                 middle[x + 1],
                 middle[x + 2],
                 down[x],
                 down[x + 1],
                 down[x + 2]},
                low,
                high
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
    // The resulting set is order-independent; a device may propagate to a fixed point instead.
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
            edges[x] = classes[x] == 2 ? 255 : 0;
        }
    }

    const auto edges = level.edges_plane();
    const kernels::Plane<std::uint8_t>
        thinning{level.thinning[0], level.width, level.height, level.thinning.step1()};
    // Only write the interior: scratch keeps its zero border, edges keeps raw Canny
#pragma omp parallel for schedule(static)
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            thinning.row(y)[x] = kernels::thinning_at(edges.as_const(), x, y, kernels::kThinning1);
        }
    }
#pragma omp parallel for schedule(static)
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            edges.row(y)[x] = kernels::thinning_at(thinning.as_const(), x, y, kernels::kThinning2);
        }
    }
}

EdgesHost Backend::host_edges(Buffers& level) {
    // Return a read-only view of the thinned edges
    return EdgesHost{level.edges_plane().as_const()};
}

} // namespace cctag::portable::cpu

#ifdef CCTAG_TEST
#include <boost/ut.hpp>

namespace cctag::portable::tests::edges_stage {

using namespace boost::ut;

inline suite<"edges_stage"> edges_stage_suite = [] {
    "edges retain weak pixels only when connected to a strong pixel on a one column image"_test =
        [] {
        cpu::Buffers level;
        level.ensure(1, 7);
        level.dx = (cv::Mat1s(7, 1) << 11, 3, 10, 2, 3, 3, 0);
        level.dy.setTo(0);
        cctag::Parameters params(3);
        // Horizontal gradients compare with zero outside this one-column image.
        // The low-threshold tie splits off the final weak component.
        const std::array<std::uint8_t, 7> expected = {255, 255, 255, 0, 0, 0, 0};
        cpu::Backend::edges(level, params);
        for (int y = 0; y < 7; ++y) {
            expect(eq(level.edges(y, 0), expected[y])) << "row" << y;
        }
        std::swap(params._cannyThrLow, params._cannyThrHigh);
        cpu::Backend::edges(level, params);
        for (int y = 0; y < 7; ++y) {
            expect(eq(level.edges(y, 0), expected[y])) << "swapped thresholds at row" << y;
        }

        // A second frame has no strong pixel to retain the weak component
        level.dx.setTo(3);
        cpu::Backend::edges(level, params);
        expect(eq(cv::countNonZero(level.edges), 0));

        level.ensure(1, 1);
        level.dx.setTo(-32768);
        level.dy.setTo(32767);
        cpu::Backend::edges(level, params);
        expect(eq(level.edges(0, 0), 255));
    };

    "edges keep the raw Canny border and use a zero border for the second thinning pass"_test = [] {
        cpu::Buffers level;
        level.ensure(3, 3);
        // NMS retains the centre, its upper neighbour and its right neighbour.
        // Pass one keeps the centre (entry 152). With a copied border, pass two
        // would remove it; the zero scratch border leaves an isolated pixel.
        level.dx = (cv::Mat1s(3, 3) << 0, 11, 0, 0, 11, 0, 0, 0, 0);
        level.dy = (cv::Mat1s(3, 3) << 0, 0, 0, 0, 0, 11, 0, 0, 0);
        const cctag::Parameters params(3);
        cpu::Backend::edges(level, params);
        const cv::Mat1b expected = (cv::Mat1b(3, 3) << 0, 255, 0, 0, 255, 255, 0, 0, 0);
        expect(eq(cv::countNonZero(level.edges != expected), 0));

        // Reusing scratch must also clear the previous interior edge
        level.dx.setTo(0);
        level.dy.setTo(0);
        cpu::Backend::edges(level, params);
        expect(eq(cv::countNonZero(level.edges), 0));
    };
};

} // namespace cctag::portable::tests::edges_stage
#endif // CCTAG_TEST
