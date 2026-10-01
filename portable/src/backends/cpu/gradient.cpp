/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "backends/cpu/backend.hpp"
#include "kernels/frontend.hpp"

namespace cctag::portable::cpu {

// Separable 9-tap filters with replicated borders: each axis is smoothed across
// and differentiated along, sharing one vertical-pass scratch plane
void Backend::gradient(Buffers& level) {
    level.derivative_scratch.create(level.height, level.width);
    const auto source = level.src_plane().as_const();
    const kernels::Plane<const float> scratch{
        level.derivative_scratch[0],
        level.width,
        level.height,
        level.derivative_scratch.step1()
    };
    for (int axis = 0; axis < 2; ++axis) {
        const auto& vertical = axis == 0 ? kernels::gaussian_kernel : kernels::derivative_kernel;
        const auto& horizontal = axis == 0 ? kernels::derivative_kernel : kernels::gaussian_kernel;
#pragma omp parallel for schedule(static)
        for (int y = 0; y < int(level.height); ++y) {
            for (int x = 0; x < int(level.width); ++x) {
                level.derivative_scratch(y, x) = kernels::filter_at(source, x, y, true, vertical);
            }
        }
        auto& result = axis == 0 ? level.dx : level.dy;
#pragma omp parallel for schedule(static)
        for (int y = 0; y < int(level.height); ++y) {
            for (int x = 0; x < int(level.width); ++x) {
                result(y, x) =
                    static_cast<std::int16_t>(kernels::filter_at(scratch, x, y, false, horizontal));
            }
        }
    }
}

} // namespace cctag::portable::cpu

#ifdef CCTAG_TEST
#include <boost/ut.hpp>

#include <opencv2/core.hpp>

namespace cctag::portable::tests::gradient_stage {

using namespace boost::ut;

inline suite<"gradient_stage"> gradient_stage_suite = [] {
    "gradient of a constant plane is exactly zero, including a single pixel"_test = [] {
        // The fused multiply-adds of the antisymmetric derivative taps must cancel exactly,
        // and replicated borders must not create a gradient at the image edge. The level is
        // reused across sizes and values, so stale scratch would also show up here.
        cpu::Buffers level;
        for (const auto size : {cv::Size(1, 1), cv::Size(37, 23)}) {
            level.ensure(size.width, size.height);
            for (const int value : {0, 255, 127}) {
                level.src.setTo(value);
                cpu::Backend::gradient(level);
                expect(eq(cv::countNonZero(level.dx), 0)) << value;
                expect(eq(cv::countNonZero(level.dy), 0)) << value;
            }
        }
    };
};

} // namespace cctag::portable::tests::gradient_stage
#endif // CCTAG_TEST
