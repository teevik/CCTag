/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// The `pyramid` stage: the level-0 copy, then per coarser level the legacy's `cv::resize`
// (`Level.cpp:67`, default `INTER_LINEAR`, which takes the area fast path on an exact halving).
#include "backends/cpu/backend.hpp"

#include <opencv2/imgproc.hpp>

namespace cctag::portable::cpu {

void Backend::load(Buffers& level0, kernels::Plane<const std::uint8_t> input) {
    const cv::Mat1b view(
        static_cast<int>(input.height),
        static_cast<int>(input.width),
        const_cast<std::uint8_t*>(input.data),
        input.stride
    );
    view.copyTo(level0.src);
}

void Backend::pyramid(Buffers& coarser, const Buffers& finer) {
    cv::resize(finer.src, coarser.src, coarser.src.size());
}

PyramidHost Backend::host_pyramid(Buffers& level) {
    return PyramidHost{level.src_plane().as_const()};
}

} // namespace cctag::portable::cpu
