/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// The `gradient` stage: the legacy's two `cv::filter2D` calls (`filter/cvRecode.cpp`) with the
// 9x9 derivative kernel and its transpose, replicate borders, output saturated to int16.
#include "kernels/gradient.hpp"

#include "backends/cpu/backend.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace cctag::portable::cpu {

namespace {

/// The 9x9 derivative kernel as a matrix of its own; `dy` is the transpose.
cv::Mat1f derivative_kernel() {
    cv::Mat1f kernel(9, 9);
    std::copy_n(&kernels::kDerivativeKernel[0][0], kernel.total(), kernel.begin());
    return kernel;
}

const cv::Mat1f kKernelDx = derivative_kernel();
const cv::Mat1f kKernelDy = kKernelDx.t();

} // namespace

void Backend::gradient(Buffers& level) {
    const cv::Point anchor{-1, -1};
    const double delta{0};
    cv::filter2D(level.src, level.dx, CV_16SC1, kKernelDx, anchor, delta, cv::BORDER_REPLICATE);
    cv::filter2D(level.src, level.dy, CV_16SC1, kKernelDy, anchor, delta, cv::BORDER_REPLICATE);
}

GradientHost Backend::host_gradient(Buffers& level) {
    return GradientHost{level.dx_plane().as_const(), level.dy_plane().as_const()};
}

} // namespace cctag::portable::cpu
