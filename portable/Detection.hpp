/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_DETECTION_HPP
#define CCTAG_PORTABLE_DETECTION_HPP

#include <cctag/ICCTag.hpp>

#include <cstddef>

namespace cctag {
namespace portable {

/// Detects the markers of a gray image, with the pyramid built on the SYCL device and the stages
/// not yet ported run on the CPU. Takes the arguments of cctag::cctagDetection that it uses.
void detect(
    boost::ptr_list<ICCTag>& markers,
    std::size_t frame,
    const cv::Mat& graySrc,
    const Parameters& providedParams,
    const CCTagMarkersBank* bank,
    Probe* probe
);

} // namespace portable
} // namespace cctag

#endif
