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

/// Detects the CCTag markers of a gray image
void detect(
    boost::ptr_list<ICCTag>& markers,
    int pipe_id,
    std::size_t frame,
    const cv::Mat& gray_src,
    const Parameters& provided_params,
    const CCTagMarkersBank* bank,
    Probe* probe
);

} // namespace portable
} // namespace cctag

#endif
