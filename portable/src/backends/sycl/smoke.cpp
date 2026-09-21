/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/ICCTag.hpp>
#include <cctag/Probe.hpp>

int main(int argc, char** argv) {
    boost::ptr_list<cctag::ICCTag> detections;
    cctag::Parameters params(3);
    const cv::Mat1b image(64, 64, std::uint8_t{127});

    cctag::cctagDetection(detections, 0, 1, image, params);

    return 0;
}
