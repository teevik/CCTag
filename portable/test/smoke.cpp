/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/ICCTag.hpp>

#include <cstdlib>

int main() {
    boost::ptr_list<cctag::ICCTag> detections;
    const cv::Mat image;

    cctag::Parameters parameters(3);
    cctag::cctagDetection(detections, 0, 0, image, parameters);
    if (!detections.empty()) {
        return EXIT_FAILURE;
    }

    cctag::cctagDetection(detections, 0, 0, image, 3);
    return detections.empty() ? EXIT_SUCCESS : EXIT_FAILURE;
}
