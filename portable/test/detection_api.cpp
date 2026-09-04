/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// Both public cctagDetection overloads accept an empty image and return no detection candidates.
// This checks linking and empty-input handling; it does not exercise detection on an image.
#include <cctag/ICCTag.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(detection_api_suite)

BOOST_AUTO_TEST_CASE(parameters_overload_returns_no_detection_candidates_for_an_empty_image) {
    boost::ptr_list<cctag::ICCTag> detections;
    const cv::Mat image;

    cctag::Parameters parameters(3);
    cctag::cctagDetection(detections, 0, 0, image, parameters);
    BOOST_CHECK(detections.empty());
}

BOOST_AUTO_TEST_CASE(crown_count_overload_returns_no_detection_candidates_for_an_empty_image) {
    boost::ptr_list<cctag::ICCTag> detections;
    const cv::Mat image;
    cctag::cctagDetection(detections, 0, 0, image, 3);
    BOOST_CHECK(detections.empty());
}

BOOST_AUTO_TEST_SUITE_END()
