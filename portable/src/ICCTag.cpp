/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/ICCTag.hpp>

namespace cctag {

void cctagDetection(
    boost::ptr_list<ICCTag>& markers,
    int pipeId,
    std::size_t frame,
    const cv::Mat& graySrc,
    std::size_t nRings,
    logtime::Mgmt* durations,
    const std::string& parameterFile,
    const std::string& cctagBankFilename
) {
    (void)pipeId;
    (void)frame;
    (void)graySrc;
    (void)nRings;
    (void)durations;
    (void)parameterFile;
    (void)cctagBankFilename;
    markers.clear();
}

void cctagDetection(
    boost::ptr_list<ICCTag>& markers,
    int pipeId,
    std::size_t frame,
    const cv::Mat& graySrc,
    const Parameters& params,
    logtime::Mgmt* durations,
    const CCTagMarkersBank* pBank,
    Probe* probe
) {
    (void)pipeId;
    (void)frame;
    (void)graySrc;
    (void)params;
    (void)durations;
    (void)pBank;
    (void)probe;
    markers.clear();
}

} // namespace cctag
