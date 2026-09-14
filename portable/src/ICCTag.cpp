/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "host/context.hpp"
#include "host/detect.hpp"

#include <cctag/ICCTag.hpp>
#include <cctag/Probe.hpp>

#if defined(CCTAG_PORTABLE_BACKEND_CPU)
#include "backends/cpu/backend.hpp"
namespace cctag::portable {
using SelectedBackend = cpu::Backend;
}
#else
#error "CCTAG_PORTABLE_BACKEND_<NAME> must be defined for exactly one execution backend"
#endif

#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace cctag {

namespace {

using SelectedContext = portable::Context<portable::SelectedBackend>;

std::mutex registry_mutex;
std::map<int, std::unique_ptr<SelectedContext>> registry;

/// Returns the pipe's persistent context, creating it on first use
SelectedContext& context_for(int pipeId) {
    const std::lock_guard<std::mutex> lock(registry_mutex);
    auto& slot = registry[pipeId];
    if (!slot) {
        slot = std::make_unique<SelectedContext>();
    }
    return *slot;
}

} // namespace

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
    // TODO(markers stage): Load `parameterFile` through Params.cpp and
    // `cctagBankFilename` through CCTagMarkersBank
    // Reject file arguments until loading is supported
    if (!parameterFile.empty() || !cctagBankFilename.empty()) {
        throw std::logic_error(
            "cctagDetection: parameter and bank files are not supported by the portable pipeline "
            "yet"
        );
    }
    const Parameters params(nRings);
    cctagDetection(markers, pipeId, frame, graySrc, params, durations, nullptr, nullptr);
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
    (void)frame;
    (void)durations;
    (void)pBank;
    markers.clear();

    if (graySrc.empty()) {
        return;
    }

    if (graySrc.type() != CV_8UC1) {
        throw std::invalid_argument("cctagDetection: the input image must be 8-bit single-channel");
    }

    const portable::kernels::Plane<const std::uint8_t> input{
        .data = graySrc.ptr<std::uint8_t>(0),
        .width = static_cast<std::uint32_t>(graySrc.cols),
        .height = static_cast<std::uint32_t>(graySrc.rows),
        .stride = graySrc.step[0],
    };

    portable::detect(context_for(pipeId), input, params, probe);
}

} // namespace cctag
