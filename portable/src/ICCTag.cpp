/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// The only portable-pipeline file that sees cv::Mat and boost::ptr_list. Owns the pipeId -> context
// registry and instantiates the execution backend selected by CCTAG_PORTABLE_BACKEND.
#include <cctag/ICCTag.hpp>
#include <cctag/Probe.hpp>

#include "host/context.hpp"
#include "host/detect.hpp"

#if defined(CCTAG_PORTABLE_BACKEND_CPU)
#include "backends/cpu/backend.hpp"
namespace cctag::portable {
using SelectedBackend = cpu::Backend;
}
#elif defined(CCTAG_PORTABLE_BACKEND_STUB)
#include "backends/stub/backend.hpp"
namespace cctag::portable {
using SelectedBackend = stub::Backend;
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

/// One persistent context per pipe; concurrent detections on the same pipe are the caller's
/// responsibility, as in the legacy pipeline.
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
    (void)parameterFile;
    (void)cctagBankFilename;
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
        graySrc.ptr<std::uint8_t>(0),
        static_cast<std::uint32_t>(graySrc.cols),
        static_cast<std::uint32_t>(graySrc.rows),
        graySrc.step[0],
    };
    portable::detect(context_for(pipeId), input, params, probe);
}

} // namespace cctag
