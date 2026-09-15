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

#include <boost/archive/xml_iarchive.hpp>

#if defined(CCTAG_PORTABLE_BACKEND_CPU)
#include "backends/cpu/backend.hpp"
namespace cctag::portable {
using SelectedBackend = cpu::Backend;
}
#else
#error "CCTAG_PORTABLE_BACKEND_<NAME> must be defined for exactly one execution backend"
#endif

#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace cctag {

namespace {

using SelectedContext = portable::Context<portable::SelectedBackend>;

/// Owns the public API's copy of a portable detection candidate
class DetectionCandidate final : public ICCTag {
  public:
    explicit DetectionCandidate(const portable::Marker& marker) :
        ellipse(
            Point2d<Eigen::Vector3f>(marker.outer_ellipse.cx, marker.outer_ellipse.cy),
            marker.outer_ellipse.a,
            marker.outer_ellipse.b,
            marker.outer_ellipse.angle
        ) {
        _x = marker.center.x();
        _y = marker.center.y();
        _id = marker.id;
        _status = marker.status;
    }

    float x() const override {
        return _x;
    }
    float y() const override {
        return _y;
    }
    MarkerID id() const override {
        return _id;
    }
    int getStatus() const override {
        return _status;
    }
    const numerical::geometry::Ellipse& rescaledOuterEllipse() const override {
        return ellipse;
    }
    ICCTag* clone() const override {
        return new DetectionCandidate(*this);
    }

  private:
    numerical::geometry::Ellipse ellipse;
};

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
    Parameters params(nRings);
    if (!parameterFile.empty()) {
        std::ifstream input(parameterFile);
        if (!input) {
            throw std::invalid_argument(
                "cctagDetection: cannot open parameter file " + parameterFile
            );
        }
        boost::archive::xml_iarchive archive(input);
        archive >> boost::serialization::make_nvp("CCTagsParams", params);
    }
    if (cctagBankFilename.empty()) {
        cctagDetection(markers, pipeId, frame, graySrc, params, durations, nullptr, nullptr);
    } else {
        const CCTagMarkersBank bank(cctagBankFilename);
        cctagDetection(markers, pipeId, frame, graySrc, params, durations, &bank, nullptr);
    }
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

    const Parameters& effective_params = Parameters::OverrideLoaded ? Parameters::Override : params;
    auto& context = context_for(pipeId);
    if (pBank) {
        context.bank.set(*pBank);
    } else {
        context.bank.custom = false;
    }
    portable::detect(context, input, effective_params, probe);
    for (const auto& marker : context.markers) {
        markers.push_back(new DetectionCandidate(marker));
    }
}

} // namespace cctag
