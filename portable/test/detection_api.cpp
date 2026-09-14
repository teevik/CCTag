/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/ICCTag.hpp>
#include <cctag/Probe.hpp>

#include <boost/ut.hpp>

#include <limits>
#include <stdexcept>
#include <vector>

using namespace boost::ut;

namespace {

struct PyramidProbe : cctag::Probe {
    std::vector<cv::Size> sizes;

    void pyramid(std::uint32_t, const cctag::Plane& src) override {
        sizes.emplace_back(static_cast<int>(src.width), static_cast<int>(src.height));
    }
};

// Restore the shared override even when a detection throws or an assertion fails
struct OverrideState {
    bool loaded = cctag::Parameters::OverrideLoaded;
    std::size_t levels = cctag::Parameters::Override._numberOfProcessedMultiresLayers;

    ~OverrideState() {
        cctag::Parameters::OverrideLoaded = loaded;
        cctag::Parameters::Override._numberOfProcessedMultiresLayers = levels;
    }
};

} // namespace

inline suite<"detection_api"> detection_api_suite = [] {
    "parameters overload returns no detection candidates for an empty image"_test = [] {
        boost::ptr_list<cctag::ICCTag> detections;
        const cv::Mat image;

        cctag::Parameters parameters(3);
        cctag::cctagDetection(detections, 0, 0, image, parameters);
        expect(detections.empty());
    };

    "crown count overload returns no detection candidates for an empty image"_test = [] {
        boost::ptr_list<cctag::ICCTag> detections;
        const cv::Mat image;
        cctag::cctagDetection(detections, 0, 0, image, 3);
        expect(detections.empty());
    };

    "parameters overload selects the global override only when loaded"_test = [] {
        boost::ptr_list<cctag::ICCTag> detections;
        const cv::Mat1b image(16, 16, std::uint8_t{0});
        const cctag::Parameters parameters(3);
        const OverrideState restore;
        cctag::Parameters::Override._numberOfProcessedMultiresLayers = 2;

        for (const bool loaded : {false, true, false}) {
            cctag::Parameters::OverrideLoaded = loaded;
            PyramidProbe probe;
            cctag::cctagDetection(detections, 10, 0, image, parameters, nullptr, nullptr, &probe);
            const std::size_t expected = loaded ? 2 : parameters._numberOfProcessedMultiresLayers;
            expect(eq(probe.sizes.size(), expected));
        }
    };

    "detection rejects images too small for the configured pyramid"_test = [] {
        boost::ptr_list<cctag::ICCTag> detections;
        const cctag::Parameters parameters(3);

        for (const cv::Size size : {cv::Size{7, 16}, cv::Size{16, 7}, cv::Size{1, 1}}) {
            const cv::Mat1b image(size, std::uint8_t{0});
            expect(throws<std::invalid_argument>([&] {
                cctag::cctagDetection(detections, 11, 0, image, parameters);
            }));
        }
        // The crown-count overload must enforce the same default-level requirement.
        expect(throws<std::invalid_argument>([&] {
            cctag::cctagDetection(detections, 11, 0, cv::Mat1b(7, 16, std::uint8_t{0}), 3);
        }));
    };

    "detection rejects zero and excessive processed pyramid level counts"_test = [] {
        boost::ptr_list<cctag::ICCTag> detections;
        const cv::Mat1b image(16, 16, std::uint8_t{0});
        cctag::Parameters parameters(3);

        for (const std::size_t count : {std::size_t{0}, std::numeric_limits<std::size_t>::max()}) {
            parameters._numberOfProcessedMultiresLayers = count;
            expect(throws<std::invalid_argument>([&] {
                cctag::cctagDetection(detections, 12, 0, image, parameters);
            }));
        }
    };

    "a detection pipe recovers after rejection and supports one pixel pyramid levels"_test = [] {
        boost::ptr_list<cctag::ICCTag> detections;
        cctag::Parameters parameters(3);
        const cv::Mat1b image(8, 8, std::uint8_t{0});
        const std::vector<cv::Size> expected{{8, 8}, {4, 4}, {2, 2}, {1, 1}};

        for (int frame = 0; frame < 2; ++frame) {
            PyramidProbe probe;
            cctag::cctagDetection(
                detections,
                13,
                frame,
                image,
                parameters,
                nullptr,
                nullptr,
                &probe
            );
            expect(probe.sizes == expected);
            expect(throws<std::invalid_argument>([&] {
                cctag::cctagDetection(
                    detections,
                    13,
                    frame,
                    cv::Mat1b(7, 8, std::uint8_t{0}),
                    parameters
                );
            }));
        }

        parameters._numberOfProcessedMultiresLayers = 1;
        for (const cv::Size size : {cv::Size{1, 1}, cv::Size{1, 7}, cv::Size{7, 1}}) {
            PyramidProbe probe;
            cctag::cctagDetection(
                detections,
                13,
                2,
                cv::Mat1b(size, std::uint8_t{0}),
                parameters,
                nullptr,
                nullptr,
                &probe
            );
            expect(probe.sizes == std::vector<cv::Size>{size});
        }
    };
};
