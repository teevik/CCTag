/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "portable/Detection.hpp"

#include "portable/Pipeline.hpp"

#include <cctag/CCTag.hpp>
#include <cctag/Canny.hpp>
#include <cctag/Detection.hpp>
#include <cctag/Fitting.hpp>
#include <cctag/Identification.hpp>
#include <cctag/Multiresolution.hpp>
#include <cctag/Probe.hpp>
#include <cctag/Types.hpp>
#include <cctag/Vote.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <list>
#include <memory>
#include <vector>

namespace cctag {
namespace portable {

namespace {

using numerical::geometry::Ellipse;

/// The host images of one pyramid level
struct Level {
    cv::Mat src;
    cv::Mat dx;
    cv::Mat dy;
    cv::Mat edges;
};

Plane plane(const cv::Mat& image) {
    return Plane{
        static_cast<std::uint32_t>(image.cols),
        static_cast<std::uint32_t>(image.rows),
        image.step,
        image.data
    };
}

/// Refits a marker found on an upper level to the edge points of level 0
void projectToLevel0(CCTag& marker, const EdgePointCollection& level0) {
    const float scale = marker.scale();
    Ellipse rescaledOuterEllipse = marker.rescaledOuterEllipse();

    std::list<EdgePoint*> pointsInHull;
    selectEdgePointInEllipticHull(level0, rescaledOuterEllipse, scale, pointsInHull);
    if (pointsInHull.size() < 5) {
        return;
    }

    std::vector<EdgePoint*> points;
    float SmFinal = 1e+10;
    outlierRemoval(pointsInHull, points, SmFinal, 20.0, NO_WEIGHT, 60);
    if (points.size() < 5) {
        return;
    }

    try {
        numerical::ellipseFitting(rescaledOuterEllipse, points);

        std::vector<DirectedPoint2d<Eigen::Vector3f>> directedPoints;
        for (EdgePoint* e : points) {
            directedPoints.emplace_back(e->x(), e->y(), e->dX(), e->dY());
        }
        marker.setCenterImg(
            Point2d<Eigen::Vector3f>(marker.centerImg().x() * scale, marker.centerImg().y() * scale)
        );
        marker.setRescaledOuterEllipse(rescaledOuterEllipse);
        marker.setRescaledOuterEllipsePoints(directedPoints);
    } catch (...) {
        // The marker keeps its upper level fit
    }
}

} // namespace

void detect(
    boost::ptr_list<ICCTag>& markers,
    int pipeId,
    std::size_t frame,
    const cv::Mat& graySrc,
    const Parameters& providedParams,
    const CCTagMarkersBank* bank,
    Probe* probe
) {
    Pipeline& pipe = pipeline(pipeId);

    std::unique_ptr<CCTagMarkersBank> defaultBank;
    if (!bank) {
        defaultBank.reset(new CCTagMarkersBank(providedParams._nCrowns));
        bank = defaultBank.get();
    }

    const Parameters& params = Parameters::OverrideLoaded ? Parameters::Override : providedParams;

    std::srand(1);

    assert(graySrc.isContinuous());
    assert(graySrc.type() == CV_8U);

    const int levelCount = params._numberOfProcessedMultiresLayers;
    load(
        pipe,
        graySrc.data,
        graySrc.cols,
        graySrc.rows,
        levelCount,
        params._cannyThrLow,
        params._cannyThrHigh
    );

    std::vector<Level> levels;
    int width = graySrc.cols;
    int height = graySrc.rows;
    for (int i = 0; i < levelCount; ++i) {
        levels.push_back(
            {cv::Mat(height, width, CV_8UC1),
             cv::Mat(height, width, CV_16SC1),
             cv::Mat(height, width, CV_16SC1),
             cv::Mat(height, width, CV_8UC1)}
        );
        Level& level = levels.back();
        downloadSource(pipe, i, level.src.data);
        downloadGradient(pipe, i, level.dx.ptr<std::int16_t>(), level.dy.ptr<std::int16_t>());
        downloadEdges(pipe, i, level.edges.data);
        width /= 2;
        height /= 2;
    }

    if (probe) {
        for (int i = 0; i < levelCount; ++i) {
            probe->pyramid(i, plane(levels[i].src));
            probe->gradient(i, plane(levels[i].dx), plane(levels[i].dy));
            probe->edges(i, plane(levels[i].edges));
        }
    }

    // Every level gets a collection of the size of level 0, as in cctagMultiresDetection
    std::vector<std::unique_ptr<EdgePointCollection>> edgeCollections(levelCount);
    for (auto& edgeCollection : edgeCollections) {
        edgeCollection.reset(new EdgePointCollection(graySrc.cols, graySrc.rows));
    }

    // Detect on every level, from the top of the pyramid down
    CCTag::List found;
    for (int i = levelCount - 1; i >= 0; --i) {
        const Level& level = levels[i];
        EdgePointCollection& edgeCollection = *edgeCollections[i];

        std::vector<EdgePoint*> seeds;
        edgesPointsFromCanny(edgeCollection, level.edges, level.dx, level.dy);
        vote(edgeCollection, seeds, level.dx, level.dy, params);
        if (seeds.size() > 1) {
            std::sort(seeds.begin(), seeds.end(), receivedMoreVoteThan);
        }

        CCTag::List levelMarkers;
        cctagDetectionFromEdges(
            levelMarkers,
            edgeCollection,
            level.src,
            seeds,
            frame,
            i,
            std::pow(2.0, i),
            params,
            nullptr
        );
        for (const CCTag& marker : levelMarkers) {
            found.push_back(new CCTag(marker));
        }
    }

    for (CCTag& marker : found) {
        if (marker.pyramidLevel() > 0) {
            projectToLevel0(marker, *edgeCollections[0]);
        } else {
            marker.setRescaledOuterEllipsePoints(marker.points().back());
        }
    }

    if (params._doIdentification) {
        const cv::Mat& src = levels[0].src;
        std::vector<std::vector<ImageCut>> vSelectedCuts(found.size());
        std::vector<int> detected(found.size(), -1);

        int tagIndex = 0;
        for (const CCTag& marker : found) {
            detected[tagIndex] = identification::identify_step_1(
                tagIndex,
                marker,
                vSelectedCuts[tagIndex],
                src,
                params
            );
            ++tagIndex;
        }

        tagIndex = 0;
        for (CCTag& marker : found) {
            if (detected[tagIndex] == status::id_reliable) {
                detected[tagIndex] = identification::identify_step_2(
                    tagIndex,
                    marker,
                    vSelectedCuts[tagIndex],
                    bank->getMarkers(),
                    src,
                    nullptr,
                    params
                );
            }
            marker.setStatus(detected[tagIndex]);
            ++tagIndex;
        }
    }

    // Delete overlapping markers while keeping the best ones, in two passes as the legacy code does
    CCTag::List markersPrelim, markersFinal;
    for (const CCTag& marker : found) {
        update(markersPrelim, marker);
    }
    for (const CCTag& marker : markersPrelim) {
        update(markersFinal, marker);
    }
    markersFinal.sort();

    markers.clear();
    for (const CCTag& marker : markersFinal) {
        markers.push_back(new CCTag(marker));
    }
}

} // namespace portable
} // namespace cctag
