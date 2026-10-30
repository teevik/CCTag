/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/CCTag.hpp>
#include <cctag/Canny.hpp>
#include <cctag/Detection.hpp>
#include <cctag/Fitting.hpp>
#include <cctag/Identification.hpp>
#include <cctag/Multiresolution.hpp>
#include <cctag/Probe.hpp>
#include <cctag/Types.hpp>
#include <cctag/Vote.hpp>
#include <cctag/filter/thinning.hpp>
#include <cctag/portable/Detection.hpp>
#include <cctag/portable/Pipeline.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <list>
#include <memory>
#include <vector>

namespace cctag {
namespace portable {

namespace {

using numerical::geometry::Ellipse;

Plane plane(const cv::Mat& image) {
    return Plane{
        static_cast<std::uint32_t>(image.cols),
        static_cast<std::uint32_t>(image.rows),
        image.step,
        image.data
    };
}

// WIP: Code for unfinished stages is copied over temporarily

// TODO: Copied cctagMultiresDetection in Multiresolution.cpp
void project_to_level0(CCTag& marker, const EdgePointCollection& level0) {
    const float scale = marker.scale();
    Ellipse rescaled_outer_ellipse = marker.rescaledOuterEllipse();

    std::list<EdgePoint*> points_in_hull;
    selectEdgePointInEllipticHull(level0, rescaled_outer_ellipse, scale, points_in_hull);
    if (points_in_hull.size() < 5) {
        return;
    }

    std::vector<EdgePoint*> points;
    float sm_final = 1e+10;
    outlierRemoval(points_in_hull, points, sm_final, 20.0, NO_WEIGHT, 60);
    if (points.size() < 5) {
        return;
    }

    try {
        numerical::ellipseFitting(rescaled_outer_ellipse, points);

        std::vector<DirectedPoint2d<Eigen::Vector3f>> directed_points;
        for (EdgePoint* e : points) {
            directed_points.emplace_back(e->x(), e->y(), e->dX(), e->dY());
        }
        marker.setCenterImg(
            Point2d<Eigen::Vector3f>(marker.centerImg().x() * scale, marker.centerImg().y() * scale)
        );
        marker.setRescaledOuterEllipse(rescaled_outer_ellipse);
        marker.setRescaledOuterEllipsePoints(directed_points);
    } catch (...) {
        // The marker keeps its upper level fit
    }
}

/// The host images of one pyramid level
struct HostLevel {
    cv::Mat src;
    cv::Mat dx;
    cv::Mat dy;
    /// 255 on the thinned edges, else 0
    cv::Mat edges;
};

// TODO: temporary until hysteresis is ported to SYCL
void hysteresis(cv::Mat& map) {
    const std::ptrdiff_t width = map.cols;
    Mark* marks = map.ptr<Mark>();
    std::vector<std::ptrdiff_t> stack;
    for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(map.total()); ++i) {
        if (marks[i] == Mark::strong) {
            stack.push_back(i);
        }
    }
    while (!stack.empty()) {
        const std::ptrdiff_t i = stack.back();
        stack.pop_back();
        for (std::ptrdiff_t v = -1; v <= 1; ++v) {
            for (std::ptrdiff_t u = -1; u <= 1; ++u) {
                const std::ptrdiff_t j = i + v * width + u;
                if (marks[j] == Mark::weak) {
                    marks[j] = Mark::strong;
                    stack.push_back(j);
                }
            }
        }
    }
}

} // namespace

void detect(
    boost::ptr_list<ICCTag>& markers,
    int pipe_id,
    std::size_t frame,
    const cv::Mat& gray_src,
    const Parameters& provided_params,
    const CCTagMarkersBank* bank,
    Probe* probe
) {
    Pipeline& pipe = pipeline(pipe_id);

    std::unique_ptr<CCTagMarkersBank> default_bank;
    if (!bank) {
        default_bank.reset(new CCTagMarkersBank(provided_params._nCrowns));
        bank = default_bank.get();
    }

    const Parameters& params = Parameters::OverrideLoaded ? Parameters::Override : provided_params;

    std::srand(1);

    assert(gray_src.isContinuous());
    assert(gray_src.type() == CV_8U);

    const int level_count = params._numberOfProcessedMultiresLayers;
    run(pipe,
        gray_src.data,
        gray_src.cols,
        gray_src.rows,
        level_count,
        params._cannyThrLow,
        params._cannyThrHigh);

    // TODO: Hysteresis and thinning in SYCL
    std::vector<HostLevel> levels;
    int width = gray_src.cols;
    int height = gray_src.rows;
    for (int i = 0; i < level_count; ++i) {
        levels.push_back(
            {cv::Mat(height, width, CV_8UC1),
             cv::Mat(height, width, CV_16SC1),
             cv::Mat(height, width, CV_16SC1),
             cv::Mat()}
        );
        HostLevel& level = levels.back();
        download_source(pipe, i, level.src.data);
        download_gradient(pipe, i, level.dx.ptr<std::int16_t>(), level.dy.ptr<std::int16_t>());
        cv::Mat map(height, width, CV_8UC1);
        download_map(pipe, i, map.ptr<Mark>());
        if (probe) {
            probe->pyramid(i, plane(level.src));
            probe->gradient(i, plane(level.dx), plane(level.dy));
            probe->stage("suppress", i, plane(map));
        }
        hysteresis(map);
        if (probe) {
            probe->stage("hysteresis", i, plane(map));
        }
        level.edges = map == int(Mark::strong);
        cv::Mat temp(height, width, CV_8UC1);
        thin(level.edges, temp);
        if (probe) {
            probe->edges(i, plane(level.edges));
        }
        width /= 2;
        height /= 2;
    }

    // Every level gets a collection of the size of level 0
    std::vector<std::unique_ptr<EdgePointCollection>> edge_collections(level_count);
    for (auto& edge_collection : edge_collections) {
        edge_collection.reset(new EdgePointCollection(gray_src.cols, gray_src.rows));
    }

    // TODO: Detect edges in SYCL
    CCTag::List found;
    for (int i = level_count - 1; i >= 0; --i) {
        const HostLevel& level = levels[i];
        EdgePointCollection& edge_collection = *edge_collections[i];

        std::vector<EdgePoint*> seeds;
        edgesPointsFromCanny(edge_collection, level.edges, level.dx, level.dy);
        vote(edge_collection, seeds, level.dx, level.dy, params);
        if (seeds.size() > 1) {
            std::sort(seeds.begin(), seeds.end(), receivedMoreVoteThan);
        }

        CCTag::List level_markers;
        cctagDetectionFromEdges(
            level_markers,
            edge_collection,
            level.src,
            seeds,
            frame,
            i,
            std::pow(2.0, i),
            params,
            nullptr
        );
        for (const CCTag& marker : level_markers) {
            found.push_back(new CCTag(marker));
        }
    }

    // Bring every marker to level 0
    for (CCTag& marker : found) {
        if (marker.pyramidLevel() > 0) {
            project_to_level0(marker, *edge_collections[0]);
        } else {
            marker.setRescaledOuterEllipsePoints(marker.points().back());
        }
    }

    // TODO: Identify markers in SYCL
    if (params._doIdentification) {
        const cv::Mat& src = levels[0].src;
        std::vector<std::vector<ImageCut>> v_selected_cuts(found.size());
        std::vector<int> detected(found.size(), -1);

        int tag_index = 0;
        for (const CCTag& marker : found) {
            detected[tag_index] = identification::identify_step_1(
                tag_index,
                marker,
                v_selected_cuts[tag_index],
                src,
                params
            );
            ++tag_index;
        }

        tag_index = 0;
        for (CCTag& marker : found) {
            if (detected[tag_index] == status::id_reliable) {
                detected[tag_index] = identification::identify_step_2(
                    tag_index,
                    marker,
                    v_selected_cuts[tag_index],
                    bank->getMarkers(),
                    src,
                    nullptr,
                    params
                );
            }
            marker.setStatus(detected[tag_index]);
            ++tag_index;
        }
    }

    // TODO: In SYCL
    CCTag::List markers_prelim, markers_final;
    for (const CCTag& marker : found) {
        update(markers_prelim, marker);
    }
    for (const CCTag& marker : markers_prelim) {
        update(markers_final, marker);
    }
    markers_final.sort();

    markers.clear();
    for (const CCTag& marker : markers_final) {
        markers.push_back(new CCTag(marker));
    }
}

} // namespace portable
} // namespace cctag
