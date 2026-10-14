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
#include <limits>
#include <list>
#include <memory>
#include <stdexcept>
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

/// Refits a marker found on an upper level to the edge points of level 0
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

/// Fills the votes of a level into its edge collection as vote() in Vote.cpp does. Returns the
/// seeds by votes descending.
std::vector<EdgePoint*> fill_votes(
    EdgePointCollection& edge_collection,
    const std::vector<Edgel>& edgels,
    const Parameters& params
) {
    std::vector<std::vector<int>> voters(edgels.size());
    std::vector<EdgePoint*> seeds;
    for (std::size_t j = 0; j < edgels.size(); ++j) {
        const Edgel& e = edgels[j];
        if (e.chosen >= 0) {
            voters[e.chosen].push_back(int(j));
        }
        if (e.votes == 0) {
            continue;
        }
        EdgePoint* p = edge_collection(int(j));
        p->_flowLength = e.flow / e.votes;
        if (e.votes >= params._minVotesToSelectCandidate) {
            p->_isMax = int(e.votes);
            seeds.push_back(p);
        }
    }
    edge_collection.create_voter_lists(voters);
    // The seeds are in edge index order, so the stable sort breaks ties by it
    std::stable_sort(seeds.begin(), seeds.end(), receivedMoreVoteThan);
    return seeds;
}

/// The homography m and its inverse, as getSignals inverts it, row-major
Homography pack(const Eigen::Matrix3f& m) {
    const Eigen::Matrix3f inverse = m.inverse();
    Homography h;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            h.h[r * 3 + c] = m(r, c);
            h.inverse[r * 3 + c] = inverse(r, c);
        }
    }
    return h;
}

/// The imaged center search of refineConicFamilyGlob, with the residuals computed on the device
class DeviceCenterSearch : public identification::CenterSearch {
  public:
    explicit DeviceCenterSearch(Pipeline& pipe) :
        pipe_(pipe) {}

    bool refine(
        Eigen::Matrix3f& homography,
        Point2d<Eigen::Vector3f>& center,
        float& residual,
        const std::vector<ImageCut>& cuts,
        const Ellipse& outer_ellipse,
        const Parameters& params
    ) override {
        const std::size_t grid_n_sample = params._imagedCenterNGridSample;
        const std::size_t samples = cuts.front().imgSignal().size();
        cuts_.clear();
        for (const ImageCut& cut : cuts) {
            cuts_.push_back({cut.stop().x(), cut.stop().y(), cut.beginSig(), cut.endSig()});
        }
        upload_cuts(pipe_, cuts_, samples, grid_n_sample * grid_n_sample);

        // The neighbourhood shrinks around the best point until finer than 0.02 pixel, as
        // refineConicFamilyGlob
        float neighbour_size = params._imagedCenterNeighbourSize;
        const float max_semi_axis = std::max(outer_ellipse.a(), outer_ellipse.b());
        while (neighbour_size * max_semi_axis > 0.02) {
            identification::getNearbyPoints(
                outer_ellipse,
                center,
                nearby_points_,
                neighbour_size,
                grid_n_sample,
                identification::GRID
            );

            // As imageCenterOptimizationGlob, the points without a homography are skipped
            points_.clear();
            matrices_.clear();
            homographies_.clear();
            for (const Point2d<Eigen::Vector3f>& point : nearby_points_) {
                Eigen::Matrix3f m;
                try {
                    identification::computeHomographyFromEllipseAndImagedCenter(
                        outer_ellipse,
                        point,
                        m
                    );
                } catch (...) {
                    continue;
                }
                points_.push_back(point);
                matrices_.push_back(m);
                homographies_.push_back(pack(m));
            }
            center_costs(pipe_, homographies_, residuals_);

            // The first readable point of least residual
            residual = std::numeric_limits<float>::max();
            std::size_t best = points_.size();
            for (std::size_t i = 0; i < residuals_.size(); ++i) {
                if (residuals_[i] >= 0 && (best == points_.size() || residuals_[i] < residual)) {
                    residual = residuals_[i];
                    best = i;
                }
            }
            if (best == points_.size()) {
                return false;
            }
            center = points_[best];
            homography = matrices_[best];
            neighbour_size /= float((grid_n_sample - 1) / 2);
        }
        return true;
    }

  private:
    Pipeline& pipe_;
    std::vector<Cut> cuts_;
    std::vector<Point2d<Eigen::Vector3f>> nearby_points_;
    std::vector<Point2d<Eigen::Vector3f>> points_;
    std::vector<Eigen::Matrix3f> matrices_;
    std::vector<Homography> homographies_;
    std::vector<float> residuals_;
};

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

    // As vote() in Vote.cpp, the device votes with _angleVoting 0
    if (params._angleVoting != 0) {
        throw std::domain_error(
            "thrVotingAngle must be equal to 0 or edge points gradients have to be normalized."
        );
    }

    const int level_count = params._numberOfProcessedMultiresLayers;
    load(
        pipe,
        gray_src.data,
        gray_src.cols,
        gray_src.rows,
        level_count,
        params._cannyThrLow,
        params._cannyThrHigh,
        params._distSearch,
        params._nCrowns,
        params._ratioVoting
    );

    // The gradient and edge images are only needed by the probe
    std::vector<cv::Mat> sources;
    int width = gray_src.cols;
    int height = gray_src.rows;
    for (int i = 0; i < level_count; ++i) {
        sources.emplace_back(height, width, CV_8UC1);
        download_source(pipe, i, sources.back().data);
        if (probe) {
            cv::Mat dx(height, width, CV_16SC1);
            cv::Mat dy(height, width, CV_16SC1);
            cv::Mat edges(height, width, CV_8UC1);
            download_gradient(pipe, i, dx.ptr<std::int16_t>(), dy.ptr<std::int16_t>());
            download_edges(pipe, i, edges.data);
            probe->pyramid(i, plane(sources.back()));
            probe->gradient(i, plane(dx), plane(dy));
            probe->edges(i, plane(edges));
        }
        width /= 2;
        height /= 2;
    }

    static_assert(max_edgels == EdgePointCollection::MAX_POINTS, "max_edgels must match");

    // Every level gets a collection of the size of level 0, as in cctagMultiresDetection
    std::vector<std::unique_ptr<EdgePointCollection>> edge_collections(level_count);
    for (auto& edge_collection : edge_collections) {
        edge_collection.reset(new EdgePointCollection(gray_src.cols, gray_src.rows));
    }

    // Detect on every level, from the top of the pyramid down
    CCTag::List found;
    std::vector<Edgel> edgels;
    for (int i = level_count - 1; i >= 0; --i) {
        EdgePointCollection& edge_collection = *edge_collections[i];

        // The edge points come in scan order, so their indices are those of the device list
        download_edgels(pipe, i, edgels);
        for (const Edgel& e : edgels) {
            edge_collection.add_point(e.x, e.y, e.dx, e.dy);
        }
        for (std::size_t j = 0; j < edgels.size(); ++j) {
            EdgePoint* p = edge_collection(int(j));
            edge_collection.set_before(p, edgels[j].before);
            edge_collection.set_after(p, edgels[j].after);
        }

        const std::vector<EdgePoint*> seeds = fill_votes(edge_collection, edgels, params);

        CCTag::List level_markers;
        cctagDetectionFromEdges(
            level_markers,
            edge_collection,
            sources[i],
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

    for (CCTag& marker : found) {
        if (marker.pyramidLevel() > 0) {
            project_to_level0(marker, *edge_collections[0]);
        } else {
            marker.setRescaledOuterEllipsePoints(marker.points().back());
        }
    }

    if (params._doIdentification) {
        const cv::Mat& src = sources[0];
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

        DeviceCenterSearch center_search(pipe);
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
                    params,
                    &center_search
                );
            }
            marker.setStatus(detected[tag_index]);
            ++tag_index;
        }
    }

    // Delete overlapping markers while keeping the best ones, in two passes as the legacy code does
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
