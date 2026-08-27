/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/ImagePyramid.hpp>
#include <cctag/Probe.hpp>
#include <cctag/ProbeLegacy.hpp>
#include <cctag/Types.hpp>
#include <cctag/Vote.hpp>

#include <list>

namespace cctag {
namespace {

Plane plane(const cv::Mat& matrix)
{
    return Plane{
      static_cast<std::uint32_t>(matrix.cols), static_cast<std::uint32_t>(matrix.rows), matrix.step, matrix.data};
}

} // namespace

void probePyramid(Probe* probe, const ImagePyramid& imagePyramid)
{
    for(std::size_t level = 0; level < imagePyramid.getNbLevels(); ++level)
    {
        const Level* current = imagePyramid.getLevel(level);
        const std::uint32_t index = static_cast<std::uint32_t>(level);
        probe->pyramid(index, plane(current->getSrc()));
        probe->gradient(index, plane(current->getDx()), plane(current->getDy()));
        probe->edges(index, plane(current->getEdges()));
    }
}

void probeEdgePointsAndVote(Probe* probe,
                            std::uint32_t level,
                            EdgePointCollection& edgeCollection,
                            const std::vector<EdgePoint*>& seeds)
{
    const std::size_t count = static_cast<std::size_t>(edgeCollection.get_point_count());
    std::vector<std::int32_t> xy;
    std::vector<float> gradients;
    std::vector<std::int32_t> links;
    std::vector<std::int32_t> votersOffsets;
    std::vector<std::int32_t> votersValues;
    std::vector<std::int32_t> isMax;
    std::vector<float> flowLength;
    std::vector<std::int32_t> seedIndices;

    xy.reserve(count * 2);
    gradients.reserve(count * 2);
    links.reserve(count * 2);
    votersOffsets.reserve(count + 1);
    isMax.reserve(count);
    flowLength.reserve(count);
    seedIndices.reserve(seeds.size());
    votersOffsets.push_back(0);

    for(std::size_t index = 0; index < count; ++index)
    {
        EdgePoint* point = edgeCollection(static_cast<int>(index));
        xy.push_back(point->x());
        xy.push_back(point->y());
        gradients.push_back(point->dX());
        gradients.push_back(point->dY());
        links.push_back(edgeCollection(edgeCollection.before(point)));
        links.push_back(edgeCollection(edgeCollection.after(point)));

        const EdgePointCollection::voter_list voters = edgeCollection.voters(point);
        for(const int* voter = voters.first; voter != voters.second; ++voter)
        {
            votersValues.push_back(*voter);
        }
        votersOffsets.push_back(static_cast<std::int32_t>(votersValues.size()));
        isMax.push_back(point->_isMax);
        flowLength.push_back(point->_flowLength);
    }

    for(const EdgePoint* seed : seeds)
    {
        seedIndices.push_back(edgeCollection(seed));
    }

    probe->edge_points(level, EdgePointsView{static_cast<std::uint32_t>(count), xy.data(), gradients.data()});
    probe->vote(level,
                VoteView{links.data(),
                         votersOffsets.data(),
                         votersValues.data(),
                         isMax.data(),
                         flowLength.data(),
                         static_cast<std::uint32_t>(seedIndices.size()),
                         seedIndices.data()});
}

void probeLinking(Probe* probe,
                  std::uint32_t level,
                  const EdgePointCollection& edgeCollection,
                  const std::vector<std::unique_ptr<Candidate>>& candidates)
{
    std::vector<std::int32_t> seeds;
    std::vector<std::int32_t> segmentOffsets;
    std::vector<std::int32_t> segmentValues;
    std::vector<std::int32_t> childCounts;
    std::vector<float> averageVote;

    seeds.reserve(candidates.size());
    segmentOffsets.reserve(candidates.size() + 1);
    childCounts.reserve(candidates.size());
    averageVote.reserve(candidates.size());
    segmentOffsets.push_back(0);

    for(const std::unique_ptr<Candidate>& candidate : candidates)
    {
        seeds.push_back(edgeCollection(candidate->_seed));
        for(const EdgePoint* point : candidate->_convexEdgeSegment)
        {
            segmentValues.push_back(edgeCollection(point));
        }
        segmentOffsets.push_back(static_cast<std::int32_t>(segmentValues.size()));

        std::list<EdgePoint*> children;
        childrenOf(edgeCollection, candidate->_convexEdgeSegment, children);
        childCounts.push_back(static_cast<std::int32_t>(children.size()));
        averageVote.push_back(candidate->_averageReceivedVote);
    }

    probe->linking(level,
                   LinkingView{static_cast<std::uint32_t>(candidates.size()),
                               seeds.data(),
                               segmentOffsets.data(),
                               segmentValues.data(),
                               childCounts.data(),
                               averageVote.data()});
}

void probeCandidates(Probe* probe, const CCTag::List& candidates)
{
    std::vector<float> ellipses;
    std::vector<std::int32_t> levels;
    std::vector<float> quality;

    ellipses.reserve(candidates.size() * 5);
    levels.reserve(candidates.size());
    quality.reserve(candidates.size());
    for(const CCTag& candidate : candidates)
    {
        const numerical::geometry::Ellipse& ellipse = candidate.rescaledOuterEllipse();
        ellipses.push_back(ellipse.center().x());
        ellipses.push_back(ellipse.center().y());
        ellipses.push_back(ellipse.a());
        ellipses.push_back(ellipse.b());
        ellipses.push_back(ellipse.angle());
        levels.push_back(candidate.pyramidLevel());
        quality.push_back(candidate.quality());
    }

    probe->candidates(
      CandidatesView{static_cast<std::uint32_t>(candidates.size()), ellipses.data(), levels.data(), quality.data()});
}

void probeMarkers(Probe* probe, const CCTag::List& markers)
{
    std::vector<float> xy;
    std::vector<std::int32_t> ids;
    std::vector<std::int32_t> statuses;

    xy.reserve(markers.size() * 2);
    ids.reserve(markers.size());
    statuses.reserve(markers.size());
    for(const CCTag& marker : markers)
    {
        xy.push_back(marker.x());
        xy.push_back(marker.y());
        ids.push_back(marker.id());
        statuses.push_back(marker.getStatus());
    }

    probe->markers(MarkersView{static_cast<std::uint32_t>(markers.size()), xy.data(), ids.data(), statuses.data()});
}

} // namespace cctag
