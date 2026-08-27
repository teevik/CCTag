/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PROBE_LEGACY_HPP
#define CCTAG_PROBE_LEGACY_HPP

#include <cctag/CCTag.hpp>
#include <cctag/Candidate.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace cctag {

class EdgePoint;
class EdgePointCollection;
class ImagePyramid;
class Probe;

void probePyramid(Probe* probe, const ImagePyramid& imagePyramid);

void probeEdgePointsAndVote(Probe* probe,
                            std::uint32_t level,
                            EdgePointCollection& edgeCollection,
                            const std::vector<EdgePoint*>& seeds);

void probeLinking(Probe* probe,
                  std::uint32_t level,
                  const EdgePointCollection& edgeCollection,
                  const std::vector<std::unique_ptr<Candidate>>& candidates);

void probeCandidates(Probe* probe, const CCTag::List& candidates);
void probeMarkers(Probe* probe, const CCTag::List& markers);

} // namespace cctag

#endif
