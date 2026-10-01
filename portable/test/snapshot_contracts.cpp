/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// Public API and reuse contracts exercised with retained CUDA captures.
// Stage outputs and self-consistency across reuse and thread counts are checked by the
// snapshot comparator on portable_cuda_isolation's output, not here.
#include "backends/cpu/backend.hpp"
#include "host/context.hpp"
#include "kernels/plane.hpp"
#include "support/reference_snapshot.hpp"

#include <cctag/ICCTag.hpp>

#include <boost/ut.hpp>

#include <cstdint>
#include <iostream>
#include <vector>

using namespace boost::ut;
using namespace cctag::portable;
using namespace cctag::portable::test;

namespace {

std::vector<std::filesystem::path> snapshot_files_or_fail() {
    const std::vector<std::filesystem::path> files = reference_snapshot_files();
    expect(!files.empty()) << "no *.safetensors in " << reference_snapshots_dir()->string()
                           << fatal;
    return files;
}

} // namespace

int main(int argc, const char** argv) {
    if (!reference_snapshots_dir()) {
        std::cout << "CCTAG_REFERENCE_SNAPSHOTS is not set: enter `nix develop` or point it "
                     "at the reference-snapshot store\n";
        return 77; // Tell CTest to skip when the reference snapshot directory is unset
    }

    const suite<"snapshot_contracts"> snapshot_contracts_suite = [] {
        "public detection candidates and their clones retain the observed marker results"_test =
            [] {
            struct MarkerProbe : cctag::Probe {
                std::vector<Eigen::Vector2f> centers;
                std::vector<std::int32_t> ids;
                std::vector<std::int32_t> statuses;
                void markers(const cctag::MarkersView& view) override {
                    for (std::uint32_t i = 0; i < view.candidate_count; ++i) {
                        centers.emplace_back(
                            view.positions_xy[2 * i],
                            view.positions_xy[2 * i + 1]
                        );
                        ids.push_back(view.marker_ids[i]);
                        statuses.push_back(view.identification_statuses[i]);
                    }
                }
            };
            for (const auto& file : snapshot_files_or_fail()) {
                const ReferenceSnapshot snapshot = ReferenceSnapshot::read(file);
                auto pixels = snapshot.tensor(Stage::pyramid, 0, "src").as<std::uint8_t>();
                const cv::Mat1b image(
                    snapshot.image_height(),
                    snapshot.image_width(),
                    pixels.data()
                );
                const cctag::Parameters params = reference_parameters(snapshot);
                boost::ptr_list<cctag::ICCTag> detections;
                MarkerProbe probe;
                cctag::cctagDetection(detections, 77, 0, image, params, nullptr, nullptr, &probe);
                expect(!detections.empty()) << snapshot.problem() << fatal;
                expect(eq(detections.size(), probe.ids.size())) << fatal;
                boost::ptr_list<cctag::ICCTag> clones(detections);
                // Reuse the pipe and clear its results while the API's clones stay alive
                cctag::cctagDetection(
                    detections,
                    77,
                    1,
                    cv::Mat1b(32, 32, std::uint8_t{0}),
                    params
                );
                expect(detections.empty());
                std::size_t i = 0;
                for (const auto& clone : clones) {
                    expect(eq(clone.id(), probe.ids[i]));
                    expect(eq(clone.getStatus(), probe.statuses[i]));
                    expect(eq(clone.x(), probe.centers[i].x()));
                    expect(eq(clone.y(), probe.centers[i].y()));
                    expect(clone.rescaledOuterEllipse().a() > 0);
                    expect(clone.rescaledOuterEllipse().b() > 0);
                    ++i;
                }
            }
        };
        "a smaller candidate run exposes none of a larger previous run's candidates"_test = [] {
            for (const auto& file : snapshot_files_or_fail()) {
                const ReferenceSnapshot snapshot = ReferenceSnapshot::read(file);
                cctag::Parameters full_params = reference_parameters(snapshot);
                // Unidentified candidate markers are not deduplicated, so each active one
                // must produce exactly one detection candidate.
                full_params._doIdentification = false;
                Context<cpu::Backend> context;
                fill_context(snapshot, Stage::linking, context);

                // Each limited run follows an unlimited one, so the context's storage holds
                // more candidate markers than the limited run produces
                for (const std::size_t max_candidates_per_level : {1u, 0u}) {
                    cpu::Backend::candidates(context, full_params);
                    expect(!context.candidate_markers.view().empty())
                        << snapshot.problem() << fatal;

                    cctag::Parameters limited_params = full_params;
                    limited_params._maximumNbCandidatesLoopTwo = max_candidates_per_level;
                    cpu::Backend::candidates(context, limited_params);
                    const auto active_count = context.candidate_markers.view().size();
                    expect(le(active_count, context.levels.size() * max_candidates_per_level))
                        << snapshot.problem();

                    const CandidatesHost view = host_candidates(context);
                    expect(eq(view.n, active_count));
                    expect(eq(view.ellipses.size(), 5 * active_count));
                    expect(eq(view.levels.size(), active_count));
                    expect(eq(view.quality.size(), active_count));

                    cpu::Backend::markers(context, limited_params);
                    expect(eq(context.markers.size(), active_count));
                }
            }
        };
    };
    return cfg<override>.run({.argc = argc, .argv = argv});
}
