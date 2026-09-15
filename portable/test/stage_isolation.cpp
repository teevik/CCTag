/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// Run each CPU pipeline stage on reference inputs and compare its output with the reference
#include "backends/cpu/backend.hpp"
#include "host/context.hpp"
#include "kernels/plane.hpp"
#include "support/reference_snapshot.hpp"

#include <cctag/ICCTag.hpp>

#include <boost/ut.hpp>

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
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

namespace cctag { void prototypeReleaseContexts(); }

int main(int argc, const char** argv) {
    if (!reference_snapshots_dir()) {
        std::cout << "CCTAG_REFERENCE_SNAPSHOTS is not set: enter `nix develop` or point it "
                     "at the reference-snapshot store\n";
        return 77; // Tell CTest to skip when the reference snapshot directory is unset
    }

    const suite<"stage_isolation"> stage_isolation_suite = [] {
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
                const cctag::Parameters params(snapshot.crowns());
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
        "markers are exact across thread counts and reused contexts"_test = [] {
            const int threads = omp_get_max_threads();
            Context<cpu::Backend> reused;
            for (const auto& file : snapshot_files_or_fail()) {
                const ReferenceSnapshot snapshot = ReferenceSnapshot::read(file);
                const cctag::Parameters params(snapshot.crowns());
                Context<cpu::Backend> fresh;
                fill_context(snapshot, Stage::linking, fresh);
                omp_set_num_threads(1);
                cpu::Backend::candidates(fresh, params);
                cpu::Backend::markers(fresh, params);
                const MarkersHost expected = host_markers(fresh);
                expect(std::ranges::find(expected.statuses, 1) != expected.statuses.end())
                    << snapshot.problem() << "must exercise reliable identification";
                for (const int count : {1, 3, threads}) {
                    fill_context(snapshot, Stage::linking, reused);
                    reused.candidate_markers = fresh.candidate_markers;
                    omp_set_num_threads(count);
                    cpu::Backend::markers(reused, params);
                    const MarkersHost actual = host_markers(reused);
                    expect(std::ranges::equal(expected.xy, actual.xy))
                        << snapshot.problem() << "centers at" << count << "threads";
                    expect(std::ranges::equal(expected.ids, actual.ids))
                        << snapshot.problem() << "ids";
                    expect(std::ranges::equal(expected.statuses, actual.statuses))
                        << snapshot.problem() << "statuses";
                    expect(eq(fresh.markers.size(), reused.markers.size())) << fatal;
                    for (std::size_t i = 0; i < fresh.markers.size(); ++i) {
                        expect(fresh.markers[i].homography == reused.markers[i].homography)
                            << snapshot.problem() << "homography";
                        expect(eq(fresh.markers[i].quality, reused.markers[i].quality))
                            << snapshot.problem() << "quality";
                    }
                }
            }
            omp_set_num_threads(threads);
        };
        "candidate markers are exact across thread counts and reused contexts"_test = [] {
            const int threads = omp_get_max_threads();
            Context<cpu::Backend> reused;
            for (const auto& file : snapshot_files_or_fail()) {
                const ReferenceSnapshot snapshot = ReferenceSnapshot::read(file);
                const cctag::Parameters params(snapshot.crowns());
                Context<cpu::Backend> fresh;
                fill_context(snapshot, Stage::linking, fresh);
                omp_set_num_threads(1);
                cpu::Backend::candidates(fresh, params);
                const CandidatesHost expected = host_candidates(fresh);
                for (const int count : {1, 3, threads}) {
                    fill_context(snapshot, Stage::linking, reused);
                    omp_set_num_threads(count);
                    cpu::Backend::candidates(reused, params);
                    const CandidatesHost actual = host_candidates(reused);
                    expect(std::ranges::equal(expected.ellipses, actual.ellipses))
                        << snapshot.problem() << "ellipses at" << count << "threads";
                    expect(std::ranges::equal(expected.levels, actual.levels))
                        << snapshot.problem() << "levels";
                    expect(std::ranges::equal(expected.quality, actual.quality))
                        << snapshot.problem() << "quality";
                    expect(eq(fresh.candidate_markers.size(), reused.candidate_markers.size()))
                        << fatal;
                    for (std::size_t i = 0; i < fresh.candidate_markers.size(); ++i) {
                        const auto& a = fresh.candidate_markers[i].outer_points;
                        const auto& b = reused.candidate_markers[i].outer_points;
                        expect(eq(a.size(), b.size()))
                            << snapshot.problem() << "outer point count" << fatal;
                        expect(
                            std::equal(
                                a.begin(),
                                a.end(),
                                b.begin(),
                                [](const auto& left, const auto& right) {
                            return left.x == right.x && left.y == right.y && left.dx == right.dx
                                && left.dy == right.dy;
                        }
                            )
                        ) << snapshot.problem()
                          << "directed outer points";
                    }
                }
            }
            omp_set_num_threads(threads);
        };
        "candidates match reference snapshot from reference linking and level zero edges"_test =
            [] {
            for (const auto& file : snapshot_files_or_fail()) {
                const ReferenceSnapshot snapshot = ReferenceSnapshot::read(file);
                Context<cpu::Backend> context;
                fill_context(snapshot, Stage::linking, context);
                const cctag::Parameters params(snapshot.crowns());
                cpu::Backend::candidates(context, params);
                const auto ellipses = snapshot.tensor("candidates/ellipse").as<float>();
                const auto levels = snapshot.tensor("candidates/level").as<std::int32_t>();
                const auto quality = snapshot.tensor("candidates/quality").as<float>();
                const auto comparison = compare_candidates(
                    {static_cast<std::uint32_t>(levels.size()), ellipses, levels, quality},
                    host_candidates(context)
                );
                bool accepted = comparison.passed;
                if (const char* file = std::getenv("CCTAG_CANDIDATE_ALLOWANCE");
                    !accepted && file) {
                    const auto allowance = CandidateAllowance::read(file);
                    accepted = allowance.allows(snapshot, comparison, "fork/cpu");
                    if (accepted) {
                        std::cout << snapshot.problem() << ": " << describe(comparison)
                                  << "; accepted: " << allowance.reason << '\n';
                    }
                }
                expect(accepted) << snapshot.problem() << describe(comparison);
            }
        };
        "pyramid matches reference snapshot from each reference finer level"_test = [] {
            for (const auto& file : snapshot_files_or_fail()) {
                const ReferenceSnapshot snapshot = ReferenceSnapshot::read(file);
                Context<cpu::Backend> context;
                fill_context(snapshot, Stage::pyramid, context);
                auto& levels = context.levels;

                // Check that loading the reference image preserves its bytes at level 0
                const Tensor& image = snapshot.tensor(Stage::pyramid, 0, "src");
                const std::vector<std::uint8_t> pixels = image.as<std::uint8_t>();
                cpu::Backend::load(
                    levels[0],
                    {pixels.data(),
                     snapshot.image_width(),
                     snapshot.image_height(),
                     snapshot.image_width()}
                );
                const Mismatch loaded =
                    compare_plane<std::uint8_t>(image, levels[0].src_plane().as_const());
                expect(loaded.exact()) << snapshot.problem() << describe(image, loaded);

                for (std::uint32_t level = 1; level < levels.size(); ++level) {
                    // Restore the finer level's reference input so resize errors cannot accumulate
                    fill_level(snapshot, level - 1, Stage::pyramid, levels[level - 1]);
                    cpu::Backend::pyramid(levels[level], levels[level - 1]);
                    const Tensor& expected = snapshot.tensor(Stage::pyramid, level, "src");
                    const Mismatch mismatch =
                        compare_plane<std::uint8_t>(expected, levels[level].src_plane().as_const());
                    expect(mismatch.exact()) << snapshot.problem() << describe(expected, mismatch);
                }
            }
        };

        "gradient matches reference snapshot from reference pyramid planes"_test = [] {
            for (const auto& file : snapshot_files_or_fail()) {
                const ReferenceSnapshot snapshot = ReferenceSnapshot::read(file);
                Context<cpu::Backend> context;
                fill_context(snapshot, Stage::pyramid, context);
                auto& levels = context.levels;
                for (std::uint32_t level = 0; level < levels.size(); ++level) {
                    cpu::Backend::gradient(levels[level]);
                    const Tensor& dx = snapshot.tensor(Stage::gradient, level, "dx");
                    const Tensor& dy = snapshot.tensor(Stage::gradient, level, "dy");
                    const Mismatch dx_mismatch =
                        compare_plane<std::int16_t>(dx, levels[level].dx_plane().as_const());
                    const Mismatch dy_mismatch =
                        compare_plane<std::int16_t>(dy, levels[level].dy_plane().as_const());
                    expect(dx_mismatch.exact()) << snapshot.problem() << describe(dx, dx_mismatch);
                    expect(dy_mismatch.exact()) << snapshot.problem() << describe(dy, dy_mismatch);
                }
            }
        };
        "edges match reference snapshot from reference gradient planes"_test = [] {
            for (const auto& file : snapshot_files_or_fail()) {
                const ReferenceSnapshot snapshot = ReferenceSnapshot::read(file);
                Context<cpu::Backend> context;
                fill_context(snapshot, Stage::gradient, context);
                const cctag::Parameters params(snapshot.crowns());
                auto& levels = context.levels;
                for (std::uint32_t level = 0; level < levels.size(); ++level) {
                    cpu::Backend::edges(levels[level], params);
                    const Tensor& edges = snapshot.tensor(Stage::edges, level, "edges");
                    const Mismatch mismatch =
                        compare_plane<std::uint8_t>(edges, levels[level].edges_plane().as_const());
                    expect(mismatch.exact()) << snapshot.problem() << describe(edges, mismatch);
                }
            }
        };
        "edge points match reference snapshot from reference edges"_test = [] {
            for (const auto& file : snapshot_files_or_fail()) {
                const ReferenceSnapshot snapshot = ReferenceSnapshot::read(file);
                Context<cpu::Backend> context;
                fill_context(snapshot, Stage::edges, context);
                auto& levels = context.levels;
                for (std::uint32_t level = 0; level < levels.size(); ++level) {
                    cpu::Backend::edge_points(levels[level]);
                    const EdgePointsHost points = cpu::Backend::host_edge_points(levels[level]);
                    const Tensor& xy = snapshot.tensor(Stage::edge_points, level, "xy");
                    const Tensor& gradients =
                        snapshot.tensor(Stage::edge_points, level, "gradients");
                    expect(eq(points.n, xy.shape[0])) << snapshot.problem() << "level" << level;
                    const Mismatch xy_mismatch = compare_values<std::int32_t>(xy, points.xy);
                    const Mismatch gradient_mismatch =
                        compare_values<float>(gradients, points.gradients);
                    expect(xy_mismatch.exact()) << snapshot.problem() << describe(xy, xy_mismatch);
                    expect(gradient_mismatch.exact())
                        << snapshot.problem() << describe(gradients, gradient_mismatch);
                }
            }
        };
        "linking matches reference snapshot from reference vote"_test = [] {
            for (const auto& file : snapshot_files_or_fail()) {
                const ReferenceSnapshot snapshot = ReferenceSnapshot::read(file);
                Context<cpu::Backend> context;
                fill_context(snapshot, Stage::vote, context);
                const cctag::Parameters params(snapshot.crowns());
                for (std::uint32_t level = 0; level < context.levels.size(); ++level) {
                    cpu::Backend::linking(context.levels[level], params);
                    const LinkingHost linking = cpu::Backend::host_linking(context.levels[level]);
                    const auto compare = [&](const char* name, auto values) {
                        const Tensor& expected = snapshot.tensor(Stage::linking, level, name);
                        const Mismatch mismatch = compare_values(expected, values);
                        expect(mismatch.exact())
                            << snapshot.problem() << describe(expected, mismatch);
                    };
                    compare("seeds", linking.seeds);
                    compare("segments/offsets", linking.segment_offsets);
                    compare("segments/values", linking.segment_values);
                    compare("child_counts", linking.child_counts);
                    compare("avg_vote", linking.avg_vote);
                }
            }
        };
        "vote matches reference snapshot from reference edge points"_test = [] {
            for (const auto& file : snapshot_files_or_fail()) {
                const ReferenceSnapshot snapshot = ReferenceSnapshot::read(file);
                Context<cpu::Backend> context;
                fill_context(snapshot, Stage::edge_points, context);
                const cctag::Parameters params(snapshot.crowns());
                for (std::uint32_t level = 0; level < context.levels.size(); ++level) {
                    cpu::Backend::vote(context.levels[level], params);
                    const VoteHost vote = cpu::Backend::host_vote(context.levels[level]);
                    const auto compare = [&](const char* name, auto values) {
                        const Tensor& expected = snapshot.tensor(Stage::vote, level, name);
                        const Mismatch mismatch = compare_values(expected, values);
                        expect(mismatch.exact())
                            << snapshot.problem() << describe(expected, mismatch);
                    };
                    compare("links", vote.links);
                    compare("voters/offsets", vote.voters_offsets);
                    compare("voters/values", vote.voters_values);
                    compare("is_max", vote.is_max);
                    compare("flow_length", vote.flow_length);
                    compare("seeds", vote.seeds);
                    compare("seed_order", vote.seed_order);
                }
            }
        };
    };
    const auto status = cfg<override>.run({.argc = argc, .argv = argv});
    // THROWAWAY: consumers explicitly release Contexts before runtime teardown.
    cctag::prototypeReleaseContexts();
    return status;
}
