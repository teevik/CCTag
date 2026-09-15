/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_HOST_DETECT_HPP
#define CCTAG_PORTABLE_HOST_DETECT_HPP

#include "host/backend.hpp"
#include "host/context.hpp"
#include "host/views.hpp"
#include "kernels/plane.hpp"

#include <cctag/Params.hpp>
#include <cctag/Probe.hpp>

#include <cstdint>
#include <cstdlib>

namespace cctag::portable {

/// Times a stage using the probe's `enter` and `leave` callbacks
/// Waits for the execution backend to finish before calling `leave`
template <ExecutionBackend Backend>
class StageTiming {
  public:
    StageTiming(Context<Backend>& context, Probe* probe, const char* stage) :
        context(context),
        probe(std::getenv("PROTOTYPE_SYCL_UNTIMED_SNAPSHOT") ? nullptr : probe),
        stage(stage) {
        if (this->probe) {
            this->probe->enter(stage);
        }
    }
    void finish() {
        if (probe) {
            Backend::wait(context);
            probe->leave(stage);
        }
    }
    StageTiming(const StageTiming&) = delete;
    StageTiming& operator=(const StageTiming&) = delete;

  private:
    Context<Backend>& context;
    Probe* probe;
    const char* stage;
};

/// Runs the portable pipeline's stages across all pyramid levels
template <ExecutionBackend Backend>
void detect(
    Context<Backend>& context,
    kernels::Plane<const std::uint8_t> input,
    const Parameters& params,
    Probe* probe
) {
    context.ensure(input.width, input.height, params);
    try {
    auto& levels = context.levels;
    const std::uint32_t count = static_cast<std::uint32_t>(levels.size());

    // Load level 0 and build each coarser pyramid level
    {
        StageTiming<Backend> timing(context, probe, "pyramid");
        Backend::load(levels[0], input);
        for (std::uint32_t level = 1; level < count; ++level) {
            Backend::pyramid(levels[level], levels[level - 1]);
        }
        timing.finish();
    }
    // Compute gradients at every pyramid level
    {
        StageTiming<Backend> timing(context, probe, "gradient");
        for (std::uint32_t level = 0; level < count; ++level) {
            Backend::gradient(levels[level]);
        }
        timing.finish();
    }

    // Find and thin edges at every pyramid level
    {
        StageTiming<Backend> timing(context, probe, "edges");
        for (std::uint32_t level = 0; level < count; ++level) {
            Backend::edges(levels[level], params);
        }
        timing.finish();
    }

    // Collect edge points in canonical order at every pyramid level
    {
        StageTiming<Backend> timing(context, probe, "edge_points");
        for (std::uint32_t level = 0; level < count; ++level) {
            Backend::edge_points(levels[level]);
        }
        timing.finish();
    }

    // Link edge points and gather their votes at every pyramid level
    {
        StageTiming<Backend> timing(context, probe, "vote");
        for (std::uint32_t level = 0; level < count; ++level) {
            Backend::vote(levels[level], params);
        }
        timing.finish();
    }

    // Walk seeds and gather segments at every pyramid level
    {
        StageTiming<Backend> timing(context, probe, "linking");
        for (std::uint32_t level = 0; level < count; ++level) {
            Backend::linking(levels[level], params);
        }
        timing.finish();
    }

    // Observe host views after all stages, outside the stage timings
    if (probe && probe->observes_stages()) {
        for (std::uint32_t level = 0; level < count; ++level) {
            const PyramidHost pyramid = Backend::host_pyramid(levels[level]);
            probe->pyramid(level, probe_plane(pyramid.src));
            const GradientHost gradient = Backend::host_gradient(levels[level]);
            probe->gradient(level, probe_plane(gradient.dx), probe_plane(gradient.dy));
            const EdgesHost edges = Backend::host_edges(levels[level]);
            probe->edges(level, probe_plane(edges.edges));
            const EdgePointsHost points = Backend::host_edge_points(levels[level]);
            probe->edge_points(
                level,
                cctag::EdgePointsView{points.n, points.xy.data(), points.gradients.data()}
            );
            const VoteHost vote = Backend::host_vote(levels[level]);
            probe->vote(
                level,
                cctag::VoteView{
                    vote.links.data(),
                    vote.voters_offsets.data(),
                    vote.voters_values.data(),
                    vote.is_max.data(),
                    vote.flow_length.data(),
                    static_cast<std::uint32_t>(vote.seed_order.size()),
                    vote.seed_order.data()
                }
            );
            const LinkingHost linking = Backend::host_linking(levels[level]);
            probe->linking(
                level,
                cctag::LinkingView{
                    linking.c,
                    linking.seeds.data(),
                    linking.segment_offsets.data(),
                    linking.segment_values.data(),
                    linking.child_counts.data(),
                    linking.avg_vote.data()
                }
            );
        }
    }

    // Fit candidate markers across levels and project their outer ellipses to level zero
    {
        StageTiming<Backend> timing(context, probe, "candidates");
        Backend::candidates(context, params);
        timing.finish();
    }
    if (probe && probe->observes_stages()) {
        const CandidatesHost candidates = host_candidates(context);
        probe->candidates(
            {candidates.n,
             candidates.ellipses.data(),
             candidates.levels.data(),
             candidates.quality.data()}
        );
    }
    {
        StageTiming<Backend> timing(context, probe, "markers");
        Backend::markers(context, params);
        timing.finish();
    }
    if (probe && probe->observes_stages()) {
        const MarkersHost markers = host_markers(context);
        probe->markers({markers.n, markers.xy.data(), markers.ids.data(), markers.statuses.data()});
    }
    Backend::wait(context);
    } catch (...) {
        if constexpr (requires { Backend::invalidate(context); }) Backend::invalidate(context);
        throw;
    }
}

} // namespace cctag::portable

#ifdef CCTAG_TEST
#include "backends/cpu/backend.hpp"

#include <cctag/Params.hpp>
#include <cctag/Probe.hpp>

#include <boost/ut.hpp>

#include <omp.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace cctag::portable::tests::host_sequence {

using namespace boost::ut;

/// Copies observed planes into `tensors` and records timing events in `timing`
struct RecordingProbe : cctag::Probe {
    std::map<std::string, std::vector<std::uint8_t>> tensors;
    std::vector<std::string> timing;

    void record(const std::string& name, const cctag::Plane& plane, std::size_t element_size) {
        auto& bytes = tensors[name];
        bytes.resize(static_cast<std::size_t>(plane.width) * plane.height * element_size);
        for (std::uint32_t y = 0; y < plane.height; ++y) {
            std::memcpy(
                bytes.data() + static_cast<std::size_t>(y) * plane.width * element_size,
                static_cast<const std::uint8_t*>(plane.data) + y * plane.stride_bytes,
                static_cast<std::size_t>(plane.width) * element_size
            );
        }
    }

    void pyramid(std::uint32_t level, const cctag::Plane& src) override {
        record("pyramid/level" + std::to_string(level) + "/src", src, 1);
    }

    void gradient(std::uint32_t level, const cctag::Plane& dx, const cctag::Plane& dy) override {
        record("gradient/level" + std::to_string(level) + "/dx", dx, 2);
        record("gradient/level" + std::to_string(level) + "/dy", dy, 2);
    }

    void edges(std::uint32_t level, const cctag::Plane& edges) override {
        record("edges/level" + std::to_string(level) + "/edges", edges, 1);
    }

    void edge_points(std::uint32_t level, const cctag::EdgePointsView& points) override {
        record(
            "edge_points/level" + std::to_string(level) + "/xy",
            {2, points.point_count, 2 * sizeof(std::int32_t), points.positions_xy},
            sizeof(std::int32_t)
        );
        record(
            "edge_points/level" + std::to_string(level) + "/gradients",
            {2, points.point_count, 2 * sizeof(float), points.gradients},
            sizeof(float)
        );
    }

    void vote(std::uint32_t level, const cctag::VoteView& vote) override {
        const auto n = static_cast<std::uint32_t>(
            tensors.at("edge_points/level" + std::to_string(level) + "/xy").size()
            / (2 * sizeof(std::int32_t))
        );
        const auto prefix = "vote/level" + std::to_string(level) + "/";
        record(
            prefix + "links",
            {2, n, 2 * sizeof(std::int32_t), vote.linked_point_indices},
            sizeof(std::int32_t)
        );
        record(
            prefix + "voters/offsets",
            {1, n + 1, sizeof(std::int32_t), vote.voter_offsets},
            sizeof(std::int32_t)
        );
        record(
            prefix + "voters/values",
            {1,
             static_cast<std::uint32_t>(vote.voter_offsets[n]),
             sizeof(std::int32_t),
             vote.voter_point_indices},
            sizeof(std::int32_t)
        );
        record(
            prefix + "is_max",
            {1, n, sizeof(std::int32_t), vote.seed_vote_counts},
            sizeof(std::int32_t)
        );
        record(
            prefix + "flow_length",
            {1, n, sizeof(float), vote.mean_flow_lengths},
            sizeof(float)
        );
        record(
            prefix + "seed_order",
            {1, vote.seed_count, sizeof(std::int32_t), vote.seed_point_indices},
            sizeof(std::int32_t)
        );
    }

    void linking(std::uint32_t level, const cctag::LinkingView& linking) override {
        const auto n = linking.segment_count;
        const auto prefix = "linking/level" + std::to_string(level) + "/";
        record(
            prefix + "seeds",
            {1, n, sizeof(std::int32_t), linking.seed_point_indices},
            sizeof(std::int32_t)
        );
        record(
            prefix + "segments/offsets",
            {1, n + 1, sizeof(std::int32_t), linking.segment_offsets},
            sizeof(std::int32_t)
        );
        record(
            prefix + "segments/values",
            {1,
             static_cast<std::uint32_t>(linking.segment_offsets[n]),
             sizeof(std::int32_t),
             linking.segment_point_indices},
            sizeof(std::int32_t)
        );
        record(
            prefix + "child_counts",
            {1, n, sizeof(std::int32_t), linking.child_point_counts},
            sizeof(std::int32_t)
        );
        record(prefix + "avg_vote", {1, n, sizeof(float), linking.vote_scores}, sizeof(float));
    }

    void enter(const char* stage) override {
        timing.push_back(std::string("enter ") + stage);
    }

    void candidates(const cctag::CandidatesView& candidates) override {
        record(
            "candidates/ellipse",
            {5, candidates.candidate_count, 5 * sizeof(float), candidates.outer_ellipse_parameters},
            sizeof(float)
        );
        record(
            "candidates/level",
            {1, candidates.candidate_count, sizeof(std::int32_t), candidates.pyramid_levels},
            sizeof(std::int32_t)
        );
        record(
            "candidates/quality",
            {1, candidates.candidate_count, sizeof(float), candidates.quality_scores},
            sizeof(float)
        );
    }

    void markers(const cctag::MarkersView& markers) override {
        record(
            "markers/xy",
            {2, markers.candidate_count, 2 * sizeof(float), markers.positions_xy},
            sizeof(float)
        );
        record(
            "markers/id",
            {1, markers.candidate_count, sizeof(std::int32_t), markers.marker_ids},
            sizeof(std::int32_t)
        );
        record(
            "markers/status",
            {1, markers.candidate_count, sizeof(std::int32_t), markers.identification_statuses},
            sizeof(std::int32_t)
        );
    }

    void leave(const char* stage) override {
        timing.push_back(std::string("leave ") + stage);
    }
};

/// Creates a test image with rings and noise to vary the pixel values
inline std::vector<std::uint8_t>
test_image(std::uint32_t width, std::uint32_t height, std::uint32_t seed = 12345u) {
    // Ring width in squared pixels and alternating brightness values
    constexpr std::uint32_t squared_radius_step = 37;
    constexpr std::uint32_t dark_intensity = 40;
    constexpr std::uint32_t bright_intensity = 200;

    std::mt19937 random(seed);
    std::uniform_int_distribution<int> noise(0, 31); // Keep brightness plus noise within a byte

    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            // Alternate dark and bright rings around the top-left corner
            // Using squared distance makes the rings narrower farther from the corner
            const std::uint32_t radius_squared = x * x + y * y;
            const std::uint32_t ring_index = radius_squared / squared_radius_step;
            const std::uint32_t ring_intensity =
                ring_index % 2 == 0 ? dark_intensity : bright_intensity;

            pixels[static_cast<std::size_t>(y) * width + x] =
                static_cast<std::uint8_t>(ring_intensity + noise(random));
        }
    }
    return pixels;
}

template <ExecutionBackend Backend>
RecordingProbe
run(Context<Backend>& context,
    const std::vector<std::uint8_t>& image,
    std::uint32_t width,
    std::uint32_t height) {
    RecordingProbe probe;
    const cctag::Parameters params(3);
    detect(
        context,
        kernels::Plane<const std::uint8_t>{image.data(), width, height, width},
        params,
        &probe
    );
    return probe;
}

inline void expect_identical(const RecordingProbe& a, const RecordingProbe& b) {
    expect(eq(a.tensors.size(), b.tensors.size())) << fatal;
    for (const auto& [name, bytes] : a.tensors) {
        const auto other = b.tensors.find(name);
        expect(other != b.tensors.end()) << name << " missing" << fatal;
        expect(bytes == other->second) << name << " differs";
    }
}

template <ExecutionBackend Backend>
void expect_thread_count_does_not_change_stage_outputs() {
    const std::uint32_t w = 301, h = 173;
    const auto image = test_image(w, h);
    Context<Backend> context;
    omp_set_num_threads(1);
    const RecordingProbe single = run(context, image, w, h);
    omp_set_num_threads(3);
    expect_identical(single, run(context, image, w, h));
    omp_set_num_threads(omp_get_num_procs());
    const RecordingProbe many = run(context, image, w, h);
    expect_identical(single, many);
}

inline suite<"host_sequence"> host_sequence_suite = [] {
    "load copies the input into level zero"_test = [] {
        const std::uint32_t w = 37, h = 23;
        const auto image = test_image(w, h);
        Context<cpu::Backend> context;
        const RecordingProbe probe = run(context, image, w, h);
        expect(probe.tensors.at("pyramid/level0/src") == image);
    };

    "probe observes per level stages and whole image candidates and markers"_test = [] {
        const std::uint32_t w = 37, h = 23;
        Context<cpu::Backend> context;
        const RecordingProbe probe = run(context, test_image(w, h), w, h);
        // Seventeen tensors at each level, then three each for candidates and markers
        expect(eq(probe.tensors.size(), 74u));
        for (const auto* name : {"ellipse", "level", "quality"}) {
            expect(eq(probe.tensors.count(std::string("candidates/") + name), 1u)) << name;
        }
        for (const auto* name : {"xy", "id", "status"}) {
            expect(eq(probe.tensors.count(std::string("markers/") + name), 1u)) << name;
        }
        for (std::uint32_t level = 0; level < context.levels.size(); ++level) {
            expect(eq(probe.tensors.count("pyramid/level" + std::to_string(level) + "/src"), 1u))
                << "src at level" << level;
            expect(eq(probe.tensors.count("gradient/level" + std::to_string(level) + "/dx"), 1u))
                << "dx at level" << level;
            expect(eq(probe.tensors.count("gradient/level" + std::to_string(level) + "/dy"), 1u))
                << "dy at level" << level;
            expect(eq(probe.tensors.count("edges/level" + std::to_string(level) + "/edges"), 1u))
                << "edges at level" << level;
            expect(eq(probe.tensors.count("edge_points/level" + std::to_string(level) + "/xy"), 1u))
                << "edge point coordinates at level" << level;
            expect(
                eq(probe.tensors.count("edge_points/level" + std::to_string(level) + "/gradients"),
                   1u)
            ) << "edge point gradients at level"
              << level;
            for (const auto* name :
                 {"links",
                  "voters/offsets",
                  "voters/values",
                  "is_max",
                  "flow_length",
                  "seed_order"}) {
                expect(
                    eq(probe.tensors.count("vote/level" + std::to_string(level) + "/" + name), 1u)
                ) << name
                  << "at level" << level;
            }
            for (const auto* name :
                 {"seeds", "segments/offsets", "segments/values", "child_counts", "avg_vote"}) {
                expect(
                    eq(probe.tensors.count("linking/level" + std::to_string(level) + "/" + name),
                       1u)
                ) << name
                  << "at level" << level;
            }
        }
    };

    "probe receives timing events in stage order through markers"_test = [] {
        const std::uint32_t w = 37, h = 23;
        Context<cpu::Backend> context;
        const RecordingProbe probe = run(context, test_image(w, h), w, h);
        const std::vector<std::string> expected = {
            "enter pyramid",
            "leave pyramid",
            "enter gradient",
            "leave gradient",
            "enter edges",
            "leave edges",
            "enter edge_points",
            "leave edge_points",
            "enter vote",
            "leave vote",
            "enter linking",
            "leave linking",
            "enter candidates",
            "leave candidates",
            "enter markers",
            "leave markers"
        };
        expect(probe.timing == expected);
    };

    "context reuse does not change stage outputs"_test = [] {
        const std::uint32_t w = 64, h = 48;
        const auto first = test_image(w, h, 1u);
        const auto second = test_image(w, h, 2u);
        Context<cpu::Backend> fresh;
        const RecordingProbe expected = run(fresh, first, w, h);
        Context<cpu::Backend> reused;
        run(reused, first, w, h);
        run(reused, second, w, h);
        const RecordingProbe third = run(reused, first, w, h);
        expect_identical(expected, third);
    };

    "timing only probe receives timing events without stage data"_test = [] {
        struct TimingOnlyProbe : RecordingProbe {
            bool observes_stages() const override {
                return false;
            }
        };
        const std::uint32_t w = 37, h = 23;
        const auto image = test_image(w, h);
        Context<cpu::Backend> context;
        const RecordingProbe observed = run(context, image, w, h);
        TimingOnlyProbe timed;
        const Parameters params(3);
        detect<cpu::Backend>(context, {image.data(), w, h, w}, params, &timed);
        expect(timed.tensors.empty());
        expect(timed.timing == observed.timing);
    };

    "cpu thread count does not change stage outputs"_test = [] {
        expect_thread_count_does_not_change_stage_outputs<cpu::Backend>();
    };
};

} // namespace cctag::portable::tests::host_sequence
#endif // CCTAG_TEST

#endif
