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

namespace cctag::portable {

/// Times a stage using the probe's `enter` and `leave` callbacks
/// Waits for the execution backend to finish before calling `leave`
template <ExecutionBackend Backend>
class StageTiming {
  public:
    StageTiming(Context<Backend>& context, Probe* probe, const char* stage) :
        context(context),
        probe(probe),
        stage(stage) {
        if (probe) {
            probe->enter(stage);
        }
    }
    ~StageTiming() {
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
    auto& levels = context.levels;
    const std::uint32_t count = static_cast<std::uint32_t>(levels.size());

    // Load level 0 and build each coarser pyramid level
    {
        StageTiming<Backend> timing(context, probe, "pyramid");
        Backend::load(levels[0], input);
        for (std::uint32_t level = 1; level < count; ++level) {
            Backend::pyramid(levels[level], levels[0]);
        }
    }
    // Compute gradients at every pyramid level
    {
        StageTiming<Backend> timing(context, probe, "gradient");
        for (std::uint32_t level = 0; level < count; ++level) {
            Backend::gradient(levels[level]);
        }
    }

    // Find and thin edges at every pyramid level
    {
        StageTiming<Backend> timing(context, probe, "edges");
        for (std::uint32_t level = 0; level < count; ++level) {
            Backend::edges(levels[level], params);
        }
    }

    // Collect edge points in canonical order at every pyramid level
    {
        StageTiming<Backend> timing(context, probe, "edge_points");
        for (std::uint32_t level = 0; level < count; ++level) {
            Backend::edge_points(levels[level]);
        }
    }

    // Link edge points and gather their votes at every pyramid level
    {
        StageTiming<Backend> timing(context, probe, "vote");
        for (std::uint32_t level = 0; level < count; ++level) {
            Backend::vote(levels[level], params);
        }
    }

    // Walk seeds and gather segments at every pyramid level
    {
        StageTiming<Backend> timing(context, probe, "linking");
        for (std::uint32_t level = 0; level < count; ++level) {
            Backend::linking(levels[level], params);
        }
    }

    // Observe host views after all stages, outside the stage timings
    if (probe && probe->observes_stages()) {
        for (std::uint32_t level = 0; level < count; ++level) {
            const SnapshotViews views = Backend::snapshot_views(levels[level]);
            observe(*probe, level, views);
        }
    }

    // Fit candidate markers across levels and project their outer ellipses to level zero
    {
        StageTiming<Backend> timing(context, probe, "candidates");
        Backend::candidates(context, params);
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
    }
    if (probe && probe->observes_stages()) {
        const MarkersHost markers = host_markers(context);
        probe->markers({markers.n, markers.xy.data(), markers.ids.data(), markers.statuses.data()});
    }
}

} // namespace cctag::portable

#ifdef CCTAG_TEST
#include "backends/cpu/backend.hpp"

#include <cctag/Params.hpp>
#include <cctag/Probe.hpp>

#include <boost/ut.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace cctag::portable::tests::host_sequence {

using namespace boost::ut;

/// Records which stage observations and timing events the probe receives
struct RecordingProbe : cctag::Probe {
    std::vector<std::string> observations;
    std::vector<std::string> timing;

    void observed(const char* stage, std::uint32_t level) {
        observations.push_back(std::string(stage) + " level" + std::to_string(level));
    }
    void pyramid(std::uint32_t level, const cctag::Plane&) override {
        observed("pyramid", level);
    }
    void gradient(std::uint32_t level, const cctag::Plane&, const cctag::Plane&) override {
        observed("gradient", level);
    }
    void edges(std::uint32_t level, const cctag::Plane&) override {
        observed("edges", level);
    }
    void edge_points(std::uint32_t level, const cctag::EdgePointsView&) override {
        observed("edge_points", level);
    }
    void vote(std::uint32_t level, const cctag::VoteView&) override {
        observed("vote", level);
    }
    void linking(std::uint32_t level, const cctag::LinkingView&) override {
        observed("linking", level);
    }
    void candidates(const cctag::CandidatesView&) override {
        observations.emplace_back("candidates");
    }
    void markers(const cctag::MarkersView&) override {
        observations.emplace_back("markers");
    }
    void enter(const char* stage) override {
        timing.push_back(std::string("enter ") + stage);
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

inline void
run(Context<cpu::Backend>& context,
    const std::vector<std::uint8_t>& image,
    std::uint32_t width,
    std::uint32_t height,
    cctag::Probe* probe = nullptr) {
    const cctag::Parameters params(3);
    detect(
        context,
        kernels::Plane<const std::uint8_t>{image.data(), width, height, width},
        params,
        probe
    );
}

/// Copies the per-level integer stage outputs, keyed by stage, level and name. Floating-point
/// outputs are left out, since their self-consistency is checked by the snapshot comparator.
inline std::map<std::string, std::vector<std::int32_t>>
discrete_outputs(Context<cpu::Backend>& context) {
    std::map<std::string, std::vector<std::int32_t>> outputs;
    for (std::size_t i = 0; i < context.levels.size(); ++i) {
        const SnapshotViews views = cpu::Backend::snapshot_views(context.levels[i]);
        const auto level = "level" + std::to_string(i) + "/";
        const auto plane = [&](const std::string& name, auto source) {
            auto& values = outputs[level + name];
            for (std::uint32_t y = 0; y < source.height; ++y) {
                values.insert(values.end(), source.row(y), source.row(y) + source.width);
            }
        };
        const auto values = [&](const std::string& name, std::span<const std::int32_t> source) {
            outputs[level + name].assign(source.begin(), source.end());
        };
        plane("src", views.pyramid.src);
        plane("dx", views.gradient.dx);
        plane("dy", views.gradient.dy);
        plane("edges", views.edges.edges);
        values("edge_points", views.edge_points.xy);
        values("links", views.vote.links);
        values("voters_offsets", views.vote.voters_offsets);
        values("voters_values", views.vote.voters_values);
        values("is_max", views.vote.is_max);
        values("seeds", views.vote.seeds);
        values("seed_order", views.vote.seed_order);
        values("linking_seeds", views.linking.seeds);
        values("segment_offsets", views.linking.segment_offsets);
        values("segment_values", views.linking.segment_values);
        values("child_counts", views.linking.child_counts);
    }
    return outputs;
}

inline suite<"host_sequence"> host_sequence_suite = [] {
    "load copies the input into level zero"_test = [] {
        const std::uint32_t w = 37, h = 23;
        const auto image = test_image(w, h);
        Context<cpu::Backend> context;
        run(context, image, w, h);
        const auto src = cpu::Backend::snapshot_views(context.levels[0]).pyramid.src;
        for (std::uint32_t y = 0; y < h; ++y) {
            expect(
                std::ranges::equal(
                    std::span(src.row(y), w),
                    std::span(image).subspan(std::size_t(y) * w, w)
                )
            ) << "row"
              << y;
        }
    };

    "probe observes each per level stage once at every level, then candidates and markers"_test =
        [] {
        const std::uint32_t w = 37, h = 23;
        Context<cpu::Backend> context;
        RecordingProbe probe;
        run(context, test_image(w, h), w, h, &probe);
        std::vector<std::string> expected{"candidates", "markers"};
        for (std::uint32_t level = 0; level < context.levels.size(); ++level) {
            for (const char* stage :
                 {"pyramid", "gradient", "edges", "edge_points", "vote", "linking"}) {
                expected.push_back(std::string(stage) + " level" + std::to_string(level));
            }
        }
        // Each observation is promised once; the order between callbacks is not
        std::ranges::sort(expected);
        std::ranges::sort(probe.observations);
        expect(probe.observations == expected);
    };

    "probe receives timing events in stage order through markers"_test = [] {
        const std::uint32_t w = 37, h = 23;
        Context<cpu::Backend> context;
        RecordingProbe probe;
        run(context, test_image(w, h), w, h, &probe);
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

    "timing only probe receives timing events without stage observations"_test = [] {
        struct TimingOnlyProbe : RecordingProbe {
            bool observes_stages() const override {
                return false;
            }
        };
        const std::uint32_t w = 37, h = 23;
        const auto image = test_image(w, h);
        Context<cpu::Backend> context;
        RecordingProbe observing;
        run(context, image, w, h, &observing);
        TimingOnlyProbe timed;
        run(context, image, w, h, &timed);
        expect(timed.observations.empty());
        expect(timed.timing == observing.timing);
    };

    "an intervening frame does not change discrete stage outputs"_test = [] {
        const std::uint32_t w = 64, h = 48;
        const auto first = test_image(w, h, 1u);
        const auto second = test_image(w, h, 2u);
        Context<cpu::Backend> fresh;
        run(fresh, first, w, h);
        const auto expected = discrete_outputs(fresh);
        // The comparison is only meaningful if the image reaches the graph stages
        expect(!expected.at("level0/seed_order").empty()) << fatal;
        expect(!expected.at("level0/segment_values").empty()) << fatal;

        Context<cpu::Backend> reused;
        run(reused, first, w, h);
        run(reused, second, w, h);
        run(reused, first, w, h);
        const auto actual = discrete_outputs(reused);

        expect(eq(actual.size(), expected.size())) << fatal;
        for (const auto& [name, values] : expected) {
            expect(actual.at(name) == values) << name << "differs";
        }
    };
};

} // namespace cctag::portable::tests::host_sequence
#endif // CCTAG_TEST

#endif
