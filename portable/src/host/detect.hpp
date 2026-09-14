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
            Backend::pyramid(levels[level], levels[level - 1]);
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

    // Observe host views after all stages, outside the stage timings
    if (probe) {
        for (std::uint32_t level = 0; level < count; ++level) {
            const PyramidHost pyramid = Backend::host_pyramid(levels[level]);
            probe->pyramid(level, probe_plane(pyramid.src));
            const GradientHost gradient = Backend::host_gradient(levels[level]);
            probe->gradient(level, probe_plane(gradient.dx), probe_plane(gradient.dy));
            const EdgesHost edges = Backend::host_edges(levels[level]);
            probe->edges(level, probe_plane(edges.edges));
        }
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
void expect_thread_count_does_not_change_pyramid_gradient_or_edges_planes() {
    const std::uint32_t w = 301, h = 173;
    const auto image = test_image(w, h);
    Context<Backend> context;
    omp_set_num_threads(1);
    const RecordingProbe single = run(context, image, w, h);
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

    "probe observes pyramid gradient and edges planes at every level"_test = [] {
        const std::uint32_t w = 37, h = 23;
        Context<cpu::Backend> context;
        const RecordingProbe probe = run(context, test_image(w, h), w, h);
        expect(eq(probe.tensors.size(), 16u)); // `src`, `dx`, `dy` and `edges` at each of 4 levels
        for (std::uint32_t level = 0; level < context.levels.size(); ++level) {
            expect(eq(probe.tensors.count("pyramid/level" + std::to_string(level) + "/src"), 1u))
                << "src at level" << level;
            expect(eq(probe.tensors.count("gradient/level" + std::to_string(level) + "/dx"), 1u))
                << "dx at level" << level;
            expect(eq(probe.tensors.count("gradient/level" + std::to_string(level) + "/dy"), 1u))
                << "dy at level" << level;
            expect(eq(probe.tensors.count("edges/level" + std::to_string(level) + "/edges"), 1u))
                << "edges at level" << level;
        }
    };

    "probe receives pyramid then gradient then edges timing events"_test = [] {
        const std::uint32_t w = 37, h = 23;
        Context<cpu::Backend> context;
        const RecordingProbe probe = run(context, test_image(w, h), w, h);
        const std::vector<std::string> expected = {
            "enter pyramid",
            "leave pyramid",
            "enter gradient",
            "leave gradient",
            "enter edges",
            "leave edges"
        };
        expect(probe.timing == expected);
    };

    "context reuse does not change pyramid gradient or edges planes"_test = [] {
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

    "cpu thread count does not change pyramid gradient or edges planes"_test = [] {
        expect_thread_count_does_not_change_pyramid_gradient_or_edges_planes<cpu::Backend>();
    };
};

} // namespace cctag::portable::tests::host_sequence
#endif // CCTAG_TEST

#endif
