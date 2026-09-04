/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// The seam (ADR 0001): context sizing, a specialising backend bit-identical to the delegating one,
// the probe's stage order, context reuse across frames, and the thread count.
#define BOOST_TEST_MODULE testPortableSeam

#define BOOST_TEST_DYN_LINK

#include "backends/cpu/backend.hpp"
#include "backends/stub/backend.hpp"
#include "host/backend.hpp"
#include "host/context.hpp"
#include "host/detect.hpp"

#include <cctag/Params.hpp>
#include <cctag/Probe.hpp>

#include <boost/test/unit_test.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace cctag::portable;

namespace {

static_assert(ExecutionBackend<cpu::Backend>);
static_assert(ExecutionBackend<stub::Backend>);

/// Copies every plane it is shown, keyed like the stage snapshot's tensor names.
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
    void enter(const char* stage) override {
        timing.push_back(std::string("enter ") + stage);
    }
    void leave(const char* stage) override {
        timing.push_back(std::string("leave ") + stage);
    }
};

/// A deterministic, textured test image; odd dimensions exercise the generic resize path.
std::vector<std::uint8_t>
test_image(std::uint32_t width, std::uint32_t height, std::uint32_t seed = 12345u) {
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height);
    std::uint32_t state = seed;
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            state = state * 1664525u + 1013904223u;
            const int ring = ((x * x + y * y) / 37) % 2 ? 200 : 40;
            pixels[static_cast<std::size_t>(y) * width + x] =
                static_cast<std::uint8_t>(ring + (state >> 27));
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

void expect_identical(const RecordingProbe& a, const RecordingProbe& b) {
    BOOST_REQUIRE_EQUAL(a.tensors.size(), b.tensors.size());
    for (const auto& [name, bytes] : a.tensors) {
        const auto other = b.tensors.find(name);
        BOOST_REQUIRE_MESSAGE(other != b.tensors.end(), name << " missing");
        BOOST_CHECK_MESSAGE(bytes == other->second, name << " differs");
    }
}

template <ExecutionBackend Backend>
void expect_thread_count_does_not_change_the_result() {
#ifdef _OPENMP
    const std::uint32_t w = 301, h = 173;
    const auto image = test_image(w, h);
    Context<Backend> context;
    omp_set_num_threads(1);
    const RecordingProbe single = run(context, image, w, h);
    omp_set_num_threads(omp_get_num_procs());
    const RecordingProbe many = run(context, image, w, h);
    expect_identical(single, many);
#else
    BOOST_TEST_MESSAGE("OpenMP is off: the sequential build has nothing to compare");
#endif
}

} // namespace

BOOST_AUTO_TEST_SUITE(seam_suite)

BOOST_AUTO_TEST_CASE(context_sizes_levels_by_integer_halving) {
    Context<cpu::Backend> context;
    const cctag::Parameters params(3);
    context.ensure(37, 23, params);
    BOOST_REQUIRE_EQUAL(context.levels.size(), params._numberOfProcessedMultiresLayers);
    BOOST_CHECK_EQUAL(context.levels[1].width, 18u);
    BOOST_CHECK_EQUAL(context.levels[1].height, 11u);
    BOOST_CHECK_EQUAL(context.levels[3].width, 4u);
    BOOST_CHECK_EQUAL(context.levels[3].height, 2u);
    const auto* before = context.levels[0].src.data;
    context.ensure(37, 23, params);
    BOOST_CHECK_EQUAL(before, context.levels[0].src.data); // steady state: no reallocation
}

BOOST_AUTO_TEST_CASE(stub_backend_matches_the_cpu_backend_bit_for_bit) {
    const std::uint32_t w = 37, h = 23;
    const auto image = test_image(w, h);
    Context<cpu::Backend> cpu_context;
    Context<stub::Backend> stub_context;
    const RecordingProbe baseline = run(cpu_context, image, w, h);
    const RecordingProbe stub = run(stub_context, image, w, h);
    BOOST_CHECK(baseline.tensors.at("pyramid/level0/src") == image);
    expect_identical(baseline, stub);
}

BOOST_AUTO_TEST_CASE(probe_sees_every_level_of_each_stage_in_pipeline_order) {
    const std::uint32_t w = 37, h = 23;
    Context<cpu::Backend> context;
    const RecordingProbe probe = run(context, test_image(w, h), w, h);
    BOOST_CHECK_EQUAL(probe.tensors.size(), 12u); // 4 levels x (src, dx, dy)
    const std::vector<std::string> expected =
        {"enter pyramid", "leave pyramid", "enter gradient", "leave gradient"};
    BOOST_CHECK_EQUAL_COLLECTIONS(
        probe.timing.begin(),
        probe.timing.end(),
        expected.begin(),
        expected.end()
    );
}

BOOST_AUTO_TEST_CASE(context_reuse_does_not_leak_the_previous_frame) {
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
}

BOOST_AUTO_TEST_CASE(thread_count_does_not_change_the_result_on_the_cpu_backend) {
    expect_thread_count_does_not_change_the_result<cpu::Backend>();
}

BOOST_AUTO_TEST_CASE(thread_count_does_not_change_the_result_on_the_stub_backend) {
    expect_thread_count_does_not_change_the_result<stub::Backend>();
}

BOOST_AUTO_TEST_SUITE_END()
