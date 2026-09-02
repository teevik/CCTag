#define BOOST_TEST_MODULE testPortableSeam

#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "backends/cpu/backend.hpp"
#include "backends/stub/backend.hpp"
#include "host/context.hpp"
#include "host/detect.hpp"
#include "kernels/gradient.hpp"
#include "kernels/resize.hpp"

#include <cctag/Params.hpp>
#include <cctag/Probe.hpp>

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
std::vector<std::uint8_t> test_image(std::uint32_t width, std::uint32_t height) {
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height);
    std::uint32_t state = 12345u;
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

template <class Backend>
RecordingProbe run(Context<Backend>& context, const std::vector<std::uint8_t>& image, std::uint32_t width, std::uint32_t height) {
    RecordingProbe probe;
    const cctag::Parameters params(3);
    detect(context, kernels::Plane<const std::uint8_t>{image.data(), width, height, width}, params, &probe);
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

} // namespace

BOOST_AUTO_TEST_SUITE(kernels_suite)

BOOST_AUTO_TEST_CASE(halving_rounds_half_up) {
    // 4x2 -> 2x1; sums 4*1+2 -> 1, 4*254+2+... -> saturates at 255 never triggers.
    const std::uint8_t src[2][4] = {{1, 1, 253, 255}, {1, 1, 255, 255}};
    BOOST_CHECK_EQUAL(kernels::halve_at(&src[0][0], 4, 0, 0), 1);
    BOOST_CHECK_EQUAL(kernels::halve_at(&src[0][0], 4, 1, 0), 255);
    const std::uint8_t tie[2][2] = {{0, 1}, {0, 1}}; // sum 2 -> (2 + 2) >> 2 = 1
    BOOST_CHECK_EQUAL(kernels::halve_at(&tie[0][0], 2, 0, 0), 1);
}

BOOST_AUTO_TEST_CASE(rounding_is_nearest_even_and_clamped) {
    BOOST_CHECK_EQUAL(kernels::round_to_int16(-3.5f), -4);
    BOOST_CHECK_EQUAL(kernels::round_to_int16(34.5f), 34);
    BOOST_CHECK_EQUAL(kernels::round_to_int16(2.5f), 2);
    BOOST_CHECK_EQUAL(kernels::round_to_int16(-2.5f), -2);
    BOOST_CHECK_EQUAL(kernels::round_to_int16(0.4999999f), 0);
    BOOST_CHECK_EQUAL(kernels::round_to_int16(40000.f), 32767);
    BOOST_CHECK_EQUAL(kernels::round_to_int16(-40000.f), -32768);
}

BOOST_AUTO_TEST_CASE(gradient_taps_are_the_kernel_without_zeros) {
    BOOST_CHECK_EQUAL(kernels::kDxTaps.dx[0], -4);
    BOOST_CHECK_EQUAL(kernels::kDxTaps.dy[0], -4);
    BOOST_CHECK_EQUAL(kernels::kDxTaps.dx[4], 1); // column 4 skipped
    BOOST_CHECK_EQUAL(kernels::kDyTaps.dy[36], 1); // row 4 skipped: rows 0-3 hold 36 taps
    BOOST_CHECK_EQUAL(kernels::kDyTaps.coefficient[0], kernels::kDxTaps.coefficient[0]);
    BOOST_CHECK_EQUAL(kernels::kDyTaps.coefficient[1], kernels::kDerivativeKernel[1][0]);
}

BOOST_AUTO_TEST_CASE(gradient_of_a_constant_image_is_zero_and_of_a_step_is_antisymmetric) {
    const std::uint32_t w = 16, h = 16;
    std::vector<std::uint8_t> flat(w * h, 77);
    BOOST_CHECK_EQUAL(kernels::gradient_at(flat.data(), w, w, h, 7, 7, kernels::kDxTaps), 0);
    std::vector<std::uint8_t> step(w * h);
    for (std::uint32_t y = 0; y < h; ++y) {
        for (std::uint32_t x = 0; x < w; ++x) {
            step[y * w + x] = x < 8 ? 0 : 255;
        }
    }
    const std::int16_t left = kernels::gradient_at(step.data(), w, w, h, 7, 7, kernels::kDxTaps);
    const std::int16_t right = kernels::gradient_at(step.data(), w, w, h, 8, 7, kernels::kDxTaps);
    BOOST_CHECK_GT(left, 0);
    BOOST_CHECK_EQUAL(left, right);
    BOOST_CHECK_EQUAL(kernels::gradient_at(step.data(), w, w, h, 7, 7, kernels::kDyTaps), 0);
}

BOOST_AUTO_TEST_CASE(linear_axis_tables_match_opencv_for_an_odd_halving) {
    // 5 -> 2: scale 2.5; d=0: fx = 0.75 -> sx 0, alpha (512, 1536); d=1: fx = 3.25 -> sx 3, (1536, 512).
    std::int32_t ofs[2];
    std::int16_t alpha[4];
    const std::uint32_t max = kernels::linear_axis_tables(5, 2, ofs, alpha);
    BOOST_CHECK_EQUAL(max, 2u);
    BOOST_CHECK_EQUAL(ofs[0], 0);
    BOOST_CHECK_EQUAL(alpha[0], 512);
    BOOST_CHECK_EQUAL(alpha[1], 1536);
    BOOST_CHECK_EQUAL(ofs[1], 3);
    BOOST_CHECK_EQUAL(alpha[2], 1536);
    BOOST_CHECK_EQUAL(alpha[3], 512);
}

BOOST_AUTO_TEST_SUITE_END()

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
    BOOST_CHECK(!context.levels[1].exact_halving); // 37x23 -> 18x11
    BOOST_CHECK(!context.levels[2].exact_halving); // 18x11 -> 9x5
    BOOST_CHECK(!context.levels[3].exact_halving); // 9x5 -> 4x2
    const auto* before = context.levels[0].src.data();
    context.ensure(37, 23, params);
    BOOST_CHECK_EQUAL(before, context.levels[0].src.data()); // steady state: no reallocation
}

BOOST_AUTO_TEST_CASE(stub_delegation_and_specialisation_match_the_baseline_bit_for_bit) {
    const std::uint32_t w = 37, h = 23;
    const auto image = test_image(w, h);
    Context<cpu::Backend> cpu_context;
    Context<stub::Backend> stub_context;
    const RecordingProbe baseline = run(cpu_context, image, w, h);
    const RecordingProbe stub = run(stub_context, image, w, h);
    BOOST_REQUIRE_EQUAL(baseline.tensors.size(), 12u); // 4 levels x (src, dx, dy)
    expect_identical(baseline, stub);
    BOOST_CHECK(baseline.tensors.at("pyramid/level0/src") == image);
    BOOST_CHECK_EQUAL(baseline.timing.size(), 4u);
    BOOST_CHECK_EQUAL(baseline.timing[0], "enter pyramid");
    BOOST_CHECK_EQUAL(baseline.timing[3], "leave gradient");
}

BOOST_AUTO_TEST_CASE(second_frame_on_the_same_context_is_identical) {
    const std::uint32_t w = 64, h = 48;
    const auto image = test_image(w, h);
    Context<cpu::Backend> context;
    const RecordingProbe first = run(context, image, w, h);
    const RecordingProbe second = run(context, image, w, h);
    expect_identical(first, second);
    BOOST_CHECK(context.levels[1].exact_halving);
}

BOOST_AUTO_TEST_CASE(thread_count_does_not_change_the_result) {
#ifdef _OPENMP
    const std::uint32_t w = 301, h = 173;
    const auto image = test_image(w, h);
    Context<cpu::Backend> context;
    omp_set_num_threads(1);
    const RecordingProbe single = run(context, image, w, h);
    omp_set_num_threads(omp_get_num_procs());
    const RecordingProbe many = run(context, image, w, h);
    expect_identical(single, many);
#else
    BOOST_TEST_MESSAGE("OpenMP is off: the sequential build has nothing to compare");
#endif
}

BOOST_AUTO_TEST_SUITE_END()
