/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// Stage isolation (thesis #67): each test runs one stage function of the CPU backend on the
// reference's input to that stage — the reference snapshot's stage N−1 tensors, loaded straight
// into the stage buffers — and compares its output element-exact against the reference's stage N
// tensors. One test per stage function; the build adds one per stage it grows.
//
// The tests need the reference-snapshot store (`$CCTAG_REFERENCE_SNAPSHOTS`, exported by
// `nix develop` and the nix checks) and are skipped with a message without it; the loader's own
// tests run anywhere.
#define BOOST_TEST_MODULE testPortableStageIsolation

#define BOOST_TEST_DYN_LINK

#include "backends/cpu/backend.hpp"
#include "host/context.hpp"
#include "kernels/plane.hpp"
#include "reference_snapshot.hpp"

#include <boost/test/unit_test.hpp>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cctag::portable;
using namespace cctag::portable::test;

namespace {

boost::test_tools::assertion_result store_available(boost::unit_test::test_unit_id) {
    boost::test_tools::assertion_result result(reference_snapshots_dir().has_value());
    if (!result) {
        result.message() << "CCTAG_REFERENCE_SNAPSHOTS is not set: enter `nix develop` or point it "
                            "at the reference-snapshot store";
    }
    return result;
}

std::vector<std::filesystem::path> snapshot_files_or_fail() {
    const std::vector<std::filesystem::path> files = reference_snapshot_files();
    BOOST_REQUIRE_MESSAGE(
        !files.empty(),
        "no *.safetensors in " << reference_snapshots_dir()->string()
    );
    return files;
}

template <class T>
kernels::Plane<const T> as_const(kernels::Plane<T> plane) {
    return {plane.data, plane.width, plane.height, plane.stride};
}

/// A hand-built safetensors file: `u64` header length, the JSON header padded to eight bytes with
/// spaces (as the Rust writer does), then the data section.
std::vector<std::uint8_t> safetensors(std::string header, const std::vector<std::uint8_t>& data) {
    while (header.size() % 8 != 0) {
        header.push_back(' ');
    }
    std::vector<std::uint8_t> bytes(8);
    const std::uint64_t length = header.size();
    for (int i = 0; i < 8; ++i) {
        bytes[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(length >> (8 * i));
    }
    bytes.insert(bytes.end(), header.begin(), header.end());
    bytes.insert(bytes.end(), data.begin(), data.end());
    return bytes;
}

} // namespace

BOOST_AUTO_TEST_SUITE(snapshot_loader_suite)

BOOST_AUTO_TEST_CASE(reads_a_hand_built_snapshot) {
    // A U8 [2, 3] plane followed by an I16 [1, 2] plane that starts at an odd byte offset, with
    // an escaped metadata value; whitespace and key order vary from the Rust writer's on purpose.
    const std::string header =
        R"({"pyramid/level0/src": {"dtype": "U8", "shape": [2, 3], "data_offsets": [0, 6]},)"
        R"( "gradient/level0/dx": {"dtype":"I16","shape":[1,2],"data_offsets":[7,11]},)"
        R"( "pad": {"dtype":"U8","shape":[1],"data_offsets":[6,7]},)"
        R"( "__metadata__": {"problem": "å\"x", "crowns": "3", "stages": "pyramid,gradient"}})";
    const std::vector<std::uint8_t> data = {1, 2, 3, 4, 5, 6, 9, 0xFE, 0xFF, 0x02, 0x00};
    const ReferenceSnapshot snapshot = ReferenceSnapshot::from_bytes(safetensors(header, data));

    BOOST_CHECK_EQUAL(snapshot.problem(), "\xC3\xA5\"x");
    BOOST_CHECK_EQUAL(snapshot.crowns(), 3u);
    BOOST_REQUIRE_EQUAL(snapshot.stages().size(), 2u);
    BOOST_CHECK(snapshot.stages()[1] == Stage::gradient);
    BOOST_CHECK(snapshot.has(Stage::gradient));
    BOOST_CHECK(!snapshot.has(Stage::vote));

    const Tensor& src = snapshot.tensor(Stage::pyramid, 0, "src");
    BOOST_CHECK(src.dtype == Dtype::u8);
    BOOST_REQUIRE_EQUAL(src.shape.size(), 2u);
    BOOST_CHECK_EQUAL(src.shape[0], 2u);
    BOOST_CHECK_EQUAL(src.shape[1], 3u);
    const std::vector<std::uint8_t> pixels = src.as<std::uint8_t>();
    BOOST_CHECK_EQUAL(pixels.size(), 6u);
    BOOST_CHECK_EQUAL(pixels[5], 6);

    const std::vector<std::int16_t> dx = snapshot.tensor("gradient/level0/dx").as<std::int16_t>();
    BOOST_REQUIRE_EQUAL(dx.size(), 2u);
    BOOST_CHECK_EQUAL(dx[0], -2);
    BOOST_CHECK_EQUAL(dx[1], 2);

    BOOST_CHECK_THROW(snapshot.tensor("gradient/level0/dy"), std::runtime_error);
    BOOST_CHECK_THROW(src.as<std::int32_t>(), std::runtime_error);
    BOOST_CHECK_THROW(snapshot.meta("image_width"), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(rejects_offsets_that_disagree_with_dtype_and_shape) {
    const std::string header =
        R"({"pyramid/level0/src":{"dtype":"U8","shape":[2,3],"data_offsets":[0,5]},"__metadata__":{}})";
    BOOST_CHECK_THROW(
        ReferenceSnapshot::from_bytes(safetensors(header, std::vector<std::uint8_t>(5))),
        std::runtime_error
    );
    const std::string overrun =
        R"({"pyramid/level0/src":{"dtype":"U8","shape":[2,3],"data_offsets":[0,6]},"__metadata__":{}})";
    BOOST_CHECK_THROW(
        ReferenceSnapshot::from_bytes(safetensors(overrun, std::vector<std::uint8_t>(5))),
        std::runtime_error
    );
}

BOOST_AUTO_TEST_CASE(fills_and_compares_planes_element_exact) {
    // `src` and `dx` for a 3x2 level; `dy` is missing on purpose.
    const std::string header =
        R"({"pyramid/level0/src":{"dtype":"U8","shape":[2,3],"data_offsets":[0,6]},)"
        R"("gradient/level0/dx":{"dtype":"I16","shape":[2,3],"data_offsets":[6,18]},)"
        R"("__metadata__":{"x":"y"}})";
    const ReferenceSnapshot snapshot = ReferenceSnapshot::from_bytes(
        safetensors(header, {1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
    );
    const Tensor& src = snapshot.tensor(Stage::pyramid, 0, "src");

    cpu::Buffers buffers;
    buffers.ensure(3, 2, 0, 0);
    fill_level(snapshot, 0, Stage::pyramid, buffers);
    BOOST_CHECK_EQUAL(buffers.src[4], 5);
    BOOST_CHECK(compare_plane<std::uint8_t>(src, as_const(buffers.src_plane())).exact());

    buffers.src[4] = 0;
    buffers.src[5] = 0;
    const Mismatch mismatch = compare_plane<std::uint8_t>(src, as_const(buffers.src_plane()));
    BOOST_CHECK_EQUAL(mismatch.count, 2u);
    BOOST_CHECK_EQUAL(mismatch.first, 4u);
    BOOST_CHECK_EQUAL(
        describe(src, mismatch),
        "pyramid/level0/src: 2 of 6 elements differ, first at index 4 (y 1, x 1)"
    );

    cpu::Buffers wrong_size;
    wrong_size.ensure(2, 3, 0, 0);
    BOOST_CHECK_THROW(fill_level(snapshot, 0, Stage::pyramid, wrong_size), std::runtime_error);
    // `dy` is absent: the gradient fill fails; beyond gradient the loader has no buffers yet.
    BOOST_CHECK_THROW(fill_level(snapshot, 0, Stage::gradient, buffers), std::runtime_error);
    BOOST_CHECK_THROW(fill_level(snapshot, 0, Stage::edges, buffers), std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(stage_isolation_suite)

BOOST_AUTO_TEST_CASE(
    pyramid_from_the_reference_finer_level,
    *boost::unit_test::precondition(store_available)
) {
    for (const auto& file : snapshot_files_or_fail()) {
        const ReferenceSnapshot snapshot = ReferenceSnapshot::read(file);
        BOOST_TEST_CONTEXT("problem " << snapshot.problem()) {
            Context<cpu::Backend> context;
            fill_context(snapshot, Stage::pyramid, context);
            auto& levels = context.levels;

            // Level 0 is the load: the reference image in, the same bytes out.
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
                compare_plane<std::uint8_t>(image, as_const(levels[0].src_plane()));
            BOOST_CHECK_MESSAGE(loaded.exact(), describe(image, loaded));

            for (std::uint32_t level = 1; level < levels.size(); ++level) {
                BOOST_TEST_CONTEXT("level " << level) {
                    // The finer level holds this test's own output from the previous iteration:
                    // restore the reference's before it becomes the input.
                    fill_level(snapshot, level - 1, Stage::pyramid, levels[level - 1]);
                    cpu::Backend::pyramid(levels[level], levels[level - 1]);
                    const Tensor& expected = snapshot.tensor(Stage::pyramid, level, "src");
                    const Mismatch mismatch =
                        compare_plane<std::uint8_t>(expected, as_const(levels[level].src_plane()));
                    BOOST_CHECK_MESSAGE(mismatch.exact(), describe(expected, mismatch));
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(
    gradient_from_the_reference_src,
    *boost::unit_test::precondition(store_available)
) {
    for (const auto& file : snapshot_files_or_fail()) {
        const ReferenceSnapshot snapshot = ReferenceSnapshot::read(file);
        BOOST_TEST_CONTEXT("problem " << snapshot.problem()) {
            Context<cpu::Backend> context;
            fill_context(snapshot, Stage::pyramid, context);
            auto& levels = context.levels;
            for (std::uint32_t level = 0; level < levels.size(); ++level) {
                BOOST_TEST_CONTEXT("level " << level) {
                    cpu::Backend::gradient(levels[level]);
                    const Tensor& dx = snapshot.tensor(Stage::gradient, level, "dx");
                    const Tensor& dy = snapshot.tensor(Stage::gradient, level, "dy");
                    const Mismatch dx_mismatch =
                        compare_plane<std::int16_t>(dx, as_const(levels[level].dx_plane()));
                    const Mismatch dy_mismatch =
                        compare_plane<std::int16_t>(dy, as_const(levels[level].dy_plane()));
                    BOOST_CHECK_MESSAGE(dx_mismatch.exact(), describe(dx, dx_mismatch));
                    BOOST_CHECK_MESSAGE(dy_mismatch.exact(), describe(dy, dy_mismatch));
                }
            }
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
