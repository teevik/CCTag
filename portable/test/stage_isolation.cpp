/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// Run each CPU pipeline stage on reference inputs and compare its output exactly
#include "backends/cpu/backend.hpp"
#include "host/context.hpp"
#include "kernels/plane.hpp"
#include "support/reference_snapshot.hpp"

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

    const suite<"stage_isolation"> stage_isolation_suite = [] {
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
    };
    return cfg<override>.run({.argc = argc, .argv = argv});
}
