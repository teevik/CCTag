// THROWAWAY: actual SYCL gradient on the complete reference-pyramid corpus.
#include "backends/prototype_sycl/backend.hpp"
#include "host/context.hpp"
#include "support/reference_snapshot.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    using namespace cctag::portable;
    using namespace cctag::portable::test;
    using B = prototype_sycl::Backend;
    try {
        auto files = reference_snapshot_files();
        if (files.empty()) throw std::runtime_error("reference snapshots required");
        Context<B> context;
        std::size_t total = 0;
        // A/B/C/A checks reconfiguration and reuse, in addition to each reference input.
        files.push_back(files.front());
        for (const auto& file : files) {
            const auto snapshot = ReferenceSnapshot::read(file);
            const cctag::Parameters params(snapshot.crowns());
            context.ensure(snapshot.image_width(), snapshot.image_height(), params);
            for (std::size_t i = 0; i < context.levels.size(); ++i) {
                auto& level = context.levels[i];
                copy_plane(snapshot.tensor(Stage::pyramid, i, "src"), level.host.src_plane());
                B::gradient(level);
            }
            for (std::size_t i = 0; i < context.levels.size(); ++i) {
                auto& level = context.levels[i];
                const auto view = B::host_gradient(level);
                const auto& dx = snapshot.tensor(Stage::gradient, i, "dx");
                const auto& dy = snapshot.tensor(Stage::gradient, i, "dy");
                const auto mx = compare_plane<std::int16_t>(dx, view.dx);
                const auto my = compare_plane<std::int16_t>(dy, view.dy);
                std::cout << snapshot.problem() << " level=" << i << " "
                          << describe(dx, mx) << " " << describe(dy, my) << '\n';
                if (!mx.exact() || !my.exact()) throw std::runtime_error("gradient differs from reference");
                total += mx.total + my.total;
                const auto again = B::host_gradient(level);
                if (again.dx.data != view.dx.data || again.dy.data != view.dy.data)
                    throw std::runtime_error("repeated host view replaced live storage");
            }
            B::wait(context);
        }
        // Independent complete candidate inputs exercise the borrowed host-stage seam.
        for (const auto& file : reference_snapshot_files()) {
            const auto snapshot = ReferenceSnapshot::read(file);
            const cctag::Parameters params(snapshot.crowns());
            Context<cpu::Backend> baseline;
            fill_context(snapshot, Stage::linking, baseline);
            cpu::Backend::candidates(baseline, params);
            context.ensure(snapshot.image_width(), snapshot.image_height(), params);
            for (std::size_t i = 0; i < context.levels.size(); ++i)
                fill_level(snapshot, i, Stage::linking, context.levels[i].host);
            B::candidates(context, params);
            const auto expected_candidates = host_candidates(baseline);
            const auto actual_candidates = host_candidates(context);
            if (!std::ranges::equal(expected_candidates.ellipses, actual_candidates.ellipses)
                || !std::ranges::equal(expected_candidates.levels, actual_candidates.levels)
                || !std::ranges::equal(expected_candidates.quality, actual_candidates.quality))
                throw std::runtime_error("borrowed candidates inputs changed baseline output");
            // Copy complete candidate records only in this delegation check, including
            // original center and ordered directed points absent from reference snapshots.
            context.candidate_markers = baseline.candidate_markers;
            cpu::Backend::markers(baseline, params);
            B::markers(context, params);
            const auto expected = host_markers(baseline), actual = host_markers(context);
            if (!std::ranges::equal(expected.xy, actual.xy)
                || !std::ranges::equal(expected.ids, actual.ids)
                || !std::ranges::equal(expected.statuses, actual.statuses))
                throw std::runtime_error("markers delegation changed identical complete inputs");
            for (std::size_t i = 0; i < baseline.markers.size(); ++i)
                if (baseline.markers[i].homography != context.markers[i].homography
                    || baseline.markers[i].quality != context.markers[i].quality)
                    throw std::runtime_error("markers delegation changed homography/quality");
            std::cout << "PASS: " << snapshot.problem()
                      << " borrowed candidate inputs and complete-input marker delegation\n";
        }
        std::cout << "PASS: " << total << " gradient values; full corpus plus A/B/C/A reuse\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
