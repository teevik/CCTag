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
        std::cout << "PASS: " << total << " gradient values; full corpus plus A/B/C/A reuse\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
