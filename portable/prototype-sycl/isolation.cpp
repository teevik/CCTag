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
        if (std::getenv("PROTOTYPE_SYCL_SECTION") && std::string(std::getenv("PROTOTYPE_SYCL_SECTION")) != "gradient") {
            for (const auto& file : files) {
                const auto snapshot=ReferenceSnapshot::read(file);
                const cctag::Parameters params(snapshot.crowns());
                context.ensure(snapshot.image_width(),snapshot.image_height(),params);
                for(std::size_t i=0;i<context.levels.size();++i) {
                    auto& level=context.levels[i];
                    fill_level(snapshot,i,Stage::gradient,level.host);
                    B::upload_gradient(level);B::edges(level,params);
                    auto expected=snapshot.tensor(Stage::edges,i,"edges");
                    auto mismatch=compare_plane<std::uint8_t>(expected,B::host_edges(level).edges);
                    std::cout<<"edges "<<snapshot.problem()<<" level="<<i<<" "<<describe(expected,mismatch)<<'\n';
                    if(!mismatch.exact())throw std::runtime_error("reference-fed edges differs");
                    fill_level(snapshot,i,Stage::edges,level.host);
                    B::upload_edges(level);B::edge_points(level);
                    auto points=B::host_edge_points(level);
                    auto xy=compare_values<std::int32_t>(snapshot.tensor(Stage::edge_points,i,"xy"),points.xy);
                    auto gradients=compare_values<float>(snapshot.tensor(Stage::edge_points,i,"gradients"),points.gradients);
                    if(!xy.exact()||!gradients.exact())throw std::runtime_error("reference-fed edge points differs");
                    cpu::Buffers baseline;baseline.ensure(level.host.width,level.host.height,snapshot.image_width(),snapshot.image_height());
                    fill_level(snapshot,i,Stage::edge_points,baseline);
                    if(cv::countNonZero(baseline.edge_map!=level.host.edge_map))throw std::runtime_error("host edge map differs");
                    // Invoke the real CPU continuation through the SYCL adapter.
                    // Compact voting must not depend on any gradient plane values.
                    if(level.host.prototype_compact_vote){level.host.dx.setTo(123);level.host.dy.setTo(-456);}
                    B::vote(level,params);
                    auto v=B::host_vote(level);
                    auto check=[&]<class T>(const char* name,std::span<const T> values){
                        if(!compare_values<T>(snapshot.tensor(Stage::vote,i,name),values).exact())throw std::runtime_error(std::string("vote differs: ")+name);
                    };
                    check("links",v.links);check("voters/offsets",v.voters_offsets);check("voters/values",v.voters_values);
                    check("is_max",v.is_max);check("flow_length",v.flow_length);check("seeds",v.seeds);check("seed_order",v.seed_order);
                    std::cout<<"PASS: "<<snapshot.problem()<<" level="<<i<<" reference-fed edge_points, complete edge map and vote boundary\n";
                }
            }
            prototype_sycl::Buffers small;small.bind(*context.execution);
            // Hand-constructed 8-connectivity, isolated weak components, and retained-state changes.
            small.ensure(7,5,7,5);
            std::vector<int> classes(35,0),expected(35,0);
            classes[0]=2;classes[8]=classes[16]=classes[24]=1;
            classes[6]=classes[13]=1;classes[34]=2;
            expected=classes;expected[8]=expected[16]=expected[24]=2;
            if(B::hysteresis_case(small,classes)!=expected)throw std::runtime_error("independent diagonal connectivity");
            classes.assign(35,1);expected=classes;
            if(B::hysteresis_case(small,classes)!=expected)throw std::runtime_error("unseeded weak component");
            classes[0]=2;expected.assign(35,2);
            if(B::hysteresis_case(small,classes)!=expected)throw std::runtime_error("dense connected component");
            small.ensure(1,1025,1,1025);classes.assign(1025,1);classes[1024]=2;expected.assign(1025,2);
            if(B::hysteresis_case(small,classes)!=expected)throw std::runtime_error("long reverse chain");
            std::cout<<"PASS: independent diagonal, disconnected, unseeded, dense and 1025-pixel chain hysteresis\n";
            small.ensure(1,7,1,7);
            small.host.dx=(cv::Mat1s(7,1)<<11,3,10,2,3,3,0);small.host.dy.setTo(0);
            cctag::Parameters params(3);
            for(int swapped=0;swapped<2;++swapped){
                B::upload_gradient(small);B::edges(small,params);auto view=B::host_edges(small);
                std::array<unsigned char,7> values{255,255,255,0,0,0,0};
                for(int y=0;y<7;++y)if(view.edges.row(y)[0]!=values[y])throw std::runtime_error("threshold tie");
                std::swap(params._cannyThrLow,params._cannyThrHigh);
            }
            small.ensure(3,3,3,3);
            small.host.dx=(cv::Mat1s(3,3)<<0,11,0,0,11,0,0,0,0);
            small.host.dy=(cv::Mat1s(3,3)<<0,0,0,0,0,11,0,0,0);
            B::upload_gradient(small);B::edges(small,params);
            auto border=B::host_edges(small);std::array<unsigned char,9> values{0,255,0,0,255,255,0,0,0};
            for(int y=0;y<3;++y)for(int x=0;x<3;++x)if(border.edges.row(y)[x]!=values[y*3+x])throw std::runtime_error("thinning border");
            small.ensure(5,4,5,4);
            cv::Rect region(2,0,5,4);
            small.host.edges=cv::Mat1b(4,9,std::uint8_t{0})(region);
            small.host.dx=cv::Mat1s(4,9,std::int16_t{0})(region);
            small.host.dy=cv::Mat1s(4,9,std::int16_t{0})(region);
            cv::Mat1i storage(4,9,-7);small.host.edge_map=storage(region);
            for(int repeat=0;repeat<3;++repeat){
                small.host.edges.setTo(0);
                if(repeat!=1){small.host.edges(0,0)=255;small.host.edges(0,4)=255;small.host.edges(2,1)=255;small.host.edges(3,4)=255;}
                small.host.edges(2,3)=1;small.host.dx(0,0)=-32768;small.host.dy(0,0)=32767;
                B::upload_edges(small);B::edge_points(small);auto points=B::host_edge_points(small);
                std::vector<int> coords=repeat==1?std::vector<int>{}:std::vector<int>{0,0,4,0,1,2,4,3};
                if(!std::ranges::equal(coords,points.xy))throw std::runtime_error("canonical point order");
                if(repeat!=1&&(points.gradients[0]!=-32768.f||points.gradients[1]!=32767.f))throw std::runtime_error("point gradients");
                cv::Mat1i expected_map(4,5,-1);for(unsigned i=0;i<points.n;++i)expected_map(coords[2*i+1],coords[2*i])=i;
                if(cv::countNonZero(expected_map!=small.host.edge_map))throw std::runtime_error("complete map rewrite");
                if(cv::countNonZero(storage.colRange(0,2)!=-7)||cv::countNonZero(storage.colRange(7,9)!=-7))throw std::runtime_error("stride padding changed");
            }
            small.ensure(1,1,1,1);small.host.dx.setTo(-32768);small.host.dy.setTo(32767);
            B::upload_gradient(small);B::edges(small,params);
            if(B::host_edges(small).edges.row(0)[0]!=255)throw std::runtime_error("extreme gradient magnitude");
            std::cout<<"PASS: threshold ties, swapped thresholds, thinning border, pitched canonical compaction, empty/reuse and extreme gradients\n";
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
