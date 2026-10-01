/*
 * Copyright 2026, Simula Research Laboratory
 * SPDX-License-Identifier: MPL-2.0
 */
// portable_cuda_isolation: runs each CPU stage on inputs restored from a CUDA reference
// capture and writes one snapshot per stage, then two full-pipeline snapshots at different
// OpenMP thread counts. It applies no acceptance rule. The snapshot comparator does.
#include "backends/cpu/backend.hpp"
#include "host/context.hpp"
#include "host/detect.hpp"
#include "support/reference_snapshot.hpp"

#include <cctag/cctag_config.hpp>

#include <boost/json.hpp>

#include <omp.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <utility>

using namespace cctag::portable;
using namespace cctag::portable::test;

namespace {

constexpr int kStageCount = int(Stage::markers) + 1;

struct OwnedTensor {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::vector<std::uint8_t> bytes;
};
using Tensors = std::map<std::string, OwnedTensor>;

/// Adds a copy of `values` as a tensor of the given shape
template <class T>
void put(
    Tensors& tensors,
    std::string name,
    std::vector<std::uint64_t> shape,
    std::span<const T> values
) {
    OwnedTensor tensor{
        dtype_name(dtype_of<T>()),
        std::move(shape),
        std::vector<std::uint8_t>(values.size_bytes())
    };
    if (!values.empty()) {
        std::memcpy(tensor.bytes.data(), values.data(), values.size_bytes());
    }
    tensors.emplace(std::move(name), std::move(tensor));
}

/// Adds a copy of a plane as a `[height, width]` tensor
template <class T>
void plane(Tensors& tensors, std::string name, kernels::Plane<const T> source) {
    std::vector<T> values;
    for (std::uint32_t y = 0; y < source.height; ++y) {
        values.insert(values.end(), source.row(y), source.row(y) + source.width);
    }
    put<T>(tensors, std::move(name), {source.height, source.width}, values);
}

/// Collects one stage's outputs under the snapshot's tensor names. Candidates and markers
/// are sorted by (level, y, x) and (id, y, x) so their order does not depend on detection.
Tensors observe(Context<cpu::Backend>& context, Stage stage) {
    Tensors tensors;
    for (std::size_t i = 0; i < context.levels.size(); ++i) {
        auto& level = context.levels[i];
        const std::string p = std::string(stage_name(stage)) + "/level" + std::to_string(i) + "/";
        switch (stage) {
            case Stage::pyramid:
                plane(tensors, p + "src", level.src_plane().as_const());
                break;
            case Stage::gradient:
                plane(tensors, p + "dx", level.dx_plane().as_const());
                plane(tensors, p + "dy", level.dy_plane().as_const());
                break;
            case Stage::edges:
                plane(tensors, p + "edges", level.edges_plane().as_const());
                break;
            case Stage::edge_points:
                put<std::int32_t>(tensors, p + "xy", {level.n, 2}, level.xy);
                put<float>(tensors, p + "gradients", {level.n, 2}, level.gradients);
                break;
            case Stage::vote:
                put<std::int32_t>(tensors, p + "links", {level.n, 2}, level.links);
                put<std::int32_t>(
                    tensors,
                    p + "voters/offsets",
                    {level.n + 1},
                    level.voters_offsets
                );
                put<std::int32_t>(
                    tensors,
                    p + "voters/values",
                    {level.voters_values.size()},
                    level.voters_values
                );
                put<std::int32_t>(tensors, p + "is_max", {level.n}, level.is_max);
                put<float>(tensors, p + "flow_length", {level.n}, level.flow_length);
                put<std::int32_t>(tensors, p + "seeds", {level.seeds.size()}, level.seeds);
                put<std::int32_t>(
                    tensors,
                    p + "seed_order",
                    {level.seed_order.size()},
                    level.seed_order
                );
                break;
            case Stage::linking:
                put<std::int32_t>(
                    tensors,
                    p + "seeds",
                    {level.link_seeds.size()},
                    level.link_seeds
                );
                put<std::int32_t>(
                    tensors,
                    p + "segments/offsets",
                    {level.segment_offsets.size()},
                    level.segment_offsets
                );
                put<std::int32_t>(
                    tensors,
                    p + "segments/values",
                    {level.segment_values.size()},
                    level.segment_values
                );
                put<std::int32_t>(
                    tensors,
                    p + "child_counts",
                    {level.child_counts.size()},
                    level.child_counts
                );
                put<float>(tensors, p + "avg_vote", {level.avg_vote.size()}, level.avg_vote);
                break;
            default:
                break;
        }
    }
    if (stage == Stage::candidates) {
        auto view = host_candidates(context);
        std::vector<std::size_t> order(view.n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
            return std::tie(view.levels[a], view.ellipses[5 * a + 1], view.ellipses[5 * a])
                < std::tie(view.levels[b], view.ellipses[5 * b + 1], view.ellipses[5 * b]);
        });
        std::vector<float> ellipse, quality;
        std::vector<std::int32_t> levels;
        for (auto i : order) {
            ellipse.insert(
                ellipse.end(),
                view.ellipses.begin() + 5 * i,
                view.ellipses.begin() + 5 * i + 5
            );
            quality.push_back(view.quality[i]);
            levels.push_back(view.levels[i]);
        }
        put<float>(tensors, "candidates/ellipse", {view.n, 5}, ellipse);
        put<float>(tensors, "candidates/quality", {view.n}, quality);
        put<std::int32_t>(tensors, "candidates/level", {view.n}, levels);
    }
    if (stage == Stage::markers) {
        auto view = host_markers(context);
        std::vector<std::size_t> order(view.n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
            return std::tie(view.ids[a], view.xy[2 * a + 1], view.xy[2 * a])
                < std::tie(view.ids[b], view.xy[2 * b + 1], view.xy[2 * b]);
        });
        std::vector<float> xy;
        std::vector<std::int32_t> ids, statuses;
        for (auto i : order) {
            xy.insert(xy.end(), {view.xy[2 * i], view.xy[2 * i + 1]});
            ids.push_back(view.ids[i]);
            statuses.push_back(view.statuses[i]);
        }
        put<float>(tensors, "markers/xy", {view.n, 2}, xy);
        put<std::int32_t>(tensors, "markers/id", {view.n}, ids);
        put<std::int32_t>(tensors, "markers/status", {view.n}, statuses);
    }
    return tensors;
}

/// Writes a safetensors snapshot with the reference's identity metadata. Unless `full`, the
/// reference tensors of the stages before `last` are included as the restored inputs.
void write(
    const ReferenceSnapshot& reference,
    Tensors values,
    Stage last,
    const std::filesystem::path& path,
    bool full = false
) {
    if (!full) {
        for (const auto& [name, tensor] : reference.tensors()) {
            auto stage = parse_stage(name.substr(0, name.find('/')));
            if (stage && int(*stage) < int(last)) {
                values.emplace(
                    name,
                    OwnedTensor{
                        dtype_name(tensor.dtype),
                        tensor.shape,
                        {tensor.bytes.begin(), tensor.bytes.end()}
                    }
                );
            }
        }
    }
    boost::json::object header, metadata;
    for (const char* key :
         {"format",
          "format_version",
          "levels",
          "processed_levels",
          "image_sha256",
          "image_width",
          "image_height",
          "problem",
          "crowns",
          "params_sha256",
          "effective_parameters",
          "debug_dir",
          "semantic_params_version",
          "semantic_params_sha256"}) {
        metadata[key] = reference.meta(key);
    }
    boost::json::object reference_metadata;
    for (const auto& [key, value] : reference.metadata()) {
        reference_metadata[key] = value;
    }
    metadata["diagnostic_reference_metadata"] = boost::json::serialize(reference_metadata);
    metadata["fork_commit"] = "";
    metadata["pinned"] = "false";
    metadata["source_tree_sha256"] = CCTAG_SOURCE_TREE_SHA256;
    metadata["detector_build_settings"] = CCTAG_BUILD_SETTINGS;
    metadata["diagnostic_openmp_threads"] = std::to_string(omp_get_max_threads());
    metadata["variant"] = "portable/cpu";
    metadata["pipeline"] = "portable";
    metadata["diagnostic_algorithm"] = "cuda";
    metadata["diagnostic_mode"] = full ? "full-pipeline" : "independent-stage-inputs";
    metadata["diagnostic_reference_problem"] = reference.problem();
    std::string stages;
    for (int i = 0; i <= int(last); ++i) {
        if (i) {
            stages += ',';
        }
        stages += stage_name(Stage(i));
    }
    metadata["stages"] = stages;
    header["__metadata__"] = metadata;
    std::uint64_t offset = 0;
    for (const auto& [name, tensor] : values) {
        boost::json::array shape;
        for (auto extent : tensor.shape) {
            shape.emplace_back(extent);
        }
        header[name] = boost::json::object{
            {"dtype", tensor.dtype},
            {"shape", shape},
            {"data_offsets", boost::json::array{offset, offset + tensor.bytes.size()}}
        };
        offset += tensor.bytes.size();
    }
    std::string json = boost::json::serialize(header);
    while (json.size() % 8) {
        json += ' ';
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("cannot write diagnostic snapshot");
    }
    for (int i = 0; i < 8; ++i) {
        out.put(char(std::uint64_t(json.size()) >> (8 * i)));
    }
    out.write(json.data(), json.size());
    for (const auto& [_, tensor] : values) {
        out.write(reinterpret_cast<const char*>(tensor.bytes.data()), tensor.bytes.size());
    }
    if (!out) {
        throw std::runtime_error("failed diagnostic snapshot write");
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: portable_cuda_isolation REFERENCE.safetensors OUTPUT_DIRECTORY\n";
        return 2;
    }
    try {
        const auto reference = ReferenceSnapshot::read(argv[1]);
        if (reference.meta("variant") != "reference/cuda"
            || reference.meta("replay_version") != "1") {
            throw std::runtime_error("complete CUDA reference capture required");
        }
        const auto params = reference_parameters(reference);
        const std::filesystem::path output(argv[2]);
        std::filesystem::create_directories(output);
        for (int index = 0; index < kStageCount; ++index) {
            const auto stage = Stage(index);
            Context<cpu::Backend> context;
            if (stage == Stage::markers) {
                fill_identification(reference, context);
            } else {
                fill_context(reference, index == 0 ? Stage::pyramid : Stage(index - 1), context);
            }
            switch (stage) {
                case Stage::pyramid:
                    for (std::size_t i = 1; i < context.levels.size(); ++i) {
                        cpu::Backend::pyramid(context.levels[i], context.levels[0]);
                    }
                    break;
                case Stage::gradient:
                    for (auto& level : context.levels) {
                        cpu::Backend::gradient(level);
                    }
                    break;
                case Stage::edges:
                    for (auto& level : context.levels) {
                        cpu::Backend::edges(level, params);
                    }
                    break;
                case Stage::edge_points:
                    for (auto& level : context.levels) {
                        cpu::Backend::edge_points(level);
                    }
                    break;
                case Stage::vote:
                    for (auto& level : context.levels) {
                        cpu::Backend::vote(level, params);
                    }
                    break;
                case Stage::linking:
                    for (auto& level : context.levels) {
                        cpu::Backend::linking(level, params);
                    }
                    break;
                case Stage::candidates:
                    cpu::Backend::candidates(context, params);
                    break;
                case Stage::markers:
                    cpu::Backend::markers(context, params);
                    break;
            }
            write(
                reference,
                observe(context, stage),
                stage,
                output / (std::string(stage_name(stage)) + ".safetensors")
            );
            std::cout << "captured independently isolated " << stage_name(stage) << '\n';
            if (stage == Stage::candidates) {
                // Randomized fits may differ from a particular CUDA capture. Check
                // their useful output through the real identification continuation.
                auto tensors = observe(context, Stage::candidates);
                cpu::Backend::markers(context, params);
                tensors.merge(observe(context, Stage::markers));
                write(reference, std::move(tensors), Stage::markers,
                      output / "candidates-markers.safetensors");
            }
        }
        // The repeat reuses the context with a different OpenMP thread count
        const int threads = omp_get_max_threads();
        const std::pair<const char*, int> runs[] = {
            {"full.safetensors", threads},
            {"full-repeat.safetensors", threads == 1 ? 3 : 1},
        };
        Context<cpu::Backend> full;
        const auto pixels = reference.tensor(Stage::pyramid, 0, "src").as<std::uint8_t>();
        const kernels::Plane<const std::uint8_t> image{
            pixels.data(),
            reference.image_width(),
            reference.image_height(),
            reference.image_width()
        };
        for (const auto& [name, thread_count] : runs) {
            omp_set_num_threads(thread_count);
            detect(full, image, params, nullptr);
            Tensors tensors;
            for (int i = 0; i < kStageCount; ++i) {
                tensors.merge(observe(full, Stage(i)));
            }
            write(reference, std::move(tensors), Stage::markers, output / name, true);
        }
        std::cout << "captured full pipeline; no numerical acceptance rule was applied\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
