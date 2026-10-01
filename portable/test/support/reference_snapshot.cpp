/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "reference_snapshot.hpp"

#include <cctag/Params.hpp>

#include <boost/json.hpp>
#include <boost/json/src.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <type_traits>

namespace cctag::portable::test {

namespace {

Dtype parse_dtype(const std::string& name, const std::string& tensor) {
    if (name == "U8") {
        return Dtype::u8;
    }
    if (name == "I16") {
        return Dtype::i16;
    }
    if (name == "I32") {
        return Dtype::i32;
    }
    if (name == "F32") {
        return Dtype::f32;
    }
    throw std::runtime_error(tensor + ": unsupported dtype " + name);
}

constexpr const char* kStageNames[] = {
    "pyramid",
    "gradient",
    "edges",
    "edge_points",
    "vote",
    "linking",
    "candidates",
    "markers",
};

} // namespace

const char* stage_name(Stage stage) {
    return kStageNames[static_cast<int>(stage)];
}

std::optional<Stage> parse_stage(const std::string& name) {
    for (int i = 0; i < static_cast<int>(std::size(kStageNames)); ++i) {
        if (name == kStageNames[i]) {
            return static_cast<Stage>(i);
        }
    }
    return std::nullopt;
}

const char* dtype_name(Dtype dtype) {
    switch (dtype) {
        case Dtype::u8:
            return "U8";
        case Dtype::i16:
            return "I16";
        case Dtype::i32:
            return "I32";
        case Dtype::f32:
            return "F32";
    }
    return "?";
}

std::size_t dtype_size(Dtype dtype) {
    switch (dtype) {
        case Dtype::u8:
            return 1;
        case Dtype::i16:
            return 2;
        case Dtype::i32:
        case Dtype::f32:
            return 4;
    }
    return 0;
}

std::size_t Tensor::elements() const {
    std::size_t count = 1;
    for (const std::uint64_t extent : shape) {
        count *= static_cast<std::size_t>(extent);
    }
    return count;
}

void Tensor::expect_dtype(Dtype expected) const {
    if (dtype != expected) {
        throw std::runtime_error(
            name + ": expected " + dtype_name(expected) + ", the tensor is " + dtype_name(dtype)
        );
    }
}

ReferenceSnapshot ReferenceSnapshot::read(const std::filesystem::path& file) {
    std::ifstream stream(file, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open " + file.string());
    }
    std::vector<std::uint8_t> bytes(
        (std::istreambuf_iterator<char>(stream)),
        std::istreambuf_iterator<char>()
    );
    return from_bytes(std::move(bytes));
}

ReferenceSnapshot ReferenceSnapshot::from_bytes(std::vector<std::uint8_t> bytes) {
    ReferenceSnapshot snapshot;
    snapshot.bytes = std::move(bytes);
    const std::vector<std::uint8_t>& data = snapshot.bytes;
    if (data.size() < 8) {
        throw std::runtime_error("safetensors: file shorter than its header length field");
    }
    std::uint64_t header_length = 0;
    for (int i = 7; i >= 0; --i) {
        header_length = (header_length << 8) | data[static_cast<std::size_t>(i)];
    }
    if (header_length > data.size() - 8) {
        throw std::runtime_error("safetensors: header length exceeds the file");
    }
    const std::string_view header(
        reinterpret_cast<const char*>(data.data() + 8),
        static_cast<std::size_t>(header_length)
    );
    const std::span<const std::uint8_t> payload(
        data.data() + 8 + header_length,
        data.size() - 8 - header_length
    );

    boost::json::value document;
    try {
        document = boost::json::parse(header);
    } catch (const boost::system::system_error& error) {
        throw std::runtime_error(
            std::string("safetensors: malformed JSON header: ") + error.what()
        );
    }
    if (!document.is_object()) {
        throw std::runtime_error("safetensors: the JSON header is not an object");
    }
    const auto integer = [](const boost::json::value& value, const std::string& what) {
        if (!(value.is_uint64() || (value.is_int64() && value.get_int64() >= 0))) {
            throw std::runtime_error(what + ": expected a non-negative JSON integer");
        }
        return value.to_number<std::uint64_t>();
    };
    const auto string = [](const boost::json::value& value, const std::string& what) {
        if (!value.is_string()) {
            throw std::runtime_error(what + ": expected a JSON string");
        }
        return std::string(value.get_string());
    };
    for (const auto& [name, entry] : document.get_object()) {
        if (name == "__metadata__") {
            if (!entry.is_object()) {
                throw std::runtime_error("__metadata__: expected a JSON object");
            }
            for (const auto& [key, value] : entry.get_object()) {
                snapshot.metadata_entries.emplace(
                    key,
                    string(value, "__metadata__." + std::string(key))
                );
            }
            continue;
        }
        const std::string tensor_name(name);
        if (!entry.is_object()) {
            throw std::runtime_error(tensor_name + ": expected a JSON object");
        }
        const boost::json::object& fields = entry.get_object();
        const auto field = [&](const char* key) -> const boost::json::value& {
            const boost::json::value* found = fields.if_contains(key);
            if (found == nullptr) {
                throw std::runtime_error(tensor_name + ": missing " + key);
            }
            return *found;
        };
        Tensor tensor;
        tensor.name = tensor_name;
        tensor.dtype = parse_dtype(string(field("dtype"), tensor_name + ".dtype"), tensor_name);
        const boost::json::value& shape = field("shape");
        if (!shape.is_array()) {
            throw std::runtime_error(tensor_name + ".shape: expected a JSON array");
        }
        for (const boost::json::value& extent : shape.get_array()) {
            tensor.shape.push_back(integer(extent, tensor_name + ".shape"));
        }
        const boost::json::value& offsets = field("data_offsets");
        if (!offsets.is_array() || offsets.get_array().size() != 2) {
            throw std::runtime_error(tensor_name + ": data_offsets must have two entries");
        }
        const std::uint64_t begin = integer(offsets.get_array()[0], tensor_name + ".data_offsets");
        const std::uint64_t end = integer(offsets.get_array()[1], tensor_name + ".data_offsets");
        if (begin > end || end > payload.size()) {
            throw std::runtime_error(tensor_name + ": data_offsets outside the data section");
        }
        if (end - begin != tensor.elements() * dtype_size(tensor.dtype)) {
            throw std::runtime_error(tensor_name + ": data_offsets disagree with dtype and shape");
        }
        tensor.bytes =
            payload.subspan(static_cast<std::size_t>(begin), static_cast<std::size_t>(end - begin));
        snapshot.tensor_entries.emplace(tensor_name, std::move(tensor));
    }
    if (snapshot.metadata_entries.empty()) {
        throw std::runtime_error("safetensors: no __metadata__ (not a stage snapshot)");
    }
    return snapshot;
}

const std::string& ReferenceSnapshot::meta(const std::string& key) const {
    const auto found = metadata_entries.find(key);
    if (found == metadata_entries.end()) {
        throw std::runtime_error("__metadata__." + key + " is absent");
    }
    return found->second;
}

std::uint32_t ReferenceSnapshot::meta_u32(const std::string& key) const {
    const std::string& text = meta(key);
    std::uint32_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::runtime_error("__metadata__." + key + " is not an integer: " + text);
    }
    return value;
}

std::vector<Stage> ReferenceSnapshot::stages() const {
    std::vector<Stage> result;
    std::stringstream list(meta("stages"));
    std::string item;
    while (std::getline(list, item, ',')) {
        const std::optional<Stage> stage = parse_stage(item);
        if (!stage) {
            throw std::runtime_error("__metadata__.stages names an unknown stage: " + item);
        }
        result.push_back(*stage);
    }
    return result;
}

bool ReferenceSnapshot::has(Stage stage) const {
    const std::vector<Stage> listed = stages();
    return std::find(listed.begin(), listed.end(), stage) != listed.end();
}

const Tensor& ReferenceSnapshot::tensor(const std::string& name) const {
    const auto found = tensor_entries.find(name);
    if (found == tensor_entries.end()) {
        throw std::runtime_error("tensor " + name + " is absent from the snapshot");
    }
    return found->second;
}

const Tensor&
ReferenceSnapshot::tensor(Stage stage, std::uint32_t level, const std::string& name) const {
    return tensor(std::string(stage_name(stage)) + "/level" + std::to_string(level) + "/" + name);
}

std::optional<std::filesystem::path> reference_snapshots_dir() {
    const char* value = std::getenv("CCTAG_REFERENCE_SNAPSHOTS");
    if (value == nullptr || *value == '\0') {
        return std::nullopt;
    }
    return std::filesystem::path(value);
}

std::vector<std::filesystem::path> reference_snapshot_files() {
    std::vector<std::filesystem::path> files;
    const std::optional<std::filesystem::path> dir = reference_snapshots_dir();
    if (!dir) {
        return files;
    }
    for (const auto& entry : std::filesystem::directory_iterator(*dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

namespace {
/// Throws unless every index is below `limit`, or is -1 when `permit_absent`
void validate_indices(
    std::span<const std::int32_t> indices,
    std::size_t limit,
    const char* name,
    bool permit_absent = false
) {
    if (!std::ranges::all_of(indices, [&](auto value) {
        return (permit_absent && value == -1) || (value >= 0 && std::size_t(value) < limit);
    })) {
        throw std::runtime_error(std::string(name) + ": index outside captured collection");
    }
}
/// Throws unless `offsets` delimit `rows` rows of `values` and every value is below `limit`
void validate_csr(
    std::span<const std::int32_t> offsets,
    std::span<const std::int32_t> values,
    std::size_t rows,
    std::size_t limit,
    const char* name
) {
    if (offsets.size() != rows + 1 || offsets.front() != 0 || offsets.back() < 0
        || std::size_t(offsets.back()) != values.size() || !std::ranges::is_sorted(offsets)) {
        throw std::runtime_error(std::string(name) + ": invalid CSR offsets");
    }
    validate_indices(values, limit, name);
}
void validate_finite(std::span<const float> values, const char* name) {
    if (!std::ranges::all_of(values, [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error(std::string(name) + ": nonfinite replay value");
    }
}
} // namespace

void fill_level(
    const ReferenceSnapshot& snapshot,
    std::uint32_t level,
    Stage upto,
    cpu::Buffers& buffers
) {
    copy_plane<std::uint8_t>(snapshot.tensor(Stage::pyramid, level, "src"), buffers.src_plane());
    if (upto == Stage::pyramid) {
        return;
    }
    copy_plane<std::int16_t>(snapshot.tensor(Stage::gradient, level, "dx"), buffers.dx_plane());
    copy_plane<std::int16_t>(snapshot.tensor(Stage::gradient, level, "dy"), buffers.dy_plane());
    if (upto == Stage::gradient) {
        return;
    }
    copy_plane<std::uint8_t>(snapshot.tensor(Stage::edges, level, "edges"), buffers.edges_plane());
    if (upto == Stage::edges) {
        return;
    }
    const Tensor& xy = snapshot.tensor(Stage::edge_points, level, "xy");
    const Tensor& gradients = snapshot.tensor(Stage::edge_points, level, "gradients");
    if (xy.shape.size() != 2 || xy.shape[1] != 2 || xy.shape[0] > cpu::kMaxEdgePoints
        || gradients.shape != xy.shape) {
        throw std::runtime_error("edge_points: expected matching [n, 2] coordinates and gradients");
    }
    buffers.xy = xy.as<std::int32_t>();
    buffers.gradients = gradients.as<float>();
    buffers.n = static_cast<std::uint32_t>(xy.shape[0]);
    validate_finite(buffers.gradients, "edge point gradients");

    // The edge map is not captured, so restore it from the reference's canonical coordinates
    buffers.edge_map.setTo(-1);
    for (std::uint32_t index = 0; index < buffers.n; ++index) {
        const auto x = buffers.xy[2 * index];
        const auto y = buffers.xy[2 * index + 1];
        if (x < 0 || y < 0 || static_cast<std::uint32_t>(x) >= buffers.width
            || static_cast<std::uint32_t>(y) >= buffers.height) {
            throw std::runtime_error("edge_points: coordinate outside the pyramid level");
        }
        if (buffers.edge_map(y, x) != -1) {
            throw std::runtime_error("edge_points: duplicate coordinate");
        }
        buffers.edge_map(y, x) = static_cast<std::int32_t>(index);
    }
    if (upto == Stage::edge_points) {
        return;
    }
    buffers.links = snapshot.tensor(Stage::vote, level, "links").as<std::int32_t>();
    buffers.voters_offsets =
        snapshot.tensor(Stage::vote, level, "voters/offsets").as<std::int32_t>();
    buffers.voters_values = snapshot.tensor(Stage::vote, level, "voters/values").as<std::int32_t>();
    if (snapshot.metadata().contains("replay_version")) {
        const auto prefix = "replay/vote/level" + std::to_string(level) + "/";
        buffers.voters_offsets = snapshot.tensor(prefix + "voter_offsets").as<std::int32_t>();
        buffers.voters_values = snapshot.tensor(prefix + "voter_values").as<std::int32_t>();
    }
    buffers.is_max = snapshot.tensor(Stage::vote, level, "is_max").as<std::int32_t>();
    buffers.flow_length = snapshot.tensor(Stage::vote, level, "flow_length").as<float>();
    buffers.seeds = snapshot.tensor(Stage::vote, level, "seeds").as<std::int32_t>();
    buffers.seed_order = snapshot.tensor(Stage::vote, level, "seed_order").as<std::int32_t>();
    if (buffers.links.size() != 2 * std::size_t(buffers.n) || buffers.is_max.size() != buffers.n
        || buffers.flow_length.size() != buffers.n) {
        throw std::runtime_error("vote: replay row count differs from edge points");
    }
    validate_indices(buffers.links, buffers.n, "vote links", true);
    validate_csr(buffers.voters_offsets, buffers.voters_values, buffers.n, buffers.n, "voters");
    validate_indices(buffers.seeds, buffers.n, "vote seeds");
    validate_indices(buffers.seed_order, buffers.n, "vote seed order");
    auto ordered_seeds = buffers.seed_order;
    auto seed_set = buffers.seeds;
    std::ranges::sort(ordered_seeds);
    std::ranges::sort(seed_set);
    if (ordered_seeds != seed_set
        || std::adjacent_find(seed_set.begin(), seed_set.end()) != seed_set.end()) {
        throw std::runtime_error("vote: seed order must be a permutation of unique seeds");
    }
    validate_finite(buffers.flow_length, "vote flow length");
    if (upto == Stage::vote) {
        return;
    }
    buffers.link_seeds = snapshot.tensor(Stage::linking, level, "seeds").as<std::int32_t>();
    buffers.segment_offsets =
        snapshot.tensor(Stage::linking, level, "segments/offsets").as<std::int32_t>();
    buffers.segment_values =
        snapshot.tensor(Stage::linking, level, "segments/values").as<std::int32_t>();
    buffers.child_counts =
        snapshot.tensor(Stage::linking, level, "child_counts").as<std::int32_t>();
    buffers.avg_vote = snapshot.tensor(Stage::linking, level, "avg_vote").as<float>();
    const auto segment_count = buffers.link_seeds.size();
    validate_indices(buffers.link_seeds, buffers.n, "linking seeds");
    validate_csr(
        buffers.segment_offsets,
        buffers.segment_values,
        segment_count,
        buffers.n,
        "linking segments"
    );
    if (buffers.child_counts.size() != segment_count || buffers.avg_vote.size() != segment_count
        || !std::ranges::all_of(buffers.child_counts, [](auto value) { return value >= 0; })) {
        throw std::runtime_error("linking: invalid replay row counts");
    }
    validate_finite(buffers.avg_vote, "linking average vote");
    if (snapshot.metadata().contains("replay_version")) {
        const auto prefix = "replay/linking/level" + std::to_string(level) + "/";
        buffers.children_offsets = snapshot.tensor(prefix + "child_offsets").as<std::int32_t>();
        buffers.children_values = snapshot.tensor(prefix + "child_values").as<std::int32_t>();
        buffers.loop_one_order = snapshot.tensor(prefix + "order").as<std::int32_t>();
        if (buffers.children_offsets.size() != buffers.link_seeds.size() + 1
            || buffers.loop_one_order.size() != buffers.link_seeds.size()) {
            throw std::runtime_error("linking replay row count differs from captured segments");
        }
        validate_csr(
            buffers.children_offsets,
            buffers.children_values,
            segment_count,
            buffers.n,
            "linking children"
        );
        validate_indices(buffers.loop_one_order, segment_count, "linking order");
        auto order = buffers.loop_one_order;
        std::ranges::sort(order);
        if (std::adjacent_find(order.begin(), order.end()) != order.end()) {
            throw std::runtime_error("linking replay order is not a permutation");
        }
        for (std::size_t i = 0; i < segment_count; ++i) {
            if (buffers.child_counts[i]
                != buffers.children_offsets[i + 1] - buffers.children_offsets[i]) {
                throw std::runtime_error(
                    "linking replay child count differs from ordered children"
                );
            }
        }
        return;
    }
    // Without replay tensors, rebuild the children and candidate order from segments and votes
    const auto count = buffers.link_seeds.size();
    buffers.children_offsets.assign(1, 0);
    buffers.children_values.clear();
    std::vector<std::size_t> acceptance_rank(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto segment = std::span<const std::int32_t>(buffers.segment_values)
                                 .subspan(
                                     buffers.segment_offsets[i],
                                     buffers.segment_offsets[i + 1] - buffers.segment_offsets[i]
                                 );
        const auto minimum = kernels::child_vote_min(segment, buffers.voters_offsets);
        for (const auto point : segment) {
            const auto begin = buffers.voters_offsets[point],
                       end = buffers.voters_offsets[point + 1];
            if (end - begin >= minimum) {
                buffers.children_values.insert(
                    buffers.children_values.end(),
                    buffers.voters_values.begin() + begin,
                    buffers.voters_values.begin() + end
                );
            }
        }
        const auto children = buffers.children_values.size() - buffers.children_offsets.back();
        if (children != static_cast<std::size_t>(buffers.child_counts[i])) {
            throw std::runtime_error(
                "linking: reference child count disagrees with the vote graph"
            );
        }
        buffers.children_offsets.push_back(
            static_cast<std::int32_t>(buffers.children_values.size())
        );
        const auto seed =
            std::find(buffers.seed_order.begin(), buffers.seed_order.end(), buffers.link_seeds[i]);
        if (seed == buffers.seed_order.end()) {
            throw std::runtime_error("linking: reference segment seed is absent from seed_order");
        }
        acceptance_rank[i] = seed - buffers.seed_order.begin();
    }
    buffers.loop_one_order.resize(count);
    std::iota(buffers.loop_one_order.begin(), buffers.loop_one_order.end(), 0);
    std::sort(buffers.loop_one_order.begin(), buffers.loop_one_order.end(), [&](auto a, auto b) {
        return buffers.avg_vote[a] > buffers.avg_vote[b]
            || (buffers.avg_vote[a] == buffers.avg_vote[b]
                && acceptance_rank[a] > acceptance_rank[b]);
    });
    if (upto == Stage::linking) {
        return;
    }
    throw std::logic_error(
        std::string("fill_level: the stage buffers stop at linking, add the fill for ")
        + stage_name(upto) + " together with its buffers"
    );
}

void fill_context(const ReferenceSnapshot& snapshot, Stage upto, Context<cpu::Backend>& context) {
    const cctag::Parameters params = reference_parameters(snapshot);
    context.ensure(snapshot.image_width(), snapshot.image_height(), params);
    if (context.levels.size() != snapshot.processed_levels()) {
        throw std::runtime_error(
            "the context has " + std::to_string(context.levels.size())
            + " levels, the snapshot's processed_levels is "
            + std::to_string(snapshot.processed_levels())
        );
    }
    for (std::uint32_t level = 0; level < context.levels.size(); ++level) {
        fill_level(snapshot, level, upto, context.levels[level]);
    }
}

cctag::Parameters reference_parameters(const ReferenceSnapshot& snapshot) {
    cctag::Parameters params(snapshot.crowns());
    if (!snapshot.metadata().contains("replay_version")) {
        return params;
    }
    if (snapshot.meta("replay_version") != "1" || snapshot.meta("semantic_params_version") != "1") {
        throw std::runtime_error("unsupported CUDA replay schema");
    }
    const auto values = boost::json::parse(snapshot.meta("effective_parameters")).as_object();
    const auto field = [&]<class T>(const char* name, T& target) {
        if constexpr (std::is_same_v<T, bool>) {
            const int value = boost::json::value_to<int>(values.at(name));
            if (value != 0 && value != 1) {
                throw std::runtime_error("invalid boolean replay parameter");
            }
            target = value;
        } else {
            target = boost::json::value_to<T>(values.at(name));
        }
    };
#define P(name) field(#name, params.name)
    P(_cannyThrLow);
    P(_cannyThrHigh);
    P(_distSearch);
    P(_thrGradientMagInVote);
    P(_angleVoting);
    P(_ratioVoting);
    P(_averageVoteMin);
    P(_thrMedianDistanceEllipse);
    P(_maximumNbSeeds);
    P(_maximumNbCandidatesLoopTwo);
    P(_nCrowns);
    P(_nCircles);
    P(_minPointsSegmentCandidate);
    P(_minVotesToSelectCandidate);
    P(_threshRobustEstimationOfOuterEllipse);
    P(_ellipseGrowingEllipticHullWidth);
    P(_windowSizeOnInnerEllipticSegment);
    P(_numberOfMultiresLayers);
    P(_numberOfProcessedMultiresLayers);
    P(_nSamplesOuterEllipse);
    P(_numCutsInIdentStep);
    P(_numSamplesOuterEdgePointsRefinement);
    P(_cutsSelectionTrials);
    P(_sampleCutLength);
    P(_imagedCenterNGridSample);
    P(_imagedCenterNeighbourSize);
    P(_minIdentProba);
    P(_useLMDif);
    P(_searchForAnotherSegment);
    P(_writeOutput);
    P(_doIdentification);
    P(_maxEdges);
    P(_useCuda);
    P(_pinnedCounters);
    P(_pinnedNearbyPoints);
#undef P
    params._debugDir = snapshot.meta("debug_dir");
    if (params._numberOfProcessedMultiresLayers != snapshot.processed_levels()
        || params._nCrowns != snapshot.crowns()) {
        throw std::runtime_error("replay parameter identity differs");
    }
    return params;
}

void fill_identification(const ReferenceSnapshot& snapshot, Context<cpu::Backend>& context) {
    fill_context(snapshot, Stage::pyramid, context);
    if (snapshot.meta("replay_version") != "1") {
        throw std::runtime_error("identification requires replay version 1");
    }
    const std::string prefix = "replay/identification/";
    const auto floats = [&](const char* name, std::size_t rows, std::size_t columns) {
        const auto& tensor = snapshot.tensor(prefix + name);
        if (tensor.shape != std::vector<std::uint64_t>{rows, columns}) {
            throw std::runtime_error("invalid identification replay shape: " + tensor.name);
        }
        auto values = tensor.as<float>();
        if (!std::ranges::all_of(values, [](float value) { return std::isfinite(value); })) {
            throw std::runtime_error("nonfinite identification replay: " + tensor.name);
        }
        return values;
    };
    const auto& centers_tensor = snapshot.tensor(prefix + "centers");
    if (centers_tensor.shape.size() != 2) {
        throw std::runtime_error("invalid replay centers");
    }
    const std::size_t count = centers_tensor.shape[0];
    const auto centers = floats("centers", count, 2), ellipse = floats("ellipse", count, 5),
               conics = floats("ellipse_matrix", count, 9),
               homographies = floats("homography", count, 9), scales = floats("scale", count, 1),
               qualities = floats("quality", count, 1);
    const auto ints = [&](const char* name, std::size_t rows) {
        const auto& tensor = snapshot.tensor(prefix + name);
        if (tensor.shape != std::vector<std::uint64_t>{rows}) {
            throw std::runtime_error("invalid replay integer shape");
        }
        return tensor.as<std::int32_t>();
    };
    const auto levels = ints("level", count), ids = ints("id", count),
               statuses = ints("status", count), offsets = ints("point_offsets", count + 1);
    const auto& points_tensor = snapshot.tensor(prefix + "directed_points");
    if (points_tensor.shape.size() != 2) {
        throw std::runtime_error("invalid replay points");
    }
    const auto points = floats("directed_points", points_tensor.shape[0], 4);
    if (offsets.front() != 0 || offsets.back() < 0
        || std::size_t(offsets.back()) != points.size() / 4 || !std::ranges::is_sorted(offsets)) {
        throw std::runtime_error("invalid replay point offsets");
    }
    context.candidate_markers.clear();
    for (std::size_t i = 0; i < count; ++i) {
        CandidateMarker candidate;
        candidate.center = {centers[2 * i], centers[2 * i + 1]};
        candidate.level = levels[i];
        candidate.id = ids[i];
        candidate.status = statuses[i];
        candidate.scale = scales[i];
        candidate.quality = qualities[i];
        if (candidate.level < 0 || std::size_t(candidate.level) >= context.levels.size()
            || candidate.scale <= 0
            || !candidate.rescaled_outer_ellipse.set_parameters(
                ellipse[5 * i],
                ellipse[5 * i + 1],
                ellipse[5 * i + 2],
                ellipse[5 * i + 3],
                ellipse[5 * i + 4]
            )) {
            throw std::runtime_error("invalid replay candidate geometry");
        }
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                candidate.rescaled_outer_ellipse.conic(row, col) = conics[9 * i + 3 * row + col];
                candidate.homography(row, col) = homographies[9 * i + 3 * row + col];
            }
        }
        for (auto j = offsets[i]; j < offsets[i + 1]; ++j) {
            candidate.outer_points.push_back(
                {points[4 * j], points[4 * j + 1], points[4 * j + 2], points[4 * j + 3]}
            );
        }
        context.candidate_markers.push_back(candidate);
    }
    auto bank_offsets = snapshot.tensor(prefix + "bank_offsets").as<std::int32_t>();
    auto bank_values = snapshot.tensor(prefix + "bank_values").as<float>();
    if (bank_offsets.size() < 2 || bank_offsets.front() != 0 || bank_offsets.back() < 0
        || std::size_t(bank_offsets.back()) != bank_values.size()
        || !std::ranges::is_sorted(bank_offsets)) {
        throw std::runtime_error("invalid replay bank offsets");
    }
    for (std::size_t i = 1; i < bank_offsets.size(); ++i) {
        if (bank_offsets[i] - bank_offsets[i - 1] != 2 * snapshot.crowns() - 1) {
            throw std::runtime_error("invalid replay bank width");
        }
    }
    if (!std::ranges::all_of(bank_values, [](float value) {
        return std::isfinite(value) && value > 0;
    })) {
        throw std::runtime_error("invalid replay bank ratios");
    }
    context.bank.custom = true;
    context.bank.ratios = std::move(bank_values);
    context.bank.offsets.assign(bank_offsets.begin(), bank_offsets.end());
}

} // namespace cctag::portable::test

#ifdef CCTAG_TEST
#include <boost/ut.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cctag::portable::test {

namespace {

/// Builds safetensors bytes from a JSON header and tensor data
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

/// Builds a snapshot from named tensors, laying out the header and byte offsets itself.
/// The parser tests use hand-built bytes instead, so a layout error here cannot hide there.
struct SnapshotBuilder {
    // A stage snapshot must have metadata
    boost::json::object metadata{{"problem", "test"}};
    std::map<std::string, std::pair<boost::json::object, std::vector<std::uint8_t>>> tensors;

    template <class T>
    SnapshotBuilder&
    add(const std::string& name, std::vector<std::uint64_t> shape, const std::vector<T>& values) {
        std::vector<std::uint8_t> bytes(values.size() * sizeof(T));
        if (!values.empty()) {
            std::memcpy(bytes.data(), values.data(), bytes.size());
        }
        tensors[name] = {
            {{"dtype", dtype_name(dtype_of<T>())},
             {"shape", boost::json::array(shape.begin(), shape.end())}},
            std::move(bytes)
        };
        return *this;
    }

    /// Adds the four planes of a `width` by 1 level 0, with every pixel marked as an edge
    SnapshotBuilder& edge_row(std::uint64_t width) {
        add<std::uint8_t>("pyramid/level0/src", {1, width}, std::vector<std::uint8_t>(width));
        add<std::int16_t>("gradient/level0/dx", {1, width}, std::vector<std::int16_t>(width));
        add<std::int16_t>("gradient/level0/dy", {1, width}, std::vector<std::int16_t>(width));
        return add<std::uint8_t>(
            "edges/level0/edges",
            {1, width},
            std::vector<std::uint8_t>(width, 255)
        );
    }

    ReferenceSnapshot build() const {
        boost::json::object header{{"__metadata__", metadata}};
        std::vector<std::uint8_t> data;
        for (const auto& [name, tensor] : tensors) {
            auto entry = tensor.first;
            entry["data_offsets"] = {data.size(), data.size() + tensor.second.size()};
            header[name] = entry;
            data.insert(data.end(), tensor.second.begin(), tensor.second.end());
        }
        return ReferenceSnapshot::from_bytes(safetensors(boost::json::serialize(header), data));
    }
};

/// Two edge points with hand-chosen vote and linking outputs. The vote and linking orders
/// deliberately differ from index order, so a fill that sorts or rebuilds them is caught.
SnapshotBuilder two_point_linking() {
    SnapshotBuilder snapshot;
    snapshot.edge_row(2)
        .add<std::int32_t>("edge_points/level0/xy", {2, 2}, {0, 0, 1, 0})
        .add<float>("edge_points/level0/gradients", {2, 2}, {0, 0, 0, 0})
        .add<std::int32_t>("vote/level0/links", {2, 2}, {-1, 1, 0, -1})
        .add<std::int32_t>("vote/level0/voters/offsets", {3}, {0, 1, 2})
        .add<std::int32_t>("vote/level0/voters/values", {2}, {1, 0})
        .add<std::int32_t>("vote/level0/is_max", {2}, {1, 1})
        .add<float>("vote/level0/flow_length", {2}, {1.5f, 2.5f})
        .add<std::int32_t>("vote/level0/seeds", {2}, {0, 1})
        .add<std::int32_t>("vote/level0/seed_order", {2}, {1, 0})
        .add<std::int32_t>("linking/level0/seeds", {1}, {1})
        .add<std::int32_t>("linking/level0/segments/offsets", {2}, {0, 2})
        .add<std::int32_t>("linking/level0/segments/values", {2}, {1, 0})
        .add<std::int32_t>("linking/level0/child_counts", {1}, {2})
        .add<float>("linking/level0/avg_vote", {1}, {2.f});
    return snapshot;
}

/// Adds replay tensors whose voter order differs from the canonical `voters/values`
SnapshotBuilder two_point_replay() {
    SnapshotBuilder snapshot = two_point_linking();
    snapshot.metadata["replay_version"] = "1";
    snapshot.add<std::int32_t>("replay/vote/level0/voter_offsets", {3}, {0, 1, 2})
        .add<std::int32_t>("replay/vote/level0/voter_values", {2}, {0, 1})
        .add<std::int32_t>("replay/linking/level0/child_offsets", {2}, {0, 2})
        .add<std::int32_t>("replay/linking/level0/child_values", {2}, {1, 0})
        .add<std::int32_t>("replay/linking/level0/order", {1}, {0});
    return snapshot;
}

} // namespace

using namespace boost::ut;

inline suite<"snapshot_support"> snapshot_support_suite = [] {
    "reads metadata and unaligned tensor values from hand built bytes"_test = [] {
        // Place the I16 plane at an odd byte offset to check reading unaligned values
        const std::string header =
            R"({"pyramid/level0/src": {"dtype": "U8", "shape": [2, 3], "data_offsets": [0, 6]},)"
            R"( "gradient/level0/dx": {"dtype": "I16", "shape": [1, 2], "data_offsets": [7, 11]},)"
            R"( "pad": {"dtype": "U8", "shape": [1], "data_offsets": [6, 7]},)"
            R"( "__metadata__": {"problem": "01", "crowns": "3", "stages": "pyramid,gradient"}})";
        const std::vector<std::uint8_t> data = {1, 2, 3, 4, 5, 6, 9, 0xFE, 0xFF, 0x02, 0x00};
        const ReferenceSnapshot snapshot = ReferenceSnapshot::from_bytes(safetensors(header, data));

        expect(eq(snapshot.problem(), std::string{"01"}));
        expect(eq(snapshot.crowns(), 3u));
        expect(eq(snapshot.stages().size(), 2u)) << fatal;
        expect(snapshot.stages()[1] == Stage::gradient);
        expect(snapshot.has(Stage::gradient));
        expect(!snapshot.has(Stage::vote));

        const Tensor& src = snapshot.tensor(Stage::pyramid, 0, "src");
        expect(src.dtype == Dtype::u8);
        expect(eq(src.shape.size(), 2u)) << fatal;
        expect(eq(src.shape[0], 2u));
        expect(eq(src.shape[1], 3u));
        const std::vector<std::uint8_t> pixels = src.as<std::uint8_t>();
        expect(eq(pixels.size(), 6u));
        expect(eq(pixels[5], 6));

        const std::vector<std::int16_t> dx =
            snapshot.tensor("gradient/level0/dx").as<std::int16_t>();
        expect(eq(dx.size(), 2u)) << fatal;
        expect(eq(dx[0], -2));
        expect(eq(dx[1], 2));

        expect(throws<std::runtime_error>([&] { (void)snapshot.tensor("gradient/level0/dy"); }));
        expect(throws<std::runtime_error>([&] { (void)src.as<std::int32_t>(); }));
        expect(throws<std::runtime_error>([&] { (void)snapshot.meta("image_width"); }));
    };

    "rejects tensor byte counts that disagree with shape or exceed the file"_test = [] {
        const auto snapshot = [](const char* offsets, std::size_t data_size) {
            const std::string header = std::string(R"({"src":{"dtype":"U8","shape":[2,3],)")
                + R"("data_offsets":)" + offsets + R"(},"__metadata__":{"problem":"test"}})";
            return ReferenceSnapshot::from_bytes(
                safetensors(header, std::vector<std::uint8_t>(data_size))
            );
        };
        // Six bytes for a 2x3 U8 tensor is valid, so the cases below fail only on their bytes
        expect(nothrow([&] { (void)snapshot("[0,6]", 6); }));
        expect(throws<std::runtime_error>([&] { (void)snapshot("[0,5]", 6); }));
        expect(throws<std::runtime_error>([&] { (void)snapshot("[0,6]", 5); }));
    };

    "fills planes and rejects planes that are missing or the wrong size"_test = [] {
        SnapshotBuilder builder;
        builder.add<std::uint8_t>("pyramid/level0/src", {2, 3}, {1, 2, 3, 4, 5, 6})
            .add<std::int16_t>("gradient/level0/dx", {2, 3}, std::vector<std::int16_t>(6));
        const ReferenceSnapshot snapshot = builder.build();

        cpu::Buffers buffers;
        buffers.ensure(3, 2);
        fill_level(snapshot, 0, Stage::pyramid, buffers);
        expect(eq(buffers.src(1, 1), 5));
        // The gradient stage also needs `dy`, which the snapshot lacks
        expect(throws<std::runtime_error>([&] {
            fill_level(snapshot, 0, Stage::gradient, buffers);
        }));

        cpu::Buffers transposed;
        transposed.ensure(2, 3);
        expect(throws<std::runtime_error>([&] {
            fill_level(snapshot, 0, Stage::pyramid, transposed);
        }));
    };

    "fills edge points from the snapshot and rebuilds the edge map from them"_test = [] {
        SnapshotBuilder builder;
        builder.edge_row(2)
            .add<std::int32_t>("edge_points/level0/xy", {1, 2}, {1, 0})
            .add<float>("edge_points/level0/gradients", {1, 2}, {-1.f, 2.f});
        const ReferenceSnapshot snapshot = builder.build();
        cpu::Buffers buffers;
        buffers.ensure(2, 1);
        buffers.edge_map.setTo(7);
        fill_level(snapshot, 0, Stage::edge_points, buffers);
        expect(eq(buffers.n, 1u));
        expect(buffers.xy == std::vector<std::int32_t>{1, 0});
        // The gradient planes are zero, so these values show the gradients were copied
        expect(buffers.gradients == std::vector<float>{-1.f, 2.f});
        expect(eq(buffers.edge_map(0, 0), -1));
        expect(eq(buffers.edge_map(0, 1), 0));
        // The snapshot has no vote tensors
        expect(throws<std::runtime_error>([&] { fill_level(snapshot, 0, Stage::vote, buffers); }));
    };

    "fill rejects edge point coordinates outside the level"_test = [] {
        cpu::Buffers buffers;
        buffers.ensure(2, 1);
        for (const auto& xy : {std::vector<std::int32_t>{2, 0}, std::vector<std::int32_t>{0, 1}}) {
            SnapshotBuilder builder;
            builder.edge_row(2)
                .add<std::int32_t>("edge_points/level0/xy", {1, 2}, xy)
                .add<float>("edge_points/level0/gradients", {1, 2}, {0.f, 0.f});
            const ReferenceSnapshot snapshot = builder.build();
            expect(throws<std::runtime_error>([&] {
                fill_level(snapshot, 0, Stage::edge_points, buffers);
            }));
        }
    };

    "fills an empty edge point collection and clears the previous edge map"_test = [] {
        SnapshotBuilder builder;
        builder.edge_row(1)
            .add<std::int32_t>("edge_points/level0/xy", {0, 2}, {})
            .add<float>("edge_points/level0/gradients", {0, 2}, {});
        cpu::Buffers buffers;
        buffers.ensure(1, 1);
        buffers.n = 1;
        buffers.xy = {0, 0};
        buffers.gradients = {1.f, -1.f};
        buffers.edge_map.setTo(0);
        fill_level(builder.build(), 0, Stage::edge_points, buffers);
        expect(eq(buffers.n, 0u));
        expect(buffers.xy.empty());
        expect(buffers.gradients.empty());
        expect(eq(buffers.edge_map(0, 0), -1));
    };

    "fills vote and linking without changing their stored orders"_test = [] {
        const ReferenceSnapshot snapshot = two_point_linking().build();
        cpu::Buffers buffers;
        buffers.ensure(2, 1);
        fill_level(snapshot, 0, Stage::vote, buffers);
        expect(buffers.links == std::vector<std::int32_t>{-1, 1, 0, -1});
        expect(buffers.voters_offsets == std::vector<std::int32_t>{0, 1, 2});
        expect(buffers.voters_values == std::vector<std::int32_t>{1, 0});
        expect(buffers.is_max == std::vector<std::int32_t>{1, 1});
        expect(buffers.flow_length == std::vector<float>{1.5f, 2.5f});
        expect(buffers.seeds == std::vector<std::int32_t>{0, 1});
        expect(buffers.seed_order == std::vector<std::int32_t>{1, 0});

        fill_level(snapshot, 0, Stage::linking, buffers);
        const LinkingHost linking = buffers.linking_view();
        expect(eq(linking.c, 1u));
        expect(std::ranges::equal(linking.seeds, std::array{1}));
        expect(std::ranges::equal(linking.segment_offsets, std::array{0, 2}));
        expect(std::ranges::equal(linking.segment_values, std::array{1, 0}));
        expect(std::ranges::equal(linking.child_counts, std::array{2}));
        expect(std::ranges::equal(linking.avg_vote, std::array{2.f}));
    };

    "fill supports stages only through linking"_test = [] {
        cpu::Buffers buffers;
        buffers.ensure(2, 1);
        expect(throws<std::logic_error>([&] {
            fill_level(two_point_linking().build(), 0, Stage::candidates, buffers);
        }));
    };

    "replay tensors replace the canonical voter order and supply children and order"_test = [] {
        cpu::Buffers buffers;
        buffers.ensure(2, 1);
        fill_level(two_point_replay().build(), 0, Stage::linking, buffers);
        expect(buffers.voters_values == std::vector<std::int32_t>{0, 1});
        expect(buffers.children_values == std::vector<std::int32_t>{1, 0});
        expect(buffers.loop_one_order == std::vector<std::int32_t>{0});
    };

    "fill rejects replay indices and offsets out of range"_test = [] {
        const std::vector<std::pair<std::string, std::vector<std::int32_t>>> corrupted = {
            {"replay/vote/level0/voter_offsets", {127, 1, 2}},
            {"replay/vote/level0/voter_values", {127, 1}},
            {"replay/linking/level0/child_offsets", {127, 2}},
            {"replay/linking/level0/child_values", {127, 0}},
            {"replay/linking/level0/order", {127}},
        };
        cpu::Buffers buffers;
        buffers.ensure(2, 1);
        for (const auto& [name, values] : corrupted) {
            SnapshotBuilder builder = two_point_replay();
            builder.add<std::int32_t>(name, {values.size()}, values);
            const ReferenceSnapshot snapshot = builder.build();
            expect(throws<std::runtime_error>([&] {
                fill_level(snapshot, 0, Stage::linking, buffers);
            })) << name;
        }
    };
};

} // namespace cctag::portable::test
#endif // CCTAG_TEST
