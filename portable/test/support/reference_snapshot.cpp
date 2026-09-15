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

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numbers>
#include <numeric>
#include <sstream>
#include <tuple>

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

struct CandidateRow {
    std::array<float, 5> ellipse;
    std::int32_t level;
    float quality;
    std::size_t index = 0;
};

float center_distance(const CandidateRow& a, const CandidateRow& b) {
    const float x = a.ellipse[0] - b.ellipse[0], y = a.ellipse[1] - b.ellipse[1];
    return std::sqrt(x * x + y * y);
}

std::vector<CandidateRow> candidate_rows(CandidatesHost candidates) {
    if (candidates.ellipses.size() != std::size_t{candidates.n} * 5
        || candidates.levels.size() != candidates.n || candidates.quality.size() != candidates.n
        || !std::ranges::all_of(candidates.ellipses, [](float value) {
        return std::isfinite(value);
    }) || !std::ranges::all_of(candidates.quality, [](float value) {
        return std::isfinite(value);
    })) {
        throw std::runtime_error("candidates: inconsistent row counts or non-finite values");
    }
    std::vector<CandidateRow> rows;
    for (std::size_t i = 0; i < candidates.n; ++i) {
        CandidateRow row;
        std::copy_n(candidates.ellipses.begin() + 5 * i, 5, row.ellipse.begin());
        row.index = i;
        row.level = candidates.levels[i];
        row.quality = candidates.quality[i];
        rows.push_back(row);
    }
    // The snapshot probe canonicalizes before Rules::Tolerant deduplicates by quality
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        return std::tie(a.level, a.ellipse[1], a.ellipse[0])
            < std::tie(b.level, b.ellipse[1], b.ellipse[0]);
    });
    std::vector<CandidateRow> deduplicated;
    for (const auto& row : rows) {
        bool found = false;
        for (auto& current : deduplicated) {
            if (center_distance(row, current)
                < 0.5f * std::max(row.ellipse[3], current.ellipse[3])) {
                if (row.quality > current.quality) {
                    current = row;
                }
                found = true;
            }
        }
        if (!found) {
            deduplicated.push_back(row);
        }
    }
    return deduplicated;
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

    // The edge map is not captured; restore it from the reference's canonical coordinates
    buffers.edge_map.setTo(-1);
    for (std::uint32_t index = 0; index < buffers.n; ++index) {
        const auto x = buffers.xy[2 * index];
        const auto y = buffers.xy[2 * index + 1];
        if (x < 0 || y < 0 || static_cast<std::uint32_t>(x) >= buffers.width
            || static_cast<std::uint32_t>(y) >= buffers.height) {
            throw std::runtime_error("edge_points: coordinate outside the pyramid level");
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
    buffers.is_max = snapshot.tensor(Stage::vote, level, "is_max").as<std::int32_t>();
    buffers.flow_length = snapshot.tensor(Stage::vote, level, "flow_length").as<float>();
    buffers.seeds = snapshot.tensor(Stage::vote, level, "seeds").as<std::int32_t>();
    buffers.seed_order = snapshot.tensor(Stage::vote, level, "seed_order").as<std::int32_t>();
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
    // Rebuild the unstored children and candidate order from reference segments and votes
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
    const cctag::Parameters params(snapshot.crowns());
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

std::string describe(const Tensor& reference, const Mismatch& mismatch) {
    std::ostringstream out;
    out << reference.name << ": " << mismatch.count << " of " << mismatch.total
        << " elements differ";
    if (!mismatch.exact()) {
        out << ", first at index " << mismatch.first;
        if (reference.shape.size() == 2 && reference.shape[1] != 0) {
            out << " (y " << mismatch.first / reference.shape[1] << ", x "
                << mismatch.first % reference.shape[1] << ")";
        }
    }
    return out.str();
}

CandidateComparison
compare_candidates(CandidatesHost reference_view, CandidatesHost candidate_view) {
    const auto reference = candidate_rows(reference_view);
    const auto candidate = candidate_rows(candidate_view);
    CandidateComparison result;
    std::vector<std::tuple<float, std::size_t, std::size_t>> distances;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        for (std::size_t j = 0; j < candidate.size(); ++j) {
            distances.emplace_back(center_distance(reference[i], candidate[j]), i, j);
        }
    }
    std::sort(distances.begin(), distances.end());
    std::vector<bool> matched_reference(reference.size()), matched_candidate(candidate.size());
    for (const auto& [distance, i, j] : distances) {
        if (distance > 5.f || matched_reference[i] || matched_candidate[j]) {
            continue;
        }
        matched_reference[i] = matched_candidate[j] = true;
        const auto& left = reference[i].ellipse;
        const auto& right = candidate[j].ellipse;
        result.center_drift = std::max(result.center_drift, distance);
        result.pairs_passed &= distance <= 1.f;
        for (int axis : {2, 3}) {
            const float drift = std::abs(left[axis] - right[axis]);
            result.axis_drift = std::max(result.axis_drift, drift);
            result.pairs_passed &= drift <= 3.f || drift / std::abs(left[axis]) <= 0.05f;
        }
        if (left[2] != 0 && left[3] / left[2] > 1.1f) {
            const float angle =
                std::abs(std::remainder(left[4] - right[4], std::numbers::pi_v<float>));
            result.angle_drift = std::max(result.angle_drift, angle);
            result.pairs_passed &= angle <= 0.05f;
        }
    }
    for (std::size_t i = 0; i < reference.size(); ++i) {
        if (!matched_reference[i]) {
            result.unmatched_reference.push_back(reference[i].index);
        }
    }
    result.extra = std::count(matched_candidate.begin(), matched_candidate.end(), false);
    result.passed = result.pairs_passed && result.unmatched_reference.empty();
    return result;
}

std::string describe(const CandidateComparison& comparison) {
    std::ostringstream out;
    out << "candidates: " << comparison.unmatched_reference.size() << " unmatched reference rows, "
        << comparison.extra << " extra rows; worst drift: center " << comparison.center_drift
        << " px, axis " << comparison.axis_drift << " px, angle " << comparison.angle_drift
        << " rad";
    return out.str();
}

std::string snapshot_hash(const ReferenceSnapshot& snapshot) {
    using Digest = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;
    auto digest = [] {
        Digest result(EVP_MD_CTX_new(), EVP_MD_CTX_free);
        if (!result || EVP_DigestInit_ex(result.get(), EVP_sha256(), nullptr) != 1) {
            throw std::runtime_error("cannot initialize SHA-256");
        }
        return result;
    };
    auto update = [](const Digest& hash, const void* data, std::size_t size) {
        if (EVP_DigestUpdate(hash.get(), data, size) != 1) {
            throw std::runtime_error("cannot update SHA-256");
        }
    };
    auto finish = [](const Digest& hash) {
        std::array<unsigned char, 32> bytes;
        if (EVP_DigestFinal_ex(hash.get(), bytes.data(), nullptr) != 1) {
            throw std::runtime_error("cannot finish SHA-256");
        }
        return bytes;
    };
    auto hash = digest();
    const auto& stages = snapshot.meta("stages");
    update(hash, stages.data(), stages.size());
    for (const auto* name : kStageNames) {
        const Stage stage = *parse_stage(name);
        if (!snapshot.has(stage)) {
            continue;
        }
        auto stage_hash = digest();
        for (const auto& [key, tensor] : snapshot.tensors()) {
            if (!key.starts_with(std::string(name) + "/")) {
                continue;
            }
            update(stage_hash, key.c_str(), key.size() + 1);
            const std::string dtype = dtype_name(tensor.dtype);
            update(stage_hash, dtype.c_str(), dtype.size() + 1);
            for (const auto dimension : tensor.shape) {
                std::array<std::uint8_t, 8> bytes;
                for (int i = 0; i < 8; ++i) {
                    bytes[i] = static_cast<std::uint8_t>(dimension >> (8 * i));
                }
                update(stage_hash, bytes.data(), bytes.size());
            }
            update(stage_hash, "", 1);
            update(stage_hash, tensor.bytes.data(), tensor.bytes.size());
        }
        const auto bytes = finish(stage_hash);
        update(hash, bytes.data(), bytes.size());
    }
    std::string result = "sha256:";
    for (const auto byte : finish(hash)) {
        result += "0123456789abcdef"[byte >> 4];
        result += "0123456789abcdef"[byte & 15];
    }
    return result;
}

CandidateAllowance CandidateAllowance::read(const std::filesystem::path& file) {
    std::ifstream input(file);
    if (!input) {
        throw std::runtime_error("cannot read candidate allowance: " + file.string());
    }
    const std::string text{std::istreambuf_iterator<char>(input), {}};
    const auto value = boost::json::parse(text);
    const auto& object = value.as_object();
    return {
        boost::json::value_to<std::string>(object.at("variant")),
        boost::json::value_to<std::string>(object.at("snapshot_hash")),
        boost::json::value_to<std::size_t>(object.at("row")),
        boost::json::value_to<std::int32_t>(object.at("level")),
        boost::json::value_to<std::array<float, 5>>(object.at("ellipse")),
        boost::json::value_to<std::string>(object.at("reason")),
    };
}

bool CandidateAllowance::allows(
    const ReferenceSnapshot& reference,
    const CandidateComparison& comparison,
    std::string_view candidate_variant
) const {
    if (candidate_variant != variant || !comparison.pairs_passed || comparison.extra != 0
        || comparison.unmatched_reference != std::vector<std::size_t>{row}
        || test::snapshot_hash(reference) != snapshot_hash) {
        return false;
    }
    const auto levels = reference.tensor("candidates/level").as<std::int32_t>();
    const auto ellipses = reference.tensor("candidates/ellipse").as<float>();
    return row < levels.size() && levels[row] == level && row < ellipses.size() / 5
        && std::equal(ellipse.begin(), ellipse.end(), ellipses.begin() + 5 * row);
}

} // namespace cctag::portable::test

#ifdef CCTAG_TEST
#include <boost/ut.hpp>

#include <cstdint>
#include <limits>
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

struct CandidateFixture {
    std::vector<float> ellipses;
    std::vector<std::int32_t> levels;
    std::vector<float> quality;

    CandidateFixture(std::initializer_list<CandidateRow> rows) {
        for (const auto& row : rows) {
            ellipses.insert(ellipses.end(), row.ellipse.begin(), row.ellipse.end());
            levels.push_back(row.level);
            quality.push_back(row.quality);
        }
    }

    CandidatesHost view() const {
        return {static_cast<std::uint32_t>(levels.size()), ellipses, levels, quality};
    }
};

CandidateComparison compare_one(std::array<float, 5> reference, std::array<float, 5> candidate) {
    const CandidateFixture left{{reference, 0, 1.f}}, right{{candidate, 0, 1.f}};
    return compare_candidates(left.view(), right.view());
}

} // namespace

using namespace boost::ut;

inline suite<"snapshot_support"> snapshot_support_suite = [] {
    "snapshot content hash excludes metadata and follows the canonical byte contract"_test = [] {
        const std::string header =
            R"({"pyramid/level0/src":{"dtype":"U8","shape":[1,3],"data_offsets":[0,3]},)"
            R"("__metadata__":{"stages":"pyramid","problem":"test"}})";
        const auto snapshot = ReferenceSnapshot::from_bytes(safetensors(header, {1, 2, 3}));
        expect(eq(
            snapshot_hash(snapshot),
            std::string{"sha256:5ed943897df4f73004d2f3bc1d304280805dd0b89db1ba7c19fb33ff9fc5d972"}
        ));
        auto renamed = header;
        renamed.replace(renamed.find("test"), 4, "renamed");
        expect(
            eq(snapshot_hash(snapshot),
               snapshot_hash(ReferenceSnapshot::from_bytes(safetensors(renamed, {1, 2, 3}))))
        );
        expect(
            snapshot_hash(snapshot)
            != snapshot_hash(ReferenceSnapshot::from_bytes(safetensors(header, {1, 2, 4})))
        );
    };

    "candidate allowance accepts only its missing raw row and preserves other failures"_test = [] {
        const CandidateFixture reference{{{100, 0, 1, 2, 0}, 1, 1}, {{0, 0, 1, 2, 0}, 0, 1}};
        const CandidateFixture candidate{{{0, 0, 1, 2, 0}, 0, 1}};
        const std::string header =
            R"({"candidates/ellipse":{"dtype":"F32","shape":[2,5],"data_offsets":[0,40]},)"
            R"("candidates/level":{"dtype":"I32","shape":[2],"data_offsets":[40,48]},)"
            R"("candidates/quality":{"dtype":"F32","shape":[2],"data_offsets":[48,56]},)"
            R"("__metadata__":{"stages":"candidates"}})";
        std::vector<std::uint8_t> data(56);
        std::memcpy(data.data(), reference.ellipses.data(), 40);
        std::memcpy(data.data() + 40, reference.levels.data(), 8);
        std::memcpy(data.data() + 48, reference.quality.data(), 8);
        const auto snapshot = ReferenceSnapshot::from_bytes(safetensors(header, data));
        const CandidateAllowance
            allowance{"test/cpu", snapshot_hash(snapshot), 0, 1, {100, 0, 1, 2, 0}, "reviewed"};
        const auto comparison = compare_candidates(reference.view(), candidate.view());
        expect(!comparison.passed);
        expect(comparison.unmatched_reference == std::vector<std::size_t>{0});
        expect(allowance.allows(snapshot, comparison, "test/cpu"));
        expect(!comparison.passed); // Acceptance never changes the raw verdict
        expect(!allowance.allows(snapshot, comparison, "another/cpu"));
        for (int changed = 0; changed < 4; ++changed) {
            auto stale = allowance;
            if (changed == 0) {
                stale.snapshot_hash += "0";
            }
            if (changed == 1) {
                stale.row = 1;
            }
            if (changed == 2) {
                stale.level = 0;
            }
            if (changed == 3) {
                stale.ellipse[0] += 1;
            }
            expect(!stale.allows(snapshot, comparison, "test/cpu"));
        }
        data[55] ^= 1; // Even an unrelated tensor change invalidates the snapshot pin
        const auto moved = ReferenceSnapshot::from_bytes(safetensors(header, data));
        expect(!allowance.allows(moved, comparison, "test/cpu"));
        for (const CandidateFixture& bad : {
                 CandidateFixture{},
                 CandidateFixture{{{100, 0, 1, 2, 0}, 1, 1}},
                 CandidateFixture{{{2, 0, 1, 2, 0}, 0, 1}},
                 CandidateFixture{{{0, 0, 1, 6, 0}, 0, 1}},
                 CandidateFixture{{{0, 0, 1, 2, 0}, 0, 1}, {{200, 0, 1, 2, 0}, 0, 1}},
             }) {
            expect(!allowance.allows(
                snapshot,
                compare_candidates(reference.view(), bad.view()),
                "test/cpu"
            ));
        }
    };

    "candidate matching keeps the highest quality duplicate and allows extra rows"_test = [] {
        const CandidateFixture reference{{{0, 0, 10, 20, 0}, 0, 5}};
        const CandidateFixture candidate{
            {{4, 0, 6, 12, 1}, 1, 1},
            {{100, 0, 10, 20, 0}, 0, 1},
            {{0, 0, 10, 20, 0}, 2, 5},
        };
        const auto comparison = compare_candidates(reference.view(), candidate.view());
        expect(comparison.passed);
        expect(eq(comparison.unmatched_reference.size(), 0u));
        expect(eq(comparison.extra, 1u));
        expect(eq(comparison.center_drift, 0.f));
    };

    "candidate matching is one to one and requires every reference representative"_test = [] {
        const CandidateFixture reference{{{0, 0, 1, 1, 0}, 0, 1}, {{1, 0, 1, 1, 0}, 0, 1}};
        const CandidateFixture candidate{{{0.5f, 0, 1, 1, 0}, 0, 1}};
        const auto comparison = compare_candidates(reference.view(), candidate.view());
        expect(!comparison.passed);
        expect(eq(comparison.unmatched_reference.size(), 1u));
        expect(eq(comparison.extra, 0u));
        const CandidateFixture empty{};
        expect(!compare_candidates(reference.view(), empty.view()).passed);
        expect(compare_candidates(empty.view(), candidate.view()).passed);
    };

    "candidate matching distinguishes center tolerance from the pairing radius"_test = [] {
        const std::array<float, 5> reference{0, 0, 10, 20, 0};
        expect(compare_one(reference, {1, 0, 10, 20, 0}).passed);
        expect(!compare_one(reference, {1.01f, 0, 10, 20, 0}).passed);
        const auto paired = compare_one(reference, {5, 0, 10, 20, 0});
        expect(!paired.passed);
        expect(eq(paired.unmatched_reference.size(), 0u));
        const auto outside = compare_one(reference, {5.01f, 0, 10, 20, 0});
        expect(!outside.passed);
        expect(eq(outside.unmatched_reference.size(), 1u));
        expect(eq(outside.extra, 1u));
    };

    "candidate axes accept either absolute or relative tolerance"_test = [] {
        for (const int axis : {2, 3}) {
            std::array<float, 5> reference{0, 0, 10, 20, 0};
            auto candidate = reference;
            candidate[axis] += 3.f;
            expect(compare_one(reference, candidate).passed);
            candidate[axis] += 0.01f;
            expect(!compare_one(reference, candidate).passed);
            reference[axis] = 100.f;
            candidate = reference;
            candidate[axis] = 105.f;
            expect(compare_one(reference, candidate).passed);
            candidate[axis] = 105.01f;
            expect(!compare_one(reference, candidate).passed);
        }
    };

    "candidate angles wrap modulo pi and ignore nearly circular reference ellipses"_test = [] {
        const std::array<float, 5> reference{0, 0, 10, 20, 0};
        expect(compare_one(reference, {0, 0, 10, 20, 0.05f}).passed);
        expect(!compare_one(reference, {0, 0, 10, 20, 0.0501f}).passed);
        expect(compare_one(reference, {0, 0, 10, 20, std::numbers::pi_v<float>}).passed);
        expect(compare_one({0, 0, 10, 11, 0}, {0, 0, 10, 11, 1}).passed);
        expect(!compare_one({0, 0, 10, 11.01f, 0}, {0, 0, 10, 11.01f, 1}).passed);
    };

    "candidate matching rejects malformed rows and non-finite values"_test = [] {
        const CandidateFixture reference{{{0, 0, 10, 20, 0}, 0, 1}};
        auto malformed = reference.view();
        malformed.ellipses = malformed.ellipses.first(4);
        expect(throws<std::runtime_error>([&] {
            (void)compare_candidates(reference.view(), malformed);
        }));
        CandidateFixture nonfinite = reference;
        nonfinite.quality[0] = std::numeric_limits<float>::quiet_NaN();
        expect(throws<std::runtime_error>([&] {
            (void)compare_candidates(reference.view(), nonfinite.view());
        }));
        nonfinite = reference;
        nonfinite.ellipses[0] = std::numeric_limits<float>::infinity();
        expect(throws<std::runtime_error>([&] {
            (void)compare_candidates(nonfinite.view(), reference.view());
        }));
    };

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
        const std::string header =
            R"({"pyramid/level0/src":{"dtype":"U8","shape":[2,3],"data_offsets":[0,5]},"__metadata__":{}})";
        expect(throws<std::runtime_error>([&] {
            (void)ReferenceSnapshot::from_bytes(safetensors(header, std::vector<std::uint8_t>(5)));
        }));
        const std::string overrun =
            R"({"pyramid/level0/src":{"dtype":"U8","shape":[2,3],"data_offsets":[0,6]},"__metadata__":{}})";
        expect(throws<std::runtime_error>([&] {
            (void)ReferenceSnapshot::from_bytes(safetensors(overrun, std::vector<std::uint8_t>(5)));
        }));
    };

    "fills edge planes and edge point collections from reference snapshot bytes"_test = [] {
        const std::string header =
            R"({"pyramid/level0/src":{"dtype":"U8","shape":[1,2],"data_offsets":[0,2]},)"
            R"("gradient/level0/dx":{"dtype":"I16","shape":[1,2],"data_offsets":[2,6]},)"
            R"("gradient/level0/dy":{"dtype":"I16","shape":[1,2],"data_offsets":[6,10]},)"
            R"("edges/level0/edges":{"dtype":"U8","shape":[1,2],"data_offsets":[10,12]},)"
            R"("edge_points/level0/xy":{"dtype":"I32","shape":[1,2],"data_offsets":[12,20]},)"
            R"("edge_points/level0/gradients":{"dtype":"F32","shape":[1,2],"data_offsets":[20,28]},)"
            R"("__metadata__":{"schema_version":"1"}})";
        const std::vector<std::uint8_t> data = {3, 4, 0, 0, 0, 0, 0, 0, 0,   0,   0, 255, 1, 0,
                                                0, 0, 0, 0, 0, 0, 0, 0, 128, 191, 0, 0,   0, 64};
        const ReferenceSnapshot snapshot = ReferenceSnapshot::from_bytes(safetensors(header, data));
        cpu::Buffers buffers;
        buffers.ensure(2, 1);
        fill_level(snapshot, 0, Stage::edges, buffers);
        expect(eq(buffers.edges(0, 0), 0));
        expect(eq(buffers.edges(0, 1), 255));
        buffers.edge_map.setTo(7);
        fill_level(snapshot, 0, Stage::edge_points, buffers);
        expect(eq(buffers.n, 1u));
        expect(buffers.xy == std::vector<std::int32_t>{1, 0});
        // These differ from the gradient planes: the loader must copy, not recompute
        expect(buffers.gradients == std::vector<float>{-1.f, 2.f});
        expect(eq(buffers.edge_map(0, 0), -1));
        expect(eq(buffers.edge_map(0, 1), 0));
        expect(throws<std::runtime_error>([&] {
            (void)fill_level(snapshot, 0, Stage::vote, buffers);
        }));

        // Reject coordinates outside either axis before indexing the edge map
        for (const std::size_t offset : {12u, 16u}) {
            auto outside = data;
            outside[offset] = 2;
            const ReferenceSnapshot invalid =
                ReferenceSnapshot::from_bytes(safetensors(header, outside));
            expect(throws<std::runtime_error>([&] {
                fill_level(invalid, 0, Stage::edge_points, buffers);
            }));
        }
    };

    "fills vote and linking without changing their stored orders"_test = [] {
        const std::string header =
            R"({"pyramid/level0/src":{"dtype":"U8","shape":[1,2],"data_offsets":[0,2]},)"
            R"("gradient/level0/dx":{"dtype":"I16","shape":[1,2],"data_offsets":[2,6]},)"
            R"("gradient/level0/dy":{"dtype":"I16","shape":[1,2],"data_offsets":[6,10]},)"
            R"("edges/level0/edges":{"dtype":"U8","shape":[1,2],"data_offsets":[10,12]},)"
            R"("edge_points/level0/xy":{"dtype":"I32","shape":[2,2],"data_offsets":[12,28]},)"
            R"("edge_points/level0/gradients":{"dtype":"F32","shape":[2,2],"data_offsets":[28,44]},)"
            R"("vote/level0/links":{"dtype":"I32","shape":[2,2],"data_offsets":[44,60]},)"
            R"("vote/level0/voters/offsets":{"dtype":"I32","shape":[3],"data_offsets":[60,72]},)"
            R"("vote/level0/voters/values":{"dtype":"I32","shape":[2],"data_offsets":[72,80]},)"
            R"("vote/level0/is_max":{"dtype":"I32","shape":[2],"data_offsets":[80,88]},)"
            R"("vote/level0/flow_length":{"dtype":"F32","shape":[2],"data_offsets":[88,96]},)"
            R"("vote/level0/seeds":{"dtype":"I32","shape":[2],"data_offsets":[96,104]},)"
            R"("vote/level0/seed_order":{"dtype":"I32","shape":[2],"data_offsets":[104,112]},)"
            R"("linking/level0/seeds":{"dtype":"I32","shape":[1],"data_offsets":[112,116]},)"
            R"("linking/level0/segments/offsets":{"dtype":"I32","shape":[2],"data_offsets":[116,124]},)"
            R"("linking/level0/segments/values":{"dtype":"I32","shape":[2],"data_offsets":[124,132]},)"
            R"("linking/level0/child_counts":{"dtype":"I32","shape":[1],"data_offsets":[132,136]},)"
            R"("linking/level0/avg_vote":{"dtype":"F32","shape":[1],"data_offsets":[136,140]},)"
            R"("__metadata__":{"schema_version":"1"}})";
        std::vector<std::uint8_t> data = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255};
        // Hand-built little-endian words: coordinates, gradients, vote and linking tensors
        for (const std::uint32_t word :
             {0u,          0u, 1u, 0u, 0u, 0u, 0u, 0u, 0xffffffffu, 1u,          0u,
              0xffffffffu, 0u, 1u, 2u, 1u, 0u, 1u, 1u, 0x3fc00000u, 0x40200000u, 0u,
              1u,          1u, 0u, 1u, 0u, 2u, 1u, 0u, 2u,          0x40000000u}) {
            for (int byte = 0; byte < 4; ++byte) {
                data.push_back(static_cast<std::uint8_t>(word >> (8 * byte)));
            }
        }
        const ReferenceSnapshot snapshot = ReferenceSnapshot::from_bytes(safetensors(header, data));
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
        const LinkingHost linking = cpu::Backend::host_linking(buffers);
        expect(eq(linking.c, 1u));
        expect(std::ranges::equal(linking.seeds, std::array{1}));
        expect(std::ranges::equal(linking.segment_offsets, std::array{0, 2}));
        expect(std::ranges::equal(linking.segment_values, std::array{1, 0}));
        expect(std::ranges::equal(linking.child_counts, std::array{2}));
        expect(std::ranges::equal(linking.avg_vote, std::array{2.f}));
        expect(throws<std::logic_error>([&] {
            fill_level(snapshot, 0, Stage::candidates, buffers);
        }));
    };

    "fills an empty edge point collection and removes the previous edge map"_test = [] {
        const std::string header =
            R"({"pyramid/level0/src":{"dtype":"U8","shape":[1,1],"data_offsets":[0,1]},)"
            R"("gradient/level0/dx":{"dtype":"I16","shape":[1,1],"data_offsets":[1,3]},)"
            R"("gradient/level0/dy":{"dtype":"I16","shape":[1,1],"data_offsets":[3,5]},)"
            R"("edges/level0/edges":{"dtype":"U8","shape":[1,1],"data_offsets":[5,6]},)"
            R"("edge_points/level0/xy":{"dtype":"I32","shape":[0,2],"data_offsets":[6,6]},)"
            R"("edge_points/level0/gradients":{"dtype":"F32","shape":[0,2],"data_offsets":[6,6]},)"
            R"("__metadata__":{"schema_version":"1"}})";
        const ReferenceSnapshot snapshot =
            ReferenceSnapshot::from_bytes(safetensors(header, {0, 0, 0, 0, 0, 0}));
        cpu::Buffers buffers;
        buffers.ensure(1, 1);
        buffers.n = 1;
        buffers.xy = {0, 0};
        buffers.gradients = {1.f, -1.f};
        buffers.edge_map.setTo(0);
        fill_level(snapshot, 0, Stage::edge_points, buffers);
        expect(eq(buffers.n, 0u));
        expect(buffers.xy.empty());
        expect(buffers.gradients.empty());
        expect(eq(buffers.edge_map(0, 0), -1));
    };

    "fills stage buffers and reports exact plane mismatches"_test = [] {
        // Include `src` and `dx` for a 3x2 level, leaving out `dy` to check missing data
        const std::string header =
            R"({"pyramid/level0/src":{"dtype":"U8","shape":[2,3],"data_offsets":[0,6]},)"
            R"("gradient/level0/dx":{"dtype":"I16","shape":[2,3],"data_offsets":[6,18]},)"
            R"("__metadata__":{"x":"y"}})";
        const ReferenceSnapshot snapshot = ReferenceSnapshot::from_bytes(
            safetensors(header, {1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
        );
        const Tensor& src = snapshot.tensor(Stage::pyramid, 0, "src");

        cpu::Buffers buffers;
        buffers.ensure(3, 2);
        fill_level(snapshot, 0, Stage::pyramid, buffers);
        expect(eq(buffers.src(1, 1), 5));
        expect(compare_plane<std::uint8_t>(src, buffers.src_plane().as_const()).exact());

        buffers.src(1, 1) = 0;
        buffers.src(1, 2) = 0;
        const Mismatch mismatch = compare_plane<std::uint8_t>(src, buffers.src_plane().as_const());
        expect(eq(mismatch.count, 2u));
        expect(eq(mismatch.first, 4u)); // First difference at (y 1, x 1)

        cpu::Buffers wrong_size;
        wrong_size.ensure(2, 3);
        expect(throws<std::runtime_error>([&] {
            (void)fill_level(snapshot, 0, Stage::pyramid, wrong_size);
        }));
        // Both fills need the missing `dy` plane and must fail
        expect(throws<std::runtime_error>([&] {
            (void)fill_level(snapshot, 0, Stage::gradient, buffers);
        }));
        expect(throws<std::runtime_error>([&] {
            (void)fill_level(snapshot, 0, Stage::edges, buffers);
        }));
    };
};

} // namespace cctag::portable::test
#endif // CCTAG_TEST
