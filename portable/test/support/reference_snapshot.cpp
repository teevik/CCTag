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
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <sstream>

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
