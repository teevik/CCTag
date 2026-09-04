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
#include <boost/json/src.hpp> // header-only Boost.JSON: the one translation unit that includes it

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace cctag::portable::test {

namespace {

Dtype parse_dtype(const std::string& name, const std::string& tensor) {
    if (name == "U8") return Dtype::u8;
    if (name == "I16") return Dtype::i16;
    if (name == "I32") return Dtype::i32;
    if (name == "F32") return Dtype::f32;
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
    for (int i = 0; i < static_cast<int>(std::size(kStageNames)); ++i)
        if (name == kStageNames[i]) return static_cast<Stage>(i);
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
    for (const std::uint64_t extent : shape)
        count *= static_cast<std::size_t>(extent);
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
    if (!stream) throw std::runtime_error("cannot open " + file.string());
    std::vector<std::uint8_t> bytes(
        (std::istreambuf_iterator<char>(stream)),
        std::istreambuf_iterator<char>()
    );
    return from_bytes(std::move(bytes));
}

ReferenceSnapshot ReferenceSnapshot::from_bytes(std::vector<std::uint8_t> bytes) {
    ReferenceSnapshot snapshot;
    snapshot.bytes_ = std::move(bytes);
    const std::vector<std::uint8_t>& data = snapshot.bytes_;
    if (data.size() < 8)
        throw std::runtime_error("safetensors: file shorter than its header length field");
    std::uint64_t header_length = 0;
    for (int i = 7; i >= 0; --i)
        header_length = (header_length << 8) | data[static_cast<std::size_t>(i)];
    if (header_length > data.size() - 8)
        throw std::runtime_error("safetensors: header length exceeds the file");
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
    if (!document.is_object())
        throw std::runtime_error("safetensors: the JSON header is not an object");
    const auto integer = [](const boost::json::value& value, const std::string& what) {
        if (!(value.is_uint64() || (value.is_int64() && value.get_int64() >= 0)))
            throw std::runtime_error(what + ": expected a non-negative JSON integer");
        return value.to_number<std::uint64_t>();
    };
    const auto string = [](const boost::json::value& value, const std::string& what) {
        if (!value.is_string()) throw std::runtime_error(what + ": expected a JSON string");
        return std::string(value.get_string());
    };
    for (const auto& [name, entry] : document.get_object()) {
        if (name == "__metadata__") {
            if (!entry.is_object())
                throw std::runtime_error("__metadata__: expected a JSON object");
            for (const auto& [key, value] : entry.get_object())
                snapshot.metadata_.emplace(key, string(value, "__metadata__." + std::string(key)));
            continue;
        }
        const std::string tensor_name(name);
        if (!entry.is_object()) throw std::runtime_error(tensor_name + ": expected a JSON object");
        const boost::json::object& fields = entry.get_object();
        const auto field = [&](const char* key) -> const boost::json::value& {
            const boost::json::value* found = fields.if_contains(key);
            if (found == nullptr) throw std::runtime_error(tensor_name + ": missing " + key);
            return *found;
        };
        Tensor tensor;
        tensor.name = tensor_name;
        tensor.dtype = parse_dtype(string(field("dtype"), tensor_name + ".dtype"), tensor_name);
        const boost::json::value& shape = field("shape");
        if (!shape.is_array())
            throw std::runtime_error(tensor_name + ".shape: expected a JSON array");
        for (const boost::json::value& extent : shape.get_array())
            tensor.shape.push_back(integer(extent, tensor_name + ".shape"));
        const boost::json::value& offsets = field("data_offsets");
        if (!offsets.is_array() || offsets.get_array().size() != 2)
            throw std::runtime_error(tensor_name + ": data_offsets must have two entries");
        const std::uint64_t begin = integer(offsets.get_array()[0], tensor_name + ".data_offsets");
        const std::uint64_t end = integer(offsets.get_array()[1], tensor_name + ".data_offsets");
        if (begin > end || end > payload.size())
            throw std::runtime_error(tensor_name + ": data_offsets outside the data section");
        if (end - begin != tensor.elements() * dtype_size(tensor.dtype))
            throw std::runtime_error(tensor_name + ": data_offsets disagree with dtype and shape");
        tensor.bytes =
            payload.subspan(static_cast<std::size_t>(begin), static_cast<std::size_t>(end - begin));
        snapshot.tensors_.emplace(tensor_name, std::move(tensor));
    }
    if (snapshot.metadata_.empty())
        throw std::runtime_error("safetensors: no __metadata__ (not a stage snapshot)");
    return snapshot;
}

const std::string& ReferenceSnapshot::meta(const std::string& key) const {
    const auto found = metadata_.find(key);
    if (found == metadata_.end()) throw std::runtime_error("__metadata__." + key + " is absent");
    return found->second;
}

std::uint32_t ReferenceSnapshot::meta_u32(const std::string& key) const {
    const std::string& text = meta(key);
    std::uint32_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size())
        throw std::runtime_error("__metadata__." + key + " is not an integer: " + text);
    return value;
}

std::vector<Stage> ReferenceSnapshot::stages() const {
    std::vector<Stage> result;
    std::stringstream list(meta("stages"));
    std::string item;
    while (std::getline(list, item, ',')) {
        const std::optional<Stage> stage = parse_stage(item);
        if (!stage) throw std::runtime_error("__metadata__.stages names an unknown stage: " + item);
        result.push_back(*stage);
    }
    return result;
}

bool ReferenceSnapshot::has(Stage stage) const {
    const std::vector<Stage> listed = stages();
    return std::find(listed.begin(), listed.end(), stage) != listed.end();
}

const Tensor& ReferenceSnapshot::tensor(const std::string& name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end())
        throw std::runtime_error("tensor " + name + " is absent from the snapshot");
    return found->second;
}

const Tensor&
ReferenceSnapshot::tensor(Stage stage, std::uint32_t level, const std::string& name) const {
    return tensor(std::string(stage_name(stage)) + "/level" + std::to_string(level) + "/" + name);
}

std::optional<std::filesystem::path> reference_snapshots_dir() {
    const char* value = std::getenv("CCTAG_REFERENCE_SNAPSHOTS");
    if (value == nullptr || *value == '\0') return std::nullopt;
    return std::filesystem::path(value);
}

std::vector<std::filesystem::path> reference_snapshot_files() {
    std::vector<std::filesystem::path> files;
    const std::optional<std::filesystem::path> dir = reference_snapshots_dir();
    if (!dir) return files;
    for (const auto& entry : std::filesystem::directory_iterator(*dir))
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors")
            files.push_back(entry.path());
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
    if (upto == Stage::pyramid) return;
    copy_plane<std::int16_t>(snapshot.tensor(Stage::gradient, level, "dx"), buffers.dx_plane());
    copy_plane<std::int16_t>(snapshot.tensor(Stage::gradient, level, "dy"), buffers.dy_plane());
    if (upto == Stage::gradient) return;
    throw std::logic_error(
        std::string("fill_level: the stage buffers stop at gradient; add the fill for ")
        + stage_name(upto) + " when the build adds its buffers"
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
    for (std::uint32_t level = 0; level < context.levels.size(); ++level)
        fill_level(snapshot, level, upto, context.levels[level]);
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
