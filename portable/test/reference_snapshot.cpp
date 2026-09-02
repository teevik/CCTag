/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "reference_snapshot.hpp"

#include <cctag/Params.hpp>

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <variant>

namespace cctag::portable::test {

namespace {

// --- A minimal JSON reader, enough for a safetensors header: objects, arrays, strings,
// --- integers and doubles, `true`/`false`/`null`.

struct Json;
using JsonObject = std::map<std::string, Json>;
using JsonArray = std::vector<Json>;

struct Json {
    std::variant<std::nullptr_t, bool, std::uint64_t, double, std::string, JsonArray, JsonObject>
        value;

    const JsonObject& object(const std::string& what) const {
        if (const auto* object = std::get_if<JsonObject>(&value)) {
            return *object;
        }
        throw std::runtime_error(what + ": expected a JSON object");
    }
    const JsonArray& array(const std::string& what) const {
        if (const auto* array = std::get_if<JsonArray>(&value)) {
            return *array;
        }
        throw std::runtime_error(what + ": expected a JSON array");
    }
    const std::string& string(const std::string& what) const {
        if (const auto* string = std::get_if<std::string>(&value)) {
            return *string;
        }
        throw std::runtime_error(what + ": expected a JSON string");
    }
    std::uint64_t integer(const std::string& what) const {
        if (const auto* integer = std::get_if<std::uint64_t>(&value)) {
            return *integer;
        }
        throw std::runtime_error(what + ": expected a non-negative JSON integer");
    }
};

class JsonReader {
  public:
    explicit JsonReader(std::string_view text) :
        text_(text) {}

    Json document() {
        Json result = value();
        skip_whitespace();
        if (at_ != text_.size()) {
            fail("trailing characters after the JSON document");
        }
        return result;
    }

  private:
    std::string_view text_;
    std::size_t at_ = 0;

    [[noreturn]] void fail(const std::string& what) const {
        throw std::runtime_error("JSON header at byte " + std::to_string(at_) + ": " + what);
    }

    void skip_whitespace() {
        while (at_ < text_.size()
               && (text_[at_] == ' ' || text_[at_] == '\n' || text_[at_] == '\r'
                   || text_[at_] == '\t')) {
            ++at_;
        }
    }

    char peek() {
        skip_whitespace();
        if (at_ >= text_.size()) {
            fail("unexpected end of input");
        }
        return text_[at_];
    }

    void expect(char c) {
        if (peek() != c) {
            fail(std::string("expected '") + c + "'");
        }
        ++at_;
    }

    bool consume(std::string_view literal) {
        if (text_.substr(at_, literal.size()) == literal) {
            at_ += literal.size();
            return true;
        }
        return false;
    }

    Json value() {
        switch (peek()) {
            case '{':
                return Json{object()};
            case '[':
                return Json{array()};
            case '"':
                return Json{string()};
            case 't':
                if (consume("true")) {
                    return Json{true};
                }
                fail("bad literal");
            case 'f':
                if (consume("false")) {
                    return Json{false};
                }
                fail("bad literal");
            case 'n':
                if (consume("null")) {
                    return Json{nullptr};
                }
                fail("bad literal");
            default:
                return number();
        }
    }

    JsonObject object() {
        JsonObject result;
        expect('{');
        if (peek() == '}') {
            ++at_;
            return result;
        }
        for (;;) {
            std::string key = string();
            expect(':');
            result.emplace(std::move(key), value());
            if (peek() == ',') {
                ++at_;
                continue;
            }
            expect('}');
            return result;
        }
    }

    JsonArray array() {
        JsonArray result;
        expect('[');
        if (peek() == ']') {
            ++at_;
            return result;
        }
        for (;;) {
            result.push_back(value());
            if (peek() == ',') {
                ++at_;
                continue;
            }
            expect(']');
            return result;
        }
    }

    std::uint32_t hex4() {
        if (at_ + 4 > text_.size()) {
            fail("truncated \\u escape");
        }
        std::uint32_t code = 0;
        const auto [end, error] =
            std::from_chars(text_.data() + at_, text_.data() + at_ + 4, code, 16);
        if (error != std::errc{} || end != text_.data() + at_ + 4) {
            fail("bad \\u escape");
        }
        at_ += 4;
        return code;
    }

    static void append_utf8(std::string& out, std::uint32_t code) {
        if (code < 0x80) {
            out.push_back(static_cast<char>(code));
        } else if (code < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else if (code < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (code >> 18)));
            out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        }
    }

    std::string string() {
        expect('"');
        std::string result;
        for (;;) {
            if (at_ >= text_.size()) {
                fail("unterminated string");
            }
            const char c = text_[at_++];
            if (c == '"') {
                return result;
            }
            if (c != '\\') {
                result.push_back(c);
                continue;
            }
            if (at_ >= text_.size()) {
                fail("unterminated escape");
            }
            const char escape = text_[at_++];
            switch (escape) {
                case '"':
                case '\\':
                case '/':
                    result.push_back(escape);
                    break;
                case 'b':
                    result.push_back('\b');
                    break;
                case 'f':
                    result.push_back('\f');
                    break;
                case 'n':
                    result.push_back('\n');
                    break;
                case 'r':
                    result.push_back('\r');
                    break;
                case 't':
                    result.push_back('\t');
                    break;
                case 'u': {
                    std::uint32_t code = hex4();
                    if (code >= 0xD800 && code < 0xDC00 && consume("\\u")) {
                        const std::uint32_t low = hex4();
                        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                    }
                    append_utf8(result, code);
                    break;
                }
                default:
                    fail("bad escape");
            }
        }
    }

    Json number() {
        const std::size_t start = at_;
        bool integral = true;
        while (at_ < text_.size()) {
            const char c = text_[at_];
            if (c == '.' || c == 'e' || c == 'E') {
                integral = false;
            } else if (!(c == '-' || c == '+' || (c >= '0' && c <= '9'))) {
                break;
            }
            ++at_;
        }
        const std::string_view token = text_.substr(start, at_ - start);
        if (token.empty()) {
            fail("expected a value");
        }
        if (integral && token[0] != '-') {
            std::uint64_t integer = 0;
            const auto [end, error] =
                std::from_chars(token.data(), token.data() + token.size(), integer);
            if (error == std::errc{} && end == token.data() + token.size()) {
                return Json{integer};
            }
        }
        double real = 0;
        const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), real);
        if (error != std::errc{} || end != token.data() + token.size()) {
            fail("bad number");
        }
        return Json{real};
    }
};

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
    snapshot.bytes_ = std::move(bytes);
    const std::vector<std::uint8_t>& data = snapshot.bytes_;
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

    const Json document = JsonReader(header).document();
    for (const auto& [name, entry] : document.object("header")) {
        if (name == "__metadata__") {
            for (const auto& [key, value] : entry.object("__metadata__")) {
                snapshot.metadata_.emplace(key, value.string("__metadata__." + key));
            }
            continue;
        }
        const JsonObject& fields = entry.object(name);
        const auto field = [&](const char* key) -> const Json& {
            const auto found = fields.find(key);
            if (found == fields.end()) {
                throw std::runtime_error(name + ": missing " + key);
            }
            return found->second;
        };
        Tensor tensor;
        tensor.name = name;
        tensor.dtype = parse_dtype(field("dtype").string(name + ".dtype"), name);
        for (const Json& extent : field("shape").array(name + ".shape")) {
            tensor.shape.push_back(extent.integer(name + ".shape"));
        }
        const JsonArray& offsets = field("data_offsets").array(name + ".data_offsets");
        if (offsets.size() != 2) {
            throw std::runtime_error(name + ": data_offsets must have two entries");
        }
        const std::uint64_t begin = offsets[0].integer(name + ".data_offsets");
        const std::uint64_t end = offsets[1].integer(name + ".data_offsets");
        if (begin > end || end > payload.size()) {
            throw std::runtime_error(name + ": data_offsets outside the data section");
        }
        if (end - begin != tensor.elements() * dtype_size(tensor.dtype)) {
            throw std::runtime_error(name + ": data_offsets disagree with dtype and shape");
        }
        tensor.bytes =
            payload.subspan(static_cast<std::size_t>(begin), static_cast<std::size_t>(end - begin));
        snapshot.tensors_.emplace(name, std::move(tensor));
    }
    if (snapshot.metadata_.empty()) {
        throw std::runtime_error("safetensors: no __metadata__ (not a stage snapshot)");
    }
    return snapshot;
}

const std::string& ReferenceSnapshot::meta(const std::string& key) const {
    const auto found = metadata_.find(key);
    if (found == metadata_.end()) {
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
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) {
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
