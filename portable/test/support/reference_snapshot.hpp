/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_TEST_REFERENCE_SNAPSHOT_HPP
#define CCTAG_PORTABLE_TEST_REFERENCE_SNAPSHOT_HPP

// Reads reference snapshots, fills CPU stage buffers and compares stage outputs exactly

#include "backends/cpu/backend.hpp"
#include "host/context.hpp"
#include "kernels/plane.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace cctag::portable::test {

/// Pipeline stages in execution order, also used as prefixes in tensor names
enum class Stage {
    pyramid,
    gradient,
    edges,
    edge_points,
    vote,
    linking,
    candidates,
    markers,
};

const char* stage_name(Stage stage);
/// Parses a stage name, returning no value if it is unknown
std::optional<Stage> parse_stage(const std::string& name);

/// Element types supported by the stage snapshot format
enum class Dtype {
    u8,
    i16,
    i32,
    f32,
};

const char* dtype_name(Dtype dtype);
std::size_t dtype_size(Dtype dtype);

template <class T>
constexpr Dtype dtype_of();
template <>
constexpr Dtype dtype_of<std::uint8_t>() {
    return Dtype::u8;
}
template <>
constexpr Dtype dtype_of<std::int16_t>() {
    return Dtype::i16;
}
template <>
constexpr Dtype dtype_of<std::int32_t>() {
    return Dtype::i32;
}
template <>
constexpr Dtype dtype_of<float>() {
    return Dtype::f32;
}

/// One stage snapshot tensor, with a view of its bytes in the snapshot's storage
struct Tensor {
    std::string name;
    Dtype dtype = Dtype::u8;
    std::vector<std::uint64_t> shape;
    std::span<const std::uint8_t> bytes;

    std::size_t elements() const;

    /// Copies values into aligned storage since tensor bytes may be unaligned
    /// Throws if `T` does not match the tensor's element type
    template <class T>
    std::vector<T> as() const {
        expect_dtype(dtype_of<T>());
        std::vector<T> values(elements());
        if (!values.empty()) {
            std::memcpy(values.data(), bytes.data(), bytes.size());
        }
        return values;
    }

  private:
    void expect_dtype(Dtype expected) const;
};

/// Holds a reference snapshot's file bytes, metadata and tensors indexed by full name
class ReferenceSnapshot {
  public:
    static ReferenceSnapshot read(const std::filesystem::path& file);
    /// Parses safetensors bytes: an 8-byte header length, a JSON header and tensor data
    static ReferenceSnapshot from_bytes(std::vector<std::uint8_t> bytes);

    ReferenceSnapshot(ReferenceSnapshot&&) = default;
    ReferenceSnapshot& operator=(ReferenceSnapshot&&) = default;
    ReferenceSnapshot(const ReferenceSnapshot&) = delete;
    ReferenceSnapshot& operator=(const ReferenceSnapshot&) = delete;

    const std::map<std::string, std::string>& metadata() const {
        return metadata_entries;
    }
    /// Returns a metadata field, throwing if it is missing
    const std::string& meta(const std::string& key) const;
    std::uint32_t meta_u32(const std::string& key) const;

    std::string problem() const {
        return meta("problem");
    }
    std::uint32_t image_width() const {
        return meta_u32("image_width");
    }
    std::uint32_t image_height() const {
        return meta_u32("image_height");
    }
    std::uint32_t processed_levels() const {
        return meta_u32("processed_levels");
    }
    std::uint32_t crowns() const {
        return meta_u32("crowns");
    }
    /// Returns the stages listed in `__metadata__.stages`, preserving their order
    std::vector<Stage> stages() const;
    bool has(Stage stage) const;

    const std::map<std::string, Tensor>& tensors() const {
        return tensor_entries;
    }
    bool has_tensor(const std::string& name) const {
        return tensor_entries.count(name) != 0;
    }
    /// Returns a tensor by full name, throwing if it is missing
    const Tensor& tensor(const std::string& name) const;
    /// Returns the tensor named `<stage>/level<level>/<name>`, throwing if it is missing
    const Tensor& tensor(Stage stage, std::uint32_t level, const std::string& name) const;

  private:
    ReferenceSnapshot() = default;

    std::vector<std::uint8_t> bytes;
    std::map<std::string, std::string> metadata_entries;
    std::map<std::string, Tensor> tensor_entries;
};

/// Returns the reference snapshot directory from `CCTAG_REFERENCE_SNAPSHOTS`
/// Returns no value if the variable is unset or empty
std::optional<std::filesystem::path> reference_snapshots_dir();
/// Lists `.safetensors` files in the reference snapshot directory, sorted by name
/// Returns an empty list if `CCTAG_REFERENCE_SNAPSHOTS` is unset or empty
std::vector<std::filesystem::path> reference_snapshot_files();

/// Copies a `[height, width]` tensor into a plane, respecting its row stride
/// Throws if the element type or dimensions differ
template <class T>
void copy_plane(const Tensor& reference, kernels::Plane<T> plane) {
    if (reference.shape.size() != 2 || reference.shape[0] != plane.height
        || reference.shape[1] != plane.width) {
        throw std::runtime_error(
            reference.name + ": shape does not match the plane's " + std::to_string(plane.height)
            + "x" + std::to_string(plane.width)
        );
    }
    const std::vector<T> values = reference.template as<T>();
    for (std::uint32_t y = 0; y < plane.height; ++y) {
        std::memcpy(
            plane.row(y),
            values.data() + static_cast<std::size_t>(y) * plane.width,
            static_cast<std::size_t>(plane.width) * sizeof(T)
        );
    }
}

/// Fills one level's stage buffers with reference outputs through `upto`, inclusive
/// Requires buffers sized by `Buffers::ensure` and supports stages through `edge_points`
void fill_level(
    const ReferenceSnapshot& snapshot,
    std::uint32_t level,
    Stage upto,
    cpu::Buffers& buffers
);

/// Sizes the context using the reference snapshot's image dimensions and crown count
/// Fills each level through `upto`, inclusive, checking the level count and plane dimensions
void fill_context(const ReferenceSnapshot& snapshot, Stage upto, Context<cpu::Backend>& context);

/// Counts differing elements in an exact comparison
struct Mismatch {
    /// Number of elements compared
    std::size_t total = 0;
    /// Number of elements whose bytes differ
    std::size_t count = 0;
    /// Flat index of the first difference, valid when `count` is nonzero
    std::size_t first = 0;

    bool exact() const {
        return count == 0;
    }
};

/// Compares a host plane with a `[height, width]` tensor, checking each element's bytes
template <class T>
Mismatch compare_plane(const Tensor& reference, kernels::Plane<const T> plane) {
    if (reference.shape.size() != 2 || reference.shape[0] != plane.height
        || reference.shape[1] != plane.width) {
        throw std::runtime_error(
            reference.name + ": shape does not match the plane's " + std::to_string(plane.height)
            + "x" + std::to_string(plane.width)
        );
    }
    const std::vector<T> expected = reference.template as<T>();
    Mismatch mismatch;
    mismatch.total = expected.size();
    for (std::uint32_t y = 0; y < plane.height; ++y) {
        const T* row = plane.row(y);
        for (std::uint32_t x = 0; x < plane.width; ++x) {
            const std::size_t index = static_cast<std::size_t>(y) * plane.width + x;
            if (std::memcmp(&row[x], &expected[index], sizeof(T)) != 0) {
                if (mismatch.count == 0) {
                    mismatch.first = index;
                }
                ++mismatch.count;
            }
        }
    }
    return mismatch;
}

/// Compares contiguous values with a tensor of the same element count, checking their bytes
template <class T>
Mismatch compare_values(const Tensor& reference, std::span<const T> values) {
    if (reference.elements() != values.size()) {
        throw std::runtime_error(
            reference.name + ": " + std::to_string(reference.elements()) + " elements, got "
            + std::to_string(values.size())
        );
    }
    const std::vector<T> expected = reference.template as<T>();
    Mismatch mismatch;
    mismatch.total = expected.size();
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (std::memcmp(&values[index], &expected[index], sizeof(T)) != 0) {
            if (mismatch.count == 0) {
                mismatch.first = index;
            }
            ++mismatch.count;
        }
    }
    return mismatch;
}

/// Describes the mismatch count and first differing index, adding `(y, x)` for a plane
std::string describe(const Tensor& reference, const Mismatch& mismatch);

} // namespace cctag::portable::test

#endif
