/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_TEST_REFERENCE_SNAPSHOT_HPP
#define CCTAG_PORTABLE_TEST_REFERENCE_SNAPSHOT_HPP

// Test support for stage isolation: reads a reference snapshot (the safetensors file the legacy
// pipeline writes with every stage's outputs for one problem) and fills the CPU backend's stage
// buffers with the reference snapshot's outputs, so one stage function can run on its reference
// input and be compared element-exact against its reference output. A whole-run
// comparison cannot tell a wrong stage from a stage fed something subtly different upstream; this
// can.
//
// The header is parsed with Boost.JSON (header-only); the data section is viewed in place.

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

/// The pipeline stages, in pipeline order; tensor names start with the stage name.
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
/// Inverse of `stage_name`; empty for an unknown name.
std::optional<Stage> parse_stage(const std::string& name);

/// The four element types a stage snapshot may contain.
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

/// One tensor of a snapshot.
struct Tensor {
    std::string name;
    Dtype dtype = Dtype::u8;
    std::vector<std::uint64_t> shape;
    std::span<const std::uint8_t> bytes;

    std::size_t elements() const;

    /// The values as `T`, copied out: the file gives no alignment guarantee, so the bytes are
    /// never reinterpreted in place. Throws when `T` is not the tensor's dtype.
    template <class T>
    std::vector<T> as() const {
        expect_dtype(dtype_of<T>());
        std::vector<T> values(elements());
        std::memcpy(values.data(), bytes.data(), bytes.size());
        return values;
    }

  private:
    void expect_dtype(Dtype expected) const;
};

/// A parsed stage snapshot: `__metadata__` plus every tensor by its full name.
class ReferenceSnapshot {
  public:
    static ReferenceSnapshot read(const std::filesystem::path& file);
    /// Parses an in-memory safetensors file: `u64` header length, JSON header, data section.
    static ReferenceSnapshot from_bytes(std::vector<std::uint8_t> bytes);

    ReferenceSnapshot(ReferenceSnapshot&&) = default;
    ReferenceSnapshot& operator=(ReferenceSnapshot&&) = default;
    ReferenceSnapshot(const ReferenceSnapshot&) = delete;
    ReferenceSnapshot& operator=(const ReferenceSnapshot&) = delete;

    const std::map<std::string, std::string>& metadata() const {
        return metadata_;
    }
    /// A `__metadata__` field; throws when absent.
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
    /// `__metadata__.stages`, in file order.
    std::vector<Stage> stages() const;
    bool has(Stage stage) const;

    const std::map<std::string, Tensor>& tensors() const {
        return tensors_;
    }
    bool has_tensor(const std::string& name) const {
        return tensors_.count(name) != 0;
    }
    /// Throws when the tensor is absent.
    const Tensor& tensor(const std::string& name) const;
    /// `<stage>/level<level>/<name>`.
    const Tensor& tensor(Stage stage, std::uint32_t level, const std::string& name) const;

  private:
    ReferenceSnapshot() = default;

    std::vector<std::uint8_t> bytes_;
    std::map<std::string, std::string> metadata_;
    std::map<std::string, Tensor> tensors_;
};

/// `$CCTAG_REFERENCE_SNAPSHOTS` when set and non-empty: the directory of reference snapshots,
/// one `<problem>.safetensors` per test image.
std::optional<std::filesystem::path> reference_snapshots_dir();
/// Every `*.safetensors` in the store, sorted by name. Empty when the store is unset.
std::vector<std::filesystem::path> reference_snapshot_files();

/// Copies a `[height, width]` tensor into a plane of the same element type and dimensions. Throws
/// on a dtype or shape mismatch.
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

/// Fills one level's stage buffers (already sized by `Buffers::ensure`) with the reference's
/// outputs of every stage up to and including `upto`. The buffers stop at `gradient` so far; each
/// new stage extends this with one more fill.
void fill_level(
    const ReferenceSnapshot& snapshot,
    std::uint32_t level,
    Stage upto,
    cpu::Buffers& buffers
);

/// Sizes `context` for the snapshot's image and parameters (`Parameters(crowns)`), checks every
/// level's dimensions against the reference's, and fills every level up to and including `upto`.
void fill_context(const ReferenceSnapshot& snapshot, Stage upto, Context<cpu::Backend>& context);

/// The result of an element-exact comparison.
struct Mismatch {
    std::size_t total = 0;
    std::size_t count = 0;
    std::size_t first = 0;

    bool exact() const {
        return count == 0;
    }
};

/// Element-exact comparison of a host plane against a `[height, width]` tensor.
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

/// Element-exact comparison of contiguous values against a tensor of the same element count.
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

/// One line for a test message: the tensor, the mismatch count and where the first one is
/// (as `(y, x)` when the tensor is a plane).
std::string describe(const Tensor& reference, const Mismatch& mismatch);

} // namespace cctag::portable::test

#endif
