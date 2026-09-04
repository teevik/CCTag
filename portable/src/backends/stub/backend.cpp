/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "backends/stub/backend.hpp"

#include "host/backend.hpp"
#include "host/parallel.hpp"
#include "kernels/gradient.hpp"

#include <cstring>

namespace cctag::portable::stub {

static_assert(ExecutionBackend<Backend>);

namespace {

constexpr std::size_t kPitchAlignment = 64;

/// Copies a staging plane into a pitched device plane.
template <class T>
void to_device(std::vector<T>& device, std::size_t pitch, kernels::Plane<const T> staging) {
    for (std::uint32_t y = 0; y < staging.height; ++y)
        std::memcpy(device.data() + y * pitch, staging.row(y), staging.width * sizeof(T));
}

/// Fills a staging plane from a pitched device plane.
template <class T>
void to_staging(const std::vector<T>& device, std::size_t pitch, kernels::Plane<T> staging) {
    for (std::uint32_t y = 0; y < staging.height; ++y)
        std::memcpy(staging.row(y), device.data() + y * pitch, staging.width * sizeof(T));
}

void src_to_device(Buffers& level) {
    to_device<std::uint8_t>(level.device_src, level.pitch, level.staging.src_plane().as_const());
}

void src_to_staging(const Buffers& level) {
    to_staging<std::uint8_t>(level.device_src, level.pitch, level.staging.src_plane());
}

} // namespace

void Buffers::ensure(std::uint32_t level_width, std::uint32_t level_height) {
    width = level_width;
    height = level_height;
    pitch =
        (static_cast<std::size_t>(width) + kPitchAlignment - 1) / kPitchAlignment * kPitchAlignment;
    device_src.resize(pitch * height);
    device_dx.resize(pitch * height);
    device_dy.resize(pitch * height);
    staging.ensure(level_width, level_height);
}

void Backend::load(Buffers& level0, kernels::Plane<const std::uint8_t> input) {
    cpu::Backend::load(level0.staging, input); // delegated
    src_to_device(level0);
}

void Backend::pyramid(Buffers& coarser, const Buffers& finer) {
    src_to_staging(finer);
    cpu::Backend::pyramid(coarser.staging, finer.staging); // delegated
    src_to_device(coarser);
}

void Backend::gradient(Buffers& level) {
    // Specialised: column-major, parallel over columns; a different loop, the same element
    // function, writing pitched device planes.
    const std::uint8_t* source = level.device_src.data();
    const std::size_t pitch = level.pitch;
    const int columns = static_cast<int>(level.width);
    const std::uint32_t rows = level.height;
    CCTAG_PORTABLE_PARALLEL_FOR_STATIC
    for (int x = 0; x < columns; ++x) {
        for (std::uint32_t y = 0; y < rows; ++y) {
            const std::size_t at = y * pitch + x;
            level.device_dx[at] =
                kernels::gradient_at(source, pitch, level.width, rows, x, y, kernels::kDxTaps);
            level.device_dy[at] =
                kernels::gradient_at(source, pitch, level.width, rows, x, y, kernels::kDyTaps);
        }
    }
}

PyramidHost Backend::host_pyramid(Buffers& level) {
    src_to_staging(level);
    return cpu::Backend::host_pyramid(level.staging);
}

GradientHost Backend::host_gradient(Buffers& level) {
    to_staging<std::int16_t>(level.device_dx, level.pitch, level.staging.dx_plane());
    to_staging<std::int16_t>(level.device_dy, level.pitch, level.staging.dy_plane());
    return cpu::Backend::host_gradient(level.staging);
}

} // namespace cctag::portable::stub
