/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "backends/stub/backend.hpp"

#include "host/parallel.hpp"
#include "kernels/gradient.hpp"

#include <cstring>

namespace cctag::portable::stub {

namespace {

constexpr std::size_t kPitchAlignment = 64;

template <class T>
void upload(std::vector<T>& device, std::size_t pitch, const kernels::Plane<const T>& host) {
    for (std::uint32_t y = 0; y < host.height; ++y) {
        std::memcpy(device.data() + y * pitch, host.row(y), host.width * sizeof(T));
    }
}

template <class T>
void download(const std::vector<T>& device, std::size_t pitch, kernels::Plane<T> host) {
    for (std::uint32_t y = 0; y < host.height; ++y) {
        std::memcpy(host.row(y), device.data() + y * pitch, host.width * sizeof(T));
    }
}

void upload_src(Buffers& level) {
    const cpu::Buffers& staging = level.staging;
    upload<std::uint8_t>(
        level.device_src, level.pitch, {staging.src.data(), staging.width, staging.height, staging.width}
    );
}

void download_src(const Buffers& level) {
    download<std::uint8_t>(level.device_src, level.pitch, level.staging.src_plane());
}

} // namespace

void Buffers::ensure(
    std::uint32_t level_width,
    std::uint32_t level_height,
    std::uint32_t from_width,
    std::uint32_t from_height
) {
    width = level_width;
    height = level_height;
    pitch = (static_cast<std::size_t>(width) + kPitchAlignment - 1) / kPitchAlignment * kPitchAlignment;
    device_src.resize(pitch * height);
    device_dx.resize(pitch * height);
    device_dy.resize(pitch * height);
    // The staging copy carries the baseline's own glue (resize tables) for delegated stages.
    staging.ensure(level_width, level_height, from_width, from_height);
}

void Backend::load(Buffers& level0, kernels::Plane<const std::uint8_t> input) {
    cpu::Backend::load(level0.staging, input); // delegated
    upload_src(level0);
}

void Backend::pyramid(Buffers& coarser, const Buffers& finer) {
    download_src(finer);
    cpu::Backend::pyramid(coarser.staging, finer.staging); // delegated
    upload_src(coarser);
}

void Backend::gradient(Buffers& level) {
    // Specialised: column-major, parallel over columns — a different loop, the same element
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

HostViews Backend::host(const Buffers& level) {
    download_src(level);
    download<std::int16_t>(level.device_dx, level.pitch, level.staging.dx_plane());
    download<std::int16_t>(level.device_dy, level.pitch, level.staging.dy_plane());
    return cpu::Backend::host(level.staging);
}

} // namespace cctag::portable::stub
