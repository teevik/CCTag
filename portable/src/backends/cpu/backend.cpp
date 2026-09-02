/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "backends/cpu/backend.hpp"

#include "host/parallel.hpp"
#include "kernels/gradient.hpp"
#include "kernels/resize.hpp"

#include <cstring>

namespace cctag::portable::cpu {

void Buffers::ensure(
    std::uint32_t level_width,
    std::uint32_t level_height,
    std::uint32_t from_width,
    std::uint32_t from_height
) {
    width = level_width;
    height = level_height;
    const std::size_t area = static_cast<std::size_t>(width) * height;
    src.resize(area);
    dx.resize(area);
    dy.resize(area);

    finer_height = from_height;
    exact_halving = from_width == 2 * width && from_height == 2 * height;
    if (exact_halving || from_width == 0) {
        xofs.clear();
        ialpha.clear();
        yofs.clear();
        ibeta.clear();
        return;
    }
    xofs.resize(width);
    ialpha.resize(2 * static_cast<std::size_t>(width));
    yofs.resize(height);
    ibeta.resize(2 * static_cast<std::size_t>(height));
    xmax = kernels::linear_axis_tables(from_width, width, xofs.data(), ialpha.data());
    ymax = kernels::linear_axis_tables(from_height, height, yofs.data(), ibeta.data());
}

void Backend::load(Buffers& level0, kernels::Plane<const std::uint8_t> input) {
    const int rows = static_cast<int>(level0.height);
    CCTAG_PORTABLE_PARALLEL_FOR_STATIC
    for (int y = 0; y < rows; ++y) {
        std::memcpy(level0.src_plane().row(y), input.row(y), level0.width);
    }
}

void Backend::pyramid(Buffers& coarser, const Buffers& finer) {
    const std::uint8_t* source = finer.src.data();
    const std::size_t stride = finer.width;
    const int rows = static_cast<int>(coarser.height);
    const std::uint32_t columns = coarser.width;
    if (coarser.exact_halving) {
        CCTAG_PORTABLE_PARALLEL_FOR_STATIC
        for (int y = 0; y < rows; ++y) {
            std::uint8_t* out = coarser.src_plane().row(y);
            for (std::uint32_t x = 0; x < columns; ++x) {
                out[x] = kernels::halve_at(source, stride, x, y);
            }
        }
        return;
    }
    const kernels::LinearAxis xs{coarser.xofs.data(), coarser.ialpha.data(), coarser.xmax};
    const kernels::LinearAxis ys{coarser.yofs.data(), coarser.ibeta.data(), coarser.ymax};
    CCTAG_PORTABLE_PARALLEL_FOR_STATIC
    for (int y = 0; y < rows; ++y) {
        std::uint8_t* out = coarser.src_plane().row(y);
        for (std::uint32_t x = 0; x < columns; ++x) {
            out[x] = kernels::bilinear_at(source, stride, finer.height, x, y, xs, ys);
        }
    }
}

void Backend::gradient(Buffers& level) {
    const std::uint8_t* source = level.src.data();
    const std::size_t stride = level.width;
    const int rows = static_cast<int>(level.height);
    const std::uint32_t columns = level.width;
    CCTAG_PORTABLE_PARALLEL_FOR_STATIC
    for (int y = 0; y < rows; ++y) {
        std::int16_t* dx = level.dx_plane().row(y);
        std::int16_t* dy = level.dy_plane().row(y);
        for (std::uint32_t x = 0; x < columns; ++x) {
            dx[x] = kernels::gradient_at(source, stride, columns, level.height, x, y, kernels::kDxTaps);
            dy[x] = kernels::gradient_at(source, stride, columns, level.height, x, y, kernels::kDyTaps);
        }
    }
}

HostViews Backend::host(const Buffers& level) {
    return HostViews{
        {level.src.data(), level.width, level.height, level.width},
        {level.dx.data(), level.width, level.height, level.width},
        {level.dy.data(), level.width, level.height, level.width},
    };
}

} // namespace cctag::portable::cpu
