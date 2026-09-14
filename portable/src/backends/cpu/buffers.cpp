/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "backends/cpu/backend.hpp"

namespace cctag::portable::cpu {

void Buffers::ensure(std::uint32_t level_width, std::uint32_t level_height) {
    width = level_width;
    height = level_height;
    const int rows = static_cast<int>(height);
    const int cols = static_cast<int>(width);

    // Only reallocate if the size has changed
    src.create(rows, cols);
    dx.create(rows, cols);
    dy.create(rows, cols);
}

} // namespace cctag::portable::cpu
