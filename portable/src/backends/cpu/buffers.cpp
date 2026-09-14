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
    edges.create(rows, cols);
    magnitude.create(rows + 2, cols + 2);
    nms_class.create(rows + 2, cols + 2);
    thinning.create(rows, cols);

    // The stages overwrite the interiors and never write these zero borders
    magnitude.setTo(0);
    nms_class.setTo(0);
    thinning.setTo(0);
    hysteresis_stack.clear();
    hysteresis_stack.reserve(static_cast<std::size_t>(width) * height);
}

} // namespace cctag::portable::cpu
