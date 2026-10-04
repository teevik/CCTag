/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "portable/Pipeline.hpp"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <iostream>
#include <vector>

namespace cctag {
namespace portable {

namespace {

/// The device images of one pyramid level, tightly packed
struct Level {
    std::uint32_t width;
    std::uint32_t height;
    std::uint8_t* source;
};

} // namespace

struct Pipeline::Impl {
    sycl::queue queue{sycl::property::queue::in_order{}};
    std::vector<Level> levels;

    void allocate(std::uint32_t width, std::uint32_t height, std::uint32_t count);
    void release();
    void downsample(const Level& src, const Level& dst);
};

void Pipeline::Impl::allocate(std::uint32_t width, std::uint32_t height, std::uint32_t count) {
    if (levels.size() == count && count > 0 && levels[0].width == width
        && levels[0].height == height) {
        return;
    }

    release();
    for (std::uint32_t i = 0; i < count; ++i) {
        levels.push_back(
            {width, height, sycl::malloc_device<std::uint8_t>(std::size_t(width) * height, queue)}
        );
        width /= 2;
        height /= 2;
    }
}

void Pipeline::Impl::release() {
    queue.wait();
    for (const Level& level : levels) {
        sycl::free(level.source, queue);
    }
    levels.clear();
}

/// Halves an image by averaging every 2x2 block, as cv::resize does when it halves
void Pipeline::Impl::downsample(const Level& src, const Level& dst) {
    const std::uint8_t* in = src.source;
    std::uint8_t* out = dst.source;
    const std::size_t inWidth = src.width;
    const std::size_t outWidth = dst.width;

    queue.parallel_for(sycl::range<2>(dst.height, dst.width), [=](sycl::id<2> id) {
        const std::size_t y = id[0];
        const std::size_t x = id[1];
        const std::uint8_t* top = in + 2 * y * inWidth + 2 * x;
        const std::uint8_t* bottom = top + inWidth;
        const std::uint32_t sum = top[0] + top[1] + bottom[0] + bottom[1];
        out[y * outWidth + x] = static_cast<std::uint8_t>((sum + 2) / 4);
    });
}

Pipeline::Pipeline() :
    _impl(new Impl) {
    std::cerr << "SYCL device: " << _impl->queue.get_device().get_info<sycl::info::device::name>()
              << std::endl;
}

Pipeline::~Pipeline() {
    _impl->release();
}

void Pipeline::load(
    const std::uint8_t* image,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t levels
) {
    _impl->allocate(width, height, levels);

    const std::vector<Level>& pyramid = _impl->levels;
    _impl->queue.memcpy(pyramid[0].source, image, std::size_t(width) * height);
    for (std::size_t i = 1; i < pyramid.size(); ++i) {
        _impl->downsample(pyramid[i - 1], pyramid[i]);
    }
}

void Pipeline::downloadSource(std::uint32_t level, std::uint8_t* dst) {
    const Level& src = _impl->levels[level];
    _impl->queue.memcpy(dst, src.source, std::size_t(src.width) * src.height).wait();
}

} // namespace portable
} // namespace cctag
