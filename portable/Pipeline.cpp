/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "portable/Pipeline.hpp"

#include <sycl/sycl.hpp>

#include <array>
#include <cstddef>
#include <iostream>
#include <map>
#include <mutex>
#include <vector>

namespace cctag {
namespace portable {

namespace {

/// A pixel of the edge map: weak or strong until hysteresis, then edge or none
enum class Mark : std::uint8_t {
    none = 0,
    weak = 1,
    strong = 2,
    edge = 255
};

/// The device images of one pyramid level, tightly packed
struct Level {
    std::uint32_t width;
    std::uint32_t height;
    std::uint8_t* source;
    std::int16_t* dx;
    std::int16_t* dy;
    /// The border is always none, so the kernels after suppress need no bounds checks
    Mark* map;
};

/// A 9-tap filter, centred on the 5th tap
using Taps = std::array<float, 9>;

/// The two halves of h_gauss_filter in cuda/frame_02_gaussian.cu, whose outer product is the 9x9
/// kernel of cvRecodedCanny
constexpr Taps gauss{
    0.000053390535453f,
    0.001768051711852f,
    0.021539279301849f,
    0.096532352630054f,
    0.159154943091895f,
    0.096532352630054f,
    0.021539279301849f,
    0.001768051711852f,
    0.000053390535453f
};
constexpr Taps derivative{
    -0.002683701023220f,
    -0.066653979229454f,
    -0.541341132946452f,
    -1.213061319425269f,
    0.0f,
    1.213061319425269f,
    0.541341132946452f,
    0.066653979229454f,
    0.002683701023220f
};

/// tan(22.5°) and tan(67.5°), the bounds of the diagonal sector of non-maximum suppression
constexpr float tan22 = 0.41421356f;
constexpr float tan67 = 2.41421356f;

/// A LUT of filter/thinning.cpp at the indices with the centre set, one bit per neighbourhood
using Lut = std::array<std::uint32_t, 8>;

constexpr Lut thin1{
    0x3333337f,
    0x33ff31ff,
    0xffffffff,
    0xffffffff,
    0x3323ffff,
    0xffffffff,
    0xff22ffff,
    0xffffffff
};
constexpr Lut thin2{
    0xfffff7ff,
    0xfffff5ff,
    0xeceef4ff,
    0xeceef4ff,
    0xffefffff,
    0xffffffff,
    0xeceef4ff,
    0xeceef4fe
};

/// Branch points a hysteresis walker remembers to come back to
constexpr int walkStack = 64;

} // namespace

struct Pipeline {
    sycl::queue queue{sycl::property::queue::in_order{}};
    std::vector<Level> levels;
    /// Temporary image the size of level 0, can be reused between passes
    float* intermediate = nullptr;
};

namespace {

void release(Pipeline& pipeline) {
    pipeline.queue.wait();
    for (const Level& level : pipeline.levels) {
        sycl::free(level.source, pipeline.queue);
        sycl::free(level.dx, pipeline.queue);
        sycl::free(level.dy, pipeline.queue);
        sycl::free(level.map, pipeline.queue);
    }
    pipeline.levels.clear();
    sycl::free(pipeline.intermediate, pipeline.queue);
    pipeline.intermediate = nullptr;
}

void allocate(Pipeline& pipeline, std::uint32_t width, std::uint32_t height, std::uint32_t count) {
    const std::vector<Level>& levels = pipeline.levels;
    if (levels.size() == count && count > 0 && levels[0].width == width
        && levels[0].height == height) {
        return;
    }

    release(pipeline);
    pipeline.intermediate = sycl::malloc_device<float>(std::size_t(width) * height, pipeline.queue);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::size_t size = std::size_t(width) * height;
        pipeline.levels.push_back(
            {width,
             height,
             sycl::malloc_device<std::uint8_t>(size, pipeline.queue),
             sycl::malloc_device<std::int16_t>(size, pipeline.queue),
             sycl::malloc_device<std::int16_t>(size, pipeline.queue),
             sycl::malloc_device<Mark>(size, pipeline.queue)}
        );
        width /= 2;
        height /= 2;
    }
}

/// Halves an image by averaging every 2x2 block, as cv::resize does when it halves
void downsample(sycl::queue& queue, const Level& src, const Level& dst) {
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

/// Filters every column of the source image, replicating the border rows
void filterColumns(sycl::queue& queue, const Level& level, const Taps taps, float* out) {
    const std::uint8_t* in = level.source;
    const int width = level.width;
    const int height = level.height;

    queue.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> id) {
        const int y = id[0];
        const int x = id[1];
        float sum = 0.0f;
        for (int k = 0; k < 9; ++k) {
            const int row = sycl::clamp(y + k - 4, 0, height - 1);
            sum += taps[k] * in[row * width + x];
        }
        out[y * width + x] = sum;
    });
}

/// Filters every row of an intermediate image, replicating the border columns and rounding to
/// nearest as cv::saturate_cast does
void filterRows(
    sycl::queue& queue,
    const Level& level,
    const float* in,
    const Taps taps,
    std::int16_t* out
) {
    const int width = level.width;
    const int height = level.height;

    queue.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> id) {
        const int y = id[0];
        const int x = id[1];
        float sum = 0.0f;
        for (int k = 0; k < 9; ++k) {
            const int column = sycl::clamp(x + k - 4, 0, width - 1);
            sum += taps[k] * in[y * width + column];
        }
        out[y * width + x] = static_cast<std::int16_t>(sycl::rint(sum));
    });
}

/// Computes dx and dy of a level as cvRecodedCanny does, with separable filters
void gradient(Pipeline& pipeline, const Level& level) {
    filterColumns(pipeline.queue, level, gauss, pipeline.intermediate);
    filterRows(pipeline.queue, level, pipeline.intermediate, derivative, level.dx);
    filterColumns(pipeline.queue, level, derivative, pipeline.intermediate);
    filterRows(pipeline.queue, level, pipeline.intermediate, gauss, level.dy);
}

/// The gradient magnitude at index i, rounded
int magnitude(const std::int16_t* dx, const std::int16_t* dy, int i) {
    const int gx = dx[i];
    const int gy = dy[i];
    return sycl::rint(sycl::sqrt(float(gx * gx + gy * gy)));
}

/// Non-maximum suppression: marks the pixels whose gradient magnitude peaks along the gradient
/// direction strong above high and weak above low, and everything else, including the whole
/// border, none
void suppress(sycl::queue& queue, const Level& level, float low, float high) {
    const std::int16_t* dx = level.dx;
    const std::int16_t* dy = level.dy;
    Mark* map = level.map;
    const int width = level.width;
    const int height = level.height;

    queue.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> id) {
        const int y = id[0];
        const int x = id[1];
        const int i = y * width + x;
        const bool border = x == 0 || y == 0 || x == width - 1 || y == height - 1;
        const int m = magnitude(dx, dy, i);
        if (border || m <= low) {
            map[i] = Mark::none;
            return;
        }

        // The index step to the next pixel along the gradient: right when the gradient is within
        // 22.5° of horizontal, down when within 22.5° of vertical, else down-right or down-left
        const int gx = dx[i];
        const int gy = dy[i];
        const float ax = sycl::abs(gx);
        const float ay = sycl::abs(gy);
        int step;
        bool diagonal = false;
        if (ay < tan22 * ax) {
            step = 1;
        } else if (ay > tan67 * ax) {
            step = width;
        } else {
            const bool sameSign = (gx < 0) == (gy < 0);
            step = sameSign ? width + 1 : width - 1;
            diagonal = true;
        }

        // A tie with the neighbour before suppresses the pixel. A tie with the one after keeps it,
        // so one pixel of a plateau survives, except along a diagonal, where cvRecodedCanny
        // suppresses both; keeping one there adds 5-10 % edge pixels.
        const int before = magnitude(dx, dy, i - step);
        const int after = magnitude(dx, dy, i + step);
        const bool maximum = m > before && (diagonal ? m > after : m >= after);
        map[i] = !maximum ? Mark::none : m > high ? Mark::strong : Mark::weak;
    });
}

/// The 8-neighbours of an interior pixel with a mark the pixel itself does not have: the index of
/// the first in scan order, or -1, and how many there are
struct Neighbours {
    int first;
    int count;
};

Neighbours neighbours(const Mark* map, int width, int i, Mark mark) {
    Neighbours found{-1, 0};
    for (int v = -1; v <= 1; ++v) {
        for (int u = -1; u <= 1; ++u) {
            const int j = i + v * width + u;
            if (map[j] == mark) {
                if (found.first < 0) {
                    found.first = j;
                }
                ++found.count;
            }
        }
    }
    return found;
}

/// Every weak pixel with a strong 8-neighbour becomes strong and walks its chain depth first,
/// promoting the first weak neighbour and stacking the pixel if it has more. Sets pending if a full
/// stack makes a walker leave a weak neighbour behind.
void walk(sycl::queue& queue, const Level& level, std::uint32_t* pending) {
    Mark* map = level.map;
    const int width = level.width;
    const std::size_t size = std::size_t(level.width) * level.height;

    queue.parallel_for(sycl::range<1>(size), [=](sycl::id<1> id) {
        int i = id[0];
        if (map[i] != Mark::weak || neighbours(map, width, i, Mark::strong).count == 0) {
            return;
        }
        map[i] = Mark::strong;
        int stack[walkStack];
        int top = 0;
        for (;;) {
            const Neighbours weak = neighbours(map, width, i, Mark::weak);
            if (weak.first < 0) {
                if (top == 0) {
                    return;
                }
                i = stack[--top];
                continue;
            }
            if (weak.count > 1) {
                if (top < walkStack) {
                    stack[top++] = i;
                } else {
                    *pending = 1;
                }
            }
            map[weak.first] = Mark::strong;
            i = weak.first;
        }
    });
}

/// Makes every weak pixel of the map connected to a strong one strong, so strong marks exactly
/// the edges. Walks until no walker leaves work behind, usually after one launch. Walkers race on
/// the map, which is safe as pixels only turn strong.
void hysteresis(Pipeline& pipeline, const Level& level) {
    sycl::queue& queue = pipeline.queue;
    std::uint32_t* pending = sycl::malloc_device<std::uint32_t>(1, queue);
    std::uint32_t any = 1;
    while (any) {
        queue.memset(pending, 0, sizeof(any));
        walk(queue, level, pending);
        queue.memcpy(&any, pending, sizeof(any)).wait();
    }
    sycl::free(pending, queue);
}

/// One imageIter of filter/thinning.cpp over a whole level, where in marks edges with edge: each
/// interior pixel of out is marked edge where the LUT keeps one, else none, and the border is none
void thinPass(
    sycl::queue& queue,
    const Level& level,
    const Mark* in,
    Mark edge,
    const Lut lut,
    Mark* out
) {
    const int width = level.width;
    const int height = level.height;

    queue.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> id) {
        const int y = id[0];
        const int x = id[1];
        if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
            out[y * width + x] = Mark::none;
            return;
        }
        const auto at = [=](int u, int v) -> std::uint32_t {
            return in[(y + v) * width + x + u] == edge;
        };
        const std::uint32_t index = at(-1, -1) | at(-1, 0) << 1 | at(-1, 1) << 2 | at(0, -1) << 3
            | at(0, 1) << 4 | at(1, -1) << 5 | at(1, 0) << 6 | at(1, 1) << 7;
        const bool keep = at(0, 0) && ((lut[index / 32] >> index % 32) & 1);
        out[y * width + x] = keep ? Mark::edge : Mark::none;
    });
}

/// Thins the strong pixels of the map in place into the edge image, marked edge or none, as thin
/// does on the CPU
void thin(Pipeline& pipeline, const Level& level) {
    auto* temp = reinterpret_cast<Mark*>(pipeline.intermediate);
    thinPass(pipeline.queue, level, level.map, Mark::strong, thin1, temp);
    thinPass(pipeline.queue, level, temp, Mark::edge, thin2, level.map);
}

/// Computes the thinned edge image of a level as cvRecodedCanny and thin do, with thresholds on
/// the gradient magnitude
void edges(Pipeline& pipeline, const Level& level, float low, float high) {
    suppress(pipeline.queue, level, low, high);
    hysteresis(pipeline, level);
    thin(pipeline, level);
}

} // namespace

Pipeline& pipeline(int pipeId) {
    // Never destroyed: AdaptiveCpp cannot free device memory from a static destructor
    static std::map<int, Pipeline>& pipelines = *new std::map<int, Pipeline>;
    static std::mutex mutex;

    const std::lock_guard<std::mutex> lock(mutex);
    const auto inserted = pipelines.try_emplace(pipeId);
    Pipeline& pipeline = inserted.first->second;
    if (inserted.second) {
        std::cerr << "SYCL device: "
                  << pipeline.queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    }
    return pipeline;
}

void load(
    Pipeline& pipeline,
    const std::uint8_t* image,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t levels,
    float cannyThrLow,
    float cannyThrHigh
) {
    allocate(pipeline, width, height, levels);

    const std::vector<Level>& pyramid = pipeline.levels;
    pipeline.queue.memcpy(pyramid[0].source, image, std::size_t(width) * height);
    for (std::size_t i = 1; i < pyramid.size(); ++i) {
        downsample(pipeline.queue, pyramid[i - 1], pyramid[i]);
    }
    // cvRecodedCanny scales the thresholds by 256 to compare with the magnitude
    const float low = cannyThrLow * 256;
    const float high = cannyThrHigh * 256;
    for (const Level& level : pyramid) {
        gradient(pipeline, level);
        edges(pipeline, level, low, high);
    }
}

void downloadSource(Pipeline& pipeline, std::uint32_t level, std::uint8_t* dst) {
    const Level& src = pipeline.levels[level];
    pipeline.queue.memcpy(dst, src.source, std::size_t(src.width) * src.height).wait();
}

void downloadGradient(Pipeline& pipeline, std::uint32_t level, std::int16_t* dx, std::int16_t* dy) {
    const Level& src = pipeline.levels[level];
    const std::size_t size = std::size_t(src.width) * src.height;
    pipeline.queue.memcpy(dx, src.dx, size * sizeof(std::int16_t));
    pipeline.queue.memcpy(dy, src.dy, size * sizeof(std::int16_t)).wait();
}

void downloadEdges(Pipeline& pipeline, std::uint32_t level, std::uint8_t* dst) {
    const Level& src = pipeline.levels[level];
    pipeline.queue.memcpy(dst, src.map, std::size_t(src.width) * src.height).wait();
}

} // namespace portable
} // namespace cctag
