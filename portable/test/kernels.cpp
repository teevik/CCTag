/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// Element functions against the OpenCV call they reproduce. Each element function the build lands
// adds one case here; like every portable test, they run under the reference's OPENCV_CPU_DISABLE
// mask (set by ctest), where OpenCV's results are the scalar ones.
#define BOOST_TEST_MODULE testPortableKernels

#define BOOST_TEST_DYN_LINK

#include "kernels/gradient.hpp"

#include <boost/test/unit_test.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>

using namespace cctag::portable;

namespace {

/// A deterministic, textured image; odd dimensions so no border coincides with a tap stride.
cv::Mat1b test_image(int width, int height) {
    cv::Mat1b image(height, width);
    std::uint32_t state = 12345u;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            state = state * 1664525u + 1013904223u;
            const int ring = ((x * x + y * y) / 37) % 2 ? 200 : 40;
            image(y, x) = static_cast<std::uint8_t>(ring + (state >> 27));
        }
    }
    return image;
}

/// `gradient_at` against `filter2D` at every pixel of a `width` x `height` image.
void expect_gradient_at_reproduces_filter2d(int width, int height) {
    const cv::Mat1b image = test_image(width, height);
    const cv::Mat1f kernel_dx(9, 9, const_cast<float*>(&kernels::kDerivativeKernel[0][0]));
    const cv::Mat1f kernel_dy = kernel_dx.t();
    cv::Mat1s dx, dy;
    cv::filter2D(image, dx, CV_16SC1, kernel_dx, cv::Point{-1, -1}, 0.0, cv::BORDER_REPLICATE);
    cv::filter2D(image, dy, CV_16SC1, kernel_dy, cv::Point{-1, -1}, 0.0, cv::BORDER_REPLICATE);

    const std::size_t stride = image.step1();
    const auto w = static_cast<std::uint32_t>(width);
    const auto h = static_cast<std::uint32_t>(height);
    for (std::uint32_t y = 0; y < h; ++y) {
        for (std::uint32_t x = 0; x < w; ++x) {
            BOOST_TEST_CONTEXT(width << "x" << height << " pixel (" << x << ", " << y << ")") {
                BOOST_CHECK_EQUAL(
                    kernels::gradient_at(image[0], stride, w, h, x, y, kernels::kDxTaps),
                    dx(static_cast<int>(y), static_cast<int>(x))
                );
                BOOST_CHECK_EQUAL(
                    kernels::gradient_at(image[0], stride, w, h, x, y, kernels::kDyTaps),
                    dy(static_cast<int>(y), static_cast<int>(x))
                );
            }
        }
    }
}

} // namespace

BOOST_AUTO_TEST_SUITE(kernels_suite)

BOOST_AUTO_TEST_CASE(gradient_at_reproduces_filter2d_at_every_pixel) {
    expect_gradient_at_reproduces_filter2d(37, 23);
}

// Every size up to the kernel's 9x9 footprint and a little beyond: images narrower than the kernel
// replicate one border pixel across several taps, and both sides of a pixel can clamp at once.
BOOST_AUTO_TEST_CASE(gradient_at_reproduces_filter2d_on_images_smaller_than_the_kernel) {
    for (int height = 1; height <= 12; ++height)
        for (int width = 1; width <= 12; ++width)
            expect_gradient_at_reproduces_filter2d(width, height);
}

BOOST_AUTO_TEST_SUITE_END()
