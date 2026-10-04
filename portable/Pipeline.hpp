/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_PIPELINE_HPP
#define CCTAG_PORTABLE_PIPELINE_HPP

#include <cstdint>
#include <memory>

namespace cctag {
namespace portable {

/// The stages of a detection that run on a SYCL device
class Pipeline {
  public:
    Pipeline();
    ~Pipeline();

    /// Uploads a tightly packed gray image and builds the pyramid, each level half the size of the
    /// one above
    void load(
        const std::uint8_t* image,
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t levels
    );

    /// Copies the source image of a level into a tightly packed host image
    void downloadSource(std::uint32_t level, std::uint8_t* dst);

  private:
    // Keeps SYCL out of the headers that host code includes
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace portable
} // namespace cctag

#endif
