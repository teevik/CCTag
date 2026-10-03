/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_PIPELINE_HPP
#define CCTAG_PORTABLE_PIPELINE_HPP

#include <memory>

namespace cctag {
namespace portable {

/// The stages of a detection that run on a SYCL device
class Pipeline
{
  public:
    Pipeline();
    ~Pipeline();

  private:
    // Keeps SYCL out of the headers that host code includes
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace portable
} // namespace cctag

#endif
