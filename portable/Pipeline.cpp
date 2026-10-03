/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "portable/Pipeline.hpp"

#include <sycl/sycl.hpp>

#include <iostream>

namespace cctag {
namespace portable {

struct Pipeline::Impl
{
    sycl::queue queue;
};

Pipeline::Pipeline()
  : _impl(new Impl)
{
    std::cerr << "SYCL device: " << _impl->queue.get_device().get_info<sycl::info::device::name>() << std::endl;
}

Pipeline::~Pipeline() = default;

} // namespace portable
} // namespace cctag
