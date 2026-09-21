/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "backends/sycl/backend.hpp"

#include <sycl/sycl.hpp>

#include <exception>
#include <iostream>

namespace cctag::portable::sycl_backend {

void Backend::load(Buffers&, kernels::Plane<const std::uint8_t>) {
    const sycl::device device = sycl::device(sycl::gpu_selector_v);
    sycl::queue queue(device, [](sycl::exception_list errors) {
        for (const auto& error : errors) {
            std::rethrow_exception(error);
        }
    });

    std::clog << "Platform: " << device.get_platform().get_info<sycl::info::platform::name>()
              << '\n'
              << "Device: " << device.get_info<sycl::info::device::name>() << std::endl;

    int result = 0;
    sycl::buffer<int, 1> output(&result, sycl::range<1>(1));
    queue.submit([&](sycl::handler& handler) {
        auto value = output.get_access<sycl::access::mode::write>(handler);
        handler.single_task([=]() { value[0] = 42; });
    });
    queue.wait_and_throw();
    const sycl::host_accessor value(output, sycl::read_only);
    if (value[0] != 42) {
        throw std::runtime_error("smoke kernel returned an unexpected result");
    }
    std::clog << "PASS: SYCL kernel wrote " << value[0] << ".\n";
}

} // namespace cctag::portable::sycl_backend
