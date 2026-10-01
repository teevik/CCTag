/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_BACKENDS_SYCL_BACKEND_HPP
#define CCTAG_PORTABLE_BACKENDS_SYCL_BACKEND_HPP

#include "host/backend.hpp"

#include <stdexcept>

namespace cctag::portable::sycl_backend {

struct Buffers {
    void ensure(std::uint32_t, std::uint32_t) {}
};

struct Backend {
    using Buffers = sycl_backend::Buffers;

    static void load(Buffers&, kernels::Plane<const std::uint8_t>);
    static void pyramid(Buffers&, const Buffers&) {}
    static void gradient(Buffers&) {}
    static void edges(Buffers&, const Parameters&) {}
    static void edge_points(Buffers&) {}
    static void vote(Buffers&, const Parameters&) {}
    static void linking(Buffers&, const Parameters&) {}
    static void candidates(Context<Backend>&, const Parameters&) {}
    static void markers(Context<Backend>&, const Parameters&) {}

    static SnapshotViews snapshot_views(Buffers&) {
        throw std::logic_error("not implemented");
    }

    static void wait(Context<Backend>&) {}
};

static_assert(ExecutionBackend<Backend>);

} // namespace cctag::portable::sycl_backend

#endif
