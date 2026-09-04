/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_HOST_PARALLEL_HPP
#define CCTAG_PORTABLE_HOST_PARALLEL_HPP

// The CPU backend's parallel idiom (ADR 0003): a visible loop carrying an OpenMP pragma. The macro
// is empty unless the translation unit is compiled with OpenMP, which is the sequential build.
// Each iteration writes only the outputs its own element owns, so the schedule never changes a
// result; `OMP_NUM_THREADS=1` is the sequential run of the same binary.

#if defined(_OPENMP) && defined(CCTAG_PORTABLE_PARALLEL)
#define CCTAG_PORTABLE_PRAGMA(x) _Pragma(#x)
#define CCTAG_PORTABLE_PARALLEL_FOR_STATIC CCTAG_PORTABLE_PRAGMA(omp parallel for schedule(static))
#define CCTAG_PORTABLE_PARALLEL_FOR_DYNAMIC CCTAG_PORTABLE_PRAGMA(omp parallel for schedule(dynamic, 1))
#else
#define CCTAG_PORTABLE_PARALLEL_FOR_STATIC
#define CCTAG_PORTABLE_PARALLEL_FOR_DYNAMIC
#endif

#endif
