/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef CCTAG_PORTABLE_KERNELS_PCG32_HPP
#define CCTAG_PORTABLE_KERNELS_PCG32_HPP

#include <array>
#include <bit>
#include <cstdint>

namespace cctag::portable::kernels {

/// One candidate's pcg32 setseq stream, continued from loop two into loop three
struct Pcg32 {
    std::uint64_t state = 0;
    std::uint64_t increment = 1;
};

inline std::uint32_t pcg32_next(Pcg32& random) {
    const auto previous = random.state;
    random.state = previous * 6364136223846793005ULL + random.increment;
    const auto value = static_cast<std::uint32_t>(((previous >> 18) ^ previous) >> 27);
    return std::rotr(value, static_cast<int>(previous >> 59));
}

inline void pcg32_seed(Pcg32& random, std::uint64_t seed, std::uint64_t stream) {
    random.state = 0;
    random.increment = (stream << 1) | 1;
    pcg32_next(random);
    random.state += seed;
    pcg32_next(random);
}

/// Draws below a positive bound without modulo bias, as the legacy bounded_rand does
inline std::uint32_t pcg32_bounded(Pcg32& random, std::uint32_t bound) {
    const std::uint32_t threshold = -bound % bound;
    for (;;) {
        const auto value = pcg32_next(random);
        if (value >= threshold) {
            return value % bound;
        }
    }
}

/// Draws five distinct indices in draw order; count must be at least five
inline std::array<std::int32_t, 5> rand_5_k(Pcg32& random, std::uint32_t count) {
    std::array<std::int32_t, 5> indices;
    for (int i = 0; i < 5; ++i) {
        bool duplicate;
        do {
            indices[i] = static_cast<std::int32_t>(pcg32_bounded(random, count));
            duplicate = false;
            for (int j = 0; j < i; ++j) {
                duplicate |= indices[j] == indices[i];
            }
        } while (duplicate);
    }
    return indices;
}

} // namespace cctag::portable::kernels

#ifdef CCTAG_TEST
#include <boost/ut.hpp>

namespace cctag::portable::tests::pcg32 {

using namespace boost::ut;

inline suite<"pcg32"> pcg32_suite = [] {
    "pcg32 reproduces the legacy setseq sequence"_test = [] {
        kernels::Pcg32 random;
        kernels::pcg32_seed(random, 42, 54);
        // Recorded from the legacy's vendored pcg32(42, 54)
        for (const auto expected :
             {0xa15c02b7u, 0x7b47f409u, 0xba1d3330u, 0x83d2f293u, 0xbfa4784bu, 0xcbed606eu}) {
            expect(eq(kernels::pcg32_next(random), expected));
        }
    };

    "bounded draws reject the biased tail and retain the candidate stream"_test = [] {
        kernels::Pcg32 random;
        kernels::pcg32_seed(random, 271828, (std::uint64_t{3} << 32) | 42);
        // A bound just above 2^31 rejects about half the draws; values from the legacy generator
        for (const auto expected : {356052018u, 760635042u, 314693708u, 641323914u, 1797617292u}) {
            expect(eq(kernels::pcg32_bounded(random, 0x80000001u), expected));
        }
        expect(eq(kernels::pcg32_next(random), 3130194086u));
    };

    "five distinct draws preserve the legacy draw order across duplicate retries"_test = [] {
        kernels::Pcg32 random;
        kernels::pcg32_seed(random, 271828, (std::uint64_t{3} << 32) | 42);
        expect(kernels::rand_5_k(random, 5) == std::array<std::int32_t, 5>{2, 0, 1, 3, 4});
        expect(eq(kernels::pcg32_next(random), 726611558u));
    };
};

} // namespace cctag::portable::tests::pcg32
#endif

#endif
