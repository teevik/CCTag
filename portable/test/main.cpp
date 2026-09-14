/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <boost/ut.hpp>

int main(int argc, const char** argv) {
    return boost::ut::cfg<boost::ut::override>.run({.argc = argc, .argv = argv});
}
