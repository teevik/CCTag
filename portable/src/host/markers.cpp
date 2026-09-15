/*
 * Copyright 2026, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "host/markers.hpp"

#include <cctag/CCTagMarkersBank.hpp>

namespace cctag::portable {

void MarkerBank::ensure(std::size_t count) {
    if (custom || crowns == count) {
        return;
    }
    set(CCTagMarkersBank(count));
    crowns = count;
    custom = false;
}

void MarkerBank::set(const CCTagMarkersBank& bank) {
    crowns = 0;
    ratios.clear();
    offsets.clear();
    offsets.push_back(0);
    for (const auto& marker : bank.getMarkers()) {
        ratios.insert(ratios.end(), marker.begin(), marker.end());
        offsets.push_back(ratios.size());
    }
    custom = true;
}

} // namespace cctag::portable
