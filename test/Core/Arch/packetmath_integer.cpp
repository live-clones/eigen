// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "packetmath_test_helpers.h"

// =============================================================================
// Tests for packetmath_integer
// =============================================================================
TEST(PacketmathIntegerTest, Basic) {
  g_first_pass = true;
  for (int i = 0; i < g_repeat; i++) {
    test::runner<int8_t>::run();
    test::runner<uint8_t>::run();
    test::runner<int16_t>::run();
    test::runner<uint16_t>::run();
    test::runner<int32_t>::run();
    test::runner<uint32_t>::run();
    test::runner<int64_t>::run();
    test::runner<uint64_t>::run();
    g_first_pass = false;
  }
}
