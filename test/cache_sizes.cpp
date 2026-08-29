// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#ifdef EIGEN_CPU_CACHE_SYSFS
#include <cstdlib>
#endif

void cache_sizes_plausible() {
  // Whatever the platform reports has to be a plausible data-cache size: a stray unit suffix
  // would otherwise be read as a handful of bytes and silently shrink every blocking size.
  std::ptrdiff_t l1 = -1, l2 = -1, l3 = -1, l3_per_cpu = -1;
  internal::queryCacheSizes(l1, l2, l3, l3_per_cpu);
  for (std::ptrdiff_t size : {l1, l2, l3, l3_per_cpu})
    if (size > 0) VERIFY(size >= 1024);
  // A share is one CPU's slice of one L3 instance, so it can never exceed the reported L3.
  VERIFY(l3_per_cpu >= 0);
  if (l3_per_cpu > 0) VERIFY(l3 > 0 && l3_per_cpu <= l3);
}

#ifdef EIGEN_CPU_CACHE_SYSFS

void cache_sizes_parsers() {
  VERIFY(internal::parseCpuCacheSize("64K\n") == 64 * 1024);
  VERIFY(internal::parseCpuCacheSize("2048K\n") == 2048 * 1024);
  VERIFY(internal::parseCpuCacheSize("32M\n") == 32 * 1024 * 1024);
  // A size Eigen cannot make sense of has to read as "unknown", never as a few bytes.
  VERIFY(internal::parseCpuCacheSize("bogus") == 0);
  VERIFY(internal::parseCpuCacheSize("0K") == 0);
  // A bare count is bytes, not kibibytes.
  VERIFY(internal::parseCpuCacheSize("512\n") == 512);

  VERIFY(internal::parseCpuListCount("0\n") == 1);
  VERIFY(internal::parseCpuListCount("0-3\n") == 4);
  VERIFY(internal::parseCpuListCount("0-3,8-11\n") == 8);
  VERIFY(internal::parseCpuListCount("0,2,4\n") == 3);
  VERIFY(internal::parseCpuListCount("") == 0);
  VERIFY(internal::parseCpuListCount("3-0") == 0);
  VERIFY(internal::parseCpuListCount("0-3\r\n") == 4);
  // Trailing text means the format is not the one assumed, so the count is not trustworthy:
  // a small count here would inflate l3_per_cpu rather than leave the share unknown.
  VERIFY(internal::parseCpuListCount("0-3junk") == 0);
  VERIFY(internal::parseCpuListCount("0-3 8-11") == 0);
  VERIFY(internal::parseCpuListCount("0-3,") == 0);
  VERIFY(internal::parseCpuListCount("0-3,junk") == 0);
  VERIFY(internal::parseCpuListCount("junk") == 0);
}

void cache_sizes_sysfs() {
  // On a kernel that publishes the topology, Eigen has to pick it up instead of falling back to
  // its compiled-in defaults. glibc's sysconf answers only on x86, so before the sysfs fallback
  // this failed on every other Linux architecture.
  std::ptrdiff_t l1, l2, l3, l3_per_cpu;
  internal::queryCacheSizes(l1, l2, l3, l3_per_cpu);
  char probe[32];
  if (internal::readCpuCacheAttribute(0, "size", probe)) VERIFY(l1 > 0);
  // The share is the one the same detection pass derived, not a second, unmanaged query.
  VERIFY(l3_per_cpu == internal::queryCpuCacheTopologySysfs().l3_per_cpu);
}

#endif  // EIGEN_CPU_CACHE_SYSFS

EIGEN_DECLARE_TEST(cache_sizes) {
  CALL_SUBTEST(cache_sizes_plausible());
#ifdef EIGEN_CPU_CACHE_SYSFS
  CALL_SUBTEST(cache_sizes_parsers());
  CALL_SUBTEST(cache_sizes_sysfs());
#endif
}
