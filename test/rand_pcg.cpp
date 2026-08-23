// Build the rand test against the opt-in PCG backend.
//
// EIGEN_USE_PCG_RANDOM is what turns on EIGEN_HAS_THREAD_LOCAL_RANDOM, and with
// it the thread-safety subtest and the reproducibility subtest's use of the new
// generator.  Without a target that defines it, none of the behavior this
// backend exists to provide is compiled at all.  `rand` still covers the
// std::rand() fallback, so both configurations are tested.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#define EIGEN_USE_PCG_RANDOM
#include "rand.cpp"
