// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

// stream_ordering with the per-thread default stream, as nvcc's
// --default-stream per-thread selects: the module never passes the default
// stream, so its ordering must not depend on which one that is.

#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#include "stream_ordering.cpp"
