# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# this test seems to never have been successful on x87, so is considered to contain a FP-related bug.
# see thread: "non-linear optimization test summary"
ei_add_test(NonLinearOptimization)