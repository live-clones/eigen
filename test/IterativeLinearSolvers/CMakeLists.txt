# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

ei_add_test(conjugate_gradient)
ei_add_test(incomplete_cholesky)
ei_add_test(incomplete_LUT)
ei_add_test(bicgstab)
ei_add_test(lscg)
ei_add_test(lsmr)

# Promoted upstream from unsupported/Eigen/IterativeSolvers into Eigen proper.
ei_add_test(bicgstabl)
ei_add_test(dgmres)
ei_add_test(gmres)
ei_add_test(idrs)
ei_add_test(idrstabl)
ei_add_test(minres)
