# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

eigen_add_benchmark(bench_cholesky bench_cholesky.cpp)
eigen_add_benchmark(bench_cholesky_double bench_cholesky.cpp DEFINITIONS SCALAR=double)
eigen_add_benchmark(bench_cholesky_inverse bench_cholesky_inverse.cpp)
eigen_add_benchmark(bench_cholesky_inverse_double bench_cholesky_inverse.cpp DEFINITIONS SCALAR=double)
eigen_add_benchmark(bench_bunchkaufman bench_bunchkaufman.cpp DEFINITIONS SCALAR=double)
eigen_add_benchmark(bench_bunchkaufman_cplx bench_bunchkaufman.cpp DEFINITIONS "SCALAR=std::complex<double>")
