// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Answers oneMKL's CPUID vendor check with "Intel" from inside the binary.
//
// oneMKL chooses its kernels from the vendor string as well as the feature
// bits: on a CPU that is not GenuineIntel it takes a conservative path, which on
// Zen 5 with AVX-512 runs dgemm at half of Eigen's rate (measured 2026-09). The
// library obtains the predicate through mkl_serv_get_cpu_true(), an exported
// function that returns a pointer to it, called through the PLT; a definition in
// the executable therefore interposes on libmkl_core's own, provided the symbol
// is in the executable's dynamic symbol table (vendors.cmake adds
// --export-dynamic-symbol for it). Only the `mkl_bypass` arm compiles this file.
//
// Intel does not support the configuration. MKL_DEBUG_CPU_TYPE, the variable that
// used to do this, was removed in MKL 2020 Update 1, and the older override point
// mkl_serv_intel_cpu_true() no longer exists in 2025.3.

extern "C" {

static int eigenBenchCpuIsIntel() { return 1; }

void* mkl_serv_get_cpu_true() { return reinterpret_cast<void*>(&eigenBenchCpuIsIntel); }

}  // extern "C"
