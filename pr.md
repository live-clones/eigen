# PR: ARM SME2 GEMM Kernel Optimization for Apple M4

This PR introduces a high-performance GEMM (General Matrix-Matrix Multiplication) kernel utilizing the **ARM Scalable Matrix Extension (SME2)**, specifically optimized for the Apple M4 family of processors.

## Summary of Changes

- **SME2 GEMM Kernel**: Implemented a new GEMM kernel in `Eigen/src/Core/arch/SME/GeneralBlockPanelKernel.h` using SME2 `svmopa` (outer product) instructions.
- **Architectural Integration**: 
    - Updated `ConfigureVectorization.h` to automatically enable SME support if the compiler targets an SME-capable architecture (detecting `__ARM_FEATURE_SME`).
    - Enabled **NEON-SME Coexistence**: Unlike previous iterations, this implementation allows SME GEMM to coexist with NEON packet math. This ensures that while GEMM is accelerated via SME, all other vector operations (additions, divisions, etc.) continue to benefit from NEON vectorization rather than falling back to scalar code.
    - Updated `Architecture::Target` logic to prioritize SME blocking (16x16) for matrix products when available.
- **Performance Optimizations**:
    - Vectorized the store-back loop using Vertical/Horizontal ZA tile reads to match Eigen's memory layout.
    - Eliminated streaming mode switching overhead (`SMSTOP`/`SMSTART`) by separating pointer preparation from the compute loop.

## Performance Benchmarks

Benchmarks were performed on an **Apple M4 Pro (14 cores)** using a 4096x4096x4096 float matrix multiplication (Single Thread). Results are averaged over **10 runs**.

| Implementation | Time (mean ± stddev) | GFLOPS | Speedup |
| :--- | :--- | :--- | :--- |
| **NEON (Baseline)** | 1.2130s ± 0.0027s | ~113.3 | 1.00x |
| **SME (Optimized)** | **0.3933s ± 0.0033s** | **~349.5** | **~3.08x** |

*Note: The extremely low standard deviation (<1%) confirms the stability and reliability of the performance gains.*

## Correctness Verification

Correctness has been verified across multiple matrix sizes (including odd boundaries) using Eigen's internal test suite and custom validation tools.

### 1. Official Unit Test
The following command runs the official Eigen product unit test specialized for SME:
```bash
/opt/homebrew/Cellar/llvm@20/20.1.8/bin/clang++ -O3 -std=c++17 -I. 
    -mcpu=apple-m4+sme+nosve 
    test/product_sme.cpp -o test/product_sme_exe
./test/product_sme_exe
```
**Result:** `Test Passed` (Verified 10/10 repetitions for random sizes).

### 2. Large Scale Verification
Verified 1024x1024 random matrix multiplication against Eigen's reference implementation:
```bash
# Correctness Check Output
Size: 1024x1024
Error: 0
EIGEN_VECTORIZE_SME is defined (SUCCESS)
```

## How to use in Downstream (e.g., XLA/TensorFlow)

This implementation is designed to be "Plug-and-Play". Downstream projects targeting the Apple M4 or ARMv9-A SME-capable CPUs will automatically utilize the SME GEMM kernel without any additional macro definitions:

```bash
# Compilation flag to enable SME in Eigen automatically
-mcpu=apple-m4 
# OR
-march=armv9-a+sme
```

## Cleanup & Refactoring
- Cleaned up experimental headers and temporary test files.
- Separated `run_sme_gemm_impl` (locally streaming) from `run_sme_gemm` (driver) to ensure stability in non-streaming environments.
