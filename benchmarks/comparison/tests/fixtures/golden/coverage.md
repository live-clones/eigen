# Benchmark comparison coverage

This manifest states exactly which machine, library and operation combinations were measured. Anything absent from the tables is listed here with the reason it is absent, so a partial dataset is self-describing.

- measured arm cells: **12**
- not measured arm cells: **2**
- unaccounted (harness bug): **0**

## Configurations

| Config | Machine | ISA | Compiler | Threads | Eigen commit |
|---|---|---|---|---|---|
| `m4pro__aarch64-neon__appleclang17__e2a2fda17__t1` | Apple M4 Pro | aarch64-neon | AppleClang 17.0.0 | 1 | `e2a2fda17` |

## Operations

| BLAS/LAPACK | Eigen | Arms | Measured | Not measured | Unaccounted |
|---|---|---|---|---|---|
| FullPivLU | `Eigen::FullPivLU<MatrixType> lu(A)` | accelerate, eigen | 1 | 1 | 0 |
| GEMM | `C.noalias() += A * B` | accelerate, eigen | 11 | 1 | 0 |

## Not measured, by reason

| Config | BLAS/LAPACK | Arm | Scalar | Reason | Cells | Shapes |
|---|---|---|---|---|---|---|
| `m4pro__aarch64-neon__appleclang17__e2a2fda17__t1` | FullPivLU | accelerate | f64 | `no_reference_equivalent` | 1 | 512x512 |
| `m4pro__aarch64-neon__appleclang17__e2a2fda17__t1` | GEMM | accelerate | f64 | `reference_routine_absent` | 1 | 128x128x128 |

Details:

- `no_reference_equivalent`: No LAPACK driver: complete-pivoting LU (the deprecated ?gefa family aside) has no supported netlib equivalent.
- `reference_routine_absent`: The reference build in this contribution did not register m=n=k=128
