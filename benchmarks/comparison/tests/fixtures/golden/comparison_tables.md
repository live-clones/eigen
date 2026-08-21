# Eigen versus Apple Accelerate on dense linear algebra

Rows are keyed by the BLAS/LAPACK mnemonic, with the equivalent Eigen spelling beside it, so a reader who navigates by BLAS/LAPACK naming can find the Eigen expression that performs the same work.

Every rate is GFLOP/s computed from the flop count in `ops.toml`, identical for both arms of a comparison. A ratio above 1 means Eigen is faster.

This dataset is partial by construction. A combination that was not measured is shown as `n/a` with a footnote giving the reason; an operation with no reference counterpart is shown as `—` and named as such. Neither is ever rendered as zero. The coverage manifest lists every such combination.

Merged 2026-08-20T00:00:00Z by reduce.py 1.0.0.

### double on Apple M4 Pro, aarch64-neon, AppleClang 17.0.0, 1 thread(s)

Apple M4 Pro, aarch64-neon, AppleClang 17.0.0, 1 thread(s); scalar type double

| BLAS/LAPACK | Eigen | Size | Eigen GFLOP/s | Apple Accelerate GFLOP/s | Eigen vs Apple Accelerate |
|---|---|---|---|---|---|
| FullPivLU [a] | `Eigen::FullPivLU<MatrixType> lu(A)` | m=512 n=512 | 31.50 | — [b] | no LAPACK counterpart [b] |
| GEMM (dgemm) | `C.noalias() += A * B` | m=24 n=24 k=24 | 8.00 | 5.00 | x1.60 |
| GEMM (dgemm) | `C.noalias() += A * B` | m=32 n=32 k=32 | 14.00 | 10.00 | x1.40 |
| GEMM (dgemm) | `C.noalias() += A * B` | m=48 n=48 k=48 | 28.00 | 30.00 | x0.93 |
| GEMM (dgemm) | `C.noalias() += A * B` | m=64 n=64 k=64 | 45.00 | 60.00 | x0.75 |
| GEMM (dgemm) | `C.noalias() += A * B` | m=96 n=96 k=96 | 78.00 | 90.00 | x0.87 (inconclusive) [c] |
| GEMM (dgemm) | `C.noalias() += A * B` | m=128 n=128 k=128 | 96.00 | n/a [d] | n/a [e] |

## Notes

- **[a]** No LAPACK driver: complete-pivoting LU (the deprecated ?gefa family aside) has no supported netlib equivalent. The flop count for FullPivLU is a convention, not a hardware-truthful operation count. It is applied identically to both arms, so the ratio is meaningful even though the absolute rate is not a machine-efficiency figure.
- **[b]** No LAPACK driver: complete-pivoting LU (the deprecated ?gefa family aside) has no supported netlib equivalent.
- **[c]** The two arms' [median - MAD, median + MAD] intervals overlap: the difference is smaller than the observed run-to-run variation and is neither a win nor a regression.
- **[d]** The reference build in this contribution did not register m=n=k=128
- **[e]** At least one arm of this comparison was not measured; see the coverage manifest.
