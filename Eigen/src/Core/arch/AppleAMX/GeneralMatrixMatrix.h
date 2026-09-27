// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_APPLE_AMX_GENERALMATRIXMATRIX_H
#define EIGEN_APPLE_AMX_GENERALMATRIXMATRIX_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

#ifndef EIGEN_APPLE_AMX_MIN_RESULT_SIZE
#define EIGEN_APPLE_AMX_MIN_RESULT_SIZE 256
#endif

namespace Eigen {
namespace internal {
namespace apple_amx {

// Instruction encodings and register model: P. Cawley, "Apple AMX instruction set", https://github.com/corsix/amx.
template <int Op>
EIGEN_ALWAYS_INLINE void issue(uint64_t operand) {
  __asm__ volatile(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" : : "i"(Op), "r"(operand) : "memory");
}
EIGEN_ALWAYS_INLINE void set() { __asm__ volatile("nop\nnop\nnop\n.word (0x201000 + (17 << 5) + 0)" : : : "memory"); }
EIGEN_ALWAYS_INLINE void clr() { __asm__ volatile("nop\nnop\nnop\n.word (0x201000 + (17 << 5) + 1)" : : : "memory"); }
EIGEN_ALWAYS_INLINE void ldx(uint64_t o) { issue<0>(o); }
EIGEN_ALWAYS_INLINE void ldy(uint64_t o) { issue<1>(o); }
EIGEN_ALWAYS_INLINE void sty(uint64_t o) { issue<3>(o); }
EIGEN_ALWAYS_INLINE void ldz(uint64_t o) { issue<4>(o); }
EIGEN_ALWAYS_INLINE void stz(uint64_t o) { issue<5>(o); }
EIGEN_ALWAYS_INLINE void extry(uint64_t o) { issue<9>(o); }
template <typename T>
EIGEN_ALWAYS_INLINE void fma(uint64_t o);
template <>
EIGEN_ALWAYS_INLINE void fma<float>(uint64_t o) {
  issue<12>(o);
}
template <>
EIGEN_ALWAYS_INLINE void fma<double>(uint64_t o) {
  issue<10>(o);
}

// Load/store operand: address in bits 0-55, register from bit 56, pair (bit 62), quad from the M2 on (bits 62 and 60).
constexpr uint64_t kPair = uint64_t(1) << 62, kQuad = (uint64_t(1) << 62) | (uint64_t(1) << 60);
EIGEN_ALWAYS_INLINE uint64_t address(const void* p) {
  return reinterpret_cast<uint64_t>(p) & ((uint64_t(1) << 56) - 1);
}
EIGEN_ALWAYS_INLINE uint64_t xy(const void* p, int reg, uint64_t multiple = 0) {
  return address(p) | uint64_t(reg) << 56 | multiple;
}
EIGEN_ALWAYS_INLINE uint64_t zrow(const void* p, int row, bool pair = false) {
  return address(p) | uint64_t(row) << 56 | (pair ? kPair : 0);
}
// Matrix-mode fma: z[zr + s*j][i] += x[i] * y[j], s = 4 (fp32) or 8 (fp64); x and y offsets in bytes.
constexpr uint64_t fma_op(int zr, int xoff, int yoff) {
  return uint64_t(zr) << 20 | uint64_t(xoff & 0x1FF) << 10 | uint64_t(yoff & 0x1FF);
}

// AMX generation from hw.cpufamily (mach/machine.h), since macOS traps user reads of MIDR_EL1; 0: none known.
inline int detect_generation() {
  uint32_t family = 0;
  std::size_t size = sizeof(family);
  if (sysctlbyname("hw.cpufamily", &family, &size, nullptr, 0) != 0) return 0;
  switch (family) {
    case 0x1b588bb3u:  // FIRESTORM_ICESTORM: M1 family
      return 1;
    case 0xda33d83du:  // BLIZZARD_AVALANCHE: M2 family
      return 2;
    case 0xfa33415eu:  // IBIZA: M3
    case 0x5f4dea93u:  // LOBOS: M3 Pro
    case 0x72015832u:  // PALMA: M3 Max
      return 3;
    case 0x6f5129acu:  // DONAN: M4
    case 0x17d5b93au:  // BRAVA: M4 Pro, M4 Max
      return 4;
    default:
      return 0;
  }
}
inline int generation() {
  static const int g = detect_generation();
  return g;
}
// The kernels use the quad loads of the M2 and later.
inline bool usable() { return generation() >= 2; }

// NEON crossover of the GEMM, rows cols depth sizeof(Scalar) >= this, measured on the M2 and M4 Pro (M3: M2's).
inline double gemm_min_work_bytes() {
#ifdef EIGEN_APPLE_AMX_MIN_WORK_BYTES
  return EIGEN_APPLE_AMX_MIN_WORK_BYTES;
#else
  return generation() >= 4 ? double(1 << 17) : double(1 << 18);
#endif
}

// Units a product spreads over, one per performance cluster; 0 when the kernels cannot run.
inline int units() {
  static const int u = usable() ? numext::maxi(1, queryPerformanceClusters()) : 0;
  return u;
}

// Grain of the split of a product over units: a multiple of both sides of every C block.
constexpr Index kSplitGrain = 32;

// C block of the row-major core problem: mr x nr in the Z accumulators.
template <typename T>
struct block;
template <>
struct block<float> {
  static constexpr int mr = 32, nr = 32;
};
template <>
struct block<double> {
  static constexpr int mr = 16, nr = 32;
};

EIGEN_ALWAYS_INLINE void prefetch_l2(const void* p, Index bytes) {
  for (Index l = 0; l < bytes; l += 128) __builtin_prefetch(static_cast<const char*>(p) + l, 0, 2);
}

// Packed depth is zero padded to a multiple of 4; a B panel is followed by 256 bytes so panels do not share L1 sets.
constexpr Index kDepthStep = 4;
template <typename T>
EIGEN_ALWAYS_INLINE Index b_stride(Index kp) {
  return kp * Index(block<T>::nr) + Index(256 / sizeof(T));
}
EIGEN_ALWAYS_INLINE Index round_up(Index x, Index m) { return (x + m - 1) / m * m; }

// fp32 C row 16p + j is in Z rows 4j + 2p + {0, 1}; fp64 C row 8p + j in Z rows 8j + 4p + {0, ..., 3}.
template <bool Store>
EIGEN_ALWAYS_INLINE void z_io(float* C, Index ldc) {
  const bool pair = ((reinterpret_cast<uintptr_t>(C) | uintptr_t(ldc * Index(sizeof(float)))) & 127) == 0;
  for (int p = 0; p < 2; ++p)
    for (int j = 0; j < 16; ++j) {
      float* row = C + Index(16 * p + j) * ldc;
      const int z = 4 * j + 2 * p;
      if (pair) {
        if (Store)
          stz(zrow(row, z, true));
        else
          ldz(zrow(row, z, true));
      } else if (Store) {
        stz(zrow(row, z));
        stz(zrow(row + 16, z + 1));
      } else {
        ldz(zrow(row, z));
        ldz(zrow(row + 16, z + 1));
      }
    }
}
template <bool Store>
EIGEN_ALWAYS_INLINE void z_io(double* C, Index ldc) {
  const bool pair = ((reinterpret_cast<uintptr_t>(C) | uintptr_t(ldc * Index(sizeof(double)))) & 127) == 0;
  for (int p = 0; p < 2; ++p)
    for (int j = 0; j < 8; ++j) {
      double* row = C + Index(8 * p + j) * ldc;
      const int z = 8 * j + 4 * p;
      if (pair) {
        if (Store) {
          stz(zrow(row, z, true));
          stz(zrow(row + 16, z + 2, true));
        } else {
          ldz(zrow(row, z, true));
          ldz(zrow(row + 16, z + 2, true));
        }
      } else {
        for (int q = 0; q < 4; ++q)
          if (Store)
            stz(zrow(row + 8 * q, z + q));
          else
            ldz(zrow(row + 8 * q, z + q));
      }
    }
}

// B panel read from its row-major source: rows from kb on read zeros.
template <typename T>
struct source_panel {
  const T* src;
  Index ldb;
  Index kb;
};
alignas(256) static const float kZeros[64] = {};
template <typename T>
EIGEN_ALWAYS_INLINE void source_rows(const source_panel<T>& s, Index r0, int nrows) {
  constexpr int kRowBytes = block<T>::nr * int(sizeof(T));
  for (int u = 0; u < nrows; ++u) {
    const Index r = r0 + u;
    const char* p = r < s.kb ? reinterpret_cast<const char*>(s.src + r * s.ldb) : reinterpret_cast<const char*>(kZeros);
    if (r + 32 < s.kb) prefetch_l2(s.src + (r + 32) * s.ldb, kRowBytes);
    for (int q = 0; q < kRowBytes / 64; ++q) ldx(xy(p + 64 * q, u * kRowBytes / 64 + q));
  }
}

// Four depth steps: 512-byte slabs of packed A (Y) and B (X).
template <typename T>
struct step4;
template <>
struct step4<float> {
  template <bool Source>
  static EIGEN_ALWAYS_INLINE void run(const char* a, const char* b, const source_panel<float>* s, Index k) {
    if (Source) {
      source_rows(*s, k, 4);
    } else {
      ldx(xy(b, 0, kQuad));
      ldx(xy(b + 256, 4, kQuad));
    }
    ldy(xy(a, 0, kQuad));
    ldy(xy(a + 256, 4, kQuad));
    for (int d = 0; d < 4; ++d) {
      fma<float>(fma_op(0, 128 * d, 128 * d));
      fma<float>(fma_op(1, 128 * d + 64, 128 * d));
      fma<float>(fma_op(2, 128 * d, 128 * d + 64));
      fma<float>(fma_op(3, 128 * d + 64, 128 * d + 64));
    }
  }
};
template <>
struct step4<double> {
  static EIGEN_ALWAYS_INLINE void two(int dx, int dy) {
    for (int h = 0; h < 2; ++h)
      for (int q = 0; q < 4; ++q) fma<double>(fma_op(4 * h + q, 256 * dx + 64 * q, 128 * dy + 64 * h));
  }
  template <bool Source>
  static EIGEN_ALWAYS_INLINE void run(const char* a, const char* b, const source_panel<double>* s, Index k) {
    ldy(xy(a, 0, kQuad));
    ldy(xy(a + 256, 4, kQuad));
    if (Source) {
      source_rows(*s, k, 2);
    } else {
      ldx(xy(b, 0, kQuad));
      ldx(xy(b + 256, 4, kQuad));
    }
    two(0, 0);
    two(1, 1);
    if (Source) {
      source_rows(*s, k + 2, 2);
    } else {
      ldx(xy(b + 512, 0, kQuad));
      ldx(xy(b + 768, 4, kQuad));
    }
    two(0, 2);
    two(1, 3);
  }
};

// mr x nr block of C += A panel (kp x mr) * B panel (kp x nr); pfa: next A panel to prefetch.
template <typename T, bool Source>
EIGEN_ALWAYS_INLINE void kernel_body(Index kp, const T* Ap, const T* Bp, T* C, Index ldc, const source_panel<T>* s,
                                     const char* pfa) {
  constexpr Index kSlabA = 4 * block<T>::mr * sizeof(T), kSlabB = 4 * block<T>::nr * sizeof(T);
  const char* a = reinterpret_cast<const char*>(Ap);
  const char* b = reinterpret_cast<const char*>(Bp);
  z_io<false>(C, ldc);
  for (Index k = 0; k < kp; k += 4, a += kSlabA, b += kSlabB) {
    step4<T>::template run<Source>(a, b, s, k);
    if (pfa) prefetch_l2(pfa + k * Index(block<T>::mr * sizeof(T)), kSlabA);
  }
  z_io<true>(C, ldc);
}
template <typename T>
EIGEN_DONT_INLINE void kernel(Index kp, const T* Ap, const T* Bp, T* C, Index ldc, const source_panel<T>* s,
                              const char* pfa) {
  if (s)
    kernel_body<T, true>(kp, Ap, Bp, C, ldc, s, pfa);
  else
    kernel_body<T, false>(kp, Ap, Bp, C, ldc, s, pfa);
}
// Partial block through an aligned scratch tile.
template <typename T>
EIGEN_DONT_INLINE void kernel_edge(Index kp, const T* Ap, const T* Bp, T* C, Index ldc, Index rows, Index cols,
                                   const source_panel<T>* s) {
  constexpr int kMr = block<T>::mr, kNr = block<T>::nr;
  alignas(128) T t[kMr * kNr];
  for (Index r = 0; r < rows; ++r) std::memcpy(t + r * kNr, C + r * ldc, std::size_t(cols) * sizeof(T));
  kernel<T>(kp, Ap, Bp, t, kNr, s, nullptr);
  for (Index r = 0; r < rows; ++r) std::memcpy(C + r * ldc, t + r * kNr, std::size_t(cols) * sizeof(T));
}

// 4 x 4 transposition: rows s, s + ld, ... to rows d, d + dstride, ..., scaled by alpha when scale.
EIGEN_ALWAYS_INLINE void transpose4(const float* s, Index ld, float* d, Index dstride, float alpha, bool scale) {
  float32x4_t r0 = vld1q_f32(s), r1 = vld1q_f32(s + ld), r2 = vld1q_f32(s + 2 * ld), r3 = vld1q_f32(s + 3 * ld);
  if (scale) {
    r0 = vmulq_n_f32(r0, alpha);
    r1 = vmulq_n_f32(r1, alpha);
    r2 = vmulq_n_f32(r2, alpha);
    r3 = vmulq_n_f32(r3, alpha);
  }
  const float32x4x2_t t01 = vtrnq_f32(r0, r1), t23 = vtrnq_f32(r2, r3);
  vst1q_f32(d, vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0])));
  vst1q_f32(d + dstride, vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1])));
  vst1q_f32(d + 2 * dstride, vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0])));
  vst1q_f32(d + 3 * dstride, vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1])));
}
EIGEN_ALWAYS_INLINE void transpose4(const double* s, Index ld, double* d, Index dstride, double alpha, bool scale) {
  for (int h = 0; h < 2; ++h)
    for (int g = 0; g < 2; ++g) {
      float64x2_t r0 = vld1q_f64(s + Index(2 * h) * ld + 2 * g), r1 = vld1q_f64(s + Index(2 * h + 1) * ld + 2 * g);
      if (scale) {
        r0 = vmulq_n_f64(r0, alpha);
        r1 = vmulq_n_f64(r1, alpha);
      }
      vst1q_f64(d + Index(2 * g) * dstride + 2 * h, vzip1q_f64(r0, r1));
      vst1q_f64(d + Index(2 * g + 1) * dstride + 2 * h, vzip2q_f64(r0, r1));
    }
}

// 32 depth steps of a full A panel through Z: rows in with ldz, each depth step out as a Z column (extrv) to Y.
template <typename T>
EIGEN_ALWAYS_INLINE void transpose_panel_amx(const T* A, Index lda, T* dst) {
  constexpr int kLanes = 64 / int(sizeof(T)), kTiles = kLanes == 16 ? 4 : 8, kMr = 2 * kLanes;
  constexpr uint64_t kExtractColumn = (uint64_t(1) << 63) | (uint64_t(kLanes == 16 ? 8 : 1) << 11) |
                                      (uint64_t(1) << 26) | (uint64_t(1) << 10);
  for (int m = 0; m < kMr; ++m)  // row m, depth group g to tile 2g + m / kLanes
    for (int g = 0; g < 32 / kLanes; ++g)
      ldz(zrow(A + m * lda + g * kLanes, kTiles * (m % kLanes) + 2 * g + m / kLanes));
  for (int k = 0; k < 32; ++k) {
    const int t = 2 * (k / kLanes), c = k % kLanes, y = 128 * (k & 3);
    extry(kExtractColumn | uint64_t(c * kTiles + t) << 20 | uint64_t(y));
    extry(kExtractColumn | uint64_t(c * kTiles + t + 1) << 20 | uint64_t(y + 64));
    sty(xy(dst + k * kMr, 2 * (k & 3), kPair));
  }
}

// Row-major A block (mb x kb) to panels Ap[panel][k][r] of mr rows, alpha applied, zero padded.
template <typename T>
EIGEN_DONT_INLINE void pack_a_rows(Index mb, Index kb, Index kp, const T* A, Index lda, T alpha, T* Ap) {
  constexpr Index kMr = block<T>::mr, kLine = 128 / sizeof(T), kAheadChunks = 4;
  const bool scale = alpha != T(1);
  for (Index p0 = 0; p0 < mb; p0 += kMr) {
    const Index rows = numext::mini(kMr, mb - p0), r4 = rows / 4 * 4, k4 = kb / 4 * 4;
    T* dst = Ap + p0 * kp;
    Index kz = 0;
    if (rows == kMr && !scale)
      for (; kz + 32 <= kb; kz += 32) {
        // AMX waits in order on each miss: prefetch whole chunks ahead, into the next panel too.
        Index kf = kz + 32 * kAheadChunks, pf0 = p0;
        if (kf + 32 > kb) {
          kf -= kb / 32 * 32;
          pf0 += kMr;
        }
        if (pf0 + kMr <= mb && kf >= 0 && kf + 32 <= kb)
          for (Index r = 0; r < kMr; ++r) prefetch_l2(A + (pf0 + r) * lda + kf, 32 * Index(sizeof(T)));
        transpose_panel_amx<T>(A + p0 * lda + kz, lda, dst + kz * kMr);
      }
    for (Index k0 = kz; k0 < k4; k0 += kLine) {
      const Index k1 = numext::mini(k4, k0 + kLine);
      for (Index r = 0; r < r4; r += 4) {
        const T* src = A + (p0 + r) * lda;
        if (k0 + 2 * kLine < kb)
          for (Index u = 0; u < 4; ++u) __builtin_prefetch(src + u * lda + k0 + 2 * kLine, 0, 3);
        for (Index k = k0; k < k1; k += 4) transpose4(src + k, lda, dst + k * kMr + r, kMr, alpha, scale);
      }
    }
    for (Index r = 0; r < r4; ++r)
      for (Index k = k4; k < kb; ++k) dst[k * kMr + r] = alpha * A[(p0 + r) * lda + k];
    for (Index r = r4; r < rows; ++r)
      for (Index k = 0; k < kb; ++k) dst[k * kMr + r] = alpha * A[(p0 + r) * lda + k];
    if (rows < kMr)
      for (Index k = 0; k < kb; ++k) std::memset(dst + k * kMr + rows, 0, std::size_t(kMr - rows) * sizeof(T));
    std::memset(dst + kb * kMr, 0, std::size_t((kp - kb) * kMr) * sizeof(T));
  }
}

// Column-major A block to the same panels: each depth step is a contiguous copy.
template <typename T>
EIGEN_DONT_INLINE void pack_a_cols(Index mb, Index kb, Index kp, const T* A, Index lda, T alpha, T* Ap) {
  constexpr Index kMr = block<T>::mr;
  for (Index p0 = 0; p0 < mb; p0 += kMr) {
    const Index rows = numext::mini(kMr, mb - p0);
    T* dst = Ap + p0 * kp;
    for (Index k = 0; k < kb; ++k) {
      const T* s = A + k * lda + p0;
      T* d = dst + k * kMr;
      for (Index r = 0; r < rows; ++r) d[r] = alpha * s[r];
      for (Index r = rows; r < kMr; ++r) d[r] = T(0);
    }
    std::memset(dst + kb * kMr, 0, std::size_t((kp - kb) * kMr) * sizeof(T));
  }
}

// Row-major B block (kb x nb) to panels of nr columns in source row order, so that each row streams contiguously.
template <typename T>
EIGEN_DONT_INLINE void pack_b_rows(Index kb, Index kp, Index nb, const T* B, Index ldb, T* Bp) {
  constexpr Index kNr = block<T>::nr, kFloats = kNr * sizeof(T) / 4;
  const Index bs = b_stride<T>(kp), full = nb / kNr * kNr;
  for (Index k = 0; k < kb; ++k) {
    const T* s = B + k * ldb;
    if (k + 8 < kb) prefetch_l2(s + 8 * ldb, nb * Index(sizeof(T)));
    for (Index jj = 0; jj < full; jj += kNr) {
      const float* sf = reinterpret_cast<const float*>(s + jj);
      float* d = reinterpret_cast<float*>(Bp + jj / kNr * bs + k * kNr);
      for (Index c = 0; c < kFloats; c += 16) vst1q_f32_x4(d + c, vld1q_f32_x4(sf + c));
    }
    if (full < nb) {
      T* d = Bp + full / kNr * bs + k * kNr;
      std::memcpy(d, s + full, std::size_t(nb - full) * sizeof(T));
      std::memset(d + (nb - full), 0, std::size_t(kNr - (nb - full)) * sizeof(T));
    }
  }
  for (Index jj = 0; jj < nb; jj += kNr)
    std::memset(Bp + jj / kNr * bs + kb * kNr, 0, std::size_t((kp - kb) * kNr) * sizeof(T));
}

// Column-major B block to the same panels by 4 x 4 transpositions.
template <typename T>
EIGEN_DONT_INLINE void pack_b_cols(Index kb, Index kp, Index nb, const T* B, Index ldb, T* Bp) {
  constexpr Index kNr = block<T>::nr;
  const Index bs = b_stride<T>(kp), k4 = kb / 4 * 4;
  for (Index jj = 0; jj < nb; jj += kNr) {
    const Index cols = numext::mini(kNr, nb - jj), c4 = cols / 4 * 4;
    T* dst = Bp + jj / kNr * bs;
    for (Index c = 0; c < c4; c += 4) {
      const T* src = B + (jj + c) * ldb;
      for (Index k = 0; k < k4; k += 4) transpose4(src + k, ldb, dst + k * kNr + c, kNr, T(1), false);
      for (Index k = k4; k < kb; ++k)
        for (Index u = 0; u < 4; ++u) dst[k * kNr + c + u] = src[u * ldb + k];
    }
    for (Index c = c4; c < cols; ++c)
      for (Index k = 0; k < kb; ++k) dst[k * kNr + c] = B[(jj + c) * ldb + k];
    if (cols < kNr)
      for (Index k = 0; k < kb; ++k) std::memset(dst + k * kNr + cols, 0, std::size_t(kNr - cols) * sizeof(T));
    std::memset(dst + kb * kNr, 0, std::size_t((kp - kb) * kNr) * sizeof(T));
  }
}

// Block near `size` (a multiple of step) that splits total into equal blocks.
EIGEN_ALWAYS_INLINE Index balance(Index total, Index size, Index step) {
  if (size >= total) return round_up(total, step);
  const Index blocks = (total + size - 1) / size;
  return numext::mini(round_up(size, step), round_up((total + blocks - 1) / blocks, step));
}

// Row-major core problem C (M x N, ldc) += alpha A (M x K) B (K x N); A and B row- or column-major.
template <typename T>
struct problem {
  Index M, N, K;
  T alpha;
  const T* A;
  Index lda;
  bool a_row_major;
  const T* B;
  Index ldb;
  bool b_row_major;
  T* C;
  Index ldc;
};

// Blocking measured on the M2: depth of about 4 KB per row, A block up to 8 MB, B block the rest of 6 MB.
template <typename T>
void default_blocking(Index M, Index N, Index K, Index& mc, Index& nc, Index& kc) {
  constexpr Index kMr = block<T>::mr, kNr = block<T>::nr, kBytes = sizeof(T);
  kc = balance(K, 4096 / kBytes, kDepthStep);
  mc = balance(M, numext::maxi(kMr, (Index(8) << 20) / (kc * kBytes) / kMr * kMr), kMr);
  const Index rest = numext::maxi(Index(0), (Index(6) << 20) - mc * kc * kBytes);
  nc = balance(N, numext::maxi(8 * kNr, rest / (kc * kBytes) / kNr * kNr), kNr);
}

// mc and nc are multiples of mr and nr, kc of kDepthStep.
template <typename T>
EIGEN_DONT_INLINE void run(const problem<T>& p, Index mc, Index nc, Index kc) {
  constexpr Index kMr = block<T>::mr, kNr = block<T>::nr, kBytes = sizeof(T);
  eigen_internal_assert(mc % kMr == 0 && nc % kNr == 0 && kc % kDepthStep == 0);
  const Index M = p.M, N = p.N, K = p.K;
  // B is read from its source when all of it takes at most 2 MB (fp64 always, fp32 for M <= 128).
  const bool b_source = p.b_row_major && K * N * kBytes <= (Index(2) << 20) && (kBytes == 8 || M <= 128);

  const std::size_t a_size = std::size_t(mc * kc), b_size = std::size_t(nc / kNr * b_stride<T>(kc));
  const std::size_t a_bytes = (a_size * sizeof(T) + 255) & ~std::size_t(255);
  void* buffer = handmade_aligned_malloc(a_bytes + b_size * sizeof(T), 256);
  if (buffer == nullptr) throw_std_bad_alloc();
  T* Ap = static_cast<T*>(buffer);
  T* Bp = static_cast<T*>(static_cast<void*>(static_cast<char*>(buffer) + a_bytes));

  set();
  for (Index i = 0; i < M; i += mc) {
    const Index mb = numext::mini(mc, M - i);
    for (Index k = 0; k < K; k += kc) {
      const Index kb = numext::mini(kc, K - k), kp = round_up(kb, kDepthStep), bs = b_stride<T>(kp);
      if (p.a_row_major)
        pack_a_rows<T>(mb, kb, kp, p.A + i * p.lda + k, p.lda, p.alpha, Ap);
      else
        pack_a_cols<T>(mb, kb, kp, p.A + k * p.lda + i, p.lda, p.alpha, Ap);
      for (Index j = 0; j < N; j += nc) {
        const Index nb = numext::mini(nc, N - j);
        const T* Bs = p.b_row_major ? p.B + k * p.ldb + j : p.B + j * p.ldb + k;
        if (!p.b_row_major)
          pack_b_cols<T>(kb, kp, nb, Bs, p.ldb, Bp);
        else if (!b_source)
          pack_b_rows<T>(kb, kp, nb, Bs, p.ldb, Bp);
        else if (nb % kNr)  // the partial panel is packed even when B is read from its source
          pack_b_rows<T>(kb, kp, nb % kNr, Bs + nb / kNr * kNr, p.ldb, Bp + nb / kNr * bs);
        T* Cb = p.C + i * p.ldc + j;
        for (Index ii = 0; ii < mb; ii += kMr) {
          const Index rows = numext::mini(kMr, mb - ii);
          const T* A_panel = Ap + ii * kp;
          for (Index jj = 0; jj < nb; jj += kNr) {
            const Index cols = numext::mini(kNr, nb - jj);
            T* Ct = Cb + ii * p.ldc + jj;
            const T* B_panel = Bp + jj / kNr * bs;
            // The next C tile towards L2: the tile start otherwise waits for ldz.
            if (jj + kNr < nb || ii + kMr < mb) {
              const T* next = jj + kNr < nb ? Ct + kNr : Cb + (ii + kMr) * p.ldc;
              for (Index r = 0; r < kMr; ++r) prefetch_l2(next + r * p.ldc, kNr * kBytes);
            }
            const source_panel<T> s{Bs + jj, p.ldb, kb};
            const source_panel<T>* source = b_source && cols == kNr ? &s : nullptr;
            if (rows < kMr || cols < kNr) {
              kernel_edge<T>(kp, A_panel, B_panel, Ct, p.ldc, rows, cols, source);
              continue;
            }
            // The next A panel of this block (the first again for the next j) at the first column.
            const T* next_a = jj != 0 ? nullptr : ii + kMr < mb ? A_panel + kMr * kp : (j + nc < N ? Ap : nullptr);
            kernel<T>(kp, A_panel, B_panel, Ct, p.ldc, source, reinterpret_cast<const char*>(next_a));
          }
        }
      }
    }
  }
  clr();
  handmade_aligned_free(buffer);
}

}  // namespace apple_amx

// Real float and double products run on AMX; every other scalar pair keeps the generic kernel.
template <typename LhsScalar, typename RhsScalar>
struct apple_amx_pair
    : bool_constant<std::is_same<LhsScalar, RhsScalar>::value &&
                    (std::is_same<LhsScalar, float>::value || std::is_same<LhsScalar, double>::value)> {};

// Column-major res (rows x cols) += alpha lhs rhs as the row-major core problem res^T += alpha rhs^T lhs^T.
template <int LhsStorageOrder, int RhsStorageOrder, typename Scalar, typename Index>
bool apple_amx_gemm(Index rows, Index cols, Index depth, const Scalar* lhs, Index lhsStride, const Scalar* rhs,
                    Index rhsStride, Scalar* res, Index resIncr, Index resStride, Scalar alpha) {
  const double area = double(rows) * double(cols);
  if (resIncr != 1 || area < EIGEN_APPLE_AMX_MIN_RESULT_SIZE ||
      !apple_amx::usable() || area * double(depth) * sizeof(Scalar) < apple_amx::gemm_min_work_bytes())
    return false;
  apple_amx::problem<Scalar> p;
  p.M = cols;
  p.N = rows;
  p.K = depth;
  p.alpha = alpha;
  p.A = rhs;
  p.lda = rhsStride;
  p.a_row_major = RhsStorageOrder == ColMajor;
  p.B = lhs;
  p.ldb = lhsStride;
  p.b_row_major = LhsStorageOrder == ColMajor;
  p.C = res;
  p.ldc = resStride;
  Index mc, nc, kc;
  apple_amx::default_blocking<Scalar>(p.M, p.N, p.K, mc, nc, kc);
  apple_amx::run(p, mc, nc, kc);
  return true;
}
template <int LhsStorageOrder, int RhsStorageOrder, typename Scalar, typename Index>
EIGEN_ALWAYS_INLINE bool apple_amx_run(std::true_type, Index rows, Index cols, Index depth, const Scalar* lhs,
                                       Index lhsStride, const Scalar* rhs, Index rhsStride, Scalar* res, Index resIncr,
                                       Index resStride, Scalar alpha) {
  return apple_amx_gemm<LhsStorageOrder, RhsStorageOrder>(rows, cols, depth, lhs, lhsStride, rhs, rhsStride, res,
                                                          resIncr, resStride, alpha);
}
template <int LhsStorageOrder, int RhsStorageOrder, typename LhsScalar, typename RhsScalar, typename ResScalar,
          typename Index>
EIGEN_ALWAYS_INLINE bool apple_amx_run(std::false_type, Index, Index, Index, const LhsScalar*, Index, const RhsScalar*,
                                       Index, ResScalar*, Index, Index, ResScalar) {
  return false;
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_APPLE_AMX_GENERALMATRIXMATRIX_H
