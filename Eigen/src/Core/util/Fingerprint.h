// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#ifndef EIGEN_FINGERPRINT_H
#define EIGEN_FINGERPRINT_H

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "../InternalHeaderCheck.h"

namespace Eigen {
namespace internal {

// 128-bit content fingerprint for cache-invalidation keys: detects content
// changes that pointer identity cannot, such as an array rewritten in place
// or a new array landing in a recycled allocation.
//
// Fingerprint equality is trusted without comparing the fingerprinted bytes
// (unlike a hash-table hash, where key comparison resolves collisions), so
// the collision probability is the correctness bar. Construction: two
// xor-rotate-multiply lanes over alternating premixed 64-bit words, each
// finalized with the splitmix64 avalanche (constants from Vigna's
// public-domain splitmix64; implemented independently). A difference
// confined to one 8-byte stripe must collide its lane (~2^-64 per input
// pair); differences spanning both stripes must collide both lanes
// (~2^-128). Non-cryptographic: this guards against accidental staleness,
// not adversarial inputs.
//
// In-process use only. Values depend on endianness and may change between
// Eigen versions; never persist them.
struct Fingerprint128 {
  std::uint64_t lane0;
  std::uint64_t lane1;

  friend bool operator==(const Fingerprint128& a, const Fingerprint128& b) {
    return a.lane0 == b.lane0 && a.lane1 == b.lane1;
  }
  friend bool operator!=(const Fingerprint128& a, const Fingerprint128& b) { return !(a == b); }
};

// Streaming builder: append() any number of byte ranges, then finalize().
// Each append is length-framed, so chunk boundaries are significant:
// append("ab"), append("c") differs from append("a"), append("bc") and from
// append("abc"), and appending an empty range is not a no-op.
class Fingerprint128Builder {
 public:
  void append(const void* data, std::size_t bytes) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = bytes / 16; i > 0; --i, p += 16) {
      std::uint64_t w0, w1;
      std::memcpy(&w0, p, 8);
      std::memcpy(&w1, p + 8, 8);
      step(w0, w1);
    }
    const std::size_t rem = bytes % 16;
    if (rem > 0) {
      unsigned char tail[16] = {};
      std::memcpy(tail, p, rem);
      std::uint64_t w0, w1;
      std::memcpy(&w0, tail, 8);
      std::memcpy(&w1, tail + 8, 8);
      step(w0, w1);
    }
    // Length frame: separates chunk boundaries and distinguishes the
    // zero-padded tail from genuine zero bytes.
    step(static_cast<std::uint64_t>(bytes), static_cast<std::uint64_t>(bytes));
  }

  // The avalanche is a bijection, so equal fingerprints imply equal lane
  // states. Non-destructive: appending after finalize() is valid.
  Fingerprint128 finalize() const { return Fingerprint128{avalanche(h0_), avalanche(h1_)}; }

 private:
  static std::uint64_t rotl(std::uint64_t x, unsigned r) { return (x << r) | (x >> (64 - r)); }

  static std::uint64_t avalanche(std::uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
  }

  // Diffuses a word before it is injected into a lane (first half of the
  // splitmix64 avalanche). Without it, a high-bit difference passes through
  // the lane multiply intact (2^63 * k = 2^63 mod 2^64 for any odd k), so a
  // one-bit change in one word could be canceled deterministically by an
  // aligned one-bit change in the next word of the same stripe. The premix
  // spreads any single-word difference across many bits first, and is
  // independent of the serial lane state, so it pipelines into multiplier
  // slots the lane chain leaves idle.
  static std::uint64_t premix(std::uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    return x;
  }

  void step(std::uint64_t w0, std::uint64_t w1) {
    // Odd multipliers keep the per-word step a bijection of the lane state;
    // the rotation moves high-bit differences into carry-propagating
    // positions before the multiply. The lanes use unrelated multipliers so
    // a both-stripe difference must collide both independently.
    constexpr std::uint64_t kMul0 = 0x9e3779b97f4a7c15ULL;  // 2^64 / golden ratio
    constexpr std::uint64_t kMul1 = 0xff51afd7ed558ccdULL;  // MurmurHash3 finalizer constant (public domain)
    h0_ = rotl(h0_ ^ premix(w0), 27) * kMul0;
    h1_ = rotl(h1_ ^ premix(w1), 31) * kMul1;
  }

  // Distinct arbitrary seeds per lane (first 128 fractional bits of pi).
  std::uint64_t h0_ = 0x243f6a8885a308d3ULL;
  std::uint64_t h1_ = 0x13198a2e03707344ULL;
};

inline Fingerprint128 fingerprint128(const void* data, std::size_t bytes) {
  Fingerprint128Builder builder;
  builder.append(data, bytes);
  return builder.finalize();
}

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_FINGERPRINT_H
