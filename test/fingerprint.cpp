// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2026 Rasmus Munk Larsen <rmlarsen@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <cstdint>
#include <cstring>
#include <set>
#include <utility>
#include <vector>

#include <Eigen/src/Core/util/Fingerprint.h>

using Eigen::internal::Fingerprint128;
using Eigen::internal::Fingerprint128Builder;

static std::pair<std::uint64_t, std::uint64_t> as_pair(const Fingerprint128& fp) {
  return std::make_pair(fp.lane0, fp.lane1);
}

static std::vector<unsigned char> make_buffer(std::size_t n) {
  std::vector<unsigned char> buf(n);
  for (std::size_t i = 0; i < n; ++i) buf[i] = static_cast<unsigned char>(i * 131 + 7);
  return buf;
}

static void test_determinism_and_equality() {
  const std::vector<unsigned char> buf = make_buffer(1000);
  const Fingerprint128 a = Fingerprint128::digest(buf.data(), buf.size());
  const Fingerprint128 b = Fingerprint128::digest(buf.data(), buf.size());
  VERIFY(a == b);
  VERIFY(!(a != b));

  // digest() is a single-append builder.
  Fingerprint128Builder builder;
  builder.append(buf.data(), buf.size());
  VERIFY(builder.finalize() == a);
  // finalize() does not consume the state.
  VERIFY(builder.finalize() == a);

  const Fingerprint128 c = Fingerprint128::digest(buf.data(), buf.size() - 1);
  VERIFY(a != c);
  VERIFY(!(a == c));
}

// Every single-bit flip yields a fingerprint distinct from the original and
// from every other flip. The buffer length exercises full 16-byte blocks
// plus a partial tail.
static void test_bit_flips_are_distinct() {
  std::vector<unsigned char> buf = make_buffer(40);
  std::set<std::pair<std::uint64_t, std::uint64_t>> seen;
  VERIFY(seen.insert(as_pair(Fingerprint128::digest(buf.data(), buf.size()))).second);
  for (std::size_t byte = 0; byte < buf.size(); ++byte) {
    for (int bit = 0; bit < 8; ++bit) {
      buf[byte] ^= 1u << bit;
      VERIFY(seen.insert(as_pair(Fingerprint128::digest(buf.data(), buf.size()))).second);
      buf[byte] ^= 1u << bit;
    }
  }
}

// All-zero buffers of every length up to several blocks are pairwise
// distinct: the length frame separates the zero-padded tail from genuine
// zero bytes.
static void test_zero_buffer_lengths_are_distinct() {
  const std::vector<unsigned char> zeros(64, 0);
  std::set<std::pair<std::uint64_t, std::uint64_t>> seen;
  for (std::size_t len = 0; len <= zeros.size(); ++len) {
    VERIFY(seen.insert(as_pair(Fingerprint128::digest(zeros.data(), len))).second);
  }
}

// Chunk boundaries are significant: the same concatenated bytes split
// differently across append() calls give different fingerprints, and empty
// appends are not no-ops. This is what lets a caller fingerprint several
// arrays without separator bytes.
static void test_append_boundaries_are_framed() {
  const unsigned char data[3] = {1, 2, 3};
  std::set<std::pair<std::uint64_t, std::uint64_t>> seen;
  // {3, 0} and {0, 3} check that leading and trailing empty appends are
  // significant too.
  static constexpr std::size_t splits[][2] = {{3, 0}, {0, 3}, {1, 2}, {2, 1}};
  for (const auto& split : splits) {
    Fingerprint128Builder builder;
    builder.append(data, split[0]);
    builder.append(data + split[0], split[1]);
    VERIFY(seen.insert(as_pair(builder.finalize())).second);
  }
  // All two-append variants differ from the unsplit single append.
  VERIFY(seen.insert(as_pair(Fingerprint128::digest(data, 3))).second);
}

// Regression for the two-flip differential a bare xor-rotate-multiply step
// admits: a high-bit difference passes through the lane multiply intact, so
// without the word premix, flipping bit b in one word and bit (b + rot) mod
// 64 in the next word of the same stripe canceled deterministically for
// b + rot = 63, reproducing the unmodified fingerprint.
static void test_adjacent_word_flip_pairs_are_distinct() {
  std::vector<unsigned char> buf = make_buffer(48);
  const auto flip = [&buf](int stripe, int block, int bit) {
    buf[block * 16 + stripe * 8 + bit / 8] ^= 1u << (bit % 8);
  };
  std::set<std::pair<std::uint64_t, std::uint64_t>> seen;
  VERIFY(seen.insert(as_pair(Fingerprint128::digest(buf.data(), buf.size()))).second);
  for (int stripe = 0; stripe < 2; ++stripe) {
    const int rot = stripe == 0 ? 27 : 31;
    for (int block = 0; block + 1 < 3; ++block) {
      for (int bit = 0; bit < 64; ++bit) {
        const int image = (bit + rot) % 64;
        flip(stripe, block, bit);
        flip(stripe, block + 1, image);
        VERIFY(seen.insert(as_pair(Fingerprint128::digest(buf.data(), buf.size()))).second);
        flip(stripe, block + 1, image);
        flip(stripe, block, bit);
      }
    }
  }
}

// The fingerprint is a function of the bytes, not of the buffer's address
// or alignment.
static void test_alignment_invariance() {
  const std::vector<unsigned char> buf = make_buffer(40);
  const Fingerprint128 reference = Fingerprint128::digest(buf.data(), buf.size());
  std::vector<unsigned char> arena(buf.size() + 16);
  for (std::size_t offset : {1, 3, 7, 9, 15}) {
    std::memcpy(arena.data() + offset, buf.data(), buf.size());
    VERIFY(Fingerprint128::digest(arena.data() + offset, buf.size()) == reference);
  }
}

// Documents the striped construction: bytes [0,8) of each 16-byte block feed
// lane0 and bytes [8,16) feed lane1, so a single-bit flip avalanches its own
// lane (roughly half its 64 bits) and leaves the other lane untouched.
static void test_single_lane_avalanche() {
  std::vector<unsigned char> buf = make_buffer(32);
  const Fingerprint128 reference = Fingerprint128::digest(buf.data(), buf.size());
  for (std::size_t byte = 0; byte < buf.size(); ++byte) {
    const bool feeds_lane0 = (byte / 8) % 2 == 0;
    for (int bit = 0; bit < 8; ++bit) {
      buf[byte] ^= 1u << bit;
      const Fingerprint128 flipped = Fingerprint128::digest(buf.data(), buf.size());
      buf[byte] ^= 1u << bit;
      const int distance0 = Eigen::internal::popcount(reference.lane0 ^ flipped.lane0);
      const int distance1 = Eigen::internal::popcount(reference.lane1 ^ flipped.lane1);
      VERIFY_IS_EQUAL(feeds_lane0 ? distance1 : distance0, 0);
      const int affected = feeds_lane0 ? distance0 : distance1;
      VERIFY(affected >= 12 && affected <= 52);
    }
  }
}

EIGEN_DECLARE_TEST(fingerprint) {
  CALL_SUBTEST(test_determinism_and_equality());
  CALL_SUBTEST(test_bit_flips_are_distinct());
  CALL_SUBTEST(test_zero_buffer_lengths_are_distinct());
  CALL_SUBTEST(test_append_boundaries_are_framed());
  CALL_SUBTEST(test_adjacent_word_flip_pairs_are_distinct());
  CALL_SUBTEST(test_alignment_invariance());
  CALL_SUBTEST(test_single_lane_avalanche());
}
