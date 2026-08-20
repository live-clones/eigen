// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

#include "main.h"

#include <limits>

#include <Eigen/Eigenvalues>

template <typename RealScalar>
void wilkinson_shift_avoids_squared_subdiagonal() {
  using std::sqrt;

  volatile RealScalar normal_min = (std::numeric_limits<RealScalar>::min)();
  const RealScalar subdiag_before = sqrt(normal_min) / RealScalar(16);
  for (int td_sign = -1; td_sign <= 1; td_sign += 2) {
    for (int subdiag_sign = -1; subdiag_sign <= 1; subdiag_sign += 2) {
      const RealScalar td = RealScalar(td_sign) * subdiag_before * NumTraits<RealScalar>::epsilon();
      RealScalar diag[2] = {td, -td};
      RealScalar subdiag[1] = {RealScalar(subdiag_sign) * subdiag_before};

      internal::tridiagonal_qr_step<RealScalar, RealScalar, int>(diag, subdiag, 0, 1, nullptr, 2);

      // A Wilkinson shift nearly diagonalizes this 2x2 block in one step. Reassociating its fallback through e^2
      // under fast-math makes the correction underflow under FTZ and instead leaves the off-diagonal unchanged.
      VERIFY(numext::abs(subdiag[0]) <= NumTraits<RealScalar>::epsilon() * subdiag_before);
    }
  }
}

EIGEN_DECLARE_TEST(eigensolver_selfadjoint_fastmath) {
  CALL_SUBTEST(wilkinson_shift_avoids_squared_subdiagonal<float>());
  CALL_SUBTEST(wilkinson_shift_avoids_squared_subdiagonal<double>());
}
