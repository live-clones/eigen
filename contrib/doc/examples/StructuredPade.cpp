// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// The [m/m] Pade approximant p(x)/q(x) of exp(x), q(0) = 1, from its Taylor
// coefficients c_k = 1/k!. Matching the series through degree 2m gives the
// denominator from a Hankel system,
//   sum_{j=1}^{m} c_{m+i-j} q_j = -c_{m+i},  i = 1..m,
// written with H(i,j) = c_{i+j+1} (j reversed), and the numerator by a
// truncated convolution p_i = sum_{j<=i} c_{i-j} q_j.

#include <contrib/Eigen/StructuredMatrices>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace Eigen;

int main() {
  std::cout << " m  |p(1)/q(1) - e|  |p(-2)/q(-2) - e^-2|\n";
  for (int m : {2, 4, 6, 8}) {
    VectorXd c(2 * m + 1);
    c[0] = 1;
    for (int k = 1; k <= 2 * m; ++k) c[k] = c[k - 1] / k;
    // H(i, j) = c_{i+j+1} for i, j = 0..m-1: first column c_1..c_m, last row c_m..c_{2m-1}.
    const auto H = makeHankel(c.segment(1, m), c.segment(m, m));
    // The unknowns are q_m, ..., q_1 (reversed): row i reads c_{i+1} q_m + ... + c_{m+i} q_1.
    const VectorXd reversed = H.solve(-c.segment(m + 1, m));
    VectorXd q(m + 1), p(m + 1);
    q[0] = 1;
    q.tail(m) = reversed.reverse();
    for (int i = 0; i <= m; ++i) p[i] = c.head(i + 1).reverse().dot(q.head(i + 1));
    const auto ratio = [&](double t) {
      double num = 0, den = 0;
      for (int i = m; i >= 0; --i) {
        num = num * t + p[i];
        den = den * t + q[i];
      }
      return num / den;
    };
    std::cout << std::setw(2) << m << "  " << std::setw(15) << std::abs(ratio(1) - std::exp(1.0)) << "  "
              << std::abs(ratio(-2) - std::exp(-2.0)) << "\n";
  }
}
