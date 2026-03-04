static bool eigen_did_assert = false;
#define eigen_assert(X)                                                                \
  if (!eigen_did_assert && !(X)) {                                                     \
    std::cout << "### Assertion raised in " << __FILE__ << ":" << __LINE__ << ":\n" #X \
              << "\n### The following would happen without assertions:\n";             \
    eigen_did_assert = true;                                                           \
  }

#include <iostream>
#include <cassert>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;
using namespace std;

int main(int, char**) {
  cout.precision(3);
// intentionally remove indentation of snippet
{
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

int array[24];
for (int i = 0; i < 24; ++i) array[i] = i;

cout << "Original column-major matrix:\n" << Map<Matrix<int, 6, 4> >(array) << endl;
cout << "Every other row:\n" << Map<Matrix<int, 3, 4>, Unaligned, InnerStride<2> >(array) << endl;

}
  return 0;
}
