// SPDX-License-Identifier: MPL-2.0

#include "../Eigen/Core"

#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
#define ROWS Dynamic
#define COLS Dynamic
#else
#define ROWS 3
#define COLS 1
#endif

using namespace Eigen;

int main() { Matrix<int, ROWS, COLS>{1, 2, 3}; }
