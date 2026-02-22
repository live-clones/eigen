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
Array<double, 1, 3> x(8, 25, 3), e(1. / 3., 0.5, 2.);
cout << "[" << x << "]^[" << e << "] = " << x.pow(e) << endl;   // using ArrayBase::pow
cout << "[" << x << "]^[" << e << "] = " << pow(x, e) << endl;  // using Eigen::pow

}
  return 0;
}
