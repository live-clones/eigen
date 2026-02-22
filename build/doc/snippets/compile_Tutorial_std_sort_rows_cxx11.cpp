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
ArrayXXi A = ArrayXXi::Random(4, 4).abs();
cout << "Here is the initial matrix A:\n" << A << "\n";
for (auto row : A.rowwise()) std::sort(row.begin(), row.end());
cout << "Here is the sorted matrix A:\n" << A << "\n";

}
  return 0;
}
