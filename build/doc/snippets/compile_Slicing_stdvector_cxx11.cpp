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
std::vector<int> ind{4, 2, 5, 5, 3};
MatrixXi A = MatrixXi::Random(4, 6);
cout << "Initial matrix A:\n" << A << "\n\n";
cout << "A(all,ind):\n" << A(Eigen::placeholders::all, ind) << "\n\n";

}
  return 0;
}
