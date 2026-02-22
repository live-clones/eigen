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
Vector3d v = Vector3d::Random();
Projective3d P(Matrix4d::Random());
cout << "v                                   = [" << v.transpose() << "]^T" << endl;
cout << "h.homogeneous()                     = [" << v.homogeneous().transpose() << "]^T" << endl;
cout << "(P * v.homogeneous())               = [" << (P * v.homogeneous()).transpose() << "]^T" << endl;
cout << "(P * v.homogeneous()).hnormalized() = [" << (P * v.homogeneous()).eval().hnormalized().transpose() << "]^T"
     << endl;

}
  return 0;
}
