#include <Eigen/CXX11/Tensor>
#include <iostream>

int main() {
  {
    // Some initialization, let's go to the next comment.
    Eigen::TensorFixedSize<float, Eigen::Sizes<3, 5>> A;
    Eigen::TensorFixedSize<float, Eigen::Sizes<5, 2, 6, 4>> B;
    Eigen::TensorFixedSize<float, Eigen::Sizes<4, 2, 3>> C;
    A.setRandom();
    B.setRandom();
    C.setRandom();

    Eigen::TensorFixedSize<float, Eigen::Sizes<2, 3, 4>> R0;
    Eigen::TensorFixedSize<float, Eigen::Sizes<2, 3, 4>> R1;
    Eigen::TensorFixedSize<float, Eigen::Sizes<2, 3, 4>> R2;
    Eigen::TensorFixedSize<float, Eigen::Sizes<2, 3, 4>> R3;

    // Sometimes, conventional tensor expressions can be difficult to read.
    // Without indexed notation, one would write something like this
    R0 = (A.contract(B.chip(1, 2), std::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 0)}) +
          C.shuffle(std::array<int, 3>{2, 1, 0}))
             .shuffle(std::array<int, 3>{1, 0, 2});

    // It is not clear whether the first or the second dimension of B is
    // chipped. Because of the shuffling and chipping, it is difficult to
    // follow which dimensions are contracted or added. If the expression is
    // wrong, I get no compilation error but an assertion failure at runtime
    // (a crash).

    // To write clearer expressions, one can use an indexed expression, which is
    // an extension of the Einstein notation. This way, the above expression can
    // be equivalently rewritten as:
    using namespace Eigen::tensor_indices;
    R1(i, j, k) = A(j, l) * B(l, i, 1, k) + C(k, i, j);

    // A bunch of single letter indices can be imported with
    // `using namespace Eigen::tensor_indices;` or one can defined his own
    // indices with `TensorIndex<some_letter_or_number> some_index_name;`.
    Eigen::TensorIndex<'i'> ii;
    Eigen::TensorIndex<'j'> jj;
    Eigen::TensorIndex<'k'> kk;
    Eigen::TensorIndex<'l'> ll;  // here, one could use Eigen::TensorIndex<'l'>
                                 // my_very_long_index_name_which_is_equal_to_l_because_they_have_the_same_type;
    R2(ii, jj, kk) = A(jj, ll) * B(ll, ii, 1, kk) + C(kk, ii, jj);

    // By definition, an indexed expression does the same computation as if it
    // was in nested for loops.

    // loop over every index in result
    for (Eigen::DenseIndex i = 0; i < R3.dimension(0); ++i)
      for (Eigen::DenseIndex j = 0; j < R3.dimension(1); ++j)
        for (Eigen::DenseIndex k = 0; k < R3.dimension(2); ++k) {
          // assign zero
          R3(i, j, k) = 0.0f;
          // add terms without any other index
          R3(i, j, k) += C(k, i, j);
          // for terms with other indices, add the sum along those other indices
          for (Eigen::DenseIndex l = 0; l < A.dimension(1); ++l) {
            R3(i, j, k) += A(j, l) * B(l, i, 1, k);
          }
        }

    std::cout << "Let's check that R0 == R1 == R2 == R3 :" << std::endl;
    for (Eigen::DenseIndex i = 0; i < R3.dimension(0); ++i)
      for (Eigen::DenseIndex j = 0; j < R3.dimension(1); ++j)
        for (Eigen::DenseIndex k = 0; k < R3.dimension(2); ++k) {
          std::cout << R0(i, j, k) << " == " << R1(i, j, k) << " == " << R2(i, j, k) << " == " << R3(i, j, k)
                    << std::endl;
        }
  }

  {
    // For convenicene it is possible to directly assign a zeroth order
    // TensorIndexed, i.e., a scalar
    Eigen::TensorFixedSize<float, Eigen::Sizes<2, 3>> A;
    Eigen::TensorFixedSize<float, Eigen::Sizes<2, 3>> B;
    A.setRandom();
    B.setRandom();
    using namespace Eigen::tensor_indices;
    float result = A(i, j) * B(i, j);
  }

  {
    // The correct ordering of shuffling indices can be hard for tensor operations
    // involving mutliple indices
    Eigen::TensorFixedSize<float, Eigen::Sizes<2, 3, 4, 1>> A;
    A.setRandom();

    Eigen::TensorFixedSize<float, Eigen::Sizes<3, 1, 4, 2>> B1;
    B1 = A.shuffle(std::array<int, 4>{3, 0, 2, 1});

    // Using index notation complicated shuffling expressions get really verbose and
    // easy to read
    Eigen::TensorFixedSize<float, Eigen::Sizes<3, 1, 4, 2>> B2;
    using namespace Eigen::tensor_indices;
    B2(i, j, k, l) = A(l, i, k, j);
  }

  {
    // Chipping is made very easy since we use some fixed index in a given dimension
    // and use and `TensorIndex` for all other dimensions
    Eigen::TensorFixedSize<float, Eigen::Sizes<2, 3, 4>> A;
    A.setRandom();

    using namespace Eigen::tensor_indices;

    Eigen::TensorFixedSize<float, Eigen::Sizes<3, 4>> B0;
    B0(j, k) = A(1, j, k);  // first slice in dimension 0

    Eigen::TensorFixedSize<float, Eigen::Sizes<2, 4>> B1;
    B1(i, k) = A(i, 0, k);  // zeroth slice in dimension 1

    Eigen::TensorFixedSize<float, Eigen::Sizes<2, 3>> B2;
    B2(i, j) = A(i, j, 2);  // second slice in dimension 2
  }

  {
    // Tensor contractions can be really cumbersome and annoying to write:
    Eigen::TensorFixedSize<float, Eigen::Sizes<4, 3, 5>> A;
    Eigen::TensorFixedSize<float, Eigen::Sizes<4, 5>> B;
    A.setRandom();
    B.setRandom();

    Eigen::TensorFixedSize<float, Eigen::Sizes<3>> C0;
    C0 = A.contract(B, std::array<Eigen::IndexPair<int>, 2>{Eigen::IndexPair<int>(0, 0), Eigen::IndexPair<int>(2, 1)});

    // Using indexed notation this gets really verbose and easy to read
    Eigen::TensorFixedSize<float, Eigen::Sizes<3>> C1;
    using namespace Eigen::tensor_indices;
    C1(j) = A(i, j, k) * B(i, k);

    // The very same applies for tensor pruducts. For computing the [dyadic
    // product](https://en.wikipedia.org/wiki/Dyadics) of two first order tensors we currently have to write
    Eigen::TensorFixedSize<float, Eigen::Sizes<3>> a;
    Eigen::TensorFixedSize<float, Eigen::Sizes<3>> b;
    a.setRandom();
    b.setRandom();

    Eigen::TensorFixedSize<float, Eigen::Sizes<3, 3>> ab0;
    ab0 = A.contract(B, std::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 0)});

    // Using index notation this gets again really simple
    Eigen::TensorFixedSize<float, Eigen::Sizes<3, 3>> ab1;
    using namespace Eigen::tensor_indices;
    ab1(i, j) = a(i) * b(j);
  }

  {
    // Obviously, simple addition and subtraction is also supported which is only usefull in combination with shuffling
    // or contraction
    Eigen::TensorFixedSize<float, Eigen::Sizes<2, 3>> A;
    Eigen::TensorFixedSize<float, Eigen::Sizes<2, 3>> B;
    Eigen::TensorFixedSize<float, Eigen::Sizes<2, 2>> C;
    A.setRandom();
    B.setRandom();
    C.setRandom();

    Eigen::TensorFixedSize<float, Eigen::Sizes<2, 2>> D;
    using namespace Eigen::tensor_indices;
    D(i, j) = A(i, k) * B(k, j) + C(i, j);
    D(i, j) = A(i, k) * B(k, j) - C(i, j);
  }

  return 0;
}
