#include "main.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <type_traits>

// Asserts the expression `T1 = T2` returns `T1&`
template <typename T1, typename T2>
void check_assignment_return_type() {
  using T1AssignmentOperatorReturnType = decltype(std::declval<T1>() = std::declval<const T2&>());
  static_assert(std::is_same<T1&, T1AssignmentOperatorReturnType>::value);

  if constexpr (!std::is_same<T1&, T1AssignmentOperatorReturnType>::value) {
    using X = typename T1AssignmentOperatorReturnType::show_this_type_in_error_message;
    X* unused;
  }
}

template <typename Derived, typename OtherDerived, typename Scalar>
void test_assignment_operators() {
  using ReturnByValueOtherDerived = decltype(std::declval<OtherDerived>().matrix().fullPivHouseholderQr().matrixQ());

#define CHECK_ASSIGNMENT_OPERATORS_DenseBase(Type)                   \
  do {                                                               \
    check_assignment_return_type<Type, DenseBase<Derived>>();        \
    check_assignment_return_type<Type, DenseBase<OtherDerived>>();   \
    check_assignment_return_type<Type, EigenBase<OtherDerived>>();   \
    check_assignment_return_type<Type, ReturnByValueOtherDerived>(); \
  } while (0)

  CHECK_ASSIGNMENT_OPERATORS_DenseBase(DenseBase<Derived>);

#define CHECK_ASSIGNMENT_OPERATORS_ArrayBase(Type)            \
  do {                                                        \
    check_assignment_return_type<Type, ArrayBase<Derived>>(); \
    check_assignment_return_type<Type, Scalar>();             \
    CHECK_ASSIGNMENT_OPERATORS_DenseBase(Type);               \
  } while (0)

  CHECK_ASSIGNMENT_OPERATORS_ArrayBase(ArrayBase<Derived>);
  // Classes derived from ArrayBase:
  // ArrayWrapper
  CHECK_ASSIGNMENT_OPERATORS_ArrayBase(decltype(std::declval<Derived>().array()));

#define CHECK_ASSIGNMENT_OPERATORS_MatrixBase(Type)                  \
  do {                                                               \
    check_assignment_return_type<Type, MatrixBase<Derived>>();       \
    check_assignment_return_type<Type, DenseBase<OtherDerived>>();   \
    check_assignment_return_type<Type, EigenBase<OtherDerived>>();   \
    check_assignment_return_type<Type, ReturnByValueOtherDerived>(); \
    CHECK_ASSIGNMENT_OPERATORS_DenseBase(Type);                      \
  } while (0)

  CHECK_ASSIGNMENT_OPERATORS_MatrixBase(MatrixBase<Derived>);
  // Classes derived from MatrixBase:
  // MatrixWrapper
  CHECK_ASSIGNMENT_OPERATORS_MatrixBase(decltype(std::declval<Array2d>().matrix()));

#define CHECK_ASSIGNMENT_OPERATORS_dense_xpr_base(Type)        \
  do {                                                         \
    if constexpr (std::is_base_of<ArrayBase<Type>, Type>::value) {  \
      CHECK_ASSIGNMENT_OPERATORS_ArrayBase(Type);              \
    }                                                          \
    if constexpr (std::is_base_of<MatrixBase<Type>, Type>::value) { \
      CHECK_ASSIGNMENT_OPERATORS_MatrixBase(Type);             \
    }                                                          \
  } while (0)

  // Classes derived from dense_xpr_base, assignable:
  // TODO: Block, if DirectAccessBit is false
  // CwiseUnaryView
  CHECK_ASSIGNMENT_OPERATORS_dense_xpr_base(decltype(std::declval<Array2cd>().real()));
  CHECK_ASSIGNMENT_OPERATORS_dense_xpr_base(decltype(std::declval<Matrix2cd>().real()));
  // Diagonal
  CHECK_ASSIGNMENT_OPERATORS_dense_xpr_base(decltype(std::declval<Matrix2d>().diagonal()));
  // Reverse
  CHECK_ASSIGNMENT_OPERATORS_dense_xpr_base(decltype(std::declval<Matrix2d>().reverse()));
  // Transpose
  CHECK_ASSIGNMENT_OPERATORS_dense_xpr_base(decltype(std::declval<Matrix2d>().transpose()));

#define CHECK_ASSIGNMENT_OPERATORS_PlainObjectBase(Type)             \
  do {                                                               \
    check_assignment_return_type<Type, PlainObjectBase<Derived>>();  \
    check_assignment_return_type<Type, ReturnByValueOtherDerived>(); \
    check_assignment_return_type<Type, EigenBase<OtherDerived>>();   \
    CHECK_ASSIGNMENT_OPERATORS_dense_xpr_base(Type);                 \
  } while (0)

  CHECK_ASSIGNMENT_OPERATORS_PlainObjectBase(PlainObjectBase<Derived>);

  // Classes derived from PlainObjectBase:
  // Matrix
  CHECK_ASSIGNMENT_OPERATORS_PlainObjectBase(Matrix2d);
  // Array
  CHECK_ASSIGNMENT_OPERATORS_PlainObjectBase(Array2d);

#define CHECK_ASSIGNMENT_OPERATORS_MapBase(Type)            \
  do {                                                      \
    check_assignment_return_type<Type, MapBase<Derived>>(); \
    CHECK_ASSIGNMENT_OPERATORS_dense_xpr_base(Type);        \
  } while (0)

  CHECK_ASSIGNMENT_OPERATORS_MapBase(MapBase<Derived>);
  // Classes derived from MapBase
  // Reshaped
  CHECK_ASSIGNMENT_OPERATORS_MapBase(decltype(std::declval<Derived>().reshaped()));

#define CHECK_ASSIGNMENT_OPERATORS_Block(Type) \
  do {                                         \
    CHECK_ASSIGNMENT_OPERATORS_MapBase(Type);  \
  } while (0)

  CHECK_ASSIGNMENT_OPERATORS_Block(decltype(std::declval<Derived>().block(0, 0, 1, 1)));
  // Classes derived from Block:
  // VectorBlock
  CHECK_ASSIGNMENT_OPERATORS_Block(decltype(std::declval<Vector2d>().segment(0, 1)));

#define CHECK_ASSIGNMENT_OPERATORS_Map(Type)  \
  do {                                        \
    CHECK_ASSIGNMENT_OPERATORS_MapBase(Type); \
  } while (0)

  CHECK_ASSIGNMENT_OPERATORS_Map(Map<Derived>);

#define CHECK_ASSIGNMENT_OPERATORS_RefBase(Type) \
  do {                                           \
    CHECK_ASSIGNMENT_OPERATORS_MapBase(Type);    \
  } while (0)

  // RefBase: No external usage
  // Classes derived from RefBase:
  // Ref
  CHECK_ASSIGNMENT_OPERATORS_RefBase(Ref<Derived>);

  // NoAlias
  if constexpr (std::is_base_of_v<ArrayBase<Derived>, Derived>) {
    check_assignment_return_type<NoAlias<Derived, ArrayBase>, ArrayBase<Derived>>();
  }
  if constexpr (std::is_base_of_v<MatrixBase<Derived>, Derived>) {
    check_assignment_return_type<NoAlias<Derived, MatrixBase>, MatrixBase<Derived>>();
  }

  // PermutationBase: No external visibility
  // PermutationBase::operator=(PermutationBase<OtherDerived>)
  // PermutationBase::operator=(TranspositionsBase<OtherDerived>)

  // TranspositionsBase: No external visibility
  // TranspositionsBase::operator=(TranspositionsBase<OtherDerived>)

  // TriangularViewImpl::operator=(TriangularBase<OtherDerived>)
  // TriangularViewImpl::operator=(MatrixBase<OtherDerived>)
  check_assignment_return_type<TriangularViewImpl<Matrix2d, Lower, Dense>,
                               TriangularBase<TriangularView<OtherDerived, Lower>>>();
  check_assignment_return_type<TriangularViewImpl<Matrix2d, Lower, Dense>, MatrixBase<OtherDerived>>();
  check_assignment_return_type<TriangularViewImpl<Matrix2d, Lower, Dense>,
                               TriangularViewImpl<Matrix2d, Lower, Dense>>();
  check_assignment_return_type<TriangularView<Matrix2d, Lower>, TriangularBase<TriangularView<OtherDerived, Lower>>>();
  check_assignment_return_type<TriangularView<Matrix2d, Lower>, MatrixBase<OtherDerived>>();

  // VectorwiseOp::operator=(DenseBase<OtherDerived>)
  check_assignment_return_type<VectorwiseOp<Matrix2d, Vertical>, DenseBase<OtherDerived>>();

  // Quaternion
  check_assignment_return_type<QuaternionBase<Quaternion<double>>, QuaternionBase<Quaternion<double>>>();
  check_assignment_return_type<QuaternionBase<Quaternion<double>>, QuaternionBase<Map<Quaternion<double>>>>();
  check_assignment_return_type<QuaternionBase<Quaternion<double>>, AngleAxis<double>>();
  check_assignment_return_type<QuaternionBase<Quaternion<double>>, Matrix3d>();
  check_assignment_return_type<QuaternionBase<Quaternion<double>>, MatrixBase<Matrix3d>>();
  check_assignment_return_type<Quaternion<double>, Quaternion<double>>();
  check_assignment_return_type<Quaternion<double>, Map<Quaternion<double>>>();
  check_assignment_return_type<Quaternion<double>, AngleAxis<double>>();
  check_assignment_return_type<Quaternion<double>, MatrixBase<Matrix3d>>();
  check_assignment_return_type<Quaternion<double>, Matrix3d>();
  check_assignment_return_type<Map<Quaternion<double>>, Map<Quaternion<double>>>();
  check_assignment_return_type<Map<Quaternion<double>>, Quaternion<double>>();
  check_assignment_return_type<Map<Quaternion<double>>, AngleAxis<double>>();
  check_assignment_return_type<Map<Quaternion<double>>, MatrixBase<Matrix3d>>();
  check_assignment_return_type<Map<Quaternion<double>>, Matrix3d>();

  // TODO sparse_matrix_block_impl::operator=(SparseMatrixBase<OtherDerived>)
  // TODO sparse_matrix_block_impl::operator=(BlockType)
  // TODO SparseMatrixBase::operator=(EigenBase<OtherDerived>)
  // TODO SparseMatrixBase::operator=(ReturnByValue<OtherDerived>)
  // TODO SparseMatrixBase::operator=(SparseMatrixBase<OtherDerived>)
  // TODO SparseMatrixBase::operator=(Derived)
  // TODO TensorBase::operator=(OtherDerived)
  // TODO TensorStorage::operator=(Self)
  // TODO SkylineMatrixBase::operator=(Derived)
  // TODO SkylineMatrixBase::operator=(SkylineMatrixBase<OtherDerived>)
  // TODO SkylineMatrixBase::operator=(SkylineProduct<Lhs, Rhs, SkylineTimeSkylineProduct>)
}

EIGEN_DECLARE_TEST(assignment_operators) {
  test_assignment_operators<Matrix2d, MatrixXd, double>();
  test_assignment_operators<Array2d, ArrayXd, double>();
  test_assignment_operators<Matrix2d, ArrayXXd, double>();
  test_assignment_operators<Array2d, MatrixXd, double>();
}
