// main.h adds instrumentation which breaks constexpr so we do not run any tests in here,
// this is strictly compile-time.
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace Eigen;

inline void error_if_not_constexpr() {} // not constexpr
#if EIGEN_COMP_CXXVER >= 20
consteval
#else
constexpr
#endif
void assert_constexpr(bool b) {
  if (!b) error_if_not_constexpr();
}

constexpr bool zeroSized()
{
  constexpr Matrix<float, 0, 0> m0;
  static_assert(m0.size() == 0, "");

  constexpr Matrix<float, 0, 0> m1;
  static_assert(m0 == m1, "");
  static_assert(!(m0 != m1), "");

  constexpr Array<int, 0, 0> a0;
  static_assert(a0.size() == 0, "");

  constexpr Array<int, 0, 0> a1;
  static_assert((a0 == a1).all(), "");
  static_assert((a0 != a1).count() == 0, "");

  constexpr Array<float, 0, 0> af;
  static_assert(m0 == af.matrix(), "");
  static_assert((m0.array() == af).all(), "");
  static_assert(m0.array().matrix() == m0, "");

  return true;
}

static_assert(zeroSized(), "");

static constexpr double static_data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

constexpr bool maps()
{
  constexpr Map<const Vector4d> m(static_data);
  static_assert(m(0) == 1, "");
  constexpr Map<const Array<double, 4, 1>> a(static_data);
  static_assert(m == a.matrix(), "");
  static_assert(m.size() == 4, "");
  static_assert(a.size() == 4, "");
  static_assert(m.rows() == 4 && m.cols() == 1, "");
  return true;
}

static_assert(maps(), "");

constexpr bool nc_maps()
{
  bool result = true;

  double d[] = {1, 2, 3, 4};
  Map<Vector4d> m(d);
  result = result && (m.x() == 1 && m.y() == 2 && m.z() == 3 && m.w() == 4);

  float array[3] = {};
  auto v = Vector3f::Map(array);
  v.fill(10);
  result = result && (v.array() == 10).all();

  return result;
}

constexpr bool blocks()
{
  constexpr Map<const Matrix2d> m(static_data);
  constexpr auto block = m.block<2,1>(0, 1);

  constexpr Map<const Vector2d> v(static_data + 2);
  static_assert(block == v, "");

  return true;
}

static_assert(blocks(), "");

constexpr bool diagonal_row_columns()
{
  constexpr Map<const Matrix2d> m(static_data);
  static_assert(m.block<2,1>(0, 1) == m.col(1), "");
  static_assert(m.block<1,2>(1, 0) == m.row(1), "");
  static_assert(m.diagonal()(0) == 1 && m.diagonal()(1) == 4, "");
  return true;
}

static_assert(diagonal_row_columns(), "");

static constexpr int static_data_antisym[] = {
  0, 1, -1,
  -1, 0, 1,
  1, -1, 0 };

constexpr bool transpose_unaryminus()
{
  constexpr Map<const Matrix<int, 3, 3>> m(static_data_antisym);

  static_assert(m.transpose() == -m, "");
  static_assert(-m.transpose() == m, "");
  static_assert((-m).transpose() == m, "");

  static_assert(m.transpose() != m, "");
  static_assert(-m.transpose() != -m, "");
  static_assert((-m).transpose() != -m, "");

  return true;
}

static_assert(transpose_unaryminus(), "");

constexpr bool reductions()
{
  constexpr Map<const Matrix<int, 3, 3>> m(static_data_antisym);
  static_assert(m.size() == 9, "");

#if EIGEN_COMP_CXXVER >= 20
  static_assert(m.sum() == 0);
  static_assert(m.trace() == 0);
  static_assert(m.mean() == 0);
  static_assert(m.prod() == 0);
  static_assert(m.minCoeff() == -1);
  static_assert(m.maxCoeff() == 1);

  static_assert(m.rowwise().sum() == Vector3i::Zero());
  static_assert((((m.array() == m.array().transpose()).colwise()).count() == Array<long, 1, 3>::Ones()).all());

  static_assert(m.squaredNorm() == 6);
#endif

  return true;
}

static_assert(reductions(), "");

constexpr bool scalar_mult_div()
{
  constexpr Map<const Matrix2d> m(static_data);

  static_assert((m * 2)(0,0) == 2, "");
  static_assert((m / 2)(1,1) == 2*m(0,0), "");
#if EIGEN_COMP_CXXVER >= 20
  static_assert((2 * m).sum() == 2 * 4 * 5 / 2, "");
#endif

  constexpr double c = 8;
  static_assert((m * c)(0,0) == 8, "");
  static_assert((m.array() / c).matrix() == 1/c * m, "");
  return true;
}

static_assert(scalar_mult_div(), "");

constexpr bool determinant()
{
  constexpr Map<const Matrix<double, 1, 1>> m1(static_data);
  static_assert(m1.determinant() == 1, "");

  constexpr Map<const Matrix<double, 2, 2>> m2(static_data);
  static_assert(m2.determinant() == 1*4 - 2*3, "");

  constexpr Map<const Matrix<double, 3, 3>> m3(static_data);
  static_assert(m3.determinant() == 1*(5*9 - 8*6) - 2*(4*9 - 6*7) + 3*(4*8 - 5*7), "");

  static_assert(Matrix4d::Identity().determinant() == 1, "");

#if EIGEN_COMP_CXXVER >= 20
  constexpr Matrix4d M{{0,1,0,0},{0,0,1,0},{0,0,0,1},{1,0,0,0}};
  static_assert(M.determinant() == -1);
#endif

  return true;
}

static_assert(determinant(), "");

constexpr bool constant_identity()
{
#if EIGEN_COMP_CXXVER >= 20
  static_assert(Matrix<float, 8, 7>::Constant(5).sum() == 5 * 7 * 8);
#endif
  static_assert(Matrix3f::Zero()(0,0) == 0, "");
  static_assert(Matrix4d::Ones()(3,3) == 1, "");
  static_assert(Matrix2i::Identity()(0,0) == 1 && Matrix2i::Identity()(1,0) == 0, "");
#if EIGEN_COMP_CXXVER >= 20
  static_assert(Matrix<int, 5, 5>::Identity().trace() == 5);
#endif
  static_assert(Matrix<float, Dynamic, Dynamic>::Ones(2,3).size() == 6, "");
  static_assert(Matrix<float, Dynamic, 1>::Zero(10).rows() == 10, "");
#if EIGEN_COMP_CXXVER >= 20
  static_assert(Matrix<int, 1, Dynamic>::Constant(20, 20).sum() == 400);
#endif

  return true;
}

static_assert(constant_identity(), "");

constexpr bool dynamic_basics()
{
  // This verifies that we only calculate the entry that we need.
  static_assert(Matrix<double, Dynamic, Dynamic>::Identity(50000,50000).array()(25,25) == 1, "");

  static_assert(Matrix4d::Identity().block(1,1,2,2)(0,1) == 0, "");
  static_assert(MatrixXf::Identity(50,50).transpose() == MatrixXf::Identity(50, 50), "");

  constexpr Map<const MatrixXi> dynMap(static_data_antisym, 3, 3);
  constexpr Map<const Matrix3i> staticMap(static_data_antisym);
  static_assert(dynMap == staticMap, "");
  static_assert(dynMap.transpose() != staticMap, "");
  // e.g. this hits an assertion at compile-time that would otherwise fail at runtime.
  //static_assert(dynMap != staticMap.block(1,2,0,0));

  return true;
}

static_assert(dynamic_basics(), "");

constexpr bool sums()
{
  constexpr Map<const Matrix<double, 4, 4>> m(static_data);
  constexpr auto b(m.block<2,2>(0,0)); // 1 2 5 6
  constexpr Map<const Matrix2d> m2(static_data); // 1 2 3 4

  static_assert((b + m2).col(0) == 2*Map<const Vector2d>(static_data), "");
  static_assert(b + m2 == m2 + b, "");

  static_assert((b - m2).col(0) == Vector2d::Zero(), "");
  static_assert((b - m2).col(1) == 2*Vector2d::Ones(), "");

  static_assert((2*b - m2).col(0) == b.col(0), "");
  static_assert((b - 2*m2).col(0) == -b.col(0), "");

  static_assert((b - m2 + b + m2 - 2*b) == Matrix2d::Zero(), "");

  return true;
}

static_assert(sums(), "");

constexpr bool unit_vectors()
{
  static_assert(Vector4d::UnitX()(0) == 1, "");
  static_assert(Vector4d::UnitY()(1) == 1, "");
  static_assert(Vector4d::UnitZ()(2) == 1, "");
  static_assert(Vector4d::UnitW()(3) == 1, "");

  static_assert(Vector4d::UnitX().dot(Vector4d::UnitX()) == 1, "");
  static_assert(Vector4d::UnitX().dot(Vector4d::UnitY()) == 0, "");
  static_assert((Vector4d::UnitX() + Vector4d::UnitZ()).dot(Vector4d::UnitY() + Vector4d::UnitW()) == 0, "");

  return true;
}

static_assert(unit_vectors(), "");

constexpr bool construct_from_other()
{
#if EIGEN_COMP_CXXVER >= 20
  constexpr Matrix2d m{Map<const Matrix2d>(static_data)};
  static_assert(m == Map<const Matrix2d>(static_data));

  constexpr Matrix2d mt{Map<const Matrix2d>(static_data).transpose()};
  static_assert(m == mt.transpose());
  static_assert(m.transpose() == mt);
  static_assert(m.diagonal() == mt.diagonal());

  constexpr Matrix3i a{Map<const Matrix3i>(static_data_antisym)};
  constexpr Matrix3i a_plus_at{a + a.transpose()};
  static_assert(a_plus_at == Matrix3i::Zero());

  constexpr Matrix3i at{Map<const Matrix<int, 3, 3, RowMajor>>(a.data())};
  static_assert(a + at == Matrix3i::Zero());
  static_assert(a + at.transpose() == 2 * a);
  static_assert(a.diagonal() == at.diagonal());

  constexpr Matrix3i aa{a};
  static_assert(aa == at.transpose());
#endif

  return true;
}

static_assert(construct_from_other(), "");

constexpr bool construct_from_values()
{
#if EIGEN_COMP_CXXVER >= 20
  constexpr Matrix<float, 1, 1> m11(55);
  static_assert(m11.x() == 55);

  constexpr Vector2i m21(3.1, 42e-2);
  static_assert(m21.sum() == 3);

  constexpr Vector3d m31(2.7, 18e-3, 1e-4);
  static_assert(m31.sum() == 2.7 + 0.018 + 0.0001);

  constexpr Vector4d m41(1, 2, 3, 4);
  static_assert(m41.x() == 1 && m41.y() == 2 && m41.z() == 3 && m41.w() == 4);

  constexpr Matrix<int, 1, 6> a(1, 2, 3, 4, 5, 6);
  static_assert(a.sum() == 6*7/2);

  constexpr Matrix3d m33{{0, 1, 0}, {0, 0, -1}, {-1, 0, 0}};
  static_assert(m33.determinant() == 1);

  constexpr double data[] = { 1, 1, 2, 2 };
  constexpr Matrix2d md(data);
  static_assert(md.sum() == 6);
#endif

  return true;
}

static_assert(construct_from_values(), "");

constexpr bool triangular()
{
  bool result = true;
#if EIGEN_COMP_CXXVER >= 20
  constexpr Map<const Matrix4d> m(static_data);
  constexpr Matrix4d m2{m.triangularView<Upper>()};
  constexpr Vector4d m3{m.triangularView<Upper>().solve(m.col(0))};
  result = result && m3(0) != 0;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      result = result && m2(i,j) == (i < j) ? 0 : m(i,j);
#endif
  return result;
}

static_assert(triangular(), "");

constexpr bool nc_construct_from_values()
{
  bool result = true;

#if EIGEN_COMP_CXXVER >= 20
  Matrix<float, 1, 1> m11(55);
  result = result && (m11.x() == 55);

  Matrix<float, 1, 1> maa(22);
  m11.swap(maa);
  result = result && (m11.x() == 22 && maa.x() == 55);
#endif
  return result;
}

constexpr bool nc_crossproduct()
{
  bool result = true;
#if EIGEN_COMP_CXXVER >= 20
  Vector3d X(Vector3d::UnitX());
  Vector3d Y(Vector3d::UnitY());
  Vector3d Z(Vector3d::UnitZ());

  result = result && X.cross(Y) == Z;
#endif
  return result;
}

constexpr bool nc_cast()
{
  bool result = true;
#if EIGEN_COMP_CXXVER >= 20
  constexpr Vector3f v(Vector3d::UnitY().cast<float>());

  result = result && v == Vector3f::UnitY();
#endif

  return result;
}

constexpr bool nc_product()
{
  bool result = true;
#if EIGEN_COMP_CXXVER >= 20
#if !EIGEN_COMP_GNUC_STRICT
  // Doesn't work with GCC at least up to version 11.2
  constexpr Matrix2i a{{0,-1},{1,0}};
  constexpr Matrix2i b{{0,1},{-1,0}};
  Vector2d v{{2,-2}};

  result = result && a*b == Matrix2i::Identity();
  result = result && (a.cast<double>() * v == Vector2d::Constant(2));
  result = result && (2*a*b) == 2*Matrix2i::Identity();
  result = result && 2*a*b == a*2*b;

  constexpr Matrix3i c{{-2,0,0},{0,-2,0},{0,0,2}};
  result = result && (c.block(0, 0, 2, 2) * a == 2 * b);
  result = result && (2 * a == c.block(0, 0, 2, 2) * b);
  result = result && (c.block<2,2>(1,1) * a == Matrix2i{{0,2},{2,0}});

  constexpr Map<const Matrix<double,2,2,RowMajor>> m(static_data);
  result = result && (m * a.cast<double>() == Matrix2d{{2, -1}, {4, -3}});

  result = result && (a * a.transpose() == Matrix2i::Identity());
  result = result && (a.transpose() * a == Matrix2i::Identity());
  result = result && (a * b).transpose() == Matrix2i::Identity();
  result = result && (a.transpose() * b.transpose()).transpose() == Matrix2i::Identity();
#endif
#endif
  return result;
}

static constexpr double static_data_quat[] = { 0, 1, 1, 0 };

constexpr bool nc_quat_mult()
{
  bool result = static_data_quat[3] == 0; // Silence warning about unused with C++14

#if EIGEN_COMP_CXXVER >= 20
  constexpr Map<const Quaterniond> mqyz(static_data_quat);
  result = result && mqyz.coeffs() == Vector4d::UnitY() + Vector4d::UnitZ();

  Quaterniond q1{1,0,0,0};
  Quaterniond q2{0,1,0,0};

  result = result && q1*q2 == q2;
  result = result && q1 == Quaternionf::Identity().cast<double>();

  Quaterniond q3{1,-1,1,-1};
  result = result && q3.squaredNorm() == 4;
  result = result && q3.dot(q1) == 1 && q2.dot(q3) == -1;

  Vector3d vy = Vector3d::UnitY();
  result = result && q2*vy == -vy;
  result = result && q2.toRotationMatrix()*vy == -vy;

  constexpr double d[] = { 0, 0, 1, 0 };
  Quaterniond qz(d);
  result = result && qz.z() == 1 && qz * Vector3d::UnitZ() == Vector3d::UnitZ();
  result = result && qz * qz.conjugate() == decltype(qz)::Identity();
  result = result && q3 * qz * qz.conjugate() * q3.inverse() == decltype(qz)::Identity();

  Quaternionf qf{.5,.5,.5,.5};
  Quaterniond qd(qf);
  result = result && qd * Vector3d::UnitX() == Vector3d::UnitY();
#endif
  return result;
}

// Run tests that aren't explicitly constexpr.
constexpr bool test_nc()
{
  assert_constexpr(nc_maps());
  assert_constexpr(nc_construct_from_values());
  assert_constexpr(nc_crossproduct());
  assert_constexpr(nc_cast());
  assert_constexpr(nc_product());
  assert_constexpr(nc_quat_mult());
  return true;
}

int main()
{
  return !test_nc();
}
