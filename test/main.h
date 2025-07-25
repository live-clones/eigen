// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2008 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstdlib>
#include <cerrno>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <typeinfo>
#include <type_traits>
#include <functional>
#ifdef EIGEN_USE_SYCL
#include <CL/sycl.hpp>
#endif

// The following includes of STL headers have to be done _before_ the
// definition of macros min() and max().  The reason is that many STL
// implementations will not work properly as the min and max symbols collide
// with the STL functions std::min() and std::max().  The STL headers may check
// for the macro definition of min/max and issue a warning or undefine the
// macros.
//
// Still, Windows defines min() and max() in windef.h as part of the regular
// Windows system interfaces and many other Windows APIs depend on these
// macros being available.  To prevent the macro expansion of min/max and to
// make Eigen compatible with the Windows environment all function calls of
// std::min() and std::max() have to be written with parenthesis around the
// function name.
//
// All STL headers used by Eigen should be included here.  Because main.h is
// included before any Eigen header and because the STL headers are guarded
// against multiple inclusions, no STL header will see our own min/max macro
// definitions.
#include <limits>
#include <algorithm>
// Disable ICC's std::complex operator specializations so we can use our own.
#define _OVERRIDE_COMPLEX_SPECIALIZATION_ 1
#include <complex>
#include <deque>
#include <queue>
#include <cassert>
#include <list>
#if __cplusplus >= 201103L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201103L)
#include <random>
#include <chrono>
#endif
#if __cplusplus > 201703L
// libstdc++ 9's <memory> indirectly uses max() via <bit>.
// libstdc++ 10's <memory> indirectly uses max() via ranges headers.
#include <memory>
// libstdc++ 11's <thread> indirectly uses max() via semaphore headers.
#include <thread>
#endif

// Configure GPU.
#if defined(EIGEN_USE_HIP)
#if defined(__HIPCC__) && !defined(EIGEN_NO_HIP)
#define EIGEN_HIPCC __HIPCC__
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#endif
#elif defined(__CUDACC__) && !defined(EIGEN_NO_CUDA)
#define EIGEN_CUDACC __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#if CUDA_VERSION >= 7050
#include <cuda_fp16.h>
#endif
#endif

#if defined(EIGEN_CUDACC) || defined(EIGEN_HIPCC)
#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#endif

// To test that all calls from Eigen code to std::min() and std::max() are
// protected by parenthesis against macro expansion, the min()/max() macros
// are defined here and any not-parenthesized min/max call will cause a
// compiler error.
#if !defined(__HIPCC__) && !defined(EIGEN_USE_SYCL) && !defined(EIGEN_POCKETFFT_DEFAULT)
//
// HIP header files include the following files
//  <thread>
//  <regex>
//  <unordered_map>
// which seem to contain not-parenthesized calls to "max"/"min", triggering the following check and causing the compile
// to fail
//
// Including those header files before the following macro definition for "min" / "max", only partially resolves the
// issue This is because other HIP header files also define "isnan" / "isinf" / "isfinite" functions, which are needed
// in other headers.
//
// So instead choosing to simply disable this check for HIP
//
#define min(A, B) please_protect_your_min_with_parentheses
#define max(A, B) please_protect_your_max_with_parentheses
#define isnan(X) please_protect_your_isnan_with_parentheses
#define isinf(X) please_protect_your_isinf_with_parentheses
#define isfinite(X) please_protect_your_isfinite_with_parentheses
#endif

// test possible conflicts
struct real {};
struct imag {};

#ifdef M_PI
#undef M_PI
#endif
#define M_PI please_use_EIGEN_PI_instead_of_M_PI

#define FORBIDDEN_IDENTIFIER \
  (this_identifier_is_forbidden_to_avoid_clashes) this_identifier_is_forbidden_to_avoid_clashes
// B0 is defined in POSIX header termios.h
#define B0 FORBIDDEN_IDENTIFIER
#define I FORBIDDEN_IDENTIFIER

// _res is defined by resolv.h
#define _res FORBIDDEN_IDENTIFIER

// Unit tests calling Eigen's blas library must preserve the default blocking size
// to avoid troubles.
#ifndef EIGEN_NO_DEBUG_SMALL_PRODUCT_BLOCKS
#define EIGEN_DEBUG_SMALL_PRODUCT_BLOCKS
#endif

// shuts down ICC's remark #593: variable "XXX" was set but never used
#define TEST_SET_BUT_UNUSED_VARIABLE(X) EIGEN_UNUSED_VARIABLE(X)

#ifdef TEST_ENABLE_TEMPORARY_TRACKING

static long int nb_temporaries;
static long int nb_temporaries_on_assert = -1;

#ifdef TEST_IGNORE_STACK_ALLOCATED_TEMPORARY
inline void on_temporary_creation(long int size, int SizeAtCompileTime) {
  // ignore stack-allocated temporaries
  if (SizeAtCompileTime != -1) return;
#else
inline void on_temporary_creation(long int size, int) {
#endif
  // here's a great place to set a breakpoint when debugging failures in this test!
  if (size != 0) nb_temporaries++;
  if (nb_temporaries_on_assert > 0) assert(nb_temporaries < nb_temporaries_on_assert);
}

#define EIGEN_DENSE_STORAGE_CTOR_PLUGIN \
  { on_temporary_creation(size, Size); }

#define VERIFY_EVALUATION_COUNT(XPR, N)                            \
  {                                                                \
    nb_temporaries = 0;                                            \
    XPR;                                                           \
    if (nb_temporaries != (N)) {                                   \
      std::cerr << "nb_temporaries == " << nb_temporaries << "\n"; \
    }                                                              \
    VERIFY((#XPR) && nb_temporaries == (N));                       \
  }

#endif

#include "split_test_helper.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

// On windows CE, NDEBUG is automatically defined <assert.h> if NDEBUG is not defined.
#ifndef DEBUG
#define DEBUG
#endif

#define DEFAULT_REPEAT 10

namespace Eigen {
static std::vector<std::string> g_test_stack;
// level == 0 <=> abort if test fail
// level >= 1 <=> warning message to std::cerr if test fail
static int g_test_level = 0;
static int g_repeat = 1;
static unsigned int g_seed = 0;
static bool g_has_set_repeat = false, g_has_set_seed = false;

class EigenTest {
 public:
  EigenTest() : m_func(0) {}
  EigenTest(const char* a_name, void (*func)(void)) : m_name(a_name), m_func(func) {
    get_registered_tests().push_back(this);
  }
  const std::string& name() const { return m_name; }
  void operator()() const { m_func(); }

  static const std::vector<EigenTest*>& all() { return get_registered_tests(); }

 protected:
  static std::vector<EigenTest*>& get_registered_tests() {
    static std::vector<EigenTest*>* ms_registered_tests = new std::vector<EigenTest*>();
    return *ms_registered_tests;
  }
  std::string m_name;
  void (*m_func)(void);
};

// Declare and register a test, e.g.:
//    EIGEN_DECLARE_TEST(mytest) { ... }
// will create a function:
//    void test_mytest() { ... }
// that will be automatically called.
#define EIGEN_DECLARE_TEST(X)                                                              \
  void EIGEN_CAT(test_, X)();                                                              \
  static EigenTest EIGEN_CAT(test_handler_, X)(EIGEN_MAKESTRING(X), &EIGEN_CAT(test_, X)); \
  void EIGEN_CAT(test_, X)()
}  // namespace Eigen

#define TRACK std::cerr << __FILE__ << " " << __LINE__ << std::endl

#define EIGEN_DEFAULT_IO_FORMAT IOFormat(4, 0, "  ", "\n", "", "", "", "")

#if (defined(_CPPUNWIND) || defined(__EXCEPTIONS)) && !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && \
    !defined(__SYCL_DEVICE_ONLY__)
#define EIGEN_EXCEPTIONS
#endif

#ifndef EIGEN_NO_ASSERTION_CHECKING

namespace Eigen {
static const bool should_raise_an_assert = false;

// Used to avoid to raise two exceptions at a time in which
// case the exception is not properly caught.
// This may happen when a second exceptions is triggered in a destructor.
static bool no_more_assert = false;
static bool report_on_cerr_on_assert_failure = true;

struct eigen_assert_exception {
  eigen_assert_exception(void) {}
  ~eigen_assert_exception() { Eigen::no_more_assert = false; }
};

struct eigen_static_assert_exception {
  eigen_static_assert_exception(void) {}
  ~eigen_static_assert_exception() { Eigen::no_more_assert = false; }
};
}  // namespace Eigen
// If EIGEN_DEBUG_ASSERTS is defined and if no assertion is triggered while
// one should have been, then the list of executed assertions is printed out.
//
// EIGEN_DEBUG_ASSERTS is not enabled by default as it
// significantly increases the compilation time
// and might even introduce side effects that would hide
// some memory errors.
#ifdef EIGEN_DEBUG_ASSERTS

namespace Eigen {
namespace internal {
static bool push_assert = false;
}
static std::vector<std::string> eigen_assert_list;
}  // namespace Eigen
#define eigen_assert(a)                                                                                             \
  if ((!(a)) && (!no_more_assert)) {                                                                                \
    if (report_on_cerr_on_assert_failure) std::cerr << #a << " " __FILE__ << "(" << __LINE__ << ")\n";              \
    Eigen::no_more_assert = true;                                                                                   \
    EIGEN_THROW_X(Eigen::eigen_assert_exception());                                                                 \
  } else if (Eigen::internal::push_assert) {                                                                        \
    eigen_assert_list.push_back(std::string(EIGEN_MAKESTRING(__FILE__) " (" EIGEN_MAKESTRING(__LINE__) ") : " #a)); \
  }

#ifdef EIGEN_EXCEPTIONS
#define VERIFY_RAISES_ASSERT(a)                                                                                  \
  {                                                                                                              \
    Eigen::no_more_assert = false;                                                                               \
    Eigen::eigen_assert_list.clear();                                                                            \
    Eigen::internal::push_assert = true;                                                                         \
    Eigen::report_on_cerr_on_assert_failure = false;                                                             \
    try {                                                                                                        \
      a;                                                                                                         \
      std::cerr << "One of the following asserts should have been triggered:\n";                                 \
      for (uint ai = 0; ai < eigen_assert_list.size(); ++ai) std::cerr << "  " << eigen_assert_list[ai] << "\n"; \
      VERIFY(Eigen::should_raise_an_assert&& #a);                                                                \
    } catch (Eigen::eigen_assert_exception) {                                                                    \
      Eigen::internal::push_assert = false;                                                                      \
      VERIFY(true);                                                                                              \
    }                                                                                                            \
    Eigen::report_on_cerr_on_assert_failure = true;                                                              \
    Eigen::internal::push_assert = false;                                                                        \
  }
#endif  // EIGEN_EXCEPTIONS

#elif !defined(__CUDACC__) && !defined(__HIPCC__) && !defined(__SYCL_DEVICE_ONLY__)  // EIGEN_DEBUG_ASSERTS
#define eigen_assert(a)                               \
  if ((!(a)) && (!no_more_assert)) {                  \
    Eigen::no_more_assert = true;                     \
    if (report_on_cerr_on_assert_failure) {           \
      eigen_plain_assert(a);                          \
    } else {                                          \
      EIGEN_THROW_X(Eigen::eigen_assert_exception()); \
    }                                                 \
  }

#ifdef EIGEN_EXCEPTIONS
#define VERIFY_RAISES_ASSERT(a)                      \
  {                                                  \
    Eigen::no_more_assert = false;                   \
    Eigen::report_on_cerr_on_assert_failure = false; \
    try {                                            \
      a;                                             \
      VERIFY(Eigen::should_raise_an_assert&& #a);    \
    } catch (Eigen::eigen_assert_exception&) {       \
      VERIFY(true);                                  \
    }                                                \
    Eigen::report_on_cerr_on_assert_failure = true;  \
  }
#endif  // EIGEN_EXCEPTIONS
#endif  // EIGEN_DEBUG_ASSERTS

#ifndef VERIFY_RAISES_ASSERT
#define VERIFY_RAISES_ASSERT(a) std::cout << "Can't VERIFY_RAISES_ASSERT( " #a " ) with exceptions disabled\n";
#endif

#if !defined(__CUDACC__) && !defined(__HIPCC__) && !defined(SYCL_DEVICE_ONLY)
#define EIGEN_USE_CUSTOM_ASSERT
#endif

#else  // EIGEN_NO_ASSERTION_CHECKING

#define VERIFY_RAISES_ASSERT(a) \
  {}

#endif  // EIGEN_NO_ASSERTION_CHECKING

#if !defined(EIGEN_TESTING_CONSTEXPR) && !defined(EIGEN_TESTING_PLAINOBJECT_CTOR)
#define EIGEN_INTERNAL_DEBUGGING
#endif
#include <Eigen/Core>

inline void verify_impl(bool condition, const char* testname, const char* file, int line,
                        const char* condition_as_string) {
  if (!condition) {
    if (Eigen::g_test_level > 0) std::cerr << "WARNING: ";
    std::cerr << "Test " << testname << " failed in " << file << " (" << line << ")" << std::endl
              << "    " << condition_as_string << std::endl;
    std::cerr << "Stack:\n";
    const int test_stack_size = static_cast<int>(Eigen::g_test_stack.size());
    for (int i = test_stack_size - 1; i >= 0; --i) std::cerr << "  - " << Eigen::g_test_stack[i] << "\n";
    std::cerr << "\n";
    if (Eigen::g_test_level == 0) abort();
  }
}

#define VERIFY(a) ::verify_impl(a, g_test_stack.back().c_str(), __FILE__, __LINE__, EIGEN_MAKESTRING(a))

#define VERIFY_GE(a, b) ::verify_impl(a >= b, g_test_stack.back().c_str(), __FILE__, __LINE__, EIGEN_MAKESTRING(a >= b))
#define VERIFY_LE(a, b) ::verify_impl(a <= b, g_test_stack.back().c_str(), __FILE__, __LINE__, EIGEN_MAKESTRING(a <= b))

#define VERIFY_IS_EQUAL(a, b) VERIFY(test_is_equal(a, b, true))
#define VERIFY_IS_NOT_EQUAL(a, b) VERIFY(test_is_equal(a, b, false))
#define VERIFY_IS_APPROX(a, b) VERIFY(verifyIsApprox(a, b))
#define VERIFY_IS_NOT_APPROX(a, b) VERIFY(!test_isApprox(a, b))
#define VERIFY_IS_MUCH_SMALLER_THAN(a, b) VERIFY(test_isMuchSmallerThan(a, b))
#define VERIFY_IS_NOT_MUCH_SMALLER_THAN(a, b) VERIFY(!test_isMuchSmallerThan(a, b))
#define VERIFY_IS_APPROX_OR_LESS_THAN(a, b) VERIFY(test_isApproxOrLessThan(a, b))
#define VERIFY_IS_NOT_APPROX_OR_LESS_THAN(a, b) VERIFY(!test_isApproxOrLessThan(a, b))
#define VERIFY_IS_CWISE_EQUAL(a, b) VERIFY(verifyIsCwiseApprox(a, b, true))
#define VERIFY_IS_CWISE_APPROX(a, b) VERIFY(verifyIsCwiseApprox(a, b, false))

#define VERIFY_IS_UNITARY(a) VERIFY(test_isUnitary(a))

#define STATIC_CHECK(COND) EIGEN_STATIC_ASSERT((COND), EIGEN_INTERNAL_ERROR_PLEASE_FILE_A_BUG_REPORT)

#define CALL_SUBTEST(FUNC)                          \
  do {                                              \
    g_test_stack.push_back(EIGEN_MAKESTRING(FUNC)); \
    FUNC;                                           \
    g_test_stack.pop_back();                        \
  } while (0)

// Forward declarations to avoid ICC warnings
#if EIGEN_COMP_ICC

template <typename T>
std::string type_name();

namespace Eigen {

template <typename T, typename U>
bool test_is_equal(const T& actual, const U& expected, bool expect_equal = true);

}  // end namespace Eigen

#endif  // EIGEN_COMP_ICC

namespace Eigen {

template <typename T1, typename T2>
std::enable_if_t<internal::is_same<T1, T2>::value, bool> is_same_type(const T1&, const T2&) {
  return true;
}

template <typename T>
inline typename NumTraits<T>::Real test_precision() {
  return NumTraits<T>::dummy_precision();
}
template <>
inline float test_precision<float>() {
  return 1e-3f;
}
template <>
inline double test_precision<double>() {
  return 1e-6;
}
template <>
inline long double test_precision<long double>() {
  return 1e-6l;
}
template <>
inline float test_precision<std::complex<float> >() {
  return test_precision<float>();
}
template <>
inline double test_precision<std::complex<double> >() {
  return test_precision<double>();
}
template <>
inline long double test_precision<std::complex<long double> >() {
  return test_precision<long double>();
}

#define EIGEN_TEST_SCALAR_TEST_OVERLOAD(TYPE)                                          \
  inline bool test_isApprox(TYPE a, TYPE b) {                                          \
    return numext::equal_strict(a, b) || ((numext::isnan)(a) && (numext::isnan)(b)) || \
           (internal::isApprox(a, b, test_precision<TYPE>()));                         \
  }                                                                                    \
  inline bool test_isCwiseApprox(TYPE a, TYPE b, bool exact) {                         \
    return numext::equal_strict(a, b) || ((numext::isnan)(a) && (numext::isnan)(b)) || \
           (!exact && internal::isApprox(a, b, test_precision<TYPE>()));               \
  }                                                                                    \
  inline bool test_isMuchSmallerThan(TYPE a, TYPE b) {                                 \
    return internal::isMuchSmallerThan(a, b, test_precision<TYPE>());                  \
  }                                                                                    \
  inline bool test_isApproxOrLessThan(TYPE a, TYPE b) {                                \
    return internal::isApproxOrLessThan(a, b, test_precision<TYPE>());                 \
  }

EIGEN_TEST_SCALAR_TEST_OVERLOAD(short)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(unsigned short)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(int)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(unsigned int)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(long)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(unsigned long)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(long long)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(unsigned long long)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(float)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(double)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(half)
EIGEN_TEST_SCALAR_TEST_OVERLOAD(bfloat16)

#undef EIGEN_TEST_SCALAR_TEST_OVERLOAD

#ifndef EIGEN_TEST_NO_COMPLEX
inline bool test_isApprox(const std::complex<float>& a, const std::complex<float>& b) {
  return internal::isApprox(a, b, test_precision<std::complex<float> >());
}
inline bool test_isMuchSmallerThan(const std::complex<float>& a, const std::complex<float>& b) {
  return internal::isMuchSmallerThan(a, b, test_precision<std::complex<float> >());
}

inline bool test_isApprox(const std::complex<double>& a, const std::complex<double>& b) {
  return internal::isApprox(a, b, test_precision<std::complex<double> >());
}
inline bool test_isMuchSmallerThan(const std::complex<double>& a, const std::complex<double>& b) {
  return internal::isMuchSmallerThan(a, b, test_precision<std::complex<double> >());
}

#ifndef EIGEN_TEST_NO_LONGDOUBLE
inline bool test_isApprox(const std::complex<long double>& a, const std::complex<long double>& b) {
  return internal::isApprox(a, b, test_precision<std::complex<long double> >());
}
inline bool test_isMuchSmallerThan(const std::complex<long double>& a, const std::complex<long double>& b) {
  return internal::isMuchSmallerThan(a, b, test_precision<std::complex<long double> >());
}
#endif
#endif

#ifndef EIGEN_TEST_NO_LONGDOUBLE
inline bool test_isApprox(const long double& a, const long double& b) {
  bool ret = internal::isApprox(a, b, test_precision<long double>());
  if (!ret)
    std::cerr << std::endl << "    actual   = " << a << std::endl << "    expected = " << b << std::endl << std::endl;
  return ret;
}

inline bool test_isMuchSmallerThan(const long double& a, const long double& b) {
  return internal::isMuchSmallerThan(a, b, test_precision<long double>());
}
inline bool test_isApproxOrLessThan(const long double& a, const long double& b) {
  return internal::isApproxOrLessThan(a, b, test_precision<long double>());
}
#endif  // EIGEN_TEST_NO_LONGDOUBLE

// test_relative_error returns the relative difference between a and b as a real scalar as used in isApprox.
template <typename T1, typename T2>
typename NumTraits<typename T1::RealScalar>::NonInteger test_relative_error(const EigenBase<T1>& a,
                                                                            const EigenBase<T2>& b) {
  using std::sqrt;
  typedef typename NumTraits<typename T1::RealScalar>::NonInteger RealScalar;
  typename internal::nested_eval<T1, 2>::type ea(a.derived());
  typename internal::nested_eval<T2, 2>::type eb(b.derived());
  return sqrt(RealScalar((ea.matrix() - eb.matrix()).cwiseAbs2().sum()) /
              RealScalar((std::min)(eb.cwiseAbs2().sum(), ea.cwiseAbs2().sum())));
}

template <typename T1, typename T2>
typename T1::RealScalar test_relative_error(const T1& a, const T2& b, const typename T1::Coefficients* = 0) {
  return test_relative_error(a.coeffs(), b.coeffs());
}

template <typename T1, typename T2>
typename T1::Scalar test_relative_error(const T1& a, const T2& b, const typename T1::MatrixType* = 0) {
  return test_relative_error(a.matrix(), b.matrix());
}

template <typename S, int D>
S test_relative_error(const Translation<S, D>& a, const Translation<S, D>& b) {
  return test_relative_error(a.vector(), b.vector());
}

template <typename S, int D, int O>
S test_relative_error(const ParametrizedLine<S, D, O>& a, const ParametrizedLine<S, D, O>& b) {
  return (std::max)(test_relative_error(a.origin(), b.origin()), test_relative_error(a.origin(), b.origin()));
}

template <typename S, int D>
S test_relative_error(const AlignedBox<S, D>& a, const AlignedBox<S, D>& b) {
  return (std::max)(test_relative_error((a.min)(), (b.min)()), test_relative_error((a.max)(), (b.max)()));
}

template <typename Derived>
class SparseMatrixBase;
template <typename T1, typename T2>
typename T1::RealScalar test_relative_error(const MatrixBase<T1>& a, const SparseMatrixBase<T2>& b) {
  return test_relative_error(a, b.toDense());
}

template <typename Derived>
class SparseMatrixBase;
template <typename T1, typename T2>
typename T1::RealScalar test_relative_error(const SparseMatrixBase<T1>& a, const MatrixBase<T2>& b) {
  return test_relative_error(a.toDense(), b);
}

template <typename Derived>
class SparseMatrixBase;
template <typename T1, typename T2>
typename T1::RealScalar test_relative_error(const SparseMatrixBase<T1>& a, const SparseMatrixBase<T2>& b) {
  return test_relative_error(a.toDense(), b.toDense());
}

template <typename T1, typename T2>
typename NumTraits<typename NumTraits<T1>::Real>::NonInteger test_relative_error(
    const T1& a, const T2& b, std::enable_if_t<internal::is_arithmetic<typename NumTraits<T1>::Real>::value, T1>* = 0) {
  typedef typename NumTraits<typename NumTraits<T1>::Real>::NonInteger RealScalar;
  return numext::sqrt(RealScalar(numext::abs2(a - b)) /
                      (numext::mini)(RealScalar(numext::abs2(a)), RealScalar(numext::abs2(b))));
}

template <typename T>
T test_relative_error(const Rotation2D<T>& a, const Rotation2D<T>& b) {
  return test_relative_error(a.angle(), b.angle());
}

template <typename T>
T test_relative_error(const AngleAxis<T>& a, const AngleAxis<T>& b) {
  return (std::max)(test_relative_error(a.angle(), b.angle()), test_relative_error(a.axis(), b.axis()));
}

template <typename Type1, typename Type2>
inline bool test_isApprox(const Type1& a, const Type2& b, typename Type1::Scalar* = 0)  // Enabled for Eigen's type only
{
  return a.isApprox(b, test_precision<typename Type1::Scalar>());
}

// get_test_precision is a small wrapper to test_precision allowing to return the scalar precision for either scalars or
// expressions
template <typename T>
typename NumTraits<typename T::Scalar>::Real get_test_precision(const T&, const typename T::Scalar* = 0) {
  return test_precision<typename NumTraits<typename T::Scalar>::Real>();
}

template <typename T>
typename NumTraits<T>::Real get_test_precision(
    const T&, std::enable_if_t<internal::is_arithmetic<typename NumTraits<T>::Real>::value, T>* = 0) {
  return test_precision<typename NumTraits<T>::Real>();
}

// verifyIsApprox is a wrapper to test_isApprox that outputs the relative difference magnitude if the test fails.
template <typename Type1, typename Type2>
inline bool verifyIsApprox(const Type1& a, const Type2& b) {
  bool ret = test_isApprox(a, b);
  if (!ret) {
    std::cerr << "Difference too large wrt tolerance " << get_test_precision(a)
              << ", relative error is: " << test_relative_error(a, b) << std::endl;
  }
  return ret;
}

// verifyIsCwiseApprox is a wrapper to test_isCwiseApprox that outputs the relative difference magnitude if the test
// fails.
template <typename Type1, typename Type2>
inline bool verifyIsCwiseApprox(const Type1& a, const Type2& b, bool exact) {
  bool ret = test_isCwiseApprox(a, b, exact);
  if (!ret) {
    if (exact) {
      std::cerr << "Values are not an exact match";
    } else {
      std::cerr << "Difference too large wrt tolerance " << get_test_precision(a);
    }
    std::cerr << ", relative error is: " << test_relative_error(a, b) << std::endl;
  }
  return ret;
}

// The idea behind this function is to compare the two scalars a and b where
// the scalar ref is a hint about the expected order of magnitude of a and b.
// WARNING: the scalar a and b must be positive
// Therefore, if for some reason a and b are very small compared to ref,
// we won't issue a false negative.
// This test could be: abs(a-b) <= eps * ref
// However, it seems that simply comparing a+ref and b+ref is more sensitive to true error.
template <typename Scalar, typename ScalarRef>
inline bool test_isApproxWithRef(const Scalar& a, const Scalar& b, const ScalarRef& ref) {
  return test_isApprox(a + ref, b + ref);
}

template <typename Derived1, typename Derived2>
inline bool test_isMuchSmallerThan(const MatrixBase<Derived1>& m1, const MatrixBase<Derived2>& m2) {
  return m1.isMuchSmallerThan(m2, test_precision<typename internal::traits<Derived1>::Scalar>());
}

template <typename Derived>
inline bool test_isMuchSmallerThan(const MatrixBase<Derived>& m,
                                   const typename NumTraits<typename internal::traits<Derived>::Scalar>::Real& s) {
  return m.isMuchSmallerThan(s, test_precision<typename internal::traits<Derived>::Scalar>());
}

template <typename Derived>
inline bool test_isUnitary(const MatrixBase<Derived>& m) {
  return m.isUnitary(test_precision<typename internal::traits<Derived>::Scalar>());
}

// Checks component-wise, works with infs and nans.
template <typename Derived1, typename Derived2>
bool test_isCwiseApprox(const DenseBase<Derived1>& m1, const DenseBase<Derived2>& m2, bool exact) {
  if (m1.rows() != m2.rows()) {
    return false;
  }
  if (m1.cols() != m2.cols()) {
    return false;
  }
  for (Index r = 0; r < m1.rows(); ++r) {
    for (Index c = 0; c < m1.cols(); ++c) {
      if (m1(r, c) != m2(r, c) && !((numext::isnan)(m1(r, c)) && (numext::isnan)(m2(r, c))) &&
          (exact || !test_isApprox(m1(r, c), m2(r, c)))) {
        return false;
      }
    }
  }
  return true;
}

template <typename Derived1, typename Derived2>
bool test_isCwiseApprox(const SparseMatrixBase<Derived1>& m1, const SparseMatrixBase<Derived2>& m2, bool exact) {
  return test_isCwiseApprox(m1.toDense(), m2.toDense(), exact);
}

template <typename T, typename U>
bool test_is_equal(const T& actual, const U& expected, bool expect_equal) {
  if (numext::equal_strict(actual, expected) == expect_equal) return true;
  // false:
  std::cerr << "\n    actual   = " << actual << "\n    expected " << (expect_equal ? "= " : "!=") << expected << "\n\n";
  return false;
}

/**
 * Check if number is "not a number" (NaN).
 *
 * @tparam T input type
 * @param x input value
 * @return true, if input value is "not a number" (NaN)
 */
template <typename T>
bool isNotNaN(const T& x) {
  return x == x;
}

/**
 * Check if number is plus infinity.
 *
 * @tparam T input type
 * @param x input value
 * @return true, if input value is plus infinity
 */
template <typename T>
bool isPlusInf(const T& x) {
  return x > NumTraits<T>::highest();
}

/**
 * Check if number is minus infinity.
 *
 * @tparam T input type
 * @param x input value
 * @return true, if input value is minus infinity
 */
template <typename T>
bool isMinusInf(const T& x) {
  return x < NumTraits<T>::lowest();
}

}  // end namespace Eigen

#include "random_matrix_helper.h"

template <typename T>
struct GetDifferentType;

template <>
struct GetDifferentType<float> {
  typedef double type;
};
template <>
struct GetDifferentType<double> {
  typedef float type;
};
template <typename T>
struct GetDifferentType<std::complex<T> > {
  typedef std::complex<typename GetDifferentType<T>::type> type;
};

template <typename T>
std::string type_name(T) {
  return typeid(T).name();
}
template <>
std::string type_name<float>(float) {
  return "float";
}
template <>
std::string type_name<double>(double) {
  return "double";
}
template <>
std::string type_name<long double>(long double) {
  return "long double";
}
template <>
std::string type_name<Eigen::half>(Eigen::half) {
  return "half";
}
template <>
std::string type_name<Eigen::bfloat16>(Eigen::bfloat16) {
  return "bfloat16";
}
template <>
std::string type_name<int8_t>(int8_t) {
  return "int8_t";
}
template <>
std::string type_name<int16_t>(int16_t) {
  return "int16_t";
}
template <>
std::string type_name<int32_t>(int32_t) {
  return "int32_t";
}
template <>
std::string type_name<int64_t>(int64_t) {
  return "int64_t";
}
template <>
std::string type_name<uint8_t>(uint8_t) {
  return "uint8_t";
}
template <>
std::string type_name<uint16_t>(uint16_t) {
  return "uint16_t";
}
template <>
std::string type_name<uint32_t>(uint32_t) {
  return "uint32_t";
}
template <>
std::string type_name<uint64_t>(uint64_t) {
  return "uint64_t";
}
template <>
std::string type_name<std::complex<float> >(std::complex<float>) {
  return "complex<float>";
}
template <>
std::string type_name<std::complex<double> >(std::complex<double>) {
  return "complex<double>";
}
template <>
std::string type_name<std::complex<long double> >(std::complex<long double>) {
  return "complex<long double>";
}
template <>
std::string type_name<std::complex<int> >(std::complex<int>) {
  return "complex<int>";
}
template <typename T>
std::string type_name() {
  return type_name(T());
}

using namespace Eigen;

/**
 * Set number of repetitions for unit test from input string.
 *
 * @param str input string
 */
inline void set_repeat_from_string(const char* str) {
  errno = 0;
  g_repeat = int(strtoul(str, 0, 10));
  if (errno || g_repeat <= 0) {
    std::cout << "Invalid repeat value " << str << std::endl;
    exit(EXIT_FAILURE);
  }
  g_has_set_repeat = true;
}

/**
 * Set seed for randomized unit tests from input string.
 *
 * @param str input string
 */
inline void set_seed_from_string(const char* str) {
  errno = 0;
  g_seed = int(strtoul(str, 0, 10));
  if (errno || g_seed == 0) {
    std::cout << "Invalid seed value " << str << std::endl;
    exit(EXIT_FAILURE);
  }
  g_has_set_seed = true;
}

int main(int argc, char* argv[]) {
  g_has_set_repeat = false;
  g_has_set_seed = false;
  bool need_help = false;

  for (int i = 1; i < argc; i++) {
    if (argv[i][0] == 'r') {
      if (g_has_set_repeat) {
        std::cout << "Argument " << argv[i] << " conflicting with a former argument" << std::endl;
        return 1;
      }
      set_repeat_from_string(argv[i] + 1);
    } else if (argv[i][0] == 's') {
      if (g_has_set_seed) {
        std::cout << "Argument " << argv[i] << " conflicting with a former argument" << std::endl;
        return 1;
      }
      set_seed_from_string(argv[i] + 1);
    } else {
      need_help = true;
    }
  }

  if (need_help) {
    std::cout << "This test application takes the following optional arguments:" << std::endl;
    std::cout << "  rN     Repeat each test N times (default: " << DEFAULT_REPEAT << ")" << std::endl;
    std::cout << "  sN     Use N as seed for random numbers (default: based on current time)" << std::endl;
    std::cout << std::endl;
    std::cout << "If defined, the environment variables EIGEN_REPEAT and EIGEN_SEED" << std::endl;
    std::cout << "will be used as default values for these parameters." << std::endl;
    return 1;
  }

  char* env_EIGEN_REPEAT = getenv("EIGEN_REPEAT");
  if (!g_has_set_repeat && env_EIGEN_REPEAT) set_repeat_from_string(env_EIGEN_REPEAT);
  char* env_EIGEN_SEED = getenv("EIGEN_SEED");
  if (!g_has_set_seed && env_EIGEN_SEED) set_seed_from_string(env_EIGEN_SEED);

  if (!g_has_set_seed) g_seed = (unsigned int)time(NULL);
  if (!g_has_set_repeat) g_repeat = DEFAULT_REPEAT;

  std::cout << "Initializing random number generator with seed " << g_seed << std::endl;
  std::stringstream ss;
  ss << "Seed: " << g_seed;
  g_test_stack.push_back(ss.str());
  srand(g_seed);
  std::cout << "Repeating each test " << g_repeat << " times" << std::endl;

  VERIFY(EigenTest::all().size() > 0);

  for (std::size_t i = 0; i < EigenTest::all().size(); ++i) {
    const EigenTest& current_test = *EigenTest::all()[i];
    Eigen::g_test_stack.push_back(current_test.name());
    current_test();
    Eigen::g_test_stack.pop_back();
  }

  return 0;
}

// These warning are disabled here such that they are still ON when parsing Eigen's header files.
#if defined __INTEL_COMPILER
// remark #383: value copied to temporary, reference to temporary used
//  -> this warning is raised even for legal usage as: g_test_stack.push_back("foo"); where g_test_stack is a
//  std::vector<std::string>
// remark #1418: external function definition with no prior declaration
//  -> this warning is raised for all our test functions. Declaring them static would fix the issue.
// warning #279: controlling expression is constant
// remark #1572: floating-point equality and inequality comparisons are unreliable
#pragma warning disable 279 383 1418 1572
#endif

#ifdef _MSC_VER
// 4503 - decorated name length exceeded, name was truncated
#pragma warning(disable : 4503)
#endif

#include "gpu_test_helper.h"

#ifndef EIGEN_TEST_MAX_SIZE
#define EIGEN_TEST_MAX_SIZE 320
#endif
