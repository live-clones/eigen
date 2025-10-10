/* This contains a limited subset of the typedefs exposed by f2c
   for use by the Eigen BLAS C-only implementation.
*/

#ifndef __EIGEN_DATATYPES_H__
#define __EIGEN_DATATYPES_H__

#if defined(_WIN32)
#if defined(EIGEN_BLAS_BUILD_DLL)
#define EIGEN_BLAS_API __declspec(dllexport)
#elif defined(EIGEN_BLAS_LINK_DLL)
#define EIGEN_BLAS_API __declspec(dllimport)
#else
#define EIGEN_BLAS_API
#endif
#else
#define EIGEN_BLAS_API
#endif

typedef int integer;
typedef unsigned int uinteger;
typedef float real;
typedef double doublereal;
typedef struct {
  real r, i;
} complex;
typedef struct {
  doublereal r, i;
} doublecomplex;
typedef int logical;

#define abs(x) ((x) >= 0 ? (x) : -(x))
#define dabs(x) (doublereal) abs(x)
#define min(a, b) ((a) <= (b) ? (a) : (b))
#define max(a, b) ((a) >= (b) ? (a) : (b))
#define dmin(a, b) (doublereal) min(a, b)
#define dmax(a, b) (doublereal) max(a, b)

#endif
