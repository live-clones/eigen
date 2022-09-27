#ifndef EIGEN_LIBDIVIDESUPPORT_H
#define EIGEN_LIBDIVIDESUPPORT_H

#ifdef LIBDIVIDE_H
EIGEN_STATIC_ASSERT(std::false_type::value,
                    INCLUDE_THIS_FILE_BEFORE_LIBDIVIDE.H)
#endif  // LIBDIVIDE_H

namespace Eigen {
namespace internal {
#define IS_LIBDIVIDE_TYPE(T)(                      \
   Eigen::internal::is_same<T, uint16_t>::value || \
   Eigen::internal::is_same<T,  int16_t>::value || \
   Eigen::internal::is_same<T, uint32_t>::value || \
   Eigen::internal::is_same<T,  int32_t>::value || \
   Eigen::internal::is_same<T, uint64_t>::value || \
   Eigen::internal::is_same<T,  int64_t>::value)

#define SRC_IS_REPRESENTABLE_AS_DST(SRC, DST)                                        \
   ( Eigen::NumTraits<SRC>::IsInteger && Eigen::NumTraits<DST>::IsInteger) && \
   (!Eigen::NumTraits<SRC>::IsSigned  || Eigen::NumTraits<DST>::IsSigned)  && \
   ( Eigen::NumTraits<SRC>::digits()  <= Eigen::NumTraits<DST>::digits())

#define IS_LIBDIVIDE_COMPATIBLE(NUM, DEN) (IS_LIBDIVIDE_TYPE(NUM)) && (SRC_IS_REPRESENTABLE_AS_DST(DEN, NUM))
}  // namespace internal
}  // namespace Eigen

#ifdef EIGEN_VECTORIZE_SSE2
#define LIBDIVIDE_SSE2
#endif  // EIGEN_VECTORIZE_SSE2

#ifdef EIGEN_VECTORIZE_AVX2
#define LIBDIVIDE_AVX2
#endif  // EIGEN_VECTORIZE_AVX2

#ifdef EIGEN_VECTORIZE_AVX512
#define LIBDIVIDE_AVX512
#endif  // EIGEN_VECTORIZE_AVX512

#ifdef EIGEN_VECTORIZE_NEON
#define LIBDIVIDE_NEON
#endif  // EIGEN_VECTORIZE_NEON

#define LIBDIVIDE_INLINE EIGEN_STRONG_INLINE

#include <libdivide.h>

namespace Eigen {
namespace internal {

    template <typename SrcScalar, typename DstScalar>
    struct integer_is_representable {
        static constexpr bool SrcIsInteger = Eigen::NumTraits<SrcScalar>::IsInteger;
        static constexpr bool SrcIsSigned =  Eigen::NumTraits<SrcScalar>::IsSigned;
        static constexpr int  SrcDigits =    Eigen::NumTraits<SrcScalar>::digits();
        static constexpr bool DstIsInteger = Eigen::NumTraits<DstScalar>::IsInteger;
        static constexpr bool DstIsSigned =  Eigen::NumTraits<DstScalar>::IsSigned;
        static constexpr int  DstDigits =    Eigen::NumTraits<DstScalar>::digits();
        static constexpr bool value =        
            (SrcIsInteger && DstIsInteger) &&
            (!SrcIsSigned || DstIsSigned) &&
            (SrcDigits <= DstDigits);
    };

    template <typename SrcScalar, typename DstScalar>
    static constexpr bool integer_is_representable_v =
        integer_is_representable<SrcScalar, DstScalar>::value;

    template <typename Scalar>
    struct is_libdivide_type {
        static constexpr bool value = false;
    };
    template <>
    struct is_libdivide_type<uint16_t> {
        static constexpr bool value = true;
    };
    template <>
    struct is_libdivide_type<int16_t> {
        static constexpr bool value = true;
    };
    template <>
    struct is_libdivide_type<uint32_t> {
        static constexpr bool value = true;
    };
    template <>
    struct is_libdivide_type<int32_t> {
        static constexpr bool value = true;
    };
    template <>
    struct is_libdivide_type<uint64_t> {
        static constexpr bool value = true;
    };
    template <>
    struct is_libdivide_type<int64_t> {
        static constexpr bool value = true;
    };

    template <typename Scalar>
    static constexpr bool is_libdivide_type_v = is_libdivide_type<Scalar>::value;

    template <typename Scalar, typename DivisorScalar>
    static constexpr bool is_libdivide_compatible_v = is_libdivide_type_v<Scalar> && integer_is_representable_v<DivisorScalar, Scalar>;

template <typename Scalar, typename DivisorScalar>
struct scalar_libdivide_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  scalar_libdivide_op(const DivisorScalar& divisor)
      : m_divisor(static_cast<Scalar>(divisor)) {
    EIGEN_STATIC_ASSERT((is_libdivide_compatible_v<Scalar,DivisorScalar>),
                        SCALAR_MUST_BE_LIBDIVIDE_TYPE_AND_DIVISOR_MUST_BE_REPRESENTABLE_AS_SCALAR);
    eigen_assert(divisor != 0);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& a) const {
    return m_divisor.divide(a);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a) const {
    return m_divisor.divide(a);
  }

 protected:
  const libdivide::divider<Scalar> m_divisor;
};

template <typename Scalar, typename DivisorScalar>
struct functor_traits<scalar_libdivide_op<Scalar, DivisorScalar>> {
  enum {

    PacketAccess = !is_same<typename packet_traits<Scalar>::type,
                            Scalar>::value,  // if the packet is defined, then
                                             // we're good to go
    Cost = scalar_div_cost<Scalar,
                           PacketAccess>::value  // its less, but close enough
  };
};
}  // namespace internal

template <typename Derived, typename DivisorScalar>
using CwiseLibdivideReturnType = std::enable_if_t<internal::is_libdivide_type_v<typename Derived::Scalar> && internal::integer_is_representable_v<DivisorScalar, typename Derived::Scalar>,
CwiseUnaryOp<internal::scalar_libdivide_op<typename Derived::Scalar, DivisorScalar>,const Derived>>;

template <typename Derived, typename DivisorScalar>
CwiseLibdivideReturnType<Derived, DivisorScalar> operator/(
    const ArrayBase<Derived>& lhs, const DivisorScalar& rhs) {
  typedef typename Derived::Scalar Scalar;
  return CwiseLibdivideReturnType<Derived, DivisorScalar>(
      lhs.derived(), internal::scalar_libdivide_op<Scalar, DivisorScalar>(rhs));
}
template <typename Derived, typename DivisorScalar>
CwiseLibdivideReturnType<Derived, DivisorScalar> operator/(
    const MatrixBase<Derived>& lhs, const DivisorScalar& rhs) {
  typedef typename Derived::Scalar Scalar;
  return CwiseLibdivideReturnType<Derived, DivisorScalar>(
      lhs.derived(), internal::scalar_libdivide_op<Scalar, DivisorScalar>(rhs));
}
}  // namespace Eigen

#endif  // EIGEN_LIBDIVIDESUPPORT_H