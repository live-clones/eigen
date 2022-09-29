#ifndef EIGEN_LIBDIVIDESUPPORT_H
#define EIGEN_LIBDIVIDESUPPORT_H

#ifdef LIBDIVIDE_H
EIGEN_STATIC_ASSERT(std::false_type::value,
                    INCLUDE_THIS_FILE_BEFORE_LIBDIVIDE.H)
#endif  // LIBDIVIDE_H

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
        static constexpr bool value =        
            ( Eigen::NumTraits<SrcScalar>::IsInteger && Eigen::NumTraits<DstScalar>::IsInteger) &&
            (!Eigen::NumTraits<SrcScalar>::IsSigned  || Eigen::NumTraits<DstScalar>::IsSigned ) &&
            ( Eigen::NumTraits<SrcScalar>::digits()  <= Eigen::NumTraits<DstScalar>::digits() );
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
    static constexpr bool is_valid_libdivide_op =
        is_libdivide_type_v<Scalar> && integer_is_representable_v<DivisorScalar, Scalar>;

    template <typename Scalar, typename DivisorScalar>
    struct scalar_libdivide_op {
      EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
      scalar_libdivide_op(const DivisorScalar& divisor)
          : m_divisor(static_cast<Scalar>(divisor)) {
        EIGEN_STATIC_ASSERT((is_valid_libdivide_op<Scalar, DivisorScalar>),
                            SCALAR_MUST_BE_LIBDIVIDE_TYPE_AND_DIVISOR_MUST_BE_REPRESENTABLE_AS_SCALAR);
        eigen_assert(divisor != 0);
      }
      EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
      Scalar operator()(const Scalar& a) const { return m_divisor.divide(a); }
      template <typename Packet>
      EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
      Packet packetOp  (const Packet& a) const { return m_divisor.divide(a); }
     protected:
      const libdivide::divider<Scalar> m_divisor;
    };

template <typename Scalar, typename DivisorScalar>
    struct functor_traits<scalar_libdivide_op<Scalar, DivisorScalar>> {
      enum {
        PacketAccess = !is_same<typename packet_traits<Scalar>::type, Scalar>::value,
        Cost = scalar_div_cost<Scalar, PacketAccess>::value / 2
      };
    };
}  // namespace internal

template <typename Derived, typename DivisorScalar>
using CwiseLibdivideReturnType = std::enable_if_t<internal::is_valid_libdivide_op<typename Derived::Scalar, DivisorScalar>,
                                                  CwiseUnaryOp<internal::scalar_libdivide_op<typename Derived::Scalar, DivisorScalar>, const Derived>>;

template <typename Derived, typename DivisorScalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
CwiseLibdivideReturnType<Derived, DivisorScalar> libdivideQuotient(const DenseBase<Derived>& lhs, const DivisorScalar& rhs) {
    const internal::scalar_libdivide_op<typename Derived::Scalar, DivisorScalar> libdivideOp(rhs);
    const CwiseLibdivideReturnType<Derived, DivisorScalar> libdivideExpr(lhs.derived(), libdivideOp);
    return libdivideExpr;
}
template <typename Derived, typename DivisorScalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
CwiseLibdivideReturnType<Derived, DivisorScalar> operator/(const ArrayBase<Derived>& lhs, const DivisorScalar& rhs) {
    return libdivideQuotient(lhs.derived(), rhs);
}
template <typename Derived, typename DivisorScalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
CwiseLibdivideReturnType<Derived, DivisorScalar> operator/(const MatrixBase<Derived>& lhs, const DivisorScalar& rhs) {
    return libdivideQuotient(lhs.derived(), rhs);
}

}  // namespace Eigen

#endif  // EIGEN_LIBDIVIDESUPPORT_H