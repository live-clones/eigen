
#ifndef EIGEN_CORE_ARCH_HVX_GENERAL_BLOCK_KERNEL_H
#define EIGEN_CORE_ARCH_HVX_GENERAL_BLOCK_KERNEL_H

namespace Eigen {
namespace internal {

template <bool ConjLhs_, bool ConjRhs_, int PacketSize_>
class gebp_traits<float, float, ConjLhs_, ConjRhs_, Architecture::Target,
                  PacketSize_>
    : public gebp_traits<float, float, ConjLhs_, ConjRhs_,
                         Architecture::Generic, PacketSize_> {
 public:
  typedef Packet32qf AccPacket;

  EIGEN_STRONG_INLINE void initAcc(Packet32qf& p) { p = pzero<Packet32qf>(p); }

  template <typename LaneIdType>
  EIGEN_STRONG_INLINE void madd(const Packet32f& a, const Packet32f& b,
                                Packet32qf& c, Packet32f& /*tmp*/,
                                const LaneIdType&) const {
    c = pmadd(a, b, c);
  }

  template <typename LaneIdType>
  EIGEN_STRONG_INLINE void madd(const Packet32f& a,
                                const QuadPacket<Packet32f>& b, Packet32qf& c,
                                Packet32f& tmp, const LaneIdType& lane) const {
    madd(a, b.get(lane), c, tmp, lane);
  }

  EIGEN_STRONG_INLINE void acc(const Packet32qf& c, const Packet32f& alpha,
                               Packet32f& r) const {
    r = pmadd(c, alpha, r);
  }
};

}  // end namespace internal
}  // end namespace Eigen

#endif  // EIGEN_CORE_ARCH_HVX_GENERAL_BLOCK_KERNEL_H
