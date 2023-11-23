// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2023 Zang Ruochen <zangruochen@loongson.cn>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_COMPLEX_LSX_H
#define EIGEN_COMPLEX_LSX_H

// IWYU pragma: private
#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

//---------- float ----------
struct Packet2cf
{
  EIGEN_STRONG_INLINE Packet2cf() {}
  EIGEN_STRONG_INLINE explicit Packet2cf(const __m128& a) : v(a) {}
  Packet4f v;
};

template<> struct packet_traits<std::complex<float> >  : default_packet_traits
{
  typedef Packet2cf type;
  typedef Packet2cf half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 2,

    HasAdd    = 1,
    HasSub    = 1,
    HasMul    = 1,
    HasDiv    = 1,
    HasNegate = 1,
    HasSqrt   = 1,
    HasAbs    = 0,
    HasAbs2   = 0,
    HasMin    = 0,
    HasMax    = 0,
    HasSetLinear = 0
  };
};

template<> struct unpacket_traits<Packet2cf> {
  typedef std::complex<float> type;
  typedef Packet2cf half;
  typedef Packet4f as_real;
  enum {
    size=2,
    alignment=Aligned16,
    vectorizable=true,
    masked_load_available=false,
    masked_store_available=false
  };
};

template<> EIGEN_STRONG_INLINE Packet2cf padd<Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(__lsx_vfadd_s(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet2cf psub<Packet2cf>(const Packet2cf& a, const Packet2cf& b) { return Packet2cf(__lsx_vfsub_s(a.v,b.v)); }

template<> EIGEN_STRONG_INLINE Packet2cf pnegate(const Packet2cf& a)
{
  const uint32_t b[4]= {0x80000000u,0x80000000u,0x80000000u,0x80000000u};
  Packet4i mask = (Packet4i)__lsx_vld(b,0);
  Packet2cf res;
  res.v=(Packet4f)__lsx_vxor_v((__m128i)a.v,mask);
  return res;
}

template<> EIGEN_STRONG_INLINE Packet2cf pconj(const Packet2cf& a)
{
  const uint32_t b[4] = {0x00000000u, 0x80000000u, 0x00000000u, 0x80000000u};
  Packet4i mask = (__m128i)__lsx_vld(b, 0);
  Packet2cf res;
  res.v=(Packet4f)__lsx_vxor_v((__m128i)a.v,mask);
  return res;
}

template <>
EIGEN_STRONG_INLINE Packet2cf pmul<Packet2cf>(const Packet2cf& a, const Packet2cf& b) {
	Packet4f part0_tmp = (Packet4f)__lsx_vfmul_s(a.v, b.v);
	Packet4f part0 = __lsx_vfsub_s(part0_tmp, (__m128)__lsx_vshuf4i_w(part0_tmp, 0x31));
	Packet4f part1_tmp = __lsx_vfmul_s((__m128)__lsx_vshuf4i_w(a.v, 0xb1), b.v);
	Packet4f part1 = __lsx_vfadd_s(part1_tmp, (__m128)__lsx_vshuf4i_w(part1_tmp, 0x31));
	Packet2cf res;
	res.v = (Packet4f)__lsx_vpackev_w((__m128i)part1, (__m128i)part0);
	return res;
}

template<> EIGEN_STRONG_INLINE Packet2cf ptrue  <Packet2cf>(const Packet2cf& a) { return Packet2cf(ptrue(Packet4f(a.v))); }

template<> EIGEN_STRONG_INLINE Packet2cf pand<Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  Packet2cf res;
  res.v = (Packet4f)__lsx_vand_v((__m128i)a.v,(__m128i)b.v);
  return res;
}

template<> EIGEN_STRONG_INLINE Packet2cf por<Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  Packet2cf res;
  res.v = (Packet4f)__lsx_vor_v((__m128i)a.v,(__m128i)b.v);
  return res;
}

template<> EIGEN_STRONG_INLINE Packet2cf pxor<Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  Packet2cf res;
  res.v = (Packet4f)__lsx_vxor_v((__m128i)a.v,(__m128i)b.v);
  return res;
}

template<> EIGEN_STRONG_INLINE Packet2cf pandnot<Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  Packet2cf res;
  res.v = (Packet4f)__lsx_vandn_v((__m128i)b.v,(__m128i)a.v);
  return res;
}

template<> EIGEN_STRONG_INLINE Packet2cf pload <Packet2cf> (const std::complex<float>* from) { EIGEN_DEBUG_ALIGNED_LOAD return Packet2cf(pload<Packet4f>(&numext::real_ref(*from))); }

template<> EIGEN_STRONG_INLINE Packet2cf ploadu<Packet2cf> (const std::complex<float>* from) { EIGEN_DEBUG_UNALIGNED_LOAD return Packet2cf(ploadu<Packet4f>(&numext::real_ref(*from))); }

template<> EIGEN_STRONG_INLINE Packet2cf pset1<Packet2cf>(const std::complex<float>&  from)
{
  float f0 = from.real(), f1 = from.imag();
  Packet4f re = {f0, f0, f0, f0};
  Packet4f im = {f1, f1, f1, f1};
  return Packet2cf((Packet4f)__lsx_vilvl_w((__m128i)im, (__m128i)re));
}

template<> EIGEN_STRONG_INLINE Packet2cf ploaddup<Packet2cf>(const std::complex<float>* from) { return pset1<Packet2cf>(*from); }

template<> EIGEN_STRONG_INLINE void pstore <std::complex<float> >(std::complex<float> *   to, const Packet2cf& from) { EIGEN_DEBUG_ALIGNED_STORE pstore(&numext::real_ref(*to), Packet4f(from.v)); }

template<> EIGEN_STRONG_INLINE void pstoreu<std::complex<float> >(std::complex<float> *   to, const Packet2cf& from) { EIGEN_DEBUG_UNALIGNED_STORE pstoreu(&numext::real_ref(*to), Packet4f(from.v)); }

template<> EIGEN_DEVICE_FUNC inline Packet2cf pgather<std::complex<float>, Packet2cf>(const std::complex<float>* from, Index stride)
{
  Packet2cf res;
  float real1 = from[0].real(), imag1 = from[0].imag();
  float real2 = from[stride].real(), imag2 = from[stride].imag();
  Packet4f tmp = {real1, imag1, real2, imag2};
  res.v = tmp;
  return res;
}

template<> EIGEN_DEVICE_FUNC inline void pscatter<std::complex<float>, Packet2cf>(std::complex<float>* to, const Packet2cf& from, Index stride)
{
  *to = std::complex<float>(from.v[0], from.v[1]);
  to += stride;
  *to = std::complex<float>(from.v[2], from.v[3]);
}

template<> EIGEN_STRONG_INLINE void prefetch<std::complex<float> >(const std::complex<float> *   addr) { __builtin_prefetch(addr); }

template<> EIGEN_STRONG_INLINE std::complex<float> pfirst<Packet2cf>(const Packet2cf& a)
{
  EIGEN_ALIGN16 std::complex<float> res[2];
  __lsx_vst(a.v, res, 0);
  return res[0];
}

template<> EIGEN_STRONG_INLINE Packet2cf preverse(const Packet2cf& a)
{
  Packet2cf res;
  res.v = (Packet4f)__lsx_vshuf4i_w(a.v, 0x4e);
  return res;
}

template<> EIGEN_STRONG_INLINE std::complex<float> predux<Packet2cf>(const Packet2cf& a)
{
  return pfirst(Packet2cf(__lsx_vfadd_s(a.v, vec4f_movehl(a.v,a.v))));
}

template<> EIGEN_STRONG_INLINE std::complex<float> predux_mul<Packet2cf>(const Packet2cf& a)
{
  return pfirst(pmul(a, Packet2cf(vec4f_movehl(a.v,a.v))));
}

EIGEN_STRONG_INLINE Packet2cf pcplxflip/* <Packet2cf> */(const Packet2cf& x)
{
  return Packet2cf(vec4f_swizzle1(x.v, 1, 0, 3, 2));
}

EIGEN_MAKE_CONJ_HELPER_CPLX_REAL(Packet2cf,Packet4f)

template<> EIGEN_STRONG_INLINE Packet2cf pdiv<Packet2cf>(const Packet2cf& a, const Packet2cf& b)
{
  return pdiv_complex(a, b);
}

//---------- double ----------
struct Packet1cd
{
  EIGEN_STRONG_INLINE Packet1cd() {}
  EIGEN_STRONG_INLINE explicit Packet1cd(const __m128d& a) : v(a) {}
  Packet2d v;
};

template<> struct packet_traits<std::complex<double> >  : default_packet_traits
{
  typedef Packet1cd type;
  typedef Packet1cd half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 0,
    size = 1,

    HasAdd    = 1,
    HasSub    = 1,
    HasMul    = 1,
    HasDiv    = 1,
    HasNegate = 1,
    HasSqrt   = 1,
    HasAbs    = 0,
    HasAbs2   = 0,
    HasMin    = 0,
    HasMax    = 0,
    HasSetLinear = 0
  };
};

template<> struct unpacket_traits<Packet1cd> {
  typedef std::complex<double> type;
  typedef Packet1cd half;
  typedef Packet2d as_real;
  enum {
    size=1,
    alignment=Aligned16,
    vectorizable=true,
    masked_load_available=false,
    masked_store_available=false
  };
};

template<> EIGEN_STRONG_INLINE Packet1cd padd<Packet1cd>(const Packet1cd& a, const Packet1cd& b) { return Packet1cd(__lsx_vfadd_d(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet1cd psub<Packet1cd>(const Packet1cd& a, const Packet1cd& b) { return Packet1cd(__lsx_vfsub_d(a.v,b.v)); }
template<> EIGEN_STRONG_INLINE Packet1cd pnegate(const Packet1cd& a) { return Packet1cd(pnegate(Packet2d(a.v))); }

template<> EIGEN_STRONG_INLINE Packet1cd pconj(const Packet1cd& a)
{
  const uint64_t tmp[2]= {0x0000000000000000u,0x8000000000000000u};
  __m128i mask = __lsx_vld(tmp,0);
  Packet1cd res;
  res.v = (Packet2d)__lsx_vxor_v((__m128i)a.v, mask);
  return res;
}

template<> EIGEN_STRONG_INLINE Packet1cd pmul<Packet1cd>(const Packet1cd& a, const Packet1cd& b)
{
  Packet2d tmp_real = __lsx_vfmul_d(a.v, b.v);
  Packet2d real = __lsx_vfsub_d(tmp_real, preverse(tmp_real));

  Packet2d tmp_imag = __lsx_vfmul_d(preverse(a.v), b.v);
  Packet2d imag = (__m128d)__lsx_vfadd_d((__m128d)tmp_imag, preverse(tmp_imag));
  Packet1cd res;
  res.v = (__m128d)__lsx_vilvl_d((__m128i)imag, (__m128i)real);
  return res;
}

template<> EIGEN_STRONG_INLINE Packet1cd ptrue  <Packet1cd>(const Packet1cd& a) { return Packet1cd(ptrue(Packet2d(a.v))); }
template<> EIGEN_STRONG_INLINE Packet1cd pand   <Packet1cd>(const Packet1cd& a, const Packet1cd& b)
{
	Packet1cd res;
	res.v = (Packet2d)__lsx_vand_v((__m128i)a.v, (__m128i)b.v);
	return res;
}
template<> EIGEN_STRONG_INLINE Packet1cd por    <Packet1cd>(const Packet1cd& a, const Packet1cd& b)
{
	Packet1cd res;
	res.v = (Packet2d)__lsx_vor_v((__m128i)a.v,(__m128i)b.v);
	return res;
}
template<> EIGEN_STRONG_INLINE Packet1cd pxor   <Packet1cd>(const Packet1cd& a, const Packet1cd& b)
{
       Packet1cd res;
       res.v = (Packet2d)__lsx_vxor_v((__m128i)a.v,(__m128i)b.v);
       return res;
}
template<> EIGEN_STRONG_INLINE Packet1cd pandnot<Packet1cd>(const Packet1cd& a, const Packet1cd& b)
{
	Packet1cd res;
	res.v = (Packet2d)__lsx_vandn_v((__m128i)b.v,(__m128i)a.v);
	return res;
}

// FIXME force unaligned load, this is a temporary fix
template<> EIGEN_STRONG_INLINE Packet1cd pload <Packet1cd>(const std::complex<double>* from)
{ EIGEN_DEBUG_ALIGNED_LOAD return Packet1cd(pload<Packet2d>((const double*)from)); }
template<> EIGEN_STRONG_INLINE Packet1cd ploadu<Packet1cd>(const std::complex<double>* from)
{ EIGEN_DEBUG_UNALIGNED_LOAD return Packet1cd(ploadu<Packet2d>((const double*)from)); }
template<> EIGEN_STRONG_INLINE Packet1cd pset1<Packet1cd>(const std::complex<double>&  from)
{ /* here we really have to use unaligned loads :( */ return ploadu<Packet1cd>(&from); }

template<> EIGEN_STRONG_INLINE Packet1cd ploaddup<Packet1cd>(const std::complex<double>* from) { return pset1<Packet1cd>(*from); }

// FIXME force unaligned store, this is a temporary fix
template<> EIGEN_STRONG_INLINE void pstore <std::complex<double> >(std::complex<double> *   to, const Packet1cd& from) { EIGEN_DEBUG_ALIGNED_STORE pstore((double*)to, Packet2d(from.v)); }
template<> EIGEN_STRONG_INLINE void pstoreu<std::complex<double> >(std::complex<double> *   to, const Packet1cd& from) { EIGEN_DEBUG_UNALIGNED_STORE pstoreu((double*)to, Packet2d(from.v)); }

template<> EIGEN_STRONG_INLINE void prefetch<std::complex<double> >(const std::complex<double> *   addr) { __builtin_prefetch(addr); }

template<> EIGEN_STRONG_INLINE std::complex<double>  pfirst<Packet1cd>(const Packet1cd& a)
{
  EIGEN_ALIGN16 double res[2];
  __lsx_vst(a.v, res, 0);
  return std::complex<double>(res[0],res[1]);
}

template<> EIGEN_STRONG_INLINE Packet1cd preverse(const Packet1cd& a)
{
	return a;
}

template<> EIGEN_STRONG_INLINE std::complex<double> predux<Packet1cd>(const Packet1cd& a)
{
  return pfirst(a);
}

template<> EIGEN_STRONG_INLINE std::complex<double> predux_mul<Packet1cd>(const Packet1cd& a)
{
  return pfirst(a);
}

EIGEN_MAKE_CONJ_HELPER_CPLX_REAL(Packet1cd,Packet2d)

template<> EIGEN_STRONG_INLINE Packet1cd pdiv<Packet1cd>(const Packet1cd& a, const Packet1cd& b)
{
  return pdiv_complex(a, b);
}

EIGEN_STRONG_INLINE Packet1cd pcplxflip/* <Packet1cd> */(const Packet1cd& x)
{
  return Packet1cd(preverse(Packet2d(x.v)));
}

EIGEN_DEVICE_FUNC inline void
ptranspose(PacketBlock<Packet2cf,2>& kernel) {
  Packet4f tmp1 = (Packet4f)__lsx_vilvl_w((__m128i)kernel.packet[1].v, (__m128i)kernel.packet[0].v);
  Packet4f tmp2 = (Packet4f)__lsx_vilvh_w((__m128i)kernel.packet[1].v, (__m128i)kernel.packet[0].v);
  kernel.packet[0].v = (Packet4f)__lsx_vshuf4i_w(tmp1, 0xd8);
  kernel.packet[1].v = (Packet4f)__lsx_vshuf4i_w(tmp2, 0xd8);
}

template<> EIGEN_STRONG_INLINE Packet2cf pcmp_eq(const Packet2cf& a, const Packet2cf& b)
{
  Packet4f eq = (Packet4f)__lsx_vfcmp_ceq_s(a.v, b.v);
  return Packet2cf(pand<Packet4f>(eq, vec4f_swizzle1(eq, 1, 0, 3, 2)));
}

template<> EIGEN_STRONG_INLINE Packet1cd pcmp_eq(const Packet1cd& a, const Packet1cd& b)
{
  Packet2d eq = (Packet2d)__lsx_vfcmp_ceq_d(a.v, b.v);
  return Packet1cd(pand<Packet2d>(eq, preverse(eq)));
}

template<> EIGEN_DEVICE_FUNC inline Packet2cf pselect(const Packet2cf& mask, const Packet2cf& a, const Packet2cf& b)
{
  Packet2cf res;
  res.v = (Packet4f)__lsx_vbitsel_v((__m128i)b.v, (__m128i)a.v, (__m128i)mask.v);
  return res;
}

template<> EIGEN_STRONG_INLINE Packet1cd psqrt<Packet1cd>(const Packet1cd& a) {
  return psqrt_complex<Packet1cd>(a);
}

template<> EIGEN_STRONG_INLINE Packet2cf psqrt<Packet2cf>(const Packet2cf& a) {
  return psqrt_complex<Packet2cf>(a);
}

} // end namespace internal
} // end namespace Eigen

#endif // EIGEN_COMPLEX_LSX_H
