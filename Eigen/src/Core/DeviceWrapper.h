// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2023 The Dumbass <cs.schlosser@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_DEVICEWRAPPER_H
#define EIGEN_DEVICEWRAPPER_H

namespace Eigen {
template <typename XprType, typename Device>
struct DeviceWrapper : public XprType {
  using CleanedXprType = internal::remove_all_t<XprType>;

  EIGEN_DEVICE_FUNC DeviceWrapper(EigenBase<CleanedXprType>& xpr, const Device& device)
      : m_xpr(xpr.derived()), m_device(device) {}
  EIGEN_DEVICE_FUNC DeviceWrapper(const EigenBase<CleanedXprType>& xpr, const Device& device)
      : m_xpr(xpr.derived()), m_device(device) {}

  template <typename OtherDerived>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE XprType& operator=(const EigenBase<OtherDerived>& other);
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE XprType& xpr() { return m_xpr; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const XprType& xpr() const { return m_xpr; }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Device& device() const { return m_device; }
  NoAlias<DeviceWrapper, EigenBase> noalias() { return NoAlias<DeviceWrapper, EigenBase>(*this); }

  XprType& m_xpr;
  const Device& m_device;
};

namespace internal {

// unless otherwise specified, use the default evaluation strategy
template <typename Dst, typename Src, typename Func, typename Device>
struct call_assignment_no_alias_device {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR static void run(DeviceWrapper<Dst, Device>& dst, const Src& src,
                                                                        const Func& func) {
    call_assignment_no_alias(dst.xpr(), src, func);
  }
};

// insert thread pool device specializations here, e.g. , preferably in another header
//template <typename Dst, typename Src, typename Func>
//struct call_assignment_no_alias_device<Dst, Src, Func, DoomsDayDevice>
//{
//  static void kill_all_humans();
//};

template <typename Dst, typename Src, typename Func, typename Device>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE EIGEN_CONSTEXPR void call_assignment_no_alias(DeviceWrapper<Dst, Device>& dst,
                                                                                    const Src& src, const Func& func) {
  call_assignment_no_alias_device<Dst, Src, Func, Device>::run(dst, src, func);
}
}  // namespace internal

template <typename Derived, typename Device>
template <typename OtherDerived>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived& DeviceWrapper<Derived, Device>::operator=(
    const EigenBase<OtherDerived>& other) {
  internal::call_assignment(*this, other.derived());
  return xpr();
}

template <typename Derived>
template <typename Device>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DeviceWrapper<Derived, Device> EigenBase<Derived>::useDevice(
    const Device& device) {
  return DeviceWrapper<Derived, Device>(derived(), device);
}

template <typename Derived>
template <typename Device>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE DeviceWrapper<const Derived, Device> EigenBase<Derived>::useDevice(
    const Device& device) const {
  return DeviceWrapper<const Derived, Device>(derived(), device);
}
}  // namespace Eigen

#endif