
// A Scalar that simulates an integer with arbitrary numerical properties
template <int Radix, int Digits, bool Signed>
struct TrickyInteger {
  using DataType = std::conditional_t<Signed, int64_t, uint64_t>;
  operator DataType() const { return m_data; }
  TrickyInteger() : m_data(0) {}
  template <typename T>
  TrickyInteger(const T& other) {
    set(other);
  }
  template <typename T>
  inline TrickyInteger& operator=(const T& other) {
    set(other);
    return *this;
  }
  constexpr inline static TrickyInteger get_highest() {
    TrickyInteger res;
    for (int d = 0; d < Digits; d++) {
      res.m_data *= Radix;
      res.m_data += (Radix - 1);
    }
    return res;
  }
  template <bool IsSigned = Signed, std::enable_if_t<IsSigned, bool> = true>
  constexpr inline static TrickyInteger get_lowest() {
    TrickyInteger res = get_highest();
    res.m_data = -res.m_data;
    return res;
  }
  template <bool IsSigned = Signed, std::enable_if_t<!IsSigned, bool> = true>
  constexpr inline static TrickyInteger get_lowest() {
    return TrickyInteger();
  }

 private:
  template <typename T>
  inline void set(const T& other) {
    m_data = other;
    VERIFY(m_data >= get_lowest() && m_data <= get_highest());
  }
  DataType m_data;
};

namespace Eigen {
template <int Radix, int Digits, bool Signed>
struct NumTraits<TrickyInteger<Radix, Digits, Signed>> {
  using T = TrickyInteger<Radix, Digits, Signed>;
  static constexpr int IsInteger = 1, IsSigned = Signed, IsComplex = 0, RequireInitialization = 1, AddCost = 1,
                       MulCost = 1;
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static inline int radix() { return Radix; }
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static inline int digits() { return Digits; }
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static inline T highest() { return T::get_highest(); }
  EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static inline T lowest() { return T::get_lowest(); }
};
}  // namespace Eigen