// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2011-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2011-2012 Jitse Niesen <jitse@maths.leeds.ac.uk>
// Copyright (C) 2018 David Tellenbach <david.tellenbach@tellnotes.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef EIGEN_HERMITIAN_EVALUATORS_H
#define EIGEN_HERMITIAN_EVALUATORS_H


namespace Eigen {
// NOTE: struct HermitianMatrixCoeffReturnHelper cannot be member of namespace
// internal since it is the return type of HermitianBase::coeffRef()

// Scalar is no complex type, i.e., a type for that 
// internal::is_complex<Scalar>::value is false
template<typename Scalar, class NestedExpression>
struct HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression, 0> {

  HermitianMatrixCoeffReturnHelper(int row, int col, const NestedExpression* nested, const bool isStoredValue)
    : m_isStoredValue(isStoredValue), m_nested(nested), m_row(row), m_col(col) {}

  operator Scalar() const {
    return m_nested->coeff(m_row, m_col);
  }

  HermitianMatrixCoeffReturnHelper& operator=(const Scalar& value) {
    const_cast<NestedExpression*>(m_nested)->coeffRef(m_row, m_col) = value;
    return *this;
  }

  template<typename OtherScalar, typename OtherNestedExpression>
  OtherScalar operator+(const HermitianMatrixCoeffReturnHelper<OtherScalar, OtherNestedExpression, 1>& rhs) const {
    return static_cast<Scalar>(*this) + static_cast<OtherScalar>(rhs);
  }

  template<typename OtherScalar, typename OtherNestedExpression>
  OtherScalar operator-(const HermitianMatrixCoeffReturnHelper<OtherScalar, OtherNestedExpression, 1>& rhs) const {
    return static_cast<Scalar>(*this) - static_cast<OtherScalar>(rhs);
  }

  template<typename OtherScalar, typename OtherNestedExpression>
  OtherScalar operator*(const HermitianMatrixCoeffReturnHelper<OtherScalar, OtherNestedExpression, 1>& rhs) const {
    return static_cast<Scalar>(*this) * static_cast<OtherScalar>(rhs);
  }

  template<typename OtherScalar, typename OtherNestedExpression>
  OtherScalar operator/(const HermitianMatrixCoeffReturnHelper<OtherScalar, OtherNestedExpression, 1>& rhs) const {
    return static_cast<Scalar>(*this) / static_cast<OtherScalar>(rhs);
  }

  template<typename OtherScalar>
  HermitianMatrixCoeffReturnHelper& operator+=(const OtherScalar& other) {
    const_cast<NestedExpression*>(m_nested)->coeffRef(m_row, m_col) += other;
    return *this;
  }

  template<typename OtherScalar>
  HermitianMatrixCoeffReturnHelper& operator-=(const OtherScalar& other) {
    const_cast<NestedExpression*>(m_nested)->coeffRef(m_row, m_col) -= other;
    return *this;
  }

  template<typename OtherScalar>
  HermitianMatrixCoeffReturnHelper& operator*=(const OtherScalar& other) {
    const_cast<NestedExpression*>(m_nested)->coeffRef(m_row, m_col) *= other;
    return *this;
  }

  template<typename OtherScalar>
  HermitianMatrixCoeffReturnHelper& operator/=(const OtherScalar& other) {
    const_cast<NestedExpression*>(m_nested)->coeffRef(m_row, m_col) /= other;
    return *this;
  }

private:
  const bool m_isStoredValue;
  const NestedExpression* m_nested;
  const int m_row;
  const int m_col;
};

// Scalar is a complex type, i.e., a type for that 
// internal::is_complex<Scalar>::value is true
template<typename Scalar, class NestedExpression>
struct HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression, 1> : public Scalar {

  HermitianMatrixCoeffReturnHelper(int row, int col, const NestedExpression* nested, const bool isStoredValue)
    : Scalar((isStoredValue)?(nested->coeff(row, col)):(std::conj(nested->coeff(row, col)))), 
    m_isStoredValue(isStoredValue), m_nested(nested), m_row(row), m_col(col) {}

  HermitianMatrixCoeffReturnHelper& operator=(const Scalar& value) {
    if (m_isStoredValue) {
      const_cast<NestedExpression*>(m_nested)->coeffRef(m_row, m_col) = value;
    } else {
      Scalar tmp = std::conj(value);
      const_cast<NestedExpression*>(m_nested)->coeffRef(m_row, m_col) = tmp;
    }
    return *this;
  }

  operator Scalar() {
    if (m_isStoredValue) {
      return m_nested->coeffRef(m_row, m_col);
    } else {
      return std::conj(m_nested->coeffRef(m_row, m_col));
    }
  }

  template<typename OtherScalar, class OtherNestedExpression>
  Scalar operator+(const HermitianMatrixCoeffReturnHelper<OtherScalar, OtherNestedExpression>& other) const {
    return *this + static_cast<OtherScalar>(other);
  }

  template<typename OtherScalar, class OtherNestedExpression>
  Scalar operator-(const HermitianMatrixCoeffReturnHelper<OtherScalar, OtherNestedExpression>& other) const {
    return *this - static_cast<OtherScalar>(other);
  }

  template<typename OtherScalar, class OtherNestedExpression>
  Scalar operator*(const HermitianMatrixCoeffReturnHelper<OtherScalar, OtherNestedExpression>& other) const {
    return *this * static_cast<OtherScalar>(other);
  }

  template<typename OtherScalar, class OtherNestedExpression>
  Scalar operator/(const HermitianMatrixCoeffReturnHelper<OtherScalar, OtherNestedExpression>& other) const {
    return *this / static_cast<OtherScalar>(other);
  }

  template<typename OtherScalar, class OtherNestedExpression>
  bool operator==(const HermitianMatrixCoeffReturnHelper<OtherScalar, OtherNestedExpression>& other) const {
    return *this == static_cast<OtherScalar>(other);
  }

  template<typename OtherScalar, class OtherNestedExpression>
  bool operator!=(const HermitianMatrixCoeffReturnHelper<OtherScalar, OtherNestedExpression>& other) const {
    return *this != static_cast<OtherScalar>(other);
  }

  template<typename OtherScalar>
  HermitianMatrixCoeffReturnHelper& operator+=(const OtherScalar& other) {
    if (m_isStoredValue) {
      const_cast<NestedExpression*>(m_nested)->coeffRef(m_row, m_col) += other;
    } else {
      Scalar tmp = std::conj(other);
      const_cast<NestedExpression*>(m_nested)->coeffRef(m_row, m_col) += tmp;
    }
    return *this;
  }

  template<typename OtherScalar>
  HermitianMatrixCoeffReturnHelper& operator-=(const OtherScalar& other) {
    if (m_isStoredValue) {
      const_cast<NestedExpression*>(m_nested)->coeffRef(m_row, m_col) -= other;
    } else {
      Scalar tmp = std::conj(other);
      const_cast<NestedExpression*>(m_nested)->coeffRef(m_row, m_col) -= tmp;
    }
    return *this;
  }

  template<typename OtherScalar>
  HermitianMatrixCoeffReturnHelper& operator*=(const OtherScalar& other) {
    if (m_isStoredValue) {
      const_cast<NestedExpression*>(m_nested)->coeffRef(m_row, m_col) *= other;
    } else {
      Scalar tmp = std::conj(other);
      const_cast<NestedExpression*>(m_nested)->coeffRef(m_row, m_col) *= tmp;
    }
    return *this;
  }

  template<typename OtherScalar>
  HermitianMatrixCoeffReturnHelper& operator/=(const OtherScalar& other) {
    if (m_isStoredValue) {
      const_cast<NestedExpression*>(m_nested)->coeffRef(m_row, m_col) /= other;
    } else {
      Scalar tmp = std::conj(other);
      const_cast<NestedExpression*>(m_nested)->coeffRef(m_row, m_col) /= tmp;
    }
    return *this;
  }
 
private:
  const bool m_isStoredValue;
  const NestedExpression* m_nested;
  const int m_row;
  const int m_col;
};

namespace internal {

template<typename Derived>
struct evaluator<HermitianBase<Derived> > : evaluator_base<HermitianBase<Derived> >
{
  typedef HermitianBase<Derived> HermitianType;
  typedef typename HermitianType::Scalar Scalar;
  typedef typename HermitianType::CoeffReturnType CoeffReturnType;
  typedef typename internal::traits<Derived>::NestedExpression NestedExpression;

  enum {
    UpLo = HermitianType::UpLo,
    RowsAtCompileTime = HermitianType::RowsAtCompileTime,
    ColsAtCompileTime = HermitianType::ColsAtCompileTime,
    SizeAtCompileTime = HermitianType::SizeAtCompileTime,
    Flags = (unsigned int)(NoPreferredStorageOrderBit & LvalueBit)
  };

#if EIGEN_HAS_CXX11
  EIGEN_DEVICE_FUNC
  evaluator() : m_nested(nullptr) {}
#else
  EIGEN_DEVICE_FUNC
  evaluator() : m_nested(0) {}
#endif

  EIGEN_DEVICE_FUNC evaluator(const HermitianType& m)
    : m_nested(&m.nestedExpression()), m_dimension(m.cols()) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression> coeff(Index row, Index col) const
  {
    // Even dimension and upper triangular storage
    if (UpLo == Upper && !(m_dimension % 2)) {
      if (row <= col) {
        if (col < m_dimension / 2) {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(m_dimension / 2 + col + 1, row, m_nested, true);
        } else {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(row, col - m_dimension / 2, m_nested, true);
        }
      } else {
        if (col <= row) {
          if (row < m_dimension / 2) {
            return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(m_dimension / 2 + row + 1, col, m_nested, false);
          } else {
            return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(col, row - m_dimension / 2, m_nested, false);
          }
        }
      }
    }

    // Even dimension and lower triangular storage
    if (UpLo == Lower && !(m_dimension % 2)) {
      if (row < col) {
        if (row < m_dimension / 2) {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(col + 1, row, m_nested, false);
        } else {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(row - m_dimension / 2, col - m_dimension / 2,
                                                                            m_nested, false);
        }
      } else {
        if (col < m_dimension / 2) {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(row + 1, col, m_nested, true);
        } else {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(col - m_dimension / 2, row - m_dimension / 2,
                                                                            m_nested, true);
        }
      }
    }
  
    // Odd dimension and upper triangular storage
    if (UpLo == Upper && (m_dimension % 2)) {
      if (row <= col) {
        if (col < (m_dimension - 1) / 2) {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>((m_dimension - 1) / 2 + col + 1, row,
                                                                            m_nested, true);
        } else {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(row, col - (m_dimension - 1) / 2,
                                                                            m_nested, true);
        }
      } else {
        if (row < (m_dimension - 1) / 2) {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>((m_dimension - 1) / 2 + row + 1, col,
                                                                            m_nested, false);
        } else {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(col, row - (m_dimension - 1) / 2,
                                                                            m_nested, false);
        }
      }
    }

    // Odd dimension and lower triangular storage
    if (UpLo == Lower && (m_dimension % 2)) {
      if (row < col) {
        if (row < (m_dimension + 1) / 2) {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(col, row, m_nested, false);
        } else {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(row - (m_dimension - 1)  / 2 - 1,
                                                                            col - (m_dimension + 1)  / 2 + 1,
                                                                            m_nested, false);
        }
      } else {
        if (col < (m_dimension + 1) / 2) {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(row, col, m_nested, true);
        } else {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(col - (m_dimension - 1)  / 2 - 1,
                                                                            row - (m_dimension + 1)  / 2 + 1,
                                                                            m_nested, true);
        }
      }
    }
    // Some compilers might mock without this fallback return value
    return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(0, 0, m_nested, true);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression> coeffRef(Index row, Index col) {
    // Even dimension and upper triangular storage
    if (UpLo == Upper && !(m_dimension % 2)) {
      if (row <= col) {
        if (col < m_dimension / 2) {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(m_dimension / 2 + col + 1, row, m_nested, true);        
        } else {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(row, col - m_dimension / 2, m_nested, true);          
        }
      } else {
        if (col <= row) {
          if (row < m_dimension / 2) {
            return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(m_dimension / 2 + row + 1, col, m_nested, false);            
          } else {
            return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(col, row - m_dimension / 2, m_nested, false);            
          }
        }
      }
    }

    // Even dimension and lower triangular storage
    if (UpLo == Lower && !(m_dimension % 2)) {
      if (row < col) {
        if (row < m_dimension / 2) {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(col + 1, row, m_nested, false);
        } else {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(row - m_dimension / 2, col - m_dimension / 2,
                                                                            m_nested, false);
        }
      } else {
        if (col < m_dimension / 2) {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(row + 1, col, m_nested, true);
        } else {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(col - m_dimension / 2, row - m_dimension / 2,
                                                                            m_nested, true);
        }
      }
    }

    // Odd dimension and upper triangular storage
    if (UpLo == Upper && (m_dimension % 2)) {
      if (row <= col) {
        if (col < (m_dimension - 1) / 2) {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>((m_dimension - 1) / 2 + col + 1, row,
                                                                            m_nested, true);          
        } else {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(row, col - (m_dimension - 1) / 2,
                                                                            m_nested, true);
        }
      } else {
        if (row < (m_dimension - 1) / 2) {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>((m_dimension - 1) / 2 + row + 1, col,
                                                                            m_nested, false);          
        } else {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(col, row - (m_dimension - 1) / 2,
                                                                            m_nested, false);
        }        
      }
    }

    // Odd dimension and lower triangular storage
    if (UpLo == Lower && (m_dimension % 2)) {
      if (row < col) {
        if (row < (m_dimension + 1) / 2) {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(col, row, m_nested, false);          
        } else {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(row - (m_dimension - 1)  / 2 - 1,
                                                                            col - (m_dimension + 1)  / 2 + 1,
                                                                            m_nested, false);          
        }
      } else {
        if (col < (m_dimension + 1) / 2) {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(row, col, m_nested, true);          
        } else {
          return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(col - (m_dimension - 1)  / 2 - 1,
                                                                            row - (m_dimension + 1)  / 2 + 1,
                                                                            m_nested, true);          
        }
      }
    }
    // Some compilers might mock without this fallback return value
    return HermitianMatrixCoeffReturnHelper<Scalar, NestedExpression>(0, 0, m_nested, true);
  }

protected:
  const NestedExpression* m_nested;
  Index m_dimension;
};


} // namespace internal

} // end namespace Eigen

#endif // EIGEN_COREEVALUATORS_H