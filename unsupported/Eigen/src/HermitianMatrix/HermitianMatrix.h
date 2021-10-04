// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018 David Tellenbach <david.tellenbach@tellnotes.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_HERMITIANMATRIX_H
#define EIGEN_HERMITIANMATRIX_H

namespace Eigen {


template<typename Lhs, typename Rhs, int Option>
class ProductImpl<Lhs,Rhs,Option,HermitianShape>
  : public internal::hermitian_product_base<Lhs,Rhs,Option>
{
  private:
    typedef typename Lhs::NestedExpression NestedExpression;

  public:
    typedef typename internal::hermitian_product_base<Lhs, Rhs, Option> Base;

    EIGEN_DEVICE_FUNC
    inline const NestedExpression nestedExpression() const 
    {
      // In the case of a product we just want a brand new nested expression
      NestedExpression nested;
      return nested;
    }

 
};


namespace internal {

#ifndef EIGEN_PARSED_BY_DOXYGEN
// Dimension: even
template<typename _Scalar, int _Dimension, unsigned int _UpLo, int _Storage, int _MaxDimension>
struct traits<HermitianMatrix<_Scalar, _Dimension, _UpLo, _Storage, _MaxDimension, 0> >
  : traits<Matrix<_Scalar, _Dimension, _Dimension, _Storage, _MaxDimension, _MaxDimension> >
{
  typedef Matrix<_Scalar, _Dimension + 1, _Dimension / 2, _Storage,  _MaxDimension + 1, _MaxDimension / 2> NestedExpression;
  typedef _Scalar Scalar;
  typedef Matrix<_Scalar, _Dimension, _Dimension, _Storage, _MaxDimension, _MaxDimension> DenseType;
  typedef HermitianMatrix<_Scalar, Dynamic, _UpLo, _Storage, _MaxDimension> PlainObject;
  typedef HermitianShape StorageKind;
  typedef HermitianXpr XprKind;
  enum {
    Flags = LvalueBit | NoPreferredStorageOrderBit | NestByRefBit,
    UpLo = _UpLo,
    RowsAtCompileTime = _Dimension,
    ColsAtCompileTime = _Dimension,
    MaxRowsAtCompileTime = _MaxDimension,
    MaxColsAtCompileTime = _MaxDimension
  };
};

// Dimension: odd
template<typename _Scalar, int _Dimension, unsigned int _UpLo, int _Storage, int _MaxDimension>
struct traits<HermitianMatrix<_Scalar, _Dimension, _UpLo, _Storage, _MaxDimension, 1> >
  : traits<Matrix<_Scalar, _Dimension, _Dimension, _Storage, _MaxDimension, _MaxDimension> >
{
  typedef Matrix<_Scalar, _Dimension, (_Dimension + 1) / 2, _Storage, _MaxDimension, (_MaxDimension + 1) / 2> NestedExpression;
  typedef _Scalar Scalar;
  typedef Matrix<_Scalar, _Dimension, _Dimension, _Storage, _MaxDimension, _MaxDimension> DenseType;
  typedef HermitianMatrix<_Scalar, Dynamic, _UpLo, _Storage, _MaxDimension> PlainObject;
  typedef HermitianShape StorageKind;
  typedef HermitianXpr XprKind;

  enum {
    Flags = LvalueBit | NoPreferredStorageOrderBit | NestByRefBit,
    UpLo = _UpLo,
    RowsAtCompileTime = _Dimension,
    ColsAtCompileTime = _Dimension,
    MaxRowsAtCompileTime = _MaxDimension,
    MaxColsAtCompileTime = _MaxDimension
  };
};

// Dimension::Eigen::Dynamic
template<typename _Scalar, unsigned int _UpLo, int _Storage, int _MaxDimension>
struct traits<HermitianMatrix<_Scalar, Dynamic, _UpLo, _Storage, _MaxDimension> >
  : traits<Matrix<_Scalar, Dynamic, Dynamic, _Storage, _MaxDimension, _MaxDimension> >
{
  typedef Matrix<_Scalar, Dynamic, Dynamic, _Storage, _MaxDimension, _MaxDimension> NestedExpression;
  typedef Matrix<_Scalar, Dynamic, Dynamic, _Storage, _MaxDimension, _MaxDimension> DenseType;
  typedef HermitianMatrix<_Scalar, Dynamic, _UpLo, _Storage, _MaxDimension> PlainObject;
  typedef _Scalar Scalar;
  typedef HermitianShape StorageKind;
  typedef HermitianXpr XprKind;

  enum {
    Flags = LvalueBit | NoPreferredStorageOrderBit | NestByRefBit,
    UpLo = _UpLo,
    RowsAtCompileTime = Dynamic,
    ColsAtCompileTime = Dynamic,
    MaxRowsAtCompileTime = _MaxDimension,
    MaxColsAtCompileTime = _MaxDimension
  };
};
#endif

template<typename DstXprType, typename Lhs, typename Rhs>
struct hermitian_prod_impl {

  typedef typename internal::traits<DstXprType>::Scalar Scalar;

  static void run(DstXprType& dst, const Lhs& lhs, const Rhs& rhs)
  {
    const Index dimension = dst.cols();
    
    const Matrix<Scalar, ((Lhs::Dimension == Dynamic)?(Dynamic):(Lhs::Dimension / 2)) , ((Lhs::Dimension == Dynamic)?(Dynamic):(Lhs::Dimension / 2))>& tmp = lhs.nestedExpression().block(1, 0, dimension / 2, dimension / 2).template selfadjointView<Lower>();
    const Matrix<Scalar, (Rhs::Dimension == Dynamic)?(Dynamic):(Rhs::Dimension / 2) , (Rhs::Dimension == Dynamic)?(Dynamic):(Rhs::Dimension / 2)>& tmp2 = lhs.nestedExpression().block(0, 0, dimension / 2, dimension / 2).template selfadjointView<Upper>();
    #pragma omp parallel sections
    {
    #pragma omp section 
    {
    dst.nestedExpression().block(1, 0, dimension / 2, dimension / 2).template triangularView<Lower>() = tmp * rhs.nestedExpression().block(1, 0, dimension / 2, dimension / 2).template selfadjointView<Lower>()
      + lhs.nestedExpression().block(dimension/2 + 1, 0, dimension / 2, dimension / 2).transpose() 
      * rhs.nestedExpression().block(dimension/2 + 1, 0, dimension/2, dimension/2);
    }
    #pragma omp section
    {
    dst.nestedExpression().block(dimension/2 + 1, 0, dimension/2, dimension/2) = lhs.nestedExpression().block(dimension/2 + 1, 0, dimension/2, dimension/2)
      * rhs.nestedExpression().block(1, 0, dimension/2, dimension/2).template selfadjointView<Lower>()
      + lhs.nestedExpression().block(0, 0, dimension/2, dimension/2).template selfadjointView<Upper>()
      * rhs.nestedExpression().block(dimension/2 + 1, 0, dimension/2, dimension/2);
    }
    #pragma omp section
    {
    dst.nestedExpression().block(0, 0, dimension/2, dimension/2).template triangularView<Upper>() = lhs.nestedExpression().block(dimension/2 + 1, 0, dimension/2, dimension/2)
      * rhs.nestedExpression().block(dimension/2 + 1, 0, dimension/2, dimension/2).transpose()
      + tmp2 * rhs.nestedExpression().block(0, 0, dimension/2, dimension/2).template selfadjointView<Upper>(); 
    }
    }
    
  }
};
} // namespace internal

/** \class Hermitian
  * \ingroup HermitianMatrix_Module
  *
  * \brief Represents a hermitian matrix with its storage
  *
  * \param _Scalar the type of coefficients
  * \param _Dimension the dimension of the matrix, or Eigen::Dynamic
  * \param _UpLo can take the values Eigen::Upper to store only the upper triagonal part of the matrix or Eigen::Lower
  *        to store only the lower triangular part respectively. This parameter is optional and defaults to
  *        Eigen::Upper.
  * \param _Storage can take the values Eigen::RowMajor for row-major storage of the underlying Eigen::Matrix or
  *        Eigen::ColMajor for column-major storage respectively. This parameter is optional and defaults to 
  *        Eigen::ColMajor.
  * \param _MaxDimension the dimension of the matrix or Eigen::Dynamic. This parameter is option and defaults to 
  *        _Dimension.
  */
template<typename _Scalar, int _Dimension, unsigned int _UpLo, int _Storage, int _MaxDimension, int _Remainder>
class HermitianMatrix
  : public HermitianBase<HermitianMatrix<_Scalar, _Dimension, _UpLo, _Storage, _MaxDimension> >
{
  public:
#ifndef EIGEN_PARSED_BY_DOXYGEN
    typedef typename internal::traits<HermitianMatrix>::NestedExpression NestedExpression;
    typedef typename internal::traits<HermitianMatrix>::StorageKind StorageKind;
    typedef typename internal::traits<HermitianMatrix>::StorageIndex StorageIndex;
    typedef HermitianBase<HermitianMatrix<_Scalar, _Dimension, _UpLo, _Storage, _MaxDimension> > Base;
#endif

    enum {
      Dimension = _Dimension,
      UpLo = _UpLo
    };

    /** Default constructor without initialization */
    EIGEN_DEVICE_FUNC
    inline HermitianMatrix()
      : m_dimension(_Dimension == Dynamic ? 0 : _Dimension),
        m_odd(_Dimension == Dynamic ? 0 : _Dimension % 2)
    {}

    /**  Constructs a hermitian matrix with given dimension */
    EIGEN_DEVICE_FUNC
    explicit inline HermitianMatrix(Index dim)
      : m_odd(dim % 2), m_dimension(dim)
    {
      if (dim % 2) {
        m_nested.resize(dim, (dim + 1) / 2);
      } else {
        m_nested.resize(dim + 1, dim / 2);
      }
    }

    /** Constructs a HermitianMatrix by passing a dense matrix */
    EIGEN_DEVICE_FUNC
    template<typename OtherDerived>
    explicit inline HermitianMatrix(const MatrixBase<OtherDerived>& other)
      : m_dimension(other.cols()), m_odd(other.cols() % 2)
    {
      // Replace with generic template argument check 

      // _UpLo must not differ from Eigen::Lower and Eigen::Upper
      EIGEN_STATIC_ASSERT(_UpLo == Upper || _UpLo == Lower,
                          HERMITIANMATRIX_ACCEPTS_UPPER_AND_LOWER_MODE_ONLY);

      // other is quadratic matrix and its size matches the size of this
      EIGEN_STATIC_ASSERT(OtherDerived::RowsAtCompileTime == _Dimension
                          && OtherDerived::RowsAtCompileTime == OtherDerived::ColsAtCompileTime, 
                          YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES);

      if (_Dimension == Dynamic) {
        if (m_odd) {
          m_nested.resize(m_dimension, (m_dimension + 1) / 2);
        } else {
          m_nested.resize(m_dimension + 1, m_dimension / 2);
        }
      }

      // Dimension: Even
      // UpLo:      Upper
      if (_UpLo == Upper && !m_odd) {
        for (Index row = 0; row < m_dimension; ++row) {
          for (Index col = row; col < m_dimension; ++col) {
            if (col < (m_dimension / 2)) {
              m_nested(m_dimension / 2 + col + 1, row) = other(row, col);
            } else {
              m_nested(row, col - m_dimension / 2) = other(row, col);
            }
          }
        }
        return;
      }

      // Dimension: Even
      // UpLo:      Lower
      if (_UpLo == Lower && !m_odd) {
        for (Index row = 0; row < m_dimension; ++row) {
          for (Index col = 0; col <= row; ++col) {
            if (col < m_dimension / 2) {
              m_nested(row +1, col) = other(row, col);
            } else {
              m_nested(col - m_dimension / 2, row - m_dimension / 2) = other(row, col);
            }
          }
        }
        return;
      }
      
      // Dimension: odd
      // UpLo:      Upper
      if (_UpLo == Upper && m_odd) {
        for (Index row = 0; row < m_dimension; ++row) {
          for (Index col = row; col < m_dimension; ++col) {
            if (col < ((m_dimension - 1) / 2)) {
              m_nested((m_dimension - 1)/2 + col + 1, row) = other(row, col);
            } else {
              m_nested(row, col - (m_dimension - 1) / 2) = other(row, col);
            }
          }
        }
        return;
      }

      // Dimension: Odd
      // UpLo:      Lower
      if (_UpLo == Lower && m_odd) {
        for (Index row = 0; row < m_dimension; ++row) {
          for (Index col = 0; col <= row; ++col) {
            if (col < (other.cols() + 1) / 2) {
              m_nested(row, col) = other(row, col);
            } else {
              m_nested(col - (m_dimension - 1)  / 2 - 1, row - (m_dimension + 1)  / 2 + 1) = other(row, col);
            }
          }
        }
        return;
      }
    }

    /** Copy constructor */
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    inline HermitianMatrix(const HermitianBase<OtherDerived>& other)
      : m_nested(other.nestedExpression()), m_dimension(other.cols()), m_odd(other.cols() % 2) {}

#ifndef EIGEN_PARSED_BY_DOXYGEN
    /** copy constructor. prevent a default copy constructor from hiding the other templated constructor */
    explicit inline HermitianMatrix(const HermitianMatrix& other)
      : m_nested(other.nestedExpression()), m_dimension(other.m_dimension),
        m_odd(other.m_dimension % 2) {}
#endif

    template<typename OtherDerived>
    explicit inline HermitianMatrix(const CwiseBinaryOp<internal::scalar_sum_op<_Scalar, typename internal::traits<OtherDerived>::Scalar >, 
                           const HermitianMatrix<_Scalar, _Dimension, _UpLo, _Storage, _MaxDimension>, const OtherDerived >& other)
      : m_dimension(other.lhs().cols()), m_odd(other.lhs().cols() % 2),
        m_nested(other.lhs().nestedExpression() + other.rhs().nestedExpression()){}

    template<typename OtherDerived>
    explicit inline HermitianMatrix(const CwiseBinaryOp<internal::scalar_difference_op<_Scalar, typename internal::traits<OtherDerived>::Scalar >, 
                           const HermitianMatrix<_Scalar, _Dimension, _UpLo, _Storage, _MaxDimension>, const OtherDerived >& other)
      : m_dimension(other.lhs().cols()), m_odd(other.lhs().cols() % 2)
    {
      m_nested = other.lhs().nestedExpression() - other.rhs().nestedExpression();
    }

    /** Copy assignment constructor */
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    HermitianMatrix& operator=(const HermitianBase<OtherDerived>& other)
    {
      return Base::operator=(other);
    }

    /** Copy assignment constructor */
    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    HermitianMatrix& operator=(const DenseBase<OtherDerived>& other)
    {
      return Base::operator=(other);
    }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    HermitianMatrix&
    operator=(const CwiseBinaryOp<internal::scalar_sum_op<_Scalar, typename internal::traits<OtherDerived>::Scalar>, 
              const HermitianMatrix, const OtherDerived >& other)
    {
      m_dimension = other.lhs().cols();
      m_odd = m_dimension % 2;

      // Just add and assign the nested expressions of the left and right hand
      // side of the abstract expression
      m_nested = other.lhs().nestedExpression() + other.rhs().nestedExpression();
      return *this;
    }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    HermitianMatrix&
    operator=(const CwiseBinaryOp<internal::scalar_difference_op<_Scalar, typename internal::traits<OtherDerived>::Scalar>, 
              const HermitianMatrix, const OtherDerived >& other)
    {
      m_dimension = other.lhs().cols();
      m_odd = m_dimension % 2;

      // Just add and assign the nested expressions of the left and right hand
      // side of the abstract expression
      m_nested = other.lhs().nestedExpression() - other.rhs().nestedExpression();
      return *this;
    }

    template<typename OtherDerived>
    EIGEN_DEVICE_FUNC
    HermitianMatrix&
    operator=(const Product<const HermitianMatrix<_Scalar, _Dimension, _UpLo, _Storage, _MaxDimension>, 
              const OtherDerived, LazyProduct>& product) {
      m_dimension = product.lhs().cols();
      m_odd = m_dimension % 2;
      
      if (_Dimension == Dynamic) {
        if (m_odd) {
          m_nested.resize(m_dimension, (m_dimension + 1) / 2);
        } else {
          m_nested.resize(m_dimension + 1, m_dimension / 2);
        }
      }
      // TODO:: Distinguish between dynamic and fixed dimension
      //  -> Dynamic or fixed dimension for temporaries
      //  -> Dynamic or fixed form of block expression

      // Even dimension
      if (!m_odd) {
        if (_UpLo == Lower) {
          const typename HermitianMatrix<_Scalar, _Dimension, _UpLo, _Storage, _MaxDimension>::NestedExpression& nestedLhs = product.lhs().nestedExpression();
          const typename OtherDerived::NestedExpression& nestedRhs = product.rhs().nestedExpression();

          const Matrix<_Scalar, Dynamic, Dynamic>& tmp = nestedLhs.block(1, 0, m_dimension/2, m_dimension/2).template selfadjointView<Lower>();
          const Matrix<_Scalar, Dynamic, Dynamic>& tmp2 = nestedLhs.block(0, 0, m_dimension/2, m_dimension/2).template selfadjointView<Upper>();
          #pragma omp parallel sections
          {
          #pragma omp section 
          {
          this->nestedExpression().block(1, 0, m_dimension/2, m_dimension/2).template triangularView<Lower>() = tmp * nestedRhs.block(1, 0, m_dimension/2, m_dimension/2).template selfadjointView<Lower>()
            + nestedLhs.block(m_dimension/2 + 1, 0, m_dimension/2, m_dimension/2).transpose() 
            * nestedRhs.block(m_dimension/2 + 1, 0, m_dimension/2, m_dimension/2);
          }
          #pragma omp section
          {
          this->nestedExpression().block(m_dimension/2 + 1, 0, m_dimension/2, m_dimension/2) = nestedLhs.block(m_dimension/2 + 1, 0, m_dimension/2, m_dimension/2)
            * nestedRhs.block(1, 0, m_dimension/2, m_dimension/2).template selfadjointView<Lower>()
            + nestedLhs.block(0, 0, m_dimension/2, m_dimension/2).template selfadjointView<Upper>()
            * nestedRhs.block(m_dimension/2 + 1, 0, m_dimension/2, m_dimension/2);
          }
          #pragma omp section
          {
          this->nestedExpression().block(0, 0, m_dimension/2, m_dimension/2).template triangularView<Upper>() = nestedLhs.block(m_dimension/2 + 1, 0, m_dimension/2, m_dimension/2)
            * nestedRhs.block(m_dimension/2 + 1, 0, m_dimension/2, m_dimension/2).transpose()
            + tmp2 
            * nestedRhs.block(0, 0, m_dimension/2, m_dimension/2).template selfadjointView<Upper>(); 
          }
          }
          
        } else if (_UpLo == Upper) {

          return *this;
        }
      }
      return *this;
      
    }

#ifndef EIGEN_PARSED_BY_DOXYGEN
    /** copy assignment constructor. Prevent a default copy constructor from 
     *  hiding the other templated constructor */
    // EIGEN_DEVICE_FUNC
    // HermitianMatrix& operator=(const HermitianMatrix& other)
    // {
    //   m_dimension = other.cols();
    //   m_odd = m_dimension % 2;
    //   m_nested = other.nestedExpression();
    //   return *this;
    // }
#endif

    /** \returns the number of columns. \sa rows(), ColsAtCompileTime*/
    EIGEN_DEVICE_FUNC
    Index cols()
    {
      return m_dimension;
    }

    /** \returns the number of columns. \sa rows(), ColsAtCompileTime*/
    EIGEN_DEVICE_FUNC
    Index cols() const
    {
      return m_dimension;
    }

    /** \returns the number of rows. \sa cols(), RowsAtCompileTime */
    EIGEN_DEVICE_FUNC
    Index rows()
    {
      return m_dimension;
    }

    /** \returns the number of rows. \sa cols(), RowsAtCompileTime */
    EIGEN_DEVICE_FUNC
    Index rows() const
    {
      return m_dimension;
    }

    /** \returns a const reference to the nested expression */
    EIGEN_DEVICE_FUNC
    const NestedExpression& nestedExpression() const
    {
      return m_nested;
    }

    /** \returns a reference to the nested expression */
    EIGEN_DEVICE_FUNC
    NestedExpression& nestedExpression() {
      return m_nested;
    }

    EIGEN_DEVICE_FUNC
    inline void resize(Index dim)
    {
      if (dim % 2) {
        m_nested.resize(dim, (dim + 1) / 2);
      } else {
        m_nested.resize(dim + 1, dim / 2);
      }
      m_dimension = dim;
    }

    /** Sets all coefficients to zero. */
    EIGEN_DEVICE_FUNC
    inline void setZero()
    {
      m_nested.setZero();
    }

    EIGEN_DEVICE_FUNC
    inline static HermitianMatrix Random()
    {
      HermitianMatrix ret;
      ret.nestedExpression() = NestedExpression::Random();
      return ret;
    }

    EIGEN_DEVICE_FUNC
    inline static HermitianMatrix Random(Index dim)
    {
      HermitianMatrix ret;
      ret.nestedExpression() = NestedExpression::Random(dim, dim);
    }
    
  private:
    NestedExpression m_nested;
    bool m_odd;
    Index m_dimension;
};

namespace internal {

template<> struct storage_kind_to_shape<HermitianShape> { typedef HermitianShape Shape; };

struct Hermitian2Dense {};
struct Hermitian2Hermitian {};
struct Dense2Hermitian {};

template<> struct AssignmentKind<HermitianShape, HermitianShape> { typedef Hermitian2Hermitian Kind; };
template<> struct AssignmentKind<DenseShape,     HermitianShape> { typedef Hermitian2Dense Kind; };
template<> struct AssignmentKind<HermitianShape, DenseShape>     { typedef Dense2Hermitian Kind; };

// Hermitian to Hermitian assignment
template< typename DstXprType, typename SrcXprType, typename Functor>
struct Assignment<DstXprType, SrcXprType, Functor, Hermitian2Hermitian>
{
  static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<typename DstXprType::Scalar,typename SrcXprType::Scalar> &/*func*/)
  {
    Index dstDim = src.rows();
    if(dst.rows() != dstDim) {
      dst.resize(dstDim);
    }
      
    eigen_assert(dst.rows() == src.rows() && dst.cols() == src.cols());
    dst.nestedExpression() = src.nestedExpression();
  }
  
  static void run(DstXprType &dst, const SrcXprType &src, const internal::add_assign_op<typename DstXprType::Scalar,typename SrcXprType::Scalar> &/*func*/)
  { dst.nestedExpression() += src.nestedExpression(); }
  
  static void run(DstXprType &dst, const SrcXprType &src, const internal::sub_assign_op<typename DstXprType::Scalar,typename SrcXprType::Scalar> &/*func*/)
  { dst.nestedExpression() -= src.nestedExpression(); }
};

// Hermitian matrix to Dense assignment
template< typename DstXprType, typename SrcXprType, typename Functor>
struct Assignment<DstXprType, SrcXprType, Functor, Hermitian2Dense>
{
  static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<typename DstXprType::Scalar,typename SrcXprType::Scalar> &/*func*/)
  {
    const Index dim = src.cols();
    if (!DstXprType::IsRowMajor) {
        for (Index col = 0; col < dim; ++col) {
          for (Index row = col; row < dim; ++row) {
            typename SrcXprType::Scalar val = src.coeff(row, col);
            dst.coeffRef(row, col) = val;
            dst.coeffRef(col, row) = val;
          }
        }
    } else {
      for (Index row = 0; row < dim; ++row) {
        for (Index col = 0; col < row; ++col) {
          typename SrcXprType::Scalar val = src.coeff(row, col);
          dst.coeffRef(row, col) = val;
          dst.coeffRef(col, row) = val;
        }
      }
    }
  }
  
  static void run(DstXprType &dst, const SrcXprType &src, const internal::add_assign_op<typename DstXprType::Scalar,typename SrcXprType::Scalar> &/*func*/)
  { dst.nestedExpression() += src.nestedExpression(); }
  
  static void run(DstXprType &dst, const SrcXprType &src, const internal::sub_assign_op<typename DstXprType::Scalar,typename SrcXprType::Scalar> &/*func*/)
  { dst.nestedExpression() -= src.nestedExpression(); }
};

// Hermitian product to Dense assignment
template< typename DstXprType, typename _Scalar, int _Dimension, unsigned int _UpLo, int _Storage, int _MaxDimension, typename Functor>
struct Assignment<DstXprType, Product<HermitianMatrix<_Scalar, _Dimension, _UpLo, _Storage, _MaxDimension>, HermitianMatrix<_Scalar, _Dimension, _UpLo, _Storage, _MaxDimension>, LazyProduct> , Functor, Dense2Dense>
{
  typedef Product<HermitianMatrix<_Scalar, _Dimension, _UpLo, _Storage, _MaxDimension>, HermitianMatrix<_Scalar, _Dimension, _UpLo, _Storage, _MaxDimension>, LazyProduct > SrcXprType;
  
  static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<typename DstXprType::Scalar,typename SrcXprType::Scalar> &/*func*/)
  {
    dst = src.lhs().toDenseMatrix() * src.rhs().toDenseMatrix();
  }

};

// Hermitian product to hermitian assignment
template< typename DstXprType, typename _Scalar, int _Dimension, unsigned int _UpLo, int _Storage, int _MaxDimension, typename Functor>
struct Assignment<DstXprType, Product<HermitianMatrix<_Scalar, _Dimension, _UpLo, _Storage, _MaxDimension>, HermitianMatrix<_Scalar, _Dimension, _UpLo, _Storage, _MaxDimension>, LazyProduct> , Functor, Dense2Hermitian>
{
  typedef Product<HermitianMatrix<_Scalar, _Dimension, _UpLo, _Storage, _MaxDimension>, HermitianMatrix<_Scalar, _Dimension, _UpLo, _Storage, _MaxDimension>, LazyProduct > SrcXprType;

  static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<typename DstXprType::Scalar,typename SrcXprType::Scalar> &/*func*/)
  {
    hermitian_prod_impl<DstXprType, HermitianMatrix<_Scalar, _Dimension, _UpLo, _Storage, _MaxDimension>, HermitianMatrix<_Scalar, _Dimension, _UpLo, _Storage, _MaxDimension> >::run(dst, src.lhs(), src.rhs());
  }
};

} // namespace internal

// dense * hermitian
template<typename Derived>
template<typename HermitianDerived>
EIGEN_DEVICE_FUNC inline const Product<Derived, HermitianDerived, LazyProduct>
MatrixBase<Derived>::operator*(const HermitianBase<HermitianDerived>& hermitian) const
{
  return Product<Derived, HermitianDerived, LazyProduct>(derived(), hermitian.derived());
}

}  // namespace Eigen

#endif  // EIGEN_HERMITIANMATRIX_H
