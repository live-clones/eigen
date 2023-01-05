/*
 Copyright (c) 2011, Intel Corporation. All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of Intel Corporation nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 ********************************************************************************
 *   Content : Eigen bindings to LAPACKe
 *    Householder QR decomposition of a matrix with column pivoting based on
 *    LAPACKE_?geqp3 function.
 ********************************************************************************
*/

#ifndef EIGEN_COLPIVOTINGHOUSEHOLDERQR_LAPACKE_H
#define EIGEN_COLPIVOTINGHOUSEHOLDERQR_LAPACKE_H

#include "./InternalHeaderCheck.h"

namespace Eigen {

#if defined(EIGEN_USE_LAPACKE)

    template<typename Scalar>
    inline lapack_int call_geqp3(int matrix_layout, lapack_int m, lapack_int n, Scalar* a, lapack_int lda, lapack_int* jpvt, Scalar* tau);
    template<>
    inline lapack_int call_geqp3(int matrix_layout, lapack_int m, lapack_int n, float* a, lapack_int lda, lapack_int* jpvt, float* tau)
    { return LAPACKE_sgeqp3(matrix_layout, m, n, a, lda, jpvt, tau); }
    template<>
    inline lapack_int call_geqp3(int matrix_layout, lapack_int m, lapack_int n, double* a, lapack_int lda, lapack_int* jpvt, double* tau)
    { return LAPACKE_dgeqp3(matrix_layout, m, n, a, lda, jpvt, tau); }
    template<>
    inline lapack_int call_geqp3(int matrix_layout, lapack_int m, lapack_int n, lapack_complex_float* a, lapack_int lda, lapack_int* jpvt, lapack_complex_float* tau)
    { return LAPACKE_cgeqp3(matrix_layout, m, n, a, lda, jpvt, tau); }
    template<>
    inline lapack_int call_geqp3(int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, lapack_int lda, lapack_int* jpvt, lapack_complex_double* tau)
    { return LAPACKE_zgeqp3(matrix_layout, m, n, a, lda, jpvt, tau); }

    template <typename MatrixType>
    struct ColPivHouseholderQR_LAPACKE_impl {
      typedef typename MatrixType::Scalar Scalar;
      typedef typename MatrixType::RealScalar RealScalar;
      typedef typename internal::lapacke_helpers::translate_type_imp<Scalar>::type LapackeType;
      static constexpr lapack_int LapackeStorage = MatrixType::IsRowMajor ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;

      typedef typename internal::plain_diag_type<MatrixType>::type HCoeffsType;
      typedef PermutationMatrix<Dynamic, Dynamic, lapack_int> PermutationType;
      typedef typename internal::plain_row_type<MatrixType, Index>::type IntRowVectorType;

      static void run(MatrixType& qr, HCoeffsType& hCoeffs, PermutationType& colsPermutation,
                      IntRowVectorType& colsTranspositions, Index& nonzero_pivots, RealScalar& maxpivot,
                      bool usePrescribedThreshold, RealScalar prescribedThreshold, Index& det_p, bool& isInitialized) {
        using std::abs;
        
        isInitialized = false;
        hCoeffs.resize(qr.diagonalSize());
        colsTranspositions.resize(qr.cols());
        nonzero_pivots = 0;
        maxpivot = RealScalar(0);
        colsPermutation.resize(qr.cols());
        colsPermutation.indices().setZero();

        lapack_int rows = internal::convert_index<lapack_int, Index>(qr.rows());
        lapack_int cols = internal::convert_index<lapack_int, Index>(qr.cols());
        LapackeType* qr_data = (LapackeType*)(qr.data());
        lapack_int lda = internal::convert_index<lapack_int, Index>(qr.outerStride());
        lapack_int* perm_data = colsPermutation.indices().data();
        LapackeType* hCoeffs_data = (LapackeType*)(hCoeffs.data());

        lapack_int info = call_geqp3(LapackeStorage, rows, cols, qr_data, lda, perm_data, hCoeffs_data);
        if (info != 0) return;

        maxpivot = qr.diagonal().cwiseAbs().maxCoeff();
        hCoeffs.adjointInPlace();
        RealScalar defaultThreshold = NumTraits<Scalar>::epsilon() * RealScalar(qr.diagonalSize());
        RealScalar threshold = usePrescribedThreshold ? prescribedThreshold : defaultThreshold;
        RealScalar premultiplied_threshold = abs(maxpivot) * threshold;
        for (Index i = 0; i < qr.diagonalSize(); i++)
          if (abs(qr.coeff(i, i)) > premultiplied_threshold) nonzero_pivots++;
        colsPermutation.indices().array() -= lapack_int(1);
        det_p = colsPermutation.determinant();

        isInitialized = true;
      };
    };

    #define EIGEN_LAPACKE_QR_COLPIV(Scalar, StorageOrder)                                                       \
    template <>                                                                                                 \
    void ColPivHouseholderQR<Matrix<Scalar,Dynamic,Dynamic,StorageOrder>, lapack_int>::computeInPlace() {       \
    ColPivHouseholderQR_LAPACKE_impl<MatrixType>::run(m_qr, m_hCoeffs, m_colsPermutation, m_colsTranspositions, \
                                                      m_nonzero_pivots, m_maxpivot, m_usePrescribedThreshold,   \
                                                      m_prescribedThreshold, m_det_p, m_isInitialized); }       \

    EIGEN_LAPACKE_QR_COLPIV(float,    ColMajor)
    EIGEN_LAPACKE_QR_COLPIV(double,   ColMajor)
    EIGEN_LAPACKE_QR_COLPIV(scomplex, ColMajor)
    EIGEN_LAPACKE_QR_COLPIV(dcomplex, ColMajor)
    EIGEN_LAPACKE_QR_COLPIV(float,    RowMajor)
    EIGEN_LAPACKE_QR_COLPIV(double,   RowMajor)
    EIGEN_LAPACKE_QR_COLPIV(scomplex, RowMajor)
    EIGEN_LAPACKE_QR_COLPIV(dcomplex, RowMajor)

#endif
}  // end namespace Eigen

#endif  // EIGEN_COLPIVOTINGHOUSEHOLDERQR_LAPACKE_H
