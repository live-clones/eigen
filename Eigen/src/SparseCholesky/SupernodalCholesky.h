#ifndef EIGEN_SUPERNODAL_CHOLESKY_H
#define EIGEN_SUPERNODAL_CHOLESKY_H

namespace Eigen {

template <typename MatrixType_, int UpLo_, typename Ordering_>
class SupernodalCholeskyLLT : public SparseSolverBase<SupernodalCholeskyLLT<MatrixType_, UpLo_, Ordering_>> {
  using Self = SupernodalCholeskyLLT<MatrixType_, UpLo_, Ordering_>;
  using Base = SparseSolverBase<Self>;
  using Base::m_isInitialized;

 public:
  using MatrixType = MatrixType_;
  enum { UpLo = UpLo_ };
  using Scalar = typename MatrixType::Scalar;
  using RealScalar = typename MatrixType::RealScalar;
  using StorageIndex = typename MatrixType::StorageIndex;

  using Helper = Eigen::internal::simpl_chol_helper<Scalar, StorageIndex>;
  using HelperSN = Eigen::internal::supernodal_chol_helper<Scalar, StorageIndex>;

  using VectorType = Matrix<Scalar, Dynamic, 1>;
  using VectorI = Matrix<StorageIndex, Dynamic, 1>;

  using MatrixL = SparseMatrix<Scalar, ColMajor, StorageIndex>;
  using MatrixU = SparseMatrix<Scalar, ColMajor, StorageIndex>;

  using CholMatrixType = SparseMatrix<Scalar, ColMajor, StorageIndex>;

  SupernodalCholeskyLLT() = default;

  SupernodalCholeskyLLT& setShift(Scalar beta) {
    m_beta = beta;
    return *this;
  }

  SupernodalCholeskyLLT& analyzePattern(const MatrixType& A) {
    clear_();
    m_info = Eigen::Success;

    MatrixType Au = A.template triangularView<Eigen::Upper>();
    Au.makeCompressed();

    const StorageIndex n = static_cast<StorageIndex>(Au.cols());
    m_n = n;

    VectorI hadjOuter(n + 1);
    hadjOuter.setZero();
    VectorI tmp(n);
    Helper::calc_hadj_outer(n, A, hadjOuter.data());
    VectorI hadjInner(hadjOuter[n]);
    Helper::calc_hadj_inner(n, A, hadjOuter.data(), hadjInner.data(), tmp.data());

    m_parent.resize(n);
    Helper::calc_etree(n, A, m_parent.data(), tmp.data());

    VectorI firstChild(n), firstSibling(n), post(n), dfs(n);
    Helper::calc_lineage(n, m_parent.data(), firstChild.data(), firstSibling.data());
    Helper::calc_post(n, m_parent.data(), firstChild.data(), firstSibling.data(), post.data(), dfs.data());

    VectorI prevLeaf(n);
    prevLeaf.setConstant(Helper::kEmpty);
    m_colcount.resize(n);
    Helper::calc_colcount(n, hadjOuter.data(), hadjInner.data(), m_parent.data(), prevLeaf.data(), tmp.data(),
                          post.data(), m_colcount.data(), false);

    m_supe = HelperSN::compute_supernodes(m_parent, m_colcount);
    m_s_parent = HelperSN::compute_supernodal_etree(m_parent, m_supe.supernodes);

    HelperSN::allocate_supernodal_factor(m_supe.supernodes, m_supe.Snz, m_Lpx, m_Li, m_Lpi, m_ssize, m_xsize);

    VectorI ApVec(n + 1);
    ApVec = Eigen::Map<const VectorI>(Au.outerIndexPtr(), n + 1);
    VectorI AiVec(Au.nonZeros());
    AiVec = Eigen::Map<const VectorI>(Au.innerIndexPtr(), Au.nonZeros());

    HelperSN::build_supernodal_pattern_from_A(m_supe.supernodes, m_supe.sn_id, m_s_parent, ApVec, AiVec, nullptr, m_Li,
                                              m_Lpi);

    m_Lx.resize(m_Lpx[m_supe.supernodes.size() - 1]);
    m_symbolic_ok = true;
    return *this;
  }

  SupernodalCholeskyLLT& factorize(const MatrixType& A) {
    m_numeric_ok = false;
    m_info = Eigen::Success;

    if (!m_symbolic_ok) {
      m_info = Eigen::InvalidInput;
      return *this;
    }

    MatrixType Al = A.template triangularView<Eigen::Lower>();
    Al.makeCompressed();

    bool ok = HelperSN::numeric_from_A(Al, m_supe, m_Lpi, m_Lpx, m_Li, m_beta, m_Lx);
    if (!ok) {
      m_info = Eigen::NumericalIssue;
      return *this;
    }

    m_L = HelperSN::export_sparse_lower(m_supe.supernodes, m_supe.Snz, m_Lpi, m_Lpx, m_Li, m_Lx, m_n);

    m_numeric_ok = true;
    return *this;
  }

  SupernodalCholeskyLLT& compute(const MatrixType& A) { return analyzePattern(A).factorize(A); }

  const MatrixL& matrixL() const { return m_L; }
  ComputationInfo info() const { return m_info; }

 private:
  void clear_() {
    m_L.resize(0, 0);
    m_parent.resize(0);
    m_colcount.resize(0);
    m_supe.supernodes.resize(0);
    m_supe.sn_id.resize(0);
    m_supe.Snz.resize(0);
    m_s_parent.resize(0);
    m_Lpi.resize(0);
    m_Lpx.resize(0);
    m_Li.resize(0);
    m_Lx.resize(0);
    m_symbolic_ok = false;
    m_numeric_ok = false;
    m_n = 0;
    m_ssize = m_xsize = 0;
  }

  MatrixL m_L;
  VectorI m_parent;
  VectorI m_colcount;

  HelperSN::Supernodes m_supe;

  VectorI m_s_parent;
  VectorI m_Lpi, m_Lpx, m_Li;
  VectorType m_Lx;
  StorageIndex m_n = 0, m_ssize = 0, m_xsize = 0;
  Scalar m_beta = Scalar(0);
  bool m_symbolic_ok = false, m_numeric_ok = false;
  ComputationInfo m_info = Success;
};

namespace internal {

template <typename _MatrixType, int _UpLo, typename _Ordering>
struct traits<SupernodalCholeskyLLT<_MatrixType, _UpLo, _Ordering>> {
  using MatrixType = _MatrixType;
  using Scalar = typename MatrixType::Scalar;
  using StorageIndex = typename MatrixType::StorageIndex;
  enum { UpLo = _UpLo };
  using OrderingType = _Ordering;

  using MatrixL = SparseMatrix<Scalar, ColMajor, StorageIndex>;
  using MatrixU = SparseMatrix<Scalar, ColMajor, StorageIndex>;
};

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_SUPERNODAL_CHOLESKY_H