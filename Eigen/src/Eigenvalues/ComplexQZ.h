// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Alexey Korepanov
// Copyright (C) 2025 Ludwig Striet <ludwig.striet@mathematik.uni-freiburg.de>
//
// This Source Code Form is subject to the terms of the
// Mozilla Public License v. 2.0. If a copy of the MPL
// was not distributed with this file, You can obtain one at
// https://mozilla.org/MPL/2.0/.
//
// Derived from: Eigen/src/Eigenvalues/RealQZ.h

#ifndef EIGEN_COMPLEX_QZ_H_
#define EIGEN_COMPLEX_QZ_H_ 

//#include <complex>
#include <iostream>

#include "../../Core"
#include "../../SparseCore"
#include "../../SparseQR"

namespace Eigen {

template <typename RealScalar> class ComplexQZ {

	public:

		//using RealScalar = double;
		using Scalar = std::complex<RealScalar>;
		using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
		using RealMat2 = Eigen::Matrix<RealScalar, 2, 2>;
		using RealMat3 = Eigen::Matrix<RealScalar, 3, 3>;
		using Vec = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
		using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
		using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
		using Mat2 = Eigen::Matrix<Scalar, 2, 2>;

		// TODO How do we define these properly?
		static bool is_negligible(const Scalar x, const RealScalar tol = Eigen::NumTraits<RealScalar>::epsilon())
		{
			return x.real()*x.real() + x.imag()*x.imag() < tol*tol;
		}

		Mat getMatrixQ() const;
		Mat getMatrixS() const;
		Mat getMatrixZ() const;
		Mat getMatrixT() const;

		// Let's be compatible with Eigen...
		Mat matrixQ() const { return m_Q; };
		Mat matrixS() const { return m_S; };
		Mat matrixZ() const { return m_Z; };
		Mat matrixT() const { return m_T; };

		static void printStructure(const Mat& M) {
			for (int i = 0; i < M.rows(); i++) {
				for (int j = 0; j < M.cols(); j++) {
					std::cout << (!is_negligible(M(i,j)) ? "* " : "0 ");
				}
				std::cout << '\n';
			}
			std::cout << std::flush;
		}

		static inline Mat2 computeZk2(const Mat& b);

		ComplexQZ()  {

		};

		ComplexQZ(const Mat& A, const Mat& B, bool computeQZ = true) : m_verbose(false), _A(A), _B(B),
			_computeQZ(computeQZ)
		{
			compute(A, B, computeQZ);
		}

		void setVerbose(bool verbose) {
			m_verbose = verbose;
		}

		void compute(const Mat& A, const Mat& B, bool computeQZ = true);

	private:

		//static RealScalar EPS;
		bool _computeQZ;

		bool m_verbose;

		void reduce_quasitriangular_S();

		// The above method QZ_step(p, q) first computes Q_step and and Z_step block
		// matrices, creates a large matrix, and actually does the step afterwards. 
		// This method is thought to do the steps directly on the matrices S, T, Q, Z
		void do_QZ_step(int p, int q, int iter);

		// Assume we have a 0 at T(k, k) that we want to push to T(l, l), which will
		// give a 0-entry at S(l,l-1)

		const Mat _A, _B;
		Mat m_S, m_T,
				m_Q, m_Z;
		int _n;

		RealScalar m_normOfT, m_normOfS;

		// This is basically taken from from Eigen3::RealQZ
		void hessenbergTriangular();

		void computeNorms();

		int findSmallSubdiagEntry(int l);
		int findSmallDiagEntry(int f, int l);

		void test_decomposition(bool verbose = false);

		void push_down_zero_ST(int k, int l);

}; // End of class 

//template<> double ComplexQZ<double>::EPS = 1e-15;
//template<> mpfr::mpreal ComplexQZ<mpfr::mpreal>::EPS = 1e-15;

template <typename RealScalar>
void ComplexQZ<RealScalar>::compute(const Mat& A, const Mat& B, bool computeQZ) {

	_computeQZ = computeQZ;
	_n = A.rows();

	assert(A.cols() == _n && B.rows() == _n && B.cols() == _n);

	// Copy A and B, these will be the matrices on which we operate later
	m_S = A;
	m_T = B;

	// This will initialize Q and Z and bring _S,_T to hessenberg-triangular form
	if (m_verbose)
		std::cout << "Computing Hessenberg Triangular Form..." << std::flush;
	hessenbergTriangular();
	if (m_verbose)
		std::cout << "done." << std::endl;
	//std::cout << "Running compute method..." << std::endl;
	// We assume that we already have that A is upper-Hessenberg and B is
	// upper-triangular.  This is legitimate, as we know how we can compute this
	// (we can use the respective part of the Eigen::RealQZ algorithm)

	//int q = 0;
	int l = _n-1,
			f,
			local_iter = 0;

	int maxIters = 400;

	computeNorms();

	if (m_verbose)
		std::cout << "Computing the QZ steps..." << std::endl;
	while (l > 0 && local_iter < maxIters) {
		if (m_verbose)
			std::cout << "\rl = " << l << "   " << std::flush;	
		f = findSmallSubdiagEntry(l);

		// Subdiag entry is small -> delete
		if (f > 0) {
			m_S.coeffRef(f, f-1) = Scalar(0);
		}

		if (f == l) { // One root found
			l--;
			local_iter = 0;
		}
		else if (f == l-1) { // Two roots found
			l -= 2;
			local_iter = 0;
			// TODO is it necessary that we split-off the rows NOW? Probably, we can do it later.
		}
		else {
			int z = findSmallDiagEntry(f, l);
			if (z >= f) {
				push_down_zero_ST(z, l);
			}
			else {
				do_QZ_step(f, _n-l-1, local_iter);
				local_iter++;
			}
		}
	}
	if (m_verbose)
		std::cout << "\rl = " << l << " done" << std::endl;

	if (local_iter == maxIters) {
		std::cout << "No convergence was achieved after " << local_iter << " iterations." << std::endl;
	}

	//std::cout << "Running compute() method using EPS = " << EPS << "..." << std::endl;
	//std::cout << "iter = " << iter << std::endl;	

	reduce_quasitriangular_S();

}

template <typename RealScalar>
void ComplexQZ<RealScalar>::reduce_quasitriangular_S() {

	//Assume we are sure that we have T upper-diagonal and S
	//reduced-upper-hessenberg now.
	if (m_verbose)
		std::cout << "Reducing quasitriangular S..." << std::endl;
	for (int i = 0; i < _n-1; i++) { // i is column index

		if (m_verbose)
			std::cout << "\ri = " << i << "/" << _n-1 << std::flush;
		
		if (!is_negligible(m_S(i+1, i))) {

			// We have found a non-zero on the subdiagonal and want to eliminate it
			if (is_negligible(m_T(i,i)) || is_negligible(m_T(i+1,i+1)))
				continue;

			Mat2 A = m_S.block(i, i, 2, 2),
					 B = m_T.block(i, i, 2, 2);

			Scalar mu = A(0,0)/B(0,0);
			Scalar a12_bar = A(0,1) - mu*B(0,1);
			Scalar a22_bar = A(1,1) - mu*B(1,1);

			Scalar p = Scalar(1)/Scalar(2)*( a22_bar/B(1,1) - B(0,1)*A(1,0)/(B(0,0) * B(1,1)) );

			// TODO DO WE NEED this assertion?
			//assert(!is_negligible(p.real()));

			RealScalar sgn_p = p.real() >= RealScalar(0) ? RealScalar(1) : RealScalar(-1);

			Scalar q = A(1,0)*a12_bar/(B(0,0)*B(1,1));

			Scalar r = p*p+q;

			Scalar lambda = mu + p + sgn_p*std::sqrt(r);
			//assert(is_negligible( (A-lambda*B).determinant() ));

			Mat2 E = A-lambda*B;

			int l;
			E.rowwise().norm().maxCoeff(&l);

			Eigen::JacobiRotation<Scalar> G;
			G.makeGivens(E(l, 1), E(l, 0));

			m_S.applyOnTheRight(i, i+1, G.adjoint());
			m_T.applyOnTheRight(i, i+1, G.adjoint());

			if (_computeQZ)
				m_Z.applyOnTheLeft(i, i+1, G);

			Mat2 Si = m_S.block(i,i,2,2), Ti = m_T.block(i,i,2,2);
			Mat2 C = Si.norm() < (lambda*Ti).norm() ? Si : lambda*Ti;

			G.makeGivens(C(0,0), C(1,0));
			m_S.applyOnTheLeft(i, i+1, G.adjoint());
			m_T.applyOnTheLeft(i, i+1, G.adjoint());

			if (_computeQZ)
				m_Q.applyOnTheRight(i, i+1, G);
		
		}
	}

	if (m_verbose)
		std::cout << "done." << std::endl;

}

template <typename RealScalar>
void ComplexQZ<RealScalar>::test_decomposition(bool verbose) {

	if (_computeQZ) {
		RealScalar err_A = (_A - m_Q*m_S*m_Z).cwiseAbs().maxCoeff();
		RealScalar err_B = (_B - m_Q*m_T*m_Z).cwiseAbs().maxCoeff();

		if (verbose) {
			std::cout << "err_A = " << err_A << std::endl;	
			std::cout << "err_B = " << err_B << std::endl;	
		}
	}
};

// This is basically taken from from Eigen3::RealQZ
template <typename RealScalar>
void ComplexQZ<RealScalar>::hessenbergTriangular() {

	//const Index dim = m_S.cols();

	// Count number of non-zero entries of m_T. We will use this to decide if we
	// are going to use a HouseholderQR decomposition for dense matrices or a 
	// SparseQR decomposition for sparse matrices.

	using Triplet = Eigen::Triplet<Scalar>;
	std::vector<Triplet> T_triplets;

	for (int i = 0; i < m_T.rows(); i++) {
		for (int j = 0; j < m_T.cols(); j++) {
			if (std::abs(m_T(i,j)) > Eigen::NumTraits<RealScalar>::epsilon())
				T_triplets.push_back(Triplet(i, j, m_T(i,j)));
		}
	}

	Eigen::SparseMatrix<Scalar, Eigen::ColMajor> T_sparse(m_T.rows(), m_T.cols());
	T_sparse.setFromTriplets(T_triplets.begin(), T_triplets.end());
	T_sparse.makeCompressed();

	Eigen::SparseQR<Eigen::SparseMatrix<Scalar, Eigen::ColMajor>, Eigen::NaturalOrdering<int> >
		sparseQR;

	if (m_verbose)
		std::cout << "Computing QR decomposition of T..." << std::flush;
	sparseQR.setPivotThreshold(RealScalar(0)); // This prevends algorithm from doing pivoting
	sparseQR.compute(T_sparse);
	if (m_verbose)
		std::cout << "done" << std::endl;

	// perform QR decomposition of T, overwrite T with R, save Q
	//Eigen::HouseholderQR<Mat> qrT(m_T);
	m_T = sparseQR.matrixR();
	m_T.template triangularView<Eigen::StrictlyLower>().setZero();

	m_Q = sparseQR.matrixQ();
	// overwrite S with Q* S
	//m_S.applyOnTheLeft(m_Q.adjoint()); // Correct line
	//m_S.rowwise().applyOnTheLeft(sparseQR.matrixQ().adjoint());
	m_S = sparseQR.matrixQ().adjoint() *m_S;

	m_Z = Mat::Identity(_n, _n);


	int total_steps = 0, steps = 0;
	for (int j=0; j<=_n-3; j++) {
		for (int i=_n-1; i>=j+2; i--) {
			total_steps++;
		}
	}

	// reduce S to upper Hessenberg with Givens rotations
	for (int j=0; j<=_n-3; j++) {
		for (int i=_n-1; i>=j+2; i--) {
			Eigen::JacobiRotation<Scalar> G;
			// kill S(i,j)
			//if(!numext::is_exactly_zero(_S.coeff(i, j)))
			if(m_S.coeff(i, j) != Scalar(0))
			{
				// This is the adapted code
				G.makeGivens(m_S.coeff(i-1,j), m_S.coeff(i,j), &m_S.coeffRef(i-1, j));
				m_S.coeffRef(i, j) = Scalar(0);

				m_T.rightCols(_n-i+1).applyOnTheLeft(i-1,i,G.adjoint());
				m_S.rightCols(_n-j-1).applyOnTheLeft(i-1,i,G.adjoint());

				// This is what we want to achieve
				assert(is_negligible(m_S(i, j)));
				m_S(i, j) = Scalar(0);

				// update Q
				if (_computeQZ)
					m_Q.applyOnTheRight(i-1,i,G);
			}

			//if(!numext::is_exactly_zero(m_T.coeff(i, i - 1)))
			if(m_T.coeff(i, i - 1) != Scalar(0))
			{
				// Compute rotation and update matrix T
				G.makeGivens(m_T.coeff(i,i), m_T.coeff(i,i-1), &m_T.coeffRef(i,i));
				m_T.topRows(i).applyOnTheRight(i-1,i,G.adjoint());
				m_T.coeffRef(i, i-1) = Scalar(0);

				// Update matrix S
				// This works (and we can, probably, not apply this only to some top rows).
				m_S.applyOnTheRight(i-1,i,G.adjoint());

				assert(is_negligible(m_T(i,i-1)));
				m_T(i, i-1) = Scalar(0);
				// update Z
				if (_computeQZ)
					m_Z.applyOnTheLeft(i-1,i,G);

			}
			steps++;
			if (m_verbose)
				std::cout << "\rdone: " << steps << "/" << total_steps << std::flush;
		}
	}
}

template <typename RealScalar>
inline typename ComplexQZ<RealScalar>::Mat2 ComplexQZ<RealScalar>::computeZk2(const Mat& b) {
	assert(b.rows() == 1 && b.cols() == 2);

	// All of this can probably be shortened, but it works for the moment.
	RealMat2 S;
	S << 0, 1,
			 1, 0;
	//PermutationMatrix<int> 
	//Eigen::PermutationMatrix<2, 2, int> S(Eigen::Vector2i(1, 0));

	Mat bprime = S*b.adjoint();

	Eigen::JacobiRotation<Scalar> J;
	J.makeGivens(bprime(0), bprime(1));

	Mat2 Z = S;
	Z.applyOnTheLeft(0, 1, J);
	Z = S*Z;

	return Z;
}

template <typename RealScalar>
void ComplexQZ<RealScalar>::do_QZ_step(int p, int q, int iter) {

	const std::function<Scalar(int, int)> a = [p,this](int i, int j) {
		return m_S(p+i-1, p+j-1);
	};
	const std::function<Scalar(int, int)> b = [p, this](int i, int j) {
		return m_T(p+i-1, p+j-1);
	};

	const int m = _n-p-q; // Size of the inner block

	Scalar x,y,z;

	// TODO in the original Eigen-Implementation, they do "exceptional shifts" some
	// times...why?
	Scalar W1 = a(m-1,m-1)/b(m-1,m-1) - a(1,1)/b(1,1),
				 W2 = a(m,m)/b(m,m) - a(1,1)/b(1,1),
				 W3 = a(m,m-1)/b(m-1,m-1);
	// Taken from bachelor thesis Gnutzmann/Skript by Stefan Funken
	x = (W1*W2 - a(m-1,m)/b(m,m) * W3 + W3*b(m-1,m)/b(m,m)*a(1,1)/b(1,1))
		* b(1,1)/a(2,1) + a(1,2)/b(2,2) - a(1,1)/b(1,1)*b(1,2)/b(2,2);
	y = (a(2,2)/b(2,2) - a(1,1)/b(1,1)) - a(2,1)/b(1,1)*b(1,2)/b(2,2) - W1 - W2+ W3*(b(m-1,m)/b(m,m));
	z = a(3,2)/b(2,2);

	Vec ws1(2*_n), ws2(2*_n); // Temporary data
	Mat X(3, 1);
	const Eigen::PermutationMatrix<3, 3, int> S3(Eigen::Vector3i(2, 0, 1));

	for (int k = p; k < p+m-2; k++) {
		X << x, y, z;

		Vec2 ess; Scalar tau; RealScalar beta;

		X.makeHouseholder(ess, tau, beta);

		// The permutations are needed because the makeHouseHolder-method computes
		// the householder transformation in a way that the vector is reflected to
		// (1 0 ... 0) instead of (0 ... 0 1)
		// These lines works
		//_S.middleRows(k, 3).applyHouseholderOnTheLeft(ess, tau, ws.data());
		//_T.middleRows(k, 3).applyHouseholderOnTheLeft(ess, tau, ws.data());
		m_S.middleRows(k, 3).rightCols(std::min(_n, _n-k+1)).applyHouseholderOnTheLeft(ess, tau, ws1.data());
		m_T.middleRows(k, 3).rightCols(_n-k).applyHouseholderOnTheLeft(ess, tau, ws1.data());

		if (_computeQZ)
			m_Q.middleCols(k, 3).applyHouseholderOnTheRight(ess, std::conj(tau), ws1.data());

		// Compute Matrix Zk1 s.t. (b(k+2,k) ... b(k+2, k+2)) Zk1 = (0,0,*)
		Vec3 bprime = (m_T.block(k+2, k, 1, 3)*S3).adjoint();
		bprime.makeHouseholder(ess, tau, beta);

		m_S.middleCols(k, 3).topRows(std::min(k+4, _n)).applyOnTheRight(S3);
		m_S.middleCols(k, 3).topRows(std::min(k+4, _n)).applyHouseholderOnTheRight(ess, std::conj(tau), ws1.data());
		m_S.middleCols(k, 3).topRows(std::min(k+4, _n)).applyOnTheRight(S3.transpose());

		m_T.middleCols(k, 3).topRows(std::min(k+3, _n)).applyOnTheRight(S3);
		m_T.middleCols(k, 3).topRows(std::min(k+3, _n)).applyHouseholderOnTheRight(ess, std::conj(tau), ws2.data());
		m_T.middleCols(k, 3).topRows(std::min(k+3, _n)).applyOnTheRight(S3.transpose());

		if (_computeQZ) {
			m_Z.middleRows(k, 3).applyOnTheLeft(S3.transpose());
			m_Z.middleRows(k, 3).applyHouseholderOnTheLeft(ess, tau, ws1.data());
			m_Z.middleRows(k, 3).applyOnTheLeft(S3);
		}

		Mat2 Zk2 = computeZk2(m_T.block(k+1, k, 1, 2));
		m_S.middleCols(k, 2).topRows(std::min(k+4, _n)).applyOnTheRight(Zk2);
		m_T.middleCols(k, 2).topRows(std::min(k+3, _n)).applyOnTheRight(Zk2);

		if (_computeQZ)
			m_Z.middleRows(k, 2).applyOnTheLeft(Zk2.adjoint());

		x = m_S(k+1, k);
		y = m_S(k+2, k);
		if (k < p+m-3) {
			z = m_S(k+3, k);
		}
	};

	//Find a Householdermartirx Qn1 s.t. Qn1 (x y)^T = (* 0)
	Eigen::JacobiRotation<Scalar> J;
	J.makeGivens(x, y);

	m_S.middleRows(p+m-2, 2).applyOnTheLeft(0, 1, J.adjoint());
	m_T.middleRows(p+m-2, 2).applyOnTheLeft(0, 1, J.adjoint());

	if (_computeQZ)
		m_Q.middleCols(p+m-2, 2).applyOnTheRight(0, 1, J);

	// Find a Householdermatrix Zn1 s.t. (b(n,n-1) b(n,n)) * Zn1 = (0 *)
	Mat2 Zn1 = computeZk2(m_T.block(p+m-1, p+m-2, 1, 2));
	//{
	//	Mat b = _T.block(p+m-1, p+m-2, 1, 2);
	//	b = b*Zn1;
	//	// TODO actually, it should be...why isn't it?
	//	//assert(is_negligible(b(0)));
	//	assert(std::abs(b(0)) < 1e-10);
	//}

	m_S.middleCols(p+m-2, 2).applyOnTheRight(Zn1);
	m_T.middleCols(p+m-2, 2).applyOnTheRight(Zn1);

	if (_computeQZ)
		m_Z.middleRows(p+m-2, 2).applyOnTheLeft(Zn1.adjoint());
}


// We have a 0 at T(k,k) and want to "push it down" to T(l,l)
template <typename RealScalar>
void ComplexQZ<RealScalar>::push_down_zero_ST(int k, int l) {
	// Test Preconditions
	//assert(is_negligible(_T(k, k), EPS*m_normOfT));
	//assert(is_negligible(_T(k, k)));

	Eigen::JacobiRotation<Scalar> J;
	for (int j = k+1; j <= l; j++) {

		// Create a 0 at _T(j, j)
		J.makeGivens(m_T(j-1,j), m_T(j,j), &m_T.coeffRef(j-1, j));
		m_T.rightCols(_n-j-1).applyOnTheLeft(j-1, j, J.adjoint());
		m_T.coeffRef(j, j) = Scalar(0);

		//assert(j < _n+2); // This ensures n-j+2 > 0
		//assert(2 <= j); // This ensures n-j+2 <= n
		//_S.rightCols(_n-j+2).applyOnTheLeft(j-1, j, J.adjoint());
		m_S.applyOnTheLeft(j-1, j, J.adjoint());

		if (_computeQZ)
			m_Q.applyOnTheRight(j-1, j, J);

		// Delete the non-desired non-zero at _S(j, j-2)
		// TODO The following line sometimes made the program crash. The entering
		// if-statement seems to fix this issue (which somehow makes sense...)
		if (j > 1) {
			J.makeGivens(std::conj(m_S(j, j-1)), std::conj(m_S(j, j-2)));
			m_S.applyOnTheRight(j-1, j-2, J);
			//assert(is_negligible(_S(j, j-2)));
			m_S(j, j-2) = Scalar(0);
			m_T.applyOnTheRight(j-1, j-2, J);

			if (_computeQZ)
				m_Z.applyOnTheLeft(j-1, j-2, J.adjoint());
		}
	}

	// Assume we have the desired structure now, up to the non-zero entry at
	// _S(l, l-1) which we will delete through a last right-jacobi-rotation
	J.makeGivens(std::conj(m_S(l, l)), std::conj(m_S(l, l-1)));
	m_S.topRows(l+1).applyOnTheRight(l, l-1, J);
	//assert(is_negligible(_S(l, l-1)));
	//if (!is_negligible(_S(l,l-1))) {
	//	std::cout << "_S(l,l-1) = " << _S(l,l-1) << std::endl;	
	//	std::cout << "(Should be 0)" << std::endl;
	//	std::cin.ignore();
	//}
	m_S(l, l-1) = Scalar(0);
	m_T.topRows(l+1).applyOnTheRight(l, l-1, J);

	if (_computeQZ)
		m_Z.applyOnTheLeft(l, l-1, J.adjoint());

	// Test Postconditions
	assert(is_negligible(m_T(l,l)) && is_negligible(m_S(l,l-1)));
	m_T(l, l) = Scalar(0);
	m_S(l, l-1) = Scalar(0);
};

template <typename RealScalar>
void ComplexQZ<RealScalar>::computeNorms()
{
	const int size = m_S.cols();
	m_normOfS = RealScalar(0);
	m_normOfT = RealScalar(0);
	for (int j = 0; j < size; ++j)
	{
		m_normOfS += m_S.col(j).segment(0, (std::min)(size,j+2)).cwiseAbs().sum();
		m_normOfT += m_T.row(j).segment(j, size - j).cwiseAbs().sum();
	}

	//std::cout << "m_normOfT = " << m_normOfT << std::endl;	
	//std::cout << "m_normOfS = " << m_normOfS << std::endl;	
	//std::cin.ignore();
};

// Copied from Eigen3 RealQZ implementation
template<typename RealScalar>
inline int ComplexQZ<RealScalar>::findSmallSubdiagEntry(int iu)
{
	using std::abs;
	int res = iu;
	while (res > 0)
	{
		RealScalar s = abs(m_S.coeff(res-1,res-1)) + abs(m_S.coeff(res,res));
		if (s == Scalar(0))
			s = m_normOfS;
		if (abs(m_S.coeff(res,res-1)) < Eigen::NumTraits<RealScalar>::epsilon() * s)
			break;
		res--;
	}
	return res;
}

// Copied from Eigen3 RealQZ implementation
template<typename RealScalar> inline int ComplexQZ<RealScalar>::findSmallDiagEntry(int f, int l)
{
	using std::abs;
	int res = l;
	while (res >= f) {
		if (abs(m_T.coeff(res,res)) <= Eigen::NumTraits<RealScalar>::epsilon() * m_normOfT)
			break;
		res--;
	}
	return res;
}


template <typename RealScalar>
typename ComplexQZ<RealScalar>::Mat ComplexQZ<RealScalar>::getMatrixQ() const {
	return m_Q;
}

template <typename RealScalar>
typename ComplexQZ<RealScalar>::Mat ComplexQZ<RealScalar>::getMatrixZ() const {
	return m_Z;
}

template <typename RealScalar>
typename ComplexQZ<RealScalar>::Mat ComplexQZ<RealScalar>::getMatrixT() const {
	return m_T;
}

template <typename RealScalar>
typename ComplexQZ<RealScalar>::Mat ComplexQZ<RealScalar>::getMatrixS() const {
	return m_S;
}

}



#endif // _COMPLEX_QZ_H_
