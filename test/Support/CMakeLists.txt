if(CHOLMOD_FOUND AND EIGEN_BUILD_BLAS AND EIGEN_BUILD_LAPACK)
  ei_add_test(cholmod_support "" "${CHOLMOD_ALL_LIBS}")
endif()

if(UMFPACK_FOUND AND EIGEN_BUILD_BLAS)
  ei_add_test(umfpack_support "" "${UMFPACK_ALL_LIBS}")
endif()

if(KLU_FOUND AND EIGEN_BUILD_BLAS)
  ei_add_test(klu_support "" "${KLU_ALL_LIBS}")
endif()

if(SuperLU_FOUND AND EIGEN_BUILD_BLAS)
  ei_add_test(superlu_support "" "${SUPERLU_ALL_LIBS}")
endif()

if(PARDISO_FOUND)
  ei_add_test(pardiso_support "" "${PARDISO_ALL_LIBS}")
endif()

if(PASTIX_FOUND AND (SCOTCH_FOUND OR METIS_FOUND))
  ei_add_test(pastix_support "" "${PASTIX_ALL_LIBS}")
endif()

if(SPQR_FOUND AND CHOLMOD_FOUND AND EIGEN_BUILD_BLAS AND EIGEN_BUILD_LAPACK)
  ei_add_test(spqr_support "" "${SPQR_ALL_LIBS}")
endif()

if(METIS_FOUND)
  ei_add_test(metis_support "" "${METIS_LIBRARIES}")
endif()

if(Accelerate_FOUND)
  ei_add_test(accelerate_support "" "${Accelerate_ALL_LIBS}")
endif()

if(QT4_FOUND)
  ei_add_test(qtvector "" "${QT_QTCORE_LIBRARY}")
endif()
