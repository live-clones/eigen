# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

set(Eigen_HEADERS
  AdolcForward
  AlignedVector3
  ArpackSupport
  AutoDiff
  BVH
  EulerAngles
  FFT
  GPU
  IterativeSolvers
  KroneckerProduct
  LevenbergMarquardt
  MatrixFunctions
  MPRealSupport
  NNLS
  NonLinearOptimization
  NumericalDiff
  OpenGLSupport
  Polynomials
  SparseExtra
  SpecialFunctions
  Splines
  Tensor
  TensorSymmetry
  ThreadPool
  )

install(FILES
  ${Eigen_HEADERS}
  DESTINATION ${INCLUDE_INSTALL_DIR}/contrib/Eigen COMPONENT Devel
  )

install(DIRECTORY src DESTINATION ${INCLUDE_INSTALL_DIR}/contrib/Eigen COMPONENT Devel FILES_MATCHING PATTERN "*.h")
