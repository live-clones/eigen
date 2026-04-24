# GPU module tests. Included from unsupported/test/CMakeLists.txt inside the
# if(CUDA_FOUND AND EIGEN_TEST_CUDA) block.

find_package(CUDAToolkit QUIET)

# ei_add_gpu_test(<name> [EXTRA_LIBS <lib>...] [EXTRA_INCLUDES <dir>...] [EXTRA_DEFINES <def>...])
#
# Declares a GPU test executable built from <name>.cpp, registers it with
# CTest under the "Official;gpu" labels, links Eigen3 + CUDA runtime + any
# EXTRA_LIBS, wires it into buildtests/buildtests_gpu, and sets
# SKIP_RETURN_CODE=77 so tests can signal "skipped".
#
# GPU module tests are plain .cpp (host-compiled) by design: they orchestrate
# NVIDIA library calls and never launch kernels themselves, which avoids NVCC
# instantiating Eigen CPU packet ops for CUDA vector types.
function(ei_add_gpu_test test_name)
  cmake_parse_arguments(ARG "" "" "EXTRA_LIBS;EXTRA_INCLUDES;EXTRA_DEFINES" ${ARGN})
  add_executable(${test_name} ${test_name}.cpp)
  target_include_directories(${test_name} PRIVATE
    "${CUDA_TOOLKIT_ROOT_DIR}/include"
    "${CMAKE_CURRENT_BINARY_DIR}"
    ${ARG_EXTRA_INCLUDES})
  target_link_libraries(${test_name}
    Eigen3::Eigen CUDA::cudart_static ${ARG_EXTRA_LIBS})
  target_compile_definitions(${test_name} PRIVATE
    EIGEN_TEST_MAX_SIZE=${EIGEN_TEST_MAX_SIZE}
    EIGEN_TEST_PART_ALL=1
    ${ARG_EXTRA_DEFINES})
  add_test(NAME ${test_name} COMMAND ${test_name})
  add_dependencies(buildtests ${test_name})
  add_dependencies(buildtests_gpu ${test_name})
  set_property(TEST ${test_name} APPEND PROPERTY LABELS "Official;gpu")
  set_property(TEST ${test_name} PROPERTY SKIP_RETURN_CODE 77)
endfunction()

# DeviceMatrix core: CUDA runtime + cuBLAS + cuSOLVER (for BLAS-1 ops via GpuContext).
ei_add_gpu_test(device_matrix
  EXTRA_LIBS CUDA::cublas CUDA::cusolver CUDA::npps CUDA::nppc)

# cuBLAS integration. OFF by default; later MRs in the stack opt in.
option(EIGEN_TEST_CUBLAS "Test cuBLAS integration" OFF)
if(EIGEN_TEST_CUBLAS AND TARGET CUDA::cublas)
  ei_add_gpu_test(cublas EXTRA_LIBS CUDA::cublas CUDA::cusolver)
endif()

# cuSOLVER dense factorizations (LLT, LU, QR, SVD, self-adjoint eigensolver).
option(EIGEN_TEST_CUSOLVER "Test cuSOLVER integration" OFF)
if(EIGEN_TEST_CUSOLVER AND TARGET CUDA::cusolver)
  foreach(_cusolver_test IN ITEMS cusolver_llt cusolver_lu cusolver_qr cusolver_svd cusolver_eigen)
    ei_add_gpu_test(${_cusolver_test} EXTRA_LIBS CUDA::cusolver CUDA::cublas)
  endforeach()
endif()

# cuFFT (part of the CUDA toolkit, no separate option needed).
if(TARGET CUDA::cufft)
  ei_add_gpu_test(cufft EXTRA_LIBS CUDA::cufft CUDA::cublas)
endif()

# cuSPARSE SpMV (part of the CUDA toolkit) and the end-to-end CG test that
# exercises Eigen::ConjugateGradient with DeviceMatrix + device-resident SpMV.
if(TARGET CUDA::cusparse)
  ei_add_gpu_test(cusparse_spmv
    EXTRA_LIBS CUDA::cusparse CUDA::cublas CUDA::cusolver)
  ei_add_gpu_test(cg
    EXTRA_LIBS CUDA::cusparse CUDA::cublas CUDA::cusolver CUDA::npps CUDA::nppc)
endif()

option(EIGEN_TEST_CUSPARSE "Test cuSPARSE integration" OFF)
if(EIGEN_TEST_CUSPARSE AND TARGET CUDA::cusparse)
  ei_add_gpu_test(cusparse EXTRA_LIBS CUDA::cusparse)
endif()

# cuDSS sparse direct solvers -- distributed separately from the CUDA Toolkit.
option(EIGEN_TEST_CUDSS "Test cuDSS sparse solver integration" OFF)
if(EIGEN_TEST_CUDSS)
  find_path(CUDSS_INCLUDE_DIR cudss.h
    HINTS ${CUDSS_DIR}/include ${CUDA_TOOLKIT_ROOT_DIR}/include /usr/include)
  find_library(CUDSS_LIBRARY cudss
    HINTS ${CUDSS_DIR}/lib ${CUDSS_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib64 /usr/lib/x86_64-linux-gnu)
  if(CUDSS_INCLUDE_DIR AND CUDSS_LIBRARY)
    message(STATUS "cuDSS found: ${CUDSS_LIBRARY}")
    foreach(_cudss_test IN ITEMS cudss_llt cudss_ldlt cudss_lu)
      ei_add_gpu_test(${_cudss_test}
        EXTRA_LIBS CUDA::cusolver CUDA::cublas ${CUDSS_LIBRARY}
        EXTRA_INCLUDES ${CUDSS_INCLUDE_DIR}
        EXTRA_DEFINES EIGEN_CUDSS=1)
    endforeach()
  else()
    message(WARNING "EIGEN_TEST_CUDSS=ON but cuDSS not found. Set CUDSS_DIR.")
  endif()
endif()
