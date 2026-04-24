# GPU module tests. Included from unsupported/test/CMakeLists.txt inside the
# if(CUDA_FOUND AND EIGEN_TEST_CUDA) block.

find_package(CUDAToolkit)

# ei_add_gpu_test(<name> [EXTRA_LIBS <lib>...])
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
  cmake_parse_arguments(ARG "" "" "EXTRA_LIBS" ${ARGN})
  add_executable(${test_name} ${test_name}.cpp)
  target_include_directories(${test_name} PRIVATE
    "${CUDA_TOOLKIT_ROOT_DIR}/include"
    "${CMAKE_CURRENT_BINARY_DIR}")
  target_link_libraries(${test_name}
    Eigen3::Eigen CUDA::cudart_static ${ARG_EXTRA_LIBS})
  target_compile_definitions(${test_name} PRIVATE
    EIGEN_TEST_MAX_SIZE=${EIGEN_TEST_MAX_SIZE}
    EIGEN_TEST_PART_ALL=1)
  add_test(NAME ${test_name} COMMAND ${test_name})
  add_dependencies(buildtests ${test_name})
  add_dependencies(buildtests_gpu ${test_name})
  set_property(TEST ${test_name} APPEND PROPERTY LABELS "Official;gpu")
  set_property(TEST ${test_name} PROPERTY SKIP_RETURN_CODE 77)
endfunction()

# DeviceMatrix core: CUDA runtime only.
ei_add_gpu_test(device_matrix)

# cuBLAS integration (BLAS-3, triangular solves, rank-k updates, ...).
option(EIGEN_TEST_CUBLAS "Test cuBLAS integration" ON)
if(EIGEN_TEST_CUBLAS AND TARGET CUDA::cublas)
  ei_add_gpu_test(cublas EXTRA_LIBS CUDA::cublas CUDA::cusolver)
endif()

# cuSOLVER dense factorizations (LLT, LU, QR, SVD, self-adjoint eigensolver).
option(EIGEN_TEST_CUSOLVER "Test cuSOLVER integration" ON)
if(EIGEN_TEST_CUSOLVER AND TARGET CUDA::cusolver)
  foreach(_cusolver_test IN ITEMS cusolver_llt cusolver_lu cusolver_qr cusolver_svd cusolver_eigen)
    ei_add_gpu_test(${_cusolver_test} EXTRA_LIBS CUDA::cusolver CUDA::cublas)
  endforeach()
endif()
