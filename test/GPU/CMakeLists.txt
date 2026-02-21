if(CUDA_FOUND AND EIGEN_TEST_CUDA)
  # Make sure to compile without the -pedantic, -Wundef, -Wnon-virtual-dtor
  # and -fno-check-new flags since they trigger thousands of compilation warnings
  # in the CUDA runtime
  string(REPLACE "-pedantic" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-Wundef" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-Wnon-virtual-dtor" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-fno-check-new" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

  if(EIGEN_TEST_CUDA_CLANG)
    string(APPEND CMAKE_CXX_FLAGS " --cuda-path=${CUDA_TOOLKIT_ROOT_DIR}")
    foreach(GPU IN LISTS EIGEN_CUDA_COMPUTE_ARCH)
      string(APPEND CMAKE_CXX_FLAGS " --cuda-gpu-arch=sm_${GPU}")
    endforeach()
    string(APPEND CMAKE_CXX_FLAGS " ${EIGEN_CUDA_CXX_FLAGS}")
  else()
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(NVCC_ARCH_FLAGS)
    list(LENGTH EIGEN_CUDA_COMPUTE_ARCH EIGEN_CUDA_COMPUTE_ARCH_SIZE)
    if(EIGEN_CUDA_COMPUTE_ARCH_SIZE)
      list(GET EIGEN_CUDA_COMPUTE_ARCH 0 EIGEN_CUDA_COMPUTE_DEFAULT)
      set(NVCC_ARCH_FLAGS " -arch=sm_${EIGEN_CUDA_COMPUTE_DEFAULT}")
    endif()
    foreach(ARCH IN LISTS EIGEN_CUDA_COMPUTE_ARCH)
      string(APPEND NVCC_ARCH_FLAGS " -gencode arch=compute_${ARCH},code=sm_${ARCH}")
    endforeach()
    set(CUDA_NVCC_FLAGS  "--expt-relaxed-constexpr -Xcudafe \"--display_error_number\" ${NVCC_ARCH_FLAGS} ${CUDA_NVCC_FLAGS} ${EIGEN_CUDA_CXX_FLAGS}")
    cuda_include_directories("${CMAKE_CURRENT_BINARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/include")
  endif()

  set(EIGEN_ADD_TEST_FILENAME_EXTENSION  "cu")

  ei_add_test(gpu_example)
  ei_add_test(gpu_basic)

  unset(EIGEN_ADD_TEST_FILENAME_EXTENSION)

endif()

# HIP unit tests
if (EIGEN_TEST_HIP)

  set(ROCM_PATH "/opt/rocm" CACHE STRING "Path to the ROCm installation.")

  if (EXISTS ${ROCM_PATH}/hip)
    set(HIP_PATH ${ROCM_PATH}/hip)
    list(APPEND CMAKE_MODULE_PATH ${HIP_PATH}/cmake)
  elseif (EXISTS ${ROCM_PATH}/lib/cmake/hip)
    set(HIP_PATH ${ROCM_PATH})
    list(APPEND CMAKE_MODULE_PATH ${HIP_PATH}/lib/cmake/hip)
  else ()
    message(FATAL_ERROR "EIGEN_TEST_HIP is ON, but could not find the ROCm installation under ${ROCM_PATH}")
  endif()

  find_package(HIP REQUIRED)
  if (HIP_FOUND)
    execute_process(COMMAND ${HIP_PATH}/bin/hipconfig --platform OUTPUT_VARIABLE HIP_PLATFORM)

    if ((${HIP_PLATFORM} STREQUAL "hcc") OR (${HIP_PLATFORM} STREQUAL "amd"))

      include_directories(${HIP_PATH}/include)

      set(EIGEN_ADD_TEST_FILENAME_EXTENSION  "cu")
      ei_add_test(gpu_basic)
      ei_add_test(gpu_example)
      unset(EIGEN_ADD_TEST_FILENAME_EXTENSION)

    elseif ((${HIP_PLATFORM} STREQUAL "nvcc") OR (${HIP_PLATFORM} STREQUAL "nvidia"))
      message(FATAL_ERROR "HIP_PLATFORM = nvcc is not supported within Eigen")
    else ()
      message(FATAL_ERROR "Unknown HIP_PLATFORM = ${HIP_PLATFORM}")
    endif()
  endif()
endif()
