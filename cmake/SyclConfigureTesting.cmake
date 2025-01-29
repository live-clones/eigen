# SPDX-License-Identifier: MPL-2.0

set(CMAKE_CXX_STANDARD 17)
# Forward CMake options as preprocessor definitions
if(EIGEN_SYCL_USE_DEFAULT_SELECTOR)
    add_definitions(-DEIGEN_SYCL_USE_DEFAULT_SELECTOR=${EIGEN_SYCL_USE_DEFAULT_SELECTOR})
endif()
if(EIGEN_SYCL_NO_LOCAL_MEM)
    add_definitions(-DEIGEN_SYCL_NO_LOCAL_MEM=${EIGEN_SYCL_NO_LOCAL_MEM})
endif()
if(EIGEN_SYCL_LOCAL_MEM)
    add_definitions(-DEIGEN_SYCL_LOCAL_MEM=${EIGEN_SYCL_LOCAL_MEM})
endif()
if(EIGEN_SYCL_MAX_GLOBAL_RANGE)
    add_definitions(-DEIGEN_SYCL_MAX_GLOBAL_RANGE=${EIGEN_SYCL_MAX_GLOBAL_RANGE})
endif()
if(EIGEN_SYCL_LOCAL_THREAD_DIM0)
    add_definitions(-DEIGEN_SYCL_LOCAL_THREAD_DIM0=${EIGEN_SYCL_LOCAL_THREAD_DIM0})
endif()
if(EIGEN_SYCL_LOCAL_THREAD_DIM1)
    add_definitions(-DEIGEN_SYCL_LOCAL_THREAD_DIM1=${EIGEN_SYCL_LOCAL_THREAD_DIM1})
endif()
if(EIGEN_SYCL_REG_M)
    add_definitions(-DEIGEN_SYCL_REG_M=${EIGEN_SYCL_REG_M})
endif()
if(EIGEN_SYCL_REG_N)
    add_definitions(-DEIGEN_SYCL_REG_N=${EIGEN_SYCL_REG_N})
endif()
if(EIGEN_SYCL_ASYNC_EXECUTION)
    add_definitions(-DEIGEN_SYCL_ASYNC_EXECUTION=${EIGEN_SYCL_ASYNC_EXECUTION})
endif()
if(EIGEN_SYCL_DISABLE_SKINNY)
    add_definitions(-DEIGEN_SYCL_DISABLE_SKINNY=${EIGEN_SYCL_DISABLE_SKINNY})
endif()
if(EIGEN_SYCL_DISABLE_DOUBLE_BUFFER)
    add_definitions(-DEIGEN_SYCL_DISABLE_DOUBLE_BUFFER=${EIGEN_SYCL_DISABLE_DOUBLE_BUFFER})
endif()
if(EIGEN_SYCL_DISABLE_SCALAR)
    add_definitions(-DEIGEN_SYCL_DISABLE_SCALAR=${EIGEN_SYCL_DISABLE_SCALAR})
endif()
if(EIGEN_SYCL_DISABLE_GEMV)
    add_definitions(-DEIGEN_SYCL_DISABLE_GEMV=${EIGEN_SYCL_DISABLE_GEMV})
endif()
if(EIGEN_SYCL_DISABLE_ARM_GPU_CACHE_OPTIMISATION)
    add_definitions(-DEIGEN_SYCL_DISABLE_ARM_GPU_CACHE_OPTIMISATION=${EIGEN_SYCL_DISABLE_ARM_GPU_CACHE_OPTIMISATION})
endif()

if(EIGEN_SYCL_ComputeCpp)
    if(MSVC)
        list(APPEND COMPUTECPP_USER_FLAGS -DWIN32)
    else()
        list(APPEND COMPUTECPP_USER_FLAGS -Wall)
    endif()
    # The following flags are not supported by Clang and can cause warnings
    # if used with -Werror so they are removed here.
    if(COMPUTECPP_USE_COMPILER_DRIVER)
        set(CMAKE_CXX_COMPILER ${ComputeCpp_DEVICE_COMPILER_EXECUTABLE})
        string(REPLACE "-Wlogical-op" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
        string(REPLACE "-Wno-psabi" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    endif()
    list(APPEND COMPUTECPP_USER_FLAGS
            -DEIGEN_NO_ASSERTION_CHECKING=1
            -no-serial-memop
            -Xclang
            -cl-mad-enable)
endif(EIGEN_SYCL_ComputeCpp)
