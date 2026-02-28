# CMake Toolchain File for Hexagon Cross-Compilation
# Based on: https://github.com/quic/toolchain_for_hexagon/blob/master/examples/README.md
# License: BSD 3-Clause (same as Hexagon toolchain project)
#
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=HexagonToolchain.cmake [other options] <source_dir>
#
# Environment variables (optional):
#   HEXAGON_TOOLCHAIN_ROOT - Path to toolchain root directory
#   HEXAGON_ARCH - Architecture version (default: v68)

# Toolchain configuration
if(NOT DEFINED HEXAGON_TOOLCHAIN_ROOT)
    if(DEFINED ENV{HEXAGON_TOOLCHAIN_ROOT})
        set(HEXAGON_TOOLCHAIN_ROOT $ENV{HEXAGON_TOOLCHAIN_ROOT})
    else()
        # Default toolchain location
        set(HEXAGON_TOOLCHAIN_ROOT "/home/developer/workspace/tools/clang+llvm-20.1.4-cross-hexagon-unknown-linux-musl")
    endif()
endif()

# Architecture version
if(NOT DEFINED HEXAGON_ARCH)
    if(DEFINED ENV{HEXAGON_ARCH})
        set(HEXAGON_ARCH $ENV{HEXAGON_ARCH})
    else()
        set(HEXAGON_ARCH "v68")
    endif()
endif()

# Auto-detect host architecture and set appropriate paths
execute_process(COMMAND uname -m OUTPUT_VARIABLE HOST_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
if(HOST_ARCH STREQUAL "aarch64")
    # ARM64 native paths
    set(HEXAGON_TOOLCHAIN_BIN "${HEXAGON_TOOLCHAIN_ROOT}/aarch64-linux-musl/bin")
    set(HEXAGON_SYSROOT "${HEXAGON_TOOLCHAIN_ROOT}/aarch64-linux-musl/target/hexagon-unknown-linux-musl")
    message(STATUS "Using ARM64 native toolchain paths")
else()
    # x86_64 paths (default)
    set(HEXAGON_TOOLCHAIN_BIN "${HEXAGON_TOOLCHAIN_ROOT}/x86_64-linux-gnu/bin")
    set(HEXAGON_SYSROOT "${HEXAGON_TOOLCHAIN_ROOT}/x86_64-linux-gnu/target/hexagon-unknown-linux-musl")
    message(STATUS "Using x86_64 toolchain paths")
endif()

# Verify toolchain exists
if(NOT EXISTS "${HEXAGON_TOOLCHAIN_BIN}")
    message(FATAL_ERROR 
        "Hexagon toolchain not found at: ${HEXAGON_TOOLCHAIN_BIN}\n"
        "Please set HEXAGON_TOOLCHAIN_ROOT to the correct path or ensure toolchain is extracted."
    )
endif()

message(STATUS "Using Hexagon toolchain: ${HEXAGON_TOOLCHAIN_ROOT}")
message(STATUS "Hexagon architecture: ${HEXAGON_ARCH}")

# System identification
set(CMAKE_SYSTEM_NAME Linux CACHE STRING "")
set(CMAKE_SYSTEM_PROCESSOR hexagon CACHE STRING "")

# Target specification
set(CMAKE_C_COMPILER_TARGET hexagon-unknown-linux-musl CACHE STRING "")
set(CMAKE_CXX_COMPILER_TARGET hexagon-unknown-linux-musl CACHE STRING "")

# Compiler settings
set(CMAKE_C_COMPILER "${HEXAGON_TOOLCHAIN_BIN}/hexagon-unknown-linux-musl-clang" CACHE FILEPATH "")
set(CMAKE_CXX_COMPILER "${HEXAGON_TOOLCHAIN_BIN}/hexagon-unknown-linux-musl-clang++" CACHE FILEPATH "")
set(CMAKE_ASM_COMPILER "${HEXAGON_TOOLCHAIN_BIN}/hexagon-unknown-linux-musl-clang" CACHE FILEPATH "")

# Binary utilities
set(CMAKE_AR "${HEXAGON_TOOLCHAIN_BIN}/hexagon-unknown-linux-musl-ar" CACHE FILEPATH "")
set(CMAKE_RANLIB "${HEXAGON_TOOLCHAIN_BIN}/hexagon-unknown-linux-musl-ranlib" CACHE FILEPATH "")
set(CMAKE_OBJCOPY "${HEXAGON_TOOLCHAIN_BIN}/hexagon-unknown-linux-musl-objcopy" CACHE FILEPATH "")
set(CMAKE_OBJDUMP "${HEXAGON_TOOLCHAIN_BIN}/hexagon-unknown-linux-musl-objdump" CACHE FILEPATH "")
set(CMAKE_NM "${HEXAGON_TOOLCHAIN_BIN}/hexagon-unknown-linux-musl-nm" CACHE FILEPATH "")
set(CMAKE_STRIP "${HEXAGON_TOOLCHAIN_BIN}/hexagon-unknown-linux-musl-strip" CACHE FILEPATH "")

# Cross-compilation settings
set(CMAKE_CROSSCOMPILING ON CACHE BOOL "")

# QEMU emulator configuration with alignment tolerance options
# Default: Strict alignment (hardware-accurate)
# Test mode: Relaxed alignment for compatibility with Eigen test suite
option(EIGEN_TEST_QEMU_RELAX_ALIGNMENT "Enable relaxed memory alignment in QEMU for testing" OFF)

if(EIGEN_TEST_QEMU_RELAX_ALIGNMENT)
    message(STATUS "QEMU alignment: RELAXED (test mode) - similar to x86-64 tolerance")
    # QEMU options for relaxed alignment:
    # -d guest_errors: Log but don't crash on guest OS errors (like misaligned access)
    # -cpu any: Use generic CPU model (may be more tolerant)
    # Note: These flags make QEMU more tolerant of alignment issues for testing
    set(CMAKE_CROSSCOMPILING_EMULATOR "${HEXAGON_TOOLCHAIN_BIN}/qemu-hexagon;-d;guest_errors;-cpu;any" CACHE FILEPATH "")
    add_definitions(-DEIGEN_QEMU_RELAXED_ALIGNMENT=1)
else()
    message(STATUS "QEMU alignment: STRICT (hardware-accurate)")
    set(CMAKE_CROSSCOMPILING_EMULATOR "${HEXAGON_TOOLCHAIN_BIN}/qemu-hexagon" CACHE FILEPATH "")
endif()

# Architecture-specific settings
set(CMAKE_SIZEOF_VOID_P 4 CACHE STRING "")
set(CMAKE_C_COMPILER_FORCED ON CACHE BOOL "")
set(CMAKE_CXX_COMPILER_FORCED ON CACHE BOOL "")

# C++ standard library features (required for LLVM runtimes)
set(CMAKE_CXX_COMPILE_FEATURES cxx_std_17;cxx_std_14;cxx_std_11;cxx_std_03;cxx_std_98 CACHE STRING "")

# Sysroot configuration
set(CMAKE_SYSROOT "${HEXAGON_SYSROOT}" CACHE PATH "")
set(CMAKE_FIND_ROOT_PATH "${HEXAGON_SYSROOT}" CACHE PATH "")

# Search paths (prevent CMake from looking in host system)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Hexagon-specific compiler flags
# Based on: https://gitlab.com/libeigen/eigen/-/blame/3cd872b7c3700eee35616f74afe3ff56166dd7f5/cmake/HexagonToolchain.cmake#L120
set(HEXAGON_BASE_FLAGS "-m${HEXAGON_ARCH} -G0 -fPIC --target=hexagon-unknown-linux-musl")

# HVX (Hexagon Vector eXtensions) support - conditional based on EIGEN_TEST_HVX option
# Default: OFF (disabled) for baseline compatibility
# Enable with: -DEIGEN_TEST_HVX=ON
option(EIGEN_TEST_HVX "Enable HVX SIMD vectorization support for Eigen tests" OFF)

if(EIGEN_TEST_HVX)
    message(STATUS "HVX support: ENABLED (EIGEN_TEST_HVX=ON)")
    set(HEXAGON_HVX_FLAGS "-mhvx -mhvx-length=128B")
    set(HEXAGON_BASE_FLAGS "${HEXAGON_BASE_FLAGS} ${HEXAGON_HVX_FLAGS}")
    
    # Add preprocessor definition for HVX-aware code
    add_definitions(-DEIGEN_TEST_HVX=1)
else()
    message(STATUS "HVX support: DISABLED (default). Use -DEIGEN_TEST_HVX=ON to enable.")
endif()

# C++ standard library settings for libc++
set(HEXAGON_CXX_FLAGS "${HEXAGON_BASE_FLAGS} -stdlib=libc++")

# Linker flags with correct sysroot
set(HEXAGON_LINKER_FLAGS "-fuse-ld=lld --sysroot=${HEXAGON_SYSROOT}")

# Apply flags
set(CMAKE_C_FLAGS_INIT "${HEXAGON_BASE_FLAGS}" CACHE STRING "")
set(CMAKE_CXX_FLAGS_INIT "${HEXAGON_CXX_FLAGS}" CACHE STRING "")
set(CMAKE_EXE_LINKER_FLAGS_INIT "${HEXAGON_LINKER_FLAGS}" CACHE STRING "")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${HEXAGON_LINKER_FLAGS}" CACHE STRING "")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "${HEXAGON_LINKER_FLAGS}" CACHE STRING "")

# Cache variables for user override
set(HEXAGON_TOOLCHAIN_ROOT "${HEXAGON_TOOLCHAIN_ROOT}" CACHE PATH "Path to Hexagon toolchain root")
set(HEXAGON_ARCH "${HEXAGON_ARCH}" CACHE STRING "Hexagon architecture version")

# Option to enable Eigen test alignment
option(ENABLE_EIGEN_TEST_ALIGN "Enable Eigen test alignment" ON)
if(ENABLE_EIGEN_TEST_ALIGN)
    add_definitions(-DEIGEN_TEST_ALIGN)
endif()

# Pre-configure standard math library for cross-compilation
# This avoids the CMake test that tries to compile and run a test program
set(STANDARD_MATH_LIBRARY_FOUND TRUE CACHE BOOL "Math library found" FORCE)
set(STANDARD_MATH_LIBRARY "m" CACHE STRING "Standard math library" FORCE)

# Override the specific cache variables that CMake tests
set(standard_math_library_linked_to_automatically FALSE CACHE BOOL "Math library auto-linked" FORCE)
set(standard_math_library_linked_to_as_m TRUE CACHE BOOL "Math library via -lm" FORCE)