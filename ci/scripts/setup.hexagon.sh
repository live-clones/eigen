#!/bin/bash

# Install and configure Hexagon toolchain for CI
# This file is part of Eigen, a lightweight C++ template library
# for linear algebra.
#
# Copyright (C) 2023, The Eigen Authors
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

set -e

# Color output for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Hexagon Toolchain Setup ===${NC}"

# Configuration
HEXAGON_TOOLCHAIN_VERSION=${EIGEN_CI_HEXAGON_TOOLCHAIN_VERSION:-"20.1.4"}
HEXAGON_ARCH=${EIGEN_CI_HEXAGON_ARCH:-"v68"}
TOOLCHAIN_DIR="/opt/hexagon-toolchain"
CACHE_DIR=".cache/hexagon-toolchain"

# Create cache directory
mkdir -p ${CACHE_DIR}

# Check if toolchain is already cached
CACHED_TOOLCHAIN="${CACHE_DIR}/clang+llvm-${HEXAGON_TOOLCHAIN_VERSION}-cross-hexagon-unknown-linux-musl.tar.xz"

if [ -f "${CACHED_TOOLCHAIN}" ]; then
    echo -e "${GREEN}Using cached Hexagon toolchain ${HEXAGON_TOOLCHAIN_VERSION}${NC}"
    TOOLCHAIN_ARCHIVE="${CACHED_TOOLCHAIN}"
else
    echo -e "${YELLOW}Downloading Hexagon toolchain ${HEXAGON_TOOLCHAIN_VERSION}...${NC}"
    TOOLCHAIN_URL="https://github.com/quic/toolchain_for_hexagon/releases/download/clang-${HEXAGON_TOOLCHAIN_VERSION}/clang+llvm-${HEXAGON_TOOLCHAIN_VERSION}-cross-hexagon-unknown-linux-musl.tar.xz"
    
    # Download with retry mechanism
    TOOLCHAIN_ARCHIVE="/tmp/hexagon-toolchain.tar.xz"
    for attempt in 1 2 3; do
        if wget --timeout=30 --tries=3 -O "${TOOLCHAIN_ARCHIVE}" "${TOOLCHAIN_URL}"; then
            # Cache the downloaded toolchain
            cp "${TOOLCHAIN_ARCHIVE}" "${CACHED_TOOLCHAIN}"
            echo -e "${GREEN}Toolchain downloaded and cached successfully${NC}"
            break
        else
            echo -e "${YELLOW}Download attempt ${attempt} failed, retrying...${NC}"
            if [ ${attempt} -eq 3 ]; then
                echo -e "${RED}Failed to download Hexagon toolchain after 3 attempts${NC}"
                exit 1
            fi
            sleep 5
        fi
    done
fi

# Extract toolchain
echo -e "${YELLOW}Extracting Hexagon toolchain to ${TOOLCHAIN_DIR}...${NC}"
mkdir -p ${TOOLCHAIN_DIR}
tar -xf ${TOOLCHAIN_ARCHIVE} -C ${TOOLCHAIN_DIR} --strip-components=1

# Verify toolchain extraction
if [ ! -d "${TOOLCHAIN_DIR}/x86_64-linux-gnu" ]; then
    echo -e "${RED}Error: Toolchain extraction failed - missing x86_64-linux-gnu directory${NC}"
    exit 1
fi

# Set up environment variables
echo -e "${YELLOW}Configuring Hexagon environment...${NC}"
export HEXAGON_TOOLCHAIN_ROOT=${TOOLCHAIN_DIR}
export HEXAGON_ARCH=${HEXAGON_ARCH}
export PATH="${TOOLCHAIN_DIR}/x86_64-linux-gnu/bin:${PATH}"

# Hexagon toolchain paths
HEXAGON_BIN_DIR="${TOOLCHAIN_DIR}/x86_64-linux-gnu/bin"
HEXAGON_SYSROOT="${TOOLCHAIN_DIR}/x86_64-linux-gnu/target/hexagon-unknown-linux-musl"

# Verify required binaries exist
REQUIRED_BINARIES=(
    "hexagon-unknown-linux-musl-clang"
    "hexagon-unknown-linux-musl-clang++"
    "qemu-hexagon"
)

for binary in "${REQUIRED_BINARIES[@]}"; do
    if [ ! -f "${HEXAGON_BIN_DIR}/${binary}" ]; then
        echo -e "${RED}Error: Required binary ${binary} not found in toolchain${NC}"
        exit 1
    fi
done

# Configure cross-compilation variables for Eigen CI
export EIGEN_CI_CROSS_C_COMPILER="${HEXAGON_BIN_DIR}/hexagon-unknown-linux-musl-clang"
export EIGEN_CI_CROSS_CXX_COMPILER="${HEXAGON_BIN_DIR}/hexagon-unknown-linux-musl-clang++"
export EIGEN_CI_CROSS_INSTALL="wget tar xz-utils"

# Set up sysroot and library paths
export EIGEN_CI_HEXAGON_SYSROOT="${HEXAGON_SYSROOT}"
export QEMU_LD_PREFIX="${HEXAGON_SYSROOT}"

# Configure CMake toolchain path
CMAKE_TOOLCHAIN_FILE="${PWD}/cmake/HexagonToolchain.cmake"
if [ ! -f "${CMAKE_TOOLCHAIN_FILE}" ]; then
    echo -e "${YELLOW}Warning: CMake toolchain file not found at ${CMAKE_TOOLCHAIN_FILE}${NC}"
    echo -e "${YELLOW}This may cause build failures if not using eigen-tools integration${NC}"
fi

# Hexagon-specific compiler flags
HEXAGON_BASE_FLAGS="-m${HEXAGON_ARCH} -G0 -fPIC --target=hexagon-unknown-linux-musl --sysroot=${HEXAGON_SYSROOT}"
HEXAGON_HVX_FLAGS="-mhvx -mhvx-length=${EIGEN_CI_HEXAGON_HVX_LENGTH:-128B}"

# Export flags for use in build scripts
export EIGEN_CI_HEXAGON_BASE_FLAGS="${HEXAGON_BASE_FLAGS}"
export EIGEN_CI_HEXAGON_HVX_FLAGS="${HEXAGON_HVX_FLAGS}"

# Set up ccache if available
if command -v ccache >/dev/null 2>&1; then
    echo -e "${GREEN}Setting up ccache for Hexagon builds${NC}"
    export CCACHE_DIR="${PWD}/.cache/ccache"
    export CCACHE_MAXSIZE="2G"
    export CCACHE_COMPRESS="true"
    mkdir -p ${CCACHE_DIR}
    
    # Create ccache symlinks for Hexagon compilers
    CCACHE_BIN_DIR="${PWD}/.cache/ccache-bin"
    mkdir -p ${CCACHE_BIN_DIR}
    ln -sf $(which ccache) ${CCACHE_BIN_DIR}/hexagon-unknown-linux-musl-clang
    ln -sf $(which ccache) ${CCACHE_BIN_DIR}/hexagon-unknown-linux-musl-clang++
    export PATH="${CCACHE_BIN_DIR}:${PATH}"
fi

# Print configuration summary
echo -e "${GREEN}=== Hexagon Toolchain Configuration ===${NC}"
echo -e "Toolchain Version: ${GREEN}${HEXAGON_TOOLCHAIN_VERSION}${NC}"
echo -e "Hexagon Architecture: ${GREEN}${HEXAGON_ARCH}${NC}"
echo -e "Toolchain Root: ${GREEN}${HEXAGON_TOOLCHAIN_ROOT}${NC}"
echo -e "C Compiler: ${GREEN}${EIGEN_CI_CROSS_C_COMPILER}${NC}"
echo -e "C++ Compiler: ${GREEN}${EIGEN_CI_CROSS_CXX_COMPILER}${NC}"
echo -e "Sysroot: ${GREEN}${HEXAGON_SYSROOT}${NC}"
echo -e "HVX Length: ${GREEN}${EIGEN_CI_HEXAGON_HVX_LENGTH:-128B}${NC}"

# Verify compiler functionality
echo -e "${YELLOW}Verifying Hexagon compiler...${NC}"
if ${EIGEN_CI_CROSS_C_COMPILER} --version >/dev/null 2>&1; then
    echo -e "${GREEN}Hexagon C compiler verification successful${NC}"
else
    echo -e "${RED}Error: Hexagon C compiler verification failed${NC}"
    exit 1
fi

if ${EIGEN_CI_CROSS_CXX_COMPILER} --version >/dev/null 2>&1; then
    echo -e "${GREEN}Hexagon C++ compiler verification successful${NC}"
else
    echo -e "${RED}Error: Hexagon C++ compiler verification failed${NC}"
    exit 1
fi

# Verify QEMU Hexagon emulator
echo -e "${YELLOW}Verifying QEMU Hexagon emulator...${NC}"
QEMU_HEXAGON="${HEXAGON_BIN_DIR}/qemu-hexagon"
if ${QEMU_HEXAGON} -version >/dev/null 2>&1; then
    echo -e "${GREEN}QEMU Hexagon emulator verification successful${NC}"
else
    echo -e "${YELLOW}Warning: QEMU Hexagon emulator verification failed${NC}"
    echo -e "${YELLOW}Tests requiring emulation may fail${NC}"
fi

echo -e "${GREEN}Hexagon toolchain setup completed successfully${NC}" 