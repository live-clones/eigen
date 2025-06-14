#!/bin/bash

# Comprehensive validation script for Hexagon builds and tests
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
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Hexagon Build & Test Validation ===${NC}"

# Configuration
TOOLCHAIN_DIR="/opt/hexagon-toolchain"
BUILD_DIR=${EIGEN_CI_BUILDDIR:-.build}
VALIDATION_RESULTS_DIR="${BUILD_DIR}/validation"
VALIDATION_LOG="${VALIDATION_RESULTS_DIR}/validation.log"

# Create validation results directory
mkdir -p "${VALIDATION_RESULTS_DIR}"

# Initialize validation log
cat > "${VALIDATION_LOG}" << EOF
# Hexagon Validation Report
Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Architecture: ${EIGEN_CI_HEXAGON_ARCH:-v68}
Toolchain Version: ${EIGEN_CI_HEXAGON_TOOLCHAIN_VERSION:-20.1.4}
Build Directory: ${BUILD_DIR}

EOF

# Validation counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
WARNING_CHECKS=0
FAILED_CHECKS=0

# Validation function
validate_check() {
    local check_name="$1"
    local check_command="$2"
    local is_critical="${3:-true}"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    echo -e "${YELLOW}Validating: ${check_name}${NC}"
    echo "VALIDATION: ${check_name}" >> "${VALIDATION_LOG}"
    
    if eval "${check_command}" >> "${VALIDATION_LOG}" 2>&1; then
        echo -e "${GREEN}✓ ${check_name} - PASSED${NC}"
        echo "RESULT: PASSED" >> "${VALIDATION_LOG}"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        if [ "${is_critical}" = "true" ]; then
            echo -e "${RED}✗ ${check_name} - FAILED (Critical)${NC}"
            echo "RESULT: FAILED (Critical)" >> "${VALIDATION_LOG}"
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
            return 1
        else
            echo -e "${YELLOW}⚠ ${check_name} - WARNING${NC}"
            echo "RESULT: WARNING" >> "${VALIDATION_LOG}"
            WARNING_CHECKS=$((WARNING_CHECKS + 1))
            return 0
        fi
    fi
    
    echo "---" >> "${VALIDATION_LOG}"
}

echo -e "${PURPLE}=== Toolchain Validation ===${NC}"

# Validate toolchain installation
validate_check "Toolchain Directory Exists" \
    "[ -d '${TOOLCHAIN_DIR}' ]"

validate_check "Hexagon Compiler Available" \
    "[ -f '${TOOLCHAIN_DIR}/x86_64-linux-gnu/bin/hexagon-unknown-linux-musl-clang++' ]"

validate_check "QEMU Hexagon Emulator Available" \
    "[ -f '${TOOLCHAIN_DIR}/x86_64-linux-gnu/bin/qemu-hexagon' ]"

validate_check "Hexagon Sysroot Available" \
    "[ -d '${TOOLCHAIN_DIR}/x86_64-linux-gnu/target/hexagon-unknown-linux-musl' ]"

# Test compiler functionality
validate_check "Hexagon Compiler Version Check" \
    "${TOOLCHAIN_DIR}/x86_64-linux-gnu/bin/hexagon-unknown-linux-musl-clang++ --version"

validate_check "QEMU Hexagon Version Check" \
    "${TOOLCHAIN_DIR}/x86_64-linux-gnu/bin/qemu-hexagon -version"

echo -e "${PURPLE}=== Build Environment Validation ===${NC}"

# Validate build environment
validate_check "Build Directory Exists" \
    "[ -d '${BUILD_DIR}' ]"

validate_check "CMake Configuration Present" \
    "[ -f '${BUILD_DIR}/CMakeCache.txt' ]"

validate_check "Ninja Build File Present" \
    "[ -f '${BUILD_DIR}/build.ninja' ]"

validate_check "CMake Toolchain Configuration" \
    "grep -q 'CMAKE_TOOLCHAIN_FILE.*HexagonToolchain' '${BUILD_DIR}/CMakeCache.txt'"

validate_check "Hexagon Target Architecture" \
    "grep -q 'hexagon-unknown-linux-musl' '${BUILD_DIR}/CMakeCache.txt'"

echo -e "${PURPLE}=== Binary Validation ===${NC}"

# Find and validate Hexagon binaries
if [ -d "${BUILD_DIR}" ]; then
    # Count built binaries
    HEXAGON_BINARIES=$(find "${BUILD_DIR}" -type f -executable -exec file {} \; 2>/dev/null | grep -c "QUALCOMM DSP6" || echo "0")
    
    validate_check "Hexagon Binaries Built" \
        "[ ${HEXAGON_BINARIES} -gt 0 ]"
    
    echo "Found ${HEXAGON_BINARIES} Hexagon binaries" >> "${VALIDATION_LOG}"
    
    # Test specific binaries if they exist
    TEST_BINARIES=(
        "test/array_cwise"
        "test/basicstuff" 
        "test/linearalgebra"
        "test/matrix"
        "test/packetmath"
    )
    
    for binary in "${TEST_BINARIES[@]}"; do
        if [ -f "${BUILD_DIR}/${binary}" ]; then
            validate_check "Binary: ${binary}" \
                "file '${BUILD_DIR}/${binary}' | grep -q 'QUALCOMM DSP6'" \
                "false"
        fi
    done
fi

echo -e "${PURPLE}=== QEMU Emulation Validation ===${NC}"

# Test QEMU emulation with a simple binary
if [ -f "${BUILD_DIR}/test/basicstuff" ]; then
    QEMU_HEXAGON="${TOOLCHAIN_DIR}/x86_64-linux-gnu/bin/qemu-hexagon"
    SYSROOT="${TOOLCHAIN_DIR}/x86_64-linux-gnu/target/hexagon-unknown-linux-musl"
    
    export QEMU_LD_PREFIX="${SYSROOT}"
    
    validate_check "QEMU Basic Execution Test" \
        "timeout 30 ${QEMU_HEXAGON} ${BUILD_DIR}/test/basicstuff --help" \
        "false"
else
    echo -e "${YELLOW}⚠ No basicstuff binary found for QEMU validation${NC}"
    echo "QEMU validation skipped - no test binary available" >> "${VALIDATION_LOG}"
fi

echo -e "${PURPLE}=== Eigen Library Validation ===${NC}"

# Validate Eigen installation if present
INSTALL_DIR="${BUILD_DIR}/install"
if [ -d "${INSTALL_DIR}" ]; then
    validate_check "Eigen Headers Installed" \
        "[ -d '${INSTALL_DIR}/include/eigen3' ]"
    
    validate_check "Eigen Core Header Available" \
        "[ -f '${INSTALL_DIR}/include/eigen3/Eigen/Core' ]"
    
    validate_check "Eigen Dense Header Available" \
        "[ -f '${INSTALL_DIR}/include/eigen3/Eigen/Dense' ]"
        
    # Count Eigen header files
    EIGEN_HEADERS=$(find "${INSTALL_DIR}/include/eigen3" -name "*.h" -o -name "Core" -o -name "Dense" 2>/dev/null | wc -l)
    echo "Found ${EIGEN_HEADERS} Eigen header files" >> "${VALIDATION_LOG}"
fi

echo -e "${PURPLE}=== Performance & Resource Validation ===${NC}"

# Check build performance metrics
if [ -f "${BUILD_DIR}/build.ninja" ]; then
    BUILD_TARGETS=$(grep -c "^build " "${BUILD_DIR}/build.ninja" 2>/dev/null || echo "0")
    validate_check "Build Targets Generated" \
        "[ ${BUILD_TARGETS} -gt 0 ]" \
        "false"
    
    echo "Generated ${BUILD_TARGETS} build targets" >> "${VALIDATION_LOG}"
fi

# Check disk usage
BUILD_SIZE=$(du -sh "${BUILD_DIR}" 2>/dev/null | cut -f1 || echo "unknown")
echo "Build directory size: ${BUILD_SIZE}" >> "${VALIDATION_LOG}"

echo -e "${PURPLE}=== HVX (Vector Extensions) Validation ===${NC}"

# Check for HVX-specific artifacts
if [ -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    validate_check "HVX Support Enabled" \
        "grep -q 'mhvx' '${BUILD_DIR}/CMakeCache.txt'" \
        "false"
    
    validate_check "HVX Vector Length Configuration" \
        "grep -q 'mhvx-length=128B' '${BUILD_DIR}/CMakeCache.txt'" \
        "false"
fi

echo -e "${PURPLE}=== Test Infrastructure Validation ===${NC}"

# Validate CTest configuration
if [ -f "${BUILD_DIR}/CTestTestfile.cmake" ]; then
    validate_check "CTest Configuration Present" \
        "[ -f '${BUILD_DIR}/CTestTestfile.cmake' ]"
    
    # Count available tests
    TEST_COUNT=$(ctest --show-only --build-config Release 2>/dev/null | grep -c "Test #" || echo "0")
    validate_check "CTest Tests Available" \
        "[ ${TEST_COUNT} -gt 0 ]" \
        "false"
    
    echo "Found ${TEST_COUNT} CTest tests" >> "${VALIDATION_LOG}"
fi

# Generate validation summary
echo -e "${CYAN}=== Validation Summary ===${NC}"
echo "Total Checks: ${TOTAL_CHECKS}"
echo -e "${GREEN}Passed: ${PASSED_CHECKS}${NC}"
echo -e "${YELLOW}Warnings: ${WARNING_CHECKS}${NC}"
echo -e "${RED}Failed: ${FAILED_CHECKS}${NC}"

# Calculate success rate
if [ ${TOTAL_CHECKS} -gt 0 ]; then
    SUCCESS_RATE=$(( (PASSED_CHECKS + WARNING_CHECKS) * 100 / TOTAL_CHECKS ))
    echo "Success Rate: ${SUCCESS_RATE}%"
    
    # Add summary to log
    cat >> "${VALIDATION_LOG}" << EOF

=== VALIDATION SUMMARY ===
Total Checks: ${TOTAL_CHECKS}
Passed: ${PASSED_CHECKS}
Warnings: ${WARNING_CHECKS}
Failed: ${FAILED_CHECKS}
Success Rate: ${SUCCESS_RATE}%

Generated at: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
EOF
    
    # Create JSON summary for CI consumption
    cat > "${VALIDATION_RESULTS_DIR}/validation-summary.json" << EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "architecture": "${EIGEN_CI_HEXAGON_ARCH:-v68}",
  "toolchain_version": "${EIGEN_CI_HEXAGON_TOOLCHAIN_VERSION:-20.1.4}",
  "total_checks": ${TOTAL_CHECKS},
  "passed_checks": ${PASSED_CHECKS},
  "warning_checks": ${WARNING_CHECKS},
  "failed_checks": ${FAILED_CHECKS},
  "success_rate": ${SUCCESS_RATE},
  "build_directory": "${BUILD_DIR}",
  "validation_log": "${VALIDATION_LOG}"
}
EOF
    
    echo -e "${CYAN}Validation results saved to: ${VALIDATION_RESULTS_DIR}${NC}"
    echo -e "${CYAN}Detailed log: ${VALIDATION_LOG}${NC}"
    echo -e "${CYAN}JSON summary: ${VALIDATION_RESULTS_DIR}/validation-summary.json${NC}"
    
    # Set exit code based on critical failures
    if [ ${FAILED_CHECKS} -gt 0 ]; then
        echo -e "${RED}❌ Validation failed due to critical errors${NC}"
        exit 1
    elif [ ${WARNING_CHECKS} -gt 0 ]; then
        echo -e "${YELLOW}⚠️ Validation completed with warnings${NC}"
        exit 0
    else
        echo -e "${GREEN}✅ All validations passed successfully${NC}"
        exit 0
    fi
else
    echo -e "${RED}❌ No validation checks were performed${NC}"
    exit 1
fi 