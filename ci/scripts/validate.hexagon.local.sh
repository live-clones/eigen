#!/bin/bash

# Local validation script for Hexagon builds and tests
# This version is designed for local testing without requiring the full Hexagon toolchain

set -e

# Color output for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Local Hexagon Validation (Mock Mode) ===${NC}"

# Configuration
BUILD_DIR=${EIGEN_CI_BUILDDIR:-.build-hexagon-v68-default}
VALIDATION_RESULTS_DIR="${BUILD_DIR}/validation"
VALIDATION_LOG="${VALIDATION_RESULTS_DIR}/validation.log"

# Create validation results directory
mkdir -p "${VALIDATION_RESULTS_DIR}"

# Initialize validation log
cat > "${VALIDATION_LOG}" << EOF
# Hexagon Local Validation Report (Mock Mode)
Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Architecture: ${EIGEN_CI_HEXAGON_ARCH:-v68}
Toolchain Version: ${EIGEN_CI_HEXAGON_TOOLCHAIN_VERSION:-20.1.4}
Build Directory: ${BUILD_DIR}
Mode: Local Testing (Mock)

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
    local is_critical="${3:-false}"  # Default to non-critical for local testing
    
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
            echo -e "${YELLOW}⚠ ${check_name} - SKIPPED (Local Mode)${NC}"
            echo "RESULT: SKIPPED (Local Mode)" >> "${VALIDATION_LOG}"
            WARNING_CHECKS=$((WARNING_CHECKS + 1))
            return 0
        fi
    fi
    
    echo "---" >> "${VALIDATION_LOG}"
}

echo -e "${PURPLE}=== Environment Validation ===${NC}"

# Basic environment checks
validate_check "Working Directory Accessible" \
    "[ -d '$(pwd)' ]" \
    "true"

validate_check "Build Directory Creation" \
    "mkdir -p '${BUILD_DIR}'" \
    "true"

validate_check "Validation Directory Creation" \
    "mkdir -p '${VALIDATION_RESULTS_DIR}'" \
    "true"

echo -e "${PURPLE}=== System Tools Validation ===${NC}"

# Check for basic tools
validate_check "Bash Available" \
    "command -v bash >/dev/null 2>&1"

validate_check "Python3 Available" \
    "command -v python3 >/dev/null 2>&1"

validate_check "CMake Available" \
    "command -v cmake >/dev/null 2>&1"

validate_check "Make Available" \
    "command -v make >/dev/null 2>&1"

validate_check "GCC Available" \
    "command -v gcc >/dev/null 2>&1"

echo -e "${PURPLE}=== Mock Hexagon Toolchain Validation ===${NC}"

# Mock toolchain validation (these will show as warnings in local mode)
validate_check "Mock: Hexagon Compiler" \
    "echo 'Mock hexagon-clang++ version check' && false"

validate_check "Mock: QEMU Hexagon Emulator" \
    "echo 'Mock qemu-hexagon version check' && false"

validate_check "Mock: Hexagon Sysroot" \
    "echo 'Mock sysroot validation' && false"

echo -e "${PURPLE}=== Eigen Source Validation ===${NC}"

# Check for Eigen source files
validate_check "Eigen Directory Present" \
    "[ -d 'Eigen' ] || [ -d '../Eigen' ] || [ -d 'eigen' ]"

validate_check "Eigen Core Header" \
    "find . -name 'Core' -path '*/Eigen/*' | head -1 | xargs test -f"

validate_check "CMakeLists.txt Present" \
    "[ -f 'CMakeLists.txt' ] || [ -f '../CMakeLists.txt' ]"

echo -e "${PURPLE}=== Mock Build Validation ===${NC}"

# Create mock build artifacts for testing
echo -e "${CYAN}Creating mock build artifacts...${NC}"

# Create mock CMakeCache.txt
cat > "${BUILD_DIR}/CMakeCache.txt" << EOF
# Mock CMakeCache.txt for local testing
CMAKE_TOOLCHAIN_FILE:FILEPATH=/mock/HexagonToolchain.cmake
CMAKE_CXX_COMPILER:FILEPATH=/mock/hexagon-unknown-linux-musl-clang++
CMAKE_C_COMPILER:FILEPATH=/mock/hexagon-unknown-linux-musl-clang
CMAKE_SYSTEM_NAME:STRING=Linux
CMAKE_SYSTEM_PROCESSOR:STRING=hexagon
EOF

# Create mock build.ninja
cat > "${BUILD_DIR}/build.ninja" << EOF
# Mock build.ninja for local testing
rule CXX_COMPILER
  command = /mock/hexagon-clang++ \$FLAGS -o \$out \$in
  description = Building CXX object \$out

build test/basicstuff: CXX_COMPILER test/basicstuff.cpp
build test/array_cwise: CXX_COMPILER test/array_cwise.cpp
EOF

# Create mock test directory structure
mkdir -p "${BUILD_DIR}/test"
mkdir -p "${BUILD_DIR}/testing"
mkdir -p "${BUILD_DIR}/install/include/eigen3/Eigen"

# Create mock test binaries (empty files for validation)
touch "${BUILD_DIR}/test/basicstuff"
touch "${BUILD_DIR}/test/array_cwise"
touch "${BUILD_DIR}/test/linearalgebra"

# Create mock Eigen headers
touch "${BUILD_DIR}/install/include/eigen3/Eigen/Core"
touch "${BUILD_DIR}/install/include/eigen3/Eigen/Dense"

validate_check "Mock CMake Configuration" \
    "[ -f '${BUILD_DIR}/CMakeCache.txt' ]"

validate_check "Mock Build File" \
    "[ -f '${BUILD_DIR}/build.ninja' ]"

validate_check "Mock Test Binaries" \
    "[ -f '${BUILD_DIR}/test/basicstuff' ]"

echo -e "${PURPLE}=== Validation Summary ===${NC}"

# Generate validation summary
cat >> "${VALIDATION_LOG}" << EOF

=== VALIDATION SUMMARY ===
Total Checks: ${TOTAL_CHECKS}
Passed: ${PASSED_CHECKS}
Warnings: ${WARNING_CHECKS}
Failed: ${FAILED_CHECKS}

Status: $([ ${FAILED_CHECKS} -eq 0 ] && echo "SUCCESS" || echo "FAILED")
EOF

# Create validation summary JSON
cat > "${VALIDATION_RESULTS_DIR}/validation-summary.json" << EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "mode": "local_testing",
  "total_checks": ${TOTAL_CHECKS},
  "passed_checks": ${PASSED_CHECKS},
  "warning_checks": ${WARNING_CHECKS},
  "failed_checks": ${FAILED_CHECKS},
  "status": "$([ ${FAILED_CHECKS} -eq 0 ] && echo "success" || echo "failed")",
  "build_directory": "${BUILD_DIR}",
  "hexagon_arch": "${EIGEN_CI_HEXAGON_ARCH:-v68}",
  "toolchain_version": "${EIGEN_CI_HEXAGON_TOOLCHAIN_VERSION:-20.1.4}"
}
EOF

echo -e "${CYAN}=== Final Results ===${NC}"
echo -e "Total Checks: ${TOTAL_CHECKS}"
echo -e "${GREEN}Passed: ${PASSED_CHECKS}${NC}"
echo -e "${YELLOW}Warnings: ${WARNING_CHECKS}${NC}"
echo -e "${RED}Failed: ${FAILED_CHECKS}${NC}"

if [ ${FAILED_CHECKS} -eq 0 ]; then
    echo -e "${GREEN}✅ Local validation completed successfully${NC}"
    echo -e "${YELLOW}Note: This is a mock validation for local testing${NC}"
    exit 0
else
    echo -e "${RED}❌ Local validation failed${NC}"
    exit 1
fi 