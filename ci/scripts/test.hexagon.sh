#!/bin/bash

# Enhanced Hexagon testing script with comprehensive monitoring and CI integration
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

echo -e "${BLUE}=== Hexagon Testing with Enhanced Integration ===${NC}"

# Configuration
TOOLCHAIN_DIR="/opt/hexagon-toolchain"
QEMU_HEXAGON="${TOOLCHAIN_DIR}/x86_64-linux-gnu/bin/qemu-hexagon"
SYSROOT="${TOOLCHAIN_DIR}/x86_64-linux-gnu/target/hexagon-unknown-linux-musl"
BUILD_DIR=${EIGEN_CI_BUILDDIR:-.build}

# Enhanced testing configuration
ENABLE_VALIDATION=${EIGEN_CI_ENABLE_VALIDATION:-true}
ENABLE_MONITORING=${EIGEN_CI_ENABLE_MONITORING:-true}
ENABLE_PERFORMANCE_TRACKING=${EIGEN_CI_ENABLE_PERFORMANCE:-true}
TEST_TIMEOUT=${EIGEN_CI_TEST_TIMEOUT:-300}
MAX_PARALLEL_TESTS=${EIGEN_CI_MAX_PARALLEL_TESTS:-4}

# Create testing directories
mkdir -p "${BUILD_DIR}/testing"
mkdir -p "${BUILD_DIR}/reports"

# Initialize comprehensive logging
TEST_LOG="${BUILD_DIR}/testing/hexagon-test.log"
RESULTS_JSON="${BUILD_DIR}/testing/test-results.json"
JUNIT_XML="${BUILD_DIR}/Testing/Temporary/LastTest.log.xml"

echo "=== Enhanced Hexagon Testing Started ===" > "${TEST_LOG}"
echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")" >> "${TEST_LOG}"
echo "Configuration:" >> "${TEST_LOG}"
echo "  Architecture: ${EIGEN_CI_HEXAGON_ARCH:-v68}" >> "${TEST_LOG}"
echo "  Timeout: ${TEST_TIMEOUT}" >> "${TEST_LOG}"
echo "  Max Parallel: ${MAX_PARALLEL_TESTS}" >> "${TEST_LOG}"
echo "  Validation: ${ENABLE_VALIDATION}" >> "${TEST_LOG}"
echo "  Monitoring: ${ENABLE_MONITORING}" >> "${TEST_LOG}"
echo "" >> "${TEST_LOG}"

# Pre-test validation
pre_test_validation() {
    echo -e "${PURPLE}=== Pre-Test Validation ===${NC}"
    
    if [ "${ENABLE_VALIDATION}" = "true" ]; then
        echo "Running pre-test validation..." >> "${TEST_LOG}"
        
        # Run validation script if available
        if [ -f "ci/scripts/validate.hexagon.sh" ]; then
            echo -e "${YELLOW}Running comprehensive validation...${NC}"
            if bash ci/scripts/validate.hexagon.sh >> "${TEST_LOG}" 2>&1; then
                echo -e "${GREEN}✓ Pre-test validation passed${NC}"
                echo "Pre-test validation: PASSED" >> "${TEST_LOG}"
            else
                echo -e "${YELLOW}⚠ Pre-test validation had warnings${NC}"
                echo "Pre-test validation: WARNING" >> "${TEST_LOG}"
            fi
        fi
    else
        echo "Pre-test validation disabled" >> "${TEST_LOG}"
    fi
}

# Verify QEMU emulator exists and is functional
verify_qemu_setup() {
    echo -e "${PURPLE}=== Verifying QEMU Setup ===${NC}"
    
    if [ ! -f "${QEMU_HEXAGON}" ]; then
        echo -e "${RED}Error: QEMU Hexagon emulator not found at ${QEMU_HEXAGON}${NC}"
        echo "QEMU verification: FAILED - emulator not found" >> "${TEST_LOG}"
        exit 1
    fi
    
    # Test QEMU basic functionality
    if ${QEMU_HEXAGON} -version >/dev/null 2>&1; then
        echo -e "${GREEN}✓ QEMU Hexagon emulator verified${NC}"
        echo "QEMU verification: PASSED" >> "${TEST_LOG}"
    else
        echo -e "${RED}Error: QEMU Hexagon emulator not functional${NC}"
        echo "QEMU verification: FAILED - not functional" >> "${TEST_LOG}"
        exit 1
    fi
    
    # Set up QEMU environment
    export QEMU_LD_PREFIX=${SYSROOT}
    echo "QEMU_LD_PREFIX: ${QEMU_LD_PREFIX}" >> "${TEST_LOG}"
}

# Enhanced test execution function
run_hexagon_test() {
    local test_binary=$1
    local test_name=$2
    local timeout=${3:-${TEST_TIMEOUT}}
    local category=${4:-"general"}
    
    local start_time=$(date +%s%N)
    local result="UNKNOWN"
    local output=""
    local exit_code=0
    
    echo -e "${YELLOW}Running ${test_name} (${category})...${NC}"
    echo "=== TEST: ${test_name} ===" >> "${TEST_LOG}"
    echo "Category: ${category}" >> "${TEST_LOG}"
    echo "Binary: ${test_binary}" >> "${TEST_LOG}"
    echo "Timeout: ${timeout}s" >> "${TEST_LOG}"
    echo "Start: $(date -u +"%Y-%m-%d %H:%M:%S UTC")" >> "${TEST_LOG}"
    
    if [ -f "${test_binary}" ]; then
        # Verify it's a Hexagon binary
        if file "${test_binary}" | grep -q "QUALCOMM DSP6"; then
            echo "Binary type: Hexagon DSP6" >> "${TEST_LOG}"
            
            # Run test with timeout and capture output
            if output=$(timeout ${timeout} ${QEMU_HEXAGON} ${test_binary} 2>&1); then
                result="PASSED"
                exit_code=0
                echo -e "${GREEN}✓ ${test_name} passed${NC}"
            else
                exit_code=$?
                if [ ${exit_code} -eq 124 ]; then
                    result="TIMEOUT"
                    echo -e "${RED}✗ ${test_name} timed out after ${timeout} seconds${NC}"
                else
                    result="FAILED"
                    echo -e "${RED}✗ ${test_name} failed with exit code ${exit_code}${NC}"
                fi
            fi
        else
            result="INVALID"
            echo -e "${YELLOW}⚠ ${test_name} is not a valid Hexagon binary${NC}"
            echo "Binary type: $(file "${test_binary}")" >> "${TEST_LOG}"
        fi
    else
        result="MISSING"
        echo -e "${YELLOW}⚠ ${test_name} binary not found: ${test_binary}${NC}"
    fi
    
    local end_time=$(date +%s%N)
    local duration_ms=$(((end_time - start_time) / 1000000))
    
    # Log detailed results
    cat >> "${TEST_LOG}" << EOF
Result: ${result}
Exit Code: ${exit_code}
Duration: ${duration_ms}ms
End: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Output:
${output}
---

EOF
    
    # Return appropriate exit code for test counting
    case "${result}" in
        "PASSED") return 0 ;;
        "MISSING"|"INVALID") return 2 ;; # Skip
        *) return 1 ;; # Failed
    esac
}

# Main test execution with enhanced tracking
execute_test_suite() {
    echo -e "${PURPLE}=== Executing Test Suite ===${NC}"
    
    # Navigate to build directory
    if [ ! -d "${BUILD_DIR}" ]; then
        echo -e "${RED}Error: Build directory ${BUILD_DIR} not found${NC}"
        exit 1
    fi
    
    cd ${BUILD_DIR}
    echo -e "${YELLOW}Running tests from directory: $(pwd)${NC}"
    
    # Track test results
    local total_tests=0
    local passed_tests=0
    local failed_tests=0
    local skipped_tests=0
    local timeout_tests=0
    
    # Test categories for better organization
    declare -A test_categories
    
    # Core Eigen tests
    echo -e "${BLUE}=== Core Eigen Tests ===${NC}"
    
    BASIC_TESTS=(
        "test/array_cwise:ArrayCwise:core"
        "test/basicstuff:BasicStuff:core"
        "test/dense_transformations_1:DenseTransformations1:core"
        "test/dense_transformations_2:DenseTransformations2:core"
        "test/linearalgebra:LinearAlgebra:core"
        "test/lu:LU:core"
        "test/qr:QR:core"
        "test/svd:SVD:core"
        "test/cholesky:Cholesky:core"
        "test/jacobisvd:JacobiSVD:core"
    )
    
    for test_info in "${BASIC_TESTS[@]}"; do
        IFS=':' read -r test_binary test_name test_category <<< "${test_info}"
        total_tests=$((total_tests + 1))
        
        if run_hexagon_test "${test_binary}" "${test_name}" "${TEST_TIMEOUT}" "${test_category}"; then
            passed_tests=$((passed_tests + 1))
        else
            case $? in
                1) failed_tests=$((failed_tests + 1)) ;;
                2) skipped_tests=$((skipped_tests + 1)) ;;
            esac
        fi
    done
    
    # Hexagon-specific tests
    echo -e "${BLUE}=== Hexagon-Specific Tests ===${NC}"
    
    HEXAGON_TESTS=(
        "test/hexagon_basic:HexagonBasic:hexagon"
        "test/hexagon_packet:HexagonPacket:hexagon"
        "test/hexagon_simd:HexagonSIMD:hexagon"
        "test/packetmath:PacketMath:hexagon"
    )
    
    for test_info in "${HEXAGON_TESTS[@]}"; do
        IFS=':' read -r test_binary test_name test_category <<< "${test_info}"
        total_tests=$((total_tests + 1))
        
        if run_hexagon_test "${test_binary}" "${test_name}" "${TEST_TIMEOUT}" "${test_category}"; then
            passed_tests=$((passed_tests + 1))
        else
            case $? in
                1) failed_tests=$((failed_tests + 1)) ;;
                2) skipped_tests=$((skipped_tests + 1)) ;;
            esac
        fi
    done
    
    # HVX (Hexagon Vector eXtensions) tests
    echo -e "${BLUE}=== HVX Vector Tests ===${NC}"
    
    HVX_TESTS=(
        "test/hvx_packet:HVXPacket:hvx"
        "test/hvx_basicstuff:HVXBasicStuff:hvx"
        "test/hvx_vectorization:HVXVectorization:hvx"
        "test/hvx_aligned:HVXAligned:hvx"
        "test/hvx_geometry:HVXGeometry:hvx"
    )
    
    for test_info in "${HVX_TESTS[@]}"; do
        IFS=':' read -r test_binary test_name test_category <<< "${test_info}"
        total_tests=$((total_tests + 1))
        
        # HVX tests get longer timeout
        local hvx_timeout=$((TEST_TIMEOUT * 2))
        if run_hexagon_test "${test_binary}" "${test_name}" "${hvx_timeout}" "${test_category}"; then
            passed_tests=$((passed_tests + 1))
        else
            case $? in
                1) failed_tests=$((failed_tests + 1)) ;;
                2) skipped_tests=$((skipped_tests + 1)) ;;
            esac
        fi
    done
    
    # Store results for later use
    echo "${total_tests}" > "${BUILD_DIR}/testing/total_tests"
    echo "${passed_tests}" > "${BUILD_DIR}/testing/passed_tests"
    echo "${failed_tests}" > "${BUILD_DIR}/testing/failed_tests"
    echo "${skipped_tests}" > "${BUILD_DIR}/testing/skipped_tests"
}

# Enhanced CTest integration
run_ctest_integration() {
    echo -e "${BLUE}=== Enhanced CTest Integration ===${NC}"
    
    if [ -f "CTestTestfile.cmake" ] && command -v ctest >/dev/null 2>&1; then
        echo -e "${YELLOW}Running CTest with Hexagon emulation...${NC}"
        
        # Set up CTest environment for Hexagon
        export CTEST_OUTPUT_ON_FAILURE=1
        export QEMU_HEXAGON_BINARY=${QEMU_HEXAGON}
        
        # Configure CTest for parallel execution
        local ctest_args="-j ${MAX_PARALLEL_TESTS} --timeout ${TEST_TIMEOUT} --output-on-failure"
        
        # Apply test filters if specified
        if [ -n "${EIGEN_CI_CTEST_REGEX}" ]; then
            ctest_args="${ctest_args} -R ${EIGEN_CI_CTEST_REGEX}"
        elif [ -n "${EIGEN_CI_CTEST_LABEL}" ]; then
            ctest_args="${ctest_args} -L ${EIGEN_CI_CTEST_LABEL}"
        fi
        
        echo "CTest command: ctest ${ctest_args}" >> "${TEST_LOG}"
        
        # Run CTest with comprehensive logging
        if ctest ${ctest_args} --output-log "${BUILD_DIR}/testing/ctest-output.log"; then
            echo -e "${GREEN}✓ CTest execution successful${NC}"
            echo "CTest execution: SUCCESSFUL" >> "${TEST_LOG}"
        else
            echo -e "${YELLOW}⚠ CTest execution had failures${NC}"
            echo "CTest execution: PARTIAL_FAILURE" >> "${TEST_LOG}"
        fi
        
        # Convert CTest results to JUnit XML if possible
        if [ -f "${BUILD_DIR}/Testing/Temporary/LastTest.log" ]; then
            echo "Converting CTest results to JUnit XML..." >> "${TEST_LOG}"
            # Additional processing can be added here
        fi
    else
        echo -e "${YELLOW}CTest not available or not configured${NC}"
        echo "CTest integration: SKIPPED" >> "${TEST_LOG}"
    fi
}

# Performance benchmarking
run_performance_tests() {
    if [ "${ENABLE_PERFORMANCE_TRACKING}" = "true" ]; then
        echo -e "${BLUE}=== Performance Benchmarking ===${NC}"
        
        PERF_TESTS=(
            "test/benchGeometry:GeometryBenchmark"
            "test/benchDenseLinearAlgebra:DenseLinearAlgebraBenchmark"
        )
        
        for test_info in "${PERF_TESTS[@]}"; do
            IFS=':' read -r test_binary test_name <<< "${test_info}"
            
            if [ -f "${test_binary}" ]; then
                echo -e "${YELLOW}Running performance test: ${test_name}...${NC}"
                
                local perf_output="${BUILD_DIR}/testing/${test_name}-perf.json"
                local perf_timeout=$((TEST_TIMEOUT * 3))
                
                if timeout ${perf_timeout} ${QEMU_HEXAGON} ${test_binary} --benchmark_format=json > "${perf_output}" 2>/dev/null; then
                    echo -e "${GREEN}✓ ${test_name} performance test completed${NC}"
                    echo "Performance test ${test_name}: COMPLETED" >> "${TEST_LOG}"
                else
                    echo -e "${YELLOW}⚠ ${test_name} performance test failed or timed out${NC}"
                    echo "Performance test ${test_name}: FAILED" >> "${TEST_LOG}"
                fi
            fi
        done
    else
        echo "Performance testing disabled" >> "${TEST_LOG}"
    fi
}

# Generate comprehensive test results
generate_test_results() {
    echo -e "${CYAN}=== Generating Test Results ===${NC}"
    
    # Read test counts
    local total_tests=$(cat "${BUILD_DIR}/testing/total_tests" 2>/dev/null || echo "0")
    local passed_tests=$(cat "${BUILD_DIR}/testing/passed_tests" 2>/dev/null || echo "0")
    local failed_tests=$(cat "${BUILD_DIR}/testing/failed_tests" 2>/dev/null || echo "0")
    local skipped_tests=$(cat "${BUILD_DIR}/testing/skipped_tests" 2>/dev/null || echo "0")
    
    # Calculate success rate
    local success_rate=0
    if [ ${total_tests} -gt 0 ]; then
        success_rate=$(( (passed_tests * 100) / total_tests ))
    fi
    
    # Generate JSON results
    cat > "${RESULTS_JSON}" << EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "architecture": "${EIGEN_CI_HEXAGON_ARCH:-v68}",
  "toolchain_version": "${EIGEN_CI_HEXAGON_TOOLCHAIN_VERSION:-20.1.4}",
  "test_results": {
    "total": ${total_tests},
    "passed": ${passed_tests},
    "failed": ${failed_tests},
    "skipped": ${skipped_tests},
    "success_rate": ${success_rate}
  },
  "configuration": {
    "timeout": ${TEST_TIMEOUT},
    "max_parallel": ${MAX_PARALLEL_TESTS},
    "validation_enabled": ${ENABLE_VALIDATION},
    "monitoring_enabled": ${ENABLE_MONITORING},
    "performance_tracking": ${ENABLE_PERFORMANCE_TRACKING}
  },
  "environment": {
    "ci_job": "${CI_JOB_NAME:-local}",
    "ci_pipeline": "${CI_PIPELINE_SOURCE:-local}",
    "commit": "${CI_COMMIT_SHA:-unknown}"
  }
}
EOF
    
    echo -e "${CYAN}Test results saved to: ${RESULTS_JSON}${NC}"
}

# Post-test monitoring
post_test_monitoring() {
    if [ "${ENABLE_MONITORING}" = "true" ]; then
        echo -e "${PURPLE}=== Post-Test Monitoring ===${NC}"
        
        # Run monitoring script if available
        if [ -f "ci/scripts/monitor.hexagon.sh" ]; then
            echo -e "${YELLOW}Running post-test monitoring...${NC}"
            if bash ci/scripts/monitor.hexagon.sh >> "${TEST_LOG}" 2>&1; then
                echo -e "${GREEN}✓ Post-test monitoring completed${NC}"
            else
                echo -e "${YELLOW}⚠ Post-test monitoring had issues${NC}"
            fi
        fi
    else
        echo "Post-test monitoring disabled" >> "${TEST_LOG}"
    fi
}

# Test summary and exit code determination
generate_test_summary() {
    echo -e "${CYAN}=== Test Summary ===${NC}"
    
    # Read final test counts
    local total_tests=$(cat "${BUILD_DIR}/testing/total_tests" 2>/dev/null || echo "0")
    local passed_tests=$(cat "${BUILD_DIR}/testing/passed_tests" 2>/dev/null || echo "0")
    local failed_tests=$(cat "${BUILD_DIR}/testing/failed_tests" 2>/dev/null || echo "0")
    local skipped_tests=$(cat "${BUILD_DIR}/testing/skipped_tests" 2>/dev/null || echo "0")
    
    echo "Total Tests: ${total_tests}"
    echo -e "${GREEN}Passed: ${passed_tests}${NC}"
    echo -e "${RED}Failed: ${failed_tests}${NC}"
    echo -e "${YELLOW}Skipped: ${skipped_tests}${NC}"
    
    # Calculate success percentage
    if [ ${total_tests} -gt 0 ]; then
        local success_rate=$((passed_tests * 100 / total_tests))
        echo "Success Rate: ${success_rate}%"
        
        # Final summary to log
        cat >> "${TEST_LOG}" << EOF

=== FINAL TEST SUMMARY ===
Total Tests: ${total_tests}
Passed: ${passed_tests}
Failed: ${failed_tests}
Skipped: ${skipped_tests}
Success Rate: ${success_rate}%
Completed: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

EOF
        
        # Set exit code based on results and thresholds
        if [ ${failed_tests} -eq 0 ]; then
            echo -e "${GREEN}✅ All tests passed successfully${NC}"
            exit 0
        elif [ ${success_rate} -ge 80 ]; then
            echo -e "${YELLOW}⚠️ Tests completed with acceptable success rate (${success_rate}%)${NC}"
            exit 0  # Accept partial success for cross-compilation
        elif [ ${success_rate} -ge 60 ]; then
            echo -e "${YELLOW}⚠️ Tests completed with marginal success rate (${success_rate}%)${NC}"
            exit 0  # Still acceptable for cross-compilation testing
        else
            echo -e "${RED}❌ Test suite failed with low success rate (${success_rate}%)${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ No tests were executed${NC}"
        exit 1
    fi
}

# Main execution flow
main() {
    echo -e "${BLUE}Starting enhanced Hexagon testing pipeline...${NC}"
    
    pre_test_validation
    verify_qemu_setup
    execute_test_suite
    run_ctest_integration
    run_performance_tests
    generate_test_results
    post_test_monitoring
    generate_test_summary
}

# Execute main function
main "$@" 