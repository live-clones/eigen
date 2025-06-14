#!/bin/bash

# Hexagon CI Test Monitoring and Reporting Script
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

echo -e "${BLUE}=== Hexagon CI Test Monitoring ===${NC}"

# Configuration
BUILD_DIR=${EIGEN_CI_BUILDDIR:-.build}
MONITOR_DIR="${BUILD_DIR}/monitoring"
MONITOR_LOG="${MONITOR_DIR}/monitor.log"
TEST_RESULTS_JSON="${MONITOR_DIR}/test-results.json"
PERFORMANCE_JSON="${MONITOR_DIR}/performance.json"

# Create monitoring directory
mkdir -p "${MONITOR_DIR}"

# Initialize monitoring
echo "=== Hexagon CI Test Monitoring Started ===" > "${MONITOR_LOG}"
echo "Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")" >> "${MONITOR_LOG}"
echo "Architecture: ${EIGEN_CI_HEXAGON_ARCH:-v68}" >> "${MONITOR_LOG}"
echo "Build Directory: ${BUILD_DIR}" >> "${MONITOR_LOG}"
echo "" >> "${MONITOR_LOG}"

# System information collection
collect_system_info() {
    echo -e "${PURPLE}=== Collecting System Information ===${NC}"
    
    cat >> "${MONITOR_LOG}" << EOF
=== SYSTEM INFORMATION ===
Hostname: $(hostname)
Architecture: $(uname -m)
Kernel: $(uname -r)
CPU Count: $(nproc)
Memory: $(free -h | grep '^Mem:' | awk '{print $2}')
Disk Space: $(df -h . | tail -1 | awk '{print $4}' | sed 's/G/ GB/')

EOF
}

# Build monitoring
monitor_build_status() {
    echo -e "${PURPLE}=== Monitoring Build Status ===${NC}"
    
    if [ -f "${BUILD_DIR}/build.ninja" ]; then
        BUILD_TARGETS=$(grep -c "^build " "${BUILD_DIR}/build.ninja" 2>/dev/null || echo "0")
        echo "Build targets: ${BUILD_TARGETS}" >> "${MONITOR_LOG}"
        
        # Check for any build failures in recent logs
        if [ -f "${BUILD_DIR}/.ninja_log" ]; then
            RECENT_BUILDS=$(tail -20 "${BUILD_DIR}/.ninja_log" | wc -l)
            echo "Recent build operations: ${RECENT_BUILDS}" >> "${MONITOR_LOG}"
        fi
    else
        echo "No build.ninja found - build may not be configured" >> "${MONITOR_LOG}"
    fi
}

# Test execution monitoring
monitor_test_execution() {
    echo -e "${PURPLE}=== Monitoring Test Execution ===${NC}"
    
    local test_start_time=$(date +%s)
    local tests_run=0
    local tests_passed=0
    local tests_failed=0
    local tests_timeout=0
    
    echo "=== TEST EXECUTION MONITORING ===" >> "${MONITOR_LOG}"
    echo "Start time: $(date -u +"%Y-%m-%d %H:%M:%S UTC")" >> "${MONITOR_LOG}"
    
    # Monitor CTest if available
    if [ -f "${BUILD_DIR}/CTestTestfile.cmake" ] && command -v ctest >/dev/null 2>&1; then
        echo -e "${YELLOW}Monitoring CTest execution...${NC}"
        
        # Create CTest monitoring log
        CTEST_LOG="${MONITOR_DIR}/ctest-monitor.log"
        
        # Run CTest with timeout and monitoring
        cd "${BUILD_DIR}"
        
        if timeout 1800 ctest --output-on-failure --timeout 300 -j 4 > "${CTEST_LOG}" 2>&1; then
            echo -e "${GREEN}CTest execution completed successfully${NC}"
            
            # Parse CTest results
            if grep -q "tests passed" "${CTEST_LOG}"; then
                tests_passed=$(grep "tests passed" "${CTEST_LOG}" | head -1 | grep -o '[0-9]*' | head -1 || echo "0")
            fi
            
            if grep -q "tests failed" "${CTEST_LOG}"; then
                tests_failed=$(grep "tests failed" "${CTEST_LOG}" | head -1 | grep -o '[0-9]*' | head -1 || echo "0")
            fi
            
            tests_run=$((tests_passed + tests_failed))
            
        else
            echo -e "${RED}CTest execution failed or timed out${NC}"
            tests_timeout=1
        fi
    else
        echo -e "${YELLOW}CTest not available for monitoring${NC}"
    fi
    
    local test_end_time=$(date +%s)
    local test_duration=$((test_end_time - test_start_time))
    
    # Log test results
    cat >> "${MONITOR_LOG}" << EOF
=== TEST RESULTS ===
Tests Run: ${tests_run}
Tests Passed: ${tests_passed}
Tests Failed: ${tests_failed}
Tests Timeout: ${tests_timeout}
Duration: ${test_duration} seconds
End time: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

EOF
    
    # Create JSON test results
    cat > "${TEST_RESULTS_JSON}" << EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "architecture": "${EIGEN_CI_HEXAGON_ARCH:-v68}",
  "tests": {
    "total": ${tests_run},
    "passed": ${tests_passed},
    "failed": ${tests_failed},
    "timeout": ${tests_timeout}
  },
  "duration_seconds": ${test_duration},
  "success_rate": $([ ${tests_run} -gt 0 ] && echo "scale=2; ${tests_passed} * 100 / ${tests_run}" | bc || echo "0")
}
EOF
}

# QEMU performance monitoring
monitor_qemu_performance() {
    echo -e "${PURPLE}=== Monitoring QEMU Performance ===${NC}"
    
    local qemu_tests=0
    local qemu_successful=0
    local total_execution_time=0
    
    echo "=== QEMU PERFORMANCE MONITORING ===" >> "${MONITOR_LOG}"
    
    # Test QEMU with various binaries
    TEST_BINARIES=(
        "test/basicstuff"
        "test/array_cwise"
        "test/linearalgebra"
    )
    
    QEMU_HEXAGON="/opt/hexagon-toolchain/x86_64-linux-gnu/bin/qemu-hexagon"
    SYSROOT="/opt/hexagon-toolchain/x86_64-linux-gnu/target/hexagon-unknown-linux-musl"
    export QEMU_LD_PREFIX="${SYSROOT}"
    
    for binary in "${TEST_BINARIES[@]}"; do
        if [ -f "${BUILD_DIR}/${binary}" ]; then
            echo -e "${YELLOW}Testing QEMU with ${binary}...${NC}"
            qemu_tests=$((qemu_tests + 1))
            
            local start_time=$(date +%s%N)
            
            if timeout 60 ${QEMU_HEXAGON} "${BUILD_DIR}/${binary}" --help >/dev/null 2>&1; then
                local end_time=$(date +%s%N)
                local execution_time=$(((end_time - start_time) / 1000000)) # Convert to milliseconds
                
                qemu_successful=$((qemu_successful + 1))
                total_execution_time=$((total_execution_time + execution_time))
                
                echo "QEMU test ${binary}: SUCCESS (${execution_time}ms)" >> "${MONITOR_LOG}"
            else
                echo "QEMU test ${binary}: FAILED" >> "${MONITOR_LOG}"
            fi
        fi
    done
    
    # Calculate average execution time
    local avg_execution_time=0
    if [ ${qemu_successful} -gt 0 ]; then
        avg_execution_time=$((total_execution_time / qemu_successful))
    fi
    
    cat >> "${MONITOR_LOG}" << EOF
=== QEMU PERFORMANCE RESULTS ===
Total QEMU tests: ${qemu_tests}
Successful: ${qemu_successful}
Failed: $((qemu_tests - qemu_successful))
Average execution time: ${avg_execution_time}ms
Total execution time: ${total_execution_time}ms

EOF
    
    # Create performance JSON
    cat > "${PERFORMANCE_JSON}" << EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "qemu_performance": {
    "total_tests": ${qemu_tests},
    "successful": ${qemu_successful},
    "failed": $((qemu_tests - qemu_successful)),
    "average_execution_time_ms": ${avg_execution_time},
    "total_execution_time_ms": ${total_execution_time}
  }
}
EOF
}

# Resource usage monitoring
monitor_resource_usage() {
    echo -e "${PURPLE}=== Monitoring Resource Usage ===${NC}"
    
    cat >> "${MONITOR_LOG}" << EOF
=== RESOURCE USAGE ===
Current Memory Usage:
$(free -h)

Current Disk Usage:
$(df -h "${BUILD_DIR}")

Build Directory Size:
$(du -sh "${BUILD_DIR}" 2>/dev/null || echo "unknown")

Cache Directory Sizes:
EOF
    
    # Check various cache directories
    CACHE_DIRS=(
        ".cache/ccache"
        ".cache/hexagon-toolchain"
        "${BUILD_DIR}/CMakeFiles"
    )
    
    for cache_dir in "${CACHE_DIRS[@]}"; do
        if [ -d "${cache_dir}" ]; then
            cache_size=$(du -sh "${cache_dir}" 2>/dev/null | cut -f1 || echo "unknown")
            echo "${cache_dir}: ${cache_size}" >> "${MONITOR_LOG}"
        fi
    done
    
    echo "" >> "${MONITOR_LOG}"
}

# CI integration monitoring
monitor_ci_integration() {
    echo -e "${PURPLE}=== Monitoring CI Integration ===${NC}"
    
    cat >> "${MONITOR_LOG}" << EOF
=== CI INTEGRATION STATUS ===
Pipeline Source: ${CI_PIPELINE_SOURCE:-local}
Job Name: ${CI_JOB_NAME:-local}
Commit SHA: ${CI_COMMIT_SHA:-unknown}
Branch: ${CI_COMMIT_REF_NAME:-unknown}
MR Labels: ${CI_MERGE_REQUEST_LABELS:-none}

EOF
    
    # Check for CI-specific artifacts
    CI_ARTIFACTS=(
        "${BUILD_DIR}/Testing"
        "${MONITOR_DIR}"
        "${BUILD_DIR}/validation"
    )
    
    echo "CI Artifacts Status:" >> "${MONITOR_LOG}"
    for artifact in "${CI_ARTIFACTS[@]}"; do
        if [ -d "${artifact}" ]; then
            artifact_size=$(du -sh "${artifact}" 2>/dev/null | cut -f1 || echo "unknown")
            echo "  ${artifact}: present (${artifact_size})" >> "${MONITOR_LOG}"
        else
            echo "  ${artifact}: missing" >> "${MONITOR_LOG}"
        fi
    done
}

# Generate comprehensive report
generate_report() {
    echo -e "${CYAN}=== Generating Comprehensive Report ===${NC}"
    
    local report_file="${MONITOR_DIR}/hexagon-ci-report.md"
    
    cat > "${report_file}" << EOF
# Hexagon CI Test Report

**Generated:** $(date -u +"%Y-%m-%d %H:%M:%S UTC")  
**Architecture:** ${EIGEN_CI_HEXAGON_ARCH:-v68}  
**Job:** ${CI_JOB_NAME:-local}  
**Commit:** ${CI_COMMIT_SHA:-unknown}  

## Test Results Summary

EOF
    
    # Add test results if available
    if [ -f "${TEST_RESULTS_JSON}" ]; then
        echo "### Test Execution" >> "${report_file}"
        echo "\`\`\`json" >> "${report_file}"
        cat "${TEST_RESULTS_JSON}" >> "${report_file}"
        echo "\`\`\`" >> "${report_file}"
        echo "" >> "${report_file}"
    fi
    
    # Add performance results if available
    if [ -f "${PERFORMANCE_JSON}" ]; then
        echo "### QEMU Performance" >> "${report_file}"
        echo "\`\`\`json" >> "${report_file}"
        cat "${PERFORMANCE_JSON}" >> "${report_file}"
        echo "\`\`\`" >> "${report_file}"
        echo "" >> "${report_file}"
    fi
    
    # Add validation results if available
    if [ -f "${BUILD_DIR}/validation/validation-summary.json" ]; then
        echo "### Validation Results" >> "${report_file}"
        echo "\`\`\`json" >> "${report_file}"
        cat "${BUILD_DIR}/validation/validation-summary.json" >> "${report_file}"
        echo "\`\`\`" >> "${report_file}"
        echo "" >> "${report_file}"
    fi
    
    echo "### Detailed Logs" >> "${report_file}"
    echo "- [Full Monitor Log](monitor.log)" >> "${report_file}"
    if [ -f "${BUILD_DIR}/validation/validation.log" ]; then
        echo "- [Validation Log](../validation/validation.log)" >> "${report_file}"
    fi
    if [ -f "${MONITOR_DIR}/ctest-monitor.log" ]; then
        echo "- [CTest Log](ctest-monitor.log)" >> "${report_file}"
    fi
    
    echo -e "${GREEN}Report generated: ${report_file}${NC}"
}

# Main monitoring execution
main() {
    echo -e "${BLUE}Starting comprehensive Hexagon CI monitoring...${NC}"
    
    collect_system_info
    monitor_build_status
    monitor_test_execution
    monitor_qemu_performance
    monitor_resource_usage
    monitor_ci_integration
    generate_report
    
    echo -e "${CYAN}=== Monitoring Complete ===${NC}"
    echo -e "${CYAN}Results saved to: ${MONITOR_DIR}${NC}"
    echo -e "${CYAN}Monitor log: ${MONITOR_LOG}${NC}"
    
    # Print summary
    if [ -f "${TEST_RESULTS_JSON}" ]; then
        echo -e "${YELLOW}Test Results Summary:${NC}"
        cat "${TEST_RESULTS_JSON}" | python3 -m json.tool 2>/dev/null || cat "${TEST_RESULTS_JSON}"
    fi
}

# Execute monitoring
main "$@" 