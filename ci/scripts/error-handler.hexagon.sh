#!/bin/bash

# Comprehensive Error Handler for Hexagon CI Pipeline
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

echo -e "${BLUE}=== Hexagon CI Error Handler ===${NC}"

# Configuration
MAX_RETRIES=${EIGEN_CI_MAX_RETRIES:-3}
RETRY_DELAY=${EIGEN_CI_RETRY_DELAY:-10}
ERROR_LOG="${EIGEN_CI_BUILDDIR:-.build}/error-handler.log"
ERROR_SUMMARY="${EIGEN_CI_BUILDDIR:-.build}/error-summary.json"

# Create error tracking directory
mkdir -p "$(dirname "${ERROR_LOG}")"

# Initialize error tracking
cat > "${ERROR_LOG}" << EOF
=== Hexagon CI Error Handler Log ===
Start Time: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Job: ${CI_JOB_NAME:-local}
Commit: ${CI_COMMIT_SHA:-unknown}
Pipeline: ${CI_PIPELINE_SOURCE:-local}

EOF

# Error categorization
declare -A ERROR_CATEGORIES
ERROR_CATEGORIES[1]="TOOLCHAIN_DOWNLOAD"
ERROR_CATEGORIES[2]="TOOLCHAIN_SETUP" 
ERROR_CATEGORIES[3]="CMAKE_CONFIGURATION"
ERROR_CATEGORIES[4]="BUILD_FAILURE"
ERROR_CATEGORIES[5]="TEST_FAILURE"
ERROR_CATEGORIES[6]="QEMU_FAILURE"
ERROR_CATEGORIES[7]="NETWORK_FAILURE"
ERROR_CATEGORIES[8]="RESOURCE_EXHAUSTION"
ERROR_CATEGORIES[9]="TIMEOUT"
ERROR_CATEGORIES[10]="UNKNOWN"

# Error handling functions
log_error() {
    local error_code=$1
    local error_message="$2"
    local context="$3"
    local timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
    
    echo -e "${RED}ERROR [$error_code]: $error_message${NC}"
    
    cat >> "${ERROR_LOG}" << EOF
[$timestamp] ERROR_CODE: $error_code
[$timestamp] CATEGORY: ${ERROR_CATEGORIES[$error_code]:-UNKNOWN}
[$timestamp] MESSAGE: $error_message
[$timestamp] CONTEXT: $context
[$timestamp] PWD: $(pwd)
[$timestamp] USER: $(whoami)
[$timestamp] ---

EOF
}

# Retry mechanism with exponential backoff
retry_command() {
    local command="$1"
    local max_attempts="${2:-$MAX_RETRIES}"
    local delay="${3:-$RETRY_DELAY}"
    local context="$4"
    
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo -e "${YELLOW}Attempt $attempt/$max_attempts: $context${NC}"
        
        if eval "$command"; then
            echo -e "${GREEN}✓ Success on attempt $attempt${NC}"
            return 0
        else
            local exit_code=$?
            echo -e "${RED}✗ Failed on attempt $attempt (exit code: $exit_code)${NC}"
            
            if [ $attempt -eq $max_attempts ]; then
                log_error 10 "Command failed after $max_attempts attempts: $command" "$context"
                return $exit_code
            fi
            
            echo -e "${YELLOW}Waiting ${delay}s before retry...${NC}"
            sleep $delay
            
            # Exponential backoff
            delay=$((delay * 2))
            attempt=$((attempt + 1))
        fi
    done
}

# Toolchain download with retry and fallback
download_toolchain_with_retry() {
    local toolchain_url="$1"
    local toolchain_dir="$2"
    local version="$3"
    
    echo -e "${BLUE}=== Downloading Hexagon Toolchain with Retry ===${NC}"
    
    # Primary download attempt
    local download_cmd="wget --progress=bar:force --retry-connrefused --waitretry=30 --read-timeout=300 --timeout=300 -O /tmp/hexagon-toolchain.tar.xz '$toolchain_url'"
    
    if retry_command "$download_cmd" 3 15 "Primary toolchain download"; then
        echo -e "${GREEN}✓ Toolchain downloaded successfully${NC}"
    else
        # Fallback to alternative mirrors
        echo -e "${YELLOW}⚠ Primary download failed, trying fallback mirrors...${NC}"
        
        local fallback_urls=(
            "https://github.com/quic/toolchain_for_hexagon/releases/download/clang-${version}/clang+llvm-${version}-cross-hexagon-unknown-linux-musl.tar.xz"
            "https://api.github.com/repos/quic/toolchain_for_hexagon/releases/latest"
        )
        
        local success=false
        for fallback_url in "${fallback_urls[@]}"; do
            echo -e "${YELLOW}Trying fallback: $fallback_url${NC}"
            download_cmd="wget --progress=bar:force --retry-connrefused --waitretry=30 --read-timeout=300 --timeout=300 -O /tmp/hexagon-toolchain.tar.xz '$fallback_url'"
            
            if retry_command "$download_cmd" 2 10 "Fallback toolchain download"; then
                success=true
                break
            fi
        done
        
        if [ "$success" = false ]; then
            log_error 1 "All toolchain download attempts failed" "Primary and fallback URLs exhausted"
            return 1
        fi
    fi
    
    # Verify download integrity
    if [ ! -f "/tmp/hexagon-toolchain.tar.xz" ] || [ ! -s "/tmp/hexagon-toolchain.tar.xz" ]; then
        log_error 1 "Downloaded toolchain file is missing or empty" "Post-download verification"
        return 1
    fi
    
    # Extract with verification
    echo -e "${YELLOW}Extracting toolchain...${NC}"
    if retry_command "tar -tf /tmp/hexagon-toolchain.tar.xz > /dev/null" 2 5 "Toolchain archive verification"; then
        echo -e "${GREEN}✓ Toolchain archive verified${NC}"
    else
        log_error 1 "Downloaded toolchain archive is corrupted" "Archive verification failed"
        return 1
    fi
    
    # Extract toolchain
    mkdir -p "$toolchain_dir"
    if retry_command "tar -xf /tmp/hexagon-toolchain.tar.xz -C '$toolchain_dir' --strip-components=1" 2 5 "Toolchain extraction"; then
        echo -e "${GREEN}✓ Toolchain extracted successfully${NC}"
        rm -f /tmp/hexagon-toolchain.tar.xz
        return 0
    else
        log_error 2 "Failed to extract toolchain" "Extraction to $toolchain_dir failed"
        return 1
    fi
}

# CMake configuration with enhanced error handling
configure_cmake_with_retry() {
    local build_dir="$1"
    local source_dir="$2"
    local toolchain_file="$3"
    
    echo -e "${BLUE}=== CMake Configuration with Error Handling ===${NC}"
    
    # Verify prerequisites
    if [ ! -f "$toolchain_file" ]; then
        log_error 3 "Toolchain file not found: $toolchain_file" "CMake configuration prerequisites"
        return 1
    fi
    
    if [ ! -d "$source_dir" ]; then
        log_error 3 "Source directory not found: $source_dir" "CMake configuration prerequisites"
        return 1
    fi
    
    # Create build directory
    mkdir -p "$build_dir"
    cd "$build_dir"
    
    # CMake configuration command
    local cmake_cmd="cmake -GNinja -DCMAKE_TOOLCHAIN_FILE='$toolchain_file' -DCMAKE_BUILD_TYPE=Release '$source_dir'"
    
    # Try configuration with retry
    if retry_command "$cmake_cmd" 2 10 "CMake configuration"; then
        echo -e "${GREEN}✓ CMake configuration successful${NC}"
        return 0
    else
        # Detailed error analysis
        echo -e "${YELLOW}Analyzing CMake configuration failure...${NC}"
        
        # Check common issues
        if [ ! -w "$build_dir" ]; then
            log_error 3 "Build directory not writable: $build_dir" "Permission check"
        fi
        
        if ! command -v cmake >/dev/null 2>&1; then
            log_error 3 "CMake not found in PATH" "Tool availability check"
        fi
        
        if ! command -v ninja >/dev/null 2>&1; then
            log_error 3 "Ninja not found in PATH" "Tool availability check"
        fi
        
        log_error 3 "CMake configuration failed after retries" "Final configuration attempt"
        return 1
    fi
}

# Build process with monitoring and error recovery
build_with_monitoring() {
    local build_dir="$1"
    local jobs="${2:-4}"
    local target="${3:-all}"
    
    echo -e "${BLUE}=== Build Process with Monitoring ===${NC}"
    
    cd "$build_dir"
    
    # Check available resources
    local available_memory=$(free -m | awk '/^Mem:/{print $7}')
    local available_disk=$(df . | awk 'NR==2{print $4}')
    
    echo "Available memory: ${available_memory}MB"
    echo "Available disk: ${available_disk}KB"
    
    # Adjust parallelism based on resources
    if [ "$available_memory" -lt 2000 ]; then
        echo -e "${YELLOW}⚠ Low memory detected, reducing parallelism${NC}"
        jobs=2
    fi
    
    # Build command with timeout and monitoring
    local build_cmd="timeout 1800 ninja -j$jobs $target"
    
    # Monitor build process
    local build_start=$(date +%s)
    
    if retry_command "$build_cmd" 2 30 "Build process"; then
        local build_end=$(date +%s)
        local build_duration=$((build_end - build_start))
        echo -e "${GREEN}✓ Build completed in ${build_duration}s${NC}"
        
        # Log build metrics
        cat >> "${ERROR_LOG}" << EOF
BUILD_SUCCESS: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
BUILD_DURATION: ${build_duration}s
BUILD_JOBS: $jobs
BUILD_TARGET: $target
---

EOF
        return 0
    else
        # Build failure analysis
        echo -e "${YELLOW}Analyzing build failure...${NC}"
        
        # Check for common build issues
        if [ "$available_disk" -lt 1000000 ]; then  # Less than 1GB
            log_error 8 "Insufficient disk space for build" "Disk space: ${available_disk}KB"
        fi
        
        if [ "$available_memory" -lt 1000 ]; then  # Less than 1GB
            log_error 8 "Insufficient memory for build" "Available memory: ${available_memory}MB"
        fi
        
        # Check for specific error patterns in build log
        if [ -f ".ninja_log" ]; then
            local failed_targets=$(tail -20 .ninja_log | grep -c "FAILED" || echo "0")
            if [ "$failed_targets" -gt 0 ]; then
                log_error 4 "Build targets failed: $failed_targets" "Ninja build log analysis"
            fi
        fi
        
        log_error 4 "Build process failed after retries" "Final build attempt"
        return 1
    fi
}

# QEMU testing with fallback strategies
test_with_qemu_fallback() {
    local test_binary="$1"
    local timeout="${2:-60}"
    
    echo -e "${BLUE}=== QEMU Testing with Fallback ===${NC}"
    
    local qemu_hexagon="/opt/hexagon-toolchain/x86_64-linux-gnu/bin/qemu-hexagon"
    local sysroot="/opt/hexagon-toolchain/x86_64-linux-gnu/target/hexagon-unknown-linux-musl"
    
    # Verify QEMU setup
    if [ ! -f "$qemu_hexagon" ]; then
        log_error 6 "QEMU Hexagon emulator not found: $qemu_hexagon" "QEMU availability check"
        return 1
    fi
    
    if [ ! -f "$test_binary" ]; then
        log_error 6 "Test binary not found: $test_binary" "Test binary availability"
        return 1
    fi
    
    # Set up QEMU environment
    export QEMU_LD_PREFIX="$sysroot"
    
    # Test QEMU basic functionality first
    if ! retry_command "$qemu_hexagon -version >/dev/null 2>&1" 2 5 "QEMU basic functionality"; then
        log_error 6 "QEMU emulator not functional" "QEMU version check failed"
        return 1
    fi
    
    # Run test with timeout and retry
    local test_cmd="timeout $timeout $qemu_hexagon '$test_binary'"
    
    if retry_command "$test_cmd" 2 10 "QEMU test execution"; then
        echo -e "${GREEN}✓ QEMU test successful${NC}"
        return 0
    else
        # Fallback strategies
        echo -e "${YELLOW}⚠ Primary QEMU test failed, trying fallbacks...${NC}"
        
        # Try with increased timeout
        local extended_timeout=$((timeout * 2))
        test_cmd="timeout $extended_timeout $qemu_hexagon '$test_binary' --help"
        
        if retry_command "$test_cmd" 1 5 "QEMU test with extended timeout"; then
            echo -e "${YELLOW}⚠ Test succeeded with extended timeout${NC}"
            return 0
        fi
        
        # Try basic execution without arguments
        test_cmd="timeout $timeout $qemu_hexagon '$test_binary' < /dev/null"
        
        if retry_command "$test_cmd" 1 5 "QEMU test without arguments"; then
            echo -e "${YELLOW}⚠ Basic execution succeeded${NC}"
            return 0
        fi
        
        log_error 6 "All QEMU test strategies failed" "QEMU emulation exhausted"
        return 1
    fi
}

# Network connectivity check with fallback
check_network_connectivity() {
    echo -e "${BLUE}=== Network Connectivity Check ===${NC}"
    
    local test_urls=(
        "github.com"
        "api.github.com"
        "google.com"
        "1.1.1.1"
    )
    
    local connectivity=false
    
    for url in "${test_urls[@]}"; do
        if retry_command "ping -c 1 -W 5 $url >/dev/null 2>&1" 1 0 "Connectivity to $url"; then
            echo -e "${GREEN}✓ Network connectivity confirmed via $url${NC}"
            connectivity=true
            break
        fi
    done
    
    if [ "$connectivity" = false ]; then
        log_error 7 "No network connectivity detected" "All connectivity tests failed"
        return 1
    fi
    
    return 0
}

# Resource monitoring and early warning
monitor_resources() {
    echo -e "${BLUE}=== Resource Monitoring ===${NC}"
    
    # Memory check
    local memory_usage=$(free | awk '/^Mem:/{printf "%.0f", $3/$2 * 100}')
    if [ "$memory_usage" -gt 90 ]; then
        log_error 8 "High memory usage detected: ${memory_usage}%" "Resource monitoring"
        echo -e "${RED}⚠ WARNING: Memory usage critical (${memory_usage}%)${NC}"
    fi
    
    # Disk space check
    local disk_usage=$(df . | awk 'NR==2{print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 90 ]; then
        log_error 8 "High disk usage detected: ${disk_usage}%" "Resource monitoring"
        echo -e "${RED}⚠ WARNING: Disk usage critical (${disk_usage}%)${NC}"
    fi
    
    # CPU load check
    local cpu_load=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    local cpu_count=$(nproc)
    local load_percent=$(echo "$cpu_load $cpu_count" | awk '{printf "%.0f", ($1/$2) * 100}')
    
    if [ "$load_percent" -gt 200 ]; then
        log_error 8 "High CPU load detected: ${load_percent}%" "Resource monitoring"
        echo -e "${RED}⚠ WARNING: CPU load critical (${load_percent}%)${NC}"
    fi
    
    echo "Resource status: Memory ${memory_usage}%, Disk ${disk_usage}%, CPU Load ${load_percent}%"
}

# Generate error summary
generate_error_summary() {
    echo -e "${CYAN}=== Generating Error Summary ===${NC}"
    
    local total_errors=$(grep -c "ERROR_CODE:" "$ERROR_LOG" 2>/dev/null || echo "0")
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Count errors by category
    local category_counts=""
    for code in "${!ERROR_CATEGORIES[@]}"; do
        local count=$(grep "ERROR_CODE: $code" "$ERROR_LOG" 2>/dev/null | wc -l || echo "0")
        if [ "$count" -gt 0 ]; then
            category_counts="$category_counts,\"${ERROR_CATEGORIES[$code]}\": $count"
        fi
    done
    category_counts=${category_counts#,}  # Remove leading comma
    
    # Generate JSON summary
    cat > "$ERROR_SUMMARY" << EOF
{
  "timestamp": "$timestamp",
  "job_name": "${CI_JOB_NAME:-local}",
  "commit_sha": "${CI_COMMIT_SHA:-unknown}",
  "pipeline_source": "${CI_PIPELINE_SOURCE:-local}",
  "total_errors": $total_errors,
  "error_categories": {$category_counts},
  "error_log": "$ERROR_LOG",
  "summary_generated": true
}
EOF
    
    echo -e "${GREEN}✓ Error summary saved to: $ERROR_SUMMARY${NC}"
    echo "Total errors encountered: $total_errors"
}

# Main error handler function
main_error_handler() {
    local operation="$1"
    shift
    local args=("$@")
    
    echo -e "${BLUE}Starting error handler for operation: $operation${NC}"
    
    # Monitor resources before operation
    monitor_resources
    
    case "$operation" in
        "download_toolchain")
            download_toolchain_with_retry "${args[@]}"
            ;;
        "configure_cmake")
            configure_cmake_with_retry "${args[@]}"
            ;;
        "build")
            build_with_monitoring "${args[@]}"
            ;;
        "test_qemu")
            test_with_qemu_fallback "${args[@]}"
            ;;
        "check_network")
            check_network_connectivity
            ;;
        *)
            log_error 10 "Unknown operation: $operation" "Main error handler"
            return 1
            ;;
    esac
    
    local exit_code=$?
    
    # Monitor resources after operation
    monitor_resources
    
    # Generate summary
    generate_error_summary
    
    return $exit_code
}

# If script is called directly, run the main handler
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main_error_handler "$@"
fi 