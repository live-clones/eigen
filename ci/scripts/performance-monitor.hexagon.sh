#!/bin/bash

# Performance Monitor and Optimizer for Hexagon CI Pipeline
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

echo -e "${CYAN}=== Hexagon CI Performance Monitor ===${NC}"

# Configuration
PERFORMANCE_LOG="${EIGEN_CI_BUILDDIR:-.build}/performance-monitor.log"
METRICS_JSON="${EIGEN_CI_BUILDDIR:-.build}/performance-metrics.json"
OPTIMIZATION_REPORT="${EIGEN_CI_BUILDDIR:-.build}/optimization-report.txt"
MONITOR_INTERVAL=${EIGEN_MONITOR_INTERVAL:-5}  # seconds
ENABLE_PROFILING=${EIGEN_ENABLE_PROFILING:-false}

# Create monitoring directories
mkdir -p "$(dirname "${PERFORMANCE_LOG}")"

# Initialize performance log
cat > "${PERFORMANCE_LOG}" << EOF
=== Hexagon CI Performance Monitor Log ===
Start Time: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Job: ${CI_JOB_NAME:-local}
Commit: ${CI_COMMIT_SHA:-unknown}
Pipeline: ${CI_PIPELINE_SOURCE:-local}
Monitoring Interval: ${MONITOR_INTERVAL}s
Profiling Enabled: ${ENABLE_PROFILING}

EOF

# Global metrics storage
declare -A METRICS
METRICS[start_time]=$(date +%s)
METRICS[peak_memory]=0
METRICS[peak_cpu]=0
METRICS[peak_load]=0
METRICS[total_disk_io]=0
METRICS[total_network_io]=0

# System information collection
collect_system_info() {
    echo -e "${BLUE}=== Collecting System Information ===${NC}"
    
    local system_info=$(cat << EOF
System Information:
  OS: $(uname -s) $(uname -r)
  Architecture: $(uname -m)
  CPU Info: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
  CPU Cores: $(nproc)
  Total Memory: $(free -h | awk '/^Mem:/{print $2}')
  Available Memory: $(free -h | awk '/^Mem:/{print $7}')
  Disk Space: $(df -h . | awk 'NR==2{print $2 " total, " $4 " available"}')
  Swap: $(free -h | awk '/^Swap:/{print $2}')
  Load Average: $(uptime | awk -F'load average:' '{print $2}')
  
Docker Information (if available):
EOF
)
    
    echo "$system_info"
    echo "$system_info" >> "${PERFORMANCE_LOG}"
    
    # Docker info if available
    if command -v docker &> /dev/null && docker info &> /dev/null; then
        local docker_info="  Docker Version: $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo 'N/A')"
        echo "$docker_info"
        echo "$docker_info" >> "${PERFORMANCE_LOG}"
    fi
    
    echo ""
}

# Resource monitoring function
monitor_resources() {
    local duration="$1"
    local output_file="$2"
    local process_name="${3:-}"
    
    echo -e "${YELLOW}Monitoring resources for ${duration}s...${NC}"
    
    local end_time=$(($(date +%s) + duration))
    local sample_count=0
    
    # Initialize monitoring output
    echo "timestamp,cpu_percent,memory_mb,memory_percent,disk_read_mb,disk_write_mb,network_rx_mb,network_tx_mb,load_1min,processes" > "$output_file"
    
    while [ $(date +%s) -lt $end_time ]; do
        local timestamp=$(date +%s)
        
        # CPU usage
        local cpu_percent=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
        
        # Memory usage
        local memory_info=$(free -m)
        local memory_used=$(echo "$memory_info" | awk '/^Mem:/{print $3}')
        local memory_total=$(echo "$memory_info" | awk '/^Mem:/{print $2}')
        local memory_percent=$(echo "$memory_used $memory_total" | awk '{printf "%.1f", ($1/$2)*100}')
        
        # Disk I/O
        local disk_stats=$(cat /proc/diskstats | awk '{read+=$6; write+=$10} END {printf "%.2f %.2f", read/2048, write/2048}')
        local disk_read=$(echo "$disk_stats" | awk '{print $1}')
        local disk_write=$(echo "$disk_stats" | awk '{print $2}')
        
        # Network I/O
        local network_stats=$(cat /proc/net/dev | tail -n +3 | awk '{rx+=$2; tx+=$10} END {printf "%.2f %.2f", rx/1048576, tx/1048576}')
        local network_rx=$(echo "$network_stats" | awk '{print $1}')
        local network_tx=$(echo "$network_stats" | awk '{print $2}')
        
        # Load average
        local load_1min=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
        
        # Process count
        local process_count=$(ps aux | wc -l)
        
        # Update peak metrics
        if (( $(echo "$cpu_percent > ${METRICS[peak_cpu]}" | bc -l 2>/dev/null || echo "0") )); then
            METRICS[peak_cpu]=$cpu_percent
        fi
        
        if [ "$memory_used" -gt "${METRICS[peak_memory]}" ]; then
            METRICS[peak_memory]=$memory_used
        fi
        
        if (( $(echo "$load_1min > ${METRICS[peak_load]}" | bc -l 2>/dev/null || echo "0") )); then
            METRICS[peak_load]=$load_1min
        fi
        
        # Log sample
        echo "$timestamp,$cpu_percent,$memory_used,$memory_percent,$disk_read,$disk_write,$network_rx,$network_tx,$load_1min,$process_count" >> "$output_file"
        
        sample_count=$((sample_count + 1))
        sleep "$MONITOR_INTERVAL"
    done
    
    echo -e "${GREEN}✓ Collected $sample_count samples${NC}"
}

# Process-specific monitoring
monitor_process() {
    local process_name="$1"
    local duration="$2"
    local output_file="$3"
    
    echo -e "${YELLOW}Monitoring process '$process_name' for ${duration}s...${NC}"
    
    local end_time=$(($(date +%s) + duration))
    local sample_count=0
    
    # Initialize process monitoring output
    echo "timestamp,pid,cpu_percent,memory_mb,threads,files_open,context_switches" > "$output_file"
    
    while [ $(date +%s) -lt $end_time ]; do
        local timestamp=$(date +%s)
        
        # Find processes matching name
        local pids=$(pgrep -f "$process_name" 2>/dev/null || echo "")
        
        if [ -n "$pids" ]; then
            for pid in $pids; do
                if [ -d "/proc/$pid" ]; then
                    # Process CPU and memory
                    local proc_stats=$(ps -o pid,pcpu,rss,nlwp -p "$pid" --no-headers 2>/dev/null || echo "")
                    if [ -n "$proc_stats" ]; then
                        local proc_cpu=$(echo "$proc_stats" | awk '{print $2}')
                        local proc_memory=$(echo "$proc_stats" | awk '{printf "%.2f", $3/1024}')  # Convert KB to MB
                        local proc_threads=$(echo "$proc_stats" | awk '{print $4}')
                        
                        # Open files
                        local open_files=$(ls /proc/$pid/fd 2>/dev/null | wc -l || echo "0")
                        
                        # Context switches
                        local context_switches=$(grep voluntary_ctxt_switches /proc/$pid/status 2>/dev/null | awk '{print $2}' || echo "0")
                        
                        echo "$timestamp,$pid,$proc_cpu,$proc_memory,$proc_threads,$open_files,$context_switches" >> "$output_file"
                    fi
                fi
            done
        fi
        
        sample_count=$((sample_count + 1))
        sleep "$MONITOR_INTERVAL"
    done
    
    echo -e "${GREEN}✓ Collected $sample_count process samples${NC}"
}

# Performance benchmarking
run_performance_benchmark() {
    local benchmark_type="$1"
    local build_dir="$2"
    
    echo -e "${BLUE}=== Running Performance Benchmark: $benchmark_type ===${NC}"
    
    local benchmark_start=$(date +%s)
    local benchmark_log="${EIGEN_CI_BUILDDIR:-.build}/benchmark-${benchmark_type}.log"
    
    case "$benchmark_type" in
        "compile_time")
            benchmark_compile_time "$build_dir" "$benchmark_log"
            ;;
        "test_execution")
            benchmark_test_execution "$build_dir" "$benchmark_log"
            ;;
        "memory_usage")
            benchmark_memory_usage "$build_dir" "$benchmark_log"
            ;;
        "disk_io")
            benchmark_disk_io "$build_dir" "$benchmark_log"
            ;;
        *)
            echo -e "${RED}Unknown benchmark type: $benchmark_type${NC}"
            return 1
            ;;
    esac
    
    local benchmark_end=$(date +%s)
    local benchmark_duration=$((benchmark_end - benchmark_start))
    
    echo -e "${GREEN}✓ Benchmark completed in ${benchmark_duration}s${NC}"
    echo "Benchmark ($benchmark_type): ${benchmark_duration}s" >> "${PERFORMANCE_LOG}"
}

# Compile time benchmarking
benchmark_compile_time() {
    local build_dir="$1"
    local log_file="$2"
    
    echo "=== Compile Time Benchmark ===" > "$log_file"
    echo "Start: $(date)" >> "$log_file"
    
    cd "$build_dir"
    
    # Clean build for accurate timing
    if [ -f "build.ninja" ]; then
        ninja clean >/dev/null 2>&1 || true
    fi
    
    # Time the build process
    local compile_start=$(date +%s.%N)
    
    # Monitor compilation
    local monitor_file="${build_dir}/compile-monitor.csv"
    monitor_resources 300 "$monitor_file" "ninja" &
    local monitor_pid=$!
    
    # Run build
    if ninja -j$(nproc) 2>&1 | tee -a "$log_file"; then
        local compile_end=$(date +%s.%N)
        local compile_time=$(echo "$compile_end - $compile_start" | bc)
        
        echo "Compilation successful in ${compile_time}s" >> "$log_file"
        METRICS[compile_time]=$compile_time
        
        # Stop monitoring
        kill $monitor_pid 2>/dev/null || true
        
        # Analyze compile metrics
        if [ -f "$monitor_file" ]; then
            local avg_cpu=$(tail -n +2 "$monitor_file" | awk -F',' '{sum+=$2; count++} END {printf "%.1f", sum/count}')
            local avg_memory=$(tail -n +2 "$monitor_file" | awk -F',' '{sum+=$3; count++} END {printf "%.1f", sum/count}')
            
            echo "Average CPU during compilation: ${avg_cpu}%" >> "$log_file"
            echo "Average Memory during compilation: ${avg_memory}MB" >> "$log_file"
            
            METRICS[compile_avg_cpu]=$avg_cpu
            METRICS[compile_avg_memory]=$avg_memory
        fi
        
        echo -e "${GREEN}✓ Compilation benchmark: ${compile_time}s${NC}"
    else
        kill $monitor_pid 2>/dev/null || true
        echo -e "${RED}✗ Compilation failed${NC}"
        return 1
    fi
}

# Test execution benchmarking
benchmark_test_execution() {
    local build_dir="$1"
    local log_file="$2"
    
    echo "=== Test Execution Benchmark ===" > "$log_file"
    echo "Start: $(date)" >> "$log_file"
    
    cd "$build_dir"
    
    # Find test binaries
    local test_binaries=$(find . -name "*test*" -type f -executable 2>/dev/null | head -5)
    
    if [ -z "$test_binaries" ]; then
        echo "No test binaries found" >> "$log_file"
        return 0
    fi
    
    local total_test_time=0
    local test_count=0
    
    for test_binary in $test_binaries; do
        echo "Testing: $test_binary" >> "$log_file"
        
        local test_start=$(date +%s.%N)
        
        # Run test with timeout
        if timeout 60 "$test_binary" >> "$log_file" 2>&1; then
            local test_end=$(date +%s.%N)
            local test_time=$(echo "$test_end - $test_start" | bc)
            
            echo "Test completed in ${test_time}s" >> "$log_file"
            total_test_time=$(echo "$total_test_time + $test_time" | bc)
            test_count=$((test_count + 1))
            
            echo -e "${GREEN}✓ $test_binary: ${test_time}s${NC}"
        else
            echo -e "${YELLOW}⚠ $test_binary: timeout or failed${NC}"
            echo "Test failed or timed out" >> "$log_file"
        fi
    done
    
    if [ $test_count -gt 0 ]; then
        local avg_test_time=$(echo "$total_test_time / $test_count" | bc -l)
        echo "Total test time: ${total_test_time}s" >> "$log_file"
        echo "Average test time: ${avg_test_time}s" >> "$log_file"
        
        METRICS[total_test_time]=$total_test_time
        METRICS[avg_test_time]=$avg_test_time
        METRICS[test_count]=$test_count
    fi
}

# Memory usage benchmarking
benchmark_memory_usage() {
    local build_dir="$1"
    local log_file="$2"
    
    echo "=== Memory Usage Benchmark ===" > "$log_file"
    echo "Start: $(date)" >> "$log_file"
    
    # Memory stress test
    echo "Running memory usage analysis..." >> "$log_file"
    
    # Monitor memory during a representative operation (e.g., cmake configure)
    local memory_log="${build_dir}/memory-usage.csv"
    monitor_resources 60 "$memory_log" &
    local monitor_pid=$!
    
    # Simulate memory-intensive operation
    cd "$build_dir"
    cmake --build . --target clean >/dev/null 2>&1 || true
    cmake .. >/dev/null 2>&1 || true
    
    sleep 30  # Let monitoring collect data
    kill $monitor_pid 2>/dev/null || true
    
    # Analyze memory usage
    if [ -f "$memory_log" ]; then
        local peak_memory=$(tail -n +2 "$memory_log" | awk -F',' 'BEGIN{max=0} {if($3>max) max=$3} END{print max}')
        local avg_memory=$(tail -n +2 "$memory_log" | awk -F',' '{sum+=$3; count++} END {printf "%.1f", sum/count}')
        local peak_memory_percent=$(tail -n +2 "$memory_log" | awk -F',' 'BEGIN{max=0} {if($4>max) max=$4} END{print max}')
        
        echo "Peak Memory Usage: ${peak_memory}MB (${peak_memory_percent}%)" >> "$log_file"
        echo "Average Memory Usage: ${avg_memory}MB" >> "$log_file"
        
        METRICS[peak_memory_usage]=$peak_memory
        METRICS[avg_memory_usage]=$avg_memory
        METRICS[peak_memory_percent]=$peak_memory_percent
        
        echo -e "${GREEN}✓ Memory benchmark: Peak ${peak_memory}MB, Avg ${avg_memory}MB${NC}"
    fi
}

# Disk I/O benchmarking
benchmark_disk_io() {
    local build_dir="$1"
    local log_file="$2"
    
    echo "=== Disk I/O Benchmark ===" > "$log_file"
    echo "Start: $(date)" >> "$log_file"
    
    cd "$build_dir"
    
    # Test write performance
    echo "Testing write performance..." >> "$log_file"
    local write_start=$(date +%s.%N)
    dd if=/dev/zero of=test_write.tmp bs=1M count=100 2>/dev/null || true
    local write_end=$(date +%s.%N)
    local write_time=$(echo "$write_end - $write_start" | bc)
    
    # Test read performance
    echo "Testing read performance..." >> "$log_file"
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true  # Clear cache if possible
    local read_start=$(date +%s.%N)
    dd if=test_write.tmp of=/dev/null bs=1M 2>/dev/null || true
    local read_end=$(date +%s.%N)
    local read_time=$(echo "$read_end - $read_start" | bc)
    
    # Calculate throughput
    local write_throughput=$(echo "scale=2; 100 / $write_time" | bc)
    local read_throughput=$(echo "scale=2; 100 / $read_time" | bc)
    
    echo "Write Performance: ${write_time}s (${write_throughput} MB/s)" >> "$log_file"
    echo "Read Performance: ${read_time}s (${read_throughput} MB/s)" >> "$log_file"
    
    METRICS[disk_write_time]=$write_time
    METRICS[disk_read_time]=$read_time
    METRICS[disk_write_throughput]=$write_throughput
    METRICS[disk_read_throughput]=$read_throughput
    
    # Cleanup
    rm -f test_write.tmp
    
    echo -e "${GREEN}✓ Disk I/O benchmark: Write ${write_throughput} MB/s, Read ${read_throughput} MB/s${NC}"
}

# Generate optimization recommendations
generate_optimization_recommendations() {
    echo -e "${CYAN}=== Generating Optimization Recommendations ===${NC}"
    
    cat > "$OPTIMIZATION_REPORT" << EOF
=== Hexagon CI Performance Optimization Report ===
Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Job: ${CI_JOB_NAME:-local}

PERFORMANCE METRICS:
EOF
    
    # Add collected metrics
    for metric in "${!METRICS[@]}"; do
        echo "  $metric: ${METRICS[$metric]}" >> "$OPTIMIZATION_REPORT"
    done
    
    cat >> "$OPTIMIZATION_REPORT" << EOF

OPTIMIZATION RECOMMENDATIONS:

1. BUILD PERFORMANCE:
   - Use ccache to speed up incremental builds
   - Consider using Docker build cache layers
   - Use parallel testing with appropriate concurrency limits
   - Enable ninja build system for faster incremental builds

2. CI CONFIGURATION:
   - Cache Hexagon toolchain between builds
   - Use LTO (Link Time Optimization) for release builds
   - Consider using mold or lld for faster linking

3. INFRASTRUCTURE:
   - Use runners with NVMe SSDs for build workloads
   - Ensure adequate RAM (recommend 4GB+ for parallel builds)
   - Consider using larger VM instances for better performance

EOF
    
    echo -e "${GREEN}✓ Optimization report saved to: $OPTIMIZATION_REPORT${NC}"
}

# Generate comprehensive metrics JSON
generate_metrics_json() {
    echo -e "${CYAN}=== Generating Performance Metrics JSON ===${NC}"
    
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local end_time=$(date +%s)
    local total_duration=$((end_time - METRICS[start_time]))
    
    cat > "$METRICS_JSON" << EOF
{
  "timestamp": "$timestamp",
  "job_name": "${CI_JOB_NAME:-local}",
  "commit_sha": "${CI_COMMIT_SHA:-unknown}",
  "pipeline_source": "${CI_PIPELINE_SOURCE:-local}",
  "total_duration_seconds": $total_duration,
  "system_info": {
    "os": "$(uname -s)",
    "architecture": "$(uname -m)",
    "cpu_cores": $(nproc),
    "total_memory_mb": $(free -m | awk '/^Mem:/{print $2}'),
    "available_memory_mb": $(free -m | awk '/^Mem:/{print $7}')
  },
  "performance_metrics": {
EOF
    
    # Add all collected metrics
    local first=true
    for metric in "${!METRICS[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> "$METRICS_JSON"
        fi
        echo -n "    \"$metric\": \"${METRICS[$metric]}\"" >> "$METRICS_JSON"
    done
    
    cat >> "$METRICS_JSON" << EOF

  },
  "recommendations": [
    "Enable ccache for faster incremental builds",
    "Use parallel builds with appropriate concurrency",
    "Consider SSD storage for build workloads",
    "Monitor memory usage and adjust parallelism accordingly"
  ],
  "report_files": {
    "performance_log": "$PERFORMANCE_LOG",
    "optimization_report": "$OPTIMIZATION_REPORT",
    "metrics_json": "$METRICS_JSON"
  }
}
EOF
    
    echo -e "${GREEN}✓ Metrics JSON saved to: $METRICS_JSON${NC}"
}

# Main performance monitoring function
run_performance_monitoring() {
    local operation="$1"
    local duration="${2:-300}"  # Default 5 minutes
    local build_dir="${3:-.build}"
    
    echo -e "${BLUE}Starting performance monitoring for: $operation${NC}"
    
    # Collect system information
    collect_system_info
    
    # Start resource monitoring
    local monitor_file="${build_dir}/resource-monitor.csv"
    monitor_resources "$duration" "$monitor_file"
    
    # Generate reports
    generate_optimization_recommendations
    generate_metrics_json
    
    echo -e "${GREEN}✓ Performance monitoring completed${NC}"
    echo -e "${CYAN}Reports available:${NC}"
    echo -e "  Performance Log: $PERFORMANCE_LOG"
    echo -e "  Metrics JSON: $METRICS_JSON"
    echo -e "  Optimization Report: $OPTIMIZATION_REPORT"
}

# If script is called directly, run the performance monitoring
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    run_performance_monitoring "$@"
fi 