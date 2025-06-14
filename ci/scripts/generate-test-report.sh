#!/bin/bash

# Comprehensive Test Report Generation for Hexagon CI
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

echo -e "${BLUE}=== Generating Hexagon Test Report ===${NC}"

# Configuration
BUILD_DIR=${EIGEN_CI_BUILDDIR:-.build}
REPORT_DIR="${BUILD_DIR}/reports"
FINAL_REPORT="${REPORT_DIR}/hexagon-test-report.html"
JSON_REPORT="${REPORT_DIR}/hexagon-test-report.json"

# Create reports directory
mkdir -p "${REPORT_DIR}"

# Collect all available data
echo -e "${YELLOW}Collecting test data...${NC}"

# Initialize report data
report_timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
job_name=${CI_JOB_NAME:-"local"}
commit_sha=${CI_COMMIT_SHA:-"unknown"}
architecture=${EIGEN_CI_HEXAGON_ARCH:-"v68"}
toolchain_version=${EIGEN_CI_HEXAGON_TOOLCHAIN_VERSION:-"20.1.4"}

# Collect test results
test_results=""
if [ -f "${BUILD_DIR}/testing/test-results.json" ]; then
    test_results=$(cat "${BUILD_DIR}/testing/test-results.json")
    echo -e "${GREEN}✓ Found test results${NC}"
else
    test_results='{"test_results": {"total": 0, "passed": 0, "failed": 0, "skipped": 0}}'
    echo -e "${YELLOW}⚠ No test results found${NC}"
fi

# Collect validation results
validation_results=""
if [ -f "${BUILD_DIR}/validation/validation-summary.json" ]; then
    validation_results=$(cat "${BUILD_DIR}/validation/validation-summary.json")
    echo -e "${GREEN}✓ Found validation results${NC}"
else
    validation_results='{"validation": {"total_checks": 0, "passed_checks": 0, "failed_checks": 0}}'
    echo -e "${YELLOW}⚠ No validation results found${NC}"
fi

# Collect monitoring data
monitoring_data=""
if [ -f "${BUILD_DIR}/monitoring/performance.json" ]; then
    monitoring_data=$(cat "${BUILD_DIR}/monitoring/performance.json")
    echo -e "${GREEN}✓ Found monitoring data${NC}"
else
    monitoring_data='{"qemu_performance": {"total_tests": 0, "successful": 0, "failed": 0}}'
    echo -e "${YELLOW}⚠ No monitoring data found${NC}"
fi

# Generate comprehensive JSON report
echo -e "${YELLOW}Generating JSON report...${NC}"

cat > "${JSON_REPORT}" << EOF
{
  "report_metadata": {
    "timestamp": "${report_timestamp}",
    "job_name": "${job_name}",
    "commit_sha": "${commit_sha}",
    "architecture": "${architecture}",
    "toolchain_version": "${toolchain_version}",
    "ci_pipeline": "${CI_PIPELINE_SOURCE:-local}",
    "ci_branch": "${CI_COMMIT_REF_NAME:-unknown}"
  },
  "test_results": ${test_results},
  "validation_results": ${validation_results},
  "monitoring_data": ${monitoring_data},
  "logs": {
    "test_log": "${BUILD_DIR}/testing/hexagon-test.log",
    "validation_log": "${BUILD_DIR}/validation/validation.log",
    "monitor_log": "${BUILD_DIR}/monitoring/monitor.log"
  }
}
EOF

echo -e "${GREEN}✓ JSON report generated: ${JSON_REPORT}${NC}"

# Generate simple text report for now
echo -e "${YELLOW}Generating text report...${NC}"

TEXT_REPORT="${REPORT_DIR}/hexagon-test-report.txt"

cat > "${TEXT_REPORT}" << EOF
HEXAGON CI TEST REPORT
======================

Metadata:
- Timestamp: ${report_timestamp}
- Job: ${job_name}
- Commit: ${commit_sha}
- Architecture: ${architecture}
- Toolchain: ${toolchain_version}
- Pipeline: ${CI_PIPELINE_SOURCE:-local}
- Branch: ${CI_COMMIT_REF_NAME:-unknown}

Test Results:
$(echo "${test_results}" | python3 -m json.tool 2>/dev/null || echo "No test results available")

Validation Results:
$(echo "${validation_results}" | python3 -m json.tool 2>/dev/null || echo "No validation results available")

Monitoring Data:
$(echo "${monitoring_data}" | python3 -m json.tool 2>/dev/null || echo "No monitoring data available")

Log Files:
- Test Log: ${BUILD_DIR}/testing/hexagon-test.log
- Validation Log: ${BUILD_DIR}/validation/validation.log
- Monitor Log: ${BUILD_DIR}/monitoring/monitor.log

EOF

echo -e "${GREEN}✓ Text report generated: ${TEXT_REPORT}${NC}"

# Generate JUnit XML report for GitLab CI integration
echo -e "${YELLOW}Generating JUnit XML report...${NC}"

JUNIT_XML="${BUILD_DIR}/testing/test-results.xml"

# Extract test data for JUnit XML
total_tests=$(echo "${test_results}" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('test_results', {}).get('total', 0))" 2>/dev/null || echo "0")
passed_tests=$(echo "${test_results}" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('test_results', {}).get('passed', 0))" 2>/dev/null || echo "0")
failed_tests=$(echo "${test_results}" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('test_results', {}).get('failed', 0))" 2>/dev/null || echo "0")
skipped_tests=$(echo "${test_results}" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('test_results', {}).get('skipped', 0))" 2>/dev/null || echo "0")

cat > "${JUNIT_XML}" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="HexagonTests" tests="${total_tests}" failures="${failed_tests}" skipped="${skipped_tests}" time="0">
    <properties>
        <property name="architecture" value="${architecture}"/>
        <property name="toolchain_version" value="${toolchain_version}"/>
        <property name="job_name" value="${job_name}"/>
        <property name="commit_sha" value="${commit_sha}"/>
    </properties>
    <testcase classname="HexagonTests" name="OverallTestExecution" time="0">
        $(if [ "${failed_tests}" -gt 0 ]; then
            echo "<failure message=\"${failed_tests} tests failed out of ${total_tests} total tests\"/>"
        fi)
    </testcase>
    <system-out><![CDATA[
Hexagon Test Report Summary:
- Total Tests: ${total_tests}
- Passed: ${passed_tests}
- Failed: ${failed_tests}
- Skipped: ${skipped_tests}
- Architecture: ${architecture}
- Toolchain: ${toolchain_version}
    ]]></system-out>
</testsuite>
EOF

echo -e "${GREEN}✓ JUnit XML report generated: ${JUNIT_XML}${NC}"

# Create summary file for CI consumption
echo -e "${YELLOW}Creating CI summary...${NC}"

CI_SUMMARY="${REPORT_DIR}/ci-summary.txt"

cat > "${CI_SUMMARY}" << EOF
HEXAGON CI TEST SUMMARY
========================
Job: ${job_name}
Commit: ${commit_sha}
Architecture: ${architecture}
Toolchain: ${toolchain_version}
Timestamp: ${report_timestamp}

TEST RESULTS:
- Total Tests: ${total_tests}
- Passed: ${passed_tests}
- Failed: ${failed_tests}
- Skipped: ${skipped_tests}

VALIDATION RESULTS:
- Total Checks: $(echo "${validation_results}" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('total_checks', 0))" 2>/dev/null || echo "0")
- Passed: $(echo "${validation_results}" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('passed_checks', 0))" 2>/dev/null || echo "0")
- Failed: $(echo "${validation_results}" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('failed_checks', 0))" 2>/dev/null || echo "0")

QEMU PERFORMANCE:
- QEMU Tests: $(echo "${monitoring_data}" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('qemu_performance', {}).get('total_tests', 0))" 2>/dev/null || echo "0")
- Successful: $(echo "${monitoring_data}" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('qemu_performance', {}).get('successful', 0))" 2>/dev/null || echo "0")

REPORT FILES:
- Text Report: ${TEXT_REPORT}
- JSON Report: ${JSON_REPORT}
- JUnit XML: ${JUNIT_XML}
EOF

echo -e "${GREEN}✓ CI summary created: ${CI_SUMMARY}${NC}"

echo -e "${CYAN}=== Report Generation Complete ===${NC}"
echo -e "${CYAN}Reports generated in: ${REPORT_DIR}${NC}"
echo -e "${CYAN}Main report: ${TEXT_REPORT}${NC}"
echo -e "${CYAN}JSON data: ${JSON_REPORT}${NC}"
echo -e "${CYAN}JUnit XML: ${JUNIT_XML}${NC}"
echo -e "${CYAN}CI Summary: ${CI_SUMMARY}${NC}" 