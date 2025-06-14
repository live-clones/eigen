# Hexagon CI Pipeline Guide

## Overview

The Hexagon CI pipeline provides comprehensive support for building and testing Eigen with Qualcomm Hexagon DSP toolchain. This guide covers setup, configuration, troubleshooting, and best practices for the Hexagon CI integration.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Configuration](#configuration)
4. [Build Variants](#build-variants)
5. [Testing](#testing)
6. [Performance Monitoring](#performance-monitoring)
7. [Error Handling](#error-handling)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)
10. [Advanced Usage](#advanced-usage)

## Quick Start

### Prerequisites

- GitLab Runner with Docker support
- Internet access for toolchain download
- Minimum 4GB RAM and 20GB disk space
- Linux environment (Ubuntu 20.04+ recommended)

### Basic Setup

1. **Enable Hexagon builds in your pipeline**:
```yaml
include:
  - local: 'ci/build.hexagon.gitlab-ci.yml'
  - local: 'ci/test.hexagon.gitlab-ci.yml'
```

2. **Configure variables** (optional):
```yaml
variables:
  HEXAGON_TOOLCHAIN_VERSION: "17.0.0"
  HEXAGON_CACHE_ENABLED: "true"
  EIGEN_HEXAGON_JOBS: "4"
```

3. **Run your first Hexagon build**:
```bash
# Trigger a pipeline or push to your branch
git push origin feature/hexagon-support
```

## Pipeline Architecture

### Build Stages

The Hexagon CI pipeline consists of three main stages:

1. **Setup Stage** (`setup_hexagon`)
   - Downloads and installs Hexagon toolchain
   - Sets up QEMU emulation environment
   - Configures build environment

2. **Build Stage** (`build_hexagon_*`)
   - Compiles Eigen with various Hexagon configurations
   - Supports multiple build variants (Debug, Release, HVX, etc.)
   - Generates build artifacts

3. **Test Stage** (`test_hexagon_*`)
   - Runs unit tests under QEMU emulation
   - Performs performance benchmarks
   - Validates build artifacts

### File Structure

```
ci/
├── build.hexagon.gitlab-ci.yml      # Build job definitions
├── test.hexagon.gitlab-ci.yml       # Test job definitions
├── scripts/
│   ├── setup.hexagon.sh             # Toolchain setup
│   ├── test.hexagon.sh              # Test execution
│   ├── validate.hexagon.sh          # Validation checks
│   ├── monitor.hexagon.sh           # Monitoring
│   ├── error-handler.hexagon.sh     # Error handling
│   └── performance-monitor.hexagon.sh # Performance monitoring
├── docs/
│   ├── hexagon-ci-guide.md          # This guide
│   └── troubleshooting.md           # Troubleshooting guide
└── cmake/
    └── HexagonToolchain.cmake       # CMake toolchain file
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HEXAGON_TOOLCHAIN_VERSION` | `17.0.0` | Hexagon toolchain version |
| `HEXAGON_CACHE_ENABLED` | `true` | Enable toolchain caching |
| `EIGEN_HEXAGON_JOBS` | `4` | Number of parallel build jobs |
| `EIGEN_HEXAGON_TIMEOUT` | `1800` | Build timeout in seconds |
| `EIGEN_QEMU_TIMEOUT` | `300` | QEMU test timeout in seconds |
| `EIGEN_CI_MAX_RETRIES` | `3` | Maximum retry attempts |
| `EIGEN_CI_RETRY_DELAY` | `10` | Delay between retries (seconds) |
| `EIGEN_ENABLE_PROFILING` | `false` | Enable performance profiling |
| `EIGEN_MONITOR_INTERVAL` | `5` | Monitoring sample interval (seconds) |

### Toolchain Configuration

The Hexagon toolchain can be configured through several methods:

1. **Automatic Download** (default):
```yaml
variables:
  HEXAGON_TOOLCHAIN_VERSION: "17.0.0"
  HEXAGON_TOOLCHAIN_URL: "https://developer.qualcomm.com/download/hexagon/..."
```

2. **Pre-installed Toolchain**:
```yaml
variables:
  HEXAGON_TOOLCHAIN_PATH: "/opt/hexagon-toolchain"
  HEXAGON_SKIP_DOWNLOAD: "true"
```

3. **Custom Mirror**:
```yaml
variables:
  HEXAGON_TOOLCHAIN_URL: "https://your-mirror.com/hexagon-toolchain.tar.xz"
```

### Build Configuration

Configure build options through CMake variables:

```yaml
variables:
  CMAKE_BUILD_TYPE: "Release"
  EIGEN_BUILD_TESTING: "ON"
  EIGEN_BUILD_PKGCONFIG: "ON"
  HEXAGON_ENABLE_HVX: "ON"
  HEXAGON_TARGET_ARCH: "v73"
```

## Build Variants

### Available Build Jobs

1. **hexagon_v68** - Hexagon v68 architecture
2. **hexagon_v73** - Hexagon v73 architecture (default)
3. **hexagon_hvx** - HVX (Hexagon Vector eXtensions) enabled
4. **hexagon_debug** - Debug build with symbols
5. **hexagon_optimized** - Optimized release build
6. **hexagon_minimal** - Minimal build (core only)
7. **hexagon_full** - Full build with all features

### Build Customization

You can customize builds by modifying the job definitions:

```yaml
hexagon_custom:
  extends: .hexagon_build_template
  variables:
    HEXAGON_TARGET_ARCH: "v68"
    CMAKE_BUILD_TYPE: "RelWithDebInfo"
    EIGEN_BUILD_TESTING: "OFF"
    HEXAGON_ENABLE_HVX: "OFF"
```

## Testing

### Test Categories

1. **Unit Tests** - Core Eigen functionality
2. **Performance Tests** - Benchmark critical operations
3. **Regression Tests** - Prevent regressions
4. **Integration Tests** - End-to-end scenarios

### QEMU Emulation

Tests run under QEMU Hexagon emulation:

```bash
# Manual test execution
./ci/scripts/test.hexagon.sh build/test_binary

# With custom timeout
EIGEN_QEMU_TIMEOUT=600 ./ci/scripts/test.hexagon.sh build/test_binary
```

### Test Configuration

```yaml
test_hexagon_custom:
  extends: .hexagon_test_template
  variables:
    EIGEN_QEMU_TIMEOUT: "600"
    EIGEN_TEST_CATEGORIES: "unit,performance"
    EIGEN_TEST_PARALLEL: "4"
```

## Performance Monitoring

### Monitoring Features

- **Resource Usage**: CPU, memory, disk I/O monitoring
- **Build Performance**: Compilation time tracking
- **Test Performance**: Execution time analysis
- **System Health**: Load average, process monitoring

### Enabling Monitoring

```yaml
variables:
  EIGEN_ENABLE_PROFILING: "true"
  EIGEN_MONITOR_INTERVAL: "5"
```

### Monitoring Reports

Generated reports:
- `performance-monitor.log` - Detailed performance log
- `performance-metrics.json` - Structured metrics data
- `optimization-report.txt` - Optimization recommendations

### Manual Monitoring

```bash
# Monitor build process
./ci/scripts/performance-monitor.hexagon.sh resource_monitor 300 .build

# Monitor specific process
./ci/scripts/performance-monitor.hexagon.sh process_monitor 300 .build ninja
```

## Error Handling

### Automatic Error Recovery

The pipeline includes comprehensive error handling:

- **Retry Mechanism**: Automatic retry with exponential backoff
- **Fallback Strategies**: Alternative download mirrors, extended timeouts
- **Resource Monitoring**: Early warning for resource exhaustion
- **Graceful Degradation**: Continue with partial functionality

### Error Categories

1. **TOOLCHAIN_DOWNLOAD** - Toolchain download failures
2. **TOOLCHAIN_SETUP** - Toolchain installation issues
3. **CMAKE_CONFIGURATION** - CMake configuration problems
4. **BUILD_FAILURE** - Compilation errors
5. **TEST_FAILURE** - Test execution failures
6. **QEMU_FAILURE** - QEMU emulation issues
7. **NETWORK_FAILURE** - Network connectivity problems
8. **RESOURCE_EXHAUSTION** - Insufficient resources
9. **TIMEOUT** - Operation timeouts

### Error Handling Configuration

```yaml
variables:
  EIGEN_CI_MAX_RETRIES: "3"
  EIGEN_CI_RETRY_DELAY: "10"
  EIGEN_CI_TIMEOUT: "1800"
```

## Troubleshooting

### Common Issues

#### 1. Toolchain Download Failure

**Symptoms**: Download timeouts, corrupted archives
**Solutions**:
- Check network connectivity
- Verify toolchain URL
- Use alternative mirrors
- Increase timeout values

```bash
# Verify connectivity
./ci/scripts/error-handler.hexagon.sh check_network

# Manual download test
wget --timeout=300 $HEXAGON_TOOLCHAIN_URL
```

#### 2. Build Failures

**Symptoms**: Compilation errors, missing dependencies
**Solutions**:
- Check toolchain installation
- Verify CMake configuration
- Review build logs
- Adjust parallel jobs

```bash
# Validate toolchain
./ci/scripts/validate.hexagon.sh

# Debug build
CMAKE_VERBOSE_MAKEFILE=ON ninja -v
```

#### 3. Test Failures

**Symptoms**: QEMU crashes, test timeouts
**Solutions**:
- Increase QEMU timeout
- Check binary compatibility
- Verify QEMU setup
- Run tests individually

```bash
# Test QEMU setup
qemu-hexagon --version

# Manual test run
./ci/scripts/test.hexagon.sh build/test_binary --verbose
```

#### 4. Resource Issues

**Symptoms**: Out of memory, disk space errors
**Solutions**:
- Reduce parallel jobs
- Increase runner resources
- Clean build artifacts
- Monitor resource usage

```bash
# Check resources
free -h
df -h
./ci/scripts/performance-monitor.hexagon.sh resource_monitor 60
```

### Debug Mode

Enable debug mode for detailed logging:

```yaml
variables:
  EIGEN_DEBUG_MODE: "true"
  EIGEN_VERBOSE_LOGGING: "true"
```

### Log Analysis

Important log files:
- `setup-hexagon.log` - Toolchain setup log
- `build-hexagon.log` - Build process log
- `test-hexagon.log` - Test execution log
- `error-handler.log` - Error handling log
- `performance-monitor.log` - Performance monitoring log

## Best Practices

### Performance Optimization

1. **Use Caching**:
```yaml
variables:
  HEXAGON_CACHE_ENABLED: "true"
cache:
  key: "hexagon-toolchain-v17.0.0"
  paths:
    - /opt/hexagon-toolchain/
```

2. **Optimize Parallel Jobs**:
```yaml
variables:
  EIGEN_HEXAGON_JOBS: "$(nproc)"  # Use all available cores
```

3. **Use Appropriate Timeouts**:
```yaml
variables:
  EIGEN_HEXAGON_TIMEOUT: "1800"  # 30 minutes for builds
  EIGEN_QEMU_TIMEOUT: "300"      # 5 minutes for tests
```

### Resource Management

1. **Monitor Resource Usage**:
```yaml
variables:
  EIGEN_ENABLE_PROFILING: "true"
```

2. **Set Resource Limits**:
```yaml
hexagon_build:
  resource_group: high_memory
  variables:
    EIGEN_HEXAGON_JOBS: "2"  # Reduce for memory-constrained runners
```

### Reliability

1. **Enable Error Handling**:
```yaml
variables:
  EIGEN_CI_MAX_RETRIES: "3"
  EIGEN_CI_RETRY_DELAY: "10"
```

2. **Use Health Checks**:
```yaml
before_script:
  - ./ci/scripts/validate.hexagon.sh
```

3. **Implement Monitoring**:
```yaml
after_script:
  - ./ci/scripts/performance-monitor.hexagon.sh full_build
```

### Security

1. **Verify Downloads**:
- Use checksums for toolchain verification
- Verify download sources
- Use secure download protocols

2. **Limit Permissions**:
- Run builds with minimal privileges
- Avoid running as root
- Use Docker security features

## Advanced Usage

### Custom Toolchain

To use a custom Hexagon toolchain:

1. **Upload toolchain to secure location**
2. **Configure custom URL**:
```yaml
variables:
  HEXAGON_TOOLCHAIN_URL: "https://your-secure-mirror.com/hexagon-toolchain.tar.xz"
  HEXAGON_TOOLCHAIN_CHECKSUM: "sha256:abc123..."
```

### Multi-Architecture Builds

Build for multiple Hexagon architectures:

```yaml
.hexagon_matrix:
  parallel:
    matrix:
      - HEXAGON_TARGET_ARCH: ["v68", "v73", "v75"]
        CMAKE_BUILD_TYPE: ["Release", "Debug"]
```

### Custom Test Suites

Create custom test configurations:

```yaml
test_hexagon_custom:
  extends: .hexagon_test_template
  script:
    - ./ci/scripts/test.hexagon.sh --category=performance --timeout=600
    - ./ci/scripts/test.hexagon.sh --category=regression --parallel=2
```

### Integration with External Tools

Integrate with external analysis tools:

```yaml
hexagon_analysis:
  extends: .hexagon_build_template
  script:
    - ./ci/scripts/setup.hexagon.sh
    - cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
    - clang-tidy -p . ../Eigen/**/*.h
    - cppcheck --enable=all ../Eigen/
```

## Conclusion

The Hexagon CI pipeline provides a robust, scalable solution for building and testing Eigen with Qualcomm Hexagon DSP toolchain. By following this guide and implementing the recommended best practices, you can achieve reliable, high-performance CI builds.

For additional support:
- Review the troubleshooting guide
- Check the error handling logs
- Consult the performance monitoring reports
- Reach out to the Eigen development team

## Appendix

### Useful Commands

```bash
# Setup commands
./ci/scripts/setup.hexagon.sh
./ci/scripts/validate.hexagon.sh

# Build commands
mkdir -p build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../ci/cmake/HexagonToolchain.cmake ..
ninja -j$(nproc)

# Test commands
./ci/scripts/test.hexagon.sh build/test_binary
./ci/scripts/test.hexagon.sh --category=unit --timeout=300

# Monitoring commands
./ci/scripts/performance-monitor.hexagon.sh resource_monitor 300
./ci/scripts/monitor.hexagon.sh

# Error handling
./ci/scripts/error-handler.hexagon.sh check_network
./ci/scripts/error-handler.hexagon.sh download_toolchain $URL $DIR $VERSION
```

### Reference Links

- [Qualcomm Hexagon SDK Documentation](https://developer.qualcomm.com/software/hexagon-dsp-sdk)
- [Eigen Documentation](https://eigen.tuxfamily.org/dox/)
- [GitLab CI/CD Documentation](https://docs.gitlab.com/ee/ci/)
- [QEMU Hexagon Documentation](https://qemu.readthedocs.io/en/latest/system/target-hexagon.html) 