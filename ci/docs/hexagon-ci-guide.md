# Hexagon DSP CI/CD Guide (Updated for Dockerfile-based CI)

## Overview

This guide covers the Hexagon DSP cross-compilation CI/CD pipeline for Eigen. The pipeline has been modernized to use a Dockerfile-based approach that provides:

- **Pre-built Docker images** with Hexagon toolchain pre-installed
- **Simplified CI configuration** with minimal scripting
- **Consistent environment** across local development and CI
- **Faster build times** with cached toolchain setup

## Key Components

### Core Pipeline Jobs

1. **Docker Image Build** (`build:docker:hexagon`)
   - Builds CI Docker image with Hexagon toolchain 20.1.4
   - Pre-installs all dependencies and configures environment
   - Pushes to GitLab container registry

2. **Cross-compilation Builds** (`build:linux:hexagon:*:ci`)
   - Uses pre-built Docker image for consistent environment
   - Supports multiple Hexagon architectures (v68, v73)
   - Includes debug, optimized, and HVX variants

3. **QEMU-based Testing** (`test:linux:hexagon:*`)
   - Runs compiled tests under QEMU Hexagon emulation
   - Supports parallel test execution
   - Generates comprehensive test reports

### File Structure (Updated)

```
ci/
├── Dockerfile                       # CI environment definition
├── build.hexagon.gitlab-ci.yml      # Build job definitions  
├── test.hexagon.gitlab-ci.yml       # Test job definitions
├── test-hexagon-setup.sh            # CI environment validation script
├── scripts/
│   ├── vars.linux.sh               # Linux environment variables
│   ├── common.linux.before_script.sh # Common Linux setup
│   ├── test.linux.script.sh        # Generic Linux test script
│   ├── build.linux.script.sh       # Generic Linux build script
│   └── backup-*/                   # Backup of removed scripts
├── docs/
│   ├── hexagon-ci-guide.md         # This guide
│   └── troubleshooting.md          # Troubleshooting guide
└── cmake/
    └── HexagonToolchain.cmake      # CMake toolchain file
```

**Note**: For CI configuration validation, use `eigen-tools/test_ci_tools.py` which provides comprehensive GitLab CI analysis and validation.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EIGEN_CI_HEXAGON_TOOLCHAIN_VERSION` | `20.1.4` | Hexagon toolchain version |
| `EIGEN_CI_HEXAGON_ARCH` | `v68` | Target Hexagon architecture |
| `EIGEN_CI_BUILDDIR` | `.build` | Build directory name |
| `EIGEN_CI_TEST_TIMEOUT` | `300` | Test timeout in seconds |
| `EIGEN_CI_MAX_PARALLEL_TESTS` | `4` | Number of parallel test jobs |
| `HEXAGON_SYSROOT` | Auto-detected | Hexagon sysroot path |

### Docker Configuration

The Dockerfile automatically configures:

```dockerfile
# Pre-installs Hexagon toolchain 20.1.4
ARG HEXAGON_TOOLCHAIN_VERSION=20.1.4
# Sets up environment variables
ENV HEXAGON_TOOLCHAIN_ROOT=/opt/hexagon-toolchain
ENV HEXAGON_SYSROOT=/opt/hexagon-toolchain/x86_64-linux-gnu/target/hexagon-unknown-linux-musl
# Adds setup script
COPY setup-hexagon-env.sh /usr/local/bin/
```

### Build Configuration

Configure builds through GitLab CI variables:

```yaml
variables:
  EIGEN_CI_HEXAGON_ARCH: "v68"  # or "v73"
  EIGEN_CI_BUILDDIR: ".build-hexagon-v68"
  EIGEN_CI_ADDITIONAL_ARGS: "-DCMAKE_BUILD_TYPE=Release -DEIGEN_TEST_HVX=ON"
```

## Build Variants

### Available Build Jobs

1. **build:linux:hexagon:v68:ci** - Standard v68 build
2. **build:linux:hexagon:v68:hvx:ci** - v68 with HVX enabled
3. **build:linux:hexagon:v68:debug:ci** - Debug build
4. **build:linux:hexagon:v68:optimized:ci** - Optimized for performance
5. **build:linux:hexagon:v73:ci** - Hexagon v73 architecture
6. **build:linux:hexagon:minimal:ci** - Minimal build for quick validation

### Build Customization

Extend base configuration for custom builds:

```yaml
build:custom:hexagon:
  extends: .build:linux:hexagon:ci
  variables:
    EIGEN_CI_HEXAGON_ARCH: v68
    EIGEN_CI_BUILDDIR: .build-custom
    EIGEN_CI_ADDITIONAL_ARGS: "-DCMAKE_BUILD_TYPE=RelWithDebInfo -DEIGEN_TEST_CUSTOM=ON"
```

## Testing

### Test Categories

1. **Smoke Tests** (`test:linux:hexagon:smoke`) - Quick validation
2. **Standard Tests** (`test:linux:hexagon:v68:default`) - Core functionality  
3. **HVX Tests** (`test:linux:hexagon:v68:hvx`) - Vector operations
4. **Performance Tests** (`test:linux:hexagon:performance`) - Benchmarks
5. **Integration Tests** (`test:linux:hexagon:integration`) - End-to-end validation

### QEMU Emulation

Tests run under QEMU Hexagon emulation using the pre-configured environment:

```bash
# Environment is automatically set up in Docker image
source /usr/local/bin/setup-hexagon-env.sh

# Validation script ensures everything works
/workspace/eigen-mirror/ci/test-hexagon-setup.sh

# Tests run with ctest
cd ${EIGEN_CI_BUILDDIR}
ctest --output-on-failure --timeout ${EIGEN_CI_TEST_TIMEOUT} -j${EIGEN_CI_MAX_PARALLEL_TESTS}
```

### Test Configuration

Customize test execution:

```yaml
test:custom:hexagon:
  extends: .test:linux:hexagon
  variables:
    EIGEN_CI_TEST_TIMEOUT: "600"
    EIGEN_CI_MAX_PARALLEL_TESTS: "2"
    EIGEN_CI_CTEST_REGEX: "matrix|array"  # Run specific tests
```

## Local Development

### Using the Docker Environment

Build and run the CI environment locally:

```bash
# Build Docker image
docker build -f ci/Dockerfile -t eigen-hexagon:local .

# Run interactive container
docker run -it --rm -v $(pwd):/workspace/eigen-mirror eigen-hexagon:local bash

# Inside container - validate setup
/workspace/eigen-mirror/ci/test-hexagon-setup.sh

# Build Eigen
source /usr/local/bin/setup-hexagon-env.sh
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/HexagonToolchain.cmake ..
ninja
```

### Manual Validation

Use the validation script to check your Hexagon environment:

```bash
# Hexagon environment validation (used in CI)
./ci/test-hexagon-setup.sh

# CI configuration validation (use eigen-tools)
cd ../eigen-tools && python3 test_ci_tools.py
```

## Performance Monitoring

### Built-in Monitoring

The pipeline includes automatic monitoring:

- **Build Time Tracking** - Ninja build progress
- **Test Execution Metrics** - ctest timing and results  
- **Resource Usage** - Docker container metrics
- **Artifact Size Tracking** - Build output sizes

### Monitoring Configuration

```yaml
variables:
  EIGEN_CI_ENABLE_MONITORING: "true"
  EIGEN_CI_ENABLE_PERFORMANCE: "true"
  EIGEN_CI_ENABLE_VALIDATION: "true"
```

## Error Handling

### Automatic Error Recovery

The modernized pipeline provides:

- **Pre-validated Environment** - Docker image ensures consistent setup
- **Simplified Error Paths** - Fewer failure points with pre-built toolchain
- **Clear Error Messages** - Validation scripts provide specific guidance
- **Fast Feedback** - Smoke tests catch issues early

### Common Issues and Solutions

1. **Docker Build Failures**
   - Check Dockerfile syntax
   - Verify toolchain download URL
   - Ensure sufficient disk space

2. **Cross-compilation Issues**  
   - Run validation script: `./ci/test-hexagon-setup.sh`
   - Check CMake toolchain file
   - Verify sysroot configuration

3. **QEMU Test Failures**
   - Check QEMU version compatibility
   - Verify binary file format
   - Increase test timeout if needed

### Error Handling Configuration

```yaml
variables:
  EIGEN_CI_TEST_TIMEOUT: "300"        # Increase for slow tests
  EIGEN_CI_MAX_PARALLEL_TESTS: "2"    # Reduce if resource-constrained
```

## Migration from Legacy Scripts

### Removed Components

The following scripts have been **deprecated and removed** (backed up in `ci/scripts/backup-*/`):

- ❌ `setup.hexagon.sh` → ✅ Pre-configured in Dockerfile
- ❌ `validate.hexagon.sh` → ✅ Replaced by `test-hexagon-setup.sh`  
- ❌ `test.hexagon.sh` → ✅ Direct `ctest` execution
- ❌ `monitor.hexagon.sh` → ✅ GitLab CI built-in monitoring
- ❌ `performance-monitor.hexagon.sh` → ✅ Simplified metrics collection
- ❌ `error-handler.hexagon.sh` → ✅ Simplified error handling
- ❌ `alert.hexagon.sh` → ✅ GitLab CI notifications

### Migration Guide

If you have scripts that reference the old components:

1. **Environment Setup**: Use `source /usr/local/bin/setup-hexagon-env.sh`
2. **Hexagon Validation**: Use `/workspace/eigen-mirror/ci/test-hexagon-setup.sh` (CI) or `./ci/test-hexagon-setup.sh` (local)
3. **CI Configuration Validation**: Use `eigen-tools/test_ci_tools.py`
4. **Testing**: Use direct `ctest` commands
5. **Monitoring**: Use GitLab CI built-in metrics and artifacts

## Troubleshooting

### Quick Diagnostics

```bash
# Validate Hexagon environment
./ci/test-hexagon-setup.sh

# Validate CI configuration (use eigen-tools)
cd ../eigen-tools && python3 test_ci_tools.py

# Test toolchain in Docker
docker run --rm eigen-hexagon:local /workspace/eigen-mirror/ci/test-hexagon-setup.sh

# Check Docker image
docker run --rm eigen-hexagon:latest hexagon-unknown-linux-musl-clang++ --version
```

## Conclusion

The modernized Hexagon CI pipeline provides a robust, scalable solution for building and testing Eigen with Qualcomm Hexagon DSP toolchain using Docker containers. The new approach offers:

- **Faster Setup** - Pre-built Docker images eliminate toolchain download time
- **Consistent Environment** - Same toolchain version across all builds
- **Simplified Maintenance** - Fewer scripts and configuration files to manage
- **Better Reliability** - Pre-validated environment reduces failure points

For additional support:
- Review the CI configuration files
- Use the validation scripts
- Check GitLab CI pipeline logs
- Consult the Eigen development team

## Quick Reference

### Essential Commands

```bash
# Local Development
docker build -f ci/Dockerfile -t eigen-hexagon:local .
docker run -it --rm -v $(pwd):/workspace/eigen-mirror eigen-hexagon:local bash

# Environment Setup (inside container)
source /usr/local/bin/setup-hexagon-env.sh
/workspace/eigen-mirror/ci/test-hexagon-setup.sh

# Build Commands  
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/HexagonToolchain.cmake ..
ninja

# Testing
ctest --output-on-failure --timeout 300 -j4

# CI Configuration Validation (use eigen-tools)
cd ../eigen-tools && python3 test_ci_tools.py
```

### Key Files

- `ci/Dockerfile` - CI environment definition
- `ci/build.hexagon.gitlab-ci.yml` - Build jobs
- `ci/test.hexagon.gitlab-ci.yml` - Test jobs  
- `ci/test-hexagon-setup.sh` - Hexagon environment validation (used in CI)
- `cmake/HexagonToolchain.cmake` - CMake cross-compilation setup
- `../eigen-tools/test_ci_tools.py` - CI configuration validation

### Reference Links

- [Codelinaro Hexagon Toolchain](https://github.com/quic/toolchain_for_hexagon)
- [Eigen Documentation](https://eigen.tuxfamily.org/dox/)
- [GitLab CI/CD Documentation](https://docs.gitlab.com/ee/ci/)
- [Docker Documentation](https://docs.docker.com/) 