# Hexagon DSP CI/CD Guide

## Overview

This guide covers the Hexagon DSP cross-compilation CI/CD pipeline for Eigen. The pipeline uses a dedicated Docker-based approach that provides:

- **Dedicated Hexagon Docker images** with toolchain v20.1.4 pre-installed
- **Direct environment variables** set in Dockerfile (no runtime setup scripts)
- **Consistent environment** across local development and CI
- **Streamlined configuration** with minimal complexity

## Architecture

### Key Components

1. **Dedicated Docker Environment** (`ci/Dockerfile`)
   - Pre-installs Hexagon toolchain 20.1.4
   - Sets environment variables directly
   - Validates toolchain during build
   - No conditional logic or runtime setup scripts

2. **Cross-compilation Builds** (`build.hexagon.gitlab-ci.yml`)
   - Multiple Hexagon architectures (v68, v73)
   - Debug, release, and HVX-optimized variants
   - Automated ccache for faster builds

3. **QEMU-based Testing** (`test.hexagon.gitlab-ci.yml`)
   - Runs compiled tests under QEMU Hexagon emulation
   - Parallel test execution with configurable timeouts
   - Comprehensive test reporting and artifact collection

### File Structure

```
ci/
├── Dockerfile                       # Dedicated Hexagon CI environment
├── build.hexagon.gitlab-ci.yml      # Build job definitions  
├── test.hexagon.gitlab-ci.yml       # Test job definitions
├── test-hexagon-setup.sh            # Environment validation script
├── README.md                        # CI overview
└── docs/
    └── hexagon-ci-guide.md          # This guide
```

## Environment Configuration

### Pre-configured Environment Variables

The Dockerfile sets these environment variables directly:

```bash
HEXAGON_TOOLCHAIN_VERSION=20.1.4
HEXAGON_TOOLCHAIN_ROOT=/opt/hexagon-toolchain
HEXAGON_TOOLCHAIN_BIN=/opt/hexagon-toolchain/bin
HEXAGON_SYSROOT=/opt/hexagon-toolchain/x86_64-linux-gnu/target/hexagon-unknown-linux-musl
PATH="${HEXAGON_TOOLCHAIN_BIN}:${PATH}"
QEMU_LD_PREFIX="${HEXAGON_SYSROOT}"
EIGEN_CI_CROSS_C_COMPILER="${HEXAGON_TOOLCHAIN_BIN}/hexagon-unknown-linux-musl-clang"
EIGEN_CI_CROSS_CXX_COMPILER="${HEXAGON_TOOLCHAIN_BIN}/hexagon-unknown-linux-musl-clang++"
```

### CI Configuration Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EIGEN_CI_HEXAGON_ARCH` | `v68` | Target Hexagon architecture |
| `EIGEN_CI_BUILDDIR` | `.build` | Build directory name |
| `EIGEN_CI_TEST_TIMEOUT` | `300` | Test timeout in seconds |
| `EIGEN_CI_MAX_PARALLEL_TESTS` | `4` | Number of parallel test jobs |
| `EIGEN_CI_BUILD_TARGET` | `buildtests` | CMake target to build |

## Build Variants

### Available Build Jobs

1. **build:linux:hexagon:v68:ci** - Standard v68 release build
2. **build:linux:hexagon:v68:hvx:ci** - v68 with HVX enabled
3. **build:linux:hexagon:v68:debug:ci** - Debug build
4. **build:linux:hexagon:v68:optimized:ci** - Optimized for performance
5. **build:linux:hexagon:v73:ci** - Hexagon v73 architecture
6. **build:linux:hexagon:minimal:ci** - Minimal build for quick validation

### Build Configuration Examples

Extend base configuration for custom builds:

```yaml
build:custom:hexagon:
  extends: .build:linux:hexagon:ci
  variables:
    EIGEN_CI_HEXAGON_ARCH: v68
    EIGEN_CI_BUILDDIR: .build-custom
    EIGEN_CI_ADDITIONAL_ARGS: "-DCMAKE_BUILD_TYPE=RelWithDebInfo -DEIGEN_TEST_CUSTOM=ON"
```

## Local Development with Docker

### Quick Start

Build and use the Hexagon CI environment locally:

```bash
# Navigate to Eigen repository
cd /path/to/eigen-mirror

# Build the Docker image
docker build -f ci/Dockerfile -t eigen-hexagon:local .

# Run interactive container with workspace mounted
docker run -it --rm -v $(pwd):/workspace eigen-hexagon:local bash
```

### Building Eigen with CMake (Inside Container)

Once inside the Docker container, you can build Eigen using the pre-configured environment:

```bash
# Verify environment is ready
echo "🔧 Hexagon Environment Information"
echo "Toolchain: ${HEXAGON_TOOLCHAIN_ROOT}"
echo "Compilers: ${EIGEN_CI_CROSS_C_COMPILER}"
echo "Sysroot: ${HEXAGON_SYSROOT}"

# Verify toolchain works
${EIGEN_CI_CROSS_C_COMPILER} --version
${EIGEN_CI_CROSS_CXX_COMPILER} --version

# Create build directory
mkdir -p .build-hexagon && cd .build-hexagon

# Configure with CMake using Hexagon toolchain
cmake \
  -DCMAKE_TOOLCHAIN_FILE=${PWD}/../cmake/HexagonToolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DEIGEN_TEST_HEXAGON=ON \
  -DBUILD_TESTING=ON \
  -GNinja \
  ..

# Build the tests
ninja buildtests

# Or build specific targets
ninja eigen_buildtests_packetmath
ninja eigen_buildtests_basicstuff
```

### Building with HVX Support

For HVX-enabled builds:

```bash
# Configure with HVX support
cmake \
  -DCMAKE_TOOLCHAIN_FILE=${PWD}/../cmake/HexagonToolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DEIGEN_TEST_HEXAGON=ON \
  -DEIGEN_TEST_HVX=ON \
  -DBUILD_TESTING=ON \
  -GNinja \
  ..

# Build HVX tests
ninja buildtests
```

### Testing with QEMU

Run tests using QEMU emulation:

```bash
# Run all tests
ctest --output-on-failure --timeout 300 -j4

# Run specific test categories
ctest --output-on-failure -R "packetmath|hvx"

# Run tests with verbose output
ctest --output-on-failure --verbose -R "basicstuff"

# Run performance tests (if available)
ctest --output-on-failure -L "Performance"
```

### Development Workflow Example

Complete development workflow in the Docker container:

```bash
# 1. Start container with workspace mounted
docker run -it --rm -v $(pwd):/workspace eigen-hexagon:local bash

# 2. Create and configure build directory
mkdir -p .build-dev && cd .build-dev
cmake \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/HexagonToolchain.cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -DEIGEN_TEST_HEXAGON=ON \
  -DBUILD_TESTING=ON \
  -GNinja \
  ..

# 3. Build incrementally during development
ninja eigen_buildtests_packetmath

# 4. Test specific functionality
ctest --output-on-failure -R "packetmath" --verbose

# 5. Build optimized version for performance testing
cd .. && mkdir -p .build-optimized && cd .build-optimized
cmake \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/HexagonToolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
  -DEIGEN_TEST_HEXAGON=ON \
  -DEIGEN_TEST_HVX=ON \
  -DBUILD_TESTING=ON \
  -GNinja \
  ..

ninja buildtests
ctest --output-on-failure --timeout 600
```

## Testing

### Test Categories

1. **Smoke Tests** (`test:linux:hexagon:smoke`) - Quick validation (basic tests only)
2. **Standard Tests** (`test:linux:hexagon:v68:default`) - Core functionality  
3. **HVX Tests** (`test:linux:hexagon:v68:hvx`) - Vector operations
4. **Debug Tests** (`test:linux:hexagon:v68:debug`) - Debug build validation
5. **Performance Tests** (`test:linux:hexagon:performance`) - Benchmarks
6. **Integration Tests** (`test:linux:hexagon:integration`) - End-to-end validation

### QEMU Configuration

Tests run under QEMU Hexagon emulation with pre-configured environment:

```bash
# Environment variables are already set in Docker image
echo "Using QEMU with sysroot: ${QEMU_LD_PREFIX}"

# Validation ensures QEMU is working
/workspace/ci/test-hexagon-setup.sh

# Tests run with appropriate timeouts
ctest --output-on-failure --timeout ${EIGEN_CI_TEST_TIMEOUT} -j${EIGEN_CI_MAX_PARALLEL_TESTS}
```

### Custom Test Configuration

```yaml
test:custom:hexagon:
  extends: .test:linux:hexagon
  variables:
    EIGEN_CI_TEST_TIMEOUT: "600"        # Longer timeout
    EIGEN_CI_MAX_PARALLEL_TESTS: "2"    # Fewer parallel jobs
    EIGEN_CI_CTEST_REGEX: "matrix|array" # Run specific tests
  script:
    - cd ${EIGEN_CI_BUILDDIR:-.build}
    - ctest --output-on-failure --timeout ${EIGEN_CI_TEST_TIMEOUT} -j${EIGEN_CI_MAX_PARALLEL_TESTS} -R "${EIGEN_CI_CTEST_REGEX}"
```

## CMake Integration

### HexagonToolchain.cmake

The CMake toolchain file (`cmake/HexagonToolchain.cmake`) automatically configures:

- Cross-compilation settings for Hexagon target
- Compiler and linker flags
- Sysroot configuration
- HVX-specific optimizations (when enabled)

### CMake Usage Examples

```bash
# Basic configuration
cmake -DCMAKE_TOOLCHAIN_FILE=cmake/HexagonToolchain.cmake ..

# With specific build type
cmake \
  -DCMAKE_TOOLCHAIN_FILE=cmake/HexagonToolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  ..

# With testing enabled
cmake \
  -DCMAKE_TOOLCHAIN_FILE=cmake/HexagonToolchain.cmake \
  -DBUILD_TESTING=ON \
  -DEIGEN_TEST_HEXAGON=ON \
  ..

# With HVX support
cmake \
  -DCMAKE_TOOLCHAIN_FILE=cmake/HexagonToolchain.cmake \
  -DEIGEN_TEST_HVX=ON \
  ..

# Custom compiler flags
cmake \
  -DCMAKE_TOOLCHAIN_FILE=cmake/HexagonToolchain.cmake \
  -DCMAKE_CXX_FLAGS="-O3 -march=hexagon -mcpu=hexagonv68" \
  ..
```

## Performance Monitoring

### Built-in Monitoring

The CI pipeline includes:

- **Build Time Tracking** - Ninja build progress and timing
- **Test Execution Metrics** - CTest timing and pass/fail rates
- **Artifact Size Tracking** - Binary sizes and test coverage
- **Resource Usage** - Docker container resource consumption

### Monitoring Configuration

```yaml
variables:
  EIGEN_CI_ENABLE_MONITORING: "true"
  EIGEN_CI_ENABLE_PERFORMANCE: "true"
  EIGEN_CI_ENABLE_VALIDATION: "true"
```

## Error Handling & Troubleshooting

### Environment Validation

Use the validation script to verify the Hexagon environment:

```bash
# Inside Docker container
/workspace/ci/test-hexagon-setup.sh

# Local validation (if container is built)
docker run --rm eigen-hexagon:local /workspace/ci/test-hexagon-setup.sh
```

### Common Issues and Solutions

1. **Docker Build Failures**
   ```bash
   # Check Dockerfile syntax
   docker build --no-cache -f ci/Dockerfile .
   
   # Verify toolchain download
   wget --timeout=60 --tries=3 https://artifacts.codelinaro.org/artifactory/codelinaro-toolchain-for-hexagon/20.1.4/clang+llvm-20.1.4-cross-hexagon-unknown-linux-musl.tar.zst
   ```

2. **Cross-compilation Issues**
   ```bash
   # Verify environment
   echo $HEXAGON_TOOLCHAIN_ROOT
   echo $EIGEN_CI_CROSS_C_COMPILER
   ls -la $EIGEN_CI_CROSS_C_COMPILER
   
   # Test compiler
   ${EIGEN_CI_CROSS_C_COMPILER} --version
   ```

3. **QEMU Test Failures**
   ```bash
   # Check QEMU setup
   echo $QEMU_LD_PREFIX
   ls -la ${QEMU_LD_PREFIX}/lib
   
   # Test simple binary
   echo 'int main(){return 0;}' | ${EIGEN_CI_CROSS_C_COMPILER} -x c - -o test
   qemu-hexagon test
   ```

4. **CMake Configuration Issues**
   ```bash
   # Verify toolchain file
   ls -la cmake/HexagonToolchain.cmake
   
   # Test basic configuration
   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/HexagonToolchain.cmake --debug-output .
   ```

### Debugging Tips

```bash
# Check build logs in detail
ninja -v

# Run specific test with maximum verbosity
ctest --output-on-failure --verbose -R "specific_test_name"

# Check environment variables
env | grep HEXAGON
env | grep EIGEN_CI

# Verify binary architecture
file path/to/test_binary
hexdump -C path/to/test_binary | head
```

## Quick Reference

### Essential Commands

```bash
# Build Docker image
docker build -f ci/Dockerfile -t eigen-hexagon:local .

# Start development container
docker run -it --rm -v $(pwd):/workspace eigen-hexagon:local bash

# Configure and build (inside container)
mkdir .build && cd .build
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/HexagonToolchain.cmake ..
ninja buildtests

# Run tests
ctest --output-on-failure --timeout 300 -j4

# Validate environment
/workspace/ci/test-hexagon-setup.sh
```

### Key Build Targets

```bash
ninja buildtests              # Build all test binaries
ninja eigen_buildtests_*      # Build specific test categories
ninja install                 # Install Eigen headers
```

### Useful CTest Commands

```bash
ctest --output-on-failure                    # Run all tests with output on failure
ctest -R "packetmath|basicstuff"            # Run specific test patterns
ctest -L "Hexagon"                           # Run tests with specific label
ctest --timeout 600 -j2                     # Custom timeout and parallelism
ctest --verbose -R "test_name"               # Verbose output for specific test
```

### Key Files

- `ci/Dockerfile` - Dedicated Hexagon CI environment
- `ci/build.hexagon.gitlab-ci.yml` - Build job definitions
- `ci/test.hexagon.gitlab-ci.yml` - Test job definitions
- `ci/test-hexagon-setup.sh` - Environment validation script
- `cmake/HexagonToolchain.cmake` - CMake cross-compilation setup

### Reference Links

- [Codelinaro Hexagon Toolchain](https://github.com/quic/toolchain_for_hexagon)
- [Eigen Documentation](https://eigen.tuxfamily.org/dox/)
- [GitLab CI/CD Documentation](https://docs.gitlab.com/ee/ci/)
- [Docker Documentation](https://docs.docker.com/) 