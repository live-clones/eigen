# Eigen CI Configuration for Hexagon DSP

This directory contains the CI configuration for building and testing Eigen with Hexagon DSP support using the open source Hexagon toolchain.

## Updated Architecture (Dedicated Hexagon Support)

The CI environment has been streamlined for dedicated Hexagon toolchain usage:

### Key Improvements

1. **Dedicated Hexagon Dockerfile**: Removed conditional `ENABLE_HEXAGON` checks since this is now a Hexagon-specific CI environment
2. **Direct Environment Variables**: Environment variables are set directly in the Dockerfile instead of using a runtime setup script
3. **Simplified CI Pipeline**: Removed dependency on `setup-hexagon-env.sh` script across all CI jobs

### Environment Variables Set in Dockerfile

- `HEXAGON_TOOLCHAIN_ROOT=/opt/hexagon-toolchain`
- `HEXAGON_TOOLCHAIN_BIN=/opt/hexagon-toolchain/bin`  
- `HEXAGON_SYSROOT=/opt/hexagon-toolchain/x86_64-linux-gnu/target/hexagon-unknown-linux-musl`
- `QEMU_LD_PREFIX=${HEXAGON_SYSROOT}`
- `EIGEN_CI_CROSS_C_COMPILER=${HEXAGON_TOOLCHAIN_BIN}/hexagon-unknown-linux-musl-clang`
- `EIGEN_CI_CROSS_CXX_COMPILER=${HEXAGON_TOOLCHAIN_BIN}/hexagon-unknown-linux-musl-clang++`

### Files

- `Dockerfile` - Main CI container with Hexagon toolchain
- `build.hexagon.gitlab-ci.yml` - Build jobs for different Hexagon configurations
- `test.hexagon.gitlab-ci.yml` - Test jobs with QEMU emulation support
- `test-hexagon-setup.sh` - Environment validation script

### Usage

The CI automatically builds Docker images and runs cross-compilation tests for Hexagon DSP targets including v68 and v73 architectures with HVX support.

## Eigen CI infrastructure

Eigen's CI infrastructure uses three stages:
  1. A `checkformat` stage to verify MRs satisfy proper formatting style, as
     defined by `clang-format`.
  2. A `build` stage to build the unit-tests.
  3. A `test` stage to run the unit-tests.

For merge requests, only a small subset of tests are built/run, and only on a
small subset of platforms.  This is to reduce our overall testing infrastructure
resource usage.  In addition, we have nightly jobs that build and run the full
suite of tests on most officially supported platforms.
