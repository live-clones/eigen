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

### MSVC coverage for merge requests

The nightly Windows jobs need the dedicated Windows runners, so merge requests
get their MSVC coverage from `build:windows:cross:...`, which cross-compiles the
smoke tests on a hosted Linux runner by running the real MSVC toolchain under
Wine, and `test:windows:cross:...`, which runs the resulting binaries on a
hosted Windows runner.  Both use hosted runners and pre-baked images, so they
also run in forks.

The toolchain is not part of the CI image (see
`ci/docker/ubuntu-24.04-amd64-msvc-wine-build/`): shipping it there would
redistribute it, which nobody has established the Visual Studio license terms
permit.  It is downloaded from Microsoft's installer manifests by
`ci/scripts/setup.msvc-wine.sh` on each run instead.  Nothing is licensed or
activated at run time: the compiler builds with no network access once
installed.
