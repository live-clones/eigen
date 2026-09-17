## Eigen CI infrastructure

Eigen's CI infrastructure uses three stages:
  1. A `checkformat` stage to verify MRs satisfy proper formatting style, as
     defined by `clang-format`.
  2. A `build` stage to build the unit-tests.
  3. A `test` stage to run the unit-tests.

For merge requests, only a small subset of tests are built/run, and only on a
small subset of platforms.  This is to reduce our overall testing infrastructure
resource usage.  In addition, a weekly scheduled pipeline builds and runs the
full suite of tests on most officially supported platforms.

## Persistent compiler cache

Self-hosted runners can configure a persistent host directory for ccache to
avoid the 5 GB GitLab archive limit and eliminate compression overhead.
Set the standard `CCACHE_*` environment variables in `/etc/gitlab-runner/config.toml`:

```toml
[[runners]]
  environment = [
    "CCACHE_DIR=/ccache",
    "CCACHE_MAXSIZE=50G",
  ]
  [runners.docker]
    volumes = ["/var/cache/eigen-ccache:/ccache:rw", "/cache"]
```

The YAML templates prefix their defaults with `EIGEN_CI_CCACHE_*`
(`EIGEN_CI_CCACHE_DIR`, `EIGEN_CI_CCACHE_MAXSIZE`, `EIGEN_CI_CCACHE_BASEDIR`,
`EIGEN_CI_CCACHE_COMPRESSLEVEL`) so that runner-level `environment = [...]`
settings in `config.toml` are not shadowed by GitLab CI.  When a runner sets
standard `CCACHE_DIR`, the build scripts (`build.linux.script.sh` and
`build.windows.script.ps1`) preserve the runner's value, leaving
`${CI_PROJECT_DIR}/.ccache` empty so GitLab's archive step is a no-op.
Concurrent jobs may share one local directory on a POSIX or NTFS filesystem,
but not over network shares.
