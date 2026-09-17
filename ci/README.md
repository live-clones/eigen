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
Set the standard `CCACHE_*` environment variables in the runner's `config.toml`:

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
`${CI_PROJECT_DIR}/.ccache` absent so GitLab's `restore_cache` and
`archive_cache` steps are no-ops.

If the runner has already cached these jobs locally, `restore_cache` will still
extract the existing archive into `${CI_PROJECT_DIR}/.ccache` before the script
runs, and `archive_cache` (`when: always`) will re-archive those untouched
files on every job.  When switching an existing runner to a host `CCACHE_DIR`,
clear the runner's local cache storage once (the `/cache` volumes, or
`clear-docker-cache` for the Docker executor); once `.ccache/` is absent,
`cache-archiver` reports `No files to cache` and nothing recreates the archive.

Because a runner with a host `CCACHE_DIR` writes nothing to the GitLab cache, it
does not refresh the unscoped `<slug>-ccache` pools that first MR builds on
hosted runners fall back to (which are updated only by the weekly scheduled
default-branch builds, as `master` push pipelines build documentation only).
Runners taking the scheduled full builds (`saas-linux-2xlarge-amd64`) should
therefore stay on the GitLab cache unless another mechanism refreshes those
shared pools.

Concurrent jobs may share one local directory on a POSIX or NTFS filesystem,
but not over network shares.  The build scripts record per-job cache hits and
misses via `CCACHE_STATSLOG` and `ccache --show-log-stats` so that concurrent
jobs sharing a cache directory do not zero or mix each other's counters.
