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

If the runner has already cached these jobs locally, `restore_cache` still
extracts the stale pool into `${CI_PROJECT_DIR}/.ccache` on every job (and
`archive_cache` re-archives it whenever a job's primary key has no archive yet,
such as the first run under a new `-mr<iid>` key).  When switching an existing
runner to a host `CCACHE_DIR`, clear the runner's local cache storage once (the
`/cache` volumes, or `clear-docker-cache` for the Docker executor); once
`.ccache/` is absent, `cache-archiver` reports `No files to cache` and nothing
recreates the archive.  A runner with a distributed `[runners.cache]` backend
has no equivalent one-time clear—any other runner writing the same bucket
recreates the archive—so a host directory only fits cleanly on runners with a
local-only cache, where it also costs the shared pool nothing.

Concurrent jobs may share one local directory on a POSIX or NTFS filesystem,
but not over network shares.  The build scripts record per-job cache hits and
misses via `CCACHE_STATSLOG` and `ccache --show-log-stats` so that concurrent
jobs sharing a cache directory do not zero or mix each other's counters.

## Remote compiler cache

The Linux and Windows build jobs compile through sccache, which chains a local
disk directory with a Google Cloud Storage level that every runner reaches.
A job that may write stores each object in GCS as soon as it is compiled, so
the work of a job that dies on a lost VM or is canceled survives for its
retry. A GitLab
CI/CD variable holding a short-lived OAuth token selects the GCS area and its
mode; the first one set wins:

| Variable | Variable scope | GCS area | Mode |
|---|---|---|---|
| `EIGEN_GCS_CACHE_TOKEN_RW` | protected | shared | read-write |
| `EIGEN_GCS_CACHE_TOKEN_MR` | unprotected | merge request | read-write |
| `EIGEN_GCS_CACHE_TOKEN_RO` | unprotected | shared | read-only |

With none of them set, sccache uses the local directory alone. The shared area
is the bucket `EIGEN_CI_SCCACHE_GCS_BUCKET` (default `eigen-gitlab-ci-cache`).
The merge-request area is the bucket `EIGEN_CI_SCCACHE_GCS_MR_BUCKET` (default:
the shared bucket) under the key prefix `EIGEN_CI_SCCACHE_GCS_MR_KEY_PREFIX`
(default `mr`). Protected builds never read the merge-request area, so an
object written by a merge-request job cannot reach a master or scheduled build.
That guarantee rests on the credential, not on the scripts: the principal
behind `EIGEN_GCS_CACHE_TOKEN_MR` must be able to write the merge-request area
and nothing else, either as a separate bucket or through an IAM condition on
the object name, such as
`resource.name.startsWith("projects/_/buckets/<bucket>/objects/mr/")`. An
age-based lifecycle rule on that area bounds its size.

`SCCACHE_MULTILEVEL_CHAIN` names each backend once, so a job reads a single GCS
area: a merge-request job holding the read-write token reads the merge-request
area only, not the shared one.
