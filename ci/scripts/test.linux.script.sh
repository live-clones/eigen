#!/bin/bash
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

set -x

rootdir=`pwd`
cd ${EIGEN_CI_BUILDDIR}

target=""
if [[ ${EIGEN_CI_CTEST_REGEX} ]]; then
  target="-R ${EIGEN_CI_CTEST_REGEX}"
elif [[ ${EIGEN_CI_CTEST_LABEL} ]]; then
  target="-L ${EIGEN_CI_CTEST_LABEL}"
fi

exclude=""
if [[ -n "${EIGEN_CI_CTEST_EXCLUDE}" ]]; then
  exclude="-E ${EIGEN_CI_CTEST_EXCLUDE}"
fi

set +x

# Content-addressed pass cache (see ci/scripts/test_cache.py): skip tests
# whose executable, emulator, and image fingerprint match a first-attempt
# pass recorded by an earlier run of this job.  Merge-request pipelines
# only: scheduled and web runs keep re-running identical binaries so the
# clock-seeded RNG keeps exploring fresh seeds.  Sharded jobs must not
# skip -- dropping tests from the filtered list would shift the
# `-I index,,total` partition and could leave tests unrun in every shard.
testcache_dir="${rootdir}/.testcache"
# Scratch files live under Testing/Temporary, which the artifact excludes
# already filter out of the test job's build-directory upload.
testcache_tmp="Testing/Temporary"
testcache_active=false
if [[ "${EIGEN_CI_TEST_CACHE:-on}" == "on" \
      && "${CI_PIPELINE_SOURCE:-}" == "merge_request_event" \
      && "${CI_NODE_TOTAL:-1}" -le 1 ]] && command -v python3 >/dev/null 2>&1; then
  testcache_active=true
  mkdir -p "${testcache_dir}" "${testcache_tmp}"
  # The fingerprint must cover everything that can change a test's outcome
  # while its binary stays identical: the image and its shared libraries
  # (every lib* package is a superset of anything a test can dynamically
  # load, cross sysroots included), the checked-in CI configuration (where
  # QEMU_CPU, EIGEN_REPEAT, sanitizer options and the like are set), and
  # the same variables read directly in case a pipeline sets them outside
  # the tree.  Job-to-job differences are already isolated by the
  # per-job-name cache key; this guards against the same job's environment
  # changing over time.  Each part degrades to empty rather than failing
  # the job.
  cfg_fp=$( (cd "${rootdir}" && find ci .gitlab-ci.yml -type f | LC_ALL=C sort \
             | xargs sha256sum | sha256sum | cut -d' ' -f1) 2>/dev/null || true)
  pkg_fp=$(dpkg-query -W 'lib*' 2>/dev/null | sha256sum | cut -d' ' -f1 || true)
  env_fp=$(env 2>/dev/null | grep -E '^(EIGEN_REPEAT=|EIGEN_SEED=|QEMU_|ASAN_|UBSAN_|LSAN_|TSAN_|MSAN_|LD_LIBRARY_PATH=|LD_PRELOAD=|EIGEN_CI_CTEST_ARGS=)' \
           | LC_ALL=C sort | tr '\n' ',' || true)
  fingerprint="${CI_JOB_IMAGE:-}|$(ldd --version 2>/dev/null | head -n 1 || true)|${cfg_fp}|${pkg_fp}|${env_fp}"
  ctest --show-only=json-v1 ${target} ${exclude} > "${testcache_tmp}/testcache_tests.json" \
    && python3 "${rootdir}/ci/scripts/test_cache.py" plan \
         --tests-json "${testcache_tmp}/testcache_tests.json" \
         --manifest "${testcache_dir}/passed.txt" \
         --fingerprint "${fingerprint}" \
         --skip-out "${testcache_tmp}/testcache_skip.txt" \
         --keys-out "${testcache_tmp}/testcache_keys.txt" \
    || testcache_active=false
  if [[ "${testcache_active}" == "true" && -s "${testcache_tmp}/testcache_skip.txt" ]]; then
    skip_regex="^($(paste -sd'|' "${testcache_tmp}/testcache_skip.txt"))$"
    if [[ -n "${EIGEN_CI_CTEST_EXCLUDE}" ]]; then
      exclude="-E (${EIGEN_CI_CTEST_EXCLUDE})|${skip_regex}"
    else
      exclude="-E ${skip_regex}"
    fi
  fi
fi

EIGEN_CI_CTEST_PARALLEL=${EIGEN_CI_CTEST_PARALLEL:-${NPROC}}
# Total attempts for flaky tests (passed to ctest --repeat until-pass:N).
EIGEN_CI_CTEST_REPEAT=${EIGEN_CI_CTEST_REPEAT:-3}
# Per-test timeout for the retry phase. Retries exist to absorb seed-dependent
# flakes, which pass quickly when they pass; a test that hit the initial
# per-test timeout can never pass a full-length retry and would only burn up
# to EIGEN_CI_CTEST_REPEAT more timeouts (this pushed the qemu-emulated jobs
# past their job caps). A later --timeout on the ctest command line overrides
# an earlier one from EIGEN_CI_CTEST_ARGS.
EIGEN_CI_CTEST_RETRY_TIMEOUT=${EIGEN_CI_CTEST_RETRY_TIMEOUT:-600}
ctest_cmd="ctest ${EIGEN_CI_CTEST_ARGS} --parallel ${EIGEN_CI_CTEST_PARALLEL} --output-on-failure --no-compress-output --build-noclean ${target} ${exclude}"

echo "Running initial tests..."
# The job sources this script and GitLab Runner runs it under errexit, so
# a bare failing ctest would abort the job before the retry logic; capture
# the status through || instead.
initial_exit=0
${ctest_cmd} -T test || initial_exit=$?

# Fold first-attempt results into the pass cache.  The dashboard run's
# Test.xml is the authoritative status source: only "passed" is recorded,
# so failures, timeouts and SKIP_RETURN_CODE skips ("failed"/"notrun")
# never enter the manifest, and a ctest that died without writing results
# records nothing.  Passes obtained in the retry phase below are
# deliberately not recorded: a seed-flaky test keeps re-running.
if [[ "${testcache_active}" == "true" ]]; then
  tag=$(head -n 1 Testing/TAG 2>/dev/null || true)
  python3 "${rootdir}/ci/scripts/test_cache.py" record \
      --manifest "${testcache_dir}/passed.txt" \
      --keys "${testcache_tmp}/testcache_keys.txt" \
      --skip "${testcache_tmp}/testcache_skip.txt" \
      --test-xml "Testing/${tag}/Test.xml" || true
fi

if [[ ${initial_exit} -eq 0 ]]; then
  echo "Tests passed on the first attempt."
  exit_code=0
else
  echo "Initial tests failed with exit code ${initial_exit}. Retrying up to ${EIGEN_CI_CTEST_REPEAT} times..."
  if ${ctest_cmd} --rerun-failed --repeat until-pass:${EIGEN_CI_CTEST_REPEAT} --timeout ${EIGEN_CI_CTEST_RETRY_TIMEOUT}; then
    echo "Tests passed on retry."
    # 42 = passed-on-retry; .test:linux / .test:windows whitelist it via
    # allow_failure.exit_codes so the job is marked as a soft warning.
    exit_code=42
  else
    exit_code=$?
  fi
fi

set -x

cd ${rootdir}

set +x

exit $exit_code
