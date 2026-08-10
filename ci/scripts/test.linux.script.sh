#!/bin/bash
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

set -x

rootdir=`pwd`

# The affected-tests tier (see scripts/affected_tests.py) passes its CTest
# filter as a file rather than a variable so the regex is not bounded by CI
# variable limits.  "ALL" means run everything the paired build produced,
# "NONE" means the merge request affects no test at all.
if [[ -n "${EIGEN_CI_CTEST_REGEX_FILE}" ]]; then
  regex_file="${EIGEN_CI_CTEST_REGEX_FILE}"
  [[ "${regex_file}" = /* ]] || regex_file="${rootdir}/${regex_file}"
  # Fail loudly rather than falling through: a missing selection would
  # otherwise silently run the whole suite against a partial build.
  if [[ ! -f "${regex_file}" ]]; then
    echo "EIGEN_CI_CTEST_REGEX_FILE=${EIGEN_CI_CTEST_REGEX_FILE} does not exist." >&2
    echo "The select:tests artifact is missing; refusing to guess a test filter." >&2
    exit 1
  fi
  selection=$(cat "${regex_file}")
  case "${selection}" in
    NONE)
      echo "No tests are affected by this merge request; nothing to run."
      set +x
      return 0 2>/dev/null || exit 0
      ;;
    ALL)
      EIGEN_CI_CTEST_REGEX=""
      ;;
    *)
      EIGEN_CI_CTEST_REGEX="${selection}"
      ;;
  esac
fi

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
if ${ctest_cmd} -T test; then
  echo "Tests passed on the first attempt."
  exit_code=0
else
  echo "Initial tests failed with exit code $?. Retrying up to ${EIGEN_CI_CTEST_REPEAT} times..."
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
