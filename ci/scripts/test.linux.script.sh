#!/bin/bash

set -x

# Enter build directory.
rootdir=`pwd`
cd ${EIGEN_CI_BUILDDIR}

target=""
if [[ ${EIGEN_CI_CTEST_REGEX} ]]; then
  target="-R ${EIGEN_CI_CTEST_REGEX}"
elif [[ ${EIGEN_CI_CTEST_LABEL} ]]; then
  target="-L ${EIGEN_CI_CTEST_LABEL}"
fi

ctest_cmd="ctest ${EIGEN_CI_CTEST_ARGS} --parallel ${NPROC} --output-on-failure --no-compress-output --build-noclean ${target}"

exit_code=1

# Run the initial CTest command
echo "Running initial tests..."
if ${ctest_cmd} -T test; then
  echo "Tests passed on the first attempt."
  exit_code=0
else
  # Initial run failed, start retries
  echo "Initial tests failed with exit code $?. Retrying up to ${EIGEN_CI_CTEST_REPEAT} times..."
  retries_left=${EIGEN_CI_CTEST_REPEAT}
  retry=0
  while [[ ${retry} -lt ${EIGEN_CI_CTEST_REPEAT} ]]; do
    echo "Attempting retry ${retry} of ${EIGEN_CI_CTEST_REPEAT}..."
    # Rerun only the tests that failed previously
    if ${ctest_cmd} --rerun-failed; then
      echo "Tests passed on retry."
      exit_code=42
      break # Exit the retry loop on success
    fi
    ((retry++))
  done
fi


# Return to root directory.
cd ${rootdir}

set +x

exit $exit_code
