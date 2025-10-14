#!/bin/bash

ctest_cmd="ctest ${EIGEN_CI_CTEST_ARGS} --parallel ${NPROC} --output-on-failure --no-compress-output --build-noclean ${target}" 
max_retries=3
exit_code=1

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

set +x

if ${ctest_cmd} -T test; then
  echo "Tests passed on the first attempt."
  exit_code=0
else
  echo "Initial tests failed with exit code $?. Retrying up to ${max_retries} times..."
  if ${ctest_cmd} --rerun-failed --repeat until-pass:${max_retries}; then
    echo "Tests passed on retry."
    exit_code=42
  fi
fi

set -x

# Return to root directory.
cd ${rootdir}

set +x

exit $exit_code
