#!/bin/bash
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

set -x

echo "Running ${CI_JOB_NAME}"

# Get architecture and display CI configuration.
export ARCH=`uname -m`
export NPROC=`nproc`
# Memory the job may use, in MB: the cgroup limit when the executor sets one,
# otherwise the machine's RAM.  cgroup v2 writes the literal "max" when no
# limit applies and v1 a sentinel far above RAM; both fall through to MemTotal.
mem_mb=$(awk '/^MemTotal:/ {print int($2 / 1024)}' /proc/meminfo)
for cgroup_file in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
  if [[ -r ${cgroup_file} ]] && read -r limit_mb < "${cgroup_file}" && [[ ${limit_mb} =~ ^[0-9]+$ ]]; then
    limit_mb=$((limit_mb / 1024 / 1024))
    ((limit_mb < mem_mb)) && mem_mb=${limit_mb}
    break
  fi
done
export MEM_MB=${mem_mb}
echo "arch=$ARCH, target=${EIGEN_CI_TARGET_ARCH}"
echo "Processors: ${NPROC}"
echo "Memory (MB): ${MEM_MB}"
echo "CI Variables:"
export | grep EIGEN

# Set noninteractive, otherwise tzdata may be installed and prompt for a
# geographical region.
export DEBIAN_FRONTEND=noninteractive
if [[ "${EIGEN_CI_SKIP_APT}" != "true" ]]; then
  apt-get update -y > /dev/null
  # python3 drives the test pass cache; only the test jobs (the jobs that
  # set EIGEN_CI_TEST_CACHE) consume it, so build jobs skip the install.
  packages="ninja-build cmake git xsltproc ccache"
  if [[ "${EIGEN_CI_TEST_CACHE}" == "on" ]]; then
    packages="${packages} python3"
  fi
  apt-get install -y --no-install-recommends ${packages} > /dev/null
fi

# Install required dependencies and set up compilers.
# These are required even for testing to ensure that dynamic runtime libraries
# are available.
if [[ "$ARCH" == "${EIGEN_CI_TARGET_ARCH}" || "${EIGEN_CI_TARGET_ARCH}" == "any" ]]; then
  if [[ "${EIGEN_CI_SKIP_APT}" != "true" ]]; then
    apt-get install -y --no-install-recommends ${EIGEN_CI_INSTALL} > /dev/null;
  fi
  export EIGEN_CI_CXX_IMPLICIT_INCLUDE_DIRECTORIES="";
  export EIGEN_CI_CXX_COMPILER_TARGET="";
else
  if [[ "${EIGEN_CI_SKIP_APT}" != "true" ]]; then
    apt-get install -y --no-install-recommends ${EIGEN_CI_CROSS_INSTALL} > /dev/null;
  fi
  export EIGEN_CI_C_COMPILER=${EIGEN_CI_CROSS_C_COMPILER};
  export EIGEN_CI_CXX_COMPILER=${EIGEN_CI_CROSS_CXX_COMPILER};
  export EIGEN_CI_CXX_COMPILER_TARGET=${EIGEN_CI_CROSS_TARGET_TRIPLE};
  # Tell the compiler where to find headers and libraries if using clang.
  # NOTE: this breaks GCC since it messes with include path order
  #       (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=70129)
  if [[ "${EIGEN_CI_CROSS_CXX_COMPILER}" == *"clang"* ]]; then
    export CPLUS_INCLUDE_PATH="/usr/${EIGEN_CI_CROSS_TARGET_TRIPLE}/include";
    export LIBRARY_PATH="/usr/${EIGEN_CI_CROSS_TARGET_TRIPLE}/lib64";
  fi
fi

echo "Compilers: ${EIGEN_CI_C_COMPILER} ${EIGEN_CI_CXX_COMPILER}"

if [ -n "$EIGEN_CI_BEFORE_SCRIPT" ]; then eval "$EIGEN_CI_BEFORE_SCRIPT"; fi

set +x
