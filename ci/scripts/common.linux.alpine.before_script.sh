#!/bin/bash

set -x

echo "Running ${CI_JOB_NAME}"

# Get architecture and display CI configuration.
export ARCH=`uname -m`
export EIGEN_CI_TARGET_ARCH=`uname -m`
export NPROC=`nproc`
echo "arch=$ARCH, target=${EIGEN_CI_TARGET_ARCH}"
echo "Processors: ${NPROC}"
echo "CI Variables:"
export | grep EIGEN

apk add libgcc binutils samurai cmake git musl-dev ${EIGEN_CI_INSTALL}

export EIGEN_CI_CXX_IMPLICIT_INCLUDE_DIRECTORIES="";
export EIGEN_CI_CXX_COMPILER_TARGET="";

echo "Compilers: ${EIGEN_CI_C_COMPILER} ${EIGEN_CI_CXX_COMPILER}"

if [ -n "$EIGEN_CI_BEFORE_SCRIPT" ]; then eval "$EIGEN_CI_BEFORE_SCRIPT"; fi

set +x
