# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
#
# CI image for amd64 Linux build/test jobs (Ubuntu 24.04).
# Covers: x86-64 native builds and tests (gcc-10/13, clang-14/16/19),
#         ASAN/UBSAN builds and tests (libclang-rt-19-dev, llvm-19),
#         ARM/ppc64le/loongarch64 cross-compilation and QEMU test execution.
#
# Rebuild and push when this file changes:
#   docker buildx build --platform linux/amd64 \
#     -t registry.gitlab.com/libeigen/eigen/ubuntu-24.04-amd64:latest \
#     --push ci/docker/ubuntu-24.04-amd64/

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      cmake ninja-build git \
      gcc-10 g++-10 \
      gcc-13 g++-13 \
      clang-14 \
      clang-16 \
      clang-19 \
      libclang-rt-19-dev \
      llvm-19 \
      g++-10-arm-linux-gnueabihf \
      g++-14-powerpc64le-linux-gnu \
      g++-14-loongarch64-linux-gnu \
      qemu-user-static \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
