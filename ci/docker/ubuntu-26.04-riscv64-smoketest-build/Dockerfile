# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
#
# Slim CI image for riscv64 MR smoke test build jobs (Ubuntu 26.04).
# Builds run natively on the riscv runner; no cross-compilers needed.
#
# Rebuild and push when this file changes:
#   docker buildx build --platform linux/riscv64 \
#     -t registry.gitlab.com/libeigen/eigen/ubuntu-26.04-riscv64-smoketest-build:latest \
#     --push ci/docker/ubuntu-26.04-riscv64-smoketest-build/

FROM ubuntu:26.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      cmake ninja-build git \
      gcc-15 g++-15 \
      clang-21 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
