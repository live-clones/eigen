# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
#
# Slim CI image for amd64 MR smoke test jobs (Ubuntu 24.04).
# Covers only what the two x86-64 smoke test configurations need:
#   gcc-10 native builds and clang-14 native builds.
#
# Rebuild and push when this file changes:
#   docker buildx build --platform linux/amd64 \
#     -t registry.gitlab.com/libeigen/eigen/ubuntu-24.04-amd64-smoketest:latest \
#     --push ci/docker/ubuntu-24.04-amd64-smoketest/

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      cmake ninja-build git \
      gcc-10 g++-10 \
      clang-14 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
