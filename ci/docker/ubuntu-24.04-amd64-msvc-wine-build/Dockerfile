# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
#
# CI image for cross-compiling the Windows unit tests on an amd64 Linux runner
# (Ubuntu 24.04), by running the MSVC toolchain under Wine.
#
# Contains Wine and the msvc-wine scripts (ISC licensed), but deliberately not
# the MSVC toolchain: Visual Studio is not redistributable, so the compiler
# itself is fetched from Microsoft's installer manifests at job time by
# ci/scripts/setup.msvc-wine.sh.
#
# Packages:
#   wine, wine64 - runs cl.exe/link.exe
#   winbind      - Wine needs it for the separate-PDB debug info CMake probes with
#   msitools     - unpacks the MSI/CAB packages vsdownload.py downloads
#   python3      - runs vsdownload.py
#
# Rebuild and push when this file changes:
#   docker buildx build --platform linux/amd64 \
#     -t registry.gitlab.com/libeigen/eigen/ubuntu-24.04-amd64-msvc-wine-build:latest \
#     --push ci/docker/ubuntu-24.04-amd64-msvc-wine-build/

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MSVC_WINE_COMMIT=514f8ea34842cd6d831804d0e9658d3a32870ae1
ARG MSVC_VS_MAJOR=17

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      ca-certificates cmake git ninja-build python3 \
      msitools winbind wine wine64 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/mstorsjo/msvc-wine.git /opt/msvc-wine && \
    git -C /opt/msvc-wine checkout ${MSVC_WINE_COMMIT} && \
    rm -rf /opt/msvc-wine/.git

# Pin which Visual Studio release the toolchain is resolved from. Without this
# every job re-resolves the current release channel, so the serviced compiler
# build can change under an unchanged .gitlab-ci.yml; rebuilding this image is
# then the deliberate way to move to a newer one.
RUN mkdir -p /opt/msvc-manifest && cd /tmp && \
    /opt/msvc-wine/vsdownload.py --accept-license --major ${MSVC_VS_MAJOR} \
        --save-manifest --print-version && \
    manifest=$(ls *.manifest) && \
    printf '%s\n' "${manifest%.manifest}" > /opt/msvc-manifest/version.txt && \
    mv "${manifest}" /opt/msvc-manifest/vs.manifest
