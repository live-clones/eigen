# Sets up the MSVC toolchain on a Linux runner, running under Wine.
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
#
# Source (don't execute) this: it puts the MSVC wrappers (cl, link, ...) on PATH.
#
# The toolchain is fetched from Microsoft's own installer manifests via
# msvc-wine (https://github.com/mstorsjo/msvc-wine).  Visual Studio is not
# redistributable, so it cannot be baked into a public CI image and is
# downloaded per job instead (~800 MB, ~3 GB unpacked).

# The msvc-wine scripts (ISC licensed) ship in the CI image; fall back to a
# clone so this also works outside it, e.g. when reproducing a failure locally.
msvc_wine_dir="${EIGEN_CI_MSVC_WINE_DIR:-/opt/msvc-wine}"
msvc_dir="${EIGEN_CI_MSVC_DIR:-${PWD}/.msvc}"

if [ ! -x "${msvc_dir}/bin/x64/cl" ]; then
  if [ ! -f "${msvc_wine_dir}/vsdownload.py" ]; then
    msvc_wine_dir="${PWD}/.msvc-wine"
    rm -rf "${msvc_wine_dir}"
    git clone -q https://github.com/mstorsjo/msvc-wine.git "${msvc_wine_dir}"
    # Keep in sync with ci/docker/ubuntu-24.04-amd64-msvc-wine-build/Dockerfile.
    git -C "${msvc_wine_dir}" checkout -q \
        "${EIGEN_CI_MSVC_WINE_COMMIT:-514f8ea34842cd6d831804d0e9658d3a32870ae1}"
  fi

  # Resolve from the manifest pinned in the CI image when there is one, so the
  # serviced compiler build only changes when that image is rebuilt.  Outside
  # the image, fall back to whatever the release channel currently offers.
  msvc_manifest="${EIGEN_CI_MSVC_MANIFEST:-/opt/msvc-manifest/vs.manifest}"
  if [ -f "${msvc_manifest}" ]; then
    echo "Using pinned installer manifest:" \
         "$(cat /opt/msvc-manifest/version.txt 2>/dev/null || echo unknown)"
    channel_args="--manifest ${msvc_manifest}"
  else
    channel_args="--major ${EIGEN_CI_MSVC_VS_MAJOR:-17}"
  fi

  # Note: --msvc-version takes the Visual Studio product version, not the
  # toolset version.  17.8 selects the v143 toolset (MSVC 14.38), matching the
  # hosted Windows runners the cross-compiled tests run on.
  "${msvc_wine_dir}/vsdownload.py" --accept-license                 \
      ${channel_args}                                               \
      --msvc-version "${EIGEN_CI_MSVC_VS_VERSION:-16.11}"           \
      --architecture x64 --with-atl no --with-asan no               \
      --dest "${msvc_dir}"
  "${msvc_wine_dir}/install.sh" "${msvc_dir}"
fi

export PATH="${msvc_dir}/bin/x64:${PATH}"
export WINEDEBUG=-all
export WINEPREFIX="${WINEPREFIX:-${PWD}/.wine}"
