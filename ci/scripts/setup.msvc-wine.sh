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

msvc_wine_dir="${EIGEN_CI_MSVC_WINE_DIR:-${PWD}/.msvc-wine}"
msvc_dir="${EIGEN_CI_MSVC_DIR:-${PWD}/.msvc}"

if [ ! -x "${msvc_dir}/bin/x64/cl" ]; then
  rm -rf "${msvc_wine_dir}"
  git clone -q https://github.com/mstorsjo/msvc-wine.git "${msvc_wine_dir}"
  git -C "${msvc_wine_dir}" checkout -q "${EIGEN_CI_MSVC_WINE_COMMIT}"

  # Note: --msvc-version takes the Visual Studio product version, not the
  # toolset version.  16.11 selects the v142 toolset (MSVC 14.29), the same
  # compiler the native Windows jobs pin with -vcvars_ver.
  "${msvc_wine_dir}/vsdownload.py" --accept-license                 \
      --major "${EIGEN_CI_MSVC_VS_MAJOR}"                           \
      --msvc-version "${EIGEN_CI_MSVC_VS_VERSION}"                  \
      --architecture x64 --with-atl no --with-asan no               \
      --dest "${msvc_dir}"
  "${msvc_wine_dir}/install.sh" "${msvc_dir}"
fi

export PATH="${msvc_dir}/bin/x64:${PATH}"
export WINEDEBUG=-all
export WINEPREFIX="${WINEPREFIX:-${PWD}/.wine}"
