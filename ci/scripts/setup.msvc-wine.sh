# Sets up the MSVC toolchain on a Linux runner, running under Wine.
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
#
# Source (don't execute) this: it puts the MSVC wrappers (cl, link, ...) on PATH.
#
# The toolchain is fetched from Microsoft's own installer manifests via
# msvc-wine (https://github.com/mstorsjo/msvc-wine), per job (~800 MB, ~3 GB
# unpacked).  Eigen is MPL-2.0, an OSI-approved license, so the Visual Studio
# Community terms cover using MSVC to develop and test it; they do not permit
# sharing or otherwise distributing it, so it stays out of the CI image and out
# of the job artifacts, and lives only in this workspace while the job runs.
#
# Nothing is licensed or activated at run time -- --accept-license is a local
# flag, and the installed compiler builds with no network access.

# The msvc-wine scripts (ISC licensed) ship in the CI image; fall back to a
# clone so this also works outside it, e.g. when reproducing a failure locally.
msvc_wine_dir="${EIGEN_CI_MSVC_WINE_DIR:-/opt/msvc-wine}"
msvc_dir="${EIGEN_CI_MSVC_DIR:-${PWD}/.msvc}"

if [ ! -x "${msvc_dir}/bin/x64/cl" ]; then
  if [ ! -f "${msvc_wine_dir}/vsdownload.py" ]; then
    # Check out the commit the CI image is built from, read from the one file
    # both sides use so they cannot drift; the Dockerfile has it in its build
    # context.
    msvc_wine_commit="${EIGEN_CI_MSVC_WINE_COMMIT}"
    if [ -z "${msvc_wine_commit}" ]; then
      commit_file="$(dirname "${BASH_SOURCE[0]}")/../docker"
      commit_file="${commit_file}/ubuntu-24.04-amd64-msvc-wine-build"
      msvc_wine_commit=$(cat "${commit_file}/msvc-wine-commit.txt") || return 1
    fi
    msvc_wine_dir="${PWD}/.msvc-wine"
    rm -rf "${msvc_wine_dir}"
    git clone -q https://github.com/mstorsjo/msvc-wine.git "${msvc_wine_dir}"
    git -C "${msvc_wine_dir}" checkout -q "${msvc_wine_commit}"
  fi

  # Resolve from the manifest pinned in the CI image when there is one, so the
  # serviced compiler build only changes when that image is rebuilt.  Outside
  # the image, fall back to whatever the release channel currently offers.
  msvc_manifest="${EIGEN_CI_MSVC_MANIFEST:-/opt/msvc-manifest/vs.manifest}"
  if [ -f "${msvc_manifest}" ]; then
    # The version is written next to the manifest, so follow it if the manifest
    # was overridden rather than reporting the image's version for someone
    # else's file.
    msvc_manifest_version=$(cat "$(dirname "${msvc_manifest}")/version.txt" \
                            2>/dev/null)
    printf 'Using pinned installer manifest: %s\n' \
           "${msvc_manifest_version:-unknown}"
    channel_args="--manifest ${msvc_manifest}"
  else
    channel_args="--major ${EIGEN_CI_MSVC_VS_MAJOR:-17}"
  fi

  # Note: --msvc-version takes the Visual Studio product version, not the
  # toolset version.  17.14 selects the v143 toolset (MSVC 14.44), matching the
  # hosted Windows runners the cross-compiled tests run on.
  "${msvc_wine_dir}/vsdownload.py" --accept-license                 \
      ${channel_args}                                               \
      --msvc-version "${EIGEN_CI_MSVC_VS_VERSION:-17.14}"           \
      --architecture x64 --with-atl no --with-asan no               \
      --dest "${msvc_dir}"
  # Creating the Wine prefix logs
  #   wine: failed to open L"C:\windows\syswow64\rundll32.exe": c0000135
  # because the image installs 64-bit Wine only, so wineboot cannot register the
  # 32-bit subsystem.  Nothing here needs it -- the compiler, the linker and the
  # tests they produce are all x64 -- and WINEARCH=win64 does not suppress it.
  "${msvc_wine_dir}/install.sh" "${msvc_dir}"
fi

# The generated wrappers record the toolset they point at.  Surface it so the
# Windows test job can compare it against the runtime it has.
EIGEN_CI_MSVC_TOOLSET=$(sed -n 's/^MSVCVER=//p' "${msvc_dir}/bin/x64/msvcenv.sh")
export EIGEN_CI_MSVC_TOOLSET

export PATH="${msvc_dir}/bin/x64:${PATH}"
export WINEDEBUG=-all
export WINEPREFIX="${WINEPREFIX:-${PWD}/.wine}"
