# Find Visual Studio installation directory.
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
$VS_INSTALL_DIR = &"${Env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath

# Run VCVarsAll.bat initialization script and extract environment variables.
# http://allen-mack.blogspot.com/2008/03/replace-visual-studio-command-prompt.html
cmd.exe /c "`"${VS_INSTALL_DIR}\VC\Auxiliary\Build\vcvarsall.bat`" $EIGEN_CI_MSVC_ARCH -vcvars_ver=$EIGEN_CI_MSVC_VER & set" |
  foreach {
    if ($_ -match "^([^=]+)=(.*)$") {
      set-item -force -LiteralPath "ENV:\$($Matches[1])" -value "$($Matches[2])"
    }
  }

# Create and enter build directory.
$rootdir = Get-Location
if (-Not (Test-Path ${EIGEN_CI_BUILDDIR})) {
    mkdir $EIGEN_CI_BUILDDIR
}
cd $EIGEN_CI_BUILDDIR

# We need to split EIGEN_CI_ADDITIONAL_ARGS, otherwise they are interpreted
# as a single argument.  Split by space, unless double-quoted.
$split_args = [regex]::Split(${EIGEN_CI_ADDITIONAL_ARGS}, ' (?=(?:[^"]|"[^"]*")*$)' )

# Locate ccache when the job enables it (mirrors build.linux.script.sh: the
# GitLab cache holds ${CCACHE_DIR}, keyed on file content and compiler, so it
# hits even after a re-checkout re-stamps every source mtime). Prefer a
# runner-installed ccache, then a previously downloaded copy restored from the
# .ccache-bin cache; otherwise download the release pinned below and cache it
# for subsequent jobs on this runner. The SHA-256 must match, so a failed or
# tampered download only means building without ccache, never running an
# unverified binary. Caching MSVC needs ccache >= 4.8, and only /Z7 or no
# debug information is cacheable; these MinSizeRel builds emit none.
$ccache_exe = ""
if ("${EIGEN_CI_CCACHE}" -eq "on") {
  $ccache_version = "4.13.6"
  $ccache_sha256 = "3d7cebb05850ad704e197b3f1d3f0f924ab6c9fdfc561578e146184fe9d89380"
  $ccache_bindir = Join-Path ${rootdir} ".ccache-bin"
  $system_ccache = Get-Command ccache -ErrorAction SilentlyContinue
  if ($system_ccache) {
    $ccache_exe = $system_ccache.Source
  } elseif (Test-Path (Join-Path $ccache_bindir "ccache.exe")) {
    $ccache_exe = Join-Path $ccache_bindir "ccache.exe"
  } else {
    $zip = Join-Path ([System.IO.Path]::GetTempPath()) "ccache-${ccache_version}.zip"
    try {
      $ProgressPreference = "SilentlyContinue"
      Invoke-WebRequest "https://github.com/ccache/ccache/releases/download/v${ccache_version}/ccache-${ccache_version}-windows-x86_64.zip" -OutFile $zip
      if ((Get-FileHash $zip -Algorithm SHA256).Hash -eq $ccache_sha256) {
        $unpack = Join-Path ([System.IO.Path]::GetTempPath()) "ccache-unpack"
        Expand-Archive $zip -DestinationPath $unpack -Force
        New-Item -ItemType Directory -Force -Path $ccache_bindir | Out-Null
        Copy-Item (Join-Path $unpack "ccache-${ccache_version}-windows-x86_64/ccache.exe") $ccache_bindir
        $ccache_exe = Join-Path $ccache_bindir "ccache.exe"
      } else {
        Write-Warning "ccache download failed SHA-256 verification; building without ccache."
      }
    } catch {
      Write-Warning "ccache download failed ($_); building without ccache."
    }
  }
}

$launchers = @()
if ($ccache_exe) {
  # Forward slashes: CMake treats the launcher as a path-valued cache entry.
  $ccache_cmake = $ccache_exe -replace '\\', '/'
  $launchers = "-DCMAKE_C_COMPILER_LAUNCHER=${ccache_cmake}",
               "-DCMAKE_CXX_COMPILER_LAUNCHER=${ccache_cmake}"
  & $ccache_exe --zero-stats
}

# Configure build.
cmake -G Ninja -DCMAKE_BUILD_TYPE=MinSizeRel `
      -DEIGEN_TEST_CUSTOM_CXX_FLAGS="${EIGEN_CI_TEST_CUSTOM_CXX_FLAGS}" `
      ${launchers} ${split_args} "${rootdir}"

$target = ""
if (${EIGEN_CI_BUILD_TARGET}) {
  $target = "--target ${EIGEN_CI_BUILD_TARGET}"
}

# Windows builds sometimes fail due heap errors. In that case, try
# building the rest, then try to build again with a single thread.
cmake --build . ${target} -- -k0 || cmake --build . ${target} -- -k0 -j1

$success = $LASTEXITCODE

# Hit/miss summary for judging what the cache pays for on this job. Runs on
# failures too: the cache is pushed even then (cache:when: always), so the
# stats still describe what the next attempt can reuse.
if ($ccache_exe) {
  & $ccache_exe --show-stats
}

# Return to root directory.
cd ${rootdir}

# Explicitly propagate exit code to indicate pass/failure of build command.
if($success -ne 0) { Exit $success }
