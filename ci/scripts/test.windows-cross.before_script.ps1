# Repoints a Linux-configured build tree at its Windows location.
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
#
# The build tree was configured on a Linux runner, so the absolute paths CMake
# baked into it refer to the Linux build directory.  Rewrite them to the current
# one: CTestTestfile.cmake so ctest can find the test executables, and
# DartConfiguration.tcl so `ctest -T test` can initialize the dashboard and
# write Testing/, which the JUnit conversion in the after script needs.

$builddir = (Get-Item ${EIGEN_CI_BUILDDIR}).FullName -replace '\\','/'
$stamp = Join-Path ${EIGEN_CI_BUILDDIR} '.eigen_ci_builddir'
$origdir = (Get-Content $stamp -First 1).Trim()

Write-Host "Repointing CTest files: ${origdir} -> ${builddir}"

Get-ChildItem -Path ${EIGEN_CI_BUILDDIR} -Recurse -File |
  Where-Object { $_.Name -in 'CTestTestfile.cmake', 'DartConfiguration.tcl' } |
  ForEach-Object {
    (Get-Content $_.FullName -Raw).Replace($origdir, $builddir) |
      Set-Content $_.FullName -NoNewline
  }

# The binaries link the MSVC runtime dynamically, and that runtime has to be at
# least as new as the toolset they were built with.  Report both, so a run that
# fails because the two drifted apart says so instead of looking like 166 broken
# tests.  Nothing is enforced here: the runners are the moving part, and this
# job is allowed to fail.
$builtWith = 'unknown'
$toolsetStamp = Join-Path ${EIGEN_CI_BUILDDIR} '.eigen_ci_msvc_toolset'
if (Test-Path $toolsetStamp) {
  $builtWith = (Get-Content $toolsetStamp -First 1).Trim()
}

$runnerHas = 'unknown'
$vswhere = "${Env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
  $vsPath = & $vswhere -latest -property installationPath
  $versionFile = Join-Path $vsPath 'VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt'
  if ($vsPath -and (Test-Path $versionFile)) {
    $runnerHas = (Get-Content $versionFile -First 1).Trim()
  }
}

Write-Host "MSVC toolset: built with ${builtWith}, this runner has ${runnerHas}"

if ($builtWith -ne 'unknown' -and $runnerHas -ne 'unknown' -and $builtWith -ne $runnerHas) {
  Write-Host "WARNING: the build and the runner have drifted apart."
  if ([version]$builtWith -gt [version]$runnerHas) {
    Write-Host ("WARNING: the binaries are newer than this runner's runtime, so " +
                "they will fail to load. Set EIGEN_CI_MSVC_VS_VERSION in " +
                "ci/build.windows.gitlab-ci.yml to the Visual Studio version " +
                "this runner reports.")
  }
}
