# Repoints a Linux-configured build tree at its Windows location.
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
#
# The build tree was configured on a Linux runner, so the absolute paths CMake
# baked into CTestTestfile.cmake refer to the Linux build directory.  Rewrite
# them to the current one, otherwise ctest cannot find any test executable.

$builddir = (Get-Item ${EIGEN_CI_BUILDDIR}).FullName -replace '\\','/'
$stamp = Join-Path ${EIGEN_CI_BUILDDIR} '.eigen_ci_builddir'
$origdir = (Get-Content $stamp -First 1).Trim()

Write-Host "Repointing CTest files: ${origdir} -> ${builddir}"

Get-ChildItem -Path ${EIGEN_CI_BUILDDIR} -Filter CTestTestfile.cmake -Recurse |
  ForEach-Object {
    (Get-Content $_.FullName -Raw).Replace($origdir, $builddir) |
      Set-Content $_.FullName -NoNewline
  }
