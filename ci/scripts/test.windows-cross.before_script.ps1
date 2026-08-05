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
