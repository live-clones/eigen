# Change to build directory.
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
$rootdir = Get-Location
cd ${EIGEN_CI_BUILDDIR}

# No dashboard run to convert when ctest never reached its test phase; an
# empty report is a parse error, so write none.  See the Linux after_script.
if ((Test-Path Testing\TAG) -and (Get-Content Testing\TAG | select -first 1)) {
  $TEST_TAG = Get-Content Testing\TAG | select -first 1
  if (Test-Path Testing\$TEST_TAG\Test.xml) {
    # PowerShell equivalent to xsltproc:
    $XSL_FILE = Resolve-Path "..\ci\CTest2JUnit.xsl"
    $INPUT_FILE = Resolve-Path Testing\$TEST_TAG\Test.xml
    $OUTPUT_FILE = Join-Path -Path $pwd -ChildPath JUnitTestResults_$CI_JOB_ID.xml
    $xslt = New-Object System.Xml.Xsl.XslCompiledTransform;
    $xslt.Load($XSL_FILE)
    $xslt.Transform($INPUT_FILE,$OUTPUT_FILE)
  }
}

# Return to root directory.
cd ${rootdir}
