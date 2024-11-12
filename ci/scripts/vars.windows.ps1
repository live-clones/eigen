# SPDX-License-Identifier: MPL-2.0

# Initialize default variables used by the CI.
$EIGEN_CI_ADDITIONAL_ARGS       = ""
$EIGEN_CI_BEFORE_SCRIPT         = ""
$EIGEN_CI_BUILD_TARGET          = "buildtests"
$EIGEN_CI_BUILDDIR              = ".build"
$EIGEN_CI_MSVC_ARCH             = "x64"
$EIGEN_CI_MSVC_VER              = "14.29"
$EIGEN_CI_TEST_CUSTOM_CXX_FLAGS = "/d2ReducedOptimizeHugeFunctions /DEIGEN_STRONG_INLINE=inline /Os"
$EIGEN_CI_TEST_LABEL            = "Official"
$EIGEN_CI_TEST_REGEX            = ""
