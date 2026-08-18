# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# The complement of find_package_version_range: without a rejection case, a
# version file that accepted everything would pass that test.
#
# Both constraints come from doc/TopicCMakeGuide.dox and from the semantics
# cmake/Eigen3ConfigVersion.cmake.in documents:
#   - a bare "3.4" restricts to 3.4.z, so it must not match a later major;
#   - a major above the installed one must not match either.

bs_install_eigen()
bs_eigen_package_version(major minor patch)

if(major LESS 4)
  bs_fail("this scenario assumes Eigen's major version is at least 4, got ${major}")
endif()

bs_configure_expect_failure("consumer pinned to 3.4" "${BS_CONSUMER_DIR}/installed"
                            "${WORK_DIR}/consumer-old" output
                            "-DCMAKE_PREFIX_PATH=${BS_PREFIX}"
                            "-DEIGEN_EXPECTED_PREFIX=${BS_PREFIX}"
                            "-DEIGEN_VERSION_SPEC=3.4"
                            ${BS_FIND_PACKAGE_ISOLATION})
if(NOT output MATCHES "Eigen3")
  bs_fail("configure failed for some reason other than the version check\n----\n${output}\n----")
endif()

math(EXPR too_new "${major}+1")
bs_configure_expect_failure("consumer requiring ${too_new}" "${BS_CONSUMER_DIR}/installed"
                            "${WORK_DIR}/consumer-new" output
                            "-DCMAKE_PREFIX_PATH=${BS_PREFIX}"
                            "-DEIGEN_EXPECTED_PREFIX=${BS_PREFIX}"
                            "-DEIGEN_VERSION_SPEC=${too_new}"
                            ${BS_FIND_PACKAGE_ISOLATION})
if(NOT output MATCHES "Eigen3")
  bs_fail("configure failed for some reason other than the version check\n----\n${output}\n----")
endif()
