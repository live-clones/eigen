# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# doc/TopicCMakeGuide.dox: "we also support a range spanning major versions",
# spelled find_package(Eigen3 3.4...5 REQUIRED NO_MODULE).  The claim rests on
# the hand-written cmake/Eigen3ConfigVersion.cmake.in, whose inclusive upper
# bound admits everything below <max>+1, so the range has to be exercised
# against a real installed package rather than reasoned about.
#
# The upper bound is taken from Eigen's own major version, which is what makes
# the documented "3.4...5" span the current release; writing 5 here instead
# would turn this into a failing test the day Eigen 6 ships.
#
# test/buildsystem/CMakeLists.txt registers this scenario only for CMake 3.19
# and newer, which is where find_package learned range syntax.

bs_install_eigen()
bs_eigen_package_version(major minor patch)

bs_configure("consumer with range 3.4...${major}" "${BS_CONSUMER_DIR}/installed"
             "${WORK_DIR}/consumer"
             "-DCMAKE_PREFIX_PATH=${BS_PREFIX}"
             "-DEIGEN_EXPECTED_PREFIX=${BS_PREFIX}"
             "-DEIGEN_VERSION_SPEC=3.4...${major}"
             ${BS_FIND_PACKAGE_ISOLATION})
bs_build_and_run_consumer("consumer with a version range" "${WORK_DIR}/consumer")
