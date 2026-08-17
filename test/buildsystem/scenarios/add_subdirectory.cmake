# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# doc/TopicCMakeGuide.dox: Eigen can be embedded without installing it first,
# and the guide's "Disabling Eigen build options" snippet forces
# EIGEN_BUILD_TESTING, EIGEN_BUILD_DOC, and BUILD_TESTING off beforehand.  The
# consumer project sets exactly those, so this covers both claims: Eigen3::Eigen
# is usable from a sub-project, and the option overrides take effect.

bs_configure("embedding consumer" "${BS_CONSUMER_DIR}/subproject" "${WORK_DIR}/consumer"
             "-DEIGEN_SOURCE_DIR=${EIGEN_SOURCE_DIR}")
bs_build_and_run_consumer("embedding consumer" "${WORK_DIR}/consumer")

# The forced options must actually reach Eigen: with EIGEN_BUILD_TESTING off,
# Eigen registers no tests in the enclosing project.
bs_run(WHAT "list tests of the embedding consumer" EXPECT_RESULT 0 OUTPUT_VARIABLE tests
       COMMAND ${CMAKE_CTEST_COMMAND} --test-dir "${WORK_DIR}/consumer" -N)
if(tests MATCHES "Test *#1:")
  bs_fail("EIGEN_BUILD_TESTING=OFF was ignored; Eigen registered tests:\n----\n${tests}\n----")
endif()
