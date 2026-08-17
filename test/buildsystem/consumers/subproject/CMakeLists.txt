# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# Embeds Eigen the way doc/TopicCMakeGuide.dox describes for FetchContent:
# the options are forced before Eigen is added, then Eigen3::Eigen is linked.
# FetchContent_MakeAvailable reduces to this add_subdirectory call, so driving
# add_subdirectory directly tests the same path without a network fetch.
#
# EIGEN_USE_EXCLUDE_FROM_ALL selects the variant the guide's EXCLUDE_FROM_ALL
# section describes.

cmake_minimum_required(VERSION 3.17)
project(eigen_subproject_consumer CXX)

set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

if(EIGEN_USE_EXCLUDE_FROM_ALL)
  add_subdirectory("${EIGEN_SOURCE_DIR}" eigen-build EXCLUDE_FROM_ALL)
else()
  add_subdirectory("${EIGEN_SOURCE_DIR}" eigen-build)
endif()

add_executable(consumer ../main.cpp)
target_link_libraries(consumer Eigen3::Eigen)

# The consumer's own install rule must keep working whatever Eigen's does, so
# the scenarios can tell "Eigen installed nothing" apart from "nothing ran".
install(TARGETS consumer RUNTIME DESTINATION bin)
