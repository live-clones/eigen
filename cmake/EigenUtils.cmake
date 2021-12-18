# automatically parse the version number
function(extract_eigen_version SOURCE_FILE)
  file(READ "${SOURCE_FILE}" _eigen_version_header)
  string(REGEX MATCH "define[ \t]+EIGEN_WORLD_VERSION[ \t]+([0-9]+)" _eigen_world_version_match "${_eigen_version_header}")
  set(EIGEN_WORLD_VERSION "${CMAKE_MATCH_1}" PARENT_SCOPE)
  string(REGEX MATCH "define[ \t]+EIGEN_MAJOR_VERSION[ \t]+([0-9]+)" _eigen_major_version_match "${_eigen_version_header}")
  set(EIGEN_MAJOR_VERSION "${CMAKE_MATCH_1}" PARENT_SCOPE)
  string(REGEX MATCH "define[ \t]+EIGEN_MINOR_VERSION[ \t]+([0-9]+)" _eigen_minor_version_match "${_eigen_version_header}")
  set(EIGEN_MINOR_VERSION "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()

function (extract_git_version REPOSITORY)
  # if we are not in a git clone
  if(IS_DIRECTORY ${REPOSITORY}/.git)
    # if the git program is absent or this will leave the EIGEN_GIT_REVNUM string empty,
    # but won't stop CMake.
    execute_process(COMMAND git ls-remote --refs -q ${REPOSITORY} HEAD OUTPUT_VARIABLE EIGEN_GIT_OUTPUT)
  endif()

  # extract the git rev number from the git output...
  if(EIGEN_GIT_OUTPUT)
    string(REGEX MATCH "^([0-9;a-f]+).*" EIGEN_GIT_CHANGESET_MATCH "${EIGEN_GIT_OUTPUT}")
    set(EIGEN_GIT_REVNUM "${CMAKE_MATCH_1}" PARENT_SCOPE)
  endif()
endfunction()