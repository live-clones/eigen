# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# Cross-library comparison benchmarks: one executable per bench_*_compare.cpp,
# each carrying an Eigen arm and, when benchmarks/CMakeLists.txt selected a
# reference library through vendors.cmake, that library's arm. run.py builds
# and drives them; README.md says how.

# ---------------------------------------------------------------------------
# Build metadata the binaries publish (bench_compare.h publishArmContext)
# ---------------------------------------------------------------------------

set(_build_type "${CMAKE_BUILD_TYPE}")
if(NOT _build_type)
  set(_build_type "Release")
endif()
string(TOUPPER "${_build_type}" _build_type_upper)
set(EIGEN_BENCH_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_build_type_upper}}")
foreach(_flag -O3 -DNDEBUG)
  if(NOT EIGEN_BENCH_CXX_FLAGS MATCHES "(^| )${_flag}( |$)")
    string(APPEND EIGEN_BENCH_CXX_FLAGS " ${_flag}")
  endif()
endforeach()
string(STRIP "${EIGEN_BENCH_CXX_FLAGS}" EIGEN_BENCH_CXX_FLAGS)

# ISA target key, inferred from the flags unless the profile pins it (it cannot
# be inferred from -march=native).
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _processor)
if(_processor MATCHES "aarch64|arm64")
  if(EIGEN_BENCH_CXX_FLAGS MATCHES "sme")
    set(_isa_target "aarch64-sme")
  elseif(EIGEN_BENCH_CXX_FLAGS MATCHES "sve")
    set(_isa_target "aarch64-sve")
  else()
    set(_isa_target "aarch64-neon")
  endif()
elseif(_processor MATCHES "x86_64|amd64")
  if(EIGEN_BENCH_CXX_FLAGS MATCHES "avx512")
    set(_isa_target "x86-64-avx512")
  elseif(EIGEN_BENCH_CXX_FLAGS MATCHES "avx2")
    set(_isa_target "x86-64-avx2")
  else()
    set(_isa_target "x86-64")
  endif()
else()
  set(_isa_target "${_processor}")
endif()
set(EIGEN_BENCH_ISA_TARGET "${_isa_target}" CACHE STRING "ISA target key recorded in the results")

set(EIGEN_BENCH_EIGEN_COMMIT "")
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(COMMAND ${GIT_EXECUTABLE} --no-optional-locks -C "${EIGEN_SOURCE_DIR}" rev-parse HEAD
                  OUTPUT_VARIABLE EIGEN_BENCH_EIGEN_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET RESULT_VARIABLE _git_status)
  if(NOT _git_status EQUAL 0)
    set(EIGEN_BENCH_EIGEN_COMMIT "")
  endif()
endif()

# ---------------------------------------------------------------------------
# The reference library: version query and LAPACK
# ---------------------------------------------------------------------------

# The version query is declared in bench_compare.h and called only when this
# check links it; defining the macro on a guess would turn a missing query into
# a link failure.
set(_reference_version_define "")
eigen_bench_resolve_linkables("${EIGEN_BENCH_BLAS_LINK_LIBRARIES}" _blas_linkables_for_check)
if(EIGEN_BENCH_BLAS_FOUND AND EIGEN_BENCH_BLAS_VERSION_RUNTIME_SYMBOL AND _blas_linkables_for_check)
  include(CheckCXXSourceCompiles)
  set(_symbol "${EIGEN_BENCH_BLAS_VERSION_RUNTIME_SYMBOL}")
  string(TOUPPER "${_symbol}" _symbol_upper)
  set(CMAKE_REQUIRED_LIBRARIES ${_blas_linkables_for_check})
  set(CMAKE_REQUIRED_QUIET TRUE)
  check_cxx_source_compiles(
    "extern \"C\" void ${_symbol}();\nint main() { ${_symbol}(); return 0; }"
    EIGEN_BENCH_LINKS_${_symbol_upper})
  unset(CMAKE_REQUIRED_LIBRARIES)
  unset(CMAKE_REQUIRED_QUIET)
  if(EIGEN_BENCH_LINKS_${_symbol_upper})
    set(_reference_version_define "EIGEN_BENCH_HAVE_${_symbol_upper}")
  else()
    message(STATUS "Comparison benchmarks: ${_symbol}() does not link; the version comes from the profile")
  endif()
endif()

# Where ?potrf comes from: the vendor's own library, the vendor's separate
# LAPACK package, a LAPACK found beside the vendor's BLAS (AOCL's libflame next
# to libblis), or an unrelated LAPACK elsewhere on the system. Only the last
# owes an attribution note, and it is recorded rather than flattened.
set(EIGEN_BENCH_LAPACK_AVAILABLE FALSE)
set(EIGEN_BENCH_LAPACK_PROVIDER "")
set(EIGEN_BENCH_LAPACK_LINK_LIBRARIES "")
if(EIGEN_BENCH_BLAS_FOUND)
  if("lapack" IN_LIST EIGEN_BENCH_BLAS_PROVIDES)
    set(EIGEN_BENCH_LAPACK_AVAILABLE TRUE)
    set(EIGEN_BENCH_LAPACK_PROVIDER "vendor")
    if(EIGEN_BENCH_BLAS_LAPACK_PACKAGE)
      find_package(${EIGEN_BENCH_BLAS_LAPACK_PACKAGE} QUIET)
      if(${EIGEN_BENCH_BLAS_LAPACK_PACKAGE}_FOUND AND EIGEN_BENCH_BLAS_LAPACK_PACKAGE_TARGETS)
        set(EIGEN_BENCH_LAPACK_LINK_LIBRARIES ${EIGEN_BENCH_BLAS_LAPACK_PACKAGE_TARGETS})
      else()
        set(EIGEN_BENCH_LAPACK_AVAILABLE FALSE)
        set(EIGEN_BENCH_LAPACK_PROVIDER "")
        message(STATUS "Comparison benchmarks: ${EIGEN_BENCH_BLAS_LAPACK_PACKAGE} not found; LAPACK operations build Eigen-only")
      endif()
    endif()
  else()
    find_package(LAPACK QUIET)
    if(LAPACK_FOUND)
      set(EIGEN_BENCH_LAPACK_AVAILABLE TRUE)
      set(EIGEN_BENCH_LAPACK_PROVIDER "separate")
      if(TARGET LAPACK::LAPACK)
        set(EIGEN_BENCH_LAPACK_LINK_LIBRARIES LAPACK::LAPACK)
      else()
        set(EIGEN_BENCH_LAPACK_LINK_LIBRARIES ${LAPACK_LIBRARIES})
      endif()
      eigen_bench_resolve_linkables("${EIGEN_BENCH_LAPACK_LINK_LIBRARIES}" _lapack_paths)
      if(EIGEN_BENCH_BLAS_LIBRARY_PATH)
        get_filename_component(_blas_dir "${EIGEN_BENCH_BLAS_LIBRARY_PATH}" DIRECTORY)
        foreach(_lapack_path IN LISTS _lapack_paths)
          # FindLAPACK lists the BLAS it was linked with too; only the LAPACK
          # libraries themselves say where the routines come from.
          get_filename_component(_lapack_real "${_lapack_path}" REALPATH)
          if(_lapack_real STREQUAL EIGEN_BENCH_BLAS_LIBRARY_PATH OR _lapack_path IN_LIST EIGEN_BENCH_BLAS_LIBRARIES)
            continue()
          endif()
          get_filename_component(_lapack_dir "${_lapack_path}" DIRECTORY)
          if(_lapack_dir STREQUAL _blas_dir)
            set(EIGEN_BENCH_LAPACK_PROVIDER "vendor")
          endif()
        endforeach()
      endif()
      message(STATUS "Comparison benchmarks: lapack from ${_lapack_paths} (${EIGEN_BENCH_LAPACK_PROVIDER})")
    else()
      message(STATUS "Comparison benchmarks: no LAPACK found; LAPACK operations build Eigen-only")
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------

function(_eigen_bench_append_string_define list_var name value)
  set(result "${${list_var}}")
  string(REPLACE "\\" "\\\\" value "${value}")
  string(REPLACE "\"" "\\\"" value "${value}")
  list(APPEND result "${name}=\"${value}\"")
  set(${list_var} "${result}" PARENT_SCOPE)
endfunction()

set(_comparison_definitions "")
_eigen_bench_append_string_define(_comparison_definitions EIGEN_BENCH_COMPILER_ID "${CMAKE_CXX_COMPILER_ID}")
_eigen_bench_append_string_define(_comparison_definitions EIGEN_BENCH_COMPILER_VERSION "${CMAKE_CXX_COMPILER_VERSION}")
_eigen_bench_append_string_define(_comparison_definitions EIGEN_BENCH_CXX_STANDARD "${CMAKE_CXX_STANDARD}")
_eigen_bench_append_string_define(_comparison_definitions EIGEN_BENCH_CXX_FLAGS "${EIGEN_BENCH_CXX_FLAGS}")
_eigen_bench_append_string_define(_comparison_definitions EIGEN_BENCH_ISA_TARGET "${EIGEN_BENCH_ISA_TARGET}")
_eigen_bench_append_string_define(_comparison_definitions EIGEN_BENCH_EIGEN_COMMIT "${EIGEN_BENCH_EIGEN_COMMIT}")

# What makes a source register its reference arm; withheld from a binary whose
# reference routine lives in a family this build cannot call.
set(_comparison_reference_definitions "")
if(EIGEN_BENCH_BLAS_FOUND)
  # A bare token: bench_compare.h stringizes it.
  list(APPEND _comparison_reference_definitions "EIGEN_BENCH_REFERENCE_ARM=${EIGEN_BENCH_BLAS_ARM}")
  _eigen_bench_append_string_define(_comparison_reference_definitions EIGEN_BENCH_REFERENCE_LIBRARY_NAME "${EIGEN_BENCH_BLAS_DISPLAY_NAME}")
  _eigen_bench_append_string_define(_comparison_reference_definitions EIGEN_BENCH_REFERENCE_VERSION_FALLBACK "${EIGEN_BENCH_REFERENCE_VERSION}")
  _eigen_bench_append_string_define(_comparison_reference_definitions EIGEN_BENCH_REFERENCE_PATH "${EIGEN_BENCH_BLAS_LIBRARY_PATH}")
  if(EIGEN_BENCH_BLAS_INTERFACE_WIDTH STREQUAL "ilp64")
    list(APPEND _comparison_reference_definitions "EIGEN_BENCH_REFERENCE_ILP64=1")
  endif()
  if(_reference_version_define)
    list(APPEND _comparison_reference_definitions "${_reference_version_define}=1")
  endif()
endif()

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

add_custom_target(bench_comparison_all COMMENT "All comparison benchmark binaries")

if(CMAKE_VERSION VERSION_LESS 3.12)
  file(GLOB _comparison_sources "${CMAKE_CURRENT_SOURCE_DIR}/bench_*_compare.cpp")
else()
  file(GLOB _comparison_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/bench_*_compare.cpp")
endif()
list(SORT _comparison_sources)

foreach(_source IN LISTS _comparison_sources)
  get_filename_component(_target "${_source}" NAME_WE)

  # A source whose reference routine is LAPACK says so in a marker comment
  # ("// EIGEN_BENCH_REFERENCE_FAMILY: lapack"). Without a LAPACK to link it
  # is built Eigen-only: with the reference arm it would declare and call a
  # routine that does not resolve, and the link failure would lose both arms.
  file(STRINGS "${_source}" _family_line REGEX "EIGEN_BENCH_REFERENCE_FAMILY: *lapack")
  set(_has_reference ${EIGEN_BENCH_BLAS_FOUND})
  if(_family_line AND NOT EIGEN_BENCH_LAPACK_AVAILABLE)
    set(_has_reference FALSE)
    if(EIGEN_BENCH_BLAS_FOUND)
      message(STATUS "Comparison benchmarks: ${_target} is built Eigen-only; ${EIGEN_BENCH_BLAS_ARM} supplies no LAPACK here")
    endif()
  endif()

  set(_target_definitions ${_comparison_definitions})
  if(_has_reference)
    list(APPEND _target_definitions ${_comparison_reference_definitions})
  endif()
  eigen_add_benchmark(${_target} "${_source}" DEFINITIONS ${_target_definitions})
  if(_has_reference)
    target_link_libraries(${_target} PRIVATE ${EIGEN_BENCH_BLAS_LINK_LIBRARIES})
    if(_family_line)
      target_link_libraries(${_target} PRIVATE ${EIGEN_BENCH_LAPACK_LINK_LIBRARIES})
    endif()
    # An arm that changes how the library behaves from inside the binary (the
    # oneMKL vendor-check bypass) carries that code as a source of its own, so
    # the binary is the record of what was measured.
    if(EIGEN_BENCH_BLAS_ARM_SOURCES)
      target_sources(${_target} PRIVATE ${EIGEN_BENCH_BLAS_ARM_SOURCES})
    endif()
    if(EIGEN_BENCH_BLAS_ARM_LINK_OPTIONS)
      # Through target_link_libraries rather than target_link_options, which
      # needs CMake 3.13 where benchmarks/CMakeLists.txt promises 3.10.
      target_link_libraries(${_target} PRIVATE ${EIGEN_BENCH_BLAS_ARM_LINK_OPTIONS})
    endif()
  endif()
  add_dependencies(bench_comparison_all ${_target})

  # Smoke tests: every registered name is well formed, and one iteration of
  # every tiny cell runs, which validates the reference arm against Eigen.
  add_test(NAME ${_target}_names COMMAND ${_target} --benchmark_list_tests=true)
  set_tests_properties(${_target}_names PROPERTIES FAIL_REGULAR_EXPRESSION "[<>]" PASS_REGULAR_EXPRESSION "/eigen/")
  add_test(NAME ${_target}_smoke COMMAND ${_target} --benchmark_min_time=1x --benchmark_repetitions=1
           "--benchmark_filter=:(2|3|4|6|8|12|16)(/|$)")
  set_tests_properties(${_target}_smoke PROPERTIES TIMEOUT 300 FAIL_REGULAR_EXPRESSION "ERROR OCCURRED")
endforeach()

# ---------------------------------------------------------------------------
# What run.py needs to know about this build
# ---------------------------------------------------------------------------

function(_eigen_bench_json_escape value out)
  string(REPLACE "\\" "\\\\" value "${value}")
  string(REPLACE "\"" "\\\"" value "${value}")
  string(REPLACE "\n" "\\n" value "${value}")
  set(${out} "${value}" PARENT_SCOPE)
endfunction()

function(_eigen_bench_json_string_array values out)
  set(items "")
  foreach(value IN LISTS values)
    _eigen_bench_json_escape("${value}" escaped)
    list(APPEND items "\"${escaped}\"")
  endforeach()
  list(JOIN items ", " joined)
  set(${out} "[${joined}]" PARENT_SCOPE)
endfunction()

foreach(_name ARM DISPLAY_NAME LIBRARY_PATH NOTES)
  _eigen_bench_json_escape("${EIGEN_BENCH_BLAS_${_name}}" _json_${_name})
endforeach()
_eigen_bench_json_escape("${EIGEN_BENCH_REFERENCE_VERSION}" _json_version_fallback)
_eigen_bench_json_string_array("${EIGEN_BENCH_BLAS_LIBRARIES}" _json_libraries)
eigen_bench_resolve_linkables("${EIGEN_BENCH_LAPACK_LINK_LIBRARIES}" _json_lapack_paths)
_eigen_bench_json_string_array("${_json_lapack_paths}" _json_lapack_libraries)
if(EIGEN_BENCH_LAPACK_AVAILABLE)
  set(_json_lapack_available "true")
else()
  set(_json_lapack_available "false")
endif()
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/vendor_info.json" CONTENT "{
  \"arm\": \"${_json_ARM}\",
  \"library_name\": \"${_json_DISPLAY_NAME}\",
  \"library_version_fallback\": \"${_json_version_fallback}\",
  \"library_path\": \"${_json_LIBRARY_PATH}\",
  \"libraries\": ${_json_libraries},
  \"lapack\": {\"available\": ${_json_lapack_available}, \"provider\": \"${EIGEN_BENCH_LAPACK_PROVIDER}\", \"libraries\": ${_json_lapack_libraries}},
  \"notes\": \"${_json_NOTES}\",
  \"isa_target\": \"${EIGEN_BENCH_ISA_TARGET}\",
  \"eigen_commit\": \"${EIGEN_BENCH_EIGEN_COMMIT}\"
}
")
