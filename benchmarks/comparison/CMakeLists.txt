# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# Cross-library comparison benchmarks. See CONTRACTS.md for the name grammar,
# the result schema, and the harness CLIs; vendors.cmake for the vendor table.
#
# One executable per bench_*.cpp in this directory, named after the source stem.
# That is the same rule run.py applies to the `source` key of ops.toml
# (OpsRegistry.target_for), so the registry decides which operations are expected
# and the sources present decide what is built; the two are cross-checked below
# rather than one being derived from the other.

# ---------------------------------------------------------------------------
# Build metadata the benchmark binaries publish (CONTRACTS.md section 3.3)
# ---------------------------------------------------------------------------

set(_build_type "${CMAKE_BUILD_TYPE}")
if(NOT _build_type)
  set(_build_type "Release")
endif()
string(TOUPPER "${_build_type}" _build_type_upper)
# result_schema.json asks for the flags a reader can reconstruct the compile
# command from, so eigen_add_benchmark's own -O3 -DNDEBUG belong in the record --
# once each, since a Release configuration already carries them.
set(EIGEN_BENCH_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_build_type_upper}}")
foreach(_flag -O3 -DNDEBUG)
  if(NOT EIGEN_BENCH_CXX_FLAGS MATCHES "(^| )${_flag}( |$)")
    string(APPEND EIGEN_BENCH_CXX_FLAGS " ${_flag}")
  endif()
endforeach()
string(STRIP "${EIGEN_BENCH_CXX_FLAGS}" EIGEN_BENCH_CXX_FLAGS)

# ISA target key, using the ci/scripts naming that result_schema.json documents.
# It cannot be derived when the flags say -march=native, hence the cache override.
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
elseif(_processor MATCHES "ppc64le|powerpc64le")
  set(_isa_target "ppc64le-vsx")
else()
  set(_isa_target "${_processor}")
endif()
set(EIGEN_BENCH_ISA_TARGET "${_isa_target}" CACHE STRING
    "ISA target key recorded in the result provenance")

# The Eigen revision under test. Read-only git; untracked files are ignored
# because they cannot affect the compiled benchmark.
set(EIGEN_BENCH_EIGEN_COMMIT "")
set(EIGEN_BENCH_EIGEN_DIRTY "false")
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(COMMAND ${GIT_EXECUTABLE} --no-optional-locks -C "${EIGEN_SOURCE_DIR}" rev-parse HEAD
                  OUTPUT_VARIABLE EIGEN_BENCH_EIGEN_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET RESULT_VARIABLE _git_status)
  if(NOT _git_status EQUAL 0)
    set(EIGEN_BENCH_EIGEN_COMMIT "")
  endif()
  execute_process(COMMAND ${GIT_EXECUTABLE} --no-optional-locks -C "${EIGEN_SOURCE_DIR}"
                          status --porcelain --untracked-files=no
                  OUTPUT_VARIABLE _git_dirty OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET RESULT_VARIABLE _git_status)
  if(_git_status EQUAL 0 AND _git_dirty)
    set(EIGEN_BENCH_EIGEN_DIRTY "true")
  endif()
endif()

set(EIGEN_BENCH_OPS_TOML "${CMAKE_CURRENT_SOURCE_DIR}/ops.toml")
file(SHA256 "${EIGEN_BENCH_OPS_TOML}" EIGEN_BENCH_OPS_TOML_SHA256)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${EIGEN_BENCH_OPS_TOML}")

# ---------------------------------------------------------------------------
# Which operations ops.toml expects a binary for
# ---------------------------------------------------------------------------

# CMake has no TOML parser, and only three keys are needed -- `status` and
# `source` from an [ops.<KEY>] table and `kind` from its [ops.<KEY>.reference]
# sub-table -- so a line scan is enough. `kind` decides what is built, so a
# reformatted ops.toml this scan stopped reading would silently drop the
# reference arm of every LAPACK operation; tests/test_ops_registry.py mirrors the
# scan against a real TOML parse and fails when the two disagree.
file(READ "${EIGEN_BENCH_OPS_TOML}" _ops_content)
string(REPLACE ";" "\;" _ops_content "${_ops_content}")
# A CRLF checkout would otherwise leave \r on every line and the anchored table
# regex below would match nothing, reported as "no [ops.*] tables in ops.toml".
string(REGEX REPLACE "\r" "" _ops_content "${_ops_content}")
string(REPLACE "\n" ";" _ops_content "${_ops_content}")

set(_op_keys "")
set(_current_op "")
set(_current_reference_op "")
foreach(_line IN LISTS _ops_content)
  if(_line MATCHES "^\\[ops\\.([A-Z][A-Z0-9_]*)\\][ \t]*$")
    set(_current_op "${CMAKE_MATCH_1}")
    set(_current_reference_op "")
    list(APPEND _op_keys "${_current_op}")
    set(_op_${_current_op}_status "")
    set(_op_${_current_op}_source "")
    set(_op_${_current_op}_reference_kind "")
  elseif(_line MATCHES "^\\[ops\\.([A-Z][A-Z0-9_]*)\\.reference\\][ \t]*$")
    set(_current_op "")
    set(_current_reference_op "${CMAKE_MATCH_1}")
  elseif(_line MATCHES "^\\[")
    # Any other table header, including [ops.<KEY>.<sub>], ends the op's
    # top-level keys.
    set(_current_op "")
    set(_current_reference_op "")
  elseif(_current_op AND _line MATCHES "^status[ \t]*=[ \t]*\"([^\"]*)\"")
    set(_op_${_current_op}_status "${CMAKE_MATCH_1}")
  elseif(_current_op AND _line MATCHES "^source[ \t]*=[ \t]*\"([^\"]*)\"")
    set(_op_${_current_op}_source "${CMAKE_MATCH_1}")
  elseif(_current_reference_op AND _line MATCHES "^kind[ \t]*=[ \t]*\"([^\"]*)\"")
    # The interface family the reference routine lives in: "blas", "cblas",
    # "lapack", or "none" for an operation with no reference counterpart.
    set(_op_${_current_reference_op}_reference_kind "${CMAKE_MATCH_1}")
  endif()
endforeach()
if(NOT _op_keys)
  message(FATAL_ERROR "No [ops.*] tables found in ${EIGEN_BENCH_OPS_TOML}")
endif()

# ---------------------------------------------------------------------------
# Does the reference library answer a version query at runtime?
# ---------------------------------------------------------------------------

# bench_compare.h requires that EIGEN_BENCH_HAVE_<QUERY> be defined only
# after a check that actually links the symbol: defining it on a guess turns a
# missing query into a link failure. The declaration below deliberately does not
# match the real prototype -- it only has to resolve at link time, and the header
# declares the true signature in the translation unit that calls it.
set(_reference_version_define "")
if(EIGEN_BENCH_BLAS_FOUND AND EIGEN_BENCH_BLAS_VERSION_RUNTIME_SYMBOL)
  include(CheckCXXSourceCompiles)
  set(_symbol "${EIGEN_BENCH_BLAS_VERSION_RUNTIME_SYMBOL}")
  string(TOUPPER "${_symbol}" _symbol_upper)
  set(CMAKE_REQUIRED_LIBRARIES ${EIGEN_BENCH_BLAS_LINK_LIBRARIES})
  set(CMAKE_REQUIRED_QUIET TRUE)
  check_cxx_source_compiles(
    "extern \"C\" void ${_symbol}();\nint main() { ${_symbol}(); return 0; }"
    EIGEN_BENCH_LINKS_${_symbol_upper})
  unset(CMAKE_REQUIRED_LIBRARIES)
  unset(CMAKE_REQUIRED_QUIET)
  if(EIGEN_BENCH_LINKS_${_symbol_upper})
    set(_reference_version_define "EIGEN_BENCH_HAVE_${_symbol_upper}")
    message(STATUS "Comparison benchmarks: ${_symbol}() links; the binary reports the library's own version")
  else()
    message(STATUS "Comparison benchmarks: ${_symbol}() does not link; falling back to the configure-time version")
  endif()
endif()

# ---------------------------------------------------------------------------
# Compile definitions shared by every comparison target
# ---------------------------------------------------------------------------

# A definition whose value is a C string literal. The value may contain spaces
# and parentheses, so it is quoted here rather than stringized in the header;
# only EIGEN_BENCH_REFERENCE_ARM is passed as a bare token.
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
_eigen_bench_append_string_define(_comparison_definitions EIGEN_BENCH_EIGEN_DIRTY "${EIGEN_BENCH_EIGEN_DIRTY}")
_eigen_bench_append_string_define(_comparison_definitions EIGEN_BENCH_OPS_TOML_SHA256 "${EIGEN_BENCH_OPS_TOML_SHA256}")

# Separate from _comparison_definitions, which every target gets: these are what
# make a source register its reference arm, and the gate below withholds them
# from a target whose routines live in a family this build cannot call.
set(_comparison_reference_definitions "")

set(_comparison_arms eigen)
if(EIGEN_BENCH_BLAS_FOUND)
  list(APPEND _comparison_arms "${EIGEN_BENCH_BLAS_ARM}")
  # A bare token: bench_compare.h stringizes it, so no quoting has to
  # survive the CMake command line (CONTRACTS.md section 3.2).
  list(APPEND _comparison_reference_definitions "EIGEN_BENCH_REFERENCE_ARM=${EIGEN_BENCH_BLAS_ARM}")
  _eigen_bench_append_string_define(_comparison_reference_definitions EIGEN_BENCH_REFERENCE_LIBRARY_NAME "${EIGEN_BENCH_BLAS_DISPLAY_NAME}")
  _eigen_bench_append_string_define(_comparison_reference_definitions EIGEN_BENCH_REFERENCE_VERSION_FALLBACK "${EIGEN_BENCH_BLAS_VERSION}")
  _eigen_bench_append_string_define(_comparison_reference_definitions EIGEN_BENCH_REFERENCE_PATH "${EIGEN_BENCH_BLAS_LIBRARY_PATH}")
  _eigen_bench_append_string_define(_comparison_reference_definitions EIGEN_BENCH_REFERENCE_INTERFACE "${EIGEN_BENCH_BLAS_INTERFACE}")
  _eigen_bench_append_string_define(_comparison_reference_definitions EIGEN_BENCH_REFERENCE_THREADING "${EIGEN_BENCH_BLAS_THREADING}")
  # The Fortran BLAS integer width the reference arm must pass its arguments as.
  # Getting it wrong corrupts every argument silently, so it is derived from the
  # vendor entry that selected the library, never assumed.
  if(EIGEN_BENCH_BLAS_INTERFACE STREQUAL "ilp64")
    list(APPEND _comparison_reference_definitions "EIGEN_BENCH_REFERENCE_ILP64=1")
  endif()
  if(_reference_version_define)
    list(APPEND _comparison_reference_definitions "${_reference_version_define}=1")
  endif()
  if(EIGEN_BENCH_BLAS_COMPILE_DEFINITIONS)
    list(APPEND _comparison_reference_definitions ${EIGEN_BENCH_BLAS_COMPILE_DEFINITIONS})
  endif()
endif()

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

add_custom_target(bench_comparison_all COMMENT "All comparison benchmark binaries")

if(CMAKE_VERSION VERSION_LESS 3.12)
  file(GLOB _comparison_sources "${CMAKE_CURRENT_SOURCE_DIR}/bench_*.cpp")
else()
  file(GLOB _comparison_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/bench_*.cpp")
endif()
list(SORT _comparison_sources)

# Needed by two test registrations below (the ops.toml reconciliation and the
# harness suite), so it must be found before the target loop, not after it.
find_package(Python3 QUIET COMPONENTS Interpreter)

# LAPACK, for the operations whose reference is a LAPACK routine rather than a
# BLAS one (POTRF, GETRF, GEQRF, GESDD, SYEV in ops.toml). Most vendors ship both
# families in one library -- Accelerate, MKL and OpenBLAS all do, and their table
# rows say so in PROVIDES -- so BLAS::BLAS usually already resolves ?potrf and
# nothing extra is linked. netlib is the case that does not: its row declares
# `PROVIDES blas cblas`, and without this a LAPACK comparison against it would
# fail at link time with no explanation. Finding it here means the build can say
# whether the family is available; run.py turns a missing family into an explicit
# `reference_routine_absent` rather than a silent gap.
set(EIGEN_BENCH_LAPACK_AVAILABLE FALSE)
# Where ?potrf and friends come from: the BLAS vendor's own library, or a
# separately found LAPACK. A published row must not attribute a routine to the
# vendor named in its column header when a different package supplied it, so this
# is recorded rather than flattened into "the reference library".
set(EIGEN_BENCH_LAPACK_PROVIDER "")
if(EIGEN_BENCH_BLAS_FOUND)
  if("lapack" IN_LIST EIGEN_BENCH_BLAS_PROVIDES)
    # The vendor states it carries LAPACK in the library already linked.
    set(EIGEN_BENCH_LAPACK_AVAILABLE TRUE)
    set(EIGEN_BENCH_LAPACK_PROVIDER "vendor")
    set(EIGEN_BENCH_LAPACK_LINK_LIBRARIES "")
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
      message(STATUS "  lapack:  ${EIGEN_BENCH_LAPACK_LINK_LIBRARIES} (the reference BLAS does not carry it)")
    else()
      message(STATUS "  lapack:  not found; LAPACK-referenced operations will report reference_routine_absent")
    endif()
  endif()
endif()

# The interface families this build can actually CALL, as against the ones the
# vendor row declares: a vendor that ships no LAPACK gains the family when
# find_package(LAPACK) supplies one elsewhere. run.py turns an absent family into
# an explicit `reference_routine_absent` cell.
set(EIGEN_BENCH_REFERENCE_FAMILIES "")
if(EIGEN_BENCH_BLAS_FOUND)
  set(EIGEN_BENCH_REFERENCE_FAMILIES ${EIGEN_BENCH_BLAS_PROVIDES})
  if(EIGEN_BENCH_LAPACK_AVAILABLE AND NOT "lapack" IN_LIST EIGEN_BENCH_REFERENCE_FAMILIES)
    list(APPEND EIGEN_BENCH_REFERENCE_FAMILIES lapack)
  endif()
endif()

# Which operations each source carries, and therefore which reference families the
# binary built from it needs. ops.toml's `source` key is the same rule run.py
# applies (OpsRegistry.target_for), so the mapping is read from the registry, not
# guessed from the file name.
function(_eigen_bench_target_ops target out_var)
  set(_ops "")
  foreach(_op IN LISTS _op_keys)
    if(NOT _op_${_op}_source)
      continue()
    endif()
    get_filename_component(_op_target "${_op_${_op}_source}" NAME_WE)
    if(_op_target STREQUAL target)
      list(APPEND _ops "${_op}")
    endif()
  endforeach()
  set(${out_var} "${_ops}" PARENT_SCOPE)
endfunction()

set(_comparison_targets "")
foreach(_source IN LISTS _comparison_sources)
  get_filename_component(_target "${_source}" NAME_WE)

  # A source whose reference routine lives in a family this build cannot call --
  # ?potrf against a vendor shipping BLAS but no LAPACK -- must be built WITHOUT
  # the reference arm. With EIGEN_BENCH_REFERENCE_ARM it declares and calls a
  # routine that does not resolve, so the target fails to LINK and both arms of
  # the operation are reported build_failed, losing the Eigen arm too.
  set(_target_missing_families "")
  set(_target_reference_ops "")
  _eigen_bench_target_ops("${_target}" _target_ops_here)
  foreach(_op IN LISTS _target_ops_here)
    set(_kind "${_op_${_op}_reference_kind}")
    if(NOT _kind OR _kind STREQUAL "none")
      continue()
    endif()
    list(APPEND _target_reference_ops "${_op}")
    if(NOT _kind IN_LIST EIGEN_BENCH_REFERENCE_FAMILIES)
      list(APPEND _target_missing_families "${_kind}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _target_missing_families)

  set(_target_definitions ${_comparison_definitions})
  set(_target_has_reference FALSE)
  if(EIGEN_BENCH_BLAS_FOUND AND NOT _target_missing_families)
    set(_target_has_reference TRUE)
    list(APPEND _target_definitions ${_comparison_reference_definitions})
  elseif(EIGEN_BENCH_BLAS_FOUND)
    message(STATUS
      "Comparison benchmarks: ${_target} is built Eigen-only -- ${_target_reference_ops} needs "
      "${_target_missing_families}, which ${EIGEN_BENCH_BLAS_ARM} does not provide here. Its reference "
      "cells are reported as reference_routine_absent.")
    # The gate is per binary because a define is. A source mixing an operation
    # whose family is present with one whose family is not would lose BOTH
    # reference arms, so say so and split the source rather than publish the
    # narrower claim silently.
    set(_target_available_families "")
    foreach(_op IN LISTS _target_reference_ops)
      if("${_op_${_op}_reference_kind}" IN_LIST EIGEN_BENCH_REFERENCE_FAMILIES)
        list(APPEND _target_available_families "${_op}")
      endif()
    endforeach()
    if(_target_available_families)
      message(WARNING
        "ops.toml puts ${_target_available_families} in the same source as an operation whose reference "
        "family is unavailable, so their reference arms are dropped too. Give them their own bench_*.cpp.")
    endif()
  endif()

  eigen_add_benchmark(${_target} "${_source}" DEFINITIONS ${_target_definitions})
  if(_target_has_reference)
    target_link_libraries(${_target} PRIVATE ${EIGEN_BENCH_BLAS_LINK_LIBRARIES})
    if(EIGEN_BENCH_LAPACK_LINK_LIBRARIES)
      target_link_libraries(${_target} PRIVATE ${EIGEN_BENCH_LAPACK_LINK_LIBRARIES})
    endif()
    if(EIGEN_BENCH_BLAS_INCLUDE_DIRS)
      target_include_directories(${_target} PRIVATE ${EIGEN_BENCH_BLAS_INCLUDE_DIRS})
    endif()
    if(EIGEN_BENCH_BLAS_COMPILE_OPTIONS)
      target_compile_options(${_target} PRIVATE ${EIGEN_BENCH_BLAS_COMPILE_OPTIONS})
    endif()
  endif()
  set(_target_${_target}_has_reference ${_target_has_reference})
  add_dependencies(bench_comparison_all ${_target})
  list(APPEND _comparison_targets "${_target}")
endforeach()

if(NOT _comparison_targets)
  message(STATUS "Comparison benchmarks: no bench_*.cpp source in ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# An implemented operation whose source is absent produces no binary, and run.py
# would look for a target that was never created. Say so at configure time rather
# than at measurement time.
foreach(_op IN LISTS _op_keys)
  if(NOT _op_${_op}_status STREQUAL "implemented")
    continue()
  endif()
  set(_source "${_op_${_op}_source}")
  if(NOT _source)
    message(WARNING "ops.toml marks ${_op} implemented but names no source; run.py cannot resolve a target for it")
    continue()
  endif()
  get_filename_component(_expected_target "${_source}" NAME_WE)
  if(NOT _expected_target IN_LIST _comparison_targets)
    message(WARNING
      "ops.toml marks ${_op} implemented with source '${_source}', which run.py "
      "resolves to target '${_expected_target}', but no such source exists here. "
      "Built targets: ${_comparison_targets}")
  endif()
endforeach()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Shapes small enough that one iteration of every registered cell is a test and
# not a measurement.
set(EIGEN_BENCH_COMPARISON_SMOKE_FILTER ":(2|3|4|6|8|12|16)(/|$)"
    CACHE STRING "--benchmark_filter for the comparison smoke tests")
mark_as_advanced(EIGEN_BENCH_COMPARISON_SMOKE_FILTER)

foreach(_target IN LISTS _comparison_targets)
  # The names are the join key between the binaries and the reducer, so the
  # registration itself is what needs testing; --benchmark_list_tests prints every
  # registered name and measures nothing. An angle bracket in that output means a
  # BENCHMARK_TEMPLATE registration lost its ->Name(), which is exactly the
  # illegal-character case CONTRACTS.md section 1.1 forbids. The check stays on
  # '<>' alone because ctest matches the regex against stderr too, where Google
  # Benchmark's own warnings live.
  add_test(NAME ${_target}_names COMMAND ${_target} --benchmark_list_tests=true)
  set_tests_properties(${_target}_names PROPERTIES
    FAIL_REGULAR_EXPRESSION "[<>]"
    PASS_REGULAR_EXPRESSION "/eigen/")

  # Reconcile the registrations this binary actually carries against ops.toml,
  # in both directions. The pytest suite does the same against a committed
  # capture, which cannot notice that THIS binary's grid drifted -- regenerate
  # the capture from a drifted build and the snapshot check passes for good.
  if(Python3_Interpreter_FOUND)
    add_test(NAME ${_target}_reconciles
             COMMAND ${Python3_EXECUTABLE}
                     "${CMAKE_CURRENT_SOURCE_DIR}/tests/reconcile_registrations.py"
                     "$<TARGET_FILE:${_target}>"
                     --ops-toml "${CMAKE_CURRENT_SOURCE_DIR}/ops.toml")
  endif()

  add_test(NAME ${_target}_smoke COMMAND ${_target}
           --benchmark_min_time=1x
           --benchmark_repetitions=1
           "--benchmark_filter=${EIGEN_BENCH_COMPARISON_SMOKE_FILTER}")
  set_tests_properties(${_target}_smoke PROPERTIES TIMEOUT 300)
endforeach()

# The flop formulas every reported rate is scaled by. Not a benchmark: it has its
# own main() and links only the Google Benchmark headers that bench_common.h
# needs, so it must not go through eigen_add_benchmark.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_flops.cpp")
  add_executable(comparison_test_flops "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_flops.cpp")
  target_include_directories(comparison_test_flops PRIVATE ${EIGEN_SOURCE_DIR})
  target_link_libraries(comparison_test_flops PRIVATE benchmark::benchmark)
  add_test(NAME comparison_flops COMMAND comparison_test_flops)
endif()

# The cross-arm agreement predicate, which decides whether a reference kernel's
# output is a result or a defect. Same shape as the flop test above: its own
# main(), no GoogleTest, and no vendor library, since the predicate is pure.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_validation.cpp")
  add_executable(comparison_test_validation "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_validation.cpp")
  target_include_directories(comparison_test_validation PRIVATE ${EIGEN_SOURCE_DIR})
  target_link_libraries(comparison_test_validation PRIVATE benchmark::benchmark)
  add_test(NAME comparison_validation COMMAND comparison_test_validation)
endif()

# The harness's own tests, when a Python interpreter with pytest is available.
# An interpreter alone is not enough: `python3 -m pytest` on an image that has no
# pytest turns a missing optional developer dependency into a red build for work
# that has nothing to do with Python. jsonschema and matplotlib stay optional --
# the suite skips the cases that need them (tests/conftest.py).
if(CMAKE_VERSION VERSION_LESS 3.12)
  file(GLOB _harness_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.py")
else()
  file(GLOB _harness_tests CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.py")
endif()
if(Python3_FOUND AND _harness_tests)
  execute_process(COMMAND ${Python3_EXECUTABLE} -m pytest --version
                  RESULT_VARIABLE _pytest_status OUTPUT_QUIET ERROR_QUIET)
  if(_pytest_status EQUAL 0)
    add_test(NAME comparison_harness_python
             COMMAND ${Python3_EXECUTABLE} -m pytest -q "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    set_tests_properties(comparison_harness_python PROPERTIES
      WORKING_DIRECTORY "${EIGEN_SOURCE_DIR}")
  else()
    message(STATUS "Comparison benchmarks: ${Python3_EXECUTABLE} has no pytest; the harness tests are not registered")
  endif()
endif()

# ---------------------------------------------------------------------------
# Metadata for run.py
# ---------------------------------------------------------------------------

# run.py must not have to guess what the build decided. Everything it needs --
# which arms exist, which binary carries which operation, and how the reference
# library identifies itself -- is written to comparison/vendor_info.json in the
# build tree at generate time.
eigen_bench_json_string_array("${_comparison_arms}" _json_arms)
eigen_bench_json_string_array("${EIGEN_BENCH_BLAS_LIBRARIES}" _json_libraries)
eigen_bench_json_string_array("${EIGEN_BENCH_BLAS_INCLUDE_DIRS}" _json_include_dirs)
eigen_bench_json_string_array("${EIGEN_BENCH_BLAS_PROVIDES}" _json_provides)
eigen_bench_json_string_array("${EIGEN_BENCH_REFERENCE_FAMILIES}" _json_families)
eigen_bench_json_string_array("${EIGEN_BENCH_LAPACK_LINK_LIBRARIES}" _json_lapack_libraries)
if(EIGEN_BENCH_LAPACK_AVAILABLE)
  set(_json_lapack_available "true")
else()
  set(_json_lapack_available "false")
endif()

set(_json_targets "")
foreach(_target IN LISTS _comparison_targets)
  set(_target_ops "")
  foreach(_op IN LISTS _op_keys)
    if(_op_${_op}_status STREQUAL "implemented" AND _op_${_op}_source)
      get_filename_component(_expected_target "${_op_${_op}_source}" NAME_WE)
      if(_expected_target STREQUAL _target)
        list(APPEND _target_ops "${_op}")
      endif()
    endif()
  endforeach()
  eigen_bench_json_string_array("${_target_ops}" _json_target_ops)
  # Per binary, because the gate is: a target built Eigen-only carries no
  # reference registrations however the rest of this file describes the vendor.
  if(_target_${_target}_has_reference)
    set(_json_target_reference_arm "${EIGEN_BENCH_BLAS_ARM}")
  else()
    set(_json_target_reference_arm "")
  endif()
  list(APPEND _json_targets
"    {
      \"target\": \"${_target}\",
      \"ops\": ${_json_target_ops},
      \"reference_arm\": \"${_json_target_reference_arm}\",
      \"executable\": \"$<TARGET_FILE:${_target}>\"
    }")
endforeach()
list(JOIN _json_targets ",\n" _json_targets)

foreach(_name DISPLAY_NAME BLA_VENDOR ARM VERSION VERSION_SOURCE VERSION_RUNTIME
              LIBRARY_PATH INTERFACE THREADING THREAD_ENV CBLAS_HEADER NOTES)
  eigen_bench_json_escape("${EIGEN_BENCH_BLAS_${_name}}" _json_${_name})
endforeach()
eigen_bench_json_escape("${EIGEN_BENCH_CXX_FLAGS}" _json_cxx_flags)
eigen_bench_json_escape("${CMAKE_CXX_COMPILER}" _json_cxx_compiler)
if(EIGEN_BENCH_BLAS_FOUND)
  set(_json_reference_available "true")
else()
  set(_json_reference_available "false")
endif()

set(_vendor_info "{
  \"schema_version\": \"1.0.0\",
  \"generated_by\": \"benchmarks/comparison/CMakeLists.txt\",
  \"build_dir\": \"${CMAKE_BINARY_DIR}\",
  \"build_target\": \"bench_comparison_all\",
  \"cmake_build_type\": \"${_build_type}\",
  \"cxx_compiler_id\": \"${CMAKE_CXX_COMPILER_ID}\",
  \"cxx_compiler_version\": \"${CMAKE_CXX_COMPILER_VERSION}\",
  \"cxx_compiler_path\": \"${_json_cxx_compiler}\",
  \"cxx_standard\": ${CMAKE_CXX_STANDARD},
  \"cxx_flags\": \"${_json_cxx_flags}\",
  \"isa_target\": \"${EIGEN_BENCH_ISA_TARGET}\",
  \"eigen_commit\": \"${EIGEN_BENCH_EIGEN_COMMIT}\",
  \"eigen_dirty\": ${EIGEN_BENCH_EIGEN_DIRTY},
  \"ops_toml_sha256\": \"${EIGEN_BENCH_OPS_TOML_SHA256}\",
  \"arms\": ${_json_arms},
  \"reference\": {
    \"available\": ${_json_reference_available},
    \"lapack_available\": ${_json_lapack_available},
    \"arm\": \"${_json_ARM}\",
    \"library_name\": \"${_json_DISPLAY_NAME}\",
    \"library_version\": \"${_json_VERSION}\",
    \"version_source\": \"${_json_VERSION_SOURCE}\",
    \"version_runtime_query\": \"${_json_VERSION_RUNTIME}\",
    \"library_path\": \"${_json_LIBRARY_PATH}\",
    \"libraries\": ${_json_libraries},
    \"include_dirs\": ${_json_include_dirs},
    \"cblas_header\": \"${_json_CBLAS_HEADER}\",
    \"bla_vendor\": \"${_json_BLA_VENDOR}\",
    \"interface\": \"${_json_INTERFACE}\",
    \"threading_model\": \"${_json_THREADING}\",
    \"thread_env\": \"${_json_THREAD_ENV}\",
    \"provides\": ${_json_provides},
    \"families\": ${_json_families},
    \"lapack\": {
      \"available\": ${_json_lapack_available},
      \"provider\": \"${EIGEN_BENCH_LAPACK_PROVIDER}\",
      \"libraries\": ${_json_lapack_libraries}
    },
    \"notes\": \"${_json_NOTES}\"
  },
  \"targets\": [
${_json_targets}
  ]
}
")

if(CMAKE_CONFIGURATION_TYPES)
  set(_vendor_info_path "${CMAKE_CURRENT_BINARY_DIR}/vendor_info-$<CONFIG>.json")
else()
  set(_vendor_info_path "${CMAKE_CURRENT_BINARY_DIR}/vendor_info.json")
endif()
file(GENERATE OUTPUT "${_vendor_info_path}" CONTENT "${_vendor_info}")
