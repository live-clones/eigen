# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# Vendor table for the cross-library benchmark comparison harness.
#
# Adding a BLAS implementation is ONE eigen_bench_declare_vendor() call below and
# nothing else: no new logic here, no change in comparison/CMakeLists.txt, no
# change in run.py. The declared key is the `arm` field of the benchmark name
# grammar in CONTRACTS.md section 1, so it must match ^[a-z][a-z0-9_]*$.
#
# Each entry answers the six questions the harness asks of a reference library:
#   1. how does find_package(BLAS) locate it            -> BLA_VENDOR / PACKAGE
#   2. what else must be found first                    -> PACKAGE, PACKAGE_TARGETS
#   3. where is its cblas.h                             -> CBLAS_HEADER + hints
#   4. what does a target link against                  -> BLAS::BLAS / EXTRA_LIBRARIES
#   5. how is its version obtained                      -> VERSION_* probes
#   6. which env var caps its thread count              -> THREAD_ENV
#
# The version probe matters: result_schema.json makes `library_version` mandatory
# for a reference arm because vendors change kernels without changing the API.
# CMake supplies the configure-time answer as EIGEN_BENCH_REFERENCE_VERSION_FALLBACK;
# a library with a runtime query (VERSION_RUNTIME) overrides it in the binary.

include_guard(GLOBAL)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

set(EIGEN_BENCH_VENDOR_FIELDS
    DISPLAY_NAME        # human-readable name, lands in provenance.arms[].library_name
    ALIASES             # additional spellings accepted by EIGEN_BENCH_BLAS_VENDOR
    BLA_VENDOR          # BLA_VENDOR candidates, tried in order
    PACKAGE             # optional config package to find first
    PACKAGE_TARGETS     # imported targets that package provides
    EXTRA_LIBRARIES     # extra link libraries (on top of BLAS::BLAS)
    COMPILE_DEFINITIONS # definitions any target using this vendor's headers needs
    COMPILE_OPTIONS     # ditto, for options
    CBLAS_HEADER        # header the C++ arm includes, normally cblas.h
    CBLAS_HINTS         # extra directories to search for it
    CBLAS_PATH_SUFFIXES # PATH_SUFFIXES for that search
    INTERFACE_WIDTH     # lp64 | ilp64
    THREADING           # sequential | openmp | pthreads | tbb | gcd
    THREAD_ENV          # environment variable that caps the thread count
    PROVIDES            # subset of blas cblas lapack lapacke
    LIBRARY_PATH_FALLBACK  # used when BLAS_LIBRARIES holds no absolute path
    VERSION_PKGCONFIG   # pkg-config module name
    VERSION_HEADER      # header carrying the version
    VERSION_HEADER_REGEX  # regex with one capture group applied to that header
    VERSION_COMMAND     # command whose stdout carries the version
    VERSION_COMMAND_REGEX # regex with one capture group applied to that stdout
    VERSION_TEMPLATE    # decorates the raw version, @VERSION@ is substituted
    VERSION_RUNTIME     # runtime query the C++ arm should prefer, recorded verbatim
    VERSION_RUNTIME_SYMBOL # symbol implementing it; link-checked before it is used
    PLATFORM            # CMake variable that must be true for this vendor
    MIN_CMAKE           # minimum CMake version whose FindBLAS knows this vendor
    AUTO_PRIORITY       # rank in the auto-detection order; lower is tried first
    NOTES)

function(eigen_bench_declare_vendor key)
  cmake_parse_arguments(V "" "" "${EIGEN_BENCH_VENDOR_FIELDS}" ${ARGN})
  if(V_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "eigen_bench_declare_vendor(${key}): unknown field(s) ${V_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT key MATCHES "^[a-z][a-z0-9_]*$")
    message(FATAL_ERROR "eigen_bench_declare_vendor(${key}): arm keys must match ^[a-z][a-z0-9_]*$")
  endif()
  foreach(field IN LISTS EIGEN_BENCH_VENDOR_FIELDS)
    set_property(GLOBAL PROPERTY EIGEN_BENCH_VENDOR_${key}_${field} "${V_${field}}")
  endforeach()
  set_property(GLOBAL APPEND PROPERTY EIGEN_BENCH_VENDOR_KEYS "${key}")
endfunction()

# eigen_bench_vendor_get(<key> <FIELD> <out-var>)
function(eigen_bench_vendor_get key field out)
  get_property(value GLOBAL PROPERTY EIGEN_BENCH_VENDOR_${key}_${field})
  set(${out} "${value}" PARENT_SCOPE)
endfunction()

function(eigen_bench_vendor_keys out)
  get_property(keys GLOBAL PROPERTY EIGEN_BENCH_VENDOR_KEYS)
  set(${out} "${keys}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

# Apple ships cblas.h inside the framework rather than on the default include
# path, so `#include <cblas.h>` only compiles with this directory added by hand.
# CMAKE_OSX_SYSROOT is the SDK CMake itself compiles against; fall back to xcrun
# and finally to the system framework for a sysroot-less configuration.
set(_eigen_bench_accelerate_hints "")
if(APPLE)
  set(_eigen_bench_sdk "${CMAKE_OSX_SYSROOT}")
  if(NOT _eigen_bench_sdk)
    find_program(EIGEN_BENCH_XCRUN xcrun)
    mark_as_advanced(EIGEN_BENCH_XCRUN)
    if(EIGEN_BENCH_XCRUN)
      execute_process(COMMAND ${EIGEN_BENCH_XCRUN} --show-sdk-path
                      OUTPUT_VARIABLE _eigen_bench_sdk OUTPUT_STRIP_TRAILING_WHITESPACE
                      ERROR_QUIET)
    endif()
  endif()
  set(_eigen_bench_veclib "System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers")
  if(_eigen_bench_sdk)
    list(APPEND _eigen_bench_accelerate_hints "${_eigen_bench_sdk}/${_eigen_bench_veclib}")
  endif()
  list(APPEND _eigen_bench_accelerate_hints "/${_eigen_bench_veclib}")
endif()

eigen_bench_declare_vendor(accelerate
  AUTO_PRIORITY 10
  DISPLAY_NAME "Apple Accelerate"
  ALIASES apple veclib nas
  BLA_VENDOR Apple
  PLATFORM APPLE
  CBLAS_HEADER cblas.h
  CBLAS_HINTS ${_eigen_bench_accelerate_hints}
  # cblas_* carries a deprecation attribute since macOS 13.3 pointing at the
  # ILP64 headers; the LP64 entry points remain the supported ones and are what
  # the lp64 arm must measure, so the diagnostic is silenced rather than obeyed.
  COMPILE_OPTIONS $<$<CXX_COMPILER_ID:AppleClang,Clang>:-Wno-deprecated-declarations>
  INTERFACE_WIDTH lp64
  THREADING gcd
  THREAD_ENV VECLIB_MAXIMUM_THREADS
  PROVIDES blas cblas lapack
  LIBRARY_PATH_FALLBACK "/System/Library/Frameworks/Accelerate.framework"
  VERSION_COMMAND sw_vers -productVersion
  VERSION_TEMPLATE "Accelerate (macOS @VERSION@)"
  NOTES "Accelerate exposes no version query; the macOS product version is the only identifier of the shipped kernels.")

eigen_bench_declare_vendor(openblas
  AUTO_PRIORITY 60
  DISPLAY_NAME "OpenBLAS"
  ALIASES open_blas goto
  BLA_VENDOR OpenBLAS
  CBLAS_HEADER cblas.h
  CBLAS_HINTS $ENV{OPENBLAS_HOME} $ENV{OPENBLAS_ROOT} $ENV{OpenBLAS_HOME}
              /opt/homebrew/opt/openblas /usr/local/opt/openblas /opt/OpenBLAS /usr
  CBLAS_PATH_SUFFIXES include include/openblas openblas
  INTERFACE_WIDTH lp64
  THREADING pthreads
  THREAD_ENV OPENBLAS_NUM_THREADS
  PROVIDES blas cblas lapack lapacke
  VERSION_PKGCONFIG openblas
  VERSION_HEADER openblas_config.h
  VERSION_HEADER_REGEX "OPENBLAS_VERSION[ \t]+\" *OpenBLAS ([^ \"]+)"
  VERSION_TEMPLATE "OpenBLAS @VERSION@"
  VERSION_RUNTIME "openblas_get_config()"
  VERSION_RUNTIME_SYMBOL openblas_get_config)

eigen_bench_declare_vendor(mkl
  AUTO_PRIORITY 40
  DISPLAY_NAME "Intel oneMKL"
  ALIASES intel onemkl intel10_64lp_seq
  # Sequential LP64 first: a comparison run pins threads itself, and a threaded
  # MKL silently ignoring MKL_NUM_THREADS=1 would make the eigen arm look slow.
  BLA_VENDOR Intel10_64lp_seq Intel10_64lp Intel10_64_dyn
  CBLAS_HEADER mkl_cblas.h
  CBLAS_HINTS $ENV{MKLROOT} $ENV{ONEAPI_ROOT}/mkl/latest /opt/intel/oneapi/mkl/latest
  CBLAS_PATH_SUFFIXES include
  INTERFACE_WIDTH lp64
  THREADING sequential
  THREAD_ENV MKL_NUM_THREADS
  PROVIDES blas cblas lapack lapacke
  VERSION_PKGCONFIG mkl-dynamic-lp64-seq
  VERSION_HEADER mkl_version.h
  VERSION_HEADER_REGEX "__INTEL_MKL_BUILD_DATE[ \t]+([0-9]+)"
  VERSION_TEMPLATE "Intel oneMKL @VERSION@"
  VERSION_RUNTIME "mkl_get_version_string()"
  VERSION_RUNTIME_SYMBOL mkl_get_version_string)

eigen_bench_declare_vendor(aocl
  AUTO_PRIORITY 50
  DISPLAY_NAME "AOCL-BLIS"
  ALIASES amd aocl_mt
  # FindBLAS learned AOCL in 3.27; FLAME (plain BLIS) is the older spelling and
  # resolves the same libblis on an AOCL install.
  BLA_VENDOR AOCL AOCL_mt FLAME
  CBLAS_HEADER cblas.h
  CBLAS_HINTS $ENV{AOCL_ROOT} $ENV{BLIS_HOME} /opt/AMD/aocl /opt/aocl
  CBLAS_PATH_SUFFIXES include include/blis include/amdzen blis
  INTERFACE_WIDTH lp64
  THREADING openmp
  THREAD_ENV BLIS_NUM_THREADS
  PROVIDES blas cblas lapack
  VERSION_PKGCONFIG blis
  VERSION_HEADER blis.h
  VERSION_HEADER_REGEX "BLIS_VERSION_STR[ \t]+\"([^\"]+)\""
  VERSION_TEMPLATE "AOCL-BLIS @VERSION@"
  VERSION_RUNTIME "bli_info_get_version_str()"
  VERSION_RUNTIME_SYMBOL bli_info_get_version_str)

eigen_bench_declare_vendor(blis
  AUTO_PRIORITY 70
  DISPLAY_NAME "BLIS"
  ALIASES flame
  BLA_VENDOR FLAME
  CBLAS_HEADER cblas.h
  CBLAS_HINTS $ENV{BLIS_HOME} /opt/blis /usr/local /usr
  CBLAS_PATH_SUFFIXES include include/blis blis
  INTERFACE_WIDTH lp64
  THREADING openmp
  THREAD_ENV BLIS_NUM_THREADS
  PROVIDES blas cblas
  VERSION_PKGCONFIG blis
  VERSION_HEADER blis.h
  VERSION_HEADER_REGEX "BLIS_VERSION_STR[ \t]+\"([^\"]+)\""
  VERSION_TEMPLATE "BLIS @VERSION@"
  VERSION_RUNTIME "bli_info_get_version_str()"
  VERSION_RUNTIME_SYMBOL bli_info_get_version_str)

eigen_bench_declare_vendor(armpl
  AUTO_PRIORITY 20
  DISPLAY_NAME "Arm Performance Libraries"
  ALIASES arm arm_mp
  BLA_VENDOR Arm Arm_mp
  MIN_CMAKE 3.18
  CBLAS_HEADER cblas.h
  CBLAS_HINTS $ENV{ARMPL_DIR} $ENV{ARMPL_ROOT} $ENV{ARMPL_HOME} /opt/arm/armpl
  CBLAS_PATH_SUFFIXES include include_lp64 include/armpl armpl
  INTERFACE_WIDTH lp64
  THREADING openmp
  THREAD_ENV ARMPL_NUM_THREADS
  PROVIDES blas cblas lapack
  VERSION_PKGCONFIG armpl
  VERSION_HEADER armpl.h
  VERSION_HEADER_REGEX "ARMPL_BUILD_STRING[ \t]+\"([^\"]+)\""
  VERSION_TEMPLATE "ArmPL @VERSION@"
  VERSION_RUNTIME "armplversion()"
  VERSION_RUNTIME_SYMBOL armplversion)

eigen_bench_declare_vendor(nvpl
  AUTO_PRIORITY 30
  DISPLAY_NAME "NVPL BLAS"
  ALIASES nvidia nvpl_blas
  # FindBLAS learned NVPL in 4.1; the shipped config package works everywhere and
  # is tried first.
  BLA_VENDOR NVPL
  PACKAGE nvpl_blas
  PACKAGE_TARGETS nvpl::blas_lp64_seq
  MIN_CMAKE 4.1
  CBLAS_HEADER nvpl_blas_cblas.h
  CBLAS_HINTS $ENV{NVPL_ROOT} $ENV{NVPL_BLAS_ROOT} /opt/nvidia/nvpl
  CBLAS_PATH_SUFFIXES include include/nvpl nvpl
  INTERFACE_WIDTH lp64
  THREADING sequential
  THREAD_ENV NVPL_BLAS_NUM_THREADS
  PROVIDES blas cblas lapack
  VERSION_PKGCONFIG nvpl_blas
  VERSION_HEADER nvpl_blas.h
  VERSION_HEADER_REGEX "NVPL_BLAS_VERSION_STRING[ \t]+\"([^\"]+)\""
  VERSION_TEMPLATE "NVPL BLAS @VERSION@"
  VERSION_RUNTIME "nvpl_blas_get_version()"
  VERSION_RUNTIME_SYMBOL nvpl_blas_get_version)

eigen_bench_declare_vendor(netlib
  AUTO_PRIORITY 90
  DISPLAY_NAME "Netlib reference BLAS"
  ALIASES generic reference refblas
  BLA_VENDOR Generic
  CBLAS_HEADER cblas.h
  CBLAS_HINTS /usr/local /usr
  CBLAS_PATH_SUFFIXES include include/cblas cblas
  INTERFACE_WIDTH lp64
  THREADING sequential
  THREAD_ENV OMP_NUM_THREADS
  PROVIDES blas cblas
  VERSION_PKGCONFIG cblas
  VERSION_TEMPLATE "Netlib reference BLAS @VERSION@"
  NOTES "The reference implementation exposes no version query; the packaged version is the best available identifier.")

# Not in the table: `eigenblas`, Eigen's own blas/ shim. It is an arm key in the
# name grammar, but it exports only the Fortran ABI, no cblas.h, so it cannot be
# driven through the CBLAS calls the reference arm makes. Adding it needs a
# Fortran-ABI arm in the C++ layer first, not another row here.

# Auto-detection order. Deliberately platform-first: on a machine that has both,
# the vendor tuned for that hardware is the meaningful reference. The ranking is
# a per-row AUTO_PRIORITY rather than a second list of keys here, so that adding
# a vendor really is one eigen_bench_declare_vendor() call: a hand-maintained
# list would leave a newly declared key out of `auto` entirely, and being a CACHE
# entry it would also stay stale in every already-configured build tree.
set(EIGEN_BENCH_VENDOR_AUTO_ORDER "" CACHE STRING
    "Vendor keys tried first when EIGEN_BENCH_BLAS_VENDOR=auto; empty means the \
declared AUTO_PRIORITY order. Declared keys this list omits are still tried, after it.")
mark_as_advanced(EIGEN_BENCH_VENDOR_AUTO_ORDER)

# Left-pad a non-negative integer to a fixed width for lexicographic sorting.
function(_eigen_bench_zero_pad value out)
  set(padded "00000000${value}")
  string(LENGTH "${padded}" length)
  math(EXPR start "${length} - 8")
  string(SUBSTRING "${padded}" ${start} 8 padded)
  set(${out} "${padded}" PARENT_SCOPE)
endfunction()

# eigen_bench_vendor_auto_order(<out>)
#
# Every declared key, most preferred first: the user's override (if any) first in
# the order given, then the rest by AUTO_PRIORITY and declaration order. A key
# that is not declared is dropped here with a warning rather than at each use.
function(eigen_bench_vendor_auto_order out)
  eigen_bench_vendor_keys(keys)
  set(ranked "")
  set(index 0)
  foreach(key IN LISTS keys)
    eigen_bench_vendor_get(${key} AUTO_PRIORITY priority)
    if(NOT priority MATCHES "^[0-9]+$")
      # An undeclared priority sorts after every declared one but before the
      # last resort, so a new row is reachable in auto mode without editing it in.
      set(priority 500)
    endif()
    math(EXPR index "${index} + 1")
    # Zero-padded so that plain lexicographic list(SORT) -- the only sort CMake
    # 3.10 has -- orders numerically; the index makes it stable.
    _eigen_bench_zero_pad("${priority}" priority_padded)
    _eigen_bench_zero_pad("${index}" index_padded)
    list(APPEND ranked "${priority_padded}:${index_padded}:${key}")
  endforeach()
  list(SORT ranked)

  set(ordered "")
  foreach(key IN LISTS EIGEN_BENCH_VENDOR_AUTO_ORDER)
    if(key IN_LIST keys)
      list(APPEND ordered "${key}")
    else()
      message(WARNING "EIGEN_BENCH_VENDOR_AUTO_ORDER names unknown vendor '${key}'; ignored")
    endif()
  endforeach()
  foreach(entry IN LISTS ranked)
    string(REGEX REPLACE "^[0-9]+:[0-9]+:" "" key "${entry}")
    if(NOT key IN_LIST ordered)
      list(APPEND ordered "${key}")
    endif()
  endforeach()
  set(${out} "${ordered}" PARENT_SCOPE)
endfunction()

set(EIGEN_BENCH_BLAS_VENDOR "auto" CACHE STRING
    "Reference BLAS for the comparison benchmarks: auto, none, or a vendor key/alias")
eigen_bench_vendor_keys(_eigen_bench_all_keys)
set_property(CACHE EIGEN_BENCH_BLAS_VENDOR PROPERTY STRINGS auto none ${_eigen_bench_all_keys})

# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

# Resolve a user-supplied spelling to a table key. Sets <out> to the key, or to
# the input unchanged when nothing matches, so the caller can report it.
function(eigen_bench_normalize_vendor request out)
  string(TOLOWER "${request}" needle)
  eigen_bench_vendor_keys(keys)
  foreach(key IN LISTS keys)
    if(needle STREQUAL key)
      set(${out} "${key}" PARENT_SCOPE)
      return()
    endif()
    eigen_bench_vendor_get(${key} ALIASES aliases)
    eigen_bench_vendor_get(${key} BLA_VENDOR bla_vendors)
    foreach(alias IN LISTS aliases bla_vendors)
      string(TOLOWER "${alias}" alias)
      if(needle STREQUAL alias)
        set(${out} "${key}" PARENT_SCOPE)
        return()
      endif()
    endforeach()
  endforeach()
  set(${out} "${needle}" PARENT_SCOPE)
endfunction()

# Locate the vendor's CBLAS header directory. `libraries` is BLAS_LIBRARIES, from
# which <libdir>/../include is derived: that covers every relocatable install
# (spack, conda, homebrew, a manual --prefix) without a per-install hint.
function(_eigen_bench_find_cblas_include key libraries out)
  eigen_bench_vendor_get(${key} CBLAS_HEADER header)
  if(NOT header)
    set(header cblas.h)
  endif()
  eigen_bench_vendor_get(${key} CBLAS_HINTS hints)
  eigen_bench_vendor_get(${key} CBLAS_PATH_SUFFIXES suffixes)
  if(NOT suffixes)
    set(suffixes include)
  endif()
  foreach(lib IN LISTS libraries)
    if(IS_ABSOLUTE "${lib}" AND EXISTS "${lib}")
      get_filename_component(libdir "${lib}" DIRECTORY)
      get_filename_component(prefix "${libdir}" DIRECTORY)
      list(APPEND hints "${prefix}" "${libdir}")
    endif()
  endforeach()
  set(cache_var EIGEN_BENCH_CBLAS_INCLUDE_DIR_${key})
  find_path(${cache_var} NAMES ${header} HINTS ${hints}
            PATH_SUFFIXES ${suffixes} "" NO_DEFAULT_PATH)
  if(NOT ${cache_var})
    find_path(${cache_var} NAMES ${header} HINTS ${hints} PATH_SUFFIXES ${suffixes} "")
  endif()
  mark_as_advanced(${cache_var})
  # Empty rather than find_path's <VAR>-NOTFOUND sentinel: the answer is recorded
  # verbatim in vendor_info.json's include_dirs, where the sentinel would read as
  # a directory of that name.
  if(${cache_var})
    set(${out} "${${cache_var}}" PARENT_SCOPE)
  else()
    set(${out} "" PARENT_SCOPE)
  endif()
endfunction()

# Version probe: pkg-config, then the vendor header, then a command. Sets
# <out_version> to "" when every probe fails, which the caller turns into
# "unknown" plus a provenance gap.
function(_eigen_bench_probe_version key include_dir out_version out_source)
  set(raw "")
  set(source "")

  eigen_bench_vendor_get(${key} VERSION_PKGCONFIG pc_module)
  if(pc_module)
    find_program(EIGEN_BENCH_PKG_CONFIG NAMES pkg-config pkgconf)
    mark_as_advanced(EIGEN_BENCH_PKG_CONFIG)
    if(EIGEN_BENCH_PKG_CONFIG)
      execute_process(COMMAND ${EIGEN_BENCH_PKG_CONFIG} --modversion ${pc_module}
                      OUTPUT_VARIABLE raw OUTPUT_STRIP_TRAILING_WHITESPACE
                      ERROR_QUIET RESULT_VARIABLE status)
      if(status EQUAL 0 AND raw)
        set(source "pkg-config ${pc_module}")
      else()
        set(raw "")
      endif()
    endif()
  endif()

  eigen_bench_vendor_get(${key} VERSION_HEADER header)
  if(NOT raw AND header AND include_dir)
    eigen_bench_vendor_get(${key} VERSION_HEADER_REGEX regex)
    set(header_path "")
    if(EXISTS "${include_dir}/${header}")
      set(header_path "${include_dir}/${header}")
    else()
      file(GLOB header_candidates "${include_dir}/*/${header}")
      if(header_candidates)
        list(GET header_candidates 0 header_path)
      endif()
    endif()
    if(header_path AND regex)
      file(STRINGS "${header_path}" matched REGEX "${regex}" LIMIT_COUNT 1)
      if(matched AND matched MATCHES "${regex}")
        set(raw "${CMAKE_MATCH_1}")
        set(source "${header_path}")
      endif()
    endif()
  endif()

  eigen_bench_vendor_get(${key} VERSION_COMMAND command)
  if(NOT raw AND command)
    eigen_bench_vendor_get(${key} VERSION_COMMAND_REGEX regex)
    execute_process(COMMAND ${command} OUTPUT_VARIABLE output
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
                    RESULT_VARIABLE status)
    if(status EQUAL 0 AND output)
      if(regex)
        if(output MATCHES "${regex}")
          set(raw "${CMAKE_MATCH_1}")
        endif()
      else()
        string(REGEX REPLACE "[\r\n].*" "" raw "${output}")
      endif()
      if(raw)
        string(REPLACE ";" " " command_string "${command}")
        set(source "${command_string}")
      endif()
    endif()
  endif()

  if(raw)
    eigen_bench_vendor_get(${key} VERSION_TEMPLATE template)
    if(template)
      string(REPLACE "@VERSION@" "${raw}" raw "${template}")
    endif()
  endif()
  set(${out_version} "${raw}" PARENT_SCOPE)
  set(${out_source} "${source}" PARENT_SCOPE)
endfunction()

# eigen_bench_find_cblas_include(<blas-libraries> <out-dirs> <out-vendor-key>)
#
# For a BLAS that was found WITHOUT going through the table -- benchmarks/Tuning's
# bench_blas_gemm links whatever find_package(BLAS) reports -- ask each vendor
# entry in turn where its CBLAS header lives and take the first answer. Without
# this the include of <cblas.h> only compiles where the header happens to sit on
# the default include path, which on macOS is nowhere.
function(eigen_bench_find_cblas_include libraries out_dirs out_key)
  eigen_bench_vendor_auto_order(ordered)
  foreach(key IN LISTS ordered)
    eigen_bench_vendor_get(${key} PLATFORM platform)
    if(platform AND NOT ${platform})
      continue()
    endif()
    _eigen_bench_find_cblas_include(${key} "${libraries}" dir)
    if(dir)
      set(${out_dirs} "${dir}" PARENT_SCOPE)
      set(${out_key} "${key}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${out_dirs} "" PARENT_SCOPE)
  set(${out_key} "" PARENT_SCOPE)
endfunction()

# _eigen_bench_blas_link_libraries(<blas-libraries> <blas-linker-flags> <out>)
#
# What a target must link for the BLAS find_package(BLAS) JUST returned.
#
# FindBLAS creates BLAS::BLAS on its first success and never revises it, so in
# auto mode the target can still describe an EARLIER candidate that was tried and
# then rejected. Linking it there would build against one library while every
# provenance field -- arm key, library_name, library_version, library_path --
# named another, which is exactly the mislabelling this harness exists to
# prevent, and nothing downstream could detect it. The target is therefore
# adopted only when it agrees with this call's own BLAS_LIBRARIES.
function(_eigen_bench_blas_link_libraries libraries linker_flags out)
  if(TARGET BLAS::BLAS)
    get_target_property(target_libraries BLAS::BLAS INTERFACE_LINK_LIBRARIES)
    if(NOT target_libraries)
      set(target_libraries "")
    endif()
    if("${target_libraries}" STREQUAL "${libraries}")
      set(${out} BLAS::BLAS PARENT_SCOPE)
      return()
    endif()
  endif()
  set(result "${libraries}")
  # BLAS_LINKER_FLAGS is a space-separated string; BLAS::BLAS would have carried
  # it for us, so it has to be restored by hand on this path.
  string(STRIP "${linker_flags}" linker_flags)
  if(linker_flags)
    separate_arguments(flag_list NATIVE_COMMAND "${linker_flags}")
    list(APPEND result ${flag_list})
  endif()
  set(${out} "${result}" PARENT_SCOPE)
endfunction()

# eigen_bench_select_blas_vendor()
#
# Honours EIGEN_BENCH_BLAS_VENDOR, and EIGEN_BENCH_REFERENCE (the spelling the
# machine configs in machines/*.toml pass, CONTRACTS.md section 7) as its alias.
# Defines, in the caller's scope:
#   EIGEN_BENCH_BLAS_FOUND        TRUE when a reference library is usable
#   EIGEN_BENCH_BLAS_ARM          arm key, e.g. accelerate; "" when not found
#   EIGEN_BENCH_BLAS_LINK_LIBRARIES  what a target links against
#   EIGEN_BENCH_BLAS_INCLUDE_DIRS    directory holding the vendor's cblas.h
#   EIGEN_BENCH_BLAS_{DISPLAY_NAME,BLA_VENDOR,LIBRARIES,LIBRARY_PATH,VERSION,
#                     VERSION_SOURCE,VERSION_RUNTIME,INTERFACE,THREADING,
#                     VERSION_RUNTIME_SYMBOL,THREAD_ENV,PROVIDES,
#                     COMPILE_DEFINITIONS,COMPILE_OPTIONS,
#                     CBLAS_HEADER,NOTES}
# and leaves BLAS_FOUND/BLAS_LIBRARIES set as find_package(BLAS) would.
function(eigen_bench_select_blas_vendor)
  set(request "${EIGEN_BENCH_BLAS_VENDOR}")
  if(DEFINED EIGEN_BENCH_REFERENCE AND NOT "${EIGEN_BENCH_REFERENCE}" STREQUAL "")
    eigen_bench_normalize_vendor("${EIGEN_BENCH_REFERENCE}" reference_key)
    if(request STREQUAL "auto")
      set(request "${reference_key}")
    else()
      eigen_bench_normalize_vendor("${request}" request_key)
      if(NOT request_key STREQUAL reference_key)
        message(FATAL_ERROR
          "EIGEN_BENCH_BLAS_VENDOR=${EIGEN_BENCH_BLAS_VENDOR} and "
          "EIGEN_BENCH_REFERENCE=${EIGEN_BENCH_REFERENCE} select different vendors "
          "(${request_key} vs ${reference_key}). Set only one.")
      endif()
    endif()
  endif()

  set(EIGEN_BENCH_BLAS_FOUND FALSE PARENT_SCOPE)
  set(EIGEN_BENCH_BLAS_ARM "" PARENT_SCOPE)
  foreach(field DISPLAY_NAME BLA_VENDOR LIBRARIES LINK_LIBRARIES INCLUDE_DIRS
                LIBRARY_PATH VERSION VERSION_SOURCE VERSION_RUNTIME
                VERSION_RUNTIME_SYMBOL INTERFACE
                THREADING THREAD_ENV PROVIDES COMPILE_DEFINITIONS COMPILE_OPTIONS
                CBLAS_HEADER NOTES)
    set(EIGEN_BENCH_BLAS_${field} "" PARENT_SCOPE)
  endforeach()

  if(request STREQUAL "none" OR request STREQUAL "OFF" OR request STREQUAL "NONE")
    message(STATUS "Comparison benchmarks: reference BLAS disabled (EIGEN_BENCH_BLAS_VENDOR=none)")
    return()
  endif()

  eigen_bench_vendor_keys(all_keys)
  if(request STREQUAL "auto" OR request STREQUAL "")
    eigen_bench_vendor_auto_order(candidates)
    set(required FALSE)
  else()
    eigen_bench_normalize_vendor("${request}" candidates)
    if(NOT candidates IN_LIST all_keys)
      message(FATAL_ERROR
        "EIGEN_BENCH_BLAS_VENDOR=${request} is not a known vendor. "
        "Known keys: ${all_keys} (also auto, none).")
    endif()
    set(required TRUE)
  endif()

  set(unusable_routes "")
  foreach(key IN LISTS candidates)
    eigen_bench_vendor_get(${key} PLATFORM platform)
    if(platform AND NOT ${platform})
      continue()
    endif()
    # MIN_CMAKE is the release whose FindBLAS knows this BLA_VENDOR spelling, so
    # it gates that route alone: a vendor shipping a config package is reachable
    # through PACKAGE below on any CMake.
    eigen_bench_vendor_get(${key} MIN_CMAKE min_cmake)
    set(bla_vendor_usable TRUE)
    if(min_cmake AND CMAKE_VERSION VERSION_LESS min_cmake)
      set(bla_vendor_usable FALSE)
    endif()

    set(found FALSE)
    set(link_libraries "")
    eigen_bench_vendor_get(${key} PACKAGE package)
    eigen_bench_vendor_get(${key} PACKAGE_TARGETS package_targets)
    if(package)
      find_package(${package} QUIET)
      if(${package}_FOUND AND package_targets)
        set(found TRUE)
        set(link_libraries ${package_targets})
        set(BLAS_LIBRARIES ${package_targets})
        set(selected_bla_vendor "")
      endif()
    endif()

    eigen_bench_vendor_get(${key} BLA_VENDOR bla_vendors)
    if(NOT found AND bla_vendor_usable)
      foreach(bla_vendor IN LISTS bla_vendors)
        set(BLA_VENDOR ${bla_vendor})
        unset(BLAS_FOUND)
        unset(BLAS_LIBRARIES)
        find_package(BLAS QUIET)
        if(BLAS_FOUND)
          set(found TRUE)
          set(selected_bla_vendor "${bla_vendor}")
          _eigen_bench_blas_link_libraries("${BLAS_LIBRARIES}" "${BLAS_LINKER_FLAGS}" link_libraries)
          break()
        endif()
      endforeach()
    endif()
    if(NOT found)
      # Record why, so the single failure message below can say which routes were
      # tried rather than a bare "not found" that sends someone looking for a
      # library which is in fact installed.
      if(NOT bla_vendor_usable)
        list(APPEND unusable_routes
             "${key}: FindBLAS route needs CMake >= ${min_cmake}, this is ${CMAKE_VERSION}")
      endif()
      continue()
    endif()

    eigen_bench_vendor_get(${key} EXTRA_LIBRARIES extra_libraries)
    if(extra_libraries)
      list(APPEND link_libraries ${extra_libraries})
    endif()

    # Advisory, never a gate on selection: the comparison arm calls the Fortran
    # BLAS (dgemm_) directly and includes no vendor header at all (CONTRACTS.md
    # section 9.2). Only Tuning/bench_blas_gemm.cpp needs it, and that target
    # guards on this variable. Refusing a perfectly usable library here -- or, in
    # auto mode, silently stepping past it -- would reject NVPL resolved through
    # its config package, and any module/spack/conda MKL whose headers ship
    # separately, for a header nothing in this directory includes.
    _eigen_bench_find_cblas_include(${key} "${BLAS_LIBRARIES}" include_dir)
    eigen_bench_vendor_get(${key} CBLAS_HEADER cblas_header)

    _eigen_bench_probe_version(${key} "${include_dir}" version version_source)
    if(NOT version)
      set(version "unknown")
      set(version_source "")
    endif()

    set(library_path "")
    foreach(lib IN LISTS BLAS_LIBRARIES)
      if(IS_ABSOLUTE "${lib}" AND EXISTS "${lib}")
        set(library_path "${lib}")
        break()
      endif()
    endforeach()
    if(NOT library_path)
      eigen_bench_vendor_get(${key} LIBRARY_PATH_FALLBACK library_path)
    endif()

    foreach(field DISPLAY_NAME INTERFACE_WIDTH THREADING THREAD_ENV PROVIDES
                  COMPILE_DEFINITIONS COMPILE_OPTIONS VERSION_RUNTIME
                  VERSION_RUNTIME_SYMBOL NOTES)
      eigen_bench_vendor_get(${key} ${field} ${field}_value)
    endforeach()

    set(EIGEN_BENCH_BLAS_FOUND TRUE PARENT_SCOPE)
    set(EIGEN_BENCH_BLAS_ARM "${key}" PARENT_SCOPE)
    set(EIGEN_BENCH_BLAS_DISPLAY_NAME "${DISPLAY_NAME_value}" PARENT_SCOPE)
    set(EIGEN_BENCH_BLAS_BLA_VENDOR "${selected_bla_vendor}" PARENT_SCOPE)
    set(EIGEN_BENCH_BLAS_LIBRARIES "${BLAS_LIBRARIES}" PARENT_SCOPE)
    set(EIGEN_BENCH_BLAS_LINK_LIBRARIES "${link_libraries}" PARENT_SCOPE)
    set(EIGEN_BENCH_BLAS_INCLUDE_DIRS "${include_dir}" PARENT_SCOPE)
    set(EIGEN_BENCH_BLAS_CBLAS_HEADER "${cblas_header}" PARENT_SCOPE)
    set(EIGEN_BENCH_BLAS_LIBRARY_PATH "${library_path}" PARENT_SCOPE)
    set(EIGEN_BENCH_BLAS_VERSION "${version}" PARENT_SCOPE)
    set(EIGEN_BENCH_BLAS_VERSION_SOURCE "${version_source}" PARENT_SCOPE)
    set(EIGEN_BENCH_BLAS_VERSION_RUNTIME "${VERSION_RUNTIME_value}" PARENT_SCOPE)
    set(EIGEN_BENCH_BLAS_VERSION_RUNTIME_SYMBOL "${VERSION_RUNTIME_SYMBOL_value}" PARENT_SCOPE)
    set(EIGEN_BENCH_BLAS_INTERFACE "${INTERFACE_WIDTH_value}" PARENT_SCOPE)
    set(EIGEN_BENCH_BLAS_THREADING "${THREADING_value}" PARENT_SCOPE)
    set(EIGEN_BENCH_BLAS_THREAD_ENV "${THREAD_ENV_value}" PARENT_SCOPE)
    set(EIGEN_BENCH_BLAS_PROVIDES "${PROVIDES_value}" PARENT_SCOPE)
    set(EIGEN_BENCH_BLAS_COMPILE_DEFINITIONS "${COMPILE_DEFINITIONS_value}" PARENT_SCOPE)
    set(EIGEN_BENCH_BLAS_COMPILE_OPTIONS "${COMPILE_OPTIONS_value}" PARENT_SCOPE)
    set(EIGEN_BENCH_BLAS_NOTES "${NOTES_value}" PARENT_SCOPE)
    # Legacy consumers (benchmarks/Tuning) read these.
    set(BLAS_FOUND TRUE PARENT_SCOPE)
    set(BLAS_LIBRARIES "${BLAS_LIBRARIES}" PARENT_SCOPE)

    message(STATUS "Comparison benchmarks: reference arm '${key}' = ${DISPLAY_NAME_value}, version '${version}'")
    message(STATUS "  link:    ${link_libraries}")
    if(include_dir)
      message(STATUS "  ${cblas_header}: ${include_dir}/${cblas_header}")
    else()
      message(STATUS "  ${cblas_header}: not found; the comparison arm does not need it, "
                     "but targets that include it are skipped")
    endif()
    return()
  endforeach()

  if(required)
    set(detail "")
    if(unusable_routes)
      list(JOIN unusable_routes "; " detail)
      set(detail " (${detail})")
    endif()
    message(FATAL_ERROR "EIGEN_BENCH_BLAS_VENDOR=${request} was requested but not found${detail}")
  endif()
  message(STATUS "Comparison benchmarks: no reference BLAS found; building the eigen arm only")
endfunction()

# JSON-escape a string for the metadata file the harness reads.
function(eigen_bench_json_escape value out)
  string(REPLACE "\\" "\\\\" value "${value}")
  string(REPLACE "\"" "\\\"" value "${value}")
  string(REPLACE "\n" "\\n" value "${value}")
  string(REPLACE "\t" "\\t" value "${value}")
  set(${out} "${value}" PARENT_SCOPE)
endfunction()

# Render a CMake list as a JSON array of strings.
function(eigen_bench_json_string_array values out)
  set(items "")
  foreach(value IN LISTS values)
    eigen_bench_json_escape("${value}" value)
    list(APPEND items "\"${value}\"")
  endforeach()
  list(JOIN items ", " rendered)
  set(${out} "[${rendered}]" PARENT_SCOPE)
endfunction()
