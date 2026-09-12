# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# The reference libraries the comparison benchmarks can be built against, one
# row per arm key, and the selection that turns -DEIGEN_BENCH_REFERENCE=<key>
# into link libraries. Included by benchmarks/CMakeLists.txt before its own
# find_package(BLAS): FindBLAS creates the BLAS::BLAS target once and never
# revises it, so the table's search has to run first.
#
# A machine profile (machines/<id>.toml) names the arm and may pin the
# libraries explicitly (-DBLAS_LIBRARIES=..., -DLAPACK_LIBRARIES=...) where
# FindBLAS cannot find them; see README.md.

include_guard(GLOBAL)

set(EIGEN_BENCH_VENDOR_FIELDS
    DISPLAY_NAME        # human-readable name, lands in the result file
    ALIASES             # additional spellings accepted by EIGEN_BENCH_REFERENCE
    BLA_VENDOR          # FindBLAS BLA_VENDOR candidates, tried in order
    PACKAGE             # config package to find instead, with its imported targets
    PACKAGE_TARGETS
    LAPACK_PACKAGE      # LAPACK config package when it is a separate library
    LAPACK_PACKAGE_TARGETS
    INTERFACE_WIDTH     # lp64 | ilp64
    THREAD_ENV          # environment variable that caps the library's thread count
    PROVIDES            # subset of: blas lapack -- what the linked library itself carries
    VERSION_RUNTIME_SYMBOL  # version query the binary calls if it links (bench_compare.h)
    PLATFORM            # CMake variable that must be true for this vendor
    ARM_SOURCES         # extra sources compiled into every binary linking this arm
    ARM_LINK_OPTIONS    # extra link options for those binaries
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

function(eigen_bench_vendor_get key field out)
  get_property(value GLOBAL PROPERTY EIGEN_BENCH_VENDOR_${key}_${field})
  set(${out} "${value}" PARENT_SCOPE)
endfunction()

function(eigen_bench_vendor_keys out)
  get_property(keys GLOBAL PROPERTY EIGEN_BENCH_VENDOR_KEYS)
  set(${out} "${keys}" PARENT_SCOPE)
endfunction()

# Resolve a link list into file paths a try_compile can use: an IMPORTED target
# is invisible inside the mini-project check_cxx_source_compiles generates. A
# config package's target carries the path in IMPORTED_LOCATION; FindBLAS's
# BLAS::BLAS is INTERFACE IMPORTED with the real libraries one level down.
function(eigen_bench_resolve_linkables in_list out)
  set(result "")
  foreach(entry IN LISTS in_list)
    _eigen_bench_resolve_linkable("${entry}" 0 resolved)
    if(resolved)
      list(APPEND result ${resolved})
    endif()
  endforeach()
  set(${out} "${result}" PARENT_SCOPE)
endfunction()

function(_eigen_bench_resolve_linkable entry depth out)
  set(${out} "" PARENT_SCOPE)
  if(depth GREATER 8)
    return()
  endif()
  if(NOT TARGET "${entry}")
    set(${out} "${entry}" PARENT_SCOPE)
    return()
  endif()
  get_target_property(aliased "${entry}" ALIASED_TARGET)
  if(aliased)
    _eigen_bench_resolve_linkable("${aliased}" "${depth}" resolved)
    set(${out} "${resolved}" PARENT_SCOPE)
    return()
  endif()
  get_target_property(configurations "${entry}" IMPORTED_CONFIGURATIONS)
  if(NOT configurations)
    set(configurations "")
  endif()
  set(location "")
  foreach(suffix "" ${configurations})
    if(suffix)
      get_target_property(location "${entry}" IMPORTED_LOCATION_${suffix})
    else()
      get_target_property(location "${entry}" IMPORTED_LOCATION)
    endif()
    if(location)
      break()
    endif()
  endforeach()
  if(location)
    set(${out} "${location}" PARENT_SCOPE)
    return()
  endif()
  set(result "")
  math(EXPR next_depth "${depth} + 1")
  get_target_property(interface_libraries "${entry}" INTERFACE_LINK_LIBRARIES)
  if(interface_libraries)
    foreach(dependency IN LISTS interface_libraries)
      _eigen_bench_resolve_linkable("${dependency}" "${next_depth}" resolved)
      if(resolved)
        list(APPEND result ${resolved})
      endif()
    endforeach()
  endif()
  set(${out} "${result}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

eigen_bench_declare_vendor(accelerate
  DISPLAY_NAME "Apple Accelerate"
  ALIASES apple veclib
  BLA_VENDOR Apple
  PLATFORM APPLE
  INTERFACE_WIDTH lp64
  THREAD_ENV VECLIB_MAXIMUM_THREADS
  PROVIDES blas lapack
  NOTES "Accelerate exposes no version query; record the macOS version in the profile's version fallback.")

eigen_bench_declare_vendor(openblas
  DISPLAY_NAME "OpenBLAS"
  ALIASES open_blas goto
  BLA_VENDOR OpenBLAS
  INTERFACE_WIDTH lp64
  THREAD_ENV OPENBLAS_NUM_THREADS
  PROVIDES blas lapack
  VERSION_RUNTIME_SYMBOL openblas_get_config)

# Sequential LP64 first: a run pins threads itself, and a threaded MKL silently
# ignoring MKL_NUM_THREADS=1 would make the eigen arm look slow.
eigen_bench_declare_vendor(mkl
  DISPLAY_NAME "Intel oneMKL"
  ALIASES intel onemkl
  BLA_VENDOR Intel10_64lp_seq Intel10_64lp Intel10_64_dyn
  INTERFACE_WIDTH lp64
  THREAD_ENV MKL_NUM_THREADS
  PROVIDES blas lapack
  VERSION_RUNTIME_SYMBOL mkl_get_version_string
  NOTES "On a non-Intel CPU oneMKL takes a conservative code path chosen from the CPUID vendor string.")

# The same oneMKL with its CPUID vendor check answered "Intel" from inside the
# binary (mkl_vendor_bypass.cpp): on Zen 5 the as-shipped row runs dgemm at half
# of Eigen's rate and this row at parity (measured 2026-09). Intel does not
# support the configuration; a page carries it under this name, never as MKL.
eigen_bench_declare_vendor(mkl_bypass
  DISPLAY_NAME "Intel oneMKL, vendor check bypassed"
  ALIASES mkl_unlocked
  BLA_VENDOR Intel10_64lp_seq Intel10_64lp Intel10_64_dyn
  INTERFACE_WIDTH lp64
  THREAD_ENV MKL_NUM_THREADS
  PROVIDES blas lapack
  VERSION_RUNTIME_SYMBOL mkl_get_version_string
  ARM_SOURCES ${CMAKE_CURRENT_LIST_DIR}/mkl_vendor_bypass.cpp
  # The executable's definition must reach the dynamic symbol table to interpose
  # on the library's own; ld exports it on its own only when a shared library
  # leaves the symbol undefined, and libmkl_core defines it.
  ARM_LINK_OPTIONS -Wl,--export-dynamic-symbol=mkl_serv_get_cpu_true
  NOTES "oneMKL's CPUID vendor check is answered 'Intel' from inside the binary; unsupported by Intel, published only under this name.")

# LAPACK is AOCL-libFLAME, a separate libflame.so; FindLAPACK's AOCL route
# resolves it when BLA_VENDOR is AOCL, or pass -DLAPACK_LIBRARIES explicitly.
# AOCL keeps its libraries under lib_LP64/, which no default search prefix
# covers: pass -DCMAKE_LIBRARY_PATH=<prefix>/lib_LP64 as well.
eigen_bench_declare_vendor(aocl
  DISPLAY_NAME "AOCL-BLIS"
  ALIASES amd
  BLA_VENDOR AOCL AOCL_mt FLAME
  INTERFACE_WIDTH lp64
  THREAD_ENV BLIS_NUM_THREADS
  PROVIDES blas
  VERSION_RUNTIME_SYMBOL bli_info_get_version_str)

eigen_bench_declare_vendor(blis
  DISPLAY_NAME "BLIS"
  ALIASES flame
  BLA_VENDOR FLAME
  INTERFACE_WIDTH lp64
  THREAD_ENV BLIS_NUM_THREADS
  PROVIDES blas
  VERSION_RUNTIME_SYMBOL bli_info_get_version_str)

eigen_bench_declare_vendor(armpl
  DISPLAY_NAME "Arm Performance Libraries"
  ALIASES arm
  BLA_VENDOR Arm Arm_mp
  INTERFACE_WIDTH lp64
  THREAD_ENV ARMPL_NUM_THREADS
  PROVIDES blas lapack
  VERSION_RUNTIME_SYMBOL armplversion)

eigen_bench_declare_vendor(nvpl
  DISPLAY_NAME "NVPL BLAS"
  ALIASES nvidia
  BLA_VENDOR NVPL
  PACKAGE nvpl_blas
  PACKAGE_TARGETS nvpl::blas_lp64_seq
  LAPACK_PACKAGE nvpl_lapack
  LAPACK_PACKAGE_TARGETS nvpl::lapack_lp64_seq
  INTERFACE_WIDTH lp64
  THREAD_ENV NVPL_BLAS_NUM_THREADS
  PROVIDES blas lapack
  VERSION_RUNTIME_SYMBOL nvpl_blas_get_version)

# On Debian and Ubuntu /usr/lib/<triplet>/libblas.so is an update-alternatives
# symlink that a tuned BLAS repoints, so name the reference build by path
# (-DBLAS_LIBRARIES=/usr/lib/<triplet>/blas/libblas.so.3) rather than trusting
# BLA_VENDOR=Generic; the result file records the resolved path either way.
eigen_bench_declare_vendor(netlib
  DISPLAY_NAME "Netlib reference BLAS"
  ALIASES generic reference
  BLA_VENDOR Generic
  INTERFACE_WIDTH lp64
  THREAD_ENV OMP_NUM_THREADS
  PROVIDES blas
  NOTES "The reference implementation exposes no version query; the packaged version is the identifier.")

# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

function(eigen_bench_normalize_vendor request out)
  string(TOLOWER "${request}" needle)
  eigen_bench_vendor_keys(keys)
  foreach(key IN LISTS keys)
    eigen_bench_vendor_get(${key} ALIASES aliases)
    eigen_bench_vendor_get(${key} BLA_VENDOR bla_vendors)
    foreach(alias IN LISTS key aliases bla_vendors)
      string(TOLOWER "${alias}" alias)
      if(needle STREQUAL alias)
        set(${out} "${key}" PARENT_SCOPE)
        return()
      endif()
    endforeach()
  endforeach()
  set(${out} "${needle}" PARENT_SCOPE)
endfunction()

# eigen_bench_select_blas_vendor()
#
# Honours EIGEN_BENCH_REFERENCE (a key or alias from the table; unset or "none"
# builds the Eigen arm only). Defines, in the caller's scope:
#   EIGEN_BENCH_BLAS_FOUND            TRUE when a reference library is usable
#   EIGEN_BENCH_BLAS_ARM              arm key
#   EIGEN_BENCH_BLAS_LINK_LIBRARIES   what a target links against
#   EIGEN_BENCH_BLAS_LIBRARIES        the library files
#   EIGEN_BENCH_BLAS_LIBRARY_PATH     the first library, symlinks resolved
#   EIGEN_BENCH_BLAS_<FIELD>          every table field of the selected row
# and leaves BLAS_FOUND/BLAS_LIBRARIES set as find_package(BLAS) would.
function(eigen_bench_select_blas_vendor)
  set(EIGEN_BENCH_BLAS_FOUND FALSE PARENT_SCOPE)
  set(EIGEN_BENCH_BLAS_ARM "" PARENT_SCOPE)
  if(NOT DEFINED EIGEN_BENCH_REFERENCE OR "${EIGEN_BENCH_REFERENCE}" STREQUAL "" OR
     "${EIGEN_BENCH_REFERENCE}" STREQUAL "none")
    return()
  endif()
  eigen_bench_normalize_vendor("${EIGEN_BENCH_REFERENCE}" key)
  eigen_bench_vendor_keys(keys)
  if(NOT key IN_LIST keys)
    message(FATAL_ERROR "EIGEN_BENCH_REFERENCE=${EIGEN_BENCH_REFERENCE} names no vendor; known: ${keys}")
  endif()
  eigen_bench_vendor_get(${key} PLATFORM platform)
  if(platform AND NOT ${platform})
    message(FATAL_ERROR "EIGEN_BENCH_REFERENCE=${key} needs ${platform}, which is false on this host")
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
    endif()
  endif()
  if(NOT found)
    eigen_bench_vendor_get(${key} BLA_VENDOR bla_vendors)
    foreach(bla_vendor IN LISTS bla_vendors)
      set(BLA_VENDOR ${bla_vendor})
      # A -DBLAS_LIBRARIES=<path> on the command line is a cache entry and
      # survives this; FindBLAS then validates it instead of searching.
      unset(BLAS_FOUND)
      unset(BLAS_LIBRARIES)
      find_package(BLAS QUIET)
      if(BLAS_FOUND)
        set(found TRUE)
        # BLAS::BLAS is created on FindBLAS's first success and never revised;
        # adopt it only when it describes this call's own BLAS_LIBRARIES.
        set(link_libraries "${BLAS_LIBRARIES}")
        if(TARGET BLAS::BLAS)
          get_target_property(target_libraries BLAS::BLAS INTERFACE_LINK_LIBRARIES)
          if("${target_libraries}" STREQUAL "${BLAS_LIBRARIES}")
            set(link_libraries BLAS::BLAS)
          endif()
        endif()
        string(STRIP "${BLAS_LINKER_FLAGS}" linker_flags)
        if(linker_flags AND NOT link_libraries STREQUAL "BLAS::BLAS")
          separate_arguments(flag_list NATIVE_COMMAND "${linker_flags}")
          list(APPEND link_libraries ${flag_list})
        endif()
        break()
      endif()
    endforeach()
  endif()
  if(NOT found)
    message(FATAL_ERROR "EIGEN_BENCH_REFERENCE=${key} was requested but not found; pass -DBLAS_LIBRARIES=<path> "
                        "(and -DLAPACK_LIBRARIES=<path> for a LAPACK-referenced operation) to name it")
  endif()

  eigen_bench_resolve_linkables("${link_libraries}" library_files)
  set(library_path "")
  foreach(lib IN LISTS library_files)
    if(IS_ABSOLUTE "${lib}" AND EXISTS "${lib}")
      get_filename_component(library_path "${lib}" REALPATH)
      break()
    endif()
  endforeach()

  set(EIGEN_BENCH_BLAS_FOUND TRUE PARENT_SCOPE)
  set(EIGEN_BENCH_BLAS_ARM "${key}" PARENT_SCOPE)
  set(EIGEN_BENCH_BLAS_LINK_LIBRARIES "${link_libraries}" PARENT_SCOPE)
  set(EIGEN_BENCH_BLAS_LIBRARIES "${library_files}" PARENT_SCOPE)
  set(EIGEN_BENCH_BLAS_LIBRARY_PATH "${library_path}" PARENT_SCOPE)
  foreach(field IN LISTS EIGEN_BENCH_VENDOR_FIELDS)
    eigen_bench_vendor_get(${key} ${field} value)
    set(EIGEN_BENCH_BLAS_${field} "${value}" PARENT_SCOPE)
  endforeach()
  set(BLAS_FOUND TRUE PARENT_SCOPE)
  set(BLAS_LIBRARIES "${BLAS_LIBRARIES}" PARENT_SCOPE)
  eigen_bench_vendor_get(${key} DISPLAY_NAME display_name)
  message(STATUS "Comparison benchmarks: reference arm '${key}' = ${display_name}, ${library_path}")
endfunction()
