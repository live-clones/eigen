// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Registration machinery for the cross-library comparison benchmarks.
//
// One REGISTER_COMPARISON_POINT line emits both the Eigen arm and the
// reference-library arm of an operation at one shared grid point, with benchmark
// names that follow the grammar in CONTRACTS.md section 1:
//
//     op "/" arm "/" scalar ( "/" dimname ":" value )+ [ "/threads:" n ]
//
// The reference arm registers only when the build linked a vendor library, so the
// same source compiles and runs as an Eigen-only benchmark with no vendor present.
//
// Everything the harness cannot infer about the reference library — its key, name,
// version, interface width and threading model — is published into the Google
// Benchmark JSON context by publishArmContext(), which EIGEN_BENCH_COMPARISON_MAIN
// calls before benchmark::Initialize().

#ifndef EIGEN_BENCHMARKS_COMPARISON_BENCH_COMPARE_H
#define EIGEN_BENCHMARKS_COMPARISON_BENCH_COMPARE_H

#include <benchmark/benchmark.h>

#include <Eigen/Core>
#include <complex>
#include <cstdlib>
#include <string>

// ---------------------------------------------------------------------------
// Build-time inputs
// ---------------------------------------------------------------------------
// EIGEN_BENCH_REFERENCE_ARM is the only macro that changes what is registered.
// CMake passes it as a BARE TOKEN (-DEIGEN_BENCH_REFERENCE_ARM=openblas) so that
// no quoting has to survive the CMake command line; the header stringizes it.
// Every other macro below is optional metadata; each has a defined fallback so a
// hand-built binary still produces a well-formed context.

#define EIGEN_BENCH_STRINGIZE_(x) #x
#define EIGEN_BENCH_STRINGIZE(x) EIGEN_BENCH_STRINGIZE_(x)

#ifdef EIGEN_BENCH_REFERENCE_ARM
#define EIGEN_BENCH_REFERENCE_ARM_STR EIGEN_BENCH_STRINGIZE(EIGEN_BENCH_REFERENCE_ARM)
#else
#define EIGEN_BENCH_REFERENCE_ARM_STR ""
#endif

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------
// REGISTER_COMPARISON_POINT registers ONE grid point of one operation for both
// arms, adjacently. Google Benchmark runs instances in registration order, so
// registering a whole arm's grid and then the other arm's -- which is what an
// arrow-chain grid applied to two Benchmark objects does -- puts minutes of
// thermal and background drift between the two numbers a ratio is formed from,
// systematically and always in the same direction. Step 4 of
// .agents/benchmarking.md requires the alternating order instead; this is the
// registration-time form of it. A caller therefore drives the macro from a grid
// expressed as a list of points:
//
//     #define FOO_DIM_NAMES {"m", "n"}
//     #define FOO_POINT(...)
//         REGISTER_COMPARISON_POINT(FOO, f64, double, BM_FooEigen, BM_FooReference,
//                                   FOO_DIM_NAMES, __VA_ARGS__)
//     #define FOO_SIZES(POINT) POINT(8,8) POINT(16,16)
//     FOO_SIZES(FOO_POINT)
//
// DIM_NAMES must arrive as the NAME of an object-like macro, never as a literal
// {"m", "n"}: braces do not protect commas from macro argument splitting. For
// the same reason the two arms are spelled out here rather than delegated to a
// shared per-arm macro, which would split DIM_NAMES on the way down.
//
// ->Name() supplies the first three name fields; Google Benchmark appends the
// dimensions from ArgNames/Args and the optional /threads:N.
//
// SCALAR_TYPE is a single macro argument, so a type spelled with a comma
// (std::complex<float>) must be passed through an alias; eigen_bench::c32_t and
// eigen_bench::c64_t below exist for that.

#ifdef EIGEN_BENCH_REFERENCE_ARM
#define REGISTER_COMPARISON_POINT(MNEMONIC, SCALAR_TAG, SCALAR_TYPE, EIGEN_FN, REF_FN, DIM_NAMES, ...) \
  BENCHMARK_TEMPLATE(EIGEN_FN, SCALAR_TYPE)                                                            \
      ->Name(#MNEMONIC "/eigen/" #SCALAR_TAG)                                                          \
      ->ArgNames(DIM_NAMES)                                                                            \
      ->Args({__VA_ARGS__});                                                                           \
  BENCHMARK_TEMPLATE(REF_FN, SCALAR_TYPE)                                                              \
      ->Name(#MNEMONIC "/" EIGEN_BENCH_REFERENCE_ARM_STR "/" #SCALAR_TAG)                              \
      ->ArgNames(DIM_NAMES)                                                                            \
      ->Args({__VA_ARGS__});
#else
#define REGISTER_COMPARISON_POINT(MNEMONIC, SCALAR_TAG, SCALAR_TYPE, EIGEN_FN, REF_FN, DIM_NAMES, ...) \
  BENCHMARK_TEMPLATE(EIGEN_FN, SCALAR_TYPE)                                                            \
      ->Name(#MNEMONIC "/eigen/" #SCALAR_TAG)                                                          \
      ->ArgNames(DIM_NAMES)                                                                            \
      ->Args({__VA_ARGS__});
#endif

namespace eigen_bench {

// Aliases for the scalar tags of ops.toml whose C++ spelling contains a comma.
using c32_t = std::complex<float>;
using c64_t = std::complex<double>;

// Integer width of the reference library's Fortran BLAS/LAPACK interface.
// Eigen::BlasIndex (Eigen/src/Core/util/BlasTypes.h) is already that switch,
// keyed off EIGEN_64BIT_BLAS; aliasing it rather than restating it keeps ONE
// BLAS integer width in the binary. Two independent switches would let these
// benchmarks and Eigen's own BLAS backend disagree, which passes the compiler
// and the linker and then corrupts every by-pointer Fortran argument.
using BlasInt = Eigen::BlasIndex;

#ifdef EIGEN_BENCH_REFERENCE_ILP64
// Deliberately a hard error rather than a silent narrowing: CMake selects the
// reference arm's width from the vendor table, and Eigen only produces a 64-bit
// BlasIndex when the ABI-affecting EIGEN_64BIT_BLAS is defined for every
// translation unit.
static_assert(sizeof(BlasInt) == 8,
              "EIGEN_BENCH_REFERENCE_ILP64 selects a 64-bit-integer reference BLAS, but Eigen::BlasIndex is 32-bit: "
              "the build must define EIGEN_64BIT_BLAS as well.");
#endif

// ---------------------------------------------------------------------------
// Reference-library version queries
// ---------------------------------------------------------------------------
// Declared here rather than included from a vendor header so that a build only
// needs the library, not its development headers. CMake must define the matching
// EIGEN_BENCH_HAVE_* macro only after a check that actually compiles and links
// the symbol; defining it on a guess turns a missing query into a link failure.

}  // namespace eigen_bench

#if defined(EIGEN_BENCH_HAVE_OPENBLAS_GET_CONFIG)
extern "C" char* openblas_get_config(void);
#endif
#if defined(EIGEN_BENCH_HAVE_MKL_GET_VERSION_STRING)
extern "C" void mkl_get_version_string(char* buffer, int length);
#endif
#if defined(EIGEN_BENCH_HAVE_BLI_INFO_GET_VERSION_STR)
extern "C" const char* bli_info_get_version_str(void);
#endif
#if defined(EIGEN_BENCH_HAVE_ARMPLVERSION)
extern "C" void armplversion(int* major, int* minor, int* patch, const char** tag);
#endif
#if defined(EIGEN_BENCH_HAVE_NVPL_BLAS_GET_VERSION)
extern "C" int nvpl_blas_get_version(void);
#endif

namespace eigen_bench {

namespace detail {

inline std::string jsonEscape(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 2);
  for (char c : value) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          static const char* kHex = "0123456789abcdef";
          out += "\\u00";
          out += kHex[(static_cast<unsigned char>(c) >> 4) & 0xF];
          out += kHex[static_cast<unsigned char>(c) & 0xF];
        } else {
          out += c;
        }
        break;
    }
  }
  return out;
}

// Thread-count environment variables to report. The list must cover every
// variable run.py's THREAD_COUNT_ENV_VARS and THREAD_FIXED_ENV set, or a run
// records a cap the harness applied as absent; the affinity variables after
// them are not set by run.py but change what a thread count means, so a value
// the environment carried in is worth reporting. No CMake file defines
// EIGEN_BENCH_THREAD_ENV_VARS today, so this list is the one that is used; the
// override is kept because the vendor table already carries a per-vendor
// THREAD_ENV field to generate it from.
#ifndef EIGEN_BENCH_THREAD_ENV_VARS
#define EIGEN_BENCH_THREAD_ENV_VARS                                                            \
  "OMP_NUM_THREADS,OPENBLAS_NUM_THREADS,GOTO_NUM_THREADS,MKL_NUM_THREADS,BLIS_NUM_THREADS,"    \
  "NVPL_BLAS_NUM_THREADS,ARMPL_NUM_THREADS,VECLIB_MAXIMUM_THREADS,ACCELERATE_MAXIMUM_THREADS," \
  "BLAS_NUM_THREADS,EIGEN_NB_THREADS,OMP_DYNAMIC,MKL_DYNAMIC,OMP_PROC_BIND,OMP_PLACES"
#endif

// A JSON object literal holding only the variables that are actually set, so an
// empty object is a positive statement that the environment carried none.
inline std::string threadEnvJson() {
  const std::string names = EIGEN_BENCH_THREAD_ENV_VARS;
  std::string out = "{";
  bool first = true;
  std::string::size_type pos = 0;
  while (pos <= names.size()) {
    const std::string::size_type comma = names.find(',', pos);
    const std::string::size_type end = comma == std::string::npos ? names.size() : comma;
    const std::string name = names.substr(pos, end - pos);
    if (!name.empty()) {
      const char* value = std::getenv(name.c_str());
      if (value != nullptr) {
        if (!first) out += ",";
        first = false;
        out += "\"" + jsonEscape(name) + "\":\"" + jsonEscape(value) + "\"";
      }
    }
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  out += "}";
  return out;
}

inline std::string compilerId() {
#if defined(EIGEN_BENCH_COMPILER_ID)
  return EIGEN_BENCH_COMPILER_ID;
#elif defined(__INTEL_LLVM_COMPILER)
  return "IntelLLVM";
#elif defined(__apple_build_version__)
  return "AppleClang";
#elif defined(__clang__)
  return "Clang";
#elif defined(_MSC_VER)
  return "MSVC";
#elif defined(__GNUC__)
  return "GNU";
#else
  return "";
#endif
}

inline std::string compilerVersion() {
#if defined(EIGEN_BENCH_COMPILER_VERSION)
  return EIGEN_BENCH_COMPILER_VERSION;
#elif defined(__clang__)
  return EIGEN_BENCH_STRINGIZE(__clang_major__) "." EIGEN_BENCH_STRINGIZE(__clang_minor__) "." EIGEN_BENCH_STRINGIZE(
      __clang_patchlevel__);
#elif defined(_MSC_VER)
  return EIGEN_BENCH_STRINGIZE(_MSC_FULL_VER);
#elif defined(__GNUC__)
  return EIGEN_BENCH_STRINGIZE(__GNUC__) "." EIGEN_BENCH_STRINGIZE(__GNUC_MINOR__) "." EIGEN_BENCH_STRINGIZE(
      __GNUC_PATCHLEVEL__);
#else
  return "";
#endif
}

inline std::string cxxStandard() {
#if defined(EIGEN_BENCH_CXX_STANDARD)
  return EIGEN_BENCH_CXX_STANDARD;
#else
  // MSVC without /Zc:__cplusplus reports 199711L; the CMake-supplied value is
  // the one to trust when it is present.
  return __cplusplus >= 202302L   ? "23"
         : __cplusplus >= 202002L ? "20"
         : __cplusplus >= 201703L ? "17"
         : __cplusplus >= 201402L ? "14"
                                  : "";
#endif
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Arm metadata
// ---------------------------------------------------------------------------

// The vendor key that field 1 of this binary's reference-arm names carries;
// "" when no reference library is linked.
inline const char* referenceArmKey() { return EIGEN_BENCH_REFERENCE_ARM_STR; }

inline bool hasReference() { return referenceArmKey()[0] != '\0'; }

// Display name for tables. Defaults to the arm key so that adding a vendor needs
// no edit here; CMake supplies the prettier spelling.
inline std::string referenceLibraryName() {
#if defined(EIGEN_BENCH_REFERENCE_LIBRARY_NAME)
  return EIGEN_BENCH_REFERENCE_LIBRARY_NAME;
#else
  return referenceArmKey();
#endif
}

// Best available source first: the library's own query, then Eigen's version
// macros for the eigenblas arm, then the CMake-supplied fallback, then
// "unknown" — which obliges run.py to record a provenance gap.
inline std::string referenceLibraryVersion() {
  if (!hasReference()) return "";
#if defined(EIGEN_BENCH_HAVE_OPENBLAS_GET_CONFIG)
  {
    const char* config = openblas_get_config();
    if (config != nullptr && config[0] != '\0') return config;
  }
#elif defined(EIGEN_BENCH_HAVE_MKL_GET_VERSION_STRING)
  {
    char buffer[256] = {0};
    mkl_get_version_string(buffer, static_cast<int>(sizeof(buffer)) - 1);
    std::string version(buffer);
    // The library pads the buffer with spaces.
    const std::string::size_type last = version.find_last_not_of(' ');
    if (last != std::string::npos) return version.substr(0, last + 1);
  }
#elif defined(EIGEN_BENCH_HAVE_BLI_INFO_GET_VERSION_STR)
  {
    const char* version = bli_info_get_version_str();
    if (version != nullptr && version[0] != '\0') return version;
  }
#elif defined(EIGEN_BENCH_HAVE_ARMPLVERSION)
  {
    int major = 0, minor = 0, patch = 0;
    const char* tag = nullptr;
    armplversion(&major, &minor, &patch, &tag);
    std::string version = std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
    if (tag != nullptr && tag[0] != '\0') version += std::string(" ") + tag;
    return version;
  }
#elif defined(EIGEN_BENCH_HAVE_NVPL_BLAS_GET_VERSION)
  {
    const int packed = nvpl_blas_get_version();
    if (packed > 0) {
      return std::to_string(packed / 10000) + "." + std::to_string((packed / 100) % 100) + "." +
             std::to_string(packed % 100);
    }
  }
#endif
  if (std::string(referenceArmKey()) == "eigenblas") {
    return std::to_string(EIGEN_WORLD_VERSION) + "." + std::to_string(EIGEN_MAJOR_VERSION) + "." +
           std::to_string(EIGEN_MINOR_VERSION);
  }
#if defined(EIGEN_BENCH_REFERENCE_VERSION_FALLBACK)
  {
    const std::string fallback = EIGEN_BENCH_REFERENCE_VERSION_FALLBACK;
    if (!fallback.empty()) return fallback;
  }
#endif
  return "unknown";
}

inline std::string referenceLibraryPath() {
#if defined(EIGEN_BENCH_REFERENCE_PATH)
  return EIGEN_BENCH_REFERENCE_PATH;
#else
  return "";
#endif
}

inline std::string referenceInterface() {
#if defined(EIGEN_BENCH_REFERENCE_INTERFACE)
  return EIGEN_BENCH_REFERENCE_INTERFACE;
#else
  return hasReference() ? (sizeof(BlasInt) == 8 ? "ilp64" : "lp64") : "";
#endif
}

inline std::string referenceThreading() {
#if defined(EIGEN_BENCH_REFERENCE_THREADING)
  return EIGEN_BENCH_REFERENCE_THREADING;
#else
  return "";
#endif
}

// Publishes the eigen_bench.* keys into the Google Benchmark JSON context.
// MUST run before benchmark::Initialize().
inline void publishArmContext() {
  using benchmark::AddCustomContext;
  AddCustomContext("eigen_bench.schema_version", "1.0.0");
  AddCustomContext("eigen_bench.reference_arm", referenceArmKey());
  AddCustomContext("eigen_bench.reference_library_name", hasReference() ? referenceLibraryName() : std::string());
  AddCustomContext("eigen_bench.reference_library_version", hasReference() ? referenceLibraryVersion() : std::string());
  AddCustomContext("eigen_bench.reference_library_path", referenceLibraryPath());
  AddCustomContext("eigen_bench.reference_interface", referenceInterface());
  AddCustomContext("eigen_bench.reference_threading", referenceThreading());
#if defined(EIGEN_BENCH_EIGEN_COMMIT)
  AddCustomContext("eigen_bench.eigen_commit", EIGEN_BENCH_EIGEN_COMMIT);
#else
  AddCustomContext("eigen_bench.eigen_commit", "");
#endif
#if defined(EIGEN_BENCH_EIGEN_DIRTY)
  AddCustomContext("eigen_bench.eigen_dirty", EIGEN_BENCH_EIGEN_DIRTY);
#else
  AddCustomContext("eigen_bench.eigen_dirty", "");
#endif
  AddCustomContext("eigen_bench.compiler_id", detail::compilerId());
  AddCustomContext("eigen_bench.compiler_version", detail::compilerVersion());
  AddCustomContext("eigen_bench.cxx_standard", detail::cxxStandard());
#if defined(EIGEN_BENCH_CXX_FLAGS)
  AddCustomContext("eigen_bench.cxx_flags", EIGEN_BENCH_CXX_FLAGS);
#else
  AddCustomContext("eigen_bench.cxx_flags", "");
#endif
#if defined(EIGEN_BENCH_ISA_TARGET)
  AddCustomContext("eigen_bench.isa_target", EIGEN_BENCH_ISA_TARGET);
#else
  AddCustomContext("eigen_bench.isa_target", "");
#endif
  AddCustomContext("eigen_bench.eigen_nb_threads", std::to_string(Eigen::nbThreads()));
  // Without OpenMP, Eigen ignores EIGEN_NB_THREADS and nbThreads() is always 1,
  // so a reader cannot interpret the thread count without knowing which it is.
#ifdef EIGEN_HAS_OPENMP
  AddCustomContext("eigen_bench.eigen_has_openmp", "true");
#else
  AddCustomContext("eigen_bench.eigen_has_openmp", "false");
#endif
  AddCustomContext("eigen_bench.thread_env", detail::threadEnvJson());
#if defined(EIGEN_BENCH_OPS_TOML_SHA256)
  AddCustomContext("eigen_bench.ops_toml_sha256", EIGEN_BENCH_OPS_TOML_SHA256);
#else
  AddCustomContext("eigen_bench.ops_toml_sha256", "");
#endif
}

}  // namespace eigen_bench

// Defining main here is safe alongside the benchmark_main archive that
// eigen_add_benchmark links: that archive member is only extracted while `main`
// is still undefined.
#define EIGEN_BENCH_COMPARISON_MAIN()                                   \
  int main(int argc, char** argv) {                                     \
    ::eigen_bench::publishArmContext();                                 \
    ::benchmark::Initialize(&argc, argv);                               \
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1; \
    ::benchmark::RunSpecifiedBenchmarks();                              \
    ::benchmark::Shutdown();                                            \
    return 0;                                                           \
  }                                                                     \
  static_assert(true, "")

#endif  // EIGEN_BENCHMARKS_COMPARISON_BENCH_COMPARE_H
