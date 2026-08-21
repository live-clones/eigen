// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Registration machinery for the cross-library comparison benchmarks. Benchmark
// names follow the grammar in CONTRACTS.md section 1, which run.py parses back:
//
//     op "/" arm "/" scalar ( "/" dimname ":" value )+ [ "/threads:" n ]
//
// The reference arm registers only when the build linked a vendor library, so the
// same source is a valid Eigen-only benchmark with no vendor present.

#ifndef EIGEN_BENCHMARKS_COMPARISON_BENCH_COMPARE_H
#define EIGEN_BENCHMARKS_COMPARISON_BENCH_COMPARE_H

#include <benchmark/benchmark.h>

#include <Eigen/Core>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// EIGEN_BENCH_REFERENCE_ARM is the only macro that changes what is registered,
// and CMake passes it as a BARE TOKEN (-DEIGEN_BENCH_REFERENCE_ARM=openblas) so
// that no quoting has to survive the CMake command line. Every other macro here
// is optional metadata with a defined fallback, so a hand-built binary still
// produces a well-formed context.

#define EIGEN_BENCH_STRINGIZE_(x) #x
#define EIGEN_BENCH_STRINGIZE(x) EIGEN_BENCH_STRINGIZE_(x)

#ifdef EIGEN_BENCH_REFERENCE_ARM
#define EIGEN_BENCH_REFERENCE_ARM_STR EIGEN_BENCH_STRINGIZE(EIGEN_BENCH_REFERENCE_ARM)
#else
#define EIGEN_BENCH_REFERENCE_ARM_STR ""
#endif

// Both arms of ONE grid point, registered adjacently. Google Benchmark runs
// instances in registration order, so a whole arm's grid followed by the other's
// -- what an arrow-chain grid applied to two Benchmark objects produces -- puts
// minutes of thermal and background drift between the two numbers a ratio is
// formed from, always in the same direction. .agents/benchmarking.md step 4
// requires the alternating order; this is its registration-time form, which is
// why callers drive the macro from a list of points rather than a range.
//
// DIM_NAMES must arrive as the NAME of an object-like macro, never as a literal
// {"m", "n"}: braces do not protect commas from macro argument splitting. For
// the same reason the two arms are spelled out rather than delegated to a shared
// per-arm macro, which would split DIM_NAMES on the way down. SCALAR_TYPE is one
// argument for the same reason; eigen_bench::c32_t and c64_t below exist so a
// comma-spelled type can be passed.

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

// Integer width of the reference library's Fortran interface. Eigen::BlasIndex
// (Eigen/src/Core/util/BlasTypes.h) is already that switch, keyed off
// EIGEN_64BIT_BLAS; aliasing it keeps ONE BLAS integer width in the binary. Two
// independent switches would let these benchmarks and Eigen's own BLAS backend
// disagree, which passes the compiler and the linker and then corrupts every
// by-pointer Fortran argument.
using BlasInt = Eigen::BlasIndex;

#ifdef EIGEN_BENCH_REFERENCE_ILP64
// A hard error rather than a silent narrowing: EIGEN_64BIT_BLAS is ABI-affecting
// and must be defined for every translation unit, so the vendor table selecting
// an ILP64 arm is not on its own enough to produce a 64-bit BlasIndex.
static_assert(sizeof(BlasInt) == 8,
              "EIGEN_BENCH_REFERENCE_ILP64 selects a 64-bit-integer reference BLAS, but Eigen::BlasIndex is 32-bit: "
              "the build must define EIGEN_64BIT_BLAS as well.");
#endif

// Why a cell was skipped is not free text: result_schema.json enumerates the
// vocabulary, and the reasons mean opposite things on a published page. "Too
// large for this machine" is a fact about the host; "the result disagrees with
// Eigen" is a defect in the library. SkipWithError offers only one free-text
// channel, so the reason travels in a machine-readable envelope that run.py
// parses. A message with no envelope is a genuine runtime error, which is what a
// plain SkipWithError elsewhere in the tree already means.
#define EIGEN_BENCH_SKIP_ENVELOPE_OPEN "[eigen-bench:skip:"
#define EIGEN_BENCH_SKIP_ENVELOPE_CLOSE "] "

// `reason` must be one of result_schema.json's not_measured reasons; run.py
// checks the token against that enum and falls back to runtime_error if it does
// not recognise it, so a typo degrades rather than inventing a category.
inline void skipWithReason(benchmark::State& state, const char* reason, const std::string& detail) {
  state.SkipWithError(std::string(EIGEN_BENCH_SKIP_ENVELOPE_OPEN) + reason + EIGEN_BENCH_SKIP_ENVELOPE_CLOSE + detail);
}

// Bytes a benchmark may allocate for its operands, zero meaning unenforced.
// Google Benchmark writes its output only after every benchmark in the
// invocation has finished, so a cell that outgrows the host does not fail alone:
// the allocation throws or the OS kills the process, and every cell already
// measured is lost with it.
inline std::size_t memoryBudgetBytes() {
  const char* raw = std::getenv("EIGEN_BENCH_MEMORY_BUDGET_BYTES");
  if (raw == nullptr) return 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(raw, &end, 10);
  if (end == raw || value == 0) return 0;
  return static_cast<std::size_t>(value);
}

// Call BEFORE allocating. `bytes` is what the operands will occupy; a caller
// that under-reports gets no protection, so it should count every array that is
// live at once, not just the largest.
inline bool skipIfOverMemoryBudget(benchmark::State& state, double bytes) {
  const std::size_t budget = memoryBudgetBytes();
  if (budget == 0 || bytes <= static_cast<double>(budget)) return false;
  // Scaled to a unit that has significant digits at both ends: a GiB-only
  // rendering prints "0.00 GiB, budget is 0.00 GiB" for a small budget, which
  // states nothing on a page that has to justify why a cell is missing.
  const double mib = 1024.0 * 1024.0;
  const bool use_gib = bytes >= 1024.0 * mib && static_cast<double>(budget) >= 1024.0 * mib;
  const double scale = use_gib ? 1024.0 * mib : mib;
  const char* unit = use_gib ? " GiB" : " MiB";
  std::ostringstream message;
  message.setf(std::ios::fixed);
  message.precision(2);
  message << "operands need " << bytes / scale << unit << ", budget is " << static_cast<double>(budget) / scale << unit;
  skipWithReason(state, "out_of_memory", message.str());
  return true;
}

template <typename Scalar>
using ColMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Scalar>
using ColVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

// Does a dimension survive the narrowing to the reference arm's integer width?
// Eigen::Index is 64-bit on every platform these benchmarks run on, so an LP64
// reference BLAS narrows every dimension on the way out. The grid in ops.toml
// stays well inside 32 bits today, but a machine file that adds a larger group
// must not silently pass a truncated extent to Fortran: a caller checks this
// before it converts, and skips the point with an error if it fails.
inline bool fitsBlasInt(Eigen::Index value) {
  return value >= 0 &&
         static_cast<std::uintmax_t>(value) <= static_cast<std::uintmax_t>((std::numeric_limits<BlasInt>::max)());
}

// A structured skip, so the cell is filed as a property of the build rather than
// as a library failure: the width is decided by EIGEN_64BIT_BLAS and the vendor
// table, not by any kernel.
template <typename... Dims>
bool skipIfDimsExceedBlasInt(benchmark::State& state, Dims... dims) {
  const Eigen::Index values[] = {static_cast<Eigen::Index>(dims)...};
  for (const Eigen::Index value : values) {
    if (!fitsBlasInt(value)) {
      skipWithReason(state, "shape_unsupported",
                     "dimension " + std::to_string(value) + " does not fit the reference library's " +
                         std::to_string(8 * sizeof(BlasInt)) + "-bit integer width");
      return true;
    }
  }
  return false;
}

// Memoises the untimed correctness check per shape. Correctness is a
// deterministic property of (Scalar, kernel, shape), but Google Benchmark enters
// a body once per repetition plus several times while it searches for an
// iteration count, and the check runs a full untimed operation -- so at the
// large end of a grid, re-checking every entry costs more than the measurement.
// Held as a function-local static of a function template, it is already per
// (Scalar, Kernel); the mutex keeps the set consistent under a threaded runner.
template <typename Key>
class ValidatedShapes {
 public:
  bool contains(const Key& key) const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return set_.count(key) != 0;
  }
  // Only after the check passed: a failure must not mark the shape good for the
  // entries that follow it.
  void insert(const Key& key) {
    const std::lock_guard<std::mutex> guard(mutex_);
    set_.insert(key);
  }

 private:
  mutable std::mutex mutex_;
  std::set<Key> set_;
};

// One numerical policy for every comparison: relative to the larger of the two
// norms, scaled by sqrt(contraction length) because that is where the error
// accumulates.
//
// Both non-finite cases must be rejected explicitly, and neither falls out of
// the relative test. Every comparison against NaN is false, so `<=` under the
// negation rejects it; but an infinite entry makes BOTH sides infinite -- the
// error norm because inf minus a finite number is inf, the bound because
// tolerance * inf is inf -- and `inf <= inf` is true. A kernel that overflowed
// or returned uninitialised memory would be cached as validated and its rate
// published. Hence the finiteness test on all three norms before the bound.
template <typename Derived, typename OtherDerived>
bool agreesWithEigen(const Eigen::MatrixBase<Derived>& expected, const Eigen::MatrixBase<OtherDerived>& actual,
                     Eigen::Index contraction_length) {
  using RealScalar = typename Eigen::NumTraits<typename Derived::Scalar>::Real;
  const RealScalar expected_norm = expected.norm();
  const RealScalar actual_norm = actual.norm();
  const RealScalar error = (actual - expected).norm();
  if (!(Eigen::numext::isfinite)(expected_norm) || !(Eigen::numext::isfinite)(actual_norm) ||
      !(Eigen::numext::isfinite)(error)) {
    return false;
  }
  const RealScalar tolerance =
      RealScalar(64) * Eigen::numext::sqrt(RealScalar(contraction_length)) * Eigen::NumTraits<RealScalar>::epsilon();
  const RealScalar magnitude = Eigen::numext::maxi(expected_norm, actual_norm);
  return !!(error <= tolerance * magnitude);
}

// Version queries declared rather than included, so a build needs the reference
// library but not its development headers. CMake must define the matching
// EIGEN_BENCH_HAVE_* macro only after a check that compiles AND links the
// symbol; defining it on a guess turns a missing query into a link failure.

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

// Must cover every variable run.py's THREAD_COUNT_ENV_VARS and THREAD_FIXED_ENV
// set, or a run records a cap the harness applied as absent. The affinity
// variables at the end are not set by run.py, but they change what a thread
// count means, so a value the environment carried in is worth reporting.
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

// Wraps the console reporter purely to notice SkipWithError. Google Benchmark
// records an errored run in its output but does not make the process fail, so
// without this a benchmark whose result disagrees with Eigen is indistinguishable
// from a passing one to ctest.
class ErrorTrackingReporter : public ::benchmark::ConsoleReporter {
 public:
  void ReportRuns(const std::vector<Run>& reports) override {
    for (const Run& run : reports) {
      if (run.skipped == ::benchmark::internal::SkippedWithError && !carriesSkipEnvelope(run.report_label) &&
          !carriesSkipEnvelope(run.skip_message)) {
        errored_ = true;
      }
    }
    ConsoleReporter::ReportRuns(reports);
  }
  bool anyErrored() const { return errored_; }

 private:
  // A structured skip is not an error: skipIfOverMemoryBudget and
  // skipIfDimsExceedBlasInt report facts about the host and the build, each with
  // its own not_measured reason. Only an unlabelled SkipWithError -- a kernel
  // disagreeing with Eigen -- makes the process fail, so that an over-budget
  // cell does not colour the exit status of every cell measured beside it.
  static bool carriesSkipEnvelope(const std::string& message) {
    return message.rfind(EIGEN_BENCH_SKIP_ENVELOPE_OPEN, 0) == 0;
  }

  bool errored_ = false;
};

}  // namespace eigen_bench

// Defining main here is safe alongside the benchmark_main archive that
// eigen_add_benchmark links: that archive member is only extracted while `main`
// is still undefined.
#define EIGEN_BENCH_COMPARISON_MAIN()                                        \
  int main(int argc, char** argv) {                                          \
    ::eigen_bench::ErrorTrackingReporter reporter;                           \
    ::eigen_bench::publishArmContext();                                      \
    ::benchmark::Initialize(&argc, argv);                                    \
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;      \
    ::benchmark::RunSpecifiedBenchmarks(&reporter);                          \
    ::benchmark::Shutdown();                                                 \
    /* Google Benchmark exits 0 for a run it recorded as errored, which   */ \
    /* would leave the smoke test green with the reference arm disagreeing*/ \
    /* with Eigen on every shape.                                         */ \
    return reporter.anyErrored() ? 1 : 0;                                    \
  }                                                                          \
  static_assert(true, "")

#endif  // EIGEN_BENCHMARKS_COMPARISON_BENCH_COMPARE_H
