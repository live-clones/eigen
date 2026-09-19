// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

// Registration machinery for the cross-library comparison benchmarks. Benchmark
// names follow the grammar run.py parses back:
//
//     op "/" arm "/" scalar ( "/" dimname ":" value )+
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

// EIGEN_BENCH_REFERENCE_ARM is the only macro that changes what is registered,
// and CMake passes it as a BARE TOKEN (-DEIGEN_BENCH_REFERENCE_ARM=openblas) so
// that no quoting has to survive the CMake command line. Every other macro here
// is metadata with a defined fallback.

#define EIGEN_BENCH_STRINGIZE_(x) #x
#define EIGEN_BENCH_STRINGIZE(x) EIGEN_BENCH_STRINGIZE_(x)

#ifdef EIGEN_BENCH_REFERENCE_ARM
#define EIGEN_BENCH_REFERENCE_ARM_STR EIGEN_BENCH_STRINGIZE(EIGEN_BENCH_REFERENCE_ARM)
#else
#define EIGEN_BENCH_REFERENCE_ARM_STR ""
#endif

// Both arms of ONE grid point, registered adjacently. Google Benchmark runs
// instances in registration order, so a whole arm's grid followed by the other's
// would put minutes of thermal and background drift between the two numbers a
// ratio is formed from, always in the same direction.
//
// DIM_NAMES must arrive as the NAME of an object-like macro, never as a literal
// {"m", "n"}: braces do not protect commas from macro argument splitting. For
// the same reason SCALAR_TYPE is one argument; eigen_bench::c32_t and c64_t
// below exist so a comma-spelled type can be passed.
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

// Aliases for the scalar tags whose C++ spelling contains a comma.
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
static_assert(sizeof(BlasInt) == 8,
              "EIGEN_BENCH_REFERENCE_ILP64 selects a 64-bit-integer reference BLAS, but Eigen::BlasIndex is 32-bit: "
              "the build must define EIGEN_64BIT_BLAS as well.");
#endif

// Bytes a benchmark may allocate for its operands, zero meaning unenforced.
// Google Benchmark writes its output only after every benchmark in the
// invocation has finished, so a cell that outgrows the host loses every cell
// already measured beside it; a skipped cell loses only itself.
inline std::size_t memoryBudgetBytes() {
  const char* raw = std::getenv("EIGEN_BENCH_MEMORY_BUDGET_BYTES");
  if (raw == nullptr) return 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(raw, &end, 10);
  if (end == raw || value == 0) return 0;
  return static_cast<std::size_t>(value);
}

// Call BEFORE allocating, with every array that is live at once counted.
inline bool skipIfOverMemoryBudget(benchmark::State& state, double bytes) {
  const std::size_t budget = memoryBudgetBytes();
  if (budget == 0 || bytes <= static_cast<double>(budget)) return false;
  const double mib = 1024.0 * 1024.0;
  std::ostringstream message;
  message.setf(std::ios::fixed);
  message.precision(1);
  message << "over memory budget: operands need " << bytes / mib << " MiB, EIGEN_BENCH_MEMORY_BUDGET_BYTES is "
          << static_cast<double>(budget) / mib << " MiB";
  state.SkipWithError(message.str());
  return true;
}

template <typename Scalar>
using ColMatrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Scalar>
using ColVector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

// Eigen::Index is 64-bit, so an LP64 reference BLAS narrows every dimension on
// the way out; a caller checks before it converts.
inline bool fitsBlasInt(Eigen::Index value) {
  return value >= 0 &&
         static_cast<std::uintmax_t>(value) <= static_cast<std::uintmax_t>((std::numeric_limits<BlasInt>::max)());
}

template <typename... Dims>
bool skipIfDimsExceedBlasInt(benchmark::State& state, Dims... dims) {
  const Eigen::Index values[] = {static_cast<Eigen::Index>(dims)...};
  for (const Eigen::Index value : values) {
    if (!fitsBlasInt(value)) {
      state.SkipWithError("dimension " + std::to_string(value) + " does not fit the reference library's " +
                          std::to_string(8 * sizeof(BlasInt)) + "-bit integer width");
      return true;
    }
  }
  return false;
}

// Memoises the untimed correctness check per shape. Google Benchmark enters a
// body once per repetition plus several times while it searches for an
// iteration count, and the check runs a full untimed operation, so at the large
// end of a grid re-checking every entry would cost more than the measurement.
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
// accumulates. Non-finite values are rejected explicitly: every comparison
// against NaN is false, but an infinite entry makes both the error and the bound
// infinite and `inf <= inf` holds.
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

}  // namespace eigen_bench

// Version queries declared rather than included, so a build needs the reference
// library but not its development headers. CMake defines the matching
// EIGEN_BENCH_HAVE_* macro only after a check that links the symbol.
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

inline const char* referenceArmKey() { return EIGEN_BENCH_REFERENCE_ARM_STR; }

inline bool hasReference() { return referenceArmKey()[0] != '\0'; }

// The library's own answer first, then the version CMake found at configure
// time, then "unknown".
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
#if defined(EIGEN_BENCH_REFERENCE_VERSION_FALLBACK)
  {
    const std::string fallback = EIGEN_BENCH_REFERENCE_VERSION_FALLBACK;
    if (!fallback.empty()) return fallback;
  }
#endif
  return "unknown";
}

// Everything run.py records about the build, published into the Google
// Benchmark JSON context. MUST run before benchmark::Initialize().
inline void publishArmContext() {
  using benchmark::AddCustomContext;
#define EIGEN_BENCH_CONTEXT_(key, macro) AddCustomContext("eigen_bench." key, macro)
#if defined(EIGEN_BENCH_COMPILER_ID)
  EIGEN_BENCH_CONTEXT_("compiler_id", EIGEN_BENCH_COMPILER_ID);
#endif
#if defined(EIGEN_BENCH_COMPILER_VERSION)
  EIGEN_BENCH_CONTEXT_("compiler_version", EIGEN_BENCH_COMPILER_VERSION);
#endif
#if defined(EIGEN_BENCH_CXX_STANDARD)
  EIGEN_BENCH_CONTEXT_("cxx_standard", EIGEN_BENCH_CXX_STANDARD);
#endif
#if defined(EIGEN_BENCH_CXX_FLAGS)
  EIGEN_BENCH_CONTEXT_("cxx_flags", EIGEN_BENCH_CXX_FLAGS);
#endif
#if defined(EIGEN_BENCH_ISA_TARGET)
  EIGEN_BENCH_CONTEXT_("isa_target", EIGEN_BENCH_ISA_TARGET);
#endif
#if defined(EIGEN_BENCH_EIGEN_COMMIT)
  EIGEN_BENCH_CONTEXT_("eigen_commit", EIGEN_BENCH_EIGEN_COMMIT);
#endif
#if defined(EIGEN_BENCH_REFERENCE_LIBRARY_NAME)
  EIGEN_BENCH_CONTEXT_("reference_library_name", EIGEN_BENCH_REFERENCE_LIBRARY_NAME);
#endif
#if defined(EIGEN_BENCH_REFERENCE_PATH)
  EIGEN_BENCH_CONTEXT_("reference_library_path", EIGEN_BENCH_REFERENCE_PATH);
#endif
#undef EIGEN_BENCH_CONTEXT_
  AddCustomContext("eigen_bench.reference_arm", referenceArmKey());
  AddCustomContext("eigen_bench.reference_library_version", referenceLibraryVersion());
  AddCustomContext("eigen_bench.eigen_version", EIGEN_VERSION_STRING);
}

}  // namespace eigen_bench

// Defining main here is safe alongside the benchmark_main archive that
// eigen_add_benchmark links: that archive member is only extracted while `main`
// is still undefined. A reference result that disagrees with Eigen's is reported
// through SkipWithError, which run.py records per cell.
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
