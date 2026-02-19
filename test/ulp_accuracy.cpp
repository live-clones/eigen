// Standalone tool to measure ULP accuracy of Eigen's vectorized math functions
// against either MPFR (high-precision reference) or std C++ math functions.
//
// Usage:
//   ./ulp_accuracy --func=sin --lo=0 --hi=6.2832 --threads=16
//   ./ulp_accuracy --func=exp --threads=16
//   ./ulp_accuracy --func=sin --ref=mpfr
//   ./ulp_accuracy --list
//
// Build:
//   cd build && cmake .. && make ulp_accuracy

#include <Eigen/Core>
#include <unsupported/Eigen/SpecialFunctions>

#ifdef EIGEN_HAS_MPFR
#include <mpfr.h>
#endif

#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// ULP distance (signed and absolute)
// ---------------------------------------------------------------------------

// Maps IEEE 754 float bits to a linear integer scale where adjacent floats
// are adjacent integers. The mapping is strictly monotonic:
//   -inf -> most negative, -0.0 -> -1, +0.0 -> 0, +inf -> most positive.
static inline int64_t float_to_linear(float x) {
  int32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  if (bits < 0) {
    // Negative floats: map so that -0.0 (0x80000000) -> -1,
    // -FLT_TRUE_MIN (0x80000001) -> -2, etc.
    bits = static_cast<int32_t>(INT32_MIN) - bits - 1;
  }
  return static_cast<int64_t>(bits);
}

// Returns (eigen_val - ref_val) in ULP space.
// Positive means Eigen overestimates, negative means it underestimates.
static inline int64_t signed_ulp_error(float eigen_val, float ref_val) {
  if (eigen_val == ref_val) return 0;  // also handles -0.0 == +0.0
  bool e_nan = std::isnan(eigen_val), r_nan = std::isnan(ref_val);
  if (e_nan && r_nan) return 0;
  if (e_nan || r_nan) return INT64_MAX;
  if (std::isinf(eigen_val) || std::isinf(ref_val)) return INT64_MAX;
  return float_to_linear(eigen_val) - float_to_linear(ref_val);
}

// ---------------------------------------------------------------------------
// Per-thread accumulator
// ---------------------------------------------------------------------------

struct alignas(128) ThreadResult {
  int64_t max_abs_ulp = 0;
  float max_ulp_at = 0.0f;
  float max_ulp_eigen = 0.0f;
  float max_ulp_ref = 0.0f;
  double abs_ulp_sum = 0.0;
  uint64_t count = 0;

  // Signed histogram: bins for errors in [-hist_width, +hist_width],
  // plus two overflow bins for < -hist_width and > +hist_width.
  // Layout: [<-W] [-W] [-W+1] ... [0] ... [W-1] [W] [>W]
  // Total bins = 2*hist_width + 3
  int hist_width = 0;
  std::vector<uint64_t> hist;

  void init(int w) {
    hist_width = w;
    hist.assign(2 * w + 3, 0);
  }

  void record(int64_t signed_err, float x, float eigen_val, float ref_val) {
    int64_t abs_err = signed_err < 0 ? -signed_err : signed_err;
    if (signed_err == INT64_MAX) abs_err = INT64_MAX;

    if (abs_err > max_abs_ulp) {
      max_abs_ulp = abs_err;
      max_ulp_at = x;
      max_ulp_eigen = eigen_val;
      max_ulp_ref = ref_val;
    }
    if (abs_err != INT64_MAX) {
      abs_ulp_sum += static_cast<double>(abs_err);
    }
    count++;

    // Histogram bin
    int bin;
    if (signed_err == INT64_MAX || signed_err > hist_width) {
      bin = 2 * hist_width + 2;  // overflow high
    } else if (signed_err < -hist_width) {
      bin = 0;  // overflow low
    } else {
      bin = static_cast<int>(signed_err) + hist_width + 1;
    }
    hist[bin]++;
  }
};

// ---------------------------------------------------------------------------
// Function registry
// ---------------------------------------------------------------------------

using EigenEval = std::function<void(Eigen::Ref<Eigen::ArrayXf>, const Eigen::Ref<const Eigen::ArrayXf>&)>;
using StdEval = std::function<float(float)>;

#ifdef EIGEN_HAS_MPFR
using MpfrEval = std::function<int(mpfr_t, const mpfr_t, mpfr_rnd_t)>;
#endif

struct FuncEntry {
  std::string name;
  EigenEval eigen_eval;
  StdEval std_eval;
#ifdef EIGEN_HAS_MPFR
  MpfrEval mpfr_eval;
#endif
  float default_lo;
  float default_hi;
};

// Helper: logistic = 1/(1+exp(-x))
static float std_logistic(float x) {
  if (x >= 0) {
    float e = std::exp(-x);
    return 1.0f / (1.0f + e);
  } else {
    float e = std::exp(x);
    return e / (1.0f + e);
  }
}

#ifdef EIGEN_HAS_MPFR
static int mpfr_logistic(mpfr_t rop, const mpfr_t op, mpfr_rnd_t rnd) {
  mpfr_t tmp, one;
  mpfr_init2(tmp, mpfr_get_prec(rop));
  mpfr_init2(one, mpfr_get_prec(rop));
  mpfr_set_ui(one, 1, rnd);
  mpfr_neg(tmp, op, rnd);
  mpfr_exp(tmp, tmp, rnd);
  mpfr_add(tmp, tmp, one, rnd);
  int ret = mpfr_div(rop, one, tmp, rnd);
  mpfr_clear(tmp);
  mpfr_clear(one);
  return ret;
}

static int mpfr_rsqrt(mpfr_t rop, const mpfr_t op, mpfr_rnd_t rnd) { return mpfr_rec_sqrt(rop, op, rnd); }

static int mpfr_exp2_wrap(mpfr_t rop, const mpfr_t op, mpfr_rnd_t rnd) {
  mpfr_t two;
  mpfr_init2(two, mpfr_get_prec(rop));
  mpfr_set_ui(two, 2, rnd);
  int ret = mpfr_pow(rop, two, op, rnd);
  mpfr_clear(two);
  return ret;
}

static int mpfr_log2_wrap(mpfr_t rop, const mpfr_t op, mpfr_rnd_t rnd) { return mpfr_log2(rop, op, rnd); }

// ndtri via Newton iteration on erf.
static int mpfr_ndtri(mpfr_t rop, const mpfr_t op, mpfr_rnd_t rnd) {
  mpfr_prec_t prec = mpfr_get_prec(rop);
  mpfr_t p, x, target, erf_x, diff, ex2, sqrtpi_2, sqrt2, tmp;
  mpfr_inits2(prec + 32, p, x, target, erf_x, diff, ex2, sqrtpi_2, sqrt2, tmp, (mpfr_ptr)0);

  mpfr_set(p, op, rnd);
  mpfr_mul_ui(target, p, 2, rnd);
  mpfr_sub_ui(target, target, 1, rnd);

  mpfr_const_pi(sqrtpi_2, rnd);
  mpfr_sqrt(sqrtpi_2, sqrtpi_2, rnd);
  mpfr_div_ui(sqrtpi_2, sqrtpi_2, 2, rnd);

  mpfr_set_ui(sqrt2, 2, rnd);
  mpfr_sqrt(sqrt2, sqrt2, rnd);

  double pd = mpfr_get_d(p, MPFR_RNDN);
  double t = 2.0 * pd - 1.0;
  double guess = 0.0;
  if (std::abs(t) < 0.7) {
    guess = t * 0.886226925452758;
  } else {
    double s = (t > 0) ? 1.0 : -1.0;
    double u = -std::log((1.0 - std::abs(t)) / 2.0);
    guess = s * std::sqrt(u) * 0.8;
  }
  mpfr_set_d(x, guess, rnd);

  for (int i = 0; i < 50; i++) {
    mpfr_erf(erf_x, x, rnd);
    mpfr_sub(diff, target, erf_x, rnd);
    if (mpfr_zero_p(diff)) break;
    mpfr_sqr(ex2, x, rnd);
    mpfr_exp(ex2, ex2, rnd);
    mpfr_mul(tmp, sqrtpi_2, ex2, rnd);
    mpfr_mul(tmp, tmp, diff, rnd);
    mpfr_add(x, x, tmp, rnd);
  }

  mpfr_mul(rop, sqrt2, x, rnd);
  mpfr_clears(p, x, target, erf_x, diff, ex2, sqrtpi_2, sqrt2, tmp, (mpfr_ptr)0);
  return 0;
}
#endif  // EIGEN_HAS_MPFR

static float std_ndtri(float p) {
  if (p <= 0.0f) return -std::numeric_limits<float>::infinity();
  if (p >= 1.0f) return std::numeric_limits<float>::infinity();
  if (p == 0.5f) return 0.0f;
  float sign = 1.0f;
  float pp = p;
  if (pp > 0.5f) {
    pp = 1.0f - pp;
    sign = -1.0f;
  }
  float t = std::sqrt(-2.0f * std::log(pp));
  float c0 = 2.515517f, c1 = 0.802853f, c2 = 0.010328f;
  float d1 = 1.432788f, d2 = 0.189269f, d3 = 0.001308f;
  float num = c0 + t * (c1 + t * c2);
  float den = 1.0f + t * (d1 + t * (d2 + t * d3));
  return sign * (t - num / den);
}

static std::vector<FuncEntry> build_func_table() {
  std::vector<FuncEntry> table;

#ifdef EIGEN_HAS_MPFR
#define ADD_FUNC(name, eigen_expr, std_expr, mpfr_fn, lo, hi)                                                      \
  table.push_back(                                                                                                 \
      {#name, [](Eigen::Ref<Eigen::ArrayXf> out, const Eigen::Ref<const Eigen::ArrayXf>& a) { out = eigen_expr; }, \
       [](float x) -> float { return std_expr; },                                                                  \
       [](mpfr_t r, const mpfr_t o, mpfr_rnd_t d) { return mpfr_fn(r, o, d); }, lo, hi})
#else
#define ADD_FUNC(name, eigen_expr, std_expr, mpfr_fn, lo, hi)                                                      \
  table.push_back(                                                                                                 \
      {#name, [](Eigen::Ref<Eigen::ArrayXf> out, const Eigen::Ref<const Eigen::ArrayXf>& a) { out = eigen_expr; }, \
       [](float x) -> float { return std_expr; }, lo, hi})
#endif

  constexpr float kInf = std::numeric_limits<float>::infinity();

  // clang-format off
  ADD_FUNC(sin,   a.sin(),   std::sin(x),   mpfr_sin,   -kInf, kInf);
  ADD_FUNC(cos,   a.cos(),   std::cos(x),   mpfr_cos,   -kInf, kInf);
  ADD_FUNC(tan,   a.tan(),   std::tan(x),   mpfr_tan,   -kInf, kInf);
  ADD_FUNC(asin,  a.asin(),  std::asin(x),  mpfr_asin,  -kInf, kInf);
  ADD_FUNC(acos,  a.acos(),  std::acos(x),  mpfr_acos,  -kInf, kInf);
  ADD_FUNC(atan,  a.atan(),  std::atan(x),  mpfr_atan,  -kInf, kInf);

  ADD_FUNC(sinh,  a.sinh(),  std::sinh(x),  mpfr_sinh,  -kInf, kInf);
  ADD_FUNC(cosh,  a.cosh(),  std::cosh(x),  mpfr_cosh,  -kInf, kInf);
  ADD_FUNC(tanh,  a.tanh(),  std::tanh(x),  mpfr_tanh,  -kInf, kInf);
  ADD_FUNC(asinh, a.asinh(), std::asinh(x), mpfr_asinh, -kInf, kInf);
  ADD_FUNC(acosh, a.acosh(), std::acosh(x), mpfr_acosh, -kInf, kInf);
  ADD_FUNC(atanh, a.atanh(), std::atanh(x), mpfr_atanh, -kInf, kInf);

  ADD_FUNC(exp,   a.exp(),     std::exp(x),    mpfr_exp,       -kInf, kInf);
  ADD_FUNC(exp2,  a.exp2(),    std::exp2(x),   mpfr_exp2_wrap, -kInf, kInf);
  ADD_FUNC(expm1, a.expm1(),   std::expm1(x),  mpfr_expm1,     -kInf, kInf);
  ADD_FUNC(log,   a.log(),     std::log(x),    mpfr_log,       -kInf, kInf);
  ADD_FUNC(log1p, a.log1p(),   std::log1p(x),  mpfr_log1p,     -kInf, kInf);
  ADD_FUNC(log10, a.log10(),   std::log10(x),  mpfr_log10,     -kInf, kInf);
  ADD_FUNC(log2,  a.log2(),    std::log2(x),   mpfr_log2_wrap, -kInf, kInf);

  ADD_FUNC(erf,   a.erf(),    std::erf(x),    mpfr_erf,      -kInf, kInf);
  ADD_FUNC(erfc,  a.erfc(),   std::erfc(x),   mpfr_erfc,     -kInf, kInf);
  ADD_FUNC(lgamma, a.lgamma(), std::lgamma(x), mpfr_lngamma, -kInf, kInf);

  ADD_FUNC(logistic, a.logistic(), std_logistic(x), mpfr_logistic, -kInf, kInf);
  ADD_FUNC(ndtri,    a.ndtri(),    std_ndtri(x),    mpfr_ndtri,    -kInf, kInf);

  ADD_FUNC(sqrt,  a.sqrt(),  std::sqrt(x),       mpfr_sqrt,  -kInf, kInf);
  ADD_FUNC(cbrt,  a.cbrt(),  std::cbrt(x),       mpfr_cbrt,  -kInf, kInf);
  ADD_FUNC(rsqrt, a.rsqrt(), 1.0f/std::sqrt(x),  mpfr_rsqrt, -kInf, kInf);
  // clang-format on

#undef ADD_FUNC
  return table;
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

static void worker(const FuncEntry& func, float lo, float hi, int batch_size, bool use_mpfr, ThreadResult& result) {
  Eigen::ArrayXf input(batch_size);
  Eigen::ArrayXf eigen_out(batch_size);
  std::vector<float> ref_out(batch_size);

#ifdef EIGEN_HAS_MPFR
  mpfr_t mp_in, mp_out;
  if (use_mpfr) {
    mpfr_init2(mp_in, 128);
    mpfr_init2(mp_out, 128);
  }
#else
  (void)use_mpfr;
#endif

  auto process_batch = [&](int n, const Eigen::ArrayXf& in, const Eigen::ArrayXf& eig) {
    for (int i = 0; i < n; i++) {
#ifdef EIGEN_HAS_MPFR
      if (use_mpfr) {
        mpfr_set_flt(mp_in, in[i], MPFR_RNDN);
        func.mpfr_eval(mp_out, mp_in, MPFR_RNDN);
        ref_out[i] = mpfr_get_flt(mp_out, MPFR_RNDN);
      } else
#endif
      {
        ref_out[i] = func.std_eval(in[i]);
      }
    }
    for (int i = 0; i < n; i++) {
      int64_t err = signed_ulp_error(eig[i], ref_out[i]);
      result.record(err, in[i], eig[i], ref_out[i]);
    }
  };

  int idx = 0;
  float x = lo;
  for (;;) {
    input[idx] = x;
    idx++;

    if (idx == batch_size) {
      func.eigen_eval(eigen_out, input);
      process_batch(batch_size, input, eigen_out);
      idx = 0;
    }

    if (x == hi) break;
    x = std::nextafter(x, std::numeric_limits<float>::infinity());
  }

  // Process remaining partial batch
  if (idx > 0) {
    auto partial_in = input.head(idx);
    auto partial_eigen = eigen_out.head(idx);
    func.eigen_eval(partial_eigen, partial_in);
    process_batch(idx, input, eigen_out);
  }

#ifdef EIGEN_HAS_MPFR
  if (use_mpfr) {
    mpfr_clear(mp_in);
    mpfr_clear(mp_out);
  }
#endif
}

// ---------------------------------------------------------------------------
// Range splitting for threads
// ---------------------------------------------------------------------------

static uint64_t count_floats_in_range(float lo, float hi) {
  if (lo > hi) return 0;
  return static_cast<uint64_t>(float_to_linear(hi) - float_to_linear(lo)) + 1;
}

static float advance_float(float x, uint64_t n) {
  int64_t lin = float_to_linear(x);
  lin += static_cast<int64_t>(n);
  // Reverse the float_to_linear mapping.
  int32_t ibits;
  if (lin < 0) {
    ibits = static_cast<int32_t>(INT32_MIN) - static_cast<int32_t>(lin) - 1;
  } else {
    ibits = static_cast<int32_t>(lin);
  }
  float result;
  std::memcpy(&result, &ibits, sizeof(result));
  return result;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

static void print_usage() {
  std::printf(
      "Usage: ulp_accuracy [options]\n"
      "  --func=NAME    Function to test (required unless --list)\n"
      "  --lo=VAL       Start of range (default: -inf)\n"
      "  --hi=VAL       End of range (default: +inf)\n"
      "  --threads=N    Number of threads (default: all cores)\n"
      "  --batch=N      Batch size for Eigen eval (default: 4096)\n"
      "  --ref=MODE     Reference: 'std' (default) or 'mpfr'\n"
      "  --hist_width=N Histogram half-width in ULPs (default: 10)\n"
      "  --list         List available functions\n");
}

int main(int argc, char* argv[]) {
  std::string func_name;
  float lo = std::numeric_limits<float>::quiet_NaN();
  float hi = std::numeric_limits<float>::quiet_NaN();
  int num_threads = static_cast<int>(std::thread::hardware_concurrency());
  int batch_size = 4096;
  int hist_width = 10;
  std::string ref_mode;
  bool list_funcs = false;

  if (num_threads == 0) num_threads = 4;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.substr(0, 7) == "--func=") {
      func_name = arg.substr(7);
    } else if (arg.substr(0, 5) == "--lo=") {
      std::string val = arg.substr(5);
      if (val == "inf")
        lo = std::numeric_limits<float>::infinity();
      else if (val == "-inf")
        lo = -std::numeric_limits<float>::infinity();
      else
        lo = std::stof(val);
    } else if (arg.substr(0, 5) == "--hi=") {
      std::string val = arg.substr(5);
      if (val == "inf")
        hi = std::numeric_limits<float>::infinity();
      else if (val == "-inf")
        hi = -std::numeric_limits<float>::infinity();
      else
        hi = std::stof(val);
    } else if (arg.substr(0, 10) == "--threads=") {
      num_threads = std::stoi(arg.substr(10));
    } else if (arg.substr(0, 8) == "--batch=") {
      batch_size = std::stoi(arg.substr(8));
    } else if (arg.substr(0, 6) == "--ref=") {
      ref_mode = arg.substr(6);
    } else if (arg.substr(0, 13) == "--hist_width=") {
      hist_width = std::stoi(arg.substr(13));
    } else if (arg == "--list") {
      list_funcs = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage();
      return 0;
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
      print_usage();
      return 1;
    }
  }

  auto table = build_func_table();

  if (list_funcs) {
    std::printf("Available functions:\n");
    for (const auto& f : table) {
      std::printf("  %s\n", f.name.c_str());
    }
    return 0;
  }

  if (func_name.empty()) {
    std::fprintf(stderr, "Error: --func=NAME is required (use --list to see available functions)\n");
    return 1;
  }

  const FuncEntry* entry = nullptr;
  for (const auto& f : table) {
    if (f.name == func_name) {
      entry = &f;
      break;
    }
  }
  if (!entry) {
    std::fprintf(stderr, "Error: unknown function '%s' (use --list to see available functions)\n", func_name.c_str());
    return 1;
  }

  // Determine reference mode (default: std)
  bool use_mpfr = false;
  if (ref_mode.empty() || ref_mode == "std") {
    use_mpfr = false;
  } else if (ref_mode == "mpfr") {
#ifdef EIGEN_HAS_MPFR
    use_mpfr = true;
#else
    std::fprintf(stderr, "Error: MPFR support not compiled in. Use --ref=std or rebuild with MPFR.\n");
    return 1;
#endif
  } else {
    std::fprintf(stderr, "Error: --ref must be 'std' or 'mpfr'\n");
    return 1;
  }

  // Determine range (default: full float range including infinities)
  if (std::isnan(lo)) lo = entry->default_lo;
  if (std::isnan(hi)) hi = entry->default_hi;

  uint64_t total_floats = count_floats_in_range(lo, hi);

  std::printf("Function: %s\n", func_name.c_str());
  std::printf("Range: [%g, %g]\n", double(lo), double(hi));
  std::printf("Floats to test: %lu\n", static_cast<unsigned long>(total_floats));
  std::printf("Reference: %s\n", use_mpfr ? "MPFR (128-bit)" : "std C++ math");
  std::printf("Threads: %d\n", num_threads);
  std::printf("Batch size: %d\n", batch_size);
  std::printf("\n");
  std::fflush(stdout);

  // Split range across threads
  if (static_cast<uint64_t>(num_threads) > total_floats) {
    num_threads = static_cast<int>(total_floats);
  }
  if (num_threads < 1) num_threads = 1;

  // Heap-allocate each ThreadResult separately to avoid false sharing.
  std::vector<std::unique_ptr<ThreadResult>> results;
  results.reserve(num_threads);
  for (int t = 0; t < num_threads; t++) {
    results.push_back(std::make_unique<ThreadResult>());
    results.back()->init(hist_width);
  }

  std::vector<std::thread> threads;
  uint64_t floats_per_thread = total_floats / num_threads;
  float chunk_lo = lo;

  for (int t = 0; t < num_threads; t++) {
    float chunk_hi;
    if (t == num_threads - 1) {
      chunk_hi = hi;
    } else {
      chunk_hi = advance_float(chunk_lo, floats_per_thread - 1);
    }
    threads.emplace_back(worker, std::cref(*entry), chunk_lo, chunk_hi, batch_size, use_mpfr, std::ref(*results[t]));
    chunk_lo = std::nextafter(chunk_hi, std::numeric_limits<float>::infinity());
  }

  auto start_time = std::chrono::steady_clock::now();
  for (auto& t : threads) t.join();
  auto end_time = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(end_time - start_time).count();

  // Reduce per-thread results
  ThreadResult global;
  global.init(hist_width);
  for (int t = 0; t < num_threads; t++) {
    const auto& r = *results[t];
    if (r.max_abs_ulp > global.max_abs_ulp) {
      global.max_abs_ulp = r.max_abs_ulp;
      global.max_ulp_at = r.max_ulp_at;
      global.max_ulp_eigen = r.max_ulp_eigen;
      global.max_ulp_ref = r.max_ulp_ref;
    }
    global.abs_ulp_sum += r.abs_ulp_sum;
    global.count += r.count;
    for (size_t b = 0; b < global.hist.size(); b++) {
      global.hist[b] += r.hist[b];
    }
  }

  double mean_ulp = global.count > 0 ? global.abs_ulp_sum / global.count : 0.0;

  std::printf("Results:\n");
  std::printf("  Floats tested: %lu\n", static_cast<unsigned long>(global.count));
  std::printf("  Time: %.2f seconds (%.1f Mfloats/s)\n", elapsed, global.count / elapsed / 1e6);
  if (global.max_abs_ulp == INT64_MAX) {
    std::printf("  Max |ULP error|: inf\n");
  } else {
    std::printf("  Max |ULP error|: %ld\n", static_cast<long>(global.max_abs_ulp));
  }
  std::printf("    at x = %.9g (Eigen=%.9g, ref=%.9g)\n", double(global.max_ulp_at), double(global.max_ulp_eigen),
              double(global.max_ulp_ref));
  std::printf("  Mean |ULP error|: %.4f\n", mean_ulp);
  std::printf("\n");

  // Print signed error histogram
  std::printf("Signed ULP error histogram [-%d, +%d]:\n", hist_width, hist_width);
  int nbins = 2 * hist_width + 3;
  for (int b = 0; b < nbins; b++) {
    if (global.hist[b] == 0) continue;
    double pct = 100.0 * global.hist[b] / global.count;
    if (b == 0) {
      std::printf("  <%-4d: %12lu (%7.3f%%)\n", -hist_width, static_cast<unsigned long>(global.hist[b]), pct);
    } else if (b == nbins - 1) {
      std::printf("  >%-4d: %12lu (%7.3f%%)\n", hist_width, static_cast<unsigned long>(global.hist[b]), pct);
    } else {
      int err = b - hist_width - 1;
      std::printf("  %-5d: %12lu (%7.3f%%)\n", err, static_cast<unsigned long>(global.hist[b]), pct);
    }
  }

  return 0;
}
