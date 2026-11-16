#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Standalone reproduction of the legacy MFM reaction-coordinate generator's
// device loop.  The only A/B change is where the per-trajectory bottom OU
// state lives: dynamic shared memory (baseline) versus thread-private __half
// storage (candidate).  The half conversion and Philox call order are kept.

#ifndef MFM_DBOT
#define MFM_DBOT 37
#endif
constexpr int DBOT = MFM_DBOT;
constexpr int OUTPUTS = 17;

#ifndef MFM_CONST_COEFF
#define MFM_CONST_COEFF 0
#endif

#ifndef MFM_CACHE_INIT_BOT
#define MFM_CACHE_INIT_BOT 0
#endif

#ifndef MFM_REUSE_SHARED_REDUCE
#define MFM_REUSE_SHARED_REDUCE 0
#endif

#if MFM_CONST_COEFF
__constant__ float c_coeff[DBOT];
#endif

__device__ __forceinline__ float coeff_at(const float* shared_coeff, int j) {
#if MFM_CONST_COEFF
  return c_coeff[j];
#else
  return shared_coeff[j];
#endif
}

__device__ __forceinline__ void potential_tan(
    float x1, float x2, float x3, float inv_norm, float norm,
    float inv_temperature, float c1, float c2, float c3, float c4,
    float& g1, float& g2, float& g3) {
  const float xn1 = x1 * inv_norm;
  const float xn2 = x2 * inv_norm;
  const float xn3 = x3 * inv_norm;
  const float q1_1 = -0.43301270189221932338f;
  const float q1_2 = -0.75f;
  const float q1_3 = -0.33333333333333333333f;
  const float q2_1 = 0.86602540378443864676f;
  const float q2_2 = 0.0f;
  const float q2_3 = -0.33333333333333333333f;
  const float q3_1 = -0.43301270189221932338f;
  const float q3_2 = 0.75f;
  const float q3_3 = -0.33333333333333333333f;
  const float q4_1 = 0.0f;
  const float q4_2 = 0.0f;
  const float q4_3 = 1.0f;
  const float t1 = fmaf(q1_1, xn1, fmaf(q1_2, xn2, q1_3 * xn3));
  const float t2 = fmaf(q2_1, xn1, fmaf(q2_2, xn2, q2_3 * xn3));
  const float t3 = fmaf(q3_1, xn1, fmaf(q3_2, xn2, q3_3 * xn3));
  const float t4 = fmaf(q4_1, xn1, fmaf(q4_2, xn2, q4_3 * xn3));
  const float s1 = c1 * __expf(c1 * t1);
  const float s2 = c2 * __expf(c2 * t2);
  const float s3 = c3 * __expf(c3 * t3);
  const float s4 = c4 * __expf(c4 * t4);
  const float v1_1 = s1 * (q1_1 - t1 * xn1);
  const float v1_2 = s1 * (q1_2 - t1 * xn2);
  const float v1_3 = s1 * (q1_3 - t1 * xn3);
  const float v2_1 = s2 * (q2_1 - t2 * xn1);
  const float v2_2 = s2 * (q2_2 - t2 * xn2);
  const float v2_3 = s2 * (q2_3 - t2 * xn3);
  const float v3_1 = s3 * (q3_1 - t3 * xn1);
  const float v3_2 = s3 * (q3_2 - t3 * xn2);
  const float v3_3 = s3 * (q3_3 - t3 * xn3);
  const float v4_1 = s4 * (q4_1 - t4 * xn1);
  const float v4_2 = s4 * (q4_2 - t4 * xn2);
  const float v4_3 = s4 * (q4_3 - t4 * xn3);
  const float scale = -inv_temperature * inv_norm;
  g1 = (v1_1 + v2_1 + v3_1 + v4_1) * scale;
  g2 = (v1_2 + v2_2 + v3_2 + v4_2) * scale;
  g3 = (v1_3 + v2_3 + v3_3 + v4_3) * scale;
}

__device__ __forceinline__ void hit_tests(
    float x1, float x2, float x3, float norm, float inv_hyp, float hypotxy,
    float sum_bot2, float delta_r, float tan_el_lo, float tan_el_hi,
    float cos_dtheta, float tan_dtheta, bool& hit_a, bool& hit_b,
    bool& hit_c, bool& hit_d) {
  const bool flag_norm = (fabsf(norm - 1.0f) < delta_r) &&
                         (sum_bot2 < delta_r * delta_r);
  const float ratio = x3 * inv_hyp;
  const float ux = x1 * inv_hyp;
  const float uy = x2 * inv_hyp;
  const bool el_ok = (ratio >= tan_el_lo) && (ratio <= tan_el_hi);
  const float dot_a = fmaf(ux, -0.5f, uy * -0.86602540378443864676f);
  const float dot_b = ux;
  const float dot_c = fmaf(ux, -0.5f, uy * 0.86602540378443864676f);
  hit_a = flag_norm && el_ok && (dot_a > cos_dtheta);
  hit_b = flag_norm && el_ok && (dot_b > cos_dtheta);
  hit_c = flag_norm && el_ok && (dot_c > cos_dtheta);
  hit_d = flag_norm && (x3 > 0.0f) && (hypotxy < x3 * tan_dtheta);
}

template <int MODE>
struct BottomState;

template <>
struct BottomState<0> {
  __half* shared = nullptr;
  int stride = 0;
  int tid = 0;
  __device__ __forceinline__ void bind(__half* p, int s, int t) {
    shared = p; stride = s; tid = t;
  }
  __device__ __forceinline__ void init(const float* src) {
    for (int j = 0; j < DBOT; ++j)
      shared[j * stride + tid] = __float2half_rn(src[j]);
  }
  __device__ __forceinline__ __half load(int j) const {
    return shared[j * stride + tid];
  }
  __device__ __forceinline__ void store(int j, __half v) {
    shared[j * stride + tid] = v;
  }
};

template <>
struct BottomState<1> {
  __half value[DBOT];
  __device__ __forceinline__ void bind(__half*, int, int) {}
  __device__ __forceinline__ void init(const float* src) {
    for (int j = 0; j < DBOT; ++j) value[j] = __float2half_rn(src[j]);
  }
  __device__ __forceinline__ __half load(int j) const { return value[j]; }
  __device__ __forceinline__ void store(int j, __half v) { value[j] = v; }
};

template <>
struct BottomState<2> {
  // Slot zero stores component zero by itself; slots 1.. hold pairs.
  // This is ceil((DBOT + 1) / 2), not ceil(DBOT / 2) for even DBOT.
  __half2 value[1 + DBOT / 2];
  __device__ __forceinline__ void bind(__half*, int, int) {}
  __device__ __forceinline__ void init(const float* src) {
    value[0] = __halves2half2(__float2half_rn(src[0]), __float2half_rn(0.0f));
    for (int j = 1; j < DBOT; j += 2) {
      const __half lo = __float2half_rn(src[j]);
      const __half hi = (j + 1 < DBOT) ? __float2half_rn(src[j + 1])
                                       : __float2half_rn(0.0f);
      value[(j + 1) / 2] = __halves2half2(lo, hi);
    }
  }
  __device__ __forceinline__ __half load(int j) const {
    const __half2 v = value[j == 0 ? 0 : (j + 1) / 2];
    return (j == 0 || (j & 1)) ? __low2half(v) : __high2half(v);
  }
  __device__ __forceinline__ void store(int j, __half v) {
    const int p = j == 0 ? 0 : (j + 1) / 2;
    __half2 old = value[p];
    value[p] = (j == 0 || (j & 1)) ? __halves2half2(v, __high2half(old))
                                   : __halves2half2(__low2half(old), v);
  }
  __device__ __forceinline__ __half2 load_pair(int j) const {
    return value[(j + 1) / 2];
  }
  __device__ __forceinline__ void store_pair(int j, __half lo, __half hi) {
    value[(j + 1) / 2] = __halves2half2(lo, hi);
  }
};

template <int MODE>
__global__ void mfm_kernel(
    const float* __restrict__ x_top, const float* __restrict__ x_bot,
    const float* __restrict__ coeff, int n_curve, int n_sims, int n_t,
    float dt, float term_t, float c1tmp, float sigma, float delta_theta,
    float delta_r, float tan_el_lo, float tan_el_hi, float cos_dtheta,
    float tan_dtheta, std::uint64_t base_seed, float temperature, float c1,
    float c2, float c3, float c4, std::uint64_t curve_offset,
    std::uint64_t path_stride,
    float* __restrict__ output) {
  const int curve = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);

#if MFM_CONST_COEFF
  const float* sh_coeff = nullptr;
#else
  __shared__ float sh_coeff[DBOT];
  for (int j = tid; j < DBOT; j += blockDim.x)
    sh_coeff[j] = coeff[j];
#endif
#if !MFM_CONST_COEFF
  __syncthreads();
#endif

#if MFM_CACHE_INIT_BOT
  __shared__ float sh_init_bot[DBOT];
  for (int j = tid; j < DBOT; j += blockDim.x)
    sh_init_bot[j] = x_bot[DBOT * curve + j];
  __syncthreads();
#endif
  extern __shared__ unsigned char raw[];
  float* reduce = reinterpret_cast<float*>(raw);
  __half* sh_bot = reinterpret_cast<__half*>(raw);
#if !MFM_REUSE_SHARED_REDUCE
  sh_bot = reinterpret_cast<__half*>(raw +
      sizeof(float) * OUTPUTS * blockDim.x);
#endif
  BottomState<MODE> bot;
  bot.bind(sh_bot, blockDim.x, tid);

  float x10 = x_top[3 * curve + 0];
  float x20 = x_top[3 * curve + 1];
  float x30 = x_top[3 * curve + 2];
  float totals[OUTPUTS] = {};

  for (int sim = tid; sim < n_sims; sim += blockDim.x) {
#if MFM_CACHE_INIT_BOT
    bot.init(sh_init_bot);
#else
    bot.init(x_bot + DBOT * curve);
#endif

    curandStatePhilox4_32_10_t state;
    const std::uint64_t subseq =
        (curve_offset + static_cast<std::uint64_t>(curve)) *
            path_stride +
        static_cast<std::uint64_t>(sim);
    curand_init(static_cast<unsigned long long>(base_seed),
                static_cast<unsigned long long>(subseq), 0ULL, &state);

    float x1 = x10, x2 = x20, x3 = x30;
    int first[4] = {n_t + 1, n_t + 1, n_t + 1, n_t + 1};
    int spent[4] = {0, 0, 0, 0};

    for (int it = 0; it < n_t; ++it) {
      const int it1 = it + 1;
      const float4 r0 = curand_normal4(&state);
      const float r2 = fmaf(x1, x1, fmaf(x2, x2, x3 * x3));
      const float inv_norm = rsqrtf(r2);
      const float norm = r2 * inv_norm;
      float g1, g2, g3;
      potential_tan(x1, x2, x3, inv_norm, norm, 1.0f / temperature,
                    c1, c2, c3, c4, g1, g2, g3);
      const float xc1 = x1 * inv_norm;
      const float xc2 = x2 * inv_norm;
      const float xc3 = x3 * inv_norm;
      g1 = fmaf(c1tmp, x1 - xc1, g1);
      g2 = fmaf(c1tmp, x2 - xc2, g2);
      g3 = fmaf(c1tmp, x3 - xc3, g3);
      x1 = fmaf(-dt, g1, x1) + sigma * r0.x;
      x2 = fmaf(-dt, g2, x2) + sigma * r0.y;
      x3 = fmaf(-dt, g3, x3) + sigma * r0.z;

      float sum_bot2 = 0.0f;
      float b0 = __half2float(bot.load(0));
      float bj = fmaf(b0, coeff_at(sh_coeff, 0), sigma * r0.w);
      bot.store(0, __float2half_rn(bj));
      sum_bot2 = bj * bj;

      int j = 1;
      for (; j + 3 < DBOT; j += 4) {
        const float4 r = curand_normal4(&state);
        if constexpr (MODE == 2) {
          const __half2 h12 = bot.load_pair(j);
          const float b1 = __low2float(h12);
          const float b2 = __high2float(h12);
          const float bj1 = fmaf(b1, coeff_at(sh_coeff, j + 0), sigma * r.x);
          const float bj2 = fmaf(b2, coeff_at(sh_coeff, j + 1), sigma * r.y);
          bot.store_pair(j, __float2half_rn(bj1), __float2half_rn(bj2));
          sum_bot2 = fmaf(bj1, bj1, sum_bot2);
          sum_bot2 = fmaf(bj2, bj2, sum_bot2);

          const __half2 h34 = bot.load_pair(j + 2);
          const float b3 = __low2float(h34);
          const float b4 = __high2float(h34);
          const float bj3 = fmaf(b3, coeff_at(sh_coeff, j + 2), sigma * r.z);
          const float bj4 = fmaf(b4, coeff_at(sh_coeff, j + 3), sigma * r.w);
          bot.store_pair(j + 2, __float2half_rn(bj3), __float2half_rn(bj4));
          sum_bot2 = fmaf(bj3, bj3, sum_bot2);
          sum_bot2 = fmaf(bj4, bj4, sum_bot2);
        } else {
          float b = __half2float(bot.load(j + 0));
          bj = fmaf(b, coeff_at(sh_coeff, j + 0), sigma * r.x);
          bot.store(j + 0, __float2half_rn(bj));
          sum_bot2 = fmaf(bj, bj, sum_bot2);
          b = __half2float(bot.load(j + 1));
          bj = fmaf(b, coeff_at(sh_coeff, j + 1), sigma * r.y);
          bot.store(j + 1, __float2half_rn(bj));
          sum_bot2 = fmaf(bj, bj, sum_bot2);
          b = __half2float(bot.load(j + 2));
          bj = fmaf(b, coeff_at(sh_coeff, j + 2), sigma * r.z);
          bot.store(j + 2, __float2half_rn(bj));
          sum_bot2 = fmaf(bj, bj, sum_bot2);
          b = __half2float(bot.load(j + 3));
          bj = fmaf(b, coeff_at(sh_coeff, j + 3), sigma * r.w);
          bot.store(j + 3, __float2half_rn(bj));
          sum_bot2 = fmaf(bj, bj, sum_bot2);
        }
      }
      if (j < DBOT) {
        const float2 r = curand_normal2(&state);
        float b = __half2float(bot.load(j));
        bj = fmaf(b, coeff_at(sh_coeff, j), sigma * r.x);
        bot.store(j, __float2half_rn(bj));
        sum_bot2 = fmaf(bj, bj, sum_bot2);
        ++j;
        if (j < DBOT) {
          b = __half2float(bot.load(j));
          bj = fmaf(b, coeff_at(sh_coeff, j), sigma * r.y);
          bot.store(j, __float2half_rn(bj));
          sum_bot2 = fmaf(bj, bj, sum_bot2);
          ++j;
        }
        if (j < DBOT) {
          b = __half2float(bot.load(j));
          bj = fmaf(b, coeff_at(sh_coeff, j), sigma * curand_normal(&state));
          bot.store(j, __float2half_rn(bj));
          sum_bot2 = fmaf(bj, bj, sum_bot2);
        }
      }

      const float hypotxy2 = fmaf(x1, x1, x2 * x2);
      const float r2_now = fmaf(x3, x3, hypotxy2);
      const float inv_norm_now = rsqrtf(r2_now);
      const float norm_now = r2_now * inv_norm_now;
      const float inv_hyp = rsqrtf(fmaxf(hypotxy2, 1e-20f));
      const float hypotxy = hypotxy2 * inv_hyp;
      bool ha, hb, hc, hd;
      hit_tests(x1, x2, x3, norm_now, inv_hyp, hypotxy, sum_bot2,
                delta_r, tan_el_lo, tan_el_hi, cos_dtheta, tan_dtheta,
                ha, hb, hc, hd);
      const bool hit[4] = {ha, hb, hc, hd};
      for (int q = 0; q < 4; ++q) {
        if (hit[q]) {
          if (it1 < first[q]) first[q] = it1;
          ++spent[q];
        }
      }
    }

    const float fst[4] = {
        first[0] <= n_t ? dt * first[0] : term_t,
        first[1] <= n_t ? dt * first[1] : term_t,
        first[2] <= n_t ? dt * first[2] : term_t,
        first[3] <= n_t ? dt * first[3] : term_t};
    const int first_any = min(min(first[0], first[1]), min(first[2], first[3]));
    const bool any = first_any <= n_t;
    for (int q = 0; q < 4; ++q) {
      totals[q] += dt * spent[q];
      totals[4 + q] += fst[q];
      totals[8 + q] += first[q] <= n_t ? 1.0f : 0.0f;
      totals[13 + q] += any && first[q] == first_any ? 1.0f : 0.0f;
    }
    totals[12] += any ? 1.0f : 0.0f;
  }

#if MFM_REUSE_SHARED_REDUCE
  // MODE=0 used the first shared-memory region for bottom state.  Every
  // thread has finished its trajectory before this barrier, so that region
  // can be reused for the final reduction.
  __syncthreads();
  reduce = reinterpret_cast<float*>(raw);
 #endif
  for (int q = 0; q < OUTPUTS; ++q) reduce[q * blockDim.x + tid] = totals[q];
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      for (int q = 0; q < OUTPUTS; ++q)
        reduce[q * blockDim.x + tid] += reduce[q * blockDim.x + tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0)
    for (int q = 0; q < OUTPUTS; ++q) output[q * n_curve + curve] = reduce[q * blockDim.x];
}

static void check(cudaError_t e, const char* where) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(e));
    std::exit(2);
  }
}

static const char* env_text(const char* name) {
  const char* value = std::getenv(name);
  return value && *value ? value : nullptr;
}

static int env_int(const char* name, int fallback) {
  const char* value = env_text(name);
  return value ? std::atoi(value) : fallback;
}

static float env_float(const char* name, float fallback) {
  const char* value = env_text(name);
  return value ? std::strtof(value, nullptr) : fallback;
}

static std::uint64_t env_u64(const char* name, std::uint64_t fallback) {
  const char* value = env_text(name);
  return value ? static_cast<std::uint64_t>(std::strtoull(value, nullptr, 10))
               : fallback;
}

static void read_f32_exact(const char* path, float* destination, size_t count) {
  std::FILE* fp = std::fopen(path, "rb");
  if (!fp) {
    std::perror(path);
    std::exit(2);
  }
  const size_t read = std::fread(destination, sizeof(float), count, fp);
  const int trailing = std::fgetc(fp);
  std::fclose(fp);
  if (read != count || trailing != EOF) {
    std::fprintf(stderr, "input size mismatch for %s\n", path);
    std::exit(2);
  }
}

struct SimulationConfig {
  float dt;
  float term_t;
  float c1tmp;
  float sigma;
  float delta_theta;
  float delta_r;
  float tan_el_lo;
  float tan_el_hi;
  float cos_dtheta;
  float tan_dtheta;
  std::uint64_t base_seed;
  float temperature;
  float c1;
  float c2;
  float c3;
  float c4;
  std::uint64_t curve_offset;
  std::uint64_t path_stride;
};

template <int MODE>
float run(float* d_top, float* d_bot, float* d_coeff, float* d_out,
          int n_curve, int n_sims, int n_t, int threads,
          const SimulationConfig& config) {
  const size_t reduce_bytes = sizeof(float) * OUTPUTS * threads;
  const size_t bot_bytes = sizeof(__half) * DBOT * threads;
#if MFM_REUSE_SHARED_REDUCE
  const size_t sh = MODE == 0 ? std::max(reduce_bytes, bot_bytes) : reduce_bytes;
#else
  const size_t sh = reduce_bytes + (MODE == 0 ? bot_bytes : 0);
#endif
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "event create");
  check(cudaEventCreate(&stop), "event create");
  check(cudaEventRecord(start), "event start");
  mfm_kernel<MODE><<<n_curve, threads, sh>>>(
      d_top, d_bot, d_coeff, n_curve, n_sims, n_t,
      config.dt, config.term_t, config.c1tmp, config.sigma,
      config.delta_theta, config.delta_r, config.tan_el_lo,
      config.tan_el_hi, config.cos_dtheta, config.tan_dtheta,
      config.base_seed, config.temperature, config.c1, config.c2,
      config.c3, config.c4, config.curve_offset, config.path_stride, d_out);
  check(cudaGetLastError(), "kernel launch");
  check(cudaEventRecord(stop), "event stop");
  check(cudaEventSynchronize(stop), "event sync");
  float ms = 0.0f;
  check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms;
}

static void report_parity(const std::vector<float>& reference,
                          const std::vector<float>& candidate,
                          const char* label, int threads) {
  float max_abs = 0.0f;
  int different = 0;
  for (size_t i = 0; i < reference.size(); ++i) {
    max_abs = std::max(max_abs, std::fabs(reference[i] - candidate[i]));
    different += std::memcmp(&reference[i], &candidate[i], sizeof(float)) != 0;
  }
  std::printf("threads %d %s parity different=%d/%zu max_abs=%.9g\n",
              threads, label, different, reference.size(), max_abs);
}

int main(int argc, char** argv) {
  const int n_curve = env_int("MFM_NCURVE", 128);
  const int n_sims = env_int("MFM_NSIMS", 32768);
  const int n_t = env_int("MFM_NT", 1000);
  if (n_curve <= 0 || n_sims <= 0 || n_t <= 0) {
    std::fprintf(stderr, "MFM_NCURVE, MFM_NSIMS, and MFM_NT must be positive\n");
    return 2;
  }
  std::vector<float> top(3 * n_curve), bot(DBOT * n_curve), coeff(DBOT);
  const char* top_path = env_text("MFM_TOP_F32");
  const char* bot_path = env_text("MFM_BOT_F32");
  const char* coeff_path = env_text("MFM_COEFF_F32");
  const bool custom_inputs = top_path || bot_path || coeff_path;
  if (custom_inputs) {
    if (!top_path || !bot_path || !coeff_path) {
      std::fprintf(stderr,
                   "MFM_TOP_F32, MFM_BOT_F32, and MFM_COEFF_F32 are required together\n");
      return 2;
    }
    read_f32_exact(top_path, top.data(), top.size());
    read_f32_exact(bot_path, bot.data(), bot.size());
    read_f32_exact(coeff_path, coeff.data(), coeff.size());
  } else {
    for (int c = 0; c < n_curve; ++c) {
      top[3*c+0] = 0.2f; top[3*c+1] = 0.3f; top[3*c+2] = 0.9f;
      for (int j = 0; j < DBOT; ++j) bot[DBOT*c+j] = 0.05f * (j + 1);
    }
    for (int j = 0; j < DBOT; ++j) coeff[j] = 0.97f - 0.001f * j;
  }
  const float dt = env_float("MFM_DT", 0.002f);
  const float tangential_strength = env_float("MFM_TANGENTIAL_STRENGTH", 1.0f);
  const SimulationConfig config{
      dt,
      env_float("MFM_TERM_T", static_cast<float>(n_t) * dt),
      env_float("MFM_C1TMP", 1.0f),
      env_float("MFM_SIGMA", 0.4f),
      env_float("MFM_DELTA_THETA", 0.2f),
      env_float("MFM_DELTA_R", 0.8f),
      env_float("MFM_TAN_EL_LO", -0.5f),
      env_float("MFM_TAN_EL_HI", 0.0f),
      env_float("MFM_COS_DTHETA", 0.8660254f),
      env_float("MFM_TAN_DTHETA", 0.202710f),
      env_u64("MFM_BASE_SEED", 1234567ULL),
      env_float("MFM_TEMPERATURE", 10.0f),
      tangential_strength * env_float("MFM_C1", 0.7f),
      tangential_strength * env_float("MFM_C2", 0.9f),
      tangential_strength * env_float("MFM_C3", 1.1f),
      tangential_strength * env_float("MFM_C4", 1.3f),
      env_u64("MFM_CURVE_OFFSET", 0ULL),
      env_u64("MFM_PATH_STRIDE", static_cast<std::uint64_t>(n_sims))};
  float *d_top = nullptr, *d_bot = nullptr, *d_coeff = nullptr, *d_out = nullptr;
  check(cudaMalloc(&d_top, top.size() * sizeof(float)), "malloc top");
  check(cudaMalloc(&d_bot, bot.size() * sizeof(float)), "malloc bot");
  check(cudaMalloc(&d_coeff, coeff.size() * sizeof(float)), "malloc coeff");
  check(cudaMalloc(&d_out, OUTPUTS * n_curve * sizeof(float)), "malloc output");
  check(cudaMemcpy(d_top, top.data(), top.size()*sizeof(float), cudaMemcpyHostToDevice), "copy top");
  check(cudaMemcpy(d_bot, bot.data(), bot.size()*sizeof(float), cudaMemcpyHostToDevice), "copy bot");
  check(cudaMemcpy(d_coeff, coeff.data(), coeff.size()*sizeof(float), cudaMemcpyHostToDevice), "copy coeff");
#if MFM_CONST_COEFF
  check(cudaMemcpyToSymbol(c_coeff, coeff.data(), coeff.size() * sizeof(float)),
        "copy constant coeff");
#endif

  // A single-mode path makes repeated timing and Nsight Compute runs isolate
  // one storage implementation instead of mixing kernels in one process.
  if (argc >= 2) {
    const int mode = std::atoi(argv[1]);
    const int threads = argc >= 3 ? std::atoi(argv[2]) : 256;
    if (threads <= 0 || threads > 1024 || (threads & (threads - 1)) != 0) {
      std::fprintf(stderr, "threads must be a power of two in [1, 1024]\n");
      return 2;
    }
    std::vector<float> out(OUTPUTS * n_curve);
    auto launch_once = [&]() -> float {
      switch (mode) {
        case 0: return run<0>(d_top, d_bot, d_coeff, d_out, n_curve, n_sims, n_t, threads, config);
        case 1: return run<1>(d_top, d_bot, d_coeff, d_out, n_curve, n_sims, n_t, threads, config);
        case 2: return run<2>(d_top, d_bot, d_coeff, d_out, n_curve, n_sims, n_t, threads, config);
        default:
          std::fprintf(stderr, "mode must be 0 (shared), 1 (local), or 2 (packed)\n");
          std::exit(2);
      }
    };

    // Production uses one deterministic launch.  The old path below is kept
    // for explicit microbenchmarks; it must never be used by the dataset
    // runner because it repeats the full Monte Carlo calculation seven times.
    const bool write_once = argc >= 4 &&
        (std::strcmp(argv[3], "write") == 0 ||
         std::strcmp(argv[3], "once") == 0);
    if (write_once) {
      const float elapsed_ms = launch_once();
      std::printf("once mode %d threads %d elapsed_ms %.3f\n",
                  mode, threads, elapsed_ms);
      check(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "copy single-mode output");
      if (argc >= 5) {
        std::FILE* fp = std::fopen(argv[4], "wb");
        if (!fp) { std::perror("fopen output"); return 2; }
        const size_t nw = std::fwrite(out.data(), sizeof(float), out.size(), fp);
        std::fclose(fp);
        if (nw != out.size()) { std::fprintf(stderr, "short output write\n"); return 2; }
      }
      return 0;
    }

    const float warmup = launch_once();
    (void)warmup;
    std::vector<float> samples;
    samples.reserve(7);
    for (int rep = 0; rep < 7; ++rep) {
      samples.push_back(launch_once());
    }
    std::sort(samples.begin(), samples.end());
    std::printf("single mode %d threads %d min_ms %.3f median_ms %.3f max_ms %.3f\n",
                mode, threads, samples.front(), samples[samples.size() / 2], samples.back());
    check(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "copy single-mode output");
    if (argc >= 4 && std::strcmp(argv[3], "write") == 0 && argc >= 5) {
      std::FILE* fp = std::fopen(argv[4], "wb");
      if (!fp) { std::perror("fopen output"); return 2; }
      const size_t nw = std::fwrite(out.data(), sizeof(float), out.size(), fp);
      std::fclose(fp);
      if (nw != out.size()) { std::fprintf(stderr, "short output write\n"); return 2; }
    } else if (argc >= 4) {
      std::vector<float> ref(out.size());
      std::FILE* fp = std::fopen(argv[3], "rb");
      if (!fp) { std::perror("fopen reference"); return 2; }
      const size_t nr = std::fread(ref.data(), sizeof(float), ref.size(), fp);
      std::fclose(fp);
      if (nr != ref.size()) { std::fprintf(stderr, "short reference read\n"); return 2; }
      float max_abs = 0.0f;
      int different = 0;
      for (size_t i = 0; i < out.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(out[i] - ref[i]));
        different += std::memcmp(&out[i], &ref[i], sizeof(float)) != 0;
      }
      std::printf("reference parity different=%d/%zu max_abs=%.9g\n",
                  different, out.size(), max_abs);
    }
    cudaFree(d_top); cudaFree(d_bot); cudaFree(d_coeff); cudaFree(d_out);
    return 0;
  }

  for (int threads : {128, 256}) {
    std::vector<float> baseline(OUTPUTS*n_curve), local(OUTPUTS*n_curve), packed(OUTPUTS*n_curve);
    run<0>(d_top, d_bot, d_coeff, d_out, n_curve, n_sims, n_t, threads, config);
    check(cudaMemcpy(baseline.data(), d_out, baseline.size()*sizeof(float), cudaMemcpyDeviceToHost), "copy baseline");
    run<1>(d_top, d_bot, d_coeff, d_out, n_curve, n_sims, n_t, threads, config);
    check(cudaMemcpy(local.data(), d_out, local.size()*sizeof(float), cudaMemcpyDeviceToHost), "copy local");
    run<2>(d_top, d_bot, d_coeff, d_out, n_curve, n_sims, n_t, threads, config);
    check(cudaMemcpy(packed.data(), d_out, packed.size()*sizeof(float), cudaMemcpyDeviceToHost), "copy packed");
    report_parity(baseline, local, "local_half", threads);
    report_parity(baseline, packed, "packed_half2", threads);
    for (int rep = 0; rep < 3; ++rep) {
      float a = run<0>(d_top, d_bot, d_coeff, d_out, n_curve, n_sims, n_t, threads, config);
      float b = run<1>(d_top, d_bot, d_coeff, d_out, n_curve, n_sims, n_t, threads, config);
      float c = run<2>(d_top, d_bot, d_coeff, d_out, n_curve, n_sims, n_t, threads, config);
      std::printf("threads %d rep %d shared_half_ms %.3f local_half_ms %.3f packed_half2_ms %.3f local_speedup %.5f packed_speedup %.5f\n",
                  threads, rep, a, b, c, a / b, a / c);
    }
  }
  cudaFree(d_top); cudaFree(d_bot); cudaFree(d_coeff); cudaFree(d_out);
  return 0;
}
