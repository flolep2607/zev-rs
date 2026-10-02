// Dev-only: time gdn_bf16 (src/kev/kernels.cu, as zev launches it) against candidate rewrites on Kev-4B's shapes
// (HK 16, HV 32, DK = DV = 128), and check each candidate's output against it.
// nvcc -O3 -std=c++17 -arch=sm_89 csrc/gdn_bench.cu -o /work/gdnbench
#include "../src/kev/kernels.cu"
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// ---- candidate v2: per-token q/k norms and gates computed once (gdn_prep), FMA chains split four ways ----------------

// One warp per (token, key head): qn = q / |q| / sqrt(DK), kn = k / |k|; one thread per (token, value head): decay, beta.
extern "C" __global__ void gdn_prep(const bf16* qkv, const bf16* proj, int ld, int a_off, int b_off, const float* a_neg,
                                    const float* dt_bias, float* qn, float* kn, float* gate, long T, int HK, int HV) {
    const int C = 2 * HK * DK + HV * DV;
    const long w = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (w < T * HK) {
        const long tok = w / HK;
        const int kh = w % HK;
        const bf16* row = qkv + tok * C;
        float q[4], k[4], sq = 0.f, sk = 0.f;
        for (int r = 0; r < 4; r++) {
            q[r] = bf2f(row[kh * DK + lane + 32 * r]);
            k[r] = bf2f(row[HK * DK + kh * DK + lane + 32 * r]);
            sq += q[r] * q[r];
            sk += k[r] * k[r];
        }
        const float iq = rsqrtf(warp_sum(sq) + 1e-6f) * rsqrtf((float)DK), ik = rsqrtf(warp_sum(sk) + 1e-6f);
        for (int r = 0; r < 4; r++) {
            qn[w * DK + lane + 32 * r] = q[r] * iq;
            kn[w * DK + lane + 32 * r] = k[r] * ik;
        }
    }
    const long g = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g < T * HV) {
        const long tok = g / HV;
        const int h = g % HV;
        const float x = bf2f(proj[tok * ld + a_off + h]) + dt_bias[h];
        const float sp = fmaxf(x, 0.f) + logf(1.f + expf(-fabsf(x)));
        gate[2 * g] = expf(sp * a_neg[h]);
        gate[2 * g + 1] = 1.f / (1.f + expf(-bf2f(proj[tok * ld + b_off + h])));
    }
}

// Same warp layout as gdn_body (lane = column lane/4 x key slice lane%4, 32 state entries in registers), reading the
// prepared values. No state cache (the bench has none).
extern "C" __global__ void gdn_v2(const bf16* qkv, const float* qn, const float* kn, const float* gate, float* out,
                                  const unsigned* cu, const unsigned* list, int HK, int HV) {
    const int tiles = DV / 8;
    const int h = blockIdx.x / tiles, tile = blockIdx.x % tiles, lane = threadIdx.x;
    const unsigned b = list[blockIdx.y];
    const int col = lane >> 2, ks = lane & 3, j = tile * 8 + col, kh = h / (HV / HK);
    const int C = 2 * HK * DK + HV * DV;
    const long start = cu[b];
    const int len = cu[b + 1] - start;
    __shared__ float qs[DK], kv[DK];
    float S[32];
#pragma unroll
    for (int r = 0; r < 32; r++) S[r] = 0.f;
    for (int t = 0; t < len; t++) {
        const long tok = start + t;
        const float* qr = qn + (tok * HK + kh) * DK;
        const float* kr = kn + (tok * HK + kh) * DK;
        float q4[4], k4[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            q4[r] = qr[lane + 32 * r];
            k4[r] = kr[lane + 32 * r];
        }
        const float v = bf2f(qkv[tok * C + 2 * HK * DK + h * DV + j]);
        const float decay = gate[2 * (tok * HV + h)], beta = gate[2 * (tok * HV + h) + 1];
        __syncwarp();
#pragma unroll
        for (int r = 0; r < 4; r++) {
            qs[lane + 32 * r] = q4[r];
            kv[lane + 32 * r] = k4[r];
        }
        __syncwarp();
        float m4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int r = 0; r < 32; r++) {
            S[r] *= decay;
            m4[r & 3] += S[r] * kv[4 * r + ks];
        }
        float mem = (m4[0] + m4[1]) + (m4[2] + m4[3]);
        mem += __shfl_xor_sync(FULL, mem, 1);
        mem += __shfl_xor_sync(FULL, mem, 2);
        const float delta = (v - mem) * beta;
        float a4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int r = 0; r < 32; r++) {
            S[r] += kv[4 * r + ks] * delta;
            a4[r & 3] += S[r] * qs[4 * r + ks];
        }
        float acc = (a4[0] + a4[1]) + (a4[2] + a4[3]);
        acc += __shfl_xor_sync(FULL, acc, 1);
        acc += __shfl_xor_sync(FULL, acc, 2);
        if (ks == 0) out[(tok * HV + h) * DV + j] = acc;
    }
}


// ---- candidate v3: 4 warps per block share one cp.async-staged copy of q, k, v and the gates, 8 tokens at a time ---
#define V3_W 4
#define V3_CH 8
__device__ __forceinline__ void cpa16(void* dst, const void* src) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void cpa8(void* dst, const void* src) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

// grid (HV * (DV / 8 / V3_W), listed sequences); block V3_W warps; warp w owns value columns [8 * tile, 8 * tile + 8)
// of head h, tile = (blockIdx.x % (DV / 8 / V3_W)) * V3_W + w. rd / wr: per-sequence state in / out (0 = none).
extern "C" __global__ void __launch_bounds__(32 * V3_W) gdn_v3(const bf16* qkv, const float* qn, const float* kn, const float* gate,
        float* out, const unsigned* cu, const unsigned* list, const unsigned long long* rd, const unsigned long long* wr, int HK,
        int HV, int lg) {
    constexpr int TPH = DV / 8 / V3_W;  // blocks per head
    const int h = blockIdx.x / TPH, w = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
    const int tile = (blockIdx.x % TPH) * V3_W + w, col = lane >> 2, ks = lane & 3, j = tile * 8 + col, kh = h / (HV / HK);
    const int jb = (blockIdx.x % TPH) * V3_W * 8;  // the block's first value column
    const int C = 2 * HK * DK + HV * DV;
    const unsigned b = list[blockIdx.y];
    const long start = cu[b];
    const int len = cu[b + 1] - start;
    __shared__ __align__(16) float qk[2][V3_CH][2 * DK];   // q then k, normalized
    __shared__ __align__(16) bf16 vv[2][V3_CH][V3_W * 8];  // the block's value columns
    __shared__ __align__(16) float gg[2][V3_CH][2];        // decay, beta
    float S[32];
    const long soff = ((long)lg * HV + h) * DK * DV + j;
    const float* s0 = rd[b] ? (const float*)rd[b] + soff : nullptr;
#pragma unroll
    for (int r = 0; r < 32; r++) S[r] = s0 ? s0[(long)(4 * r + ks) * DV] : 0.f;
    auto stage = [&](int c, int buf) {
        const int t0 = c * V3_CH, n = min(V3_CH, len - t0);
        // q and k: n tokens x 2 x DK floats = n x 64 chunks of 16 bytes
        for (int e = tid; e < n * 64; e += 32 * V3_W) {
            const int i = e >> 6, part = e & 63, which = part >> 5, c4 = part & 31;
            const long tok = start + t0 + i;
            const float* src = (which ? kn : qn) + (tok * HK + kh) * DK + 4 * c4;
            cpa16(&qk[buf][i][which * DK + 4 * c4], src);
        }
        // v: n tokens x 32 bf16 = n x 4 chunks; gates: n x 8 bytes
        for (int e = tid; e < n * 4; e += 32 * V3_W) {
            const int i = e >> 2, c8 = e & 3;
            cpa16(&vv[buf][i][8 * c8], qkv + (start + t0 + i) * C + 2 * HK * DK + h * DV + jb + 8 * c8);
        }
        for (int i = tid; i < n; i += 32 * V3_W) cpa8(&gg[buf][i][0], gate + 2 * ((start + t0 + i) * HV + h));
        cpa_commit();
    };
    const int chunks = (len + V3_CH - 1) / V3_CH;
    if (chunks > 0) stage(0, 0);
    for (int c = 0; c < chunks; c++) {
        const int buf = c & 1;
        if (c + 1 < chunks) {
            stage(c + 1, buf ^ 1);
            cpa_wait<1>();
        } else {
            cpa_wait<0>();
        }
        __syncthreads();
        const int n = min(V3_CH, len - c * V3_CH);
        for (int i = 0; i < n; i++) {
            const float* kv = &qk[buf][i][DK];
            const float* qs = &qk[buf][i][0];
            const float decay = gg[buf][i][0], beta = gg[buf][i][1];
            const float v = bf2f(vv[buf][i][w * 8 + col]);
            float m4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int r = 0; r < 32; r++) {
                S[r] *= decay;
                m4[r & 3] += S[r] * kv[4 * r + ks];
            }
            float mem = (m4[0] + m4[1]) + (m4[2] + m4[3]);
            mem += __shfl_xor_sync(FULL, mem, 1);
            mem += __shfl_xor_sync(FULL, mem, 2);
            const float delta = (v - mem) * beta;
            float a4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int r = 0; r < 32; r++) {
                S[r] += kv[4 * r + ks] * delta;
                a4[r & 3] += S[r] * qs[4 * r + ks];
            }
            float acc = (a4[0] + a4[1]) + (a4[2] + a4[3]);
            acc += __shfl_xor_sync(FULL, acc, 1);
            acc += __shfl_xor_sync(FULL, acc, 2);
            if (ks == 0) out[((start + c * V3_CH + i) * HV + h) * DV + j] = acc;
        }
        __syncthreads();  // this buffer is restaged two chunks on
    }
    if (wr[b]) {
        float* st = (float*)wr[b] + soff;
#pragma unroll
        for (int r = 0; r < 32; r++) st[(long)(4 * r + ks) * DV] = S[r];
    }
}

// ---- candidate v4: v3 with contiguous key slices (lane ks owns keys 32 ks .. 32 ks + 31), read as float4 from a
// padded shared layout (slice stride 36 floats: the four slices start on different banks) ------------------------------
#define V4_SL 36
template <int CH>
__global__ void __launch_bounds__(32 * V3_W) gdn_v4(const bf16* qkv, const float* qn, const float* kn, const float* gate,
        float* out, const unsigned* cu, const unsigned* list, const unsigned long long* rd, const unsigned long long* wr, int HK,
        int HV, int lg) {
    constexpr int TPH = DV / 8 / V3_W;
    const int h = blockIdx.x / TPH, w = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
    const int tile = (blockIdx.x % TPH) * V3_W + w, col = lane >> 2, ks = lane & 3, j = tile * 8 + col, kh = h / (HV / HK);
    const int jb = (blockIdx.x % TPH) * V3_W * 8;
    const int C = 2 * HK * DK + HV * DV;
    const unsigned b = list[blockIdx.y];
    const long start = cu[b];
    const int len = cu[b + 1] - start;
    __shared__ __align__(16) float qk[2][CH][2][4 * V4_SL];
    __shared__ __align__(16) bf16 vv[2][CH][V3_W * 8];
    __shared__ __align__(16) float gg[2][CH][2];
    float S[32];
    const long soff = ((long)lg * HV + h) * DK * DV + j;
    const float* s0 = rd[b] ? (const float*)rd[b] + soff : nullptr;
#pragma unroll
    for (int r = 0; r < 32; r++) S[r] = s0 ? s0[(long)(32 * ks + r) * DV] : 0.f;
    auto stage = [&](int c, int buf) {
        const int t0 = c * CH, n = min(CH, len - t0);
        for (int e = tid; e < n * 64; e += 32 * V3_W) {
            const int i = e >> 6, part = e & 63, which = part >> 5, c4 = part & 31;
            const long tok = start + t0 + i;
            cpa16(&qk[buf][i][which][(c4 >> 3) * V4_SL + (c4 & 7) * 4], (which ? kn : qn) + (tok * HK + kh) * DK + 4 * c4);
        }
        for (int e = tid; e < n * 4; e += 32 * V3_W) {
            const int i = e >> 2, c8 = e & 3;
            cpa16(&vv[buf][i][8 * c8], qkv + (start + t0 + i) * C + 2 * HK * DK + h * DV + jb + 8 * c8);
        }
        for (int i = tid; i < n; i += 32 * V3_W) cpa8(&gg[buf][i][0], gate + 2 * ((start + t0 + i) * HV + h));
        cpa_commit();
    };
    const int chunks = (len + CH - 1) / CH;
    if (chunks > 0) stage(0, 0);
    for (int c = 0; c < chunks; c++) {
        const int buf = c & 1;
        if (c + 1 < chunks) {
            stage(c + 1, buf ^ 1);
            cpa_wait<1>();
        } else {
            cpa_wait<0>();
        }
        __syncthreads();
        const int n = min(CH, len - c * CH);
        for (int i = 0; i < n; i++) {
            const float4* k4 = reinterpret_cast<const float4*>(&qk[buf][i][1][ks * V4_SL]);
            const float4* q4 = reinterpret_cast<const float4*>(&qk[buf][i][0][ks * V4_SL]);
            const float decay = gg[buf][i][0], beta = gg[buf][i][1];
            const float v = bf2f(vv[buf][i][w * 8 + col]);
            float m4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int r = 0; r < 8; r++) {
                const float4 kk = k4[r];
                S[4 * r] *= decay; S[4 * r + 1] *= decay; S[4 * r + 2] *= decay; S[4 * r + 3] *= decay;
                m4[0] += S[4 * r] * kk.x; m4[1] += S[4 * r + 1] * kk.y; m4[2] += S[4 * r + 2] * kk.z; m4[3] += S[4 * r + 3] * kk.w;
            }
            float mem = (m4[0] + m4[1]) + (m4[2] + m4[3]);
            mem += __shfl_xor_sync(FULL, mem, 1);
            mem += __shfl_xor_sync(FULL, mem, 2);
            const float delta = (v - mem) * beta;
            float a4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int r = 0; r < 8; r++) {
                const float4 kk = k4[r], qq = q4[r];
                S[4 * r] += kk.x * delta; S[4 * r + 1] += kk.y * delta; S[4 * r + 2] += kk.z * delta; S[4 * r + 3] += kk.w * delta;
                a4[0] += S[4 * r] * qq.x; a4[1] += S[4 * r + 1] * qq.y; a4[2] += S[4 * r + 2] * qq.z; a4[3] += S[4 * r + 3] * qq.w;
            }
            float acc = (a4[0] + a4[1]) + (a4[2] + a4[3]);
            acc += __shfl_xor_sync(FULL, acc, 1);
            acc += __shfl_xor_sync(FULL, acc, 2);
            if (ks == 0) out[((start + c * CH + i) * HV + h) * DV + j] = acc;
        }
        __syncthreads();
    }
    if (wr[b]) {
        float* st = (float*)wr[b] + soff;
#pragma unroll
        for (int r = 0; r < 32; r++) st[(long)(32 * ks + r) * DV] = S[r];
    }
}

// ---- candidate v5: v4 with W warps per block and lazy decay (S = D * St, one scalar D per head) ---- // was v4: v3 with contiguous key slices (lane ks owns keys 32 ks .. 32 ks + 31), read as float4 from a
// padded shared layout (slice stride 36 floats: the four slices start on different banks) ------------------------------
template <int CH, int W>
__global__ void __launch_bounds__(32 * W) gdn_v5(const bf16* qkv, const float* qn, const float* kn, const float* gate,
        float* out, const unsigned* cu, const unsigned* list, const unsigned long long* rd, const unsigned long long* wr, int HK,
        int HV, int lg) {
    constexpr int TPH = DV / 8 / W;
    const int h = blockIdx.x / TPH, w = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
    const int tile = (blockIdx.x % TPH) * W + w, col = lane >> 2, ks = lane & 3, j = tile * 8 + col, kh = h / (HV / HK);
    const int jb = (blockIdx.x % TPH) * W * 8;
    const int C = 2 * HK * DK + HV * DV;
    const unsigned b = list[blockIdx.y];
    const long start = cu[b];
    const int len = cu[b + 1] - start;
    __shared__ __align__(16) float qk[2][CH][2][4 * V4_SL];
    __shared__ __align__(16) bf16 vv[2][CH][W * 8];
    __shared__ __align__(16) float gg[2][CH][2];
    float S[32], D = 1.f;  // the true state is D * S
    const long soff = ((long)lg * HV + h) * DK * DV + j;
    const float* s0 = rd[b] ? (const float*)rd[b] + soff : nullptr;
#pragma unroll
    for (int r = 0; r < 32; r++) S[r] = s0 ? s0[(long)(32 * ks + r) * DV] : 0.f;
    auto stage = [&](int c, int buf) {
        const int t0 = c * CH, n = min(CH, len - t0);
        for (int e = tid; e < n * 64; e += 32 * W) {
            const int i = e >> 6, part = e & 63, which = part >> 5, c4 = part & 31;
            const long tok = start + t0 + i;
            cpa16(&qk[buf][i][which][(c4 >> 3) * V4_SL + (c4 & 7) * 4], (which ? kn : qn) + (tok * HK + kh) * DK + 4 * c4);
        }
        for (int e = tid; e < n * 4; e += 32 * W) {
            const int i = e >> 2, c8 = e & 3;
            cpa16(&vv[buf][i][8 * c8], qkv + (start + t0 + i) * C + 2 * HK * DK + h * DV + jb + 8 * c8);
        }
        for (int i = tid; i < n; i += 32 * W) cpa8(&gg[buf][i][0], gate + 2 * ((start + t0 + i) * HV + h));
        cpa_commit();
    };
    const int chunks = (len + CH - 1) / CH;
    if (chunks > 0) stage(0, 0);
    for (int c = 0; c < chunks; c++) {
        const int buf = c & 1;
        if (c + 1 < chunks) {
            stage(c + 1, buf ^ 1);
            cpa_wait<1>();
        } else {
            cpa_wait<0>();
        }
        __syncthreads();
        const int n = min(CH, len - c * CH);
        for (int i = 0; i < n; i++) {
            const float4* k4 = reinterpret_cast<const float4*>(&qk[buf][i][1][ks * V4_SL]);
            const float4* q4 = reinterpret_cast<const float4*>(&qk[buf][i][0][ks * V4_SL]);
            const float decay = gg[buf][i][0], beta = gg[buf][i][1];
            const float v = bf2f(vv[buf][i][w * 8 + col]);
            D *= decay;
            if (D < 1e-18f) {  // fold D back in before 1 / D can overflow; warp-uniform (one decay per head and token)
#pragma unroll
                for (int r = 0; r < 32; r++) S[r] *= D;
                D = 1.f;
            }
            float m4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int r = 0; r < 8; r++) {
                const float4 kk = k4[r];
                m4[0] += S[4 * r] * kk.x; m4[1] += S[4 * r + 1] * kk.y; m4[2] += S[4 * r + 2] * kk.z; m4[3] += S[4 * r + 3] * kk.w;
            }
            float mem = (m4[0] + m4[1]) + (m4[2] + m4[3]);
            mem += __shfl_xor_sync(FULL, mem, 1);
            mem += __shfl_xor_sync(FULL, mem, 2);
            mem *= D;
            const float delta = (v - mem) * beta / D;
            float a4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int r = 0; r < 8; r++) {
                const float4 kk = k4[r], qq = q4[r];
                S[4 * r] += kk.x * delta; S[4 * r + 1] += kk.y * delta; S[4 * r + 2] += kk.z * delta; S[4 * r + 3] += kk.w * delta;
                a4[0] += S[4 * r] * qq.x; a4[1] += S[4 * r + 1] * qq.y; a4[2] += S[4 * r + 2] * qq.z; a4[3] += S[4 * r + 3] * qq.w;
            }
            float acc = (a4[0] + a4[1]) + (a4[2] + a4[3]);
            acc += __shfl_xor_sync(FULL, acc, 1);
            acc += __shfl_xor_sync(FULL, acc, 2);
            if (ks == 0) out[((start + c * CH + i) * HV + h) * DV + j] = acc * D;
        }
        __syncthreads();
    }
    if (wr[b]) {
        float* st = (float*)wr[b] + soff;
#pragma unroll
        for (int r = 0; r < 32; r++) st[(long)(32 * ks + r) * DV] = S[r] * D;
    }
}

static unsigned short f2bf_host(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}

template <class F>
float timeit(F f, int reps = 5) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    f();
    cudaEventRecord(a);
    for (int i = 0; i < reps; i++) f();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    return ms / reps;
}

int main(int argc, char** argv) {
    const int HK = 16, HV = 32, NSEQ = argc > 1 ? atoi(argv[1]) : 20, LEN = argc > 2 ? atoi(argv[2]) : 1550;
    const int C = 2 * HK * DK + HV * DV, LD = C + HV * DV + 2 * HV;
    const long T = (long)NSEQ * LEN;
    std::vector<unsigned short> hq((size_t)T * C), hp((size_t)T * LD);
    unsigned s = 12345;
    auto rnd = [&] { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xffff) / 65536.f - 0.5f; };
    for (auto& x : hq) x = f2bf_host(rnd());
    for (auto& x : hp) x = f2bf_host(2.f * rnd());
    std::vector<float> han(HV), hdb(HV);
    for (int h = 0; h < HV; h++) { han[h] = -0.5f - 0.05f * h; hdb[h] = 0.1f * (h % 5); }
    std::vector<unsigned> hcu(NSEQ + 1), hlist(NSEQ);
    for (int i = 0; i <= NSEQ; i++) hcu[i] = i * LEN;
    for (int i = 0; i < NSEQ; i++) hlist[i] = i;
    bf16 *qkv, *proj; float *an, *db, *out, *out2, *qn, *kn, *gate; unsigned *cu, *list; unsigned long long* nulls;
    cudaMalloc(&qkv, hq.size() * 2); cudaMalloc(&proj, hp.size() * 2); cudaMalloc(&an, HV * 4); cudaMalloc(&db, HV * 4);
    cudaMalloc(&out, T * HV * DV * 4); cudaMalloc(&out2, T * HV * DV * 4); cudaMalloc(&cu, (NSEQ + 1) * 4); cudaMalloc(&list, NSEQ * 4);
    cudaMalloc(&nulls, NSEQ * 8); cudaMemset(nulls, 0, NSEQ * 8);
    cudaMalloc(&qn, T * HK * DK * 4); cudaMalloc(&kn, T * HK * DK * 4); cudaMalloc(&gate, T * HV * 8);
    cudaMemcpy(qkv, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(proj, hp.data(), hp.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(an, han.data(), HV * 4, cudaMemcpyHostToDevice); cudaMemcpy(db, hdb.data(), HV * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(cu, hcu.data(), hcu.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(list, hlist.data(), hlist.size() * 4, cudaMemcpyHostToDevice);
    const int a_off = C + HV * DV + HV, b_off = C + HV * DV;
    const dim3 grid(HV * DV / 8, NSEQ);
    float base = timeit([&] { gdn_bf16<<<grid, 32>>>(qkv, proj, LD, a_off, b_off, an, db, out, cu, list, nulls, nulls, HK, HV, 0); });
    printf("gdn_bf16 (current): %.3f ms for %d x %d tokens  (%.2f us per token step)\n", base, NSEQ, LEN, base * 1e3 / LEN);
    const long prep_threads = T * HK * 32 > T * HV ? T * HK * 32 : T * HV;
    float v2 = timeit([&] {
        gdn_prep<<<(prep_threads + 255) / 256, 256>>>(qkv, proj, LD, a_off, b_off, an, db, qn, kn, gate, T, HK, HV);
        gdn_v2<<<grid, 32>>>(qkv, qn, kn, gate, out2, cu, list, HK, HV);
    });
    float v2main = timeit([&] { gdn_v2<<<grid, 32>>>(qkv, qn, kn, gate, out2, cu, list, HK, HV); });
    std::vector<float> o1(T * HV * DV), o2(T * HV * DV);
    cudaMemcpy(o1.data(), out, o1.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(o2.data(), out2, o2.size() * 4, cudaMemcpyDeviceToHost);
    double md = 0, mref = 0;
    for (size_t i = 0; i < o1.size(); i++) { md = fmax(md, fabs(o1[i] - o2[i])); mref = fmax(mref, fabs(o1[i])); }
    printf("gdn_v2 (prep + main): %.3f ms (main %.3f): %.2fx; max |diff| %.3g of max |out| %.3g\n", v2, v2main, base / v2, md, mref);
    float *out3;
    cudaMalloc(&out3, T * HV * DV * 4);
    const dim3 grid3(HV * (DV / 8 / V3_W), NSEQ);
    float v3main = timeit([&] { gdn_v3<<<grid3, 32 * V3_W>>>(qkv, qn, kn, gate, out3, cu, list, nulls, nulls, HK, HV, 0); });
    std::vector<float> o3(T * HV * DV);
    cudaMemcpy(o3.data(), out3, o3.size() * 4, cudaMemcpyDeviceToHost);
    md = 0;
    for (size_t i = 0; i < o1.size(); i++) md = fmax(md, fabs(o1[i] - o3[i]));
    printf("gdn_v3 (prep + main): %.3f ms (main %.3f): %.2fx; max |diff| %.3g\n", v2 - v2main + v3main, v3main, base / (v2 - v2main + v3main), md);
    auto run4 = [&](auto kern, const char* name) {
        cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        float ms = timeit([&] { kern<<<grid3, 32 * V3_W>>>(qkv, qn, kn, gate, out3, cu, list, nulls, nulls, HK, HV, 0); });
        cudaMemcpy(o3.data(), out3, o3.size() * 4, cudaMemcpyDeviceToHost);
        double d = 0;
        for (size_t i = 0; i < o1.size(); i++) d = fmax(d, fabs(o1[i] - o3[i]));
        int nb = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kern, 32 * V3_W, 0);
        printf("%s (prep + main): %.3f ms (main %.3f): %.2fx; max |diff| %.3g; %d blocks/SM\n", name, v2 - v2main + ms, ms, base / (v2 - v2main + ms), d, nb);
    };
    run4(gdn_v4<6>, "gdn_v4<6>");
    auto run5 = [&](auto kern, int W, const char* name) {
        cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        const dim3 g5(HV * (DV / 8 / W), NSEQ);
        float ms = timeit([&] { kern<<<g5, 32 * W>>>(qkv, qn, kn, gate, out3, cu, list, nulls, nulls, HK, HV, 0); });
        cudaMemcpy(o3.data(), out3, o3.size() * 4, cudaMemcpyDeviceToHost);
        double d = 0, rel = 0;
        for (size_t i = 0; i < o1.size(); i++) { d = fmax(d, fabs(o1[i] - o3[i])); }
        int nb = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kern, 32 * W, 0);
        printf("%s (prep + main): %.3f ms (main %.3f): %.2fx; max |diff| %.3g; %d blocks/SM\n", name, v2 - v2main + ms, ms, base / (v2 - v2main + ms), d, nb);
    };
    run5(gdn_v5<6, 4>, 4, "gdn_v5<6,4>");
    run5(gdn_v5<6, 8>, 8, "gdn_v5<6,8>");
    run5(gdn_v5<4, 16>, 16, "gdn_v5<4,16>");
    run5(gdn_v5<8, 8>, 8, "gdn_v5<8,8>");
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}
