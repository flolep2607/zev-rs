// Dev-only: chunked gated delta rule on tensor cores (gdn_c1 + gdn_c2) against gdn_fast_bf16 (src/kev/kernels.cu)
// on Kev-4B's shapes. nvcc -O3 -std=c++17 -arch=sm_89 csrc/gdnc_bench.cu -o /work/gdncbench
#include "../src/kev/kernels.cu"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

#ifndef GCF_MINB
#define GCF_MINB 1
#endif
#define CC 32    // chunk
#define CLD 136  // bf16 row stride, 128-wide tiles
#define CLP 40   // bf16 row stride, 32-wide tiles
#define CLU 72   // bf16 row stride, 64-wide tiles

// G1: one block of 4 warps per (32-token chunk, key head), covering the key head's HV / HK value heads: Q K^T and
// K K^T are computed once for them. Writes per (token, value head) row: W = T (beta gamma K), U = T (beta V), Q gamma,
// K gamma_C / gamma (bf16 [T, HV, 128]), P = (Q K^T) * exp(g_r - g_s) for s <= r (bf16 [T, HV, CC]), g (cumulative
// log decay in the chunk, f32 [T, HV]). T = (I + A)^-1 with A_rs = beta_r (k_r . k_s) exp(g_r - g_s), s < r.
extern "C" __global__ void __launch_bounds__(128) gdn_c1(const float* qn, const float* kn, const bf16* vin, int vld, const float* gate,
                                                        const unsigned* tiles, const unsigned* cu, bf16* Wout, bf16* Uout, bf16* Pout,
                                                        float* gout, bf16* Qgout, bf16* Kdout, int HK, int HV) {
    __shared__ __align__(16) bf16 Kb[CC * CLD], Qb[CC * CLD], Xb[CC * CLD], Vb[CC * CLD], Tb[CC * CLP], Ps[CC * CLP];
    __shared__ float Af[CC][CC + 1], gl[CC], be[CC];
    const int tile = blockIdx.x, kh = blockIdx.y, G = HV / HK;
    const unsigned b = tiles[2 * tile], t0 = tiles[2 * tile + 1];
    const long start = cu[b];
    const int len = cu[b + 1] - start, n = min(CC, len - (int)t0);
    const long row0 = start + t0;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const int rb = 16 * (w & 1);
    for (int e = tid; e < CC * 32; e += 128) {
        const int r = e >> 5, c4 = e & 31;
        const bool ok = r < n;
        const float4 kk = ok ? *(const float4*)(kn + ((row0 + r) * HK + kh) * DK + 4 * c4) : make_float4(0.f, 0.f, 0.f, 0.f);
        const float4 qq = ok ? *(const float4*)(qn + ((row0 + r) * HK + kh) * DK + 4 * c4) : make_float4(0.f, 0.f, 0.f, 0.f);
        *(uint2*)(Kb + r * CLD + 4 * c4) = make_uint2(pack_bf16(kk.x, kk.y), pack_bf16(kk.z, kk.w));
        *(uint2*)(Qb + r * CLD + 4 * c4) = make_uint2(pack_bf16(qq.x, qq.y), pack_bf16(qq.z, qq.w));
    }
    __syncthreads();
    // K K^T and Q K^T once for the key head: warp w -> rows 16 (w & 1), columns 16 (w >> 1)
    float kk2[2][4] = {}, qk2[2][4] = {};
    {
        const int cb = 16 * (w >> 1);
#pragma unroll
        for (int kc = 0; kc < 8; kc++) {
            unsigned ak[4], aq[4], bb[4];
            ldsm4(ak, Kb + (rb + (lane & 15)) * CLD + 16 * kc + 8 * (lane >> 4));
            ldsm4(aq, Qb + (rb + (lane & 15)) * CLD + 16 * kc + 8 * (lane >> 4));
            ldsm4(bb, Kb + (cb + (lane & 7) + 8 * (lane >> 4)) * CLD + 16 * kc + 8 * ((lane >> 3) & 1));
            mma16816(kk2[0], ak, bb[0], bb[1]);
            mma16816(kk2[1], ak, bb[2], bb[3]);
            mma16816(qk2[0], aq, bb[0], bb[1]);
            mma16816(qk2[1], aq, bb[2], bb[3]);
        }
    }
    for (int hi = 0; hi < G; hi++) {
        const int h = kh * G + hi;
        __syncthreads();  // the previous value head is done with the shared tiles
        if (tid < CC) {
            const int r = tid;
            const bool ok = r < n;
            gl[r] = ok ? logf(fmaxf(gate[2 * ((row0 + r) * HV + h)], 1e-30f)) : 0.f;
            be[r] = ok ? gate[2 * ((row0 + r) * HV + h) + 1] : 0.f;
        }
        for (int e = tid; e < CC * 16; e += 128) {
            const int r = e >> 4, c8 = e & 15;
            *(uint4*)(Vb + r * CLD + 8 * c8) = r < n ? *(const uint4*)(vin + (row0 + r) * vld + h * DV + 8 * c8) : make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
        if (w == 0) {
            float x = gl[lane];
            for (int o = 1; o < 32; o <<= 1) {
                const float y = __shfl_up_sync(FULL, x, o);
                if (lane >= o) x += y;
            }
            gl[lane] = x;
        }
        __syncthreads();
        {
            const int cb = 16 * (w >> 1);
#pragma unroll
            for (int nb = 0; nb < 2; nb++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int r = rb + gid + 8 * (e >> 1), s2 = cb + 8 * nb + 2 * tig + (e & 1);
                    const float dec = s2 <= r ? expf(gl[r] - gl[s2]) : 0.f;
                    Af[r][s2] = s2 < r ? be[r] * kk2[nb][e] * dec : 0.f;
                    Ps[r * CLP + s2] = f2bf(qk2[nb][e] * dec);
                }
        }
        __syncthreads();
        if (w == 0) {  // T = (I + A)^-1 by forward substitution: lane j owns column j, in registers (fully unrolled)
            const int j = lane;
            float tc[CC];
#pragma unroll
            for (int r = 0; r < CC; r++) {
                float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
#pragma unroll
                for (int s2 = 0; s2 < r; s2++) {
                    const float t = s2 >= j ? Af[r][s2] * tc[s2] : 0.f;
                    if ((s2 & 3) == 0) p0 += t; else if ((s2 & 3) == 1) p1 += t; else if ((s2 & 3) == 2) p2 += t; else p3 += t;
                }
                tc[r] = r < j ? 0.f : (r == j ? 1.f : -((p0 + p1) + (p2 + p3)));
            }
#pragma unroll
            for (int r = 0; r < CC; r++) Tb[r * CLP + j] = f2bf(tc[r]);
        }
        {  // X = beta gamma K, beta V in place; Q gamma and K gamma_C / gamma straight to global
            const float gC = gl[n - 1];
            for (int e = tid; e < CC * 32; e += 128) {
                const int r = e >> 5, c4 = 4 * (e & 31);
                const float sx = be[r] * expf(gl[r]), sq = expf(gl[r]), sk = expf(gC - gl[r]);
                float k4[4], q4[4], v4[4];
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    k4[i] = bf2f(Kb[r * CLD + c4 + i]);
                    q4[i] = bf2f(Qb[r * CLD + c4 + i]);
                    v4[i] = bf2f(Vb[r * CLD + c4 + i]) * be[r];
                }
                *(uint2*)(Xb + r * CLD + c4) = make_uint2(pack_bf16(k4[0] * sx, k4[1] * sx), pack_bf16(k4[2] * sx, k4[3] * sx));
                *(uint2*)(Vb + r * CLD + c4) = make_uint2(pack_bf16(v4[0], v4[1]), pack_bf16(v4[2], v4[3]));
                if (r < n) {
                    const long o = ((row0 + r) * HV + h) * DK + c4;
                    *(uint2*)(Qgout + o) = make_uint2(pack_bf16(q4[0] * sq, q4[1] * sq), pack_bf16(q4[2] * sq, q4[3] * sq));
                    *(uint2*)(Kdout + o) = make_uint2(pack_bf16(k4[0] * sk, k4[1] * sk), pack_bf16(k4[2] * sk, k4[3] * sk));
                }
            }
        }
        __syncthreads();
        {  // W = T X, U = T (beta V): warp w -> rows 16 (w & 1), columns 64 (w >> 1) .. + 64
            const int cb = 64 * (w >> 1);
            float wa[8][4] = {}, ua[8][4] = {};
#pragma unroll
            for (int kc = 0; kc < 2; kc++) {
                unsigned at[4];
                ldsm4(at, Tb + (rb + (lane & 15)) * CLP + 16 * kc + 8 * (lane >> 4));
#pragma unroll
                for (int nb = 0; nb < 8; nb += 2) {
                    unsigned bk[4], bv[4];
                    ldsm4t(bk, Xb + (16 * kc + (lane & 15)) * CLD + cb + 8 * nb + 8 * (lane >> 4));
                    ldsm4t(bv, Vb + (16 * kc + (lane & 15)) * CLD + cb + 8 * nb + 8 * (lane >> 4));
                    mma16816(wa[nb], at, bk[0], bk[1]);
                    mma16816(wa[nb + 1], at, bk[2], bk[3]);
                    mma16816(ua[nb], at, bv[0], bv[1]);
                    mma16816(ua[nb + 1], at, bv[2], bv[3]);
                }
            }
            __syncthreads();  // every warp is done reading X / V: stage W into X and U into V
#pragma unroll
            for (int nb = 0; nb < 8; nb++)
#pragma unroll
                for (int e = 0; e < 4; e += 2) {
                    const int r = rb + gid + 8 * (e >> 1), c = cb + 8 * nb + 2 * tig;
                    *(unsigned*)(Xb + r * CLD + c) = pack_bf16(wa[nb][e], wa[nb][e + 1]);
                    *(unsigned*)(Vb + r * CLD + c) = pack_bf16(ua[nb][e], ua[nb][e + 1]);
                }
        }
        __syncthreads();
        for (int e = tid; e < CC * 16; e += 128) {
            const int r = e >> 4, c8 = e & 15;
            if (r >= n) continue;
            *(uint4*)(Wout + ((row0 + r) * HV + h) * DK + 8 * c8) = *(const uint4*)(Xb + r * CLD + 8 * c8);
            *(uint4*)(Uout + ((row0 + r) * HV + h) * DV + 8 * c8) = *(const uint4*)(Vb + r * CLD + 8 * c8);
            if (c8 < 4) *(uint4*)(Pout + ((row0 + r) * HV + h) * CC + 8 * c8) = *(const uint4*)(Ps + r * CLP + 8 * c8);
        }
        if (tid < n) gout[(row0 + tid) * HV + h] = gl[tid];
    }
}

extern "C" __global__ void __launch_bounds__(128) gdn_c1_timed(long long* tm, const float* qn, const float* kn, const bf16* vin, int vld, const float* gate,
                                                        const unsigned* tiles, const unsigned* cu, bf16* Wout, bf16* Uout, bf16* Pout,
                                                        float* gout, bf16* Qgout, bf16* Kdout, int HK, int HV) {
    __shared__ __align__(16) bf16 Kb[CC * CLD], Qb[CC * CLD], Xb[CC * CLD], Vb[CC * CLD], Tb[CC * CLP], Ps[CC * CLP];
    __shared__ float Af[CC][CC + 1], gl[CC], be[CC];
    const int tile = blockIdx.x, kh = blockIdx.y, G = HV / HK;
    const unsigned b = tiles[2 * tile], t0 = tiles[2 * tile + 1];
    const long start = cu[b];
    const int len = cu[b + 1] - start, n = min(CC, len - (int)t0);
    const long row0 = start + t0;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const int rb = 16 * (w & 1);
    if (threadIdx.x == 0) tm[(blockIdx.y * gridDim.x + blockIdx.x) * 24 + 0] = clock64();
    for (int e = tid; e < CC * 32; e += 128) {
        const int r = e >> 5, c4 = e & 31;
        const bool ok = r < n;
        const float4 kk = ok ? *(const float4*)(kn + ((row0 + r) * HK + kh) * DK + 4 * c4) : make_float4(0.f, 0.f, 0.f, 0.f);
        const float4 qq = ok ? *(const float4*)(qn + ((row0 + r) * HK + kh) * DK + 4 * c4) : make_float4(0.f, 0.f, 0.f, 0.f);
        *(uint2*)(Kb + r * CLD + 4 * c4) = make_uint2(pack_bf16(kk.x, kk.y), pack_bf16(kk.z, kk.w));
        *(uint2*)(Qb + r * CLD + 4 * c4) = make_uint2(pack_bf16(qq.x, qq.y), pack_bf16(qq.z, qq.w));
    }
    __syncthreads();
        if (threadIdx.x == 0) tm[(blockIdx.y * gridDim.x + blockIdx.x) * 24 + 1] = clock64();
    // K K^T and Q K^T once for the key head: warp w -> rows 16 (w & 1), columns 16 (w >> 1)
    float kk2[2][4] = {}, qk2[2][4] = {};
    {
        const int cb = 16 * (w >> 1);
#pragma unroll
        for (int kc = 0; kc < 8; kc++) {
            unsigned ak[4], aq[4], bb[4];
            ldsm4(ak, Kb + (rb + (lane & 15)) * CLD + 16 * kc + 8 * (lane >> 4));
            ldsm4(aq, Qb + (rb + (lane & 15)) * CLD + 16 * kc + 8 * (lane >> 4));
            ldsm4(bb, Kb + (cb + (lane & 7) + 8 * (lane >> 4)) * CLD + 16 * kc + 8 * ((lane >> 3) & 1));
            mma16816(kk2[0], ak, bb[0], bb[1]);
            mma16816(kk2[1], ak, bb[2], bb[3]);
            mma16816(qk2[0], aq, bb[0], bb[1]);
            mma16816(qk2[1], aq, bb[2], bb[3]);
        }
    }
    for (int hi = 0; hi < G; hi++) {
        const int h = kh * G + hi;
        __syncthreads();  // the previous value head is done with the shared tiles
        if (threadIdx.x == 0) tm[(blockIdx.y * gridDim.x + blockIdx.x) * 24 + 2] = clock64();
        if (tid < CC) {
            const int r = tid;
            const bool ok = r < n;
            gl[r] = ok ? logf(fmaxf(gate[2 * ((row0 + r) * HV + h)], 1e-30f)) : 0.f;
            be[r] = ok ? gate[2 * ((row0 + r) * HV + h) + 1] : 0.f;
        }
        for (int e = tid; e < CC * 16; e += 128) {
            const int r = e >> 4, c8 = e & 15;
            *(uint4*)(Vb + r * CLD + 8 * c8) = r < n ? *(const uint4*)(vin + (row0 + r) * vld + h * DV + 8 * c8) : make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
        if (threadIdx.x == 0) tm[(blockIdx.y * gridDim.x + blockIdx.x) * 24 + 3] = clock64();
        if (w == 0) {
            float x = gl[lane];
            for (int o = 1; o < 32; o <<= 1) {
                const float y = __shfl_up_sync(FULL, x, o);
                if (lane >= o) x += y;
            }
            gl[lane] = x;
        }
        __syncthreads();
        if (threadIdx.x == 0) tm[(blockIdx.y * gridDim.x + blockIdx.x) * 24 + 4] = clock64();
        {
            const int cb = 16 * (w >> 1);
#pragma unroll
            for (int nb = 0; nb < 2; nb++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int r = rb + gid + 8 * (e >> 1), s2 = cb + 8 * nb + 2 * tig + (e & 1);
                    const float dec = s2 <= r ? expf(gl[r] - gl[s2]) : 0.f;
                    Af[r][s2] = s2 < r ? be[r] * kk2[nb][e] * dec : 0.f;
                    Ps[r * CLP + s2] = f2bf(qk2[nb][e] * dec);
                }
        }
        __syncthreads();
        if (threadIdx.x == 0) tm[(blockIdx.y * gridDim.x + blockIdx.x) * 24 + 5] = clock64();
        if (w == 0) {  // T = (I + A)^-1 by forward substitution: lane j owns column j, in registers (fully unrolled)
            const int j = lane;
            float tc[CC];
#pragma unroll
            for (int r = 0; r < CC; r++) {
                float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
#pragma unroll
                for (int s2 = 0; s2 < r; s2++) {
                    const float t = s2 >= j ? Af[r][s2] * tc[s2] : 0.f;
                    if ((s2 & 3) == 0) p0 += t; else if ((s2 & 3) == 1) p1 += t; else if ((s2 & 3) == 2) p2 += t; else p3 += t;
                }
                tc[r] = r < j ? 0.f : (r == j ? 1.f : -((p0 + p1) + (p2 + p3)));
            }
#pragma unroll
            for (int r = 0; r < CC; r++) Tb[r * CLP + j] = f2bf(tc[r]);
        }
        {  // X = beta gamma K, beta V in place; Q gamma and K gamma_C / gamma straight to global
            const float gC = gl[n - 1];
            for (int e = tid; e < CC * 32; e += 128) {
                const int r = e >> 5, c4 = 4 * (e & 31);
                const float sx = be[r] * expf(gl[r]), sq = expf(gl[r]), sk = expf(gC - gl[r]);
                float k4[4], q4[4], v4[4];
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    k4[i] = bf2f(Kb[r * CLD + c4 + i]);
                    q4[i] = bf2f(Qb[r * CLD + c4 + i]);
                    v4[i] = bf2f(Vb[r * CLD + c4 + i]) * be[r];
                }
                *(uint2*)(Xb + r * CLD + c4) = make_uint2(pack_bf16(k4[0] * sx, k4[1] * sx), pack_bf16(k4[2] * sx, k4[3] * sx));
                *(uint2*)(Vb + r * CLD + c4) = make_uint2(pack_bf16(v4[0], v4[1]), pack_bf16(v4[2], v4[3]));
                if (r < n) {
                    const long o = ((row0 + r) * HV + h) * DK + c4;
                    *(uint2*)(Qgout + o) = make_uint2(pack_bf16(q4[0] * sq, q4[1] * sq), pack_bf16(q4[2] * sq, q4[3] * sq));
                    *(uint2*)(Kdout + o) = make_uint2(pack_bf16(k4[0] * sk, k4[1] * sk), pack_bf16(k4[2] * sk, k4[3] * sk));
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) tm[(blockIdx.y * gridDim.x + blockIdx.x) * 24 + 6] = clock64();
        {  // W = T X, U = T (beta V): warp w -> rows 16 (w & 1), columns 64 (w >> 1) .. + 64
            const int cb = 64 * (w >> 1);
            float wa[8][4] = {}, ua[8][4] = {};
#pragma unroll
            for (int kc = 0; kc < 2; kc++) {
                unsigned at[4];
                ldsm4(at, Tb + (rb + (lane & 15)) * CLP + 16 * kc + 8 * (lane >> 4));
#pragma unroll
                for (int nb = 0; nb < 8; nb += 2) {
                    unsigned bk[4], bv[4];
                    ldsm4t(bk, Xb + (16 * kc + (lane & 15)) * CLD + cb + 8 * nb + 8 * (lane >> 4));
                    ldsm4t(bv, Vb + (16 * kc + (lane & 15)) * CLD + cb + 8 * nb + 8 * (lane >> 4));
                    mma16816(wa[nb], at, bk[0], bk[1]);
                    mma16816(wa[nb + 1], at, bk[2], bk[3]);
                    mma16816(ua[nb], at, bv[0], bv[1]);
                    mma16816(ua[nb + 1], at, bv[2], bv[3]);
                }
            }
            __syncthreads();  // every warp is done reading X / V: stage W into X and U into V
        if (threadIdx.x == 0) tm[(blockIdx.y * gridDim.x + blockIdx.x) * 24 + 7] = clock64();
#pragma unroll
            for (int nb = 0; nb < 8; nb++)
#pragma unroll
                for (int e = 0; e < 4; e += 2) {
                    const int r = rb + gid + 8 * (e >> 1), c = cb + 8 * nb + 2 * tig;
                    *(unsigned*)(Xb + r * CLD + c) = pack_bf16(wa[nb][e], wa[nb][e + 1]);
                    *(unsigned*)(Vb + r * CLD + c) = pack_bf16(ua[nb][e], ua[nb][e + 1]);
                }
        }
        __syncthreads();
        if (threadIdx.x == 0) tm[(blockIdx.y * gridDim.x + blockIdx.x) * 24 + 8] = clock64();
        for (int e = tid; e < CC * 16; e += 128) {
            const int r = e >> 4, c8 = e & 15;
            if (r >= n) continue;
            *(uint4*)(Wout + ((row0 + r) * HV + h) * DK + 8 * c8) = *(const uint4*)(Xb + r * CLD + 8 * c8);
            *(uint4*)(Uout + ((row0 + r) * HV + h) * DV + 8 * c8) = *(const uint4*)(Vb + r * CLD + 8 * c8);
            if (c8 < 4) *(uint4*)(Pout + ((row0 + r) * HV + h) * CC + 8 * c8) = *(const uint4*)(Ps + r * CLP + 8 * c8);
        }
        if (tid < n) gout[(row0 + tid) * HV + h] = gl[tid];
    }
}


// G2: grid (HV * 2, listed sequences), block 4 warps covering value columns [64 vt, 64 vt + 64) of head h; warp w
// owns 16 of them as rows of hT = S^T (f32 accumulators, 16 x 128). Per chunk: V_new^T = U^T - hT W^T;
// O^T = hT (Q gamma)^T + V_new^T P^T; hT = gamma_C hT + V_new^T (K gamma_C / gamma).
extern "C" __global__ void __launch_bounds__(128) gdn_c2(const bf16* Qgin, const bf16* Kdin, const bf16* Win, const bf16* Uin, const bf16* Pin,
                                                        const float* gin, float* out, const unsigned* cu, const unsigned* list,
                                                        const unsigned long long* rd, const unsigned long long* wr, int HK, int HV, int lg) {
    __shared__ __align__(16) bf16 Wb[CC * CLD], Qg[CC * CLD], Kd[CC * CLD], Pb[CC * CLP], Ub[CC * CLU];
    __shared__ float gl[CC];
    const int h = blockIdx.x >> 1, vt = blockIdx.x & 1, kh = h / (HV / HK);
    const unsigned b = list[blockIdx.y];
    const long start = cu[b];
    const int len = cu[b + 1] - start;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const int dvl = 16 * w;  // this warp's first value column within the block's 64
    float hT[16][4];
    const long soff = ((long)lg * HV + h) * DK * DV;
    const float* s0 = rd[b] ? (const float*)rd[b] + soff : nullptr;
#pragma unroll
    for (int nb = 0; nb < 16; nb++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int dv = 64 * vt + dvl + gid + 8 * (e >> 1), dk = 8 * nb + 2 * tig + (e & 1);
            hT[nb][e] = s0 ? s0[(long)dk * DV + dv] : 0.f;
        }
    for (int t0 = 0; t0 < len; t0 += CC) {
        const int n = min(CC, len - t0);
        const long row0 = start + t0;
        __syncthreads();
        if (tid < CC) gl[tid] = tid < n ? gin[(row0 + tid) * HV + h] : 0.f;
        __syncthreads();
        const float gC = gl[n - 1];
        for (int e = tid; e < CC * 16; e += 128) {  // W, Q gamma, K gamma_C / gamma rows: 16 chunks of 8 bf16
            const int r = e >> 4, c8 = e & 15;
            const bool ok = r < n;
            const long src = ((row0 + r) * HV + h) * DK + 8 * c8;
            *(uint4*)(Wb + r * CLD + 8 * c8) = ok ? *(const uint4*)(Win + src) : make_uint4(0, 0, 0, 0);
            *(uint4*)(Qg + r * CLD + 8 * c8) = ok ? *(const uint4*)(Qgin + src) : make_uint4(0, 0, 0, 0);
            *(uint4*)(Kd + r * CLD + 8 * c8) = ok ? *(const uint4*)(Kdin + src) : make_uint4(0, 0, 0, 0);
        }
        for (int e = tid; e < CC * 4; e += 128) {  // P rows: 4 chunks of 8
            const int r = e >> 2, c8 = e & 3;
            *(uint4*)(Pb + r * CLP + 8 * c8) = r < n ? *(const uint4*)(Pin + ((row0 + r) * HV + h) * CC + 8 * c8) : make_uint4(0, 0, 0, 0);
        }
        for (int e = tid; e < CC * 8; e += 128) {  // U rows, this block's 64 columns
            const int r = e >> 3, c8 = e & 7;
            *(uint4*)(Ub + r * CLU + 8 * c8) = r < n ? *(const uint4*)(Uin + ((row0 + r) * HV + h) * DV + 64 * vt + 8 * c8) : make_uint4(0, 0, 0, 0);
        }
        __syncthreads();
        unsigned ha[8][4];
#pragma unroll
        for (int kc = 0; kc < 8; kc++) {
            ha[kc][0] = pack_bf16(hT[2 * kc][0], hT[2 * kc][1]);
            ha[kc][1] = pack_bf16(hT[2 * kc][2], hT[2 * kc][3]);
            ha[kc][2] = pack_bf16(hT[2 * kc + 1][0], hT[2 * kc + 1][1]);
            ha[kc][3] = pack_bf16(hT[2 * kc + 1][2], hT[2 * kc + 1][3]);
        }
        float mw[4][4] = {}, o[4][4] = {};
#pragma unroll
        for (int kc = 0; kc < 8; kc++)
#pragma unroll
            for (int nb = 0; nb < 4; nb += 2) {
                unsigned bw[4], bq[4];
                ldsm4(bw, Wb + (8 * nb + (lane & 7) + 8 * (lane >> 4)) * CLD + 16 * kc + 8 * ((lane >> 3) & 1));
                ldsm4(bq, Qg + (8 * nb + (lane & 7) + 8 * (lane >> 4)) * CLD + 16 * kc + 8 * ((lane >> 3) & 1));
                mma16816(mw[nb], ha[kc], bw[0], bw[1]);
                mma16816(mw[nb + 1], ha[kc], bw[2], bw[3]);
                mma16816(o[nb], ha[kc], bq[0], bq[1]);
                mma16816(o[nb + 1], ha[kc], bq[2], bq[3]);
            }
        // V_new^T = U^T - hT W^T, then as bf16 A fragments over the chunk (2 k-steps)
        float vn[4][4];
#pragma unroll
        for (int nb = 0; nb < 4; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int dv = dvl + gid + 8 * (e >> 1), c = 8 * nb + 2 * tig + (e & 1);
                vn[nb][e] = bf2f(Ub[c * CLU + dv]) - mw[nb][e];
            }
        unsigned va[2][4];
#pragma unroll
        for (int kc = 0; kc < 2; kc++) {
            va[kc][0] = pack_bf16(vn[2 * kc][0], vn[2 * kc][1]);
            va[kc][1] = pack_bf16(vn[2 * kc][2], vn[2 * kc][3]);
            va[kc][2] = pack_bf16(vn[2 * kc + 1][0], vn[2 * kc + 1][1]);
            va[kc][3] = pack_bf16(vn[2 * kc + 1][2], vn[2 * kc + 1][3]);
        }
#pragma unroll
        for (int kc = 0; kc < 2; kc++)
#pragma unroll
            for (int nb = 0; nb < 4; nb += 2) {
                unsigned bp[4];
                ldsm4(bp, Pb + (8 * nb + (lane & 7) + 8 * (lane >> 4)) * CLP + 16 * kc + 8 * ((lane >> 3) & 1));
                mma16816(o[nb], va[kc], bp[0], bp[1]);
                mma16816(o[nb + 1], va[kc], bp[2], bp[3]);
            }
#pragma unroll
        for (int nb = 0; nb < 4; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int dv = 64 * vt + dvl + gid + 8 * (e >> 1), c = 8 * nb + 2 * tig + (e & 1);
                if (c < n) out[((row0 + c) * HV + h) * DV + dv] = o[nb][e];
            }
        const float gamC = expf(gC);
#pragma unroll
        for (int nb = 0; nb < 16; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) hT[nb][e] *= gamC;
#pragma unroll
        for (int kc = 0; kc < 2; kc++)
#pragma unroll
            for (int nb = 0; nb < 16; nb += 2) {
                unsigned bk[4];
                ldsm4t(bk, Kd + (16 * kc + (lane & 15)) * CLD + 8 * nb + 8 * (lane >> 4));
                mma16816(hT[nb], va[kc], bk[0], bk[1]);
                mma16816(hT[nb + 1], va[kc], bk[2], bk[3]);
            }
    }
    if (wr[b]) {
        float* st = (float*)wr[b] + soff;
#pragma unroll
        for (int nb = 0; nb < 16; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int dv = 64 * vt + dvl + gid + 8 * (e >> 1), dk = 8 * nb + 2 * tig + (e & 1);
                st[(long)dk * DV + dv] = hT[nb][e];
            }
    }
}


// Fused chunked gated delta rule: one block of 8 warps per (listed sequence, value head) runs every 32-token chunk
// start to finish. Chunk-local (shared memory): A = beta K K^T e^(g_r - g_s) (s < r), P = Q K^T e^(g_r - g_s) (s <= r),
// T = (I + A)^-1, W = T (beta gamma K), U = T (beta V), Q gamma, K gamma_C / gamma. State: warp w keeps rows
// [16 w, 16 w + 16) of hT = S^T as f32 mma accumulators; V_new^T = U^T - hT W^T, O^T = hT (Q gamma)^T + V_new^T P^T,
// hT = gamma_C hT + V_new^T (K gamma_C / gamma). Nothing chunk-local leaves the SM.
extern "C" __global__ void __launch_bounds__(256, GCF_MINB) gdn_cf(const float* qn, const float* kn, const bf16* vin, int vld, const float* gate, float* out,
                                                        const unsigned* cu, const unsigned* list, const unsigned long long* rd,
                                                        const unsigned long long* wr, int HK, int HV, int lg) {
    __shared__ __align__(16) bf16 Kb[CC * CLD], Qb[CC * CLD], Vb[CC * CLD], Xb[CC * CLD], Tb[CC * CLP], Ps[CC * CLP];
    __shared__ float Af[CC][CC + 1], gl[CC], be[CC];
    const int h = blockIdx.x, kh = h / (HV / HK);
    const unsigned b = list[blockIdx.y];
    const long start = cu[b];
    const int len = cu[b + 1] - start;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    float hT[16][4];
    const long soff = ((long)lg * HV + h) * DK * DV;
    const float* s0 = rd[b] ? (const float*)rd[b] + soff : nullptr;
#pragma unroll
    for (int nb = 0; nb < 16; nb++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int dv = 16 * w + gid + 8 * (e >> 1), dk = 8 * nb + 2 * tig + (e & 1);
            hT[nb][e] = s0 ? s0[(long)dk * DV + dv] : 0.f;
        }
    for (int t0 = 0; t0 < len; t0 += CC) {
        const int n = min(CC, len - t0);
        const long row0 = start + t0;
        __syncthreads();  // the previous chunk is done with every tile
        for (int e = tid; e < CC * 32; e += 256) {
            const int r = e >> 5, c4 = e & 31;
            const bool ok = r < n;
            const float4 kk = ok ? *(const float4*)(kn + ((row0 + r) * HK + kh) * DK + 4 * c4) : make_float4(0.f, 0.f, 0.f, 0.f);
            const float4 qq = ok ? *(const float4*)(qn + ((row0 + r) * HK + kh) * DK + 4 * c4) : make_float4(0.f, 0.f, 0.f, 0.f);
            *(uint2*)(Kb + r * CLD + 4 * c4) = make_uint2(pack_bf16(kk.x, kk.y), pack_bf16(kk.z, kk.w));
            *(uint2*)(Qb + r * CLD + 4 * c4) = make_uint2(pack_bf16(qq.x, qq.y), pack_bf16(qq.z, qq.w));
        }
        for (int e = tid; e < CC * 16; e += 256) {
            const int r = e >> 4, c8 = e & 15;
            *(uint4*)(Vb + r * CLD + 8 * c8) = r < n ? *(const uint4*)(vin + (row0 + r) * vld + h * DV + 8 * c8) : make_uint4(0, 0, 0, 0);
        }
        if (tid < CC) {
            const bool ok = tid < n;
            gl[tid] = ok ? logf(fmaxf(gate[2 * ((row0 + tid) * HV + h)], 1e-30f)) : 0.f;
            be[tid] = ok ? gate[2 * ((row0 + tid) * HV + h) + 1] : 0.f;
        }
        __syncthreads();
        if (w == 0) {
            float x = gl[lane];
            for (int o = 1; o < 32; o <<= 1) {
                const float y = __shfl_up_sync(FULL, x, o);
                if (lane >= o) x += y;
            }
            gl[lane] = x;
        }
        __syncthreads();
        {  // warps 0-3: A tiles, warps 4-7: P tiles (16 x 16 each)
            const int t = w & 3, rb = 16 * (t & 1), cb = 16 * (t >> 1);
            const bf16* Am = w < 4 ? Kb : Qb;
            float acc[2][4] = {};
#pragma unroll
            for (int kc = 0; kc < 8; kc++) {
                unsigned aa[4], bb[4];
                ldsm4(aa, Am + (rb + (lane & 15)) * CLD + 16 * kc + 8 * (lane >> 4));
                ldsm4(bb, Kb + (cb + (lane & 7) + 8 * (lane >> 4)) * CLD + 16 * kc + 8 * ((lane >> 3) & 1));
                mma16816(acc[0], aa, bb[0], bb[1]);
                mma16816(acc[1], aa, bb[2], bb[3]);
            }
#pragma unroll
            for (int nb = 0; nb < 2; nb++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int r = rb + gid + 8 * (e >> 1), s2 = cb + 8 * nb + 2 * tig + (e & 1);
                    const float dec = s2 <= r ? expf(gl[r] - gl[s2]) : 0.f;
                    if (w < 4) Af[r][s2] = s2 < r ? be[r] * acc[nb][e] * dec : 0.f;
                    else Ps[r * CLP + s2] = f2bf(acc[nb][e] * dec);
                }
        }
        __syncthreads();
        if (w == 0) {  // T = (I + A)^-1 by forward substitution: lane j owns column j, in registers
            const int j = lane;
            float tc[CC];
#pragma unroll
            for (int r = 0; r < CC; r++) {
                float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
#pragma unroll
                for (int s2 = 0; s2 < r; s2++) {
                    const float t = s2 >= j ? Af[r][s2] * tc[s2] : 0.f;
                    if ((s2 & 3) == 0) p0 += t; else if ((s2 & 3) == 1) p1 += t; else if ((s2 & 3) == 2) p2 += t; else p3 += t;
                }
                tc[r] = r < j ? 0.f : (r == j ? 1.f : -((p0 + p1) + (p2 + p3)));
            }
#pragma unroll
            for (int r = 0; r < CC; r++) Tb[r * CLP + j] = f2bf(tc[r]);
        } else {  // meanwhile: X = beta gamma K, Kd = K gamma_C / gamma (over K), Q gamma (over Q), beta V (over V)
            const float gC = gl[n - 1];
            for (int e = tid - 32; e < CC * 32; e += 224) {
                const int r = e >> 5, c4 = 4 * (e & 31);
                const float sq = expf(gl[r]), sx = be[r] * sq, sk = expf(gC - gl[r]);
                float k4[4], q4[4], v4[4];
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    k4[i] = bf2f(Kb[r * CLD + c4 + i]);
                    q4[i] = bf2f(Qb[r * CLD + c4 + i]);
                    v4[i] = bf2f(Vb[r * CLD + c4 + i]) * be[r];
                }
                *(uint2*)(Xb + r * CLD + c4) = make_uint2(pack_bf16(k4[0] * sx, k4[1] * sx), pack_bf16(k4[2] * sx, k4[3] * sx));
                *(uint2*)(Kb + r * CLD + c4) = make_uint2(pack_bf16(k4[0] * sk, k4[1] * sk), pack_bf16(k4[2] * sk, k4[3] * sk));
                *(uint2*)(Qb + r * CLD + c4) = make_uint2(pack_bf16(q4[0] * sq, q4[1] * sq), pack_bf16(q4[2] * sq, q4[3] * sq));
                *(uint2*)(Vb + r * CLD + c4) = make_uint2(pack_bf16(v4[0], v4[1]), pack_bf16(v4[2], v4[3]));
            }
        }
        __syncthreads();
        {  // W = T X (into Xb), U = T (beta V) (into Vb): warp w -> rows 16 (w & 1), columns 32 (w >> 1) .. + 32
            const int rb = 16 * (w & 1), cb = 32 * (w >> 1);
            float wa[4][4] = {}, ua[4][4] = {};
#pragma unroll
            for (int kc = 0; kc < 2; kc++) {
                unsigned at[4];
                ldsm4(at, Tb + (rb + (lane & 15)) * CLP + 16 * kc + 8 * (lane >> 4));
#pragma unroll
                for (int nb = 0; nb < 4; nb += 2) {
                    unsigned bk[4], bv[4];
                    ldsm4t(bk, Xb + (16 * kc + (lane & 15)) * CLD + cb + 8 * nb + 8 * (lane >> 4));
                    ldsm4t(bv, Vb + (16 * kc + (lane & 15)) * CLD + cb + 8 * nb + 8 * (lane >> 4));
                    mma16816(wa[nb], at, bk[0], bk[1]);
                    mma16816(wa[nb + 1], at, bk[2], bk[3]);
                    mma16816(ua[nb], at, bv[0], bv[1]);
                    mma16816(ua[nb + 1], at, bv[2], bv[3]);
                }
            }
            __syncthreads();
#pragma unroll
            for (int nb = 0; nb < 4; nb++)
#pragma unroll
                for (int e = 0; e < 4; e += 2) {
                    const int r = rb + gid + 8 * (e >> 1), c = cb + 8 * nb + 2 * tig;
                    *(unsigned*)(Xb + r * CLD + c) = pack_bf16(wa[nb][e], wa[nb][e + 1]);
                    *(unsigned*)(Vb + r * CLD + c) = pack_bf16(ua[nb][e], ua[nb][e + 1]);
                }
        }
        __syncthreads();
        // state: warp w owns hT rows [16 w, 16 w + 16); W in Xb, U in Vb, Q gamma in Qb, K gamma_C / gamma in Kb
        float mw[4][4] = {}, o[4][4] = {};
#pragma unroll
        for (int kc = 0; kc < 8; kc++) {
            const unsigned ha[4] = {pack_bf16(hT[2 * kc][0], hT[2 * kc][1]), pack_bf16(hT[2 * kc][2], hT[2 * kc][3]),
                                    pack_bf16(hT[2 * kc + 1][0], hT[2 * kc + 1][1]), pack_bf16(hT[2 * kc + 1][2], hT[2 * kc + 1][3])};
#pragma unroll
            for (int nb = 0; nb < 4; nb += 2) {
                unsigned bw[4], bq[4];
                ldsm4(bw, Xb + (8 * nb + (lane & 7) + 8 * (lane >> 4)) * CLD + 16 * kc + 8 * ((lane >> 3) & 1));
                ldsm4(bq, Qb + (8 * nb + (lane & 7) + 8 * (lane >> 4)) * CLD + 16 * kc + 8 * ((lane >> 3) & 1));
                mma16816(mw[nb], ha, bw[0], bw[1]);
                mma16816(mw[nb + 1], ha, bw[2], bw[3]);
                mma16816(o[nb], ha, bq[0], bq[1]);
                mma16816(o[nb + 1], ha, bq[2], bq[3]);
            }
        }
        unsigned va[2][4];
#pragma unroll
        for (int kc = 0; kc < 2; kc++)
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                const int nb = 2 * kc + hh;
                float v0[4];
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int dv = 16 * w + gid + 8 * (e >> 1), c = 8 * nb + 2 * tig + (e & 1);
                    v0[e] = bf2f(Vb[c * CLD + dv]) - mw[nb][e];
                }
                va[kc][2 * hh] = pack_bf16(v0[0], v0[1]);
                va[kc][2 * hh + 1] = pack_bf16(v0[2], v0[3]);
            }
#pragma unroll
        for (int kc = 0; kc < 2; kc++)
#pragma unroll
            for (int nb = 0; nb < 4; nb += 2) {
                unsigned bp[4];
                ldsm4(bp, Ps + (8 * nb + (lane & 7) + 8 * (lane >> 4)) * CLP + 16 * kc + 8 * ((lane >> 3) & 1));
                mma16816(o[nb], va[kc], bp[0], bp[1]);
                mma16816(o[nb + 1], va[kc], bp[2], bp[3]);
            }
#pragma unroll
        for (int nb = 0; nb < 4; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int dv = 16 * w + gid + 8 * (e >> 1), c = 8 * nb + 2 * tig + (e & 1);
                if (c < n) out[((row0 + c) * HV + h) * DV + dv] = o[nb][e];
            }
        const float gamC = expf(gl[n - 1]);
#pragma unroll
        for (int nb = 0; nb < 16; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) hT[nb][e] *= gamC;
#pragma unroll
        for (int kc = 0; kc < 2; kc++)
#pragma unroll
            for (int nb = 0; nb < 16; nb += 2) {
                unsigned bk[4];
                ldsm4t(bk, Kb + (16 * kc + (lane & 15)) * CLD + 8 * nb + 8 * (lane >> 4));
                mma16816(hT[nb], va[kc], bk[0], bk[1]);
                mma16816(hT[nb + 1], va[kc], bk[2], bk[3]);
            }
    }
    if (wr[b]) {
        float* st = (float*)wr[b] + soff;
#pragma unroll
        for (int nb = 0; nb < 16; nb++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int dv = 16 * w + gid + 8 * (e >> 1), dk = 8 * nb + 2 * tig + (e & 1);
                st[(long)dk * DV + dv] = hT[nb][e];
            }
    }
}

static unsigned short h2bf(float f) { unsigned u; memcpy(&u, &f, 4); u += 0x7fffu + ((u >> 16) & 1u); return (unsigned short)(u >> 16); }
template <class F> float timeit(F f, int reps = 5) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); f();
    cudaEventRecord(a); for (int i = 0; i < reps; i++) f(); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); return ms / reps;
}

int main(int argc, char** argv) {
    const int HK = 16, HV = 32, NSEQ = argc > 1 ? atoi(argv[1]) : 20, LEN = argc > 2 ? atoi(argv[2]) : 1550;
    const long T = (long)NSEQ * LEN;
    unsigned s = 99;
    auto rnd = [&] { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xffff) / 65536.f; };
    auto nrm = [&] { float u1 = fmaxf(rnd(), 1e-7f), u2 = rnd(); return sqrtf(-2 * logf(u1)) * cosf(6.2831853f * u2); };
    std::vector<float> hq(T * HK * DK), hk(T * HK * DK), hgate(T * HV * 2);
    for (long r = 0; r < T * HK; r++) {
        double sq = 0, sk = 0;
        for (int c = 0; c < DK; c++) { hq[r * DK + c] = nrm(); hk[r * DK + c] = nrm(); sq += hq[r * DK + c] * hq[r * DK + c]; sk += hk[r * DK + c] * hk[r * DK + c]; }
        for (int c = 0; c < DK; c++) { hq[r * DK + c] /= sqrt(sq) * sqrt((double)DK); hk[r * DK + c] /= sqrt(sk); }
    }
    for (long i = 0; i < T * HV; i++) { hgate[2 * i] = 0.6f + 0.4f * rnd(); hgate[2 * i + 1] = rnd(); }
    std::vector<unsigned short> hv(T * HV * DV);
    for (auto& x : hv) x = h2bf(nrm());
    std::vector<unsigned> hcu(NSEQ + 1), hlist(NSEQ), tiles;
    for (int b = 0; b <= NSEQ; b++) hcu[b] = b * LEN;
    for (int b = 0; b < NSEQ; b++) { hlist[b] = b; for (int t0 = 0; t0 < LEN; t0 += CC) { tiles.push_back(b); tiles.push_back(t0); } }
    const int nt = tiles.size() / 2;
    float *qn, *kn, *gate, *o1, *o2, *g; bf16 *v, *W, *U, *P; unsigned *cu, *list, *dt; unsigned long long* nulls;
    cudaMalloc(&qn, hq.size() * 4); cudaMalloc(&kn, hk.size() * 4); cudaMalloc(&gate, hgate.size() * 4); cudaMalloc(&v, hv.size() * 2);
    cudaMalloc(&o1, T * HV * DV * 4); cudaMalloc(&o2, T * HV * DV * 4); cudaMalloc(&g, T * HV * 4);
    cudaMalloc(&W, T * HV * DK * 2); cudaMalloc(&U, T * HV * DV * 2); cudaMalloc(&P, T * HV * CC * 2);
    bf16 *Qgb, *Kdb;
    cudaMalloc(&Qgb, T * HV * DK * 2); cudaMalloc(&Kdb, T * HV * DK * 2);
    cudaMalloc(&cu, hcu.size() * 4); cudaMalloc(&list, hlist.size() * 4); cudaMalloc(&dt, tiles.size() * 4); cudaMalloc(&nulls, NSEQ * 8);
    cudaMemset(nulls, 0, NSEQ * 8);
    cudaMemcpy(qn, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(kn, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(gate, hgate.data(), hgate.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(v, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(cu, hcu.data(), hcu.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(list, hlist.data(), hlist.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dt, tiles.data(), tiles.size() * 4, cudaMemcpyHostToDevice);
    cudaFuncSetAttribute(gdn_fast_bf16, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(gdn_c2, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(gdn_c1, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    auto ref = [&] { gdn_fast_bf16<<<dim3(HV * DV / 8 / 4, NSEQ), 128>>>(v, HV * DV, qn, kn, gate, o1, cu, list, nulls, nulls, HK, HV, 0); };
    auto c1 = [&] { gdn_c1<<<dim3(nt, HK), 128>>>(qn, kn, v, HV * DV, gate, dt, cu, W, U, P, g, Qgb, Kdb, HK, HV); };
    auto c2 = [&] { gdn_c2<<<dim3(HV * 2, NSEQ), 128>>>(Qgb, Kdb, W, U, P, g, o2, cu, list, nulls, nulls, HK, HV, 0); };
    float tr = timeit(ref), t1 = timeit(c1), t2 = timeit(c2);
    int b1, b2; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b1, gdn_c1, 128, 0); cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b2, gdn_c2, 128, 0);
    std::vector<float> a1(T * HV * DV), a2(T * HV * DV);
    cudaMemcpy(a1.data(), o1, a1.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(a2.data(), o2, a2.size() * 4, cudaMemcpyDeviceToHost);
    double md = 0, mref = 0, se = 0, sr = 0;
    for (size_t i = 0; i < a1.size(); i++) { const double d = a1[i] - a2[i]; md = fmax(md, fabs(d)); mref = fmax(mref, fabs(a1[i])); se += d * d; sr += (double)a1[i] * a1[i]; }
    printf("gdn_fast_bf16: %.3f ms\ngdn_c1 %.3f ms (%d blocks/SM) + gdn_c2 %.3f ms (%d blocks/SM) = %.3f ms: %.2fx\n", tr, t1, b1, t2, b2, t1 + t2, tr / (t1 + t2));
    {
        long long* tm; const int NM = 24;
        cudaMalloc(&tm, (size_t)nt * HK * NM * 8); cudaMemset(tm, 0, (size_t)nt * HK * NM * 8);
        gdn_c1_timed<<<dim3(nt, HK), 128>>>(tm, qn, kn, v, HV * DV, gate, dt, cu, W, U, P, g, Qgb, Kdb, HK, HV);
        cudaDeviceSynchronize();
        std::vector<long long> ht((size_t)nt * HK * NM);
        cudaMemcpy(ht.data(), tm, ht.size() * 8, cudaMemcpyDeviceToHost);
        std::vector<double> d(NM, 0); int cnt = 0;
        for (size_t bl = 0; bl < (size_t)nt * HK; bl++) {
            const long long* t = &ht[bl * NM];
            for (int i = 1; i < NM && t[i]; i++) d[i] += t[i] - t[i - 1];
            cnt++;
        }
        printf("gdn_c1 cycles per phase (mean over blocks):");
        for (int i = 1; i < NM; i++) if (d[i] > 0) printf(" [%d] %.0f", i, d[i] / cnt);
        printf("\n");
    }
    {
        float* o3; cudaMalloc(&o3, T * HV * DV * 4);
        cudaFuncSetAttribute(gdn_cf, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        auto cf = [&] { gdn_cf<<<dim3(HV, NSEQ), 256>>>(qn, kn, v, HV * DV, gate, o3, cu, list, nulls, nulls, HK, HV, 0); };
        float tf = timeit(cf);
        int bf; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bf, gdn_cf, 256, 0);
        std::vector<float> a3(T * HV * DV), r1(T * HV * DV);
        cudaMemcpy(a3.data(), o3, a3.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(r1.data(), o1, r1.size() * 4, cudaMemcpyDeviceToHost);
        double se = 0, sr = 0, md = 0;
        for (size_t i = 0; i < a3.size(); i++) { const double d = r1[i] - a3[i]; se += d * d; sr += (double)r1[i] * r1[i]; md = fmax(md, fabs(d)); }
        printf("gdn_cf (fused): %.3f ms (%d blocks/SM): %.2fx gdn_fast; max |diff| %.3g, relative rms error %.3g\n", tf, bf, tr / tf, md, sqrt(se / sr));
    }
    printf("max |diff| %.3g of max |out| %.3g; relative rms error %.3g\n%s\n", md, mref, sqrt(se / sr), cudaGetErrorString(cudaGetLastError()));
}
