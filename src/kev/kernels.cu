// Packed-varlen kernels for the decoder layers (see kernels.rs). Compiled with NVRTC.
// T is float or bf16 (the model dtype); math is f32. Per-pass metadata:
//   tok2seq[N], pos[N], cu[B+1] (sequence starts), plen[B] (past length), tiles[2*T] (sequence, first query),
//   tier lists (sequence ids), and per-sequence cache pointers rd[4][B] (past: conv, rec, k, v) and wr[4][B] (keep).

#define DK 128
#define DV 128
#define NEG_INF __int_as_float(0xff800000)
#define FULL 0xffffffffu

typedef unsigned short bf16;
typedef unsigned int u32;
typedef unsigned long long u64;

__device__ __forceinline__ float bf2f(bf16 x) { return __uint_as_float(((unsigned)x) << 16); }
__device__ __forceinline__ bf16 f2bf(float f) {
    unsigned u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16)(u >> 16);
}
__device__ __forceinline__ float ldf(const float* p, long i) { return p[i]; }
__device__ __forceinline__ float ldf(const bf16* p, long i) { return bf2f(p[i]); }
__device__ __forceinline__ void stf(float* p, long i, float v) { p[i] = v; }
__device__ __forceinline__ void stf(bf16* p, long i, float v) { p[i] = f2bf(v); }
template <typename T> __device__ __forceinline__ float rnd(float v);
template <> __device__ __forceinline__ float rnd<float>(float v) { return v; }
template <> __device__ __forceinline__ float rnd<bf16>(float v) { return bf2f(f2bf(v)); }

// round to f16 (nearest even) and back: llama.cpp's f16 KV cache and flash-attention operands
__device__ __forceinline__ float rh(float x) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x));
    float r;
    asm("cvt.f32.f16 %0, %1;" : "=f"(r) : "h"(h));
    return r;
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
    return v;
}
// Sum over the block (blockDim.x a multiple of 32). Every thread gets the result.
__device__ float block_sum(float v) {
    __shared__ float part[32];
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (l == 0) part[w] = v;
    __syncthreads();
    v = l < nw ? part[l] : 0.f;
    return warp_sum(v);
}

// ---------------------------------------------------------------------------------------------------------------
// Residual add + RMSNorm. One block per token: xo = rnd(x + m) (when m), no = norm(xo) * w.
// ---------------------------------------------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ float residual(const T* x, const T* m, long i, float scale) {
    float s = ldf(x, i);
    if (m) {
        s = rnd<T>(s + ldf(m, i));
        if (scale != 1.f) s = rnd<T>(s * scale);
    }
    return s;
}
template <typename T>
__device__ void add_norm_body(const T* x, const T* m, const float* w, T* xo, T* no, int H, float eps, int rounded, float scale) {
    const long row = (long)blockIdx.x * H;
    float ss = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float s = residual(x, m, row + i, scale);
        ss += s * s;
    }
    ss = block_sum(ss);
    const float r = rsqrtf(ss / H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float s = residual(x, m, row + i, scale);
        if (m) stf(xo, row + i, s);
        const float wi = w ? w[i] : 1.f;
        stf(no, row + i, rounded ? rnd<T>(s * r) * wi : s * r * wi);
    }
}
extern "C" __global__ void add_norm_bf16(const bf16* x, const bf16* m, const float* w, bf16* xo, bf16* no, int H, float eps, int rounded,
                                         float scale) {
    add_norm_body(x, m, w, xo, no, H, eps, rounded, scale);
}
extern "C" __global__ void add_norm_f32(const float* x, const float* m, const float* w, float* xo, float* no, int H, float eps, int rounded,
                                        float scale) {
    add_norm_body(x, m, w, xo, no, H, eps, rounded, scale);
}

// ---------------------------------------------------------------------------------------------------------------
// act(gate) * up over gu = [gate | up] per token.
// ---------------------------------------------------------------------------------------------------------------
template <typename T>
__device__ void act_body(const T* gu, T* out, long total, int I, int gelu, int round_act) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const long t = idx / I;
    const int i = idx % I;
    const float g = ldf(gu, t * 2 * I + i), u = ldf(gu, t * 2 * I + I + i);
    float a = gelu ? 0.5f * g * (1.f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g))) : g / (1.f + expf(-g));
    if (round_act) a = rnd<T>(a);
    stf(out, idx, a * u);
}
extern "C" __global__ void act_mul_bf16(const bf16* gu, bf16* out, long total, int I, int gelu, int round_act) {
    act_body(gu, out, total, I, gelu, round_act);
}
extern "C" __global__ void act_mul_f32(const float* gu, float* out, long total, int I, int gelu, int round_act) {
    act_body(gu, out, total, I, gelu, round_act);
}

// ---------------------------------------------------------------------------------------------------------------
// q/k RMSNorm + RoPE (rotate-half over pairs (i, i + half), i < half) + optional value norm, one block of hd threads
// per (token, head); key blocks also write the value and, for state-building sequences, the cache.
// ---------------------------------------------------------------------------------------------------------------
template <typename T>
__device__ void prep_body(const T* p, int ld, const float* qw, const float* kw, int vnorm, int rounded, float eps, const float* inv,
                          int half, T* q, T* k, T* v, int nh, int nkv, int qs, int koff, int voff, int kv_eq, const u32* tok2seq,
                          const u32* pos, const u32* cu, const u64* wk, const u64* wv, long kvoff, int f16) {
    __shared__ float y[1024];
    const int hd = blockDim.x, d = threadIdx.x;
    const long n = blockIdx.x;
    const bool isq = blockIdx.y < nh;
    const int h = isq ? blockIdx.y : blockIdx.y - nh;
    const long base = n * ld + (isq ? (long)h * qs : koff + (long)h * hd);
    const float x = ldf(p, base + d);
    const float* w = isq ? qw : kw;
    float val = x;
    if (w) {
        const float ss = block_sum(x * x);
        const float xn = x * rsqrtf(ss / hd + eps);
        val = rounded ? rnd<T>(rnd<T>(xn) * w[d]) : rnd<T>(xn * w[d]);
    }
    y[d] = val;
    __syncthreads();
    float o = val;
    if (d < 2 * half) {
        const float f = (float)pos[n] * inv[d % half];
        const float c = cosf(f), s = sinf(f);
        o = d < half ? y[d] * c - y[d + half] * s : y[d] * c + y[d - half] * s;
    }
    if (f16) o = rh(o);
    if (isq) {
        stf(q, (n * nh + h) * hd + d, o);
        return;
    }
    const long kvw = (long)nkv * hd;
    stf(k, n * kvw + (long)h * hd + d, o);
    float vv = kv_eq ? x : ldf(p, n * ld + voff + (long)h * hd + d);
    if (vnorm) {
        const float ss = block_sum(vv * vv);
        vv = rnd<T>(vv * rsqrtf(ss / hd + eps));
    }
    if (f16) vv = rh(vv);
    stf(v, n * kvw + (long)h * hd + d, vv);
    const u32 b = tok2seq[n];
    if (wk[b]) {
        const long S = cu[b + 1] - cu[b], j = n - cu[b];
        const long at = S * kvoff + j * kvw + (long)h * hd + d;
        stf((T*)wk[b], at, o);
        stf((T*)wv[b], at, vv);
    }
}
#define PREP_ARGS(T)                                                                                                         \
    const T *p, int ld, const float *qw, const float *kw, int vnorm, int rounded, float eps, const float *inv, int half, T *q, \
        T *k, T *v, int nh, int nkv, int qs, int koff, int voff, int kv_eq, const u32 *tok2seq, const u32 *pos, const u32 *cu,  \
        const u64 *wk, const u64 *wv, long kvoff, int f16
#define PREP_CALL prep_body(p, ld, qw, kw, vnorm, rounded, eps, inv, half, q, k, v, nh, nkv, qs, koff, voff, kv_eq, tok2seq, pos, cu, wk, wv, kvoff, f16)
extern "C" __global__ void qkv_prep_bf16(PREP_ARGS(bf16)) { PREP_CALL; }
extern "C" __global__ void qkv_prep_f32(PREP_ARGS(float)) { PREP_CALL; }

// ---------------------------------------------------------------------------------------------------------------
// Attention over packed sequences, f32 math (online softmax, no score matrix). One block (4 warps) per (tile of 16
// queries of one sequence, query head). Keys: the sequence's past (its state's cache, plen[b] positions) then its own
// tokens, causal; optional sliding window and softcap. Warp w owns queries 4w..4w+3; in the score phase lane l owns
// key l of the 32-key tile, in the value phase dims l, l+32, ... of each of its queries.
// Dynamic shared memory: Q [16][HD] f32, then K^T [HD][32] / V [32][HD] f32 (one reused buffer).
// ---------------------------------------------------------------------------------------------------------------
template <int HD, typename T>
__device__ void attn_body(const T* q, const T* kn, const T* vn, T* out, const T* gate, int gld, int gs, const u32* tiles, const u32* cu,
                          const u32* plen, const u64* rk, const u64* rv, long kvoff, int nh, int nkv, float scale, float softcap,
                          int window, int f16) {
    // f16: llama.cpp's MMA flash attention on Turing/Ampere: Q (times the scale), K and V in f16, KQ and the softmax
    // sums in f32, P rounded to f16, V.P accumulated in f16 (one rounding per 16 keys, rescaled in f16)
    constexpr int BQ = 16, BK = 32, RW = 4, DPL = HD / 32;
    extern __shared__ float sm[];
    float* Qs = sm;
    float* KV = sm + BQ * HD;
    const int tile = blockIdx.x, h = blockIdx.y, g = h / (nh / nkv);
    const int b = tiles[2 * tile], q0 = tiles[2 * tile + 1];
    const int start = cu[b], n = cu[b + 1] - start, P = plen[b];
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const long kvw = (long)nkv * HD;
    const T* pk = P ? (const T*)rk[b] + (long)P * kvoff : nullptr;
    const T* pv = P ? (const T*)rv[b] + (long)P * kvoff : nullptr;
    for (int e = tid; e < BQ * HD; e += 128) {
        const int r = e / HD, d = e % HD;
        const float qv = q0 + r < n ? ldf(q, ((long)(start + q0 + r) * nh + h) * HD + d) : 0.f;
        Qs[e] = f16 ? rh(rh(qv) * rh(scale)) : qv;
    }
    if (f16) scale = 1.f;
    float m[RW], l[RW], acc[RW][DPL];
#pragma unroll
    for (int r = 0; r < RW; r++) {
        m[r] = NEG_INF;
        l[r] = 0.f;
#pragma unroll
        for (int i = 0; i < DPL; i++) acc[r][i] = 0.f;
    }
    const int qlast = min(q0 + BQ, n) - 1;
    const int kend = P + qlast + 1;
    int kbeg = 0;
    if (window > 0) {
        kbeg = max(0, P + q0 - window + 1);
        kbeg -= kbeg % BK;
    }
    for (int k0 = kbeg; k0 < kend; k0 += BK) {
        __syncthreads();
        for (int e = tid; e < BK * HD; e += 128) {  // K^T: consecutive threads take consecutive keys
            const int kk = e % BK, d = e / BK, j = k0 + kk;
            float x = 0.f;
            if (j < kend) x = j < P ? ldf(pk, (long)j * kvw + (long)g * HD + d) : ldf(kn, (long)(start + j - P) * kvw + (long)g * HD + d);
            KV[d * BK + kk] = x;
        }
        __syncthreads();
        float s[RW];
#pragma unroll
        for (int r = 0; r < RW; r++) s[r] = 0.f;
#pragma unroll 4
        for (int d = 0; d < HD; d += 4) {
            const float k0v = KV[d * BK + lane], k1v = KV[(d + 1) * BK + lane], k2v = KV[(d + 2) * BK + lane], k3v = KV[(d + 3) * BK + lane];
#pragma unroll
            for (int r = 0; r < RW; r++) {
                const float4 qv = *(const float4*)&Qs[(w * RW + r) * HD + d];
                s[r] += qv.x * k0v + qv.y * k1v + qv.z * k2v + qv.w * k3v;
            }
        }
        const int j = k0 + lane;
        float pr[RW];
#pragma unroll
        for (int r = 0; r < RW; r++) {
            const int i = q0 + w * RW + r;  // query index in the sequence
            const int qp = P + (i < n ? i : 0);
            const bool ok = j <= qp && j < kend && (window <= 0 || qp - j < window);
            float x = s[r] * scale;
            if (softcap > 0.f) x = softcap * tanhf(x / softcap);
            x = ok ? x : NEG_INF;
            const float mn = fmaxf(m[r], warp_max(x));
            const float alpha = mn == NEG_INF ? 1.f : expf(m[r] - mn);
            pr[r] = ok ? expf(x - mn) : 0.f;
            l[r] = l[r] * alpha + pr[r];
            m[r] = mn;
            if (f16) {
                pr[r] = rh(pr[r]);
                const float ah = rh(alpha);
#pragma unroll
                for (int i2 = 0; i2 < DPL; i2++) acc[r][i2] = rh(acc[r][i2] * ah);
            } else {
#pragma unroll
                for (int i2 = 0; i2 < DPL; i2++) acc[r][i2] *= alpha;
            }
        }
        __syncthreads();
        for (int e = tid; e < BK * HD; e += 128) {  // V: coalesced rows
            const int kk = e / HD, d = e % HD, jj = k0 + kk;
            float x = 0.f;
            if (jj < kend) x = jj < P ? ldf(pv, (long)jj * kvw + (long)g * HD + d) : ldf(vn, (long)(start + jj - P) * kvw + (long)g * HD + d);
            KV[kk * HD + d] = x;
        }
        __syncthreads();
        if (f16) {
            for (int j16 = 0; j16 < BK; j16 += 16) {
                float part[RW][DPL];
#pragma unroll
                for (int r = 0; r < RW; r++)
#pragma unroll
                    for (int i2 = 0; i2 < DPL; i2++) part[r][i2] = 0.f;
                for (int jj = j16; jj < j16 + 16; jj++) {
                    float pj[RW];
#pragma unroll
                    for (int r = 0; r < RW; r++) pj[r] = __shfl_sync(FULL, pr[r], jj);
#pragma unroll
                    for (int i2 = 0; i2 < DPL; i2++) {
                        const float vv = KV[jj * HD + lane + 32 * i2];
#pragma unroll
                        for (int r = 0; r < RW; r++) part[r][i2] += pj[r] * vv;
                    }
                }
#pragma unroll
                for (int r = 0; r < RW; r++)
#pragma unroll
                    for (int i2 = 0; i2 < DPL; i2++) acc[r][i2] = rh(acc[r][i2] + part[r][i2]);
            }
        } else {
            for (int jj = 0; jj < BK; jj++) {
                float pj[RW];
#pragma unroll
                for (int r = 0; r < RW; r++) pj[r] = __shfl_sync(FULL, pr[r], jj);
#pragma unroll
                for (int i2 = 0; i2 < DPL; i2++) {
                    const float vv = KV[jj * HD + lane + 32 * i2];
#pragma unroll
                    for (int r = 0; r < RW; r++) acc[r][i2] += pj[r] * vv;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RW; r++) {
        const int i = q0 + w * RW + r;
        const float z = warp_sum(l[r]);
        if (i >= n) continue;
        const long tok = start + i;
#pragma unroll
        for (int i2 = 0; i2 < DPL; i2++) {
            const int d = lane + 32 * i2;
            float o = acc[r][i2] / z;
            if (gate) {
                const float gv = ldf(gate, tok * gld + (long)h * gs + d);
                o = rnd<T>(o) * (1.f / (1.f + expf(-gv)));
            }
            stf(out, (tok * nh + h) * HD + d, o);
        }
    }
}
#define ATTN_ARGS(T)                                                                                                             \
    const T *q, const T *kn, const T *vn, T *out, const T *gate, int gld, int gs, const u32 *tiles, const u32 *cu, const u32 *plen, \
        const u64 *rk, const u64 *rv, long kvoff, int nh, int nkv, float scale, float softcap, int window, int f16
#define ATTN_CALL(HD) attn_body<HD>(q, kn, vn, out, gate, gld, gs, tiles, cu, plen, rk, rv, kvoff, nh, nkv, scale, softcap, window, f16)
extern "C" __global__ void __launch_bounds__(128) attn64_bf16(ATTN_ARGS(bf16)) { ATTN_CALL(64); }
extern "C" __global__ void __launch_bounds__(128) attn64_f32(ATTN_ARGS(float)) { ATTN_CALL(64); }
extern "C" __global__ void __launch_bounds__(128) attn128_bf16(ATTN_ARGS(bf16)) { ATTN_CALL(128); }
extern "C" __global__ void __launch_bounds__(128) attn128_f32(ATTN_ARGS(float)) { ATTN_CALL(128); }
extern "C" __global__ void __launch_bounds__(128) attn256_bf16(ATTN_ARGS(bf16)) { ATTN_CALL(256); }
extern "C" __global__ void __launch_bounds__(128) attn256_f32(ATTN_ARGS(float)) { ATTN_CALL(256); }
extern "C" __global__ void __launch_bounds__(128) attn512_bf16(ATTN_ARGS(bf16)) { ATTN_CALL(512); }
extern "C" __global__ void __launch_bounds__(128) attn512_f32(ATTN_ARGS(float)) { ATTN_CALL(512); }

// ---------------------------------------------------------------------------------------------------------------
// DeltaNet causal conv1d + SiLU (thread per token and channel), continuing each sequence's past conv state.
// ---------------------------------------------------------------------------------------------------------------
template <typename T>
__device__ void conv_body(const T* p, int ld, const float* w, T* out, long total, int C, int K, const u32* tok2seq, const u32* cu,
                          const u64* rd, int lg) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int c = idx % C;
    const long tok = idx / C;
    const u32 b = tok2seq[tok];
    const long start = cu[b], t = tok - start;
    const T* prev = rd[b] ? (const T*)rd[b] + (long)lg * (K - 1) * C : nullptr;
    float acc = 0.f;
    for (int k = 0; k < K; k++) {
        const long tau = t - (K - 1) + k;
        const float x = tau >= 0 ? ldf(p, (start + tau) * ld + c) : (prev ? ldf(prev, (K - 1 + tau) * C + c) : 0.f);
        acc += w[k * C + c] * x;
    }
    acc = rnd<T>(acc);
    stf(out, idx, acc / (1.f + expf(-acc)));
}
extern "C" __global__ void conv_bf16(const bf16* p, int ld, const float* w, bf16* out, long total, int C, int K, const u32* tok2seq,
                                     const u32* cu, const u64* rd, int lg) {
    conv_body(p, ld, w, out, total, C, K, tok2seq, cu, rd, lg);
}
extern "C" __global__ void conv_f32(const float* p, int ld, const float* w, float* out, long total, int C, int K, const u32* tok2seq,
                                    const u32* cu, const u64* rd, int lg) {
    conv_body(p, ld, w, out, total, C, K, tok2seq, cu, rd, lg);
}

// The last K-1 conv inputs of each listed (state-building) sequence -> its conv cache, bits copied.
template <typename T>
__device__ void tail_body(const T* p, int ld, int C, int K, const u32* cu, const u32* list, long total, const u64* rd, const u64* wr, int lg) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int c = idx % C;
    const int jj = (idx / C) % (K - 1);
    const u32 b = list[idx / ((long)C * (K - 1))];
    const long start = cu[b], n = cu[b + 1] - start, tau = n - (K - 1) + jj;
    const long lay = (long)lg * (K - 1) * C;
    T v;
    if (tau >= 0) v = p[(start + tau) * ld + c];
    else if (rd[b]) v = ((const T*)rd[b])[lay + (K - 1 + tau) * C + c];
    else v = (T)0;
    ((T*)wr[b])[lay + (long)jj * C + c] = v;
}
extern "C" __global__ void conv_tail_bf16(const bf16* p, int ld, int C, int K, const u32* cu, const u32* list, long total, const u64* rd,
                                          const u64* wr, int lg) {
    tail_body(p, ld, C, K, cu, list, total, rd, wr, lg);
}
extern "C" __global__ void conv_tail_f32(const float* p, int ld, int C, int K, const u32* cu, const u32* list, long total, const u64* rd,
                                         const u64* wr, int lg) {
    tail_body(p, ld, C, K, cu, list, total, rd, wr, lg);
}

// ---------------------------------------------------------------------------------------------------------------
// Gated delta rule. One warp per (listed sequence, value head, 8 value columns). Lane = column (lane / 4) x key slice
// (lane % 4); a lane keeps its 32 state entries in registers, so the delta-rule reductions are two shuffles. Gates and
// q/k L2 norms are computed here; the next token's inputs load while the current one is processed.
// ---------------------------------------------------------------------------------------------------------------
template <typename T>
__device__ void gdn_body(const T* __restrict__ qkv, const T* __restrict__ proj, int ld, int a_off, int b_off, const float* a_neg,
                         const float* dt_bias, float* __restrict__ out, const u32* cu, const u32* list, const u64* rd, const u64* wr, int HK,
                         int HV, int lg) {
    const int tiles = DV / 8;
    const int h = blockIdx.x / tiles, tile = blockIdx.x % tiles, lane = threadIdx.x;
    const u32 b = list[blockIdx.y];
    const int col = lane >> 2, ks = lane & 3;
    const int j = tile * 8 + col, kh = h / (HV / HK);
    const int C = 2 * HK * DK + HV * DV;
    const long start = cu[b];
    const int len = cu[b + 1] - start;
    __shared__ float qs[DK];
    __shared__ float kv[DK];
    float S[32];
    const long soff = ((long)lg * HV + h) * DK * DV + j;
    const float* s0 = rd[b] ? (const float*)rd[b] + soff : nullptr;
#pragma unroll
    for (int r = 0; r < 32; r++) S[r] = s0 ? s0[(long)(4 * r + ks) * DV] : 0.f;
    const float qscale = rsqrtf((float)DK), an = a_neg[h], db = dt_bias[h];
    float nq[4], nk[4], nv = 0.f, na = 0.f, nb = 0.f;
    if (len > 0) {
        const T* row = qkv + start * C;
#pragma unroll
        for (int r = 0; r < 4; r++) {
            nq[r] = ldf(row, kh * DK + lane + 32 * r);
            nk[r] = ldf(row, HK * DK + kh * DK + lane + 32 * r);
        }
        nv = ldf(row, 2 * HK * DK + h * DV + j);
        na = ldf(proj, start * ld + a_off + h);
        nb = ldf(proj, start * ld + b_off + h);
    }
    for (int t = 0; t < len; t++) {
        float q[4], k[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            q[r] = nq[r];
            k[r] = nk[r];
        }
        const float v = nv, av = na, bv = nb;
        if (t + 1 < len) {
            const long tok = start + t + 1;
            const T* row = qkv + tok * C;
#pragma unroll
            for (int r = 0; r < 4; r++) {
                nq[r] = ldf(row, kh * DK + lane + 32 * r);
                nk[r] = ldf(row, HK * DK + kh * DK + lane + 32 * r);
            }
            nv = ldf(row, 2 * HK * DK + h * DV + j);
            na = ldf(proj, tok * ld + a_off + h);
            nb = ldf(proj, tok * ld + b_off + h);
        }
        const float x = av + db;
        const float sp = fmaxf(x, 0.f) + logf(1.f + expf(-fabsf(x)));
        const float decay = expf(sp * an);
        const float beta = 1.f / (1.f + expf(-bv));
        float sq = 0.f, sk = 0.f;
#pragma unroll
        for (int r = 0; r < 4; r++) {
            sq += q[r] * q[r];
            sk += k[r] * k[r];
        }
        sq = warp_sum(sq);
        sk = warp_sum(sk);
        const float iq = rsqrtf(sq + 1e-6f) * qscale, ik = rsqrtf(sk + 1e-6f);
        __syncwarp();
#pragma unroll
        for (int r = 0; r < 4; r++) {
            qs[lane + 32 * r] = q[r] * iq;
            kv[lane + 32 * r] = k[r] * ik;
        }
        __syncwarp();
        float mem = 0.f;
#pragma unroll
        for (int r = 0; r < 32; r++) {
            S[r] *= decay;
            mem += S[r] * kv[4 * r + ks];
        }
        mem += __shfl_xor_sync(FULL, mem, 1);
        mem += __shfl_xor_sync(FULL, mem, 2);
        const float delta = (v - mem) * beta;
        float acc = 0.f;
#pragma unroll
        for (int r = 0; r < 32; r++) {
            S[r] += kv[4 * r + ks] * delta;
            acc += S[r] * qs[4 * r + ks];
        }
        acc += __shfl_xor_sync(FULL, acc, 1);
        acc += __shfl_xor_sync(FULL, acc, 2);
        if (ks == 0) out[((start + t) * HV + h) * DV + j] = acc;
    }
    if (wr[b]) {
        float* st = (float*)wr[b] + soff;
#pragma unroll
        for (int r = 0; r < 32; r++) st[(long)(4 * r + ks) * DV] = S[r];
    }
}
#define GDN_ARGS(T)                                                                                                                 \
    const T *qkv, const T *proj, int ld, int a_off, int b_off, const float *a_neg, const float *dt_bias, float *out, const u32 *cu, \
        const u32 *list, const u64 *rd, const u64 *wr, int HK, int HV, int lg
#define GDN_CALL gdn_body(qkv, proj, ld, a_off, b_off, a_neg, dt_bias, out, cu, list, rd, wr, HK, HV, lg)
extern "C" __global__ void gdn_bf16(GDN_ARGS(bf16)) { GDN_CALL; }
extern "C" __global__ void gdn_f32(GDN_ARGS(float)) { GDN_CALL; }

// ---------------------------------------------------------------------------------------------------------------
// Gated delta rule, bf16 fast path (gdn_prep_bf16 + gdn_fast_bf16): 2.24x gdn_bf16 on a 4090 at Kev-4B's shapes
// (csrc/gdn_bench.cu, 20 sequences x 1550 tokens: 12.17 -> 5.42 ms) with the same f32 recurrence. Three changes:
// the q/k norms and the gates are computed once per token (gdn_bf16 recomputed them in each of a head's 16 warps);
// the 4 warps of a block share one cp.async-staged copy of q, k, v and the gates, GF_CH tokens at a time, so a head's
// inputs are read 4x less and the block syncs once per chunk; and a lane owns 32 contiguous keys (gdn_bf16
// interleaved them) read as float4 from a padded shared layout, a quarter of the shared-memory loads. Lazy decay
// (state = D * S) measured slower: its per-step division outweighs the 32 multiplies it saves.
// ---------------------------------------------------------------------------------------------------------------
#define GF_W 4    // warps per block = value tiles of 8 columns
#define GF_CH 6   // tokens per staged chunk
#define GF_SL 36  // floats per key slice in shared memory: 32 + 4 of padding puts the 4 slices on different banks

// One warp per (token, key head): qn = q / |q| / sqrt(DK), kn = k / |k| (f32, as gdn_bf16 computes them); one thread
// per (token, value head): gate = (decay, beta).
extern "C" __global__ void gdn_prep_bf16(const bf16* qkv, const bf16* proj, int ld, int a_off, int b_off, const float* a_neg,
                                         const float* dt_bias, float* qn, float* kn, float* gate, long T, int HK, int HV) {
    const int C = 2 * HK * DK + HV * DV;
    const long w = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (w < T * HK) {
        const long tok = w / HK;
        const int kh = w % HK;
        const bf16* row = qkv + tok * C;
        float q[4], k[4], sq = 0.f, sk = 0.f;
#pragma unroll
        for (int r = 0; r < 4; r++) {
            q[r] = bf2f(row[kh * DK + lane + 32 * r]);
            k[r] = bf2f(row[HK * DK + kh * DK + lane + 32 * r]);
            sq += q[r] * q[r];
            sk += k[r] * k[r];
        }
        const float iq = rsqrtf(warp_sum(sq) + 1e-6f) * rsqrtf((float)DK), ik = rsqrtf(warp_sum(sk) + 1e-6f);
#pragma unroll
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

__device__ __forceinline__ void cpa16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
}
__device__ __forceinline__ void cpa8(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
}

// grid (HV * DV / 8 / GF_W, listed sequences), block 32 * GF_W. Warp w owns value columns [8 tile, 8 tile + 8) of
// head h; lane = column (lane / 4) x key slice (lane % 4) holding keys 32 ks .. 32 ks + 31. rd / wr as gdn_body.
// v_in is the first value column of row 0, vld its row stride (conv_bf16's q|k|v rows, or conv_prep_bf16's v rows).
extern "C" __global__ void __launch_bounds__(32 * GF_W) gdn_fast_bf16(const bf16* v_in, int vld, const float* qn, const float* kn,
        const float* gate, float* out, const u32* cu, const u32* list, const u64* rd, const u64* wr, int HK, int HV, int lg) {
    constexpr int TPH = DV / 8 / GF_W;
    const int h = blockIdx.x / TPH, w = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
    const int tile = (blockIdx.x % TPH) * GF_W + w, col = lane >> 2, ks = lane & 3, j = tile * 8 + col, kh = h / (HV / HK);
    const int jb = (blockIdx.x % TPH) * GF_W * 8;
    const u32 b = list[blockIdx.y];
    const long start = cu[b];
    const int len = cu[b + 1] - start;
    __shared__ __align__(16) float qk[2][GF_CH][2][4 * GF_SL];
    __shared__ __align__(16) bf16 vv[2][GF_CH][GF_W * 8];
    __shared__ __align__(16) float gg[2][GF_CH][2];
    float S[32];
    const long soff = ((long)lg * HV + h) * DK * DV + j;
    const float* s0 = rd[b] ? (const float*)rd[b] + soff : nullptr;
#pragma unroll
    for (int r = 0; r < 32; r++) S[r] = s0 ? s0[(long)(32 * ks + r) * DV] : 0.f;
    auto stage = [&](int c, int buf) {
        const int t0 = c * GF_CH, n = min(GF_CH, len - t0);
        for (int e = tid; e < n * 64; e += 32 * GF_W) {  // q then k: 2 x 32 chunks of 16 bytes per token
            const int i = e >> 6, part = e & 63, which = part >> 5, c4 = part & 31;
            cpa16(&qk[buf][i][which][(c4 >> 3) * GF_SL + (c4 & 7) * 4], (which ? kn : qn) + ((start + t0 + i) * HK + kh) * DK + 4 * c4);
        }
        for (int e = tid; e < n * GF_W; e += 32 * GF_W) {  // v: the block's 8 GF_W columns, 8 per chunk
            const int i = e / GF_W, c8 = e % GF_W;
            cpa16(&vv[buf][i][8 * c8], v_in + (start + t0 + i) * vld + h * DV + jb + 8 * c8);
        }
        for (int i = tid; i < n; i += 32 * GF_W) cpa8(&gg[buf][i][0], gate + 2 * ((start + t0 + i) * HV + h));
        asm volatile("cp.async.commit_group;");
    };
    const int chunks = (len + GF_CH - 1) / GF_CH;
    if (chunks > 0) stage(0, 0);
    for (int c = 0; c < chunks; c++) {
        const int buf = c & 1;
        if (c + 1 < chunks) {
            stage(c + 1, buf ^ 1);
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        const int n = min(GF_CH, len - c * GF_CH);
        for (int i = 0; i < n; i++) {
            const float4* k4 = reinterpret_cast<const float4*>(&qk[buf][i][1][ks * GF_SL]);
            const float4* q4 = reinterpret_cast<const float4*>(&qk[buf][i][0][ks * GF_SL]);
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
            if (ks == 0) out[((start + c * GF_CH + i) * HV + h) * DV + j] = acc;
        }
        __syncthreads();  // this buffer is restaged two chunks on
    }
    if (wr[b]) {
        float* st = (float*)wr[b] + soff;
#pragma unroll
        for (int r = 0; r < 32; r++) st[(long)(32 * ks + r) * DV] = S[r];
    }
}


// DeltaNet conv1d + SiLU fused with gdn_prep_bf16: one block of 512 per token writes what gdn_fast_bf16 reads (qn, kn
// f32, v bf16 [T, HV DV], gates) instead of the bf16 q|k|v row that conv_bf16 wrote and gdn_prep_bf16 read back. Each
// value is the one the unfused pair computes: the conv output rounded to bf16 first.
extern "C" __global__ void __launch_bounds__(512) conv_prep_bf16(const bf16* p, int ld, const float* w, int K, const u32* tok2seq, const u32* cu,
        const u64* rd, int lg, int a_off, int b_off, const float* a_neg, const float* dt_bias, float* qn, float* kn, bf16* vo, float* gate,
        int HK, int HV) {
    const long tok = blockIdx.x;
    const u32 b = tok2seq[tok];
    const long start = cu[b], t = tok - start;
    const int C = 2 * HK * DK + HV * DV;
    const bf16* prev = rd[b] ? (const bf16*)rd[b] + (long)lg * (K - 1) * C : nullptr;
    auto convc = [&](int c) {
        float acc = 0.f;
        for (int k = 0; k < K; k++) {
            const long tau = t - (K - 1) + k;
            const float x = tau >= 0 ? bf2f(p[(start + tau) * ld + c]) : (prev ? bf2f(prev[(K - 1 + tau) * C + c]) : 0.f);
            acc += w[k * C + c] * x;
        }
        acc = rnd<bf16>(acc);
        return bf2f(f2bf(acc / (1.f + expf(-acc))));
    };
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int kh = warp; kh < HK; kh += 16) {
        float q[4], k[4], sq = 0.f, sk = 0.f;
#pragma unroll
        for (int r = 0; r < 4; r++) {
            q[r] = convc(kh * DK + lane + 32 * r);
            k[r] = convc(HK * DK + kh * DK + lane + 32 * r);
            sq += q[r] * q[r];
            sk += k[r] * k[r];
        }
        const float iq = rsqrtf(warp_sum(sq) + 1e-6f) * rsqrtf((float)DK), ik = rsqrtf(warp_sum(sk) + 1e-6f);
        const long o = (tok * HK + kh) * DK;
#pragma unroll
        for (int r = 0; r < 4; r++) {
            qn[o + lane + 32 * r] = q[r] * iq;
            kn[o + lane + 32 * r] = k[r] * ik;
        }
    }
    for (int c = threadIdx.x; c < HV * DV; c += 512) vo[tok * HV * DV + c] = f2bf(convc(2 * HK * DK + c));
    for (int h = threadIdx.x; h < HV; h += 512) {
        const float x = bf2f(p[tok * ld + a_off + h]) + dt_bias[h];
        const float sp = fmaxf(x, 0.f) + logf(1.f + expf(-fabsf(x)));
        gate[2 * (tok * HV + h)] = expf(sp * a_neg[h]);
        gate[2 * (tok * HV + h) + 1] = 1.f / (1.f + expf(-bf2f(p[tok * ld + b_off + h])));
    }
}

// ---------------------------------------------------------------------------------------------------------------
// DeltaNet output norm: RMSNorm(o) * w * silu(z), one warp per (token, value head).
// ---------------------------------------------------------------------------------------------------------------
template <typename T>
__device__ void gnorm_body(const float* o, const T* p, int ld, int z_off, const float* w, T* out, long warps, int HV, float eps) {
    const long gw = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (gw >= warps) return;
    const long tok = gw / HV;
    const int h = gw % HV;
    const float* x = o + gw * DV;
    float v[4], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = x[lane + 32 * i];
        ss += v[i] * v[i];
    }
    const float r = rsqrtf(warp_sum(ss) / DV + eps);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int d = lane + 32 * i;
        const float z = ldf(p, tok * ld + z_off + (long)h * DV + d);
        stf(out, gw * DV + d, v[i] * r * w[d] * (z / (1.f + expf(-z))));
    }
}
extern "C" __global__ void gated_norm_bf16(const float* o, const bf16* p, int ld, int z_off, const float* w, bf16* out, long warps, int HV,
                                           float eps) {
    gnorm_body(o, p, ld, z_off, w, out, warps, HV, eps);
}
extern "C" __global__ void gated_norm_f32(const float* o, const float* p, int ld, int z_off, const float* w, float* out, long warps, int HV,
                                          float eps) {
    gnorm_body(o, p, ld, z_off, w, out, warps, HV, eps);
}

// ---------------------------------------------------------------------------------------------------------------
// Tensor-core attention (bf16 in, f32 accumulate, P rounded to bf16 for P.V as flash-attention does). One block of
// 4 warps per (tile of 64 queries of one sequence, query head); warp w owns queries 16w..16w+15. Per 32-key tile:
// S = Q K^T with mma.m16n8k16, online softmax in registers, O += P V. Same keys, mask, window and gate as attn_body.
// Dynamic shared memory: Q [64][HD+8], K [32][HD+8], V [32][HD+8] bf16 (rows padded 16 bytes against bank
// conflicts in ldmatrix).
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void mma16816(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm4(unsigned* r, const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void ldsm4t(unsigned* r, const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) { return (unsigned)f2bf(lo) | ((unsigned)f2bf(hi) << 16); }

template <int HD>
__device__ void attn_mma_body(const bf16* q, const bf16* kn, const bf16* vn, bf16* out, const bf16* gate, int gld, int gs, const u32* tiles,
                              const u32* cu, const u32* plen, const u64* rk, const u64* rv, long kvoff, int nh, int nkv, float scale,
                              float softcap, int window) {
    constexpr int BQ = 64, BK = 32, LD = HD + 8, NB = HD / 8;
    extern __shared__ __align__(16) unsigned char smraw[];
    bf16* Qs = (bf16*)smraw;
    bf16* Ks = Qs + BQ * LD;
    bf16* Vs = Ks + BK * LD;
    const int tile = blockIdx.x, h = blockIdx.y, g = h / (nh / nkv);
    const int b = tiles[2 * tile], q0 = tiles[2 * tile + 1];
    const int start = cu[b], n = cu[b + 1] - start, P = plen[b];
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const long kvw = (long)nkv * HD;
    const bf16* pk = P ? (const bf16*)rk[b] + (long)P * kvoff : nullptr;
    const bf16* pv = P ? (const bf16*)rv[b] + (long)P * kvoff : nullptr;
    constexpr int CH = HD / 8;  // 16-byte chunks per row
    for (int e = tid; e < BQ * CH; e += 128) {
        const int r = e / CH, c = e % CH;
        uint4 x = make_uint4(0, 0, 0, 0);
        if (q0 + r < n) x = *(const uint4*)(q + ((long)(start + q0 + r) * nh + h) * HD + 8 * c);
        *(uint4*)(Qs + r * LD + 8 * c) = x;
    }
    float o[NB][4];
#pragma unroll
    for (int i = 0; i < NB; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.f, l1 = 0.f;
    const int qa = q0 + 16 * w + gid, qb = qa + 8;  // this thread's two query rows
    const int pa = P + min(qa, n - 1), pb = P + min(qb, n - 1);
    const int qlast = min(q0 + BQ, n) - 1;
    const int kend = P + qlast + 1;
    int kbeg = 0;
    if (window > 0) {
        kbeg = max(0, P + q0 - window + 1);
        kbeg -= kbeg % BK;
    }
    const bool warp_live = q0 + 16 * w < n;
    for (int k0 = kbeg; k0 < kend; k0 += BK) {
        __syncthreads();
        for (int e = tid; e < 2 * BK * CH; e += 128) {
            const int which = e / (BK * CH), r = (e / CH) % BK, c = e % CH, j = k0 + r;
            uint4 x = make_uint4(0, 0, 0, 0);
            if (j < kend) {
                const bf16* src = which == 0 ? (j < P ? pk + (long)j * kvw : kn + (long)(start + j - P) * kvw)
                                             : (j < P ? pv + (long)j * kvw : vn + (long)(start + j - P) * kvw);
                x = *(const uint4*)(src + (long)g * HD + 8 * c);
            }
            *(uint4*)((which == 0 ? Ks : Vs) + r * LD + 8 * c) = x;
        }
        __syncthreads();
        if (!warp_live) continue;
        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
        for (int kc = 0; kc < HD / 16; kc++) {
            unsigned a[4];
            ldsm4(a, Qs + (16 * w + (lane & 15)) * LD + 16 * kc + 8 * (lane >> 4));
#pragma unroll
            for (int nb = 0; nb < 4; nb += 2) {
                unsigned bb[4];
                ldsm4(bb, Ks + (8 * nb + (lane & 7) + 8 * (lane >> 4)) * LD + 16 * kc + 8 * ((lane >> 3) & 1));
                mma16816(s[nb], a, bb[0], bb[1]);
                mma16816(s[nb + 1], a, bb[2], bb[3]);
            }
        }
        float mx0 = NEG_INF, mx1 = NEG_INF;
#pragma unroll
        for (int nb = 0; nb < 4; nb++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int j = k0 + 8 * nb + 2 * tig + (e & 1);
                const int qp = e < 2 ? pa : pb;
                const bool ok = j <= qp && j < kend && (window <= 0 || qp - j < window);
                float x = s[nb][e] * scale;
                if (softcap > 0.f) x = softcap * tanhf(x / softcap);
                x = ok ? x : NEG_INF;
                s[nb][e] = x;
                if (e < 2) mx0 = fmaxf(mx0, x);
                else mx1 = fmaxf(mx1, x);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 2));
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float al0 = n0 == NEG_INF ? 1.f : expf(m0 - n0), al1 = n1 == NEG_INF ? 1.f : expf(m1 - n1);
        m0 = n0;
        m1 = n1;
        l0 *= al0;
        l1 *= al1;
#pragma unroll
        for (int i = 0; i < NB; i++) {
            o[i][0] *= al0;
            o[i][1] *= al0;
            o[i][2] *= al1;
            o[i][3] *= al1;
        }
        unsigned pa_[2][4];
#pragma unroll
        for (int nb = 0; nb < 4; nb++) {
            float p[4];
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float mm = e < 2 ? n0 : n1;
                p[e] = s[nb][e] == NEG_INF ? 0.f : expf(s[nb][e] - mm);
            }
            l0 += p[0] + p[1];
            l1 += p[2] + p[3];
            const int kc = nb >> 1, hi = nb & 1;
            pa_[kc][2 * hi] = pack_bf16(p[0], p[1]);
            pa_[kc][2 * hi + 1] = pack_bf16(p[2], p[3]);
        }
#pragma unroll
        for (int kc = 0; kc < 2; kc++) {
#pragma unroll
            for (int nb = 0; nb < NB; nb += 2) {
                unsigned bb[4];
                ldsm4t(bb, Vs + (16 * kc + (lane & 15)) * LD + 8 * nb + 8 * (lane >> 4));
                mma16816(o[nb], pa_[kc], bb[0], bb[1]);
                mma16816(o[nb + 1], pa_[kc], bb[2], bb[3]);
            }
        }
    }
    if (!warp_live) return;
    l0 += __shfl_xor_sync(FULL, l0, 1);
    l0 += __shfl_xor_sync(FULL, l0, 2);
    l1 += __shfl_xor_sync(FULL, l1, 1);
    l1 += __shfl_xor_sync(FULL, l1, 2);
#pragma unroll
    for (int half = 0; half < 2; half++) {
        const int qi = half ? qb : qa;
        if (qi >= n) continue;
        const float z = half ? l1 : l0;
        const long tok = start + qi;
#pragma unroll
        for (int nb = 0; nb < NB; nb++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int d = 8 * nb + 2 * tig + e;
                float v = o[nb][2 * half + e] / z;
                if (gate) {
                    const float gv = bf2f(gate[tok * gld + (long)h * gs + d]);
                    v = rnd<bf16>(v) * (1.f / (1.f + expf(-gv)));
                }
                out[(tok * nh + h) * HD + d] = f2bf(v);
            }
        }
    }
}
#define ATTN_MMA_ARGS                                                                                                                     \
    const bf16 *q, const bf16 *kn, const bf16 *vn, bf16 *out, const bf16 *gate, int gld, int gs, const u32 *tiles, const u32 *cu, \
        const u32 *plen, const u64 *rk, const u64 *rv, long kvoff, int nh, int nkv, float scale, float softcap, int window
#define ATTN_MMA_CALL(HD) attn_mma_body<HD>(q, kn, vn, out, gate, gld, gs, tiles, cu, plen, rk, rv, kvoff, nh, nkv, scale, softcap, window)
extern "C" __global__ void __launch_bounds__(128) attn_mma64(ATTN_MMA_ARGS) { ATTN_MMA_CALL(64); }
extern "C" __global__ void __launch_bounds__(128) attn_mma128(ATTN_MMA_ARGS) { ATTN_MMA_CALL(128); }
extern "C" __global__ void __launch_bounds__(128) attn_mma256(ATTN_MMA_ARGS) { ATTN_MMA_CALL(256); }


// ---------------------------------------------------------------------------------------------------------------
// Tensor-core attention in llama.cpp's precision (fattn-mma-f16.cuh on Turing/Ampere): Q (times the scale, in f16),
// K and V in f16, KQ accumulated in f32, P in f16, V.P accumulated in f16 (mma f16.f16.f16.f16) and rescaled in f16,
// the row sums in f32. f32 in and out (the values of q, k, v are already f16-exact). Layout as attn_mma_body, with one
// K-then-V tile buffer so head dim 512 fits.
// ---------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void mma16816h(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma16816hh(unsigned* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(c[0]), "+r"(c[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pack_f16(float lo, float hi) {
    unsigned r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ unsigned hmul2(unsigned a, unsigned b) {
    unsigned r;
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
__device__ __forceinline__ float2 unpack_f16(unsigned v) {
    unsigned short lo, hi;
    asm("mov.b32 {%0, %1}, %2;" : "=h"(lo), "=h"(hi) : "r"(v));
    float a, b;
    asm("cvt.f32.f16 %0, %1;" : "=f"(a) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(b) : "h"(hi));
    return make_float2(a, b);
}
// 8 floats -> 8 f16 (each rounded, then times the f16 scale and rounded again, as ggml-cuda loads Q)
__device__ __forceinline__ uint4 ld8h(const float* p, float hs) {
    const float4 a = *(const float4*)p, c = *(const float4*)(p + 4);
    uint4 r;
    r.x = pack_f16(rh(rh(a.x) * hs), rh(rh(a.y) * hs));
    r.y = pack_f16(rh(rh(a.z) * hs), rh(rh(a.w) * hs));
    r.z = pack_f16(rh(rh(c.x) * hs), rh(rh(c.y) * hs));
    r.w = pack_f16(rh(rh(c.z) * hs), rh(rh(c.w) * hs));
    return r;
}
template <int HD>
__device__ void attn_f16_body(const float* q, const float* kn, const float* vn, float* out, const u32* tiles,
                              const u32* cu, const u32* plen, const u64* rk, const u64* rv, long kvoff, int nh, int nkv, float scale,
                              float softcap, int window) {
    constexpr int BQ = 64, BK = 32, LD = HD + 8, NB = HD / 8;
    extern __shared__ __align__(16) unsigned char smraw[];
    bf16* Qs = (bf16*)smraw;  // f16 bits
    bf16* Ks = Qs + BQ * LD;  // K, then V, of one tile (one buffer: head dim 512 fits 99 KB)
    bf16* Vs = Ks;
    const float hs = rh(scale);
    const int tile = blockIdx.x, h = blockIdx.y, g = h / (nh / nkv);
    const int b = tiles[2 * tile], q0 = tiles[2 * tile + 1];
    const int start = cu[b], n = cu[b + 1] - start, P = plen[b];
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const long kvw = (long)nkv * HD;
    const float* pk = P ? (const float*)rk[b] + (long)P * kvoff : nullptr;
    const float* pv = P ? (const float*)rv[b] + (long)P * kvoff : nullptr;
    constexpr int CH = HD / 8;  // 16-byte chunks per row
    for (int e = tid; e < BQ * CH; e += 128) {
        const int r = e / CH, c = e % CH;
        uint4 x = make_uint4(0, 0, 0, 0);
        if (q0 + r < n) x = ld8h(q + ((long)(start + q0 + r) * nh + h) * HD + 8 * c, hs);
        *(uint4*)(Qs + r * LD + 8 * c) = x;
    }
    unsigned o[NB][2];  // half2 accumulators (rows gid, gid + 8), as ggml-cuda's T_C_VKQ on Turing/Ampere
#pragma unroll
    for (int i = 0; i < NB; i++) o[i][0] = o[i][1] = 0u;
    float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.f, l1 = 0.f;
    const int qa = q0 + 16 * w + gid, qb = qa + 8;  // this thread's two query rows
    const int pa = P + min(qa, n - 1), pb = P + min(qb, n - 1);
    const int qlast = min(q0 + BQ, n) - 1;
    const int kend = P + qlast + 1;
    int kbeg = 0;
    if (window > 0) {
        kbeg = max(0, P + q0 - window + 1);
        kbeg -= kbeg % BK;
    }
    const bool warp_live = q0 + 16 * w < n;
    for (int k0 = kbeg; k0 < kend; k0 += BK) {
        __syncthreads();
        for (int e = tid; e < BK * CH; e += 128) {
            const int r = e / CH, c = e % CH, j = k0 + r;
            uint4 x = make_uint4(0, 0, 0, 0);
            if (j < kend) x = ld8h((j < P ? pk + (long)j * kvw : kn + (long)(start + j - P) * kvw) + (long)g * HD + 8 * c, 1.f);
            *(uint4*)(Ks + r * LD + 8 * c) = x;
        }
        __syncthreads();
        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
        for (int kc = 0; kc < HD / 16; kc++) {
            unsigned a[4];
            ldsm4(a, Qs + (16 * w + (lane & 15)) * LD + 16 * kc + 8 * (lane >> 4));
#pragma unroll
            for (int nb = 0; nb < 4; nb += 2) {
                unsigned bb[4];
                ldsm4(bb, Ks + (8 * nb + (lane & 7) + 8 * (lane >> 4)) * LD + 16 * kc + 8 * ((lane >> 3) & 1));
                mma16816h(s[nb], a, bb[0], bb[1]);
                mma16816h(s[nb + 1], a, bb[2], bb[3]);
            }
        }
        __syncthreads();
        for (int e = tid; e < BK * CH; e += 128) {
            const int r = e / CH, c = e % CH, j = k0 + r;
            uint4 x = make_uint4(0, 0, 0, 0);
            if (j < kend) x = ld8h((j < P ? pv + (long)j * kvw : vn + (long)(start + j - P) * kvw) + (long)g * HD + 8 * c, 1.f);
            *(uint4*)(Vs + r * LD + 8 * c) = x;
        }
        __syncthreads();
        if (!warp_live) continue;
        float mx0 = NEG_INF, mx1 = NEG_INF;
#pragma unroll
        for (int nb = 0; nb < 4; nb++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int j = k0 + 8 * nb + 2 * tig + (e & 1);
                const int qp = e < 2 ? pa : pb;
                const bool ok = j <= qp && j < kend && (window <= 0 || qp - j < window);
                float x = s[nb][e];  // the scale is folded into Q, in f16, as ggml-cuda does
                if (softcap > 0.f) x = softcap * tanhf(x / softcap);
                x = ok ? x : NEG_INF;
                s[nb][e] = x;
                if (e < 2) mx0 = fmaxf(mx0, x);
                else mx1 = fmaxf(mx1, x);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 2));
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float al0 = n0 == NEG_INF ? 1.f : expf(m0 - n0), al1 = n1 == NEG_INF ? 1.f : expf(m1 - n1);
        m0 = n0;
        m1 = n1;
        l0 *= al0;
        l1 *= al1;
        const unsigned sc0 = pack_f16(al0, al0), sc1 = pack_f16(al1, al1);
#pragma unroll
        for (int i = 0; i < NB; i++) {
            o[i][0] = hmul2(o[i][0], sc0);
            o[i][1] = hmul2(o[i][1], sc1);
        }
        unsigned pa_[2][4];
#pragma unroll
        for (int nb = 0; nb < 4; nb++) {
            float p[4];
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float mm = e < 2 ? n0 : n1;
                p[e] = s[nb][e] == NEG_INF ? 0.f : expf(s[nb][e] - mm);
            }
            l0 += p[0] + p[1];
            l1 += p[2] + p[3];
            const int kc = nb >> 1, hi = nb & 1;
            pa_[kc][2 * hi] = pack_f16(p[0], p[1]);
            pa_[kc][2 * hi + 1] = pack_f16(p[2], p[3]);
        }
#pragma unroll
        for (int kc = 0; kc < 2; kc++) {
#pragma unroll
            for (int nb = 0; nb < NB; nb += 2) {
                unsigned bb[4];
                ldsm4t(bb, Vs + (16 * kc + (lane & 15)) * LD + 8 * nb + 8 * (lane >> 4));
                mma16816hh(o[nb], pa_[kc], bb[0], bb[1]);
                mma16816hh(o[nb + 1], pa_[kc], bb[2], bb[3]);
            }
        }
    }
    if (!warp_live) return;
    l0 += __shfl_xor_sync(FULL, l0, 1);
    l0 += __shfl_xor_sync(FULL, l0, 2);
    l1 += __shfl_xor_sync(FULL, l1, 1);
    l1 += __shfl_xor_sync(FULL, l1, 2);
#pragma unroll
    for (int half = 0; half < 2; half++) {
        const int qi = half ? qb : qa;
        if (qi >= n) continue;
        const float z = half ? l1 : l0;
        const long tok = start + qi;
#pragma unroll
        for (int nb = 0; nb < NB; nb++) {
            const float2 v = unpack_f16(o[nb][half]);
            const int d = 8 * nb + 2 * tig;
            out[(tok * nh + h) * HD + d] = v.x / z;
            out[(tok * nh + h) * HD + d + 1] = v.y / z;
        }
    }
}
#define ATTN_F16_ARGS                                                                                                                   \
    const float *q, const float *kn, const float *vn, float *out, const u32 *tiles, const u32 *cu, const u32 *plen, const u64 *rk, \
        const u64 *rv, long kvoff, int nh, int nkv, float scale, float softcap, int window
#define ATTN_F16_CALL(HD) attn_f16_body<HD>(q, kn, vn, out, tiles, cu, plen, rk, rv, kvoff, nh, nkv, scale, softcap, window)
extern "C" __global__ void __launch_bounds__(128) attn_f16_64(ATTN_F16_ARGS) { ATTN_F16_CALL(64); }
extern "C" __global__ void __launch_bounds__(128) attn_f16_128(ATTN_F16_ARGS) { ATTN_F16_CALL(128); }
extern "C" __global__ void __launch_bounds__(128) attn_f16_256(ATTN_F16_ARGS) { ATTN_F16_CALL(256); }
extern "C" __global__ void __launch_bounds__(128) attn_f16_512(ATTN_F16_ARGS) { ATTN_F16_CALL(512); }

// ---------------------------------------------------------------------------------------------------------------
// GGUF Q8_0 weights, split at load into int8 values qs [rows, cols] and f32 scales d [rows, cols/32] (w = d * q).
// ---------------------------------------------------------------------------------------------------------------
template <typename T>
__device__ void dq8_body(const signed char* qs, const float* d, T* out, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    stf(out, i, d[i / 32] * (float)qs[i]);
}
extern "C" __global__ void dequant_q8_bf16(const signed char* qs, const float* d, bf16* out, long n) { dq8_body(qs, d, out, n); }
extern "C" __global__ void dequant_q8_f32(const signed char* qs, const float* d, float* out, long n) { dq8_body(qs, d, out, n); }

// Rows `ids` of a Q8_0 matrix [*, cols], dequantized (an embedding lookup, label rows).
template <typename T>
__device__ void gq8_body(const signed char* qs, const float* d, const u32* ids, T* out, long n, int cols) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long src = (long)ids[i / cols] * cols + i % cols;
    stf(out, i, d[src / 32] * (float)qs[src]);
}
extern "C" __global__ void gather_q8_bf16(const signed char* qs, const float* d, const u32* ids, bf16* out, long n, int cols) {
    gq8_body(qs, d, ids, out, n, cols);
}
extern "C" __global__ void gather_q8_f32(const signed char* qs, const float* d, const u32* ids, float* out, long n, int cols) {
    gq8_body(qs, d, ids, out, n, cols);
}

// Activations -> q8_1 blocks of 32, one warp per block (ggml-cuda quantize.cu). MMQ (more than 8 rows):
// d_inv = 127/amax, q = round(x * d_inv), d = 1/d_inv kept in f32. MMVQ (up to 8 rows): d = amax/127,
// q = round(x / d), d kept in f16. round() is half away from zero; an all-zero block has q = 0, d = 0.
extern "C" __global__ void quant_q81(const float* x, signed char* xq, float* xd, long nblk, int small) {
    const long wid = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= nblk) return;
    const float v = x[wid * 32 + lane];
    const float amax = warp_max(fabsf(v));
    signed char q = 0;
    float d = 0.f;
    if (amax != 0.f) {
        if (small) {
            d = amax / 127.f;
            q = (signed char)roundf(v / d);
            d = rh(d);
        } else {
            const float dinv = 127.f / amax;
            q = (signed char)roundf(v * dinv);
            d = 1.f / dinv;
        }
    }
    xq[wid * 32 + lane] = q;
    if (lane == 0) xd[wid] = d;
}

// out [M, N] f32 = sum over 32-blocks kb of float(sum_k xq[m,k] wq[n,k]) * dw[n,kb] * dx[m,kb], kb ascending (as
// ggml-cuda MMQ for Q8_0 x q8_1, mmq-vec-dot.cuh). int8 tensor cores (mma.sync m16n8k32 s8), 128 threads per 64x64
// output tile, warp w owns rows 32*(w/2).. and columns 32*(w%2).. as 2 x 4 m16n8 tiles; K staged 4 blocks at a time.
extern "C" __global__ void __launch_bounds__(128) gemm_q8(const int* xq, const float* xd, const int* wq, const float* wd, float* out,
                                                          int M, int N, int K) {
    constexpr int BM = 64, BN = 64, KB = 4, LDW = 36;  // row stride in 32-bit words (4 blocks of 8 words + pad)
    __shared__ int As[BM * LDW];
    __shared__ int Bs[BN * LDW];
    __shared__ float dA[BM * KB];
    __shared__ float dB[BN * KB];
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
    const int wm = (w >> 1) * 32, wn = (w & 1) * 32;
    const int nkb = K / 32, kw = K / 4;
    float acc[2][4][4];
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int b = 0; b < 4; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;
    for (int kb0 = 0; kb0 < nkb; kb0 += KB) {
        __syncthreads();
        for (int e = tid; e < BM * 32; e += 128) {
            const int r = e >> 5, c = e & 31, kb = kb0 + (c >> 3);
            As[r * LDW + c] = (m0 + r < M && kb < nkb) ? xq[(long)(m0 + r) * kw + kb0 * 8 + c] : 0;
            Bs[r * LDW + c] = (n0 + r < N && kb < nkb) ? wq[(long)(n0 + r) * kw + kb0 * 8 + c] : 0;
        }
        for (int e = tid; e < BM * KB; e += 128) {
            const int r = e / KB, j = e % KB, kb = kb0 + j;
            dA[e] = (m0 + r < M && kb < nkb) ? xd[(long)(m0 + r) * nkb + kb] : 0.f;
            dB[e] = (n0 + r < N && kb < nkb) ? wd[(long)(n0 + r) * nkb + kb] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < KB; j++) {
            if (kb0 + j >= nkb) break;
            int a[2][4], b[4][2];
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                const int r0 = wm + mi * 16 + g;
                a[mi][0] = As[r0 * LDW + j * 8 + tig];
                a[mi][1] = As[(r0 + 8) * LDW + j * 8 + tig];
                a[mi][2] = As[r0 * LDW + j * 8 + 4 + tig];
                a[mi][3] = As[(r0 + 8) * LDW + j * 8 + 4 + tig];
            }
#pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                const int nn = wn + ni * 8 + g;
                b[ni][0] = Bs[nn * LDW + j * 8 + tig];
                b[ni][1] = Bs[nn * LDW + j * 8 + 4 + tig];
            }
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
#pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    int c[4] = {0, 0, 0, 0};
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                                 : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[ni][0]), "r"(b[ni][1]));
                    const int r0 = wm + mi * 16 + g, c0 = wn + ni * 8 + 2 * tig;
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int rr = r0 + (e >> 1) * 8, cc = c0 + (e & 1);
                        acc[mi][ni][e] += (float)c[e] * dB[cc * KB + j] * dA[rr * KB + j];
                    }
                }
            }
        }
    }
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int ni = 0; ni < 4; ni++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int rr = m0 + wm + mi * 16 + g + (e >> 1) * 8, cc = n0 + wn + ni * 8 + 2 * tig + (e & 1);
                if (rr < M && cc < N) out[(long)rr * N + cc] = acc[mi][ni][e];
            }
}

// The same product with 128x128 output tiles, 8 warps (4 x 2, each 32 rows x 64 columns = 2 x 8 m16n8 tiles), and
// the int8 tiles double-buffered with cp.async (zero-filled past M, N and K). Needs K % 128 == 0 (4-block stages);
// the block scales are loaded with plain loads alongside.
__device__ __forceinline__ void cp16(void* smem, const void* g, bool ok) {
    const unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(sa), "l"(g), "r"(ok ? 16 : 0));
}
extern "C" __global__ void __launch_bounds__(256) gemm_q8_big(const int* xq, const float* xd, const int* wq, const float* wd, float* out,
                                                              int M, int N, int K) {
    constexpr int BM = 128, BN = 128, KB = 4, LDW = 36;
    extern __shared__ __align__(16) int sm8[];
    int* As = sm8;                    // [2][BM * LDW]
    int* Bs = As + 2 * BM * LDW;      // [2][BN * LDW]
    float* dA = (float*)(Bs + 2 * BN * LDW);  // [2][BM * KB]
    float* dB = dA + 2 * BM * KB;              // [2][BN * KB]
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;
    const int wm = (w >> 1) * 32, wn = (w & 1) * 64;
    const int nkb = K / 32, kw = K / 4, nst = nkb / KB;
    float acc[2][8][4];
#pragma unroll
    for (int a = 0; a < 2; a++)
#pragma unroll
        for (int b = 0; b < 8; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;
    auto load = [&](int st, int buf) {
        const int kb0 = st * KB;
        // 128 rows x 8 chunks of 16 bytes each for A and B: 1024 chunks, 4 per thread each
        for (int e = tid; e < BM * 8; e += 256) {
            const int r = e >> 3, c = e & 7;
            const bool oka = m0 + r < M, okb = n0 + r < N;
            cp16(As + buf * BM * LDW + r * LDW + 4 * c, xq + (long)(oka ? m0 + r : 0) * kw + kb0 * 8 + 4 * c, oka);
            cp16(Bs + buf * BN * LDW + r * LDW + 4 * c, wq + (long)(okb ? n0 + r : 0) * kw + kb0 * 8 + 4 * c, okb);
        }
        asm volatile("cp.async.commit_group;");
        for (int e = tid; e < BM * KB; e += 256) {
            const int r = e / KB, j = e % KB;
            dA[buf * BM * KB + e] = m0 + r < M ? xd[(long)(m0 + r) * nkb + kb0 + j] : 0.f;
            dB[buf * BN * KB + e] = n0 + r < N ? wd[(long)(n0 + r) * nkb + kb0 + j] : 0.f;
        }
    };
    load(0, 0);
    for (int st = 0; st < nst; st++) {
        const int buf = st & 1;
        if (st + 1 < nst) {
            load(st + 1, buf ^ 1);
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        const int* A = As + buf * BM * LDW;
        const int* B = Bs + buf * BN * LDW;
        const float* sa = dA + buf * BM * KB;
        const float* sb = dB + buf * BN * KB;
#pragma unroll
        for (int j = 0; j < KB; j++) {
            int a[2][4];
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                const int r0 = wm + mi * 16 + g;
                a[mi][0] = A[r0 * LDW + j * 8 + tig];
                a[mi][1] = A[(r0 + 8) * LDW + j * 8 + tig];
                a[mi][2] = A[r0 * LDW + j * 8 + 4 + tig];
                a[mi][3] = A[(r0 + 8) * LDW + j * 8 + 4 + tig];
            }
            float da[2][2];
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                da[mi][0] = sa[(wm + mi * 16 + g) * KB + j];
                da[mi][1] = sa[(wm + mi * 16 + g + 8) * KB + j];
            }
#pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                const int nn = wn + ni * 8 + g;
                const int b0 = B[nn * LDW + j * 8 + tig], b1 = B[nn * LDW + j * 8 + 4 + tig];
                const int c0 = wn + ni * 8 + 2 * tig;
                const float db0 = sb[c0 * KB + j], db1 = sb[(c0 + 1) * KB + j];
#pragma unroll
                for (int mi = 0; mi < 2; mi++) {
                    int c[4] = {0, 0, 0, 0};
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                                 : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b0), "r"(b1));
                    acc[mi][ni][0] += (float)c[0] * db0 * da[mi][0];
                    acc[mi][ni][1] += (float)c[1] * db1 * da[mi][0];
                    acc[mi][ni][2] += (float)c[2] * db0 * da[mi][1];
                    acc[mi][ni][3] += (float)c[3] * db1 * da[mi][1];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int ni = 0; ni < 8; ni++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int rr = m0 + wm + mi * 16 + g + (e >> 1) * 8, cc = n0 + wn + ni * 8 + 2 * tig + (e & 1);
                if (rr < M && cc < N) out[(long)rr * N + cc] = acc[mi][ni][e];
            }
}

// ---------------------------------------------------------------------------------------------------------------
// W8A8 projections (KEV_W8): fp8 and int8 quantize weights per output channel and activations per token, each to 8
// bits with an f32 scale (amax / 448 for e4m3, amax / 127 for int8); cuBLASLt multiplies the 8-bit operands with f32
// (fp8) or i32 (int8) accumulation, and rescale_* applies sx[m] * sw[n] (Ada's cuBLASLt has no per-row/per-column
// scale epilogue). fp8t uses one scale per tensor, which cuBLASLt applies itself: no rescale pass.

// float -> e4m3fn, round to nearest even, saturating at 448 (no NaN out). In software: NVRTC targets sm_80 here and
// the hardware cvt needs sm_89.
__device__ __forceinline__ unsigned char f2e4m3(float x) {
    const unsigned s = (__float_as_uint(x) >> 24) & 0x80u;
    const float a = fminf(fabsf(x), 448.f);
    if (!(a >= 0x1p-6f)) {                              // subnormal (and 0, NaN): m * 2^-9; m == 8 is the first normal
        return (unsigned char)(s | (unsigned)__float2int_rn(a * 512.f));
    }
    int e;
    const float f = frexpf(a, &e);                      // a = f * 2^e, f in [0.5, 1)
    int E = e + 6, m = __float2int_rn((f * 2.f - 1.f) * 8.f);
    if (m == 8) { m = 0; E += 1; }
    if (E > 15 || (E == 15 && m > 6)) { E = 15; m = 6; }
    return (unsigned char)(s | (E << 3) | m);
}

// 8 bf16 (one 16-byte load) -> floats.
__device__ __forceinline__ void ld8(const bf16* p, float* f) {
    const uint4 v = *reinterpret_cast<const uint4*>(p);
    const unsigned w[4] = {v.x, v.y, v.z, v.w};
    for (int j = 0; j < 4; j++) { f[2 * j] = __uint_as_float(w[j] << 16); f[2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u); }
}
__device__ __forceinline__ void st8(bf16* p, const float* f) {
    uint4 v;
    unsigned* w = reinterpret_cast<unsigned*>(&v);
    for (int j = 0; j < 4; j++) w[j] = (unsigned)f2bf(f[2 * j]) | ((unsigned)f2bf(f[2 * j + 1]) << 16);
    *reinterpret_cast<uint4*>(p) = v;
}
__device__ __forceinline__ unsigned char q8(float v, int mode) {
    return mode == 1 ? (unsigned char)(signed char)max(-127, min(127, __float2int_rn(v))) : f2e4m3(v);
}
__device__ __forceinline__ float block_max(float v, float* sh) {
    v = warp_max(v);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    v = threadIdx.x < (blockDim.x >> 5) ? sh[threadIdx.x] : 0.f;
    if (threadIdx.x < 32) v = warp_max(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    v = sh[0];
    __syncthreads();
    return v;
}

// One block per row of x [R, K] (K % 8 == 0): q [R, K] 8-bit, s [R] f32 = amax / qmax. mode 1 = int8, else e4m3.
// With `amax` non-null the row's own max is replaced by *amax (one scale for the whole tensor, mode fp8t).
extern "C" __global__ void __launch_bounds__(256) quant_rows_bf16(const bf16* x, unsigned char* q, float* s, int K, int mode, const float* amax_in) {
    const long r = blockIdx.x;
    const bf16* xr = x + r * K;
    __shared__ float sh[8];
    float f[8], amax;
    if (amax_in) {
        amax = *amax_in;
    } else {
        amax = 0.f;
        for (int i = threadIdx.x * 8; i < K; i += blockDim.x * 8) {
            ld8(xr + i, f);
            for (int j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(f[j]));
        }
        amax = block_max(amax, sh);
    }
    const float sc = amax > 0.f ? amax / (mode == 1 ? 127.f : 448.f) : 1.f, inv = 1.f / sc;
    if (threadIdx.x == 0) s[r] = sc;
    unsigned char* qr = q + r * K;
    for (int i = threadIdx.x * 8; i < K; i += blockDim.x * 8) {
        ld8(xr + i, f);
        uint2 o;
        unsigned char* b = reinterpret_cast<unsigned char*>(&o);
        for (int j = 0; j < 8; j++) b[j] = q8(f[j] * inv, mode);
        *reinterpret_cast<uint2*>(qr + i) = o;
    }
}

// The max |x| over a whole bf16 tensor into *out (zeroed by the caller); n % 8 == 0. Non-negative floats order as
// their bits, so an integer atomicMax is exact.
extern "C" __global__ void __launch_bounds__(256) amax_bf16(const bf16* x, long n, float* out) {
    __shared__ float sh[8];
    float f[8], m = 0.f;
    for (long i = (blockIdx.x * (long)blockDim.x + threadIdx.x) * 8; i < n; i += (long)gridDim.x * blockDim.x * 8) {
        ld8(x + i, f);
        for (int j = 0; j < 8; j++) m = fmaxf(m, fabsf(f[j]));
    }
    m = block_max(m, sh);
    if (threadIdx.x == 0) atomicMax(reinterpret_cast<int*>(out), __float_as_int(m));
}

// One block per row m of y [M, N] bf16 in place (the fp8 GEMM's unit-scale output): y *= sx[m] * sw[n]; N % 8 == 0.
extern "C" __global__ void __launch_bounds__(256) rescale_bf16(bf16* y, const float* sx, const float* sw, long total, int N) {
    bf16* yr = y + blockIdx.x * (long)N;
    const float a = sx[blockIdx.x];
    float f[8];
    for (int i = threadIdx.x * 8; i < N; i += blockDim.x * 8) {
        ld8(yr + i, f);
        const float4 w0 = *reinterpret_cast<const float4*>(sw + i), w1 = *reinterpret_cast<const float4*>(sw + i + 4);
        f[0] *= a * w0.x; f[1] *= a * w0.y; f[2] *= a * w0.z; f[3] *= a * w0.w;
        f[4] *= a * w1.x; f[5] *= a * w1.y; f[6] *= a * w1.z; f[7] *= a * w1.w;
        st8(yr + i, f);
    }
}

// One block per row m of acc [M, N] i32 (the int8 GEMM) -> y bf16 = acc * sx[m] * sw[n]; N % 8 == 0.
extern "C" __global__ void __launch_bounds__(256) rescale_i32(const int* acc, const float* sx, const float* sw, bf16* y, long total, int N) {
    const int* ar = acc + blockIdx.x * (long)N;
    bf16* yr = y + blockIdx.x * (long)N;
    const float a = sx[blockIdx.x];
    float f[8];
    for (int i = threadIdx.x * 8; i < N; i += blockDim.x * 8) {
        const int4 v0 = *reinterpret_cast<const int4*>(ar + i), v1 = *reinterpret_cast<const int4*>(ar + i + 4);
        const float4 w0 = *reinterpret_cast<const float4*>(sw + i), w1 = *reinterpret_cast<const float4*>(sw + i + 4);
        f[0] = v0.x * a * w0.x; f[1] = v0.y * a * w0.y; f[2] = v0.z * a * w0.z; f[3] = v0.w * a * w0.w;
        f[4] = v1.x * a * w1.x; f[5] = v1.y * a * w1.y; f[6] = v1.z * a * w1.z; f[7] = v1.w * a * w1.w;
        st8(yr + i, f);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Producers that write a KEV_W8=int8 projection's input directly as int8 rows with one scale each (amax / 127),
// fusing quant_rows away: the bf16 row is never written or re-read. Each quantizes exactly the value the bf16 path
// would store (rounded to bf16 first), so the result is bit-identical to producer + quant_rows.
// ---------------------------------------------------------------------------------------------------------------
__device__ float block_max_all(float v) {  // max over the block (blockDim.x a multiple of 32); every thread gets it
    __shared__ float part[32];
    v = warp_max(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (l == 0) part[w] = v;
    __syncthreads();
    v = l < nw ? part[l] : 0.f;
    return warp_max(v);
}
__device__ __forceinline__ signed char q127(float v, float inv) { return (signed char)max(-127, min(127, __float2int_rn(v * inv))); }

// add_norm_bf16 (block of 256 per token, H <= 256 * QN_MAX) with the normalized row as int8 + scale instead of bf16.
#define QN_MAX 16
extern "C" __global__ void __launch_bounds__(256) add_norm_q8_bf16(const bf16* x, const bf16* m, const float* w, bf16* xo, signed char* q,
                                                                   float* qs, int H, float eps, int rounded, float scale) {
    const long row = (long)blockIdx.x * H;
    float sv[QN_MAX], ss = 0.f;
#pragma unroll
    for (int k = 0; k < QN_MAX; k++) {
        const int i = threadIdx.x + 256 * k;
        sv[k] = i < H ? residual(x, m, row + i, scale) : 0.f;
        ss += sv[k] * sv[k];
    }
    ss = block_sum(ss);
    const float r = rsqrtf(ss / H + eps);
    float amax = 0.f;
#pragma unroll
    for (int k = 0; k < QN_MAX; k++) {
        const int i = threadIdx.x + 256 * k;
        if (i < H) {
            if (m) stf(xo, row + i, sv[k]);
            const float wi = w ? w[i] : 1.f;
            sv[k] = bf2f(f2bf(rounded ? rnd<bf16>(sv[k] * r) * wi : sv[k] * r * wi));
            amax = fmaxf(amax, fabsf(sv[k]));
        }
    }
    amax = block_max_all(amax);
    const float sc = amax > 0.f ? amax / 127.f : 1.f, inv = 1.f / sc;
    if (threadIdx.x == 0) qs[blockIdx.x] = sc;
#pragma unroll
    for (int k = 0; k < QN_MAX; k++) {
        const int i = threadIdx.x + 256 * k;
        if (i < H) q[row + i] = q127(sv[k], inv);
    }
}

// act_mul_bf16 as one block of 512 per token (I <= 512 * AQ_MAX), writing act(gate) * up as int8 + scale.
#define AQ_MAX 24
extern "C" __global__ void __launch_bounds__(512) act_mul_q8_bf16(const bf16* gu, signed char* q, float* qs, int I, int gelu, int round_act) {
    const long t = blockIdx.x;
    float av[AQ_MAX], amax = 0.f;
#pragma unroll
    for (int k = 0; k < AQ_MAX; k++) {
        const int i = threadIdx.x + 512 * k;
        av[k] = 0.f;
        if (i < I) {
            const float g = bf2f(gu[t * 2 * I + i]), u = bf2f(gu[t * 2 * I + I + i]);
            float a = gelu ? 0.5f * g * (1.f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g))) : g / (1.f + expf(-g));
            if (round_act) a = rnd<bf16>(a);
            av[k] = bf2f(f2bf(a * u));
            amax = fmaxf(amax, fabsf(av[k]));
        }
    }
    amax = block_max_all(amax);
    const float sc = amax > 0.f ? amax / 127.f : 1.f, inv = 1.f / sc;
    if (threadIdx.x == 0) qs[t] = sc;
#pragma unroll
    for (int k = 0; k < AQ_MAX; k++) {
        const int i = threadIdx.x + 512 * k;
        if (i < I) q[t * I + i] = q127(av[k], inv);
    }
}

// gated_norm_bf16 as one block per token, warp h = value head h (blockDim = 32 HV <= 1024), writing the token's
// HV * DV outputs as int8 + one scale.
extern "C" __global__ void __launch_bounds__(1024) gated_norm_q8_bf16(const float* o, const bf16* p, int ld, int z_off, const float* w,
                                                                      signed char* q, float* qs, int HV, float eps) {
    const long tok = blockIdx.x;
    const int h = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float* x = o + (tok * HV + h) * DV;
    float v[4], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = x[lane + 32 * i];
        ss += v[i] * v[i];
    }
    const float r = rsqrtf(warp_sum(ss) / DV + eps);
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int d = lane + 32 * i;
        const float z = bf2f(p[tok * ld + z_off + (long)h * DV + d]);
        v[i] = bf2f(f2bf(v[i] * r * w[d] * (z / (1.f + expf(-z)))));
        amax = fmaxf(amax, fabsf(v[i]));
    }
    amax = block_max_all(amax);
    const float sc = amax > 0.f ? amax / 127.f : 1.f, inv = 1.f / sc;
    if (threadIdx.x == 0) qs[tok] = sc;
#pragma unroll
    for (int i = 0; i < 4; i++) q[(tok * HV + h) * DV + lane + 32 * i] = q127(v[i], inv);
}
