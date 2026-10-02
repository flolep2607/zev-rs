// Dev-only: attn_fa256 against attn_mma256 (src/kev/kernels.cu) on Kev-4B's full-attention shape: nh 16, nkv 4,
// head_dim 256, output gate, sequences packed with no cached past. nvcc -O3 -std=c++17 -arch=sm_89 csrc/attn_bench.cu
#include "../src/kev/kernels.cu"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

static unsigned short h2bf(float f) { unsigned u; memcpy(&u, &f, 4); u += 0x7fffu + ((u >> 16) & 1u); return (unsigned short)(u >> 16); }
static float bf2h(unsigned short b) { unsigned u = (unsigned)b << 16; float f; memcpy(&f, &u, 4); return f; }
template <class F> float timeit(F f, int reps = 5) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); f();
    cudaEventRecord(a); for (int i = 0; i < reps; i++) f(); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b); return ms / reps;
}
int main(int argc, char** argv) {
    const int NSEQ = argc > 1 ? atoi(argv[1]) : 20, LEN = argc > 2 ? atoi(argv[2]) : 1550, NH = 16, NKV = 4, HD = 256;
    const long T = (long)NSEQ * LEN;
    unsigned s = 7;
    auto rnd = [&] { s = s * 1664525u + 1013904223u; return ((s >> 8) & 0xffff) / 65536.f - 0.5f; };
    std::vector<unsigned short> hq(T * NH * HD), hk(T * NKV * HD), hv(T * NKV * HD), hg(T * NH * 2 * HD);
    for (auto& x : hq) x = h2bf(4 * rnd()); for (auto& x : hk) x = h2bf(4 * rnd()); for (auto& x : hv) x = h2bf(rnd()); for (auto& x : hg) x = h2bf(rnd());
    std::vector<unsigned> tiles, hcu(NSEQ + 1), hpl(NSEQ, 0);
    for (int b = 0; b <= NSEQ; b++) hcu[b] = b * LEN;
    for (int b = 0; b < NSEQ; b++) for (int q0 = 0; q0 < LEN; q0 += 64) { tiles.push_back(b); tiles.push_back(q0); }
    const int ntiles = tiles.size() / 2;
    bf16 *q, *k, *v, *g, *o1, *o2; unsigned *dt, *cu, *pl; unsigned long long* nulls;
    cudaMalloc(&q, hq.size() * 2); cudaMalloc(&k, hk.size() * 2); cudaMalloc(&v, hv.size() * 2); cudaMalloc(&g, hg.size() * 2);
    cudaMalloc(&o1, T * NH * HD * 2); cudaMalloc(&o2, T * NH * HD * 2);
    cudaMalloc(&dt, tiles.size() * 4); cudaMalloc(&cu, hcu.size() * 4); cudaMalloc(&pl, hpl.size() * 4); cudaMalloc(&nulls, NSEQ * 8);
    cudaMemset(nulls, 0, NSEQ * 8);
    cudaMemcpy(q, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(k, hk.data(), hk.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(v, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(g, hg.data(), hg.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dt, tiles.data(), tiles.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(cu, hcu.data(), hcu.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(pl, hpl.data(), hpl.size() * 4, cudaMemcpyHostToDevice);
    const float scale = 1.f / 16.f;
    const int sm_mma = 128 * (HD + 8) * 2, sm_fa = 64 * (HD + 8) * 2;
    cudaFuncSetAttribute(attn_mma256, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_mma);
    const dim3 grid(ntiles, NH);
    auto run_mma = [&] { attn_mma256<<<grid, 128, sm_mma>>>(q, k, v, o1, g + HD, NH * 2 * HD, 2 * HD, dt, cu, pl, nulls, nulls, (long)NKV * HD, NH, NKV, scale, 0.f, 0); };
    auto run_fa = [&] { attn_fa256<<<grid, 128, sm_fa>>>(q, k, v, o2, g + HD, NH * 2 * HD, 2 * HD, dt, cu, pl, nulls, nulls, (long)NKV * HD, NH, NKV, scale, 0.f, 0); };
    float t1 = timeit(run_mma), t2 = timeit(run_fa);
    double flop = 0; for (int b = 0; b < NSEQ; b++) flop += 4.0 * NH * HD * (double)LEN * (LEN + 1) / 2;
    int nb1, nb2; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb1, attn_mma256, 128, sm_mma); cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb2, attn_fa256, 128, sm_fa);
    std::vector<unsigned short> a1(T * NH * HD), a2(T * NH * HD);
    cudaMemcpy(a1.data(), o1, a1.size() * 2, cudaMemcpyDeviceToHost); cudaMemcpy(a2.data(), o2, a2.size() * 2, cudaMemcpyDeviceToHost);
    long same = 0; double md = 0;
    for (size_t i = 0; i < a1.size(); i++) { same += a1[i] == a2[i]; md = fmax(md, fabs(bf2h(a1[i]) - bf2h(a2[i]))); }
    printf("attn_mma256: %.3f ms (%.1f TFLOPS, %d blocks/SM)\nattn_fa256:  %.3f ms (%.1f TFLOPS, %d blocks/SM): %.2fx; bit-identical %ld/%zu, max |diff| %.3g\n%s\n",
           t1, flop / t1 / 1e9, nb1, t2, flop / t2 / 1e9, nb2, t1 / t2, same, a1.size(), md, cudaGetErrorString(cudaGetLastError()));
}
