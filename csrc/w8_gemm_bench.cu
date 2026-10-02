// Dev-only: time W8<...> configs and cuBLASLt int8 (+ an int32 -> bf16 rescale pass) on Kev-4B's projection shapes.
// nvcc -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_89 -I$CUTLASS_DIR/include csrc/w8_gemm_bench.cu -lcublasLt -o /work/w8bench
#include "w8_gemm.cu"
#include <cublasLt.h>
#include <cstdio>
#include <vector>

__global__ void rescale_ref(const int* acc, const float* sx, const float* sw, __nv_bfloat16* y, int M, int N) {
    const int* ar = acc + (long)blockIdx.x * N;
    __nv_bfloat16* yr = y + (long)blockIdx.x * N;
    for (int i = threadIdx.x * 8; i < N; i += blockDim.x * 8) {
        int4 v0 = *(const int4*)(ar + i), v1 = *(const int4*)(ar + i + 4);
        float a = sx[blockIdx.x];
        __nv_bfloat16 o[8] = {__float2bfloat16(v0.x * a * sw[i]),     __float2bfloat16(v0.y * a * sw[i + 1]),
                              __float2bfloat16(v0.z * a * sw[i + 2]), __float2bfloat16(v0.w * a * sw[i + 3]),
                              __float2bfloat16(v1.x * a * sw[i + 4]), __float2bfloat16(v1.y * a * sw[i + 5]),
                              __float2bfloat16(v1.z * a * sw[i + 6]), __float2bfloat16(v1.w * a * sw[i + 7])};
        *(uint4*)(yr + i) = *(uint4*)o;
    }
}

template <class F>
float timeit(F f, int reps = 20) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 3; i++) f();
    cudaEventRecord(a);
    for (int i = 0; i < reps; i++) f();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    return ms / reps;
}

int main() {
    const int M = 16384;
    const int shapes[][2] = {{18432, 2560}, {2560, 9216}, {12352, 2560}, {10240, 2560}, {2560, 4096}};
    const char* names[] = {"gate_up", "down", "gdn_in", "attn_qkv", "o/out"};
    size_t maxA = (size_t)M * 9216, maxB = (size_t)18432 * 9216, maxD = (size_t)M * 18432;
    int8_t *A, *B; float *sx, *sw; __nv_bfloat16* D; int* acc; void* ws;
    cudaMalloc(&A, maxA); cudaMalloc(&B, maxB); cudaMalloc(&sx, M * 4); cudaMalloc(&sw, 18432 * 4);
    cudaMalloc(&D, maxD * 2); cudaMalloc(&acc, maxD * 4); cudaMalloc(&ws, 32 << 20);
    cudaMemset(A, 1, maxA); cudaMemset(B, 1, maxB);
    std::vector<float> ones(18432, 1e-3f);
    cudaMemcpy(sx, ones.data(), M * 4, cudaMemcpyHostToDevice); cudaMemcpy(sw, ones.data(), 18432 * 4, cudaMemcpyHostToDevice);
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    double tot[16] = {0};
    for (int si = 0; si < 5; si++) {
        const int N = shapes[si][0], K = shapes[si][1];
        const double tops = 2.0 * M * N * K / 1e12;
        // cuBLASLt int8 -> int32, plus the rescale pass
        cublasLtMatmulDesc_t desc; cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32I, CUDA_R_32I);
        int t = 1; cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &t, 4);
        cublasLtMatrixLayout_t la, lb, lc;
        cublasLtMatrixLayoutCreate(&la, CUDA_R_8I, K, N, K); cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I, K, M, K);
        cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, N, M, N);
        cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
        size_t wsz = 32 << 20; cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, 8);
        cublasLtMatmulHeuristicResult_t h; int cnt;
        cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref, 1, &h, &cnt);
        int one = 1, zero = 0;
        float lt_ms = timeit([&] { cublasLtMatmul(lt, desc, &one, B, la, A, lb, &zero, acc, lc, acc, lc, &h.algo, ws, wsz, 0); });
        float rs_ms = timeit([&] { rescale_ref<<<M, 256>>>(acc, sx, sw, D, M, N); });
        printf("%-8s N=%5d K=%4d  cuBLASLt %.3f ms (%.0f TOPS) + rescale %.3f = %.3f ms\n", names[si], N, K, lt_ms, tops / lt_ms * 1e3, rs_ms, lt_ms + rs_ms);
        tot[0] += lt_ms + rs_ms;
        int ci = 1;
#define TRY(TB_M, TB_N, TB_K, W_M, W_N, W_K, ST, SW)                                                                     \
        {                                                                                                                  \
            using C = W8<S<TB_M, TB_N, TB_K>, S<W_M, W_N, W_K>, ST, SW>;                                                    \
            int rc = C::run(A, B, sx, sw, D, M, N, K, 0);                                                                  \
            float ms = rc ? -1.f : timeit([&] { C::run(A, B, sx, sw, D, M, N, K, 0); });                                   \
            printf("   %3dx%3dx%3d w%3dx%3dx%3d st%d sw%d: %s%.3f ms (%.0f TOPS)\n", TB_M, TB_N, TB_K, W_M, W_N, W_K, ST, SW,  \
                   rc ? "FAILED " : "", ms, tops / ms * 1e3);                                                              \
            tot[ci++] += ms;                                                                                               \
        }
        TRY(128, 128, 64, 64, 64, 64, 5, 1)
        TRY(256, 128, 64, 64, 64, 64, 3, 1)
        TRY(128, 256, 64, 64, 64, 64, 3, 1)
        TRY(128, 256, 64, 64, 64, 64, 3, 8)
        TRY(256, 128, 64, 64, 64, 64, 3, 8)
        TRY(128, 256, 64, 64, 64, 64, 4, 8)
        TRY(256, 128, 64, 64, 64, 64, 4, 8)
        TRY(128, 128, 128, 64, 64, 128, 3, 8)
        TRY(256, 128, 128, 64, 64, 128, 2, 8)
        TRY(128, 256, 128, 64, 64, 128, 2, 8)
    }
    printf("sum over the 5 shapes: cuBLASLt+rescale %.3f ms; configs:", tot[0]);
    for (int i = 1; i < 11; i++) printf(" %.3f", tot[i]);
    printf("\n");
}
