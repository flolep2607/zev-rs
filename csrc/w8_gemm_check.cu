// Dev-only: kev_w8_gemm_i8 against cuBLASLt int8 -> int32 -> scale on random data, including M and N that do not
// divide the tiles. nvcc -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_89 -I$CUTLASS_DIR/include csrc/w8_gemm_check.cu -lcublasLt
#include "w8_gemm.cu"
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>
#include <vector>

__global__ void fill_i8(int8_t* p, long n, unsigned seed) {
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) {
        unsigned h = (unsigned)i * 2654435761u ^ seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        p[i] = (int8_t)((int)(h % 255) - 127);
    }
}

int main() {
    const int cases[][3] = {{16384, 18432, 2560}, {1000, 2560, 9216}, {16383, 12352, 2560}, {37, 10240, 2560}, {300, 2560, 4096}};
    int8_t *A, *B; float *sx, *sw; __nv_bfloat16* D; int* acc; void* ws;
    cudaMalloc(&A, (size_t)16384 * 9216); cudaMalloc(&B, (size_t)18432 * 9216); cudaMalloc(&sx, 16384 * 4); cudaMalloc(&sw, 18432 * 4);
    cudaMalloc(&D, (size_t)16384 * 18432 * 2); cudaMalloc(&acc, (size_t)16384 * 18432 * 4); cudaMalloc(&ws, 32 << 20);
    std::vector<float> hs(18432);
    for (int i = 0; i < 18432; i++) hs[i] = 1e-4f * (1 + (i * 37 % 97));
    cudaMemcpy(sx, hs.data(), 16384 * 4, cudaMemcpyHostToDevice);
    for (int i = 0; i < 18432; i++) hs[i] = 1e-3f * (1 + (i * 53 % 89));
    cudaMemcpy(sw, hs.data(), 18432 * 4, cudaMemcpyHostToDevice);
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    for (auto& c : cases) {
        const int M = c[0], N = c[1], K = c[2];
        fill_i8<<<1024, 256>>>(A, (long)M * K, 1234u); fill_i8<<<1024, 256>>>(B, (long)N * K, 99u);
        cublasLtMatmulDesc_t desc; cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32I, CUDA_R_32I);
        int t = 1; cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &t, 4);
        cublasLtMatrixLayout_t la, lb, lc;
        cublasLtMatrixLayoutCreate(&la, CUDA_R_8I, K, N, K); cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I, K, M, K); cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, N, M, N);
        cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
        size_t wsz = 32 << 20; cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, 8);
        cublasLtMatmulHeuristicResult_t h; int cnt; cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref, 1, &h, &cnt);
        int one = 1, zero = 0;
        cublasLtMatmul(lt, desc, &one, B, la, A, lb, &zero, acc, lc, acc, lc, &h.algo, ws, wsz, 0);
        for (int tile = 0; tile < 3; tile++) {
            int rc = kev_w8_gemm_i8(A, B, sx, sw, D, M, N, K, tile, 0);
            cudaDeviceSynchronize();
            std::vector<int> ha((size_t)M * N); std::vector<__nv_bfloat16> hd((size_t)M * N); std::vector<float> hx(M), hw(N);
            cudaMemcpy(ha.data(), acc, ha.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(hd.data(), D, hd.size() * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(hx.data(), sx, M * 4, cudaMemcpyDeviceToHost); cudaMemcpy(hw.data(), sw, N * 4, cudaMemcpyDeviceToHost);
            double worst = 0; long bad = 0, wm = -1, wn = -1;
            for (long i = 0; i < (long)M * N; i++) {
                const long m = i / N, n = i % N;
                const double ref = (double)ha[i] * hx[m] * hw[n], got = __bfloat162float(hd[i]);
                const double rel = fabs(got - ref) / (fabs(ref) + 1e-3);
                if (rel > worst) { worst = rel; wm = m; wn = n; }
                if (rel > 0.01) bad++;
            }
            printf("M=%5d N=%5d K=%4d tile %d rc=%d: worst rel err %.4g at (%ld,%ld), %ld of %ld above 1%%\n", M, N, K, tile, rc, worst, wm, wn, bad, (long)M * N);
        }
    }
}
