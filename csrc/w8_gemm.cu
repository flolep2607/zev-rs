// KEV_W8=int8 projections: y [M, N] bf16 = (x_q [M, K] int8 . w_q [N, K]^T int8) * sx[m] * sw[n], one CUTLASS kernel.
// The scales are applied in the epilogue (an Sm80 epilogue visitor tree, as vLLM's cutlass_scaled_mm does), so the
// int32 accumulator never reaches memory: cuBLASLt's int8 kernels can only write int32, and the separate rescale pass
// that then re-read it cost 17% of the forward pass. Built by build.rs with nvcc against CUTLASS (CUTLASS_DIR).

#include <cuda_runtime.h>

// Order matters: visitors.hpp leans on what the gemm and numeric-conversion headers declare (as in CUTLASS's
// examples/47_ampere_gemm_universal_streamk).
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

using namespace cute;

namespace {

using ElementA = int8_t;
using ElementB = int8_t;
using ElementD = cutlass::bfloat16_t;
using ElementAcc = int32_t;
using ElementCompute = float;
constexpr int kAlignAB = 16, kAlignD = 8, kEpiStages = 1;

template <class TB, class WS, int Stages, int Swizzle = 1>
struct W8 {
    using Map = cutlass::epilogue::threadblock::OutputTileThreadLayout<TB, WS, ElementD, kAlignD, kEpiStages>;
    using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
    // sx: one scale per row m (a column vector broadcast across n); sw: one per column n
    using ScaleX = cutlass::epilogue::threadblock::VisitorColBroadcast<Map, float, Stride<_1, _0, int32_t>>;
    using ScaleW = cutlass::epilogue::threadblock::VisitorRowBroadcast<Map, float, Stride<_0, _1, int32_t>>;
    using MulW = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementCompute, ElementCompute,
                                                                cutlass::FloatRoundStyle::round_to_nearest>;
    using EvtW = cutlass::epilogue::threadblock::Sm80EVT<MulW, ScaleW, Accum>;
    using MulX = cutlass::epilogue::threadblock::VisitorCompute<cutlass::multiplies, ElementD, ElementCompute,
                                                                cutlass::FloatRoundStyle::round_to_nearest>;
    using EvtX = cutlass::epilogue::threadblock::Sm80EVT<MulX, ScaleX, EvtW>;
    using Store = cutlass::epilogue::threadblock::VisitorAuxStore<Map, ElementD, cutlass::FloatRoundStyle::round_to_nearest,
                                                                  Stride<int64_t, _1, int64_t>>;
    using Evt = cutlass::epilogue::threadblock::Sm80EVT<Store, EvtX>;
    using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, kAlignAB,     //
        ElementB, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, kAlignAB,  //
        ElementD, cutlass::layout::RowMajor, kAlignD, ElementAcc, ElementCompute, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, TB, WS, cutlass::gemm::GemmShape<16, 8, 32>, Evt,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>, Stages, cutlass::arch::OpMultiplyAddSaturate,
        kEpiStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    static int run(const void* xq, const void* wq, const float* sx, const float* sw, void* y, int m, int n, int k,
                   cudaStream_t stream) {
        typename Evt::Arguments epi{
            {
                {sx, 0.f, {_1{}, _0{}, int32_t(m)}},  // ScaleX
                {
                    {sw, 0.f, {_0{}, _1{}, int32_t(n)}},  // ScaleW
                    {},                                   // Accum
                    {},                                   // MulW
                },
                {},  // MulX
            },
            {static_cast<ElementD*>(y), {int64_t(n), _1{}, int64_t(m) * n}},  // Store
        };
        typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k}, 1, epi,
                                      static_cast<const ElementA*>(xq), static_cast<const ElementB*>(wq), nullptr,
                                      nullptr, int64_t(m) * k, int64_t(n) * k, 0, 0, k, k, 0, 0);
        Gemm gemm;
        cutlass::Status s = gemm.can_implement(args);
        if (s != cutlass::Status::kSuccess) return 100 + int(s);
        if (Gemm::get_workspace_size(args) != 0) return 99;  // kGemm with no split-k needs none
        s = gemm.initialize(args, nullptr, stream);
        if (s != cutlass::Status::kSuccess) return 200 + int(s);
        s = gemm.run(stream);
        if (s != cutlass::Status::kSuccess) return 300 + int(s);
        return cudaGetLastError() == cudaSuccess ? 0 : 400;
    }
};

template <int M, int N, int K>
using S = cutlass::gemm::GemmShape<M, N, K>;

}  // namespace

// tile 0: 128x256x64, 4 stages; 1: 256x128x64, 4 stages; 2: 128x128x64, 5 stages (vLLM's sm80 int8 default). All
// swizzle 8: launching blocks in groups of 8 along n keeps their A and B tiles in L2. Measured on a 4090 by
// csrc/w8_gemm_bench.cu over Kev-4B's five projection shapes at M = 16384 (2026-10-02): 7.34-7.41 ms, 610-654
// TOPS (the card's int8 dense peak is 660; tile 0 is 2% ahead in the server, 18.9 vs 18.6 reranks/s); cuBLASLt int8 plus its rescale pass 13.56 ms; swizzle 1 instead of 8
// dropped down_proj to 219 TOPS. Returns 0, or a code naming the failing step.
extern "C" int kev_w8_gemm_i8(const void* xq, const void* wq, const float* sx, const float* sw, void* y, int m, int n,
                              int k, int tile, void* stream) {
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (tile) {
        case 1: return W8<S<256, 128, 64>, S<64, 64, 64>, 4, 8>::run(xq, wq, sx, sw, y, m, n, k, s);
        case 2: return W8<S<128, 128, 64>, S<64, 64, 64>, 5, 8>::run(xq, wq, sx, sw, y, m, n, k, s);
        default: return W8<S<128, 256, 64>, S<64, 64, 64>, 4, 8>::run(xq, wq, sx, sw, y, m, n, k, s);
    }
}
