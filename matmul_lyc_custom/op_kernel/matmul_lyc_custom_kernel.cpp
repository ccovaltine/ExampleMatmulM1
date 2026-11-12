#include "kernel_operator.h"

// Fixed FP16 GEMV: [1, 7168] x [7168, 16160] -> [1, 16160].
// Ascend310P3 / DAV_2002 low-level Cube implementation. No CANN MatMul API is
// called here: GM->L1->L0 data movement, Mmad and L0C->UB->GM are explicit.
// Tile sizes and core count are selected by op_host on every launch.
class MatmulLycCustomKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR a, GM_ADDR b, GM_ADDR c, AscendC::TPipe &pipe,
        uint32_t nTiles, uint32_t baseN, uint32_t baseK, uint32_t l1K,
        uint32_t kL1Tiles, uint32_t l1Prefetch)
    {
        nTiles_ = nTiles;
        baseN_ = baseN;
        baseK_ = baseK;
        l1K_ = l1K;
        kL1Tiles_ = kL1Tiles;
        l1Prefetch_ = l1Prefetch;

        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a), K);
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b), K * N);
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(c), N);

        pipe.InitBuffer(inQueueA1_, 2, FRACTAL_M * l1K_ * sizeof(half));
        pipe.InitBuffer(inQueueA2_, 2, FRACTAL_M * baseK_ * sizeof(half));
        pipe.InitBuffer(inQueueB1_, 2, l1K_ * baseN_ * sizeof(half));
        pipe.InitBuffer(inQueueB2_, 2, baseK_ * baseN_ * sizeof(half));
        pipe.InitBuffer(outQueueCO1_, 1, FRACTAL_M * baseN_ * sizeof(float));
        pipe.InitBuffer(cUbBuf_, FRACTAL_M * baseN_ * sizeof(half));
        pipe.InitBuffer(zeroUbBuf_, FRACTAL_M * l1K_ * sizeof(half));
        auto zeroUb = zeroUbBuf_.Get<half>();
        AscendC::Duplicate(zeroUb, static_cast<half>(0), FRACTAL_M * l1K_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
    }

    __aicore__ inline void Process()
    {
        // Host launches blockNum cores; N tiles are distributed over them.
        // Every tile is exactly baseN columns for this fixed shape.
        for (uint32_t tileIdx = AscendC::GetBlockIdx(); tileIdx < nTiles_;
             tileIdx += AscendC::GetBlockNum()) {
            ComputeTile(tileIdx * baseN_);
        }
    }

private:
    static constexpr uint32_t K = 7168;
    static constexpr uint32_t N = 16160;
    static constexpr uint32_t FRACTAL_M = 16;
    static constexpr uint32_t C0 = 16;

    __aicore__ inline void CopyA(uint32_t kOffset)
    {
        auto a1Local = inQueueA1_.AllocTensor<half>();
        // Build a zero-padded 16-row A tile in NZ layout. Only row zero
        // contains the original vector; ordinary Mmad can then be used.
        auto zeroUb = zeroUbBuf_.Get<half>();
        AscendC::DataCopy(a1Local, zeroUb, FRACTAL_M * l1K_);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (uint32_t k = 0; k < l1K_ / C0; ++k) {
            AscendC::DataCopy(a1Local[k * C0 * C0], aGm_[kOffset + k * C0], C0);
        }
        inQueueA1_.EnQue(a1Local);
    }

    __aicore__ inline void CopyB(uint32_t kOffset, uint32_t nOffset)
    {
        auto b1Local = inQueueB1_.AllocTensor<half>();
        // Build zZ: K fractals outer, N fractals inner, 16x16 row-major
        // inside each fractal.
        AscendC::DataCopyParams params(C0, 1, (N - C0) * sizeof(half) / 32, 0);
        for (uint32_t k = 0; k < l1K_ / C0; ++k) {
            for (uint32_t n = 0; n < baseN_ / C0; ++n) {
                uint32_t dstOffset = (k * (baseN_ / C0) + n) * C0 * C0;
                uint64_t srcOffset = static_cast<uint64_t>(kOffset + k * C0) * N + nOffset + n * C0;
                AscendC::DataCopy(b1Local[dstOffset], bGm_[srcOffset], params);
            }
        }
        inQueueB1_.EnQue(b1Local);
    }

    __aicore__ inline void SplitA(const AscendC::LocalTensor<half> &a1Local)
    {
        auto a2Local = inQueueA2_.AllocTensor<half>();
        // NZ -> ZZ for a padded 16-row A matrix.
        AscendC::LoadData2DParams params(0, baseK_ / C0, 1, 0, 0, false, 0);
        AscendC::LoadData(a2Local, a1Local, params);
        inQueueA2_.EnQue(a2Local);
    }

    __aicore__ inline void SplitB(const AscendC::LocalTensor<half> &b1Local)
    {
        auto b2Local = inQueueB2_.AllocTensor<half>();
        // zZ -> nZ: K outer, N inner; transpose within each 16x16 fractal.
        AscendC::LoadData2DParams params;
        params.repeatTimes = baseN_ / C0;
        params.srcStride = 1;
        params.dstGap = 0;
        params.ifTranspose = true;
        for (uint32_t k = 0; k < baseK_ / C0; ++k) {
            uint32_t offset = k * (baseN_ / C0) * C0 * C0;
            AscendC::LoadData(b2Local[offset], b1Local[offset], params);
        }
        inQueueB2_.EnQue(b2Local);
    }

    __aicore__ inline void ComputeTile(uint32_t nOffset)
    {
        auto c1Local = outQueueCO1_.AllocTensor<float>();
        for (uint32_t outer = 0; outer < kL1Tiles_; ++outer) {
            CopyA(outer * l1K_);
            CopyB(outer * l1K_, nOffset);
            auto a1Local = inQueueA1_.DeQue<half>();
            auto b1Local = inQueueB1_.DeQue<half>();

            for (uint32_t inner = 0; inner < l1Prefetch_; ++inner) {
                SplitA(a1Local[inner * baseK_ * FRACTAL_M]);
                SplitB(b1Local[inner * baseK_ * baseN_]);
                auto a2Local = inQueueA2_.DeQue<half>();
                auto b2Local = inQueueB2_.DeQue<half>();
                AscendC::PipeBarrier<PIPE_M>();

                AscendC::MmadParams mmadParams;
                mmadParams.m = FRACTAL_M; // Use the ordinary matrix path; only row 0 is returned.
                mmadParams.n = baseN_;
                mmadParams.k = baseK_;
                mmadParams.cmatrixInitVal = (outer == 0 && inner == 0);
                mmadParams.cmatrixSource = false;
                AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);

                inQueueA2_.FreeTensor(a2Local);
                inQueueB2_.FreeTensor(b2Local);
            }
            inQueueA1_.FreeTensor(a1Local);
            inQueueB1_.FreeTensor(b1Local);
        }
        outQueueCO1_.EnQue(c1Local);
        CopyOut(nOffset);
    }

    __aicore__ inline void CopyOut(uint32_t nOffset)
    {
        auto c1Local = outQueueCO1_.DeQue<float>();
        auto cUbLocal = cUbBuf_.Get<half>();

        // DAV_2002 does not support L0C->GM Fixpipe. Matrix-mode L0C->UB
        // performs FP32 accumulation -> FP16 conversion instead.
        AscendC::DataCopyParams matrixParams(1, baseN_ / C0, 0, 0);
        AscendC::DataCopyEnhancedParams enhancedParams(
            AscendC::BlockMode::BLOCK_MODE_MATRIX,
            AscendC::DeqScale::DEQ_NONE, 0, 0, false,
            pad_t::PAD_NONE, 0);
        AscendC::DataCopy(cUbLocal, c1Local, matrixParams, enhancedParams);
        outQueueCO1_.FreeTensor(c1Local);

        // UB still holds NZ fractals. Select row zero from each 16x16
        // fractal and write baseN contiguous FP16 values to the ND output.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopyParams ndParams(baseN_ / C0, 1, FRACTAL_M - 1, 0);
        AscendC::DataCopy(cGm_[nOffset], cUbLocal, ndParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    uint32_t nTiles_;
    uint32_t baseN_;
    uint32_t baseK_;
    uint32_t l1K_;
    uint32_t kL1Tiles_;
    uint32_t l1Prefetch_;
    AscendC::GlobalTensor<half> aGm_;
    AscendC::GlobalTensor<half> bGm_;
    AscendC::GlobalTensor<half> cGm_;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1_;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2_;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1_;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2_;
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cUbBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> zeroUbBuf_;
};

extern "C" __global__ __aicore__ void matmul_lyc_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR c,
    uint32_t nTiles, uint32_t baseN, uint32_t baseK, uint32_t l1K,
    uint32_t kL1Tiles, uint32_t l1Prefetch)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE);
    AscendC::TPipe pipe;
    MatmulLycCustomKernel op;
    op.Init(a, b, c, pipe, nTiles, baseN, baseK, l1K, kL1Tiles, l1Prefetch);
    op.Process();
    pipe.Destroy();
}
