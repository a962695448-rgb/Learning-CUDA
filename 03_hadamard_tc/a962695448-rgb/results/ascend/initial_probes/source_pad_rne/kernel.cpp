#include "kernel_operator.h"

// 仅为边界/精度探针使用保守屏障，不用于性能结论。
template<class T> class PadProbe {
public:
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, uint32_t n) {
        AscendC::GlobalTensor<T> src, dst;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input), n);
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output), n);
        pipe_.InitBuffer(queue_, 1, 272 * sizeof(T));
        auto local = queue_.AllocTensor<T>();
        const uint32_t valid_bytes = n * sizeof(T);
        const uint8_t right_pad = ((32 - valid_bytes % 32) % 32) / sizeof(T);
        AscendC::DataCopyExtParams copy{1, valid_bytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{true, 0, right_pad, static_cast<T>(0)};
        AscendC::DataCopyPad(local, src, copy, pad);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(dst, local, copy);
        AscendC::PipeBarrier<PIPE_ALL>();
        queue_.FreeTensor(local);
    }
private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> queue_;
};

template<class T> class RneProbe {
public:
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR stored, GM_ADDR readback, uint32_t n) {
        AscendC::GlobalTensor<float> src, back;
        AscendC::GlobalTensor<T> dst;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input), n);
        back.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(readback), n);
        dst.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(stored), n);
        pipe_.InitBuffer(in_, 1, 272 * sizeof(float));
        pipe_.InitBuffer(out_, 1, 272 * sizeof(T));
        pipe_.InitBuffer(back_, 1, 272 * sizeof(float));
        auto input_local = in_.AllocTensor<float>();
        auto typed_local = out_.AllocTensor<T>();
        auto float_local = back_.AllocTensor<float>();
        const uint32_t input_bytes = n * sizeof(float);
        AscendC::DataCopyExtParams input_copy{1, input_bytes, 0, 0, 0};
        const uint8_t right_pad = ((32 - (n * sizeof(float)) % 32) % 32) / sizeof(float);
        AscendC::DataCopyPadExtParams<float> pad{true, 0, right_pad, 0.0f};
        AscendC::DataCopyPad(input_local, src, input_copy, pad);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(typed_local, input_local, AscendC::RoundMode::CAST_RINT, n);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(float_local, typed_local, AscendC::RoundMode::CAST_NONE, n);
        AscendC::PipeBarrier<PIPE_ALL>();
        const uint32_t typed_bytes = n * sizeof(T);
        AscendC::DataCopyExtParams typed_copy{1, typed_bytes, 0, 0, 0};
        AscendC::DataCopyPad(dst, typed_local, typed_copy);
        AscendC::DataCopyPad(back, float_local, input_copy);
        AscendC::PipeBarrier<PIPE_ALL>();
        in_.FreeTensor(input_local);
        out_.FreeTensor(typed_local);
        back_.FreeTensor(float_local);
    }
private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> in_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_, back_;
};

extern "C" __global__ __aicore__ void ascend_pad_fp16(GM_ADDR x, GM_ADDR y, uint32_t n) {
    PadProbe<half> op; op.Run(x, y, n);
}
extern "C" __global__ __aicore__ void ascend_pad_bf16(GM_ADDR x, GM_ADDR y, uint32_t n) {
    PadProbe<bfloat16_t> op; op.Run(x, y, n);
}
extern "C" __global__ __aicore__ void ascend_rne_fp16(GM_ADDR x, GM_ADDR y, GM_ADDR back, uint32_t n) {
    RneProbe<half> op; op.Run(x, y, back, n);
}
extern "C" __global__ __aicore__ void ascend_rne_bf16(GM_ADDR x, GM_ADDR y, GM_ADDR back, uint32_t n) {
    RneProbe<bfloat16_t> op; op.Run(x, y, back, n);
}
