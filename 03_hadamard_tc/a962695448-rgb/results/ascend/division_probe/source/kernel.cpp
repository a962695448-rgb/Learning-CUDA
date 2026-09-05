#include "kernel_operator.h"

#ifndef PROBE_SCALAR_DIV
#define PROBE_SCALAR_DIV 0
#endif

// 固定256个FP32输入，与main.cpp一致；只测数值，不作性能结论。
template<bool Scalar> class DivProbe {
public:
    __aicore__ inline void Init(GM_ADDR lhs, GM_ADDR rhs, GM_ADDR output) {
        lhs_global_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lhs), kCount);
        rhs_global_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rhs), kCount);
        output_global_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output), kCount);
        pipe_.InitBuffer(lhs_queue_, 1, kCount * sizeof(float));
        pipe_.InitBuffer(rhs_queue_, 1, kCount * sizeof(float));
        pipe_.InitBuffer(output_queue_, 1, kCount * sizeof(float));
    }
    __aicore__ inline void Process() {
        auto lhs = lhs_queue_.template AllocTensor<float>();
        auto rhs = rhs_queue_.template AllocTensor<float>();
        AscendC::DataCopy(lhs, lhs_global_, kCount);
        AscendC::DataCopy(rhs, rhs_global_, kCount);
        lhs_queue_.EnQue(lhs);
        rhs_queue_.EnQue(rhs);
        lhs = lhs_queue_.template DeQue<float>();
        rhs = rhs_queue_.template DeQue<float>();
        auto output = output_queue_.template AllocTensor<float>();
#if PROBE_SCALAR_DIV
        if constexpr (Scalar) {
            // 显式MTE2->S后才读取UB；S->MTE3后才写回，不以CPU生成actual。
            auto read_event = pipe_.template AllocEventID<AscendC::HardEvent::MTE2_S>();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(read_event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(read_event);
            pipe_.template ReleaseEventID<AscendC::HardEvent::MTE2_S>(read_event);
            for (uint32_t i = 0; i < kCount; ++i) {
                const float numerator = lhs.GetValue(i);
                const float denominator = rhs.GetValue(i);
                const float quotient = numerator / denominator;
                output.SetValue(i, quotient);
            }
            auto write_event = pipe_.template AllocEventID<AscendC::HardEvent::S_MTE3>();
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(write_event);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(write_event);
            pipe_.template ReleaseEventID<AscendC::HardEvent::S_MTE3>(write_event);
        } else
#endif
        {
            AscendC::Div(output, lhs, rhs, kCount);
        }
        output_queue_.EnQue(output);
        lhs_queue_.FreeTensor(lhs);
        rhs_queue_.FreeTensor(rhs);
        output = output_queue_.template DeQue<float>();
        AscendC::DataCopy(output_global_, output, kCount);
        output_queue_.FreeTensor(output);
    }
private:
    static constexpr uint32_t kCount = 256;
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> lhs_queue_, rhs_queue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> output_queue_;
    AscendC::GlobalTensor<float> lhs_global_, rhs_global_, output_global_;
};

extern "C" __global__ __aicore__ void ascend_div_vector_probe(GM_ADDR lhs, GM_ADDR rhs, GM_ADDR output) {
    DivProbe<false> probe;
    probe.Init(lhs, rhs, output);
    probe.Process();
}

#if PROBE_SCALAR_DIV
extern "C" __global__ __aicore__ void ascend_div_scalar_probe(GM_ADDR lhs, GM_ADDR rhs, GM_ADDR output) {
    DivProbe<true> probe;
    probe.Init(lhs, rhs, output);
    probe.Process();
}
#endif
