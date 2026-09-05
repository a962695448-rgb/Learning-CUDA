#include "kernel_operator.h"

// 自写最小 NPU 向量 Add；固定长度和对齐只用于先验证编译/发射链路。
// 后续短尾 DataCopyPad 与 FP16/BF16 RNE 使用独立探针阶段。
class AddSmoke {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
        x_global_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), kCount);
        y_global_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), kCount);
        z_global_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(z), kCount);
        pipe_.InitBuffer(x_queue_, 1, kCount * sizeof(float));
        pipe_.InitBuffer(y_queue_, 1, kCount * sizeof(float));
        pipe_.InitBuffer(z_queue_, 1, kCount * sizeof(float));
    }
    __aicore__ inline void Process() {
        auto x = x_queue_.AllocTensor<float>();
        auto y = y_queue_.AllocTensor<float>();
        AscendC::DataCopy(x, x_global_, kCount);
        AscendC::DataCopy(y, y_global_, kCount);
        x_queue_.EnQue(x);
        y_queue_.EnQue(y);
        x = x_queue_.DeQue<float>();
        y = y_queue_.DeQue<float>();
        auto z = z_queue_.AllocTensor<float>();
        AscendC::Add(z, x, y, kCount);
        z_queue_.EnQue(z);
        x_queue_.FreeTensor(x);
        y_queue_.FreeTensor(y);
        z = z_queue_.DeQue<float>();
        AscendC::DataCopy(z_global_, z, kCount);
        z_queue_.FreeTensor(z);
    }
private:
    static constexpr uint32_t kCount = 256;
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> x_queue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> y_queue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> z_queue_;
    AscendC::GlobalTensor<float> x_global_, y_global_, z_global_;
};

extern "C" __global__ __aicore__ void ascend_add_smoke(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
    AddSmoke op;
    op.Init(x, y, z);
    op.Process();
}

