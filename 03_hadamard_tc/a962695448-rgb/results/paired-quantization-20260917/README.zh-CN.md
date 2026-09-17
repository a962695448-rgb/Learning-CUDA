# 成对量化的完整实验与最终采用证据

基线为 a962695448-rgb/Learning-CUDA 的 fabe4aa1beb4f623564ba02ce8cdf24af5932b11，硬件为 RTX4090（本轮 4090D 无空闲 GPU，未使用它做新测量）。source 是最终通过的内核、主机分派和验证工具。

- initial：通用 launch 中的新分派；一个旧 Hadamard 主机控制点失败，REJECT。
- isolated：隔离 quantize-only 分派；仍有一个双峰小批量主机计时失败，REJECT。
- confirmation：固定 CPU、延长采样并加入同函数 A/A 后仍有 N4/16387 分配式调用失败，REJECT。
- final：采用编译期维度的网格计算后，重跑完整正确性、同样长采样并增加该 N4 点的 A/A，全部 ACCEPT。

最终 288 条设备、288 条普通调用、9 组 A/A 均满足 5% 退化门槛，12 个目标设备分组均超过 1.10 倍。分组设备 1.361–1.425 倍、普通 allocating/out 目标约 1.069/1.117 倍；所有负结果原样保留，不相乘不同阶段比值。

64 组有限分量编码/位置/偏移/线程检查分别在原版和候选通过；128768 个有限编码被放入选定分量，不是向量组合的全穷举。700 组 packed、372/63/4 原接口、16 组流/Graph 及 N1 存储编码检查通过。最终内核与数学测试未靠更改容差获得通过。

## 复现最终版本

检出固定基线，在具备相应 NGC Torch/CUDA 的 RTX4090 上执行：

```bash
python prepare_inputs.py --baseline /path/to/Learning-CUDA --work /tmp/paired-quantization-replay
AP_CUDA_ONLY=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python /tmp/paired-quantization-replay/run_gpu.py
```

准备脚本逐个核对基线 Git blob，再叠加 source。输出目录必须不存在。run_gpu.py 与最终实测 run_gpu_final.py 字节一致，只采用文件别名；固定到一个允许的 CPU 核、Torch CPU 线程数 1、每样本 5000 次调用、每轮 9 样本。数据/掩码预存和输出预分配的边界见协议。硬件计数器或进程 RSS 没有测量，不能据此声称因果或进程内存收益。

raw summary 与 INPUT_MANIFEST 保留原运行的文件路径和环境。另一项目的输入路径仅为历史冻结记录，CUDA 复现无需准备其源码。档案不包含租赁账户或凭据。
