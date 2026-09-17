# 成对 Hadamard 两版实验与边界验证

两版候选分别在冻结协议下与 b49fff4 比较，正确性通过，但都因普通调用门槛失败而 **REJECT**。当前生产计算源码继续采用基线。新增可复现的 packed Hadamard 验证器已在基线实现上完成独立 CPU 参考的 128 项检查和 24 项流/Graph 检查。

| 候选 | 设备目标分组范围 | 设备项 | 普通调用项 | 超过 5% 的普通调用项 | 判定 |
|---|---|---:|---:|---:|---|
| 首版 | 1.264908–1.402975× | 588 | 588 | 3 | REJECT |
| 隔离融合 host 分派 | 1.264442–1.402666× | 588 | 588 | 5 | REJECT |

每版三轮，24 个方法/dtype/线程/轮次分组均超过 1.10×，全部 588 个设备项及 9 个 A/A 校准通过。门槛和样本数没有在看到结果后调低。固定地址 Graph 设备时间不能替代普通 API 调用，更不能外推为模型端到端性能。

## 两版差异和复查重点

首版在线程内完成相邻元素的第一级蝶形，再进行线程间交换。第二版保持 paired_hadamard.cuh 完全相同，仅对 fused packed 使用独立 host 分派；纯变换计算路径保持首版。第二版仍有 5 个普通调用超限点，说明本次简化不足以支持采纳。FP16 / rows65536 / N8 / out 的退化在两版均可见；其他失败点包括分配式调用。完整样本与第二版主机端符号/汇编已保留，尚未建立这些时间差的因果解释。

## 正确性范围

- 两版候选及基线各有 128 项配置：FP16/BF16、N2/4/8/16、scale1 与归一化、选定首尾分量、2B 输入和输出偏移、奇数字节 INT4 输出偏移、128/256 线程、护栏和版本。选定分量遍历 128768 个有限编码；其余分量用六类有界锚点，不是全部向量组合穷举。
- 每版候选的 24 项流/Graph 配置和故障负对照通过；第二版结束后另对基线执行完全相同的 24 项检查，通过。
- 每版 512 项有限输入的溢出、抵消、2D/4D 与偏移兼容检查，候选结果逐位等于旧 packed。该兼容检查不对非有限中间值的量化作数学意义保证。
- 每版还保留原量化分量 64 项、singleton 存储编码 131072 种、packed 700 正例/4 类拒绝/16 个上下文，以及 372 原接口正例/63 负例/4 mutation/24 上下文。重复轮次和重复调用不累计为更多独立测试。

## 复现

initial 和 isolated 各自保存源码覆盖层、固定协议、完整结果、日志及独立复算脚本。准备基线仓库 b49fff4 后，在本目录执行以下命令；默认复现 isolated，首版加 --variant initial。准备脚本核对所有基线 Git blob，重新生成明确的 CUDA-only 输入清单，原始 INPUT_MANIFEST.json 仍作为原实机会话证据保留：

```bash
python prepare_inputs.py --baseline /path/to/Learning-CUDA-b49fff4 --work /tmp/hadamard-replay
cd /tmp/hadamard-replay
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PH_CUDA_ONLY=1 python run_gpu.py
python tools/analyze.py results-remote/benchmark.json --output audit.json
```

首版 runner 可额外运行 NineToothed；上述命令明确选择 CUDA 范围。NineToothed 的 15 GPU 差分及 108 语义回归原件另存于其证据分支 6e3df8a95a3aec9b32d61de343525100b11e46b8 / docs/validation/layout-expression-gpu-20260917。

两次实机均使用 RTX4090、CUDA12.8，具体 Torch/驱动/编译器版本由 results/summary.json 记录。两次结果全部备份校验后各自关机。原始完整导出的指纹保存在 BACKUPS.json；本目录不包含租赁账户余额或私有连接信息。
