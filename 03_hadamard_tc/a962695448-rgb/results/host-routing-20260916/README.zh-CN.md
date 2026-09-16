# RTX 4090 主机路由实验：不采用候选

2026-09-16 对一个小范围候选进行验证：在路由不需要设备属性或完整 shape 时提前返回，减少主机端元数据读取。**正确性通过，性能未达到预先固定的门槛，候选不进入提交 PR 的计算源码。**

## 受测版本

- 对照：`a962695448-rgb/Learning-CUDA` 的 `4049323cf400c73bd160f19b6c8b7defc678bf01`。
- 候选：只替换 `03_hadamard_tc/a962695448-rgb/src/torch_binding.cu` 中的 `row_choice`；完整候选文件保存在本目录 `candidate_torch_binding.cu`，属于实验材料。
- 对照 binding SHA-256：`0939e6e08f06080c245a171124c1c04a2342298eb69663d4aef4b583f346bed8`。
- 候选 binding SHA-256：`3c4202f3318278500c3f5c7ad050d55cd7fb45d693cf0b2f408c9193f9aa972d`。

两个源码树的 102 个跟踪文件在 GPU 运行前后核对，只有上述候选 binding 允许不同。kernel、数值规则、路由阈值和公开参数均未改变。`SOURCE_TREES.json` 和 `SOURCE_ARCHIVES.json` 保留本轮共享验证包的原始元数据；本实验只读取其中 CUDA 条目。

## 环境与正确性

NVIDIA GeForce RTX 4090，sm89；Python 3.12.3；PyTorch `2.6.0a0+ecf3bae40a.nv25.01`；CUDA/NVCC 12.8/12.8.61；NumPy 1.26.4。结果仅属于该设备和环境，不能替代 A100 记录。

- CPU 路由对照：21,420 组，其中 10,472 组在两版中均被拒绝。这里使用可计数的元数据替身，不是 ATen/CUDA 测试。
- GPU：696 组配置和 12 类异常输入检查通过。主矩阵的 688 组覆盖 FP16/BF16、dim=1～256 的二次幂、行数 1/17/65/257、original/auto 及合法的 packed 布局、128/256 线程、默认及归一化 scale。其余 8 组是 4D 零输入/非默认 stream 与 CUDA Graph 的新旧互比。
- 新旧输出逐字节比较；主矩阵的变换还使用独立 NumPy FWHT 参考，沿用 FP16 `<0.01`、BF16 `<0.05` 的绝对误差界限；量化使用独立 CPU 参考核对舍入与打包。补充互比不扩展为跨 stream 依赖或 Graph 行为的全面保证。
- 同一配置中的多个断言不累计为更多独立用例。Graph 在这里仅用于正确性检查。

## 计时协议与结果

[`CUDA_PROTOCOL.json`](CUDA_PROTOCOL.json) 在 GPU 计时前固定。32 个目标配置与 32 个对照配置，3 个独立进程轮次；每组预热 200 次、计时 2,000 次，每配置 9 组；新旧执行顺序交替。共 3,456 条完整计时间隔记录。

这里测量带输出分配的普通 Python 位置参数接口，包括循环开销和末尾 GPU 同步；不是 kernel event、CUDA Graph 或模型端到端时间。每配置取 9 组中位数，目标集合的耗时下降率为 `100 × (1 − 1 / GM(t对照 / t候选))`，其中 GM 是未加权几何平均。它是工程筛选指标，不是业务工作负载加速比。

保留条件：**每轮**目标集合耗时下降至少 5%，且任一目标/对照配置的耗时回退不超过 2%。

| 轮次 | 目标集合耗时下降 | 观测到的最大单配置回退 | 通过门槛 |
|---|---:|---:|---|
| 1 | -1.1455% | 20.6636% | 否 |
| 2 | -0.0448% | 7.3795% | 否 |
| 3 | +0.6025% | 0.9957% | 否 |

正值表示耗时下降，负值表示耗时增加。结果显示该候选没有达到足够且稳定的收益，不能据此宣称性能提升。未改变门槛、删去慢例或用额外采样替换原轮次；原始负例完整保留。未锁定 GPU 频率或绑定 CPU 核，数据的解释限定在记录的环境和协议内。

## 复现

使用全新目录保存本实验材料，从该目录执行：

```bash
python prepare_sources.py
# 无法联网时：python prepare_sources.py --archive /path/to/cuda-source.tar.gz
python -m unittest test_validation_tools -v
python run_cuda_experiment.py --output new-results --benchmark --stage-timeout 900
```

需要一张独占使用、sm80+ 的 NVIDIA GPU、与 PyTorch 匹配的 CUDA 开发工具、C++ 编译器和 Ninja。`nvcc` 与当前 Python 环境的可执行目录须在 PATH 中。脚本不安装依赖、不租机、不自动关机。

只复算本次记录，无需 GPU：

```bash
python -c 'import json; from pathlib import Path; from analyze_cuda import analyze; p=Path("recorded"); actual=analyze(p,json.loads(Path("CUDA_PROTOCOL.json").read_text())); assert actual==json.loads((p/"cuda-analysis.json").read_text()); print(actual["accepted"],actual["observations"])'
```

预期输出为 `False 3456`。`recorded/run-summary.json` 提供阶段命令、退出码和产物摘要；`recorded/cuda-analysis.json` 提供每个配置的对照结果。完整文本回收后，11 份原始文件逐份 SHA-256 核对通过，本地复算与服务器分析逐字段一致。

`prepare_sources.py` 是归档时补充的取源工具；实际 GPU 使用的对照、候选、验证/计时脚本、固定协议和原始数据均保留。生产实现继续使用原提交；这一负例用于说明本轮优化选择，而非扩大性能结论。
