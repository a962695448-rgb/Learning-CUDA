# RTX 4090 D 限定范围自动路由验收

基线：`c591085756fec139c55f3de59c2a4ba0742dc5cc`。最终候选只为精确设备名 `NVIDIA GeForce RTX 4090 D` 增加规则：`row_layout="auto"`、dim ∈ {1,2,4,8,16}、rows ∈ [4096,65536] 时使用已有 packed kernel 与 256 线程；其余范围回退原路径。默认 original 选项和计算内核未变。

## 从剖析到保留门槛

`profile.json` 的 480 组探索记录同时测量常驻数据的 CUDA Graph 设备事件时间与普通分配式 Python 调用墙钟时间。小输入的主机调用成本显著高于设备执行时间；低维大批量的 packed 布局存在较大收益。`torch-trace.json` 和 `torch-profile.txt` 提供 CPU 分配/launch 与实际 kernel 记录。剖析器本身有开销，不能拿其三次调用耗时当稳定基准。

之后冻结 `CUDA_CONFIRMATION_PROTOCOL.json`：120 个目标配置、60 个范围外对照，三轮；每组九次旧/新交替测量。每次测量重放包含 32 次调用的 Graph 20 遍，以 CUDA event 计时。每个目标配置每轮至少快 5%，任何对照每轮不得慢 3%，并要求独立 CPU 参考及新路由执行上下文检查全部通过。

## 结果

| 轮次 | 目标配置中位数加速比范围 | 对照最大回退 |
|---|---|---|
| 0 | 1.315–7.528× | 2.176% |
| 1 | 1.314–7.521× | 0.253% |
| 2 | 1.312–7.517× | 0.253% |

180 个独立配置 × 三轮 = 540 组配对，所有保留门槛通过；九组双版本计时共 9,720 个事件计时值，记录在 `comparison.json`。输出与独立 NumPy FWHT/量化参考逐位一致，输入保持不变。

额外 16 组新路由配置覆盖 FP16/BF16、2D/4D、非默认流、每组四次变更输入后的 Graph 重放，以及每组两个错误对照；全部通过。CPU reference 与 270 个路由边界检查通过。重复测量、重放及错误对照不计为更多独立配置。

## 复现

在匹配的 NGC Torch 2.6.0a0+ecf3bae40a.nv25.01 / CUDA 12.8 / RTX 4090 D 环境中，将基线项目准备为 control，复制为 candidate，将本目录 `row_policy_4090d.hpp` 覆盖 candidate 的 `include/row_policy.hpp`。从任意目录执行：

```bash
python confirm_cuda.py --control /path/to/control --candidate /path/to/candidate --control-build /tmp/control-build --output /tmp/new-comparison --protocol CUDA_CONFIRMATION_PROTOCOL.json
```

`control`/`candidate` 均指向仓库内 `03_hadamard_tc/a962695448-rgb`。输出目录要求不存在。控制构建沿用项目 build_torch_extension；候选构建使用相同优化/ABI 标志，仅模块名与该头文件不同。

原始记录包含输入种子、各次计时、环境、扩展和源码 SHA-256；候选源码在本地与服务器逐份一致。`source/` 还保留新增 CPU 单测与 Makefile 接入，可通过 `make cpu-test` 复验。

加速比是受测配置上 auto 路径的设备执行收益，不是默认 original 调用、模型端到端或其他 GPU 的收益。数据在设备驻留且预热，不由此声称 DRAM 带宽或测得寄存器/占用率瓶颈。未覆盖的批量范围、GPU 型号和旧 A100 记录保持各自边界。
