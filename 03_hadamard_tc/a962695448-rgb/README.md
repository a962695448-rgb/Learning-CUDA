# Hadamard 变换加速（CUDA）

在输入最后一维执行 Sylvester Hadamard 变换，提供普通 CUDA 基线、warp 寄存器实现、Tensor Core 对照实现，以及变换与 INT4 量化融合。支持 FP16/BF16 和四维输入形状；验证与性能测量命令在同一可执行程序中。

本目录是独立项目，构建只需要 C++17 和 CUDA Toolkit，不依赖 PyTorch。

## 当前状态

- 已在 RTX 4060 Laptop / WSL2 上编译并验证。
- 初轮 580 组 exact-grid 用例通过；拓宽到连续随机、正态和含异常值输入后，1,876 组用例通过。
- 拓宽用例中最大绝对误差 0.0078125；35 个 warp 输出元素与已舍入矩阵参考存在容差内差异，结果如实保存在日志中。
- CPU 对实际变换输出的量化、GPU 分步量化、GPU 融合量化的 packed bytes 与 scales 全量一致。
- 本机 16 组形状/精度/scale 配置的性能扫描已记录。
- **A100 验证、profiler 记录、第三方 CUDA 库对照和最终上游提交尚未完成。**本机结果不能替代 A100 实测。

## 数学与接口约定

输入形状为 `[batch, seq, heads, dim]`，按行连续存储，前三维展平成 `rows=batch*seq*heads`。当前支持 `dim ∈ {1,2,4,8,16,32,64,128,256}`。

变换定义为：`y[row,j] = scale * sum_i H[j,i] * x[row,i]`，其中 `H[j,i] = (-1)^popcount(j & i)`。

- 默认 `scale=1`，为未归一化变换，和参考库默认 scale 对齐。
- `--normalize` 或 `--scale normalized` 使用 `scale=1/sqrt(dim)`。
- 输入和输出均为所选 FP16/BF16，内部加减与 Tensor Core 累加使用 FP32。
- 当前输入由确定性生成器按形状和精度构造；自测包含边界形状、全零、脉冲、交替符号、随机、正态和异常值模式。该命令行程序不解析外部二进制张量文件。
- 当前构建要求 sm80+，以同时验证 BF16 Tensor Core；RTX 4060 为 sm89，A100 为 sm80。

## 量化约定

首版明确采用 per-row 对称 INT4，量化整数范围 `[-7,7]`：

1. Hadamard 结果先舍入为输出 FP16/BF16。
2. 每行 `s=max(abs(y))/7`；全零行固定 `s=1`。
3. `q=clamp(round_to_nearest_even(y/s),-7,7)`。
4. 每个字节放两个 4-bit 补码数：偶数元素在低半字节，奇数元素在高半字节。N=1 时高半字节为零。

融合路径刻意保留第 1 步的舍入，使其与“先变换写回，再量化”采用相同数值语义；不能为了减少一次舍入而破坏融合一致性。每行 scale 单独输出，packed 存储确实每字节两个数。

FP8、随机符号旋转和更长维度属于当前未实现的扩展，不计入已完成能力。

## 实现路径

- `naive_global`：先转 FP32，每个蝶形阶段启动一个 kernel，最后舍入写回。用于可读基线。
- `warp`：每个 warp 处理一行；短距离蝶形使用 shuffle，跨 32 个元素的蝶形在每个线程的寄存器数组中完成。
- `tensor_core`：WMMA 执行稠密 Hadamard 矩阵乘，作为不同算法的 Tensor Core 对照；只用于 dim>=16。
- `split_int4`：warp 变换后再调用独立量化 kernel。
- `fused_int4`：一条 warp kernel 完成变换、输出精度舍入、行最大值归约、量化及打包。

当前 Tensor Core 方法做了比 FWHT 更多的运算，而且有共享内存中转。在本机测试中它比 warp 方法慢；这一结果不代表所有 Tensor Core Hadamard 算法都慢，也不作为默认最快路径。

## 构建和运行

Linux 或 WSL2，已配置 NVIDIA 驱动和 CUDA Toolkit：

```bash
# 在本目录执行；普通 Toolkit 用户设置 CUDA_HOME=/usr/local/cuda。
make CUDA_HOME="$HOME/.local/opt/cuda-12.8" ARCH=89
make cpu-test
./build/hadamard --self-test

# 单一配置。日志文件记录实际硬件与测量范围。
./build/hadamard --benchmark --batch 4 --seq 128 --heads 8 \
  --dim 256 --dtype fp16 --csv results/my-run.csv

# 完整自测、非法参数检查和 16 组基准；label 不得和已有结果混用。
python3 scripts/run_validation.py --label local-new-run --benchmark
```

A100 上重编：

```bash
make CUDA_HOME=/usr/local/cuda ARCH=80
./build/hadamard --self-test
python3 scripts/run_validation.py --label a100-first-run --benchmark
```

Makefile 会记录 NVCC 路径和编译参数；修改 ARCH 后自动重编，避免把 sm89 二进制直接当成 A100 构建。

## 如何判定正确

- 独立 CPU 参考采用 FP64 稠密矩阵求和，没有复用 GPU 的蝶形实现；再按公开输出 dtype 舍入。
- FP16 最大绝对误差严格 `<1e-2`，BF16 严格 `<5e-2`。
- 小用例对所有元素作稠密比较；大批量选 32 行作独立稠密比较，但分步/融合量化仍覆盖全部元素。
- 量化独立 oracle 对实际 warp 输出进行 CPU 量化，并与两个 GPU 路径逐字节、逐 scale 比较。
- 若变换仅在允许的误差内接近稠密参考，不会把恰在量化阈值附近的离散差异误称为融合错误；此类差异单独记录。只有 warp 变换与稠密参考完全相同时，才强制要求稠密量化也精确相等。
- CPU 专用测试还覆盖 Hadamard 二次变换还原、最近偶数舍入、补码打包及非法长度。

## 性能测量口径

- `kernel_only`：CUDA events，先预热、再多次执行取均值；排除分配、输入复制和 Hadamard 矩阵准备。
- `cpu_compute`：优化编译的单线程 FP32 FWHT，使用主机时钟，不包含重置输入的复制。
- `host_e2e`：已有缓冲区上的 pageable H2D + warp 变换 + D2H，用主机时钟计时。
- 吞吐量为输入元素数/秒，不等于 FLOP/s，也不假称物理内存带宽。
- 本机日志中的短 kernel 时间受 WSL/WDDM、频率和调度噪声影响。当前扫描是单轮重复均值，尚不是跨次运行的置信区间。

实测证据：`results/validation_sm89_expanded.log`、`results/benchmark_sm89_expanded.csv`、`results/build_sm89_dryrun.log`。初始 exact-grid 自测日志为 `results/self_test_sm89.log`，拓宽结果以 expanded 日志为准。

代表配置 `[4,128,8,256]`、未归一化、RTX 4060 Laptop：FP16 warp 约 25.00 us、融合量化约 17.06 us；独立测量时的调度差异可能使融合路径读数更低，不能只据这两个值声称增加量化反而降低变换计算量。分步量化约 32.92 us；GPU 含复制总时间约 618.12 us，CPU FWHT 约 1889.38 us。所有数值都仅对应当前 CSV 行和测量范围。

小规模不一定有端到端收益。例如 `[1,1,17,256]` FP16 的 CPU FWHT 约 7.56 us，GPU 含复制约 71.07 us；`[4,128,8,16]` 也由 CPU 更快。必须保留这些结果，不能从大规模的加速外推到所有形状。

## 后续验收

- A100 重编、完整自测、规模扫描，与 NineToothed GPU 差分统一记录环境。
- 在同一 GPU/输入/scale 下与 `fast_hadamard_transform` 对照，避免引用其 A100 公布数据冒充本项目结果。
- 用 ncu/nsys 验证 launch、访存、寄存器和 Tensor Core 路径瓶颈，考虑分解 Hadamard 的 Tensor Core 算法。
- 提交前核对训练营“包含测试”与通用“无测试代码”的措辞冲突；保留完整开发验证证据。

## 参考

- [本季度官方题目](https://github.com/InfiniTensor/Learning-CUDA/tree/2026-summer-project/03_hadamard_tc)
- [参考库及其 scale 约定](https://github.com/Dao-AILab/fast-hadamard-transform/blob/e7706faf8d1c3b9f241e36860640ad1dac644ede/README.md)

本实现自行编写；参考库用于接口和比较口径核对，尚未计入第三方 GPU 对照测试通过。
