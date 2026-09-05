# 沐曦 C500 MACA 后端

本目录是 Hadamard 项目的独立沐曦移植。当前已完成 C500 实机的最小编译、FP16/BF16 存储与舍入、简单算术、内存哨兵和两组完整 warp 的 shuffle 探针；**完整 Hadamard API 的正确性、融合 INT4 和性能验证尚待实机运行，不能据此宣布完整适配通过。**

算法起点是本项目天数后端固定提交 `a387db3332c6f9b01f128dd681848260c9691281`，沐曦代码使用独立命名空间、编译宏和结果记录。天数与本次沐曦设备都报告 warp64，不代表它们具有相同的编译、同步、舍入或性能行为；天数的 1504 组结果不能计为沐曦通过用例。

## 已核验的开发环境与实际配额

| 项目 | 当前实机/镜像信息 |
|---|---|
| 设备 | 沐曦 C500，sGPU 计算配额 **25%** |
| 可用资源配额 | `mx-smi` 明确显示 **16000 MiB** 显存配额；租赁界面标为 16GB |
| 物理设备属性 | 65536 MiB 显存、104 个 SM；这是物理属性，不能当成当前实例独占资源 |
| MACA/驱动工具报告 | MACA `3.0.0.8`，Kernel Mode `3.8.30`，`mx-smi 2.2.6` |
| 供应商镜像标签 | PyTorch `2.4`、Python `3.10.10`、MACA `3.0.0.4`；镜像标签与驱动工具报告分别记录，不合并成一个版本 |
| 编译器 | `cucc` / `mxcc 1.0.0`，版本标识 `df29922f9c` |
| 兼容运行时返回值 | runtime API 和 driver API 均返回 `11060`；这是兼容 API 的数值，不证明正在使用 NVIDIA CUDA 11.6 |
| warp | 设备报告 64；已对两个完整 warp 执行最小交换探针 |

**本实例是 25% 计算配额的 C500 sGPU，不是独占完整 C500。**后续性能必须标注该配额，不能与整卡结果直接比较，也不能简单将吞吐量乘以四外推整卡性能。尚未取得干扰隔离、固定时钟和性能计数器权限的证据。

记录环境时可运行：

```bash
mx-smi
/opt/maca/tools/cu-bridge/bin/cucc --version
/opt/maca/mxgpu_llvm/bin/mxcc --version
python3 --version
```

供应商的镜像和宿主机驱动可以分别提供组件。当前版本组合只以已运行探针为初步兼容证据，不据此承诺其他 MACA 版本组合可用。

## MACA 原生接口与本项目的 cu-bridge 路径

MACA 原生开发使用 `mxcc`、MACA 头文件与运行时接口，例如 `mc_runtime.h`、`mc_common.h` 和 `mcMalloc` 等；原生源文件需要按 MACA API 编写。这与保留 CUDA 命名的兼容源文件是两种开发入口，不能把它们的头文件和链接参数随意混用。

当前移植选择 **MACA 的 cu-bridge 编译路径**：`/opt/maca/tools/cu-bridge/bin/cucc` 编译设备程序，源码保留 `cuda_runtime.h`、`cuda_fp16.h`、`cuda_bf16.h`、`__half`、`__nv_bfloat16` 等兼容名称。已在该机器确认这些头文件及转换入口可用。保留 CUDA 名称不等于由 NVIDIA 编译器执行，也不等于 CPU 回退。

不要把天数的 `/usr/local/corex`、`-x ivcore`、`__ILUVATAR__`，或 NVIDIA 的 `-arch=sm_89` 参数带入沐曦构建。沐曦 Warp64 分支使用 `HADAMARD_METAX_WARP64` 和真实 MACA 编译环境的 `__MACACC__`；不手动伪造编译器识别宏。

## 最小实机探针已经证明什么

- FP16、BF16 各 257 个输入，包含正负值、正负零、舍入中点及边界样例。设备存储位模式与独立宿主机整数最近偶数舍入参考一致，随后转为 FP32 的简单算术结果位模式一致。
- 输入保持不变、分配区前后各 17 个元素的哨兵检查通过；覆盖只满足 2 字节对齐的半精度存储及不满线程块的尾部。
- 两个完整的 64-lane warp，对 XOR 距离 `1、2、4、8、16、32` 执行整数 `__shfl_xor` 交换，共 768 个观察值全部符合预期，哨兵检查通过。

这些结果支持继续编译和测试设备内核，不证明完整 FWHT、跨寄存器蝶形、多行/尾行同步、浮点 shuffle 组合、INT4 融合或性能已经正确。BF16 证明的是存储/转换及 FP32 计算链路，**不宣称 C500 的原生 BF16 算术加速已经由本探针证实**。原始探针日志和源码 hash 应随本平台独立证据归档，不能借用天数的日志。

## C++ API 与调用契约

接口见 [hadamard_api.h](hadamard_api.h)，实现见 [hadamard_api.cu](hadamard_api.cu)，命名空间为 `hadamard::metax`。现有声明包括：

| 入口 | 目标行为 |
|---|---|
| `transform` | FP16/BF16 Hadamard 变换，内部 FP32，允许完全原位变换 |
| `quantize_int4` | 对已存储的 FP16/BF16 数据执行每行对称 INT4 量化 |
| `transform_int4` | 融合变换与量化，先按 FP16/BF16 输出类型舍入，再量化 |

方法枚举包含 `Baseline`、共享内存 `Optimized` 和条件编译的 `Warp64`，默认仍是 `Optimized`。这些是当前代码中的实现入口，**各方法的完整沐曦实机结果仍待验证**。

输入为连续设备内存 `[rows,n]`，N 是 1～256 的 2 次幂；连续 `[batch,seq,heads,head_dim]` 可在检查乘法溢出后令 `rows=batch*seq*heads`、`n=head_dim`。API 不接受 stride 参数，不自动转换非连续视图。scale 应有限且为正，输入应有限，变换后的值应位于输出存储类型的有限范围内。

输入/变换输出各需 `rows*n` 个对应 dtype，INT4 需 `rows*ceil(n/2)` 字节，量化 scale 需 `rows` 个 float。输入及变换输出要求 2 字节对齐，scales 要求 float 对齐；量化涉及的缓冲区互不重叠。transform 只允许完全原位或互不重叠。API 不查询真实分配容量或设备归属，调用方必须保证它们有效。

INT4 约定为 `[-7,7]`，每行 `s=max(abs(x))/7`，全零行取 `s=1`；最近偶数舍入，偶数元素存低四位，N=1 的空高四位为零。融合路径与分步路径的 packed bytes 及 float scales 必须分别实机证明完全一致。

操作只在调用方 stream 上发射，不分配、不复制、不等待。检查参数/发射返回值，并在回读前检查 stream 同步的返回值。合法 `rows=0` 不发射内核，仍检查 N、scale 和方法；不能把零行成功当作设备路径验证。

Warp64 必须用真实 MACA/cu-bridge 编译，显式启用 `-DHADAMARD_METAX_WARP64`；独立 API 调用方还须在设备初始化时通过 `cudaGetDeviceProperties` 检查 `prop.warpSize == 64`。未编入该路径时，非空 Warp64 调用返回 `cudaErrorNotSupported`，不静默回退。仅通过该属性检查仍不足以替代本平台完整测试。

## 完整复现入口与待验收项

本平台已有独立的 [run_platform.py](run_platform.py) 和 [validate_and_benchmark.cu](validate_and_benchmark.cu)。以下是按当前实现准备的执行命令，**尚不是完整 API 已在沐曦实机通过的证明**。从项目 `03_hadamard_tc/a962695448-rgb` 目录执行，结果目录必须尚不存在：

```bash
# 快速检查，不能替代完整矩阵或性能验收。
python3 platforms/metax/run_platform.py --warp64 --quick --no-benchmark \
  --output results/metax/warp64_quick_01

# 完整矩阵与默认基准。
python3 platforms/metax/run_platform.py --warp64 --repeats 100 --groups 5 \
  --output results/metax/warp64_full_01

# 若需要只检查共享实现，省略 --warp64；使用新的结果目录。
python3 platforms/metax/run_platform.py --no-benchmark \
  --output results/metax/shared_full_01
```

`--maca-root` 默认为 `/opt/maca`，`--compiler` 默认为该目录下 `tools/cu-bridge/bin/cucc`。runner 保留 `cucc` 入口名称，不因解析符号链接而改变兼容模式；实际编译为 cucc、C++17、`-O2`、项目 include 路径和两个 `.cu` 文件，Warp64 模式额外定义 `HADAMARD_METAX_WARP64`，由 cucc 处理兼容运行库链接。

runner 只为本次子进程设置 `MACA_PATH`、指向 cu-bridge 的 `CUDA_PATH`/`CUCC_PATH`、工具搜索路径及存在的 SDK 库目录；保留继承环境，不安装/替换驱动或框架。`--warp64` 是构建选项，生成的程序会在设备运行前检查真实 warp 宽度。不满足条件时应明确失败，不能把该条件静默跳过后写成支持。

按当前循环设计，Warp64 完整模式应执行 1504 个参数组合、180 项 API 契约检查，另有 runner 的 14 项 CLI 拒绝检查；基准为九条路径，每轮 1350 组样本。不启用 Warp64 时为 1500 个参数组合、122 项 API 契约检查、12 项未编入路径检查和六条基准路径。**这些是计划完成数，实际通过数以本平台运行后的 JSON/退出码为准，不是已完成数据。**

每个新目录保存 `build.log`、`invalid_*.log`、`validation.log/json`、可选 `benchmark.log/csv` 和 `run_summary.json`。摘要包含本次沐曦源码 hash、Git HEAD/状态、MACA 环境、编译器、各阶段命令/退出码及产物 hash。`adapted_from` 是算法来源的天数提交，不能误认成当前沐曦实际测试源码版本。后续多轮须使用新的文件/目录；同一正确性矩阵重复运行不累计成新用例覆盖。

| 验收项 | 当前状态 | 需要补充的结果 |
|---|---|---|
| 最小设备编译、存储/舍入/算术、整数 shuffle 探针 | 已通过上述有限范围 | 本平台原始日志和探针源码 hash 归档 |
| 全部方法实际编译 | 待完整 API 实机运行 | cucc 命令、编译器/链接环境、构建日志与退出码 |
| FP16/BF16 完整矩阵 | 待运行 | 所有支持 N、两种 scale、多个行数/尾行、分布与种子、离群值、独立 FP64 稠密 oracle |
| API、内存边界与 INT4 | 待运行 | 原位/重叠/对齐/溢出/空指针/零行、哨兵、最近偶数舍入、全部分步/融合 bytes 与 scale |
| Warp64 的跨行与网格复用 | 待运行 | 单 warp/不满 CTA、多 CTA、超过网格上限时的多行复用；不能只复用两个完整 warp 的探针结果 |
| 性能 | 未测量，无加速结论 | 同一 25% sGPU 配额下多轮、相同输入语义的基线/共享/Warp64/分步/融合样本，保留退化 |
| 公布与复现 | 待完整运行结束 | 固定源码提交、环境与配额、公开原始结果清单及 SHA256 |

正确性沿用项目已舍入参考定义：独立 FP64 稠密结果经 FP32 转换后舍入到输出 dtype，与实际输出比较，FP16 绝对误差严格小于 `1e-2`、BF16 严格小于 `5e-2`。另行保留相对未舍入 FP64 的误差，不混用两列阈值；不得扩大阈值、缩小输入或跳过 BF16 来掩盖失败。

性能应固定分配与输入，区分设备事件区间和包含主机/设备复制的端到端时间，记录预热、重复次数、独立运行次数及全部原始样本。逻辑读写量换算的 GB/s 不是实测物理显存带宽。天数和沐曦处于不同硬件/配额，不直接用两张表相除宣称平台优劣。

公开材料排除实例访问地址、租赁编号和凭据；保留足以复现的设备型号、配额、软件版本、命令和日志。国产适配是 Hadamard 的扩展工作，不替代九齿项目的 A100 验收，也不等于已获训练营加分或上游合并。
