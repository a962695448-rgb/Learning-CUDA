# 天数智芯 COREX 后端

此目录包含已取得 MR-V100 实机证据的共享内存原型，以及在其基础上新增的可复用 C++ API、优化候选和验证/基准 runner。**下方 324 组合实机结果只属于 `shared_baseline.cu`，不证明新增 API 或优化候选已经通过天数实机验证。**新增版本的完整实机结果待补充，没有宣称硬件原生 BF16 计算或最终性能优势。

## 首轮原型：构建与运行

在项目的 `03_hadamard_tc/a962695448-rgb` 目录执行，使用与驱动匹配的完整 COREX 开发环境：

```bash
mkdir -p build/iluvatar
/usr/local/corex/bin/clang++ -x ivcore -std=c++17 -O2 \
  -Iinclude platforms/iluvatar/shared_baseline.cu \
  -L/usr/local/corex/lib -Wl,-rpath,/usr/local/corex/lib -lcudart \
  -o build/iluvatar/shared_baseline
./build/iluvatar/shared_baseline
```

若供应商镜像使用其他 SDK 路径，调整路径。不要照搬 NVIDIA 的 `-arch=sm_89`；如需要指定天数目标架构，应使用该 SDK 和实机确认过的编译参数。

## 首轮原型：算法与验证范围

- 每个 block 处理一行，线程数为 `max(64, N)`，`N` 为 1～256 的 2 次幂。
- 全部蝶形加减使用 FP32；共享内存上每层先读取、全 block 同步、写回、再次同步。所有线程参加同步，不依赖 32 或 64 的 warp 宽度，也不依赖 WMMA。
- 变换后先按输出 FP16/BF16 舍入，再量化。每行对称 INT4 为 `[-7,7]`，采用最近偶数舍入，偶数元素放低四位；零行 scale 为 1，`N=1` 的空高四位为 0。
- 所有传输和 kernel 使用同一个显式非阻塞 stream，回传后同步，再在 CPU 上检查。
- 每种精度 162 组变换用例：9 种 N、3 种行数（1/17/257）、2 种 scale（1 和 `1/sqrt(N)`）、零值/脉冲/确定性随机输入。`N=1` 的两种 scale 数值相同，但两条参数组合均执行。
- 每组检查所有元素的 CPU FWHT 误差，并对首行/中间行/末行使用独立 FP64 稠密 Hadamard oracle；阈值严格小于 FP16 `1e-2`、BF16 `5e-2`。
- 对真实 GPU 变换输出运行 CPU 量化，检查全部打包字节和 scale 与 GPU 分步、融合路径完全一致；另有每种精度一组正负半整数最近偶数舍入测试。
- 随机输入在 `[-0.25,0.25)`；这只是初步移植范围，不等同于主线更广分布、离群值、多随机种子的完整验收。

输出包含真实设备名称、warp 宽度、运行时版本、驱动版本、每种精度用例数、元素数、最大绝对误差；失败以非零退出码退出。主程序始终尝试两种精度，不通过静默跳过 BF16 制造通过结果。

## 首轮原型：已取得的 MR-V100 实机证据

2026-09-05 在真实天数智芯 MR-V100 32GB 上完成了首轮 COREX 原生编译与运行，编译和验证进程均以退出码 0 结束。设备报告 `warp=64`、`runtime=10020`、`driver=10020`；后两个数值是兼容运行时 API 返回值，不代表使用 NVIDIA CUDA 10.2 编译器。实机编译器为 `/usr/local/corex/bin/clang++` 18.1.8，使用 `-x ivcore -std=c++17 -O2 -Iinclude -lcudart`。

| 精度 | 变换用例 | 检查元素 | 最大绝对误差 | 半整数舍入 | CPU / 分步 / 融合 INT4 |
|---|---:|---:|---:|---|---|
| FP16 | 162 | 843,150 | 0.00385761261 | 通过 | 全部字节与 scale 一致 |
| BF16 | 162 | 843,150 | 0.0301055908 | 通过 | 全部字节与 scale 一致 |

此处误差是 GPU 输出与**未舍入到 FP16/BF16 的 FP32 CPU FWHT，以及抽样 FP64 稠密 oracle**的最大绝对差。这个定义包含最终存储精度的舍入误差，不能直接与其他实验中采用已舍入参考输出的最大误差横向比较。

通过结果证明本程序的 BF16 存取/转换与 FP32 累加路径在该实机和 SDK 上可用；不证明硬件原生 BF16 算术，也不意味着国产平台适配已全部完成。

首轮源码的汇总 `printf` 触发 COREX 格式检查警告。最终版改用主机 C++ 输出流，算法和用例保持不变，并重新在 MR-V100 编译运行：编译无警告、退出码 0，324 组变换及两组舍入检查再次通过。最终源码 SHA256 为 `d8b00f6e60feaf0dd450ca66255e05bd7e0e01a060d911a9fc7c61779023f180`。

原始验证输出、空的成功编译日志、设备环境与哈希清单保存在 [results/iluvatar](../../results/iluvatar/manifest.json)。重复运行同一批组合不计为新增独立用例。

## 新增 C++ API 与优化候选（待天数实机验收）

| 文件 | 职责 |
|---|---|
| [hadamard_api.h](hadamard_api.h) | FP16/BF16 变换、INT4 量化及融合变换量化的公开接口和内存契约 |
| [hadamard_api.cu](hadamard_api.cu) | 同一接口下的 `Method::Baseline`、`Method::Optimized` 两套设备实现 |
| [validate_and_benchmark.cu](validate_and_benchmark.cu) | 独立 FP64 稠密参考、接口检查、内存边界检查及设备事件计时 |
| [run_platform.py](run_platform.py) | COREX 构建、CLI 拒绝测试、正确性验证、基准汇总及文件哈希记录 |

`Baseline` 保留每个元素对应一个线程、每层两次同步和线程 0 串行最大值归约。`Optimized` 让每个线程独占一个蝶形的两个输入/输出，每层仅在结束时同步，并用并行最大值归约计算量化 scale。两者均不依赖 NVIDIA warp32 或天数 warp64 的隐式同步，不包含 WMMA。`Optimized` 是实现名称；是否更快由同机测量决定，不能由名称推断。

API 使用 `hadamard::iluvatar` 命名空间，提供 `__half` 和 `__nv_bfloat16` 重载：

- `transform`：Hadamard 变换，允许完全原位操作 `input == output`；不允许其他部分重叠。
- `quantize_int4`：量化已经舍入到 FP16/BF16 的变换输出。
- `transform_int4`：融合变换与量化，变换后先舍入到对应存储类型，保持与分步路径相同的量化语义。

输入为连续设备内存 `[rows, n]`，`n` 是 1～256 的 2 次幂。连续 `[batch, seq, heads, head_dim]` 按 `rows=batch*seq*heads`、`n=head_dim` 传入；乘法溢出由调用方先检查。API 不接受 stride 参数，也不会为非连续视图自动整理内存。`scale` 必须有限且为正；输入应有限，变换结果应在输出存储类型的有限范围内。

调用方负责分配当前设备上容量足够的缓冲区：输入/变换输出各 `rows*n` 个对应 dtype，INT4 输出 `rows*ceil(n/2)` 字节，scale 输出 `rows` 个 float。输入和变换输出要求 2 字节对齐，scale 要求 float 对齐；量化接口的各缓冲区互不重叠。API 检查形状、scale、方法、对齐、整数/地址范围溢出及缓冲区重叠，**不查询真实分配容量或设备归属**。

全部操作在调用方 stream 上异步发射，不分配、不复制、不等待。返回值用于检查参数/发射错误，执行错误由调用方同步 stream 时检查。`rows=0` 合法：检查 `n`、`scale`、方法后返回，不检查指针或发射内核；CLI 为便于实际测试，只接受正形状。

调用示例（缓冲区及 stream 已由调用方创建）：

```cpp
#include "platforms/iluvatar/hadamard_api.h"

cudaError_t transform_then_quantize(
    const __half* input, __half* output, std::uint8_t* packed,
    float* scales, std::size_t rows, cudaStream_t stream) {
    namespace api = hadamard::iluvatar;
    auto status = api::transform(input, output, rows, 128, 1.0f, stream,
                                 api::Method::Optimized);
    if (status != cudaSuccess) return status;
    return api::quantize_int4(output, packed, scales, rows, 128, stream,
                              api::Method::Optimized);
}
```

只需要量化输出时，可用 `transform_int4(input, packed, scales, rows, 128, 1.0f, stream, method)` 替换两次调用。不要在前一调用结束前释放缓冲区；读取主机结果前还需安排复制及 stream 同步。这是原生 C++ API，当前不提供天数 PyTorch Python 扩展。

## 新版构建与可复现运行

从项目 `03_hadamard_tc/a962695448-rgb` 目录执行。runner 只需要 Python 标准库和已配套安装的 COREX SDK，不安装或替换框架、驱动。结果目录必须尚不存在，避免覆盖先前证据。

```bash
# 快速检查工具链和小矩阵；即使通过，也不代表完整验收。
python3 platforms/iluvatar/run_platform.py \
  --quick --no-benchmark --output results/iluvatar/api_quick_01

# 正式矩阵 + 默认基准：不要加 --quick。
python3 platforms/iluvatar/run_platform.py \
  --repeats 100 --groups 5 --output results/iluvatar/api_full_01
```

默认编译器是 `/usr/local/corex/bin/clang++`，SDK 根目录是 `/usr/local/corex`。可通过 `--corex-root /实际/SDK/路径` 或 `--compiler /实际/编译器/路径` 指定；二者应属于匹配的安装。runner 使用 `-x ivcore -std=c++17 -O2`，链接配套 `libcudart`；不使用 NVIDIA `-arch=sm_89`，也不启用 fast-math。

需要仅构建可执行程序时：

```bash
mkdir -p build/iluvatar
/usr/local/corex/bin/clang++ -x ivcore -std=c++17 -O2 -Iinclude \
  platforms/iluvatar/hadamard_api.cu \
  platforms/iluvatar/validate_and_benchmark.cu \
  -L/usr/local/corex/lib -Wl,-rpath,/usr/local/corex/lib -lcudart \
  -o build/iluvatar/validate_and_benchmark

# 直接运行默认完整矩阵。
./build/iluvatar/validate_and_benchmark --validate \
  --json build/iluvatar/validation.json

# 定位一个连续四维形状；这是定向验证，full_matrix 为 false。
./build/iluvatar/validate_and_benchmark --validate --dtype bf16 \
  --batch 2 --seq 17 --heads 3 --dim 128 \
  --json build/iluvatar/validation_custom.json

# 已构建程序的独立基准进程；每轮使用新的文件名。
./build/iluvatar/validate_and_benchmark --benchmark --groups 5 --repeats 100 \
  --csv build/iluvatar/benchmark_run02.csv
```

直接执行二进制会按指定路径写文件，应使用新文件名；只有 runner 对整个结果目录实施“不覆盖既有目录”的检查。二进制参数错误返回 2，设备/正确性失败返回 1；runner 只有请求的所有阶段及对应退出码检查成功才返回 0。快速模式和 `--no-benchmark` 下的成功都不能当作完整性能验收。

runner 的主要输出：

| 文件 | 内容 |
|---|---|
| `build.log`、`invalid_*.log` | 编译原始输出、CLI 拒绝测试原始输出 |
| `validation.log`、`validation.json` | 设备信息、进度、实际用例数、误差与 API 契约检查数；JSON 标明 `full_matrix` |
| `benchmark.log`、`benchmark.csv` | 基准进度和每组未经筛选的事件计时样本 |
| `run_summary.json` | 源文件 SHA256、Git HEAD/状态、编译器信息、各阶段命令/退出码、结果摘要及产物哈希 |
| `validate_and_benchmark` | 本轮编译的本机可执行文件；其 SHA256 单独记录 |

`run_summary.json` 中的 Git HEAD 是执行时读取值；若源码尚未提交，必须同时依据 `git_status` 和源文件 SHA256 追溯，不能仅将 HEAD 当成实际测试代码。发布日志前核查访问地址等私有信息；包含私有信息的原件与本机二进制可单独保存，公开副本使用自己的哈希。

## 新版完整验证设计与实机结果待补区

以下为当前源码实现的测试矩阵，**不是已取得的天数测试结果**：

- FP16/BF16 分别遍历 `N={1,2,4,8,16,32,64,128,256}`、`rows={1,3,17,257}`，scale 为 1 和 `1/sqrt(N)`；N=1 的重复 scale 只计算一次。
- 输入包括均匀分布 `[-1,1)`、标准差 0.5 的正态分布、带幅度 8 离群值的均匀分布、全零和单位脉冲。前三种使用种子 `123、8042、15961`；零值/脉冲没有随机性，只计一次。
- 全量模式另外执行 `rows=65537`、N=1/2 的用例，覆盖 block 循环处理多行的同步和网格上限路径。按当前循环静态计算，每 dtype 750 组合、共 1500；最终应以真实 `validation.json` 的完成计数为准。
- 所有元素与独立 FP64 稠密 Hadamard 矩阵结果比较。验收参考先经 FP32 转换，再舍入到输出 dtype，与主线已舍入参考约定一致；FP16 绝对误差严格小于 `1e-2`，BF16 严格小于 `5e-2`。同时保留相对未舍入 FP64 结果的最大误差，区分存储精度误差；不与首轮原型的误差列直接混用。
- 基线、优化和优化原位变换逐字节一致；两种方法的分步/融合 INT4 对照 CPU 对实际设备变换结果的量化，packed bytes 和 scale 必须完全一致。分配前后哨兵及输入副本用于检查边界写越界和输入被改动。
- API 契约覆盖非法 N/scale/方法、空指针、部分重叠、不对齐、大小溢出、零行及手写正负半整数舍入预期。runner 另执行 14 个 CLI 拒绝用例，要求实际返回 2。

基准默认覆盖两 dtype、N=64/128/256、rows=1/17/257/4096/16384。大形状以 batch/seq/heads 的乘积表示。每形状比较基线/优化的 transform、split、fused 六条路径，各先预热 10 次；默认 5 组，每组重复 100 次，组间轮换方法顺序。统计使用全部组样本，报告中位数、最小值、最大值以及包含退化的基线/优化比值。

计时使用同一显式 stream 的兼容设备事件，区间不含分配、主机/设备复制或 CPU 参考；输入只读重复使用，属于缓存可复用的测量。**`kernel_us` 是该发射序列的事件区间均值，不是端到端延迟；`logical_GBs` 是逻辑张量读写量估计，不是硬件测得的物理显存带宽。**每个大基准形状另外核验量化一致性及首/中/末行稠密参考。最终优化结论需补充多次独立进程运行的稳定性、退化形状、实际配额/共享状态及环境信息。

| 待核验内容 | 当前状态 | 应填写的证据 |
|---|---|---|
| 新版实际 COREX 编译 | 待天数实机运行确认 | 源码版本/哈希、编译器版本、完整命令、日志和退出码 |
| 两 dtype 完整矩阵与接口检查 | 待天数实机运行确认 | `validation.json`、`validation.log`、实际组合/检查数与两种误差 |
| 三轮独立基准及退化分析 | 待天数实机运行确认 | 每轮完整 CSV/日志、统计及测量边界 |
| 可复现归档 | 待上述运行完成 | 公开固定 commit、本地与公开原始结果清单/SHA256 |

首轮已有证据仍保留在前文及 [初版清单](../../results/iluvatar/manifest.json)。新增版本发布前，应按实机结果更新上表；不把尚未执行的设计用例写成“通过”。

## 后续待完成

1. 已保存并复测当前基线；后续每个优化版本继续单独保留源码、SDK、日志和哈希。
2. 在真实 COREX 环境构建新增 C++ API，并运行完整输入/精度矩阵、接口负例及独立参考验证。
3. 取得共享内存优化候选相对基线的真实计时，保留所有退化样例；如进一步探索 warp shuffle，另行验证其宽度和掩码语义，不沿用 NVIDIA warp32 假设。
4. 保存多轮设备事件性能记录，补充独占/共享配额和工具权限；若报告端到端性能，另行实现并明确包含的分配/复制/同步边界。
5. 完成 API/runner 的实机复现报告与归档。框架扩展和原生 BF16 算术支持属于额外能力，未实现或未取证时明确保持未验证状态。

国产适配结果不能替代九齿项目要求的 A100 验证。
