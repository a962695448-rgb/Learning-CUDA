# 壁仞 106M 原生 SUPA 后端

本目录是 Hadamard 项目的独立壁仞移植。当前已在壁仞 106M 上完成原生 SUPA 编译及有限探针：FP16/BF16 各 257 个存储/舍入/算术样例、内存哨兵，以及两个完整 warp 的 320 个整数 XOR 交换观察。**完整 Hadamard API、融合 INT4 和性能尚待实机运行；这些探针不是完整适配通过的证明。**

壁仞当前设备为 warp32，天数和沐曦已经验证的 Warp64 路径不能直接套用。实现、测试、版本和原始结果均需独立归档，不继承其他平台的通过数量或速度比。

## 当前实机环境与资源边界

| 项目 | 已观察到的状态 |
|---|---|
| 设备 | 壁仞 106M，界面标称显存 32GB |
| 设备 API 可见显存 | 34,091,302,912 字节；保留该原始数值，不把标称容量和可见容量混为一列 |
| 执行资源 | 64 个 SM、warpSize=32、每 block 最大 1024 线程、共享内存 32 KiB |
| SDK | SUPA SDK `1.10.0.1.rc1` |
| 编译器 | `brcc 1.10.0`，Clang `16.0.1` |
| 设备管理工具 | BRSMI `1.10.2`、Driver `1.10.1`、SUPA `1.10` |
| 运行库 | 原生 `libsupa-runtime.so` |
| CUDA 兼容层 | 当前环境未找到 SUDA，不以 CUDA API 兼容层为前提 |
| Python 框架 | 当前 PyTorch 为 `2.8+cpu`，不能据此宣称已提供壁仞 GPU PyTorch 扩展 |
| 虚拟化显示 | SVI 显示 Disabled；该显示本身不能证明没有其他任务共享资源或已经获得独占整卡 |

SDK、驱动、设备工具和框架版本分别记录，不将其合并为一个推测版本。后续基准还需要记录采样时的设备状态、共享/独占证据及工具权限；不能只凭 SM 数、显存容量或 SVI 状态宣称独占与固定时钟。

## 原生 SUPA 路线

本平台使用真实 `brcc` 编译 `.su` 源文件，包含已安装 SDK 的 `supa.h`、`device/supa_fp16.h`、`device/supa_bf16.h`，链接本机 `libsupa-runtime.so`。项目源文件只引用厂商接口，**不将私人 SDK 头文件实现复制进公开仓库**。使用者需要单独安装相应 SDK。

运行时使用 SUPA 的 `suError_t`、`suStream_t`、`suEvent_t` 和 `su*` 函数；FP16/BF16 存储类型为 `float16`、`bfloat16`。这不是给 CUDA 函数名简单加宏，也不是将 CPU 结果作为设备输出。

迁移时已经核对的接口差异包括：

- 设备分配使用 `suMallocDevice`，不能推测成 `suMalloc`。
- `suMemcpyAsync` 的参数顺序为 `(dst, src, bytes, stream, kind)`，stream 位于方向枚举之前；不能沿用 CUDA 顺序。
- 使用 SUPA 自己的 stream/event 句柄和错误枚举，不与 CUDA 句柄混传。显式检查发射返回值和后续 stream 同步的执行错误。
- 初始化、上传、kernel、回读放在同一个显式 stream，不能假设 SUPA 与 CUDA 有相同的默认 stream 隐式依赖规则。
- 编译器使用真实 SUPA 宏 `__SUPACC__`，设备分支由 `__SUPA_ARCH__` 区分；不伪造 `__CUDACC__`、`__ILUVATAR__` 或 `__MACACC__`。

不能将天数的 `-x ivcore`、沐曦的 cucc/cu-bridge 或 NVIDIA 的 `-arch=sm_89` 命令照搬到本平台。最终复现命令应以独立 runner 的真实 brcc 及 include/lib 路径为准。

## BF16 最近偶数舍入：先保留失败，再修正语义

最初直接使用原生 BF16 转换时，实机样例输入 FP32 位模式 `0x3f818000`：独立最近偶数舍入参考期望 BF16 `0x3f82`，设备却给出 `0x3f81`，探针明确失败。该输入恰好位于两个 BF16 表示值中点，低候选尾位为奇数，因此按最近偶数规则应选高候选。

这条记录证明**该设备/SDK 的这一路原生转换不能在项目里直接当成已经满足 RNE 的操作**。不能把参考答案改成 `0x3f81`、放宽阈值或隐藏 BF16 来绕过失败，也不把一个反例泛化为所有未来 SDK 的行为。

修正使用本项目自行实现的位级 RNE helper：先从 FP32 位模式计算最近偶数的 BF16 表示，再得到一个 BF16 可精确表示的 FP32 值，最后转为 SUPA 的 `bfloat16` 存储。合法位转换使用固定大小的 `memcpy`，不通过违反别名规则的指针强转读取。此实现遵循格式定义，不复制厂商 SDK 的转换代码。

输入生成、设备变换输出、独立参考输出及融合量化的中间舍入必须统一到同一 RNE 契约；尤其不能用原生 BF16 主机构造器的不同舍入行为生成“参考”，否则设备和参考同时犯错也可能误判通过。最终仍须通过独立 FP64 数值参考与位模式/边界检查，而不是只比较两个共享同一路 helper 的实现。

当前已保留原失败与修正探针两个独立源文件版本，原始编译/运行日志将在本平台结果目录归档。修正后探针通过不等于完整 Hadamard 已通过，也**不证明硬件原生 BF16 算术加速**；本项目目标是 BF16 存储/转换与 FP32 内部加减。

## 最小实机探针的准确范围

- FP16、BF16 各 257 个样例，覆盖随机正负数、正负零、舍入中点和部分边界值。修正后的 BF16 存储位模式、随后转换为 FP32 的简单算术与独立宿主机参考一致。
- 设备缓冲区前后各 17 个元素的哨兵、输入保持和仅 2 字节对齐的半精度存储检查通过，覆盖不满线程块的尾部。
- 两个完整的 32-lane warp，对 XOR 距离 `1、2、4、8、16` 独立检查，共 320 个观察值通过。每个 warp 使用可区分的数据，防止跨 warp 取错值被相同输入掩盖。

这里的 shuffle 通过结果仅覆盖已运行的完整 warp/整数交换。它不等于浮点蝶形、多寄存器交互、尾行、跨 CTA 网格复用或融合量化已经通过。SUPA 接口名称带有 `_sync` 时，也不能未经核验就套用 CUDA 对活跃线程 mask 的保证；完整 warp 参与、无效逻辑元素填零和分支收敛仍需在实现中保证。

## 完整 API、复现入口与待验收项

独立接口见 [hadamard_api.h](hadamard_api.h)，实现见 [hadamard_api.su](hadamard_api.su)，命名空间为 `hadamard::biren`。声明提供 `transform`、`quantize_int4`、`transform_int4` 的 `float16`/`bfloat16` 重载，返回 `suError_t`，使用 `suStream_t`。方法为 `Baseline`、共享内存 `Optimized`、条件编译的 `Warp32`，默认仍为 `Optimized`；目前是代码入口，完整实机通过状态仍待取得。

接口为原生 C++ 异步调用：连续 FP16/BF16 `[rows,n]`，N 为 1～256 的 2 次幂，内部 FP32；连续 `[batch,seq,heads,head_dim]` 在检查乘法溢出后展平成 rows，不接受 stride 参数。scale 必须有限且为正；输入应有限，变换输出应在存储类型的有限范围内。

INT4 目标语义为 `[-7,7]`，每行 `s=max(abs(x))/7`，全零行取 1，最近偶数舍入，偶数元素在低四位，N=1 的空高四位为零；融合路径先按 FP16/BF16 存储类型舍入再量化。

调用方提供 `rows*n` 个输入/变换输出元素、`rows*ceil(n/2)` 个 packed 字节及 `rows` 个 float scales。半精度指针要求 2 字节对齐、scales 要求 float 对齐；transform 只允许完全原位或互不重叠，量化各缓冲区互不重叠。API 检查参数、对齐、大小溢出和重叠，但不查询真实分配容量或设备归属。

所有发射都使用调用方 stream，不分配、不复制、不等待。调用方检查发射错误，并在使用回读结果前同步相应 stream。合法 `rows=0` 只检查 N、scale、方法后返回，不发射内核。

Warp32 需要真实 SUPA 编译环境和 `-DHADAMARD_BIREN_WARP32`，调用方还须在初始化时经 `suGetDeviceProperties` 确认当前设备 `warpSize==32`；该查询不在每次 API 发射路径内进行。未编译支持的非空 Warp32 调用返回 `suErrorNotSupported`，不静默回退。其他同为 warp32 的设备仍不能直接继承本平台语义或性能结论。

### 准备好的复现命令（完整实机结果待取得）

runner 位于 [run_platform.py](run_platform.py)，原生验证程序为 [validate_and_benchmark.su](validate_and_benchmark.su)。从项目 `03_hadamard_tc/a962695448-rgb` 目录执行，结果目录必须是新目录：

```bash
# 快速工具链/小矩阵检查，不代表完整验收。
python3 platforms/biren/run_platform.py \
  --sdk-root /usr/local/birensupa/sdk/1.10.0.1.rc1 --warp32 --quick --no-benchmark \
  --output results/biren/warp32_quick_01

# 计划的完整矩阵及九路径基准；运行结果必须另行核验。
python3 platforms/biren/run_platform.py \
  --sdk-root /usr/local/birensupa/sdk/1.10.0.1.rc1 --warp32 --repeats 100 --groups 5 \
  --output results/biren/warp32_full_01

# 如仅复测共享路径，省略 --warp32 并使用新目录。
python3 platforms/biren/run_platform.py \
  --sdk-root /usr/local/birensupa/sdk/1.10.0.1.rc1 --no-benchmark \
  --output results/biren/shared_full_01
```

`--sdk-root` 默认 `/usr/local/birensupa/sdk/latest`，`--compiler` 默认该 SDK 下的 `brcc/bin/brcc`；runner 根据实际解析的 SDK 目录读取 `supa/include` 与 `supa/lib`。编译参数为 `-x supa -std=c++17 -O2`、项目/SDK include 路径、两个 `.su` 文件以及 `-lsupa-runtime` 和相应 rpath；`--warp32` 额外定义项目专用宏。这里的 `--warp32` 是 runner 的构建选项，不是生成二进制的运行参数。

runner 仅为子进程设置 `SUPA_PATH=SDK/supa`、`BIREN_HOME=SDK`、工具搜索路径及实际存在的 SUPA/brcc 库目录，不修改用户系统环境或替换框架。设备库搜索需要前者指向 `supa` 子目录，不能误指整个 SDK 根目录。缺失编译器、构建失败、检查失败都应以非零退出码结束；结果目录已存在则拒绝覆盖。

按当前设计，Warp32 完整模式应包含 1504 组变换、180 项 API 契约检查、14 项 CLI 拒绝检查，另外有 12 项独立写出的主机 BF16 RNE 边界检查；九路径基准每轮 1350 条组样本。**这些是代码的计划矩阵，不是壁仞实机已经通过的数字。**最终应读取 `validation.json` 和各进程退出码，不将主机检查、探针观察值或重复运行叠加为 GPU 覆盖。

输出包括构建日志、CLI 拒绝日志、验证 JSON/日志、可选基准 CSV/日志及 `run_summary.json`。摘要固定实际源文件 SHA256、Git HEAD/状态、SDK/编译器、命令和退出码；其中算法来源提交与本次实际壁仞测试提交是不同字段，不混为同一版本。

| 项目 | 当前状态 | 后续需要的证据 |
|---|---|---|
| 原生 brcc 设备编译与最小探针 | 已取得上述有限证据 | 源码 hash、原失败/修正编译及运行日志 |
| BF16 输出 RNE | helper 探针通过，整条算子链待运行 | 独立参考、半整数边界、FP16/BF16 全量样例 |
| 共享内存/warp32 Hadamard | 待完整实机验证 | 所有支持 N、多行数/尾行、两种 scale、零值/脉冲/多分布多种子/离群值 |
| 接口与内存 | 待运行 | 原位、部分重叠、对齐、溢出、空指针、零行、输入保持和前后哨兵 |
| 分步/融合 INT4 | 待运行 | CPU 对实际设备输出量化与各设备路径的 packed bytes/scales 精确一致 |
| 性能 | 尚无完整算子性能结论 | 同机、同输入语义、多独立进程的基线/优化/分步/融合事件样本，保留退化 |
| 对外复现 | 源码和 runner 已落地，完整运行待核验 | 固定实际源码版本、精确 brcc 命令、依赖/环境、日志和逐文件 hash |

完整验证沿用项目已舍入参考：独立 FP64 稠密 Hadamard 结果经 FP32 转换后，按输出 dtype 的 RNE 舍入再比较，FP16 绝对误差严格小于 `1e-2`、BF16 严格小于 `5e-2`。同时记录相对未舍入 FP64 结果的误差，不能对两列套同一阈值。

后续性能用原生 SUPA event 记录，明确排除或包含的分配、复制、同步和预热；事件区间可能含主机发射造成的空闲，不能直接冒充端到端延迟或纯内核时长。逻辑读写量换算的 GB/s 不是实测物理显存带宽。其他平台的速度比不作为壁仞结果。

公开交付只包括项目自行编写的源代码、复现入口、环境说明及脱离私人访问凭据的实验结果。私人 SDK 头文件、实例访问地址和租赁凭据不进入仓库。本次国产移植不替代九齿 A100 验收，也不表示训练营已确认加分或上游合并。
