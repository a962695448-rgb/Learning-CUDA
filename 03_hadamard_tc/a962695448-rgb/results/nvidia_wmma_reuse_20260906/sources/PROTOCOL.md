# 四 warp WMMA 输入复用：预先冻结的独立实验

状态：源代码准备与编译检查阶段。尚未执行本实验的 GPU 正确性或性能测试。这里的结果只能与本程序中的同设备、同配置、同轮次事件计时比较。

## 唯一改动与基准

基准提交为 `9f5fdc363b4149d4a211701f24ab0548084ca3e5`。`baseline/kernels.cuh` 是该提交的原始字节快照，`baseline/provenance.json` 记录 Git blob 与 SHA256。三条路径均编入同一个 C++ 程序：

| 路径 | 每个 CTA | 列划分 | 输入/输出数学 |
|---|---|---|---|
| old_wmma | 原 128 线程，warp0 计算 | 16 列 | 冻结的原 WMMA |
| four_warp_wmma | 128 线程，四个独立计算 warp | 最多 64 列 | 同原 A/H、k 循环、FP32 累加与最终 dtype RNE |
| warp128 | 原 128 线程，四行 | 每 warp 一整行 | 冻结的原 FP32 FWHT |

候选共享 `a[16*N]`，每 warp 有独立 `c[256]`，所有 CTA 线程均通过两次 barrier。N16/32 只有部分计算 warp 有效，但无效 warp 不提前返回。没有改 INT4、现有生产派发、线程块大小或其他后端。

N256 时 old/new 共享内存源码声明分别为 9/12 KiB。M17/N256 时 CTA 数从 32 降到 8，属于必须观察的小批量反例。不能把降低输入读取次数等同于实际加速，也不能依据静态寄存器信息推断实测 occupancy。

## 固定配置与留出范围

完整笛卡尔积为 N∈{16,32,64,128,256}、M∈{1,17,64,257,4096,16384}、dtype∈{FP16,BF16}、scale∈{1, float32(1/sqrt(N))}，共 120 个独立 shape/dtype/scale 配置。

预先选定筛选集：N∈{16,64,256}、M∈{17,4096} 与全部 dtype/scale，共 24 个。N16 保留在筛选集以观察无额外计算 warp 收益的反例。其余 96 个组成留出集，包括 N32、N128 及 M1/64/257/16384；不得在看到筛选时间后重新挑选留出点。`run_experiment.py --set screen/holdout/all` 输出各自固定列表。

**首次 GPU 执行前冻结的继续门槛**：筛选集全部 24 个配置的所有数值验证必须 PASS，且至少 1 个配置的候选相对 old WMMA 在三个轮次的中位时间都下降 ≥5%，才运行固定的全部 96 个留出配置。任意数值失败或没有配置满足三轮 ≥5%，即停止，不改变筛选点、阈值或挑选局部留出。通过这个门槛仅表示值得继续验证，不表示优于 warp128 或可以推广到生产。留出入口强制读取同冻结源码的筛选汇总并复核 24 个 raw CSV/validation SHA 与三轮比值；`--set all` 仅用于打印完整计划或编译，禁止绕开门槛直接执行全部数据。

数据生成 `dyadic_v1_seed_0x96269544` 固定，第一行是有符号伪随机数，随后按行混入零、全一、单脉冲、交替正负和近抵消值。所有输入先转为目标 dtype。输入幅度不超过 4，并且都是 1/128 的整数倍；FP16/BF16 存储仍保持这个性质。因此 256 项以内的每个 FP32 部分和都精确，允许建立严格的独立位级期望。

额外正确性组 `uniform24_v1_seed_0x6e4d21b3_exponents_minus12_to_0` 使用 24 位均匀随机尾数和 -12..0 的不同指数，转成目标 dtype；不假定 FP32 部分和精确。每个配置仍对 old/new 的所有元素逐位比较，并对三方法做独立 FP64 抽样：M≤32 时所有行，否则均匀抽取 32 行（包含首末行），每行所有列。沿冻结 `main.cu:305–326` 的原规则，FP64 结果转 float32 后做 dtype RNE，FP16 绝对误差严格 <0.01、BF16 严格 <0.05。各方法的 rounded/unrounded 最大误差和 rounded 位差单列；不强求这组 warp128 与 WMMA 位相同。

这些数据不能代替生产项目中极值、INT4 中点等完整验收。四 warp 对 old WMMA 的逐位对照是本候选必须满足的门槛；本次不声称 WMMA 与 FWHT 对任意输入都逐位一致。候选若采用，仍需通过生产原有完整矩阵。

## 正确性门槛

1. CPU 用独立稠密符号矩阵与 FP64 点积得到所有输出，不复用 GPU 蝶形。上述有界 dyadic 输入保证所有 FP32 和都精确；再按合同做 FP32 scale 乘法及 dtype RNE，三方法全部逐元素与其位级期望相等。
2. 单独检查候选与 old WMMA 全部输出字节完全一致。另报告相对未舍入 FP64 输出的最大绝对误差和 `error/max(1,abs(reference))`，不把 dtype 量化误差称为计算错误。
3. 计时前，分别验证从分配基址偏移 32 字节和 34 字节的输入/输出。矩阵指针始终保持 WMMA 所需对齐。每份输入、H 矩阵与输出都有前后 guard；输入及 H 必须逐字节不变，输出 guard 不得改变。
4. 非 exact 随机指数组另外验证完整 old/new 位一致、FP64 抽样及 32 字节对齐布局 guard。随后恢复原 dyadic 计时输入。三轮计时后各复查一次同配置输出/guard，不将这些复查累计成新增 shape 测试。每个配置仍只计 1 个独立配置、2 个输入组；dyadic 有 2 个 guard 布局及 3 次轮后复查，随机指数组有 1 个布局及 32 行以内的独立参考。
5. 任一 CUDA 错误、非有限值、位差或 guard 变化即退出非零；Python 汇总拒绝失败进程、缺少 PASS 字段或不完整 CSV。不能通过放宽 oracle 使实验通过。

## 计时与记录

每个配置在**同一个 C++ 进程内**执行三个轮次，不是三个独立进程。不同配置由 Python 顺序启动独立进程；CPU oracle 与任何计时不并行。轮 1/2/3 的顺序依次为 old→new→warp、new→warp→old、warp→old→new；每一轮的每个 sample 都按该顺序交错测三方法。默认每方法每轮预热 10 次、20 个事件样本，每样本连续发射 100 次相同 kernel。每配置共 180 个原始事件行；筛选集 4,320 行、完整集 21,600 行。可以在运行前整体改变样本或重复次数，但必须记录，不能只给某个方法使用不同计时设置。

CUDA event 的原始 `event_elapsed_ms` 和 `kernel_ms=event_elapsed_ms/iterations` 均保留，单位都是毫秒。所有 kernel 使用同一显式非阻塞 stream。同一进程中的输入与 H 在三方法之间共享，三份输出拥有相同大小/对齐；分配、H 构建、H2D/D2H、CPU 参考、验证及预热都在事件之外。这里使用事件包围连续 C++ kernel launch；短 kernel 可能受发射供给间隙影响，并非 CUDA Graph 数据，不与其他 Graph 测量直接相除。

每配置/轮次分别报告中位数、最小/最大值与原始样本。核心比值为同轮 `old_wmma/four_warp_wmma`；同时报告 `warp128/four_warp_wmma`，不能只战胜旧 WMMA 就省略仍慢于 FWHT 的事实。记录三轮都更快、三轮都至少降低 5% 时间的配置数以及最差负例；5% 只作为预定的实际收益摘要门槛，不是显著性检验。没有随机搜索、按结果重排数据集或隐藏 N16/32/小 M。

CSV 严格核对表头、固定字段、方法顺序、唯一 round/method/sample、有限正毫秒以及 event/kernel 单位关系。源码 SHA、完整编译命令、NVCC 版本、编译退出码/日志、二进制 SHA、进程命令/退出码及每个 raw 文件 SHA 都进入汇总。

## 编译与运行

默认命令仅检查冻结清单并打印 CPU 计划，不创建 GPU 工作：

```bash
python3 run_experiment.py --set screen
```

仅编译，不运行程序；A100 使用 sm80，4090 使用 sm89，并应在目标服务器本机重编译和记录完整命令：

```bash
python3 run_experiment.py --build-only --arch 80 --nvcc /path/to/nvcc --output /new/path/build-only
```

根代理取得统一 GPU 窗口后执行，输出目录必须不存在：

```bash
python3 run_experiment.py --execute --set screen --arch 80 --nvcc /path/to/nvcc --output /new/path/screen
python3 run_experiment.py --execute --set holdout --screen-summary /new/path/screen/run_summary.json --arch 80 --nvcc /path/to/nvcc --output /new/path/holdout
```

三方法由同一个二进制、同组 flags 编译；没有 `--use_fast_math`。仅 NVCC 编译成功不能证明任何 GPU 正确性或性能。
