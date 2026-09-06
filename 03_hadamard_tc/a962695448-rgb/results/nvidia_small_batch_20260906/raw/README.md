# NVIDIA small_batch 线程配置实验

这是隔离实验，不修改生产仓库。固定源为 `9f5fdc363b4149d4a211701f24ab0548084ca3e5`，Dao 源为 `e7706faf8d1c3b9f241e36860640ad1dac644ede`。当前状态以原始 worker JSON 为准；只有本地脚本准备不能视为 GPU 通过。

`warp_kernel` 已用 `blockDim.x / 32` 表示每 CTA 行数，行尾判断整 warp 一致。32、64、128、256 线程都保持完整 warp；蝶形、量化和 storage RNE 不动。`thread_config.patch` 仅扩展隔离 PyTorch binding 的参数校验及帮助文字，默认仍为 128。`source_manifest.json` 记录固定 Git 原始 blob 与改动后文件的字节 SHA。

## 预注册范围与停止条件

`protocol.json` 在首次 GPU 运行前冻结。首阶段只测 M={1,17,64}、N={64,256}、两 dtype、两 scale，共 24 个配置；每个配置比较 32/64/128/256 与同轮 Dao。目标是检查少 warp 的 CTA 是否改善已知 M17、N256 情况，同时包含小于/大于该 batch 的对照。

32 或 64 只有在至少一个筛选配置的三轮 Graph 中位耗时都比同轮 128 降低 ≥5%，才进入固定 holdout：M={1,3,16,17,31,32,33,64,65}、N={16,64,128,256}、两 dtype、两 scale 去掉已筛选的 24 项，剩余 120 项。筛选与 holdout 合计 144 个预定配置。无候选则停止；不追加形状，不调宽门槛。任何候选都保留全部退化，不自动写入生产派发。

性能只测 transform；fused 与 standalone INT4 只作正确性检查。每配置固定 7 个输入，所有元素比较线程版本位级输出；每个版本的 packed/scales 与独立 CPU FP32 除法、RNE、打包结果完全一致；FP64 稠密矩阵检查首/中/尾三行，并逐元素比较真正的 Dao CUDA 输出。normalized scale 只舍入为 FP32 一次，所有路径共用同值。失败保留上下文及见证并停止本进程计时。

## 测量口径

三个独立 Python 进程串行运行。每个方法用私有 CUDA Graph，保留 64 个互异输出，跨方法输出地址也互异；固定输入；先后检查全部捕获输出字节。25 次 API 预热、5 次 Graph 预热、每组 20 次 replay、5 组。记录每组完整事件区间 ms，按 1280 次调用折算 ms/us；每轮取 5 组中位数。

同一配置中各线程及 Dao 同输入、同轮比较；方法顺序与配置顺序按固定种子在三进程重排，每组轮换。eager 另测 5×200 次分配型 Python API 调用，CUDA 事件区间可能包含 host dispatch 导致的 GPU 空隙，不称纯 kernel 或 CPU 墙钟端到端延迟。Graph 也包含均摊 replay 调度，不混用 eager、旧单输出测试或其他 GPU 数据作分母。

## 执行与复算

只在协调好的单任务 GPU 窗口使用已有 CUDA/PyTorch/Dao 环境，不安装依赖、不停止无关 PID。脚本检查一张 RTX4090、sm89，并以 `MAX_JOBS=1`、`TORCH_CUDA_ARCH_LIST=8.9` 构建独立扩展。已有空闲驻留 CUDA context 会记录，不能据此声称独占硬件。

```bash
python freeze_manifest.py
python run_suite.py --phase screen --output-directory runs/screen --reference-repo /path/to/pinned/fast-hadamard-transform
python analyze_runs.py --phase screen --output derived/screen runs/screen/run1.json runs/screen/run2.json runs/screen/run3.json
# 仅当 selection.json 的 selected_threads 非空且三轮完整通过时：
python run_suite.py --phase validation --selection derived/screen/selection.json --output-directory runs/validation --reference-repo /path/to/pinned/fast-hadamard-transform
python analyze_runs.py --phase validation --selection derived/screen/selection.json --output derived/validation runs/validation/run1.json runs/validation/run2.json runs/validation/run3.json
```

每阶段保存各 worker PID、开始/结束时间、退出码、完整 log、源/协议/扩展二进制 SHA、GPU 前后快照以及所有原始组样本。独立构建目录是本实验 `build/`，`extension_cache/` 不与生产或其他项目共用。二进制和可能出现的失败 `.pt` 见证仅保留私有证据，不作为公共源码档案。

Graph 测量与 CPU oracle 沿用已实测的同口径工具思路；新脚本将比较集合从两种线程推广为四种，保留原样引用的固定参考工具文件及其 SHA。它没有借用旧结果作性能证据。
