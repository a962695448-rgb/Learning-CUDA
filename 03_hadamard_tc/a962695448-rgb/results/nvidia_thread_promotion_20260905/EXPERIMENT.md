# NVIDIA 256 线程候选的有限复核

只复核此前三轮 fixed-buffer 筛选中的 24 个稳定候选，不变更生产 API。

- 纯变换：N=16/64，M=4096/16384；两精度、两 scale。
- 融合 INT4：N=16/64，M=4096；两精度、两 scale。
- 邻近复核：对应每个目标 M 的 M−1、M+1，合计 72 个性能配置。
- 对照：同一扩展内显式 128/256 线程；变换再加入固定提交 Dao 对照。
- 内核逐字节复制不改；包装器仅暴露线程参数并限制 N=16/64。改动补丁和源文件 SHA 随附。
- 图结构：每方法各自私有 CUDA Graph 池；64 次调用各保留独立输出（量化的 packed/scales 均独立），25 次 API 预热、5 次图预热，每组重放 20 次、5 组；3 个独立进程的测量顺序轮换。
- 正确性：48 个 shape/dtype/scale ×7 个确定性模式/seed=336 个输入用例；全部元素 baseline/candidate 位一致、Dao 原严格容差、融合/分步位一致、CPU 全量 INT4 与最多三行独立 FP64 稠密参考。三次复跑不计新增独立用例。
- 验收：每轮相对同轮 128 线程中位数减少至少5%才标稳定候选；所有原目标和邻近负例保留。不拿旧 fixed-buffer 数字作分母。不能将图均摊时间称纯 kernel 延迟、端到端或物理带宽。

初次读到的服务器状态：JOBS 无运行任务，GPU utilization=0；Console CUDA 已初始化、allocated=0、reserved=2MiB，NVML 显示无法完整映射的448MiB驻留上下文。另一个项目已承诺此实验期间不新增 GPU 任务。以上不是独占GPU证明；实际每进程前后再记录 GPU 状态与驻留进程。

先编译，再依次运行三个独立进程；仅本目录可写，不修改九齿或生产 Hadamard 仓库。

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST=8.9
python run_experiment.py --build-only --reference-repo /data/infinitensor-2026/fast-hadamard-transform --output results/build.json
python run_experiment.py --run-index 1 --reference-repo /data/infinitensor-2026/fast-hadamard-transform --output results/run1.json
python run_experiment.py --run-index 2 --reference-repo /data/infinitensor-2026/fast-hadamard-transform --output results/run2.json
python run_experiment.py --run-index 3 --reference-repo /data/infinitensor-2026/fast-hadamard-transform --output results/run3.json
```
