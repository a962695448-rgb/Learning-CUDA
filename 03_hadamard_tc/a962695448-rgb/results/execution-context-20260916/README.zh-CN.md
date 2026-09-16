# RTX 4090 D 执行上下文回归（2026-09-16）

受测计算源码来自提交 `4049323cf400c73bd160f19b6c8b7defc678bf01`；原 51 份源码/构建文件保持字节一致，仅新增 `verify_execution_context.py`。代码和结果均保存 SHA-256。

## 实测结果

- 120 组配置全部通过：FP16/BF16、128/256 线程、2D/4D、dim 1/8/64/256，以及合法的 original/auto/packed/contiguous256 组合。
- 三种接口：`hadamard`、`hadamard_int4`、`quantize_int4`。
- 120 次非默认流生产者/消费者检查、480 次原地替换输入后的 Graph 重放，与独立 NumPy FWHT/量化参考逐位一致；输入保持不变。
- 每组各一个故意放错流、一个保留旧输出的负例，合计 240 个，全部检出。负例不能检出即整次失败；不计为额外正确性用例。
- 新鲜编译加验证约 128.54 秒，不能解读为内核性能。未修改计算内核，也不宣称新的加速比。

## 复现

检出上述 CUDA 提交，将本目录 `verify_execution_context.py` 复制到项目的 `scripts/` 目录，保留原构建脚本，进入 `03_hadamard_tc/a962695448-rgb`：

```bash
export PATH=/usr/local/cuda/bin:$PATH
python scripts/verify_execution_context.py --build-directory /tmp/hadamard-context-build --json /tmp/new-context-report.json
```

输出路径必须不存在。完整环境见 `environment.json`：RTX 4090 D、sm89、Torch 2.6.0a0+ecf3bae40a.nv25.01、CUDA 12.8、NumPy 1.26.4。原始用例、错误对照和源码指纹见 `cuda-context.json`。

## 测试方法与边界

数据在非默认流中延迟写入，随后调用受测接口。GPU 张量在同步结束前保持存活，提前预热两个流上的分配；错误对照刻意让消费者运行在默认流。测试使用 PyTorch 的内部测试辅助函数 `torch.cuda._sleep`，缺失该函数或当前环境使对照失去区分能力时，明确失败，不静默跳过。

Graph 在旁路流上预热并捕获，每次重放前更新同一输入内存，检查四种输入及输出和输入不可变性。规则来自 [PyTorch 2.6 CUDA 流与 Graph 文档](https://docs.pytorch.org/docs/2.6/notes/cuda.html)，辅助函数定义见 [PyTorch v2.6.0 源码](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/cuda/__init__.py)。

这是单设备代表性配置的正确性补测，未覆盖任意流调度、多 GPU 或新 A100 环境；与历史计时保持独立。
