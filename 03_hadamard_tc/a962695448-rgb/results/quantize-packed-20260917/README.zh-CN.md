# 独立 packed INT4 量化验证档案

基线：a962695448-rgb/Learning-CUDA 的 `28fb3a95ca8bcdf53a66949cfde23820bf7e4005`。

- `final`：独立入口的 700 组新正确性检查、4 组非法条件、16 组流/Graph 检查，372/63/4 项原接口检查，以及 240 设备 + 240 主机性能记录。
- `initial`：添加 row_layout 参数的第一版；设备门槛曾通过，但 21/120 条旧 out 接口主机记录退化超过 5%，因此最终未采用。原始 ACCEPT 仅指最初预设的设备门槛，不能当成最终方案已接受。
- `PROTOCOL-CUDA-FINAL.json`：复测前增加逐配置主机门槛；设备分组几何平均至少 1.10 倍、任一设备/主机配置不退化超过 5%，全部通过。
- `source`：最终实际测试的 CUDA binding 与独立正确性脚本；其余代码由验证过的基线补齐。新入口为 quantize_int4_packed / quantize_int4_packed_out，保留旧接口签名。

## 复现

在具有 NGC Torch/CUDA 编译环境的 RTX4090D 上检出基线，从本档案目录执行：

```bash
python prepare_inputs.py --baseline /path/to/Learning-CUDA --work /tmp/quantize-replay
MQ_CUDA_ONLY=1 python /tmp/quantize-replay/run_gpu.py
```

准备工具检查基线每个 Git blob，并叠加 source。目录必须不存在；不要复用旧输出目录。本轮使用 sm89、CUDA 12.8.61、Torch 2.6.0a0+ecf3bae40a.nv25.01。MQ_CUDA_ONLY=1 复现 CUDA 部分，不依赖另一项目。

设备计时来自 warmup 后的 64 节点 Graph × 10 次重放；普通 Python 调用单独以前后同步墙钟计时。分组设备几何平均为 2.87–3.17 倍，普通 allocating/out 整体几何平均约 1.36/1.89 倍；不外推为模型端到端或其他平台效果。

输入清单及运行日志按原样保存。历史清单中另一项目的路径只是原始输入记录；复现准备工具仅生成 CUDA 必要文件并重新计算输入清单。
