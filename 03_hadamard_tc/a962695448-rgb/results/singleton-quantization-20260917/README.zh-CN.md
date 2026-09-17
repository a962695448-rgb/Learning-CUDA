# 单元素量化穷举与未采用的特化

基线：a962695448-rgb/Learning-CUDA 的 f9da76087f6cbf6aad52e9effa6cf3ac3aabccf0。

生产内核保持基线。本档案 source/include/packed_rows.cuh **是未采用的实验内核**，只供重现性能结论；新的穷举工具独立验证当前生产内核，不依赖采用特化。

- 两个版本均覆盖 131072 个存储编码、128768 个有限值的独立参考；2D/4D 尾行、128/256 线程、输入不变、哨兵和版本检查通过。
- 非有限编码的逐位比较只针对同一旧 packed 路径，不扩展 API 支持范围。
- 132 条设备与 132 条主机配对记录全部满足 5% 退化限制，但目标设备分组只有 1.0266–1.0437 倍，未达 1.10 倍预设门槛，故 REJECT。
- exhaustive-control.json 是当前保留的生产版本证据；exhaustive-candidate.json 和 benchmark.json 属于未采用的候选。

## 复现实验

在具有当前 NGC Torch/CUDA 的 RTX4090D 上检出上述基线后执行：

```bash
python prepare_inputs.py --baseline /path/to/Learning-CUDA --work /tmp/singleton-replay
CQ_CUDA_ONLY=1 python /tmp/singleton-replay/run_gpu.py
```

prepare_inputs.py 逐项验证基线 Git blob，再覆盖实验 source；使用新的输出目录。run_gpu_measured.py 是实际完整工作流，run_gpu.py 仅增加 CQ_CUDA_ONLY 开关，方便跳过另一项目的差分；其 CUDA 正确性和计时部分与实测脚本字节一致。测量清单保留原始路径、环境、源码及二进制 SHA，不含租赁或凭据信息。
