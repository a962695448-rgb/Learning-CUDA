# 可复用 CUDA 输出缓冲区：验收记录（2026-09-17）

基线 `bce004ecb2097a20207080f7507e4cfd2ee49d06`。新增 hadamard_out、hadamard_int4_out、quantize_int4_out，写入调用者提供的输出并返回 None。原分配式 API、默认参数、路由和设备核函数源码不变。

## 收益与口径

普通 Python 调用、同设备、驻留输入，旧分配接口与新预分配输出接口逐组交替。输出预分配不计入 out 计时，旧接口计入其输出分配与返回对象处理；每组 2000 次调用，组前后 CUDA 同步，九组取中位数。72 个配置（54 个目标、18 个大批量对照）× 三轮 = 216 组比较，同时复查候选中的旧接口。

| 接口 | 三轮目标配置的几何平均加速比 |
|---|---|
| Hadamard | 1.747 / 1.756 / 1.751× |
| Hadamard + INT4 | 2.008 / 2.023 / 2.006× |
| INT4 | 2.030 / 2.046 / 2.035× |

目标为 rows 1/17/256、dim 8/64/256、FP16/BF16；对照 rows 65536。每方法每轮目标几何平均至少快 10%，每配置 out 或原接口均不得慢 5%，全部满足。原接口观测最大回退为 2.765%以内。不是 CUDA Graph 计时，也不代表模型端到端或包括初次缓冲区分配的收益。

## 正确性及调用契约

- 372 组正常调用配置，每组更换两份输入，与独立 NumPy 参考逐位一致。覆盖 2D/4D、FP16/BF16、所有二次幂维度、线程和合法布局组合；检查缓冲区地址、边界哨兵和版本计数。
- 63 项拒绝检查：错误形状、dtype、设备、步长、grad、lazy-negative/inference 元数据、布局与任何活动字节重叠，均在写入及版本修改前拒绝。DLPack 别名、部分重叠和两个输出重叠均覆盖。
- 三接口的已保存反向传播输入能够检测到输出被修改；inference_mode 内调用通过；预热后的 50 次调用无额外 PyTorch 输出张量显存分配。允许同一分配内字节区间互不重叠的视图。
- 24 组非默认流/Graph 配置通过，含 96 次变更输入重放和 48 个错误对照。

采用连续张量的实际字节区间判断重叠，避免依赖 storage 对象身份。原地写入更新 PyTorch 版本计数；其必要性见 [PyTorch 2.6 TensorImpl 说明](https://github.com/pytorch/pytorch/blob/v2.6.0/c10/core/TensorImpl.h)。[ATen overlap 实现](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/MemoryOverlap.cpp)说明了只判断 storage 关系的边界。

首次工具仅捕获 RuntimeError，非法布局实际返回 ValueError，导致负例工具提前退出；实现已正确拒绝该调用。扩展异常捕获后完整通过。initial/ 保留首次工具、日志和部分结果；核函数及 C++ 实现未为此改动。Triton 的已通过测试未在 CUDA 工具重试时重复运行。

## 复现

检出固定基线仓库，在匹配 RTX 4090 D / NGC Torch 2.6 / CUDA 12.8 的环境中执行：

```bash
python prepare_inputs.py --baseline /path/to/Learning-CUDA --work /tmp/out-repro
API_OPT_CUDA_ONLY=1 python /tmp/out-repro/run_gpu.py
```

工作目录须不存在。脚本逐 Git blob 核对基线并覆盖 source/ 中的候选文件，按真实测试结构生成输入清单。单独正确性入口为项目 scripts/verify_out_buffers.py。编译统一使用 sm89，精确环境、扩展、运行器、协议及输入清单指纹保存在 summary.json。

复用意味着下一次调用会覆盖旧输出；需要保留时先复制。输出不自动调整形状，不允许任意输入/输出活动字节重叠；跨流使用和释放由调用者管理同步与生命周期。CUDA Graph 重放按 PyTorch 的规则执行，主机端版本更新发生于调用/捕获阶段。本轮是单卡 NVIDIA 验证，其他设备和框架编译模式不据此声称已覆盖。
