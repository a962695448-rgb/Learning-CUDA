# A100跨卡验收：公开证据档案

固定源码`12c76d8331ef7cf3fd4c8c14a049162559be4302`已在真实NVIDIA A100-SXM4-40GB上完成验收。运行于2026-09-06北京时间02:55:05～03:05:35；所有预定阶段退出0，最终设备空闲。结论、性能边界与负例见[验收报告](../../reports/a100-validation.md)。

## 文件结构与原始字节

- `retrieved/`：严格白名单内的48个服务器原始文本文件，包含源码、构建配置、日志、阶段状态和原始采样；没有编译二进制、访问地址或连接材料。
- [原始传输清单](retrieved/transfer_manifest.json)：48个文件的服务器大小与SHA；它本身是下载时形成的客户端清单，不冒充第49个服务器结果。
- [source_manifest.json](source_manifest.json)：上传前固定的源码锚点，含11个Git源文件的commit/blob/SHA及辅助脚本SHA；与服务器记录独立比对。
- [archive_manifest.json](archive_manifest.json)：本公开档案全部文件的大小、SHA、来源类别及换行信息。
- [.gitattributes](.gitattributes)：局部`** -text`，禁止Git换行转换，保护原始字节。

所有48个raw保持原样，未做脱敏改写。私有原交付仍保留。唯一需要调整的是**派生摘要**中的本地Windows输入目录：公开分析脚本将其输出为相对路径后重新生成摘要；数学验证和数据未变。原/公开脚本及摘要双SHA、变更说明在[public_derivation_provenance.json](public_derivation_provenance.json)，不混用原始与公开摘要的SHA。

`retrieved/source/README.md`反映固定提交当时的状态；本次A100实测结论以此档案及验收报告为准。参考库README里的已发表性能数字也不计作本项目的实测。

## 原始结果入口

| 文件 | 作用 |
|---|---|
| [state.json](retrieved/results/state.json) | 固定版本、阶段PID/退出码、硬件快照和完成状态 |
| [CLI默认矩阵日志](retrieved/source/results/validation_a100_default128.log) | 原1,876矩阵、15项拒绝检查、16组基准命令 |
| [CLI显式256矩阵](retrieved/results/cli_explicit256_original_matrix.log) | 同一原1,876矩阵的第二线程设置 |
| [CLI原始CSV](retrieved/source/results/benchmark_a100_default128.csv) | 110行kernel-only、CPU compute、host end-to-end数据及us/ms |
| [dao_default.json](retrieved/results/dao_default.json) | 原1,800矩阵、12组eager与12组Graph，保留两个Graph负例 |
| [api_threads.json](retrieved/results/api_threads.json) | 默认/128/256位一致、27项线程拒绝及stream检查 |
| [run1](retrieved/results/run1.json)、[run2](retrieved/results/run2.json)、[run3](retrieved/results/run3.json) | 预定72配置的三轮全部原始样本与336个输入检查 |
| [collection_metadata.json](retrieved/results/collection_metadata.json) | 实际binary SHA与cuobjdump的sm80证据；不含binary内容 |
| [reference_source_sync.json](retrieved/results/reference_source_sync.json) | 固定Dao原Git tree/commit的恢复核对 |

## 源码与构建对应

实际编译源保存在`retrieved/source/`；[main.cu](retrieved/source/src/main.cu)、[torch_binding.cu](retrieved/source/src/torch_binding.cu)、[kernels.cuh](retrieved/source/include/kernels.cuh)、[reference.hpp](retrieved/source/include/reference.hpp)的原字节可按source_manifest逐文件核查。

[默认扩展build.ninja](retrieved/build_default/build.ninja)、[线程扩展build.ninja](retrieved/build_checker/build.ninja)和[CLI构建配置](retrieved/source/build/config.txt)记录实际ARCH80构建。固定Dao为`e7706faf8d1c3b9f241e36860640ad1dac644ede`，其[setup.py](retrieved/fast-hadamard-transform/setup.py)、[许可证](retrieved/fast-hadamard-transform/LICENSE)和构建记录一并保留；安装后的模块出处及SHA在各结果JSON中核查。

独立环境使用Torch2.5.0+cu124、NVCC12.4.99、ABI=false；CUDA程序不是由4090的sm89产物直接复用。算子验证中的GPU actual由真实CUDA kernel产生，CPU用于独立参考；CPU性能基准单列，不冒充GPU结果。

## 离线复算

```bash
# 只读取本目录retrieved中的文本证据，不启动GPU或连接服务器。
python analyze_delivery.py
```

脚本输出[derived/summary.json](derived/summary.json)、[原12组所有比较](derived/dao_all_configurations.csv)、[三轮72配置全表](derived/promotion_all_configurations.csv)、[CLI各口径全表](derived/cli_all_measurements.csv)和[总配置表](derived/all_configurations.csv)。缺失、失败或不一致会返回UNVERIFIED，不筛去负例。

复算会更新派生摘要时间戳，但不修改48个raw；如需保持公开档案不变，先复制整个目录再复算。历史`worker.py`包含当时租期的停止边界，只用于审计，不应直接当作新租期启动器。

1,876、1,800和336分别记账，重复设置/进程不增加独立算法用例。默认仍为128；256性能结论仅覆盖预定N=16/64和指定M及M±1，不保证其他输入或所有硬件更快。
