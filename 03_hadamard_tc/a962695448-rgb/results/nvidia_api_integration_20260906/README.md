# NVIDIA 可选线程接口：隔离集成验证

基于`8f75553a074d79b850294377d5aea6381e93da19`，只向CLI和PyTorch添加显式128/256线程选项，默认128不变；没有自动派发。此目录固定保留通过真实4090验证的源码快照、28个原始结果和构建记录。

## 源码对应

- [source_manifest.json](source_manifest.json)：全部实际构建/执行文件SHA。
- [main.cu](project/src/main.cu)、[torch_binding.cu](project/src/torch_binding.cu)、[verify_block_threads.py](project/scripts/verify_block_threads.py)：本次集成的三个文件。
- [kernels.cuh](project/include/kernels.cuh)、[reference.hpp](project/include/reference.hpp)、原Dao对照脚本均保持原数学与oracle，未为候选修改参考结果。
- [integration_runner.py](integration_runner.py)：串行编译、回归和六配置Graph冒烟流程；[graph_measure.py](graph_measure.py)复用前一阶段的独立输出图方法。
- [archive_manifest.json](archive_manifest.json)：归档文件与来源类别。

正式源码已固定于[`24849f6`](https://github.com/a962695448-rgb/Learning-CUDA/commit/24849f61ef06350f4e8bcd224ef93d97622c9744)。实机执行对应前述8f75553基线加补丁及原始源码SHA；提交之后仅通过只读`git show`核查10个NVIDIA构建/验证文件的LF内容与测试快照一致，该commit未另行重跑GPU。核查字段见[source_equivalence_main_lf.json](source_equivalence_main_lf.json)。

源码补丁SHA为`3b480221f0b875e856b4bb32ec8a58d24ff91f619319caa0698fe3cd5bc476e4`；对应三文件实际内容见本目录快照及source_manifest。此值用于关联当时的补丁，不代表二进制SHA或后续仓库commit。

### 原始字节与Git源码内容

测试时Windows工作树包含CRLF/LF混合换行，服务器按原字节构建；主源码经Git处理后的raw SHA可能不同。归档使用局部`.gitattributes`关闭换行转换，保留实际测试源的原字节，以匹配`source_manifest.json`。

[source_equivalence_main_lf.json](source_equivalence_main_lf.json)将测试时10个project文件与应用补丁后的主树逐文件对照：**仅将CRLF替换为LF后，全部SHA一致，没有真实代码内容差异**。例如`main.cu`测试raw为29945字节（516个CRLF），主树raw为29996字节（567个CRLF），LF规范化后都为29429字节。清单同时记录raw与LF两套SHA，不把它们混称为直接原字节一致。后续Git工作树换行可能变化，LF内容比较与原始构建SHA应分别核查。

## 验证范围

| 范围 | 结果 |
|---|---|
| 原1,876组CLI矩阵 | 默认128、显式256均PASS；最大绝对误差0.0078125，CPU/分步/融合INT4精确一致 |
| 原1,800组Dao矩阵 | PASS；每个输入默认=128=256位一致，FP16/BF16对Dao最大绝对差0 |
| 拒绝检查 | 既有15个CLI、10个张量输入；新增11个CLI和27个PyTorch线程参数检查均PASS |
| 旧调用、非默认stream | PASS；设备guard保留，单卡环境未实测多GPU切换 |
| 单位与CSV | 保留原mean_us，追加mean_ms=mean_us/1000；旧表头拒绝追加，原文件SHA不变 |

两遍1,876矩阵仍是同一范围，每个1,800输入的三种调用也不能计为5,400个不同用例。35个warp相对稠密舍入参考的容差内差异保留，没有删去负例或放宽阈值。

六个已经测过的Graph代表配置仅用于确认集成未丢失效果，256线程相对128减少耗时7.12%～25.34%。图中保留64份独立输出，所有图输出在计时前后核查；同轮比较使用5组中位数。不是新三轮大搜索，不是端到端时间，也不能外推全部M/N。

## 原始文件与重现

[integration_summary.json](results/integration_summary.json)列出全部命令、退出码、环境、源码SHA和六配置原始采样，包含us/ms中位数。[validation_integration_default.log](results/validation_integration_default.log)、[explicit256_matrix.log](results/explicit256_matrix.log)、[pytorch_1800.json](results/pytorch_1800.json)保留完整检查结果。

[transfer_manifest.json](results/transfer_manifest.json)内的28个服务器文件在归档时全部重新比对SHA/大小。独立module、目录与构建缓存未覆盖生产缓存；二进制不发布，其SHA为：

- CLI：`e34ed01a8b436abbb039884263a4eb97702939bda08ef77ef6445f2a09268163`。
- PyTorch：`b8243f890b2285585e523fab9eaee4bf35faf4452ad6351ac60bbc76d3d8f57d`。

要在相同服务器目录布局下复现完整隔离验证，请向不含`results/`的新工作目录复制`project/`、`source_manifest.json`、`integration_runner.py`和`graph_measure.py`，设置CUDA_HOME、PATH、MAX_JOBS=1，再运行`python integration_runner.py`。其他目录布局可直接按主README运行当前源码的CLI自测与可配置参考库路径的`verify_block_threads.py`，为JSON选择新文件名。原始环境日志有服务器本地文件路径但不含访问URL、凭据或编译二进制。

测试前GPUUtil=0；已有无法完整映射的448MiB驻留上下文，不作为独占证明。所有验证进程已退出0，GPU恢复空闲。A100验收独立进行。
