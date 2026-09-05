# NVIDIA 256线程候选：独立输出Graph复核

本目录固定保存2026-09-05 UTC的实验源码、三轮结果与复算程序。原24个目标和48个邻近配置三轮均满足相对同轮128线程减少耗时至少5%，范围6.53%～25.58%。默认128未因这项实验自动改变，不能外推到全部M/N或A100。

详细范围、最弱配置、环境与限制见[实验报告](RESULTS.md)，预定条件见[实验说明](EXPERIMENT.md)。不同于早期复用同一输出的筛选，每个方法图中保留64份独立输出，25次API预热、5次图预热、20次重放、5组；三轮是独立进程。所有结果包含图调度均摊开销，不是独立单kernel延迟或物理带宽。

## 原源码与修改对应

- [source_manifest.json](source_manifest.json)：构建源文件的逐字节SHA。
- [原包装器](sources/torch_binding_original.cu)、[实验包装器](sources/torch_binding_experiment.cu)、[改动差异](binding_changes.patch)：仅增加显式128/256线程入口并限制本次N=16/64。
- [kernels.cuh](sources/kernels.cuh)：原内核逐字节复制，数学未改。
- [run_experiment.py](run_experiment.py)：实际验证和图计时程序；固定版本Dao来源核查在[sources/compare_reference.py](sources/compare_reference.py)。
- [archive_manifest.json](archive_manifest.json)：本归档全部文件的SHA与来源类别。

## 三轮原始记录

[run1.json](results/run1.json)、[run2.json](results/run2.json)、[run3.json](results/run3.json)保留336个不同输入用例、72个性能配置的原始组样本和前后设备快照。复跑不增加独立用例数；128/256及CPU量化核查均未改变容差。

[transfer_manifest.json](results/transfer_manifest.json)记录10个原始文件的服务器SHA与大小，归档时全部比对通过。扩展二进制不发布，只保留SHA、构建日志与参数。未知448MiB驻留上下文的归属未能完整映射，因此不声称独占GPU。

归档使用局部`.gitattributes`关闭换行转换，保留实际测试文件的原字节。`archive_manifest.json`分别记录raw与CRLF→LF规范化SHA；后者只用于源码内容比较，不替代原始SHA，也不代表重新构建的二进制一定逐字节相同。

复算：[analyze_results.py](analyze_results.py)读取现有JSON，生成[完整72配置比较](results/comparison.csv)和[摘要](results/summary.json)。运行`python analyze_results.py`会重新生成派生文件，不覆盖三轮原始JSON。重新跑GPU实验时，先复制整个归档到新的工作目录，并使用[实验说明](EXPERIMENT.md)中的新输出路径，以保护历史记录。
