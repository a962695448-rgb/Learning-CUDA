# 2026-09-08 小维度行打包实测档案

实际生产源码：`a3703fda3cfd7aa1b45342210dc57fb030f0d904`。两卡分别构建，分别统计；A800 不能标为 A100。先读[实现、用法、完整结果及限制](../../reports/packed-rows-validation.md)。

| 入口 | 内容 |
|---|---|
| [manifest.json](manifest.json) | 304 个原始/派生文件的精确字节 SHA-256，原始 ZIP 指纹，未公开的原生时间线清单 |
| [A800 分析](a800/production_analysis.json) / [4090 分析](rtx4090/production_analysis.json) | 生产 Graph 与 eager 的全量统计和每个非正收益观测 |
| [A800 Graph CSV](a800/production_graph.csv) / [4090 Graph CSV](rtx4090/production_graph.csv) | 每配置每轮的 original128、original256、auto 毫秒及采用标志 |
| [A800 eager CSV](a800/production_eager.csv) / [4090 eager CSV](rtx4090/production_eager.csv) | Python 逐次调用的全部负收益记录，不能与 Graph 混算 |
| [A800 原始运行](a800/runs/) / [4090 原始运行](rtx4090/runs/) | 正确性、3 轮初筛、3 轮留出、3 轮独立收窄、内存/同步检查与规则接受/拒绝 |
| [A800 生产运行](a800/production/runs/) / [4090 生产运行](rtx4090/production/runs/) | 7 模式 CLI、1800 Dao 矩阵、280 定向条件、元数据/CSV、3 轮生产性能、nsys |
| [A800 环境探针](a800/hardware-check/) / [4090 环境探针](rtx4090/hardware-check/) | 真实设备/工具版本、编译运行结果与 `ERR_NVGPUCTRPERM` 原始失败 |
| [analyze_production.py](analyze_production.py) | 不依赖 GPU 的 event、单位、组中位数、配置集合和证据指纹复算 |

`a800/production/project/` 与 `rtx4090/production/project/` 为生产输入源码；设备根下的 `project/` 是旧基线，`candidate_binding.cu`/`packed_rows.cuh` 是独立候选，不能混用。各自输入 manifest 与服务器 binary fingerprint 说明实际执行版本。

原始日志数值和文本未删改，负例完整保留。只有原生 nsys/SQLite 二进制留在本地 ZIP，公开保留统计、日志、原始哈希。`restored_input_sources.json` 记录从冻结输入包补齐 CPU `.cpp` 文件的来源。源码选择、分析输出和实际 GPU 测量的职责分别标明；重复同一输入矩阵不能累计为更多独立测试。

本目录的 `.gitattributes` 禁用文本换行转换，以保持 CSV 和日志原字节。README 与清单自身不在 manifest 的被哈希文件列表内，避免循环引用。
