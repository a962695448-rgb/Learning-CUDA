# A100 规则与两卡调用方式验证档案

实际最终源码：`24dfef776ad73a4128cb6138674c5886c21e49c0`。先读[实现、调用示例、完整结果与限制](../../reports/a100-calling-validation.md)。

| 入口 | 内容 |
|---|---|
| [manifest.json](manifest.json) | 原始文件及派生分析的精确字节 SHA-256，私有原生时间线和会话元数据的省略清单 |
| [A100 校准](a100/calibration/runs/) | 450 配置正确性、三轮筛选、冻结规则、三轮留出；接受 9 条、拒绝 N16 融合，原始负例完整保留 |
| [A100 最终运行](a100/final/) / [4090 最终运行](rtx4090/) | 最终生产源码、全量正确性、三轮 Graph 与三轮调用方式对照、时间线和权限失败 |
| [A100 分析](a100_final_analysis.json) / [4090 分析](rtx4090_final_analysis.json) | 完整收益范围及所有未达 5% 的观测，回退和 eager 分开统计 |
| [A100 Graph 毫秒表](a100_final_graph.csv) / [4090 Graph 毫秒表](rtx4090_final_graph.csv) | 各轮 original128、original256、auto 及实际比较基线 |
| [A100 eager 毫秒表](a100_final_eager.csv) / [4090 eager 毫秒表](rtx4090_final_eager.csv) | 参数传递写法与旧默认调用的独立比较，不与 Graph 混算 |
| [早期 A100 调用诊断](eager_analysis.json) | 冻结 a3703fd、A100 尚无自动规则时的控制实验；不能代替最终源码数据 |
| [trace 范围复核](a100/final/checks/trace-scope-audit.json) | 原脚本误判保留 FAIL，按变换/融合模板分别检查原始 trace 的独立复核 |
| [analyze_final.py](analyze_final.py) | 从附带两份小型原始计时 ZIP 离线复算，不需要 GPU |

`a100/production/` 是原 PATH 遗漏导致 Ninja 不可见的失败运行；`a100/production_retry/` 是修正环境后的 a3703fd 完整通过；`a100/final/production/` 和 `rtx4090/production/` 才是最终 24dfef7 源码。各版本输入 manifest 和二进制哈希分别保存。

原始日志、数值和失败没有删改。原生 `.nsys-rep`、`.sqlite` 与会话元数据保存在私有完整 ZIP；公开保留其哈希及省略原因。两个最终计时 ZIP 是原样下载的小型数值包，和展开记录重复提供以简化离线复算。`.gitattributes` 防止原始 CSV/日志换行转换；README、清单自身不计入清单中的文件哈希，避免循环引用。

`verify_policy.cpp` 和 `baseline_row_policy.hpp` 是实际执行的主机新旧选择比较；前者保留当时相对工作区路径。它验证选择逻辑，不是 GPU 性能实验。全部实验和对照的数量按独立范围描述，不累计跨模式、跨进程或跨设备重复来增加测试数。

另提供仅调整 include 路径的 `verify_policy_portable.cpp`，搭配冻结的 `row_policy_24dfef7.hpp`。该副本已在主机重新编译运行，结果同为18216个旧设备组合及5项A100边界断言通过。在本目录可执行：

```bash
WORK=$(mktemp -d)
g++ -std=c++17 -O2 verify_policy_portable.cpp -o "$WORK/policy-check"
"$WORK/policy-check"
```
