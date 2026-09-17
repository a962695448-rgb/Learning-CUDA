# 验证证据与完整档案

本提交保留实现、必要测试、平台入口、简明报告及摩尔的核心复现材料。
完整开发分支和历史实验不随此次精简删除。

## 固定档案

- [完整归档提交 3676727](https://github.com/a962695448-rgb/Learning-CUDA/tree/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb)
- [历史 results 目录](https://github.com/a962695448-rgb/Learning-CUDA/tree/3676727fc21ea27bda668d743d10e52d7e138f68/03_hadamard_tc/a962695448-rgb/results)

该 results 目录的 2,586 个历史文件通过固定提交引用。日志、失败记录、数据和
实验脚本仍可在个人仓库查询；本 PR 的功能和测试不要求下载全部历史归档。
复现某次历史实验时，应使用该实验清单指定的源码、环境和输入；历史说明中的
results 路径属于完整档案检出。新运行可以在当前目录创建自己的结果目录。

## 重点审查入口

| 内容 | 入口与范围 |
|---|---|
| NVIDIA 最新生产验证 | [A100/4090 调用方式与自动规则](reports/a100-calling-validation.md)，原始数据链接固定到对应档案 |
| 独立参考与线程配置 | [A100 跨卡报告](reports/a100-validation.md)，包含固定 Dao 对照与全部负例 |
| 小维度行打包 | [A800/4090 报告](reports/packed-rows-validation.md)，按设备和 Graph/普通调用分别说明 |
| N256 显式融合布局 | [4090](reports/fused-layout-validation.md)、[A100](reports/fused-layout-a100-validation.md) |
| 五种国产平台 | 各 platforms 子目录 README，分别列出 SDK、设备、正确性和本机基线 |
| 摩尔原始计时与再生成 | [公开复现材料](platforms/moore/repro/README.zh-CN.md)，含 6,750 条计时、冻结对照和夹具生成器 |
| 代码来源与本轮检查 | [SUBMISSION_MANIFEST.json](SUBMISSION_MANIFEST.json) |

## 首轮精简提交的验证边界

初次精简阶段没有改变 CUDA/MUSA 等计算源码。保留代码的 Git blob SHA 与固定档案一致；
当时的工作是精简提交、整理文档链接和执行 CPU 检查。此前 GPU 结果仍归属于
其清单明确记录的实测版本，不将本轮整理写成新的 GPU 实验。

本地 CPU reference 检查、摩尔离线复算及数据完整性检查通过。原始 GPU 结果中
测试范围、跳过项、慢例及计时方式均保留，跨设备及不同计时口径不直接相除。

## 2026-09-17 独立量化入口

新增显式 packed 量化入口的完整范围见 [报告](reports/quantize-packed-20260917.md)。
[固定档案](https://github.com/a962695448-rgb/Learning-CUDA/tree/29bb34a0b2817e282a9554bd92c3911a3c010762/03_hadamard_tc/a962695448-rgb/results/quantize-packed-20260917) 包含最终双重性能门槛、正确性检查、源码和复现准备工具，也保存新增参数引入主机开销的初版实验。当前源文件与新测试哈希已在 SUBMISSION_MANIFEST.json 更新；未将先前的跨卡或国产平台记录当成本轮新入口的实测证据。
