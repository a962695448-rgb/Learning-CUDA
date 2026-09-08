# A100 N256 融合布局复验

实际源码来自 `35f79b9cc595b0459a6172e0e2a63dcb9f4854af`，与 `155a05a` 的融合实现一致。原始包包含冻结源码、sm80 编译产物、完整回归、三个独立进程的原始计时与 GPU 样本位模式。86 个原始文件逐字节核验后独立复算，未导入 4090 的计时样本。

- [报告](../../reports/fused-layout-a100-validation.md)
- [三轮逐配置比较](comparison.csv)、[1560 个原始计时样本](samples.csv)、[离线校验摘要](analysis.json)
- [原始证据 ZIP](raw.zip)、[内部文件清单](raw_manifest.json)、[发布清单](archive_manifest.json)

复现离线检查：解压 `raw.zip` 到新目录后，运行 `python analyze.py <解压目录> --output <新的输出目录>`。依赖 NumPy；不需要 GPU。脚本先校验冻结文件，再复算 364 份去重精确证书及全部计时换算。

原始 4090 V2 的 `protocol_v2.json`、`checks_v2.py` 按原文保留以复用数值检查。本次 `run_a100_holdout.py` 只使用其固定输入矩阵、计时设置和 `check_input` 数值逻辑；旧设备、旧二进制及旧回归记录未用作 A100 通过凭据。A100 的设备、编译目标、新回归与二进制散列均单独核验。

结果：51/52 配置三轮均减少耗时；47/52 每轮减少至少 5%。BF16 M=16385、scale=1 在第三轮退化 8.43%，前两轮分别改善 7.30%、7.30%；完整负例保留，不改为“全部稳定加速”。默认仍为 original。
