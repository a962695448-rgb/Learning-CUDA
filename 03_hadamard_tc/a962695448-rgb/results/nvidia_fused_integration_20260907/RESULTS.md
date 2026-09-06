# NVIDIA 融合布局整合与V2邻近验证

**完整生产回归通过；V2的52个邻近配置三轮全部通过，融合耗时均降低。**41/52配置在每轮都降时至少5%，总体范围4.7837%～14.3440%，没有退化。保留默认original；contiguous256是N256、128线程的显式融合选项，普通transform和quantize_only不切换。

## 真实来源与硬件

实测代码为217c30f加已审核工作字节，13源在 `raw/source_manifest.json`。`raw/derived/source_equivalence_git.json` 将这些源与提交155a05a8b957bdf558ef93a2db1e3aea9fadf36f关联：13文件LF内容一致，10文件原字节一致，另3文件只差换行。没有把旧运行冒称新提交重跑。

2026-09-07重启后容器host为6c79eb7695b5，GPU UUID为GPU-bc61c62f-2e69-f382-e981-b932b23df2aa，均与前一会话区分。实际为RTX4090/sm89、driver570.124.06、CUDA12.8、Torch2.6.0a0+ecf3bae40a.nv25.01、NumPy1.26.4。重启后重新核验原27文件、回归报告及二进制SHA一致，复用已通过的完整回归，没有重跑该旧矩阵。三轮仅用当前同机同配置配对，不使用历史时间作分母。

production binary SHA：`eb3f03f28b7f993bfc3351a8afc1022ccc5f93301c95ed3163fd437f6e1f3468`。构建命令和sm89证明在原回归日志；不发布二进制。

## 回归、原失败与V2口径

原完整回归包含：CPU参考9组；默认128、原256、候选融合三配置各运行同一1876输入矩阵；原1800三方/线程矩阵，其中200个N256是候选子集；metadata28和定向16（2/4D、对齐/2B偏移、非默认stream）通过。32个CLI、27线程/5布局/10输入拒绝检查、28行CSV格式及旧18/20列表头保护通过。各范围分开计数，重复调用不算新输入。

原V1 holdout确实失败：49配置完整、688条件完成，0个性能样本。FP16 M16383/N256、scale1、normal95811的第8191行230列，原GPU与Dao相同为cc7e，而独立精确数学值正确舍入为相邻cc7d；二者相差1ULP=0.015625。旧额外FP64-rounded门槛不满足，原FAIL、源码和全部诊断未改。

V2保留全部52矩阵、种子、输入、计时，以及官方对固定Dao的FP16<0.01、BF16<0.05阈值，量化packed/scales仍全部位精确。仅将FP64辅助门槛改为推导的前向误差证书：整数FWHT求精确值；CPU逐stage FP32加scale；整数最近偶数存储位必须等GPU；用Fraction验证E32和实际storage误差之和。E64单独记录，不加入E32冒充精确界。公式、假设与独立审阅后冻结版本在 `raw/revised_v2/certificate_notes.md`、`protocol_v2.json`。

## V2结果

M固定为2、3、16、18、63、65、255、256、258、4095、4097、16383、16385，N256、两dtype、scale1/1/16，共52配置。每进程728正确性条件（7输入×两偏移），三进程重复不算2184种新输入；FP16/BF16每轮对Dao最大绝对误差均为0。原门槛本会失败的2个条件在每轮均保留并标记，V1不追认通过。

| dtype | 配置 | 三轮每轮降时≥5% | 三轮全部耗时变化 |
|---|---:|---:|---:|
| FP16 | 26 | 15 | 降低4.7837%～13.8542% |
| BF16 | 26 | 26 | 降低5.3895%～14.3440% |

其余11个FP16配置仍为正收益，但没有全部三轮达到5%；完整比较保留在comparison.csv。无任一轮退化或>3%退化。

CUDA Graph每方法64个独立输出tuple、每组20replay、5组取中位数、三独立进程重排顺序。1560个原始事件样本全部保留，原始ms及折算每调用ms/us可复算。它衡量捕获GPU工作及均摊replay调度，不是孤立kernel或端到端耗时，也不证明任意M或其他GPU收益。没有自动形状派发。

CPU离线分析VERIFIED，核验470个唯一样本数组并重算364份唯一证书；每份报告仍独立验证全部条件、来源与位数据。完整u16样本sidecar和行级分数证书随包保留。此验证完成后停止实验；GPU最终0%/2MiB且无compute进程。

## 复算与数据边界

```bash
cd raw/revised_v2
python analyze_v2.py --regression-report ../runs/regression/regression_report.json --output recomputed \
  runs_20260907/run1.json runs_20260907/run2.json runs_20260907/run3.json
cd ../..
python verify_archive.py
```

所有raw文件保持传输原字节；两个原传输清单、源/协议/二进制hash、重启会话、原FAIL、完整回归及V2结果均在包内。`.gitattributes`禁止换行转换，外层清单核验每个其他文件。未发布8MiB完整私有输入、编译二进制或访问凭据；完整输入已单独备份。其生成器、shape/seed、全元素hash及失败行完整元素保留，可在相同运行环境重现并核对。
