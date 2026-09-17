# 已预登记的相对稳定性确认：未采用

方案在新采样前固定于1f97d437370af9f5c102797308ad5fd1a8db9bef的preregistered目录。本次没有修改计算源码或二进制，也没有重算旧数据追认通过。8份新输入、旧245份源码与3个已测.so均核对SHA；Torch/NumPy/CUDA/计算能力匹配。旧完整CLI/数值验证继续保持原有证据范围，按登记的additional_correctness还在新precheck目录重做512项兼容检查并通过。

## 新数据结果

三个独立进程、完整588设备/588普通调用、24目标分组、18同源校准均执行完毕。三轮分别ACCEPT/REJECT/ACCEPT，最终保持 **REJECT**。

唯一未过项：round1（从0计）FP16 packed_fused / rows16387 / N2 / 256线程 / allocating。

- 独立中位数比：1.031310918，组内配对比值中位数：1.029463958；都在原有5%退化限值内。
- 配对比值IQR/中位数：**11.840853%**，超过新采样前登记的10%相对稳定性限值。
- 其余设备、主机、目标分组及同源校准条件通过；设备目标分组几何平均范围1.270161–1.407428倍，仅是未采用候选在本轮RTX4090D上的特定Graph结果。

该点未显示中位耗时退化，但仍不足以满足已登记的稳定性要求。没有删去该点、提高限值、再补跑直到通过或将旧REJECT改为ACCEPT。生产内核继续保留，停止对这条候选的重复确认；后续工作转向其他有证据的热点。

## 定义与证据边界

本次事先明确将每版本绝对IQR保留为诊断标记，采纳稳定性针对配对相对时间。5%双口径约束、1.10倍设备分组、完整矩阵和三个进程没有缩减。五项合成检查覆盖共同漂移、单侧波动、真正退化、缺记录/错时间线。新数据仍未达新定义，故结论不是对先前失败的改写。

主机墙钟包含同步区间内GPU等待；CPU/线程时间与GC/调度计数用于排查，不能直接证明波动根因。绝对时间不得跨显卡或会话直接相除。

## 完整数据恢复

21份新记录，raw JSON包30069999字节，SHA256 9b44cfad1ea2680e26c8260090812b4c6f66f3010f4dc10b582e538ec1cceef3。无损恢复和逐文件校验已完成，实例随后确认关机。

```bash
python tools/recover_export.py transport/EXPORT_RECEIPT.json transport decoded
python tools/analyze.py decoded/results-remote/benchmark.json --output audit.json
```

原有源指纹、登记脚本和固定协议在同级preregistered目录，旧CLI与接口完整验证在704b0b3625cf93048b21ec47ac578f9bc0c32c0b/results/integrated-pairs-20260917。run_relative_confirmation.py仅封装已登记的512项前置检查；其结果独立输出，未覆盖旧数据或改动计时脚本。
