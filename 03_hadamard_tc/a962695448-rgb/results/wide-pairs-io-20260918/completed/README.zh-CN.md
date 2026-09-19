# 最后一轮：对齐成对读写与小批量回退

结果 ACCEPT；三轮共1050设备、2064主机、24目标组、18同源校准，无失败门槛。数值正确性和完整CLI、接口、流/Graph回归均通过。

源代码及测量协议在采样前固定于 a11ae35c5324864c4945311744e6b3aec2150e57 的同级 preregistered 目录，原预登记文件没有修改。原58配置全部保留，扩为89配置；输入/输出offset与4095/4096/4097阈值边界包含在内。

本目录25份base64分片保存38份完整原始文件（100058128字节），含全部采样、时间线、正确性、编译记录、三轮结果和实机摘要。恢复：

```bash
python3 recover_export.py FINAL_EXPORT_RECEIPT.txt . recovered
# 用同级 preregistered/tools/analyze.py 和原 PROTOCOL.json 独立复算
python3 ../preregistered/tools/analyze.py recovered/results-remote/benchmark.json --protocol ../preregistered/PROTOCOL.json --output audit-new.json
```

回收于2026-09-19，全部分片/gzip/解压内容/逐文件SHA核对；本地独立复算与远端ACCEPT一致。254冻结输入、协议和manifest指纹一致，受测二进制与只读cuobjdump诊断对应。硬件为RTX4090D，CUDA12.8.61，Torch2.6.0a0。没有在回收时重新测量，旧失败实验继续保留。服务器回收后已确认关机。
