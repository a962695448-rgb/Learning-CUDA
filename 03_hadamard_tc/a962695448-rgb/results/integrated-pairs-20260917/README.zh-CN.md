# 在既有入口内整合成对蝶形：完整验证记录

本轮保持既有kernel符号及主机分派，仅改变设备实现和Torch/CLI两处网格。完整正确性通过，三轮分别ACCEPT/REJECT/ACCEPT；最终按预先冻结协议保持REJECT，尚不采用候选。

## 结果与未过判据

- 588个设备项、24个目标分组全部通过；588个普通调用项的独立中位数比和配对比值均符合5%退化限值。
- 18项同源码模块A/A校准通过。唯一未过项是第二轮 BF16 packed Hadamard / rows65536 / N8 / out 的绝对耗时离散度。
- 该点独立中位数比0.985208、配对比值0.987443，仍在5%范围；原版IQR/中位数49.732%，候选44.103%，超过预设10%。最后判定没有改写，也未删除样本。

| 方法 | 设备目标分组几何平均范围 |
|---|---:|
| Hadamard | 1.270357–1.400775× |
| 融合INT4 | 1.314354–1.407615× |

全部普通调用的最小比值为0.964108，配对口径最小为0.964669。比值为原版/候选，不能把它与其他显卡或先前不同协议直接相除。

## 波动点的可核对事实

前12组约3.9–4.0微秒，随后两侧共同上升到约6.3微秒，末组回落。线程CPU时间基本跟随墙钟时间，36个测量块均没有GC回收记录；因此不能直接归因于GC或简单的失去CPU调度时间，也不能仅凭这些计数确定频率、驱动或其他原因。完整时间线保留。

## 正确性覆盖

- CLI候选original/packed/auto × 128/256线程的完整1876案例集合均通过。相同案例在6个配置下重复，不记为11256个独立输入案例。原版另核对original128及packed256。
- 两版均运行既有run_validation.py的15项CLI拒绝检查；候选CPU参考和行策略测试通过。
- 原版、同源副本和候选均通过128组分量/scale/偏移配置；候选24组流/Graph、512项溢出/抵消兼容通过。
- 原量化分量64组、singleton131072种存储编码、packed700正例/4拒绝/16上下文，以及372旧接口正例/63负例/4mutation/24上下文通过。未将重复调用累计为新增独立用例。

## 环境和范围

原RTX4090实例因无GPU已取消开机，本轮使用此前授权的RTX4090D；原版、同源副本、候选在同机重编。性能矩阵保持原49种配置、两种精度与线程数；改为三个新进程、18组平衡顺序、相同输出对象，并保存完整配对/CPU时间线。5%和1.10倍门槛不变，稳定性判据在运行前写入PROTOCOL.json。

实测：NVIDIA GeForce RTX 4090 D，Torch 2.6.0a0+ecf3bae40a.nv25.01、CUDA 12.8、NumPy 1.26.4。硬件和计时范围仅对应本轮，不作A100、其他国产平台或模型端到端加速声明。

245份输入运行前后相同。39份原始结果无损压缩回收，raw JSON包30503844字节，SHA256 8d9f873fb102d1749ad2b6d8df58efacaa2832e066f31bc26a504ab7a2f24022。工具独立复算REJECT/唯一稳定性失败项。实例随后已确认关机。

## 下一步边界

此时没有证据要求再次修改设备数学或网格。若继续确认，应先围绕共同漂移与相对性能稳定性制定新的、前瞻性的测量方案，保持本次REJECT原件；不得在本轮数据上改规则追认通过，也不得仅补跑失败点代替完整范围。

## 恢复与复算完整原始记录

完整记录使用无损gzip/base64分片保存，避免重复的大型时间线直接展开；没有删去任何测量块。

```bash
python tools/recover_export.py transport/EXPORT_RECEIPT.json transport decoded
python tools/analyze.py decoded/results-remote/benchmark.json --output audit.json
python -m unittest discover -s tools -p 'test_analysis.py' -v
```

恢复工具依次验证文本分片、gzip和原始JSON SHA256，再核对39份文件的内容摘要。summary/便于直接查阅CLI、源码清单和唯一失败项。原实验README作为INPUT_README.zh-CN.md保留，代表冻结时状态。

重建实机输入：从固定4516fd4仓库执行 `python prepare_inputs.py --baseline /path/to/baseline --work /tmp/integrated-replay`，全部245份文件须与原INPUT_MANIFEST相同，再从新目录执行 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py`。原版与候选在目标GPU上重新编译，不能复用其他架构成绩。
