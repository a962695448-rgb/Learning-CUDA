# V2：参考库精度与精确前向误差证书

当前只完成CPU准备，尚未运行V2 GPU矩阵。原V1协议、27文件、FAIL和全部诊断保留；13生产源及production binary不变，复用实际全回归PASS。源与提交155a05a的LF内容对应证明单独保留，不把旧工作字节测试冒称新提交重测。

## 修订原因与不变项

V1在FP16、M16383、N256、scale1、normal seed95811发现第8191行第230列：原GPU和固定Dao均为−17.96875（cc7e），CPU逐stage FP32恰到相邻FP16中点，RNE选偶数尾位；精确数学值在中点另一侧，正确舍入为−17.953125（cc7d）。原参考库差为0，独立FP64已舍入结果差1ULP=0.015625。两种FP16结果间无第三个FP16值。本例直接/经FP32转换相同，不是双舍入bug。

官方FP16/BF16要求与参考实现的绝对误差分别小于0.01/0.05。V1额外把该门槛用于独立FP64已舍入输出，此处与所选参考库门槛冲突。V2保持参考库全部元素阈值、融合/分步/CPU packed与scales位精确、全部52个M/dtype/scale配置、7个输入/种子、两偏移和计时方法不变，只把FP64辅助硬门槛替换为有推导的前向误差证书。

V1仍为FAIL：49配置完整、688输入检查完成、0个计时配置，后两进程未启动。V2输出保存在独立目录，不追认V1通过，也不引入V1性能样本。

## 证书的三个硬门槛

每输入取首/中/末行并去重。将真实FP16/BF16位提升到共同2幂分母，在Python整数上做8阶段FWHT，再乘精确scale，得到数学值。FP64 dense独立计算，其自身与精确值的差只作诊断。

令u=2^-24、eta=2^-149、gamma9=9u/(1−9u)：

`E32 = gamma9*|scale|*L1(input) + eta/2*(|scale|*255*(1+u)^8+1)`

每个输出有256个叶和255个加减节点；每条输入路径最多8次加减加1次scale舍入。上述项分别覆盖相对舍入误差与逐渐下溢的绝对误差贡献。详见 certificate_notes.md。

1. CPU每阶段使用FP32加减，最后FP32乘scale；其对精确数学值的误差必须≤E32。
2. 通过整数算法对CPUpre独立执行FP16/BF16最近偶数舍入，全部样本存储位必须等于真正GPU输出位。
3. 单独计算实际storage舍入误差delta；GPU存储值对精确值的误差必须≤E32+delta。

所有决定使用Fraction精确比较，保存分数；浮点显示界向上舍入，不参与门槛。FP64误差E64不加进E32冒充精确界。模型要求有限值、无溢出、RN32和逐渐下溢；CPU环境预检以及现有NVCC构建参数核对记录在结果中。仍保留rounded/unrounded误差、ULP、舍入路径差异及旧辅助门槛本会失败的条件计数。

已对取回的真实GPU见证3行×256元素做CPU复算，768元素证书全部通过；这是旧执行数据的CPU复核，不是V2 GPU通过。原始样本位与证书将按每轮独立保留，供离线重算。

## 待审阅后执行

```bash
python freeze_v2.py
python run_suite_v2.py --regression-report ../runs/regression/regression_report.json \
  --reference-repo /path/to/pinned/fast-hadamard-transform --output-directory runs
# 三轮完成后，使用本目录分析器离线重算，不混入旧结果。
```

仅在独立审阅和新清单冻结后使用GPU。精确绑定原回归报告、原13源/27文件、production与Dao binary SHA；默认行为及项目源码保持不变。
