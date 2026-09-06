# NVIDIA 显式融合布局生产整合验证

本目录拥有生产源的独立工作字节快照，主仓库未被本任务改动。基线为 `217c30ff5e78842cd5809de6bf78ee8a7f04fc54` 加已审阅工作区变更；13个源码的原始SHA、LF归一化SHA和工作diff记录在 `source_manifest.json`、`working_diff.patch`。快照后不会自动同步主树变化。

`protocol.json` 固定完整回归与52个新邻近M。controller、检测器和分析器先做本地静态检查，GPU窗口由协调者安排，不能把准备完成当作实机通过。

## 串行回归

```bash
python freeze_manifest.py
python run_regressions.py --output-directory runs/regression --reference-repo /path/to/pinned/fast-hadamard-transform
```

回归依次运行原CPU参考测试、原默认CLI1876+15参数负例、原256线程CLI1876、可选融合CLI1876、17个新增CLI边界条件、4个CSV格式小检查和18/20列旧文件保护、原1800三方/线程矩阵、28个metadata检查、16个生产接口/错位/非默认stream条件。候选CLI只在212个N256输入用新融合；三方矩阵的200个N256输入是原1800的子集。不同执行模式与重调用不算新输入。

原验证脚本使用不同扩展名：线程兼容模块独立记录SHA；metadata、定向检查和后续三轮性能复用同一生产扩展binary。所有文件与binary身份在回归报告中保存。每个GPU subprocess前记录连续两次空闲，最多等待60秒；每阶段保留PID、命令、退出码、完整日志。失败立即停止依赖工作，输出不覆盖。

CSV小检查只验证21列格式、ms/us换算、布局标签作用范围和旧文件字节保护；其28行不是性能优化证据。不会重跑旧16配置CLI性能矩阵或原24配置Graph研究。

## 52个固定邻近配置

仅在完整回归报告PASS后：

```bash
python run_holdout_suite.py --regression-report runs/regression/regression_report.json --output-directory runs/holdout
python analyze_holdout.py --regression-report runs/regression/regression_report.json --output derived \
  runs/holdout/run1.json runs/holdout/run2.json runs/holdout/run3.json
```

M固定为2、3、16、18、63、65、255、256、258、4095、4097、16383、16385；均不在初轮矩阵。N256、128线程、两dtype、scale1/1/16，共52配置，只比较生产API的original与contiguous256融合路径。普通transform/quantize_only没有性能搜索。

每进程先完成52×7输入×两指针偏移=728个正确性条件，含原/新/分步位一致、CPU实际量化、FP64抽样以及输入/guard不变，再计时。三进程复用相同production binary；64个独立输出tuple/Graph、20replay×5组、固定重排顺序，保留1560个原始事件样本。仅同轮同配置作分母，报告每轮至少5%的稳定收益以及所有超过3%的退化，无自动派发或跨硬件保证。

初轮24配置及独立lazy-neg问题属于先前证据，不混入52配置样本；新接口的lazy拒绝/物化行为由28项metadata回归重新验证。
