# 短尾 DataCopyPad 与 FP16/BF16 RNE 探针

独立于父目录正验证的 Add 包，不修改父目录源码。使用现场 CANN 9.0 DataCopyPad Ext/Cast count API；所有 GM 输入输出通过同一 aclrtStream 搬运，NPU kernel 采用保守屏障。本探针不用于性能评估。

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cmake -S tools/ascend_smoke/stage2 -B build/ascend_stage2 \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.0 \
  -DSOC_VERSION=Ascend910B1 -DRUN_MODE=npu -DCMAKE_BUILD_TYPE=Release
cmake --build build/ascend_stage2 -j2
./build/ascend_stage2/ascend_stage2 --stage pad --dtype fp16 --n 1
./build/ascend_stage2/ascend_stage2 --stage pad --dtype both
./build/ascend_stage2/ascend_stage2 --stage rne --dtype both
```

- `pad`：17 个短尾长度 × 2 dtype，共 34 次真实 NPU 往返；保持原始16位数据，无数值转换。
- `rne`：同样 34 个 case，FP32→dtype 的 CAST_RINT，再 CAST_NONE 返回 FP32；正负中点、相邻值与随机有限值对照自写整数 oracle。
- typed GM payload 偏移 34B（仅2B对齐），float payload 偏移68B（4B对齐）；头尾至少17元素guard。整块传输缓冲按64B取整，额外尾部同样必须保留0xa5。
- `--stage all` 执行共68个case；只有实际NPU全部通过返回0。报错保留输入bits/期望/实际及dtype/N，禁止改成截断参考掩盖RNE问题。

当前代码等待现场编译和运行，不把文档中的API存在当作此测试已通过。
