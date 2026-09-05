# Ascend 910B1 原生探针

第一阶段只验证真实 NPU 编译/发射链路：256 个 FP32 Add、32B 对齐 payload、前后 32B guard、输入不变和同一 stream 的异步复制。尚不代表短长度 DataCopyPad、2B 偏移或 FP16/BF16 RNE 已通过。

使用现场 CANN 9.0.0 提供的 CMake，不复制 SDK 源文件。`Ascend910B1` 已由主代理调用现场 `aclrtGetSocName()` 获得。

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cmake -S tools/ascend_smoke -B build/ascend_smoke \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.0 \
  -DSOC_VERSION=Ascend910B1 -DRUN_MODE=npu -DCMAKE_BUILD_TYPE=Release
cmake --build build/ascend_smoke -j2
./build/ascend_smoke/ascend_smoke --stage add
```

保存配置、编译和运行日志及源文件 SHA；只有实际 NPU 输出 `ADD_PASS` 且退出 0 才算本阶段通过。CMake 明确拒绝 CPU/模拟器模式。

后续阶段等待对照现场 DataCopyPad/Cast 原型和结构字段再添加，不以新版本网页原型替代现场 SDK。
