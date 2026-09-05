# Ascend 910B1 / CANN 9.0 FP32除法精度探针

此包仅准备实机编译与运行，**尚未证明任一路径精度通过**。不实现Hadamard，不提供CPU回退，不将编译成功视作数值通过。

沿用已实际通过的`ascend_smoke`原生ACL发射/固定对齐方式。`kernel.cpp`的actual全部由`__global__ __aicore__` kernel生成：

- `vector_div`：`AscendC::Div(LocalTensor<float>, ..., 256)`。
- `aicore_cpp_div`：设备函数内`GetValue`取数、C++ `float /`、`SetValue`写UB，并显式建立MTE2→S→MTE3事件依赖。源语言中的标量形式不保证特定物理指令；具体降低方式以编译器/实机结果为准。

## 两个独立构建目录

默认不编译scalar路径。若`float /`、标量访问或相关事件API不受该SDK/目标支持，保留其完整编译错误，vector构建仍可单独测量；不能改成host计算actual。

```bash
cmake -S . -B build-vector \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.0 \
  -DSOC_VERSION=Ascend910B1 -DRUN_MODE=npu \
  -DENABLE_SCALAR_DIV=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-vector -j1
./build-vector/ascend_div_probe --mode vector

cmake -S . -B build-scalar \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.0 \
  -DSOC_VERSION=Ascend910B1 -DRUN_MODE=npu \
  -DENABLE_SCALAR_DIV=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-scalar -j1
./build-scalar/ascend_div_probe --mode both
```

ON构建也可分别用`--mode vector`、`--mode scalar`运行。两条路径都选中时，vector出现精度差异不会阻止scalar继续检测，最终退出码仍为1。SDK路径与环境须沿用实际通过的smoke，不安装/替换驱动或工具链。

## 固定256个输入对

1. 32个非负值除以7：零、2^-24、2^-14、常见二进制分数、1附近FP16/BF16可表示值、65504，以及BF16可表示的大/小正规值。
2. 上述32个值的负号控制，包括负零。这组只测符号语义，不把负数当作真实maxabs。
3. 16种正maxabs，每种取8个正负/零/二进制分数x，执行`x / CPU_FP32(maxabs/7)`，共128对。
4. 64个普通正负除数控制。

第三组的分母是作为输入给定的host FP32参考scale，用于**隔离x/scale除法本身**；没有在NPU中串联maxabs/7与量化，也没有验证Cast/RINT或融合INT4。即使这里通过，也不能直接声称完整量化合同通过。

输入/输出范围预先固定为有限FP32正规数或零，不包含零分母、NaN、overflow或FP32 subnormal考核；不根据实机结果修改输入或期望。输入的FP16/BF16可表示值以明确FP32 bit pattern给定，不依赖尚未核实的设备Cast。

## 参考与判定

host参考使用实际FP32 `/`，输入及结果为volatile float，禁用fast-math和浮点合并，设置FE_TONEAREST，并用1/7、1/2、负零三项已知bits校准。所有最终比较按uint32 bits精确进行，不用容差，不将Div结果作为参考。

每条路径输出总mismatch、四组mismatch、前16个差异的lhs/rhs/expected/actual及各自bits；始终显示前4个样本。符号零差异单列但仍计入失败。未选中的路径标为NOT_TESTED，不能由零计数推断通过。

ACL输入上传、kernel发射、回读使用同一stream。输入完整字节与前后guard均检查；output先填充0xa5并核查两端32B guard，防止未写结果或越界被掩盖。数值不一致返回1，参数/未编译路径错误返回2。保留全部日志与构建选项，暂不进行性能计时。

API用法依据实际SDK快照：Div的Level2 FP32声明、`ascendc_compile_definitions`与SetFlag/WaitFlag/TPipe事件声明；未复制SDK实现。scalar设备编译与精度仍待实测。
