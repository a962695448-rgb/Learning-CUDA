# 在现有入口内整合成对蝶形（尚未运行）

基线为已发布4516fd4，计算源码仍对应b49fff4。当前只准备候选，尚未CUDA编译、正确性或性能验证，不发布为优化实现。

相较此前候选，本版保留 packed_rows_kernel 的名称、模板参数、函数签名以及现有 host 分派结构。仅在 Transform && N>=2 的设备分支中移入已经验证过的相邻成对蝶形，N1和quantize-only保留旧设备逻辑。shared constexpr packed_row_lanes 同时控制 Torch 和 CLI 网格，避免两处映射不一致。无新增公开API或独立host分派。

修改范围只有 include/packed_rows.cuh、src/torch_binding.cu、src/main.cu。此前主机交叉诊断未能在四配置中复现稳定超过5%的退化，不能代替本版完整验收，也未更改两次REJECT。

下一步先冻结新协议与源指纹，准备完整 CUDA/Torch 和 CLI 检查；CLI 计算入口已受影响，需要重新编译并跑原有 self-test（original/packed/auto及线程配置）、独立CPU参考和已有负例。随后执行完整形状/精度/线程/ordinary/Graph对照，保留时间线与配对样本，并控制输出缓冲差异；不得仅用四项诊断案例决定采纳。

共享常量只供编译期行宽计算，不缓存输入、参数或输出。具体编译兼容、资源占用及性能以实测为准。
