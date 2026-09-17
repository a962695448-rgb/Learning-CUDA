第一版在通用 launch 模板内插入成对量化分派，数值和设备门槛通过，但 BF16 N8/65536 行的 allocating Hadamard 对照三轮约慢 4%–5%，第二轮越过 5% 门槛。保留结果为 REJECT。第二版恢复通用 launch 与 dispatch 定义，通过 quantize-only 的显式特化隔离新分派；内核、测试矩阵、采样数量和门槛不变。
