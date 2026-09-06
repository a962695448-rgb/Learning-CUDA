# N256 数值证书：整数精确变换、FP32 阶段模型和存储 RNE

本模块仅使用 CPU NumPy 与 Python 整数/Fraction；不调用 GPU，不改变 kernel、量化公式、旧失败记录或官方 Dao 对照要求。它检查调用方提交的采样输入/输出位模式，不能替调用方证明这些输出来自哪一次 GPU 执行，也不覆盖未提交的其他行。

## 调用接口

```python
from numeric_certificate import certify_samples

certificate = certify_samples(
    input_bits,          # uint16 数值，[采样行数, 256]；一维256元素也可
    gpu_output_bits,     # 相同形状的实际输出原始uint16位模式
    dtype="fp16",       # 或 "bf16"
    scale=1.0,          # 仅1或0.0625
    dense_fp64=dense,    # 可选：实际使用的FP64 dense输出，dtype必须float64
    row_ids=sample_rows,
)
if certificate["status"] != "PASS":
    raise RuntimeError(certificate["first_failure"])
```

不提供 `dense_fp64` 时，在本模块计算 NumPy FP64 稠密 Hadamard 乘积；报告 `dense_source` 区分这两种来源。提供该参数可保留调用方原先稠密参考的实际舍入结果。

`include_element_diagnostics=True` 额外返回每个元素的详细记录。默认也会检查**所有提交元素**，但只保存每行最大存储误差的位置、最多8个舍入差异样例，以及行级最大值/计数，避免大量重复字符串。关闭详细输出不会减少样本、改变门禁或改变数值运算。

返回值为可直接 JSON 序列化的字典：

- `status`：`PASS` 或 `FAIL`；调用方必须显式检查。
- `first_failure`：首个失败代码、采样行/原行号、列，以及适用的输入位、GPU位、期望存储位、CPU预存储FP32位和精确误差/界。
- `rows`：逐行 L1、共同二进制分母、E32分项、三个门禁结果、误差最大值、E64诊断、舍入差异计数和示例。
- `summary`：通过时的全样本计数及最大误差；没有把不同进程或样本范围累加成新 GPU 用例。
- 输入位、输出位和 CPU 预存储 FP32 值按小端元素字节计算的 SHA256；由调用方关联原始证据。

涉及界的每个记录包含精确 `fraction` 字符串和 `upper_float`。后者由 `nextafter(...,+∞)` 产生保守向上展示值；**所有门禁始终使用 Fraction，不使用展示浮点数**。

## 精确值与阶段模型

1. 把 FP16/BF16 原始位解码为 FP32。每个值使用 `as_integer_ratio()` 得到精确整数比，并与原始 IEEE 位直接解释出的有理数核对，避免静默丢失 subnormal。
2. 每行提升到共同的2幂分母，使用任意精度 Python 整数完成8阶段 FWHT；乘以精确 scale 得到理想值 `y`。整个过程没有浮点加减。
3. 另一路使用 `np.float32` 显式执行相同 global bit0→7 顺序的 add/sub，每阶段落回FP32，再进行一次FP32 scale乘法，得到预存储值 `z`。没有改变蝶形操作数顺序。
4. 使用整数商、余数和奇偶位实现最近偶数舍入。对正规数、subnormal、下溢到有符号零和进位统一处理，不借用 NumPy FP16/BF16 cast 作为存储 RNE 的裁判。

整数 FWHT 对应 Sylvester Hadamard 变换。这里的 FP64 dense **不是精确值的替身**：将其每个实际 FP64 数值提升为 Fraction 后，单独记录 `E64 = |dense64-y|`。即使 E64 非零也不会加入 E32，不会据此放宽门禁。

## 固定误差界及三个门禁

采用以下固定量：

\[
u_{32}=2^{-24},\qquad\eta_{32}=2^{-149},\qquad
\gamma_9=\frac{9u_{32}}{1-9u_{32}}=\frac9{2^{24}-9}.
\]

单个输出的依赖树有256个输入叶子、255个加减节点；每个输入对输出的路径最多8次加减及一次scale。设输入行的精确绝对值和为 `L1`，固定预存储界为：

\[
E_{32}=\gamma_9|s|L1+
\frac{\eta_{32}}2\left(|s|\,255(1+u_{32})^8+1\right).
\]

推导使用有限 IEEE RN 运算模型
`fl(t)=t(1+δ)+ε`，`|δ|≤u32`、`|ε|≤eta32/2`：

- 输入项路径上最多9个相对扰动，合计由 `gamma9` 控制，逐项取绝对值得到第一项。
- 每个加减节点的绝对 underflow 残差，至输出最多经过8个后续相对扰动及一次精确系数 `|s|`。统一上界为 `|s|(1+u32)^8`，累计255个节点；最后scale自身的绝对残差再加1份 `eta32/2`。
- 上式保留完整 underflow 项；没有因为普通输入远离下溢而删去该项。

对每个元素逐一执行三个显式 `if` 门禁：

1. `|z-y| ≤ E32`，比较两个精确有理数。
2. `RNE_T(z)` 的**全部存储位**必须等于实际 GPU 位，包括零符号。
3. 记录实际 `delta_store=|RNE_T(z)-z|`，检查 `|GPUstored-y| ≤ E32+delta_store`。

第二项不可省略。第三项由前两项及三角不等式支撑；实际存储舍入误差不用于掩盖不同的 GPU 位模式。

记录 GPU 对未舍入dense、直接舍入dense、经FP32舍入dense的误差，但不再对已经舍入的dense套用 `.01/.05`。官方 Dao 全元素严格阈值和 CPU INT4 packed/scales 精确检查仍由外层 adapter 独立负责。

## 舍入诊断与前提

同时记录两组 direct/via-FP32 比较：

- 理想有理数 `y → RNE_T` 与 `y → RNE32 → RNE_T`；
- 实际 FP64 dense 数值 `dense64 → RNE_T` 与 `dense64 → RNE32 → RNE_T`。

它们是诊断，不会替换阶段模型。相邻正/负方向的存储 ULP 使用原始格式位计算；最大有限值的无限邻居显式记为 `null`。理想数学零的直接舍入采用正零，实际 CPU 阶段零的存储舍入保留符号；纯零符号差异另外计数，不冒称 double rounding。

前提是 RN ties-to-even、无运算重排/fast-math、无溢出、渐进下溢且不启用 FTZ/DAZ。模块检查 CPU 两个 RN 中点、subnormal加法与乘法，并拒绝非有限输入、输出、FP32中间值、dense或诊断中的非有限存储转换。有限个 CPU 预检不能替代 GPU 编译选项与运行来源的记录；这些仍是调用方责任。

## 已执行的 CPU 检查

保留的实测来源是旧失败目录中的 `server_raw/diagnostics/dense_rounding/diagnostic.json`。本次只复核其中**已经提供GPU位模式的一个标量**：row8191、column230，未补造缺失的整行GPU输出。

| 量 | 精确值/位模式 |
|---|---|
| 理想值 | `-75333627/4194304` |
| CPU阶段FP32 | `-17.9609375`，`c18fb000` |
| 实际GPU及CPU阶段RNE | `-17.96875`，`cc7e` |
| 理想值直接RNE | `-17.953125`，`cc7d` |
| 输入L1 | `426739835/4194304` |
| E32向上展示值 | `5.45790694458629e-05` |
| 实际预存储误差 | `5/4194304` |
| 实际存储舍入误差 | `1/128` |
| GPU对理想值误差 | `32773/4194304` |

CPU阶段值恰在相邻FP16值的中点，偶数最低位选择 `cc7e`；理想值稍高于该中点，选择 `cc7d`。该见证的直接舍入与经FP32舍入相同，不能归因为 dense 路径 double rounding。两个误差界和存储位门禁在该已观测标量上成立。

此外完成：22个独立闭式 CPU 控制（零、脉冲、常量、交替、subnormal、负零，覆盖两dtype/两scale）；1个BF16宽指数控制显示256个FP64误差非零元素仍按独立E32检查；10个整数RNE中点控制；3个拒绝控制（改坏输出位、非有限输入、FP32阶段溢出）。这些是 CPU 自检，不是新增GPU测试，也不抹去旧52配置运行的原始失败。
