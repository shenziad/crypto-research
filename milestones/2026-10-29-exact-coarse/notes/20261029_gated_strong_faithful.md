# SwiGLU 融合：精确截断、私有输入量化与匹配基线

2026-10-29，当前入口。研究只进行本地协议构造、文献查重、功能与数值实验，不开展攻击研究。尚未确认独立新颖性，尚不是端到端私有 LLM 或 CCF C+ 成果。

## 当前结论

融合利用 `g(a)*b` 对 b 的线性结构，消去激活查表与乘法之间的一次依赖交互。在同一原生后端上，与 LLAMA 混合位宽基线相比，它的在线字节数相同，交互少一次。计入私有输入量化后，4864 元素的预处理数组约 77.52 MB 对 84.47 MB；模拟每次逻辑交换 50 ms 时，本地进程中位耗时约 277.1 ms 对 330.7 ms。其本地无延迟性能接近普通32位 Beaver 基线，不能声称普遍计算加速。

向量 DPF、公开线性后处理、DCF、带掩码分段仿射函数、LLAMA 混合乘法和精确移位均是已有工具。当前结果支持继续研究算子融合与表示选择，不能将组合恒等式称为新的密码原语。后续应先解决新颖性和完整 FFN 的接口成本，再决定是否形成论文。

## 有限功能和输入契约

模型为官方 Qwen2.5-0.5B，固定 revision `060db6499f32faf8b98477b0a26969ef7d8b9987`，24层，每层中间宽度4864。每层公开 `sa=max_gate/127`、`sb=max_up/32767`，由冻结的四个校准窗口确定。公开 LUT 为

`g_i[j]=round(256*SiLU(sa_i*sint8(j)))`。

该序列化整数表本身定义有限功能，不将其当作连续 SiLU 的精确值。门控输入 a 在8位环分享，b 在32位环分享，b 的数值被规定为有符号16位整数。输出为64位环上的 `floor(g_i[a]*sint16(b)/256)` 的分享。

所有24层、全部256个索引均已枚举：`max |g|=13701`，`|g*b|<=448954368<2^31`，截断后绝对值不超过1753728。此为精确整数范围证据，见 `results/qwen_gate_integer_certificate.json`，不以测试样本的最大值代替全域证明。

新增私有输入前缀从规范化 signed32 fixed8 分享开始：

`A=round(256*a_real/sa)`，`B=round(256*b_real/sb)`；
`qa=clip(floor((sint32(A)+128)/256),-127,127)`，
`qb=clip(floor((sint32(B)+128)/256),-32767,32767)`。

二次舍入在半整数处向正无穷舍入，不等于 PyTorch 直接浮点 ties-to-even 量化。模型矩阵乘法结果如何私下转换为该规范化 fixed8 分享仍未实现；不能据此称整个 FFN 已闭环。

## 构造与正确性

融合门独立采样 ra∈Z256、rb∈Z2^32。两方同一次交换打开 `ahat=a+ra` 和 `bhat=b+rb`。以标准向量 DPF 分享隐藏点 ra、载荷 `(1,-rb)`，在全部256点求值，再以公开 `g(ahat-j)` 加权，得到 `F0=g(a)`、`F1=-rb*g(a)` 的分享。各方本地输出 `bhat*F0_i+F1_i`，其和恰为 `g(a)*b mod2^32`。

门后精确移位与符号扩展共用一次32位掩码开示。设 `h=(raw+r) mod2^32`、`theta=(r+2^31) mod2^32`，则

`floor(sint32(raw)/256)`
`= (h>>8)-(r>>8)-I((h mod256)<(r mod256))`
`  +2^24*(I(h<theta)-I(r<2^31))`。

分别用8位和32位 DCF 产生两个比较项的64位算术分享，常数由预处理分享。无需任何本地随机截断。`faithful_shift_check.py` 已穷举小环所有65536对输入/掩码，并用真实两方原生后端检查完整 signed32 边界和随机值。

输入量化前缀也用上述式子，但低位阈值改成 `(r_low+128) mod256`，常数增加 `I(r_low<128)`，得到半向上舍入的32位分享。随后再次开示新掩码的舍入值，三组32位双载荷 DCF 对分段仿射系数求值完成 signed clamp。截断与裁剪各一次交换，a、b 两支同时批处理；最后 a 直接投影到8位环。所有形状固定，不随秘密值改变。

裁剪在原始无符号坐标的系数跳点为 `M, 2^31, 2^32-M`。负边界必须属于线性分支，以保证经过掩码循环平移后的系数正确；这已由全部掩码的小环枚举覆盖，而不仅比较相等边界的函数值。

## 基线与实际计数

同一输入、整数表、输出定义及原生 AES 后端，使用三种门：

- `beaver32`：32位标量 DPF 查表，32位 Beaver 乘法。
- `mixed16`：16位标量 DPF 查表，LLAMA signed16×16→32乘法；从共同32位 b 分享本地投影，不收取虚构的转换成本。
- `fused32`：上述双载荷门。

当前原生后端尚未采用 Pika/SIGMA 的所有树打包和 GPU 优化，因此这里的“强基线”表示匹配功能、混合位宽和后端的对照，不表示已覆盖最优实现。门前、门后开示可批处理的项均已批处理。

以下字节是两方实际发送之和，不把收取再加一遍；材料是实际密钥数组存储字节，不含输入分享、模型参数和 dealer 网络。

| 功能范围 / 每元素 | Beaver32 | Mixed16 | Fused32 |
|---|---:|---:|---:|
| 已量化输入→精确64位输出：在线B | 26 | 18 | 18 |
| 同上：逻辑交换 | 3 | 3 | 2 |
| 同上：两方材料B | 2458 | 3878 | 2450 |
| fixed8私有量化→精确64位输出：在线B | 58 | 50 | 50 |
| 同上：逻辑交换 | 5 | 5 | 4 |
| 同上：两方材料B | 15946 | 17366 | 15938 |

真实 Qwen 第0/12/23层、每次4864元素，6种完整算法顺序×3层×2延迟×3算法，各设置总计108次运行。两套结果都全部通过原始乘积及精确 signed64 输出检查。

| 范围 / rank0总耗时中位数ms | Beaver32 | Mixed16 | Fused32 |
|---|---:|---:|---:|
| 已量化输入，0 ms模拟延迟 | 39.57 | 44.20 | 39.84 |
| 已量化输入，50 ms/交换 | 191.12 | 194.74 | 140.82 |
| 包含私有输入量化，0 ms | 76.24 | 81.14 | 77.06 |
| 包含私有输入量化，50 ms/交换 | 326.62 | 330.73 | 277.13 |

最后两行两方取最大耗时的中位数分别为76.26/81.16/77.19与326.65/330.81/277.23 ms，结论一致。50 ms来自本地 socket 每次交换后的人工延迟，既不是实际 WAN，也没有评估带宽竞争。计入量化后，相比 Mixed16 材料降低约8.2%、模拟延迟耗时降低约16.2%；在线字节无降低。

## 与有限协议匹配的 GPU 数值证据

RTX4060 Laptop，Float32，关闭TF32，冻结四个校准窗口；测试offset4096/4608/5120/5632，每窗129 tokens，共512个预测位置。改变全部24层门，矩阵乘法、注意力和其他运算仍为浮点。

| 数值定义 | 平均softmax KL | top-1一致率 | 平均NLL差 |
|---|---:|---:|---:|
| 直接浮点量化+精确floor | 0.00690493 | 95.117% | -0.00677317 |
| fixed8输入量化+精确floor | 0.00652707 | 95.898% | -0.00757319 |

后者逐层共60235776个元素，gate/up编码分别有约0.196%/0.203%与直接 round 不同，确实重跑了全模型数值前向。小样本 NLL 的负差不能解释为质量提升；这些结果不是完整语料困惑度、下游任务或端到端安全推理。

## 安全证明状态与历史更正

目标为可信预处理、两方半诚实、至多腐化一方。新鲜均匀环掩码隐藏每次开示；DPF/DCF 单密钥隐私和标准 offset-FSS 组合为候选提供证明路径。仍需完成组合后的联合视图、输出分享、具体 PRG 安全损失和实现层面的正式证明；当前功能检查不替代该证明。本地父进程生成全部秘密、fork子进程继承内存，当前程序是测试架，不是隔离部署。

旧 `fused_gated_lut.py` / `qwen_gated_protocol.py` 的本地随机舍入只检验过数值/边际分布，未建立标准模拟安全；已保留为历史功能模式，主线全部换成精确 FSS 移位。历史64位的34→18B、47%降幅不适用于当前混合位宽强基线。旧记录的“安全截断/端到端收益”均不得由那些实验推出。

## 查重与下一轮

- [标准DPF改进](https://eprint.iacr.org/2018/707)：向量/群载荷与求值线性组合不是本工作新原语。
- [FSS预处理框架，TCC2019](https://www.iacr.org/archive/tcc2019/11891211/11891211.pdf)：固定点门的掩码重写与预处理属于既有研究框架。
- [LLAMA，PoPETs2022](https://petsymposium.org/popets/2022/popets-2022-0109.pdf)：混合位宽乘法和符号扩展是本对照采用的现有协议。
- [Pika，PoPETs2022](https://petsymposium.org/popets/2022/popets-2022-0113.pdf)：查表的树/叶打包优化需作为后续实现基线，不能据当前标量原生后端声称最优求值性能。
- [SIGMA，PoPETs2024](https://petsymposium.org/popets/2024/popets-2024-0107.pdf)：已有私有Transformer、精确移位和复杂非线性协议。
- [FuseFSS，2026](https://arxiv.org/html/2606.09551v1)：已把单输入分段函数的比较及向量系数查表统一编译；后处理也包含秘密乘法。当前二输入线性融合相对其“先查表再Beaver”实例省一次依赖，但不是据此证明超出通用框架。
- [SPA，2026-10-23](https://arxiv.org/abs/2610.32065)：已利用SiLU/GELU的奇偶结构近似以降低FHE算术成本；因此将SiLU拆成 `x/2+even(x)` 不作为新的独立选题。

下一轮考察门控后立即接4864→896的down投影这一结构：是否能把截断推迟到收缩之后，直接保持宽环门乘积，并与已有延迟截断/通用融合进行相同接口对照。需要同时计入宽环输入量化、秘密权重矩阵乘法和最终移位，不能仅把门后截断删掉。该问题是待检验候选，没有宣称新颖或节省已实现。

### 同路线继续：收缩之后截断的数值筛查

已继续完成两种舍入位置的全24层 GPU 前向，而非仅列出计划。保持相同fixed8量化和门表，把所有down权重固定成 `Wq=round(4096*W)` 的有符号16位整数，分别定义：

`stage_k=floor(sum_j Wq_kj*floor(g(a_j)*b_j/256)/4096)`；
`late_k=floor(sum_j Wq_kj*g(a_j)*b_j/2^20)`。

两者输出物理scale均为sb。它们是不同的有限算术功能，不能将数值不同的执行时间当作同功能加速。阶段式和延后式均有四窗全模型结果：stage KL约0.00669040、top-1一致率94.336%；late KL约0.00742914、top-1一致率95.703%。指标各有变化，样本不足，不能声称延后截断提高质量。

对全部24层每个输出行，用公开有限门域的乘积上界乘以该行 `sum |Wq|` 证明所有可能累加的绝对值不超过189702076563456，仅需49个有符号位，低于2^53和2^63。GPU使用float64精确表示这些整数及其每个可能部分和，最后才转整数移位；这只是有限电路的数值求值，不是安全矩阵乘法。三套结果各自保存24层证书，峰值GPU分配约3.42 GB。

局部门输入相同时，两种整数输出的差还满足 `|late_k-stage_k|<=ceil(sum_j |Wq_kj|/4096)`；这是由每个被提前丢弃的小数位处于[0,1)得出的确定性界，不能直接外推为全模型误差界。

该结构允许继续测试56位宽环的原始门乘积以及收缩后的精确FSS移位，但要付出量化结果提升、宽环私有矩阵计算的成本。延迟截断本身不是新方法：[通信高效逻辑回归工作](https://eprint.iacr.org/2022/866) 已讨论LLAMA用较大环承载未截断乘法，随后统一截断；[Ditto](https://proceedings.mlr.press/v235/wu24d.html) 已系统处理安全量化和定点转换。下一步只考察这种SwiGLU收缩接口是否产生额外可测收益，不把通用技巧重新命名为创新。

新增 `results/gated_input_quantizer_explicit_checks.json` 还记录12次独立验证：对三个真实层的三种算法及4096项全signed32随机输入，分别重构并检查qa/qb，避免只检查最终乘积掩盖量化错误。

## 复现入口

源码位于 `experiments/`；CPU解释器 `envs/torch-cpu/bin/python`，GPU解释器重用 NEWSMY 上 `KVcacheSC/.venv/bin/python`。设置各缓存指向本目录 `toolchain/`。下载器 `fetch_qwen_gate_model.py` 固定官方revision与SHA256；语料及清单位于 `data/`。

```bash
c++ -O3 -std=c++17 -maes -msse2 -fPIC -shared experiments/dpf_aes_native.cpp -o toolchain/dpf_aes_native.so
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 envs/torch-cpu/bin/python experiments/dpf_aes_native_check.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 envs/torch-cpu/bin/python experiments/dcf_aes_native_check.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 envs/torch-cpu/bin/python experiments/faithful_shift_check.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 envs/torch-cpu/bin/python experiments/qwen_gated_strong_protocol.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 envs/torch-cpu/bin/python experiments/gated_input_quantizer_check.py
# GPU、缓存环境按上文设置后运行：
/media/shenziad/NEWSMY/KVcacheSC/.venv/bin/python experiments/qwen_gated_quality_faithful.py --fixed8-input-quantizer
```

权威结果分别为 `results/qwen_gated_strong_faithful.json`、`gated_input_quantizer_pipeline.json`、`qwen_gated_quality_faithful.json`、`qwen_gated_quality_fixed8_input.json`。二进制SHA256为 `0585a47e6a13eb46371c43860262e3286324880a45511b3e9dc97a17b4e3233d`，源码保留上游 MIT 许可。
