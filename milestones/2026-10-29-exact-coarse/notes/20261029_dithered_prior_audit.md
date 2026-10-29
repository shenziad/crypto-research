# 粗分类／细正文候选的直接先例核验

2026-10-29，持续更新。只审阅构造和功能定义，不开展攻击研究。以下“未发现相同接口”只描述已读范围，不构成全域查重或新颖性证明。

1. **概率截断本身已有直接先例。** Dalskov、Escudero、Keller 的 [Secure Evaluation of Quantized Neural Networks](https://arxiv.org/html/1910.12435) §3.2.2 已有“公开掩码和的高位，减去掩码高位”以及按输入小数部分决定向上概率的舍入。当前 z 的代数不能算新原语。
2. **ORCA 已融合随机截断和激活。** [作者托管 SP2024 原文](https://www.csa.iisc.ac.in/~arkapravab/papers/SP2024_Orca.pdf) §5.2、§5.3，PDF第8、9页，给出独立随机阈值的 stTR、符号提升和 ReLU 融合。它明确使用 DCF 处理低位关系；不能把现有随机截断功能当成省去低位比较的同接口基线。当前功能公开 rho，必须单独写明这种接口差别。原PDF、SHA256和第8页渲染已保存在 sources/primary_pdf_audit_20261029/。
3. **SIGMA 已分离粗比较和细计算。** [作者预印本](https://apanwariisc.github.io/publications/pets-2024-sigma/sigma.pdf) §5.1.1 Optimization2、Figure6（PDF第8页）把 TR 提到前面，减少 ReLU/Clip 比较位宽，最终 select 仍作用于原细 x。因此泛称“低位宽分类、保留原输入”也已有明显近邻。它计算 ReLU 减去粗绝对值 LUT 的残差；当前候选作用于任意对齐 C0 仿射样条的选段，正文为该段 P_i(A)，需证明并实测这一差别是否足以带来独立研究价值。
4. **SIGMA 的 faithful 在此处是精确 floor。** §4.2.1 定义 LRS 为 floor(x/2^f)，§4.2.2 的 TR 是丢低位并缩小环。不能把该论文中的 faithful 自动解释为任意 floor/ceil 一ULP舍入。已读作者版数学排版和原文公式；最终发表版从[官方 proceedings 镜像](https://crysp.petsymposium.org/popets/2024/popets-2024-0107.pdf)核对版本。
5. **Google affine public spline 已有细输入与公开参数功能。** `google/fss_machine_learning/applications/secure_spline/secure_spline.h` 第39–64行描述精确 MIC 选段与一次多项式，`.cc` 第300行起批量 MIC，随后对原输入做各公开斜率的乘法。cc/h/LICENSE 的 Git blob 已逐一核对，记录在 sources/google_spline_reference/source_manifest.json。所审实现不是本候选公开rho的随机选段定义；不能由此推断所有相关工作均无此设计。
6. **2021公开样条载荷压缩和2026线性模块闭包仍是基础先例。** 见既有 public_span_audit.json 与幂基证明。新的分类器没有使标准载荷基底压缩变成原创密码原语。
7. **Grotto 完整审计仍未完成。** ePrint正文下载403；已由原论文索引确认DPF前缀奇偶与样条支持，并发现作者的 `Kyle-Storrier/EzPC-LLAMA-fork`，但那是修改的 LLAMA 比较对象，不能当作 Grotto 本体代码。第三方 myl7/fss 的 parity-tree 计时把 Preprocess 排除在 point Eval 外，不能拿后者单独与本实验在线时延比较。当前没有上游 Grotto/SIGMA/FuseFSS 的匹配性能结论。

8. **Curl 已明确公开 cutoff 的理想功能。** [CAMLIS2024正式原文](https://ceur-ws.org/Vol-3920/paper02.pdf) §3.1、Figure3（PDF第9、10页）已有公开随机阈值的概率截断接口及stand-alone模拟；公开rho不能作为新密码功能贡献。阈值端点约定必须在具体对标时统一。§4.2.2、Eq.(2)和Figure7（PDF第15、16页）还已有高位选压缩表、低位做线性插值的设计。原PDF、SHA256和第9/15页渲染已保存。当前构造与该确定性插值协议并不相同，但“粗定位保留细数值”的泛化说法再次受到直接先例限制。

当前定位进一步收紧为“平滑性约束下，已有公开随机截断与FSS的具体组合、误差传播证书和实验候选”。若只剩经典概率截断与标准 FSS 的直接组合，就应保留为工程基线并继续转向，不能凭正向计时宣布 CCF C+ 成果。双种子及同种子粗正文消融已完成；第二模型继续，另核验门误差经过down后的精确偏差/方差公式，再决定是否值得构造独立的新协议。
