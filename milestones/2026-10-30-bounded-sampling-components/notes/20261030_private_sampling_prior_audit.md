# 私有采样紧邻先例核验

日期：2026-10-30。这里只比较构造、功能及成本，不做攻击研究。结论：基本OT隐藏查表已有直接先例；任意高精度alias编译是否已有同等构造仍未确认，不能把关键词未命中视为新颖性证据。

## PoPETs 2025

[Kii、Ichikawa、Miura，Lightweight Two-Party Secure Sampling Protocol for Differential Privacy](https://petsymposium.org/popets/2025/popets-2025-0003.pdf)。完整PDF已保存，756028字节，SHA256 `a1064b8858f027ce4c62f7a63aed335491089996b82e8ee11c62db5f5bebcd9a`；本地提取全文并渲染查看物理第10页（印刷页32）。

§7/Protocol3 的公开表T先随机置换，再以新鲜加法掩码隐藏表项；另一方均匀选OT索引，得到秘密分享的表样本，随后求和生成DP噪声。置换与索引由不同参与者选择，各抽样重新随机化。它已覆盖“隐藏索引+隐藏表值+OT”的组合思路；本项目采用XOR平移替代完整置换不能独立算创新。区别是该论文的表公开、目标是其卷积噪声分布，本候选由一方持有任意高精度alias表并秘密比较选择。该差异是待核验的功能差异，不自动构成论文贡献。

## AsiaCCS 2025

[Agarwal等，Prior-Based Label Differential Privacy via Secure Two-Party Computation](https://doi.org/10.1145/3708821.3736203)，并核对[作者机构页面](https://research.google/pubs/prior-based-label-differential-privacy-via-secure-two-party-computation/)。

出版商索引正文及ePrint检索到的完整§4.1.1/Figure3、§4.1.2显示：已有私有集合大小下的均匀集合采样、共享索引OT读取、局部集合置换、私有偏置Bernoulli采样，以及用这些模块实现特定非均匀RR分布。Bernoulli模块按2^f精度展开表。我们的通用共享索引OT读取和模块化组合均应视为标准组件，不主张首创。其特定RR分布不等于任意alias表；当前未找到其已明确给出本候选完整编译的证据。

下载限制：ePrint 2025/211 PDF、ACM PDF及full HTML本地请求均403，网页工具直接打开也403；检索索引能提供相关段落，但尚未取得本地完整PDF或逐页核对最终版本。不能称已完成该论文的全面审计。保留请求记录 `sources/primary_pdf_audit_20261030/prior_label_dp_download_attempts.json`；Google页面的Download链接还指向另一个DOI（3736198），最终论文DOI以标题匹配的3736203为准。

## 对本项目的决定

保留已通过的实际OT/Yao组件和同后端CDF对照，定位为可复用研究原型。局部收益、有限联合模拟和模型NLL改善是三种不同证据，均不单独证明新密码原语或完整LLM加速。下一步首先核算已秘密采样索引读取另一方私有adapter行的真实成本，再决定是否值得搭建191次完整拒绝链路；不得忽略这些行读取或改成公开候选。
