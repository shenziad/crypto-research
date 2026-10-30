# LLM Cryptography Research / LLM结构密码协议探索

利用LLM的数学结构研究密码协议，记录文献查重、可复现实验、负结果和适用边界。项目持续进行；尚未确认新的密码原语或CCF C+级独立创新。

## 阶段成果

| 日期 | 内容 | 状态 |
|---|---|---|
| 2026-10-29 | [精确共享低位修正分类与有限gate/down协议](milestones/2026-10-29-exact-coarse/notes/20261029_exact_coarse_closure.md) | 432组完整局部协议实验通过；两个Qwen模型、CPU/CUDA；已归档 |
| 2026-10-30 | [固定提案私有token采样筛选](milestones/2026-10-30-private-sampling-screen/README.md) | 1,024个留出位置、98,304个精确枚举组合；直接方案负结果，已收束 |
| 2026-10-30 | [有界私有输出采样：有限算术与秘密接受组件](milestones/2026-10-30-bounded-sampling-components/README.md) | 13组真实两方GC验证、有限采样误差预算及私有读行后端；完整链路未接通 |
| 2026-10-30 | [有界私有输出采样完整功能原型](milestones/2026-10-30-bounded-sampling-fullchain/README.md) | 12次完整两方运行、两种后端各4次交替复测；密集强基线/实际安全归约未完成 |
| 2026-10-30 | [同一有限E31分布的完整密集/拒绝采样对照](milestones/2026-10-30-matched-exp31/README.md) | 四组交替复测全部通过；当前实现通信少38.56%、墙钟少41.10%；独立创新/实际安全归约未完成 |
| 2026-10-30 | [Qwen0.5B有界输出适配器固定迁移](milestones/2026-10-30-qwen-bounded-transfer/README.md) | 验证收益保留48.8%，未过预设50%门槛；原阶段未跑测试，后续固定权重跟进见下 |
| 2026-10-30 | [Qwen冻结权重留出跟进与决策修正](milestones/2026-10-30-qwen-frozen-followup/README.md) | 两语料8192位置均改善：PTB−0.1221/Wiki−0.00626nat；继续完善质量/完整协议，旧筛选结果保留 |
| 2026-10-30 | [Qwen有限质量与原生18位选中行组件](milestones/2026-10-30-qwen-finite-components/README.md) | 8192位置保留收益、10组原生两方测试通过；完整查表/CDF95仍待连接 |
| 2026-10-31 | [Qwen完整私有head：同分布全链路对照](milestones/2026-10-31-qwen-fullchain/README.md) | 两语料功能核验＋四组交替复测；当前实现通信少48.46%、墙钟少50.91%；贡献与联合安全继续完善 |
| 2026-10-31 | [Qwen精确字段编码与完整三路径对照](milestones/2026-10-31-qwen-exact-packing/README.md) | 同一有限函数；当前编码路径较密集通信少72.13%、墙钟少65.07%；安全候选界尚待实现 |
| 2026-10-31 | [私有LUT放置部件与BFV证明义务审查](milestones/2026-10-31-private-lut-placement/README.md) | 28万图/52万映射独立核验；BFV残差与完整密文模拟仍待证明，继续完善 |

[实验汇总JSON](milestones/2026-10-29-exact-coarse/results/exact_coarse_comparison_report.json) · [受限证明草稿](milestones/2026-10-29-exact-coarse/notes/20261029_exact_coarse_joint_view.md) · [复现范围与CPU smoke](milestones/2026-10-29-exact-coarse/REPRODUCIBILITY.md)

相对同一研究实现中的直接32位分类，修正24位方案的本地CPU在线时间降低25.5%–32.9%，CUDA降低15.1%–22.5%；通信增加8n字节，仍为5次交换。这些是有限计算块结果，不能解释为整模型加速或对上游SIGMA/FuseFSS/Grotto的优势。

原始数据、代码检查点与下载资产保存在NEWSMY。这里上传阶段性源码和原始结果记录；不包含模型大权重、环境缓存、语料正文、选中token序列或第三方论文PDF。已启动的完整整数FFN新窗口测试仅收尾为工程基线；本任务后续不以MoE稀疏密码计算或量化原生密码计算为独立主线。

采样筛选表明：把校准提案包络扩大到覆盖本批留出分布后，去重仍访问至少99.94%的词表。该结论针对已测直接构造，既不是所有采样协议的下界，也不是新的CCF C+成果。研究继续围绕LLM结构与严格私有输出接口探索。

See [NOTICE](NOTICE.md) for attribution and [the reproducibility notes](milestones/2026-10-29-exact-coarse/REPRODUCIBILITY.md) for the limits of a clean clone.
