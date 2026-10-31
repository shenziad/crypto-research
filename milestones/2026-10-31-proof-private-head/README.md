# 证明候选接入真实私有head：功能闭环与通信正结果

双域AES、平衡映射、完整匹配、真实/虚拟ID完整表域、掩码哨兵、新鲜公钥零加密及218位BFV/149位注入已接入隔离后端。两语料完整私有head功能、三组控制与四组同期密集/拒绝复测全部通过；模型和有限函数保持原样。安全论证为带明确假设的草稿，尚未完成完整软件认证或独立创新确认。

| 同期路径 | 发送字节中位数 | 墙钟中位数 | 墙钟范围 |
|---|---:|---:|---:|
| 优化密集CDF95 | 3,504,947,076 | 113.974秒 | 29.932–202.320秒 |
| 证明候选拒绝路径 | 981,310,983 | 62.123秒 | 49.153–302.293秒 |

通信减少72.0021%。墙钟方向逐块混合、波动很大；不宣称稳定加速。全部8次复测保留，无剔除或重跑；低频/桌面资源快照一并记录，未建立耗时原因的因果关系。两阶段分别11/20个原生对子、948.07/1050.66秒，均在各自登记预算内。

- [实际后端和条件组合草稿](notes/20261031_fable_proof_integration_outcome.md)、[假设清单](notes/20261031_fable_proof_assumptions.json)
- [同期复测完整结论](notes/20261031_qwen_proof_matched_benchmark_outcome.md)、[复测审计](results/qwen_proof_matched_benchmark_audit.json)
- [控制结果](results/fable_proof_controls.json)、[两语料功能](results/qwen_proof_sampling.json)、[集成审计](results/fable_proof_integration_stage_audit.json)
- [集成登记](notes/20261031_fable_proof_integration_preregistration.json)、[测试前补充](notes/20261031_fable_proof_pre_run_amendment.json)、[复测登记](notes/20261031_qwen_proof_benchmark_preregistration.json)
- [可核对源码差异及构建配方](sources/fable_proof_backend_build/source_manifest.json)、[待激活公开布局补充计划](notes/20261031_fable_bucket_budget_followup_plan.md)

前序依据：[BFV均匀界与原生1024列检查](../2026-10-31-bfv-response-proof/README.md)、[放置部件和Hall候选界](../2026-10-31-private-lut-placement/README.md)、[旧精确编码结果](../2026-10-31-qwen-exact-packing/README.md)。旧版性能只对应旧后端，不与这里的证明候选混用。

本阶段保留质量和通信正结果，并继续研究墙钟波动原因与公开布局成本。固定格式支持、标准匹配/重新随机化和证明补足不声称新密码原语；CCF级贡献仍待独立证据。

自有程序相对阶段目录解析ROOT。需按前序配方准备已固定FABLE/EzPC/SEAL依赖、原生选中行/alias/CDF95二进制和Python环境；此处上传新源码/依赖差异/摘要，非干净克隆一键安装包。脚本拒绝覆盖原记录，复跑须另登记时间窗口/输出目录。大权重/语料正文/分享/币/token/二进制/第三方完整源码与PDF保留在NEWSMY。
