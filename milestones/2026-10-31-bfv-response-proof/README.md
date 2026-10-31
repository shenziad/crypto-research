# BFV候选响应：均匀系数界、模拟草稿和原生正确性

有界私有输出头的后端证明补足。固定n8192/t163841/218位密钥层（172位数据层）/M4096/149位均匀噪声，另加新鲜公钥零加密。源码对应的保守系数推导和精确有理数检查给出87位残差界；两次LUT共45个响应密文的噪声平移TV上界约1.523e-13；模切换各层的解密余量均为正。完整密文模拟另依赖明确的RLWE/PRNG假设及新鲜零加密，旧后端未被修改或认证。

六个公开合成原生电路各运行1024列，49,152个槽与独立Python参考一致。未读取/拟合noise budget，未做攻击、选密钥或重跑电路。原登记窗口在构建失败后到期（电路执行数0），原记录保留；一次60分钟续做在389.35秒内完成审计，六个电路均只执行一次。构建失败日志仍在NEWSMY。

- [推导、条件、查重与边界](notes/20261031_bfv_response_uniform_bound.md)
- [原登记](notes/20261031_bfv_response_proof_preregistration.json)、[续做登记](notes/20261031_bfv_response_proof_continuation.json)、[原未完成记录](results/bfv_response_initial_attempt.json)
- [精确算术证书](results/bfv_response_bound_certificate.json)、[原生检查](results/bfv_response_pipeline_check.json)、[阶段审计](results/bfv_response_proof_stage_audit.json)
- [构建/依赖说明](sources/bfv_response_proof_build/build_recipe.json)

此处是响应子程序的构造性证明草稿和正确性检查，不是机器检验的C++形式化证明、完整LUT输出分享模拟或完整系统安全认证。标准重新随机化/噪声隐藏/图放置均不声称新原语；CCF级独立贡献尚未确认。已有模型质量和旧后端成本正结果保留。新配置的真实两方链路及同期代价仍待测量，不能套用先前0.977GB/8.41秒数字。

程序相对本阶段解析ROOT，依赖前序已固定FABLE/SEAL构建及其utils.cpp.o；重跑须另登记新时间窗口/输出目录，默认拒绝覆盖原结果。包含公开参数输入、源码和结果摘要；二进制/公开合成大夹具/第三方完整源码与论文PDF保存在NEWSMY。本阶段不是干净克隆一键安装包。前序依赖见[精确编码阶段](../2026-10-31-qwen-exact-packing/README.md)和[放置部件阶段](../2026-10-31-private-lut-placement/README.md)。
