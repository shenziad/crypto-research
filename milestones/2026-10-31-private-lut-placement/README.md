# 私有LUT放置部件与BFV证明义务审查

有界私有输出头的构造性完善阶段，已有质量/代价结果保持原样。独立C++部件实现平衡三元组映射和完整增广匹配：280,894个小图、1,214,286个前缀与Hall穷举一致；524,798个完整表映射与Python参考一致；2048桶全部匹配，固定公开测试密钥下最大负载453。

此阶段未接入实际FABLE/GC后端，不产生新的两方性能数字或完整安全认证。BFV审查明确区分密钥层与数据层，记录现有69位噪声配置在条件平移界下所需的残差上限780，以及尚未完成的全电路残差/完整密文模拟证明。未做攻击实验或密钥/失败搜索。总阶段353秒，低于30分钟上限；0个两方对子；一次固定完整表检查。

- [完整结论和下一步](notes/20261031_private_lut_placement_outcome.md)
- [登记目标和预算](notes/20261031_private_lut_placement_preregistration.json)
- [部件结果](results/private_lut_placement_check.json)、[BFV条件算术](results/bfv_flooding_obligations.json)、[阶段审计](results/private_lut_placement_stage_audit.json)
- [构建说明](sources/private_lut_placement_build/build_recipe.json)

Python程序相对本阶段目录解析ROOT；本阶段包含自有源码、公开参数和结果摘要。需本地NumPy/cryptography和已固定SEAL4.1.1依赖；空目录复跑前需按登记预算另建登记/输出，脚本默认拒绝覆盖原记录。不是干净克隆一键安装包。大夹具、二进制、第三方完整源码保留在NEWSMY；前序依赖信息见[精确编码阶段](../2026-10-31-qwen-exact-packing/README.md)。标准Hall条件、增广匹配和噪声平移引理不声称新原语；CCF级贡献尚未确认。
