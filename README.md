# LLM Cryptography Research / LLM结构密码协议探索

利用LLM的数学结构研究密码协议，记录文献查重、可复现实验、负结果和适用边界。项目持续进行；尚未确认新的密码原语或CCF C+级独立创新。

## 阶段成果

| 日期 | 内容 | 状态 |
|---|---|---|
| 2026-10-29 | [精确共享低位修正分类与有限gate/down协议](milestones/2026-10-29-exact-coarse/notes/20261029_exact_coarse_closure.md) | 432组完整局部协议实验通过；两个Qwen模型、CPU/CUDA；已归档 |

[实验汇总JSON](milestones/2026-10-29-exact-coarse/results/exact_coarse_comparison_report.json) · [受限证明草稿](milestones/2026-10-29-exact-coarse/notes/20261029_exact_coarse_joint_view.md) · [复现范围与CPU smoke](milestones/2026-10-29-exact-coarse/REPRODUCIBILITY.md)

相对同一研究实现中的直接32位分类，修正24位方案的本地CPU在线时间降低25.5%–32.9%，CUDA降低15.1%–22.5%；通信增加8n字节，仍为5次交换。这些是有限计算块结果，不能解释为整模型加速或对上游SIGMA/FuseFSS/Grotto的优势。

原始数据、代码检查点与下载资产保存在NEWSMY。这里上传阶段性源码和原始结果记录；不包含模型大权重、环境缓存、语料正文或第三方论文PDF。新一轮联合精度分配和完整整数FFN筛查仍在本地运行，待闭环后另记里程碑，不并入本轮已完成结论。

See [NOTICE](NOTICE.md) for attribution and [the reproducibility notes](milestones/2026-10-29-exact-coarse/REPRODUCIBILITY.md) for the limits of a clean clone.
