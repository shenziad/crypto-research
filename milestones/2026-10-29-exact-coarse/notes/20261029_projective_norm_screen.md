# 延迟注意力归一化：第一轮筛查为负

2026-10-29。只进行构造、代数和表示可行性筛查，没有实现攻击，也未声称得到新密码协议。

## 候选接口

单头注意力可把分子N、正分母D保留到残差后：`x+N/D=(Dx+N)/D`。进一步

`RMSNorm(u/D)=u/sqrt(mean(u²)+epsilon*D²)`。

必须同步缩放epsilon，不能只引用epsilon为零时的缩放不变性。多头输出投影混合各头N_h/Z_h，可取 `D=product_h Z_h`、`u=D*x+sum_h(product_{g!=h}Z_g)*W_h*N_h` 消去显式倒数，但分母乘积扩大动态范围。

## 实测

Qwen2.5-0.5B有14查询头、2个KV头；共享KV不等于共享softmax分母。使用原FP32模型全部24层，4个WikiText2窗口校准按层／公开查询位置的整数log2 gauge。固定gauge后，在32个新WikiText2、32个独立PTB窗口、每窗129查询上筛查，共198144个归一化实例。直接从softmax最大概率得到max-shift分母Z_h=1/pmax_h。所有未量化的含epsilon恒等式通过。

| 公共定点小数位 | 分母量化为0 | projective范数和超出2^62 | 直接x量化的对照 | 最大相对归一化输出误差 |
|---|---:|---:|---:|---:|
| 16 | 118 | 3227 | 0 | 1.1972 |
| 24 | 12 | 56511 | 1235 | 1.0241 |
| 32 | 0 | 179232 | 196966 | 0.07194 |

32位对照本身也不适合此63位范数累加，所以不能把该列单独解读为projective方案更省环。16位直接表示在这些样本中没有累加溢出，projective则出现分母下溢、范数溢出及严重局部误差。

这是必要条件筛查：实验人为使用已经正常归一化注意力后的残差x构造u=xD，尚未加入从私有各头分子构造u的舍入和成本。即使这样仍未通过固定小环／小误差要求，所以不继续为这个直接版本搭建昂贵MPC。私有动态重标定、多字环或单头特殊模型是不同构造，未据此一并否定，也不会未经核算继续投入。

原始记录：`results/projective_norm_probe.json`；分母统计和冻结gauge：`results/projective_norm_denominators.npz`；脚本：`experiments/projective_norm_probe.py`。这里的float64统计不是大整数范围证书、全模型质量或协议加速证据。

相邻先例包括FlashAttention延迟归一化，以及[SOAL ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/a4b293979b8b521e9222d30c40246911-Paper-Conference.pdf)的域缩放／异常值处理；未把搜索没有直接命中projective跨残差接口当作新颖性证明。

下一步转向“粗区间选择、细多项式取值”的误差可认证接口：尝试只减少区间定位精度而保留乘法体的细输入，先检验数学功能、公开随机dither定义和误差界，再决定是否值得模型筛查。
