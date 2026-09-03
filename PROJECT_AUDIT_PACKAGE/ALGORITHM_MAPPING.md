# 算法、流程与公式到代码映射

## 当前数据流程

```text
随机输入 [B,128,64]
→ GroupQuery.d_x→128 投影
→ 连续每16个位置一组的均值 query
→ Quantizer 压缩为 [B,128,32]
→ 每组 codec 展平并取 q_k 个信息位
→ MotherCode 添加 128 位头部并生成每角色校验位
→ 8192 位 QPSK + AWGN + 硬判决
→ Receiver/LDPCDecoder/Erasure
→ 解码信息位与统计
```

## 核心算法映射

| 算法 | 目的 | 输入/输出 | 伪代码 | 代码位置 | 主要配置 |
|---|---|---|---|---|---|
| 分组与 query | 让同组角色共享均值信息 | `[B,M,d_x]` → `[B,M,128]` 或 `[B,8,128]` | `projected=Linear(x); grouped=reshape; query=mean(grouped); return projected+query` | `group_query.py:GroupQuery.input_to_group/group_to_query` | `d_x=64`, `M=128`, `group_size=16` |
| codec 投影 | 得到每角色 32 维 codec | `[B,M,128]` → `[B,M,32]` | `hidden=Linear(x); codec=Linear(hidden)` | `quantizer.py:Quantizer.quantize` | `d_model=128`, `codec_dim=32` |
| profile 生成 | 固定 13 种资源分配 | profile ID → `q/n/mask/hash` | `构造 q,n; 校验 sum(n)=8064; 计算 SHA-256` | `profile_generator.py:ProfileGenerator` | `K=8`, `payload_bits=8064` |
| profile 母码 | 形成固定长度发送比特 | 每角色 `q_k` 信息位 → `n_k` 码字 | `parity=u@P mod 2; code=[u,parity]; concat(header,roles)` | `mother_code.py:encode_profile` | `header=128`, 总长 `8192` |
| QPSK-AWGN | 模拟软件信道 | 比特 → 含噪 I/Q → 硬比特 | `00/01/10/11→(±1,±1)/sqrt(2); y=s+n; sign(y)` | `software_channel.py` | `snr_db=10`, `random_seed=42` |
| 擦除标记 | 标记低可靠接收位 | 接收硬比特、软 I/Q → bool mask | `mask=abs(soft)<threshold` | `erasure.py:erasure_function` | `soft_threshold=0.5` |
| Min-Sum 译码 | 在小型校验图上迭代更新 | 32 位块 → 解码块 | `固定10轮校验节点/变量节点更新` | `decoder.py:LDPCDecoder.decode` | `max_iterations=10`，实现仍固定 10 轮 |
| 数学函数/LP | 计算概率、效率和约束解 | NumPy 数组/LP 约束 → 数值结果 | `linprog(c,A_ub,b_ub,A_eq,b_eq,bounds)` | `math_functions.py` | 由调用方提供 |

## 符号到程序变量

| 数学/方案符号 | 程序变量 | 含义 | 范围/单位 | 代码 |
|---|---|---|---|---|
| `M` | `self.M` | 角色/特征点数量 | 128 | `input_mask.py`, `group_query.py` |
| `d_x` | `self.d_x` | 输入特征维度 | 64 | 配置 `toy.d_x` / `model.d_x` |
| `d_model` | `self.d_model` | 中间投影维度 | 128 | `group_query.py`, `quantizer.py` |
| `codec_dim` | `self.codec_dim` | codec 维度 | 32 | `quantizer.py` |
| `q_k` | `q_list[k]` | 角色 k 信息位数 | 由 profile 指定 | `profile_generator.py` |
| `n_k` | `n_list[k]` | 角色 k 发送段长度 | 总和 8064 | `profile_generator.py`, `mother_code.py` |
| `p_e,k` | `compute_p_e_k` 返回值 | 角色边缘错误率 | 概率 | `math_functions.py` |
| `p_e,ij` | `compute_p_e_ij` 返回值 | 联合错误概率 | 概率 | `math_functions.py` |
| `eta_k` | `compute_eta_k` 返回值 | 归一化二元不确定性 | 无量纲 | `math_functions.py` |
| `p_u` | `compute_p_u` 返回值 | 未擦除但接收错误概率 | 概率 | `math_functions.py` |

当前没有发现正式实验方案文档，因此无法建立“论文方案符号→实现”的更高层验收映射；上表仅描述现有代码。
