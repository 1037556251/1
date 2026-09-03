# 实施规范：固定资源下的可审计语义角色协同感知

## 0. 一页执行摘要

### 0.1 唯一主问题

在每帧完整编码长度、调制方式、计划信道使用数和声明范围内的执行拓扑均固定时，场景自适应是否仍能通过一组具有稳定因果功能的地址化角色，改善协同 3D 检测的风险效用、校准和困难场景表现？

### 0.2 四项必须同时成立的核心结论

1. **角色不是普通分组。** 完整方法的角色必须具有跨场景稳定、角色间可分、干预后必要的功能签名，并显著强于参数匹配的分组固定瓶颈。
2. **自适应不是增加资源。** 学习到的角色索引源描述—保护配置必须在相同完整编码长度 `N0`、相同计划信道使用数 `C0`、相同母码图和相同解码迭代数下，优于最强静态、source-mask-only 与 code-mask-only 配置。
3. **风险上界不是一句保护直觉。** 必须在独立校准集上估计 post-quantization 单角色损失、场景—失败协方差、角色联合失败交互和未检出错误，并冻结 simultaneous UCB。`β=1` safe 路径依靠 bounded loss 结构性满足 almost-sure interaction premise；LP tight 路径只是在冻结有限 support 上审计的经验包络。进入测试后分别报告二者的 coverage/slack 和 profile rule 效用，禁止把 tight 的零 violation 写成 almost-sure 证明。
4. **可预测性是结构主张，不是硬件时限。** 必须用逐帧比特审计、静态图、操作序列、tensor shape、固定迭代、FLOPs 和 memory upper envelope 证明声明范围内的资源性质；服务器 latency 只作参考，不能升级为 WCET 或车载 deadline 保证。

只满足其中一项或两项，不得对外宣称完整机制成立。

### 0.3 主任务与数据

- 任务：单远端协作者到 ego 端的 LiDAR 协同 3D 目标检测；检测类别按各数据集官方可评估类别执行，跨数据集主比较统一报告 vehicle 类。
- 主数据：OPV2V，用于角色、有限 profile、软件信道、风险上界和 overload 扫描。
- 外部诊断：V2V4Real 不再要求全量训练。若数据与缓存已可用，只运行冻结 OPV2V checkpoint 的小规模 zero-shot 方向检查；它不作为 core gate。
- 主框架：OpenCOOD，主干为 PointPillars。先用同一冻结 checkpoint 一次性提取 canonical ego BEV 与 remote pre-admission token cache；预计至少 80% 的正式训练/评估 GPU-hours 从该缓存接口开始，不再重复 backbone。首次运行前将仓库 commit、checkpoint hash 与 cache schema hash 写入 manifest。
- 主运行点：`M=128, K=8, N0=8192 bits, QPSK, L=1`。这些数值仅作为初始冻结点；最终主运行点必须按第 4 节的验证集规则选择，不能按测试结果选择。
- 训练种子：至少 5 个，固定为 `{1103, 2207, 3301, 4409, 5513}`。
- 信道随机种子：主 AWGN 每个测试序列、训练种子和信道点固定使用相同的 5 组随机实现；burst/erasure stress 固定使用相同的 3 组随机实现（common random numbers）。稀有事件精度不足时保留保守 UCB，不通过临时增加 seeds 扩张计算。

### 0.4 simulation-heavy 最低完成包

- **H1 Role：** B5 Grouped、B6 No-role-loss、B8 ISRC，5 seeds；三种 accepted interventions、synchronized permutation、held-out Hungarian 和压力复审全部保留。B4 Flat 只作 3-seed secondary。
- **H2 Profile + Bound：** P0 Uniform、P1b Static-best、P3 Source-only、P4 Code-only、P5 Learned-joint、P5b-safe、P6 Oracle，以及完成全量 LP bootstrap 后的 P5b-tight empirical controller；五点 AWGN、两个 stress 条件和完整 E2-T Bound Audit。safe pipeline 必须先独立闭环。
- **H3 Exact Contract + Overload：** 三个 `N0` 的 bit-exact `N0/C0`、统一 graph/operator/shape/iteration、FLOPs、memory upper envelope，以及 explicit fallback 对 silent truncation 的 overload frontier。
- **不属于最低完成包：** Orin/SDR、TensorRT 大扫点、端到端多 seed 微调、V2V4Real 全量训练、公开强任务基线、P1a/P2/P5g/P5c 和大规模 wall-clock latency。它们只能在三块 core 证据全部完成后运行。

## 1. 从理论约束反推实验

| 理论对象 | 首要替代解释 | 必须运行的对照 | 主要证据 | 失败时的唯一结论 |
|---|---|---|---|---|
| 固定资源内的地址化角色 | 只是位置敏感分组或 query slots | Grouped、No-role-loss；Flat 为 secondary | Stab/Sep/Nec、认证角色比例、干预热图 | 不支持“角色”机制 |
| 跨训练实例的角色可比性 | 直接比较同编号造成伪稳定 | held-out Hungarian 对齐、匹配不确定性 | 匹配频率、匹配边际、对齐后签名相似度 | 只能做单实例契约内解释 |
| 联合源描述—保护自适应 | 只是额外参数、量化变化或测试标签泄漏 | Uniform、Static-best、Source-mask-only、Code-mask-only、Oracle | 风险效用 AUC、校准 AUC、oracle-gap closure | 不支持联合自适应价值 |
| 角色加权超额风险上界 | 只是“重要角色多保护”的直觉，或暗中假设失败与场景独立 | safe `β=1` structural anchor；post-quantization 单/双/多角色干预、真实 decoder failure、undetected-error、simultaneous UCB | safe coverage/slack；tight audited-support violation/slack；bound-selected regret | 分开判定 formal anchor、empirical envelope 和 controller utility；不得互相代替 |
| 固定执行拓扑 | 不同 profile 实际切换 codec/循环 | 图哈希、操作序列哈希、固定迭代、分配审计 | 100% 拓扑一致；逐项工作量一致 | 结构性保证无效 |
| 固定容量下安全退化 | 通过静默丢弃困难场景换来稳定 | overload flag、silent truncation、fallback | overload 风险、选择性风险、最坏复杂度箱 | 固定容量不可称为安全可管理 |
| cached semantic boundary | backbone 重算或 cache 精度差异制造结果 | 同 cache_id、online-vs-cache 回放、只读 hash | cache 等价性与 downstream GPU-hour 占比 | 结论仅限无法复现的特征实现 |
| 综合价值 | 只在平均 AP 上获得小增益 | 最坏箱、overload、风险指标、exact contract | risk–resource 非支配前沿与预注册门槛 | 降级为普通压缩实验 |

## 2. 强制公平性契约

任何违反本节的结果均作废，不得进入主结果表。

### 2.1 对所有可学习方法保持一致

- 相同的原始帧、训练/验证/测试序列和数据增强随机流。
- 相同 ego 输入、同一远端协作者、相同历史长度 `L`、相同有效性掩码信息。
- 相同的冻结 PointPillars checkpoint、cache schema、特征精度和 cache 文件；核心实验不允许端到端微调 backbone。
- 相同训练 epoch、优化器类别、学习率搜索预算、early-stop 规则和最大参数搜索次数。
- 相同任务头容量。无法做到逐参数相等时，参数量误差不得超过 5%，语义边界 FLOPs 误差不得超过 10%；超出必须给出参数匹配版本。
- 相同完整编码长度 `N0`，其中包含版本号、profile index、overload/validity 标志、角色区域、校验字段和全部保护位。
- 相同调制方式和 `C0=N0/2` 个 QPSK 符号；不得只比较 semantic payload。
- 相同信道实现与随机数；每个方法在同一帧上看到完全相同的噪声/突发状态。
- 相同母码长度、校验矩阵、解码器和固定迭代数。只有掩码、shortening、puncturing 位置可以随 profile 变化。
- 用于坐标对齐的 ego/remote pose、速度和时间同步元数据对所有方法完全相同，禁止携带检测特征。该固定控制开销不塞入 `N0`，但必须在 `environment.yaml` 中记录字段、位宽、更新频率和额外 channel uses，并在 full-link accounting 中相加。
- P5 的主结果不读取即时 CSI。解析上界控制器只在量化信道箱 `γ` 可获得的条件下运行；其 3-bit 反馈、估计误差和反馈时延必须单独计账，且不得把信息条件不同的结果写成纯算法公平胜负。

### 2.2 禁止做法

- 不得为完整方法单独增加更宽的 backbone、更多历史帧、更多 ego 特征或额外标签。
- 不得让 profile selector 接触测试标签、未来帧、真实 TTC 或接收端不可获得信息。
- 不得使用每个 SNR 单独训练或单独调参的完整方法，再与单一静态模型比较。
- 不得把 padding 后长度当作完整通信开销而忽略 header、FEC、CRC 或 profile index。
- 不得按单帧目标框做统计独立样本；统计单位是场景序列。
- 不得根据 `TEST_AUDIT` 结果修改阈值、主运行点或主要终点。
- 不得只保留表现最好的 seed；所有预注册 seed 都必须报告。

## 3. 数据准备与不可泄漏划分

### 3.0 协作者选择

核心设置严格使用一个 ego 和一个远端协作者，避免协作者数量成为隐藏的可变通信预算。对每个完整序列，在首个双方均有效的帧按 pose 距离选择最近的非 ego agent，距离并列时选 agent ID 最小者；该配对在整个序列冻结。若远端暂时缺帧，使用 erasure/cold-start 语义，不在帧级改选更有利的协作者。配对结果写入 `pair_manifest.yaml`。V2V4Real 的两车设置直接使用官方配对。

### 3.1 固定数据层级

OPV2V 建立以下互斥集合，并将序列 ID 写入 `splits_manifest.yaml`：

1. `TRAIN`：参数学习。
2. `VAL_SELECT`：选择 `M/K/N0`、超参数、静态 profile、校准温度与 fallback 阈值。
3. `VAL_ALIGN`：独立训练实例之间的 Hungarian 角色对齐；禁止用于阈值选择。
4. `VAL_BOUND`：只估计风险上界各项、simultaneous UCB 和解析 profile 查找表；禁止训练网络、选择角色门槛或汇报最终效应。
5. `TEST_AUDIT`：最终一次性统计与置信区间。

使用官方 train/validation/test 划分作为外层边界。将官方 validation 按**完整序列**分为 `VAL_SELECT/VAL_ALIGN/VAL_BOUND=40%/30%/30%`；不得拆帧。若任一子集少于 20 个序列，改用 `50%/25%/25%` 并在 `split_audit.csv` 明示有效序列数，不得拆帧补样本。官方 test 完整保留为 `TEST_AUDIT`。若官方 test 标签不可本地评估，则将官方 train 按序列重划为 `65/10/10/15` 对应四个非测试集合，并将官方 validation 用作外部复核；必须在日志中说明。

可选 V2V4Real zero-shot 不建立训练/校准子集；只保存按 `SHA256(sequence_id)` 升序取前 20 个完整序列的 `optional_domain_manifest.yaml`，不得依据类别数、距离、难度或初步结果挑选。

### 3.2 分层与泄漏检查

- OPV2V：优先按 town/场景 ID 分组，确保同一路段相邻帧不跨集合。
- V2V4Real（仅在执行可选 zero-shot 诊断时）：按 route/sequence 分组，不允许同一路线的相邻采集片段跨集合。
- 生成 `split_audit.csv`，列出每组的帧数、序列数、目标密度、可见距离、遮挡代理和原始 token 数分布。
- 对所有序列路径做 SHA-256 清单；训练开始前运行交集检查，任意序列 ID 或文件哈希交叉即停止。
- 归一化统计量仅由 `TRAIN` 计算；模型选择、温度与运行阈值只由 `VAL_SELECT` 计算；角色对齐只由 `VAL_ALIGN` 计算；风险上界及解析查找表只由 `VAL_BOUND` 计算。四类工件分别写入不同目录并校验数据来源。

### 3.3 复杂度与困难场景箱

箱边界只用 `TRAIN` 分位数确定并冻结：

- 原始 token 数 `n_t`：Q1/Q2/Q3/Q4 四箱。
- ego 端不可见但协作者可见的目标数：0、1、2、≥3。
- 目标距离：0–30 m、30–50 m、50–70 m、>70 m。
- 遮挡代理：ego 点数相对协作者点数比值的四分位；若官方遮挡标签可用，同时报告官方标签。
- 场景密度：有效 3D GT 数的四分位。

主困难箱定义为：`n_t` 最高四分位，或 ego 不可见/弱可见目标数 ≥2。两个条件分别报告，不得合并后只给一个结果。

## 4. 输入接口和主运行点选择

### 4.0 唯一 cached-feature 边界

核心运行链固定为：

`cached ego BEV + cached remote pre-admission tokens → admission → role/grouped/flat encoder → quantization → mother-code → software channel → fixed decoder → receiver/task head`

缓存按以下规则一次生成、全方法共享：

1. backbone 只能来自两种预先记录的来源之一：带固定 commit/hash 的公开 OpenCOOD PointPillars checkpoint，或在 OPV2V `TRAIN` 上训练一次、只用 `VAL_SELECT` 选 checkpoint 的共同 backbone；不得为方法分别训练。冻结后对每个 agent/frame 做一次 canonical 前向，保存 ego BEV、remote BEV 中阈值前的候选 token feature、二维坐标、置信度、时间戳、agent/sequence/frame ID 和有效性。GT 不写入 feature cache。
2. cache 使用 chunked Zarr/NPZ；feature 默认 FP16、坐标/置信度 FP32。每个 split 随机抽取 `min(500,该 split 帧数)` 帧做无标签 online-vs-cache 数值回放，要求 token ID/排序完全一致、FP16 反量化后的相对 L2 误差 `<1e-3`；共同 task head 的 AP@0.7 差异 `<0.1` 点只在 `TRAIN/VAL_SELECT` 检查，禁止为 cache 验证提前读取 TEST 标签。未通过则改为 FP32 cache，不允许放宽阈值。
3. `feature_cache_manifest.yaml` 记录 backbone commit/checkpoint SHA-256、预处理配置、坐标系、dtype、shape、每个 shard hash 与生成命令。cache 一经生成只读；任何 backbone/preprocessing 变化必须全量重建并产生新 `cache_id`，不同 cache_id 的结果不得配对。
4. 缓存边界必须位于 admission/role 之前；禁止缓存 admitted `M×d_x` 之后的角色、grouped blocks、query outputs、profile logits 或 decoded tensors。B5/B6/B8 的表示模块、role loss、跨 seed emergence 和 interventions 必须在每个正式 run 中真实训练/执行，不能用预计算 role vectors 代替。
5. 所有机制方法直接读取相同 cache row；channel randomness 在 cache 之后生成。用于干预的 clean receiver inputs、counterfactual profiles 和 noise keys必须引用同一 `cache_id/frame_id`。
6. 目标是正式 GPU-hours 中至少 80% 不执行 backbone。`run_cost.csv` 分别累计 cache extraction、downstream training、intervention、codec/channel sweep 和统计时间；未达到 80% 只影响成本目标，不改变效用结论。

### 4.1 可变 token 的唯一构造

从 remote cache 读取 PointPillars BEV 候选特征与空间置信度。令超过在 `VAL_SELECT` 冻结阈值的 BEV 单元形成原始集合；每个 token 必须包含：BEV 特征、归一化二维坐标、置信度和时间戳编码。原始集合保持可变基数 `n_t`。

Admission 采用确定性排序：置信度降序，置信度相同按 `(x,y)` 字典序；保留前 `M` 个并输出 `o_t=1[n_t>M]`。禁止使用 GT 决定保留对象。记录 admission 前后每个目标的可见证据和被丢弃 token 数。

### 4.2 阶段式容量选择，避免不可计算的全因子爆炸

只在 `VAL_SELECT` 上依次执行：

1. `M ∈ {64,128,256}`，固定 `K=8,N0=8192`。选择使 overload ≤5% 的最小 `M`；若 256 仍超过 5%，保留 256 并明确进入 overload-heavy 设置。
2. `K ∈ {4,8,12,16}`，固定选中的 `M` 与 `N0=8192`。选择满足“clean-link AP@0.7 距最佳值不超过 0.5 AP 点”的最小 `K`；若并列，选择认证角色比例更高者。
3. `N0 ∈ {4096,8192,16384}` bits。在 `Es/N0=2 dB` 上，先找三档中 AP@0.7 最佳值与风险损失最小值；选择同时满足“AP 距最佳不超过 0.5 点”和“风险损失不高于最小值的 1.05 倍”的最小 `N0`。若无预算同时满足，选择风险损失最小者；仍并列时选较小 `N0`。另两档保留用于前沿实验。
4. 所有选择完成后写入 `frozen_operating_point.yaml` 并计算 SHA-256。之后不得因测试表现修改。

初始建议值 `M=128,K=8,N0=8192` 仅用于打通流程，不能跳过上述冻结步骤。

### 4.3 网络结构锁定

所有机制比较使用以下共同结构：

- token feature 先线性投影到 `d_z=128`，加入二维正弦坐标编码与一位 validity。
- 角色提取器含 2 层 masked cross-attention，每层 4 heads、FFN width 256、pre-LayerNorm；K 个 learned queries 维度均为 128。不得增加输入依赖的迭代次数。
- 每个 128-d role 经共享 `128→64→32` MLP 得到 codec vector；接收端用共享 `32→64→128` MLP 还原。量化和 FEC 作用于 32-d codec vector。
- selector 对 K 个 role 做 mean/max pooling，并拼接远端置信度均值、最大值、熵、overload，以及由远端局部预测和共同 ego 运动元数据计算的 4 维摘要（clipped minimum predicted TTC、corridor-object count、weak-visibility count、maximum closing speed），共经 `MLP(264→128→P)` 输出 profile logits。所有摘要裁剪/归一化范围由 TRAIN 冻结；P3/P4/P5 使用同一 selector 宽度。
- receiver 将解码 roles、role embedding、validity、前一帧固定 K roles 和 ego BEV features 输入 2 层、4-head cross-attention fusion，再接共同检测头。序列第一帧或远端缺帧用 learned cold-start/erasure symbol。
- B4/B5 的低秩投影形式为 `vec(P)→R→K·128`，R 由脚本选择为使参数量最接近完整角色提取器且误差≤5%的整数；若单个 R 无法同时满足 FLOPs≤10%，加入实际参与前向的 residual MLP 并重新搜索 R。禁止用不参与前向的 dummy parameters 做表面匹配。

## 5. 必须实现的基线与消融

### 5.1 系统级基线

| ID | 方法 | 实现要求 | 排除的替代解释 |
|---|---|---|---|
| B0 | Ego-only sanity | 冻结 ego cache + 同一 task head；只验证合作信息是否有正增益 | 合作输入是否有价值 | core sanity，非主对手 |
| B3 | Silent truncation | overload 时截断但不发送 overload flag，其他链路相同 | 显式 overload/fallback 是否必要 | H3 core control |
| B4 | Flat fixed vector | 低秩投影读取完整固定 `M×d` 输入，输出与 `K×d_z` 相同总维数；接收端按一个整体向量处理 | 固定维数本身足够 | secondary，3 seeds |
| B5 | Grouped fixed bottleneck | 与 B4 同级的全输入低秩投影，输出拆为 K 个有序块；接收端使用与完整方法相同的位置 embedding，无 query-to-token 竞争 | 普通有序分组足够 | H1 core，5 seeds |
| B6 | Query slots, no role loss | 保留查询提取和全部 codec，只移除跨视图角色稳定项 | query 结构本身足够 | H1 core，5 seeds |
| B8 | Full ISRC | 完整角色、有限 profile 与风险接口 | 核心对象 | H1/H2/H3 core，5 seeds |

H1 的不可删比较只有 `B5/B6/B8`。B4 保留为较低成本次要对照，不参与 role gate 的最低完成条件；B0 只需在共同 cache 上训练/评估一次完整 5-seed task head sanity。B1、可变稀疏 B2、V2X-ViT、Where2comm 和其他强任务基线从 core matrix 删除；只有现成兼容 checkpoint/cache 且不影响三条主证据链时才可作为 appendix run，不能挤占 role intervention、Bound Audit 或 exact-contract 审计预算。

### 5.2 profile 控制组

在完全相同的角色编码器和母码上比较：

- P0 `Uniform`：每个角色相同源位和保护位。
- P1a `Static-ranked`：用 `VAL_SELECT` 的角色平均干预敏感度排序，一次性冻结；secondary。
- P1b `Static-best`：遍历同一有限 profile 表，在 `VAL_SELECT` 上选择风险损失最低的单一 profile，测试时永不改变。这是主要静态对手。
- P2 `Random`：从可行表均匀抽样；随机序列固定并与帧 ID 绑定，运行 5 个 random-profile seeds 后先取均值；secondary。
- P3 `Source-mask-only`：source bit-plane mask 可自适应，mother-code transmission mask 对所有角色固定；selector 参数量与完整 selector 匹配。由于 active source bits 改变时有效码率也会变化，结果只解释为“无自适应 code mask”的对照，不声称物理保护强度绝对不变。
- P4 `Code-mask-only`：source bit-plane 固定为 6 bit/标量，仅 mother-code transmission mask 可自适应。
- P5 `Learned joint`：联合选择源位与保护位。
- P5g `Grouped-joint`：用 B5 的 grouped features 驱动与 P5 参数/FLOPs匹配的 selector，使用完全相同的 profile 表和 codec；只在 core gates 完成后作为 optional diagnostic，不进入最低完成条件。
- P5c `Learned joint + CSI`：只用于 optional 信息条件消融；在 P5 允许输入后追加同一 3-bit `γ`，其余参数搜索和反馈计账与解析控制器一致。它不替代 scene-only P5 主结果。
- P5b `Bound-selected / Eq. (23)`：不训练 selector；用 `VAL_BOUND` 冻结的 simultaneous-UCB 表按 `(a,γ)` 执行有限表最小化。`P5b-safe` 使用 `β=1`，是 bounded-interaction premise 的 formal structural anchor；`P5b-tight` 使用交互 LP 的 simultaneous β-UCB，是 audited support 上的 empirically calibrated sharp controller/reference，不具有 deterministic 或 population-wide almost-sure 资格。测试时不更新任何统计量。
- P6 `Oracle`：使用当前帧标签从同一有限表中选择最小风险配置，仅作为不可实现上界，不进入实际方法排名。

若 P6 相对 P1b 的风险改善小于 3%，说明 profile 表没有形成有意义的自适应空间；必须先重新设计 profile 表，不能把 P5 的微小差异解释为成功。

### 5.3 角色干预与负对照

必须运行下列操作，且所有操作保持其他输入、channel realization 和 receiver state 不变：

1. `Receiver permutation`：仅置换接收端角色顺序。
2. `Synchronized permutation`：同步置换 query、packet region、profile mask 和 receiver embedding；输出 logits 最大绝对误差必须 `<1e-5`。
3. `Matched null`：角色替换为训练得到的 erasure embedding，并将 validity 置零。
4. `Within-stratum shuffle`：从相同数据集、密度箱、距离箱和信道箱的另一帧替换该角色。
5. `Conditional replacement`：以其他 `K-1` 个角色的余弦距离做 kNN，从 `VAL_ALIGN` 字典选择最近但来自不同序列的角色替代；`k=20`，对 20 个候选平均。
6. `Query sharing`：K 个 query 强制共享参数，保持总参数量并补偿到后续投影层。
7. `Remove role embedding`：接收端去除角色地址 embedding。

Matched null、shuffle、conditional replacement 被预先声明为三个 `accepted intervention operators`。角色 k 只有在**每一个** accepted operator 下都分别通过第 8.3 节的 Stab、Sep、Nec 与实践效应门槛，才记为 certified；不得用“三取二”、平均效应或方向投票补救某个失败 operator。Receiver/synchronized permutation、query sharing 和 remove role embedding 只作为结构诊断，不进入单角色证书。

## 6. 母码、比特打包与信道协议

### 6.1 固定拓扑母码

- 角色向量先经共享线性投影压缩为 `d_c=32` 个标量；每个标量用对称、逐通道均匀 bit-plane 量化，最大 8 bit，因此每角色最大信息位 `k_c=256`。量化范围只用 `TRAIN` 的 0.1%/99.9% 分位确定并冻结，超界值饱和。
- 主运行点每个角色使用同一 QC-LDPC mother graph，固定 `n_c=1536`、最大信息位 `k_c=256` 和 normalized min-sum 解码迭代 `I_c=10`；归一化系数由 `VAL_SELECT` 从 `{0.65,0.75,0.85}` 中选择一次。
- 所有 profile 都执行最大位宽量化、同一 parity 生成、同一 Tanner graph 和 10 次完整迭代；禁止 early stopping。
- 减少源位时，对未使用的低有效 bit-plane 做 shortening：编码前固定为 0，解码 LLR 设为大正值。
- 减少传输保护位时，对对应 mother-code 位置做 puncturing：解码 LLR 设为 0。
- 每个角色内部数组始终保留 `n_c` 长度；profile 只控制固定 gather/scatter mask。
- profile 表离线生成后运行 rank/可解码性检查；不满足校验矩阵秩或导致系统位不可恢复的条目直接删除。

每个 `(K,N0)` 构成独立 contract，可使用适合该预算的一个 mother graph；不同预算之间允许 `n_c` 不同，但同一 `(K,N0)` 内所有 profile 和所有比较方法必须共享同一 graph。主运行点之外，选择满足 `n_c ≥ ceil((N0-128)/K)+256` 的最小可用 QC lifting size，并将 graph hash 写入 `profiles.yaml`。

### 6.1.1 主运行点 profile 表的确定性生成

主运行点固定 header 区 `H=128 bits`，角色总区为 8064 bits，均匀时每角色发送 1008 个 mother-code 位置。profile 数为 `P=1+K+K/2=13`：

1. `m=0` uniform：所有角色 `q_k=192`（每标量 6 bit）、`n_k=q_k+r_k=1008`。
2. `m=1..K` single-emphasis：目标角色 `q=256,n=1264`；按循环索引选 4 个 donor，每个 donor `q=160,n=944`；其余角色 `q=192,n=1008`。总发送位置保持 8064。
3. `m=K+1..K+K/2` paired-emphasis：不重叠循环相邻角色对各 `q=224,n=1136`；与该对不重合的 4 个 donor 各 `q=160,n=944`；剩余角色保持 `q=192,n=1008`。

其他 `(K,N0)` 不直接复制上述绝对数。先令各角色 base region 为 `floor((N0-128)/K)`，余数按角色索引从小到大各加 1。single-emphasis 取 `D=min(4,K-1)` 个循环 donor，每个 donor 减少 64 个发送位置，目标角色增加 `64D`；paired-emphasis 取 `D=min(4,K-2)` 个 donor，每个 donor减少 64，两个目标各增加 `32D`。source bits 以 32-bit 为一级，在 `[128,256]` 内同方向调整；若发送位置小于 active source bits，先减少 source 级别。脚本最终必须验证每个 profile 的 `sum(n_k)=N0-128`、`q_k≤n_k≤n_c` 和所有 mask 合法，否则该 contract 不得运行。

source mask 始终保留 sign bit 和从最高有效位向低位扩展的嵌套 bit-plane；不得优先保留最低有效位。puncturing mask 必须包含全部 active systematic positions，其余位置按固定 parity 可靠性排序补足到 `n_k`。可靠性排序只由 mother graph 和训练信道 Monte Carlo 得到，不依赖测试帧。`generate_profiles.py` 必须输出每项的 `q_k,n_k,r_k`、mask hash、总和检查和可解码性结果。

### 6.2 帧字段

每帧按以下顺序打包，所有字段计入 `N0`：

`128-bit protected control block | role_regions`

- `version` 固定 8 bit。
- `profile_id` 为 `ceil(log2 P)` bit。
- 控制原文由 8-bit version、profile ID、overload、K-bit sender-validity 和 CRC-16 组成，补零至 64 bit，再用固定 BCH(127,64) 编码并补 1 bit，形成 `H=128 bits`。CRC 在发送端覆盖“控制字段 + 按 profile 定义的全部 active source bits”，并随控制原文一起编码。所有方法一致。
- 控制块解码失败直接拒绝整帧，并把当前 K 个远端角色全部映射到同一个 declared erasure/fallback 路径；风险审计中记 `E_k=1,∀k`。每个角色在固定 10 次解码后，用 syndrome 与在 `VAL_SELECT` 冻结的 LLR-confidence 阈值生成 decoder-validity；失败角色替换为 erasure。若所有角色均声明有效但重算 CRC 失败，同样拒绝整帧并置全部 `E_k=1`；若已有角色明确失败，CRC 作为诊断记录，接收端仍按角色 validity 执行固定图，不做数据依赖重解码。此时任何未被 erasure、却与发送端 active source bits 不同且仍被 receiver 使用的角色，都必须离线计入 `U_t=1`，即使全局 CRC 已提示不一致。declared erasure/fallback 的具体张量、validity 和历史输入必须与 E1 干预及 E2-T 的 `Δ(A)` 完全复用同一函数，禁止出现第二套“审计专用 erasure”。
- 每个 profile、每个 `N0`、空输入、全满输入和边界量化值必须通过无噪声 bit-exact round-trip。单比特翻转测试的通过标准是“被正确纠正，或被 syndrome/CRC 明确检测”，不得要求所有错误静默恢复。

### 6.3 信道矩阵

主信道采用 QPSK + AWGN，比较固定符号能量，因此横轴使用 `Es/N0 ∈ {-2,0,2,4,6} dB`，不用会随有效码率变化的 `Eb/N0`。

额外压力块只保留两个预注册条件，不做全因子扫描：

- `Burst-mid`：Gilbert–Elliott hard-error，平均 `BER=1e-3`、平均 burst length `16 bits`。
- `Region-erasure-mid`：每个角色区域独立擦除概率 `0.10`，header 仍按 `Es/N0=2 dB` 的同一软件物理信道传输。

主 AWGN 每帧每信道点运行 5 个固定 channel seeds；两个压力条件各运行 3 个。先读取缓存感知 token，避免不同方法因上游随机性不同而失配。软件 bit simulation 是 core protocol；不要求 SDR、射频前端或真实无线设备。

噪声使用 counter-based PRNG，key 固定为 `(dataset, sequence_id, frame_id, channel_model, channel_seed, bit_position)`。不同长度方法共享相同 bit position 的随机数前缀；超出共同长度的部分继续按 bit position 生成。不得通过依次调用全局 RNG 造成方法顺序改变噪声。

P5b/P5c 使用的 `γ` 反馈是独立、固定长度的 3-bit reverse-link 字段，不属于前向 `N0/C0`。主公式审计先假定该字段正确可得；full-link 结果必须额外加上其调制 channel uses、估计窗口与反馈时延，并按第 7.4 节运行错箱压力。scene-only P5 不发送、也不读取此字段。

## 7. 训练流程

### 7.1 core cached training 与可选 sensitivity

**阶段 A（唯一 core training）：cached-interface training。** 冻结 PointPillars 与 feature cache；只训练 admission 后接口、B4/B5/B6/B8 表示模块、量化/信道 surrogate、selector、receiver 和共同 task head。H1/H2/H3 的所有主结论只来自阶段 A。

**阶段 B（可选 appendix）：end-to-end sensitivity。** 只有全部 core gates 完成且仍有预算时，才允许从同一 checkpoint 对 B5/B6/B8 各做 1 个预注册 seed 的端到端微调；它不进入主统计、不改变阈值，也不能替代 cached-interface 结果。默认不运行。

因此核心比较的研究边界明确从冻结 `M×d_x` semantic interface 开始；不得把 backbone 重新学习带来的收益归入角色或 profile 机制。

### 7.1.1 跨视图角色稳定训练

每个训练样本从同一 canonical cache row 在线构造两个 task-preserving feature views：对 token 坐标和 ego/remote BEV index 同步施加 `{identity,x-flip,y-flip,180°}` 中保持原 tensor shape 的 coordinate re-expression，再独立施加最多 10% 的低置信度 token dropout、feature-channel dropout≤5% 和零均值 Gaussian feature jitter（标准差为 TRAIN channel-wise std 的 1%）。变换后立即逆映射到 canonical receiver 坐标再计算任务输出和角色相似度，因此 GT、agent 配对和物理目标集合不变。dropout 只按 cache confidence 的 TRAIN 分位数决定，禁止读取 GT 判断“关键”。所有随机 key 由 `(cache_id,epoch,frame_id,view_id)` 决定。禁止 raw-point intensity augmentation、插值旋转、裁掉高置信 token 或任何要求重跑 backbone 的增强进入 core training。

角色对比温度固定 `τ=0.1`。`λ_role` 只从 `{0.05,0.10,0.20}` 选择；选择依据为 `VAL_SELECT` 风险效用与角色认证比例的词典序（先风险效用，差异小于 1% 时再看认证比例）。No-role-loss 使用相同增强和训练批次，只将 `λ_role=0`。

### 7.2 统一优化设置

- 优化器统一为 AdamW。B5/B6/B8 只允许以下六个 `(lr, weight_decay, λ_role, λ_ch, λ_cal)` 配置：`(3e-4,1e-4,0.10,0.1,0.05)`、`(1e-4,1e-4,0.10,0.1,0.05)`、`(1e-3,1e-4,0.10,0.1,0.05)`、`(3e-4,1e-3,0.10,0.1,0.05)`、`(3e-4,1e-4,0.05,0.1,0.05)`、`(3e-4,1e-4,0.20,0.5,0.10)`。缺失的 loss 项置零但搜索次数不变。B4 只复用 B5 选定的 optimizer schedule，不再单独网格搜索。
- 训练上限 60 epochs；以第 8.2 节的加权风险损失为 first-order early-stop 指标，patience 10。候选 checkpoint 风险差异小于 1% 时选 AP@0.7 更高者，再并列时选 ECE 更低者；所有方法使用相同规则。
- batch size 由显存确定后全方法锁定；如需梯度累积，比较方法使用相同有效 batch size。
- 所有 seed 完整训练；失败 run 必须保留日志并重跑同 seed，不得替换成新的“幸运 seed”。
- 混合精度设置、gradient clipping、学习率 scheduler 和数据增强写入统一配置。

统一目标为 `L_det + λ_role L_role + λ_ch L_decode + λ_cal L_cal`。`L_decode` 是 clean role 与 bit-accurate decoded role 的 Huber loss；`L_cal` 使用 detection Brier surrogate。量化 rounding 使用 straight-through estimator；LDPC 使用固定 10 次可微 normalized min-sum；selector 用 Gumbel-softmax 从温度 1.0 线性退火到 0.1，在第 40 epoch 后转 hard straight-through。测试时只用 hard argmax、真实 bit packing 和 hard CRC/erasure 逻辑。

`L_det` 对 B4/B5/B6/B8 使用同一风险权重：普通 GT 权重 1；第 8.2 节定义的关键 GT 权重为 `min(3,1+0.5w_i)`。禁止只给 B8 风险加权而让机制控制组使用不同目标。

### 7.3 profile selector 可用输入

主实验的 selector 只允许读取当前 K 个发送端角色、第 4.3 节定义的远端摘要、overload 和共同 ego 运动元数据。禁止读取即时 CSI、ego 感知特征、GT、未来帧或 receiver loss；训练时从 `[-2,6] dB` 随机采样，使 selector 学习跨信道稳健的场景风险分配，而不是获得免费信道侧信息。若额外研究 CSI-aware selector，必须单独标记，并让所有自适应对照得到相同量化 CSI，同时把反向反馈位数和反馈时延单独计账；该结果不能替代主实验。训练时若使用 differentiable surrogate，最终测试必须切换为真实 bit packing 和实际 decoder。

### 7.4 两类控制器的接口必须分开

1. **P5 scene-only：** 输入保持第 7.3 节不变，测试时不知道 `γ`；这是学习控制器的主运行条件。
2. **P5b analytical：** 输入是发送端可计算的离散风险层 `a` 与接收端反馈的量化信道箱 `γ`。风险层使用第 4.3 节已存在的发送端摘要，按优先级确定：`a=3` 为 overload；否则 `a=2` 为 clipped predicted TTC≤3 s 或 corridor-object count≥1；否则 `a=1` 为 predicted TTC≤6 s、weak-visibility count≥1 或 maximum closing speed>0；其余为 `a=0`。缺失 TTC 置为训练分布上限而非零。`a` 在 profile 选择前只计算一次，并对同一帧的所有 counterfactual m 保持不变；禁止由 receiver 输出、解码结果或 GT 反推。阈值在任何上界估计前写入 `risk_strata.yaml`。
3. `γ` 取最近的 `Es/N0∈{-2,0,2,4,6}` 箱，平局向更差箱取整；每帧用固定 3 bit 反馈。主 Bound Audit 使用真实量化箱以隔离公式本身，另做估计误差压力：`P(γ_hat=γ)=0.8`，其余概率等分给相邻箱，边界概率并入唯一相邻箱。
4. P5b-safe/P5b-tight 的查表键严格为 `(a,γ)`，表值为一个 profile ID；supported cell 同值并列时先选 `VAL_BOUND` 上 `R0` 较小者，再选最小 profile ID；unsupported cell 的全 1 并列固定回退 P1b。测试期间不允许重新估计、平滑、插值或回退到标签 oracle。
5. controller utility 比较固定为 `Static-best/P1b`、scene-only `P5`、gamma-aware `P5b-tight`、`Oracle/P6`；P5b-safe 必须先独立完成 formal-anchor 审计并并列报告。P5b-tight 只能称 empirical controller/reference。若执行 optional P5c，用它区分“解析规则质量”与“多获得 γ”两种收益；P5 与 P5b 的直接差异必须标记为不同信息条件。

## 8. 指标定义与统计方案

### 8.1 标准效用

- 3D AP 与 BEV AP，IoU `{0.5,0.7}`；vehicle 类为跨数据集主终点。
- Recall@IoU0.5，分别报告 0–30、30–50、50–70、>70 m。
- ego 弱可见目标召回：ego 点数处于训练分布最低四分位、协作者点数高于中位数的 GT。
- 推理输出校准：NMS 后保留 score≥0.1 的预测；按 score 降序与 GT 贪心匹配，IoU≥0.5 为正确。报告 15 个等频置信度箱的 detection ECE 与逐预测 Brier score；阈值和匹配规则在 `VAL_SELECT` 冻结。

### 8.2 决策导向风险

对每个 GT 目标计算相对纵向速度和 TTC。若速度不可用，用相邻 3 帧中心差分并在序列边界标记不可估计。关键目标定义为以下任一条件：

- `0 < TTC ≤ 3 s`；
- 距 ego ≤20 m 且位于 ego 行驶走廊；
- ego 弱可见而协作者清晰可见。

行驶走廊使用 ego 航向方向前方 50 m、左右各 3.5 m 的矩形；若地图车道信息可靠，额外报告基于车道多边形的结果，但矩形定义保持为跨数据集主结果。

主风险指标：

`Critical Miss Rate = 未匹配关键 GT 数 / 关键 GT 总数`。

辅助风险损失对每个漏检关键目标赋权 `min(4, 3/max(TTC,0.5))`；无有效 TTC 但满足其他条件者权重为 1。权重与阈值在查看 `TEST_AUDIT` 前冻结。

用于 early stop、profile 选择和 SNR-AUC 的“加权风险损失”定义为漏检关键目标权重之和除以关键 GT 权重之和；若预测框虽匹配但 localization error 使 IoU<0.5，按漏检处理。该指标越低越好，不能与 `1-AP` 混合成可调权重的复合分数。AP 与 ECE 始终单独报告。

Critical Miss Rate 在每个重采样集合上用“总漏检关键 GT / 总关键 GT”计算，不先平均单帧比例。没有关键 GT 的序列保留用于 AP/校准，但不向该指标贡献分子或分母；若某个完整评估子集关键 GT 总数为 0，该子集指标记为 NA，不用 0 代替，并在结果中报告有效关键 GT 数。

风险上界使用同一任务方向、但必须是逐帧有界损失：`ell_t = min(1, 本帧漏检关键目标权重之和 / max(本帧关键 GT 权重之和,1))`，因此 `0≤ell_t≤Lmax=1`。无关键 GT 帧的 `ell_t=0`，但仍参与 decoder failure、undetected error 和 `p_u` 统计。全数据效用仍按上一段 ratio-of-sums 报告；不得把逐帧均值冒充 Critical Miss Rate。`ell_t` 只服务于风险上界及其控制器，定义在 `TEST_AUDIT` 前冻结。

### 8.3 角色证书

影响签名固定为六维：分类损失变化、定位损失变化、Brier 变化、Critical Miss Rate 变化、远距离召回变化、ego 弱可见召回变化。各维用 `VAL_ALIGN` 中未干预样本的标准差归一化。

- `Stab_{k,o}`：accepted operator `o` 下，预定义场景子集之间归一化影响签名余弦相似度的最小值。
- `Sep_{k,o}`：operator `o` 下，角色 k 与最近其他角色的归一化签名欧氏距离。
- `Nec_{k,o}`：operator `o` 干预角色 k 后风险效用损失的序列均值。

对每个 `o∈{matched-null, within-stratum-shuffle, conditional-replacement}`，门槛均固定为：`LCB95(Stab_{k,o})>0.80`、`LCB95(Sep_{k,o})>0.25`、`LCB95(Nec_{k,o})>0`，且风险恶化达到 0.5 个百分点或标准化效应 `0.2 SD`。角色证书取三个 operator 判定的逻辑与；完整方法至少 75% 角色通过，且比 Grouped 和 No-role-loss 的认证比例高至少 25 个百分点，才能称为系统性角色结构。若在首个正式训练前确需改变门槛，必须在 `change_log.md` 记录理论理由；运行 `TEST_AUDIT` 后不得改变。

上述 LCB 以完整序列为 cluster，对全部 `K×3` 角色—operator 判定做 one-sided Romano–Wolf simultaneous 95% 校正；不是逐项未校正区间。实践效应门槛也按对应 simultaneous LCB 判定。E1-R 复用相同多重性口径。

E1 的 clean-link `Nec_{k,o}` 只用于角色证书。风险上界中的 `w_k^(m)` 是 profile m 完成 source quantization 后、以 declared erasure 作用于角色 k 得到的逐帧有界损失增量；二者必须存入不同字段，禁止复制或别名复用。

### 8.4 跨训练实例对齐

- 对每对训练 seed，在 `VAL_ALIGN` 上构造角色签名距离矩阵并做 Hungarian assignment。
- assignment 冻结后，只在 `TEST_AUDIT` 计算对齐后 Stab/Sep/Nec。
- 对序列做 10,000 次 bootstrap，报告每个匹配的出现频率和最优/次优总成本差。
- 匹配频率 <80% 或边际置信区间包含 0 的角色记为“跨实例对应不确定”，不得强行编号对应。

### 8.5 统计单位与检验

- 独立统计单位：完整场景序列。
- 所有方法差异采用两层 paired bootstrap，10,000 次：先重采样训练 seed，再在每个 seed 内按相同序列 ID 重采样；方法之间保持配对。主 AWGN 的 5 个、stress 的 3 个 channel seeds 先在每个序列内求均值，不作为独立样本；信道随机性另做嵌套敏感性分析。
- 推断性主要终点有四个：H1a 认证角色比例、H2a P5 相对最强核心经验对照的风险损失 SNR-AUC、H2b P5b-tight 相对 P1b 的 bounded-risk SNR-AUC、H3a overload 下 declared fallback 相对 silent truncation 的 bounded-risk 差。四者使用 Holm 校正，family-wise `α=0.05`。最坏复杂度箱 Critical Miss 仍强制报告，但作为 secondary，不再扩张主检验族。
- Bound 审计分三级，均不靠显著性检验“证明”数学结论：一级检查 P5b-safe 的 `β=1` 是否按 bounded loss 正确实现，从而结构性满足 interaction premise；二级只评估 P5b-tight 在 `VAL_BOUND` cross-fit 与 TEST audited support 上的 empirical coverage；三级评估 controller utility。tight 的 coverage、slack 或零 confirmed violation 不得升级成 unknown population 上的 almost-sure guarantee。
- AP、ECE、各 SNR 点和各消融为次要终点，报告效应值与 95% CI，不进行“挑显著”解释；若执行服务器 reference latency，只报描述统计，不进入推断性终点。
- 同时报绝对差、相对差和 CI；禁止只报 p-value。

## 9. 五组主实验及通过门槛

### E1：角色形成与替代解释排除

**输入条件：** clean link、冻结 backbone、选定 `M/K/N0`、5 seeds。  
**核心比较：** B5、B6、B8；B4 作为 3-seed secondary。  
**步骤：**

1. 训练全部方法并保存每 epoch checkpoint、query、角色 embedding。
2. 在 `VAL_ALIGN` 完成跨 seed 对齐。
3. 在 `TEST_AUDIT` 对 B5/B6/B8 运行三种 accepted interventions、synchronized permutation 和 Hungarian 审计；B4 只运行 matched-null 与 synchronized permutation，作为 flat diagnostic。
4. 输出 6×K influence heatmap、每角色 Stab/Sep/Nec 森林图、认证比例及 assignment uncertainty。
5. 检查 synchronized permutation 数值等价性。

**通过门槛：** B8 满足第 8.3 节的 75% 认证比例与 ≥25 pp 优势；同步置换误差 `<1e-5`；Grouped 即使 receiver permutation 敏感也不得自动通过证书。若只观察到 receiver permutation 掉点，E1 判定失败。

### E1-R：部署压力下的角色证书复审

**目的：** 检查 clean-link 已认证地址在信道错误与 overload 下是否仍保持可区分、稳定且必要的 downstream effect，而不是重新发现一套压力条件专用角色。  
**输入条件：** 只使用 E1 已冻结的 5-seed checkpoints、Hungarian assignment、归一化尺度和三种 accepted operators；禁止再训练、重新对齐或重设阈值。  
**压力网格：** AWGN `Es/N0={-2,2,6} dB` 使用同一 5 个 channel seeds；`Burst-mid` 使用 3 个；均按 `overload∈{0,1}` 分层。clean/no-overload 是参照格。

**步骤：**

1. 对每个 E1 clean-certified 角色，在每个压力格和每个 accepted operator 上重新计算六维签名、`Stab_{k,o}`、`Sep_{k,o}`、`Nec_{k,o}`；固定 E1 的标准化尺度，不能在压力格内重新归一化。
2. 报 `signature cosine(clean,stress)`、证书保留率、Nec 衰减比例和角色 erasure rate；结果按 overload 与非 overload 分层。
3. 同时运行 synchronized permutation，确认 channel corruption 只改变数据值而未破坏地址—packet—mask—receiver 的 gauge 等价性。
4. 报从 clean 到每个压力格的全量曲线，不得只选仍通过的角色或中等信道点。

**通过门槛：** 在 `2 dB/no-overload` 与 `6 dB/overload` 两个预注册部署格，至少 75% 的 clean-certified 角色仍对三个 accepted operators 全部通过，且跨压力签名余弦的角色中位数 LCB95>0.80；同步置换误差仍 `<1e-5`。`-2 dB` 和 burst 格作为失效边界，不要求通过但必须报告。未通过时，角色证书只限定于 clean 条件，不能描述为部署压力稳定。

### E2-T：角色加权风险上界与有限表控制器审计

**目的：** 独立检查 safe formal anchor 是否按结构性条件正确闭环，评估 tight empirical envelope 在冻结 support 上的 coverage/slack，并检验由两张冻结表选择的 profile 是否是有效 controller/reference。P5 的收益、tight 的经验 coverage 和 safe 的 structural premise 三者不能互相替代。

**输入条件：** E1 通过；角色模型、profile 表、风险层 `a`、信道箱 `γ`、loss、decoder validity、CRC 逻辑全部冻结。所有上界估计只读取 `VAL_BOUND`；每个训练 seed/checkpoint 独立生成一张上界表，不跨模型池化。生成 `bound_freeze_manifest.yaml` 和 SHA-256 后才允许访问 `TEST_AUDIT`。

#### E2-T.0 必须先通过的公式单元测试

1. error-free replay 必须满足 `E_k=0,U=0,R=R0`，逐帧最大数值误差 `<1e-8`。
2. 人工 single erasure 必须满足缓存的 loss 差与 `Δ({k})` 一致，且 `d_k=max(Δ,0)`；人工 all-role reject 必须产生 `E_k=1,∀k`。
3. 构造 8 个可手算样本验证 covariance：完全独立时 `η=0`，正/负耦合时实现值分别等于手算绝对协方差。
4. 构造边缘概率相同但联合排列不同的二角色流，验证 `p_e,i/p_e,j` 不变而 `p_e,ij` 改变；构造一个 remaining-valid bit error，验证无论 CRC diagnostic 是否告警都计 `U=1`。
5. 对随机 1,000 个 `(t,m,A)` 检查严格 `β=1` RHS 逐样本覆盖，并检查所有概率/损失/协方差值位于声明范围。任一失败不得开始 `VAL_BOUND` 估计。

#### E2-T.1 每个符号对应的可执行观测量

对每个 `(train_seed, a, γ, m)`，所有 profile 在完全相同的帧和 5 个主 AWGN channel seeds 上运行，保存逐帧逐信道实现；定义：

- `R0(m|a,γ)`：profile m 完成 source quantization/dequantization、但角色区和控制区 error-free 时 `ell_t` 的序列簇均值。即使数值不随 γ 变化，也保留 γ 维度以保证所有项条件一致；实现应检查同一 `(a,m)` 跨 γ 的 `R0` 数值误差 `<1e-8`。
- `d_{t,k}^{(m)}=[ell_t(erasure{k})-ell_t(error-free m)]_+`，`w_k^(m)=E[d_{t,k}^{(m)}|a,γ]`。erasure 在 source quantization 后施加，并调用第 6.2 节唯一 declared erasure/fallback 函数，包括相同 validity 与历史输入；禁止读取或复制 E1 clean-link `Nec`。
- `E_{t,k}^{(m,γ)}=1` 当固定 10 次解码后角色 k 被 syndrome/LLR validity 映射到 declared erasure，或 control/全有效 CRC reject 把整帧映射为全角色 erasure。`p_e,k` 为该事件概率；`p_e,ij` 为同一 frame/channel realization 内 `E_i=E_j=1` 的联合概率，不能用边缘概率乘积替代。
- `η_k^(m)=|Cov(E_{t,k}^{(m,γ)},d_{t,k}^{(m)}|a,γ)|`。协方差直接由同帧配对的失败指示与 post-quantization 单角色损失计算；不得打乱 scene、以 `E[E]E[d]` 替代或预设为零。5 个 channel seeds 是同一序列内重复测量。
- `U_t=1` 当至少一个与发送端 active source bits/控制字段不同的值仍被 receiver 使用，且该错误没有被映射到对应角色 erasure 或整帧拒绝；即使 CRC diagnostic 已告警但因已有局部 erasure 而继续处理，也按此定义检查 remaining-valid roles。`p_u=Pr(U_t=1|m,a,γ)`。bit-accurate transmitter truth 只用于离线判定 U，不提供给 receiver。
- `R(m|a,γ)`：真实信道、真实 decoder/CRC/erasure 路径下 `ell_t` 的序列簇均值。

上界逐 cell 计算为：

`B(m|a,γ)=min{Lmax, R0_bar + Σ_k(p_e,k_bar·w_k_bar + η_k_bar) + Σ_{i<j}(p_e,ij_bar·β_ij_bar) + Lmax·p_u_bar}`，其中 `Lmax=1`。注意 `η_k` 是协方差绝对值本身，公式中不再乘 `p_e,k`。

两张冻结表都严格执行 `m*(a,γ)=argmin_{m∈M} B(m|a,γ)`；safe/tight 的唯一区别是 `β_bar` 的来源。每个 `(a,γ)` 必须同时保存所有 m 的 B、被选 profile、tie-break 原因和 profile contract hash，不能只保存最终 ID。

#### E2-T.2 交互常数与 bounded-interaction 审计

1. 对每个 profile 和每个 `VAL_BOUND` 帧，缓存所有 8 个 singleton erasure；对全部 28 个角色对计算 `Δ_t({i,j})` 和正交互残差 `[Δ_t({i,j})-d_{t,i}-d_{t,j}]_+`。
2. 对 `K=8` 主运行点，在每个风险层分层抽取至少 2,000 帧（不足则全取），穷举全部 `2^K-1` 非空角色子集；其余帧至少执行全部 pair、失败集合 `A_t` 以及 single-emphasis/paired-emphasis 涉及的 top-3 子集。该步骤缓存 post-quantization roles，并把多个 intervention mask 合并成 receiver-only batch；不得为每个子集重复 backbone、量化或信道编码。
3. **safe 表：formal structural anchor。** 使用 loss 有界性给出的确定性常数 `β_ij=1`。因为 `0≤ell≤1`，任意 `|A|≥2` 时至少一个 pair 项已给出 1，从而逐样本覆盖 `Δ_t(A)≤1`；`|A|=1` 由 `d_k=[Δ({k})]_+` 覆盖。这里的 formal 资格只指 bounded-interaction premise 由构造满足，不表示有限 TEST 能“证明”整个上界定理。
4. **tight 表：empirically audited interaction envelope。** 在 `VAL_BOUND` 上解非负线性规划：最小化 `Σβ_ij`，约束每个已审计 `(t,A)` 满足 `Δ_t(A)≤Σ_{k∈A}d_{t,k}+Σ_{pairs⊂A}β_ij`；以 sequence bootstrap 对 β 加 simultaneous upper margin，生成 P5b-tight。5-fold cross-fit 与 TEST 零 confirmed violation只授予 `empirically supported on audited support` 状态，不能授予 deterministic、structural 或 unknown-population almost-sure 状态。
5. TEST 上仍对真实失败集合 `A_t` 和全部 pair 复核 bounded-interaction。任一 `Δ_t(A)` 超过 tight RHS，即把 P5b-tight 标为 `empirical-envelope violation`，保留其描述性 controller performance 但撤销 calibrated-envelope 资格；safe RHS 若出现违反，说明 bounded loss、`Δ/d` 或实现定义错误，立即停止。

仅靠增加 VAL/TEST 样本、bootstrap 次数或零 violation 不能把 P5b-tight 升格为 formal theorem object。若未来需要这种资格，必须先把理论 contract 改为概率型/期望型 interaction residual 并重新给出相应推导；执行团队不得通过实验措辞自行完成该升级。

#### E2-T.3 两类残差的受控耦合压力

该步骤只做离线 theorem-falsification replay，不训练控制器、不进入主效用排名，也不向 receiver 提供标签：

1. **scene–failure coupling：** 在每个 `(a,γ,m)` 内保留真实 decoder 产生的整行 detected-erasure mask，使用 assignment 在场景间重新配对这些 mask 与缓存的 `d_{t,k}^{(m)}`：分别最大化和最小化 `Σ_{t,k}E_{t,k}d_{t,k}`。这样 `p_e,k`、`p_e,ij` 和失败集合大小分布保持不变，只有 scene–failure dependence 改变。计算实际 replay risk、`η_k` 与去掉 η 的反事实 RHS。
2. **role–failure coupling：** 对 `VAL_BOUND` 中 `β_ij` 最大的预注册 top-5 角色对，在保持各自 failure stream 的边缘计数不变、并随机打乱其与 `d` 的对应以压低 η 后，构造 Fréchet 上/下联合排列，改变 `p_e,ij`。计算实际 replay risk、完整 RHS 与把 `p_e,ij` 错写成 `p_e,i·p_e,j` 的 RHS。
3. 每次 replay 只允许调用 declared erasure path；undetected corrupted values 不参与该反事实，以免同时改变 `p_u`。assignment 和 top-5 pair 由 `VAL_BOUND` 冻结，TEST 只执行同一规则。
4. P5b-safe 必须覆盖两种耦合极端；不覆盖即实现错误。P5b-tight 的结果只解释为这些 replay support 上的经验覆盖或违反。去 η/独立 pair 版本若仍覆盖，只说明该数据上的残差数值较小；不得为了制造失败而改变 loss 或干预。

#### E2-T.4 simultaneous UCB、稀有事件和冻结

1. 全局 one-sided simultaneous 置信预算固定为 `α=0.05`：`R0/w/η` 连续项族使用 `α_cont=0.025`，概率项族使用 `α_prob=0.025`，由 union bound 保证总 family-wise error≤0.05。不得让两个子族各自使用完整 0.05。
2. `VAL_BOUND` 内以完整序列为 cluster 做 10,000 次 one-sided Romano–Wolf max-t bootstrap，同时覆盖所有支持 cell 的 `R0,w,η` 和 tight `β`；每个 bootstrap replicate 重新求交互 LP，不能固定点估计 β。bootstrap 中同序列的全部帧与 5 个 channel seeds 整体携带，不当作独立样本。任何标准误为 0 或 bootstrap 退化的项，改用其合法范围上的 Hoeffding–Bonferroni UCB，取两者较大值。每个训练 seed 独立构造 UCB，最终跨 seed 的效应比较再按第 8.5 节两层 bootstrap。
3. `p_e,k/p_e,ij/p_u` 先在每个序列内计算事件比例，再以这些 `[0,1]` cluster means 构造 one-sided empirical-Bernstein UCB，并对全部 G 个概率项使用 Bonferroni `α_prob/G` 保证 simultaneous coverage。若所有 cluster means 都为零，额外以“该序列是否至少发生一次”为 Bernoulli 单位计算 one-sided Clopper–Pearson `α_prob/G` 上界，最终取两种上界较大者；因此零事件仍给非零上界。禁止对逐帧或 5 个 channel seeds 直接套独立二项分布。safe 表 `β=1` 无需估计误差；tight β 与连续项共享 `α_cont`，并纳入同一次 max-t family，而不是另开 0.05。
4. cell 支持门槛为至少 20 个独立序列且 500 个有效帧。失败事件数不是删 cell 的理由；稀有事件必须保留并使用精确上界。未达样本门槛的 cell 将 `B=1`、标记 `unsupported-conservative`，P5b-safe/P5b-tight 均回退到 P1b，不得跨风险层借样本。
5. 在正式冻结前做 5-fold sequence cross-fit：每折用 4 折估计全部 UCB，在剩余折只查 coverage/slack；该结果只用于发现实现错误并给 tight 表授予 `cross-fit supported` 或 `cross-fit violation` 标签，不产生 formal validity 资格。完成后用全量 `VAL_BOUND` 重估一次并冻结 `bound_terms.parquet`、`bound_ucb_table.csv`、`bound_profile_table_safe.csv`、`bound_profile_table_tight.csv`、代码 commit、配置与数据哈希。
6. UCB 只能逐项向上截断到合法范围：概率、`R0/w/β` 在 `[0,1]`，`η` 在 `[0,0.25]`；最终 RHS 再截断到 1。禁止对 TEST 结果做温度缩放、整体乘系数或 slack 校正。

#### E2-T.4.1 tight LP 与 bootstrap 的工程规范

这些优化只能减少计算时间，不能改变 10,000 次 bootstrap、sequence-cluster resampling、LP 约束集合、solver tolerance 或 simultaneous family：

1. **safe-first：** E2-T.0、P5b-safe 全 cell、safe profile table 和 safe coverage pipeline 必须先独立跑通并冻结；tight 计算失败不得阻塞 safe 结果交付。
2. **receiver batching：** 预先缓存每个 `(seed,a,m,t)` 的 post-quantization roles 与 singleton loss；255 个 subset masks 按显存分块做 receiver-only batch。保存 `frames/s`、peak memory 和 batch size，不允许为提速减少 subset/frame/profile/seed。
3. **LP matrix prebuild：** 把每个 `(seed,a,γ,m)` 的约束写成只读 CSR/CSC matrix，并按 sequence 保存 row index。bootstrap replicate 只选择重采样序列对应的 row blocks，不重新构造符号表达式。
4. **exact active-constraint generation：** 可用上一 replicate/cell 解 warm-start，并先解 active rows；随后必须向量化扫描该 replicate 的**全部**约束，加入所有 violation `>1e-10` 的 rows，循环至 full scan 零 violation。最终 full-matrix max violation、迭代数和 active-row 比例写入日志。只筛选一次或不做最终 full scan 的结果无效。
5. **固定数值规则：** solver、版本、primal/dual feasibility tolerance（均≤`1e-9`）、tie-break、线程数和随机 seed 写入 `lp_solver.yaml`。infeasible、unbounded、超 tolerance 或未收敛 replicate 不得丢弃，必须修复后以同一 replicate ID 重跑。
6. **并行但可复现：** bootstrap ID 固定为 `0..9999`，随机 key 由 `(train_seed,cell_id,fold,bootstrap_id)` 决定；可跨 CPU process/node 并行，但最终按 ID 合并，禁止以先完成的 10,000 个替代预注册 ID。
7. **先 benchmark、不得缩水：** 在 full `VAL_BOUND` matrix 上先跑独立 dry-run IDs `bench-000..bench-099`，输出 median/P95 LP time、constraint-scan time、active rows、RAM 和预计总 wall time。这 100 个 dry runs 永不进入统计 bootstrap；正式估计仍完整运行 IDs `0..9999`。若预计超出资源窗口，继续交付 safe pipeline，并把 tight 标为 `compute-pending`；不得把 10,000 降为 1,000、减少 folds/cells 或仅保留成功 replicates 后仍称 simultaneous tight UCB。

#### E2-T.5 TEST 评价与控制器比较

1. 对每个 supported `(a,γ,m)` 报 `R_hat`、冻结 `B`、absolute slack `B-R_hat`、relative slack `(B-R_hat)/max(R_hat,0.01)`、各上界项占未截断 RHS 的比例，以及序列数/帧数/关键 GT 数。TEST 的 simultaneous LCB/UCB 同样以完整序列为 cluster、在全部支持 cell 上做 one-sided max-t；退化项使用合法范围上的 Bonferroni concentration fallback。
2. coverage 同时报三种状态：`covered` 为 `R_hat≤B`；`confirmed covered` 为 TEST 风险的 simultaneous one-sided 95% UCB≤B；`confirmed violation` 为 simultaneous one-sided 95% LCB>B；其余为 inconclusive。逐 cell coverage 比例附 Clopper–Pearson 95% CI，不能把一次测试表的比例解释成重复抽样覆盖率；P5b-tight 的这些状态字段统一加前缀 `empirical_`。
3. 核心比较固定为 P1b、P5、P5b-tight、P6；按 γ 报 bounded-risk、风险 SNR-AUC、与 P6 的 profile agreement、oracle regret 和 oracle-gap closure。并列报告 P5b-safe，以及带 `γ_hat` 误差和 3-bit feedback latency 的两种 P5b 压力结果；P5c 仅在 optional run 已完成时加入。
4. 分别报 P5b-safe/P5b-tight 的 profile 使用熵和 `(a,γ)→m` 热图。若 P5b-tight 超过 90% 支持 cell 选择同一 profile，仍可报告覆盖结果，但不得称解析控制器具有实质自适应性。
5. 并列报告 full bound、`η=0`、`p_e,ij=p_e,i·p_e,j`、删除 pair term、删除 `p_u`、错误复用 clean `Nec` 六种公式审计；后五种只用于展示每个理论部件的必要性，禁止用于选择 profile 或宣称更紧上界。

**formal-anchor implementation gate：** P5b-safe 的逐样本 structural unit tests 必须 100% 通过；其冻结 bound 不得有 confirmed violation，且 point-estimate coverage≥95%。若失败，结论是 theorem-to-implementation mapping 失败或其他估计项不足，不能归因于 almost-sure β premise。  
**tight empirical-envelope gate：** 只有完整 10,000-replicate pipeline、5-fold cross-fit 和 TEST audited support 均无 confirmed violation且 point-estimate coverage≥95%，才标记 `empirically calibrated on audited support`。该 gate 通过也**不**表示 almost-sure assumption、unknown population 或 deterministic guarantee 成立。出现 violation 时保留负结果并撤销 tight-envelope 资格；出现 inconclusive cell 时标记 `insufficient precision`。  
**controller utility gate：** P5b-tight 相对 P1b 的 bounded-risk SNR-AUC 至少降低 3%、Holm-adjusted paired 95% CI 不跨 0，并闭合至少 20% 的 `(P6-P1b)` gap；同时 median absolute slack<0.20，且未截断到 1 的 supported cell 至少占 50%。若 tight envelope gate 未通过，仍可描述 controller performance，但只能称 empirical heuristic/reference，不能称 formal/certified bound-selected rule。

### E2：固定资源内联合自适应价值

**输入条件：** 冻结 E1 的角色模型；所有 profile 使用相同母码图、`N0/C0` 和 10 次迭代。  
**核心比较：** P0、P1b、P3、P4、P5、P5b-safe、P5b-tight、P6。P1a/P2/P5g/P5c 均为 optional diagnostics，不进入最低完成条件。  
**步骤：**

1. 先做 P6 oracle-gap 检查；gap <3% 时停止并重设计 profile 表。
2. 在 AWGN 五个 SNR 点和 5 channel seeds 上运行。
3. 再运行 `Burst-mid` 与 `Region-erasure-mid` 两个条件，各 3 channel seeds；不扩展成 stress 全因子。
4. 报 AP、Critical Miss Rate、风险损失、ECE、role erasure rate、CRC reject rate。
5. 计算风险损失相对 `Es/N0` 的梯形 AUC；所有方法用相同横轴。
6. 统计 P5 的 profile 使用率、选择熵、风险箱条件分布和场景内 counterfactual profile swap。若 P5 退化为单一 profile，必须由 P1b 结果验证其等价性，不得称为自适应。若运行 P5g/P5c，单独标注 diagnostic，不混入 core ranking。
7. 在不使用测试标签选择 profile 的前提下，以三个 accepted operators 中的保守值 `min_o Nec_{k,o}` 作为 clean 角色必要性，计算各风险箱中该值与平均 source/protection share 的 Spearman 相关，并用 10,000 次角色标签置换构造零分布；该分析只解释 selector 是否利用已认证角色，不得代替 post-quantization `w_k^(m)` 或效用门槛。

**通过门槛：** P5 相对风险表现最好的 `P0/P1b/P3/P4` 的风险损失 AUC 至少降低 5%，paired 95% CI 不跨 0；clean link AP@0.7 下降不超过 0.5 点；P5 至少闭合 `(P6-P1b)` oracle gap 的 30%。角色因果性由 E1 独立判定；不得用 E2 效用反推角色证书。

### E3：资源—效用—失效前沿

**输入条件：** cached-interface B8；固定第 4 节选定的 `M/K`，只扫描 `N0={4096,8192,16384}`，不再扫描相邻 `M/K` 或端到端模型。  
**比较：** B8 的显式 overload flag + declared fallback、B3 silent truncation、ego-only fallback；B5 只在主 `N0` 作为 grouped reference。  
**步骤：**

1. 对三个 `N0` 分别生成独立 exact contract，并运行逐帧 `N0/C0`、mask rank、bit round-trip、graph/operator/shape/iteration audit。
2. 在 OPV2V 主五点 AWGN 上输出 AP、Critical Miss、bounded risk、overload rate、fallback rate、FLOPs 和 memory upper envelope；最坏复杂度箱只按 TRAIN 边界统计。
3. 单独比较显式 overload + declared fallback、silent truncation 和 ego-only fallback；对 overload/non-overload 分层，不允许用多数非 overload 帧稀释风险。
4. 绘制 `bounded risk/AP – N0 – FLOPs – memory envelope` 前沿。服务器 reference latency 只作为附加列，不参与非支配判定和 gate。
5. **可选域诊断：** 若 V2V4Real 已经可用，使用第 3.1 节 manifest 的最多 20 个序列、OPV2V 冻结 checkpoint 与相同 cache schema做 zero-shot；不训练、不调参、不要求达到同方向，也不进入 pass/fail。未运行时直接标记 `not executed`，不构成缺项。

**通过门槛：** 三个 contract 的 bit/graph/operator/iteration 审计全部 100% 通过；B8 至少在一个 `N0` 上形成 risk–resource 非支配点；overload 下 declared fallback 的 bounded risk 不高于 silent truncation，paired 95% CI 上界≤0。外部域与 wall-clock latency 均不是 E3 gate。

## 10. 结构、工作量与参考运行时间审计

### 10.1 图与操作序列

在同一 TorchScript 或 ONNX 静态图上执行；二者任选其一并冻结版本，不要求 TensorRT 或边缘设备：

- 只导出一个接收 `profile_id` 的统一静态图；profile table 作为固定 `P×K×n_c` mask tensor，由一次固定 shape 的 index/gather 读取。禁止为每个 profile 分别编译一个 engine 后声称各自静态。
- 导出后保存 graph text、节点类型序列、tensor shape 列表和 SHA-256。
- 每个 profile、每个复杂度箱各抽 200 个 cached frames，记录 operator 序列；静态图解析覆盖全部节点。
- 操作类型、顺序、循环次数和 tensor shape 必须 100% 相同；仅数据值和 mask 内容允许不同。
- 记录动态内存分配次数。声明边界内 warm-up 后应为 0；若框架内部不可避免，必须证明次数与 profile/scene 无关。
- 检查 decoder 恰好执行 10 次迭代，不得因校验通过提前停止。
- 禁止数据依赖零跳过和 structured-sparsity 加速。若后端含自动 fusion，必须同时保存 fusion 前静态图与 fusion 后 trace，证明 profile 只改变 mask 值。
- 用静态 shape 和算子参数计算每帧 FLOPs/MACs；按 `parameters + persistent buffers + peak live tensors + declared backend workspace` 计算 memory upper envelope。报告值必须对 profile/scene 不变；它是结构上界，不是实际平台峰值的替代。

### 10.2 开发服务器参考 latency（非 gate）

不要求 Jetson Orin、SDR 或任何特定部署硬件。只在现有开发 GPU 上给出可复现的 reference latency，用于成本量级说明；不得解释为车载 deadline、WCET 或跨平台保证。

参考测量固定为：

- batch size 1、固定精度；200 cached frames warm-up，随后 1,000 frames ×3 轮；每轮随机化方法顺序，GPU 前后 synchronize。
- 只把 `cache read 完成后 → task output` 定义为 core boundary，分项报告 `T_admission/T_role/T_select/T_quant-code/T_decode/T_receiver`。
- 若顺手测量 raw input→output full pipeline，必须单独标记 `optional-online-reference`，不得与 cached boundary 混合。
- 报 median、P95、P99、mean 和 MAD；不设置 deadline、不计算 violation probability、不宣称 tail guarantee。
- 保存 GPU/CPU 型号、精度、driver、CUDA、PyTorch/ONNX runtime、操作系统、功耗模式和测量脚本。reference latency 不参与任何 pass/fail，也不要求不同机器复现绝对数值。

## 11. 执行顺序与停止规则

必须按以下顺序，前一 gate 未通过不得进入大规模后续计算：

1. **G0 cache/data gate：** 五类 split 无泄漏，online-vs-cache 等价性、cache hash、token/GT 可视化正确。
2. **G1 bit gate：** 所有 profile round-trip、长度、CRC 和边界测试通过。
3. **G2 topology gate：** graph/operator/iteration 完全一致。
4. **G3 sanity gate：** clean link 下 B8 不低于 Ego-only，且合作信息确实产生正增益。
5. **G4 role gate：** E1 达到逐 operator 角色证书门槛。
6. **G5 role-robustness gate：** E1-R 的两个部署格达到证书保留门槛。
7. **G6 oracle gate：** profile 表 oracle gap ≥3%。
8. **G7 formal-anchor gate：** `ell/R0/w/E/η/p_u` 单元测试与 safe `β=1` 的逐样本 structural envelope 100% 通过，safe pipeline 独立冻结。
9. **G8 empirical-tight/controller gate：** E2-T 分别给出 tight `cross-fit/TEST empirical support`、compute status 与 controller utility；不得输出统一的“theorem validity”标签。
10. **G9 adaptation gate：** E2 达到联合自适应门槛。
11. **G10 exact/frontier gate：** E3 的三个 exact contracts、overload/fallback 和 risk–resource frontier 完成。外部域与 reference latency 不属于 gate。

若 G2 失败，优先修 codec/导出实现，禁止用 latency 平均值掩盖。若 G4 失败，先检查角色机制与对照，不继续包装保护机制。若 G5 失败，后续结果只能使用 clean-contract 角色措辞。若 G6 失败，说明 profile 空间本身无价值，停止训练 selector。若 G7 失败，theorem-to-implementation mapping 无效；若 G8 tight audit 失败，safe anchor 仍可独立保留，但 tight 只能作为有明确 violation/inconclusive 标签的经验控制器。每次失败和修复都写入 `gate_log.md`。

### 11.1 训练与评估运行矩阵

1. **cache extraction：** 只运行一次冻结 PointPillars，生成 canonical feature cache、manifest 和每 split 最多 500-frame 的等价性报告；G0 未通过不得训练 downstream。
2. **接口打通：** seed 1103，B5/B6/B8 各 500 个 downstream train steps，仅检查 loss、shape、cache key、bit packing 和可视化，不产出正式指标。
3. **超参数冻结：** B5/B6/B8 使用 seeds 1103、2207 运行第 7.2 节六个配置，以两个 seed 的 `VAL_SELECT` 平均风险效用选择；同一方法选定一个配置后冻结。B4 直接复用 B5 schedule，不额外搜索。
4. **core 正式训练：** B5/B6/B8 使用全部 5 seeds；B4 使用 3 seeds。B0 训练共同 task head 的 5 seeds；B3 复用 B8 checkpoint，只改变 overload 路径，不另训。B1/B2/B7 与公开强基线不在 core matrix。
5. **E1 干预：** 只对完成的 B5/B6/B8 5-seed checkpoints 运行，不重新训练；先缓存 clean outputs，再复用相同 receiver inputs。B4 secondary 只跑精简诊断。
6. **E1-R 压力复审：** 复用 E1 checkpoints、对齐和三个 accepted operators，在固定精简压力网格只做 cached forward 与干预，不再训练。
7. **E2-T safe-first：** 对每个 `(a,γ,m)` 生成 error-free、singleton、全部 pair、预注册 multi-role 和 bit-accurate channel 记录；先独立完成 safe `β=1` UCB、profile table、cross-fit diagnostics 与 freeze manifest。
8. **E2-T tight：** 预构建 LP matrix，先跑固定 100-replicate benchmark，再执行完整 10,000 bootstrap、5-fold cross-fit 和全 `VAL_BOUND` freeze。tight 的 manifest 必须写 `compute_status` 与 `envelope_scope=audited-support-only`。
9. **E2 selector：** 角色 encoder 冻结；P3/P4/P5 各 5 seeds。P0/P1b/P5b/P6 不额外训练；P1a/P2/P5g/P5c 只在有余量时运行，状态写入 manifest。
10. **信道与 TEST：** safe freeze 完成后即可统一读取 cache 运行五点 AWGN + 两个 stress 条件；tight 只有完整 10,000-replicate freeze 完成才作为 calibrated empirical controller 入表。不得边看结果边增加 SNR 或 seeds。
11. **结构/前沿：** 完成三个 `N0` contract、graph/operator/FLOPs/memory audit 和 overload frontier。服务器 reference latency、V2V4Real zero-shot 与单 seed end-to-end sensitivity 只有在 core 完成且资源有余量时才运行，默认跳过。

### 11.2 资源下限与降载规则

- 建议最低资源：1 张 24GB 以上 GPU（2 张可并行）、32 CPU cores、128GB RAM、至少 1.5TB 可用存储；先用 100 序列 pilot 测量 cache bytes/frame，再确认总空间。无需边缘平台或无线硬件。
- 每个训练 run 保存 best、last 和固定 epochs `{10,20,40,60}`，其余 checkpoint 可在指标与恢复测试完成后归档。
- feature cache 使用 chunked compression；predictions 使用 parquet/NPZ；点云原始数据不复制进交付目录。
- 算力不足时依次删除 V2V4Real zero-shot、单 seed end-to-end sensitivity、P1a/P2/P5g/P5c、B4 的额外 interventions 和服务器 latency；不得减少 B5/B6/B8 的 5 个核心 seeds、三种 accepted interventions、Hungarian、P0/P1b/P3/P4/P5/P5b-safe/P6、G2 topology audit 或 E2-T 的 singleton/pair/真实失败集合审计。tight 若未完成 2,000-frame subset audit 或 10,000 LP bootstrap，只能标记 `compute-pending/incomplete`，不得用缩水结果；safe `β=1` pipeline 仍必须完整交付。
- 如资源仍不足，停止并提交缺口清单，不得用单 seed 或单数据结果替代完整证据。

### 11.3 软件环境冻结

- 使用容器或 Conda lockfile 固定 Python、PyTorch、CUDA、cuDNN、OpenCOOD、Zarr/NumPy 与可选 ONNX runtime；不得只交 `pip freeze` 而缺少 CUDA/driver 信息。
- 设置 Python/NumPy/PyTorch/CUDA seeds，记录 deterministic flags 和不可确定算子列表。若某算子无法确定化，必须用重复运行量化其方差。
- 所有配置经 schema 校验后生成 `config_hash`；checkpoint、prediction 和 metric 文件均携带该 hash。
- `run_all` 首先执行 preflight：数据哈希、GPU 型号、磁盘空间、profile 合法性、split 交集和依赖版本任一失败即退出非零状态。

## 12. 交付目录与文件格式

执行团队必须按以下结构交付；路径名固定，避免后续找不到证据：

```text
experiment_delivery/
  00_manifests/
    splits_manifest.yaml
    frozen_operating_point.yaml
    profiles.yaml
    risk_strata.yaml
    bound_freeze_manifest.yaml
    lp_solver.yaml
    feature_cache_manifest.yaml
    optional_runs.yaml
    optional_domain_manifest.yaml  # only if zero-shot is executed
    environment.yaml
    git_commit.txt
    change_log.md
  01_unit_tests/
    bit_roundtrip.json
    frame_length_audit.csv
    permutation_equivalence.csv
    topology_hashes.csv
    cache_equivalence.csv
    bound_formula_unit_tests.json
  02_checkpoints/{dataset}/{method}/{seed}/
  03_predictions/{dataset}/{method}/{seed}/{channel}/
  04_metrics/
    sequence_level_metrics.parquet
    role_signatures.parquet
    role_stress_signatures.parquet
    alignment_bootstrap.parquet
    bound_terms.parquet
    bound_interactions.parquet
    bound_counterfactual_stress.parquet
    tight_lp_solver_log.parquet
    latency_samples.parquet  # optional
    run_cost.csv
  05_tables/
    main_utility.csv
    role_certificate.csv
    role_stress_certificate.csv
    profile_ablation.csv
    bound_ucb_table.csv
    bound_profile_table_safe.csv
    bound_profile_table_tight.csv
    bound_coverage.csv
    bound_controller.csv
    tight_lp_benchmark.csv
    overload_fallback.csv
  06_figures/
    role_signature_heatmap.pdf
    role_certificate_forest.pdf
    role_stress_retention.pdf
    bound_actual_vs_ucb.pdf
    bound_slack_decomposition.pdf
    bound_residual_stress.pdf
    bound_controller_regret.pdf
    utility_calibration_vs_snr.pdf
    utility_predictability_frontier.pdf
    overload_selective_risk.pdf
  07_logs/
    gate_log.md
    failed_runs.csv
    server_reference_snapshot/  # optional
  08_reproduction/
    run_all.ps1
    run_all.sh
    README_EXECUTION.md
```

### 12.1 `sequence_level_metrics.parquet` 必需列

`dataset, split, cache_id, sequence_id, frame_id, method, controller_type, train_seed, channel_model, channel_seed, esn0_db, gamma_true, gamma_observed, risk_stratum_a, M, K, N0, profile_id, overload, n_raw_tokens, critical_gt, critical_miss, bounded_loss_ell, brier_sum, detection_count, boundary_latency_ms, full_latency_ms, feedback_bits, feedback_latency_ms, crc_reject, role_erasure_count, undetected_error, run_id`

`boundary_latency_ms/full_latency_ms` 在未运行可选 reference timing 时允许为 NA；不能用 0 填充。`run_cost.csv` 至少包含 `stage, method, train_seed, gpu_hours, cpu_hours, bytes_read, bytes_written, backbone_executed, cache_id`，并据此核验非 backbone GPU-hour 占比。

AP 不得拆成虚构的逐帧贡献。`03_predictions` 必须保存每个检测的 score、类别、box、GT 匹配信息；bootstrap 每次按序列重采样后从原始 predictions 重新计算 AP。可另外保存 `sequence_ap50/sequence_ap70` 作为诊断，但不替代重采样后的全局 AP。

`bound_terms.parquet` 每行固定为一个 `(train_seed,a,γ,m,k或pair,sequence_id)` 聚合，至少包含：`r0,w_k,pe_k,pe_ij,eta_k,beta_ij,beta_mode,envelope_scope,pu,term_ucb,cell_support,bootstrap_fold,config_hash`。`bound_coverage.csv` 至少包含：`beta_mode,envelope_scope,R_hat,R_lcb95,R_ucb95,B,slack_abs,slack_rel,coverage_state,n_sequences,n_frames,n_critical_gt`。safe 的 `envelope_scope=structural-bounded-loss`；tight 固定为 `audited-support-only`。任何脚本若发现 `split=TEST_AUDIT` 的记录参与 `bound_ucb_table.csv` 生成，必须退出非零状态。

`tight_lp_solver_log.parquet` 至少包含 `train_seed,cell_id,fold,bootstrap_id,n_rows,n_active_rows,n_constraint_rounds,solve_ms,scan_ms,peak_ram_mb,primal_status,dual_status,max_full_matrix_violation,solver_version`。必须恰好覆盖正式预注册 bootstrap IDs；`tight_lp_benchmark.csv` 保存 dry-run IDs `bench-000..bench-099` 的实测与全量外推，不进入统计估计。`bound_freeze_manifest.yaml` 必须分别记录 `safe_anchor_status`、`tight_compute_status`、`tight_crossfit_status`、`tight_test_status`、`tight_envelope_scope=audited-support-only` 和实际 bootstrap count。

`bound_controller.csv` 必须包含 `controller_id,beta_mode,envelope_scope,qualification_status,a,gamma,selected_profile,tie_break,actual_risk,oracle_regret`。P5b-safe 的 qualification 只能描述 formal interaction anchor 与 empirical bound audit；P5b-tight 只能使用 `empirically-supported/violation/inconclusive/compute-pending`，禁止出现 `formal/certified/almost-sure`。

### 12.2 每个图表的最小复现要求

- 每张图由一个独立脚本从 parquet/csv 生成，不读取手工编辑数字。
- 图脚本输出对应的 `data_used.csv` 和命令行参数。
- 表格必须保留所有 seed，主表同时给 mean、95% CI 和序列数。
- 所有 tight 表题、图题和 legend 必须出现 `Empirical envelope — audited support only`；safe 与 tight 不得只靠颜色区分。
- 所有原始结果只追加不覆盖；修复后用新的 `run_id`。

## 13. 实验人员逐步清单

### 开始前

- [ ] 确认 OPV2V 下载完整并生成哈希；V2V4Real 仅在选择执行 optional zero-shot 时准备。
- [ ] 冻结五类 split，运行无泄漏测试。
- [ ] 用 50 帧可视化坐标变换、协作者到 ego 对齐、token admission 和 GT。
- [ ] 确认所有方法读取同一缓存 token 和 channel realization。
- [ ] feature cache manifest、shard hashes 与每 split 最多 500-frame 的 online-vs-cache 等价性全部通过。
- [ ] cache 中没有 admitted roles、grouped/query outputs、profile logits 或 decoded tensors；role emergence 仍由每个正式 seed 独立训练。

### 实现完成后

- [ ] profile 表全部通过秩、长度、round-trip、CRC 测试。
- [ ] synchronized permutation 输出误差 `<1e-5`。
- [ ] 所有 profile 的 graph hash、操作序列和 decoder 迭代相同。
- [ ] B5/B6/B8 参数、task head 与 cache interface 一致；若运行 B4，其与 B5 的参数/FLOPs 匹配在容差内。
- [ ] E2-T.0 五类公式单元测试全部通过，erasure/fallback 只有一个共享实现。

### 训练后

- [ ] 五个固定 seed 均完成，失败 run 有记录。
- [ ] 主运行点只由 `VAL_SELECT` 决定并已哈希冻结。
- [ ] Hungarian 只使用 `VAL_ALIGN`，没有接触测试结果。
- [ ] calibration/fallback 阈值只来自 `VAL_SELECT`。
- [ ] 风险层、上界项、simultaneous UCB 与 P5b-safe/P5b-tight 表只来自 `VAL_BOUND`，并已生成 freeze manifest。
- [ ] P5 不含 CSI；P5b/P5c 的 3-bit γ、估计误差和反馈时延已单独记录。

### 统计前

- [ ] 统计单位为序列，bootstrap 为配对且使用共同随机数。
- [ ] 四个推断性主要终点和 Holm 校正已在脚本中锁定。
- [ ] E1 每个 certified role 对三个 accepted operators 均逐项通过，没有“三取二”。
- [ ] E1-R 未重新训练、重新对齐或在压力格重新归一化。
- [ ] `w_k^(m)` 来自 post-quantization intervention，未复用 clean-link `Nec`。
- [ ] `η_k` 由同帧 `(E_k,d_k)` 协方差计算；`p_e,ij` 来自联合事件而非边缘乘积。
- [ ] safe 上界使用 `β=1`，tight β-UCB 与 safe 表分开；零错误概率仍有非零 UCB。
- [ ] safe structural anchor 已逐样本验证；tight 的所有表/图均标记 `audited-support-only`，没有 almost-sure、deterministic 或 formal guarantee 措辞。
- [ ] tight LP 的 100-replicate benchmark、full-matrix constraint scan、solver tolerance 和 10,000 个预注册 bootstrap IDs 均完整；若未完整则状态为 compute-pending/incomplete。
- [ ] TEST 数据未参与 UCB、查找表、风险层、门槛或 controller 选择。
- [ ] 困难箱边界来自 TRAIN，不按结果重新切箱。
- [ ] 所有负结果、失败 seed 和非显著结果均保留。

### 最终交付前

- [ ] E1/E1-R/E2/E3 均有明确 pass/fail；E2-T 分开记录 `safe formal-anchor implementation`、`tight empirical-envelope`、`tight compute status` 与 `controller utility`，允许 insufficient precision/compute-pending。
- [ ] V2V4Real、end-to-end sensitivity 和 reference latency 均按 optional 状态明确标记，不因未执行而伪造空结果。
- [ ] 若运行 reference timing，cached boundary 与 optional full pipeline 严格分开，且未出现 deadline/WCET 表述。
- [ ] 图表可由交付脚本一键重建。
- [ ] 交付目录无原始敏感数据副本、公开链接或访问令牌。

## 14. 结果解释规则

1. **E1 通过、E1-R 失败：** 角色结论仅限 clean contract；不得写成信道/overload 下仍稳定。
2. **E1 通过、E2 失败：** 只能说明角色具有操作稳定性；删除学习联合自适应价值结论。
3. **E1 失败、E2 通过：** 只能说明固定资源配置器有效；不得归因于稳定角色。
4. **safe formal-anchor implementation 失败、P5/P5b-tight 表现好：** 只能支持经验收益；theorem-to-implementation mapping 无效。必须先定位 bounded loss、`Δ/d`、协方差、undetected event 或数据泄漏。
5. **safe 通过、tight empirical-envelope 失败：** 保留 safe structural anchor；tight 只报告 audited-support violation 和描述性 controller performance，不得称 formal、certified、deterministic 或 almost-sure bound。
6. **safe/tight 各自审计通过、controller utility 失败：** safe 仍是保守审计锚点，tight 仍只是 audited-support envelope；coverage 不能代替 profile controller 效用。
7. **若执行 P5c，且 P5b-tight 优于 P5、P5c 也同幅提升：** 主要收益来自 γ 信息，不能归因于解析规则；应比较 P5b-tight 与 P5c 的同信息条件差异。
8. **P5b-tight 大量 cell 截断到 1 或退化为单 profile：** empirical envelope/controller 不够 sharp；只能报告保守性，不得称为精细风险分配。
9. **tight 未完成 10,000 bootstrap：** 状态只能是 compute-pending/incomplete；不得用 pilot 或缩水 bootstrap 进入主表、gate 或 simultaneous-UCB 结论。
10. **E1/E2 与 E2-T safe mapping 通过、G2 失败：** 有效性可能存在，但固定拓扑结论无效，必须修复实现后重测。
11. **平均 AP 提升、风险/困难箱失败：** 视为普通平均性能增益，不足以支持高价值主张。
12. **可选 V2V4Real zero-shot 反向：** 明确记录 domain shift，但不改变 OPV2V core gate；不得隐藏已运行的负结果。未运行时不作任何跨域结论。
13. **oracle gap 小：** 自适应问题设置无实质空间；重建 profile 表而非继续优化 selector。
14. **overload 时表现差但 flag/fallback 有效：** 可描述为可检测、可管理的退化；不得描述为容量问题已解决。
15. **所有 core gate 通过：** 才能同时支持角色功能、压力稳定性、固定资源自适应、safe risk-bound anchor、empirical tight controller、结构可审计和安全退化；仍不得升级 tight 为 population-wide almost-sure guarantee，也不得升级系统为硬件 WCET、deadline 或真实 RF 保证。
16. **cache 等价性或 cache_id 配对失败：** 受影响的全部 downstream 结果作废；不得用“只是缓存误差”保留结果。
17. **non-backbone GPU-hours 未达到 80%：** 只说明成本目标未实现，必须报告原因；若 G0 等价性与三块 core 证据通过，不据此否定机制结果。
