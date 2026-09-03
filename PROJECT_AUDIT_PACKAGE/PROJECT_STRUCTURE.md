# 项目结构与核心文件说明

| 文件/目录 | 功能 | 主要输入 | 主要输出 | 是否核心 |
|---|---|---|---|---|
| `source/modules/input_mask.py` | 从配置读取 `M=128`、`d_x=64`，生成随机输入和源掩码 | 配置、batch size、profile | `[batch,128,64]` 张量、掩码 | 是 |
| `source/modules/group_query.py` | 线性投影角色特征，并按连续 16 个位置分组执行均值查询 | 输入张量、可选掩码 | `[batch,128,128]` 或每组 query | 是 |
| `source/modules/quantizer.py` | 将特征投影到 128 维，再压缩到 32 维 codec | 分组特征 | `[batch,128,32]` codec | 是 |
| `source/modules/profile_generator.py` | 确定性生成 13 个角色资源 profile、128 位掩码和 hash | 配置 | profile 字典列表 | 是 |
| `source/modules/mother_code.py` | profile 路径按角色生成信息位和固定稀疏线性校验位，并加 128 位头部 | 源比特、`q/n` 列表 | `[batch,8192]` 比特 | 是 |
| `source/modules/software_channel.py` | QPSK 调制、AWGN 加噪和硬判决解调 | 比特、SNR、seed | 接收硬比特和可选软 I/Q | 是 |
| `source/modules/erasure.py` | 根据含噪软信息阈值生成擦除位置掩码 | 接收比特、软信息 | 布尔擦除掩码 | 是 |
| `source/modules/decoder.py` | 对 32 位块执行固定 10 次 Min-Sum；长码流保留兼容路径 | 接收比特、擦除掩码 | 解码比特 | 是 |
| `source/modules/receiver.py` | 编排解码、失败检测和统一擦除标记 | 接收比特、软信息 | 解码结果和统计 | 是 |
| `source/modules/math_functions.py` | 概率、效率、协方差及小规模非负 LP 函数 | NumPy 数组和约束 | 数值结果或 LP 解 | 是 |
| `configs/toy_config.yaml` | 统一保存 toy 参数、信道参数、擦除阈值 | YAML | 运行配置 | 是 |
| `scripts/run_demo.py` | 随机张量最小完整链路演示 | 配置 | 控制台关键形状和状态 | 是 |
| `tests/` | 单元、结构、信道、母码、擦除和端到端测试 | 测试夹具 | pytest 结果 | 是 |
| `fixed_resource_comm/` | 早期目录结构和旧测试/演示代码 | 旧接口 | 遗留运行入口 | 否，遗留 |

调用主链为：`InputMask`/随机输入 → `GroupQuery` → `Quantizer` → `ProfileGenerator`
→ `MotherCode` → `SoftwareChannel` → `Receiver`（内部使用 `LDPCDecoder` 和 `Erasure`）。
