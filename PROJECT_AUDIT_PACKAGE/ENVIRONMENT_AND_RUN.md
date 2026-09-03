# 环境、配置、seed 与运行命令

## 已确认环境

- 操作系统：Windows（当前审计环境）。
- Python：3.8.10。
- PyTorch：2.4.1+cpu。
- NumPy：1.24.4。
- SciPy：1.10.1。
- PyYAML：6.0.3。
- pytest：8.3.5。
- CUDA：未使用，当前为 CPU 版本 PyTorch。
- 依赖清单：`requirements.txt`。

## 配置

主配置为 `configs/toy_config.yaml`，其中 `toy.d_x=64`、`toy.M=128`、`toy.K=8`、
`profile.payload_bits=8064`、信道为 `QPSK_AWGN`、`snr_db=10`、seed 为 42，擦除阈值为 0.5。
各模块通过 YAML 读取配置，但存在 `toy`、`model`、`input` 三处重复维度字段，审计标记为维护风险。

## 入口

```bash
python scripts/run_demo.py
python -m pytest tests/
python reproduce/audit_all.py
```

Windows 可运行 `run_demo.bat`。审计包中的复现入口是 `python reproduce/audit_all.py`。

## seed 与可复现性

`SoftwareChannel.__init__` 从配置读取 `channel.random_seed` 并调用 `torch.manual_seed`；
demo 另外调用 `torch.manual_seed(123)` 生成输入。当前没有统一设置 Python `random` 或 NumPy
全局 seed，且信道构造函数会重置 PyTorch 全局 RNG，属于确定性设计但可能影响调用方 RNG 状态。

## 风险检查

- 母码配置中的 `k=16,n=32` 与 profile 最大 `q_k=256,n_k=1264` 不一致；profile 路径使用独立的 `P_k`，旧标量路径使用 32 位 toy 码。
- 默认构造函数含有相对路径回退逻辑，已在模块中回退到项目 `configs/`；不依赖 `.venv` 才能导入，但环境仍需安装依赖。
- 未发现正式数据路径或外部数据集。
- 正式方案输入：`experiment_spec/ISRC实验方案(1).md`；截图输入：`experiment_spec/缺口审计报告.png`。
- 方案要求的 OPV2V、OpenCOOD/PointPillars、feature cache 和 lockfile 当前均未在仓库中提供。
