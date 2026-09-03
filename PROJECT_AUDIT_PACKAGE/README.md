# 固定资源通信项目审计包

本目录是当前 toy 阶段固定资源通信项目的独立审计包。建议按以下顺序阅读：

1. `AUDIT_REPORT.md`：最终审计结论。
2. `EXPERIMENT_AUDIT.md`：实验/测试执行状态。
3. `DEVIATIONS_AND_LIMITATIONS.md`：与当前可确认方案及实现边界的差异。
4. `ALGORITHM_MAPPING.md`：算法、公式与源代码映射。
5. `TEST_REPORT.md`：实际测试证据。

## 运行

在本目录中执行：

```bash
python reproduce/audit_all.py
```

该入口会检查环境、执行测试、执行三次确定性检查、运行最小完整链路，并重新生成
`results_raw/`、`results_processed/`、`manifests/`、`logs/` 和 `AUDIT_REPORT.md`。

单独运行测试：

```bash
python -m pytest tests/
```

当前包包含 `source/`、`configs/`、`tests/`、`scripts/`。`source/` 是当前有效源码快照；
`source_snapshot/` 保留项目根文件和旧目录快照，便于审计时区分当前入口与遗留实现。

## 审计输入与证据边界

正式方案和截图输入已保存在 `experiment_spec/`。方案明确要求 OPV2V、冻结
OpenCOOD/PointPillars cache、5 个训练 seed、E1/E2/E3、safe/tight 风险上界、
固定拓扑审计和正式结果链。当前仓库没有提供这些正式数据、cache、训练结果、表格或图片，
因此本包只对 toy 实现和实际运行过的测试作结论，不把 toy 结果表述为正式实验结论。
