# 实验执行审计

正式实验方案已确认并保存在 `experiment_spec/ISRC实验方案(1).md`，截图缺口审计保存在
`experiment_spec/缺口审计报告.png`。以下按照方案中的 G0-G10、E1/E1-R/E2-T/E2/E3 要求，
区分实际证据与 toy 接口测试。

| ID | 实验/测试内容 | 是否已执行 | 结果位置 | 状态 | 说明 |
|---|---|---|---|---|---|
| E01 | 根目录 pytest 全套 28 项 | 是 | `logs/pytest_run_1.log` | PASS | 实际运行通过 |
| E02 | 第二次独立 pytest | 是 | `logs/pytest_run_2.log` | PASS | 实际运行通过 |
| E03 | 第三次独立 pytest | 是 | `logs/pytest_run_3.log` | PASS | 实际运行通过 |
| E04 | random tensor 最小完整链路 | 是 | `logs/demo_run.log`、`results_raw/` | PASS | 形状链路可运行，非正式性能实验 |
| E05 | 三次相同输入/config/seed 的关键输出确定性比较 | 是 | `manifests/`、`results_raw/` | PASS | 审计入口比较 hash |
| E06 | 13 profile 长度与约束 | 是 | pytest 日志、profile 代码 | PASS | `sum(n)=8064`，总长 8192 |
| E07 | 正式数据集实验 | 否 | 无 | NOT RUN | 未发现数据集或正式数据入口 |
| E08 | 论文级 baseline、规模和多 seed 实验 | 否 | 无 | NOT RUN | 当前项目仅 toy 阶段 |
| E09 | 正式图表/论文表格生成 | 否 | `figures/`、`tables/` 为空 | NOT RUN | 没有结果绘图脚本和正式 raw 数据 |
| E10 | 更新后审计包 pytest 全套 29 项 | 是 | `logs/audit_pytest.log` | PASS | 新增 profile 单错纠正测试通过 |
| G0 | OPV2V 五类 split、cache 等价性和无泄漏 | 否 | 无 | NOT RUN | 未提供 OPV2V、cache 或 split manifest |
| G1 | 三个 N0 的 bit/CRC/round-trip 合约 | 部分 | pytest 日志 | INCONCLUSIVE | toy 仅验证 8192 位长度和局部 round-trip，未覆盖 CRC 和三个 N0 |
| G2 | 统一静态图、operator/shape/iteration 审计 | 否 | 无 | NOT RUN | 没有 TorchScript/ONNX 图、hash 或 FLOPs/memory 审计 |
| G3-G6 | sanity、角色证书、压力复审、oracle gap | 否 | 无 | NOT RUN | 没有 B0/B5/B6/B8 训练和 E1/E1-R 数据 |
| G7 | safe `beta=1` 风险上界闭环 | 否 | 无 | NOT RUN | 现有数学函数不是方案定义的 post-quantization 风险闭环 |
| G8-G9 | tight empirical envelope 与联合自适应 | 否 | 无 | NOT RUN | 没有 10,000 bootstrap、cross-fit、selector 和 TEST 结果 |
| G10 | E3 三个 N0 与 overload/fallback 前沿 | 否 | 无 | NOT RUN | 当前代码没有三个 exact contract 和正式风险前沿 |

`PASS` 仅表示对应 toy 测试或最小运行满足当前代码断言；不等同于正式实验结论。
正式方案实验项按本表计：PASS 6、FAIL 0、INCONCLUSIVE 1、NOT RUN 8。
