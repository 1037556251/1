# 测试报告

## 总计

```text
Total: 29
PASS: 29
FAIL: 0
INCONCLUSIVE: 0
NOT RUN: 0
```

证据位于 `logs/pytest_run_1.log`、`logs/pytest_run_2.log`、`logs/pytest_run_3.log` 和
`logs/audit_pytest.log`；根项目三次运行显示 `28 passed`，更新后的审计包运行显示 `29 passed`。

## 覆盖分类

| 类别 | 状态 | 证据 |
|---|---|---|
| Unit Test | PASS | 数学函数、LP、profile、母码、QPSK、擦除局部测试 |
| Integration Test | PASS | `test_structure.py`、`test_end_to_end.py`、decoder/erasure 流程 |
| Failure / Edge Case | PASS | 输入维度、擦除掩码、全零/全一、边界比例、非法约束测试 |
| Determinism Test | PASS | `test_structure.py` 的固定 seed 测试；审计入口另执行 3 次独立比较 |

## 限定

测试通过只证明当前 toy 代码和测试断言通过，不证明正式通信性能、论文实验结论或大规模 LDPC 性能。
旧 `fixed_resource_comm/` 中的脚本没有作为根目录 pytest 套件的一部分执行。
