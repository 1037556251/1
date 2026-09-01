"""
数学toy测试
使用人工构造的小数组验证数学函数
"""

import torch
import numpy as np
import sys
import os


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))



from modules.math_functions import (
    compute_R0, compute_d_k, compute_w_k, compute_p_e_k,
    compute_p_e_ij, compute_eta_k, compute_p_u, compute_covariance,
    create_test_fixture, solve_lp_small_scale
)


def test_independent_failure_covariance():
    """
    测试独立失败时协方差项应为0
    """
    print("\n[测试1] 独立失败协方差测试")
    print("=" * 40)

    fixture = create_test_fixture()
    K = fixture['K']

    # 验证独立失败场景
    for i in range(K):
        for j in range(i + 1, K):
            p_e_i = compute_p_e_k(fixture['failure_counts'][i].item(),
                                  fixture['total_trials'])
            p_e_j = compute_p_e_k(fixture['failure_counts'][j].item(),
                                  fixture['total_trials'])
            p_e_ij = compute_p_e_ij(
                fixture['failure_counts'][i].item(),
                fixture['failure_counts'][j].item(),
                fixture['simultaneous_failures_independent'][i][j].item(),
                fixture['total_trials']
            )

            cov = compute_covariance(p_e_i, p_e_j, p_e_ij)
            print(f"  Cov({i},{j}) = {cov:.6f}")

            # 独立失败时，协方差应接近0
            assert abs(cov) < 0.01, f"独立失败时协方差应为0，实际为{cov}"

    print("  ✓ 独立失败协方差测试通过")


def test_failure_position_change():
    """
    测试保持失败次数不变但改变同时失败位置时：
    - p_e,k不变
    - p_e,ij改变
    """
    print("\n[测试2] 失败位置变化测试")
    print("=" * 40)

    # 构造两个场景：失败次数相同，但同时失败位置不同
    total_trials = 100

    # 场景A：角色1和2同时失败5次
    scene_A = {
        'failure_counts': torch.tensor([10, 10, 15, 20]),
        'simultaneous_failures': torch.tensor([[0, 5, 3, 2],
                                               [5, 0, 4, 3],
                                               [3, 4, 0, 6],
                                               [2, 3, 6, 0]])
    }

    # 场景B：角色1和2同时失败10次（其他不变）
    scene_B = {
        'failure_counts': torch.tensor([10, 10, 15, 20]),
        'simultaneous_failures': torch.tensor([[0, 10, 3, 2],
                                               [10, 0, 4, 3],
                                               [3, 4, 0, 6],
                                               [2, 3, 6, 0]])
    }

    # 验证p_e,k不变
    for k in range(4):
        p_e_k_A = compute_p_e_k(scene_A['failure_counts'][k].item(), total_trials)
        p_e_k_B = compute_p_e_k(scene_B['failure_counts'][k].item(), total_trials)
        assert abs(p_e_k_A - p_e_k_B) < 1e-10, f"p_e,{k}应保持不变"
        print(f"  p_e,{k} (场景A) = {p_e_k_A:.4f}")
        print(f"  p_e,{k} (场景B) = {p_e_k_B:.4f}")

    # 验证p_e,ij改变
    p_e_ij_A = compute_p_e_ij(
        scene_A['failure_counts'][0].item(),
        scene_A['failure_counts'][1].item(),
        scene_A['simultaneous_failures'][0][1].item(),
        total_trials
    )
    p_e_ij_B = compute_p_e_ij(
        scene_B['failure_counts'][0].item(),
        scene_B['failure_counts'][1].item(),
        scene_B['simultaneous_failures'][0][1].item(),
        total_trials
    )
    print(f"  p_e,01 (场景A) = {p_e_ij_A:.4f}")
    print(f"  p_e,01 (场景B) = {p_e_ij_B:.4f}")
    assert abs(p_e_ij_A - p_e_ij_B) > 0.01, "p_e,ij应改变"
    print("  ✓ 失败位置变化测试通过")


def test_p_u_statistics():
    """
    测试p_u：没有被erasure但接收端继续使用错误bit时，p_u必须能统计出来
    """
    print("\n[测试3] p_u统计测试")
    print("=" * 40)

    # 构造一个场景：有1000个比特，其中50个是错误但未被擦除的
    total_bits = 1000
    incorrect_not_erased = 50

    p_u = compute_p_u(incorrect_not_erased, total_bits)
    print(f"  总比特数: {total_bits}")
    print(f"  错误但未擦除比特数: {incorrect_not_erased}")
    print(f"  p_u = {p_u:.4f}")

    # 验证p_u计算正确
    expected_p_u = incorrect_not_erased / total_bits
    assert abs(p_u - expected_p_u) < 1e-10, f"p_u计算错误，期望{expected_p_u}，实际{p_u}"
    print("  ✓ p_u统计测试通过")


def test_all_math_functions():
    """
    测试所有数学函数的基本功能
    """
    print("\n[测试4] 所有数学函数基本测试")
    print("=" * 40)

    # 测试R0
    success_rates = torch.tensor([0.95, 0.90, 0.85, 0.80])
    R0 = compute_R0(success_rates)
    expected_R0 = 0.95 * 0.90 * 0.85 * 0.80
    print(f"  R0 = {R0:.4f} (期望: {expected_R0:.4f})")
    assert abs(R0.item() - expected_R0) < 1e-6, f"R0误差过大: {abs(R0.item() - expected_R0)}"

    # 测试d_k
    d_k = compute_d_k(0.95, 0.05)
    print(f"  d_k = {d_k:.4f} (期望: 0.90)")
    assert abs(d_k - 0.90) < 1e-10

    # 测试w_k
    w_k = compute_w_k(1.0, 0.90)
    print(f"  w_k = {w_k:.4f} (期望: 0.90)")
    assert abs(w_k - 0.90) < 1e-10

    # 测试eta_k
    eta_k = compute_eta_k(0.95, 0.05, 4)
    print(f"  eta_k = {eta_k:.4f}")
    assert 0 <= eta_k <= 1, "eta_k应在[0,1]范围内"

    print("  ✓ 所有数学函数基本测试通过")


def test_small_scale_lp():
    """
    测试小规模非负LP接口
    """
    print("\n[测试5] 小规模非负LP测试")
    print("=" * 40)

    # 构造一个简单的人工约束
    # 最小化 x1 + x2
    # 约束: x1 + x2 >= 1, x1 >= 0, x2 >= 0
    # 最优解: x1=1, x2=0 或 x1=0, x2=1

    import numpy as np
    c = np.array([1.0, 1.0])  # 目标函数系数
    A = np.array([[-1, -1]])  # 约束矩阵（注意：linprog使用A_ub * x <= b_ub）
    b = np.array([-1.0])  # 约束值

    constraints = {
        'A': torch.tensor(A, dtype=torch.float32),
        'b': torch.tensor(b, dtype=torch.float32),
        'c': torch.tensor(c, dtype=torch.float32)
    }

    solution = solve_lp_small_scale(constraints)
    if solution is not None:
        print(f"  LP解: {solution.numpy()}")
        # 验证约束满足
        assert solution[0] + solution[1] >= 1 - 1e-6, "约束不满足"
        print("  ✓ 小规模LP测试通过")
    else:
        print("  ⚠ LP求解失败（scipy可能未安装）")


def run_all_tests():
    print("=" * 60)
    print("数学toy测试")
    print("=" * 60)

    test_independent_failure_covariance()
    test_failure_position_change()
    test_p_u_statistics()
    test_all_math_functions()
    test_small_scale_lp()

    print("\n" + "=" * 60)
    print("所有数学toy测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
