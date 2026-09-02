"""独立 NumPy 数学函数的单元测试。"""

import numpy as np

from modules.math_functions import (
    compute_R0,
    compute_d_k,
    compute_w_k,
    compute_p_e_k,
    compute_p_e_ij,
    compute_eta_k,
    compute_p_u,
    compute_covariance,
)


def test_compute_R0_hand_calculation():
    """验证 R0 等于所有独立成功概率的乘积。"""
    success_rates = np.array([0.9, 0.8, 0.5])
    expected = 0.9 * 0.8 * 0.5
    np.testing.assert_almost_equal(compute_R0(success_rates), expected)


def test_compute_d_k_and_w_k_hand_calculation():
    """验证 d_k 为差值，w_k 为 d_k 与重要性权重的乘积。"""
    success = np.array([0.9, 0.75])
    failure = np.array([0.1, 0.25])
    importance = np.array([1.0, 2.0])
    expected_d = np.array([0.8, 0.5])
    expected_w = np.array([0.8, 1.0])
    np.testing.assert_almost_equal(compute_d_k(success, failure), expected_d)
    np.testing.assert_almost_equal(compute_w_k(importance, expected_d), expected_w)


def test_compute_p_e_k_counts_are_marginal_probabilities():
    """验证改变联合失败位置不会改变两个角色的边缘失败率。"""
    failure_counts_a = np.array([10, 20])
    failure_counts_b = np.array([10, 20])
    expected = np.array([0.1, 0.2])
    np.testing.assert_almost_equal(compute_p_e_k(failure_counts_a, 100), expected)
    np.testing.assert_almost_equal(compute_p_e_k(failure_counts_b, 100), expected)


def test_independent_failures_have_zero_covariance():
    """验证独立角色满足 p_e,ij = p_e,i * p_e,j。"""
    p_i = compute_p_e_k(np.array(10), 100)
    p_j = compute_p_e_k(np.array(20), 100)
    p_ij = compute_p_e_ij(np.array(10), np.array(20), np.array(2), 100)
    # 0.10 * 0.20 = 0.02 = 2 / 100。
    np.testing.assert_almost_equal(compute_covariance(p_i, p_j, p_ij), 0.0)


def test_joint_probability_changes_when_simultaneous_count_changes():
    """验证边缘失败次数不变而重叠次数改变时，p_e,ij 会改变。"""
    p_i_a = compute_p_e_k(np.array(10), 100)
    p_i_b = compute_p_e_k(np.array(10), 100)
    p_j_a = compute_p_e_k(np.array(20), 100)
    p_j_b = compute_p_e_k(np.array(20), 100)
    p_ij_a = compute_p_e_ij(np.array(10), np.array(20), np.array(2), 100)
    p_ij_b = compute_p_e_ij(np.array(10), np.array(20), np.array(8), 100)

    np.testing.assert_almost_equal(p_i_a, p_i_b)
    np.testing.assert_almost_equal(p_j_a, p_j_b)
    np.testing.assert_almost_equal(p_ij_a, 0.02)
    np.testing.assert_almost_equal(p_ij_b, 0.08)
    assert p_ij_a != p_ij_b


def test_compute_eta_k_hand_calculation():
    """验证 eta_k 使用归一化二元熵定义。"""
    success = np.array([0.5, 1.0])
    failure = np.array([0.5, 0.0])
    expected = np.array([1.0 / np.log2(4), 0.0])
    np.testing.assert_almost_equal(compute_eta_k(success, failure, 4), expected)


def test_compute_p_u_counts_errors_not_erased():
    """验证 100 个比特中 5 个未擦除错误比特得到 p_u=0.05。"""
    incorrect_not_erased = np.array([5, 10])
    expected = np.array([0.05, 0.10])
    np.testing.assert_almost_equal(compute_p_u(incorrect_not_erased, 100), expected)
