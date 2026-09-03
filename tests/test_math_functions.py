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
    """验证 R0 是无错误帧损失 ``ell_t`` 的均值。"""
    error_free_losses = np.array([0.0, 0.5, 1.0])
    expected = (0.0 + 0.5 + 1.0) / 3.0
    np.testing.assert_almost_equal(compute_R0(error_free_losses), expected)


def test_compute_d_k_and_w_k_hand_calculation():
    """验证 d_k 截断负增量，w_k 为其逐帧均值。"""
    erasure_losses = np.array([0.2, 0.4, 0.8])
    error_free_losses = np.array([0.1, 0.5, 0.4])
    expected_d = np.array([0.1, 0.0, 0.4])
    np.testing.assert_almost_equal(
        compute_d_k(erasure_losses, error_free_losses), expected_d)
    np.testing.assert_almost_equal(compute_w_k(expected_d), 0.5 / 3.0)


def test_compute_p_e_k_counts_are_marginal_probabilities():
    """验证角色事件概率是同帧事件指示量的均值。"""
    events = np.array([1, 0, 1, 0, 0])
    np.testing.assert_almost_equal(compute_p_e_k(events), 2.0 / 5.0)


def test_independent_failures_have_zero_covariance():
    """验证手工独立事件的协方差项为零。"""
    events_i = np.array([0, 0, 1, 1])
    events_j = np.array([0, 1, 0, 1])
    p_i = compute_p_e_k(events_i)
    p_j = compute_p_e_k(events_j)
    p_ij = compute_p_e_ij(events_i, events_j)
    # p_i=p_j=1/2，p_ij=1/4，因此协方差为0。
    np.testing.assert_almost_equal(compute_covariance(p_i, p_j, p_ij), 0.0)


def test_joint_probability_changes_when_simultaneous_count_changes():
    """验证边缘概率不变而同帧重叠改变时联合概率改变。"""
    events_i = np.array([1, 1, 0, 0])
    events_j_a = np.array([1, 0, 1, 0])
    events_j_b = np.array([1, 1, 0, 0])
    p_i_a = compute_p_e_k(events_i)
    p_i_b = compute_p_e_k(events_i)
    p_j_a = compute_p_e_k(events_j_a)
    p_j_b = compute_p_e_k(events_j_b)
    p_ij_a = compute_p_e_ij(events_i, events_j_a)
    p_ij_b = compute_p_e_ij(events_i, events_j_b)

    np.testing.assert_almost_equal(p_i_a, p_i_b)
    np.testing.assert_almost_equal(p_j_a, p_j_b)
    np.testing.assert_almost_equal(p_ij_a, 0.25)
    np.testing.assert_almost_equal(p_ij_b, 0.50)
    assert p_ij_a != p_ij_b


def test_compute_eta_k_hand_calculation():
    """验证 eta_k 是同帧事件与风险增量协方差绝对值。"""
    events = np.array([0, 0, 1, 1])
    d_k = np.array([0.0, 0.0, 1.0, 1.0])
    # 两个中心化向量相同，协方差为1/4。
    np.testing.assert_almost_equal(compute_eta_k(events, d_k), 0.25)


def test_compute_p_u_counts_errors_not_erased():
    """验证 U_t 事件中两位错误且未擦除时 p_u=2/5。"""
    incorrect_not_erased = np.array([0, 1, 0, 1, 0])
    np.testing.assert_almost_equal(compute_p_u(incorrect_not_erased), 2.0 / 5.0)
