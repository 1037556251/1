"""
数学函数：实现文档1要求的R0、d_k、w_k、p_e,k、p_e,ij、eta_k、p_u
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union

NumberOrArray = Union[np.ndarray, float, int]


def _array(value: NumberOrArray) -> np.ndarray:
    """Convert NumPy-compatible input to an ndarray without changing values."""
    return np.asarray(value, dtype=float)


def _positive_denominator(value: NumberOrArray, name: str) -> np.ndarray:
    denominator = _array(value)
    if np.any(denominator <= 0):
        raise ValueError(f"{name} must be positive")
    return denominator


def compute_R0(success_rates: NumberOrArray) -> np.floating:
    """Compute the probability that all roles succeed independently.

    Args:
        success_rates: NumPy array of per-role success probabilities.
    Returns:
        NumPy scalar ``prod_k(success_rates[k])``.
    """
    return np.prod(_array(success_rates))

def compute_d_k(success_rate_k: NumberOrArray,
                failure_rate_k: NumberOrArray) -> np.ndarray:
    """Compute each role's attenuation factor ``d_k = s_k - p_e,k``."""
    return _array(success_rate_k) - _array(failure_rate_k)

def compute_w_k(importance_k: NumberOrArray,
                d_k: NumberOrArray) -> np.ndarray:
    """Compute the role weight ``w_k = importance_k * d_k``."""
    return _array(importance_k) * _array(d_k)

def compute_p_e_k(failure_count_k: NumberOrArray,
                  total_trials: NumberOrArray) -> np.ndarray:
    """Compute marginal error probability ``p_e,k = failures_k / trials``."""
    return _array(failure_count_k) / _positive_denominator(total_trials, "total_trials")

def compute_p_e_ij(failure_counts_i: NumberOrArray,
                   failure_counts_j: NumberOrArray,
                   simultaneous_failures: NumberOrArray,
                   total_trials: NumberOrArray) -> np.ndarray:
    """Compute joint error probability ``p_e,ij = failures_ij / trials``.

    The marginal counts are accepted to match the documented interface; the
    joint probability itself is determined by the simultaneous count.
    """
    _ = (_array(failure_counts_i), _array(failure_counts_j))
    return _array(simultaneous_failures) / _positive_denominator(total_trials, "total_trials")

def compute_eta_k(success_rate_k: NumberOrArray,
                  failure_rate_k: NumberOrArray,
                  total_roles: int) -> np.ndarray:
    """Compute normalized binary uncertainty ``eta_k`` for each role."""
    if total_roles <= 1:
        raise ValueError("total_roles must be greater than 1")
    eps = np.finfo(float).eps
    success = np.clip(_array(success_rate_k), eps, 1.0)
    failure = np.clip(_array(failure_rate_k), eps, 1.0)
    entropy = -(success * np.log2(success) + failure * np.log2(failure))
    return entropy / np.log2(total_roles)

def compute_p_u(incorrect_bits_not_erased: NumberOrArray,
                total_bits: NumberOrArray) -> np.ndarray:
    """Compute probability of an incorrect bit that was not erased."""
    return _array(incorrect_bits_not_erased) / _positive_denominator(total_bits, "total_bits")

def compute_covariance(p_e_i: float, p_e_j: float, p_e_ij: float) -> float:
    """
    计算协方差项：p_e,ij - p_e,i * p_e,j
    当独立失败时，协方差项应为0
    Args:
        p_e_i: 角色i的失败概率
        p_e_j: 角色j的失败概率
        p_e_ij: 同时失败概率
    Returns:
        covariance: 协方差值
    """
    return p_e_ij - p_e_i * p_e_j

def create_test_fixture() -> dict:
    """
    创建手算测试fixture
    使用人工构造的小数组验证数学函数
    """
    # 构造4个角色的小规模测试数据
    K = 4
    total_trials = 1000

    # 角色1: 成功950次，失败50次
    # 角色2: 成功900次，失败100次
    # 角色3: 成功800次，失败200次
    # 角色4: 成功700次，失败300次

    failure_counts = torch.tensor([50, 100, 200, 300])
    success_counts = torch.tensor([950, 900, 800, 700])
    total_trials = 1000

    # 同时失败次数（构造不同的场景）
    # 场景1: 独立失败（协方差应为0）
    simultaneous_failures_independent = torch.tensor([
        [0, 5, 10, 15],    # 角色1
        [5, 0, 20, 30],    # 角色2
        [10, 20, 0, 60],   # 角色3
        [15, 30, 60, 0],   # 角色4
    ])

    # 场景2: 相关失败（协方差不为0）
    simultaneous_failures_dependent = torch.tensor([
        [0, 8, 15, 20],    # 角色1
        [8, 0, 25, 35],    # 角色2
        [15, 25, 0, 70],   # 角色3
        [20, 35, 70, 0],   # 角色4
    ])

    return {
        'K': K,
        'total_trials': total_trials,
        'failure_counts': failure_counts,
        'success_counts': success_counts,
        'simultaneous_failures_independent': simultaneous_failures_independent,
        'simultaneous_failures_dependent': simultaneous_failures_dependent
    }

def solve_lp_small_scale(constraints: dict) -> Optional[torch.Tensor]:
    """Solve a small non-negative LP using ``scipy.optimize.linprog``.

    Supported keys are ``c``, optional ``A_ub/b_ub``, optional
    ``A_eq/b_eq`` and optional ``bounds``.  The legacy ``A/b`` keys are
    accepted as aliases for ``A_ub/b_ub``.
    """
    from scipy.optimize import linprog

    if "c" not in constraints:
        raise ValueError("constraints must include objective vector 'c'")

    def as_numpy(value):
        return None if value is None else np.asarray(value, dtype=float)

    c = as_numpy(constraints["c"]).reshape(-1)
    A_ub = as_numpy(constraints.get("A_ub", constraints.get("A")))
    b_ub = as_numpy(constraints.get("b_ub", constraints.get("b")))
    A_eq = as_numpy(constraints.get("A_eq"))
    b_eq = as_numpy(constraints.get("b_eq"))
    bounds = constraints.get("bounds", [(0.0, None)] * c.size)

    if A_ub is not None:
        A_ub = np.atleast_2d(A_ub)
        b_ub = np.asarray(b_ub, dtype=float).reshape(-1)
        if A_ub.shape != (b_ub.size, c.size):
            raise ValueError("A_ub and b_ub dimensions do not match c")
    if A_eq is not None:
        A_eq = np.atleast_2d(A_eq)
        b_eq = np.asarray(b_eq, dtype=float).reshape(-1)
        if A_eq.shape != (b_eq.size, c.size):
            raise ValueError("A_eq and b_eq dimensions do not match c")
    if len(bounds) != c.size:
        raise ValueError("bounds must contain one pair per variable")

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")
    return torch.tensor(result.x, dtype=torch.float32) if result.success else None
