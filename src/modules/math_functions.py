"""
数学函数：实现文档1要求的R0、d_k、w_k、p_e,k、p_e,ij、eta_k、p_u
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union

NumberOrArray = Union[np.ndarray, float, int]


def _array(value: NumberOrArray) -> np.ndarray:
    """将兼容 NumPy 的输入转换为 ndarray，且不改变数值。"""
    return np.asarray(value, dtype=float)


def _positive_denominator(value: NumberOrArray, name: str) -> np.ndarray:
    denominator = _array(value)
    if np.any(denominator <= 0):
        raise ValueError(f"{name} must be positive")
    return denominator


def compute_R0(success_rates: NumberOrArray) -> np.floating:
    """计算所有角色彼此独立且均成功的概率。

    参数：
        success_rates：每个角色成功概率组成的 NumPy 数组。
    返回：
        NumPy 标量 ``prod_k(success_rates[k])``。
    """
    return np.prod(_array(success_rates))

def compute_d_k(success_rate_k: NumberOrArray,
                failure_rate_k: NumberOrArray) -> np.ndarray:
    """计算每个角色的衰减因子 ``d_k = s_k - p_e,k``。"""
    return _array(success_rate_k) - _array(failure_rate_k)

def compute_w_k(importance_k: NumberOrArray,
                d_k: NumberOrArray) -> np.ndarray:
    """计算角色权重 ``w_k = importance_k * d_k``。"""
    return _array(importance_k) * _array(d_k)

def compute_p_e_k(failure_count_k: NumberOrArray,
                  total_trials: NumberOrArray) -> np.ndarray:
    """计算边缘错误概率 ``p_e,k = failures_k / trials``。"""
    return _array(failure_count_k) / _positive_denominator(total_trials, "total_trials")

def compute_p_e_ij(failure_counts_i: NumberOrArray,
                   failure_counts_j: NumberOrArray,
                   simultaneous_failures: NumberOrArray,
                   total_trials: NumberOrArray) -> np.ndarray:
    """计算联合错误概率 ``p_e,ij = failures_ij / trials``。

    为匹配文档接口，函数接收边缘失败次数；联合概率本身由同时失败次数决定。
    """
    _ = (_array(failure_counts_i), _array(failure_counts_j))
    return _array(simultaneous_failures) / _positive_denominator(total_trials, "total_trials")

def compute_eta_k(success_rate_k: NumberOrArray,
                  failure_rate_k: NumberOrArray,
                  total_roles: int) -> np.ndarray:
    """计算每个角色的归一化二元不确定性 ``eta_k``。"""
    if total_roles <= 1:
        raise ValueError("total_roles must be greater than 1")
    eps = np.finfo(float).eps
    success = np.clip(_array(success_rate_k), eps, 1.0)
    failure = np.clip(_array(failure_rate_k), eps, 1.0)
    entropy = -(success * np.log2(success) + failure * np.log2(failure))
    return entropy / np.log2(total_roles)

def compute_p_u(incorrect_bits_not_erased: NumberOrArray,
                total_bits: NumberOrArray) -> np.ndarray:
    """计算未被擦除但错误的比特概率。"""
    return _array(incorrect_bits_not_erased) / _positive_denominator(total_bits, "total_bits")

def compute_covariance(p_e_i: float, p_e_j: float, p_e_ij: float) -> float:
    """
    计算协方差项：p_e,ij - p_e,i * p_e,j
    当独立失败时，协方差项应为0
    参数：
        p_e_i: 角色i的失败概率
        p_e_j: 角色j的失败概率
        p_e_ij: 同时失败概率
    返回：
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
    """使用 ``scipy.optimize.linprog`` 求解小规模非负线性规划。

    支持的键包括 ``c``、可选的 ``A_ub/b_ub``、可选的 ``A_eq/b_eq`` 和
    可选的 ``bounds``。旧接口中的 ``A/b`` 可作为 ``A_ub/b_ub`` 的别名。
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
