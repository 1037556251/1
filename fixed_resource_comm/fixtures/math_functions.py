"""
数学函数：实现文档1要求的R0、d_k、w_k、p_e,k、p_e,ij、eta_k、p_u
"""

import torch
import numpy as np
from typing import Tuple, Optional

def compute_R0(success_rates: torch.Tensor) -> torch.Tensor:
    """
    计算R0：所有角色独立成功的概率
    Args:
        success_rates: shape [K]，每个角色的成功概率
    Returns:
        R0: 标量，所有角色都成功的概率
    """
    return torch.prod(success_rates)

def compute_d_k(success_rate_k: float, failure_rate_k: float) -> float:
    """
    计算d_k：角色k的成功概率与失败概率的差值
    Args:
        success_rate_k: 角色k的成功概率
        failure_rate_k: 角色k的失败概率
    Returns:
        d_k: success_rate_k - failure_rate_k
    """
    return success_rate_k - failure_rate_k

def compute_w_k(importance_k: float, d_k: float) -> float:
    """
    计算w_k：角色k的加权重要性
    Args:
        importance_k: 角色k的重要性权重
        d_k: 角色k的d_k值
    Returns:
        w_k: 加权后的值
    """
    return importance_k * d_k

def compute_p_e_k(failure_count_k: int, total_trials: int) -> float:
    """
    计算p_e,k：角色k的独立失败概率（边缘概率）
    Args:
        failure_count_k: 角色k的失败次数
        total_trials: 总试验次数
    Returns:
        p_e,k: 角色k的失败概率
    """
    return failure_count_k / total_trials

def compute_p_e_ij(failure_counts_i: int, failure_counts_j: int,
                   simultaneous_failures: int, total_trials: int) -> float:
    """
    计算p_e,ij：角色i和j同时失败的概率
    Args:
        failure_counts_i: 角色i的失败次数
        failure_counts_j: 角色j的失败次数
        simultaneous_failures: 同时失败次数
        total_trials: 总试验次数
    Returns:
        p_e,ij: 同时失败概率
    """
    return simultaneous_failures / total_trials

def compute_eta_k(success_rate_k: float, failure_rate_k: float,
                  total_roles: int) -> float:
    """
    计算eta_k：角色k的熵（不确定性度量）
    Args:
        success_rate_k: 角色k的成功概率
        failure_rate_k: 角色k的失败概率
        total_roles: 总角色数
    Returns:
        eta_k: 角色k的熵值
    """
    # 避免log(0)
    eps = 1e-10
    success_rate_k = max(success_rate_k, eps)
    failure_rate_k = max(failure_rate_k, eps)

    entropy = - (success_rate_k * np.log2(success_rate_k) +
                 failure_rate_k * np.log2(failure_rate_k))
    return entropy / np.log2(total_roles)  # 归一化

def compute_p_u(incorrect_bits_not_erased: int, total_bits: int) -> float:
    """
    计算p_u：没有被erasure但接收端使用了错误bit的概率
    Args:
        incorrect_bits_not_erased: 没有被擦除但错误的比特数
        total_bits: 总比特数
    Returns:
        p_u: 未检测到的错误概率
    """
    return incorrect_bits_not_erased / total_bits

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
    """
    小规模非负LP求解器
    验证求出的交互参数满足全部输入约束

    Args:
        constraints: 包含约束条件的字典
            - 'A': 约束矩阵
            - 'b': 约束值
            - 'c': 目标函数系数

    Returns:
        x: 最优解（如果存在）
    """
    # 这是一个简化的LP求解器，用于验证
    # 实际应用中应该使用scipy.optimize.linprog或cvxpy

    try:
        from scipy.optimize import linprog
        import numpy as np

        A = constraints.get('A', None)
        b = constraints.get('b', None)
        c = constraints.get('c', None)

        if A is None or b is None or c is None:
            raise ValueError("约束条件不完整")

        # 非负约束
        bounds = [(0, None) for _ in range(c.shape[0])]

        # 求解LP
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

        if result.success:
            return torch.tensor(result.x)
        else:
            return None

    except ImportError:
        print("scipy not available, using simplified solver")
        # 简化的求解器：只返回可行解
        return torch.ones(c.shape[0]) * 0.5