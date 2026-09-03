"""小规模非负 LP 接口的手工构造测试。"""

import numpy as np

from modules.math_functions import solve_lp_small_scale


def test_lp_inequality_and_nonnegativity():
    """在 x+y >= 4 且 x、y >= 0 的约束下最小化 x+2y。"""
    result = solve_lp_small_scale({
        "c": np.array([1.0, 2.0]),
        "A_ub": np.array([[-1.0, -1.0]]),
        "b_ub": np.array([-4.0]),
    })
    assert result is not None
    x = result.numpy()
    np.testing.assert_allclose(x, [4.0, 0.0], atol=1e-6)
    assert np.all(x >= -1e-7)
    assert np.all(np.array([[-1.0, -1.0]]) @ x <= np.array([-4.0]) + 1e-7)


def test_lp_equality_and_nonnegativity():
    """在 x+y=5 且 x、y >= 0 的约束下最小化 x+3y。"""
    result = solve_lp_small_scale({
        "c": np.array([1.0, 3.0]),
        "A_eq": np.array([[1.0, 1.0]]),
        "b_eq": np.array([5.0]),
    })
    assert result is not None
    x = result.numpy()
    np.testing.assert_allclose(x, [5.0, 0.0], atol=1e-6)
    assert np.all(x >= -1e-7)
    np.testing.assert_allclose(np.array([[1.0, 1.0]]) @ x, [5.0], atol=1e-7)


def test_lp_mixed_constraints_and_explicit_bounds():
    """验证大于等于需求、上界、可行性和非负性约束。"""
    result = solve_lp_small_scale({
        "c": np.array([1.0, 1.0]),
        "A_ub": np.array([[-1.0, -1.0]]),  # x+y >= 3
        "b_ub": np.array([-3.0]),
        "bounds": [(0.0, 2.0), (0.0, 2.0)],
    })
    assert result is not None
    x = result.numpy()
    np.testing.assert_allclose(x.sum(), 3.0, atol=1e-6)
    assert np.all(x >= -1e-7)
    assert np.all(x <= 2.0 + 1e-7)
    assert np.all(np.array([[-1.0, -1.0]]) @ x <= np.array([-3.0]) + 1e-7)
