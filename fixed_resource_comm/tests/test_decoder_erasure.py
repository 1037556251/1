"""
测试Decoder和Erasure模块
"""

import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fixed_resource_comm.modules.decoder import LDPCDecoder
from fixed_resource_comm.modules.erasure import Erasure
from fixed_resource_comm.modules.profile_generator import ProfileGenerator

# 将项目根目录添加到 sys.path

def test_decoder():
    """测试解码器模块"""
    print("=" * 60)
    print("测试Decoder模块")
    print("=" * 60)

    # 初始化解码器
    decoder = LDPCDecoder()
    print(f"解码器配置:")
    print(f"  - 算法: {decoder.algorithm}")
    print(f"  - 最大迭代次数: {decoder.max_iterations}")
    print(f"  - 校验矩阵 shape: {decoder.H.shape}")

    # 获取 profile 参数
    pg = ProfileGenerator()
    profiles = pg.get_all_profiles()
    q_bits = sum(profiles[0]['q'])   # 总信息位
    n_total = sum(profiles[0]['n'])  # 总码长

    # 生成随机接收比特（长度应为 n_total）
    batch_size = 4
    received_bits = torch.randint(0, 2, (batch_size, n_total), dtype=torch.float32)
    print(f"\n接收比特 shape: {received_bits.shape}")

    # 解码（传入必要参数）
    decoded_bits = decoder.decode(received_bits, n_total, q_bits)
    print(f"解码后比特 shape: {decoded_bits.shape}")
    assert decoded_bits.shape == (batch_size, q_bits), f"解码输出维度错误，期望 {(batch_size, q_bits)}，实际 {decoded_bits.shape}"
    print("✓ 解码输出维度正确")

    # 获取解码器信息
    info = decoder.get_matrix_info()
    print(f"\n解码器信息:")
    print(f"  - 码长 (n): {info['n']}")
    print(f"  - 信息位 (k): {info['k']}")
    print(f"  - 校验位 (m): {info['m']}")
    print("✓ 解码器信息正确")


def test_erasure():
    """测试擦除模块"""
    print("\n" + "=" * 60)
    print("测试Erasure模块")
    print("=" * 60)

    # 初始化擦除模块
    erasure = Erasure()
    print(f"擦除配置:")
    print(f"  - 擦除概率: {erasure.erasure_probability}")

    # 测试1：随机擦除
    print("\n[测试1] 随机擦除")
    batch_size = 4
    n_bits = 32

    # 生成随机比特
    bits = torch.randint(0, 2, (batch_size, n_bits), dtype=torch.float32)
    print(f"原始比特 shape: {bits.shape}")

    # 应用随机擦除
    erased_bits = erasure.apply_erasure(bits, probability=0.2)
    print(f"擦除后比特 shape: {erased_bits.shape}")

    # 计算擦除率
    erasure_rate = (erased_bits == -1).float().mean().item()
    print(f"实际擦除率: {erasure_rate:.4f} (期望: 0.2)")
    assert abs(erasure_rate - 0.2) < 0.1, "擦除率应在0.2附近"
    print("✓ 随机擦除正确")

    # 测试2：模式擦除
    print("\n[测试2] 模式擦除")
    pattern = torch.zeros(n_bits)
    pattern[:8] = 1  # 擦除前8个位置

    pattern_erased = erasure.apply_pattern_erasure(bits, pattern)
    print(f"模式擦除后 shape: {pattern_erased.shape}")

    # 验证前8位被擦除
    assert (pattern_erased[0, :8] == -1).all(), "前8位应该被擦除"
    assert (pattern_erased[0, 8:] == bits[0, 8:]).all(), "后24位应该保留"
    print("✓ 模式擦除正确")

    # 测试3：恢复擦除
    print("\n[测试3] 恢复擦除")
    recovered_bits = erasure.recover_erasure(erased_bits, bits)
    print(f"恢复后比特 shape: {recovered_bits.shape}")

    # 验证恢复完全正确
    assert (recovered_bits == bits).all(), "恢复应该完全正确"
    print("✓ 擦除恢复正确")

    # 获取擦除信息
    info = erasure.get_erasure_info()
    print(f"\n擦除信息:")
    print(f"  - 擦除概率: {info['erasure_probability']}")
    print("✓ 擦除信息正确")


if __name__ == "__main__":
    test_decoder()
    test_erasure()
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)