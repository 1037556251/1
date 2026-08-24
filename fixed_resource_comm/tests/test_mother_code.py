"""
测试Mother Code模块
验证编码、解码、缩短和打孔功能
"""

import torch
import numpy as np
from fixed_resource_comm.modules.mother_code import MotherCode
import sys
import os
# 将项目根目录添加到 sys.path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_shortening():
    """测试shortening接口"""
    print("\n[测试] Shortening接口")
    print("=" * 40)

    mother_code = MotherCode()
    batch_size = 4
    n = mother_code.n
    q_bits = mother_code.k

    # 生成测试数据
    source_bits = torch.randint(0, 2, (batch_size, q_bits), dtype=torch.float32)
    encoded_bits = mother_code.encode(source_bits, n, q_bits)

    # 测试shortening
    n_shorten = 10
    shortened_bits = mother_code.shortening(encoded_bits, n_shorten)

    # 验证：前n_shorten位应为0
    assert (shortened_bits[:, :n_shorten] == 0).all(), "前n_shorten位应被设为0"
    # 验证：后n-n_shorten位应保持不变
    assert (shortened_bits[:, n_shorten:] == encoded_bits[:, n_shorten:]).all(), "后n-n_shorten位应保持不变"
    print(f"  Shape: {encoded_bits.shape} -> {shortened_bits.shape}")
    print("  ✓ Shortening测试通过")


def test_puncturing():
    """测试puncturing接口"""
    print("\n[测试] Puncturing接口")
    print("=" * 40)

    mother_code = MotherCode()
    batch_size = 4
    n = mother_code.n
    q_bits = mother_code.k

    # 生成测试数据
    source_bits = torch.randint(0, 2, (batch_size, q_bits), dtype=torch.float32)
    encoded_bits = mother_code.encode(source_bits, n, q_bits)

    # 测试puncturing
    n_puncture = 10
    punctured_bits = mother_code.puncturing(encoded_bits, n_puncture)

    # 验证：输出shape应为 [batch, n - n_puncture]
    assert punctured_bits.shape == (batch_size, n - n_puncture), f"输出shape应为({batch_size}, {n - n_puncture})"
    print(f"  Shape: {encoded_bits.shape} -> {punctured_bits.shape}")
    print("  ✓ Puncturing测试通过")
def test_mother_code():
    """测试母码接口的所有功能"""
    print("=" * 60)
    print("测试Mother Code模块")
    print("=" * 60)

    # 初始化
    mother_code = MotherCode()
    k = mother_code.k   # 从配置读取
    n = mother_code.n   # 从配置读取

    # 测试1：基本编码
    print("\n[测试1] 基本编码")
    batch_size = 4
    q_bits = k  # 使用全信息位
    n_total = n  # 使用全码长

    # 生成随机源比特
    source_bits = torch.randint(0, 2, (batch_size, q_bits), dtype=torch.float32)
    print(f"  源比特 shape: {source_bits.shape}")
    print(f"  源比特内容 (第0个样本): {source_bits[0].numpy()}")

    # 编码
    encoded_bits = mother_code.encode(source_bits, n_total, q_bits)
    print(f"  编码后 shape: {encoded_bits.shape}")
    print(f"  编码后内容 (第0个样本): {encoded_bits[0].numpy()}")

    # 验证输出维度
    assert encoded_bits.shape == (batch_size, n_total), \
        f"编码输出维度错误: {encoded_bits.shape} != {(batch_size, n_total)}"
    print("  ✓ 编码输出维度正确")

    # 测试2：基本解码
    print("\n[测试2] 基本解码")
    decoded_bits = mother_code.decode(encoded_bits, n_total, q_bits)
    print(f"  解码后 shape: {decoded_bits.shape}")
    print(f"  解码后内容 (第0个样本): {decoded_bits[0].numpy()}")

    # 验证输出维度
    assert decoded_bits.shape == (batch_size, q_bits), \
        f"解码输出维度错误: {decoded_bits.shape} != {(batch_size, q_bits)}"
    print("  ✓ 解码输出维度正确")

    # 验证无噪声情况下的正确性
    accuracy = (decoded_bits == source_bits).float().mean()
    print(f"  无噪声解码准确率: {accuracy.item():.4f}")

    if accuracy.item() < 1.0:
        # 找出差异的位置
        diff_mask = (decoded_bits != source_bits)
        diff_count = diff_mask.sum().item()
        print(f"  ⚠ 存在 {diff_count} 个差异位置")
        if diff_count > 0:
            diff_indices = torch.nonzero(diff_mask)
            print(f"  第一个差异位置: {diff_indices[0].numpy()}")
    else:
        print("  ✓ 无噪声解码完全正确")

    # 测试3：缩短和打孔
    print("\n[测试3] 缩短和打孔")
    test_q_bits = k - 4  # 缩短少量
    test_n_total = n - 8  # 打孔少量

    source_bits_short = torch.randint(0, 2, (batch_size, test_q_bits), dtype=torch.float32)
    encoded_short = mother_code.encode(source_bits_short, test_n_total, test_q_bits)
    print(f"  缩短后编码 shape: {encoded_short.shape}")
    print(f"  期望输出: [{batch_size}, {test_n_total}]")

    # 验证输出维度
    assert encoded_short.shape == (batch_size, test_n_total), \
        f"缩短编码输出维度错误: {encoded_short.shape} != {(batch_size, test_n_total)}"
    print("  ✓ 缩短编码输出维度正确")

    # 测试缩短后的解码
    decoded_short = mother_code.decode(encoded_short, test_n_total, test_q_bits)
    accuracy_short = (decoded_short == source_bits_short).float().mean()
    print(f"  缩短后解码准确率: {accuracy_short.item():.4f}")

    # 测试4：Source Mask应用
    print("\n[测试4] Source Mask测试")
    max_n = n
    mask = torch.zeros(batch_size, 8, max_n)
    # 为角色0设置前16位为1
    mask[:, 0, :16] = 1
    # 为角色1设置前24位为1
    mask[:, 1, :24] = 1

    test_bits = torch.ones(batch_size, 8, max_n)
    masked_bits = mother_code.apply_source_mask(test_bits, mask)

    expected_nonzero = (16 + 24) * batch_size
    actual_nonzero = masked_bits.sum().item()
    print(f"  应用mask后的非零元素数: {actual_nonzero}")
    print(f"  期望的非零元素数: {expected_nonzero}")

    # 验证mask应用
    assert actual_nonzero == expected_nonzero, \
        f"Mask应用错误: {actual_nonzero} != {expected_nonzero}"
    print("  ✓ Source Mask应用正确")

    # 测试5：不同profile的编码解码
    print("\n[测试5] 不同profile的编码解码测试")
    test_profiles = [
        {'q': [k] * 8, 'n': [n] * 8, 'name': 'Uniform'},
        {'q': [k, k // 2, k // 2, k // 2, k // 2, k, k, k],
         'n': [n, n // 2, n // 2, n // 2, n // 2, n, n, n], 'name': 'Single-emphasis'},
    ]

    for profile in test_profiles:
        print(f"\n  测试 {profile['name']} profile:")
        all_correct = True
        for role_idx in range(8):
            q_bits = profile['q'][role_idx]
            n_bits = profile['n'][role_idx]

            # 生成源比特
            source_bits = torch.randint(0, 2, (batch_size, q_bits), dtype=torch.float32)

            # 编码
            encoded_bits = mother_code.encode(source_bits, n_bits, q_bits)

            # 解码
            decoded_bits = mother_code.decode(encoded_bits, n_bits, q_bits)

            # 验证维度
            assert encoded_bits.shape == (batch_size, n_bits), \
                f"角色{role_idx}编码输出维度错误"
            assert decoded_bits.shape == (batch_size, q_bits), \
                f"角色{role_idx}解码输出维度错误"

            # 计算准确率
            accuracy = (decoded_bits == source_bits).float().mean().item()
            status = "✓" if accuracy >= 1.0 else "⚠"
            if accuracy < 1.0:
                all_correct = False
            print(f"    角色{role_idx}: q={q_bits}, n={n_bits}, 准确率={accuracy:.4f} {status}")

        if all_correct:
            print(f"  ✓ {profile['name']} profile所有角色编码解码正确")
        else:
            print(f"  ⚠ {profile['name']} profile存在编码解码差异")

    # 测试6：校验矩阵信息
    print("\n[测试6] 校验矩阵信息")
    matrix_info = mother_code.get_matrix_info()
    print(f"  码长 (n): {matrix_info['n']}")
    print(f"  信息位 (k): {matrix_info['k']}")
    print(f"  校验位 (m): {matrix_info['m']}")
    print(f"  校验矩阵 shape: {matrix_info['H_shape']}")
    print(f"  校验矩阵密度: {matrix_info['H_density']:.4f}")

    # 验证矩阵信息
    assert matrix_info['n'] == n, f"码长应为{n}"
    assert matrix_info['k'] == k, f"信息位应为{k}"
    assert matrix_info['m'] == n - k, f"校验位应为{n - k}"
    assert matrix_info['H_shape'] == (n - k, n), f"校验矩阵shape应为({n - k}, {n})"
    print("  ✓ 校验矩阵信息正确")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)

if __name__ == "__main__":
    test_mother_code()
    test_puncturing()
    test_shortening()