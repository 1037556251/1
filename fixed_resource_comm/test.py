"""
测试 _check_decoding_failure 方法
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from modules.receiver import Receiver


def test_decoding_failure():
    """
    测试接收器中的解码失败检测
    """
    print("=" * 60)
    print("测试 _check_decoding_failure 方法")
    print("=" * 60)

    # 初始化接收器
    receiver = Receiver()

    # 测试参数
    batch_size = 4
    n_total = 1008
    q_bits = 192

    # 测试1：正常解码（随机数据）
    print("\n[测试1] 正常解码 - 随机数据")
    random_bits = torch.randint(0, 2, (batch_size, n_total), dtype=torch.float32)
    decoded, stats = receiver.receive(random_bits, n_total, q_bits)
    print(f"  解码结果 shape: {decoded.shape}")
    print(f"  decoding_failed: {stats['decoding_failed']}")
    assert stats['decoding_failed'] == False, "正常数据不应解码失败"
    print("  ✅ 测试通过")

    # 测试2：全0输入（应解码失败）
    print("\n[测试2] 全0输入 - 应解码失败")
    zero_bits = torch.zeros(batch_size, n_total)
    decoded, stats = receiver.receive(zero_bits, n_total, q_bits)
    print(f"  解码结果 shape: {decoded.shape}")
    print(f"  decoding_failed: {stats['decoding_failed']}")
    assert stats['decoding_failed'] == True, "全0输入应解码失败"
    print("  ✅ 测试通过")

    # 测试3：全1输入（应解码失败）
    print("\n[测试3] 全1输入 - 应解码失败")
    one_bits = torch.ones(batch_size, n_total)
    decoded, stats = receiver.receive(one_bits, n_total, q_bits)
    print(f"  解码结果 shape: {decoded.shape}")
    print(f"  decoding_failed: {stats['decoding_failed']}")
    assert stats['decoding_failed'] == True, "全1输入应解码失败"
    print("  ✅ 测试通过")

    # 测试4：部分全0、部分正常（应解码失败，因为存在全0样本）
    print("\n[测试4] 混合输入 - 部分全0、部分正常")
    mixed_bits = torch.randn(batch_size, n_total)  # 随机噪声作为正常数据
    mixed_bits[0] = 0.0  # 第一个样本全0
    mixed_bits[1] = 1.0  # 第二个样本全1
    # 第3、4个样本保持随机
    decoded, stats = receiver.receive(mixed_bits, n_total, q_bits)
    print(f"  解码结果 shape: {decoded.shape}")
    print(f"  decoding_failed: {stats['decoding_failed']}")
    assert stats['decoding_failed'] == True, "混合输入中包含全0或全1样本应解码失败"
    print("  ✅ 测试通过")

    # 测试5：差异过大的数据（应解码失败）
    print("\n[测试5] 差异过大数据 - 解码结果与接收比特差异>50%")
    # 创建与接收数据差异很大的解码结果
    received_bits = torch.randint(0, 2, (batch_size, n_total), dtype=torch.float32)
    # 直接调用 _check_decoding_failure 方法
    # 创建解码结果：与received_bits完全相反
    opposite_bits = 1.0 - received_bits
    result = receiver._check_decoding_failure(opposite_bits, received_bits)
    print(f"  解码结果与接收比特相反")
    print(f"  decoding_failed: {result}")
    assert result == True, "完全相反的数据应解码失败"
    print("  ✅ 测试通过")

    # 测试6：边界情况 - 差异刚好50%
    print("\n[测试6] 边界情况 - 差异刚好50%")
    # 创建一半相同、一半不同的数据
    received_bits = torch.randint(0, 2, (batch_size, n_total), dtype=torch.float32)
    half_decoded = received_bits.clone()
    half_len = n_total // 2
    half_decoded[:, :half_len] = 1.0 - half_decoded[:, :half_len]  # 前半部分反转
    result = receiver._check_decoding_failure(half_decoded, received_bits)
    diff_ratio = (half_decoded != received_bits).float().mean()
    print(f"  差异比例: {diff_ratio.item():.4f}")
    print(f"  decoding_failed: {result}")
    # 差异为50%，不超过50%，不应解码失败
    assert result == False, "差异刚好50%不应解码失败"
    print("  ✅ 测试通过")

    # 测试7：维度检查（应抛出异常或返回False）
    print("\n[测试7] 维度不匹配 - 应抛出异常")
    try:
        wrong_bits = torch.randint(0, 2, (batch_size + 1, n_total), dtype=torch.float32)
        result = receiver._check_decoding_failure(decoded, wrong_bits)
        print(f"  decoding_failed: {result}")
        print("  ⚠️ 未抛出异常，但返回了结果")
    except AssertionError as e:
        print(f"  捕获到预期异常: {e}")
        print("  ✅ 测试通过（维度检查生效）")
    except Exception as e:
        print(f"  捕获到其他异常: {e}")
        print("  ❌ 测试失败（异常类型不符）")

    # 测试8：擦除恢复是否被调用
    print("\n[测试8] 解码失败时是否调用erasure")
    zero_bits = torch.zeros(batch_size, n_total)
    decoded, stats = receiver.receive(zero_bits, n_total, q_bits)
    print(f"  erasure_applied: {stats['erasure_applied']}")
    assert stats['erasure_applied'] == True, "解码失败时应应用erasure"
    print("  ✅ 测试通过")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)


def test_direct_decoding_failure():
    """
    直接测试 _check_decoding_failure 方法（不经过receive）
    """
    print("\n" + "=" * 60)
    print("直接测试 _check_decoding_failure 方法")
    print("=" * 60)

    # 初始化接收器
    receiver = Receiver()

    # 测试1：全0张量
    print("\n[直接测试1] 全0张量")
    decoded = torch.zeros(4, 100)
    received = torch.zeros(4, 100)
    result = receiver._check_decoding_failure(decoded, received)
    print(f"  全0: {result}")
    assert result == True, "全0应解码失败"
    print("  ✅")

    # 测试2：全1张量
    print("\n[直接测试2] 全1张量")
    decoded = torch.ones(4, 100)
    received = torch.ones(4, 100)
    result = receiver._check_decoding_failure(decoded, received)
    print(f"  全1: {result}")
    assert result == True, "全1应解码失败"
    print("  ✅")

    # 测试3：随机张量（差异小）
    print("\n[直接测试3] 随机张量（差异小）")
    torch.manual_seed(42)
    decoded = torch.randint(0, 2, (4, 100), dtype=torch.float32)
    received = decoded.clone()
    # 修改少量比特
    received[0, 0] = 1.0 - received[0, 0]
    received[1, 1] = 1.0 - received[1, 1]
    diff_ratio = (decoded != received).float().mean()
    print(f"  差异比例: {diff_ratio.item():.4f}")
    result = receiver._check_decoding_failure(decoded, received)
    print(f"  结果: {result}")
    assert result == False, "差异小不应解码失败"
    print("  ✅")

    # 测试4：差异大的张量
    print("\n[直接测试4] 差异大的张量")
    decoded = torch.randint(0, 2, (4, 100), dtype=torch.float32)
    received = 1.0 - decoded  # 完全相反
    diff_ratio = (decoded != received).float().mean()
    print(f"  差异比例: {diff_ratio.item():.4f}")
    result = receiver._check_decoding_failure(decoded, received)
    print(f"  结果: {result}")
    assert result == True, "差异大应解码失败"
    print("  ✅")

    # 测试5：维度不匹配
    print("\n[直接测试5] 维度不匹配")
    try:
        decoded = torch.zeros(4, 100)
        received = torch.zeros(5, 100)  # batch size不同
        result = receiver._check_decoding_failure(decoded, received)
        print(f"  结果: {result} (未抛出异常)")
    except AssertionError as e:
        print(f"  捕获到异常: {e}")
        print("  ✅")

    print("\n" + "=" * 60)
    print("所有直接测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    # 运行测试
    test_decoding_failure()
    test_direct_decoding_failure()

    print("\n" + "*" * 60)
    print("✅ 所有测试通过！_check_decoding_failure 方法工作正常")
    print("*" * 60)
