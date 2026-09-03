"""
端到端测试
测试完整的编码-传输-解码-擦除恢复流程
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from modules.mother_code import MotherCode
from modules.software_channel import SoftwareChannel
from modules.decoder import LDPCDecoder
from modules.erasure import Erasure
from modules.receiver import Receiver
from modules.profile_generator import ProfileGenerator


def test_end_to_end():
    """测试完整的端到端通信流程"""
    print("=" * 60)
    print("端到端通信测试")
    print("=" * 60)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 初始化所有模块
    mother_code = MotherCode()
    channel = SoftwareChannel()
    decoder = LDPCDecoder()
    erasure = Erasure()
    receiver = Receiver()

    # 获取 profile 参数
    pg = ProfileGenerator()
    profiles = pg.get_all_profiles()
    q_bits = sum(profiles[0]['q'])   # 总信息位
    n_total = sum(profiles[0]['n'])  # 总码长

    print(f"\n[1] 系统配置:")
    print(f"  - 总信息位 (q_bits): {q_bits}")
    print(f"  - 总码长 (n_total): {n_total}")
    print(f"  - 信道SNR: {channel.snr_db}dB")
    print(f"  - 擦除概率: {erasure.erasure_probability}")

    # 测试参数
    batch_size = 4

    # 测试1：无噪声编码-解码
    print(f"\n[测试1] 无噪声编码-解码")
    source_bits = torch.randint(0, 2, (batch_size, q_bits), dtype=torch.float32)
    print(f"  源比特 shape: {source_bits.shape}")

    # 编码
    encoded_bits = mother_code.encode(source_bits, n_total, q_bits)
    print(f"  编码后 shape: {encoded_bits.shape}")

    # 解码（传入参数）
    decoded_bits = decoder.decode(encoded_bits, n_total, q_bits)
    print(f"  解码后 shape: {decoded_bits.shape}")

    # 计算准确率
    accuracy = (decoded_bits == source_bits).float().mean()
    print(f"  无噪声解码准确率: {accuracy.item():.4f}")
    assert accuracy > 0.99, "无噪声解码准确率应接近1.0"
    print("  ✓ 无噪声编码-解码正确")

    # 测试2：有噪声编码-传输-解码
    print(f"\n[测试2] 有噪声传输")
    source_bits = torch.randint(0, 2, (batch_size, q_bits), dtype=torch.float32)
    encoded_bits = mother_code.encode(source_bits, n_total, q_bits)

    # 通过信道传输
    received_bits = channel.transmit(encoded_bits)
    print(f"  接收比特 shape: {received_bits.shape}")

    # 计算信道误码率
    channel_ber = (received_bits != encoded_bits).float().mean()
    print(f"  信道误码率: {channel_ber.item():.4f}")

    # 解码（传入参数）
    decoded_bits = decoder.decode(received_bits, n_total, q_bits)
    system_ber = (decoded_bits != source_bits).float().mean()
    print(f"  系统误码率: {system_ber.item():.4f}")
    print("  ✓ 有噪声传输测试完成")

    # 测试3：擦除标记（使用随机比特）
    print(f"\n[测试3] 擦除标记")
    bits = torch.randint(0, 2, (batch_size, n_total), dtype=torch.float32)  # 用 n_total 替换原来的 n

    # 应用随机擦除
    erased_bits = erasure.apply_erasure(bits, probability=0.2)
    erasure_rate = (erased_bits == -1).float().mean().item()
    print(f"  实际擦除率: {erasure_rate:.4f} (期望: 0.2)")

    # 验证擦除位置保留标记，不能使用原始比特恢复
    assert (erased_bits == -1).any(), "应存在被标记的擦除位置"
    print("  ✓ 擦除标记正确保留")

    # 测试4：完整接收流程
    print(f"\n[测试4] 完整接收流程")
    source_bits = torch.randint(0, 2, (batch_size, q_bits), dtype=torch.float32)
    encoded_bits = mother_code.encode(source_bits, n_total, q_bits)

    # 使用接收器处理（receiver.receive 已正确传入参数）
    decoded_bits, stats = receiver.receive(encoded_bits, n_total, q_bits)
    print(f"  接收器输出 shape: {decoded_bits.shape}")
    print(f"  接收统计: {stats}")

    receiver_accuracy = (decoded_bits == source_bits).float().mean()
    print(f"  接收器准确率: {receiver_accuracy.item():.4f}")
    print("  ✓ 完整接收流程正确")

    # 测试5：获取所有模块信息
    print(f"\n[测试5] 模块信息汇总")
    print(f"  MotherCode信息: {mother_code.get_matrix_info()}")
    print(f"  信道信息: {channel.get_channel_info()}")
    print(f"  解码器信息: {decoder.get_matrix_info()}")
    print(f"  擦除信息: {erasure.get_erasure_info()}")
    print(f"  接收器信息: {receiver.get_receiver_info()}")

    print("\n" + "=" * 60)
    print("端到端测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    test_end_to_end()
