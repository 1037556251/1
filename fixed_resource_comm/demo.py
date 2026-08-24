"""
固定资源通信系统 - 完整演示
展示所有模块的完整工作流程
"""

import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.mother_code import MotherCode
from modules.software_channel import SoftwareChannel
from modules.decoder import LDPCDecoder
from modules.erasure import Erasure
from modules.receiver import Receiver
from modules.profile_generator import ProfileGenerator


def demo_complete_flow():
    print("=" * 60)
    print("固定资源通信系统 - 完整演示 (toy版本)")
    print("=" * 60)

    # 1. 初始化所有模块
    print("\n[1] 初始化模块...")
    mother_code = MotherCode()
    channel = SoftwareChannel()
    decoder = LDPCDecoder()
    erasure = Erasure()
    receiver = Receiver()
    profile_generator = ProfileGenerator()
    print("  ✓ 所有模块初始化完成")

    # 2. 生成随机输入（toy版本：直接用随机比特模拟）
    print("\n[2] 生成随机输入...")
    batch_size = 4
    M = 128
    d_x = 64
    input_tensor = torch.randn(batch_size, M, d_x)
    print(f"  input_tensor shape: {input_tensor.shape}")

    # 3. 生成profile并获取总参数
    print("\n[3] 生成profile...")
    profiles = profile_generator.get_all_profiles()          # 修改方法名
    print(f"  生成profile数: {len(profiles)}")
    first_profile = profiles[0]
    q_bits = sum(first_profile['q'])   # 总信息位
    n_total = sum(first_profile['n'])  # 总码长
    print(f"  总信息位 q_bits = {q_bits}, 总码长 n_total = {n_total}")

    # 4. 生成源比特（代替真实的量化过程）
    print("\n[4] 生成源比特...")
    source_bits = torch.randint(0, 2, (batch_size, q_bits), dtype=torch.float32)
    print(f"  source_bits shape: {source_bits.shape}")

    # 5. 编码（包含缩短和打孔内部处理）
    print("\n[5] 编码...")
    encoded_bits = mother_code.encode(source_bits, n_total, q_bits)
    print(f"  encoded_bits shape: {encoded_bits.shape}")

    # 6. 信道传输
    print("\n[6] 信道传输...")
    received_bits = channel.transmit(encoded_bits)
    print(f"  received_bits shape: {received_bits.shape}")

    # 7. 解码
    print("\n[7] 解码...")
    decoded_bits, stats = receiver.receive(received_bits, n_total, q_bits)
    print(f"  decoded_bits shape: {decoded_bits.shape}")
    print(f"  stats: {stats}")

    # 8. 擦除恢复（可选演示）
    print("\n[8] 擦除恢复演示...")
    erased_bits = erasure.apply_erasure(decoded_bits, probability=0.1)
    recovered_bits = erasure.recover_erasure(erased_bits, decoded_bits)
    print(f"  recovered_bits shape: {recovered_bits.shape}")

    # 9. 计算准确率
    accuracy = (decoded_bits == source_bits).float().mean()
    print(f"\n[结果] 系统准确率: {accuracy.item():.4f}")

    print("\n" + "=" * 60)
    print("完整演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    demo_complete_flow()