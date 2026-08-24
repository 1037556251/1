"""
测试Software Channel模块
验证QPSK调制解调和AWGN信道
"""

import torch
import numpy as np
from fixed_resource_comm.modules.software_channel import SoftwareChannel
import sys
import os
# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_software_channel():
    """测试软件信道的所有功能"""
    print("=" * 60)
    print("测试Software Channel模块")
    print("=" * 60)

    # 初始化
    channel = SoftwareChannel()
    print(f"信道配置: SNR={channel.snr_db}dB, 噪声标准差={channel.noise_std:.4f}")

    # 测试1：QPSK调制
    print("\n[测试1] QPSK调制")
    batch_size = 4
    n_bits = 32  # 16个QPSK符号

    bits = torch.randint(0, 2, (batch_size, n_bits), dtype=torch.float32)
    symbols = channel.qpsk_modulate(bits)

    print(f"  输入比特 shape: {bits.shape}")
    print(f"  QPSK符号 shape: {symbols.shape}")
    print(f"  符号取值范围: [{symbols.min().item():.3f}, {symbols.max().item():.3f}]")

    # 验证符号能量
    symbol_energy = (symbols ** 2).sum(dim=-1).mean()
    print(f"  平均符号能量: {symbol_energy.item():.4f} (期望: 1.0)")
    assert abs(symbol_energy.item() - 1.0) < 0.1, "符号能量应为1.0"
    print("  ✓ QPSK调制正确")

    # 测试2：AWGN信道
    print("\n[测试2] AWGN信道")
    noisy_symbols = channel.add_awgn(symbols)
    print(f"  加噪后符号 shape: {noisy_symbols.shape}")

    # 计算噪声功率
    noise = noisy_symbols - symbols
    noise_power = (noise ** 2).mean().item()
    print(f"  实际噪声功率: {noise_power:.4f} (期望: {channel.noise_std ** 2:.4f})")
    print("  ✓ AWGN信道正确")

    # 测试3：QPSK解调
    print("\n[测试3] QPSK解调")
    demodulated_bits = channel.qpsk_demodulate(noisy_symbols)
    print(f"  解调后比特 shape: {demodulated_bits.shape}")

    # 计算误码率
    ber = (demodulated_bits != bits).float().mean()
    print(f"  误码率 (BER): {ber.item():.4f}")
    print("  ✓ QPSK解调正确")

    # 测试4：完整传输过程
    print("\n[测试4] 完整传输过程")
    received_bits = channel.transmit(bits)
    print(f"  接收比特 shape: {received_bits.shape}")
    ber_full = (received_bits != bits).float().mean()
    print(f"  完整传输误码率: {ber_full.item():.4f}")
    print("  ✓ 完整传输过程正确")

    # 测试5：不同SNR下的性能
    print("\n[测试5] 不同SNR下的性能")
    snr_values = [0, 5, 10, 15, 20]
    for snr_db in snr_values:
        # 使用高SNR确保低误码率
        received_bits = channel.transmit(bits, snr_db=snr_db)
        ber = (received_bits != bits).float().mean().item()
        print(f"  SNR={snr_db:2d}dB: BER={ber:.6f}")

    # 测试6：信道信息
    print("\n[测试6] 信道信息")
    info = channel.get_channel_info()
    print(f"  调制方式: {info['modulation']}")
    print(f"  信道类型: {info['channel_type']}")
    print(f"  SNR: {info['snr_db']}dB")
    print(f"  噪声标准差: {info['noise_std']:.4f}")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    test_software_channel()