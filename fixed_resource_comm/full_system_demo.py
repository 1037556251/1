"""
完整通信系统演示
整合所有模块，实现完整的编码-传输-解码-擦除恢复流程
"""

import torch
import numpy as np
from modules.mother_code import MotherCode
from modules.software_channel import SoftwareChannel
from modules.decoder import LDPCDecoder
from modules.erasure import Erasure
from modules.receiver import Receiver
import time


def run_system_demo():
    """运行完整系统演示"""
    print("=" * 60)
    print("固定资源通信系统 - 完整演示")
    print("=" * 60)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 初始化所有模块
    print("\n[1] 初始化所有模块...")
    mother_code = MotherCode()
    channel = SoftwareChannel()
    decoder = LDPCDecoder()
    erasure = Erasure()
    receiver = Receiver()
    print("  ✓ 所有模块初始化完成")

    # 获取系统参数
    k = mother_code.k
    n = mother_code.n

    print(f"\n[2] 系统参数配置:")
    print(f"  - 信息位长度 (k): {k}")
    print(f"  - 母码码长 (n): {n}")
    print(f"  - 信道SNR: {channel.snr_db}dB")
    print(f"  - 擦除概率: {erasure.erasure_probability}")
    print(f"  - 解码器最大迭代次数: {decoder.max_iterations}")
    print(f"  - 码率: {k}/{n} = {k / n:.3f}")

    # 测试参数
    batch_size = 4
    q_bits = k
    n_total = n

    # 测试场景1：无噪声完美通信
    print(f"\n[测试场景1] 无噪声完美通信")
    source_bits = torch.randint(0, 2, (batch_size, q_bits), dtype=torch.float32)

    # 编码
    encoded_bits = mother_code.encode(source_bits, n_total, q_bits)

    # 直接传输（无噪声）
    start_time = time.time()
    decoded_bits, stats = receiver.receive(encoded_bits, n_total, q_bits)
    elapsed_time = time.time() - start_time

    # 计算准确率
    accuracy = (decoded_bits == source_bits).float().mean()
    print(f"  - 源比特: {source_bits[0].numpy()}")
    print(f"  - 解码比特: {decoded_bits[0].numpy()}")
    print(f"  - 准确率: {accuracy.item():.4f}")
    print(f"  - 处理时间: {elapsed_time:.4f}秒")
    if accuracy > 0.99:
        print("  ✓ 无噪声通信完全正确")
    else:
        print("  ⚠ 无噪声通信存在错误")

    # 测试场景2：有噪声信道
    print(f"\n[测试场景2] 有噪声信道 (SNR={channel.snr_db}dB)")
    source_bits = torch.randint(0, 2, (batch_size, q_bits), dtype=torch.float32)

    # 编码
    encoded_bits = mother_code.encode(source_bits, n_total, q_bits)

    # 通过有噪声信道传输
    received_bits = channel.transmit(encoded_bits)

    # 计算信道误码率
    channel_ber = (received_bits != encoded_bits).float().mean()

    # 解码
    start_time = time.time()
    decoded_bits, stats = receiver.receive(received_bits, n_total, q_bits)
    elapsed_time = time.time() - start_time

    # 计算系统误码率
    system_ber = (decoded_bits != source_bits).float().mean()

    print(f"  - 信道误码率: {channel_ber.item():.4f}")
    print(f"  - 系统误码率: {system_ber.item():.4f}")
    print(f"  - 处理时间: {elapsed_time:.4f}秒")
    print(f"  - 接收统计: {stats}")

    # 测试场景3：带擦除的通信
    print(f"\n[测试场景3] 带擦除的通信")
    source_bits = torch.randint(0, 2, (batch_size, q_bits), dtype=torch.float32)

    # 编码
    encoded_bits = mother_code.encode(source_bits, n_total, q_bits)

    # 应用擦除
    erased_bits = erasure.apply_erasure(encoded_bits, probability=0.1)
    erasure_rate = (erased_bits == -1).float().mean().item()

    # 恢复擦除
    recovered_bits = erasure.recover_erasure(erased_bits, encoded_bits)

    # 解码
    decoded_bits, stats = receiver.receive(recovered_bits, n_total, q_bits)

    # 计算准确率
    accuracy = (decoded_bits == source_bits).float().mean()

    print(f"  - 擦除率: {erasure_rate:.4f}")
    print(f"  - 恢复后准确率: {accuracy.item():.4f}")
    print(f"  - 接收统计: {stats}")

    # 测试场景4：不同SNR下的性能比较
    print(f"\n[测试场景4] 不同SNR下的性能比较")
    snr_values = [0, 5, 10, 15, 20]

    for snr_db in snr_values:
        # 设置SNR
        channel.snr_db = snr_db
        channel.noise_std = channel._calculate_noise_std()

        # 生成测试数据
        source_bits = torch.randint(0, 2, (batch_size, q_bits), dtype=torch.float32)
        encoded_bits = mother_code.encode(source_bits, n_total, q_bits)

        # 通过信道传输
        received_bits = channel.transmit(encoded_bits)

        # 解码
        decoded_bits, _ = receiver.receive(received_bits, n_total, q_bits)

        # 计算误码率
        ber = (decoded_bits != source_bits).float().mean().item()
        print(f"  SNR={snr_db:2d}dB: BER={ber:.6f}")

    # 测试场景5：批量性能测试
    print(f"\n[测试场景5] 批量性能测试")
    batch_sizes = [1, 4, 8, 16]

    for batch_size in batch_sizes:
        source_bits = torch.randint(0, 2, (batch_size, q_bits), dtype=torch.float32)
        encoded_bits = mother_code.encode(source_bits, n_total, q_bits)

        start_time = time.time()
        decoded_bits, _ = receiver.receive(encoded_bits, n_total, q_bits)
        elapsed_time = time.time() - start_time

        throughput = batch_size / elapsed_time if elapsed_time > 0 else float('inf')
        print(f"  batch_size={batch_size:2d}: 时间={elapsed_time:.4f}秒, 吞吐量={throughput:.2f}批次/秒")

    # 输出系统信息
    print(f"\n[系统信息汇总]")
    print(f"  MotherCode: {mother_code.get_matrix_info()}")
    print(f"  信道: {channel.get_channel_info()}")
    print(f"  解码器: {decoder.get_matrix_info()}")
    print(f"  擦除: {erasure.get_erasure_info()}")
    print(f"  接收器: {receiver.get_receiver_info()}")

    print("\n" + "=" * 60)
    print("系统演示完成！")
    print("=" * 60)


def run_performance_benchmark():
    """运行性能基准测试"""
    print("\n" + "=" * 60)
    print("性能基准测试")
    print("=" * 60)

    # 初始化模块
    mother_code = MotherCode()
    channel = SoftwareChannel()
    receiver = Receiver()

    k = mother_code.k
    n = mother_code.n
    batch_size = 4
    q_bits = k
    n_total = n

    # 生成测试数据
    source_bits = torch.randint(0, 2, (batch_size, q_bits), dtype=torch.float32)

    # 测试编码性能
    print("\n编码性能测试:")
    start_time = time.time()
    num_iterations = 100
    for _ in range(num_iterations):
        encoded_bits = mother_code.encode(source_bits, n_total, q_bits)
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_iterations
    print(f"  平均编码时间: {avg_time * 1000:.4f}毫秒")

    # 测试解码性能
    print("\n解码性能测试:")
    start_time = time.time()
    for _ in range(num_iterations):
        decoded_bits = receiver.decoder.decode(encoded_bits)
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_iterations
    print(f"  平均解码时间: {avg_time * 1000:.4f}毫秒")

    # 测试信道传输性能
    print("\n信道传输性能测试:")
    start_time = time.time()
    for _ in range(num_iterations):
        received_bits = channel.transmit(encoded_bits)
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_iterations
    print(f"  平均传输时间: {avg_time * 1000:.4f}毫秒")

    # 测试完整链路性能
    print("\n完整链路性能测试:")
    start_time = time.time()
    for _ in range(num_iterations):
        encoded_bits = mother_code.encode(source_bits, n_total, q_bits)
        received_bits = channel.transmit(encoded_bits)
        decoded_bits, _ = receiver.receive(received_bits, n_total, q_bits)
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_iterations
    print(f"  平均链路时间: {avg_time * 1000:.4f}毫秒")

    print("\n" + "=" * 60)
    print("性能基准测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    run_system_demo()
    run_performance_benchmark()