"""
软件信道模块
实现QPSK调制和AWGN信道模拟
"""

import torch
import numpy as np
from typing import Optional, Tuple
import os
class SoftwareChannel:
    def __init__(self, config_path: str = None):
        """初始化软件信道

        Args:
            config_path: 配置文件路径
        """
        import yaml
        if config_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(module_dir, '..', '..', 'configs', 'toy_config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 从配置读取信道参数
        channel_config = self.config.get('channel', {})
        self.snr_db = channel_config.get('snr_db', 10.0)  # 默认SNR为10dB
        self.random_seed = channel_config.get('random_seed', 42)
        torch.manual_seed(int(self.random_seed))
        self.noise_std = self._calculate_noise_std()
        self.last_noisy_symbols = None
        self.last_soft_bits = None

    def _calculate_noise_std(self) -> float:
        """根据SNR计算噪声标准差

        QPSK调制下，每个符号携带2比特
        SNR(dB) = 10 * log10(信号功率 / 噪声功率)

        Returns:
            noise_std: 噪声标准差
        """
        snr_linear = 10 ** (self.snr_db / 10.0)
        signal_power = 1.0  # QPSK信号功率归一化为1
        noise_power = signal_power / (2.0 * snr_linear)
        noise_std = np.sqrt(noise_power)
        return noise_std

    def qpsk_modulate(self, bits: torch.Tensor) -> torch.Tensor:
        """QPSK调制

        将比特流映射为QPSK符号
        映射规则：
        - 00 -> (1, 1) / sqrt(2)
        - 01 -> (1, -1) / sqrt(2)
        - 10 -> (-1, 1) / sqrt(2)
        - 11 -> (-1, -1) / sqrt(2)

        Args:
            bits: 输入比特，shape [batch, n_bits]

        Returns:
            symbols: QPSK符号，shape [batch, n_symbols, 2] (实部, 虚部)
        """
        # 修正：正确提取维度（使用索引0和1获取具体数值）
        batch_size = bits.shape[0]  # 获取批量大小（整数）
        n_bits = bits.shape[1]      # 获取总比特数（整数）
        n_symbols = n_bits // 2      # 计算QPSK符号数

        # 确保比特数是偶数
        assert n_bits % 2 == 0, f"比特数必须为偶数，当前为{n_bits}"

        # 将比特流分组为2比特一组
        bits_reshaped = bits.reshape(batch_size, n_symbols, 2)

        # QPSK映射
        # 实部: 1->-1, 0->1
        real_part = 1 - 2 * bits_reshaped[:, :, 0]  # [batch, n_symbols]
        # 虚部: 1->-1, 0->1
        imag_part = 1 - 2 * bits_reshaped[:, :, 1]  # [batch, n_symbols]

        # 归一化到单位能量
        sqrt2 = np.sqrt(2)
        symbols = torch.stack([real_part / sqrt2, imag_part / sqrt2], dim=-1)

        return symbols

    def add_awgn(self, symbols: torch.Tensor) -> torch.Tensor:
        """添加AWGN噪声

        Args:
            symbols: QPSK符号，shape [batch, n_symbols, 2]

        Returns:
            noisy_symbols: 加噪后的符号，shape [batch, n_symbols, 2]
        """
        noise = torch.randn_like(symbols) * self.noise_std
        noisy_symbols = symbols + noise
        return noisy_symbols

    def qpsk_demodulate(self, noisy_symbols: torch.Tensor) -> torch.Tensor:
        """QPSK解调（硬判决）

        将接收到的QPSK符号解调为比特流

        Args:
            noisy_symbols: 接收到的符号，shape [batch, n_symbols, 2]

        Returns:
            bits: 解调后的比特，shape [batch, n_bits]
        """
        # 修正：正确提取维度
        batch_size = noisy_symbols.shape[0]  # 获取批量大小（整数）
        n_symbols = noisy_symbols.shape[1]   # 获取符号数（整数）

        # 实部 > 0 -> 0, 实部 < 0 -> 1
        real_bits = (noisy_symbols[:, :, 0] < 0).float()  # [batch, n_symbols]
        # 虚部 > 0 -> 0, 虚部 < 0 -> 1
        imag_bits = (noisy_symbols[:, :, 1] < 0).float()  # [batch, n_symbols]

        # 合并比特流并展平为 [batch, n_bits]
        bits = torch.stack([real_bits, imag_bits], dim=-1)  # [batch, n_symbols, 2]
        bits = bits.reshape(batch_size, n_symbols * 2)  # [batch, n_bits]

        return bits

    def transmit(self, bits: torch.Tensor, snr_db: Optional[float] = None,
                 return_soft: bool = False):
        """完整传输过程：调制 + 信道 + 解调

        Args:
            bits: 输入比特，shape [batch, n_bits]
            snr_db: 可选的SNR值，如果提供则覆盖默认值

        Returns:
            received_bits: 接收到的比特，shape [batch, n_bits]
        """
        if bits.ndim != 2 or bits.shape[1] % 2:
            raise ValueError("bits must have shape [batch, even number of bits]")
        if snr_db is not None:
            self.snr_db = snr_db
            self.noise_std = self._calculate_noise_std()

        symbols = self.qpsk_modulate(bits)

        noisy_symbols = symbols if self.snr_db == float('inf') else self.add_awgn(symbols)
        self.last_noisy_symbols = noisy_symbols
        received_bits = self.qpsk_demodulate(noisy_symbols)
        self.last_soft_bits = noisy_symbols.reshape(bits.shape[0], -1)
        return (received_bits, self.last_soft_bits) if return_soft else received_bits

    def get_channel_info(self) -> dict:
        """返回信道信息"""
        return {
            'snr_db': self.snr_db,
            'noise_std': self.noise_std,
            'modulation': 'QPSK',
            'channel_type': 'AWGN'
        }
