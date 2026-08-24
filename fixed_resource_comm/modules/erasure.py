"""
擦除模块
实现统一擦除函数
"""

import torch
import numpy as np
from typing import Optional
import os

class Erasure:
    def __init__(self, config_path: str = None):
        """初始化擦除模块

        Args:
            config_path: 配置文件路径
        """
        import yaml
        if config_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(module_dir, '..', 'config', 'toy_config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 从配置读取擦除参数
        erasure_config = self.config.get('erasure', {})
        self.erasure_probability = erasure_config.get('probability', 0.1)  # 默认擦除概率
        self.erasure_pattern = None  # 擦除模式

    def apply_erasure(self, bits: torch.Tensor,
                      probability: Optional[float] = None) -> torch.Tensor:
        """应用随机擦除

        Args:
            bits: 输入比特，shape [batch, n_bits]
            probability: 擦除概率，如果为None则使用默认值

        Returns:
            erased_bits: 擦除后的比特，被擦除的位置设为-1
        """
        if probability is None:
            probability = self.erasure_probability

        batch_size = bits.shape[0]
        n_bits = bits.shape[1]

        # 生成随机擦除掩码
        erasure_mask = torch.rand(batch_size, n_bits) < probability
        erasure_mask = erasure_mask.float()

        # 应用擦除：被擦除的位置设为-1
        erased_bits = bits.clone()
        erased_bits[erasure_mask.bool()] = -1

        return erased_bits

    # def apply_pattern_erasure(self, bits: torch.Tensor,
    #                           pattern: torch.Tensor) -> torch.Tensor:
    #     """应用指定模式的擦除
    #
    #     Args:
    #         bits: 输入比特，shape [batch, n_bits]
    #         pattern: 擦除模式，shape [n_bits]，1表示擦除，0表示保留
    #
    #     Returns:
    #         erased_bits: 擦除后的比特
    #     """
    #     batch_size = bits.shape[0]
    #
    #     # 扩展模式到整个batch
    #     pattern = pattern.unsqueeze(0)
    #
    #     # 应用擦除
    #     erased_bits = bits.clone()
    #     erased_bits[pattern.bool()] = -1
    #
    #     return erased_bits
    def apply_pattern_erasure(self, bits: torch.Tensor,
                              pattern: torch.Tensor) -> torch.Tensor:
        """应用指定模式的擦除

        Args:
            bits: 输入比特，shape [batch, n_bits]
            pattern: 擦除模式，shape [n_bits]，1表示擦除，0表示保留

        Returns:
            erased_bits: 擦除后的比特
        """
        # 正确获取批量大小（整数）
        # erasure.py 第 87 行
        batch_size = bits.shape[0]   # 获取批量大小
        n_bits = bits.shape[1]  # 获取比特数

        # 扩展pattern到整个batch
        # unsqueeze(0) 将 pattern 从 (n_bits,) 变为 (1, n_bits)
        pattern = pattern.unsqueeze(0)  # 形状: (1, n_bits)
        pattern = pattern.expand(batch_size, -1)
        # 应用擦除
        erased_bits = bits.clone()
        erased_bits[pattern.bool()] = -1

        return erased_bits

    def recover_erasure(self, erased_bits: torch.Tensor,
                        known_bits: torch.Tensor) -> torch.Tensor:
        """恢复擦除的比特（简化版本：用已知比特替换）

        Args:
            erased_bits: 擦除后的比特，shape [batch, n_bits]
            known_bits: 已知的原始比特，shape [batch, n_bits]

        Returns:
            recovered_bits: 恢复后的比特
        """
        # 找到被擦除的位置
        erasure_mask = (erased_bits == -1).float()

        # 用已知比特替换擦除位置
        recovered_bits = erased_bits.clone()
        recovered_bits[erasure_mask.bool()] = known_bits[erasure_mask.bool()]

        return recovered_bits

    def get_erasure_info(self) -> dict:
        """返回擦除信息"""
        return {
            'erasure_probability': self.erasure_probability,
            'erasure_pattern': self.erasure_pattern
        }