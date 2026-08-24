"""
接收器模块
整合解码、擦除恢复和错误检测功能
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict
from .decoder import LDPCDecoder
from .erasure import Erasure
import os

class Receiver:
    def __init__(self, config_path: str = None):
        """初始化接收器

        Args:
            config_path: 配置文件路径
        """
        import yaml
        if config_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(module_dir, '..', 'config', 'toy_config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 初始化子模块
        self.decoder = LDPCDecoder(config_path)
        self.erasure = Erasure(config_path)

        # 从配置读取接收器参数
        receiver_config = self.config.get('receiver', {})
        self.enable_erasure_recovery = receiver_config.get('enable_erasure_recovery', True)
        self.enable_error_detection = receiver_config.get('enable_error_detection', True)

    def receive(self, received_bits: torch.Tensor,
                n_total: int, q_bits: int,
                use_erasure: bool = False,
                erasure_pattern: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """完整接收流程

        Args:
            received_bits: 接收到的比特，shape [batch, n_total]
            n_total: 总码长
            q_bits: 期望的信息位数
            use_erasure: 是否使用擦除恢复
            erasure_pattern: 擦除模式（可选）

        Returns:
            decoded_bits: 解码后的比特，shape [batch, q_bits]
            stats: 接收统计信息
        """
        stats = {}
        batch_size = received_bits.shape[0]  # 修正：取第一个维度

        # 1. 解码
        decoded_bits = self.decoder.decode(received_bits, n_total, q_bits)
        stats['decoded_shape'] = decoded_bits.shape

        # 2. 检测解码失败：如果解码结果全为0或全为1，视为解码失败
        decoding_failed = self._check_decoding_failure(decoded_bits, received_bits)
        stats['decoding_failed'] = decoding_failed

        if decoding_failed:
            # 统一走erasure函数处理解码失败
            erased_bits = self.erasure.apply_erasure(decoded_bits)
            stats['erasure_applied'] = True
            stats['erasure_mode'] = 'random_erasure'

            # 尝试恢复擦除
            if self.enable_erasure_recovery:
                decoded_bits = self.erasure.recover_erasure(erased_bits, decoded_bits)
                stats['recovery_applied'] = True
            else:
                stats['recovery_applied'] = False
        else:
            stats['erasure_applied'] = False

        # 3. 擦除恢复（如果需要，且未被上面的解码失败处理覆盖）
        if use_erasure and erasure_pattern is not None and not decoding_failed:
            erased_bits = self.erasure.apply_pattern_erasure(decoded_bits, erasure_pattern)
            stats['erasure_count'] = (erased_bits == -1).sum().item()

            # 尝试恢复擦除
            if self.enable_erasure_recovery:
                decoded_bits = self.erasure.recover_erasure(erased_bits, decoded_bits)
                stats['recovery_applied'] = True
            else:
                stats['recovery_applied'] = False

        # 4. 错误检测（如果需要）
        if self.enable_error_detection:
            # 简单的奇偶校验检测
            parity = decoded_bits.sum(dim=-1) % 2
            stats['parity_errors'] = (parity != 0).sum().item()

        # 5. 截取到q_bits
        if decoded_bits.shape[1] > q_bits:
            decoded_bits = decoded_bits[:, :q_bits]

        stats['final_shape'] = decoded_bits.shape

        return decoded_bits, stats

    def _check_decoding_failure(self, decoded_bits: torch.Tensor, received_bits: torch.Tensor) -> bool:
        """
        检测解码是否失败

        解码失败的标准:
        1. 解码结果全为0
        2. 解码结果全为1
        3. 解码结果与接收比特无相关性（差值过大）

        Args:
            decoded_bits: 解码后的比特
            received_bits: 接收到的原始比特

        Returns:
            bool: 是否解码失败
        """
        # 基本维度检查
        assert decoded_bits.dim() == 2 and received_bits.dim() == 2, "Expected 2D tensors"
        assert decoded_bits.shape[0] == received_bits.shape[0], "Batch sizes must match"

        batch_size = decoded_bits.shape[0]  # ✅ 修正

        # 标准1: 全0或全1（任一样本满足即失败）
        if torch.any(torch.all(decoded_bits == 0, dim=1)) or torch.any(torch.all(decoded_bits == 1, dim=1)):
            return True

        # 标准2: 差异比例 > 50%
        min_len = min(decoded_bits.shape[1], received_bits.shape[1] ) # ✅ 移除 @ref
        diff_ratio = (decoded_bits[:, :min_len] != received_bits[:, :min_len]).float().mean()
        if diff_ratio > 0.5:
            return True

        # 标准3: 相关性接近0（可选，取消注释启用）
        # x = decoded_bits[:, :min_len].flatten().float()
        # y = received_bits[:, :min_len].flatten().float()
        # denominator = torch.std(x) * torch.std(y)
        # corr = torch.corrcoef(torch.stack([x, y]))[0, 1] if denominator != 0 else 0.0
        # if torch.abs(corr) < 0.1:
        #     return True

        return False

    def get_receiver_info(self) -> Dict:
        """返回接收器信息"""
        return {
            'enable_erasure_recovery': self.enable_erasure_recovery,
            'enable_error_detection': self.enable_error_detection,
            'decoder_info': self.decoder.get_matrix_info(),
            'erasure_info': self.erasure.get_erasure_info()
        }