"""
Quantizer模块
实现向量量化，将高维表示压缩为codec vector
"""

import torch
import numpy as np
from typing import Optional


class Quantizer:
    def __init__(self, config_path: str = "config/toy_config.yaml"):
        """初始化量化器

        Args:
            config_path: 配置文件路径
        """
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 从配置读取参数
        model_config = self.config.get('model', {})
        self.d_model = model_config.get('d_model', 128)  # 输入维度
        self.codec_dim = model_config.get('codec_dim', 32)  # 压缩后维度
        self.codebook_size = model_config.get('codebook_size', 256)  # 码本大小

    def quantize(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """量化高维表示到codec vector

        Args:
            input_tensor: 输入tensor，shape [batch, K, d_model]

        Returns:
            codec_vector: 压缩后的向量，shape [batch, K, codec_dim]
        """
        batch_size = input_tensor.shape
        K = input_tensor.shape

        # 使用线性层压缩维度（简化版本）
        # 实际应用中应该使用可学习的码本或VQ-VAE
        projection = torch.nn.Linear(self.d_model, self.codec_dim)
        codec_vector = projection(input_tensor)  # [batch, K, codec_dim]

        return codec_vector

    def dequantize(self, codec_vector: torch.Tensor) -> torch.Tensor:
        """反量化：从codec vector恢复高维表示

        Args:
            codec_vector: 压缩后的向量，shape [batch, K, codec_dim]

        Returns:
            reconstructed: 重构的表示，shape [batch, K, d_model]
        """
        batch_size = codec_vector.shape
        K = codec_vector.shape

        # 使用线性层解压缩
        projection = torch.nn.Linear(self.codec_dim, self.d_model)
        reconstructed = projection(codec_vector)  # [batch, K, d_model]

        return reconstructed

    def get_codebook(self) -> torch.Tensor:
        """获取码本（简化版本返回随机码本）"""
        return torch.randn(self.codebook_size, self.codec_dim)