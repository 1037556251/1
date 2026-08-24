"""
Group/Query表示模块
实现Input/Mask到Group或Query表示的转换
"""

import torch
import numpy as np
from typing import Optional, Tuple


class GroupQuery:
    def __init__(self, config_path: str = "config/toy_config.yaml"):
        """初始化Group/Query表示模块

        Args:
            config_path: 配置文件路径
        """
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 从配置读取参数
        model_config = self.config.get('model', {})
        self.d_model = model_config.get('d_model', 128)  # 投影维度
        self.d_x = model_config.get('d_x', 64)  # 输入维度
        self.M = model_config.get('M', 128)  # 最大角色数
        self.K = model_config.get('K', 8)  # 实际角色数

    def input_to_group(self, input_tensor: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """将输入tensor转换为Group表示

        Args:
            input_tensor: 输入tensor，shape [batch, M, d_x]
            mask: 可选掩码，shape [batch, M]

        Returns:
            group_representation: Group表示，shape [batch, K, d_model]
        """
        batch_size = input_tensor.shape

        # 投影到d_model维度
        # 使用简单的线性投影（实际应用中应该用可学习的投影层）
        projection = torch.nn.Linear(self.d_x, self.d_model)
        projected = projection(input_tensor)  # [batch, M, d_model]

        # 应用掩码（如果提供）
        if mask is not None:
            mask = mask.unsqueeze(-1)  # [batch, M, 1]
            projected = projected * mask

        # 聚合到K个角色（简化：取前K个或平均池化）
        if self.M >= self.K:
            group_representation = projected[:, :self.K, :]  # 取前K个
        else:
            # 如果M小于K，重复填充
            repeats = (self.K + self.M - 1) // self.M
            group_representation = projected.repeat(1, repeats, 1)[:, :self.K, :]

        return group_representation

    def group_to_query(self, group_representation: torch.Tensor,
                       query_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """将Group表示转换为Query表示

        Args:
            group_representation: Group表示，shape [batch, K, d_model]
            query_indices: 查询索引，shape [batch, n_queries]

        Returns:
            query_representation: Query表示，shape [batch, n_queries, d_model]
        """
        if query_indices is None:
            # 默认使用所有角色作为查询
            return group_representation

        batch_size = group_representation.shape

        # 根据索引选择查询
        query_representation = torch.zeros(batch_size, query_indices.shape, self.d_model)
        for b in range(batch_size):
            for q_idx, k_idx in enumerate(query_indices[b]):
                query_representation[b, q_idx] = group_representation[b, k_idx]

        return query_representation

    def apply_permutation(self, group_representation: torch.Tensor,
                          permutation: torch.Tensor) -> torch.Tensor:
        """应用角色置换

        Args:
            group_representation: Group表示，shape [batch, K, d_model]
            permutation: 置换索引，shape [K]

        Returns:
            permuted_representation: 置换后的表示，shape [batch, K, d_model]
        """
        return group_representation[:, permutation, :]