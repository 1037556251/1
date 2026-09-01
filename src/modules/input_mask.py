"""
输入处理与掩码生成模块
负责生成随机tensor和source mask
"""

import torch
import yaml
from typing import Optional, List, Dict
import os

class InputMask:
    def __init__(self, config_path: str = None):
        """初始化InputMask模块

        Args:
            config_path: 配置文件路径
        """
        '''with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)'''
        if config_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(module_dir, '..', '..', 'configs', 'toy_config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.M = self.config['toy']['M']
        self.K = self.config['toy']['K']
        self.d_x = self.config['toy']['d_x']
        self.batch = self.config['toy']['batch']

    def generate_random_tensor(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """生成随机输入 tensor

        Args:
            batch_size: 可选的batch大小，默认使用配置文件中的值

        Returns:
            tensor: shape [batch_size, M, d_x]
        """
        if batch_size is None:
            batch_size = self.batch

        return torch.randn(batch_size, self.M, self.d_x)

    def generate_source_mask(self,
                             profile: Dict,
                             batch_size: Optional[int] = None) -> torch.Tensor:
        """根据profile生成source mask

        Source mask规则：
        - 高有效位向低有效位扩展
        - 1表示源比特位置，0表示缩短填充位置

        Args:
            profile: profile字典，包含q和n列表
            batch_size: batch大小

        Returns:
            mask: shape [batch_size, K, max_n]，其中max_n是profile中最大的n_k
        """
        if batch_size is None:
            batch_size = self.batch

        q_list = profile['q']
        n_list = profile['n']
        max_n = max(n_list)

        # 创建mask
        mask = torch.zeros(batch_size, self.K, max_n)

        for k in range(self.K):
            n_k = n_list[k]
            q_k = q_list[k]

            # 高有效位（前q_k位）是源比特，设为1
            mask[:, k, :q_k] = 1

            # 低有效位（从q_k到n_k-1）是缩短填充，保持0
            # 注意：超过n_k的部分也保持0，这是puncturing的效果

        return mask

    def get_config(self) -> Dict:
        """返回当前配置"""
        return self.config
