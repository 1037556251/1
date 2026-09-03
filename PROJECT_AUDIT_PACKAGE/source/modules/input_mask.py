"""
输入处理与掩码生成模块
负责生成随机张量和源掩码
"""

import torch
import yaml
from typing import Optional, List, Dict
import os

class InputMask:
    def __init__(self, config_path: str = None):
        """初始化InputMask模块

        参数：
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

        参数：
            batch_size: 可选的batch大小，默认使用配置文件中的值

        返回：
            tensor: shape [batch_size, M, d_x]
        """
        if batch_size is None:
            batch_size = self.batch

        return torch.randn(batch_size, self.M, self.d_x)

    def generate_source_mask(self,
                             profile: Dict,
                             batch_size: Optional[int] = None) -> torch.Tensor:
        """根据 profile 生成源掩码

        源掩码规则：
        - 高有效位向低有效位扩展
        - 1 表示源比特位置，0 表示缩短填充位置

        参数：
            profile：profile 字典，包含 q 和 n 列表
            batch_size：批次大小

        返回：
            mask: shape [batch_size, K, max_n]，其中max_n是profile中最大的n_k
        """
        if batch_size is None:
            batch_size = self.batch

        q_list = profile['q']
        n_list = profile['n']
        max_n = max(n_list)

        # 创建掩码
        mask = torch.zeros(batch_size, self.K, max_n)

        for k in range(self.K):
            n_k = n_list[k]
            q_k = q_list[k]

            # 高有效位（前q_k位）是源比特，设为1
            mask[:, k, :q_k] = 1

            # 低有效位（从q_k到n_k-1）是缩短填充，保持0
            # 注意：超过 n_k 的部分也保持 0，这是打孔效果

        return mask

    def get_config(self) -> Dict:
        """返回当前配置"""
        return self.config
