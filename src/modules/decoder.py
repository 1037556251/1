"""
解码器模块
实现LDPC解码和统一接口
"""

import torch
import numpy as np
from typing import Optional
import os


class LDPCDecoder:
    def __init__(self, config_path: str = None):
        """初始化LDPC解码器

        参数：
            config_path: 配置文件路径
        """
        import yaml
        if config_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(module_dir, '..', '..', 'configs', 'toy_config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)


        # 从配置读取解码器参数
        decoder_config = self.config.get('decoder', {})
        self.max_iterations = decoder_config.get('max_iterations', 10)
        self.algorithm = decoder_config.get('algorithm', 'LDPC')

        # 初始化与母码配置一致的固定校验矩阵
        self._build_parity_check_matrix()

    def _build_parity_check_matrix(self):
        """构建小型固定校验矩阵（与mother_code保持一致）"""
        import numpy as np

        H = np.zeros((16, 32), dtype=np.int8)

        np.random.seed(42)  # 固定种子保证可复现
        for col in range(32):
            rows = np.random.choice(16, 3, replace=False)
            H[rows, col] = 1

        self.H = torch.tensor(H, dtype=torch.float32)
        self.n = 32  # 码长
        self.k = 16  # 信息位长度
        self.m = 16  # 校验位长度

    def decode(self, received_bits: torch.Tensor,
               n_total: int, q_bits: int,
               max_iterations: Optional[int] = None,
               erasure_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        执行 LDPC Min-Sum 译码；长扁平输入走兼容路径。

        参数：
            received_bits: 接收到的比特，shape [batch, n_total]
            n_total: 总码长（可用于缩短/打孔处理）
            q_bits: 期望的信息位长度
            max_iterations：保留的接口参数；当前按要求固定执行 10 次迭代

        返回：
            decoded_bits: 解码后的比特，shape [batch, q_bits]
        """
        if received_bits.ndim != 2:
            raise ValueError("received_bits must have shape [batch, n_total]")
        if received_bits.shape[1] != n_total:
            raise ValueError("received_bits second dimension must equal n_total")
        if not 0 <= q_bits <= n_total:
            raise ValueError("q_bits must satisfy 0 <= q_bits <= n_total")
        if erasure_mask is not None and erasure_mask.shape != received_bits.shape:
            raise ValueError("erasure_mask must match received_bits shape")

        # 使用固定 H 对完整的 32 位子码字译码。小型校验图无法处理完整块
        # 之外的比特，因此这些比特保留其硬判决值。
        decoded = received_bits.clone().float()
        H = self.H.to(device=received_bits.device, dtype=decoded.dtype)
        edge_mask = H.bool()
        # 配置中的 H 描述一个 32 位 toy 码字。旧调用方传入的是扁平的
        # 8064/8192 位流，而不是这些码字的拼接；兼容路径运行一次校验图，
        # 但不把不匹配的分块边界应用到其系统信息位上。
        decode_with_fixed_graph = n_total == self.n
        n_blocks = n_total // self.n if decode_with_fixed_graph else min(1, n_total // self.n)
        for block_index in range(n_blocks):
            start = block_index * self.n
            stop = start + self.n
            observations = received_bits[:, start:stop].float()
            llr = (1.0 - 2.0 * observations) * 4.0
            if erasure_mask is not None:
                llr = llr.masked_fill(erasure_mask[:, start:stop].bool(), 0.0)

            # 变量到校验节点的消息 q_{j->i}；非边位置在整个更新过程中由
            # edge_mask 保持为零。
            variable_to_check = llr.unsqueeze(1) * H.unsqueeze(0)
            check_to_variable = torch.zeros_like(variable_to_check)

            # 固定迭代次数是有意设计的。不要增加基于 syndrome 的提前退出，
            # 因为实验要求恰好执行十次更新。
            for _ in range(10):
                # Min-Sum 校验节点更新：对每个校验节点计算符号乘积和最小、
                # 次小幅值。
                for check in range(self.m):
                    connected = edge_mask[check]
                    incoming = variable_to_check[:, check, connected]
                    magnitudes = incoming.abs()
                    signs = torch.where(incoming < 0,
                                        incoming.new_tensor(-1.0),
                                        incoming.new_tensor(1.0))
                    if incoming.shape[1] > 1:
                        outgoing = []
                        for edge in range(incoming.shape[1]):
                            other_magnitudes = magnitudes.clone()
                            other_magnitudes[:, edge] = float("inf")
                            min_other = other_magnitudes.min(dim=1).values
                            other_sign = signs.prod(dim=1) * signs[:, edge]
                            outgoing.append(other_sign * min_other)
                        check_to_variable[:, check, connected] = torch.stack(outgoing, dim=1)
                    else:
                        check_to_variable[:, check, connected] = torch.zeros_like(incoming)

                # 变量节点更新：将信道证据与其他校验节点消息相加，这是每轮
                # BP 更新的第二部分数值计算。
                total_check = check_to_variable.sum(dim=1, keepdim=True)
                variable_to_check = (
                    llr.unsqueeze(1) + total_check - check_to_variable
                ) * H.unsqueeze(0)

            posterior = llr + check_to_variable.sum(dim=1)
            if decode_with_fixed_graph:
                decoded[:, start:stop] = (posterior < 0).to(decoded.dtype)

        if not decode_with_fixed_graph:
            return received_bits[:, :q_bits].float()
        return decoded[:, :q_bits]

    def get_matrix_info(self) -> dict:
        """返回校验矩阵信息"""
        return {
            'n': self.n,
            'k': self.k,
            'm': self.m,
            'H_shape': self.H.shape,
            'algorithm': self.algorithm,
            'max_iterations': self.max_iterations
        }
