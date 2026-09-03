"""
母码接口模块。
提供母码接口、profile 分段编码、缩短和打孔操作；profile 路径使用固定稀疏校验矩阵生成校验位。
"""

import numpy as np
import torch
from typing import Dict
import yaml
import os
from typing import List, Optional, Sequence
class MotherCode:
    def __init__(self, config_path: str = None):
        """初始化母码接口
        参数：
            config_path: 配置文件路径
        """
        if config_path is None:
            # 使用相对于本文件的路径定位项目根目录下的配置文件。
            module_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(module_dir, '..', '..', 'configs', 'toy_config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)



        # 从配置读取母码参数
        mc = self.config['mother_code']
        self.k = mc['k']
        self.n = mc['n']
        self.m = self.n - self.k

        # 构建校验矩阵（使用 self.k 和 self.n）
        self._build_fixed_parity_check_matrix()

        self.K = self.config['toy']['K']
        self.max_iterations = self.config['decoder']['max_iterations']

    # def __init__(self, config_path: str = "config/toy_config.yaml"):
    # """初始化母码接口
    #        参数：
    #            config_path: 配置文件路径
    #        """
    #     import yaml
    #     with open(config_path, 'r', encoding='utf-8') as f:
    #         self.config = yaml.safe_load(f)
    #
    #     # 使用小型固定校验矩阵（Toy版本）
    #     self._build_fixed_parity_check_matrix()
    #
    #     self.K = self.config['toy']['K']  # 8个角色
    #     self.max_iterations = self.config['decoder']['max_iterations']  # 10次迭代'
    def _build_fixed_parity_check_matrix(self):
        """基于 self.k 和 self.n 构建校验矩阵"""
        n = self.n
        m = self.m  # 校验位个数 = n - k
        H = np.zeros((m, n), dtype=np.int8)
        np.random.seed(42)
        # 每列随机选3个1（保持稀疏性）
        for col in range(n):
            rows = np.random.choice(m, 3, replace=False)
            H[rows, col] = 1
        self.H = H

    def _parity_projection(self, q_bits: int, parity_bits: int) -> torch.Tensor:
        """返回用于 H=[P.T | I] 的确定性稀疏矩阵 P。"""
        if q_bits <= 0 or parity_bits < 0:
            raise ValueError("q_bits must be positive and parity_bits non-negative")
        generator = np.random.RandomState(42 + q_bits * 1009 + parity_bits)
        p = np.zeros((q_bits, parity_bits), dtype=np.int8)
        if parity_bits:
            degree = min(3, parity_bits)
            for row in range(q_bits):
                p[row, generator.choice(parity_bits, degree, replace=False)] = 1
        return torch.tensor(p, dtype=torch.float32)

    @staticmethod
    def _profile_header(profile_id: int, device: torch.device,
                        dtype: torch.dtype) -> torch.Tensor:
        """将 profile_id 编码为 32 位高位优先比特，后面补零。"""
        if not 0 <= profile_id < 2 ** 32:
            raise ValueError("profile_id must fit in 32 bits")
        bits = [(profile_id >> (31 - i)) & 1 for i in range(32)] + [0] * 96
        return torch.tensor(bits, device=device, dtype=dtype)

    def encode_profile(self, source_bits: torch.Tensor,
                       q_list: Sequence[int], n_list: Sequence[int],
                       profile_id: int = 0) -> torch.Tensor:
        """编码八个角色的载荷，并在前面添加 128 位 profile 头部。

        ``source_bits`` 按 ``[role0 q0, role1 q1, ...]`` 展平。角色 k 的码字
        为 ``[u_k, u_k @ P_k (mod 2)]``，长度恰好为 ``n_k``。由于校验部分
        由固定稀疏矩阵 P_k 生成，所得码字满足对应的校验矩阵
        ``H_k=[P_k.T | I]``。
        """
        if source_bits.ndim != 2:
            raise ValueError("source_bits must have shape [batch, sum(q_k)]")
        if len(q_list) != 8 or len(n_list) != 8:
            raise ValueError("q_list and n_list must contain 8 roles")
        q_list = [int(q) for q in q_list]
        n_list = [int(n) for n in n_list]
        if any(q <= 0 or n < q for q, n in zip(q_list, n_list)):
            raise ValueError("each role must satisfy 0 < q_k <= n_k")
        if sum(n_list) != 8064:
            raise ValueError(f"sum(n_k) must be 8064, got {sum(n_list)}")
        if source_bits.shape[1] != sum(q_list):
            raise ValueError(f"expected {sum(q_list)} source bits, got {source_bits.shape[1]}")

        encoded_roles = []
        source_offset = 0
        for q_bits, n_bits in zip(q_list, n_list):
            information = source_bits[:, source_offset:source_offset + q_bits]
            source_offset += q_bits
            parity_count = n_bits - q_bits
            P = self._parity_projection(q_bits, parity_count).to(
                device=source_bits.device, dtype=source_bits.dtype
            )
            parity = torch.remainder(information.matmul(P), 2)
            encoded_roles.append(torch.cat([information, parity], dim=1))

        header = self._profile_header(profile_id, source_bits.device, source_bits.dtype)
        header = header.unsqueeze(0).expand(source_bits.shape[0], -1)
        encoded = torch.cat([header] + encoded_roles, dim=1)
        expected_length = 128 + sum(n_list)
        if encoded.shape[1] != expected_length or expected_length != 8192:
            raise RuntimeError("profile encoding did not produce exactly 8192 bits")
        return encoded

    def encode_from_profile(self, source_bits: torch.Tensor, profile: Dict) -> torch.Tensor:
        """提供接受 ProfileGenerator 字典的便捷封装。"""
        return self.encode_profile(source_bits, profile['q'], profile['n'], profile.get('id', 0))

    def check_profile_codeword(self, encoded_bits: torch.Tensor,
                               q_list: Sequence[int], n_list: Sequence[int]) -> torch.Tensor:
        """返回每个 batch 样本的载荷校验结果（忽略头部）。"""
        if encoded_bits.ndim != 2 or encoded_bits.shape[1] != 128 + sum(n_list):
            raise ValueError("encoded_bits must include the 128-bit header")
        offset = 128
        valid = torch.ones(encoded_bits.shape[0], dtype=torch.bool, device=encoded_bits.device)
        for q_bits, n_bits in zip(q_list, n_list):
            word = encoded_bits[:, offset:offset + n_bits]
            offset += n_bits
            parity_count = n_bits - q_bits
            P = self._parity_projection(q_bits, parity_count).to(encoded_bits)
            expected = torch.remainder(word[:, :q_bits].matmul(P), 2)
            valid &= torch.all(word[:, q_bits:] == expected, dim=1)
        return valid
    # def _build_fixed_parity_check_matrix(self):
    #     """构建小型固定校验矩阵
    #
    #     使用一个简单的(32, 16) LDPC码
    #     - 码长: 32 bits
    #     - 信息位: 16 bits
    #     - 校验位: 16 bits
    #     - 码率: 1/2
    #     """
    #     # 构建一个16x32的校验矩阵（16个校验方程，32个变量节点）
    #     H = np.zeros((16, 32), dtype=np.int8)
    #
    #     # 为每列分配3个1
    #     np.random.seed(42)  # 固定种子保证可复现
    #     for col in range(32):
    #         # 选择3个不同的行
    #         rows = np.random.choice(16, 3, replace=False)
    #         H[rows, col] = 1
    #
    #     self.H = H
    #     self.n = 32  # 码长
    #     self.k = 16  # 信息位长度
    #     self.m = 16  # 校验位长度

    def _get_generator_matrix(self) -> np.ndarray:
        """从校验矩阵生成生成矩阵。

        使用系统形式的生成矩阵 G = [I | P]。

        返回：
            G：生成矩阵，形状为 (k, n)。
        """
        k = self.k
        n = self.n

        # 构建系统形式的生成矩阵
        G = np.zeros((k, n), dtype=np.int8)

        # 单位矩阵部分 (I_k)
        for i in range(k):
            G[i, i] = 1

        # 校验部分（P）使用固定种子的随机矩阵。
        np.random.seed(123)  # 固定种子保证可复现
        for i in range(k):
            for j in range(k, n):
                G[i, j] = np.random.randint(0, 2)

        return G

    # def encode(self, source_bits: torch.Tensor,
    #            n_total: int, q_bits: int) -> torch.Tensor:
    #     """执行 LDPC 编码（包含缩短和打孔）
    #
    #     编码流程：
    #     1. 缩短：将 q_bits 填充到 k 位
    #     2. LDPC编码：生成n位码字
    #     3. 打孔：从 n 位中取出 n_total 位
    #
    #     参数：
    #         source_bits: 源比特，shape [batch, q_bits]
    #         n_total: 期望输出的总比特数
    #         q_bits: 源比特数（信息位）
    #
    #     返回：
    #         encoded_bits: 编码后的比特，shape [batch, n_total]
    #     """
    #     # 修正：使用shape[0]获取batch_size（整数）
    #     # source_bits.shape 返回 torch.Size([batch, q_bits])
    #     # source_bits.shape[0] 返回 batch（整数）
    #     batch_size = source_bits.shape[0]  # 获取第一个维度（批次大小）
    #     k = self.k  # 母码信息位长度
    #     n = self.n  # 母码码长
    #
    #     # 确保source_bits是2维张量
    #     assert source_bits.dim() == 2, f"source_bits应为2维张量，实际为{source_bits.dim()}维"
    #     assert source_bits.shape[1] == q_bits, f"source_bits的第二维应为{q_bits}，实际为{source_bits.shape[1]}"
    #
    #     # 1. 缩短操作：将q_bits填充到k位
    #     # 高有效位放源比特，低有效位填充0
    #     shortened_bits = torch.zeros((batch_size, k), dtype=source_bits.dtype)
    #     q_bits_actual = min(q_bits, k)
    #     shortened_bits[:, :q_bits_actual] = source_bits[:, :q_bits_actual]
    #
    #     # 2. 执行LDPC编码
    #     G = self._get_generator_matrix()
    #     G_tensor = torch.tensor(G, dtype=source_bits.dtype)
    #
    #     # 矩阵乘法：c = u * G (mod 2)
    #     encoded_full = torch.matmul(shortened_bits, G_tensor) % 2
    #
    #     # 3. 打孔操作：从n位中取出n_total位
    #     if n_total <= n:
    #         # 从高有效位开始取
    #         encoded_bits = encoded_full[:, :n_total]
    #     else:
    #         # 如果需要更多位，重复编码
    #         repeats = (n_total + n - 1) // n
    #         encoded_full = encoded_full.repeat(1, repeats)
    #         encoded_bits = encoded_full[:, :n_total]
    #
    #     return encoded_bits
    #
    # def decode(self, received_bits: torch.Tensor,
    #            n_total: int, q_bits: int) -> torch.Tensor:
    #     """执行LDPC解码（包含解打孔和解缩短）
    #
    #     解码流程：
    #     1. 解打孔：将 n_total 位填充回 n 位
    #     2. 使用生成矩阵进行解码
    #     3. 解缩短：从 k 位中取出 q_bits 位
    #
    #     参数：
    #         received_bits: 接收到的比特，shape [batch, n_total]
    #         n_total: 接收到的总比特数
    #         q_bits: 期望的源比特数
    #
    #     返回：
    #         decoded_bits: 解码后的比特，shape [batch, q_bits]
    #     """
    #     # 修正：使用shape[0]获取batch_size（整数）
    #     # received_bits.shape 返回 torch.Size([batch, n_total])
    #     # received_bits.shape[0] 返回 batch（整数）
    #     batch_size = received_bits.shape[0]  # 获取第一个维度（批次大小）
    #     k = self.k
    #     n = self.n
    #
    #     # 确保received_bits是2维张量
    #     assert received_bits.dim() == 2, f"received_bits应为2维张量，实际为{received_bits.dim()}维"
    #     assert received_bits.shape[1] == n_total, f"received_bits的第二维应为{n_total}，实际为{received_bits.shape[1]}"
    #
    #     # 1. 解打孔：将n_total位填充回n位
    #     if n_total < n:
    #         # 缺失的位置用0填充
    #         padded_bits = torch.zeros((batch_size, n), dtype=received_bits.dtype)
    #         padded_bits[:, :n_total] = received_bits
    #     else:
    #         padded_bits = received_bits[:, :n]
    #
    #     # 2. 使用生成矩阵进行解码
    #     G = self._get_generator_matrix()
    #     G_tensor = torch.tensor(G, dtype=received_bits.dtype)
    #
    #     # 对于系统形式的编码，前k位就是信息位
    #     decoded_full = padded_bits[:, :k]
    #
    #     # 3. 解缩短：从k位中取出q_bits位
    #     q_bits_actual = min(q_bits, k)
    #     decoded_bits = decoded_full[:, :q_bits_actual]
    #
    #     return decoded_bits

    def encode(self, source_bits: torch.Tensor,
               n_total, q_bits, q_list: Optional[Sequence[int]] = None,
               n_list: Optional[Sequence[int]] = None,
               profile_id: int = 0) -> torch.Tensor:
        """
        执行编码，支持旧版标量接口或按角色 profile 编码。

        当提供 q_list/n_list（或 n_total、q_bits 为序列）时，返回值包含
        128 位头部，长度严格为
               ``128 + sum(n_k)`` 位。标量形式保留用于兼容旧调用方。
        参数：
            source_bits: [batch, q_bits]
            n_total: 总码长
            q_bits: 信息位长度
        返回：
            encoded: [batch, n_total]
        """
        if q_list is not None or n_list is not None or isinstance(n_total, (list, tuple)):
            actual_n = n_list if n_list is not None else n_total
            actual_q = q_list if q_list is not None else q_bits
            return self.encode_profile(source_bits, actual_q, actual_n, profile_id)

        batch_size = source_bits.shape[0]
        encoded = torch.zeros(batch_size, n_total, dtype=source_bits.dtype, device=source_bits.device)
        # 将信息位直接放在输出的最前面。
        encoded[:, :q_bits] = source_bits
        return encoded

    def decode(self, received_bits: torch.Tensor,
               n_total: int, q_bits: int) -> torch.Tensor:
        """
        Toy 版本：解码，直接截取前 q_bits 作为信息位。
        参数：
            received_bits: [batch, n_total]
            n_total: 总码长（仅用于接口统一）
            q_bits: 信息位长度
        返回：
            decoded: [batch, q_bits]
        """
        return received_bits[:, :q_bits]

    def apply_source_mask(self, bits: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        """应用源掩码

        参数：
            bits: 输入比特，shape [batch, K, max_n]
            mask：源掩码，shape [batch, K, max_n]

        返回：
            masked_bits：应用掩码后的比特，shape [batch, K, max_n]
        """
        return bits * mask

    def get_matrix_info(self) -> Dict:
        """返回校验矩阵信息"""
        return {
            'n': self.n,
            'k': self.k,
            'm': self.m,
            'H_shape': self.H.shape,
            'H_density': float(np.mean(self.H))
        }

    def shortening(self, encoded_bits: torch.Tensor, n_shorten: int) -> torch.Tensor:
        """缩短接口：减少源比特。

        将编码后的比特中，高有效位的比特设置为0（表示这些位置不发送信息）

        参数：
            encoded_bits: 编码后的比特，shape [batch, n]
            n_shorten: 要缩短的比特数

        返回：
            shortened_bits: 缩短后的比特，shape [batch, n]
        """
        # 高有效位向低有效位扩展，将前n_shorten位设为0
        shortened_bits = encoded_bits.clone()
        shortened_bits[:, :n_shorten] = 0
        return shortened_bits

    def puncturing(self, encoded_bits: torch.Tensor, n_puncture: int) -> torch.Tensor:
        """打孔接口：减少发送保护位。

        从编码后的比特中移除指定数量的比特（不发送这些位置）

        参数：
            encoded_bits: 编码后的比特，shape [batch, n]
            n_puncture: 要移除的比特数

        返回：
            punctured_bits: 穿刺后的比特，shape [batch, n - n_puncture]
        """
        # 移除最后n_puncture个比特（保护位）
        return encoded_bits[:, :-n_puncture]

    def shortening(self, encoded_bits: torch.Tensor, n_shorten: int) -> torch.Tensor:
        """
        减少source bits，高有效位向低有效位扩展

            参数：
            encoded_bits: 编码后的比特，shape [batch, n]
            n_shorten: 要缩短的比特数

            返回：
            shortened_bits: 缩短后的比特，shape [batch, n]
        """
        shortened_bits = encoded_bits.clone()
        # 将前n_shorten位设为0，表示这些位置不发送信息
        shortened_bits[:, :n_shorten] = 0
        return shortened_bits

    def puncturing(self, encoded_bits: torch.Tensor, n_puncture: int) -> torch.Tensor:
        """
        减少发送保护位，从编码后的比特中移除指定数量的比特

            参数：
            encoded_bits: 编码后的比特，shape [batch, n]
            n_puncture: 要移除的比特数

            返回：
            punctured_bits: 穿刺后的比特，shape [batch, n - n_puncture]
        """
        # 移除最后n_puncture个比特（保护位）
        return encoded_bits[:, :-n_puncture]
