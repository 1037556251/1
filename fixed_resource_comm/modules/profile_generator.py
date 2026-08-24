"""
Profile生成器模块
负责生成13个确定的资源分配方案（Profile）
"""

import hashlib
import numpy as np
from typing import List, Dict, Tuple
import os

class ProfileGenerator:
    def __init__(self, config_path: str = None):
        """初始化Profile生成器

        Args:
            config_path: 配置文件路径
        """
        import yaml
        if config_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(module_dir, '..', 'config', 'toy_config.yaml')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)




        self.K = self.config['toy']['K']  # 8个角色
        self.payload_bits = self.config['profile']['payload_bits']  # 8064
        self.profiles = self._generate_all_profiles()

    def _generate_mask(self, profile_id: int, profile_type: str,
                       q_list: List[int], n_list: List[int]) -> List[int]:
        """生成确定性的mask（128 bits）

        mask的生成规则：基于profile_id和hash的确定性算法
        """
        profile_str = f"{profile_id}_{profile_type}_{str(q_list)}_{str(n_list)}"
        hash_bytes = hashlib.sha256(profile_str.encode()).digest()

        # 取前128 bits作为mask
        mask = []
        for i in range(128):
            byte_idx = i // 8
            bit_idx = i % 8
            bit = (hash_bytes[byte_idx] >> bit_idx) & 1
            mask.append(bit)

        return mask
    def _get_donor_indices_for_single(self, target_role: int) -> List[int]:
        """为single-emphasis profile选择4个donor角色

        选择规则：从target_role之后按顺序选择4个角色
        """
        all_roles = list(range(self.K))
        all_roles.remove(target_role)
        # 选择离target_role最近的4个角色（按索引顺序）
        # 这里简单选择前4个
        return all_roles[:4]

    def _get_donor_indices_for_pair(self, target1: int, target2: int) -> List[int]:
        """为paired-emphasis profile选择4个donor角色"""
        all_roles = list(range(self.K))
        all_roles.remove(target1)
        all_roles.remove(target2)
        return all_roles[:4]

    def _validate_profile(self, profile_id: int, profile_type: str,
                         q_list: List[int], n_list: List[int]) -> Dict:
        """验证profile的约束条件

        验证内容：
        1. sum(n_k) == 8064
        2. q_k <= n_k 对所有k成立
        3. 生成确定性的profile ID和hash

        Args:
            profile_id: profile的ID编号
            profile_type: profile类型（uniform/single/paired）
            q_list: 每个角色的源比特数列表
            n_list: 每个角色的总比特数列表

        Returns:
            包含验证通过后的profile信息的字典
        """
        # 验证约束1：sum(n_k) == 8064
        total_n = sum(n_list)
        assert total_n == self.payload_bits, \
            f"Profile {profile_id} (type={profile_type}): sum(n)={total_n} != {self.payload_bits}"

        header_bits = 128
        N0 = total_n + header_bits
        assert N0 == 8192, f"Profile {profile_id}: N0={N0} != 8192"

        # 验证约束2：q_k <= n_k
        for k in range(self.K):
            assert q_list[k] <= n_list[k], \
                f"Profile {profile_id}: q[{k}]={q_list[k]} > n[{k}]={n_list[k]}"

        # 生成确定性的hash
        # 生成确定性的hash
        profile_str = f"{profile_id}_{profile_type}_{str(q_list)}_{str(n_list)}"
        profile_hash = hashlib.sha256(profile_str.encode()).hexdigest()[:16]

        # 生成确定性的mask（128 bits）
        mask = self._generate_mask(profile_id, profile_type, q_list, n_list)

        return {
            'id': profile_id,
            'type': profile_type,
            'q': q_list,
            'n': n_list,
            'hash': profile_hash,
            'mask': mask,  # 新增：128 bits的mask
            'validated': True,
            'n_total': total_n,  # sum(n_k) = 8064
            'N0': N0,  # N0 = 8064 + 128 = 8192
            'header': 128  # header固定128 bits
        }

    def _generate_all_profiles(self) -> List[Dict]:
        """生成全部13个profile

        Profile分布：
        - ID=0: Uniform profile（1个）
        - ID=1-8: Single-emphasis profiles（8个）
        - ID=9-12: Paired-emphasis profiles（4个）
        """
        profiles = []

        # 1. Uniform profile (ID=0)
        q = [192] * self.K
        n = [1008] * self.K
        profile0 = self._validate_profile(0, 'uniform', q, n)
        profiles.append(profile0)
        print(f"[Profile 0] Uniform profile: q={q[0]}, n={n[0]}, hash={profile0['hash']}")

        # 2. 8个Single-emphasis profiles (ID=1..8)
        for target_role in range(self.K):
            q = [192] * self.K
            n = [1008] * self.K

            # 目标角色：q=256, n=1264
            q[target_role] = 256
            n[target_role] = 1264

            # 4个donor角色：q=160, n=944
            donor_indices = self._get_donor_indices_for_single(target_role)
            for idx in donor_indices:
                q[idx] = 160
                n[idx] = 944

            # 其他角色保持uniform (q=192, n=1008)
            # 已经初始化为这个值，所以不需要额外操作

            profile_id = target_role + 1
            profile = self._validate_profile(profile_id, 'single_emphasis', q, n)
            profiles.append(profile)
            print(f"[Profile {profile_id}] Single-emphasis (target={target_role}): hash={profile['hash']}")

        # 3. 4个Paired-emphasis profiles (ID=9..12)
        # 配对方案：(0,1), (2,3), (4,5), (6,7)
        pair_targets = [(0, 1), (2, 3), (4, 5), (6, 7)]

        for pair_id, (t1, t2) in enumerate(pair_targets):
            q = [192] * self.K
            n = [1008] * self.K

            # 两个目标角色：q=224, n=1136
            q[t1] = 224
            q[t2] = 224
            n[t1] = 1136
            n[t2] = 1136

            # 4个donor角色：q=160, n=944
            donor_indices = self._get_donor_indices_for_pair(t1, t2)
            for idx in donor_indices:
                q[idx] = 160
                n[idx] = 944

            # 其他角色保持uniform (q=192, n=1008)

            profile_id = pair_id + 9
            profile = self._validate_profile(profile_id, 'paired_emphasis', q, n)
            profiles.append(profile)
            print(f"[Profile {profile_id}] Paired-emphasis (targets={t1},{t2}): hash={profile['hash']}")

        # 最终验证：确保生成了13个profile
        assert len(profiles) == 13, f"Expected 13 profiles, but got {len(profiles)}"

        return profiles

    def get_profile(self, profile_id: int) -> Dict:
        """根据ID获取profile

        Args:
            profile_id: profile的ID（0-12）

        Returns:
            profile字典
        """
        assert 0 <= profile_id < len(self.profiles), \
            f"Profile ID {profile_id} out of range (0-{len(self.profiles)-1})"
        return self.profiles[profile_id]

    def get_all_profiles(self) -> List[Dict]:
        """获取所有profile"""
        return self.profiles

    def print_profile_summary(self, profile_id: int):
        """打印profile的详细信息"""
        profile = self.get_profile(profile_id)
        print(f"\nProfile {profile['id']} ({profile['type']}):")
        print(f"  Hash: {profile['hash']}")
        print(f"  q values: {profile['q']}")
        print(f"  n values: {profile['n']}")
        print(f"  sum(n): {sum(profile['n'])}")
        print(f"  All q <= n: {all(q <= n for q, n in zip(profile['q'], profile['n']))}")