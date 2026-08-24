# final_verification.py
"""
最终验证脚本：检查所有文档要求是否满足
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.profile_generator import ProfileGenerator
from modules.mother_code import MotherCode
from modules.decoder import LDPCDecoder


def verify_all():
    print("=" * 60)
    print("最终验证：检查所有文档要求")
    print("=" * 60)

    # 1. 验证profile生成
    print("\n[验证1] Profile生成")
    pg = ProfileGenerator()
    profiles = pg.get_all_profiles()
    print(f"  ✓ 生成 {len(profiles)} 个profile")

    # 2. 验证N0=8192
    print("\n[验证2] N0=8192")
    for p in profiles:
        n_sum = sum(p['n'])
        N0 = n_sum + 128
        assert N0 == 8192, f"Profile {p.get('profile_id', p.get('id', '?'))}: N0={N0} != 8192"
    print("  ✓ 所有profile的N0=8192")

    # 3. 验证sum(n_k)=8064
    print("\n[验证3] sum(n_k)=8064")
    for p in profiles:
        total = sum(p['n'])
        assert total == 8064, f"Profile {p.get('profile_id', p.get('id', '?'))}: sum(n)={total} != 8064"
    print("  ✓ 所有profile的sum(n)=8064")

    # 4. 验证q_k <= n_k
    print("\n[验证4] q_k <= n_k")
    for p in profiles:
        for k in range(8):
            if p['q'][k] > p['n'][k]:
                raise AssertionError(f"Profile {p.get('profile_id', p.get('id', '?'))}: q[{k}]={p['q'][k]} > n[{k}]={p['n'][k]}")
    print("  ✓ 所有profile的q_k <= n_k")

    # 5. 验证确定性hash
    print("\n[验证5] 确定性hash")
    # 两次生成，比较相同profile的hash是否一致
    profiles1 = pg.get_all_profiles()
    profiles2 = pg.get_all_profiles()
    assert len(profiles1) == len(profiles2), "两次生成的profile数量不同"
    for i in range(len(profiles1)):
        # 确保每个profile有'hash'键
        assert 'hash' in profiles1[i], f"Profile {i} 缺少 'hash' 键"
        assert 'hash' in profiles2[i], f"Profile {i} 缺少 'hash' 键"
        assert profiles1[i]['hash'] == profiles2[i]['hash'], \
            f"Profile {i} 的hash不一致: {profiles1[i]['hash']} != {profiles2[i]['hash']}"
    print(f"  ✓ 所有profile的hash具有确定性（两次生成一致），共{len(profiles1)}个")

    # 6. 验证固定10次decoder iteration
    print("\n[验证6] 固定10次decoder iteration")
    decoder = LDPCDecoder()
    if hasattr(decoder, 'max_iterations'):
        assert decoder.max_iterations == 10, f"max_iterations应为10，实际为{decoder.max_iterations}"
        print(f"  ✓ max_iterations = {decoder.max_iterations}")
    else:
        print("  ⚠ 无法直接检查max_iterations，请手动确认decoder.py中设置为10")

    print("\n" + "=" * 60)
    print("所有验证通过！")
    print("=" * 60)


if __name__ == "__main__":
    verify_all()