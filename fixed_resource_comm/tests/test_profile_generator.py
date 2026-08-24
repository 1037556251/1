"""
测试Profile Generator模块
验证所有profile的约束条件
"""

from fixed_resource_comm.modules.profile_generator import ProfileGenerator
import sys
import os
# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def test_profiles():
    """测试所有profile的生成和验证"""
    print("=" * 60)
    print("测试Profile Generator")
    print("=" * 60)

    # 初始化
    generator = ProfileGenerator()

    # 获取所有profile
    profiles = generator.get_all_profiles()

    # 验证1：总共13个profile
    assert len(profiles) == 13, f"应该生成13个profile，但生成了{len(profiles)}个"
    print(f"\n✓ 生成了 {len(profiles)} 个profile")

    # 验证2：每个profile的sum(n)等于8064
    for profile in profiles:
        total_n = sum(profile['n'])
        assert total_n == 8064, f"Profile {profile['id']}: sum(n)={total_n} != 8064"
    print("✓ 所有profile的sum(n)都等于8064")

    # 验证3：每个profile的q_k <= n_k
    for profile in profiles:
        for k in range(8):
            assert profile['q'][k] <= profile['n'][k], \
                f"Profile {profile['id']}: q[{k}]={profile['q'][k]} > n[{k}]={profile['n'][k]}"
    print("✓ 所有profile的q_k <= n_k")

    # 验证4：每个profile的hash是确定性的
    # 重新生成一次，hash应该相同
    generator2 = ProfileGenerator()

    profiles2 = generator2.get_all_profiles()
    for p1, p2 in zip(profiles, profiles2):
        assert p1['hash'] == p2['hash'], \
            f"Profile {p1['id']}: hash不一致！{p1['hash']} vs {p2['hash']}"
    print("✓ 所有profile的hash是确定性的")

    # 打印所有profile的详细信息
    print("\n" + "=" * 60)
    print("所有Profile详细信息")
    print("=" * 60)
    for profile in profiles:
        generator.print_profile_summary(profile['id'])

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)

if __name__ == "__main__":
    test_profiles()