# verify_profiles.py
from fixed_resource_comm.modules.profile_generator import ProfileGenerator


def main():
    print("=== 验证所有profile的N0值 ===")

    pg = ProfileGenerator()
    profiles = pg.get_all_profiles()

    all_valid = True
    for p in profiles:
        n_sum = sum(p['n'])
        N0 = n_sum + 128  # header固定128 bits

        print(f"Profile {p['id']}:")
        print(f"  - n = {p['n']}")
        print(f"  - sum(n) = {n_sum}")
        print(f"  - header = 128")
        print(f"  - N0 = {N0}")

        if N0 != 8192:
            print(f"  ❌ 错误: N0应该为8192，实际为{N0}")
            all_valid = False
        else:
            print(f"  ✅ 正确: N0=8192")
        print()

    if all_valid:
        print("🎉 所有profile的N0=8192验证通过！")
    else:
        print("❌ 验证失败，存在N0不等于8192的profile")


if __name__ == "__main__":
    main()