"""
运行所有测试
"""
import subprocess
import sys
import os

# 确保工作目录是项目根目录（即本文件所在目录）
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def run_all_tests():
    """运行所有测试文件"""
    test_files = [
        "tests/test_mother_code.py",
        "tests/test_software_channel.py",
        "tests/test_decoder_erasure.py",
        "tests/test_end_to_end.py",
        "tests/test_profile_generator.py",
        "tests/test_structure.py",
        "tests/test_math_toys.py",
    ]

    # 调试：列出 tests/ 目录内容
    print("tests/ 目录内容:", os.listdir("tests"))

    print("=" * 60)
    print("运行所有测试")
    print("=" * 60)

    all_passed = True

    for test_file in test_files:
        print(f"\n运行测试: {test_file}")
        print("-" * 40)

        # 检查文件是否存在
        if not os.path.exists(test_file):
            print(f"✗ 文件不存在: {test_file}")
            all_passed = False
            continue

        try:
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8"
            )
            print(result.stdout)
            print(f"✓ {test_file} 测试通过")
        except subprocess.CalledProcessError as e:
            print(f"✗ {test_file} 测试失败")
            print(f"错误输出:\n{e.stderr}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！")
    else:
        print("存在测试失败，请检查错误信息")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()