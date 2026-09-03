"""
运行所有测试
"""
import subprocess
import sys
import os

# 获取当前文件所在目录（即 fixed_resource_comm/）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（fixed_resource_comm 的父目录）
PROJECT_ROOT = os.path.dirname(BASE_DIR)


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

    # 设置子进程环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + env.get('PYTHONPATH', '')
    env['PYTHONIOENCODING'] = 'utf-8'   # 强制 stdout/stderr 使用 UTF-8

    print("=" * 60)
    print("运行所有测试")
    print("=" * 60)

    all_passed = True
    for test_file in test_files:
        abs_path = os.path.join(BASE_DIR, test_file)
        print(f"\n运行测试: {test_file}")
        print("-" * 40)

        if not os.path.exists(abs_path):
            print(f"✗ 文件不存在: {abs_path}")
            all_passed = False
            continue

        try:
            # 使用 UTF-8 解码，忽略错误（避免个别特殊字符导致崩溃）
            result = subprocess.run(
                [sys.executable, abs_path],
                capture_output=True,
                text=True,
                check=True,
                env=env,
                encoding='utf-8',
                errors='ignore'   # 忽略无法解码的字符
            )
            print(result.stdout)
            print(f"✓ {test_file} 测试通过")
        except subprocess.CalledProcessError as e:
            print(f"✗ {test_file} 测试失败")
            # 错误输出也可能有特殊字符，同样忽略
            if e.stderr:
                print("错误输出:", e.stderr)
            else:
                print("（无错误输出）")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！")
    else:
        print("存在测试失败，请检查错误信息")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()