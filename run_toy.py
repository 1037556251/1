"""按固定顺序执行 Toy 版本的配置、前向、结构和数学验收。"""

from pathlib import Path
import subprocess
import sys
from typing import List


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "toy_config.yaml"


def _run_command(command: List[str], title: str) -> None:
    """运行一个验收子命令，失败时抛出异常并终止总流程。"""
    print(f"\n===== {title} =====", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _validate_profiles() -> None:
    """生成 13 个 profile，并检查长度、掩码、ID 和哈希的确定性。"""
    sys.path.insert(0, str(ROOT / "src"))
    from modules.profile_generator import ProfileGenerator

    first = ProfileGenerator(str(CONFIG)).get_all_profiles()
    second = ProfileGenerator(str(CONFIG)).get_all_profiles()
    if len(first) != 13 or len(second) != 13:
        raise AssertionError("profile 数量必须为 13")

    first_signature = []
    second_signature = []
    for profile in first:
        if sum(profile["n"]) != 8064:
            raise AssertionError(f"profile {profile['id']} 的 n_k 总和错误")
        if any(q > n for q, n in zip(profile["q"], profile["n"])):
            raise AssertionError(f"profile {profile['id']} 存在 q_k > n_k")
        if len(profile["mask"]) != 128:
            raise AssertionError(f"profile {profile['id']} 的掩码长度错误")
        first_signature.append((profile["id"], profile["hash"], profile["mask"]))
    for profile in second:
        second_signature.append((profile["id"], profile["hash"], profile["mask"]))

    if first_signature != second_signature:
        raise AssertionError("profile ID、哈希或掩码不是确定性的")
    if [item[0] for item in first_signature] != list(range(13)):
        raise AssertionError("profile ID 必须覆盖 0 到 12")
    if len({item[1] for item in first_signature}) != 13:
        raise AssertionError("profile 哈希必须唯一")
    print("已验证 13 个 profile：n_k 总和、q_k 约束、掩码、ID 和哈希均正确")


def main() -> int:
    """执行全部 Toy 验收阶段，并在任一阶段失败时返回非零状态码。"""
    try:
        print("开始执行 Toy 版本全部验收", flush=True)
        _validate_profiles()

        _run_command(
            [sys.executable, str(ROOT / "scripts" / "run_demo.py")],
            "阶段 ②：随机 Tensor 完整前向",
        )
        _run_command(
            [sys.executable, "-m", "pytest", "tests/test_structure.py", "-q"],
            "阶段 ③：结构测试",
        )
        _run_command(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/test_math_functions.py",
                "tests/test_lp_interface.py",
                "-q",
            ],
            "阶段 ④：数学 Toy 测试和 LP 约束检查",
        )
    except (subprocess.CalledProcessError, AssertionError, ImportError) as error:
        print(f"Toy 验收失败：{error}")
        return 1

    print("\n全部 Toy 测试通过（All Toy Tests PASSED）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
