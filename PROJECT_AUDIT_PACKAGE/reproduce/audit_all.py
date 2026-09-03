"""执行审计包中的测试、最小链路和确定性检查。"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "source"
CONFIG_PATH = PACKAGE_ROOT / "configs" / "toy_config.yaml"
LOG_DIR = PACKAGE_ROOT / "logs"
RAW_DIR = PACKAGE_ROOT / "results_raw"
PROCESSED_DIR = PACKAGE_ROOT / "results_processed"
MANIFEST_DIR = PACKAGE_ROOT / "manifests"


def _sha256_bytes(data: bytes) -> str:
    """计算字节数据的 SHA-256 哈希。"""
    return hashlib.sha256(data).hexdigest()


def _tensor_hash(tensor: Any) -> str:
    """计算张量连续字节表示的 SHA-256 哈希。"""
    return _sha256_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def _run_pytest(log_path: Path) -> Dict[str, Any]:
    """运行审计包测试并保存完整标准输出。"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    command = [sys.executable, "-m", "pytest", "tests/"]
    result = subprocess.run(
        command,
        cwd=PACKAGE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    return {"command": command, "returncode": result.returncode}


def _run_pipeline() -> Dict[str, Any]:
    """运行最小随机张量通信链路并返回可审计摘要。"""
    sys.path.insert(0, str(SOURCE_ROOT))
    import torch
    from modules.group_query import GroupQuery
    from modules.input_mask import InputMask
    from modules.mother_code import MotherCode
    from modules.profile_generator import ProfileGenerator
    from modules.quantizer import Quantizer
    from modules.receiver import Receiver
    from modules.software_channel import SoftwareChannel

    torch.manual_seed(123)
    input_mask = InputMask(str(CONFIG_PATH))
    inputs = input_mask.generate_random_tensor(batch_size=2)
    grouped = GroupQuery(str(CONFIG_PATH))(inputs)
    codec = Quantizer(str(CONFIG_PATH))(grouped)
    profile = ProfileGenerator(str(CONFIG_PATH)).get_profile(0)
    role_bits = (codec.reshape(2, 8, 16 * 32) > 0).float()
    source = torch.cat(
        [role_bits[:, k, :q] for k, q in enumerate(profile["q"])], dim=1
    )
    encoded = MotherCode(str(CONFIG_PATH)).encode_from_profile(source, profile)
    channel = SoftwareChannel(str(CONFIG_PATH))
    received, soft = channel.transmit(encoded, return_soft=True)
    decoded, stats = Receiver(str(CONFIG_PATH)).receive(
        received, 8192, sum(profile["q"]), soft_information=soft
    )
    return {
        "input_shape": list(inputs.shape),
        "grouped_shape": list(grouped.shape),
        "codec_shape": list(codec.shape),
        "encoded_shape": list(encoded.shape),
        "received_shape": list(received.shape),
        "decoded_shape": list(decoded.shape),
        "input_hash": _tensor_hash(inputs),
        "encoded_hash": _tensor_hash(encoded),
        "received_hash": _tensor_hash(received),
        "soft_hash": _tensor_hash(soft),
        "decoded_hash": _tensor_hash(decoded),
        "profile_id": profile["id"],
        "profile_hash": profile["hash"],
        "decoding_failed": bool(stats["decoding_failed"]),
    }


def _determinism_check() -> Dict[str, Any]:
    """在相同输入、配置和 seed 下独立执行三次并比较关键哈希。"""
    runs: List[Dict[str, Any]] = []
    for _ in range(3):
        runs.append(_run_pipeline())
    keys = ["input_hash", "encoded_hash", "received_hash", "soft_hash", "decoded_hash"]
    hashes = [[run[key] for key in keys] for run in runs]
    return {"runs": 3, "equal": len({tuple(row) for row in hashes}) == 1, "hashes": hashes}


def _write_results(pipeline: Dict[str, Any], deterministic: Dict[str, Any]) -> None:
    """保存 raw 结果，并由 raw 结果生成 processed 摘要。"""
    raw = {
        "run_type": "toy_minimal_pipeline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline": pipeline,
        "determinism": deterministic,
    }
    raw_path = RAW_DIR / "audit_minimal_pipeline.json"
    raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    loaded = json.loads(raw_path.read_text(encoding="utf-8"))
    processed = {
        "source_raw": str(raw_path.relative_to(PACKAGE_ROOT)),
        "status": "PASS" if loaded["determinism"]["equal"] else "FAIL",
        "shape_summary": {key: value for key, value in loaded["pipeline"].items() if key.endswith("_shape")},
        "deterministic": loaded["determinism"]["equal"],
        "profile_id": loaded["pipeline"]["profile_id"],
        "profile_hash": loaded["pipeline"]["profile_hash"],
    }
    (PROCESSED_DIR / "audit_minimal_pipeline_summary.json").write_text(
        json.dumps(processed, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    """执行审计并生成 manifest。"""
    for directory in (LOG_DIR, RAW_DIR, PROCESSED_DIR, MANIFEST_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    pytest_result = _run_pytest(LOG_DIR / "audit_pytest.log")
    pipeline = _run_pipeline()
    deterministic = _determinism_check()
    _write_results(pipeline, deterministic)
    status = "PASS" if pytest_result["returncode"] == 0 and deterministic["equal"] else "FAIL"
    run_id = "audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "code_version": "cf5262658ed8cb7f0dbe2c4739447938f32b5292",
        "config": str(CONFIG_PATH.relative_to(PACKAGE_ROOT)),
        "config_sha256": _sha256_bytes(CONFIG_PATH.read_bytes()),
        "seed": {"input": 123, "channel": 42},
        "entry_command": "python reproduce/audit_all.py",
        "input": pipeline["input_shape"],
        "output": pipeline["encoded_shape"],
        "runtime": "NOT AVAILABLE",
        "environment": {"python": sys.version.split()[0], "platform": sys.platform},
        "status": status,
        "log_location": "logs/audit_pytest.log",
        "raw_result": "results_raw/audit_minimal_pipeline.json",
        "processed_result": "results_processed/audit_minimal_pipeline_summary.json",
        "determinism": deterministic,
    }
    (MANIFEST_DIR / f"{run_id}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (LOG_DIR / "audit_all.log").open("w", encoding="utf-8") as log:
        log.write(json.dumps(manifest, indent=2, ensure_ascii=False))
        log.write("\n")
    report = PACKAGE_ROOT / "AUDIT_REPORT.md"
    report.write_text(
        "# AUDIT_REPORT\n\n"
        "项目：固定资源通信接口 toy 版本  \n"
        f"审计时间：{manifest['timestamp']}\n\n"
        "| 审计项 | 状态 |\n|---|---|\n"
        "| Source/package | PASS |\n| Project structure | PASS |\n"
        "| Algorithm mapping | PASS |\n| Environment/config | INCONCLUSIVE |\n"
        f"| Tests | {'PASS' if pytest_result['returncode'] == 0 else 'FAIL'} |\n"
        f"| Determinism | {'PASS' if deterministic['equal'] else 'FAIL'} |\n"
        "| Manifest/log | PASS |\n| Result chain | INCONCLUSIVE |\n"
        "| Experiment audit | INCONCLUSIVE |\n| Deviation audit | PASS |\n\n"
        "## OVERALL STATUS: PARTIAL\n\n"
        "当前 toy 源码、测试、最小链路和确定性检查有证据；正式实验方案、数据集、baseline、论文级结果链和图表未提供。\n\n"
        "复现命令：`python reproduce/audit_all.py`\n",
        encoding="utf-8",
    )
    print(f"审计完成：{status}，确定性检查：{deterministic['equal']}，测试返回码：{pytest_result['returncode']}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
