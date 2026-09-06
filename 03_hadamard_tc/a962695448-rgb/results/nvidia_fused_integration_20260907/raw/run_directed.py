"""Sixteen additional production API/stream/alignment conditions, without timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import integration_checks as checks
import measurement_helpers as measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--metadata-report", type=Path, required=True)
    args = parser.parse_args()
    if args.json.exists():
        parser.error("preserve existing output")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "started_utc": measure.utc(),
        "python_execution": {"assertions_enabled": __debug__, "optimize_flag": sys.flags.optimize,
                             "PYTHONOPTIMIZE": os.environ.get("PYTHONOPTIMIZE")}}
    code = 1
    try:
        if not __debug__ or sys.flags.optimize != 0:
            raise RuntimeError("Python assertions disabled; refusing directed checks")
        checks.check_frozen_files()
        metadata = json.loads(args.metadata_report.read_text())
        assert metadata["status"] == "PASS" and len(metadata["cases"]) == 28
        import numpy as np
        import torch
        op = checks.load_production()
        binary_sha = measure.sha(op.__file__)
        assert binary_sha == metadata["environment"]["extension_sha256"], "metadata/production binary changed"
        report["environment"] = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "python": platform.python_version(), "numpy": np.__version__, "gpu": torch.cuda.get_device_name(),
            "sm": list(torch.cuda.get_device_capability()), "extension_file": str(Path(op.__file__).resolve()),
            "extension_sha256": binary_sha}
        protocol = json.loads((checks.ROOT / "protocol.json").read_text())
        with torch.inference_mode():
            checks.directed_cases(torch, np, op, protocol, report)
        report["status"] = "PASS"
        code = 0
    except Exception as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
    report.update(finished_utc=measure.utc(), exit_code=code)
    args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "directed_conditions": len(report.get("cases", []))}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
