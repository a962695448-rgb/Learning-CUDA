#!/usr/bin/env python3
"""Fixed52 neighboring-M fused comparisons against the integrated production API."""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import traceback
import integration_checks as checks
import measurement_helpers as measure

ROOT = Path(__file__).resolve().parent


def configurations(protocol):
    domain = protocol["holdout"]
    cases = [{"dtype": dtype, "rows": rows, "dim": 256, "shape": [rows, 256],
              "normalized": normalized, "scale": 0.0625 if normalized else 1.0}
             for dtype in domain["dtypes"] for rows in domain["rows"] for normalized in domain["normalized"]]
    assert len(cases) == domain["expected_configurations"] == 52
    assert not set(domain["rows"]) & set(domain["exclude_initial_rows"])
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression-report", type=Path, required=True)
    parser.add_argument("--run-index", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.json.exists():
        parser.error("preserve existing output")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "pid": os.getpid(), "started_utc": measure.utc(),
        "run_index": args.run_index, "correctness": [], "benchmarks": [],
        "python_execution": {"assertions_enabled": __debug__, "optimize_flag": sys.flags.optimize,
                             "PYTHONOPTIMIZE": os.environ.get("PYTHONOPTIMIZE")}}
    code = 1
    try:
        if not __debug__ or sys.flags.optimize != 0:
            raise RuntimeError("Python assertions disabled; refusing validation/timing")
        report["run_manifest"] = checks.check_frozen_files()
        report["source_manifest_sha256"] = measure.sha(ROOT / "source_manifest.json")
        protocol = json.loads((ROOT / "protocol.json").read_text())
        report["protocol_sha256"] = measure.sha(ROOT / "protocol.json")
        regression = json.loads(args.regression_report.read_text())
        assert regression["status"] == "PASS" and regression["holdout_allowed"] is True
        assert regression["protocol_sha256"] == report["protocol_sha256"]
        assert regression["source_manifest_sha256"] == report["source_manifest_sha256"]
        expected_binary = regression["binaries"]["production_extension"]
        report["regression_gate"] = {"status": "PASS", "path": str(args.regression_report.resolve()),
            "sha256": measure.sha(args.regression_report), "production_extension_sha256": expected_binary["sha256"]}
        report["before"] = measure.snapshot()
        utilization = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        assert utilization.strip() and all(int(x) == 0 for x in utilization.split()), "GPU busy before CUDA import"
        import numpy as np
        import torch
        assert torch.cuda.device_count() == 1 and "RTX 4090" in torch.cuda.get_device_name()
        assert list(torch.cuda.get_device_capability()) == [8, 9]
        assert os.environ.get("TORCH_CUDA_ARCH_LIST") == "8.9" and os.environ.get("MAX_JOBS") == "1"
        op = checks.load_production()
        assert measure.sha(op.__file__) == expected_binary["sha256"]
        assert Path(op.__file__).resolve() == Path(expected_binary["path"]).resolve()
        report["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "numpy": np.__version__, "gpu": torch.cuda.get_device_name(),
            "sm": list(torch.cuda.get_device_capability()), "cpp11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
            "extension_file": str(Path(op.__file__).resolve()), "extension_sha256": measure.sha(op.__file__),
            "torch_cuda_arch_list": os.environ["TORCH_CUDA_ARCH_LIST"], "max_jobs": os.environ["MAX_JOBS"]}
        cases = configurations(protocol)
        random.Random(92700 + args.run_index).shuffle(cases)
        report["configuration_order"] = cases
        with torch.inference_mode():
            for case in cases:
                entry = {**case, "checks": []}
                report["correctness"].append(entry)
                for pattern, seeds in protocol["holdout"]["patterns_and_seeds"].items():
                    for seed in seeds:
                        for offset in (0, 2):
                            report["active_context"] = {**case, "phase": "correctness", "pattern": pattern, "seed": seed, "pointer_mod16": offset}
                            entry["checks"].append(checks.check_input(torch, np, op, case, pattern, seed, offset))
                print("CHECKED", json.dumps(case), flush=True)
            for index, case in enumerate(cases):
                report["active_context"] = {**case, "phase": "timing", "mode": "fused_int4"}
                dtype = torch.float16 if case["dtype"] == "fp16" else torch.bfloat16
                x = checks.reference_tools.make_input(torch, case["shape"], dtype, "normal", 2026, "cuda")
                assert x.data_ptr() % 16 == 0
                functions = {"original": lambda: op.hadamard_int4(x, case["scale"], 128),
                             "contiguous256": lambda: op.hadamard_int4(x, case["scale"], 128, "contiguous256")}
                timing = measure.measure_graph(torch, functions, args.run_index, index, protocol["timing"])
                report["benchmarks"].append({**case, "mode": "fused_int4", "configuration_index": index, **timing})
                gc.collect()
                print("TIMED", json.dumps(case), flush=True)
                args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        checks.check_frozen_files()
        report["summary"] = {"distinct_configurations": 52, "correctness_input_conditions": 728,
                             "graph_comparisons": 52, "initial_matrix_samples_imported": 0}
        report["status"] = "PASS"
        code = 0
    except Exception as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
    finally:
        try:
            report["after"] = measure.snapshot()
        except Exception as error:
            report["after_snapshot_error"] = repr(error)
        report.update(finished_utc=measure.utc(), exit_code=code)
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": report.get("summary"), "output": str(args.json)}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
