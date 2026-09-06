"""Unchanged52 fused configurations under separately frozen V2 correctness gates."""
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
import checks_v2 as checks

ROOT, BASE = checks.ROOT, checks.BASE
measure = checks.measure


def configurations(protocol):
    spec = protocol["holdout"]
    cases = [{"dtype": dtype, "rows": rows, "dim": 256, "shape": [rows, 256],
              "normalized": normalized, "scale": 0.0625 if normalized else 1.0}
             for dtype in spec["dtypes"] for rows in spec["rows"] for normalized in spec["normalized"]]
    if len(cases) != 52 or set(spec["rows"]) & set(spec["exclude_initial_rows"]):
        raise RuntimeError("matrix changed")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression-report", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--run-index", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    sample_file = args.json.with_name(args.json.stem + "_sample_buffers.json")
    if args.json.exists() or sample_file.exists():
        parser.error("existing results preserved")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "pid": os.getpid(), "started_utc": measure.utc(), "validation_revision": 2,
        "run_index": args.run_index, "correctness": [], "benchmarks": [], "actual_gpu_execution": False,
        "python_execution": {"assertions_enabled": __debug__, "optimize_flag": sys.flags.optimize,
                             "PYTHONOPTIMIZE": os.environ.get("PYTHONOPTIMIZE")}}
    sample_buffers = {}
    def save_samples():
        sample_file.write_text(json.dumps(sample_buffers, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        report["sample_buffers"] = {"file": sample_file.name, "sha256": measure.sha(sample_file),
                                     "buffers": len(sample_buffers), "raw_u16_arrays_retained": True}
    code = 1
    try:
        if not __debug__ or sys.flags.optimize:
            raise RuntimeError("Python assertions must remain enabled")
        protocol, manifest = checks.verify_revision_files()
        regression = checks.verify_regression(args.regression_report, protocol)
        report["protocol_sha256"] = measure.sha(ROOT / "protocol_v2.json")
        report["validation_manifest_sha256"] = measure.sha(ROOT / "manifest_v2.json")
        report["validation_manifest"] = manifest
        report["source_manifest_sha256"] = measure.sha(BASE / "source_manifest.json")
        report["original_regression_gate"] = {"status": "PASS", "report_sha256": measure.sha(args.regression_report),
            "production_extension_sha256": regression["binaries"]["production_extension"]["sha256"]}
        report["original_v1_failure"] = protocol["revision"]
        report["before"] = measure.snapshot()
        utilization = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        if not utilization.strip() or any(int(x) != 0 for x in utilization.split()):
            raise RuntimeError("GPU not idle before CUDA import")
        import numpy as np
        import torch
        import fast_hadamard_transform as reference_package
        import fast_hadamard_transform_cuda as reference_backend
        if torch.cuda.device_count() != 1 or "RTX 4090" not in torch.cuda.get_device_name() or list(torch.cuda.get_device_capability()) != [8, 9]:
            raise RuntimeError("hardware changed")
        if os.environ.get("TORCH_CUDA_ARCH_LIST") != "8.9" or os.environ.get("MAX_JOBS") != "1":
            raise RuntimeError("build environment changed")
        report["reference"] = checks.original_checks.reference_tools.provenance(reference_package, reference_backend, args.reference_repo)
        if report["reference"]["cuda_module_sha256"] != protocol["reused_regression"]["reference_binary_sha256"]:
            raise RuntimeError("fixed reference binary changed")
        op = checks.original_checks.load_production()
        if measure.sha(op.__file__) != protocol["reused_regression"]["production_binary_sha256"]:
            raise RuntimeError("production binary changed")
        report["build_model_assumptions"] = checks.verify_build_assumptions()
        report["environment"] = {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "numpy": np.__version__, "gpu": torch.cuda.get_device_name(), "sm": list(torch.cuda.get_device_capability()),
            "extension_file": str(Path(op.__file__).resolve()), "extension_sha256": measure.sha(op.__file__),
            "torch_cuda_arch_list": os.environ["TORCH_CUDA_ARCH_LIST"], "max_jobs": os.environ["MAX_JOBS"]}
        cases = configurations(protocol)
        random.Random(92700 + args.run_index).shuffle(cases)
        report["configuration_order"] = cases
        with torch.inference_mode():
            report["actual_gpu_execution"] = True
            for case in cases:
                entry = {**case, "checks": []}
                report["correctness"].append(entry)
                for pattern, seeds in protocol["holdout"]["patterns_and_seeds"].items():
                    for seed in seeds:
                        for offset in (0, 2):
                            report["active_context"] = {**case, "phase": "correctness_v2", "pattern": pattern, "seed": seed, "pointer_mod16": offset}
                            entry["checks"].append(checks.check_input(torch, np, op, reference_package.hadamard_transform,
                                                                      case, pattern, seed, offset, sample_buffers))
                print("CHECKED_V2", json.dumps(case), flush=True)
            save_samples()
            for index, case in enumerate(cases):
                report["active_context"] = {**case, "phase": "timing", "mode": "fused_int4"}
                dtype = torch.float16 if case["dtype"] == "fp16" else torch.bfloat16
                x = checks.original_checks.reference_tools.make_input(torch, case["shape"], dtype, "normal", 2026, "cuda")
                if x.data_ptr() % 16:
                    raise RuntimeError("timing input not aligned")
                functions = {"original": lambda: op.hadamard_int4(x, case["scale"], 128),
                             "contiguous256": lambda: op.hadamard_int4(x, case["scale"], 128, "contiguous256")}
                timing = measure.measure_graph(torch, functions, args.run_index, index, protocol["timing"])
                report["benchmarks"].append({**case, "mode": "fused_int4", "configuration_index": index, **timing})
                gc.collect()
                print("TIMED_V2", json.dumps(case), flush=True)
                args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        checks.verify_revision_files()
        report["summary"] = {"distinct_configurations": 52, "correctness_input_conditions": 728,
            "graph_comparisons": 52, "v1_samples_imported": 0,
            "legacy_rounded_dense_threshold_would_fail_conditions": sum(c["legacy_rounded_dense_threshold_would_fail"] for e in report["correctness"] for c in e["checks"])}
        report["status"] = "PASS"
        code = 0
    except Exception as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        if hasattr(error, "certificate"):
            report["failed_certificate"] = error.certificate
        print(report["traceback"], flush=True)
    finally:
        save_samples()
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
