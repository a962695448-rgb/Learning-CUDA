#!/usr/bin/env python3
"""Run frozen production regressions serially; permit holdout only after PASS."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "project"
CSV_FIELDS = "timestamp_utc,gpu,compute_capability,cuda_runtime,cuda_driver,batch,seq,heads,dim,dtype,scale,method,scope,repetitions,mean_us,input_elements_per_second,max_abs_error,dense_oracle_rows,warp_block_threads,mean_ms,fused_layout".split(",")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def idle_gate(record, save):
    deadline, count = time.monotonic() + 60, 0
    record["idle_samples"] = []
    while True:
        query = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
        idle = query.returncode == 0 and query.stdout.strip() and all(int(x) == 0 for x in query.stdout.split())
        record["idle_samples"].append({"utc": utc(), "exit_code": query.returncode,
            "utilization_percent": query.stdout, "stderr": query.stderr, "all_zero": bool(idle)})
        save()
        count = count + 1 if idle else 0
        if count == 2:
            record["compute_processes"] = subprocess.check_output(["nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory", "--format=csv"], text=True, timeout=10)
            return
        if time.monotonic() + 2 > deadline:
            raise RuntimeError("two idle samples not reached within60s")
        time.sleep(2)


def verify_sources():
    manifest = json.loads((ROOT / "run_manifest.json").read_text())
    for name, item in manifest["files"].items():
        path = ROOT / name
        assert path.stat().st_size == item["size"] and sha(path) == item["sha256"], name
    return manifest


def self_test_count(text, threads, layout):
    line = next(line for line in text.splitlines() if line.startswith("SELF_TEST PASS cases="))
    assert "cases=1876 " in line and "CPU/split/fused_INT4_bytes=exact scales=exact" in line
    assert f"warp_block_threads={threads}" in line and f"fused_layout={layout}" in line
    assert "sm=89" in text and "RTX 4090" in text
    if layout == "contiguous256":
        assert "fused_layout_scope=N256_only_other_N_original" in line
    return 1876


def check_csv(path, layout):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == CSV_FIELDS
        rows = list(reader)
    assert len(rows) == 7
    assert {r["method"] for r in rows} == {"naive_global", "warp", "tensor_core", "split_int4", "fused_int4", "cpu_fp32_fwht", "warp_h2d_d2h"}
    for row in rows:
        us, ms = float(row["mean_us"]), float(row["mean_ms"])
        assert us > 0 and math.isfinite(us) and math.isclose(ms, us / 1000, rel_tol=1e-9, abs_tol=1e-12)
        assert row["fused_layout"] == (layout if row["method"] == "fused_int4" else "")
        assert row["warp_block_threads"] == ("128" if row["method"] in {"warp", "split_int4", "fused_int4", "warp_h2d_d2h"} else "")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory = args.output_directory.resolve()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    target = args.output_directory / "regression_report.json"
    report = {"status": "RUNNING", "pid": os.getpid(), "started_utc": utc(), "steps": [], "checks": {}, "binaries": {},
              "python_execution": {"assertions_enabled": __debug__, "optimize_flag": sys.flags.optimize,
                                   "PYTHONOPTIMIZE": os.environ.get("PYTHONOPTIMIZE"), "child_PYTHONOPTIMIZE": "0"}}
    save = lambda: target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    env = dict(os.environ, MAX_JOBS="1", TORCH_CUDA_ARCH_LIST="8.9", OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TORCH_EXTENSIONS_DIR=str(ROOT / "extension_cache"), PYTHONOPTIMIZE="0")
    def run(name, arguments, expected=0, gpu=True, timeout=900):
        record = {"name": name, "command": list(map(str, arguments)), "expected_exit": expected}
        if record["command"][0] == sys.executable:
            record["python_assertion_launch"] = {"PYTHONOPTIMIZE": "0", "no_optimization_argument": True}
        report["steps"].append(record)
        if gpu:
            idle_gate(record, save)
        record["started_utc"] = utc()
        log = args.output_directory / (name + ".log")
        with log.open("xb") as stream:
            process = subprocess.Popen(record["command"], cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            record["pid"] = process.pid
            save()
            try:
                record["exit_code"] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                record["exit_code"] = process.returncode
                record["timed_out"] = True
        record.update(finished_utc=utc(), log=log.name, log_sha256=sha(log))
        save()
        text = log.read_text(encoding="utf-8", errors="replace")
        if record["exit_code"] != expected or record.get("timed_out"):
            raise RuntimeError(f"{name} unexpected exit; see retained log")
        return text
    code = 1
    try:
        if not __debug__ or sys.flags.optimize != 0:
            raise RuntimeError("Python assertions disabled; run without -O/-OO or optimization environment")
        report["run_manifest"] = verify_sources()
        report["source_manifest_sha256"] = sha(ROOT / "source_manifest.json")
        report["protocol_sha256"] = sha(ROOT / "protocol.json")
        protocol = json.loads((ROOT / "protocol.json").read_text())
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip().splitlines()
        assert len(gpu) == 1 and "RTX 4090" in gpu[0]
        report["gpu"] = gpu[0]
        cuda = env["CUDA_HOME"]
        build = run("build_and_cpu", ["make", "-j1", "CUDA_HOME=" + cuda, "ARCH=89", "all", "cpu-test"], gpu=False)
        assert "matrix-oracle/involution cases, rounding, packing, zero, invalid shape" in build
        report["checks"]["cpu_reference"] = "PASS"
        executable = PROJECT / "build/hadamard"
        report["binaries"]["cli"] = {"path": str(executable), "sha256": sha(executable)}
        run("cli_default", [sys.executable, PROJECT / "scripts/run_validation.py", "--label", "fused_integration_default"])
        inner = (PROJECT / "results/validation_fused_integration_default.log").read_text()
        report["checks"]["cli_default_cases"] = self_test_count(inner, 128, "original")
        assert inner.count("EXIT_CODE 2;") == 15
        report["checks"]["cli_base_rejections"] = 15
        report["checks"]["cli_original256_cases"] = self_test_count(run("cli_original256", [executable, "--self-test", "--block-threads", "256"]), 256, "original")
        report["checks"]["cli_candidate_cases"] = self_test_count(run("cli_candidate", [executable, "--self-test", "--block-threads", "128", "--fused-layout", "contiguous256"]), 128, "contiguous256")
        invalid = [["--block-threads", str(v)] for v in protocol["regression"]["cli_extra_rejections"]["thread_values"]]
        invalid += [["--block-threads"], ["--fused-layout", "unknown"], ["--fused-layout", ""], ["--fused-layout"],
                    ["--self-test", "--fused-layout", "contiguous256", "--block-threads", "256"],
                    ["--benchmark", "--fused-layout", "contiguous256", "--dim", "128"]]
        for index, arguments in enumerate(invalid):
            run(f"cli_rejection_{index:02d}", [executable, *arguments], expected=2, gpu=False)
        report["checks"]["cli_extra_rejections"] = len(invalid)
        assert len(invalid) == 17
        report["checks"]["csv_rows"] = 0
        smoke = [executable, "--benchmark", "--batch", "1", "--seq", "17", "--heads", "1", "--dim", "256", "--repetitions", "1", "--warmup", "0"]
        for dtype in ("fp16", "bf16"):
            for layout in ("original", "contiguous256"):
                csv_path = args.output_directory / f"format_{dtype}_{layout}.csv"
                run(f"csv_{dtype}_{layout}", [*smoke, "--dtype", dtype, "--fused-layout", layout, "--csv", csv_path])
                report["checks"]["csv_rows"] += check_csv(csv_path, layout)
        for columns in (18, 20):
            csv_path = args.output_directory / f"legacy_{columns}.csv"
            csv_path.write_text(",".join(CSV_FIELDS[:columns]) + "\n", encoding="utf-8")
            before = sha(csv_path)
            log = run(f"legacy_csv_{columns}", [*smoke, "--csv", csv_path], expected=1)
            assert "CSV header differs" in log and sha(csv_path) == before
        report["checks"]["legacy_csv_rejections"] = 2
        api_json, metadata_json, directed_json = [args.output_directory / n for n in ("api_matrix.json", "metadata.json", "directed.json")]
        run("api_matrix", [sys.executable, PROJECT / "scripts/verify_block_threads.py", "--reference-repo", args.reference_repo,
            "--build-directory", ROOT / "build/compatibility", "--json", api_json])
        api = json.loads(api_json.read_text())
        assert api["status"] == "PASS" and api["summary"]["cases"] == 1800 and api["summary"]["failures"] == 0
        assert api["optional_fused_layout"]["original_matrix_cases_checked"] == 200
        assert len(api["thread_value_rejections"]) == 27 and len(api["fused_layout_rejections"]) == 5 and len(api["rejected_inputs"]) == 10
        report["checks"].update(api_matrix_cases=1800, candidate_api_subset_cases=200, api_thread_rejections=27, api_layout_rejections=5, api_original_input_rejections=10)
        compat_binary = Path(api["environment"]["extension"])
        report["binaries"]["compatibility_extension"] = {"path": str(compat_binary), "sha256": sha(compat_binary)}
        assert sha(compat_binary) == api["environment"]["extension_sha256"]
        run("metadata", [sys.executable, PROJECT / "scripts/verify_tensor_metadata.py", "--build-directory", ROOT / "build/production", "--json", metadata_json])
        metadata = json.loads(metadata_json.read_text())
        assert metadata["status"] == "PASS" and len(metadata["cases"]) == 28
        report["checks"]["metadata_cases"] = 28
        run("directed_api", [sys.executable, ROOT / "run_directed.py", "--metadata-report", metadata_json, "--json", directed_json])
        directed = json.loads(directed_json.read_text())
        assert directed["status"] == "PASS" and len(directed["cases"]) == 16
        assert directed["python_execution"]["assertions_enabled"] is True and directed["python_execution"]["optimize_flag"] == 0
        assert directed["environment"]["sm"] == [8, 9]
        assert directed["environment"]["extension_sha256"] == metadata["environment"]["extension_sha256"]
        report["checks"]["targeted_api_cases"] = 16
        binary = Path(directed["environment"]["extension_file"])
        report["binaries"]["production_extension"] = {"path": str(binary), "sha256": sha(binary)}
        for name, binary_info in report["binaries"].items():
            listing = run("elf_" + name, [Path(cuda) / "bin/cuobjdump", "--list-elf", binary_info["path"]], gpu=False)
            assert "sm_89" in listing
        verify_sources()
        report["status"] = "PASS"
        report["holdout_allowed"] = True
        code = 0
    except Exception as error:
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc(), holdout_allowed=False)
        print(report["traceback"], flush=True)
    report.update(finished_utc=utc(), exit_code=code)
    save()
    print(json.dumps({"status": report["status"], "checks": report["checks"], "report": str(target)}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
