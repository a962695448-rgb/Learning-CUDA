"""仅用于此次隔离集成验证，所有写入位于本实验目录。"""
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "project"
RESULTS = ROOT / "results"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    out = {}
    for key, fields in (("gpu", "--query-gpu=name,driver_version,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw"),
                        ("processes", "--query-compute-apps=pid,process_name,used_memory")):
        p = subprocess.run(["nvidia-smi", fields, "--format=csv"], capture_output=True, text=True, timeout=15)
        out[key] = {"exit": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    return out


def main():
    RESULTS.mkdir(exist_ok=False)
    report = {"status": "RUNNING", "base_commit": "8f75553a074d79b850294377d5aea6381e93da19",
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "before": snapshot(), "stages": [], "graph_smoke": []}
    def run(label, args, expected=0):
        log = RESULTS / (label + ".log")
        start = time.monotonic()
        with log.open("xb") as stream:
            process = subprocess.run([str(x) for x in args], cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT, check=False)
        report["stages"].append({"label": label, "args": [str(x) for x in args], "exit": process.returncode,
                                 "expected_exit": expected, "elapsed_seconds": time.monotonic() - start, "log_sha256": sha(log)})
        print("STAGE", label, process.returncode, flush=True)
        if process.returncode != expected:
            raise RuntimeError(f"{label} failed: " + log.read_text(errors="replace")[-5000:])
        return log.read_text(errors="replace")
    status = 1
    try:
        source = json.loads((ROOT / "source_manifest.json").read_text())
        for name, info in source["files"].items():
            if sha(ROOT / name) != info["sha256"]:
                raise RuntimeError("source hash mismatch: " + name)
        report["source_manifest"] = source
        os.environ["MAX_JOBS"] = "1"
        os.environ["TORCH_EXTENSIONS_DIR"] = str(ROOT / "torch_cache")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
        run("build_cli_cpu", ["make", "-j1", "CUDA_HOME=" + os.environ["CUDA_HOME"], "ARCH=89", "all", "cpu-test"])
        binary = PROJECT / "build/hadamard"
        report["cli_binary_sha256"] = sha(binary)
        run("default_matrix_and_existing_cli", [sys.executable, "scripts/run_validation.py", "--label", "integration_default"])
        default_log = PROJECT / "results/validation_integration_default.log"
        (RESULTS / default_log.name).write_bytes(default_log.read_bytes())
        if "SELF_TEST PASS cases=1876" not in default_log.read_text():
            raise RuntimeError("default original1876 matrix not confirmed")
        result = run("explicit256_matrix", [binary, "--self-test", "--block-threads", "256"])
        if "SELF_TEST PASS cases=1876" not in result or "warp_block_threads=256" not in result:
            raise RuntimeError("explicit256 original1876 matrix not confirmed")
        for i, value in enumerate(("0", "32", "64", "127", "129", "512", "-1", "128.5", "x", "2147483648")):
            run("invalid_threads_" + str(i), [binary, "--self-test", "--block-threads", value], expected=2)
        run("missing_threads_value", [binary, "--block-threads"], expected=2)
        for threads in (128, 256):
            path = RESULTS / f"units_threads{threads}.csv"
            run("units_threads" + str(threads), [binary, "--benchmark", "--batch", "1", "--seq", "17", "--heads", "1",
                "--dim", "64", "--block-threads", str(threads), "--repetitions", "5", "--warmup", "1", "--csv", path])
            with path.open() as stream:
                samples = list(csv.DictReader(stream))
            assert len(samples) == 7
            for sample in samples:
                assert math.isclose(float(sample["mean_ms"]), float(sample["mean_us"]) / 1000, rel_tol=1e-10)
                affected = sample["method"] in ("warp", "split_int4", "fused_int4", "warp_h2d_d2h")
                assert sample["warp_block_threads"] == (str(threads) if affected else "")
        legacy = RESULTS / "legacy_header.csv"
        legacy.write_text("timestamp_utc,gpu,old_schema\n")
        legacy_before = sha(legacy)
        result = run("reject_legacy_csv_append", [binary, "--benchmark", "--dim", "1", "--batch", "1", "--seq", "1", "--heads", "1",
            "--repetitions", "1", "--warmup", "0", "--csv", legacy], expected=1)
        assert "CSV header differs" in result and sha(legacy) == legacy_before
        cache = ROOT / "torch_cache/block_threads_check"
        run("pytorch_1800_and_compatibility", [sys.executable, "scripts/verify_block_threads.py",
            "--reference-repo", "/data/infinitensor-2026/fast-hadamard-transform", "--build-directory", cache,
            "--json", RESULTS / "pytorch_1800.json"])
        validation = json.loads((RESULTS / "pytorch_1800.json").read_text())
        assert validation["status"] == "PASS" and validation["summary"]["cases"] == 1800
        assert validation["default_and_explicit128_and_256_bitwise_equal"]
        assert validation["non_default_stream"]["pass"]
        report["functional_summary"] = {"cli_matrix_distinct_cases": 1876, "cli_thread_modes": [128, 256],
            "dao_matrix_distinct_cases": 1800, "default_128_256_bitwise_exact": True,
            "thread_invalid_cli_cases": 11, "existing_invalid_cli_cases": 15,
            "thread_invalid_pytorch_cases": len(validation["thread_value_rejections"]), "non_default_stream": "PASS",
            "unit_conversion_and_csv_thread_scope": "PASS", "old_csv_unchanged_on_rejection": True}
        sys.path.insert(0, str(PROJECT / "scripts"))
        import torch
        import verify_block_threads
        import fast_hadamard_transform
        from graph_measure import measure_graph
        op = verify_block_threads.load_for_validation(cache)
        assert sha(op.__file__) == validation["environment"]["extension_sha256"]
        report["extension_binary_sha256"] = sha(op.__file__)
        cases = [("transform", 4096, 16, "fp16"), ("transform", 4096, 64, "bf16"),
                 ("transform", 16384, 16, "fp16"), ("transform", 16384, 64, "bf16"),
                 ("fused_int4", 4096, 16, "fp16"), ("fused_int4", 4096, 64, "bf16")]
        with torch.inference_mode():
            for mode, rows, n, dtype_name in cases:
                dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
                shape = (rows // 1024, 128, 8, n)
                x = verify_block_threads.compare_reference.make_input(torch, shape, dtype, "normal", 2026, "cuda")
                if mode == "transform":
                    functions = {"baseline128": lambda: op.hadamard(x, block_threads=128),
                                 "candidate256": lambda: op.hadamard(x, block_threads=256),
                                 "dao": lambda: fast_hadamard_transform.hadamard_transform(x, 1.0)}
                else:
                    functions = {"baseline128": lambda: op.hadamard_int4(x, block_threads=128),
                                 "candidate256": lambda: op.hadamard_int4(x, block_threads=256)}
                result = measure_graph(torch, functions, 1)
                result["median_ms"] = {name: value / 1000 for name, value in result["median_us"].items()}
                report["graph_smoke"].append({"mode": mode, "rows": rows, "dim": n, "dtype": dtype_name, **result})
                print("GRAPH_SMOKE", mode, rows, n, dtype_name, result["candidate_time_reduction_percent"], flush=True)
                gc.collect()
        report["status"] = "PASS"
        status = 0
    except Exception as error:
        import traceback
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
    finally:
        report["after"] = snapshot()
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report["artifacts"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in RESULTS.iterdir() if p.is_file()}
        (RESULTS / "integration_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "functional_summary": report.get("functional_summary"), "graph_cases": len(report["graph_smoke"])}), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
