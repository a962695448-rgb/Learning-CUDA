#!/usr/bin/env python3
"""Resume frozen workers with two recorded idle samples between processes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def wait_idle(record, save):
    deadline, consecutive = time.monotonic() + 60, 0
    record["idle_samples"] = []
    while True:
        query = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
        idle = query.returncode == 0 and query.stdout.strip() and all(int(v) == 0 for v in query.stdout.split())
        record["idle_samples"].append({"utc": now(), "exit_code": query.returncode,
            "utilization_percent": query.stdout, "stderr": query.stderr, "all_zero": bool(idle)})
        save()
        consecutive = consecutive + 1 if idle else 0
        if consecutive == 2:
            record["compute_processes"] = subprocess.check_output(["nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory", "--format=csv"], text=True, timeout=10)
            return
        if time.monotonic() + 2 > deadline:
            raise RuntimeError("GPU not continuously idle within 60 seconds; no worker launched")
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "validation"), required=True)
    parser.add_argument("--indices", type=int, nargs="+", choices=(1, 2, 3), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    if len(args.indices) != len(set(args.indices)) or (args.phase == "validation") != bool(args.selection):
        parser.error("unique indices and validation-only selection required")
    args.output_directory.mkdir(parents=True, exist_ok=False)
    status = {"status": "RUNNING", "pid": os.getpid(), "started_utc": now(), "phase": args.phase,
        "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frozen_worker_and_protocol_unchanged": True, "workers": [],
        "reason": "Original immediate run2 was rejected before CUDA import by utilization gate after run1. Preserve run1 and failed run2; add only a bounded scheduling idle wait, without changing measurement or correctness."}
    target = args.output_directory / "continuation_status.json"
    save = lambda: target.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    env = dict(os.environ, MAX_JOBS="1", TORCH_CUDA_ARCH_LIST="8.9", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TORCH_EXTENSIONS_DIR=str(ROOT / "extension_cache"))
    code = 1
    try:
        for index in args.indices:
            record = {"run_index": index}
            status["workers"].append(record)
            wait_idle(record, save)
            command = [sys.executable, "-u", str(ROOT / "run_experiment.py"), "--phase", args.phase,
                "--run-index", str(index), "--reference-repo", str(args.reference_repo),
                "--output", str(args.output_directory / f"run{index}.json")]
            if args.selection:
                command += ["--selection", str(args.selection)]
            record.update({"command": command, "started_utc": now()})
            log = args.output_directory / f"run{index}.log"
            with log.open("xb") as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                record["pid"] = process.pid
                save()
                record["exit_code"] = process.wait()
            record.update({"finished_utc": now(), "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()})
            save()
            if record["exit_code"]:
                raise RuntimeError(f"worker {index} failed; no further workers started")
        status["status"] = "PASS"
        code = 0
    except Exception as error:
        status.update({"status": "FAIL", "error": repr(error)})
    finally:
        status.update({"finished_utc": now(), "exit_code": code})
        save()
    print(json.dumps(status), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
