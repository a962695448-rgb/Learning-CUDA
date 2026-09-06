#!/usr/bin/env python3
"""Three sequential processes with a bounded, recorded idle gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


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
            raise RuntimeError("two idle samples not reached within60s; no worker started")
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    target = args.output_directory / "suite_status.json"
    status = {"status": "RUNNING", "pid": os.getpid(), "started_utc": utc(),
              "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "workers": []}
    save = lambda: target.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    env = dict(os.environ, MAX_JOBS="1", TORCH_CUDA_ARCH_LIST="8.9", OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TORCH_EXTENSIONS_DIR=str(ROOT / "extension_cache"))
    code = 1
    try:
        for index in (1, 2, 3):
            record = {"run_index": index}
            status["workers"].append(record)
            idle_gate(record, save)
            command = [sys.executable, "-u", str(ROOT / "run_experiment.py"), "--run-index", str(index),
                "--output", str(args.output_directory / f"run{index}.json"), "--reference-repo", str(args.reference_repo)]
            record.update({"command": command, "started_utc": utc()})
            log = args.output_directory / f"run{index}.log"
            with log.open("xb") as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                record["pid"] = process.pid
                save()
                record["exit_code"] = process.wait()
            record.update({"finished_utc": utc(), "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()})
            save()
            if record["exit_code"]:
                raise RuntimeError(f"worker{index} failed; no further worker started")
        status["status"] = "PASS"
        code = 0
    except Exception as error:
        status.update({"status": "FAIL", "error": repr(error)})
    finally:
        status.update({"finished_utc": utc(), "exit_code": code})
        save()
    print(json.dumps(status), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
