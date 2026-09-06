#!/usr/bin/env python3
"""Launch exactly three sequential worker processes with retained logs/exits."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "validation"), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    if (args.phase == "validation") != (args.selection is not None):
        parser.error("--selection is required only for validation")
    args.output_directory.mkdir(parents=True, exist_ok=False)
    status_path = args.output_directory / "suite_status.json"
    status = {"status": "RUNNING", "pid": os.getpid(), "phase": args.phase,
              "started_utc": now(), "workers": [], "policy": "Sequential workers; stop on nonzero, retain all partial evidence."}
    env = dict(os.environ, MAX_JOBS="1", TORCH_CUDA_ARCH_LIST="8.9",
               TORCH_EXTENSIONS_DIR=str(ROOT / "extension_cache"),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    exit_code = 1
    try:
        for index in (1, 2, 3):
            command = [sys.executable, "-u", str(ROOT / "run_experiment.py"), "--phase", args.phase,
                       "--run-index", str(index), "--reference-repo", str(args.reference_repo),
                       "--output", str(args.output_directory / f"run{index}.json")]
            if args.selection:
                command.extend(["--selection", str(args.selection)])
            log = args.output_directory / f"run{index}.log"
            record = {"run_index": index, "command": command, "started_utc": now(), "log": log.name}
            status["workers"].append(record)
            with log.open("xb") as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
                record["pid"] = process.pid
                status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
                record["exit_code"] = process.wait()
            record["finished_utc"] = now()
            record["log_sha256"] = hashlib.sha256(log.read_bytes()).hexdigest()
            status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(record), flush=True)
            if record["exit_code"]:
                raise RuntimeError(f"worker {index} returned {record['exit_code']}; no further GPU workers started")
        status["status"] = "PASS"
        exit_code = 0
    except Exception as error:
        status.update({"status": "FAIL", "error": repr(error)})
    finally:
        status["finished_utc"] = now()
        status["exit_code"] = exit_code
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
