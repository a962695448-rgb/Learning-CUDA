"""Three sequential production holdout workers, with source/regression/idle gates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from run_regressions import idle_gate, sha, utc, verify_sources

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression-report", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.regression_report = args.regression_report.resolve()
    args.output_directory = args.output_directory.resolve()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    target = args.output_directory / "suite_status.json"
    report = {"status": "RUNNING", "pid": os.getpid(), "started_utc": utc(), "workers": [],
        "python_execution": {"assertions_enabled": __debug__, "optimize_flag": sys.flags.optimize,
                             "PYTHONOPTIMIZE": os.environ.get("PYTHONOPTIMIZE"), "child_PYTHONOPTIMIZE": "0"}}
    save = lambda: target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    env = dict(os.environ, MAX_JOBS="1", TORCH_CUDA_ARCH_LIST="8.9", OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TORCH_EXTENSIONS_DIR=str(ROOT / "extension_cache"), PYTHONOPTIMIZE="0")
    code = 1
    try:
        if not __debug__ or sys.flags.optimize != 0:
            raise RuntimeError("Python assertions disabled; refusing workers")
        verify_sources()
        regression = json.loads(args.regression_report.read_text())
        assert regression["status"] == "PASS" and regression["holdout_allowed"] is True
        report["regression_report_sha256"] = sha(args.regression_report)
        for index in (1, 2, 3):
            record = {"run_index": index}
            report["workers"].append(record)
            idle_gate(record, save)
            command = [sys.executable, "-u", str(ROOT / "run_holdout.py"), "--run-index", str(index),
                       "--regression-report", str(args.regression_report), "--json", str(args.output_directory / f"run{index}.json")]
            record.update(command=command, started_utc=utc())
            log = args.output_directory / f"run{index}.log"
            with log.open("xb") as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                record["pid"] = process.pid
                save()
                record["exit_code"] = process.wait()
            record.update(finished_utc=utc(), log_sha256=sha(log))
            save()
            if record["exit_code"]:
                raise RuntimeError(f"holdout worker{index} failed; stopped")
        report["status"] = "PASS"
        code = 0
    except Exception as error:
        report.update(status="FAIL", error=repr(error))
    report.update(finished_utc=utc(), exit_code=code)
    save()
    print(json.dumps(report), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
