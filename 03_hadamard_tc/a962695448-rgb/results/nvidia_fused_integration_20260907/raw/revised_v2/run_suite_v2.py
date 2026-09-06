"""Sequential V2 processes; reuse actual regression and retain original failure."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import checks_v2 as checks

sys.path.insert(0, str(checks.BASE))
from run_regressions import idle_gate, sha, utc
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression-report", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory = args.output_directory.resolve()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "validation_revision": 2, "pid": os.getpid(), "started_utc": utc(), "workers": [],
        "python_execution": {"assertions_enabled": __debug__, "optimize_flag": sys.flags.optimize,
                             "PYTHONOPTIMIZE": os.environ.get("PYTHONOPTIMIZE"), "child_PYTHONOPTIMIZE": "0"}}
    target = args.output_directory / "suite_status_v2.json"
    save = lambda: target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    env = dict(os.environ, MAX_JOBS="1", TORCH_CUDA_ARCH_LIST="8.9", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONOPTIMIZE="0", TORCH_EXTENSIONS_DIR=str(checks.BASE / "extension_cache"))
    code = 1
    try:
        if not __debug__ or sys.flags.optimize:
            raise RuntimeError("assertions disabled")
        protocol, manifest = checks.verify_revision_files()
        checks.verify_regression(args.regression_report, protocol)
        report["protocol_sha256"] = sha(ROOT / "protocol_v2.json")
        report["validation_manifest_sha256"] = sha(ROOT / "manifest_v2.json")
        report["original_regression_sha256"] = sha(args.regression_report)
        for index in (1, 2, 3):
            record = {"run_index": index}
            report["workers"].append(record)
            idle_gate(record, save)
            command = [sys.executable, "-u", str(ROOT / "run_v2.py"), "--run-index", str(index),
                "--regression-report", str(args.regression_report.resolve()), "--reference-repo", str(args.reference_repo.resolve()),
                "--json", str(args.output_directory / f"run{index}.json")]
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
                raise RuntimeError(f"V2 worker{index} failed; dependent workers stopped")
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
