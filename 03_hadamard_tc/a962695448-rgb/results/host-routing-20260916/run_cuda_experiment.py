"""Run GPU validation and three independent host-routing benchmark processes."""

import argparse
import json
import sys
from pathlib import Path

from validation_common import ROOT, Runner, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New result directory."
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Time the candidate after validation."
    )
    parser.add_argument("--stage-timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.stage_timeout < 1:
        parser.error("--stage-timeout must be positive.")
    runner = Runner(args.output, args.stage_timeout)
    runner.record["protocol_sha256"] = digest(ROOT / "CUDA_PROTOCOL.json")
    runner.record["scope"] = (
        "Experimental host routing; production PR remains unchanged until validation and acceptance."
    )
    code = 2
    try:
        worker = [
            sys.executable,
            ROOT / "cuda_worker.py",
            "--build-root",
            runner.output / "build",
        ]
        runner.run(
            "cuda-validation",
            [
                *worker,
                "--mode",
                "validate",
                "--output",
                runner.output / "cuda-validation.json",
            ],
            ROOT,
        )
        runner.record["status"] = "VALIDATED"
        if args.benchmark:
            runner.record["status"] = "RUNNING"
            for index in (1, 2, 3):
                runner.run(
                    f"cuda-round{index}",
                    [
                        *worker,
                        "--mode",
                        "benchmark",
                        "--round",
                        str(index),
                        "--output",
                        runner.output / f"cuda-round{index}.json",
                    ],
                    ROOT,
                )
            runner.run(
                "cuda-analysis",
                [sys.executable, ROOT / "analyze_cuda.py", runner.output],
                ROOT,
            )
            result = json.loads((runner.output / "cuda-analysis.json").read_text())
            runner.record["status"] = "ACCEPTED" if result["accepted"] else "REJECTED"
            runner.record["accepted"] = result["accepted"]
        code = 0
    except Exception as error:  # noqa: BLE001 - Record a terminal failure for any GPU exception.
        validation = runner.output / "cuda-validation.json"
        unverified = (
            validation.exists()
            and json.loads(validation.read_text())["status"] == "UNVERIFIED"
        )
        runner.record["status"] = "UNVERIFIED" if unverified else "FAIL"
        runner.record["error"] = f"{type(error).__name__}: {error}"
        code = 2 if unverified else 1
    finally:
        runner.finish()
    print(json.dumps({"status": runner.record["status"], "output": str(runner.output)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
