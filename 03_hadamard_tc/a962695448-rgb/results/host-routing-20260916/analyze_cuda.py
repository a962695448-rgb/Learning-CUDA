"""Apply the preregistered acceptance gates to three paired GPU/API timing rounds."""

import argparse
import collections
import json
import math
import statistics
from pathlib import Path

from validation_common import ROOT, digest, save


def analyze(directory, protocol):
    shapes = {
        (kind, dtype, rows, dim, layout, operation)
        for kind in ("target", "control")
        for rows, dim in protocol[kind + "_shapes"]
        for dtype in protocol["dtypes"]
        for layout in protocol["layouts"]
        for operation in protocol["operations"]
    }
    expected_keys = {
        (*shape, version, group)
        for shape in shapes
        for version in ("control", "candidate")
        for group in range(protocol["groups"])
    }
    validation = json.loads((directory / "cuda-validation.json").read_text())
    if validation["status"] != "PASS" or validation["result"]["status"] != "PASS":
        raise ValueError("GPU correctness validation did not pass.")
    identity_keys = (
        "hardware",
        "numpy",
        "protocol_sha256",
        "control_binding_sha256",
        "candidate_binding_sha256",
        "binaries",
        "cuda_compiler",
    )
    identity = {key: validation[key] for key in identity_keys}
    rounds = []
    comparisons = []
    count = 0
    for round_index in range(1, protocol["rounds"] + 1):
        report = json.loads((directory / f"cuda-round{round_index}.json").read_text())
        if report["status"] != "PASS" or report["mode"] != "benchmark":
            raise ValueError("A benchmark round did not pass.")
        if {key: report[key] for key in identity_keys} != identity:
            raise ValueError("Hardware, source, binary or protocol identity changed.")
        values, seen = collections.defaultdict(list), set()
        for record in report["result"]["records"]:
            shape = tuple(
                record[key]
                for key in ("kind", "dtype", "rows", "dim", "layout", "operation")
            )
            key = (*shape, record["version"], record["group"])
            if (
                key in seen
                or key not in expected_keys
                or record["round"] != round_index
            ):
                raise ValueError("Duplicate or unexpected timing configuration.")
            seen.add(key)
            if record["repeats"] != protocol["repeats"] or record["repeats"] < 1:
                raise ValueError("Unexpected repeat count.")
            expected_us = record["elapsed_ns"] / record["repeats"] / 1000
            if (
                record["repeats"] != protocol["repeats"]
                or not math.isfinite(record["mean_us"])
                or record["mean_us"] <= 0
                or not math.isclose(record["mean_us"], expected_us, rel_tol=1e-12)
            ):
                raise ValueError("Invalid timing interval or unit conversion.")
            values[(*shape, record["version"])].append(record["mean_us"])
        if seen != expected_keys:
            raise ValueError("Missing measurements.")
        count += len(seen)
        ratios = []
        regressions = []
        for shape in sorted(shapes):
            old = statistics.median(values[(*shape, "control")])
            new = statistics.median(values[(*shape, "candidate")])
            reduction = 100 * (1 - new / old)
            comparisons.append(
                {
                    "round": round_index,
                    "kind": shape[0],
                    "dtype": shape[1],
                    "rows": shape[2],
                    "dim": shape[3],
                    "layout": shape[4],
                    "operation": shape[5],
                    "old_us": old,
                    "new_us": new,
                    "old_over_new": old / new,
                    "time_reduction_percent": reduction,
                }
            )
            if shape[0] == "target":
                ratios.append(old / new)
            regressions.append(-reduction)
        geomean = math.exp(statistics.mean(math.log(ratio) for ratio in ratios))
        aggregate_reduction = 100 * (1 - 1 / geomean)
        worst = max(regressions)
        accepted = (
            aggregate_reduction
            >= protocol["acceptance"][
                "target_geomean_time_reduction_at_least_percent_each_round"
            ]
            and worst
            <= protocol["acceptance"]["maximum_case_time_regression_percent_each_round"]
        )
        rounds.append(
            {
                "round": round_index,
                "target_unweighted_geomean_time_reduction_percent": aggregate_reduction,
                "worst_case_regression_percent": worst,
                "accepted": accepted,
            }
        )
    return {
        "status": "VERIFIED",
        "accepted": all(item["accepted"] for item in rounds),
        "identity": identity,
        "observations": count,
        "rounds": rounds,
        "comparisons": comparisons,
        "measurement": "Synchronized host wall time for allocating Python API calls, including GPU completion.",
        "scope": "The unweighted target-set statistic is an engineering gate, not an application speedup.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "CUDA_PROTOCOL.json").read_text())
    result = analyze(args.directory, protocol)
    if result["identity"]["protocol_sha256"] != digest(ROOT / "CUDA_PROTOCOL.json"):
        raise ValueError("The measured protocol differs from the local protocol.")
    output = args.directory / "cuda-analysis.json"
    if output.exists():
        raise FileExistsError(output)
    save(output, result)
    print(
        json.dumps(
            {key: result[key] for key in ("accepted", "observations", "rounds")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
