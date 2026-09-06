#!/usr/bin/env python3
"""Offline analysis of three fixed fused_int4 holdout processes.

python analyze_holdout.py --regression-report runs/regression.json --output derived \
    runs/run1.json runs/run2.json runs/run3.json

Writes analysis.json, comparison.csv, and samples.csv without changing inputs.
No GPU execution, automatic dispatch, or data from the initial 24 configurations.
Exit 0: verified evidence, including negative results. Exit 2: UNVERIFIED.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import statistics
from datetime import datetime, timezone
from fractions import Fraction


ROOT = Path(__file__).resolve().parent
BASE_COMMIT = "217c30ff5e78842cd5809de6bf78ee8a7f04fc54"
KEYS = ("dtype", "dim", "rows", "normalized")
METHODS = ("original", "contiguous256")
EXPECTED_ROWS = [2, 3, 16, 18, 63, 65, 255, 256, 258, 4095, 4097, 16383, 16385]
INITIAL_ROWS = [1, 17, 64, 257, 4096, 16384]
REGRESSION_COUNTS = {
    "cli_default_cases": 1876, "cli_original256_cases": 1876, "cli_candidate_cases": 1876,
    "api_matrix_cases": 1800, "candidate_api_subset_cases": 200, "metadata_cases": 28,
    "targeted_api_cases": 16, "cli_base_rejections": 15, "cli_extra_rejections": 17,
    "csv_rows": 28, "legacy_csv_rejections": 2,
    "api_thread_rejections": 27, "api_layout_rejections": 5, "api_original_input_rejections": 10,
}
SCOPE = ("Fused INT4 only: CUDA-event intervals / (20 replays * 64 independent retained "
         "output tuples), median of five groups per process. Captured GPU work and amortized "
         "replay scheduling; not isolated kernel, eager API, or host end-to-end latency.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path, inputs):
    data = path.read_bytes()
    inputs[str(path)] = {"bytes": len(data), "sha256": sha(data)}
    def reject(value):
        raise ValueError("Non-finite JSON value: " + value)
    return json.loads(data.decode("utf-8-sig"), parse_constant=reject)


def number(value, label, positive=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), label + ": invalid numeric value")
    require(value > 0 if positive else value >= 0, label + ": invalid range")
    return float(value)


def close(actual, expected, label):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)) and
            math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12), label + ": inconsistent value")


def case_key(case):
    require(type(case["normalized"]) is bool, "normalized must be Boolean")
    require(type(case["rows"]) is int and case["rows"] in EXPECTED_ROWS, "Row count outside holdout or from initial matrix")
    require(case["dim"] == 256 and case["shape"] == [case["rows"], 256], "Shape differs from fixed holdout")
    require(case["dtype"] in ("fp16", "bf16"), "Unexpected dtype")
    close(case["scale"], 0.0625 if case["normalized"] else 1.0, "Common scale")
    return tuple(case[name] for name in KEYS)


def verify_regression(regression, source_sha):
    require(regression["status"] == "PASS" and regression["exit_code"] == 0 and
            regression["holdout_allowed"] is True, "Full regression did not permit holdout")
    execution = regression["python_execution"]
    require(execution["assertions_enabled"] is True and execution["optimize_flag"] == 0 and
            execution["child_PYTHONOPTIMIZE"] == "0", "Regression Python assertion settings are invalid")
    require(regression["checks"]["cpu_reference"] == "PASS", "CPU reference regression did not PASS")
    require(regression["source_manifest_sha256"] == source_sha, "Regression source snapshot differs")
    for field, count in REGRESSION_COUNTS.items():
        require(regression["checks"][field] == count, "Incomplete regression check: " + field)
    for name in ("cli", "compatibility_extension", "production_extension"):
        info = regression["binaries"][name]
        require(bool(info["path"]) and re.fullmatch(r"[0-9a-f]{64}", info["sha256"]), "Missing regression binary identity: " + name)
    require(regression["steps"] and all(step["exit_code"] == step["expected_exit"] and
            step.get("finished_utc") for step in regression["steps"]), "Regression step incomplete or unexpected exit")
    return regression["binaries"]["production_extension"]["sha256"]


def verify_identity(report, protocol, protocol_sha, source_sha, regression_sha, binary_sha):
    require(report["status"] == "PASS" and report["exit_code"] == 0, "Worker failed or incomplete")
    execution = report["python_execution"]
    require(execution["assertions_enabled"] is True and execution["optimize_flag"] == 0 and
            execution["PYTHONOPTIMIZE"] == "0", "Holdout Python assertion settings are invalid")
    require(type(report["run_index"]) is int and report["run_index"] in (1, 2, 3), "Invalid run index")
    require(report["protocol_sha256"] == protocol_sha and report["source_manifest_sha256"] == source_sha, "Worker protocol/source differs")
    gate = report["regression_gate"]
    require(gate["status"] == "PASS" and gate["sha256"] == regression_sha and bool(gate["path"]), "Worker used a different/incomplete regression gate")
    require(gate["production_extension_sha256"] == binary_sha, "Worker gate production binary differs")
    environment, hardware = report["environment"], protocol["hardware"]
    require(hardware["required_name_contains"] in environment["gpu"] and environment["sm"] == hardware["required_sm"], "GPU/SM differs from fixed protocol")
    require(environment["extension_sha256"] == binary_sha, "Holdout binary differs from fully regressed production extension")
    require(environment["torch_cuda_arch_list"] == hardware["compile_arch"] and environment["max_jobs"] == hardware["max_jobs"], "Compiler setting differs")
    manifest = report["run_manifest"]["files"]
    require(manifest["protocol.json"]["sha256"] == protocol_sha and manifest["source_manifest.json"]["sha256"] == source_sha, "Frozen worker manifest identity differs")
    # Validate execution/source bytes only under this isolated experiment root.
    for name, metadata in manifest.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT), "Manifest path escapes isolated experiment")
        require(sha(path.read_bytes()) == metadata["sha256"], "Frozen local input differs: " + name)


def verify_correctness(report, protocol, expected):
    holdout = protocol["holdout"]
    input_keys = {(pattern, seed, offset) for pattern, seeds in holdout["patterns_and_seeds"].items()
                  for seed in seeds for offset in (0, 2)}
    require(len(input_keys) == 14, "Expected seven inputs at two pointer offsets")
    seen = set()
    maxima = {dtype: 0.0 for dtype in ("fp16", "bf16")}
    conditions = 0
    for case in report["correctness"]:
        key = case_key(case)
        require(key in expected and key not in seen, "Duplicate/unexpected correctness configuration")
        seen.add(key)
        entries = case["checks"]
        require(len(entries) == 14 and {(x["pattern"], x["seed"], x["pointer_mod16"]) for x in entries} == input_keys, "Correctness pattern/seed/offset matrix incomplete")
        limit = 0.01 if case["dtype"] == "fp16" else 0.05
        close(holdout["strict_abs_limit"][case["dtype"]], limit, "Protocol strict limit")
        for entry in entries:
            for flag in ("pass", "original_candidate_fused_split_exact", "input_guards_unchanged", "cpu_quantization_exact"):
                require(entry[flag] is True, "Correctness flag failed: " + flag)
            require(entry["legacy_three_arg_default_equals_explicit_original"] is True and
                    entry["non_default_stream"] is False, "Holdout API/stream metadata differs")
            require(entry["elements"] == case["rows"] * 256 and entry["quantization_rows_checked"] == case["rows"], "Incomplete all-element quantization coverage")
            require(entry["dense_rows"] == sorted({0, case["rows"] // 2, case["rows"] - 1}), "Dense oracle row subset differs")
            close(entry["strict_limit"], limit, "Reported strict limit")
            error = number(entry["dense_max_abs_error"], "Dense error")
            require(error < limit, "Dense error exceeded strict limit")
            maxima[case["dtype"]] = max(maxima[case["dtype"]], error)
            conditions += 1
    require(seen == expected and len(report["correctness"]) == 52 and conditions == 728, "Expected exactly 52 configurations/728 conditions")
    return maxima


def verify_timings(report, ordered_cases):
    order = list(ordered_cases)
    random.Random(92700 + report["run_index"]).shuffle(order)
    require([case_key(case) for case in report["configuration_order"]] == order, "Configuration order differs from fixed protocol")
    require([case_key(case) for case in report["benchmarks"]] == order, "Benchmark configuration sequence incomplete/different")
    indexed, samples = {}, []
    for index, entry in enumerate(report["benchmarks"]):
        key = case_key(entry)
        require(entry["mode"] == "fused_int4" and entry["configuration_index"] == index, "Unexpected timing mode/index")
        for field in ("samples_us", "raw_event_intervals_ms", "median_us", "median_ms"):
            require(set(entry[field]) == set(METHODS), "Only original/contiguous256 are allowed; no Dao denominator")
        for flag in ("independent_output_buffers", "cross_method_output_pointers_disjoint", "outputs_bitwise_equal_eager_before_and_after"):
            require(entry[flag] is True, "Graph buffer/output check failed: " + flag)
        for field, value in (("captured_calls_per_graph", 64), ("replays_per_group", 20), ("graph_warmup_replays", 5), ("api_warmup_calls", 25)):
            require(entry[field] == value, "Timing condition differs: " + field)
        orders, names = [], list(METHODS)
        for group in range(5):
            offset = (report["run_index"] - 1 + index + group) % 2
            orders.append(names[offset:] + names[:offset])
        require(entry["group_order"] == orders, "Method group order differs")
        stats = {}
        for method in METHODS:
            us = entry["samples_us"][method]
            ms = entry["raw_event_intervals_ms"][method]
            require(len(us) == len(ms) == 5, "Expected exactly five raw groups")
            us = [number(value, "Per-call time", positive=True) for value in us]
            ms = [number(value, "Raw event interval", positive=True) for value in ms]
            for group, (per_call, interval) in enumerate(zip(us, ms), 1):
                close(per_call, interval * 1000 / 1280, "Event conversion")
                samples.append({"run_index": report["run_index"], **dict(zip(KEYS, key)),
                    "shape": json.dumps(entry["shape"]), "scale": entry["scale"], "mode": "fused_int4",
                    "scope": "cuda_graph", "method": method, "group": group, "event_interval_ms": interval,
                    "calls_per_interval": 1280, "per_call_us": per_call, "per_call_ms": per_call / 1000})
            median = statistics.median(us)
            close(entry["median_us"][method], median, "Median microseconds")
            close(entry["median_ms"][method], median / 1000, "Median milliseconds")
            stats[method] = {"median_us": median, "range_percent_of_median": 100 * (max(us) - min(us)) / median,
                             "population_cv_percent": 100 * statistics.pstdev(us) / statistics.mean(us)}
        indexed[key] = stats
    require(len(indexed) == 52 and len(samples) == 520, "Incorrect per-process timing sample count")
    return indexed, samples


def compare_runs(indexed):
    require(set(indexed) == {1, 2, 3} and set(indexed[1]) == set(indexed[2]) == set(indexed[3]), "Three matching configuration matrices required")
    rows = []
    for key in sorted(indexed[1]):
        row = {**dict(zip(KEYS, key)), "shape": json.dumps([key[2], 256]),
               "scale": 0.0625 if key[3] else 1.0, "mode": "fused_int4", "scope": "cuda_graph"}
        gains, losses, stable_flags, regression_flags, over_three_flags = [], [], [], [], []
        for run in (1, 2, 3):
            stats = indexed[run][key]
            old, new = stats["original"]["median_us"], stats["contiguous256"]["median_us"]
            gain = 100 * (1 - new / old)
            gains.append(gain)
            losses.append(100 * (new / old - 1))
            # Exact ratios of recorded medians avoid flagging exactly 3% as >3%.
            ratio = Fraction(new) / Fraction(old)
            stable_flags.append(ratio <= Fraction(95, 100))
            regression_flags.append(ratio > 1)
            over_three_flags.append(ratio > Fraction(103, 100))
            row.update({f"run{run}_original_us": old, f"run{run}_original_ms": old / 1000,
                        f"run{run}_contiguous256_us": new, f"run{run}_contiguous256_ms": new / 1000,
                        f"run{run}_original_over_contiguous256": old / new,
                        f"run{run}_time_reduction_percent": gain,
                        f"run{run}_regression_over_3_percent": over_three_flags[-1],
                        f"run{run}_original_range_percent": stats["original"]["range_percent_of_median"],
                        f"run{run}_contiguous256_range_percent": stats["contiguous256"]["range_percent_of_median"],
                        f"run{run}_contiguous256_population_cv_percent": stats["contiguous256"]["population_cv_percent"]})
        row.update(stable_at_least_5_percent=all(stable_flags),
                   any_run_regression=any(regression_flags), every_run_regression=all(regression_flags),
                   any_run_regression_over_3_percent=any(over_three_flags),
                   every_run_regression_over_3_percent=all(over_three_flags),
                   minimum_time_reduction_percent=min(gains), maximum_time_reduction_percent=max(gains),
                   maximum_regression_percent=max(0.0, max(losses)))
        rows.append(row)
    return rows


def group_summary(rows):
    flags = ("stable_at_least_5_percent", "any_run_regression", "every_run_regression",
             "any_run_regression_over_3_percent", "every_run_regression_over_3_percent")
    return {"configurations": len(rows), **{flag: sum(row[flag] for row in rows) for flag in flags},
            "time_reduction_percent_range": [min(row["minimum_time_reduction_percent"] for row in rows), max(row["maximum_time_reduction_percent"] for row in rows)],
            "maximum_regression_percent": max(row["maximum_regression_percent"] for row in rows)}


def analyze(args, inputs):
    protocol = read_json(ROOT / "protocol.json", inputs)
    protocol_sha = inputs[str(ROOT / "protocol.json")]["sha256"]
    require(protocol["protocol_id"] == "nvidia_fused_integration_20260906_v1", "Protocol ID differs")
    holdout = protocol["holdout"]
    require(holdout["rows"] == EXPECTED_ROWS and holdout["exclude_initial_rows"] == INITIAL_ROWS and
            not set(EXPECTED_ROWS).intersection(INITIAL_ROWS), "Holdout domain differs or includes initial rows")
    require(holdout["dims"] == [256] and holdout["dtypes"] == ["fp16", "bf16"] and holdout["normalized"] == [False, True] and
            holdout["expected_configurations"] == 52 and holdout["block_threads"] == 128 and
            holdout["mode"] == "fused_int4_only" and holdout["layouts"] == list(METHODS), "Holdout contract differs")
    for field, value in (("processes", 3), ("groups", 5), ("api_warmup_calls", 25), ("captured_calls", 64), ("replays_per_group", 20), ("graph_warmup_replays", 5)):
        require(protocol["timing"][field] == value, "Timing protocol differs")
    source_path = ROOT / "source_manifest.json"
    source = read_json(source_path, inputs)
    source_sha = inputs[str(source_path)]["sha256"]
    require(source["base_commit"] == BASE_COMMIT and source["raw_working_bytes_preserved"] is True, "Reviewed source snapshot identity differs")
    for name, info in source["files"].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and sha(path.read_bytes()) == info["sha256"], "Source snapshot bytes changed: " + name)
    require(sha((ROOT / "working_diff.patch").read_bytes()) == source["working_diff_sha256"], "Working diff identity changed")
    regression = read_json(args.regression_report, inputs)
    regression_sha = inputs[str(args.regression_report)]["sha256"]
    require(regression["protocol_sha256"] == protocol_sha, "Regression protocol differs")
    binary_sha = verify_regression(regression, source_sha)
    require(len(args.runs) == 3 and len(set(args.runs)) == 3, "Exactly three distinct input reports required")
    reports = [read_json(path, inputs) for path in args.runs]
    require(len({inputs[str(path)]["sha256"] for path in args.runs}) == 3, "Duplicate input report contents")
    require({r["run_index"] for r in reports} == {1, 2, 3} and len({r["pid"] for r in reports}) == 3, "Three unique process/run identities required")
    for field in ("environment", "run_manifest", "regression_gate"):
        require(len({json.dumps(r[field], sort_keys=True) for r in reports}) == 1, "Cross-process identity differs: " + field)
    ordered = sorted(reports, key=lambda r: r["run_index"])
    for previous, current in zip(ordered, ordered[1:]):
        require(datetime.fromisoformat(previous["finished_utc"].replace("Z", "+00:00")) <=
                datetime.fromisoformat(current["started_utc"].replace("Z", "+00:00")), "Worker intervals overlap")
    cases = list(itertools.product(holdout["dtypes"], holdout["dims"], holdout["rows"], holdout["normalized"]))
    indexed, samples, maxima = {}, [], {}
    for report in reports:
        verify_identity(report, protocol, protocol_sha, source_sha, regression_sha, binary_sha)
        require(report["run_manifest"] == regression["run_manifest"], "Regression and holdout execution/source manifests differ")
        require(report["summary"] == {"distinct_configurations": 52, "correctness_input_conditions": 728,
                                      "graph_comparisons": 52, "initial_matrix_samples_imported": 0}, "Worker summary differs from fixed holdout")
        maxima[str(report["run_index"])] = verify_correctness(report, protocol, set(cases))
        timing, raw = verify_timings(report, cases)
        indexed[report["run_index"]] = timing
        samples.extend(raw)
    require(len(samples) == 1560, "Expected 52 * 2 methods * 5 groups * 3 processes = 1560 samples")
    rows = compare_runs(indexed)
    summary = {"status": "VERIFIED", "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha,
               "base_commit": BASE_COMMIT, "source_state": source["source_state"], "source_manifest_sha256": source_sha,
               "regression_report_sha256": regression_sha, "regression_checks": regression["checks"],
               "binary_identities": regression["binaries"], "production_extension_sha256": binary_sha,
               "process_runs": 3, "distinct_holdout_configurations": 52, "distinct_correctness_conditions": 728,
               "raw_timing_samples": 1560, "scope": SCOPE, "dense_max_abs_error_by_run_dtype": maxima,
               "all_52": group_summary(rows),
               "by_dtype": {dtype: group_summary([row for row in rows if row["dtype"] == dtype]) for dtype in ("fp16", "bf16")},
               "all_negative_cases": [row for row in rows if row["any_run_regression"]],
               "all_cases_with_any_regression_over_3_percent": [row for row in rows if row["any_run_regression_over_3_percent"]],
               "counting_policy": "The same 728 conditions repeat in three processes, not 2184 distinct inputs. Regression 1876 modes, original1800/its200 subset, metadata28 and targeted16 are separate scopes and are not summed. Initial24 configurations/samples are excluded.",
               "comparison_policy": "Only same-run, same-shape/dtype/scale fused Graph medians enter each original/contiguous256 comparison. No Dao, transform, initial-matrix, earlier-study or cross-GPU denominators.",
               "threshold_policy": "Decision flags compare exact rational ratios of the recorded floating-point medians; displayed percentages use floating-point arithmetic. Exactly 3% is not >3%; exactly 5% meets >=5%.",
               "claim_boundary": "No automatic selection/dispatch, arbitrary-row guarantee or universal replacement claim. The 28 CSV smoke rows in regression are format checks, not optimization performance evidence.",
               "environment": reports[0]["environment"]}
    return summary, rows, samples


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status", "mode", "scope"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--regression-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.runs = [path.resolve() for path in args.runs]
    args.regression_report = args.regression_report.resolve()
    output = args.output.resolve()
    inputs, rows, samples, summary = {}, [], [], {"status": "UNVERIFIED"}
    try:
        summary, rows, samples = analyze(args, inputs)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, ArithmeticError) as error:
        summary["reason"] = f"{type(error).__name__}: {error}"
    raw_paths = set(args.runs + [args.regression_report, ROOT / "protocol.json", ROOT / "source_manifest.json"])
    require(not any((output / name).resolve() in raw_paths for name in ("analysis.json", "comparison.csv", "samples.csv")), "Output would overwrite a raw input")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "comparison.csv", rows)
    write_csv(output / "samples.csv", samples)
    summary.update(generated_utc=datetime.now(timezone.utc).isoformat(), inputs=inputs, raw_files_modified=False,
                   analysis_script_sha256=sha(Path(__file__).read_bytes()),
                   derived_csv_sha256={name: sha((output / name).read_bytes()) for name in ("comparison.csv", "samples.csv")})
    (output / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "comparison_rows": len(rows), "raw_sample_rows": len(samples),
                      "reason": summary.get("reason"), "output": str(output)}))
    return 0 if summary["status"] == "VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
