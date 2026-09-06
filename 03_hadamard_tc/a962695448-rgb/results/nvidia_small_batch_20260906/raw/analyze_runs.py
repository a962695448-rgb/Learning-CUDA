#!/usr/bin/env python3
"""Analyze three small_batch thread_config runs without starting GPU work.

python analyze_runs.py --phase screen --output derived/screen run1.json run2.json run3.json
python analyze_runs.py --phase validation --selection derived/screen/selection.json \
    --output derived/validation holdout1.json holdout2.json holdout3.json

Writes analysis.json, comparison.csv, samples.csv, and screen selection.json.
Exit 0: verified inputs, including valid negative findings. Exit 2: UNVERIFIED.
Raw input files are never modified; no production dispatch is generated.
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
import struct
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parent
CASE_KEYS = ("dtype", "dim", "rows", "normalized")
SCOPES = {
    "graph": "CUDA-event interval / (20 replays * 64 retained independent outputs); median of five groups. Captured GPU work and amortized replay scheduling, not isolated kernel latency or host end-to-end.",
    "eager": "CUDA-event interval / 200 allocating Python API calls; median of five groups. Includes host-dispatch GPU idle gaps, not pure kernel latency or CPU wall-clock end-to-end.",
}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_json(path, files):
    data = path.read_bytes()
    files[str(path)] = {"bytes": len(data), "sha256": sha(data)}
    def reject(value):
        raise ValueError("Non-finite JSON value: " + value)
    return json.loads(data.decode("utf-8-sig"), parse_constant=reject)


def number(value, positive=False):
    check(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), "Invalid numeric value")
    check(value > 0 if positive else value >= 0, "Numeric value outside valid range")
    return float(value)


def close(actual, expected, label):
    check(math.isfinite(float(actual)) and math.isfinite(float(expected)), label + ": non-finite")
    check(math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12), label + ": recomputation differs")


def configurations(protocol, phase):
    domain = protocol["domain"]
    all_cases = set(itertools.product(domain["dtypes"], domain["dims"], domain["rows"], domain["normalized"]))
    screen = set(itertools.product(domain["dtypes"], protocol["screen"]["dims"],
                                   protocol["screen"]["rows"], domain["normalized"]))
    check(len(screen) == protocol["screen"]["expected_configurations"] == 24, "Screen domain differs")
    check(screen <= all_cases and len(all_cases - screen) == protocol["validation"]["expected_holdout_configurations"] == 120, "Holdout domain differs")
    return screen if phase == "screen" else all_cases - screen


def case_key(case):
    check(type(case["normalized"]) is bool, "normalized must be Boolean")
    check(case["shape"][-1] == case["dim"] and math.prod(case["shape"][:-1]) == case["rows"], "Shape disagrees with row/dimension fields")
    check(len(case["shape"]) in (2, 4) and all(type(x) is int and x > 0 for x in case["shape"]), "Invalid shape")
    scale = 1 / math.sqrt(case["dim"]) if case["normalized"] else 1.0
    scale = struct.unpack("f", struct.pack("f", scale))[0]
    close(case["scale"], scale, "Float32-rounded scale")
    return tuple(case[name] for name in CASE_KEYS)


def verify_correctness(report, protocol, expected, threads):
    found, signatures = set(), {}
    patterns = protocol["correctness"]["patterns_and_seeds"]
    expected_inputs = {(pattern, seed) for pattern, seeds in patterns.items() for seed in seeds}
    check(len(expected_inputs) == 7, "Correctness pattern/seed contract differs")
    maxima = {dtype: {"dao": 0.0, "dense": 0.0} for dtype in protocol["domain"]["dtypes"]}
    for case in report["correctness"]:
        key = case_key(case)
        check(key in expected and key not in found, "Unexpected/duplicate correctness case")
        found.add(key)
        signatures[key] = (tuple(case["shape"]), case["scale"])
        entries = case["checks"]
        check(len(entries) == 7 and {(x["pattern"], x["seed"]) for x in entries} == expected_inputs, "Correctness input matrix incomplete")
        limit = protocol["correctness"]["strict_abs_limit"][case["dtype"]]
        check(limit == (0.01 if case["dtype"] == "fp16" else 0.05), "Error limit differs")
        for entry in entries:
            for field in ("pass", "all_threads_transform_bitwise_exact", "all_threads_fused_split_bitwise_exact", "cpu_quantization_exact"):
                check(entry[field] is True, "Correctness check failed: " + field)
            check(entry["threads_checked"] == threads, "Incomplete correctness thread coverage")
            check(entry["elements"] == case["rows"] * case["dim"] and entry["quantization_rows_checked"] == case["rows"], "Incomplete element/quantization coverage")
            check(entry["dense_rows"] == sorted({0, case["rows"] // 2, case["rows"] - 1}), "Dense row subset differs")
            close(entry["strict_limit"], limit, "Strict error limit")
            for name in ("dao", "dense"):
                value = number(entry[name + "_max_abs_error"])
                check(value < limit, "Strict correctness error bound exceeded")
                maxima[case["dtype"]][name] = max(maxima[case["dtype"]][name], value)
    check(found == expected and len(report["correctness"]) == len(expected), "Correctness case matrix incomplete")
    return signatures, maxima


def verify_run(report, protocol, protocol_sha, phase, threads, expected):
    check(report["status"] == "PASS" and report["phase"] == phase, "Worker failed or phase differs")
    check(report["exit_code"] == 0, "Worker exit code is not zero")
    check(report["protocol_sha256"] == protocol_sha, "Frozen protocol SHA256 differs")
    check(type(report["run_index"]) is int and report["run_index"] in (1, 2, 3), "Invalid run_index")
    check(report["threads"] == threads, "Thread set/order differs")
    environment = report["environment"]
    hardware = protocol["hardware"]
    check(hardware["required_name_contains"] in environment["gpu"] and environment["sm"] == hardware["required_sm"], "Worker GPU/SM differs from protocol")
    check(environment["torch_cuda_arch_list"] == hardware["compile_arch"] and environment["max_jobs"] == hardware["max_jobs"], "Worker compiler settings differ")
    check(re.fullmatch(r"[0-9a-f]{64}", environment["extension_sha256"]), "Invalid extension SHA256")
    check(report["reference"]["commit"] == protocol["reference_commit"] and
          report["reference"]["repository"] == "https://github.com/Dao-AILab/fast-hadamard-transform", "Reference identity differs")
    check(re.fullmatch(r"[0-9a-f]{64}", report["reference"]["cuda_module_sha256"]), "Invalid reference binary SHA256")
    manifest = report["run_manifest"]["files"]
    check(manifest["protocol.json"]["sha256"] == protocol_sha, "Worker manifest protocol differs")
    for name in ("source_manifest.json", "run_experiment.py", "sources/kernels.cuh",
                 "sources/torch_binding_thread_config.cu", "sources/compare_reference.py",
                 "sources/build_torch_extension.py"):
        check(manifest[name]["sha256"] == sha((ROOT / name).read_bytes()), "Worker source differs from local fixed input: " + name)
    signatures, maxima = verify_correctness(report, protocol, expected, threads)
    methods = [f"thread{thread}" for thread in threads] + ["dao"]
    domain = protocol["domain"]
    order = [key for key in itertools.product(domain["dtypes"], domain["dims"], domain["rows"], domain["normalized"]) if key in expected]
    random.Random(92000 + report["run_index"]).shuffle(order)
    check([case_key(c) for c in report["configuration_order"]] == order, "Configuration order differs from protocol")
    check([case_key(c) for c in report["benchmarks"]] == order, "Benchmark order differs from recorded configuration order")
    timings, raw_rows = {}, []
    for case_index, case in enumerate(report["benchmarks"]):
        key = case_key(case)
        check(case["mode"] == "transform", "Unexpected benchmark mode")
        scope_order = ["graph", "eager"] if (case_index + report["run_index"]) % 2 == 0 else ["eager", "graph"]
        check(case["scope_order"] == scope_order, "Scope order differs from protocol")
        check(key in expected and key not in timings, "Unexpected/duplicate benchmark case")
        check((tuple(case["shape"]), case["scale"]) == signatures[key], "Timing and correctness shape/scale differ")
        scopes = {}
        for scope in SCOPES:
            values, result = case[scope], {}
            denominator = 20 * 64 if scope == "graph" else 200
            names = list(methods)
            random.Random(91000 + report["run_index"]).shuffle(names)
            orders = [names[(group + case_index) % len(names):] + names[:(group + case_index) % len(names)] for group in range(5)]
            check(values["group_order"] == orders, "Method/group order differs from protocol")
            check(values["api_warmup_calls"] == 25, "API warmup differs")
            if scope == "graph":
                for field in ("independent_output_buffers", "cross_method_output_pointers_disjoint", "outputs_bitwise_equal_eager_before_and_after"):
                    check(values[field] is True, "Graph output check failed: " + field)
                check(values["captured_calls_per_graph"] == 64 and values["replays_per_group"] == 20 and values["graph_warmup_replays"] == 5, "Graph timing settings differ")
            else:
                check(values["calls_per_group"] == 200, "Eager timing settings differ")
            for field in ("samples_us", "raw_event_intervals_ms", "median_us", "median_ms"):
                check(set(values[field]) == set(methods), scope + ": methods differ")
            for method in methods:
                samples, intervals = values["samples_us"][method], values["raw_event_intervals_ms"][method]
                check(len(samples) == len(intervals) == 5, "Expected five raw groups")
                samples = [number(value, positive=True) for value in samples]
                intervals = [number(value, positive=True) for value in intervals]
                for group, (us, ms) in enumerate(zip(samples, intervals), 1):
                    close(us, ms * 1000 / denominator, "Raw-event conversion")
                    raw_rows.append({"phase": phase, "run_index": report["run_index"], **dict(zip(CASE_KEYS, key)),
                                     "shape": json.dumps(case["shape"]), "scale": case["scale"],
                                     "scope": scope, "method": method, "group": group,
                                     "event_interval_ms": ms, "calls_per_interval": denominator,
                                     "per_call_us": us, "per_call_ms": us / 1000})
                median = statistics.median(samples)
                close(values["median_us"][method], median, "Median microseconds")
                close(values["median_ms"][method], median / 1000, "Median milliseconds")
                result[method] = {"median_us": median, "median_ms": median / 1000,
                                  "range_percent_of_median": 100 * (max(samples) - min(samples)) / median,
                                  "population_cv_percent": 100 * statistics.pstdev(samples) / statistics.mean(samples)}
            scopes[scope] = result
        timings[key] = {"timings": scopes, "shape": signatures[key][0], "scale": case["scale"]}
    check(set(timings) == expected and len(report["benchmarks"]) == len(expected), "Benchmark matrix incomplete")
    return timings, raw_rows, maxima


def compare_runs(indexed, phase, threads, expected):
    rows = []
    for key in sorted(expected):
        signatures = {(indexed[run][key]["shape"], indexed[run][key]["scale"]) for run in (1, 2, 3)}
        check(len(signatures) == 1, "Shape/scale changes between processes")
        shape, scale = next(iter(signatures))
        for scope in SCOPES:
            for thread in threads:
                row = {"phase": phase, **dict(zip(CASE_KEYS, key)), "shape": json.dumps(shape), "scale": scale,
                       "scope": scope, "threads": thread, "method": f"thread{thread}"}
                reductions, dao_ratios, relative_regressions, near_ties = [], [], [], []
                for run in (1, 2, 3):
                    timings = indexed[run][key]["timings"][scope]
                    candidate = timings[f"thread{thread}"]["median_us"]
                    baseline = timings["thread128"]["median_us"]
                    dao = timings["dao"]["median_us"]
                    gain = 100 * (1 - candidate / baseline)
                    dao_ratio = dao / candidate
                    reductions.append(gain)
                    dao_ratios.append(dao_ratio)
                    relative_regressions.append(100 * (candidate / baseline - 1))
                    near_ties.append(abs(candidate / dao - 1) <= 0.01)
                    row.update({f"run{run}_candidate_us": candidate, f"run{run}_candidate_ms": candidate / 1000,
                                f"run{run}_thread128_us": baseline, f"run{run}_dao_us": dao,
                                f"run{run}_time_reduction_vs128_percent": gain,
                                f"run{run}_dao_over_candidate": dao_ratio,
                                f"run{run}_time_reduction_vs_dao_percent": 100 * (1 - candidate / dao),
                                f"run{run}_near_tie_with_dao": near_ties[-1],
                                f"run{run}_range_percent_of_median": timings[f"thread{thread}"]["range_percent_of_median"],
                                f"run{run}_population_cv_percent": timings[f"thread{thread}"]["population_cv_percent"]})
                row.update(stable_at_least_5_percent=all(g >= 5 for g in reductions),
                           any_run_regression=any(g < 0 for g in reductions),
                           every_run_regression=all(g < 0 for g in reductions),
                           any_run_regression_over_3_percent=any(g < -3 for g in reductions),
                           every_run_regression_over_3_percent=all(g < -3 for g in reductions),
                           minimum_time_reduction_percent=min(reductions), maximum_time_reduction_percent=max(reductions),
                           maximum_regression_vs128_percent=max(0.0, max(relative_regressions)),
                           all_runs_no_slower_than_dao=all(r >= 1 for r in dao_ratios),
                           all_runs_faster_than_dao_by_more_than_1_percent=all(1 / r < 0.99 for r in dao_ratios),
                           any_run_near_tie_with_dao=any(near_ties))
                rows.append(row)
    return rows


def group_summary(rows):
    flags = ("stable_at_least_5_percent", "any_run_regression", "every_run_regression",
             "any_run_regression_over_3_percent", "every_run_regression_over_3_percent",
             "all_runs_no_slower_than_dao", "all_runs_faster_than_dao_by_more_than_1_percent")
    return {"case_comparisons": len(rows), **{flag: sum(row[flag] for row in rows) for flag in flags},
            "maximum_regression_vs128_percent": max(row["maximum_regression_vs128_percent"] for row in rows),
            "time_reduction_percent_range": [min(row["minimum_time_reduction_percent"] for row in rows),
                                              max(row["maximum_time_reduction_percent"] for row in rows)]}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status", "phase", "scope"]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(args, files):
    protocol = load_json(args.protocol, files)
    protocol_sha = files[str(args.protocol)]["sha256"]
    check(protocol_sha == sha((ROOT / "protocol.json").read_bytes()), "Protocol differs from local frozen input")
    check(protocol["protocol_id"] == "nvidia_small_batch_thread_config_20260906_v1", "Unexpected protocol ID")
    check(protocol["timing"]["processes"] == 3 and protocol["timing"]["groups"] == 5, "Timing protocol differs")
    check(protocol["timing"]["graph"]["captured_calls"] == 64 and protocol["timing"]["graph"]["replays_per_group"] == 20 and
          protocol["timing"]["eager"]["calls_per_group"] == 200, "Timing denominators differ")
    expected = configurations(protocol, args.phase)
    if args.phase == "screen":
        check(args.selection is None, "Screen does not consume a selection file")
        threads = protocol["screen"]["threads"]
        check(threads == [32, 64, 128, 256], "Screen thread set differs")
    else:
        check(args.selection is not None, "Validation requires --selection from verified screen analysis")
        selection = load_json(args.selection, files)
        check(selection["status"] == "PASS" and selection["phase"] == "screen" and selection["protocol_sha256"] == protocol_sha, "Screen selection is not verified under this protocol")
        check(selection["screen_configurations"] == 24 and selection["process_runs"] == 3 and
              selection["selection_scope"] == "graph" and selection["minimum_reduction_percent_in_each_run"] == 5, "Screen selection rule differs")
        selected = selection["selected_threads"]
        check(selected == sorted(set(selected)) and selected and set(selected) <= {32, 64}, "No valid selected threads; do not run a new search")
        for thread in selected:
            witnesses = selection["qualifying_graph_cases"][str(thread)]
            check(witnesses and all(len(w["time_reduction_percent_by_run"]) == 3 and
                  all(math.isfinite(v) and v >= 5 for v in w["time_reduction_percent_by_run"]) for w in witnesses), "Selection lacks three-run >=5% Graph evidence")
            check(all(tuple(w[name] for name in CASE_KEYS) in configurations(protocol, "screen") for w in witnesses), "Selection witness is outside screen domain")
        threads = sorted(set(selected + protocol["validation"]["control_threads"]))
    check(len(args.runs) == 3 and len(set(args.runs)) == 3, "Exactly three distinct run files are required")
    reports = [load_json(path, files) for path in args.runs]
    check(len({files[str(path)]["sha256"] for path in args.runs}) == 3, "Duplicate run file contents")
    check({r["run_index"] for r in reports} == {1, 2, 3}, "run_index must be exactly 1, 2, 3")
    check(len({r["pid"] for r in reports}) == 3, "Three distinct worker processes are required")
    for field in ("environment", "run_manifest", "reference"):
        check(len({json.dumps(r[field], sort_keys=True) for r in reports}) == 1, "Cross-process identity differs: " + field)
    ordered_reports = sorted(reports, key=lambda r: r["run_index"])
    for previous, current in zip(ordered_reports, ordered_reports[1:]):
        check(datetime.fromisoformat(previous["finished_utc"].replace("Z", "+00:00")) <=
              datetime.fromisoformat(current["started_utc"].replace("Z", "+00:00")), "Worker time windows overlap")
    if args.phase == "validation":
        for report in reports:
            check(report["selection"]["data"] == selection and report["selection"]["sha256"] == files[str(args.selection)]["sha256"], "Validation used a different screen selection")
    indexed, samples, maxima = {}, [], {}
    for report in reports:
        timing, raw, maximum = verify_run(report, protocol, protocol_sha, args.phase, threads, expected)
        indexed[report["run_index"]] = timing
        samples.extend(raw)
        maxima[str(report["run_index"])] = maximum
    rows = compare_runs(indexed, args.phase, threads, expected)
    summary = {"status": "VERIFIED", "phase": args.phase, "protocol_id": protocol["protocol_id"],
               "protocol_sha256": protocol_sha, "source_commit": protocol["source_commit"],
               "reference_commit": protocol["reference_commit"], "threads": threads,
               "process_runs": 3, "distinct_configurations_in_phase": len(expected),
               "distinct_correctness_inputs_in_phase": len(expected) * 7,
               "counting_note": "The same correctness inputs repeat across threads and three runs. Graph/eager are separate measurements of the same configurations, not additional independent input cases. Screen 24 and disjoint holdout 120 may only be presented as 144 predefined configurations with phase labels retained.",
               "scopes": SCOPES, "max_abs_error_by_run_dtype": maxima,
               "groups_by_scope_and_threads": {scope: {str(thread): group_summary([r for r in rows if r["scope"] == scope and r["threads"] == thread]) for thread in threads} for scope in SCOPES},
               "all_negative_cases": [r for r in rows if r["any_run_regression"]],
               "m17_n256_by_dtype_scale_scope_threads": [r for r in rows if r["rows"] == 17 and r["dim"] == 256],
               "comparison_policy": "Only same-case, same-scale, same-scope, same-run medians enter each ratio. No A100, earlier study, or cross-device denominators. No automatic dispatch.",
               "environment": reports[0]["environment"],
               "hardware_evidence_boundary": "Worker-recorded RTX4090 sm89, toolchain, binary/source hashes and nonoverlapping process windows are checked; no independent claim of exclusive physical GPU tenancy."}
    selection = None
    if args.phase == "screen":
        qualifying = {}
        for thread in protocol["selection"]["eligible_threads"]:
            qualifying[str(thread)] = [{**{name: row[name] for name in CASE_KEYS}, "scale": row["scale"],
                                       "time_reduction_percent_by_run": [row[f"run{i}_time_reduction_vs128_percent"] for i in (1, 2, 3)]}
                                      for row in rows if row["scope"] == "graph" and row["threads"] == thread and row["stable_at_least_5_percent"]]
        selection = {"status": "PASS", "phase": "screen", "protocol_sha256": protocol_sha,
                     "selected_threads": sorted(int(thread) for thread, cases in qualifying.items() if cases),
                     "qualifying_graph_cases": qualifying, "screen_configurations": 24, "process_runs": 3,
                     "selection_scope": "graph", "minimum_reduction_percent_in_each_run": 5,
                     "production_dispatch": False,
                     "input_runs": {str(path): files[str(path)] for path in args.runs}}
        selection["next_action"] = "RUN_FIXED_HOLDOUT" if selection["selected_threads"] else "STOP_NO_QUALIFYING_THREAD"
        summary["selected_threads"] = selection["selected_threads"]
    return summary, rows, samples, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--phase", choices=("screen", "validation"), required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.protocol = args.protocol.resolve()
    args.runs = [path.resolve() for path in args.runs]
    args.selection = args.selection.resolve() if args.selection else None
    output = args.output.resolve()
    files, rows, samples, selection = {}, [], [], None
    summary = {"status": "UNVERIFIED", "phase": args.phase}
    try:
        summary, rows, samples, selection = analyze(args, files)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, ArithmeticError) as error:
        summary["reason"] = f"{type(error).__name__}: {error}"
    summary.update(generated_utc=datetime.now(timezone.utc).isoformat(), inputs=files,
                   raw_files_modified=False, analysis_script_sha256=sha(Path(__file__).read_bytes()))
    targets = [output / name for name in ("analysis.json", "comparison.csv", "samples.csv", "selection.json")]
    raw_paths = set(args.runs + [args.protocol] + ([args.selection] if args.selection else []))
    check(not any(path.resolve() in raw_paths for path in targets), "Output would overwrite a raw input")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "comparison.csv", rows)
    write_csv(output / "samples.csv", samples)
    if args.phase == "screen":
        # Replace any earlier selection with UNVERIFIED on a failed rerun; never leave a stale PASS.
        selection = selection or {"status": "UNVERIFIED", "phase": "screen", "selected_threads": [],
                                  "reason": summary.get("reason"), "production_dispatch": False}
        (output / "selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary["derived_csv_sha256"] = {name: sha((output / name).read_bytes()) for name in ("comparison.csv", "samples.csv")}
    (output / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "phase": args.phase, "selected_threads": summary.get("selected_threads"),
                      "comparison_rows": len(rows), "reason": summary.get("reason"), "output": str(output)}))
    return 0 if summary["status"] == "VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
