#!/usr/bin/env python3
"""Offline three-run analysis for the fixed contiguous256 layout experiment.

python analyze_runs.py --output derived runs/run1.json runs/run2.json runs/run3.json

Pure standard library. Writes analysis.json, comparison.csv, and samples.csv;
never changes raw inputs, starts GPU work, or generates automatic dispatch.
Exit 0 means the evidence was verified, including valid negative findings.
Missing, failed, or inconsistent evidence produces UNVERIFIED and exit 2.
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


ROOT = Path(__file__).resolve().parent
CASE_KEYS = ("dtype", "dim", "rows", "normalized")
EXPECTED_COMMIT = "9f5fdc363b4149d4a211701f24ab0548084ca3e5"
EXPECTED_REFERENCE = "e7706faf8d1c3b9f241e36860640ad1dac644ede"
SCOPE = ("CUDA-event interval divided by 20 replays * 64 retained independent outputs; "
         "median of five groups per process. Captured GPU work and amortized replay scheduling; "
         "not standalone kernel latency, eager API latency, or host end-to-end.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_json(path, files):
    data = path.read_bytes()
    files[str(path)] = {"bytes": len(data), "sha256": sha(data)}
    def reject_constant(value):
        raise ValueError("Non-finite JSON constant: " + value)
    return json.loads(data.decode("utf-8-sig"), parse_constant=reject_constant)


def nonnegative(value, label, positive=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), label + ": non-finite/non-numeric")
    require(value > 0 if positive else value >= 0, label + ": invalid range")
    return float(value)


def close(actual, expected, label):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)), label + ": non-finite")
    require(math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12), label + ": recomputation differs")


def fixed_cases(protocol):
    domain = protocol["domain"]
    require(domain == {"rows": [1, 17, 64, 257, 4096, 16384], "dims": [256],
                       "dtypes": ["fp16", "bf16"], "normalized": [False, True],
                       "expected_configurations": 24}, "Fixed case domain differs")
    return list(itertools.product(domain["dtypes"], domain["dims"], domain["rows"], domain["normalized"]))


def case_key(case):
    require(type(case["normalized"]) is bool, "normalized must be Boolean")
    require(type(case["rows"]) is int and case["rows"] > 0 and case["dim"] == 256, "Invalid case dimensions")
    require(case["shape"] == [case["rows"], 256], "Shape must equal [rows,256]")
    close(case["scale"], 0.0625 if case["normalized"] else 1.0, "Shared float32 scale")
    return tuple(case[name] for name in CASE_KEYS)


def verify_identity(report, protocol, protocol_sha):
    require(report["status"] == "PASS" and report["exit_code"] == 0, "Worker incomplete or failed")
    require(type(report["run_index"]) is int and report["run_index"] in (1, 2, 3), "Invalid run index")
    require(report["protocol_sha256"] == protocol_sha, "Worker protocol SHA256 differs")
    environment, hardware = report["environment"], protocol["hardware"]
    require(hardware["required_name_contains"] in environment["gpu"] and environment["sm"] == hardware["required_sm"], "GPU/SM differs from fixed protocol")
    require(environment["torch_cuda_arch_list"] == hardware["compile_arch"] and environment["max_jobs"] == hardware["max_jobs"], "Compiler settings differ")
    require(re.fullmatch(r"[0-9a-f]{64}", environment["extension_sha256"]), "Missing extension binary SHA256")
    reference = report["reference"]
    require(reference["commit"] == EXPECTED_REFERENCE and reference["repository"] == "https://github.com/Dao-AILab/fast-hadamard-transform", "Reference identity differs")
    require(re.fullmatch(r"[0-9a-f]{64}", reference["cuda_module_sha256"]), "Missing reference binary SHA256")
    require(bool(reference["verification"]), "Reference provenance is absent")
    manifest = report["run_manifest"]["files"]
    require(manifest["protocol.json"]["sha256"] == protocol_sha, "Manifest protocol differs")
    for name in ("source_manifest.json", "run_experiment.py", "measurement_helpers.py", "helper_provenance.json", "sources/kernels.cuh",
                 "sources/contiguous256.cuh", "sources/torch_binding_contiguous256.cu",
                 "sources/compare_reference.py", "sources/build_torch_extension.py"):
        require(manifest[name]["sha256"] == sha((ROOT / name).read_bytes()), "Worker source differs from locally prepared input: " + name)


def verify_correctness(report, protocol, expected):
    patterns = protocol["correctness"]["patterns_and_seeds"]
    inputs = {(pattern, seed, offset) for pattern, seeds in patterns.items()
              for seed in seeds for offset in (0, 2)}
    require(len(inputs) == 14, "Expected seven inputs at two pointer offsets")
    require(protocol["correctness"]["input_pointer_mod16"] == [0, 2], "Pointer offset contract differs")
    found, maxima = set(), {dtype: {"dao": 0.0, "dense": 0.0} for dtype in ("fp16", "bf16")}
    for case in report["correctness"]:
        key = case_key(case)
        require(key in expected and key not in found, "Unexpected/duplicate correctness case")
        found.add(key)
        checks = case["checks"]
        require(len(checks) == 14 and {(entry["pattern"], entry["seed"], entry["pointer_mod16"]) for entry in checks} == inputs, "Correctness input/offset coverage incomplete")
        limit = 0.01 if case["dtype"] == "fp16" else 0.05
        close(protocol["correctness"]["strict_abs_limit"][case["dtype"]], limit, "Protocol strict limit")
        for entry in checks:
            require(entry["non_default_stream"] is False, "Ordinary correctness entry has unexpected stream label")
            require(entry["dao_input_pointer_mod16"] == 0 and
                    entry["dao_aligned_copy_for_offset"] is (entry["pointer_mod16"] == 2) and
                    entry["dao_copy_bitwise_equal_input"] is True, "Dao aligned-reference copy contract failed")
            for flag in ("pass", "cpu_quantization_exact", "original_new_transform_bitwise_exact",
                         "original_new_fused_split_bitwise_exact", "input_guards_unchanged"):
                require(entry[flag] is True, "Correctness flag failed: " + flag)
            require(entry["elements"] == case["rows"] * 256 and entry["quantization_rows_checked"] == case["rows"], "Element/CPU-quantization coverage differs")
            require(entry["dense_rows"] == sorted({0, case["rows"] // 2, case["rows"] - 1}), "Dense oracle row subset differs")
            close(entry["strict_limit"], limit, "Reported strict limit")
            for oracle in ("dao", "dense"):
                error = nonnegative(entry[oracle + "_max_abs_error"], oracle + " error")
                require(error < limit, "Strict numerical limit failed")
                maxima[case["dtype"]][oracle] = max(maxima[case["dtype"]][oracle], error)
    require(found == expected and len(report["correctness"]) == 24, "Correctness matrix must contain 24 configurations")
    stream_expected = {(*key, offset) for key in expected for offset in (0, 2)}
    stream_found = set()
    for entry in report["stream_checks"]:
        key = (*case_key(entry), entry["pointer_mod16"])
        require(key in stream_expected and key not in stream_found and entry["pass"] is True, "Failed/duplicate/unexpected stream check")
        stream_found.add(key)
        require(entry["non_default_stream"] is True and entry["pattern"] == "normal" and entry["seed"] == 2026, "Stream execution/input metadata differs")
        require(entry["dao_input_pointer_mod16"] == 0 and
                entry["dao_aligned_copy_for_offset"] is (entry["pointer_mod16"] == 2) and
                entry["dao_copy_bitwise_equal_input"] is True, "Stream Dao aligned-reference copy contract failed")
        for flag in ("cpu_quantization_exact", "input_guards_unchanged",
                     "original_new_transform_bitwise_exact", "original_new_fused_split_bitwise_exact"):
            require(entry[flag] is True, "Stream correctness flag failed: " + flag)
        limit = 0.01 if entry["dtype"] == "fp16" else 0.05
        close(entry["strict_limit"], limit, "Stream strict limit")
        require(nonnegative(entry["dao_max_abs_error"], "Stream Dao error") < limit and
                nonnegative(entry["dense_max_abs_error"], "Stream dense error") < limit, "Stream error bound failed")
        require(entry["elements"] == entry["rows"] * 256 and entry["quantization_rows_checked"] == entry["rows"], "Stream element/quantization coverage differs")
        require(entry["dense_rows"] == sorted({0, entry["rows"] // 2, entry["rows"] - 1}), "Stream dense row subset differs")
    require(stream_found == stream_expected and len(report["stream_checks"]) == 48, "Stream matrix incomplete")
    ties_found = set()
    for entry in report["quantization_tie_checks"]:
        key = (entry["dtype"], entry["pointer_mod16"])
        require(entry["rows"] == 17 and entry["dim"] == 256, "Standalone tie shape differs")
        require(key not in ties_found and all(entry[flag] is True for flag in ("pass", "cpu_quantization_exact", "input_guards_unchanged")), "Failed/duplicate tie check")
        ties_found.add(key)
    require(ties_found == {(dtype, offset) for dtype in ("fp16", "bf16") for offset in (0, 2)} and len(report["quantization_tie_checks"]) == 4, "Standalone quantization ties incomplete")
    return maxima


def verify_timings(report, protocol, ordered_cases):
    expected_order = list(ordered_cases)
    random.Random(92600 + report["run_index"]).shuffle(expected_order)
    require([case_key(case) for case in report["configuration_order"]] == expected_order, "Configuration order differs from protocol")
    expected_pairs = []
    for case_index, key in enumerate(expected_order):
        modes = ("transform", "fused_int4") if (case_index + report["run_index"]) % 2 == 0 else ("fused_int4", "transform")
        expected_pairs.extend((key, mode) for mode in modes)
    require([(case_key(case), case["mode"]) for case in report["benchmarks"]] == expected_pairs, "Benchmark/mode order differs or matrix is incomplete")
    indexed, raw_rows = {}, []
    for entry in report["benchmarks"]:
        key, mode = case_key(entry), entry["mode"]
        methods = protocol["timing"]["modes"][mode]
        expected_methods = ["original", "contiguous256", "dao"] if mode == "transform" else ["original", "contiguous256"]
        require(methods == expected_methods, "Mode methods differ; Dao is transform-only")
        for field in ("samples_us", "raw_event_intervals_ms", "median_us", "median_ms"):
            require(set(entry[field]) == set(methods), "Raw/median method keys differ")
        for flag in ("independent_output_buffers", "cross_method_output_pointers_disjoint", "outputs_bitwise_equal_eager_before_and_after"):
            require(entry[flag] is True, "Graph output invariant failed: " + flag)
        for field, value in (("captured_calls_per_graph", 64), ("replays_per_group", 20), ("graph_warmup_replays", 5), ("api_warmup_calls", 25)):
            require(entry[field] == value, "Graph timing condition differs: " + field)
        case_index = expected_order.index(key)
        require(entry["configuration_index"] == case_index, "Recorded configuration index differs")
        expected_modes = ["transform", "fused_int4"] if (case_index + report["run_index"]) % 2 == 0 else ["fused_int4", "transform"]
        require(entry["mode_order"] == expected_modes, "Recorded mode order differs")
        orders = []
        for group in range(5):
            offset = (report["run_index"] - 1 + case_index + group) % len(methods)
            orders.append(methods[offset:] + methods[:offset])
        require(entry["group_order"] == orders, "Method group order differs from protocol")
        stats = {}
        for method in methods:
            samples = entry["samples_us"][method]
            intervals = entry["raw_event_intervals_ms"][method]
            require(len(samples) == len(intervals) == 5, "Expected five raw event groups")
            samples = [nonnegative(value, "sample", positive=True) for value in samples]
            intervals = [nonnegative(value, "event interval", positive=True) for value in intervals]
            for group, (us, ms) in enumerate(zip(samples, intervals), 1):
                close(us, ms * 1000 / 1280, "Raw event conversion")
                raw_rows.append({"run_index": report["run_index"], **dict(zip(CASE_KEYS, key)),
                                 "shape": json.dumps(entry["shape"]), "scale": entry["scale"],
                                 "mode": mode, "scope": "cuda_graph", "method": method, "group": group,
                                 "event_interval_ms": ms, "calls_per_interval": 1280,
                                 "per_call_us": us, "per_call_ms": us / 1000})
            median = statistics.median(samples)
            close(entry["median_us"][method], median, "Median microseconds")
            close(entry["median_ms"][method], median / 1000, "Median milliseconds")
            stats[method] = {"median_us": median, "median_ms": median / 1000,
                             "range_percent_of_median": 100 * (max(samples) - min(samples)) / median,
                             "population_cv_percent": 100 * statistics.pstdev(samples) / statistics.mean(samples)}
        indexed[(*key, mode)] = stats
    require(len(indexed) == 48, "Expected 24 configurations and two modes")
    return indexed, raw_rows


def compare_runs(indexed):
    require(set(indexed) == {1, 2, 3}, "Three run indices required")
    require(set(indexed[1]) == set(indexed[2]) == set(indexed[3]), "Cross-run configuration/mode keys differ")
    rows = []
    for key in sorted(indexed[1]):
        dtype, dim, count, normalized, mode = key
        row = dict(zip((*CASE_KEYS, "mode"), key), shape=json.dumps([count, dim]),
                   scale=0.0625 if normalized else 1.0, scope="cuda_graph")
        reductions, regressions, dao_ratios, near_ties = [], [], [], []
        for run in (1, 2, 3):
            stats = indexed[run][key]
            old, new = stats["original"]["median_us"], stats["contiguous256"]["median_us"]
            gain = 100 * (1 - new / old)
            reductions.append(gain)
            regressions.append(100 * (new / old - 1))
            row.update({f"run{run}_original_us": old, f"run{run}_original_ms": old / 1000,
                        f"run{run}_contiguous256_us": new, f"run{run}_contiguous256_ms": new / 1000,
                        f"run{run}_original_over_contiguous256": old / new,
                        f"run{run}_time_reduction_percent": gain,
                        f"run{run}_original_range_percent": stats["original"]["range_percent_of_median"],
                        f"run{run}_contiguous256_range_percent": stats["contiguous256"]["range_percent_of_median"],
                        f"run{run}_contiguous256_population_cv_percent": stats["contiguous256"]["population_cv_percent"]})
            if mode == "transform":
                dao = stats["dao"]["median_us"]
                dao_ratios.append(dao / new)
                near_ties.append(abs(new / dao - 1) <= 0.01)
                row.update({f"run{run}_dao_us": dao, f"run{run}_dao_over_original": dao / old,
                            f"run{run}_dao_over_contiguous256": dao / new,
                            f"run{run}_time_reduction_vs_dao_percent": 100 * (1 - new / dao),
                            f"run{run}_near_tie_with_dao": near_ties[-1]})
        row.update(stable_at_least_5_percent=all(value >= 5 for value in reductions),
                   any_run_regression=any(value < 0 for value in reductions),
                   every_run_regression=all(value < 0 for value in reductions),
                   any_run_regression_over_3_percent=any(value < -3 for value in reductions),
                   every_run_regression_over_3_percent=all(value < -3 for value in reductions),
                   minimum_time_reduction_percent=min(reductions), maximum_time_reduction_percent=max(reductions),
                   maximum_regression_percent=max(0.0, max(regressions)))
        if mode == "transform":
            row.update(all_runs_no_slower_than_dao=all(value >= 1 for value in dao_ratios),
                       all_runs_faster_than_dao_by_more_than_1_percent=all(1 / value < 0.99 for value in dao_ratios),
                       any_run_near_tie_with_dao=any(near_ties))
        rows.append(row)
    return rows


def summarize_group(rows):
    flags = ("stable_at_least_5_percent", "any_run_regression", "every_run_regression",
             "any_run_regression_over_3_percent", "every_run_regression_over_3_percent")
    result = {"configurations": len(rows), **{flag: sum(row[flag] for row in rows) for flag in flags},
              "time_reduction_percent_range": [min(row["minimum_time_reduction_percent"] for row in rows), max(row["maximum_time_reduction_percent"] for row in rows)],
              "maximum_regression_percent": max(row["maximum_regression_percent"] for row in rows)}
    if all(row["mode"] == "transform" for row in rows):
        result.update({flag: sum(row[flag] for row in rows) for flag in ("all_runs_no_slower_than_dao", "all_runs_faster_than_dao_by_more_than_1_percent", "any_run_near_tie_with_dao")})
    return result


def analyze(args, files):
    protocol = load_json(args.protocol, files)
    protocol_sha = files[str(args.protocol)]["sha256"]
    require(protocol_sha == sha((ROOT / "protocol.json").read_bytes()), "Protocol differs from locally frozen input")
    require(protocol["protocol_id"] == "nvidia_contiguous256_20260906_v1" and protocol["source_commit"] == EXPECTED_COMMIT and
            protocol["reference_commit"] == EXPECTED_REFERENCE, "Protocol/source/reference identity differs")
    source_manifest = load_json(ROOT / "source_manifest.json", files)
    require(source_manifest["source_commit"] == EXPECTED_COMMIT, "Local source manifest commit differs")
    require(protocol["launch"] == {"block_threads": 128, "layouts": ["original", "contiguous256"], "default_layout": "original", "automatic_dispatch": False}, "Launch contract differs")
    for field, expected in (("processes", 3), ("groups", 5), ("api_warmup_calls", 25), ("captured_calls", 64), ("replays_per_group", 20), ("graph_warmup_replays", 5)):
        require(protocol["timing"][field] == expected, "Timing protocol differs: " + field)
    ordered_cases = fixed_cases(protocol)
    require(len(args.runs) == 3 and len(set(args.runs)) == 3, "Exactly three distinct run files are required")
    reports = [load_json(path, files) for path in args.runs]
    require(len({files[str(path)]["sha256"] for path in args.runs}) == 3, "Duplicate report contents")
    require({report["run_index"] for report in reports} == {1, 2, 3}, "Three unique run indices are required")
    require(len({report["pid"] for report in reports}) == 3, "Three distinct worker PIDs are required")
    for field in ("environment", "reference", "run_manifest"):
        require(len({json.dumps(report[field], sort_keys=True) for report in reports}) == 1, "Cross-process identity differs: " + field)
    sequence = sorted(reports, key=lambda report: report["run_index"])
    for previous, current in zip(sequence, sequence[1:]):
        require(datetime.fromisoformat(previous["finished_utc"].replace("Z", "+00:00")) <=
                datetime.fromisoformat(current["started_utc"].replace("Z", "+00:00")), "Worker intervals overlap")
    indexed, samples, maxima = {}, [], {}
    for report in reports:
        verify_identity(report, protocol, protocol_sha)
        require(report["summary"] == {"unique_configurations": 24, "correctness_input_cases": 336,
                                      "stream_checks": 48, "quantization_tie_checks": 4, "graph_comparisons": 48}, "Worker summary differs from fixed matrix")
        maxima[str(report["run_index"])] = verify_correctness(report, protocol, set(ordered_cases))
        timing, raw = verify_timings(report, protocol, ordered_cases)
        indexed[report["run_index"]] = timing
        samples.extend(raw)
    rows = compare_runs(indexed)
    summary = {"status": "VERIFIED", "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha,
               "source_commit": EXPECTED_COMMIT, "reference_commit": EXPECTED_REFERENCE,
               "process_runs": 3, "distinct_configurations": 24, "distinct_correctness_inputs": 336,
               "separate_stream_check_configurations": 48, "separate_quantization_tie_checks": 4,
               "counting_note": "Each of 24 configurations has seven inputs and two pointer offsets. The same 336 correctness inputs repeat across two layouts and three runs; do not multiply them into new cases. Stream and quantization-tie checks are separate scopes. Two timing modes produce 48 paired configuration/mode comparisons, not 48 distinct shapes.",
               "scope": SCOPE, "max_abs_error_by_run_dtype": maxima,
               "groups_by_mode": {mode: summarize_group([row for row in rows if row["mode"] == mode]) for mode in ("transform", "fused_int4")},
               "all_negative_cases": [row for row in rows if row["any_run_regression"]],
               "all_cases_with_any_regression_over_3_percent": [row for row in rows if row["any_run_regression_over_3_percent"]],
               "m17_n256_transform_by_dtype_scale": [row for row in rows if row["rows"] == 17 and row["mode"] == "transform"],
               "environment": reports[0]["environment"],
               "comparison_policy": "Only same-run, same-configuration and same-mode Graph medians enter each ratio. Dao comparisons exist only for transform. No earlier study, other GPU, eager, or isolated-kernel denominators.",
               "claim_boundary": "Evidence analysis only. No automatic selection/dispatch or universal replacement claim. <=1% Dao differences are near-ties. Hardware and process evidence does not establish exclusive GPU tenancy.",
               "output_alignment_boundary": protocol["correctness"]["output_alignment_boundary"]}
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
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.protocol = args.protocol.resolve()
    args.runs = [path.resolve() for path in args.runs]
    output = args.output.resolve()
    files, rows, samples = {}, [], []
    summary = {"status": "UNVERIFIED"}
    try:
        summary, rows, samples = analyze(args, files)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, ArithmeticError) as error:
        summary["reason"] = f"{type(error).__name__}: {error}"
    targets = [output / name for name in ("analysis.json", "comparison.csv", "samples.csv")]
    require(not any(path.resolve() in set(args.runs + [args.protocol]) for path in targets), "Output would overwrite raw input")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "comparison.csv", rows)
    write_csv(output / "samples.csv", samples)
    summary.update(generated_utc=datetime.now(timezone.utc).isoformat(), inputs=files, raw_files_modified=False,
                   analysis_script_sha256=sha(Path(__file__).read_bytes()),
                   derived_csv_sha256={name: sha((output / name).read_bytes()) for name in ("comparison.csv", "samples.csv")})
    (output / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "comparison_rows": len(rows), "sample_rows": len(samples),
                      "reason": summary.get("reason"), "output": str(output)}))
    return 0 if summary["status"] == "VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
