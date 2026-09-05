"""复算三轮独立输出 CUDA Graph 线程候选结果；不读取旧 fixed-buffer 数值。"""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
KEYS = ("dtype", "dim", "rows", "normalized", "mode")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = [RESULTS / f"run{run}.json" for run in (1, 2, 3)]
    reports = [json.loads(path.read_text()) for path in paths]
    indexed = []
    for i, report in enumerate(reports, start=1):
        assert report["status"] == "PASS" and report["run_index"] == i
        assert report["summary"] == {"unique_shape_dtype_scale_cases": 48,
            "correctness_input_cases": 336, "benchmark_configurations": 72}
        for case in report["correctness"]:
            assert len(case["checks"]) == 7
            for check in case["checks"]:
                assert check["all_elements_baseline_candidate_bitwise_exact"] and check["cpu_quantization_exact"]
                assert check["dao_max_abs_error"] < check["strict_limit"]
                assert check["dense_max_abs_error"] < check["strict_limit"]
        entries = {}
        for entry in report["benchmarks"]:
            key = tuple(entry[name] for name in KEYS)
            assert key not in entries
            assert entry["captured_outputs_per_graph"] == 64 and entry["independent_output_buffers"]
            assert entry["captured_outputs_bitwise_equal_eager_before_and_after"]
            assert entry["groups"] == 5 and entry["replays_per_group"] == 20
            for name, values in entry["samples_us"].items():
                assert len(values) == 5 and all(value > 0 for value in values)
                assert statistics.median(values) == entry["median_us"][name]
            entries[key] = entry
        assert len(entries) == 72
        indexed.append(entries)
    assert set(indexed[0]) == set(indexed[1]) == set(indexed[2])
    assert len({r["environment"]["extension_sha256"] for r in reports}) == 1
    assert len({r["experiment_script_sha256"] for r in reports}) == 1
    assert len({json.dumps(r["source_manifest"], sort_keys=True) for r in reports}) == 1
    rows = []
    for key in sorted(indexed[0]):
        entries = [run[key] for run in indexed]
        row = dict(zip(KEYS, key), original_target=entries[0]["original_target"])
        reductions = []
        for run, entry in enumerate(entries, start=1):
            values = entry["median_us"]
            gain = 100 * (1 - values["candidate256"] / values["baseline128"])
            reductions.append(gain)
            row.update({f"run{run}_baseline128_us": values["baseline128"],
                        f"run{run}_candidate256_us": values["candidate256"],
                        f"run{run}_time_reduction_percent": gain,
                        f"run{run}_dao_us": values.get("dao", ""),
                        f"run{run}_dao_over_candidate": values["dao"] / values["candidate256"] if "dao" in values else ""})
        row.update(min_time_reduction_percent=min(reductions), max_time_reduction_percent=max(reductions),
                   stable_at_least_5_percent=min(reductions) >= 5,
                   any_run_regression_over_3_percent=min(reductions) < -3,
                   every_run_regression_over_3_percent=max(reductions) < -3)
        rows.append(row)
    with (RESULTS / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def group(predicate):
        subset = [row for row in rows if predicate(row)]
        return {"cases": len(subset), "stable_at_least_5_percent": sum(r["stable_at_least_5_percent"] for r in subset),
                "any_run_regression_over_3_percent": sum(r["any_run_regression_over_3_percent"] for r in subset),
                "every_run_regression_over_3_percent": sum(r["every_run_regression_over_3_percent"] for r in subset),
                "minimum_time_reduction_percent": min(r["min_time_reduction_percent"] for r in subset),
                "maximum_time_reduction_percent": max(r["max_time_reduction_percent"] for r in subset)}
    summary = {"status": "THREE_RUNS_VERIFIED", "metric": "Independent-output CUDA Graph mean per call, median of 5 groups; three separate process runs paired within run. Not kernel-only or end-to-end.",
               "binary_sha256": reports[0]["environment"]["extension_sha256"],
               "experiment_script_sha256": reports[0]["experiment_script_sha256"],
               "distinct_correctness_input_cases": 336, "all_cases": group(lambda row: True),
               "original_24_targets": group(lambda row: row["original_target"]),
               "adjacent_48_cases": group(lambda row: not row["original_target"]),
               "by_mode": {mode: group(lambda row: row["mode"] == mode) for mode in ("transform", "fused_int4")},
               "strongest_cases": sorted(rows, key=lambda row: row["min_time_reduction_percent"], reverse=True)[:8],
               "weakest_cases_including_regressions": sorted(rows, key=lambda row: row["min_time_reduction_percent"])[:12],
               "environment": reports[0]["environment"],
               "limitations": ["Single RTX4090 session, not A100 or cross-date evidence.",
                   "Resident GPU context identity incompletely mapped; no exclusive-device claim.",
                   "Only N16/N64 and registered original/adjacent M ranges measured; no global optimum claim.",
                   "No division by prior fixed-buffer or previously recorded Dao timings.",
                   "Production sources and default dispatch remain unchanged."]}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {"files": {path.name: {"bytes": path.stat().st_size, "sha256": sha(path)}
                          for path in sorted(RESULTS.iterdir()) if path.is_file() and path.name != "analysis_manifest.json"}}
    (RESULTS / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("environment", "strongest_cases", "weakest_cases_including_regressions", "limitations")}, indent=2))


if __name__ == "__main__":
    main()
