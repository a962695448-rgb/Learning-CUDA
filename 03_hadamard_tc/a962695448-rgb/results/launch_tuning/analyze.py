"""复算本目录三个已归档发射实验；只用 Python 标准库。"""
import csv
import hashlib
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASE = ("mode", "rows", "dim", "dtype", "scale")
SOURCE_SHA256 = "8a9899f5180180752af20659aca31275279e3c2b1fc1d1baaea69a8b87198212"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze():
    samples = defaultdict(dict)
    raw_files = []
    environment = set()
    cases_by_run = []
    for run in (1, 2, 3):
        stem = ROOT / f"launch_sweep_4090_v2_run{run}"
        data = stem.with_suffix(".csv")
        log = stem.with_suffix(".log")
        assert "PASS: 96 shape/dtype/scale/mode cases; 384 launch configurations; 1920 raw samples." in log.read_text()
        assert "FAIL:" not in log.read_text()
        rows = list(csv.DictReader(io.StringIO(data.read_text())))
        assert len(rows) == 1920, (run, len(rows))
        seen = set()
        case_keys = set()
        for row in rows:
            case = tuple(row[k] for k in CASE)
            thread = int(row["threads"])
            group = int(row["group"])
            identity = (case, thread, group)
            assert identity not in seen
            seen.add(identity)
            assert thread in (32, 64, 128, 256) and group in range(5)
            assert row["check_status"] == "PASS"
            assert int(row["captured_calls"]) == 64 and int(row["replays"]) == 20
            value = float(row["mean_us"])
            assert math.isfinite(value) and value > 0
            samples[(case, thread)].setdefault(run, []).append(value)
            case_keys.add(case)
            environment.add((row["gpu"], row["sm"], row["cuda_runtime"]))
        assert len(case_keys) == 96
        cases_by_run.append(case_keys)
        raw_files.extend((data, log))
    assert cases_by_run[0] == cases_by_run[1] == cases_by_run[2]
    assert len(environment) == 1
    medians = {}
    for key, runs in samples.items():
        assert set(runs) == {1, 2, 3}
        assert all(len(values) == 5 for values in runs.values())
        medians[key] = [statistics.median(runs[run]) for run in (1, 2, 3)]
    assert len(medians) == 384
    records = []
    for (case, thread), times in sorted(medians.items()):
        base = medians[(case, 128)]
        reductions = [100 * (1 - time / baseline) for time, baseline in zip(times, base)]
        record = dict(zip(CASE, case))
        record["threads"] = thread
        for run in range(3):
            record[f"run{run + 1}_median_us"] = times[run]
            record[f"run{run + 1}_baseline128_us"] = base[run]
            record[f"run{run + 1}_time_reduction_percent"] = reductions[run]
        record["min_time_reduction_percent"] = min(reductions)
        record["max_time_reduction_percent"] = max(reductions)
        record["stable_at_least_5_percent"] = thread != 128 and min(reductions) >= 5
        record["any_run_regression_over_3_percent"] = thread != 128 and min(reductions) < -3
        record["all_runs_regression_over_3_percent"] = thread != 128 and max(reductions) < -3
        records.append(record)
    output_csv = ROOT / "case_medians.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    winners = [r for r in records if r["stable_at_least_5_percent"]]
    winner_cases = {tuple(r[k] for k in CASE) for r in winners}
    slower = [r for r in records if r["all_runs_regression_over_3_percent"]]
    gpu, sm, runtime = next(iter(environment))
    summary = {
        "status": "ALL_THREE_RUNS_PASS",
        "environment_observed_in_raw_csv": {"gpu": gpu, "sm": int(sm), "cuda_runtime": int(runtime)},
        "runs": 3, "distinct_cases": 96, "launch_configurations_per_run": 384,
        "samples_per_run": 1920, "total_samples": 5760,
        "definition": "Time reduction = 100*(1-candidate median/baseline128 median), paired within each process run. Stable winner requires >=5% in ALL 3 runs; no confidence interval is claimed.",
        "stable_winning_configurations": len(winners), "cases_with_stable_winner": len(winner_cases),
        "configurations_slower_over_3_percent_in_all_runs": len(slower),
        "configurations_slower_over_3_percent_in_any_run": sum(r["any_run_regression_over_3_percent"] for r in records),
        "stable_winners": winners,
        "stable_slower_examples": sorted(slower, key=lambda r: r["max_time_reduction_percent"])[:12],
        "m17_n256_transform": [r for r in records if r["mode"] == "transform" and r["rows"] == "17" and r["dim"] == "256"],
        "limitations": [
            "Same RTX4090, one experimental session, three separate processes; not cross-date or cross-device replication.",
            "Input/output addresses reused by 64 serial graph calls; 20 replays per group; fixed small working set can reuse cache.",
            "No concurrent GPU suite during this pilot according to runner; clocks, temperature, power and hardware counters were not captured.",
            "Internal launch-configuration screening only; original Dao Graph uses 64 distinct retained outputs and cannot be directly divided into these timings.",
            "No production dispatch changed; winning shapes must pass the original validation and benchmark contract before adoption.",
        ],
    }
    output_json = ROOT / "summary.json"
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    source = ROOT.parents[1] / "src" / "tune_launch.cu"
    assert sha256(source) == SOURCE_SHA256, "Experiment source differs from archived run source"
    files = raw_files + [Path(__file__), output_csv, output_json]
    manifest = {
        "date": "2026-09-05", "source": {"path": "../../src/tune_launch.cu", "sha256": SOURCE_SHA256},
        "baseline_kernel": {
            "implementation_commit": "6f8e15a2db63a1816c2da6632848a1945380cf21",
            "graph_harness_commit": "0b29fcf9031193f49319b2d4132df4d1ef6a4a74",
            "path_in_repository": "03_hadamard_tc/a962695448-rgb/include/kernels.cuh",
            "identical_git_blob_in_both_commits": "4618f3fd2964dff2341b6cdbfd402da00e55ae26",
        },
        "environment": summary["environment_observed_in_raw_csv"],
        "method": "Internal fixed-buffer graph screening; 64 serial kernel calls, 20 replays, 5 rotated-order groups per process; medians are paired against 128 threads within each process.",
        "clock_temperature_power_recorded": False,
        "ncu_counters_recorded": False,
        "production_dispatch_changed": False,
        "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256(p),
                   "kind": "raw_unchanged" if p in raw_files else "derived_or_analysis_source"} for p in files],
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("stable_winners", "stable_slower_examples", "m17_n256_transform", "limitations")}, ensure_ascii=False, indent=2))
    for record in winners:
        print("WIN", *(f"{k}={record[k]}" for k in (*CASE, "threads", "min_time_reduction_percent", "max_time_reduction_percent")))


if __name__ == "__main__":
    analyze()
