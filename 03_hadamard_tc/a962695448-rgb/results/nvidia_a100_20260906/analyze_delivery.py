#!/usr/bin/env python3
"""离线复算已下载的 A100 验收证据；只写本目录 derived，不启动测试或 GPU。

用法：python analyze_delivery.py [--input ./retrieved]
退出码：0=全部指定证据验证通过；2=UNVERIFIED（缺失、失败或不一致）。
1876 项 CLI 还需要 source/results/validation_a100_default128.log 和
results/cli_explicit256_original_matrix.log。所有原始文件保持只读。
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parent
COMMIT = "12c76d8331ef7cf3fd4c8c14a049162559be4302"
DAO_COMMIT = "e7706faf8d1c3b9f241e36860640ad1dac644ede"
KEYS = ("dtype", "dim", "rows", "normalized", "mode")
LIMITS = {"fp16": 1e-2, "bf16": 5e-2}
EAGER_SCOPE = "CUDA events around allocating eager API calls, including GPU idle gaps from host dispatch; excludes H2D/D2H/build/validation. Not kernel-only or host end-to-end."
GRAPH_SCOPE = "CUDA events around 20 replays of 64 retained independent outputs, divided by 1280 calls; median of 5 groups. Captured GPU work plus amortized replay scheduling; not isolated kernel latency or host end-to-end."
REQUIRED_STAGES = (
    "private_environment", "build_arch80_cpu_test",
    "cli_default_original_matrix_and_benchmarks", "cli_explicit256_original_matrix",
    "restore_reference", "build_install_reference",
    "dao_original1800_default128_and_12_benchmarks", "thread_api_original1800",
    "promotion_72_run1", "promotion_72_run2", "promotion_72_run3",
)


def require(condition, message):
    # 不使用可被 python -O 删除的 assert，身份与完整性检查始终执行。
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def number(value, label, positive=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool), label + ": not numeric")
    require(math.isfinite(value) and (value > 0 if positive else value >= 0), label + ": invalid value")
    return float(value)


def close(actual, expected, label, rel=1e-12):
    require(math.isclose(float(actual), float(expected), rel_tol=rel, abs_tol=1e-12), label + ": inconsistent derived value")


def validate_gpu(environment):
    require("A100" in environment.get("gpu", ""), "GPU is not identified as A100")
    cap = environment.get("compute_capability", environment.get("sm"))
    require(cap == [8, 0], "Compute capability must be [8, 0]")
    require(re.fullmatch(r"[0-9a-f]{64}", environment.get("extension_sha256", "")), "Missing extension SHA256")


def validate_reference(report):
    reference = report["reference"]
    require(reference["commit"] == DAO_COMMIT, "Wrong Dao commit")
    require(reference["repository"] == "https://github.com/Dao-AILab/fast-hadamard-transform", "Wrong reference repository")
    require(re.fullmatch(r"[0-9a-f]{64}", reference["cuda_module_sha256"]), "Missing reference binary SHA256")
    require(bool(reference.get("verification")), "Missing installed reference provenance")


def expected_dao_cases():
    return {(dtype, shape, pattern, seed, normalized)
            for dtype in LIMITS for dim in (1, 2, 4, 8, 16, 32, 64, 128, 256)
            for shape in ((1, dim), (3, dim), (17, dim), (1, 3, 7, dim), (2, 5, 13, dim))
            for pattern in ("uniform", "normal", "outlier", "zeros")
            for seed in ((2026,) if pattern == "zeros" else (2026, 95811, 314159))
            for normalized in (False, True)}


def validate_original_matrix(report):
    require(report["status"] == "PASS", "Original matrix report did not PASS")
    validate_gpu(report["environment"])
    validate_reference(report)
    require(report["reference_commit_required"] == DAO_COMMIT, "Wrong required Dao commit")
    require(report["summary"]["cases"] == 1800 and report["summary"]["failures"] == 0, "Original 1800 matrix incomplete/failed")
    found, maxima = set(), {dtype: 0.0 for dtype in LIMITS}
    for case in report["cases"]:
        key = (case["dtype"], tuple(case["shape"]), case["pattern"], case["seed"], case["normalized"])
        require(key not in found, "Duplicate original matrix case")
        found.add(key)
        limit = LIMITS[case["dtype"]]
        require(case["pass"] is True and case["fused_vs_split_int4_exact"] is True, "Failed correctness/INT4 comparison")
        close(case["strict_abs_limit"], limit, "Original strict error threshold")
        error = number(case["max_abs_error"], "Original maximum error")
        require(error < limit and case["elements_at_or_above_strict_limit"] == 0, "Original strict error bound failed")
        expected_scale = 1 / math.sqrt(case["shape"][-1]) if case["normalized"] else 1.0
        close(case["scale"], expected_scale, "Original scale", rel=1e-7)
        maxima[case["dtype"]] = max(maxima[case["dtype"]], error)
    require(len(report["cases"]) == 1800 and found == expected_dao_cases(), "Original matrix coverage differs")
    for dtype, value in maxima.items():
        close(report["summary"]["max_abs_error_by_dtype"][dtype], value, "Original summary error")
    require(report["non_default_stream"]["pass"] is True, "Non-default stream check failed")
    require(len(report["rejected_inputs"]) == 10 and all(x["pass"] is True for x in report["rejected_inputs"]), "Rejected-input tests incomplete")
    return maxima


def sample_stats(entry, names, graph=False, promotion=False):
    require(entry["groups"] == 5, "Expected five timing groups")
    require(set(entry["samples_us"]) == set(names) and set(entry["median_us"]) == set(names), "Timing methods differ")
    if graph:
        calls_key = "captured_outputs_per_graph" if promotion else "captured_calls_per_graph"
        require(entry[calls_key] == 64 and entry["replays_per_group"] == 20, "Graph replay/capture count differs")
        require(entry["api_warmup_calls"] == 25 and entry["graph_warmup_replays"] == 5, "Graph warmup differs")
        if promotion:
            require(entry["independent_output_buffers"] is True and entry["captured_outputs_bitwise_equal_eager_before_and_after"] is True, "Graph independent-buffer/correctness check failed")
        else:
            require(entry["calls_per_group"] == 1280 and entry["all_captured_outputs_equal_eager"] is True, "Graph output validation failed")
        require(set(entry["raw_event_intervals_ms"]) == set(names), "Raw event methods differ")
        require(len(entry["group_order"]) == 5 and all(sorted(order) == sorted(names) for order in entry["group_order"]), "Invalid method order")
    else:
        require(entry["repetitions_per_group"] == 200 and entry["warmup"] == 25, "Eager timing condition differs")
    result = {}
    for name in names:
        samples = entry["samples_us"][name]
        require(len(samples) == 5, "Missing raw timing samples")
        samples = [number(value, name + " sample", positive=True) for value in samples]
        median = statistics.median(samples)
        close(entry["median_us"][name], median, name + " median")
        if graph:
            intervals = entry["raw_event_intervals_ms"][name]
            require(len(intervals) == 5, "Missing raw event intervals")
            for us, ms in zip(samples, intervals):
                close(us, number(ms, "event interval", positive=True) * 1000 / 1280, "Event conversion")
        result[name] = {"median_us": median, "median_ms": median / 1000,
                        "samples_us": samples, "min_us": min(samples), "max_us": max(samples),
                        "range_percent_of_median": 100 * (max(samples) - min(samples)) / median,
                        "population_cv_percent": 100 * statistics.pstdev(samples) / statistics.mean(samples)}
    return result


def analyze_dao(report):
    maxima = validate_original_matrix(report)
    rows, summaries = [], {}
    expected = {(dtype, shape, 1.0) for dtype in LIMITS for dim in (16, 64, 256)
                for shape in ((17, dim), (4, 128, 8, dim))}
    for field, method, scope in (("benchmarks", "eager_api", EAGER_SCOPE), ("graph_benchmarks", "cuda_graph", GRAPH_SCOPE)):
        found, group = set(), []
        for entry in report[field]:
            key = (entry["dtype"], tuple(entry["shape"]), entry["scale"])
            require(key not in found and entry["method"] == method, "Duplicate/mismatched Dao benchmark")
            found.add(key)
            stats = sample_stats(entry, ("ours", "dao"), graph=method == "cuda_graph")
            ours, dao = stats["ours"]["median_us"], stats["dao"]["median_us"]
            close(entry["dao_over_ours"], dao / ours, "Dao/ours ratio")
            row = {"category": "dao_default128", "measurement": method, "dtype": entry["dtype"],
                   "shape": json.dumps(entry["shape"]), "dim": entry["shape"][-1],
                   "rows": math.prod(entry["shape"][:-1]), "scale": entry["scale"], "block_threads": 128,
                   "ours_us": ours, "ours_ms": ours / 1000, "dao_us": dao, "dao_ms": dao / 1000,
                   "dao_over_ours": dao / ours, "time_reduction_vs_dao_percent": 100 * (1 - ours / dao),
                   "ours_slower_than_dao": ours > dao, "scope": scope,
                   "raw_samples_json": json.dumps(stats, sort_keys=True)}
            rows.append(row)
            group.append(row)
        require(found == expected and len(group) == 12, method + ": expected all 12 configurations")
        summaries[method] = {"configurations": 12, "scope": scope,
                             "dao_over_ours_range": [min(x["dao_over_ours"] for x in group), max(x["dao_over_ours"] for x in group)],
                             "slower_than_dao_count": sum(x["ours_slower_than_dao"] for x in group),
                             "negative_cases": [x for x in group if x["ours_slower_than_dao"]]}
    return {"status": "VERIFIED", "original_matrix_cases": 1800, "max_abs_error_by_dtype": maxima,
            "environment": report["environment"], "measurements": summaries}, rows


def promotion_shapes():
    return {(dtype, dim, rows, norm) for dtype in LIMITS for dim in (16, 64)
            for rows in (4095, 4096, 4097, 16383, 16384, 16385) for norm in (False, True)}


def validate_promotion_shape(case):
    rows, dim = case["rows"], case["dim"]
    shape = [rows // 1024, 128, 8, dim] if rows in (4096, 16384) else [rows, dim]
    require(case["shape"] == shape, "Promotion shape differs")
    close(case["scale"], 1 / math.sqrt(dim) if case["normalized"] else 1.0, "Promotion scale")
    return tuple(case[k] for k in KEYS[:-1])


def analyze_promotion(reports, manifest, stages):
    expected_shapes = promotion_shapes()
    expected_configs = {(*key, mode) for key in expected_shapes for mode in ("transform", "fused_int4")
                        if mode == "transform" or key[2] in (4095, 4096, 4097)}
    indexed, source_ids, binaries, reference_binaries, environments = [], set(), set(), set(), set()
    for run, report in enumerate(reports, 1):
        require(report["status"] == "PASS" and report["run_index"] == run and report["build_only"] is False, "Promotion run incomplete")
        validate_gpu(report["environment"])
        validate_reference(report)
        environments.add(json.dumps(report["environment"], sort_keys=True))
        require(report["source_manifest"] == manifest and manifest["commit"] == COMMIT, "Promotion source differs from pinned worker")
        source_ids.add(report["experiment_script_sha256"])
        binaries.add(report["environment"]["extension_sha256"])
        reference_binaries.add(report["reference"]["cuda_module_sha256"])
        require(report["experiment_script_sha256"] == manifest["auxiliary_files"]["promotion_a100.py"]["sha256"], "Promotion script hash differs")
        require(report["summary"] == {"unique_shape_dtype_scale_cases": 48, "correctness_input_cases": 336, "benchmark_configurations": 72}, "Promotion summary differs")
        found = set()
        for case in report["correctness"]:
            key = validate_promotion_shape(case)
            require(key not in found, "Duplicate promotion correctness shape")
            found.add(key)
            checks = case["checks"]
            expected_checks = {(p, s) for p in ("normal", "uniform", "outlier", "zeros")
                               for s in ((2026,) if p == "zeros" else (2026, 95811))}
            require(len(checks) == 7 and {(c["pattern"], c["seed"]) for c in checks} == expected_checks, "Promotion input coverage differs")
            for check in checks:
                require(check["all_elements_baseline_candidate_bitwise_exact"] is True and check["cpu_quantization_exact"] is True, "Promotion bitwise/CPU quantization failed")
                limit = LIMITS[case["dtype"]]
                close(check["strict_limit"], limit, "Promotion strict threshold")
                require(number(check["dao_max_abs_error"], "Dao error") < limit and number(check["dense_max_abs_error"], "Dense error") < limit, "Promotion error exceeded limit")
                require(check["elements"] == case["rows"] * case["dim"] and check["quantization_rows_checked"] == case["rows"], "Incomplete promotion quantization coverage")
                require(check["dense_rows"] == [0, case["rows"] // 2, case["rows"] - 1], "Dense row sampling differs")
        require(found == expected_shapes and len(report["correctness"]) == 48, "Promotion correctness shapes differ")
        entries = {}
        for entry in report["benchmarks"]:
            key = (*validate_promotion_shape(entry), entry["mode"])
            require(key not in entries, "Duplicate promotion benchmark")
            target = entry["rows"] in ((4096, 16384) if entry["mode"] == "transform" else (4096,))
            require(entry["original_target"] is target, "Incorrect original/holdout designation")
            names = ("baseline128", "candidate256", "dao") if entry["mode"] == "transform" else ("baseline128", "candidate256")
            stats = sample_stats(entry, names, graph=True, promotion=True)
            b, c = stats["baseline128"]["median_us"], stats["candidate256"]["median_us"]
            close(entry["baseline_over_candidate"], b / c, "Promotion ratio")
            close(entry["candidate_time_reduction_percent"], 100 * (1 - c / b), "Promotion reduction")
            if "dao" in stats:
                close(entry["dao_over_candidate"], stats["dao"]["median_us"] / c, "Promotion Dao/candidate")
                close(entry["dao_over_baseline"], stats["dao"]["median_us"] / b, "Promotion Dao/baseline")
            entries[key] = (entry, stats)
        require(set(entries) == expected_configs and len(entries) == 72, "Promotion 72 configuration matrix differs")
        indexed.append(entries)
    require(len(binaries) == len(source_ids) == len(reference_binaries) == 1, "Promotion binary/source differs between runs")
    require(len(environments) == 1, "Promotion GPU/toolchain environment differs between runs")
    pids = [stages[f"promotion_72_run{i}"]["pid"] for i in (1, 2, 3)]
    require(len(set(pids)) == 3, "Promotion processes must be distinct")
    for previous, current in zip(reports, reports[1:]):
        require(datetime.fromisoformat(previous["finished_utc"].replace("Z", "+00:00")) <=
                datetime.fromisoformat(current["started_utc"].replace("Z", "+00:00")), "Promotion process windows overlap")
    rows = []
    for key in sorted(expected_configs):
        row = {"category": "promotion_128_vs_256", "measurement": "cuda_graph", **dict(zip(KEYS, key)),
               "scope": GRAPH_SCOPE, "original_target": indexed[0][key][0]["original_target"]}
        reductions, raw = [], {}
        for run, index in enumerate(indexed, 1):
            entry, stats = index[key]
            b, c = stats["baseline128"]["median_us"], stats["candidate256"]["median_us"]
            gain = 100 * (1 - c / b)
            reductions.append(gain)
            row.update({f"run{run}_baseline128_us": b, f"run{run}_candidate256_us": c,
                        f"run{run}_baseline128_ms": b / 1000, f"run{run}_candidate256_ms": c / 1000,
                        f"run{run}_time_reduction_percent": gain})
            if "dao" in stats:
                row[f"run{run}_dao_us"] = stats["dao"]["median_us"]
                row[f"run{run}_dao_over_candidate"] = stats["dao"]["median_us"] / c
            raw[f"run{run}"] = stats
        row.update(min_time_reduction_percent=min(reductions), max_time_reduction_percent=max(reductions),
                   stable_at_least_5_percent=all(x >= 5 for x in reductions),
                   any_run_regression=any(x < 0 for x in reductions), every_run_regression=all(x < 0 for x in reductions),
                   any_run_regression_over_3_percent=any(x < -3 for x in reductions),
                   every_run_regression_over_3_percent=all(x < -3 for x in reductions),
                   raw_samples_json=json.dumps(raw, sort_keys=True))
        rows.append(row)
    def group(subset):
        return {"configurations": len(subset),
                **{flag: sum(row[flag] for row in subset) for flag in (
                    "stable_at_least_5_percent", "any_run_regression", "every_run_regression",
                    "any_run_regression_over_3_percent", "every_run_regression_over_3_percent")},
                "time_reduction_percent_range": [min(r["min_time_reduction_percent"] for r in subset), max(r["max_time_reduction_percent"] for r in subset)]}
    return {"status": "VERIFIED", "scope": GRAPH_SCOPE, "process_runs": 3,
            "distinct_correctness_input_cases": 336, "counting_note": "Same 336 cases repeated in three processes; not 1008 distinct cases. Do not add to CLI/Dao scope counts.",
            "extension_sha256": next(iter(binaries)), "all_72": group(rows),
            "original_24": group([r for r in rows if r["original_target"]]),
            "adjacent_48": group([r for r in rows if not r["original_target"]]),
            "by_mode": {mode: group([r for r in rows if r["mode"] == mode]) for mode in ("transform", "fused_int4")},
            "all_regressions": [r for r in rows if r["any_run_regression"]],
            "not_stably_5_percent": [r for r in rows if not r["stable_at_least_5_percent"]],
            "environment": reports[0]["environment"]}, rows


class Delivery:
    def __init__(self, source):
        self.source, self.inputs, self.sections, self.csv_rows = source, {}, {}, []

    def read(self, name):
        path = self.source / name
        data = path.read_bytes()
        self.inputs[name] = {"bytes": len(data), "sha256": digest(data)}
        return data.decode("utf-8-sig")

    def json(self, name):
        return json.loads(self.read(name), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON value: " + value)))

    def section(self, name, function):
        try:
            result, rows = function()
            self.sections[name] = result
            self.csv_rows.extend(rows)
        except (OSError, ValueError, KeyError, TypeError, IndexError, ArithmeticError) as error:
            self.sections[name] = {"status": "UNVERIFIED", "reason": f"{type(error).__name__}: {error}"}


def validate_state(delivery):
    state = delivery.json("results/state.json")
    require(state["source_commit"] == COMMIT and state["source_manifest"]["commit"] == COMMIT, "Worker source commit mismatch")
    require(state["status"] == "PASS" and state.get("active") is None and not state.get("optional_unverified"), "Worker has failed, running, or incomplete stages")
    require(state["mandatory_correctness_and_original12"] == "PASS", "Mandatory acceptance incomplete")
    manifest = state["source_manifest"]
    # 本地准备阶段留下的源清单是下载数据之外的固定锚点；不信任结果自行声明新源码。
    local_manifest = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8-sig"))
    require(manifest == local_manifest and manifest["commit"] == COMMIT, "Worker manifest differs from locally prepared pinned source")
    stages = {entry["name"]: entry for entry in state["stages"]}
    require(len(stages) == len(state["stages"]), "Duplicate worker stage")
    for name in REQUIRED_STAGES:
        entry = stages[name]
        require(entry["exit"] == entry["expected_exit"] == 0 and not entry.get("timed_out"), "Failed required stage: " + name)
        require(name in state["completed"], "Stage absent from completed list: " + name)
    for snap in ("before", "after"):
        gpu = state[snap]["gpu"]
        require(gpu["exit"] == 0, "GPU snapshot failed")
        rows = list(csv.reader(io.StringIO(gpu["stdout"])))
        require(len(rows) == 2 and "A100" in rows[1][0], "Worker snapshot must show one A100")
        require(rows[1][2].strip() == "Disabled", "A100 MIG must be disabled")
    return state, manifest, stages


def analyze_cli(delivery, state, stages):
    for path, threads in (("source/results/validation_a100_default128.log", 128),
                          ("results/cli_explicit256_original_matrix.log", 256)):
        text = delivery.read(path)
        matches = re.findall(r"SELF_TEST PASS cases=1876[^\r\n]*", text)
        require(len(matches) == 1 and f"warp_block_threads={threads}" in matches[0], "Missing original CLI self-test PASS: " + path)
        require("CPU/split/fused_INT4_bytes=exact scales=exact" in matches[0], "CLI exact quantization evidence absent")
        if threads == 256:
            require(delivery.inputs[path]["sha256"] == stages["cli_explicit256_original_matrix"]["log_sha256"], "CLI 256 log hash mismatch")
        else:
            exits = [int(x) for x in re.findall(r"EXIT_CODE (-?\d+);", text)]
            require(exits == [0] + [2] * 15 + [0] * 16, "Original CLI validation/negative-test/benchmark exits differ")
    raw = list(csv.DictReader(io.StringIO(delivery.read("source/results/benchmark_a100_default128.csv"))))
    require(len(raw) == 110, "CLI benchmark must contain 110 measured rows")
    shapes = ((1, 1, 1, 1), (1, 1, 1, 16), (1, 1, 17, 256), (4, 128, 8, 16),
              (4, 128, 8, 64), (4, 128, 8, 256), (4, 512, 8, 256), (1, 257, 1, 256))
    expected, found, rows = set(), set(), []
    for dtype in LIMITS:
        for shape in shapes:
            methods = ["naive_global", "warp", "split_int4", "fused_int4", "cpu_fp32_fwht", "warp_h2d_d2h"]
            if shape[-1] >= 16:
                methods.append("tensor_core")
            expected.update((dtype, shape, method) for method in methods)
    for entry in raw:
        require("A100" in entry["gpu"] and entry["compute_capability"] == "80", "CLI CSV hardware mismatch")
        shape = tuple(int(entry[k]) for k in ("batch", "seq", "heads", "dim"))
        key = (entry["dtype"], shape, entry["method"])
        require(key not in found, "Duplicate CLI measurement")
        found.add(key)
        us, ms = float(entry["mean_us"]), float(entry["mean_ms"])
        number(us, "CLI microseconds", positive=True)
        close(ms, us / 1000, "CLI us/ms", rel=5e-11)
        close(entry["input_elements_per_second"], math.prod(shape) * 1e6 / us, "CLI throughput", rel=5e-11)
        close(entry["scale"], 1 / 16 if shape == (1, 257, 1, 256) else 1, "CLI scale", rel=1e-8)
        warp = entry["method"] in ("warp", "split_int4", "fused_int4", "warp_h2d_d2h")
        require(entry["warp_block_threads"] == ("128" if warp else ""), "CLI thread label differs")
        method = entry["method"]
        scope = "cpu_compute" if method == "cpu_fp32_fwht" else "host_e2e" if method == "warp_h2d_d2h" else "kernel_only"
        require(entry["scope"] == scope and int(entry["repetitions"]) == (20 if scope != "kernel_only" else 300), "CLI measurement scope/repetition differs")
        require(number(float(entry["max_abs_error"]), "CLI error") < LIMITS[entry["dtype"]], "CLI error threshold failed")
        rows.append({"category": "cli_default128", "measurement": method, "dtype": entry["dtype"],
                     "shape": json.dumps(shape), "rows": math.prod(shape[:-1]), "dim": shape[-1],
                     "scale": float(entry["scale"]), "scope": scope, "block_threads": 128 if warp else "",
                     "mean_us": us, "mean_ms": ms, "raw_csv_row_json": json.dumps(entry, sort_keys=True)})
    require(found == expected, "CLI configuration matrix differs")
    return {"status": "VERIFIED", "original_self_test_cases": 1876, "thread_settings_repeated": [128, 256],
            "counting_note": "1876 original CLI cases repeated at 128 and 256; not 3752 new distinct cases.",
            "invalid_cli_cases": 15, "benchmark_configurations": 16, "measurement_rows": 110,
            "cli_binary_sha256": state["cli_binary_sha256"],
            "scope_note": "kernel_only CUDA events; cpu_compute FP32 host compute; host_e2e pageable H2D + warp + D2H with preallocated buffers. Separate from eager/Graph."}, rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["category", "measurement", "status"]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "retrieved")
    args = parser.parse_args()
    source, output = args.input.resolve(), (ROOT / "derived").resolve()
    require(source != output and not output.is_relative_to(source), "derived must remain outside the raw input tree")
    delivery = Delivery(source)
    state = manifest = stages = None
    try:
        state, manifest, stages = validate_state(delivery)
        delivery.sections["worker_and_source"] = {"status": "VERIFIED", "source_commit": COMMIT,
            "worker_status": state["status"], "required_successful_stages": list(REQUIRED_STAGES),
            "source_manifest_sha256": digest(json.dumps(manifest, sort_keys=True).encode())}
    except (OSError, ValueError, KeyError, TypeError, IndexError, ArithmeticError) as error:
        delivery.sections["worker_and_source"] = {"status": "UNVERIFIED", "reason": f"{type(error).__name__}: {error}"}
    # 源码/执行链不完整时不把其他数据升级为可信的 A100 交付。
    if manifest is not None and stages is not None:
        delivery.section("dao_default128", lambda: analyze_dao(delivery.json("results/dao_default.json")))
        def api():
            report = delivery.json("results/api_threads.json")
            maxima = validate_original_matrix(report)
            require(report["default_and_explicit128_and_256_bitwise_equal"] is True and report["legacy_one_argument_signatures"] == "PASS", "API compatibility/bitwise check failed")
            require(len(report["thread_value_rejections"]) == 27 and all(x["pass"] is True for x in report["thread_value_rejections"]), "Explicit thread rejection matrix incomplete")
            for name in ("include/kernels.cuh", "include/reference.hpp", "src/torch_binding.cu", "scripts/compare_reference.py", "scripts/verify_block_threads.py"):
                require(report["source_sha256"][name] == manifest["files"]["source/" + name]["sha256"], "API source hash differs: " + name)
            require(not report["benchmarks"] and not report["graph_benchmarks"], "API verification unexpectedly contains timing data")
            return {"status": "VERIFIED", "original_matrix_cases": 1800, "default_128_256_bitwise_equal": True,
                    "invalid_thread_cases": 27, "max_abs_error_by_dtype": maxima,
                    "environment": report["environment"], "counting_note": "Same original Dao matrix under three API call forms; not 5400 new cases and not additive to dao_default128."}, []
        delivery.section("api_threads", api)
        delivery.section("promotion_three_runs", lambda: analyze_promotion(
            [delivery.json(f"results/run{i}.json") for i in (1, 2, 3)], manifest, stages))
        delivery.section("cli", lambda: analyze_cli(delivery, state, stages))
    else:
        for name in ("dao_default128", "api_threads", "promotion_three_runs", "cli"):
            delivery.sections[name] = {"status": "UNVERIFIED", "reason": "Pinned worker/source validation must pass first"}
    verified = all(section["status"] == "VERIFIED" for section in delivery.sections.values())
    summary = {"status": "VERIFIED_A100_DELIVERY" if verified else "UNVERIFIED",
               "generated_utc": datetime.now(timezone.utc).isoformat(), "expected_source_commit": COMMIT,
               "input_root": source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else str(source), "inputs": delivery.inputs, "sections": delivery.sections,
               "counting_policy": "CLI 1876, Dao/API 1800 and promotion 336 are separate scopes, with repeated runs/settings identified; no grand total of independent new cases.",
               "comparison_policy": "Every ratio uses matching A100 measurements within the same report/run and measurement scope. Never divide by RTX4090 or other-device values. Eager, Graph, kernel-only and host end-to-end remain separate.",
               "claim_boundary": "Offline verification of downloaded execution evidence only; no new GPU tests, global-optimum claim, official submission or upstream acceptance claim.",
               "all_negative_configurations_retained": verified,
               "csv_retention_policy": "All configurations from each VERIFIED section are emitted, including every regression; UNVERIFIED sections produce no trusted measurement rows."}
    output.mkdir(exist_ok=True)
    csv_paths = []
    for name, category in (("dao_all_configurations.csv", "dao_default128"),
                           ("promotion_all_configurations.csv", "promotion_128_vs_256"),
                           ("cli_all_measurements.csv", "cli_default128")):
        path = output / name
        write_csv(path, [row for row in delivery.csv_rows if row["category"] == category])
        csv_paths.append(path)
    all_path = output / "all_configurations.csv"
    write_csv(all_path, delivery.csv_rows)
    csv_paths.append(all_path)
    summary["derived_csvs"] = {path.name: {"bytes": path.stat().st_size, "sha256": digest(path.read_bytes())} for path in csv_paths}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "sections": {name: value["status"] for name, value in delivery.sections.items()},
                      "summary": str(output / "summary.json"), "verified_measurement_rows": len(delivery.csv_rows)}, ensure_ascii=False))
    return 0 if verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
