#!/usr/bin/env python3
"""复算固定 Ascend OFF/ON A/B；标准库实现，只读取已有 NPU 证据。"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


MATRIX = [(m, n) for m in (1, 17, 257, 4096, 16384) for n in (16, 64, 128, 256)]
DTYPES = ("fp16", "bf16")
METHODS = ("scalar_transform", "vector_transform", "scalar_split", "vector_split",
           "scalar_fused", "vector_fused", "quant_only")
VECTOR = {"vector_transform", "vector_split", "vector_fused"}
CONTROLS = set(METHODS) - VECTOR
COLUMNS = ("dtype,batch,seq,heads,dim,rows,method,group,order,repeats,kernel_us,logical_io_bytes,logical_GBs,"
           "input_working_set_bytes,seed,input_read_only,scale,block_dim,warmup,timer,vector_scale_enabled,kernel_ms").split(",")
CLASSES = ("stable_improvement_ge5", "stable_regression_ge5", "all_faster_below_stability_rule",
           "all_slower_below_stability_rule", "unchanged_all_rounds", "mixed_or_unstable")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(root, relative):
    candidate = (root / relative).resolve()
    require(root in candidate.parents, "artifact path escapes input directory: " + str(relative))
    return candidate


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        rows = list(reader)
    require(header == COLUMNS, "unexpected 22-column header: " + str(path))
    return rows


def validate_rows(rows, variant, cells):
    wanted = {(dtype, m, n, method, group) for dtype in DTYPES for m, n in cells
              for method in METHODS for group in range(5)}
    require(len(rows) == len(wanted), "unexpected observation count")
    seen, order_seen = set(), set()
    parsed = []
    for fields in rows:
        require(len(fields) == len(COLUMNS), "ragged native CSV")
        row = dict(zip(COLUMNS, fields))
        dtype, m, n, method, group = row["dtype"], int(row["rows"]), int(row["dim"]), row["method"], int(row["group"])
        key = (dtype, m, n, method, group)
        require(key in wanted and key not in seen, "duplicate or unexpected dtype/shape/method/group")
        fixed = {"batch": "1", "seq": str(m), "heads": "1", "block_dim": "32", "repeats": "5", "warmup": "3",
                 "scale": "1", "input_read_only": "true", "timer": "acl_timeline_event_ms",
                 "vector_scale_enabled": "false" if variant == "old" else "true"}
        require(all(row[k] == value for k, value in fixed.items()), "shape/parameter/timer/vector-scale mismatch")
        order = int(row["order"])
        order_key = (dtype, m, n, group, order)
        require(0 <= order < 7 and order_key not in order_seen, "invalid or repeated method order")
        ms, us = float(row["kernel_ms"]), float(row["kernel_us"])
        require(all(math.isfinite(value) and value > 0 for value in (ms, us)), "invalid native event time")
        require(math.isclose(us, ms * 1000.0, rel_tol=2e-11, abs_tol=0.0), "us/ms unit mismatch")
        seen.add(key)
        order_seen.add(order_key)
        parsed.append((dtype, m, n, method, group, ms, us))
    require(seen == wanted, "fixed matrix is incomplete")
    return parsed


def load_evidence(root, source_prefix):
    summary_path = root / "ab_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary.get("status") == "PASS", "controller did not finish PASS")
    require(str(summary.get("source_id_label", "")).startswith(source_prefix), "unexpected source-id label")
    require(summary.get("parameters") == {"rounds": 3, "block_dim": 32, "repeats": 5, "warmup": 3, "groups": 5},
            "this analysis requires the fixed block32/repeat5/warmup3/group5 experiment")
    accounting = summary["accounting"]
    for field, value in {"unique_shape_dtype_cells": 40, "expected_processes": 120, "raw_rows_per_cell_process": 70,
                         "raw_rows_per_variant_round": 1400, "expected_observations": 8400, "verified_observations": 8400}.items():
        require(accounting.get(field) == value, "controller accounting mismatch: " + field)
    binaries = summary["binaries"]
    for variant in ("old", "new"):
        initial = binaries[variant]["sha256"]
        final = summary["binary_integrity"][variant]
        require(len(initial) == 64 and all(c in "0123456789abcdef" for c in initial), "invalid binary SHA256")
        require(final.get("unchanged") is True and final.get("final_sha256") == initial, "binary changed during A/B")
        require(binaries[variant].get("expected_vector_scale_enabled") is (variant == "new"), "variant expectation mismatch")
    executions = summary["executions"]
    require(len(executions) == 120, "controller does not contain 120 processes")
    expected_order = []
    for r in range(1, 4):
        for index, (m, n) in enumerate(MATRIX):
            order = ("old", "new") if (r - 1 + index) % 2 == 0 else ("new", "old")
            expected_order.extend((r, variant, m, n) for variant in order)
    native = {}
    for sequence, (execution, expected) in enumerate(zip(executions, expected_order), 1):
        r, variant, m, n = expected
        require((execution["round"], execution["variant"], execution["rows"], execution["dim"]) == expected,
                "controller did not follow the declared interleave order")
        require(execution.get("sequence") == sequence and execution.get("returncode") == 0, "failed/misordered process")
        require(execution.get("verified_native_rows") == 70, "cell row count not verified by controller")
        require(execution.get("binary_sha256_at_start") == binaries[variant]["sha256"], "process binary identity mismatch")
        csv_path = safe_path(root, execution["native_csv"])
        log_path = safe_path(root, execution["log"])
        require(digest(csv_path) == execution["native_csv_sha256"], "native CSV SHA mismatch: " + str(csv_path))
        require(digest(log_path) == execution["log_sha256"], "process log SHA mismatch: " + str(log_path))
        rows = load_csv(csv_path)
        validate_rows(rows, variant, [(m, n)])
        native[execution["native_csv"]] = rows
    observations = {}
    merged_hashes = {}
    for variant in ("old", "new"):
        for r in range(1, 4):
            name = "%s_run%d.csv" % (variant, r)
            path = root / name
            declaration = summary["merged_csvs"][name]
            require(declaration.get("rows") == 1400, "merged declaration row count mismatch")
            require(declaration.get("numeric_rewriting") is False, "controller declared numeric rewriting")
            merged_hashes[name] = digest(path)
            require(merged_hashes[name] == declaration["sha256"], "merged CSV SHA mismatch: " + name)
            rows = load_csv(path)
            sources = ["round_%d/%s/m%d_n%d/native.csv" % (r, variant, m, n) for m, n in MATRIX]
            require(declaration["source_cells_in_order"] == sources, "unexpected native-cell merge order")
            reconstructed = [row for source in sources for row in native[source]]
            require(rows == reconstructed, "merged data differs from unfiltered original cells")
            for dtype, m, n, method, group, ms, us in validate_rows(rows, variant, MATRIX):
                observations[(variant, r, dtype, m, n, method, group)] = (ms, us)
    require(len(observations) == 8400, "unique native observation count differs from 8400")
    return summary, observations, {"ab_summary_sha256": digest(summary_path), "merged_sha256": merged_hashes,
                                    "native_csv_sha_verified": 120, "process_log_sha_verified": 120,
                                    "merged_vs_native_all_fields_exact": True}


def metrics(before, after):
    ratio = before / after
    reduction = (before - after) / before * 100.0
    require(math.isfinite(ratio) and math.isfinite(reduction), "derived comparison is nonfinite")
    return ratio, reduction


def classification(changes):
    if all(value >= 5 for value in changes):
        return CLASSES[0]
    if all(value <= -5 for value in changes):
        return CLASSES[1]
    if all(value > 0 for value in changes):
        return CLASSES[2]
    if all(value < 0 for value in changes):
        return CLASSES[3]
    if all(value == 0 for value in changes):
        return CLASSES[4]
    return CLASSES[5]


def compute(observations):
    grouped = defaultdict(list)
    for key, values in observations.items():
        grouped[key[:-1]].append((key[-1], values[0], values[1]))
    medians, timing_table = {}, []
    for key in sorted(grouped):
        variant, r, dtype, m, n, method = key
        samples = sorted(grouped[key])
        require([sample[0] for sample in samples] == list(range(5)), "median group coverage mismatch")
        values_ms = [sample[1] for sample in samples]
        values_us = [sample[2] for sample in samples]
        medians[key] = statistics.median(values_ms)
        timing_table.append({"variant": variant, "round": r, "dtype": dtype, "M": m, "N": n, "method": method,
                             "block_dim": 32, "groups": 5, "median_ms": medians[key], "median_us": statistics.median(values_us),
                             "min_group_ms": min(values_ms), "max_group_ms": max(values_ms)})
    paired, by_configuration = [], defaultdict(list)
    for r in range(1, 4):
        for dtype in DTYPES:
            for m, n in MATRIX:
                for method in METHODS:
                    old = medians[("old", r, dtype, m, n, method)]
                    new = medians[("new", r, dtype, m, n, method)]
                    ratio, reduction = metrics(old, new)
                    paired.append({"dtype": dtype, "M": m, "N": n, "method": method, "round": r,
                                   "category": "vector_path" if method in VECTOR else "control", "block_dim": 32,
                                   "off_median_ms": old, "on_median_ms": new, "off_over_on_ratio": ratio,
                                   "latency_reduction_pct": reduction, "improvement_ge5pct": reduction >= 5,
                                   "regression_ge5pct": reduction <= -5})
                    by_configuration[(dtype, m, n, method)].append(reduction)
    stability = []
    for (dtype, m, n, method), changes in sorted(by_configuration.items()):
        require(len(changes) == 3, "comparison lacks three rounds")
        stability.append({"dtype": dtype, "M": m, "N": n, "method": method,
                          "category": "vector_path" if method in VECTOR else "control",
                          "round1_reduction_pct": changes[0], "round2_reduction_pct": changes[1], "round3_reduction_pct": changes[2],
                          "median_round_reduction_pct": statistics.median(changes), "min_round_reduction_pct": min(changes),
                          "max_round_reduction_pct": max(changes), "all_rounds_faster": all(x > 0 for x in changes),
                          "all_rounds_slower": all(x < 0 for x in changes), "classification": classification(changes)})
    scalar_vector, fusion = [], []
    for variant in ("old", "new"):
        for r in range(1, 4):
            for dtype in DTYPES:
                for m, n in MATRIX:
                    common = {"variant": variant, "round": r, "dtype": dtype, "M": m, "N": n, "block_dim": 32}
                    for operation in ("transform", "split", "fused"):
                        scalar = medians[(variant, r, dtype, m, n, "scalar_" + operation)]
                        vector = medians[(variant, r, dtype, m, n, "vector_" + operation)]
                        ratio, change = metrics(scalar, vector)
                        scalar_vector.append(dict(common, operation=operation, scalar_median_ms=scalar, vector_median_ms=vector,
                                                  scalar_over_vector_ratio=ratio, latency_reduction_pct=change))
                    split = medians[(variant, r, dtype, m, n, "vector_split")]
                    fused = medians[(variant, r, dtype, m, n, "vector_fused")]
                    ratio, change = metrics(split, fused)
                    fusion.append(dict(common, vector_split_median_ms=split, vector_fused_median_ms=fused,
                                       split_over_fused_ratio=ratio, latency_reduction_pct=change))
    return timing_table, paired, stability, scalar_vector, fusion


def distribution(records):
    counts = Counter(row["classification"] for row in records)
    return {key: counts[key] for key in CLASSES}


def round_counts(paired, category):
    result = []
    for r in range(1, 4):
        values = [row["latency_reduction_pct"] for row in paired if row["category"] == category and row["round"] == r]
        result.append({"round": r, "configurations": len(values), "improvement_ge5pct": sum(x >= 5 for x in values),
                       "regression_ge5pct": sum(x <= -5 for x in values), "faster": sum(x > 0 for x in values),
                       "slower": sum(x < 0 for x in values), "equal": sum(x == 0 for x in values),
                       "median_configuration_reduction_pct": statistics.median(values)})
    return result


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "evidence/ascend-20260906/ab_analysis")
    parser.add_argument("--expected-source-id", default="f614762")
    args = parser.parse_args()
    root, output = args.input.resolve(), args.output.resolve()
    if output.exists():
        parser.error("analysis output already exists; choose a fresh path")
    try:
        controller, observations, integrity = load_evidence(root, args.expected_source_id)
        timing, paired, stable, scalar_vector, fusion = compute(observations)
        vector_rows = [row for row in stable if row["category"] == "vector_path"]
        control_rows = [row for row in stable if row["category"] == "control"]
        vector_counts, control_counts = distribution(vector_rows), distribution(control_rows)
        report = {"analysis_status": "complete", "benchmark_controller_status": "PASS", "source_id_label": controller["source_id_label"],
                  "input_directory": str(root), "analyzer_sha256": digest(Path(__file__).resolve()), "integrity": integrity,
                  "binaries": {variant: {"sha256": controller["binaries"][variant]["sha256"],
                                         "final_sha256": controller["binary_integrity"][variant]["final_sha256"]}
                               for variant in ("old", "new")},
                  "source_verification_boundary": "controller labels and recorded binary identity verified for consistency; source/build correspondence remains externally verified",
                  "scope": "interleaved independent-process OFF/ON; hot-cache ACL timeline events may include launch gaps; no CPU/NVIDIA comparison",
                  "analysis_runs_no_npu": True, "full_math_validation": "separate evidence; not repeated or inferred as new correctness coverage here",
                  "counts": {"native_observations_counted_once": 8400, "independent_processes": 120, "shape_dtype_cells": 40,
                             "median_rows": len(timing), "off_on_paired_round_rows": len(paired), "three_round_configuration_rows": len(stable),
                             "same_variant_scalar_vector_rows": len(scalar_vector), "same_variant_vector_fusion_rows": len(fusion)},
                  "rules": {"positive_reduction_pct": "(OFF_ms-ON_ms)/OFF_ms*100", "ratio": "OFF_ms/ON_ms",
                            "stable_improvement_ge5": "all three round medians reduce latency by at least 5%",
                            "stable_regression_ge5": "all three round medians increase latency by at least 5%",
                            "stability_is_not_statistical_significance": True, "confidence_intervals": "not computed"},
                  "vector_classification_counts": vector_counts, "control_classification_counts": control_counts,
                  "vector_round_counts": round_counts(paired, "vector_path"), "control_round_counts": round_counts(paired, "control"),
                  "control_interpretation": "same-build scalar paths and quant_only can change with compilation layout, execution gaps or system state; they are not assumed invariant"}
        output.mkdir(parents=True, exist_ok=False)
        tables = {"timings_median_ms.csv": timing, "off_on_paired_rounds_ms.csv": paired,
                  "off_on_three_round_stability.csv": stable, "within_variant_scalar_vector_ms.csv": scalar_vector,
                  "within_variant_vector_fusion_ms.csv": fusion}
        for name, records in tables.items():
            write_csv(output / name, records)
        report["output_tables"] = {name: {"rows": len(records), "sha256": digest(output / name)} for name, records in tables.items()}
        (output / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        text = ("# Ascend vector-scale OFF/ON 三轮复算\n\n"
                f"来源标签：`{controller['source_id_label']}`。6份合并CSV与120份cell CSV、120份进程日志的SHA均核对通过，合并内容与cell逐字段一致。共120个独立进程、8400条观测；合并文件与cell不重复计数。\n\n"
                f"120个向量路径/形状/精度组合中，三轮每轮均降低耗时至少5%的有 **{vector_counts['stable_improvement_ge5']}** 项；三轮每轮均增加耗时至少5%的有 **{vector_counts['stable_regression_ge5']}** 项。"
                f"另外，三轮均更快但未全部达到5%的有{vector_counts['all_faster_below_stability_rule']}项，三轮均更慢但未全部达到5%的有{vector_counts['all_slower_below_stability_rule']}项，混合方向或不稳定的有{vector_counts['mixed_or_unstable']}项，全相同的有{vector_counts['unchanged_all_rounds']}项。全部正负例保留在CSV。\n\n"
                f"160个控制组合中，三轮均改善至少5%的有{control_counts['stable_improvement_ge5']}项，三轮均退化至少5%的有{control_counts['stable_regression_ge5']}项，混合方向或不稳定的有{control_counts['mixed_or_unstable']}项。"
                "scalar_transform/split/fused与quant_only也可能受编译布局、发射空档和系统状态影响，不能假称完全不变，也不能把向量路径的全部变化直接归因于vector-scale。\n\n"
                "各轮使用5组原始ACL时间的中位数配对。另表完整列出每个variant内部、同block_dim=32的Scalar/Vector对比，以及Vector split/fused对比；单位均明确为ms。"
                "这些是热缓存、独立进程交错实验，ACL event区间可能包含发射空档。“稳定”仅指上述三轮5%规则，不代表统计显著性；没有生成置信区间。本分析未运行NPU、未增加正确性用例，也未混入CPU或NVIDIA结果。全量数学验证以独立验收日志为准。\n")
        (output / "结论.md").write_text(text, encoding="utf-8")
        print(json.dumps({"analysis_status": "complete", "output": str(output), "vector_counts": vector_counts,
                          "control_counts": control_counts, "vector_round_counts": report["vector_round_counts"],
                          "control_round_counts": report["control_round_counts"]}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("ANALYSIS_FAILED", error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
