#!/usr/bin/env python3
"""CPU revalidation of V2 sample buffers/certificates and three fused Graph runs.

python analyze_v2.py --regression-report ../server_raw/runs/regression/regression_report.json \
    --output derived runs/run1.json runs/run2.json runs/run3.json

The adjacent runN_sample_buffers.json files are required. Outputs are
analysis.json, comparison.csv and samples.csv. No GPU work or source edits.
Missing/failed/inconsistent evidence yields UNVERIFIED and exit 2.
"""
from datetime import datetime, timezone
from fractions import Fraction
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
sys.dont_write_bytecode = True
EXPECTED = {
    "report_sha256": "c525cc8371a6c9faa514243613b9e62e769f17ce29d20e8fbebb564b56704ace",
    "source_manifest_sha256": "ae562cf54a65f306650945e73a6985eb6989099b17cbeba3ea2f607debedb932",
    "original_run_manifest_sha256": "9ebb353585ca881f1a59864e1e097654be3f6886de7f3ddea1ef043ab6922ee9",
    "production_binary_sha256": "eb3f03f28b7f993bfc3351a8afc1022ccc5f93301c95ed3163fd437f6e1f3468",
    "reference_binary_sha256": "2e38b886e3fc6c31c3b837a4fd7354e844dd81d4a20b30f39fbd0c351d8620a4",
}
V1_FAILURE_SHA = "91b494c8c825b42ba4fad1c60c1283ea58636359e30f3ca7b462e12b935a75c1"
BASE_COMMIT = "217c30ff5e78842cd5809de6bf78ee8a7f04fc54"
DAO_COMMIT = "e7706faf8d1c3b9f241e36860640ad1dac644ede"
HARD_MAXIMA = ("prestorage_error", "storage_rounding_error", "stored_error_vs_exact")
HARD_COUNTS = ("certified_elements", "gpu_vs_exact_direct_storage_bit_differences",
               "exact_direct_vs_via_fp32_numeric_differences")
DIAGNOSTIC_MAXIMA = ("dense_fp64_error_vs_exact", "stored_error_vs_unrounded_dense",
                     "stored_error_vs_direct_rounded_dense", "stored_error_vs_via_fp32_rounded_dense")
DIAGNOSTIC_COUNTS = ("dense_fp64_inexact_elements", "dense_direct_vs_via_fp32_numeric_differences",
                     "signed_zero_only_rounding_differences")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(root, name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name and
            not name.startswith("/") and all(part not in ("", ".", "..") for part in name.split("/")), "Unsafe relative evidence path")
    path = root / name
    require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), "Evidence path escapes its root")
    return path


def read_json(path, inputs):
    data = path.read_bytes()
    inputs[str(path)] = {"bytes": len(data), "sha256": sha(data)}
    def pairs(entries):
        result = {}
        for key, value in entries:
            require(key not in result, "Duplicate JSON object key")
            result[key] = value
        return result
    def reject(value):
        raise ValueError("Non-finite JSON constant: " + value)
    return json.loads(data.decode("utf-8-sig"), object_pairs_hook=pairs, parse_constant=reject)


def validate_file_map(root, mapping):
    require(isinstance(mapping, dict) and mapping, "Missing frozen file map")
    for name, item in mapping.items():
        data = safe_path(root, name).read_bytes()
        require(len(data) == item["size"] and sha(data) == item["sha256"], "Frozen source mismatch: " + name)


def import_verified(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rational(record, label):
    value = record["fraction"]
    require(isinstance(value, str) and len(value) <= 4096 and re.fullmatch(r"-?\d+/[1-9]\d*", value), label + ": malformed fraction")
    fraction = Fraction(value)
    upper = record["upper_float"]
    require(type(upper) in (int, float) and math.isfinite(upper) and Fraction(float(upper)) >= fraction,
            label + ": display value rounds downward/nonfinite")
    return fraction


def same_fraction(recorded, computed, label):
    require(rational(recorded, label) == rational(computed, label), label + ": exact fraction differs")
    require(recorded["upper_float"] == computed["upper_float"], label + ": outward display value differs")


def verify_sources(args, inputs):
    protocol = read_json(ROOT / "protocol_v2.json", inputs)
    require(protocol["protocol_id"] == "nvidia_fused_integration_20260906_v2_reference_forward_certificate", "Wrong validation protocol")
    for name, checksum in EXPECTED.items():
        require(protocol["reused_regression"][name] == checksum, "Changed approved reused identity: " + name)
    require(protocol["revision"]["original_failure_sha256"] == V1_FAILURE_SHA and
            protocol["revision"]["source_or_math_changed"] is False and
            protocol["revision"]["matrix_or_seed_or_timing_changed"] is False, "Changed V1 retention/revision contract")
    old_manifest = read_json(BASE / "run_manifest.json", inputs)
    require(inputs[str(BASE / "run_manifest.json")]["sha256"] == EXPECTED["original_run_manifest_sha256"] and
            len(old_manifest["files"]) == 27, "Original 27-file manifest differs")
    validate_file_map(BASE, old_manifest["files"])
    original_protocol = read_json(BASE / "protocol.json", inputs)
    for name in ("rows", "dims", "dtypes", "normalized", "expected_configurations", "block_threads", "mode", "layouts",
                 "exclude_initial_rows", "patterns_and_seeds", "strict_abs_limit", "scale"):
        require(protocol["holdout"][name] == original_protocol["holdout"][name], "V2 changed a fixed original matrix field: " + name)
    require(protocol["timing"] == original_protocol["timing"], "V2 changed original timing protocol")
    source = read_json(BASE / "source_manifest.json", inputs)
    require(inputs[str(BASE / "source_manifest.json")]["sha256"] == EXPECTED["source_manifest_sha256"] and
            source["base_commit"] == BASE_COMMIT and len(source["files"]) == 13, "Original 13-source snapshot differs")
    validate_file_map(BASE, source["files"])
    manifest = read_json(ROOT / "manifest_v2.json", inputs)
    require(manifest["validation_revision"] == 2 and manifest["original_manifest_sha256"] == EXPECTED["original_run_manifest_sha256"] and
            manifest["original_source_manifest_sha256"] == EXPECTED["source_manifest_sha256"] and
            manifest["original_v1_result_remains"] == "FAIL_BEFORE_TIMING", "V2 manifest ancestry differs")
    require({"protocol_v2.json", "numeric_certificate.py", "checks_v2.py", "run_v2.py", "run_suite_v2.py", "analyze_v2.py"}
            <= set(manifest["validation_files"]), "V2 manifest omits a required execution/certificate file")
    validate_file_map(ROOT, manifest["validation_files"])
    regression = read_json(args.regression_report, inputs)
    require(inputs[str(args.regression_report)]["sha256"] == EXPECTED["report_sha256"], "Regression report is not the approved exact bytes")
    require(regression["status"] == "PASS" and regression["exit_code"] == 0 and regression["holdout_allowed"] is True,
            "Reused regression failed/incomplete")
    require(regression["source_manifest_sha256"] == EXPECTED["source_manifest_sha256"] and
            regression["binaries"]["production_extension"]["sha256"] == EXPECTED["production_binary_sha256"], "Regression source/binary differs")
    candidates = [args.v1_failure] if args.v1_failure else [BASE / "server_raw/runs/holdout/run1.json", BASE / "runs/holdout/run1.json"]
    existing = [path for path in candidates if path.is_file()]
    require(existing, "Original V1 failure evidence is unavailable")
    v1_path = next((path for path in existing if sha(path.read_bytes()) == V1_FAILURE_SHA), None)
    require(v1_path is not None, "Original V1 failure SHA differs")
    v1 = read_json(v1_path, inputs)
    require(v1["status"] == "FAIL" and v1["exit_code"] != 0 and not v1["benchmarks"], "V1 failure was relabeled or contains timing data")
    ninja_candidates = [BASE / "server_raw/build/production/build.ninja", BASE / "build/production/build.ninja"]
    ninja = next((path for path in ninja_candidates if path.is_file()), None)
    require(ninja is not None, "Actual production build.ninja is unavailable")
    ninja_bytes = ninja.read_bytes()
    inputs[str(ninja)] = {"bytes": len(ninja_bytes), "sha256": sha(ninja_bytes)}
    ninja_text = ninja_bytes.decode("utf-8")
    require(not re.search(r"(?:--|-)(?:use_fast_math|ftz(?:=|\s+)true)(?:\s|$)", ninja_text) and
            "compute_89" in ninja_text and "sm_89" in ninja_text, "Build is outside the RN/gradual-underflow model")
    return protocol, manifest, regression, {"status": "FAIL", "report_sha256": V1_FAILURE_SHA,
        "complete_configurations": sum(len(case["checks"]) == 14 for case in v1["correctness"]),
        "recorded_conditions": sum(len(case["checks"]) for case in v1["correctness"]),
        "timing_samples_imported": 0}, sha(ninja_bytes)


def verify_run_identity(report, protocol, manifest, protocol_sha, manifest_sha, ninja_sha):
    require(report["status"] == "PASS" and report["exit_code"] == 0 and report["actual_gpu_execution"] is True,
            "V2 worker failed/incomplete or did not record actual GPU execution")
    require(type(report["run_index"]) is int and report["run_index"] in (1, 2, 3) and type(report["pid"]) is int and
            report["pid"] > 0, "Invalid worker run/PID identity")
    require(report["validation_revision"] == 2 and report["protocol_sha256"] == protocol_sha and
            report["validation_manifest_sha256"] == manifest_sha and report["validation_manifest"] == manifest, "Wrong V2 identity")
    require(report["source_manifest_sha256"] == EXPECTED["source_manifest_sha256"] and
            report["original_v1_failure"] == protocol["revision"], "Source or retained V1 identity differs")
    execution = report["python_execution"]
    require(execution["assertions_enabled"] is True and execution["optimize_flag"] == 0 and execution["PYTHONOPTIMIZE"] == "0", "Assertions were disabled")
    gate = report["original_regression_gate"]
    require(gate["status"] == "PASS" and gate["report_sha256"] == EXPECTED["report_sha256"] and
            gate["production_extension_sha256"] == EXPECTED["production_binary_sha256"], "Wrong reused regression gate")
    environment = report["environment"]
    require("RTX 4090" in environment["gpu"] and environment["sm"] == [8, 9] and
            environment["extension_sha256"] == EXPECTED["production_binary_sha256"] and
            environment["torch_cuda_arch_list"] == "8.9" and environment["max_jobs"] == "1", "Hardware/production binary differs")
    reference = report["reference"]
    require(reference["commit"] == DAO_COMMIT and reference["cuda_module_sha256"] == EXPECTED["reference_binary_sha256"] and
            reference["repository"] == "https://github.com/Dao-AILab/fast-hadamard-transform" and reference["verification"], "Fixed Dao identity differs")
    assumptions = report["build_model_assumptions"]
    require(assumptions["build_ninja_sha256"] == ninja_sha and assumptions["explicit_fast_math_or_ftz_true"] is False,
            "Production build assumption evidence differs")


def load_buffers(report, report_path, inputs, array_pool):
    info = report["sample_buffers"]
    expected_name = report_path.stem + "_sample_buffers.json"
    require(info["file"] == expected_name and info["raw_u16_arrays_retained"] is True, "Sample buffer file identity differs")
    path = safe_path(report_path.parent, expected_name)
    payloads = read_json(path, inputs)
    require(inputs[str(path)]["sha256"] == info["sha256"] and len(payloads) == info["buffers"], "Sample buffer file SHA/count differs")
    for checksum, payload in payloads.items():
        require(re.fullmatch(r"[0-9a-f]{64}", checksum) is not None, "Invalid sample array digest")
        require(payload["dtype"] == "uint16_bits" and payload["shape"] in ([2, 256], [3, 256]), "Invalid sample array metadata")
        values = payload["values"]
        require(isinstance(values, list) and len(values) == payload["shape"][0] and all(isinstance(row, list) and len(row) == 256 and
                all(type(value) is int and 0 <= value <= 65535 for value in row) for row in values), "Invalid sample uint16 values")
        array = np.array(values, dtype="<u2", order="C")
        require(sha(array.tobytes()) == checksum, "Sample uint16 little-endian byte SHA differs")
        if checksum in array_pool:
            require(array.shape == array_pool[checksum].shape and np.array_equal(array, array_pool[checksum]), "Conflicting sample array under same digest")
        else:
            array.setflags(write=False)
            array_pool[checksum] = array
    return set(payloads)


def compare_certificate(recorded, fresh, case, row_ids, input_sha, output_sha):
    require(recorded["status"] == fresh["status"] == "PASS" and recorded["first_failure"] is None and
            recorded["gpu_executed"] is False, "Sample hard certificate failed")
    for field in ("certificate_version", "assumptions", "cpu_environment_precheck", "dtype", "sample_rows", "elements", "cpu_pre_f32_le_sha256"):
        require(recorded[field] == fresh[field], "Certificate identity/model/count differs: " + field)
    require(recorded["input_u16_le_sha256"] == input_sha and recorded["supplied_output_u16_le_sha256"] == output_sha and
            recorded["sample_rows"] == len(row_ids) and recorded["elements"] == len(row_ids) * 256 and recorded["dtype"] == case["dtype"], "Certificate does not reference its actual arrays")
    for field in ("scale", "unit_roundoff_u32", "eta32", "gamma9"):
        same_fraction(recorded[field], fresh[field], field)
    require(recorded["summary"]["all_three_hard_gates_passed"] is True and len(recorded["rows"]) == len(row_ids), "Incomplete hard certificate rows")
    for name in HARD_COUNTS:
        require(recorded["summary"]["counts"][name] == fresh["summary"]["counts"][name], "Hard certificate count differs: " + name)
    for name in HARD_MAXIMA:
        same_fraction(recorded["summary"]["maxima"][name], fresh["summary"]["maxima"][name], "summary." + name)
    for index, (original, actual) in enumerate(zip(recorded["rows"], fresh["rows"])):
        require(original["status"] == "PASS" and original["all_three_hard_gates_passed"] is True and
                original["sample_row_index"] == index and original["row"] == row_ids[index] and
                original["common_input_denominator"] == actual["common_input_denominator"], "Row certificate identity/hard gate differs")
        for name in ("input_l1", "relative_pre_bound", "underflow_pre_bound", "pre_bound"):
            same_fraction(original[name], actual[name], "row." + name)
        for name in HARD_MAXIMA:
            same_fraction(original["maxima"][name], actual["maxima"][name], "row.maxima." + name)
        for name in HARD_COUNTS:
            require(original["counts"][name] == actual["counts"][name], "Row hard count differs: " + name)
        observed_worst, computed_worst = original["worst_stored_error_element"], actual["worst_stored_error_element"]
        for name in ("sample_row_index", "row", "column", "gpu_bits", "expected_storage_bits", "cpu_pre_bits",
                     "exact_value", "cpu_pre_value", "gpu_stored_value", "exact_direct_storage_bits", "exact_via_fp32_storage_bits"):
            require(observed_worst[name] == computed_worst[name], "Worst-element exact certificate differs: " + name)
        for name in HARD_MAXIMA:
            same_fraction(observed_worst["errors"][name], computed_worst["errors"][name], "worst." + name)
        same_fraction(observed_worst["stored_bound"], computed_worst["stored_bound"], "worst.stored_bound")
        require(observed_worst["neighbors"] == computed_worst["neighbors"], "Worst-element storage ULP differs")
        require(original["dense_fp64_is_exact"] is (original["counts"]["dense_fp64_inexact_elements"] == 0), "Recorded dense diagnostic flag is internally inconsistent")
    # These diagnostics retain the remote BLAS result. Differences do not alter E32.
    differences = []
    for name in DIAGNOSTIC_MAXIMA:
        remote = rational(recorded["summary"]["maxima"][name], name)
        require(remote == max(rational(row["maxima"][name], "row." + name) for row in recorded["rows"]),
                "Recorded FP64 diagnostic maximum is internally inconsistent")
        local = rational(fresh["summary"]["maxima"][name], name)
        if remote != local:
            differences.append({"metric": name, "recorded_fraction": str(remote), "local_fraction": str(local)})
    for name in DIAGNOSTIC_COUNTS:
        require(all(type(row["counts"][name]) is int and 0 <= row["counts"][name] <= 256 for row in recorded["rows"]) and
                recorded["summary"]["counts"][name] == sum(row["counts"][name] for row in recorded["rows"]),
                "Recorded FP64 diagnostic count is internally inconsistent")
        if recorded["summary"]["counts"][name] != fresh["summary"]["counts"][name]:
            differences.append({"metric": name, "recorded_count": recorded["summary"]["counts"][name], "local_count": fresh["summary"]["counts"][name]})
    return differences


def analyze(args, inputs):
    protocol, manifest, regression, old_failure, ninja_sha = verify_sources(args, inputs)
    protocol_sha = inputs[str(ROOT / "protocol_v2.json")]["sha256"]
    manifest_sha = inputs[str(ROOT / "manifest_v2.json")]["sha256"]
    numeric = import_verified("verified_numeric_certificate_v2", ROOT / "numeric_certificate.py")
    timing_tools = import_verified("verified_fixed_fused_timing_helpers", BASE / "analyze_holdout.py")
    spec = protocol["holdout"]
    require(spec["rows"] == timing_tools.EXPECTED_ROWS and spec["exclude_initial_rows"] == timing_tools.INITIAL_ROWS and
            spec["dtypes"] == ["fp16", "bf16"] and spec["dims"] == [256] and spec["normalized"] == [False, True] and
            spec["expected_configurations"] == 52 and spec["block_threads"] == 128 and spec["layouts"] == ["original", "contiguous256"], "Fixed V2 matrix changed")
    for name, value in (("processes", 3), ("groups", 5), ("api_warmup_calls", 25), ("captured_calls", 64), ("replays_per_group", 20), ("graph_warmup_replays", 5)):
        require(protocol["timing"][name] == value, "Fixed V2 timing protocol changed: " + name)
    cases = [(dtype, 256, rows, norm) for dtype in spec["dtypes"] for rows in spec["rows"] for norm in spec["normalized"]]
    expected = set(cases)
    expected_inputs = {(pattern, seed, offset) for pattern, seeds in spec["patterns_and_seeds"].items() for seed in seeds for offset in (0, 2)}
    require(len(expected_inputs) == 14, "V2 input matrix changed")
    require(len(args.runs) == 3 and len(set(args.runs)) == 3, "Three distinct report files are required")
    reports = [read_json(path, inputs) for path in args.runs]
    require({report["run_index"] for report in reports} == {1, 2, 3} and len({report["pid"] for report in reports}) == 3,
            "Three distinct worker/run identities are required")
    require(len({inputs[str(path)]["sha256"] for path in args.runs}) == 3, "Duplicate report bytes")
    for field in ("environment", "reference", "validation_manifest", "original_regression_gate", "build_model_assumptions"):
        require(len({json.dumps(report[field], sort_keys=True) for report in reports}) == 1, "Cross-run identity differs: " + field)
    sequence = sorted(reports, key=lambda report: report["run_index"])
    for earlier, later in zip(sequence, sequence[1:]):
        require(datetime.fromisoformat(earlier["finished_utc"].replace("Z", "+00:00")) <=
                datetime.fromisoformat(later["started_utc"].replace("Z", "+00:00")), "V2 worker intervals overlap")
    pool, cache, indexed, all_samples = {}, {}, {}, []
    legacy_cases, diagnostic_differences, per_run = [], [], {}
    for path, report in zip(args.runs, reports):
        verify_run_identity(report, protocol, manifest, protocol_sha, manifest_sha, ninja_sha)
        available = load_buffers(report, path, inputs, pool)
        used, seen, condition_count, legacy_count = set(), set(), 0, 0
        official_maxima = {"fp16": 0.0, "bf16": 0.0}
        hard_maxima = {name: Fraction(0) for name in HARD_MAXIMA}
        for case in report["correctness"]:
            key = timing_tools.case_key(case)
            require(key in expected and key not in seen, "Unexpected/duplicate V2 correctness configuration")
            seen.add(key)
            entries = case["checks"]
            require(len(entries) == 14 and {(c["pattern"], c["seed"], c["pointer_mod16"]) for c in entries} == expected_inputs, "V2 condition matrix incomplete")
            limit = 0.01 if case["dtype"] == "fp16" else 0.05
            require(spec["strict_abs_limit"][case["dtype"]] == limit, "Official reference tolerance changed")
            row_ids = sorted({0, case["rows"] // 2, case["rows"] - 1})
            for item in entries:
                context = {"run_index": report["run_index"], **dict(zip(timing_tools.KEYS, key)),
                           "pattern": item["pattern"], "seed": item["seed"], "pointer_mod16": item["pointer_mod16"]}
                for flag in ("pass", "original_candidate_fused_split_exact", "cpu_quantization_exact", "input_guards_unchanged",
                             "legacy_three_arg_default_equals_explicit_original", "certificate_computed_on_cpu_from_actual_gpu_bits",
                             "legacy_rounded_dense_threshold_is_not_a_v2_gate", "dao_copy_bitwise_equal_input"):
                    require(item[flag] is True, "V2 exact/guard flag failed: " + flag)
                require(item["elements"] == case["rows"] * 256 and item["sample_row_ids"] == row_ids and
                        item["dao_input_pointer_mod16"] == 0 and item["dao_aligned_copy_for_offset"] is (item["pointer_mod16"] == 2), "V2 coverage/alignment differs")
                reference = item["official_reference"]
                error = reference["max_abs_error"]
                require(reference["pass"] is True and reference["strict_abs_limit"] == limit and
                        type(error) in (int, float) and math.isfinite(error) and 0 <= error < limit and
                        reference["elements_at_or_above_strict_limit"] == 0, "Unchanged official all-element reference gate failed")
                official_maxima[case["dtype"]] = max(official_maxima[case["dtype"]], error)
                in_sha, out_sha = item["sample_input_bits_sha256"], item["sample_gpu_bits_sha256"]
                require(in_sha in available and out_sha in available, "Certificate sample buffer is missing from this report's sidecar")
                used.update((in_sha, out_sha))
                require(pool[in_sha].shape == pool[out_sha].shape == (len(row_ids), 256), "Certificate sample row shape differs")
                cache_key = (case["dtype"], Fraction(str(case["scale"])), tuple(row_ids), in_sha, out_sha)
                if cache_key not in cache:
                    fresh = numeric.certify_samples(pool[in_sha], pool[out_sha], case["dtype"], case["scale"], row_ids=row_ids)
                    require(fresh["status"] == "PASS", "CPU recomputed certificate failed: " +
                            json.dumps({"condition": context, "first_failure": fresh.get("first_failure")}))
                    cache[cache_key] = fresh
                fresh = cache[cache_key]
                certificate = item["numeric_certificate"]
                differences = compare_certificate(certificate, fresh, case, row_ids, in_sha, out_sha)
                if differences:
                    diagnostic_differences.append({**context, "differences": differences, "hard_gate_effect": "none"})
                for name in HARD_MAXIMA:
                    hard_maxima[name] = max(hard_maxima[name], rational(certificate["summary"]["maxima"][name], name))
                legacy_error = rational(certificate["summary"]["maxima"]["stored_error_vs_via_fp32_rounded_dense"], "legacy diagnostic")
                legacy_fail = legacy_error >= Fraction(str(limit))
                require(item["legacy_rounded_dense_threshold_would_fail"] is legacy_fail, "Legacy diagnostic flag disagrees with its recorded fraction")
                if legacy_fail:
                    legacy_cases.append({**context, "recorded_rounded_dense_error_fraction": str(legacy_error), "legacy_limit": limit})
                    legacy_count += 1
                condition_count += 1
        require(seen == expected and len(report["correctness"]) == 52 and condition_count == 728 and used == available,
                "V2 52/728 matrix incomplete or sample sidecar has unreferenced buffers")
        require(report["summary"] == {"distinct_configurations": 52, "correctness_input_conditions": 728,
                    "graph_comparisons": 52, "v1_samples_imported": 0,
                    "legacy_rounded_dense_threshold_would_fail_conditions": legacy_count}, "V2 summary differs from actual records")
        timings, samples = timing_tools.verify_timings(report, cases)
        indexed[report["run_index"]] = timings
        for sample in samples:
            sample["validation_revision"] = 2
        all_samples.extend(samples)
        per_run[str(report["run_index"])] = {"correctness_conditions": condition_count, "legacy_would_fail_conditions": legacy_count,
            "official_reference_max_abs_error": official_maxima, "hard_maxima_exact_fractions": {name: str(value) for name, value in hard_maxima.items()},
            "sample_file_sha256": report["sample_buffers"]["sha256"], "sample_buffers": len(available)}
    require(len(all_samples) == 1560, "Expected exactly 1560 V2 timing samples")
    rows = timing_tools.compare_runs(indexed)
    for row in rows:
        row["validation_revision"] = 2
    summary = {"status": "VERIFIED", "validation_revision": 2, "protocol_sha256": protocol_sha,
        "validation_manifest_sha256": manifest_sha, "reused_regression": EXPECTED,
        "original_v1_failure_preserved": old_failure, "v1_timing_samples_imported": 0,
        "distinct_configurations": 52, "correctness_conditions_per_process": 728, "processes": 3,
        "raw_timing_samples": 1560, "unique_sample_arrays_verified": len(pool), "unique_cpu_certificate_recomputations": len(cache),
        "certificate_checks_across_reports": 3 * 728, "per_run": per_run,
        "all_52": timing_tools.group_summary(rows),
        "by_dtype": {dtype: timing_tools.group_summary([row for row in rows if row["dtype"] == dtype]) for dtype in ("fp16", "bf16")},
        "all_negative_cases": [row for row in rows if row["any_run_regression"]],
        "all_cases_with_any_regression_over_3_percent": [row for row in rows if row["any_run_regression_over_3_percent"]],
        "legacy_would_fail_diagnostics": legacy_cases, "fp64_cross_environment_diagnostic_differences": diagnostic_differences,
        "fp64_diagnostic_policy": "Original row-level FP64 diagnostics remain in each raw certificate. Cross-BLAS diagnostic differences are recorded separately; E64 never enters E32 and the old rounded-dense threshold is not a V2 gate.",
        "counting_policy": "Same728 conditions repeated across3 processes, with repeated sample mathematics cached. These are not2184 new GPU inputs. Initial24/v1 timing data are excluded. Reused regression scopes are not added to holdout case counts.",
        "cpu_revalidation_environment": {"python": sys.version.split()[0], "numpy": np.__version__,
            "numeric_certificate_sha256": manifest["validation_files"]["numeric_certificate.py"]["sha256"],
            "gpu_executed_by_analyzer": False},
        "scope": timing_tools.SCOPE, "production_dispatch_generated": False, "environment": reports[0]["environment"]}
    return summary, rows, all_samples, timing_tools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--regression-report", type=Path, required=True)
    parser.add_argument("--v1-failure", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.runs = [path.resolve() for path in args.runs]
    args.regression_report = args.regression_report.resolve()
    args.v1_failure = args.v1_failure.resolve() if args.v1_failure else None
    output = args.output.resolve()
    inputs, summary, rows, samples, timing_tools = {}, {"status": "UNVERIFIED", "validation_revision": 2}, [], [], None
    try:
        summary, rows, samples, timing_tools = analyze(args, inputs)
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, ArithmeticError) as error:
        summary["reason"] = f"{type(error).__name__}: {error}"
    protected = {path.resolve() for path in map(Path, inputs)} | set(args.runs) | {args.regression_report}
    require(not any((output / name).resolve() in protected for name in ("analysis.json", "comparison.csv", "samples.csv")), "Output would overwrite evidence")
    output.mkdir(parents=True, exist_ok=True)
    if timing_tools is not None:
        timing_tools.write_csv(output / "comparison.csv", rows)
        timing_tools.write_csv(output / "samples.csv", samples)
    else:
        for name in ("comparison.csv", "samples.csv"):
            (output / name).write_text("status,validation_revision\nUNVERIFIED,2\n", encoding="utf-8")
    summary.update(generated_utc=datetime.now(timezone.utc).isoformat(), inputs=inputs, raw_evidence_modified=False,
        analyzer_sha256=sha(Path(__file__).read_bytes()),
        derived_csv_sha256={name: sha((output / name).read_bytes()) for name in ("comparison.csv", "samples.csv")})
    (output / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "validation_revision": 2, "comparison_rows": len(rows),
                      "raw_timing_samples": len(samples), "reason": summary.get("reason"), "output": str(output)}))
    return 0 if summary["status"] == "VERIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
