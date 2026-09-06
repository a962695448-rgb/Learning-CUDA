#!/usr/bin/env python3
"""Default is a CPU-only plan. --build-only compiles; --execute uses the GPU."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
COMMIT = "9f5fdc363b4149d4a211701f24ab0548084ca3e5"
METHODS = ("old_wmma", "four_warp_wmma", "warp128")
COLUMNS = ("round", "position", "method", "sample", "rows", "n", "dtype", "scale_kind",
           "scale_float_bits", "threads", "grid_x", "grid_y", "shared_bytes",
           "input_offset_bytes", "iterations", "event_elapsed_ms", "kernel_ms", "timer",
           "validation_passed")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

def cases(which):
    result = []
    for n, rows, dtype, scale in itertools.product((16, 32, 64, 128, 256),
                                                  (1, 17, 64, 257, 4096, 16384),
                                                  ("fp16", "bf16"), ("unit", "normalized")):
        screen = n in (16, 64, 256) and rows in (17, 4096)
        if which == "all" or (which == "screen") == screen:
            result.append(dict(n=n, rows=rows, dtype=dtype, scale=scale,
                               case_id=f"n{n}_m{rows}_{dtype}_{scale}",
                               partition="screen" if screen else "holdout"))
    return result

def frozen_sources():
    manifest = json.loads((ROOT / "freeze_manifest.json").read_text(encoding="utf-8"))
    if manifest["baseline_commit"] != COMMIT:
        raise ValueError("wrong frozen baseline commit")
    for name, expected in manifest["sources_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"frozen source changed: {name}")
    return manifest

def parse_csv(path, case, samples, iterations):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("unexpected CSV schema")
        rows = list(reader)
    if len(rows) != 3 * 3 * samples:
        raise ValueError("unexpected event sample count")
    scale = 1.0 if case["scale"] == "unit" else 1.0 / math.sqrt(case["n"])
    scale_bits = struct.unpack("<I", struct.pack("<f", scale))[0]
    groups = {}
    seen = set()
    for row in rows:
        round_id, position, sample = (int(row[x]) for x in ("round", "position", "sample"))
        if not (1 <= round_id <= 3 and 1 <= position <= 3 and 1 <= sample <= samples):
            raise ValueError("invalid round/sample position")
        expected_method = METHODS[(round_id - 1 + position - 1) % 3]
        if row["method"] != expected_method:
            raise ValueError("method order differs from frozen rotation")
        key = (round_id, row["method"], sample)
        if key in seen:
            raise ValueError("duplicate event sample")
        seen.add(key)
        method = METHODS.index(row["method"])
        expected = dict(rows=case["rows"], n=case["n"], scale_float_bits=scale_bits,
                        threads=128, input_offset_bytes=32, iterations=iterations,
                        grid_x=(case["rows"] + (3 if method == 2 else 15)) // (4 if method == 2 else 16),
                        grid_y=case["n"] // 16 if method == 0 else (case["n"] + 63) // 64 if method == 1 else 1,
                        shared_bytes=0 if method == 2 else 32 * case["n"] + (1 if method == 0 else 4) * 1024)
        if any(int(row[name]) != value for name, value in expected.items()):
            raise ValueError("CSV configuration/geometry mismatch")
        if (row["dtype"] != case["dtype"] or row["scale_kind"] != case["scale"] or
                row["timer"] != "cuda_event_batched_launches" or row["validation_passed"] != "true"):
            raise ValueError("CSV scope/validation mismatch")
        event_ms, kernel_ms = (float(row[x]) for x in ("event_elapsed_ms", "kernel_ms"))
        if not all(math.isfinite(x) and x > 0 for x in (event_ms, kernel_ms)):
            raise ValueError("nonfinite/nonpositive event duration")
        if not math.isclose(event_ms / iterations, kernel_ms, rel_tol=1e-14, abs_tol=0.0):
            raise ValueError("event/kernel milliseconds disagree")
        groups.setdefault((round_id, row["method"]), []).append(kernel_ms)
    summary = []
    for (round_id, method), values in sorted(groups.items()):
        summary.append(dict(round=round_id, method=method, raw_samples_ms=values,
                            median_ms=statistics.median(values), minimum_ms=min(values), maximum_ms=max(values)))
    comparisons = []
    for round_id in range(1, 4):
        med = {m: statistics.median(groups[(round_id, m)]) for m in METHODS}
        comparisons.append(dict(round=round_id,
                                old_over_four_warp=med[METHODS[0]] / med[METHODS[1]],
                                four_warp_time_reduction=1 - med[METHODS[1]] / med[METHODS[0]],
                                warp128_over_four_warp=med[METHODS[2]] / med[METHODS[1]]))
    return dict(event_rows=len(rows), statistics=summary, same_round_comparisons=comparisons,
                all_three_rounds_faster_than_old=all(c["four_warp_time_reduction"] > 0 for c in comparisons),
                all_three_rounds_at_least_5_percent_faster_than_old=all(c["four_warp_time_reduction"] >= .05 for c in comparisons))

def screen_gate(summary_path, manifest):
    path = Path(summary_path).resolve()
    screen = json.loads(path.read_text(encoding="utf-8"))
    if (screen.get("status") != "PASS" or screen.get("partition") != "screen" or
            screen.get("sources_sha256") != manifest["sources_sha256"]):
        raise ValueError("screen has not passed with the same frozen sources")
    expected_cases = {c["case_id"]: c for c in cases("screen")}
    records = screen.get("records", [])
    if len(records) != 24 or {r["case"]["case_id"] for r in records} != set(expected_cases):
        raise ValueError("screen must contain exactly all 24 fixed configurations")
    eligible = []
    for record in records:
        case = record["case"]
        if case != expected_cases[case["case_id"]] or record.get("exit_code") != 0:
            raise ValueError("screen contains failed/mismatched configuration")
        raw_csv = path.parent / (case["case_id"] + ".csv")
        raw_validation = path.parent / (case["case_id"] + ".json")
        if sha(raw_csv) != record["csv_sha256"] or sha(raw_validation) != record["validation_sha256"]:
            raise ValueError("screen raw evidence hash mismatch")
        validation = json.loads(raw_validation.read_text(encoding="utf-8"))
        if validation != record["validation"] or validation.get("status") != "PASS":
            raise ValueError("screen numerical validation has not passed")
        comparison = parse_csv(raw_csv, case, screen["samples_per_method_round"], screen["iterations_per_event"])
        if comparison["all_three_rounds_at_least_5_percent_faster_than_old"]:
            eligible.append(case["case_id"])
    if not eligible:
        raise ValueError("STOP: no screen configuration is >=5% faster in all three rounds")
    return dict(screen_summary=str(path), screen_summary_sha256=sha(path),
                all_24_numerical_configurations_passed=True,
                threshold="at least one configuration with >=5% reduction in all three rounds",
                qualifying_configurations=eligible, fixed_holdout_configurations=96)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=("screen", "holdout", "all"), default="screen")
    parser.add_argument("--arch", choices=("80", "86", "89", "90"), default="80")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--screen-summary", type=Path, help="required evidence gate for holdout execution")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", action="store_true")
    group.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if args.execute and args.set == "all":
        parser.error("execute screen first, then gated holdout; --set all is available for planning/build-only")
    if args.execute and args.set == "holdout" and args.screen_summary is None:
        parser.error("holdout execution requires --screen-summary from the completed screen run")
    if not (1 <= args.samples <= 1000 and 1 <= args.iterations <= 10000 and 1 <= args.warmup <= 10000):
        parser.error("sample/iteration/warmup counts exceed the C++ experiment bounds")
    manifest = frozen_sources()
    gate = screen_gate(args.screen_summary, manifest) if args.execute and args.set == "holdout" else None
    plan = dict(baseline_commit=COMMIT, partition=args.set, cases=cases(args.set), rounds=3,
                methods=METHODS, samples_per_method_round=args.samples, iterations_per_event=args.iterations,
                warmup_per_method_round=args.warmup, gpu_execution_requested=args.execute,
                sources_sha256=manifest["sources_sha256"], matrix_construction_in_timing=False,
                timer="CUDA event milliseconds; batched launches; no CUDA Graph")
    if gate is not None:
        plan["screen_gate"] = gate
    if not (args.execute or args.build_only):
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    if args.output is None:
        parser.error("--output is required for build-only/execute")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = output / "benchmark"
    command = [args.nvcc, "-O3", "-std=c++17", "-lineinfo", f"-arch=sm_{args.arch}",
               "-Xptxas=-v", str(ROOT / "benchmark.cu"), "-o", str(binary)]
    plan.update(compile_command=command, arch=args.arch, records=[], status="BUILDING")
    write_json(output / "run_summary.json", plan)
    version = subprocess.run([args.nvcc, "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (output / "nvcc_version.txt").write_text(version.stdout, encoding="utf-8")
    started = time.time()
    with (output / "compile.log").open("w", encoding="utf-8") as log:
        build = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    plan.update(compile_exit_code=build.returncode, compile_elapsed_seconds=time.time()-started)
    if build.returncode:
        plan["status"] = "BUILD_FAILED"
        write_json(output / "run_summary.json", plan)
        return build.returncode
    plan["binary_sha256"] = sha(binary)
    plan["status"] = "COMPILED_ONLY"
    write_json(output / "run_summary.json", plan)
    if args.build_only:
        print(f"COMPILED_ONLY {binary}; no GPU run")
        return 0
    try:
        for case in plan["cases"]:
            prefix = output / case["case_id"]
            command = [str(binary), "--rows", str(case["rows"]), "--n", str(case["n"]),
                       "--dtype", case["dtype"], "--scale", case["scale"], "--rounds", "3",
                       "--samples", str(args.samples), "--iterations", str(args.iterations),
                       "--warmup", str(args.warmup), "--output-prefix", str(prefix)]
            with Path(str(prefix)+".log").open("w", encoding="utf-8") as log:
                run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            record = dict(case=case, command=command, exit_code=run.returncode)
            plan["records"].append(record)
            if run.returncode:
                raise RuntimeError(f"GPU process failed: {case['case_id']}")
            validation_path, csv_path = Path(str(prefix)+".json"), Path(str(prefix)+".csv")
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            flags = ("four_warp_bitwise_equal_old_wmma", "all_methods_dense_rounded_bitwise",
                     "input_and_H_unchanged", "output_guards_intact")
            if validation.get("status") != "PASS" or not all(validation.get(x) is True for x in flags):
                raise ValueError("missing correctness evidence")
            expected = dict(source_commit=COMMIT, rows=case["rows"], n=case["n"], dtype=case["dtype"],
                            scale_kind=case["scale"], unique_shape_dtype_scale_cases=1,
                            guard_layouts=[32, 34], rounds=3, samples_per_method_round=args.samples,
                            iterations_per_event=args.iterations, raw_event_rows=9*args.samples)
            if any(validation.get(k) != v for k, v in expected.items()):
                raise ValueError("validation configuration mismatch")
            general = validation.get("general_input_group", {})
            general_expected = dict(generator="uniform24_v1_seed_0x6e4d21b3_exponents_minus12_to_0",
                                    four_warp_bitwise_equal_old_wmma=True,
                                    old_new_element_comparisons=case["rows"]*case["n"],
                                    dense_rows=min(32, case["rows"]), guard_layout_bytes=32,
                                    strict_rounded_fp64_tolerance=.01 if case["dtype"] == "fp16" else .05,
                                    input_and_H_unchanged=True, output_guards_intact=True,
                                    method_order=list(METHODS))
            if any(general.get(k) != v for k, v in general_expected.items()):
                raise ValueError("general-input correctness evidence missing/mismatched")
            for field in ("rounded_max_abs_error", "unrounded_max_abs_error"):
                values = general.get(field, [])
                if len(values) != 3 or not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in values):
                    raise ValueError("invalid general-input error statistics")
            if not all(x < general_expected["strict_rounded_fp64_tolerance"] for x in general["rounded_max_abs_error"]):
                raise ValueError("general-input error exceeds original strict threshold")
            if validation.get("round_process_scope") != "three rounds in one configuration process":
                raise ValueError("round process scope is missing")
            record.update(validation_sha256=sha(validation_path), csv_sha256=sha(csv_path),
                          log_sha256=sha(str(prefix)+".log"), validation=validation,
                          analysis=parse_csv(csv_path, case, args.samples, args.iterations))
            plan["status"] = "RUNNING"
            write_json(output / "run_summary.json", plan)
            print(f"PASS {case['case_id']}", flush=True)
    except Exception as error:
        plan.update(status="FAILED", error=str(error))
        write_json(output / "run_summary.json", plan)
        raise
    plan.update(status="PASS", unique_shape_dtype_scale_cases=len(plan["cases"]),
                total_event_rows=sum(r["analysis"]["event_rows"] for r in plan["records"]))
    write_json(output / "run_summary.json", plan)
    print(f"PASS {len(plan['cases'])} unique configurations; {plan['total_event_rows']} event rows")
    return 0

if __name__ == "__main__":
    sys.exit(main())
