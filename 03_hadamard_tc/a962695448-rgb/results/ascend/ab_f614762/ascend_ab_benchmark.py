#!/usr/bin/env python3
"""固定 Ascend vector-scale OFF/ON 交错性能对照；不编译、不改变环境或源码。"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


MATRIX = [(rows, dim) for rows in (1, 17, 257, 4096, 16384) for dim in (16, 64, 128, 256)]
ROUNDS = 3
DTYPES = {"fp16", "bf16"}
METHODS = {"scalar_transform", "vector_transform", "scalar_split", "vector_split",
           "scalar_fused", "vector_fused", "quant_only"}
NATIVE_COLUMNS = (
    "dtype,batch,seq,heads,dim,rows,method,group,order,repeats,kernel_us,logical_io_bytes,logical_GBs,"
    "input_working_set_bytes,seed,input_read_only,scale,block_dim,warmup,timer,vector_scale_enabled,kernel_ms"
).split(",")


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_report(path, report):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def inventory(output, label):
    """只读取可见设备状态，不抓取全环境、凭据或假设具体硬件型号。"""
    executable = shutil.which("npu-smi")
    record = {"captured_utc": utc_now(), "available": executable is not None}
    if executable is None:
        record["state"] = "npu-smi not available on inherited PATH"
        return record
    log = output / ("npu_smi_%s.log" % label)
    record.update(command=[executable, "info"], log=log.name, returncode=None)
    try:
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.run(record["command"], stdout=stream, stderr=subprocess.STDOUT,
                                     check=False, timeout=30)
        record["returncode"] = process.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        record["error"] = str(error)
    if log.is_file():
        record["sha256"] = sha256(log)
    return record


def read_cell(path, cell, variant, block_dim, repeats, warmup, groups, expected_header=None):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        rows = list(reader)
    if header != NATIVE_COLUMNS:
        raise RuntimeError("native 22-column schema required: " + str(path))
    if expected_header is not None and header != expected_header:
        raise RuntimeError("CSV header differs from previous cell: " + str(path))
    expected_count = len(DTYPES) * len(METHODS) * groups
    if len(rows) != expected_count:
        raise RuntimeError("incomplete native CSV: expected %d rows, got %d in %s" % (expected_count, len(rows), path))
    fixed = {"batch": "1", "seq": str(cell[0]), "heads": "1", "dim": str(cell[1]), "rows": str(cell[0]),
             "block_dim": str(block_dim), "repeats": str(repeats), "warmup": str(warmup),
             "vector_scale_enabled": "false" if variant == "old" else "true",
             "timer": "acl_timeline_event_ms", "input_read_only": "true", "scale": "1"}
    seen = set()
    order_seen = {(dtype, group): set() for dtype in DTYPES for group in range(groups)}
    for fields in rows:
        if len(fields) != len(header):
            raise RuntimeError("ragged native CSV row: " + str(path))
        row = dict(zip(header, fields))
        if any(row[key] != value for key, value in fixed.items()):
            raise RuntimeError("shape/parameter/timer/build-flag mismatch: " + str(path))
        dtype, method = row["dtype"], row["method"]
        group, order = int(row["group"]), int(row["order"])
        key = (dtype, group, method)
        if dtype not in DTYPES or method not in METHODS or not 0 <= group < groups or key in seen:
            raise RuntimeError("duplicate or unexpected dtype/method/group: " + str(path))
        if not 0 <= order < len(METHODS) or order in order_seen[(dtype, group)]:
            raise RuntimeError("duplicate or unexpected launch order: " + str(path))
        microseconds, milliseconds = float(row["kernel_us"]), float(row["kernel_ms"])
        if not all(math.isfinite(value) and value > 0 for value in (microseconds, milliseconds)):
            raise RuntimeError("event us/ms must both be finite and positive: " + str(path))
        # 两列均来自原生程序，不用计算结果替换任一列；容差只覆盖12位有效数字序列化。
        if not math.isclose(microseconds, milliseconds * 1000.0, rel_tol=2e-11, abs_tol=0.0):
            raise RuntimeError("native us/ms conversion mismatch: " + str(path))
        seen.add(key)
        order_seen[(dtype, group)].add(order)
    return header, rows


def invoke(binary, variant, cell, round_number, output, args, report, manifest):
    folder = output / ("round_%d" % round_number) / variant / ("m%d_n%d" % cell)
    folder.mkdir(parents=True, exist_ok=False)
    raw = folder / "native.csv"
    log = folder / "process.log"
    command = [str(binary), "--benchmark", "--batch", "1", "--seq", str(cell[0]), "--heads", "1",
               "--dim", str(cell[1]), "--dtype", "both", "--block-dim", str(args.block_dim),
               "--repeats", str(args.repeats), "--warmup", str(args.warmup), "--groups", str(args.groups),
               "--csv", str(raw)]
    record = {"sequence": len(report["executions"]) + 1, "round": round_number, "variant": variant,
              "rows": cell[0], "dim": cell[1], "dtype": "both", "command": command, "cwd": str(folder),
              "binary_sha256_at_start": report["binaries"][variant]["sha256"],
              "log": str(log.relative_to(output)), "native_csv": str(raw.relative_to(output)),
              "started_utc": utc_now(), "returncode": None}
    report["executions"].append(record)
    save_report(manifest, report)
    print("RUN", record["sequence"], "round", round_number, variant, "M", cell[0], "N", cell[1], flush=True)
    start = time.monotonic()
    try:
        # 不传 env、不 source SDK 脚本，完整继承调用者已准备好的运行环境。
        with log.open("x", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=folder, stdout=stream, stderr=subprocess.STDOUT, check=False)
        record["returncode"] = result.returncode
    finally:
        record["finished_utc"] = utc_now()
        record["process_wall_seconds"] = time.monotonic() - start
        record["wall_time_is_not_kernel_timing"] = True
        if log.is_file():
            record["log_sha256"] = sha256(log)
            record["observed_device_stdout"] = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
                                                 if line.startswith("DEVICE ")]
        if raw.is_file():
            record["native_csv_sha256"] = sha256(raw)
        save_report(manifest, report)
    if record["returncode"] != 0:
        raise RuntimeError("benchmark process failed: " + record["log"])
    return record, raw


def merge_round(path, header, rows):
    # 保留每个原始字段字符串与追加顺序，不筛选、不排序、不重算kernel_us/kernel_ms。
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-binary", type=Path, required=True)
    parser.add_argument("--new-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", required=True, help="仅调用方标签，源码/构建hash对应关系须外部核验")
    parser.add_argument("--block-dim", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--groups", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.block_dim <= 32 or not 1 <= args.repeats <= 10000 or not 1 <= args.groups <= 10000 or not 0 <= args.warmup <= 10000:
        parser.error("block-dim must be 1..32; repeats/groups 1..10000; warmup 0..10000")
    binaries = {"old": args.old_binary.resolve(), "new": args.new_binary.resolve()}
    for name, path in binaries.items():
        if not path.is_file() or not os.access(path, os.X_OK):
            parser.error(name + " binary must exist and be executable")
    if binaries["old"] == binaries["new"]:
        parser.error("old/new must be distinct binary paths")
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory exists; choose a new path")
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "ab_summary.json"
    cell_rows = len(DTYPES) * len(METHODS) * args.groups
    report = {"status": "RUNNING", "started_utc": utc_now(), "python_version": sys.version,
              "script_sha256": sha256(Path(__file__).resolve()), "source_id_label": args.source_id,
              "source_id_verification": "label only; caller must independently verify source commit/build hashes for both binaries",
              "comparison": "interleaved independent GPU processes, NOT same-process A/B",
              "interleave_order": "old first iff (zero-based round + zero-based cell index) is even; reversed next round",
              "environment": "inherited unchanged; no compiler, environment setup script, or source mutation is executed",
              "correctness_scope": "existing binary --benchmark per-shape checks only; full-suite validation is external",
              "matrix": [{"rows": rows, "dim": dim, "dtype": "both"} for rows, dim in MATRIX],
              "parameters": {"rounds": ROUNDS, "block_dim": args.block_dim, "repeats": args.repeats,
                             "warmup": args.warmup, "groups": args.groups},
              "binaries": {name: {"path": str(path), "sha256": sha256(path),
                                  "expected_vector_scale_enabled": name == "new"} for name, path in binaries.items()},
              "accounting": {"unique_shape_dtype_cells": len(MATRIX) * len(DTYPES), "expected_processes": ROUNDS * len(MATRIX) * 2,
                             "raw_rows_per_cell_process": cell_rows, "raw_rows_per_variant_round": len(MATRIX) * cell_rows,
                             "expected_observations": ROUNDS * len(MATRIX) * 2 * cell_rows, "verified_observations": 0,
                             "counting_rule": "count native cell CSV rows once; merged copies do not add observations or correctness cases"},
              "executions": [], "merged_csvs": {}, "binary_integrity": {}, "device_inventory": {}}
    save_report(manifest, report)
    code = 1
    try:
        report["device_inventory"]["start"] = inventory(output, "start")
        save_report(manifest, report)
        common_header = None
        for round_index in range(ROUNDS):
            collected = {"old": [], "new": []}
            sources = {"old": [], "new": []}
            for cell_index, cell in enumerate(MATRIX):
                order = ("old", "new") if (round_index + cell_index) % 2 == 0 else ("new", "old")
                for variant in order:
                    record, raw = invoke(binaries[variant], variant, cell, round_index + 1, output, args, report, manifest)
                    header, rows = read_cell(raw, cell, variant, args.block_dim, args.repeats, args.warmup, args.groups, common_header)
                    if common_header is None:
                        common_header = header
                    collected[variant].extend(rows)
                    sources[variant].append(str(raw.relative_to(output)))
                    record["verified_native_rows"] = len(rows)
                    report["accounting"]["verified_observations"] += len(rows)
                    save_report(manifest, report)
            for variant in ("old", "new"):
                merged = output / ("%s_run%d.csv" % (variant, round_index + 1))
                merge_round(merged, common_header, collected[variant])
                report["merged_csvs"][merged.name] = {"sha256": sha256(merged), "rows": len(collected[variant]),
                                                     "source_cells_in_order": sources[variant], "numeric_rewriting": False}
                save_report(manifest, report)
        if len(report["executions"]) != report["accounting"]["expected_processes"]:
            raise RuntimeError("not all fixed benchmark processes ran")
        if report["accounting"]["verified_observations"] != report["accounting"]["expected_observations"]:
            raise RuntimeError("fixed observation count is incomplete")
        report["status"] = "PASS"
        code = 0
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
        report["error"] = "caller interrupted; completed artifacts retained"
        code = 130
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
        print("FAIL", error, file=sys.stderr, flush=True)
    finally:
        for name, path in binaries.items():
            try:
                final_hash = sha256(path)
                report["binary_integrity"][name] = {"final_sha256": final_hash,
                                                   "unchanged": final_hash == report["binaries"][name]["sha256"]}
            except OSError as error:
                report["binary_integrity"][name] = {"unchanged": False, "error": str(error)}
        if not all(item["unchanged"] for item in report["binary_integrity"].values()):
            report["binary_integrity_error"] = "one or more binaries changed or disappeared during the run"
            if report["status"] == "PASS":
                report["status"] = "FAIL"
                code = 1
        report["device_inventory"]["end"] = inventory(output, "end")
        report["finished_utc"] = utc_now()
        save_report(manifest, report)
    print(json.dumps({"status": report["status"], "summary": str(manifest)}, ensure_ascii=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
