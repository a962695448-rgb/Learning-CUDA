#!/usr/bin/env python3
"""壁仞小批量候选的固定留出矩阵；交错独立进程对照，不编译、不搜索参数。"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


MATRIX = [(rows, dim, dtype) for rows in (32, 63, 64, 65)
          for dim in (64, 128, 256) for dtype in ("fp16", "bf16")]
METHODS = {method + "_" + operation for method in ("baseline", "optimized", "warp32")
           for operation in ("transform", "split", "fused")}
GROUPS = 5
REPEATS = 100
ROUNDS = 3


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cell_name(cell):
    rows, dim, dtype = cell
    return "%s_m%d_n%d" % (dtype, rows, dim)


def shape_args(cell):
    rows, dim, dtype = cell
    return ["--batch", "1", "--seq", str(rows), "--heads", "1", "--dim", str(dim), "--dtype", dtype]


def save_report(path, report):
    # 只替换本次新目录中自己的运行清单；原始日志、JSON、CSV 从不覆盖。
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def invoke(command, directory, report, manifest, environment, tag, cell, phase, round_number=None):
    directory.mkdir(parents=True, exist_ok=False)
    log = directory / "process.log"
    record = {"sequence": len(report["executions"]) + 1, "phase": phase, "variant": tag,
              "cell": {"rows": cell[0], "dim": cell[1], "dtype": cell[2]}, "round": round_number,
              "command": [str(item) for item in command], "cwd": str(directory),
              "log": str(log.relative_to(manifest.parent)), "started_utc": utc_now(), "returncode": None}
    report["executions"].append(record)
    save_report(manifest, report)
    print("RUN", record["sequence"], phase, tag, cell_name(cell), "round", round_number, flush=True)
    started = time.monotonic()
    try:
        with log.open("x", encoding="utf-8") as output:
            result = subprocess.run(record["command"], cwd=directory, env=environment,
                                    stdout=output, stderr=subprocess.STDOUT, check=False)
        record["returncode"] = result.returncode
    finally:
        record["finished_utc"] = utc_now()
        record["wall_seconds"] = time.monotonic() - started
        if log.is_file():
            record["log_sha256"] = sha256(log)
        save_report(manifest, report)
    if record["returncode"] != 0:
        raise RuntimeError("nonzero process exit: " + record["log"])
    return record


def check_validation(path, cell, tag):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "PASS" or data.get("full_matrix") is not False:
        raise RuntimeError("targeted validation must PASS with full_matrix=false: " + str(path))
    if data.get("warp32_enabled") is not True:
        raise RuntimeError("validation does not report an enabled Warp32 path: " + str(path))
    # 新候选必须证实开关已生效。旧程序可能早于该元数据字段；缺失保留为未报告，不能冒充已核验。
    if tag == "new" and data.get("small_batch_warp_enabled") is not True:
        raise RuntimeError("new binary does not report small_batch_warp_enabled=true")
    if tag == "old" and data.get("small_batch_warp_enabled") is True:
        raise RuntimeError("old binary unexpectedly enables the small-batch candidate")
    if data.get("balanced_pack_enabled") is True:
        raise RuntimeError("balanced-pack must be disabled for this separate small-batch experiment")
    dtypes = data.get("dtypes", {})
    if set(dtypes) != {cell[2]} or dtypes[cell[2]].get("cases") != 22:
        raise RuntimeError("validation does not match the fixed single-dtype, 22-input-combination cell")
    return data


def read_checked_csv(path, cell, expected_header):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        records = list(reader)
    if not header or len(header) != len(set(header)):
        raise RuntimeError("missing or duplicate CSV header fields: " + str(path))
    if expected_header is not None and header != expected_header:
        raise RuntimeError("raw CSV headers differ; merge aborted: " + str(path))
    required = {"dtype", "batch", "seq", "heads", "dim", "rows", "method", "group", "order", "repeats", "kernel_us"}
    if not required.issubset(header) or len(records) != GROUPS * len(METHODS):
        raise RuntimeError("raw CSV does not contain the complete 9-method x 5-group cell")
    seen = set()
    orders = {group: set() for group in range(GROUPS)}
    for values in records:
        if len(values) != len(header):
            raise RuntimeError("ragged CSV row: " + str(path))
        row = dict(zip(header, values))
        expected = {"dtype": cell[2], "batch": "1", "seq": str(cell[0]), "heads": "1",
                    "dim": str(cell[1]), "rows": str(cell[0]), "repeats": str(REPEATS)}
        if any(row[key] != value for key, value in expected.items()):
            raise RuntimeError("CSV shape/dtype/repeat mismatch: " + str(path))
        group, order = int(row["group"]), int(row["order"])
        key = (group, row["method"])
        if group not in range(GROUPS) or row["method"] not in METHODS or key in seen:
            raise RuntimeError("duplicate or unexpected CSV method/group: " + str(path))
        if order not in range(len(METHODS)) or order in orders[group]:
            raise RuntimeError("duplicate or unexpected CSV launch order: " + str(path))
        elapsed = float(row["kernel_us"])
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError("invalid event duration: " + str(path))
        seen.add(key)
        orders[group].add(order)
    # 返回每一个原始字段，不排序、不筛选、不重新计算任何计时数据。
    return header, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-binary", type=Path, required=True)
    parser.add_argument("--new-binary", type=Path, required=True)
    parser.add_argument("--sdk-root", type=Path, default=Path("/usr/local/birensupa/sdk/latest"))
    parser.add_argument("--output", type=Path, required=True, help="必须不存在的新目录")
    parser.add_argument("--old-source-id", help="仅为调用方标签，脚本不据此宣称核验过源码提交")
    parser.add_argument("--new-source-id", help="仅为调用方标签，脚本不据此宣称核验过源码提交")
    args = parser.parse_args()
    binaries = {"old": args.old_binary.resolve(), "new": args.new_binary.resolve()}
    for name, binary in binaries.items():
        if not binary.is_file() or not os.access(binary, os.X_OK):
            parser.error(name + " binary must be an existing executable")
    if binaries["old"] == binaries["new"]:
        parser.error("old and new must be distinct binary paths")
    destination = args.output.resolve()
    if destination.exists():
        parser.error("output already exists; choose a fresh directory")
    sdk = args.sdk_root.resolve()
    supa, brcc = sdk / "supa", sdk / "brcc"
    if not (supa / "lib").is_dir() or not (brcc / "lib").is_dir():
        parser.error("SDK must contain supa/lib and brcc/lib")
    environment = dict(os.environ, SUPA_PATH=str(supa), BIREN_HOME=str(sdk))
    inherited = os.environ.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = os.pathsep.join([str(supa / "lib"), str(brcc / "lib")] + ([inherited] if inherited else []))
    environment["PATH"] = os.pathsep.join([str(brcc / "bin"), str(supa / "bin"), os.environ.get("PATH", "")])
    destination.mkdir(parents=True, exist_ok=False)
    manifest = destination / "holdout_summary.json"
    report = {"status": "RUNNING", "started_utc": utc_now(), "python_version": sys.version,
              "script_sha256": sha256(Path(__file__).resolve()),
              "scope": "fixed held-out matrix; no compilation or parameter search",
              "comparison": "interleaved separate GPU processes, NOT same-process A/B",
              "pair_order": "old first iff (zero-based round + zero-based cell index) is even; flipped next round",
              "external_quiescence": "caller must avoid concurrent compilation/GPU work; this script does not compile",
              "source_id_verification": "CLI labels only; correspondence to source commits requires external verification",
              "sdk_environment": {key: environment[key] for key in ("SUPA_PATH", "BIREN_HOME", "LD_LIBRARY_PATH")},
              "binaries": {name: {"path": str(binary), "sha256": sha256(binary),
                                  "source_id_label": getattr(args, name + "_source_id")}
                           for name, binary in binaries.items()},
              "matrix": [{"rows": m, "dim": n, "dtype": dtype} for m, n, dtype in MATRIX],
              "coverage": {"shape_dtype_cells": len(MATRIX), "unique_input_parameter_combinations": len(MATRIX) * 22,
                           "implementation_validation_processes": len(MATRIX) * 2,
                           "repeated_api_and_host_selfchecks_are_not_new_coverage": True},
              "benchmark": {"rounds": ROUNDS, "groups": GROUPS, "repeats": REPEATS,
                            "raw_rows_per_binary_round": len(MATRIX) * GROUPS * len(METHODS)},
              "executions": [], "merged_csvs": {}}
    save_report(manifest, report)
    failure_code = 1
    try:
        for index, cell in enumerate(MATRIX):
            for name in (("old", "new") if index % 2 == 0 else ("new", "old")):
                folder = destination / "validation" / name / cell_name(cell)
                output = folder / "validation.json"
                command = [binaries[name], "--validate"] + shape_args(cell) + ["--json", output]
                record = invoke(command, folder, report, manifest, environment, name, cell, "validation")
                record["validation_json"] = str(output.relative_to(destination))
                record["validation_sha256"] = sha256(output)
                save_report(manifest, report)
                record["validation"] = check_validation(output, cell, name)
                save_report(manifest, report)
        canonical_header = None
        for round_index in range(ROUNDS):
            all_rows = {"old": [], "new": []}
            raw_sources = {"old": [], "new": []}
            for index, cell in enumerate(MATRIX):
                order = ("old", "new") if (round_index + index) % 2 == 0 else ("new", "old")
                for name in order:
                    folder = destination / ("round_%d" % (round_index + 1)) / name / cell_name(cell)
                    output = folder / "raw.csv"
                    command = ([binaries[name], "--benchmark"] + shape_args(cell)
                               + ["--groups", str(GROUPS), "--repeats", str(REPEATS), "--csv", output])
                    record = invoke(command, folder, report, manifest, environment, name, cell, "benchmark", round_index + 1)
                    record["raw_csv"] = str(output.relative_to(destination))
                    record["raw_csv_sha256"] = sha256(output)
                    save_report(manifest, report)
                    header, rows = read_checked_csv(output, cell, canonical_header)
                    if canonical_header is None:
                        canonical_header = header
                    all_rows[name].extend(rows)
                    raw_sources[name].append(str(output.relative_to(destination)))
                    record["raw_row_count"] = len(rows)
                    save_report(manifest, report)
            for name in ("old", "new"):
                merged = destination / ("%s_run%d.csv" % (name, round_index + 1))
                with merged.open("x", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(canonical_header)
                    writer.writerows(all_rows[name])
                report["merged_csvs"][merged.name] = {"sha256": sha256(merged), "rows": len(all_rows[name]),
                                                     "sources_in_append_order": raw_sources[name],
                                                     "filtering_or_numeric_rewriting": False}
                save_report(manifest, report)
        for name, binary in binaries.items():
            final_hash = sha256(binary)
            report["binaries"][name]["final_sha256"] = final_hash
            if final_hash != report["binaries"][name]["sha256"]:
                raise RuntimeError(name + " binary changed while experiment was running")
        report["status"] = "PASS"
        failure_code = 0
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED"
        report["error"] = "caller interrupted the fixed experiment; all completed artifacts retained"
        failure_code = 130
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
        print("FAIL", error, file=sys.stderr, flush=True)
    finally:
        report["finished_utc"] = utc_now()
        save_report(manifest, report)
    print(json.dumps({"status": report["status"], "summary": str(manifest)}, ensure_ascii=False), flush=True)
    return failure_code


if __name__ == "__main__":
    raise SystemExit(main())
