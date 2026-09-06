#!/usr/bin/env python3
"""CPU-only audit of the frozen 4090 WMMA experiment; never starts GPU work."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import statistics
import struct
import sys

COMMIT = "9f5fdc363b4149d4a211701f24ab0548084ca3e5"
MANIFEST_SHA = "2b306ad035c9052352536344adc1c982389509f843b6231cd2c5a0f91eea0832"
TRANSFER_SHA = "24148778e5b275ca99ade64581dd14dda1c9ad3321812423211d738303bb7609"
TRANSFER_ZIP_SHA = "f8092ca0752476e78c2900c270c4487268cb0db884fe43c0ccfa5d289caaea02"
METHODS = ("old_wmma", "four_warp_wmma", "warp128")
CSV_COLUMNS = ("round", "position", "method", "sample", "rows", "n", "dtype", "scale_kind",
               "scale_float_bits", "threads", "grid_x", "grid_y", "shared_bytes", "input_offset_bytes",
               "iterations", "event_elapsed_ms", "kernel_ms", "timer", "validation_passed")

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    def reject(value):
        raise ValueError(f"nonstandard JSON constant {value} in {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def write_archive_packaging(archive):
    # Protect original raw/source byte hashes when this directory enters Git.
    (archive/".gitattributes").write_bytes(b"** -text\n")
    readme = """# WMMA 实验原始归档与离线复算

从本目录执行以下命令（Python 3.9 或更新版本，仅使用标准库）：

```bash
python analyze.py --raw-root raw --source-root sources --output ../wmma-offline-recomputed
```

输出目录必须尚不存在。命令只读取本归档，在相邻新目录生成 RESULTS.md、summary.json、全部配置轮次与负例 CSV；不访问 GPU、不编译、不联网，也不修改原始文件。

`raw/` 保留取回的 377 个原始文件及传输清单；`sources/` 提供冻结的七个源/协议文件。离线程序重算文件 size/SHA、120 个配置和 21,600 条事件数据。原传输 ZIP 没有重复放入本归档，复算会明确记录 ZIP 未在本地重算，数据分析结论不受影响。

三轮是在每个配置的同一个 C++ 进程内执行，不能视为三次独立进程重复。统计解释与负例见 RESULTS.md。二进制不包含在归档内。

`.gitattributes` 使用 `** -text`，避免 Git 自动转换换行符而破坏原字节与校验值；请保留该文件。`archive_manifest.json` 列出其余全部归档文件的摘要，清单不自包含。
"""
    (archive/"README.md").write_bytes(readme.encode("utf-8"))

def verify_transfer(root):
    manifest_path=root/"transfer_manifest.json"
    require(sha(manifest_path)==TRANSFER_SHA,"retrieval manifest differs from the root-verified transfer")
    manifest=read_json(manifest_path)
    require(len(manifest)==376,"transfer object count changed")
    selected=[]
    for name,expected in manifest.items():
        path=(root/name).resolve()
        require(path.is_relative_to(root) and path.is_file(),"unsafe/missing transfer path")
        require(path.stat().st_size==expected["bytes"] and sha(path)==expected["sha256"],f"transfer size/hash mismatch: {name}")
        selected.append(path)
    selected.append(manifest_path)
    require({p.resolve() for p in root.rglob("*") if p.is_file()}==set(selected),"retrieved inventory differs from 377 frozen files")
    archive=root.parent/"server_transfer.zip"
    if archive.exists():
        require(archive.stat().st_size==596806 and sha(archive)==TRANSFER_ZIP_SHA,"retrieved ZIP hash/size mismatch")
    return selected,dict(manifest_sha256=TRANSFER_SHA,verified_objects=376,files_including_manifest=377,
                         all_object_sizes_and_hashes_passed=True,zip_sha256=TRANSFER_ZIP_SHA,
                         zip_bytes=596806,zip_locally_rehashed=archive.exists())

def case_list(partition):
    result = []
    for n, rows, dtype, scale in itertools.product((16, 32, 64, 128, 256),
                                                  (1, 17, 64, 257, 4096, 16384),
                                                  ("fp16", "bf16"), ("unit", "normalized")):
        screen = n in (16, 64, 256) and rows in (17, 4096)
        part = "screen" if screen else "holdout"
        if partition == "all" or part == partition:
            result.append(dict(n=n, rows=rows, dtype=dtype, scale=scale,
                               case_id=f"n{n}_m{rows}_{dtype}_{scale}", partition=part))
    return result

def finite_number(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0))

def expected_scale_bits(case):
    value = 1.0 if case["scale"] == "unit" else 1 / math.sqrt(case["n"])
    return struct.unpack("<I", struct.pack("<f", value))[0]

def validate_case_json(data, case):
    count = case["rows"] * case["n"]
    expected = dict(status="PASS", source_commit=COMMIT, rows=case["rows"], n=case["n"], dtype=case["dtype"],
                    scale_kind=case["scale"], scale_float_bits=expected_scale_bits(case),
                    unique_shape_dtype_scale_cases=1, guard_layouts=[32, 34], post_round_rechecks=3,
                    four_warp_bitwise_equal_old_wmma=True, all_methods_dense_rounded_bitwise=True,
                    input_generator="dyadic_v1_seed_0x96269544", input_and_H_unchanged=True,
                    output_guards_intact=True, repeated_output_element_comparisons=count*15,
                    rounds=3, samples_per_method_round=20, iterations_per_event=100,
                    warmup_per_method_round=10, raw_event_rows=180,
                    round_process_scope="three rounds in one configuration process",
                    compute_capability="8.9", warp_size=32)
    for key, value in expected.items():
        require(data.get(key) == value and (not isinstance(value, bool) or type(data[key]) is bool),
                f"{case['case_id']}: incorrect validation field {key}")
    require("4090" in data.get("device", ""), "dataset is not from a reported 4090")
    for key in ("runtime_version", "driver_version", "sm_count", "total_global_memory"):
        require(type(data.get(key)) is int and data[key] > 0, f"invalid device metadata {key}")
    for key in ("max_unrounded_fp64_abs_error", "max_unrounded_fp64_relative_to_max_1"):
        require(finite_number(data.get(key)), f"invalid dyadic error {key}")
    require(data.get("timing_excludes") == ["H construction", "allocation", "H2D/D2H", "validation", "warmup"],
            "timing exclusions changed")
    require(data.get("timer") == "CUDA event elapsed ms / batched launch count; no CUDA Graph", "timer scope changed")
    general = data.get("general_input_group", {})
    expected_general = dict(generator="uniform24_v1_seed_0x6e4d21b3_exponents_minus12_to_0",
                            four_warp_bitwise_equal_old_wmma=True, old_new_element_comparisons=count,
                            dense_rows=min(32, case["rows"]), guard_layout_bytes=32,
                            strict_rounded_fp64_tolerance=.01 if case["dtype"] == "fp16" else .05,
                            input_and_H_unchanged=True, output_guards_intact=True, method_order=list(METHODS))
    for key, value in expected_general.items():
        require(general.get(key) == value and (not isinstance(value, bool) or type(general[key]) is bool),
                f"{case['case_id']}: incorrect general-input field {key}")
    for key in ("rounded_max_abs_error", "unrounded_max_abs_error"):
        require(len(general.get(key, [])) == 3 and all(finite_number(v) for v in general[key]), f"invalid general-input {key}")
    require(all(v < general["strict_rounded_fp64_tolerance"] for v in general["rounded_max_abs_error"]),
            "general-input error failed original strict tolerance")
    require(len(general.get("rounded_bit_mismatches", [])) == 3 and
            all(type(v) is int and 0 <= v <= general["dense_rows"] * case["n"] for v in general["rounded_bit_mismatches"]),
            "invalid general-input bit-mismatch counts")
    require(general["rounded_max_abs_error"][0] == general["rounded_max_abs_error"][1] and
            general["unrounded_max_abs_error"][0] == general["unrounded_max_abs_error"][1] and
            general["rounded_bit_mismatches"][0] == general["rounded_bit_mismatches"][1],
            "old/new general-input error metadata contradicts bitwise equivalence")

def read_case_csv(path, case):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == CSV_COLUMNS, f"CSV schema mismatch: {path}")
        rows = list(reader)
    require(len(rows) == 180, f"expected 180 event rows: {path}")
    groups = {(r, m): [] for r in (1, 2, 3) for m in METHODS}
    seen = set()
    for index, row in enumerate(rows):
        round_id = int(row["round"]); position = int(row["position"]); sample = int(row["sample"])
        expected_round = index // 60 + 1
        expected_sample = index % 60 // 3 + 1
        expected_position = index % 3 + 1
        require((round_id, sample, position) == (expected_round, expected_sample, expected_position), "event chronology changed")
        method = METHODS[(round_id + position - 2) % 3]
        require(row["method"] == method, "method rotation changed")
        key = (round_id, method, sample)
        require(key not in seen, "duplicate event row"); seen.add(key)
        method_id = METHODS.index(method)
        expected = dict(rows=case["rows"], n=case["n"], scale_float_bits=expected_scale_bits(case), threads=128,
                        input_offset_bytes=32, iterations=100,
                        grid_x=(case["rows"]+3)//4 if method_id==2 else (case["rows"]+15)//16,
                        grid_y=case["n"]//16 if method_id==0 else (case["n"]+63)//64 if method_id==1 else 1,
                        shared_bytes=0 if method_id==2 else 32*case["n"]+(1024 if method_id==0 else 4096))
        for name, value in expected.items():
            require(int(row[name]) == value, f"CSV field mismatch {name}")
        require(row["dtype"] == case["dtype"] and row["scale_kind"] == case["scale"], "CSV dtype/scale mismatch")
        require(row["timer"] == "cuda_event_batched_launches" and row["validation_passed"] == "true", "CSV validation/timer mismatch")
        event_ms, kernel_ms = float(row["event_elapsed_ms"]), float(row["kernel_ms"])
        require(finite_number(event_ms, True) and finite_number(kernel_ms, True), "nonfinite/nonpositive event")
        require(math.isclose(event_ms / 100, kernel_ms, rel_tol=1e-14, abs_tol=0), "event/kernel ms relationship mismatch")
        groups[(round_id, method)].append(kernel_ms)
    rounds = []
    for round_id in (1, 2, 3):
        med = {m: statistics.median(groups[(round_id, m)]) for m in METHODS}
        rounds.append(dict(round=round_id, old_wmma_ms=med[METHODS[0]], four_warp_wmma_ms=med[METHODS[1]], warp128_ms=med[METHODS[2]],
                           old_over_new=med[METHODS[0]]/med[METHODS[1]], warp128_over_new=med[METHODS[2]]/med[METHODS[1]],
                           reduction_vs_old=1-med[METHODS[1]]/med[METHODS[0]], reduction_vs_warp128=1-med[METHODS[1]]/med[METHODS[2]]))
    return rounds, groups

def compare_recorded_analysis(recorded, rounds, groups):
    require(recorded.get("event_rows") == 180, "recorded analysis sample count mismatch")
    saved_stats = recorded.get("statistics", [])
    require(len(saved_stats) == 9, "recorded statistics count mismatch")
    found = set()
    for stat in saved_stats:
        key = (stat["round"], stat["method"])
        require(key in groups and key not in found, "recorded statistic key mismatch"); found.add(key)
        require(stat["raw_samples_ms"] == groups[key], "recorded raw statistic values differ from CSV")
        require(stat["median_ms"] == statistics.median(groups[key]) and stat["minimum_ms"] == min(groups[key]) and
                stat["maximum_ms"] == max(groups[key]), "recorded statistic differs from independent recomputation")
    saved_rounds = recorded.get("same_round_comparisons", [])
    require(len(saved_rounds) == 3, "recorded comparison count mismatch")
    for saved, derived in zip(saved_rounds, rounds):
        require(saved["round"] == derived["round"] and saved["old_over_four_warp"] == derived["old_over_new"] and
                saved["warp128_over_four_warp"] == derived["warp128_over_new"] and
                saved["four_warp_time_reduction"] == derived["reduction_vs_old"], "recorded ratio differs from CSV")
    require(recorded.get("all_three_rounds_faster_than_old") == all(r["reduction_vs_old"] > 0 for r in rounds), "recorded win flag mismatch")
    require(recorded.get("all_three_rounds_at_least_5_percent_faster_than_old") == all(r["reduction_vs_old"] >= .05 for r in rounds), "recorded 5% flag mismatch")

def scope_summary(cases, field):
    ratios = [r[field] for c in cases for r in c["rounds"]]
    min_cell = min((r[field], c["case_id"], r["round"]) for c in cases for r in c["rounds"])
    max_cell = max((r[field], c["case_id"], r["round"]) for c in cases for r in c["rounds"])
    faster = sum(all(r[field] > 1 for r in c["rounds"]) for c in cases)
    slower = sum(all(r[field] < 1 for r in c["rounds"]) for c in cases)
    reduction = "reduction_vs_old" if field == "old_over_new" else "reduction_vs_warp128"
    return dict(configurations=len(cases), all_three_rounds_faster=faster, all_three_rounds_slower=slower,
                mixed_or_tied=len(cases)-faster-slower,
                all_three_rounds_at_least_5_percent_faster=sum(all(r[reduction] >= .05 for r in c["rounds"]) for c in cases),
                any_round_slower=sum(any(r[field] < 1 for r in c["rounds"]) for c in cases),
                min_same_round_speedup=min(ratios), max_same_round_speedup=max(ratios),
                min_case=dict(speedup=min_cell[0],case_id=min_cell[1],round=min_cell[2]),
                max_case=dict(speedup=max_cell[0],case_id=max_cell[1],round=max_cell[2]))

def load_stage(root, partition, manifest):
    directory = root / partition
    summary_file = directory / "run_summary.json"
    data = read_json(summary_file)
    planned = case_list(partition)
    require(data.get("status") == "PASS" and data.get("partition") == partition, f"{partition} is incomplete/failed")
    require(data.get("baseline_commit") == COMMIT and data.get("sources_sha256") == manifest["sources_sha256"], "source identity mismatch")
    require(data.get("cases") == planned, f"{partition} configuration list changed")
    require(data.get("rounds") == 3 and data.get("methods") == list(METHODS), "run protocol changed")
    require(data.get("samples_per_method_round") == 20 and data.get("iterations_per_event") == 100 and
            data.get("warmup_per_method_round") == 10, "timing settings differ from fixed 21600-row protocol")
    require(data.get("compile_exit_code") == 0 and data.get("arch") == "89" and data.get("gpu_execution_requested") is True, "not a successful sm89 GPU execution")
    require(data.get("matrix_construction_in_timing") is False, "matrix construction scope differs")
    require(data.get("unique_shape_dtype_scale_cases") == len(planned) and data.get("total_event_rows") == len(planned)*180, "run totals disagree")
    cmd = data.get("compile_command", [])
    require(len(cmd)==9 and cmd[1:6] == ["-O3","-std=c++17","-lineinfo","-arch=sm_89","-Xptxas=-v"] and
            cmd[6].endswith("/benchmark.cu") and cmd[7]=="-o", "compile flags/source changed")
    binary_sha = data.get("binary_sha256", "")
    require(len(binary_sha)==64 and all(x in "0123456789abcdef" for x in binary_sha), "missing binary hash")
    binary = directory / "benchmark"
    if binary.exists():
        require(sha(binary)==binary_sha, "retrieved private binary hash mismatch")
    selected = [summary_file, directory / "compile.log", directory / "nvcc_version.txt"]
    require(all(p.is_file() for p in selected), "missing compilation evidence")
    require("release 12.8" in (directory/"nvcc_version.txt").read_text(encoding="utf-8"), "compiler version differs from CUDA12.8")
    records = data.get("records", [])
    require(len(records) == len(planned), "incomplete record count")
    result = []
    for expected_case, record in zip(planned, records):
        require(record.get("case") == expected_case and record.get("exit_code") == 0, "configuration process failed/reordered")
        case = dict(expected_case)
        prefix = directory / case["case_id"]
        raw_json, raw_csv, raw_log = (Path(str(prefix)+suffix) for suffix in (".json", ".csv", ".log"))
        for path, key in ((raw_json,"validation_sha256"),(raw_csv,"csv_sha256"),(raw_log,"log_sha256")):
            require(path.is_file() and sha(path) == record.get(key), f"raw source hash mismatch: {path}")
            selected.append(path)
        validation = read_json(raw_json)
        require(validation == record.get("validation"), "embedded validation differs from original JSON")
        validate_case_json(validation, case)
        command = record.get("command", [])
        expected_args = ["--rows",str(case["rows"]),"--n",str(case["n"]),"--dtype",case["dtype"],"--scale",case["scale"],
                         "--rounds","3","--samples","20","--iterations","100","--warmup","10","--output-prefix"]
        require(len(command)==19 and command[0]==cmd[-1] and command[1:-1]==expected_args and command[-1].endswith("/"+case["case_id"]),
                "case command differs from traceable binary/protocol")
        rounds, groups = read_case_csv(raw_csv, case)
        compare_recorded_analysis(record.get("analysis", {}), rounds, groups)
        case.update(rounds=rounds,validation=validation)
        result.append(case)
    return result, selected, data, dict(stage=partition,binary_sha256=binary_sha,
                                      binary_locally_rehashed=binary.exists(),binary_in_public_archive=False,
                                      compile_command=cmd,run_summary_sha256=sha(summary_file))

def results_text(summary):
    all_scopes = summary["comparisons"]
    lines = ["# RTX 4090 四 warp WMMA 输入复用实验", "",
             "全部 120 个配置、21,600 个原始 CUDA event 行完成核验。24 个筛选配置和预先冻结的 96 个留出配置均保留；结论由 CSV 重新计算。", "",
             "| 范围 | 对照 | 三轮都快 | 三轮都慢 | 混合/相等 | 三轮均降低 ≥5% | 单轮加速比范围 |",
             "|---|---|---:|---:|---:|---:|---|" ]
    for scope,label in (("screen","筛选 24"),("holdout","留出 96"),("all","全部 120")):
        for comparator,name in (("old_wmma","旧 WMMA"),("warp128","warp128")):
            v=all_scopes[scope][comparator]
            lines.append(f"| {label} | {name} | {v['all_three_rounds_faster']} | {v['all_three_rounds_slower']} | {v['mixed_or_tied']} | {v['all_three_rounds_at_least_5_percent_faster']} | {v['min_same_round_speedup']:.4f}–{v['max_same_round_speedup']:.4f}× |")
    lines += ["", "加速比为同配置、同轮次的“对照时间 / 四 warp 时间”，大于 1 表示四 warp 更快。三轮均降低 5% 是预定的继续验证门槛，不是显著性检验。"]
    for comparator,name in (("old_wmma","旧 WMMA"),("warp128","warp128")):
        v=all_scopes["all"][comparator]; worst=v["min_case"];best=v["max_case"]
        lines += ["",f"相对{name}：最差 `{worst['case_id']}` 第 {worst['round']} 轮为 {worst['speedup']:.4f}×；最好 `{best['case_id']}` 第 {best['round']} 轮为 {best['speedup']:.4f}×。所有配置/轮次见 `case_rounds.csv`，任一轮退化的配置见 `negative_cases.csv`。"]
    narrow=summary["narrow_ranges"]
    n256=summary["by_n"]["256"]["warp128"]
    lines += ["",f"N64/128/256 的 {narrow['n64_and_larger_all_three_slower_than_warp128']} 个配置全部三轮慢于 warp128；N256 的四 warp 耗时为 warp128 的 {1/n256['max_same_round_speedup']:.2f}–{1/n256['min_same_round_speedup']:.2f} 倍。",
              "",f"局部胜例集中在 M16384 的 N16/N32。三轮同时战胜两个对照的配置有 {len(narrow['all_three_faster_than_both'])} 个，三轮相对两个对照都降低 ≥5% 的有 {len(narrow['all_three_at_least_5_percent_faster_than_both'])} 个："+
              "、".join(f"`{c}`" for c in narrow['all_three_at_least_5_percent_faster_than_both'])+"。N16 实际只有一个有效计算 warp，不能把其相对 FWHT 的收益归因为四个 warp 复用 A；这也不同于推广整个 WMMA 方案。"]
    q=summary["correctness"]
    transfer_sentence=("377 个取回文件及原 ZIP 的 size/SHA 均独立复核通过" if summary["transfer_verification"]["zip_locally_rehashed"]
                       else "377 个取回文件的 size/SHA 均独立复核通过；此次复算未提供原 ZIP，其摘要仅记录为传输来源元数据")
    lines += ["",f"数值方面：两 dtype 的 dyadic 组全部输出符合独立稠密 FP64→合同舍入的位级期望；一般随机指数组 old/new 全部 {q['general_old_new_elements']:,} 个元素逐位一致。该组 FP64 抽样覆盖 {q['general_dense_rows']:,} 行，各方法按原 FP16 <0.01 / BF16 <0.05 门槛验证；实际 rounded 最大误差与位差分开保存在 `summary.json`，没有要求这组 warp 与 WMMA 位相同。",
              "", "输入与 H 不变、输出 guard 均通过；dyadic 验证含 32/34 字节偏移，一般随机组使用 32 字节偏移。冻结 C++ 使用显式非阻塞 stream 完成初始化、传输、kernel、事件与回读；本实验未增加跨 stream 并发测试。",
              "", "每个配置在一个 C++ 进程内执行三轮，配置间是串行独立进程；不是三次独立进程重复。CPU 参考与 GPU 计时不并行。记录的是非 Graph 的 CUDA event 批量发射时间，矩阵构建、分配、拷贝和验证均排除；不能与另一套 CUDA Graph 测量直接相除。",
              "", "当前保持实验对照，不自动整合 Tensor Core 生产派发。战胜旧 WMMA 不能替代与 warp128 的比较；任何后续采用仍需要生产完整矩阵与适用范围证据。"+transfer_sentence+"，原始负例和全部 raw 文件原字节保留；二进制仅有运行器记录的 SHA，未取回也不进入公开归档。末尾 nvidia-smi 快照不是整个计时期间独占 GPU 的证明。", ""]
    return "\n".join(lines)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root",type=Path,required=True)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--archive",action="store_true")
    args=parser.parse_args()
    raw,source,output=args.raw_root.resolve(),args.source_root.resolve(),args.output.resolve()
    require(not output.exists(), "output must be a new directory")
    transferred,transfer_proof=verify_transfer(raw)
    manifest_file=source/"freeze_manifest.json"
    require(sha(manifest_file)==MANIFEST_SHA,"unexpected frozen source manifest")
    manifest=read_json(manifest_file)
    require(manifest["baseline_commit"]==COMMIT,"baseline commit mismatch")
    source_paths=[manifest_file]
    for name,expected in manifest["sources_sha256"].items():
        path=(source/name).resolve()
        require(path.is_relative_to(source) and sha(path)==expected,"frozen source file hash mismatch")
        source_paths.append(path)
        require((raw/"sources"/name).read_bytes()==path.read_bytes(),"downloaded source differs from local frozen bytes")
    require((raw/"sources/freeze_manifest.json").read_bytes()==manifest_file.read_bytes(),"downloaded source manifest differs")
    screen,selected_s,srun,sproof=load_stage(raw,"screen",manifest)
    holdout,selected_h,hrun,hproof=load_stage(raw,"holdout",manifest)
    require(srun["compile_command"][:-1]==hrun["compile_command"][:-1],"stage compile conditions changed")
    qualifying=[c["case_id"] for c in screen if all(r["reduction_vs_old"]>=.05 for r in c["rounds"])]
    require(qualifying,"holdout should not have run: no screen configuration passed the 5% gate")
    gate=hrun.get("screen_gate",{})
    require(gate.get("screen_summary_sha256")==sha(raw/"screen/run_summary.json") and
            gate.get("qualifying_configurations")==qualifying and gate.get("all_24_numerical_configurations_passed") is True and
            gate.get("fixed_holdout_configurations")==96,"recorded screen gate differs from independent recomputation")
    require(gate.get("threshold")=="at least one configuration with >=5% reduction in all three rounds","screen gate threshold changed")
    combined=screen+holdout
    require(len(combined)==120 and {c["case_id"] for c in combined}=={c["case_id"] for c in case_list("all")},"full matrix is incomplete/duplicated")
    hardware_keys=("device","compute_capability","sm_count","warp_size","total_global_memory","runtime_version","driver_version")
    hardware={k:combined[0]["validation"][k] for k in hardware_keys}
    require(all({k:c["validation"][k] for k in hardware_keys}==hardware for c in combined),"hardware/runtime metadata changed")
    comparison={}
    for name,subset in (("screen",screen),("holdout",holdout),("all",combined)):
        comparison[name]={"old_wmma":scope_summary(subset,"old_over_new"),"warp128":scope_summary(subset,"warp128_over_new")}
    by_n={str(n):{key:scope_summary([c for c in combined if c["n"]==n],field) for key,field in (("old_wmma","old_over_new"),("warp128","warp128_over_new"))} for n in (16,32,64,128,256)}
    by_rows={str(rows):{key:scope_summary([c for c in combined if c["rows"]==rows],field) for key,field in (("old_wmma","old_over_new"),("warp128","warp128_over_new"))} for rows in (1,17,64,257,4096,16384)}
    correctness=dict(unique_shape_dtype_scale_configurations=120,input_groups_per_configuration=2,
                     dyadic_unique_elements=sum(c["rows"]*c["n"] for c in combined),
                     general_old_new_elements=sum(c["validation"]["general_input_group"]["old_new_element_comparisons"] for c in combined),
                     general_dense_rows=sum(c["validation"]["general_input_group"]["dense_rows"] for c in combined),
                     repeated_checks_not_counted_as_new_configurations=True,dyadic_exact=True,general_old_new_all_bits_equal=True,
                     input_and_H_unchanged=True,output_guards_intact=True,dyadic_guard_offsets_bytes=[32,34],general_guard_offset_bytes=32)
    correctness["general_by_dtype"]={dt:{"strict_rounded_tolerance":.01 if dt=="fp16" else .05,
        "rounded_max_abs_by_method":{m:max(c["validation"]["general_input_group"]["rounded_max_abs_error"][i] for c in combined if c["dtype"]==dt) for i,m in enumerate(METHODS)},
        "unrounded_max_abs_by_method":{m:max(c["validation"]["general_input_group"]["unrounded_max_abs_error"][i] for c in combined if c["dtype"]==dt) for i,m in enumerate(METHODS)},
        "rounded_bit_mismatches_by_method":{m:sum(c["validation"]["general_input_group"]["rounded_bit_mismatches"][i] for c in combined if c["dtype"]==dt) for i,m in enumerate(METHODS)}} for dt in ("fp16","bf16")}
    narrow_ranges=dict(
        all_three_faster_than_both=[c["case_id"] for c in combined if all(r["old_over_new"]>1 and r["warp128_over_new"]>1 for r in c["rounds"])],
        all_three_at_least_5_percent_faster_than_both=[c["case_id"] for c in combined if all(r["reduction_vs_old"]>=.05 and r["reduction_vs_warp128"]>=.05 for r in c["rounds"])],
        n64_and_larger_all_three_slower_than_warp128=sum(c["n"]>=64 and all(r["warp128_over_new"]<1 for r in c["rounds"]) for c in combined))
    summary=dict(status="PASS",analysis_only_no_gpu_execution=True,baseline_commit=COMMIT,source_manifest_sha256=MANIFEST_SHA,
                 hardware=hardware,unique_configurations=120,event_rows=21600,configuration_processes=120,rounds_per_configuration_process=3,
                 three_independent_process_repetitions=False,screen_gate_recomputed_qualifying=qualifying,
                 correctness=correctness,comparisons=comparison,by_n=by_n,by_rows=by_rows,
                 compilation=[sproof,hproof],sources_sha256=manifest["sources_sha256"],transfer_verification=transfer_proof,narrow_ranges=narrow_ranges,
                 stream_evidence="frozen C++ uses explicit non-blocking stream for init/copy/launch/event/readback; no separate cross-stream concurrency test",
                 exclusions="allocation, H construction, copies, validation, warmup; non-Graph batched CUDA events",
                 no_production_integration_recommended_by_this_audit=True)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"summary.json",summary)
    fieldnames=["partition","case_id","rows","n","dtype","scale","round","old_wmma_ms","four_warp_wmma_ms","warp128_ms","old_over_new","warp128_over_new","reduction_vs_old","reduction_vs_warp128"]
    table=[];negative=[]
    for case in sorted(combined,key=lambda c:(c["n"],c["rows"],c["dtype"],c["scale"])):
        is_negative=any(r["old_over_new"]<1 or r["warp128_over_new"]<1 for r in case["rounds"])
        for row in case["rounds"]:
            record={k:case[k] for k in ("partition","case_id","rows","n","dtype","scale")};record.update(row);table.append(record)
            if is_negative:negative.append(record)
    for name,rows in (("case_rounds.csv",table),("negative_cases.csv",negative)):
        with (output/name).open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=fieldnames);writer.writeheader();writer.writerows(rows)
    (output/"RESULTS.md").write_text(results_text(summary),encoding="utf-8")
    require(set(selected_s+selected_h).issubset(set(transferred)),"stage evidence was not in original transfer")
    selected=transferred
    original_entries=[dict(relative_path=p.relative_to(raw).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in selected]
    write_json(output/"original_raw_manifest.json",dict(files=original_entries,file_count=len(original_entries),original_bytes_preserved=True))
    if args.archive:
        archive=output/"public_archive";archive.mkdir()
        write_archive_packaging(archive)
        for original in selected:
            dest=archive/"raw"/original.relative_to(raw);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,dest)
            require(sha(dest)==sha(original),"raw copy altered bytes")
        for original in source_paths:
            dest=archive/"sources"/original.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,dest)
            require(sha(dest)==sha(original),"source copy altered bytes")
        for name in ("summary.json","case_rounds.csv","negative_cases.csv","RESULTS.md","original_raw_manifest.json"):
            shutil.copyfile(output/name,archive/name)
        shutil.copyfile(Path(__file__).resolve(),archive/"analyze.py")
        entries=[]
        for path in sorted(archive.rglob("*")):
            if path.is_file():
                require(path.read_bytes()[:4]!=b"\x7fELF","binary unexpectedly entered public archive")
                entries.append(dict(path=path.relative_to(archive).as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
        write_json(archive/"archive_manifest.json",dict(files=entries,manifest_itself_excluded=True,
                   binary_files_excluded=True,source_and_raw_bytes_unchanged=True,analyzer_sha256=sha(Path(__file__))))
    print(json.dumps(dict(status="PASS",unique_configurations=120,event_rows=21600,
                          qualifying_screen_configurations=qualifying,comparisons=comparison,output=str(output)),ensure_ascii=False,indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
