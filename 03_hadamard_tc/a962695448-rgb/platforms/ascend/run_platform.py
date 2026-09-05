#!/usr/bin/env python3
"""原生 Ascend C/CANN 构建与 NPU 验证；基准须显式开启，不安装或替换 SDK。"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def capture(command, environment):
    result = subprocess.run(list(map(str, command)), cwd=ROOT, env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return {"command": list(map(str, command)), "returncode": result.returncode, "output": result.stdout}


def save(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(command, log, report, manifest, environment, expected=0):
    record = {"command": list(map(str, command)), "log": log.name, "returncode": None,
              "expected_returncode": expected}
    report["stages"].append(record)
    save(manifest, report)
    print("RUN", " ".join(record["command"]), flush=True)
    start = time.monotonic()
    try:
        with log.open("x", encoding="utf-8") as output:
            result = subprocess.run(record["command"], cwd=ROOT, env=environment,
                                    stdout=output, stderr=subprocess.STDOUT, check=False)
        record["returncode"] = result.returncode
    finally:
        record["wall_seconds"] = time.monotonic() - start
        if log.is_file():
            record["sha256"] = digest(log)
        save(manifest, report)
    if record["returncode"] != expected:
        raise RuntimeError("stage failed: " + log.name)


def environment_for(cann):
    environment = dict(os.environ)
    script = cann / "set_env.sh"
    setup = {"script": str(script), "used": script.is_file()}
    if script.is_file():
        # SDK 自带脚本只在子进程中改变环境。env 的完整结果留在内存，不写入日志。
        result = subprocess.run(["bash", "-c", 'source "$1" >/dev/null && env -0', "ascend-env", str(script)],
                                env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode:
            raise RuntimeError("CANN set_env.sh failed: " + result.stderr.decode(errors="replace")[-3000:])
        environment = {key.decode(): value.decode() for entry in result.stdout.split(b"\0") if b"=" in entry
                       for key, value in [entry.split(b"=", 1)]}
        setup["sha256"] = digest(script)
    environment.update(ASCEND_HOME_PATH=str(cann), ASCEND_TOOLKIT_HOME=str(cann), ASCEND_CANN_PACKAGE_PATH=str(cann))
    extra_bins = [str(p) for p in (cann / "bin", cann / "compiler/ccec_compiler/bin", cann / "tools/ccec_compiler/bin") if p.is_dir()]
    environment["PATH"] = os.pathsep.join(extra_bins + [environment.get("PATH", "")])
    libraries = [str(p) for p in (cann / "lib64", cann / "aarch64-linux/lib64", Path("/usr/local/Ascend/driver/lib64")) if p.is_dir()]
    inherited = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(libraries + ([inherited] if inherited else []))
    return environment, setup


def summarize(path):
    groups = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["timer"] != "acl_timeline_event_ms":
                raise RuntimeError("unexpected benchmark timing source")
            key = tuple(row[field] for field in ("dtype", "batch", "seq", "heads", "dim", "block_dim", "method"))
            groups.setdefault(key, []).append(float(row["kernel_us"]))
    if not groups:
        raise RuntimeError("empty benchmark CSV")
    series = []
    medians = {key: statistics.median(values) for key, values in groups.items()}
    for key, values in groups.items():
        series.append(dict(zip(("dtype", "batch", "seq", "heads", "dim", "block_dim", "method"), key),
                           raw_samples_us=values, median_us=statistics.median(values),
                           minimum_us=min(values), maximum_us=max(values)))
    comparisons = []
    for key, value in medians.items():
        if key[-1].startswith("scalar_"):
            peer = key[:-1] + (key[-1].replace("scalar_", "vector_"),)
            comparisons.append({"shape_dtype_blocks": key[:-1], "operation": key[-1][7:],
                                "scalar_median_us": value, "vector_median_us": medians[peer],
                                "scalar_over_vector_same_blocks": value / medians[peer]})
    return {"timer": "ACL_EVENT_TIME_LINE events; aclrtEventElapsedTime milliseconds converted to microseconds",
            "not_timed": ["allocation", "copies", "validation", "event creation/destruction", "warmup"],
            "cache_condition": "same seeded read-only inputs reused; warm-cache samples",
            "quant_only": "one shared NPU scalar-division quantization implementation; not two different algorithms",
            "logical_GBs": "logical tensor I/O estimate, not measured physical memory bandwidth",
            "series": series, "comparisons_including_slowdowns": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cann-root", type=Path, default=Path("/usr/local/Ascend/cann-9.0.0"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--build-jobs", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-stress", action="store_true", help="明确跳过大M专项，报告不宣称完整验收")
    parser.add_argument("--dtype", choices=("both", "fp16", "bf16"), default="both")
    parser.add_argument("--block-dim", type=int, default=1)
    parser.add_argument("--benchmark", action="store_true", help="验证通过后运行小规模基准矩阵")
    parser.add_argument("--pilot-benchmark", action="store_true", help="只用M17/N128先核对真实ACL event时间和耗时")
    parser.add_argument("--repeats", "--iterations", type=int, default=5)
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.block_dim <= 32 or not 1 <= args.build_jobs <= 8:
        parser.error("block-dim must be 1..32; build-jobs must be 1..8")
    if not 1 <= args.repeats <= 10000 or not 1 <= args.groups <= 10000 or not 0 <= args.warmup <= 10000:
        parser.error("repeats/groups must be 1..10000; warmup must be 0..10000")
    cann = args.cann_root.resolve()
    cmake_entry = cann / "aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake"
    if not cmake_entry.is_file():
        parser.error("missing actual Ascend C CMake entry: " + str(cmake_entry))
    destination = (args.output or ROOT / "results/ascend" / time.strftime("%Y%m%d-%H%M%S", time.gmtime())).resolve()
    build = ROOT / "build/ascend" / (destination.name + "-" + hashlib.sha256(str(destination).encode()).hexdigest()[:10])
    if destination.exists() or build.exists():
        parser.error("output/build directory already exists; use a fresh output path")
    try:
        environment, setup = environment_for(cann)
    except (OSError, RuntimeError, UnicodeError) as error:
        parser.error(str(error))
    destination.mkdir(parents=True)
    files = [SOURCE / name for name in ("CMakeLists.txt", "hadamard_api.h", "hadamard_api.cpp", "hadamard_kernel.cpp",
                                      "validate_and_benchmark.cpp", "run_platform.py")] + [ROOT / "include/reference.hpp"]
    manifest = destination / "run_summary.json"
    report = {"status": "RUNNING", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "platform": "ascend", "run_mode": "npu", "soc_target": "Ascend910B1", "host_architecture": platform.machine(),
              "cann_root": str(cann), "sdk_environment_setup": setup, "build_directory": str(build),
              "quick": args.quick, "skip_stress": args.skip_stress, "requested_dtype": args.dtype, "block_dim": args.block_dim,
              "benchmark_requested": args.benchmark or args.pilot_benchmark, "pilot_benchmark": args.pilot_benchmark,
              "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in files},
              "python_version": sys.version, "cmake_version": capture([args.cmake, "--version"], environment),
              "git_head": capture(["git", "rev-parse", "HEAD"], environment),
              "git_status": capture(["git", "status", "--short"], environment), "stages": []}
    npu_smi = shutil.which("npu-smi", path=environment.get("PATH"))
    if npu_smi:
        report["device_inventory"] = capture([npu_smi, "info"], environment)
    save(manifest, report)
    code = 1
    try:
        run([args.cmake, "-S", SOURCE, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DRUN_MODE=npu", "-DBUILD_VALIDATION=ON",
             "-DSOC_VERSION=Ascend910B1", "-DASCEND_CANN_PACKAGE_PATH=" + str(cann)],
            destination / "configure.log", report, manifest, environment)
        run([args.cmake, "--build", build, "--target", "validate_and_benchmark", "-j", str(args.build_jobs)],
            destination / "build.log", report, manifest, environment)
        binary = build / "validate_and_benchmark"
        report["binary"] = {"path": str(binary), "sha256": digest(binary)}
        invalid = [[], ["--validate", "--dim", "0"], ["--validate", "--dim", "3"], ["--validate", "--dim", "512"],
                   ["--validate", "--dtype", "fp32"], ["--validate", "--batch", "-1"], ["--validate", "--seq", "0"],
                   ["--validate", "--heads", "x"], ["--validate", "--batch", "18446744073709551615", "--seq", "2"],
                   ["--validate", "--heads", "999999999999999999999999999999"], ["--validate", "--dim"],
                   ["--validate", "--unknown", "1"], ["--benchmark", "--repeats", "0"], ["--benchmark", "--groups", "10001"],
                   ["--validate", "--block-dim", "0"], ["--validate", "--block-dim", "33"], ["--benchmark", "--warmup", "-1"]]
        for index, case in enumerate(invalid):
            run([binary] + case, destination / ("invalid_%02d.log" % index), report, manifest, environment, expected=2)
        report["cli_rejection_cases"] = len(invalid)
        command = [binary, "--validate", "--block-dim", str(args.block_dim), "--dtype", args.dtype,
                   "--json", destination / "validation.json"]
        if args.quick:
            command.append("--quick")
        if args.skip_stress:
            command.append("--skip-stress")
        run(command, destination / "validation.log", report, manifest, environment)
        validation = json.loads((destination / "validation.json").read_text(encoding="utf-8"))
        report["validation"] = validation
        if validation.get("status") != "PASS" or validation.get("execution") != "npu" or validation.get("main_block_dim") != args.block_dim:
            raise RuntimeError("validation JSON does not confirm requested NPU execution")
        if not args.quick and args.dtype == "both":
            if validation.get("full_matrix") is not True or (not args.skip_stress and validation.get("full_suite_complete") is not True):
                raise RuntimeError("full validation JSON is incomplete")
        if args.benchmark or args.pilot_benchmark:
            command = [binary, "--benchmark", "--block-dim", str(args.block_dim), "--dtype", args.dtype,
                       "--repeats", str(args.repeats), "--groups", str(args.groups), "--warmup", str(args.warmup),
                       "--csv", destination / "benchmark.csv"]
            if args.pilot_benchmark:
                command += ["--batch", "1", "--seq", "17", "--heads", "1", "--dim", "128"]
            run(command, destination / "benchmark.log", report, manifest, environment)
            report["benchmark"] = summarize(destination / "benchmark.csv")
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
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report["artifacts"] = {path.name: digest(path) for path in destination.iterdir() if path.is_file() and path != manifest}
        save(manifest, report)
    print(json.dumps({"status": report["status"], "summary": str(manifest)}, ensure_ascii=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
