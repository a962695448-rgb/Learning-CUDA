#!/usr/bin/env python3
"""固定的 128/256 线程晋级复核；独立扩展、64 独立输出的 CUDA Graph。"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources"
sys.path.insert(0, str(SOURCES))
import compare_reference as original_reference_tools


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    fields = "name,driver_version,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw"
    device = subprocess.run(["nvidia-smi", "--query-gpu=" + fields, "--format=csv"],
                            capture_output=True, text=True, timeout=15)
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
                               capture_output=True, text=True, timeout=15)
    return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device_exit": device.returncode, "device": device.stdout, "device_stderr": device.stderr,
            "process_exit": processes.returncode, "processes": processes.stdout}


def load_extension():
    from torch.utils.cpp_extension import load
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    return load(name="hadamard_thread_promotion_20260905", sources=[str(SOURCES / "torch_binding_experiment.cu")],
                extra_include_paths=[str(SOURCES)], extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo",
                    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr"], build_directory=str(build), verbose=True)


def tensors(value):
    return value if isinstance(value, tuple) else (value,)


def exact(torch, actual, expected, label):
    a, b = tensors(actual), tensors(expected)
    if len(a) != len(b) or any(x.dtype != y.dtype or x.shape != y.shape or
        not torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(a, b)):
        raise RuntimeError("bitwise mismatch: " + label)


def cpu_checks(torch, np, values, transformed, packed, scales, scale):
    n = values.shape[-1]
    rows = values.numel() // n
    indices = sorted({0, rows // 2, rows - 1})
    x = values.reshape(rows, n)[indices].float().cpu().numpy().astype(np.float64)
    h = np.array([[(-1.0 if (i & j).bit_count() % 2 else 1.0) for j in range(n)] for i in range(n)])
    dense = torch.from_numpy((x @ h * scale).astype(np.float32)).to(dtype=values.dtype)
    selected = transformed.reshape(rows, n)[indices].cpu()
    error = float((selected.float() - dense.float()).abs().max())
    limit = 1e-2 if values.dtype == torch.float16 else 5e-2
    if not math.isfinite(error) or error >= limit:
        raise RuntimeError("independent CPU dense check failed")
    y = transformed.float().cpu().numpy().reshape(rows, n)
    cpu_scales = np.max(np.abs(y), axis=1).astype(np.float32) / np.float32(7)
    cpu_scales[cpu_scales == 0] = np.float32(1)
    q = np.clip(np.rint(y / cpu_scales[:, None]), -7, 7).astype(np.int8)
    expected = ((q[:, 0::2].astype(np.uint8) & 15) |
                ((q[:, 1::2].astype(np.uint8) & 15) << 4))
    if not np.array_equal(packed.cpu().numpy().reshape(rows, n // 2), expected):
        raise RuntimeError("independent CPU packed INT4 mismatch")
    if scales.cpu().numpy().reshape(rows).tobytes() != cpu_scales.tobytes():
        raise RuntimeError("independent CPU quantization scales mismatch")
    return {"dense_rows": indices, "dense_max_abs_error": error, "strict_limit": limit,
            "quantization_rows_checked": rows, "cpu_quantization_exact": True}


def shape_for(rows, n):
    return (rows // 1024, 128, 8, n) if rows in (4096, 16384) else (rows, n)


def configurations():
    for dtype in ("fp16", "bf16"):
        for n in (16, 64):
            for rows in (4095, 4096, 4097, 16383, 16384, 16385):
                for normalized in (False, True):
                    yield {"dtype": dtype, "dim": n, "rows": rows, "shape": list(shape_for(rows, n)),
                           "normalized": normalized, "scale": 1 / math.sqrt(n) if normalized else 1.0}


def check_configuration(torch, np, op, dao, case):
    dtype = getattr(torch, "float16" if case["dtype"] == "fp16" else "bfloat16")
    checks = []
    for pattern in ("normal", "uniform", "outlier", "zeros"):
        for seed in ((2026,) if pattern == "zeros" else (2026, 95811)):
            x = original_reference_tools.make_input(torch, case["shape"], dtype, pattern, seed, "cuda")
            baseline = op.hadamard(x, case["scale"], 128)
            candidate = op.hadamard(x, case["scale"], 256)
            exact(torch, candidate, baseline, "candidate transform vs baseline")
            expected = dao(x, case["scale"])
            error = float((candidate.float() - expected.float()).abs().max())
            limit = 1e-2 if dtype == torch.float16 else 5e-2
            if not math.isfinite(error) or error >= limit:
                raise RuntimeError("Dao comparison failed")
            fused128 = op.hadamard_int4(x, case["scale"], 128)
            fused256 = op.hadamard_int4(x, case["scale"], 256)
            exact(torch, fused256, fused128, "candidate fused INT4 vs baseline")
            exact(torch, fused128, op.quantize_int4(baseline, 128), "128 fused vs split")
            exact(torch, fused256, op.quantize_int4(candidate, 256), "256 fused vs split")
            check = cpu_checks(torch, np, x, candidate, *fused256, case["scale"])
            checks.append({"pattern": pattern, "seed": seed, "dao_max_abs_error": error,
                           "elements": x.numel(), "all_elements_baseline_candidate_bitwise_exact": True, **check})
    return checks


def measure_graph(torch, functions, run_index):
    graphs, outputs, expected = {}, {}, {}
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for name, function in functions.items():
            for _ in range(25):
                function()
            expected[name] = function()
    capture_stream.synchronize()
    for name, function in functions.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            outputs[name] = [function() for _ in range(64)]
        graphs[name] = graph
    torch.cuda.synchronize()
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    for name, values in outputs.items():
        for component in range(len(tensors(values[0]))):
            if len({tensors(value)[component].data_ptr() for value in values}) != 64:
                raise RuntimeError("captured outputs do not have 64 independent buffers")
        for value in values:
            exact(torch, value, expected[name], "captured " + name)
    for graph in graphs.values():
        for _ in range(5):
            graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in functions}
    intervals = {name: [] for name in functions}
    orders, names = [], list(functions)
    for group in range(5):
        offset = (group + run_index - 1) % len(names)
        order = names[offset:] + names[:offset]
        orders.append(order)
        for name in order:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(20):
                graphs[name].replay()
            end.record()
            end.synchronize()
            ms = begin.elapsed_time(end)
            if not math.isfinite(ms) or ms <= 0:
                raise RuntimeError("invalid CUDA event timing")
            intervals[name].append(ms)
            samples[name].append(ms * 1000 / (20 * 64))
    for name, values in outputs.items():
        for value in values:
            exact(torch, value, expected[name], "after timing " + name)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    result = {"samples_us": samples, "raw_event_intervals_ms": intervals, "median_us": medians,
              "baseline_over_candidate": medians["baseline128"] / medians["candidate256"],
              "candidate_time_reduction_percent": 100 * (1 - medians["candidate256"] / medians["baseline128"]),
              "group_order": orders, "captured_outputs_per_graph": 64, "independent_output_buffers": True,
              "groups": 5, "replays_per_group": 20, "api_warmup_calls": 25,
              "graph_warmup_replays": 5, "captured_outputs_bitwise_equal_eager_before_and_after": True}
    if "dao" in medians:
        result["dao_over_candidate"] = medians["dao"] / medians["candidate256"]
        result["dao_over_baseline"] = medians["dao"] / medians["baseline128"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-repo", required=True, type=Path)
    parser.add_argument("--run-index", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve prior evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "run_index": args.run_index, "build_only": args.build_only, "before": snapshot(),
              "correctness": [], "benchmarks": [], "process_isolation_note":
              "Other project reported idle and agreed not to launch during this experiment. Preflight NVML lists a resident 448MiB context whose host PID cannot be fully mapped from the container; this is not proof of exclusive GPU tenancy."}
    status = 1
    try:
        manifest = json.loads((ROOT / "source_manifest.json").read_text())
        for name, data in manifest["files"].items():
            if sha(ROOT / name) != data["sha256"]:
                raise RuntimeError("source hash mismatch: " + name)
        report["source_manifest"] = manifest
        report["experiment_script_sha256"] = sha(__file__)
        import numpy as np
        import torch
        import fast_hadamard_transform as ref_package
        import fast_hadamard_transform_cuda as ref_backend
        report["reference"] = original_reference_tools.provenance(ref_package, ref_backend, args.reference_repo)
        op = load_extension()
        report["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "numpy": np.__version__, "gpu": torch.cuda.get_device_name(),
            "sm": list(torch.cuda.get_device_capability()), "cpp11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
            "extension_file": str(Path(op.__file__).resolve()), "extension_sha256": sha(op.__file__),
            "nvcc": subprocess.check_output([str(Path(os.environ['CUDA_HOME']) / 'bin/nvcc'), '--version'], text=True),
            "compile_concurrency": os.environ.get("MAX_JOBS"), "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST")}
        with torch.inference_mode():
            for case in ([] if args.build_only else list(configurations())):
                checks = check_configuration(torch, np, op, ref_package.hadamard_transform, case)
                report["correctness"].append({**case, "checks": checks})
                dtype = getattr(torch, "float16" if case["dtype"] == "fp16" else "bfloat16")
                x = original_reference_tools.make_input(torch, case["shape"], dtype, "normal", 2026, "cuda")
                scale = case["scale"]
                transform = {"baseline128": lambda: op.hadamard(x, scale, 128),
                             "candidate256": lambda: op.hadamard(x, scale, 256),
                             "dao": lambda: ref_package.hadamard_transform(x, scale)}
                result = measure_graph(torch, transform, args.run_index)
                report["benchmarks"].append({**case, "mode": "transform", "original_target": case["rows"] in (4096, 16384), **result})
                gc.collect()
                if case["rows"] in (4095, 4096, 4097):
                    fused = {"baseline128": lambda: op.hadamard_int4(x, scale, 128),
                             "candidate256": lambda: op.hadamard_int4(x, scale, 256)}
                    result = measure_graph(torch, fused, args.run_index)
                    report["benchmarks"].append({**case, "mode": "fused_int4", "original_target": case["rows"] == 4096, **result})
                    gc.collect()
                print("PROGRESS", json.dumps({"dtype": case["dtype"], "N": case["dim"], "M": case["rows"],
                      "normalized": case["normalized"], "benchmarks": len(report["benchmarks"])}), flush=True)
        report["status"] = "BUILD_PASS" if args.build_only else "PASS"
        report["scope"] = "64 independent retained outputs per private CUDA Graph;25 API warmups;5 graph warmups;20 replays x5 groups; fixed input/read-only addresses, all functions same condition. Times are amortized captured GPU work plus replay scheduling, not isolated kernel latency or host end-to-end. No old fixed-buffer pilot values are divided into these results."
        report["summary"] = {"unique_shape_dtype_scale_cases": len(report["correctness"]),
                             "correctness_input_cases": sum(len(c["checks"]) for c in report["correctness"]),
                             "benchmark_configurations": len(report["benchmarks"])}
        status = 0
    except Exception as error:
        import traceback
        report["status"] = "FAIL"
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr, flush=True)
    finally:
        report["after"] = snapshot()
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": report.get("summary"), "output": str(args.output)}), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
