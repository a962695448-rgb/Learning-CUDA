#!/usr/bin/env python3
"""Isolated whole-warp thread-config study; numerical checks precede timing."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import struct
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sources"))
import compare_reference as reference_tools


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def snapshot():
    result = {"utc": utc()}
    for key, option in (("gpu", "--query-gpu=name,driver_version,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw"),
                        ("processes", "--query-compute-apps=pid,process_name,used_memory")):
        run = subprocess.run(["nvidia-smi", option, "--format=csv"], capture_output=True, text=True, timeout=15)
        result[key] = {"exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
    return result


def verify_files():
    manifest = json.loads((ROOT / "run_manifest.json").read_text(encoding="utf-8"))
    for name, metadata in manifest["files"].items():
        if sha(ROOT / name) != metadata["sha256"]:
            raise RuntimeError("frozen file mismatch: " + name)
    return manifest


def configurations(protocol, phase):
    domain, screen = protocol["domain"], protocol["screen"]
    cases = []
    for dtype in domain["dtypes"]:
        for dim in domain["dims"]:
            for rows in domain["rows"]:
                is_screen = rows in screen["rows"] and dim in screen["dims"]
                if is_screen != (phase == "screen"):
                    continue
                for normalized in domain["normalized"]:
                    scale = struct.unpack("f", struct.pack("f", 1 / math.sqrt(dim) if normalized else 1.0))[0]
                    cases.append({"dtype": dtype, "dim": dim, "rows": rows, "shape": [rows, dim],
                                  "normalized": normalized, "scale": scale})
    expected = screen["expected_configurations"] if phase == "screen" else protocol["validation"]["expected_holdout_configurations"]
    if len(cases) != expected:
        raise RuntimeError("protocol configuration count mismatch")
    return cases


def load_extension():
    from torch.utils.cpp_extension import load
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    return load(name="hadamard_small_batch_thread_config_20260906", 
        sources=[str(ROOT / "sources/torch_binding_thread_config.cu")],
        extra_include_paths=[str(ROOT / "sources")], extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo", "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "--expt-relaxed-constexpr"],
        build_directory=str(build), verbose=True)


def tensors(value):
    return value if isinstance(value, tuple) else (value,)


def exact(torch, actual, expected, label):
    actual, expected = tensors(actual), tensors(expected)
    if len(actual) != len(expected):
        raise RuntimeError("component mismatch: " + label)
    for component, (a, b) in enumerate(zip(actual, expected)):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError("shape/dtype mismatch: " + label)
        ab, bb = a.contiguous().view(torch.uint8).reshape(-1), b.contiguous().view(torch.uint8).reshape(-1)
        if not torch.equal(ab, bb):
            byte = int((ab != bb).nonzero()[0, 0])
            element = byte // a.element_size()
            raise RuntimeError(f"bitwise mismatch {label}, component={component}, element={element}, "
                f"actual={a.reshape(-1)[element].item()}, expected={b.reshape(-1)[element].item()}, "
                f"actual_bytes={ab[element*a.element_size():(element+1)*a.element_size()].cpu().tolist()}, "
                f"expected_bytes={bb[element*b.element_size():(element+1)*b.element_size()].cpu().tolist()}")


def cpu_checks(torch, np, values, transformed, packed, scales, scale):
    rows, dim = values.shape
    indices = sorted({0, rows // 2, rows - 1})
    x = values[indices].float().cpu().numpy().astype(np.float64)
    signs = np.array([[(-1.0 if (i & j).bit_count() % 2 else 1.0) for j in range(dim)] for i in range(dim)])
    dense = torch.from_numpy((x @ signs * scale).astype(np.float32)).to(dtype=values.dtype)
    error = float((transformed[indices].cpu().float() - dense.float()).abs().max())
    limit = 0.01 if values.dtype == torch.float16 else 0.05
    if not math.isfinite(error) or error >= limit:
        raise RuntimeError(f"independent FP64 dense failure: error={error}, strict_limit={limit}, rows={indices}")
    y = transformed.float().cpu().numpy()
    cpu_scales = np.max(np.abs(y), axis=1).astype(np.float32) / np.float32(7)
    cpu_scales[cpu_scales == 0] = np.float32(1)
    q = np.clip(np.rint(y / cpu_scales[:, None]), -7, 7).astype(np.int8)
    expected = (q[:, 0::2].astype(np.uint8) & 15) | ((q[:, 1::2].astype(np.uint8) & 15) << 4)
    actual = packed.cpu().numpy()
    if not np.array_equal(actual, expected):
        index = tuple(int(v) for v in np.argwhere(actual != expected)[0])
        raise RuntimeError(f"CPU INT4 mismatch: row/byte={index}, actual={actual[index]}, expected={expected[index]}")
    actual_scales = scales.cpu().numpy()
    if actual_scales.tobytes() != cpu_scales.tobytes():
        index = int(np.flatnonzero(actual_scales.view(np.uint32) != cpu_scales.view(np.uint32))[0])
        raise RuntimeError(f"CPU scale bits mismatch at row={index}, actual={actual_scales[index]}, expected={cpu_scales[index]}")
    return {"dense_rows": indices, "dense_max_abs_error": error, "strict_limit": limit,
            "quantization_rows_checked": rows, "cpu_quantization_exact": True}


def check_case(torch, np, op, dao, case, threads, protocol, report, witness_dir):
    dtype = torch.float16 if case["dtype"] == "fp16" else torch.bfloat16
    entry = {**case, "checks": []}
    report["correctness"].append(entry)
    for pattern, seeds in protocol["correctness"]["patterns_and_seeds"].items():
        for seed in seeds:
            report["active_context"] = {**case, "phase": "correctness", "pattern": pattern, "seed": seed}
            x = reference_tools.make_input(torch, case["shape"], dtype, pattern, seed, "cuda")
            baseline = op.hadamard(x, case["scale"], 128)
            expected = dao(x, case["scale"])
            fused128 = op.hadamard_int4(x, case["scale"], 128)
            try:
                metric = reference_tools.metrics(torch, baseline, expected, 0.01 if dtype == torch.float16 else 0.05)
                if not metric["pass"]:
                    raise RuntimeError("Dao transform comparison failed: " + json.dumps(metric))
                for block_threads in threads:
                    report["active_context"]["block_threads"] = block_threads
                    output = op.hadamard(x, case["scale"], block_threads)
                    fused = op.hadamard_int4(x, case["scale"], block_threads)
                    split = op.quantize_int4(output, block_threads)
                    exact(torch, output, baseline, "transform vs128")
                    exact(torch, fused, fused128, "fused vs128")
                    exact(torch, fused, split, "fused vs same-thread split")
                    # Every candidate passes the same independent, all-row oracle.
                    cpu = cpu_checks(torch, np, x, output, *fused, case["scale"])
                entry["checks"].append({"pattern": pattern, "seed": seed, "pass": True,
                    "elements": x.numel(), "dao_max_abs_error": metric["max_abs_error"],
                    "all_threads_transform_bitwise_exact": True, "all_threads_fused_split_bitwise_exact": True,
                    "threads_checked": threads, **cpu})
            except Exception:
                witness_dir.mkdir(parents=True, exist_ok=True)
                path = witness_dir / f"case_{len(report['correctness']):03d}_{pattern}_{seed}.pt"
                torch.save({"input": x.cpu(), "baseline128": baseline.cpu(), "dao": expected.cpu(),
                    "context": report["active_context"]}, path)
                report["failure_witness"] = {"path": str(path), "sha256": sha(path)}
                raise


def orders_for(names, run_index, case_index, groups):
    names = list(names)
    random.Random(91000 + run_index).shuffle(names)
    return [names[(group + case_index) % len(names):] + names[:(group + case_index) % len(names)]
            for group in range(groups)]


def timed_result(torch, functions, orders, repeats, calls, invoke):
    samples, intervals = {name: [] for name in functions}, {name: [] for name in functions}
    for order in orders:
        for name in order:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(repeats):
                invoke(name)
            end.record()
            end.synchronize()
            ms = begin.elapsed_time(end)
            if not math.isfinite(ms) or ms <= 0:
                raise RuntimeError("invalid CUDA event interval")
            intervals[name].append(ms)
            samples[name].append(ms * 1000 / (repeats * calls))
    medians = {name: statistics.median(sample) for name, sample in samples.items()}
    return {"samples_us": samples, "raw_event_intervals_ms": intervals,
            "median_us": medians, "median_ms": {name: value / 1000 for name, value in medians.items()},
            "group_order": orders}


def measure_graph(torch, functions, run_index, case_index, timing):
    graph_settings = timing["graph"]
    graphs, outputs, expected = {}, {}, {}
    orders = orders_for(functions, run_index, case_index, timing["groups"])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for name in orders[0]:
            for _ in range(timing["api_warmup_calls"]):
                functions[name]()
            expected[name] = functions[name]()
    stream.synchronize()
    for name in orders[0]:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs[name] = [functions[name]() for _ in range(graph_settings["captured_calls"])]
        graphs[name] = graph
    torch.cuda.synchronize()
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    pointers = []
    for name, values in outputs.items():
        addresses = [value.data_ptr() for value in values]
        if len(set(addresses)) != graph_settings["captured_calls"]:
            raise RuntimeError("graph outputs alias: " + name)
        pointers.extend(addresses)
        for value in values:
            exact(torch, value, expected[name], "captured " + name)
    if len(set(pointers)) != len(pointers):
        raise RuntimeError("private graphs share retained output pointers")
    for graph in graphs.values():
        for _ in range(graph_settings["warmup_replays"]):
            graph.replay()
    torch.cuda.synchronize()
    result = timed_result(torch, functions, orders, graph_settings["replays_per_group"],
                          graph_settings["captured_calls"], lambda name: graphs[name].replay())
    for name, values in outputs.items():
        for value in values:
            exact(torch, value, expected[name], "after timing " + name)
    result.update({"independent_output_buffers": True, "cross_method_output_pointers_disjoint": True,
        "outputs_bitwise_equal_eager_before_and_after": True, "captured_calls_per_graph": graph_settings["captured_calls"],
        "replays_per_group": graph_settings["replays_per_group"], "graph_warmup_replays": graph_settings["warmup_replays"],
        "api_warmup_calls": timing["api_warmup_calls"],
        "scope": "CUDA-event intervals divided by 64 captured calls and 20 replays; captured GPU work plus amortized replay scheduling, not standalone kernel latency."})
    return result


def measure_eager(torch, functions, run_index, case_index, timing):
    orders = orders_for(functions, run_index, case_index, timing["groups"])
    for name in orders[0]:
        for _ in range(timing["api_warmup_calls"]):
            functions[name]()
    torch.cuda.synchronize()
    result = timed_result(torch, functions, orders, timing["eager"]["calls_per_group"], 1,
                          lambda name: functions[name]())
    result.update({"calls_per_group": timing["eager"]["calls_per_group"],
                   "api_warmup_calls": timing["api_warmup_calls"], "scope": timing["eager"]["scope"]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--run-index", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--phase", choices=("screen", "validation"), required=True)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve evidence")
    if (args.phase == "validation") != (args.selection is not None):
        parser.error("--selection is required only for validation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "started_utc": utc(), "pid": os.getpid(),
        "phase": args.phase, "run_index": args.run_index, "correctness": [], "benchmarks": []}
    code = 1
    try:
        report["run_manifest"] = verify_files()
        protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
        report["protocol_sha256"] = sha(ROOT / "protocol.json")
        threads = protocol["screen"]["threads"]
        if args.selection:
            selection = json.loads(args.selection.read_text(encoding="utf-8"))
            if selection["protocol_sha256"] != report["protocol_sha256"] or selection["phase"] != "screen":
                raise RuntimeError("selection protocol/phase mismatch")
            selected = selection["selected_threads"]
            if not selected or len(selected) != len(set(selected)) or any(t not in (32, 64) for t in selected):
                raise RuntimeError("no valid screen-qualified thread configuration")
            threads = sorted(selected + protocol["validation"]["control_threads"])
            report["selection"] = {"sha256": sha(args.selection), "data": selection}
        report["threads"] = threads
        report["before"] = snapshot()
        idle = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        if any(int(line.strip()) != 0 for line in idle.splitlines()):
            raise RuntimeError("GPU activity observed before job; coordinate rather than stopping other processes")
        import numpy as np
        import torch
        import fast_hadamard_transform as ref_package
        import fast_hadamard_transform_cuda as ref_backend
        hardware = protocol["hardware"]
        if torch.cuda.device_count() != 1 or hardware["required_name_contains"] not in torch.cuda.get_device_name() or list(torch.cuda.get_device_capability()) != hardware["required_sm"]:
            raise RuntimeError("unexpected GPU; protocol is specific to one visible RTX4090 sm89")
        if os.environ.get("TORCH_CUDA_ARCH_LIST") != hardware["compile_arch"] or os.environ.get("MAX_JOBS") != hardware["max_jobs"]:
            raise RuntimeError("require TORCH_CUDA_ARCH_LIST=8.9 and MAX_JOBS=1")
        report["reference"] = reference_tools.provenance(ref_package, ref_backend, args.reference_repo)
        op = load_extension()
        report["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "numpy": np.__version__, "gpu": torch.cuda.get_device_name(),
            "sm": list(torch.cuda.get_device_capability()), "cpp11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
            "extension_file": str(Path(op.__file__).resolve()), "extension_sha256": sha(op.__file__),
            "nvcc": subprocess.check_output([str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc"), "--version"], text=True),
            "torch_cuda_arch_list": os.environ["TORCH_CUDA_ARCH_LIST"], "max_jobs": os.environ["MAX_JOBS"]}
        cases = configurations(protocol, args.phase)
        random.Random(92000 + args.run_index).shuffle(cases)
        report["configuration_order"] = cases
        with torch.inference_mode():
            for case in cases:
                check_case(torch, np, op, ref_package.hadamard_transform, case, threads, protocol, report,
                           args.output.with_name(args.output.stem + "_witness"))
            for index, case in enumerate(cases):
                dtype = torch.float16 if case["dtype"] == "fp16" else torch.bfloat16
                x = reference_tools.make_input(torch, case["shape"], dtype, "normal", 2026, "cuda")
                functions = {f"thread{t}": (lambda t=t: op.hadamard(x, case["scale"], t)) for t in threads}
                functions["dao"] = lambda: ref_package.hadamard_transform(x, case["scale"])
                scopes = ["graph", "eager"] if (index + args.run_index) % 2 == 0 else ["eager", "graph"]
                entry = {**case, "mode": "transform", "scope_order": scopes}
                report["benchmarks"].append(entry)
                for scope in scopes:
                    report["active_context"] = {**case, "phase": "timing", "scope": scope}
                    measure = measure_graph if scope == "graph" else measure_eager
                    entry[scope] = measure(torch, functions, args.run_index, index, protocol["timing"])
                    gc.collect()
                print("PROGRESS", json.dumps({**case, "timed_configurations": len(report["benchmarks"])}), flush=True)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report["status"] = "PASS"
        report["summary"] = {"unique_configurations": len(cases),
            "correctness_input_cases": sum(len(entry["checks"]) for entry in report["correctness"]),
            "graph_configurations": len(cases), "eager_configurations": len(cases)}
        code = 0
    except Exception as error:
        report.update({"status": "FAIL", "error": repr(error), "traceback": traceback.format_exc()})
        print(report["traceback"], file=sys.stderr, flush=True)
    finally:
        try:
            report["after"] = snapshot()
        except Exception as error:
            report["after_snapshot_error"] = repr(error)
        report["finished_utc"] = utc()
        report["exit_code"] = code
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": report.get("summary"), "output": str(args.output)}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
