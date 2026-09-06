#!/usr/bin/env python3
"""Fixed N256 layout experiment; all numerical checks precede Graph timing."""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "sources"))
import compare_reference as reference_tools
import measurement_helpers as measure


def configurations(protocol):
    domain = protocol["domain"]
    result = [{"dtype": dtype, "dim": 256, "rows": rows, "shape": [rows, 256],
               "normalized": normalized, "scale": 0.0625 if normalized else 1.0}
              for dtype in domain["dtypes"] for rows in domain["rows"] for normalized in domain["normalized"]]
    assert len(result) == domain["expected_configurations"] == 24
    return result


def load_extension():
    from torch.utils.cpp_extension import load
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    return load(name="hadamard_contiguous256_20260906", sources=[str(ROOT / "sources/torch_binding_contiguous256.cu")],
        extra_include_paths=[str(ROOT / "sources")], extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo", "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "--expt-relaxed-constexpr"],
        build_directory=str(build), verbose=True)


def guarded_input(torch, values, pointer_mod16):
    prefix = 8 if pointer_mod16 == 0 else 1
    storage = torch.full((values.numel() + 16,), 123, device=values.device, dtype=values.dtype)
    x = storage[prefix:prefix + values.numel()].view(values.shape)
    x.copy_(values)
    if x.data_ptr() % 16 != pointer_mod16 or not x.is_contiguous():
        raise RuntimeError("failed to construct exact input alignment")
    return x, storage, storage.clone()


def check_case_input(torch, np, op, dao, case, pattern, seed, pointer_mod16, stream=None):
    dtype = torch.float16 if case["dtype"] == "fp16" else torch.bfloat16
    def operations():
        values = reference_tools.make_input(torch, case["shape"], dtype, pattern, seed, "cuda")
        x, storage, before = guarded_input(torch, values, pointer_mod16)
        transformed, fused, split = {}, {}, {}
        for layout in ("original", "contiguous256"):
            transformed[layout] = op.hadamard(x, case["scale"], 128, layout)
            fused[layout] = op.hadamard_int4(x, case["scale"], 128, layout)
            split[layout] = op.quantize_int4(transformed[layout], 128, layout)
        # Pinned Dao load_input uses a 16B vector load without an offset guard.
        # For offset validation only, give Dao an aligned bit-identical copy.
        # Both experimental layouts still receive the original offset tensor.
        reference_input = x if pointer_mod16 == 0 else x.clone()
        if reference_input.data_ptr() % 16:
            raise RuntimeError("Dao correctness input is not16B aligned")
        expected = dao(reference_input, case["scale"])
        measure.exact(torch, reference_input, x, "Dao reference copy preserves input bits")
        return x, storage, before, transformed, fused, split, expected
    if stream is None:
        values = operations()
    else:
        with torch.cuda.stream(stream):
            values = operations()
        stream.synchronize()
    x, storage, before, transformed, fused, split, expected = values
    try:
        measure.exact(torch, storage, before, "input and surrounding storage unchanged")
        measure.exact(torch, transformed["contiguous256"], transformed["original"], "layout transform")
        measure.exact(torch, fused["contiguous256"], fused["original"], "layout fused INT4")
        limit = 0.01 if dtype == torch.float16 else 0.05
        metric = reference_tools.metrics(torch, transformed["original"], expected, limit)
        if not metric["pass"]:
            raise RuntimeError("Dao numerical comparison failed: " + json.dumps(metric))
        cpu = []
        for layout in ("original", "contiguous256"):
            measure.exact(torch, fused[layout], split[layout], "fused vs split " + layout)
            cpu.append(measure.cpu_checks(torch, np, x, transformed[layout], *fused[layout], case["scale"]))
        return {"pass": True, "pattern": pattern, "seed": seed, "pointer_mod16": pointer_mod16,
            "elements": x.numel(), "dao_max_abs_error": metric["max_abs_error"],
            "dense_max_abs_error": max(c["dense_max_abs_error"] for c in cpu), "strict_limit": limit,
            "dense_rows": cpu[0]["dense_rows"], "quantization_rows_checked": case["rows"],
            "cpu_quantization_exact": True, "original_new_transform_bitwise_exact": True,
            "original_new_fused_split_bitwise_exact": True, "input_guards_unchanged": True,
            "dao_input_pointer_mod16": 0, "dao_aligned_copy_for_offset": pointer_mod16 == 2,
            "dao_copy_bitwise_equal_input": True,
            "non_default_stream": stream is not None}
    except Exception as error:
        error.cpu_witness = {"input": x.cpu(), "original": transformed["original"].cpu(),
                             "contiguous256": transformed["contiguous256"].cpu(), "dao": expected.cpu()}
        raise


def check_quantization_ties(torch, np, op, dtype_name, pointer_mod16):
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    levels = torch.tensor([-7, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0,
                           0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7], dtype=dtype, device="cuda")
    values = levels.repeat((17 * 256 + 16) // 17)[:17 * 256].reshape(17, 256)
    x, storage, before = guarded_input(torch, values, pointer_mod16)
    y = x.float().cpu().numpy()
    scales = np.max(np.abs(y), axis=1).astype(np.float32) / np.float32(7)
    scales[scales == 0] = np.float32(1)
    q = np.clip(np.rint(y / scales[:, None]), -7, 7).astype(np.int8)
    expected = (q[:, 0::2].astype(np.uint8) & 15) | ((q[:, 1::2].astype(np.uint8) & 15) << 4)
    for layout in ("original", "contiguous256"):
        packed, actual_scales = op.quantize_int4(x, 128, layout)
        if not np.array_equal(packed.cpu().numpy(), expected) or actual_scales.cpu().numpy().tobytes() != scales.tobytes():
            raise RuntimeError("standalone RNE tie check failed: " + layout)
    measure.exact(torch, storage, before, "standalone quantization input/guards")
    return {"dtype": dtype_name, "rows": 17, "dim": 256, "pointer_mod16": pointer_mod16,
            "pass": True, "cpu_quantization_exact": True, "input_guards_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--run-index", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("existing output must be preserved")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "pid": os.getpid(), "started_utc": measure.utc(),
        "run_index": args.run_index, "correctness": [], "stream_checks": [],
        "quantization_tie_checks": [], "benchmarks": []}
    code = 1
    try:
        report["run_manifest"] = json.loads((ROOT / "run_manifest.json").read_text())
        for name, item in report["run_manifest"]["files"].items():
            if measure.sha(ROOT / name) != item["sha256"]:
                raise RuntimeError("frozen input hash mismatch: " + name)
        protocol = json.loads((ROOT / "protocol.json").read_text())
        report["protocol_sha256"] = measure.sha(ROOT / "protocol.json")
        report["before"] = measure.snapshot()
        activity = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        if not activity.strip() or any(int(x) != 0 for x in activity.split()):
            raise RuntimeError("GPU utilization preflight is not idle; no CUDA imported")
        import numpy as np
        import torch
        import fast_hadamard_transform as ref_package
        import fast_hadamard_transform_cuda as ref_backend
        hardware = protocol["hardware"]
        if torch.cuda.device_count() != 1 or hardware["required_name_contains"] not in torch.cuda.get_device_name() or list(torch.cuda.get_device_capability()) != hardware["required_sm"]:
            raise RuntimeError("expected one RTX4090 sm89")
        if os.environ.get("TORCH_CUDA_ARCH_LIST") != "8.9" or os.environ.get("MAX_JOBS") != "1":
            raise RuntimeError("require explicit sm89 and single compile worker")
        report["reference"] = reference_tools.provenance(ref_package, ref_backend, args.reference_repo)
        op = load_extension()
        report["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "numpy": np.__version__, "gpu": torch.cuda.get_device_name(),
            "sm": list(torch.cuda.get_device_capability()), "cpp11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
            "extension_file": str(Path(op.__file__).resolve()), "extension_sha256": measure.sha(op.__file__),
            "nvcc": subprocess.check_output([str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc"), "--version"], text=True),
            "torch_cuda_arch_list": os.environ["TORCH_CUDA_ARCH_LIST"], "max_jobs": os.environ["MAX_JOBS"]}
        cases = configurations(protocol)
        random.Random(92600 + args.run_index).shuffle(cases)
        report["configuration_order"] = cases
        with torch.inference_mode():
            for case in cases:
                entry = {**case, "checks": []}
                report["correctness"].append(entry)
                for pattern, seeds in protocol["correctness"]["patterns_and_seeds"].items():
                    for seed in seeds:
                        for offset in protocol["correctness"]["input_pointer_mod16"]:
                            report["active_context"] = {**case, "phase": "correctness", "pattern": pattern, "seed": seed, "pointer_mod16": offset}
                            entry["checks"].append(check_case_input(torch, np, op, ref_package.hadamard_transform, case, pattern, seed, offset))
                for offset in protocol["correctness"]["input_pointer_mod16"]:
                    report["active_context"] = {**case, "phase": "non_default_stream", "pointer_mod16": offset}
                    stream = torch.cuda.Stream()
                    check = check_case_input(torch, np, op, ref_package.hadamard_transform, case, "normal", 2026, offset, stream)
                    report["stream_checks"].append({**case, **check})
                print("CHECKED", json.dumps(case), flush=True)
            for dtype in protocol["domain"]["dtypes"]:
                for offset in protocol["correctness"]["input_pointer_mod16"]:
                    report["active_context"] = {"phase": "quantization_ties", "dtype": dtype, "pointer_mod16": offset}
                    report["quantization_tie_checks"].append(check_quantization_ties(torch, np, op, dtype, offset))
            for index, case in enumerate(cases):
                dtype = torch.float16 if case["dtype"] == "fp16" else torch.bfloat16
                x = reference_tools.make_input(torch, case["shape"], dtype, "normal", 2026, "cuda")
                assert x.data_ptr() % 16 == 0
                mode_order = ["transform", "fused_int4"] if (index + args.run_index) % 2 == 0 else ["fused_int4", "transform"]
                for mode in mode_order:
                    report["active_context"] = {**case, "phase": "timing", "mode": mode}
                    operation = op.hadamard if mode == "transform" else op.hadamard_int4
                    functions = {layout: (lambda layout=layout: operation(x, case["scale"], 128, layout))
                                 for layout in protocol["launch"]["layouts"]}
                    if mode == "transform":
                        functions["dao"] = lambda: ref_package.hadamard_transform(x, case["scale"])
                    result = measure.measure_graph(torch, functions, args.run_index, index, protocol["timing"])
                    report["benchmarks"].append({**case, "mode": mode, "configuration_index": index,
                        "mode_order": mode_order, **result})
                    gc.collect()
                print("TIMED", json.dumps(case), flush=True)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report["summary"] = {"unique_configurations": 24, "correctness_input_cases": 336,
                              "stream_checks": 48, "quantization_tie_checks": 4, "graph_comparisons": 48}
        report["status"] = "PASS"
        code = 0
    except Exception as error:
        report.update({"status": "FAIL", "error": repr(error), "traceback": traceback.format_exc()})
        if hasattr(error, "cpu_witness"):
            path = args.output.with_name(args.output.stem + "_failure.pt")
            torch.save({**error.cpu_witness, "context": report.get("active_context")}, path)
            report["failure_witness"] = {"path": str(path), "sha256": measure.sha(path)}
        print(report["traceback"], file=sys.stderr, flush=True)
    finally:
        try:
            report["after"] = measure.snapshot()
        except Exception as error:
            report["after_snapshot_error"] = repr(error)
        report.update({"finished_utc": measure.utc(), "exit_code": code})
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": report.get("summary"), "output": str(args.output)}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
