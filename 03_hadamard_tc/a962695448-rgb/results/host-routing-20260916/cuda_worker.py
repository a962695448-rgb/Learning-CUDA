"""Run one real-GPU validation or timing process for the frozen host-routing candidate."""

import argparse
import functools
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

from validation_common import (
    ROOT,
    compiler_info,
    digest,
    gpu_environment,
    save,
    verify_sources,
)

PROJECT = Path("03_hadamard_tc/a962695448-rgb")


def load_modules(build_root, hardware):
    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = ".".join(
        map(str, hardware["compute_capability"])
    )
    modules = {}
    flags = [
        "-O3",
        "-std=c++17",
        "-lineinfo",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
    ]
    for version, directory in (
        ("control", ROOT / "sources/cuda" / PROJECT),
        ("candidate", ROOT / "candidate/cuda" / PROJECT),
    ):
        build = build_root / version
        build.mkdir(parents=True, exist_ok=True)
        modules[version] = load(
            name="hadamard_routing_" + version,
            sources=[str(directory / "src/torch_binding.cu")],
            extra_include_paths=[str(directory / "include")],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=flags,
            build_directory=str(build),
            with_cuda=True,
            verbose=True,
        )
    return modules


def invocation(module, operation, x, layout, scale=1.0, threads=128):
    if operation == "transform":
        return functools.partial(module.hadamard, x, scale, threads, layout)
    return functools.partial(
        module.hadamard_int4, x, scale, threads, "original", layout
    )


def same_bits(torch, actual, expected):
    if isinstance(actual, tuple):
        if not isinstance(expected, tuple) or len(actual) != len(expected):
            raise AssertionError("Output structures differ.")
        for left, right in zip(actual, expected):
            same_bits(torch, left, right)
        return
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError("Output shape or dtype differs.")
    if not torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    ):
        raise AssertionError("Output bits differ.")


def cpu_transform(torch, np, x, scale):
    values = x.detach().cpu().float().numpy().copy()
    n = values.shape[-1]
    stride = 1
    while stride < n:
        blocks = values.reshape(-1, n // (2 * stride), 2 * stride)
        left, right = blocks[..., :stride].copy(), blocks[..., stride:].copy()
        blocks[..., :stride] = left + right
        blocks[..., stride:] = left - right
        stride *= 2
    values = values * np.float32(scale)
    return torch.from_numpy(values).to(x.dtype)


def cpu_quantize(torch, np, rounded):
    values = rounded.detach().cpu().float().numpy()
    maximum = np.max(np.abs(values), axis=-1)
    scales = np.where(maximum == 0, np.float32(1), maximum / np.float32(7)).astype(
        np.float32
    )
    q = np.clip(np.rint(values / scales[..., None]), -7, 7).astype(np.int32)
    padded = np.zeros((*q.shape[:-1], (q.shape[-1] + 1) // 2 * 2), dtype=np.int32)
    padded[..., : q.shape[-1]] = q
    packed = ((padded[..., ::2] & 15) | ((padded[..., 1::2] & 15) << 4)).astype(
        np.uint8
    )
    return torch.from_numpy(packed), torch.from_numpy(scales)


def validate(torch, np, modules, protocol):
    rng = torch.Generator().manual_seed(protocol["seed"])
    count = 0
    for dtype in (torch.float16, torch.bfloat16):
        tolerance = 0.01 if dtype == torch.float16 else 0.05
        for rows in (1, 17, 65, 257):
            for dim in (1, 2, 4, 8, 16, 32, 64, 128, 256):
                x = torch.randn((rows, dim), generator=rng).to(dtype).cuda()
                for scale in dict.fromkeys((1.0, 1.0 / math.sqrt(dim))):
                    reference = cpu_transform(torch, np, x, scale)
                    for layout in ("original", "auto") + (
                        ("packed",) if dim <= 16 else ()
                    ):
                        for threads in (128, 256):
                            old_y = invocation(
                                modules["control"],
                                "transform",
                                x,
                                layout,
                                scale,
                                threads,
                            )()
                            new_y = invocation(
                                modules["candidate"],
                                "transform",
                                x,
                                layout,
                                scale,
                                threads,
                            )()
                            same_bits(torch, new_y, old_y)
                            error = (
                                (old_y.cpu().float() - reference.float())
                                .abs()
                                .max()
                                .item()
                            )
                            if not error < tolerance:
                                raise AssertionError(
                                    f"Independent transform error {error} exceeds {tolerance}."
                                )
                            expected_q = cpu_quantize(torch, np, old_y)
                            old_q = invocation(
                                modules["control"], "fused", x, layout, scale, threads
                            )()
                            new_q = invocation(
                                modules["candidate"], "fused", x, layout, scale, threads
                            )()
                            same_bits(torch, new_q, old_q)
                            same_bits(torch, tuple(t.cpu() for t in old_q), expected_q)
                            same_bits(
                                torch,
                                modules["candidate"].quantize_int4(new_y, threads),
                                old_q,
                            )
                            count += 1
    for dtype in (torch.float16, torch.bfloat16):
        x = torch.zeros((2, 1, 17, 256), device="cuda", dtype=dtype)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for operation in ("transform", "fused"):
                old = invocation(modules["control"], operation, x, "auto")()
                new = invocation(modules["candidate"], operation, x, "auto")()
                stream.synchronize()
                same_bits(torch, new, old)
                count += 1
        x = torch.randn((17, 64), generator=rng).to(dtype).cuda()
        for operation in ("transform", "fused"):
            outputs = {}
            graphs = []
            for name, module in modules.items():
                call = invocation(module, operation, x, "auto")
                call()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs[name] = call()
                graph.replay()
                torch.cuda.synchronize()
                graphs.append(graph)
            same_bits(torch, outputs["candidate"], outputs["control"])
            count += 1
    x = torch.ones((17, 64), device="cuda", dtype=torch.float16)
    invalid = [
        ("wrong dtype", lambda m: m.hadamard(x.float()), "dtype"),
        ("wrong rank", lambda m: m.hadamard(x.reshape(-1)), "dimensions"),
        ("non-contiguous", lambda m: m.hadamard(x.T), "contiguous"),
        ("empty", lambda m: m.hadamard(x[:0]), "nonempty"),
        ("invalid dim", lambda m: m.hadamard(x[:, :3].contiguous()), "power of two"),
        (
            "autograd",
            lambda m: m.hadamard(x.detach().requires_grad_(True)),
            "requires_grad",
        ),
        ("invalid scale", lambda m: m.hadamard(x, 0.0), "scale"),
        ("nonfinite scale", lambda m: m.hadamard(x, float("nan")), "scale"),
        ("invalid threads", lambda m: m.hadamard(x, 1.0, 64), "block_threads"),
        ("invalid layout", lambda m: m.hadamard(x, 1.0, 128, "bad"), "row_layout"),
        ("invalid packed", lambda m: m.hadamard(x, 1.0, 128, "packed"), "row_layout"),
        (
            "invalid fused layout",
            lambda m: m.hadamard_int4(x, 1.0, 128, "contiguous256"),
            "256",
        ),
    ]
    for name, call, message in invalid:
        types = []
        for module in modules.values():
            try:
                call(module)
            except Exception as error:
                if message not in str(error):
                    raise AssertionError(
                        f"{name}: unexpected diagnostic: {error}"
                    ) from error
                types.append(type(error).__name__)
            else:
                raise AssertionError(name + " was not rejected.")
        if types[0] != types[1]:
            raise AssertionError(name + ": exception types differ.")
    return {"configurations": count, "rejections": len(invalid), "status": "PASS"}


def benchmark(torch, np, modules, protocol, round_index):
    rng = torch.Generator().manual_seed(protocol["seed"])
    records = []
    for kind in ("target", "control"):
        for rows, dim in protocol[kind + "_shapes"]:
            for dtype_name in protocol["dtypes"]:
                dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
                x = torch.randn((rows, dim), generator=rng).to(dtype).cuda()
                expected_transform = cpu_transform(torch, np, x, protocol["scale"])
                for layout in protocol["layouts"]:
                    for operation in protocol["operations"]:
                        calls = {
                            key: invocation(
                                module, operation, x, layout, protocol["scale"]
                            )
                            for key, module in modules.items()
                        }
                        expected = calls["control"]()
                        same_bits(torch, calls["candidate"](), expected)
                        if operation == "transform":
                            tolerance = 0.01 if dtype == torch.float16 else 0.05
                            if (
                                not (
                                    expected.cpu().float() - expected_transform.float()
                                )
                                .abs()
                                .max()
                                .item()
                                < tolerance
                            ):
                                raise AssertionError(
                                    "Benchmark reference disagrees with CPU transform."
                                )
                        else:
                            rounded = invocation(
                                modules["control"],
                                "transform",
                                x,
                                layout,
                                protocol["scale"],
                            )()
                            same_bits(
                                torch,
                                tuple(t.cpu() for t in expected),
                                cpu_quantize(torch, np, rounded),
                            )
                        for group in range(protocol["groups"]):
                            order = ["control", "candidate"]
                            if (group + round_index) % 2:
                                order.reverse()
                            for version in order:
                                call = calls[version]
                                for _ in range(protocol["warmup"]):
                                    last = call()
                                torch.cuda.synchronize()
                                started = time.perf_counter_ns()
                                for _ in range(protocol["repeats"]):
                                    last = call()
                                torch.cuda.synchronize()
                                elapsed = time.perf_counter_ns() - started
                                same_bits(torch, last, expected)
                                records.append(
                                    {
                                        "round": round_index,
                                        "group": group,
                                        "kind": kind,
                                        "dtype": dtype_name,
                                        "rows": rows,
                                        "dim": dim,
                                        "layout": layout,
                                        "operation": operation,
                                        "version": version,
                                        "repeats": protocol["repeats"],
                                        "elapsed_ns": elapsed,
                                        "mean_us": elapsed / protocol["repeats"] / 1000,
                                    }
                                )
    return {"status": "PASS", "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "benchmark"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--round", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("The result file must not exist.")
    report = {"status": "UNVERIFIED", "mode": args.mode}
    stage = "preflight"
    try:
        import numpy as np
        import torch

        original_hashes = verify_sources("cuda", ROOT / "sources/cuda")
        proof = json.loads((ROOT / "checks/host-routing/result.json").read_text())
        candidate_root = ROOT / "candidate/cuda"
        binding = (PROJECT / "src/torch_binding.cu").as_posix()
        candidate_hashes = {}
        for name, expected in original_hashes.items():
            actual = digest(candidate_root / name)
            if actual != (proof["candidate_sha256"] if name == binding else expected):
                raise RuntimeError("Candidate source mismatch: " + name)
            candidate_hashes[name] = actual
        hardware = gpu_environment(allow_non_a100=True)
        if hardware["compute_capability"][0] < 8:
            raise RuntimeError("The frozen project requires sm80+ for its BF16 build.")
        report["hardware"] = hardware
        report["cuda_compiler"] = compiler_info()
        if not report["cuda_compiler"]["available"]:
            raise RuntimeError(
                "nvcc must be available on PATH for the paired CUDA build."
            )
        report["numpy"] = np.__version__
        protocol = json.loads((ROOT / "CUDA_PROTOCOL.json").read_text())
        report["protocol_sha256"] = digest(ROOT / "CUDA_PROTOCOL.json")
        report["control_binding_sha256"] = original_hashes[binding]
        report["candidate_binding_sha256"] = proof["candidate_sha256"]
        stage = "build"
        modules = load_modules(args.build_root, hardware)
        report["binaries"] = {
            key: {
                "sha256": digest(Path(module.__file__)),
                "name": Path(module.__file__).name,
            }
            for key, module in modules.items()
        }
        stage = args.mode
        report["result"] = (
            validate(torch, np, modules, protocol)
            if args.mode == "validate"
            else benchmark(torch, np, modules, protocol, args.round)
        )
        if verify_sources("cuda", ROOT / "sources/cuda") != original_hashes:
            raise RuntimeError("Control sources changed during execution.")
        if {
            name: digest(candidate_root / name) for name in candidate_hashes
        } != candidate_hashes:
            raise RuntimeError("Candidate sources changed during execution.")
        report["status"] = "PASS"
    except Exception as error:  # noqa: BLE001 - Preserve the report for any GPU exception.
        report["stage"] = stage
        report["status"] = "UNVERIFIED" if stage in ("preflight", "build") else "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, report)
    print(
        json.dumps(
            {"mode": args.mode, "status": report["status"], "output": str(args.output)}
        )
    )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
