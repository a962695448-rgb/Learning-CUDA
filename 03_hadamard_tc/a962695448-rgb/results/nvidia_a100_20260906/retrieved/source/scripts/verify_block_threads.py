#!/usr/bin/env python3
"""复用原1,800组Dao矩阵，核查旧默认API与显式128/256线程；不自动派发。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import compare_reference


def load_for_validation(build_directory):
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parents[1]
    build = Path(build_directory)
    build.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MAX_JOBS", "1")
    return load(name="infinitensor_hadamard_block_threads_check", sources=[str(root / "src/torch_binding.cu")],
                extra_include_paths=[str(root / "include")], extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo",
                    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr"], build_directory=str(build), verbose=True)


def bitwise_equal(torch, actual, expected, name):
    a = actual if isinstance(actual, tuple) else (actual,)
    b = expected if isinstance(expected, tuple) else (expected,)
    if len(a) != len(b) or any(x.shape != y.shape or x.dtype != y.dtype or
        not torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(a, b)):
        raise RuntimeError("default/128/256 bitwise mismatch: " + name)


class CheckedInterface:
    """同一输入核查三种调用，向原Dao对照器返回显式256结果。"""
    def __init__(self, torch, extension):
        self.torch, self.extension, self.__file__ = torch, extension, extension.__file__

    def hadamard(self, values, scale=1.0):
        default = self.extension.hadamard(values, scale)
        explicit128 = self.extension.hadamard(values, scale, block_threads=128)
        explicit256 = self.extension.hadamard(values, scale, block_threads=256)
        bitwise_equal(self.torch, default, explicit128, "transform default vs128")
        bitwise_equal(self.torch, explicit256, explicit128, "transform 256 vs128")
        return explicit256

    def hadamard_int4(self, values, scale=1.0):
        default = self.extension.hadamard_int4(values, scale)
        explicit128 = self.extension.hadamard_int4(values, scale, block_threads=128)
        explicit256 = self.extension.hadamard_int4(values, scale, block_threads=256)
        bitwise_equal(self.torch, default, explicit128, "fused default vs128")
        bitwise_equal(self.torch, explicit256, explicit128, "fused 256 vs128")
        return explicit256

    def quantize_int4(self, values):
        default = self.extension.quantize_int4(values)
        explicit128 = self.extension.quantize_int4(values, block_threads=128)
        explicit256 = self.extension.quantize_int4(values, block_threads=256)
        bitwise_equal(self.torch, default, explicit128, "quantize default vs128")
        bitwise_equal(self.torch, explicit256, explicit128, "quantize 256 vs128")
        return explicit256


def check_thread_rejections(torch, extension):
    values = torch.ones((3, 16), device="cuda", dtype=torch.float16)
    results = []
    for name in ("hadamard", "hadamard_int4", "quantize_int4"):
        for invalid in (-1, 0, 32, 64, 127, 129, 512, 128.5, "256"):
            try:
                getattr(extension, name)(values, block_threads=invalid)
            except (RuntimeError, TypeError, ValueError) as error:
                results.append({"method": name, "value": invalid, "pass": True, "error": str(error).splitlines()[0]})
            else:
                raise RuntimeError(f"invalid block_threads accepted: {name} {invalid!r}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-repo", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path, required=True, help="独立扩展构建目录")
    parser.add_argument("--json", type=Path, required=True, help="新的结果文件，禁止覆盖")
    args = parser.parse_args()
    if args.json.exists():
        parser.error("result already exists; choose a fresh output")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "RUNNING", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "reference_commit_required": compare_reference.REFERENCE_COMMIT,
              "scope": "Original 1800 input/shape/dtype/scale cases; each checks default==128==256 bitwise. Repeated calls are not new distinct cases. No performance measurement here."}
    code = 1
    try:
        import torch
        extension = load_for_validation(args.build_directory)
        sample = torch.tensor([[3.0, 1.0]], device="cuda", dtype=torch.float16)
        bitwise_equal(torch, extension.hadamard(sample), extension.hadamard(sample, 1.0, 128), "legacy one-argument transform")
        bitwise_equal(torch, extension.hadamard_int4(sample), extension.hadamard_int4(sample, 1.0, 128), "legacy one-argument fused")
        report["legacy_one_argument_signatures"] = "PASS"
        checked = CheckedInterface(torch, extension)
        # 只在本验证进程替换装载入口；不修改原脚本、原oracle或安装环境。
        compare_reference.load_extension = lambda *unused, **unused_kwargs: checked
        existing_args = argparse.Namespace(device="cuda:0", reference_repo=str(args.reference_repo),
            verbose=False, build_directory=str(args.build_directory), json=str(args.json), benchmark=False)
        code = compare_reference.run(existing_args, report)
        if code:
            raise RuntimeError("original Dao correctness matrix failed")
        report["thread_value_rejections"] = check_thread_rejections(torch, extension)
        report["default_and_explicit128_and_256_bitwise_equal"] = True
        root = Path(__file__).resolve().parents[1]
        report["source_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("include/kernels.cuh", "include/reference.hpp", "src/torch_binding.cu", "scripts/compare_reference.py", "scripts/verify_block_threads.py")}
        report["status"] = "PASS"
        code = 0
    except Exception as error:
        import traceback
        report.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
        code = 1
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": report.get("summary"),
                      "invalid_thread_cases": len(report.get("thread_value_rejections", []))}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
