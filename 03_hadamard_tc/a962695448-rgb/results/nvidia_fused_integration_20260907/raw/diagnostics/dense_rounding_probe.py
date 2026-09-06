"""Investigate the fixed failed witness without changing the frozen protocol."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import integration_checks as checks
import measurement_helpers as measure


def bits32(value):
    return f"{struct.unpack('<I', struct.pack('<f', float(value)))[0]:08x}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--reference-repo", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "pid": os.getpid(), "started_utc": measure.utc(), "timing_performed": False}
    code = 1
    try:
        if not __debug__ or sys.flags.optimize:
            raise RuntimeError("assertions required")
        checks.check_frozen_files()
        report["protocol_sha256"] = measure.sha(ROOT / "protocol.json")
        report["source_manifest_sha256"] = measure.sha(ROOT / "source_manifest.json")
        report["probe_sha256"] = measure.sha(__file__)
        failed = json.loads((ROOT / "runs/holdout/run1.json").read_text())
        case = failed["active_context"]
        assert case == {"dtype": "fp16", "rows": 16383, "dim": 256, "shape": [16383, 256],
            "normalized": False, "scale": 1.0, "phase": "correctness", "pattern": "normal", "seed": 95811, "pointer_mod16": 0}
        report["failed_case"] = case
        report["failed_report_sha256"] = measure.sha(ROOT / "runs/holdout/run1.json")
        import numpy as np
        import torch
        import fast_hadamard_transform as dao_package
        import fast_hadamard_transform_cuda as dao_backend
        report["reference"] = checks.reference_tools.provenance(dao_package, dao_backend, args.reference_repo)
        op = checks.load_production()
        report["environment"] = {"torch": torch.__version__, "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(), "sm": list(torch.cuda.get_device_capability()),
            "extension_file": str(Path(op.__file__).resolve()), "extension_sha256": measure.sha(op.__file__)}
        assert report["environment"]["extension_sha256"] == failed["environment"]["extension_sha256"]
        with torch.inference_mode():
            values = checks.reference_tools.make_input(torch, case["shape"], torch.float16, "normal", 95811, "cuda")
            x, storage, before = checks.guarded_input(torch, values, 0)
            gpu = op.hadamard(x, 1.0, 128)
            dao = dao_package.hadamard_transform(x, 1.0)
            original = op.hadamard_int4(x, 1.0, 128)
            candidate = op.hadamard_int4(x, 1.0, 128, "contiguous256")
            split = op.quantize_int4(gpu, 128)
            measure.exact(torch, storage, before, "input and guards")
            measure.exact(torch, gpu, dao, "whole tensor original GPU vs fixed Dao")
            measure.exact(torch, original, candidate, "whole tensor original/candidate fused")
            measure.exact(torch, original, split, "whole tensor fused/split")
            report["whole_tensor_checks"] = {"input_guards_unchanged": True, "gpu_dao_bitwise_equal": True,
                "original_candidate_fused_bitwise_equal": True, "fused_split_bitwise_equal": True,
                "elements": x.numel()}
            input_bits = x.cpu().numpy().view(np.uint16).copy()
            raw_path = args.output_directory / "full_input_fp16_bits.npy"
            np.save(raw_path, input_bits, allow_pickle=False)
            report["full_input"] = {"path": raw_path.name, "file_sha256": measure.sha(raw_path),
                "element_bytes_sha256": hashlib.sha256(input_bits.tobytes()).hexdigest(),
                "shape": list(input_bits.shape), "dtype": "uint16 raw IEEE FP16 bits", "size": raw_path.stat().st_size}
            for name, result in (("gpu", gpu), ("dao", dao), ("original_packed", original[0]),
                                 ("candidate_packed", candidate[0]), ("original_scales", original[1]), ("candidate_scales", candidate[1])):
                raw = result.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                report.setdefault("whole_output_element_hashes", {})[name] = hashlib.sha256(raw).hexdigest()
            indices = [0, 8191, 16382]
            sample = input_bits[indices].view(np.float16).astype(np.float32)
            cpu32 = sample.copy()
            stages = []
            for stride in (1, 2, 4, 8, 16, 32, 64, 128):
                chunks = cpu32.reshape(3, -1, stride * 2)
                a, b = chunks[..., :stride].copy(), chunks[..., stride:].copy()
                chunks[..., :stride] = np.add(a, b, dtype=np.float32)
                chunks[..., stride:] = np.subtract(a, b, dtype=np.float32)
                stages.append(cpu32.copy())
            signs = np.array([[(-1.0 if (i & j).bit_count() % 2 else 1.0) for j in range(256)] for i in range(256)], dtype=np.float64)
            dense = sample.astype(np.float64) @ signs
            direct = dense.astype(np.float16)
            via32 = dense.astype(np.float32).astype(np.float16)
            cpu32_half = cpu32.astype(np.float16)
            gpu_sample = gpu[indices].cpu().numpy()
            dao_sample = dao[indices].cpu().numpy()
            assert np.array_equal(gpu_sample.view(np.uint16), cpu32_half.view(np.uint16))
            report["sample_checks"] = {"rows": indices, "gpu_equals_cpu_stage_fp32_bits": True,
                "gpu_vs_dense_via_fp32_max_abs_error": float(np.max(np.abs(gpu_sample.astype(np.float64) - via32.astype(np.float64)))),
                "gpu_vs_direct_dense_fp16_max_abs_error": float(np.max(np.abs(gpu_sample.astype(np.float64) - direct.astype(np.float64)))),
                "dense_via_fp32_vs_direct_fp16_differing_elements": int(np.count_nonzero(via32.view(np.uint16) != direct.view(np.uint16))),
                "gpu_vs_unrounded_dense_max_abs_error": float(np.max(np.abs(gpu_sample.astype(np.float64) - dense)))}
            locations = np.argwhere(gpu_sample.view(np.uint16) != via32.view(np.uint16))
            assert len(locations) > 0
            report["differences"] = []
            for i, j in locations:
                i, j = int(i), int(j)
                exact = sum((Fraction(float(value)) * (-1 if (k & j).bit_count() % 2 else 1)
                             for k, value in enumerate(sample[i])), Fraction())
                direct_bits = int(direct.view(np.uint16)[i, j])
                candidates = []
                for raw in range(max(0, direct_bits - 3), min(65535, direct_bits + 3) + 1):
                    val = float(np.array([raw], dtype=np.uint16).view(np.float16)[0])
                    if np.isfinite(val):
                        candidates.append((abs(Fraction(val) - exact), raw & 1, raw, val))
                nearest = min(candidates)
                assert nearest[2] == direct_bits, "NumPy direct conversion differs from exact rational RNE"
                actual, expected = float(gpu_sample[i, j]), float(via32[i, j])
                report["differences"].append({"row": indices[i], "column": j,
                    "gpu": actual, "gpu_fp16_bits": f"{int(gpu_sample.view(np.uint16)[i,j]):04x}",
                    "dao": float(dao_sample[i,j]), "cpu_stage_fp32": float(cpu32[i,j]), "cpu_stage_fp32_bits": bits32(cpu32[i,j]),
                    "dense_fp64": float(dense[i,j]), "dense_exact_numerator": exact.numerator, "dense_exact_denominator": exact.denominator,
                    "dense_cast_fp32": float(np.float32(dense[i,j])), "dense_cast_fp32_bits": bits32(np.float32(dense[i,j])),
                    "dense_via_fp32_fp16": expected, "dense_via_fp32_fp16_bits": f"{int(via32.view(np.uint16)[i,j]):04x}",
                    "direct_fp64_to_fp16": float(direct[i,j]), "direct_fp16_bits": f"{direct_bits:04x}",
                    "direct_verified_against_exact_rational_rne": True,
                    "midpoint_between_gpu_and_via32": (actual + expected) / 2,
                    "exact_value_is_neighbor_midpoint": exact == Fraction((actual + expected) / 2),
                    "neighbor_fp16_gap": abs(actual - expected),
                    "fp16_ulp_toward_positive_infinity": abs(float(np.nextafter(np.float16(actual), np.float16(np.inf))) - actual),
                    "gpu_abs_error_vs_unrounded_dense": abs(actual - float(dense[i,j])),
                    "gpu_abs_error_vs_rounded_reference": abs(actual - expected),
                    "exact_gpu_error": str(abs(Fraction(actual) - exact)),
                    "exact_direct_fp16_error": str(abs(Fraction(float(direct[i,j])) - exact)),
                    "cpu_stage_values": [float(stage[i,j]) for stage in stages]})
            report["input_rows"] = [{"row": row, "values": sample[i].astype(float).tolist(),
                                     "fp16_bits": [f"{int(v):04x}" for v in input_bits[row]]} for i, row in enumerate(indices)]
            np.savez(args.output_directory / "sample_intermediates.npz", input_bits=input_bits[indices], gpu=gpu_sample,
                dao=dao_sample, cpu_stage_fp32=np.stack(stages), dense_fp64=dense, dense_direct_fp16=direct, dense_via_fp32_fp16=via32)
            report["sample_intermediates_sha256"] = measure.sha(args.output_directory / "sample_intermediates.npz")
        checks.check_frozen_files()
        report["frozen_sources_unchanged"] = True
        report["status"] = "DIAGNOSED_ORIGINAL_FAILURE_RETAINED"
        code = 0
    except Exception as error:
        import traceback
        report.update(status="DIAGNOSTIC_FAILED", error=repr(error), traceback=traceback.format_exc())
    report.update(finished_utc=measure.utc(), exit_code=code)
    (args.output_directory / "diagnostic.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "sample_checks": report.get("sample_checks"), "differences": report.get("differences"), "error": report.get("error")}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
