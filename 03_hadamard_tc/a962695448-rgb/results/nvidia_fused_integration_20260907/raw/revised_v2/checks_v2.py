"""V2 validates the reference contract and an independent exact error certificate."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE))
import integration_checks as original_checks
import measurement_helpers as measure
from numeric_certificate import certify_samples


def verify_revision_files():
    protocol = json.loads((ROOT / "protocol_v2.json").read_text())
    reuse = protocol["reused_regression"]
    if measure.sha(BASE / "run_manifest.json") != reuse["original_run_manifest_sha256"]:
        raise RuntimeError("original validation manifest changed")
    original_checks.check_frozen_files()
    if measure.sha(BASE / "source_manifest.json") != reuse["source_manifest_sha256"]:
        raise RuntimeError("production source manifest changed")
    legacy = BASE / "runs/holdout/run1.json"
    diagnostic = BASE / "diagnostics/dense_rounding/diagnostic.json"
    if measure.sha(legacy) != protocol["revision"]["original_failure_sha256"] or measure.sha(diagnostic) != protocol["revision"]["diagnostic_sha256"]:
        raise RuntimeError("original failure or diagnosis was changed")
    legacy_report = json.loads(legacy.read_text())
    if legacy_report["status"] != "FAIL" or legacy_report["benchmarks"]:
        raise RuntimeError("original V1 failure was reclassified or timing imported")
    manifest = json.loads((ROOT / "manifest_v2.json").read_text())
    for name, item in manifest["validation_files"].items():
        path = ROOT / name
        if path.stat().st_size != item["size"] or measure.sha(path) != item["sha256"]:
            raise RuntimeError("revised validation file changed: " + name)
    return protocol, manifest


def verify_regression(path, protocol):
    expected = protocol["reused_regression"]
    if measure.sha(path) != expected["report_sha256"]:
        raise RuntimeError("not the approved actual regression report")
    report = json.loads(Path(path).read_text())
    if report["status"] != "PASS" or report["exit_code"] != 0 or report["holdout_allowed"] is not True:
        raise RuntimeError("full regression did not pass")
    if report["source_manifest_sha256"] != expected["source_manifest_sha256"]:
        raise RuntimeError("regression used a different source snapshot")
    if report["binaries"]["production_extension"]["sha256"] != expected["production_binary_sha256"]:
        raise RuntimeError("regression production binary identity differs")
    return report


def verify_build_assumptions():
    path = BASE / "build/production/build.ninja"
    text = path.read_text()
    if re.search(r"(?:--|-)(?:use_fast_math|ftz(?:=|\s+)true)(?:\s|$)", text):
        raise RuntimeError("fast-math/FTZ-enabled build is outside the stated certificate model")
    if "compute_89" not in text or "sm_89" not in text:
        raise RuntimeError("expected actual sm89 production build command")
    return {"build_ninja_sha256": measure.sha(path), "explicit_fast_math_or_ftz_true": False,
            "model": "NVCC default RN32 and gradual-underflow semantics, no fast-math/FTZ opt-in; CPU RN/gradual conditions checked inside certificate"}


def store_sample_buffer(buffers, bits):
    canonical = bits.astype("<u2", copy=False)
    digest = hashlib.sha256(canonical.tobytes()).hexdigest()
    payload = {"shape": list(canonical.shape), "dtype": "uint16_bits", "values": canonical.tolist()}
    if digest in buffers and buffers[digest] != payload:
        raise RuntimeError("sample-buffer hash collision or shape mismatch")
    buffers[digest] = payload
    return digest


def check_input(torch, np, op, dao, case, pattern, seed, offset, sample_buffers):
    dtype = torch.float16 if case["dtype"] == "fp16" else torch.bfloat16
    values = original_checks.reference_tools.make_input(torch, case["shape"], dtype, pattern, seed, "cuda")
    x, storage, before = original_checks.guarded_input(torch, values, offset)
    transformed = op.hadamard(x, case["scale"], 128)
    original = op.hadamard_int4(x, case["scale"], 128)
    explicit_original = op.hadamard_int4(x, case["scale"], 128, "original")
    candidate = op.hadamard_int4(x, case["scale"], 128, "contiguous256")
    split = op.quantize_int4(transformed, 128)
    reference_input = x if offset == 0 else x.clone()
    if reference_input.data_ptr() % 16:
        raise RuntimeError("reference correctness copy is not aligned")
    reference_output = dao(reference_input, case["scale"])
    measure.exact(torch, reference_input, x, "reference correctness input bits")
    measure.exact(torch, storage, before, "original input/guards")
    measure.exact(torch, original, explicit_original, "three-argument API compatibility")
    measure.exact(torch, candidate, original, "original/candidate fused")
    measure.exact(torch, candidate, split, "fused/split")
    limit = 0.01 if dtype == torch.float16 else 0.05
    reference_metrics = original_checks.reference_tools.metrics(torch, transformed, reference_output, limit)
    if not reference_metrics["pass"]:
        raise RuntimeError("unchanged official reference threshold failed: " + json.dumps(reference_metrics))
    y = transformed.float().cpu().numpy()
    scales = np.max(np.abs(y), axis=1).astype(np.float32) / np.float32(7)
    scales[scales == 0] = np.float32(1)
    q = np.clip(np.rint(y / scales[:, None]), -7, 7).astype(np.int8)
    packed = (q[:, 0::2].astype(np.uint8) & 15) | ((q[:, 1::2].astype(np.uint8) & 15) << 4)
    if not np.array_equal(candidate[0].cpu().numpy(), packed) or candidate[1].cpu().numpy().tobytes() != scales.tobytes():
        raise RuntimeError("unchanged CPU FP32 packed/scales exact check failed")
    indices = sorted({0, case["rows"] // 2, case["rows"] - 1})
    input_bits = x[indices].cpu().contiguous().view(torch.int16).numpy().view(np.uint16)
    output_bits = transformed[indices].cpu().contiguous().view(torch.int16).numpy().view(np.uint16)
    input_hash = store_sample_buffer(sample_buffers, input_bits)
    output_hash = store_sample_buffer(sample_buffers, output_bits)
    certificate = certify_samples(input_bits, output_bits, case["dtype"], case["scale"], row_ids=indices)
    if certificate["status"] != "PASS":
        error = RuntimeError("forward-error certificate failed: " + json.dumps(certificate.get("first_failure")))
        error.certificate = certificate
        raise error
    return {"pass": True, "pattern": pattern, "seed": seed, "pointer_mod16": offset, "elements": x.numel(),
        "official_reference": reference_metrics, "dao_input_pointer_mod16": 0,
        "dao_aligned_copy_for_offset": offset == 2, "dao_copy_bitwise_equal_input": True,
        "original_candidate_fused_split_exact": True, "cpu_quantization_exact": True,
        "input_guards_unchanged": True, "legacy_three_arg_default_equals_explicit_original": True,
        "sample_input_bits_sha256": input_hash, "sample_gpu_bits_sha256": output_hash,
        "sample_row_ids": indices, "numeric_certificate": certificate,
        "certificate_computed_on_cpu_from_actual_gpu_bits": True,
        "legacy_rounded_dense_threshold_is_not_a_v2_gate": True,
        "legacy_rounded_dense_threshold_would_fail": Fraction(certificate["summary"]["maxima"]["stored_error_vs_via_fp32_rounded_dense"]["fraction"]) >= Fraction(str(limit))}
