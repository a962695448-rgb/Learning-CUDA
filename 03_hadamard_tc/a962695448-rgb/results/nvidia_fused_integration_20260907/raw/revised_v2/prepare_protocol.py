"""Create an explicit revision; preserve v1, source bytes, matrix and timing."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = json.loads((BASE / "protocol.json").read_text())
    revised = copy.deepcopy(original)
    revised["protocol_id"] = "nvidia_fused_integration_20260906_v2_reference_forward_certificate"
    revised["status_at_freeze"] = "CPU_PREPARATION_NOT_RUN"
    revised["revision"] = {
        "original_protocol_sha256": sha(BASE / "protocol.json"),
        "original_failure_sha256": sha(BASE / "server_raw/runs/holdout/run1.json"),
        "diagnostic_sha256": sha(BASE / "server_raw/diagnostics/dense_rounding/diagnostic.json"),
        "original_result": "FAIL_BEFORE_TIMING; retain original49 complete configurations/688 checks and zero benchmark samples. Never relabel v1 as PASS.",
        "reason": "Official FP16/BF16 thresholds apply to reference implementation output. The v1 auxiliary FP64-rounded threshold failed at a storage-grid midpoint despite originalGPU=Dao=CPUstageFP32. Replace only the auxiliary gate by a derived exact forward-error certificate, while retaining all matrix entries, seeds, primary reference limits and exact INT4 checks.",
        "witness": {"rows": 16383, "dim": 256, "dtype": "fp16", "scale": 1, "pattern": "normal", "seed": 95811,
                    "row": 8191, "column": 230, "gpu_bits": "cc7e", "exact_rounded_bits": "cc7d", "rounded_gap": 0.015625},
        "source_or_math_changed": False,
        "matrix_or_seed_or_timing_changed": False
    }
    revised["reused_regression"] = {
        "report_relative_to_integration_root": "runs/regression/regression_report.json",
        "report_sha256": "c525cc8371a6c9faa514243613b9e62e769f17ce29d20e8fbebb564b56704ace",
        "source_manifest_sha256": "ae562cf54a65f306650945e73a6985eb6989099b17cbeba3ea2f607debedb932",
        "original_run_manifest_sha256": "9ebb353585ca881f1a59864e1e097654be3f6886de7f3ddea1ef043ab6922ee9",
        "production_binary_sha256": "eb3f03f28b7f993bfc3351a8afc1022ccc5f93301c95ed3163fd437f6e1f3468",
        "reference_binary_sha256": "2e38b886e3fc6c31c3b837a4fd7354e844dd81d4a20b30f39fbd0c351d8620a4",
        "policy": "Reuse actual full regression PASS only after verifying the original27 file manifest, all13 source bytes, exact report SHA and actual loaded production binary. Compatibility binary remains a separate identity; no rebuild or source rewrite is required."
    }
    revised["holdout"]["correctness"] = "Same52 configurations and7patterns/seeds x0/2-byte offset: all-elements original transform vs fixed Dao with unchanged strict0.01/0.05 limits; original/candidate/split and independent CPU packed/scales remain bit-exact; inputs/guards unchanged. On first/middle/last sample rows, record FP64 rounded/unrounded differences and ULP and require the exact forward-error certificate below.728 input conditions perprocess remain unchanged."
    revised["sample_certificate"] = {
        "rows": "sorted unique{0,M//2,M-1}",
        "mathematical_reference": "Decode actual input16-bit values, lift eachrow to a common power-of-two denominator, and compute integer FWHT8stages plus exact rational scale. Independently retain FP64 dense matrix output and its deviation from exact integer result; never add that FP64 deviation to the FP32 hard bound.",
        "cpu_fp32": "Explicit NumPy float32 add/sub at eachof8stages and one float32 scale multiplication; independently apply exact integer nearest-even FP16/BF16 rounding. Every sample storage bit must equal actual GPU transform output.",
        "u32": "2^-24", "eta32": "2^-149", "gamma9": "9u32/(1-9u32)",
        "E32": "gamma9*abs(scale)*sum(abs(input)) + (eta32/2)*(abs(scale)*255*(1+u32)^8+1)",
        "derivation": "Each output depends on256 input leaves and255 add/sub nodes, with atmost8 additions/subtractions plus1scale rounding per input path. RN32 relative error products are bounded by gamma9; each add/sub absolute gradual-underflow contribution is atmosteta32/2 and passes through atmost8 remaining rounding factors, plus the final scale underflow contribution.",
        "hard_gates": ["all finite, no intermediate/output overflow, modeled IEEE RN32 and gradual underflow", "abs(CPUpre-exact_math)<=E32 using exact Fraction", "RN_dtype(CPUpre) raw bits equal GPU sample raw bits", "delta_storage=abs(RN_dtype(CPUpre)-CPUpre) recorded separately", "abs(GPUstored-exact_math)<=E32+delta_storage using exact Fraction"],
        "display": "Keep exact fractions for L1,gamma,bounds/errors. If bounds are exported as float, round outward with nextafter toward+infinity; displayed values do not decide gates.",
        "diagnostics_only": "Retain old FP64->FP32->dtype rounded error, directly correctly-rounded exact-math dtype error, unrounded FP64 error, local ULP and rounding-route differences. No fixed0.01/0.05 gate on the independently rounded FP64 auxiliary result.",
        "assumptions": "FP16/BF16 embedding in FP32 is exact; finite inputs; eight-stage FWHT then finite positive scale; IEEE round-to-nearest-even arithmetic and gradual FP32 underflow; no fast-math/FTZ-enabled build. Fail explicitly if required model conditions cannot be established."
    }
    revised["reference_input_alignment"] = "For offset2 correctness only, fixed Dao receives an aligned clone verified bit-identical to actual input; both tested fused layouts receive the real offset view. Timing uses one aligned shared input, with no clone measured."
    revised["decision"]["precondition"] = "Exact reused regression/source/binary identity and the new validation manifest verified. V2 is separate and requires independent review/freeze before execution; original v1 failure remains immutable."
    for key in ("rows", "dims", "dtypes", "normalized", "expected_configurations", "block_threads", "mode", "layouts", "exclude_initial_rows", "patterns_and_seeds", "strict_abs_limit", "scale"):
        if revised["holdout"][key] != original["holdout"][key]:
            raise RuntimeError("matrix/seed/primary-limit changed: " + key)
    if revised["timing"] != original["timing"]:
        raise RuntimeError("timing changed")
    target = ROOT / "protocol_v2.json"
    if target.exists():
        raise RuntimeError("preserve existing proposed protocol")
    target.write_text(json.dumps(revised, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PROPOSED_NOT_RUN", "protocol_sha256": sha(target), "matrix_timing_unchanged": True}))


if __name__ == "__main__":
    main()
