"""Check only the frozen production interfaces using independent CPU quantization."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "project"
sys.path.insert(0, str(PROJECT / "scripts"))
import compare_reference as reference_tools
from build_torch_extension import load_extension
import measurement_helpers as measure


def load_production():
    return load_extension(verbose=True, build_directory=str(ROOT / "build/production"))


def check_frozen_files():
    manifest = json.loads((ROOT / "run_manifest.json").read_text())
    for name, item in manifest["files"].items():
        data = (ROOT / name).read_bytes()
        if len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise RuntimeError("frozen input changed: " + name)
    return manifest


def guarded_input(torch, values, offset):
    prefix = 8 if offset == 0 else 1
    storage = torch.full((values.numel() + 16,), 123, dtype=values.dtype, device=values.device)
    x = storage[prefix:prefix + values.numel()].view(values.shape)
    x.copy_(values)
    if not x.is_contiguous() or x.data_ptr() % 16 != offset:
        raise RuntimeError("alignment fixture failed")
    return x, storage, storage.clone()


def check_input(torch, np, op, case, pattern, seed, offset, stream=None):
    dtype = torch.float16 if case["dtype"] == "fp16" else torch.bfloat16
    def operations():
        values = reference_tools.make_input(torch, case["shape"], dtype, pattern, seed, "cuda")
        x, storage, before = guarded_input(torch, values, offset)
        transformed = op.hadamard(x, case["scale"], 128)
        original = op.hadamard_int4(x, case["scale"], 128)
        explicit_original = op.hadamard_int4(x, case["scale"], 128, "original")
        candidate = op.hadamard_int4(x, case["scale"], 128, "contiguous256")
        split = op.quantize_int4(transformed, 128)
        return x, storage, before, transformed, original, explicit_original, candidate, split
    if stream is None:
        outputs = operations()
    else:
        with torch.cuda.stream(stream):
            outputs = operations()
        stream.synchronize()
    x, storage, before, transformed, original, explicit_original, candidate, split = outputs
    measure.exact(torch, storage, before, "production input/guards unchanged")
    measure.exact(torch, original, explicit_original, "legacy three-argument default vs explicit original")
    measure.exact(torch, candidate, original, "candidate fused vs original")
    measure.exact(torch, candidate, split, "candidate fused vs public split")
    rows = x.numel() // 256
    cpu = measure.cpu_checks(torch, np, x.reshape(rows, 256), transformed.reshape(rows, 256),
                             candidate[0].reshape(rows, 128), candidate[1].reshape(rows), case["scale"])
    return {"pass": True, "pattern": pattern, "seed": seed, "pointer_mod16": offset,
        "elements": x.numel(), "original_candidate_fused_split_exact": True,
        "legacy_three_arg_default_equals_explicit_original": True, "input_guards_unchanged": True,
        "non_default_stream": stream is not None, **cpu}


def directed_cases(torch, np, op, protocol, report):
    spec = protocol["regression"]["targeted_api"]
    results = []
    report["cases"] = results
    for dtype in spec["dtypes"]:
        for shape in spec["shapes"]:
            for scale in spec["scales"]:
                case = {"dtype": dtype, "shape": shape, "dim": 256,
                        "rows": shape[0] if len(shape) == 2 else shape[0] * shape[1] * shape[2],
                        "scale": scale, "normalized": scale == 0.0625}
                for offset in spec["pointer_mod16"]:
                    report["active_context"] = {**case, "pattern": "normal", "seed": 2026, "pointer_mod16": offset}
                    result = check_input(torch, np, op, case, "normal", 2026, offset, torch.cuda.Stream())
                    results.append({**case, **result})
    assert len(results) == spec["conditions"] == 16
    return results
