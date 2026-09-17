"""Compare finite inputs with overflow/cancellation against the old packed path."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "candidate/cuda/03_hadamard_tc/a962695448-rgb/scripts"))
from verify_execution_context import compare
from verify_paired_hadamard import guarded_outputs
from verify_quantize_packed import pattern


def module(version):
    name = "hadamard_transform_pairs_" + version
    path = root / "build" / version / (name + ".so")
    spec = importlib.util.spec_from_file_location(name, path)
    op = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(op)
    return op


control, candidate = module("control"), module("candidate")
torch.set_num_threads(1)
records = []
for dtype in (torch.float16, torch.bfloat16):
    for dim in (2, 4, 8, 16):
        for name in ("zero", "finite", "ties", "extreme", "subnormal", "all_max", "all_min", "cancellation"):
            shape = (1, 1, 257, dim) if name in ("extreme", "all_max", "all_min", "cancellation") else (257, dim)
            if name in ("all_max", "all_min"):
                sign = 1 if name == "all_max" else -1
                cpu = torch.full(shape, sign * torch.finfo(dtype).max, dtype=dtype)
            elif name == "cancellation":
                cpu = torch.full(shape, torch.finfo(dtype).max, dtype=dtype)
                cpu[..., dim // 2:] *= -1
            else:
                cpu = pattern(torch, shape, dtype, name)
            assert torch.all(torch.isfinite(cpu.float()))
            for scale in (1., 1 / math.sqrt(dim)):
                for offset in (0, 1):
                    pool = torch.full((cpu.numel() + offset + 1,), 37, dtype=dtype, device="cuda")
                    x = pool[offset:offset + cpu.numel()].reshape(shape)
                    x.copy_(cpu)
                    input_version = x._version
                    for threads in (128, 256):
                        print("CASE", dtype, shape, name, scale, offset, threads, flush=True)
                        options = dict(scale=scale, block_threads=threads, row_layout="packed")
                        details = {}
                        for method in ("hadamard", "hadamard_int4"):
                            original = getattr(control, method)(x, **options)
                            expected = original if isinstance(original, tuple) else (original,)
                            expected = tuple(t.cpu() for t in expected)
                            compare(torch, getattr(candidate, method)(x, **options), expected)
                            buffers, guards = guarded_outputs(torch, x, method)
                            versions = [b._version for b in buffers]
                            assert getattr(candidate, method + "_out")(x, *buffers, **options) is None
                            compare(torch, buffers, expected)
                            assert [b._version for b in buffers] == [v + 1 for v in versions]
                            for guard in guards:
                                assert torch.all(guard[:9] == 37) and torch.all(guard[-8:] == 37)
                            details[method] = {
                                "bitwise_equal_to_old_packed": True,
                                "nonfinite_output_elements": sum(int((~torch.isfinite(t.float())).sum()) for t in expected),
                            }
                        compare(torch, x, (cpu,))
                        assert x._version == input_version
                        if offset:
                            assert pool[0].item() == 37
                        assert pool[-1].item() == 37
                        records.append(dict(dtype=str(dtype), shape=shape, pattern=name,
                                            scale=scale, offset=offset, threads=threads, methods=details))
report = {"status": "PASS", "records": records,
          "scope": "Finite inputs including overflow and cancellation. Bitwise compatibility with old packed results, not a claim that INT4 values derived from nonfinite intermediates are mathematically meaningful."}
with (root / "results-remote/overflow-compatibility.json").open("x") as output:
    json.dump(report, output, indent=2)
    output.write("\n")
print("OVERFLOW_COMPATIBILITY_PASS", len(records))
