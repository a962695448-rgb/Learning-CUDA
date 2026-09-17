"""Measure fixed module/output-buffer crossovers in one fresh process."""
import argparse
import functools
import gc
import hashlib
import importlib.util
import itertools
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
protocol = json.loads((ROOT / "PROTOCOL.json").read_text())
parser = argparse.ArgumentParser()
parser.add_argument("--process", type=int, required=True)
args = parser.parse_args()
affinity = sorted(os.sched_getaffinity(0))
os.sched_setaffinity(0, {affinity[0]})
import numpy as np
import torch
torch.set_num_threads(1)
sys.path.insert(0, str(ROOT / "helpers"))
from verify_execution_context import compare, reference
from verify_paired_hadamard import transform_reference

names = {"control_a": "hd_diag_control_a", "control_b": "hd_diag_control_b", "candidate": "hd_diag_candidate"}
modules = {}
for label in protocol["load_orders"][args.process]:
    name = names[label]
    path = ROOT / "build" / label / (name + ".so")
    spec = importlib.util.spec_from_file_location(name, path)
    op = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(op)
    modules[label] = op

frequency_path = Path(f"/sys/devices/system/cpu/cpu{affinity[0]}/cpufreq/scaling_cur_freq")


def frequency():
    try:
        return int(frequency_path.read_text())
    except (OSError, ValueError):
        return None


def sample(fn):
    torch.cuda.synchronize()
    freq_before = frequency()
    before = resource.getrusage(resource.RUSAGE_SELF)
    gc_before = [generation["collections"] for generation in gc.get_stats()]
    cpu_start = time.process_time_ns()
    thread_start = time.thread_time_ns()
    start = time.perf_counter_ns()
    for _ in range(protocol["calls_per_sample"]):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter_ns()
    thread_end = time.thread_time_ns()
    cpu_end = time.process_time_ns()
    after = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "start_ns": start, "end_ns": end,
        "wall_us": (end - start) / 1000 / protocol["calls_per_sample"],
        "process_cpu_us": (cpu_end - cpu_start) / 1000 / protocol["calls_per_sample"],
        "thread_cpu_us": (thread_end - thread_start) / 1000 / protocol["calls_per_sample"],
        "gc_collection_deltas": [generation["collections"] - count for generation, count in zip(gc.get_stats(), gc_before)],
        "voluntary_switches": after.ru_nvcsw - before.ru_nvcsw,
        "involuntary_switches": after.ru_nivcsw - before.ru_nivcsw,
        "minor_faults": after.ru_minflt - before.ru_minflt,
        "major_faults": after.ru_majflt - before.ru_majflt,
        "cpu_khz_before": freq_before, "cpu_khz_after": frequency(),
    }


report = {
    "status": "RUNNING", "process": args.process,
    "load_order": protocol["load_orders"][args.process],
    "cpu_affinity": sorted(os.sched_getaffinity(0)), "records": [], "cases": [],
    "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
    "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
    "binaries": {label: hashlib.sha256(Path(op.__file__).read_bytes()).hexdigest() for label, op in modules.items()},
}
output_path = ROOT / "results-remote" / f"process-{args.process}.json"


def save():
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output_path)


save()
permutations = list(itertools.permutations(("control_a", "control_b", "candidate")))
design = [(order, rotation) for rotation in range(3) for order in permutations]
assert len(design) == protocol["groups"]
for case_index, case in enumerate(protocol["cases"]):
    dtype = getattr(torch, case["dtype"])
    rng = torch.Generator().manual_seed(protocol["seed"] + case_index)
    cpu = torch.randn((case["rows"], case["dim"]), generator=rng).to(dtype)
    x = cpu.cuda()
    original_version = x._version
    transformed = transform_reference(torch, np, cpu, protocol["scale"])
    expected = (transformed,) if case["method"] == "hadamard" else reference(torch, np, transformed, {"method": "quantize_int4"})
    pools = [tuple(torch.empty_like(v, device="cuda") for v in expected) for _ in range(3)]
    options = dict(scale=protocol["scale"], block_threads=protocol["threads"], row_layout=case["row_layout"])
    out = {label: [functools.partial(getattr(modules[label], case["method"] + "_out"), x, *pool, **options) for pool in pools] for label in names}
    allocating = {label: functools.partial(getattr(modules[label], case["method"]), x, **options) for label in names}
    for label in names:
        compare(torch, allocating[label](), expected)
        for index in range(3):
            assert out[label][index]() is None
            compare(torch, pools[index], expected)
        for fn in (allocating[label], *out[label]):
            for _ in range(protocol["warmup_calls_per_function"]):
                fn()
    torch.cuda.synchronize()
    pool_info = [[{"shape": list(t.shape), "dtype": str(t.dtype), "gpu_alignment_mod4096": t.data_ptr() % 4096} for t in pool] for pool in pools]
    for group, (order, rotation) in enumerate(design):
        conditions = protocol["conditions"]
        offset = (group + args.process) % len(conditions)
        for condition in conditions[offset:] + conditions[:offset]:
            for position, label in enumerate(order):
                pool_index = None
                if condition.startswith("shared_"):
                    pool_index = int(condition.rsplit("_", 1)[-1])
                elif condition == "rotating_separate":
                    pool_index = (("control_a", "control_b", "candidate").index(label) + rotation) % 3
                fn = allocating[label] if pool_index is None else out[label][pool_index]
                result = sample(fn)
                report["records"].append(dict(result, case=case["id"], group=group,
                    condition=condition, module=label, measurement_order=list(order), position=position,
                    pool=pool_index, pool_rotation=rotation))
    for label in names:
        compare(torch, allocating[label](), expected)
        for index in range(3):
            out[label][index]()
            compare(torch, pools[index], expected)
    compare(torch, x, (cpu,))
    assert x._version == original_version
    report["cases"].append(dict(case, correctness="PASS", pools=pool_info))
    save()
    print(args.process, case["id"], "complete", flush=True)
report["status"] = "PASS"
save()
