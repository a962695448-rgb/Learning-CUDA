import hashlib
import math
from pathlib import Path
import statistics
import subprocess
import time

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
    graph_settings = {"captured_calls": timing["captured_calls"], "replays_per_group": timing["replays_per_group"], "warmup_replays": timing["graph_warmup_replays"]}
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
        addresses = [tensor.data_ptr() for value in values for tensor in tensors(value)]
        if len(set(addresses)) != graph_settings["captured_calls"] * len(tensors(values[0])):
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

def orders_for(names, run_index, case_index, groups):
    names = list(names)
    return [names[(run_index - 1 + case_index + group) % len(names):] +
            names[:(run_index - 1 + case_index + group) % len(names)] for group in range(groups)]
