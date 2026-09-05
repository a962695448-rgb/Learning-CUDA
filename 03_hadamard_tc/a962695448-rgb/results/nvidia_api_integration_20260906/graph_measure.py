import math
import statistics

def tensors(value):
    return value if isinstance(value, tuple) else (value,)

def exact(torch, actual, expected, label):
    a, b = tensors(actual), tensors(expected)
    if len(a) != len(b) or any(x.dtype != y.dtype or x.shape != y.shape or
        not torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(a, b)):
        raise RuntimeError("bitwise mismatch: " + label)

def measure_graph(torch, functions, run_index):
    graphs, outputs, expected = {}, {}, {}
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for name, function in functions.items():
            for _ in range(25):
                function()
            expected[name] = function()
    capture_stream.synchronize()
    for name, function in functions.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            outputs[name] = [function() for _ in range(64)]
        graphs[name] = graph
    torch.cuda.synchronize()
    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()
    for name, values in outputs.items():
        for component in range(len(tensors(values[0]))):
            if len({tensors(value)[component].data_ptr() for value in values}) != 64:
                raise RuntimeError("captured outputs do not have 64 independent buffers")
        for value in values:
            exact(torch, value, expected[name], "captured " + name)
    for graph in graphs.values():
        for _ in range(5):
            graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in functions}
    intervals = {name: [] for name in functions}
    orders, names = [], list(functions)
    for group in range(5):
        offset = (group + run_index - 1) % len(names)
        order = names[offset:] + names[:offset]
        orders.append(order)
        for name in order:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(20):
                graphs[name].replay()
            end.record()
            end.synchronize()
            ms = begin.elapsed_time(end)
            if not math.isfinite(ms) or ms <= 0:
                raise RuntimeError("invalid CUDA event timing")
            intervals[name].append(ms)
            samples[name].append(ms * 1000 / (20 * 64))
    for name, values in outputs.items():
        for value in values:
            exact(torch, value, expected[name], "after timing " + name)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    result = {"samples_us": samples, "raw_event_intervals_ms": intervals, "median_us": medians,
              "baseline_over_candidate": medians["baseline128"] / medians["candidate256"],
              "candidate_time_reduction_percent": 100 * (1 - medians["candidate256"] / medians["baseline128"]),
              "group_order": orders, "captured_outputs_per_graph": 64, "independent_output_buffers": True,
              "groups": 5, "replays_per_group": 20, "api_warmup_calls": 25,
              "graph_warmup_replays": 5, "captured_outputs_bitwise_equal_eager_before_and_after": True}
    if "dao" in medians:
        result["dao_over_candidate"] = medians["dao"] / medians["candidate256"]
        result["dao_over_baseline"] = medians["dao"] / medians["baseline128"]
    return result
