"""Independently audit the frozen benchmark matrix, medians and decision."""
import argparse
import json
import math
import statistics
from pathlib import Path


def configurations(protocol):
    for kind in ("packed_hadamard", "packed_fused"):
        for rows in protocol["target_rows"]:
            for dim in protocol["target_dims"]:
                yield kind, rows, dim, True
        for rows in protocol["control_rows"]:
            for dim in protocol["target_dims"]:
                yield kind, rows, dim, False
        yield kind, 65536, 1, False
    for kind in ("original_hadamard", "original_fused"):
        for dim in (8, 256):
            yield kind, 65536, dim, False
    for dim in (2, 16):
        yield "packed_quant", 65536, dim, False
    yield "original_quant", 65536, 8, False


def key(record):
    return tuple(record[x] for x in ("round", "dtype", "kind", "rows", "dim", "threads"))


def audit(data, protocol):
    assert data["protocol"] == protocol
    expected = {}
    for round_id in range(protocol["rounds"]):
        for dtype in ("torch.float16", "torch.bfloat16"):
            for kind, rows, dim, target in configurations(protocol):
                for threads in protocol["threads"]:
                    expected[(round_id, dtype, kind, rows, dim, threads)] = target
    assert len(data["records"]) == len(expected)
    device = {key(record): record for record in data["records"]}
    assert device.keys() == expected.keys()
    limit = 1 / (1 + protocol["max_device_or_host_case_regression_percent"] / 100)
    failures = []

    def check_times(record, field):
        times = record[field]
        assert set(times) == {"control", "candidate"}
        for samples in times.values():
            assert len(samples) == protocol["groups"]
            assert all(math.isfinite(value) and value > 0 for value in samples)
        ratio = statistics.median(times["control"]) / statistics.median(times["candidate"])
        assert math.isclose(ratio, record["speedup"], rel_tol=1e-12)
        assert record["gate_passed"] == (ratio >= limit)
        if ratio < limit:
            failures.append({"key": key(record), "metric": field,
                             "method": record.get("method"), "speedup": ratio})
        return ratio

    for identity, record in device.items():
        assert record["target"] == expected[identity]
        check_times(record, "device_us")
    host_expected = {(identity, method) for identity in expected if identity[-1] == 256
                     for method in ("allocating", "out")}
    host = {(key(record), record["method"]): record for record in data["host_records"]}
    assert len(data["host_records"]) == len(host_expected)
    assert host.keys() == host_expected
    for (identity, method), record in host.items():
        assert record["target"] == expected[identity]
        check_times(record, "host_us")

    group_keys = {(round_id, dtype, threads, kind)
                  for round_id in range(protocol["rounds"])
                  for dtype in ("torch.float16", "torch.bfloat16")
                  for threads in protocol["threads"]
                  for kind in ("packed_hadamard", "packed_fused")}
    groups = {(r["round"], r["dtype"], r["threads"], r["kind"]): r for r in data["groups"]}
    assert len(data["groups"]) == len(group_keys) and groups.keys() == group_keys
    for (round_id, dtype, threads, kind), group in groups.items():
        target = [r["speedup"] for identity, r in device.items()
                  if identity[0] == round_id and identity[1] == dtype
                  and identity[-1] == threads and identity[2] == kind and r["target"]]
        mean = math.exp(statistics.mean(math.log(value) for value in target))
        assert math.isclose(mean, group["target_geomean_speedup"], rel_tol=1e-12)
        passed = mean >= protocol["minimum_each_kind_dtype_thread_round_target_geomean"]
        assert group["gate_passed"] == passed
        if not passed:
            failures.append({"group": [round_id, dtype, threads, kind], "geomean": mean})

    aa_expected = {(round_id, dtype, kind, rows, dim)
                   for round_id in range(protocol["rounds"])
                   for dtype, kind, rows, dim in [
                       ("torch.float16", "packed_hadamard", 257, 2),
                       ("torch.float16", "packed_fused", 16387, 4),
                       ("torch.bfloat16", "original_hadamard", 65536, 256)]}
    aa = {tuple(r[k] for k in ("round", "dtype", "kind", "rows", "dim")): r for r in data["calibration"]}
    assert len(data["calibration"]) == len(aa_expected) and aa.keys() == aa_expected
    for identity, record in aa.items():
        assert set(record["host_us"]) == {"a", "b"}
        for times in record["host_us"].values():
            assert len(times) == protocol["groups"]
            assert all(math.isfinite(t) and t > 0 for t in times)
        ratio = statistics.median(record["host_us"]["a"]) / statistics.median(record["host_us"]["b"])
        assert math.isclose(ratio, record["same_function_ratio"], rel_tol=1e-12)
        passed = limit <= ratio <= 1 / limit
        assert record["gate_passed"] == passed
        if not passed:
            failures.append({"aa_case": identity, "ratio": ratio})

    decision = "REJECT" if failures else "ACCEPT"
    assert data["status"] == decision
    return {"status": "AUDIT_PASS", "decision": decision, "device_records": len(device),
            "host_records": len(host), "groups": list(groups.values()),
            "aa_cases": len(aa), "failed_gates": failures}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).resolve().parents[1] / "PROTOCOL.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.results.read_text()), json.loads(args.protocol.read_text())["cuda"])
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(result["decision"], result["device_records"], result["host_records"], "failed gates:", len(result["failed_gates"]))
