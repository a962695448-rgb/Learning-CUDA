"""Audit the crossover design and describe host effects without adopting a kernel."""
import argparse
import itertools
import json
import math
import statistics
from pathlib import Path


def analyze(root):
    protocol = json.loads((root / "PROTOCOL.json").read_text())
    labels = ("control_a", "control_b", "candidate")
    design = [(order, rotation) for rotation in range(3) for order in itertools.permutations(labels)]
    expected = {(case["id"], group, condition, label) for case in protocol["cases"]
                for group in range(protocol["groups"]) for condition in protocol["conditions"] for label in labels}
    comparisons = []
    for process in range(protocol["processes"]):
        data = json.loads((root / "results-remote" / f"process-{process}.json").read_text())
        assert data["status"] == "PASS" and data["process"] == process
        assert data["load_order"] == protocol["load_orders"][process]
        assert {c["id"] for c in data["cases"] if c["correctness"] == "PASS"} == {c["id"] for c in protocol["cases"]}
        records = {(r["case"], r["group"], r["condition"], r["module"]): r for r in data["records"]}
        assert len(data["records"]) == len(expected) and records.keys() == expected
        assert all(a["end_ns"] <= b["start_ns"] for a, b in zip(data["records"], data["records"][1:]))
        for (case, group, condition, module), record in records.items():
            order, rotation = design[group]
            assert record["measurement_order"] == list(order)
            assert record["position"] == order.index(module)
            assert record["pool_rotation"] == rotation
            pool = (int(condition.rsplit("_", 1)[-1]) if condition.startswith("shared_")
                    else (labels.index(module) + rotation) % 3 if condition == "rotating_separate" else None)
            assert record["pool"] == pool
            for field in ("wall_us", "process_cpu_us", "thread_cpu_us"):
                assert math.isfinite(record[field]) and record[field] > 0
            assert record["end_ns"] > record["start_ns"]
            assert math.isclose(record["wall_us"], (record["end_ns"] - record["start_ns"]) / 1000 / protocol["calls_per_sample"], rel_tol=1e-12)
        for case in protocol["cases"]:
            for condition in protocol["conditions"]:
                for left, right in (("control_a", "control_b"), ("control_a", "candidate"), ("control_b", "candidate")):
                    a = [records[(case["id"], group, condition, left)] for group in range(protocol["groups"])]
                    b = [records[(case["id"], group, condition, right)] for group in range(protocol["groups"])]
                    ratios = [x["wall_us"] / y["wall_us"] for x, y in zip(a, b)]
                    delta = [y["wall_us"] - x["wall_us"] for x, y in zip(a, b)]
                    spread = {}
                    for label, rows in ((left, a), (right, b)):
                        times = sorted(r["wall_us"] for r in rows)
                        q1, _, q3 = statistics.quantiles(times, n=4, method="inclusive")
                        spread[label] = {"median_us": statistics.median(times), "min_us": times[0], "max_us": times[-1],
                            "iqr_percent_of_median": 100 * (q3 - q1) / statistics.median(times),
                            "context_switches": sum(r["voluntary_switches"] + r["involuntary_switches"] for r in rows),
                            "gc_collections": sum(sum(r["gc_collection_deltas"]) for r in rows)}
                    comparisons.append({"process": process, "case": case["id"], "condition": condition,
                        "left": left, "right": right,
                        "ratio_of_separate_medians": statistics.median(r["wall_us"] for r in a) / statistics.median(r["wall_us"] for r in b),
                        "median_of_group_ratios": statistics.median(ratios),
                        "median_right_minus_left_us": statistics.median(delta),
                        "right_slower_groups": sum(v > 0 for v in delta),
                        "groups": len(delta), "spread": spread})
    findings = []
    limit = 1 + protocol["diagnostic_bounds"]["aa_percent"] / 100
    for case in protocol["cases"]:
        for condition in protocol["conditions"]:
            rows = [r for r in comparisons if r["case"] == case["id"] and r["condition"] == condition]
            aa = [r for r in rows if r["left"] == "control_a" and r["right"] == "control_b"]
            candidate = [r for r in rows if r["left"] == "control_a" and r["right"] == "candidate"]
            aa_ok = all(1 / limit <= r["median_of_group_ratios"] <= limit and
                        1 / limit <= r["ratio_of_separate_medians"] <= limit for r in aa)
            broad = any(v["iqr_percent_of_median"] > protocol["diagnostic_bounds"]["block_wall_spread_percent"]
                        for r in rows for v in r["spread"].values())
            persistent_slow = aa_ok and all(r["median_of_group_ratios"] < 1 / limit and
                                           r["ratio_of_separate_medians"] < 1 / limit for r in candidate)
            persistent_fast = aa_ok and all(r["median_of_group_ratios"] > limit and
                                           r["ratio_of_separate_medians"] > limit for r in candidate)
            findings.append({"case": case["id"], "condition": condition, "same_source_modules_within_5_percent": aa_ok,
                "broad_spread_flag": broad, "persistent_candidate_slower": persistent_slow,
                "persistent_candidate_faster": persistent_fast})
    return {"status": "AUDIT_PASS", "adoption_decision": "NOT_APPLICABLE_DIAGNOSTIC_ONLY",
            "comparisons": comparisons, "findings": findings,
            "limitations": "Repeated-block observations from three processes on one machine. Static thresholds and descriptive medians are not confidence intervals or causal proof. Earlier REJECT decisions remain unchanged."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(result["status"], len(result["comparisons"]), "comparisons")
