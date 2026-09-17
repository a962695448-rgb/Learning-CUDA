"""Reproduce descriptive audits without changing the old adoption decisions."""
import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--initial-results", type=Path, required=True)
parser.add_argument("--isolated-results", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(exist_ok=False)
paths = {"initial": args.initial_results, "isolated": args.isolated_results}
trials = {label: json.loads((path / "benchmark.json").read_text()) for label, path in paths.items()}
fields = ("dtype", "kind", "rows", "dim", "method")
failed = {tuple(row[key] for key in fields) for trial in trials.values() for row in trial["host_records"] if not row["gate_passed"]}
audit = {"scope": "Descriptive only; earlier REJECT decisions remain unchanged", "cases": []}
for identity in sorted(failed):
    case = {"identity": dict(zip(fields, identity)), "observations": []}
    for label, trial in trials.items():
        for row in trial["host_records"]:
            if tuple(row[key] for key in fields) != identity:
                continue
            old, new = row["host_us"]["control"], row["host_us"]["candidate"]
            differences = [b - a for a, b in zip(old, new)]
            ratios = [a / b for a, b in zip(old, new)]
            orders = {name: [value for i, value in enumerate(differences)
                            if ("control" if (i + row["round"]) % 2 == 0 else "candidate") == name]
                      for name in ("control", "candidate")}
            case["observations"].append({"trial": label, "round": row["round"], "original_speedup": row["speedup"],
                "candidate_slower_samples": sum(value > 0 for value in differences), "samples": len(differences),
                "median_delta_us": statistics.median(differences), "min_max_delta_us": [min(differences), max(differences)],
                "paired_ratio_median": statistics.median(ratios),
                "order_delta_medians_us": {name: statistics.median(values) for name, values in orders.items()},
                "original_gate_passed": row["gate_passed"]})
    audit["cases"].append(case)
(args.output / "prior-host-audit.json").write_text(json.dumps(audit, indent=2) + "\n")


def functions(path):
    result = {}
    for block in re.split(r"(?=^[0-9a-f]+ <.+>:\n)", path.read_text(), flags=re.M):
        header = re.match(r"^[0-9a-f]+ <(.+)>:\n", block)
        if not header:
            continue
        instructions = [match.group(1) for line in block.splitlines()
                        if (match := re.match(r"^\s*[0-9a-f]+:\s+(.+)$", line))]
        result[header.group(1)] = {"instructions": instructions, "opcodes": [text.split()[0] for text in instructions],
            "calls": [re.sub(r"^\S+\s+[0-9a-f]+\s+", "", text) for text in instructions if text.split()[0] == "call"]}
    return result


assembly_paths = [args.isolated_results / f"host-assembly-{label}.txt" for label in ("control", "candidate")]
old, new = map(functions, assembly_paths)
rows = []
for name in old.keys() & new.keys():
    if "quantized" not in name and "dispatch<true, true>" not in name:
        continue
    a, b = old[name], new[name]
    rows.append({"symbol": name, "control_instructions": len(a["instructions"]), "candidate_instructions": len(b["instructions"]),
                 "same_opcode_sequence": a["opcodes"] == b["opcodes"], "same_direct_call_names": a["calls"] == b["calls"],
                 "control_calls": a["calls"], "candidate_calls": b["calls"]})
report = {"scope": "Static selected assembly, including error/cold paths. Matching opcodes/calls do not prove equal operands, callees, behavior or cost.",
          "functions": sorted(rows, key=lambda row: row["symbol"])}
(args.output / "assembly-audit.json").write_text(json.dumps(report, indent=2) + "\n")
inputs = [path / "benchmark.json" for path in paths.values()] + assembly_paths
(args.output / "INPUT_HASHES.json").write_text(json.dumps({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}, indent=2) + "\n")
print("AUDITED", len(audit["cases"]), "host points and", len(rows), "assembly functions")
