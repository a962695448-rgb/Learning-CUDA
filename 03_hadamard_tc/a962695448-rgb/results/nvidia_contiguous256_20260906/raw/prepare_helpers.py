"""Reuse the verified measurement primitives and extend output checks to tuples."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT.parent / "nvidia-small-batch-20260906/run_experiment.py"
data = source.read_bytes()
text = data.decode("utf-8")
tree = ast.parse(text)
names = ("sha", "utc", "snapshot", "tensors", "exact", "cpu_checks", "timed_result", "measure_graph")
body = "import hashlib\nimport math\nfrom pathlib import Path\nimport statistics\nimport subprocess\nimport time\n\n"
for name in names:
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    body += ast.get_source_segment(text, node) + "\n\n"
body = body.replace('graph_settings = timing["graph"]',
    'graph_settings = {"captured_calls": timing["captured_calls"], "replays_per_group": timing["replays_per_group"], "warmup_replays": timing["graph_warmup_replays"]}')
body = body.replace('addresses = [value.data_ptr() for value in values]',
    'addresses = [tensor.data_ptr() for value in values for tensor in tensors(value)]')
body = body.replace('len(set(addresses)) != graph_settings["captured_calls"]:',
    'len(set(addresses)) != graph_settings["captured_calls"] * len(tensors(values[0])):')
body += '''def orders_for(names, run_index, case_index, groups):
    names = list(names)
    return [names[(run_index - 1 + case_index + group) % len(names):] +
            names[:(run_index - 1 + case_index + group) % len(names)] for group in range(groups)]
'''
target = ROOT / "measurement_helpers.py"
assert not target.exists()
target.write_text(body, encoding="utf-8", newline="\n")
(ROOT / "helper_provenance.json").write_text(json.dumps({"source_script_sha256": hashlib.sha256(data).hexdigest(),
    "source_functions": names, "changes": ["generic tuple component pointer checks for fused outputs", "protocol key mapping", "deterministic cyclic method order"],
    "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"helper_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))
