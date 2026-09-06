"""Freeze execution files once, before compilation or GPU timing."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
names = ["protocol.json", "source_manifest.json", "helper_provenance.json", "run_experiment.py",
    "run_suite.py", "measurement_helpers.py", "analyze_runs.py", "prepare_sources.py",
    "prepare_helpers.py", "freeze_manifest.py", "adapter.patch"]
names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "sources").iterdir()) if p.is_file()]
files = {}
for name in names:
    data = (ROOT / name).read_bytes()
    if name.endswith(".py"):
        ast.parse(data, filename=name)
    files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
target = ROOT / "run_manifest.json"
assert not target.exists(), "preserve existing frozen inputs"
target.write_text(json.dumps({"files": files}, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"files": len(files), "protocol_sha256": files["protocol.json"]["sha256"],
                  "manifest_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))
