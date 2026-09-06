"""Freeze snapshot, protocol and execution code after review, before GPU use."""
import ast
import hashlib
import json
from pathlib import Path
import sys

if not __debug__ or sys.flags.optimize != 0:
    raise SystemExit("Python assertions disabled; refusing freeze")

ROOT = Path(__file__).resolve().parent
source = json.loads((ROOT / "source_manifest.json").read_text())
names = list(source["files"]) + ["source_manifest.json", "working_diff.patch", "working_status.txt",
    "protocol.json", "snapshot_sources.py", "measurement_helpers.py", "helper_provenance.json",
    "integration_checks.py", "run_directed.py", "run_regressions.py", "run_holdout.py",
    "run_holdout_suite.py", "analyze_holdout.py", "freeze_manifest.py"]
files = {}
for name in names:
    data = (ROOT / name).read_bytes()
    if name.endswith(".py"):
        ast.parse(data, filename=name)
    if name in source["files"]:
        assert hashlib.sha256(data).hexdigest() == source["files"][name]["sha256"], name
    files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
target = ROOT / "run_manifest.json"
assert not target.exists(), "preserve existing frozen execution manifest"
target.write_text(json.dumps({"files": files}, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"files": len(files), "protocol_sha256": files["protocol.json"]["sha256"],
    "manifest_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))
