"""Freeze local protocol and exact experiment input bytes before GPU work."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
names = ["protocol.json", "source_manifest.json", "run_experiment.py", "run_suite.py",
         "analyze_runs.py", "prepare_sources.py", "freeze_manifest.py", "thread_config.patch"]
names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "sources").glob("*")) if p.is_file()]
files = {}
for name in names:
    data = (ROOT / name).read_bytes()
    files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
target = ROOT / "run_manifest.json"
if target.exists():
    raise SystemExit("Existing frozen manifest is not overwritten.")
target.write_text(json.dumps({"files": files}, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"files": len(files), "protocol_sha256": files["protocol.json"]["sha256"],
                  "manifest_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))
