"""Freeze a new validation revision; never overwrite the original manifest."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not __debug__ or sys.flags.optimize:
        raise RuntimeError("assertions required")
    protocol = json.loads((ROOT / "protocol_v2.json").read_text())
    original = json.loads((BASE / "protocol.json").read_text())
    for key in ("rows", "dims", "dtypes", "normalized", "expected_configurations", "block_threads", "mode", "layouts", "exclude_initial_rows", "patterns_and_seeds", "strict_abs_limit", "scale"):
        if protocol["holdout"][key] != original["holdout"][key]:
            raise RuntimeError("matrix/seed/reference-limit changed: " + key)
    if protocol["timing"] != original["timing"]:
        raise RuntimeError("timing changed")
    baseline = json.loads((BASE / "run_manifest.json").read_text())
    if sha(BASE / "run_manifest.json") != protocol["reused_regression"]["original_run_manifest_sha256"]:
        raise RuntimeError("original manifest changed")
    for name, item in baseline["files"].items():
        if sha(BASE / name) != item["sha256"]:
            raise RuntimeError("original source changed: " + name)
    names = ["protocol_v2.json", "prepare_protocol.py", "numeric_certificate.py", "certificate_notes.md",
             "checks_v2.py", "run_v2.py", "run_suite_v2.py", "analyze_v2.py", "freeze_v2.py", "README.md"]
    files = {}
    for name in names:
        data = (ROOT / name).read_bytes()
        if name.endswith(".py"):
            ast.parse(data, filename=name)
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    target = ROOT / "manifest_v2.json"
    if target.exists():
        raise RuntimeError("preserve existing revised manifest")
    manifest = {"validation_revision": 2, "validation_files": files,
                "original_manifest_sha256": sha(BASE / "run_manifest.json"),
                "original_source_manifest_sha256": sha(BASE / "source_manifest.json"),
                "original_v1_result_remains": "FAIL_BEFORE_TIMING"}
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation_files": len(files), "protocol_sha256": sha(ROOT / "protocol_v2.json"),
                      "manifest_v2_sha256": sha(target), "old_files_unchanged": len(baseline["files"])}))


if __name__ == "__main__":
    main()
