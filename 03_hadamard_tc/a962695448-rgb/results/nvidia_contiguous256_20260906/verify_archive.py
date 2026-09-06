"""Verify public evidence bytes; no GPU or third-party package required."""
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parent
manifest = json.loads((root / "archive_manifest.json").read_text(encoding="utf-8"))
for name, item in manifest["files"].items():
    path = (root / name).resolve()
    assert path.is_relative_to(root.resolve()), name
    data = path.read_bytes()
    assert len(data) == item["size"] and hashlib.sha256(data).hexdigest() == item["sha256"], name
print(json.dumps({"status": "PASS", "files": len(manifest["files"]), "bytes": sum(v["size"] for v in manifest["files"].values())}))
