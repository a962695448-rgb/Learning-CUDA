"""Post-run read-only binary diagnostics and bounded text-transfer packaging."""
import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
pid = int(sys.argv[2])
assert not Path(f"/proc/{pid}").exists(), "Wait for the measured worker to finish"
allowed = sorted(os.sched_getaffinity(0))
os.sched_setaffinity(0, {allowed[-1]})
summary = json.loads((root / "results-remote/summary.json").read_text())
assert summary["status"] in ("PASS", "FAIL")
receipt = json.loads((root / "FINAL_EXPORT_RECEIPT.txt").read_text())
encoded = (root / "FINAL_EXPORT.base64.txt").read_bytes()
assert len(encoded) == receipt["base64_characters"]
compressed = base64.b64decode(encoded, validate=True)
assert hashlib.sha256(compressed).hexdigest() == receipt["gzip_sha256"]


def write_once(path, value):
    if path.exists():
        assert path.read_bytes() == value, path
    else:
        with path.open("xb") as stream:
            stream.write(value)


parts = []
for index, start in enumerate(range(0, len(encoded), 2000000)):
    value = encoded[start:start + 2000000]
    name = f"TRANSFER_{index:02d}.txt"
    write_once(root / name, value)
    parts.append({"name": name, "characters": len(value), "sha256": hashlib.sha256(value).hexdigest()})
transfer = dict(receipt, original_parts=receipt["parts"], parts=parts)
write_once(root / "TRANSFER_RECEIPT.txt", (json.dumps(transfer, indent=2) + "\n").encode())

diagnostics = {}
for label, folder, module in (("control", "control_a", "wi_control_a"), ("candidate", "candidate", "wi_candidate")):
    binary = root / "build" / folder / (module + ".so")
    if not binary.exists():
        diagnostics[label] = {"status": "NO_BINARY"}
        continue
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    assert digest == summary["binaries"][label]
    try:
        result = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(binary)], capture_output=True, text=True, timeout=120)
        diagnostics[label] = {"binary_sha256": digest, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except Exception as error:
        diagnostics[label] = {"binary_sha256": digest, "error": str(error)}
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == digest
value = (json.dumps(diagnostics, indent=2) + "\n").encode()
write_once(root / "RESOURCE_EXPORT.txt", value)
resource_receipt = {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
write_once(root / "RESOURCE_RECEIPT.txt", (json.dumps(resource_receipt) + "\n").encode())
print(json.dumps({"status": summary["status"], "decision": summary.get("candidate_decision"), "transfer_parts": len(parts), "characters": len(encoded), "resource": resource_receipt}))
