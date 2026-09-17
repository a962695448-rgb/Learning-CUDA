"""Rebuild one CUDA trial from a verified baseline and its recorded overlay."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--baseline", type=Path, required=True)
parser.add_argument("--work", type=Path, required=True)
parser.add_argument("--variant", choices=("initial", "isolated"), default="isolated")
args = parser.parse_args()
evidence = Path(__file__).resolve().parent
if not (evidence / "cuda-base-tree.json").exists():
    evidence = evidence / args.variant
args.work.mkdir(exist_ok=False)
control = args.work / "control/cuda"
candidate = args.work / "candidate/cuda"
entries = json.loads((evidence / "cuda-base-tree.json").read_text())["tree"]
for entry in entries:
    if entry["type"] != "blob":
        continue
    name = PurePosixPath(entry["path"])
    assert not name.is_absolute() and ".." not in name.parts
    data = (args.baseline / entry["path"]).read_bytes()
    blob = b"blob " + str(len(data)).encode() + b"\0" + data
    assert hashlib.sha1(blob).hexdigest() == entry["sha"], entry["path"]
    path = control / entry["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(int(entry["mode"], 8) & 0o777)
shutil.copytree(control, candidate)
project = candidate / "03_hadamard_tc/a962695448-rgb"
for path in (evidence / "source").rglob("*"):
    if path.is_file():
        assert not path.is_symlink()
        target = project / path.relative_to(evidence / "source")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
for name in ("PROTOCOL.json", "run_gpu.py", "cuda-base-tree.json"):
    shutil.copy2(evidence / name, args.work / name)
shutil.copytree(evidence / "tools", args.work / "tools")
# The original manifest includes archival metadata and, in v1, NineToothed.
# Record this explicit CUDA-only reconstruction without copying that manifest.
manifest = {str(p.relative_to(args.work)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.work.rglob("*")) if p.is_file()}
(args.work / "INPUT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("VERIFIED CUDA-ONLY REPLAY", len(manifest), "files; set PH_CUDA_ONLY=1 for initial")
