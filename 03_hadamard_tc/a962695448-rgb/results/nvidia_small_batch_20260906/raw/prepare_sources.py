"""Materialize fixed Git blobs and the isolated thread-configuration adapter."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1] / "Learning-CUDA"
COMMIT = "9f5fdc363b4149d4a211701f24ab0548084ca3e5"
PREFIX = "03_hadamard_tc/a962695448-rgb/"
FILES = {
    "include/kernels.cuh": "kernels.cuh",
    "src/torch_binding.cu": "torch_binding_original.cu",
    "scripts/compare_reference.py": "compare_reference.py",
    "scripts/build_torch_extension.py": "build_torch_extension.py",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    sources = ROOT / "sources"
    sources.mkdir(exist_ok=False)
    manifest = {"git_commit": COMMIT, "git_blob_bytes_preserved": True, "files": {}}
    for source, target in FILES.items():
        blob = subprocess.check_output(["git", "-C", str(REPO), "show", COMMIT + ":" + PREFIX + source])
        (sources / target).write_bytes(blob)
        manifest["files"]["sources/" + target] = {"git_path": PREFIX + source,
            "sha256": digest(blob), "size": len(blob), "modified": False}
    original = (sources / "torch_binding_original.cu").read_text(encoding="utf-8")
    old = 'block_threads == 128 || block_threads == 256, "block_threads must be 128 or 256"'
    new = ('block_threads == 32 || block_threads == 64 || block_threads == 128 || '
           'block_threads == 256, "block_threads must be 32, 64, 128 or 256"')
    assert original.count(old) == 1
    modified = original.replace(old, new).replace("block_threads=128 (default) or 256", 
        "block_threads=32, 64, 128 (default) or 256")
    data = modified.encode("utf-8")
    (sources / "torch_binding_thread_config.cu").write_bytes(data)
    manifest["files"]["sources/torch_binding_thread_config.cu"] = {
        "git_path": PREFIX + "src/torch_binding.cu", "sha256": digest(data),
        "size": len(data), "modified": True,
        "change": "Allow 32/64 as well as 128/256 in adapter validation and docstrings; default 128. Kernel bytes unchanged."}
    patch = "".join(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
        fromfile="sources/torch_binding_original.cu", tofile="sources/torch_binding_thread_config.cu"))
    (ROOT / "thread_config.patch").write_text(patch, encoding="utf-8", newline="\n")
    (ROOT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"commit": COMMIT, "files": len(manifest["files"]), "kernel_changed": False}))


if __name__ == "__main__":
    main()
