"""Freeze the reviewed production working bytes without changing its index/tree."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1] / "Learning-CUDA"
PREFIX = "03_hadamard_tc/a962695448-rgb"
PRODUCTION = REPO / PREFIX


def git(*args):
    return subprocess.check_output(["git", "-C", str(REPO), *args], stderr=subprocess.PIPE)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    destination = ROOT / "project"
    assert not destination.exists(), "preserve existing frozen snapshot"
    head = git("rev-parse", "HEAD").decode().strip()
    assert head == "217c30ff5e78842cd5809de6bf78ee8a7f04fc54", head
    names = ["Makefile"]
    for directory in ("include", "src", "scripts", "tests"):
        names += [p.relative_to(PRODUCTION).as_posix() for p in sorted((PRODUCTION / directory).rglob("*"))
                  if p.is_file() and p.suffix in (".h", ".hpp", ".cuh", ".cpp", ".cu", ".py", ".sh", ".md")]
    blobs = {name: (PRODUCTION / name).read_bytes() for name in names}
    paths = [PREFIX + "/" + p for p in ("Makefile", "include", "src", "scripts", "tests")]
    diff = git("diff", "--binary", "HEAD", "--", *paths)
    manifest = {"base_commit": head, "source_state": "reviewed working tree, not committed at snapshot",
        "raw_working_bytes_preserved": True, "diff_note": "Tracked diff is Git canonical text; untracked source files appended as LF review diff. Raw snapshot SHA is authoritative; LF hashes distinguish line-ending changes.",
        "files": {}}
    for name, data in blobs.items():
        path = PREFIX + "/" + name
        try:
            original = git("show", head + ":" + path)
        except subprocess.CalledProcessError:
            original = None
            normalized = data.decode("utf-8").replace("\r\n", "\n")
            diff += "".join(difflib.unified_diff([], normalized.splitlines(True),
                fromfile="/dev/null", tofile="b/" + path)).encode("utf-8")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest["files"]["project/" + name] = {"production_path": path, "sha256": sha(data),
            "lf_sha256": sha(data.replace(b"\r\n", b"\n")), "size": len(data),
            "base_blob_sha256": sha(original) if original is not None else None,
            "changed_from_base_bytes": original != data}
    assert head == git("rev-parse", "HEAD").decode().strip(), "HEAD changed while snapshotting"
    assert all((PRODUCTION / name).read_bytes() == data for name, data in blobs.items()), "production sources changed while snapshotting"
    (ROOT / "working_diff.patch").write_bytes(diff)
    (ROOT / "working_status.txt").write_bytes(git("status", "--short", "--", *paths))
    manifest["working_diff_sha256"] = sha(diff)
    (ROOT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"base_commit": head, "source_files": len(blobs), "bytes": sum(map(len, blobs.values())),
        "source_manifest_sha256": sha((ROOT / "source_manifest.json").read_bytes())}))


if __name__ == "__main__":
    main()
