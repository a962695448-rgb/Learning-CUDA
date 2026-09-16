"""Restore the fixed control and experimental candidate without changing production code."""

import argparse
import hashlib
import io
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

from validation_common import ROOT, digest, verify_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Optional local codeload archive.")
    args = parser.parse_args()
    control = ROOT / "sources/cuda"
    candidate = ROOT / "candidate/cuda"
    if control.exists() or candidate.exists():
        raise FileExistsError("Use a fresh directory; source trees are not overwritten.")
    specification = next(
        item
        for item in json.loads((ROOT / "SOURCE_ARCHIVES.json").read_text())
        if item["label"] == "cuda"
    )
    if args.archive is not None:
        data = args.archive.read_bytes()
    else:
        url = (
            "https://codeload.github.com/"
            + specification["repository"]
            + "/tar.gz/"
            + specification["commit"]
        )
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != specification["archive_sha256"]:
        raise RuntimeError("The source archive checksum does not match.")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for entry in archive.getmembers():
            if entry.isdir():
                continue
            if not entry.isfile():
                raise RuntimeError("Unexpected archive member type.")
            parts = Path(entry.name).parts[1:]
            if not parts or any(part in ("..", "") for part in parts):
                raise RuntimeError("Unexpected archive path.")
            output = control.joinpath(*parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(archive.extractfile(entry).read())
            output.chmod(entry.mode & 0o777)
    hashes = verify_sources("cuda", control)
    shutil.copytree(control, candidate)
    binding = "03_hadamard_tc/a962695448-rgb/src/torch_binding.cu"
    shutil.copy2(ROOT / "candidate_torch_binding.cu", candidate / binding)
    proof = json.loads((ROOT / "checks/host-routing/result.json").read_text())
    for name, expected in hashes.items():
        wanted = proof["candidate_sha256"] if name == binding else expected
        if digest(candidate / name) != wanted:
            raise RuntimeError("Candidate source mismatch: " + name)
    print("Verified control and candidate:", len(hashes), "tracked files each.")


if __name__ == "__main__":
    main()
