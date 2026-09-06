"""Create a fixed-source adapter with explicit original/contiguous256 paths."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1] / "Learning-CUDA"
COMMIT = "9f5fdc363b4149d4a211701f24ab0548084ca3e5"
PREFIX = "03_hadamard_tc/a962695448-rgb/"
SOURCES = {"include/kernels.cuh": "kernels.cuh", "src/torch_binding.cu": "torch_binding_original.cu",
           "scripts/compare_reference.py": "compare_reference.py", "scripts/build_torch_extension.py": "build_torch_extension.py"}


def main():
    directory = ROOT / "sources"
    manifest = {"source_commit": COMMIT, "files": {}}
    for path, name in SOURCES.items():
        data = subprocess.check_output(["git", "-C", str(REPO), "show", COMMIT + ":" + PREFIX + path])
        target = directory / name
        assert not target.exists()
        target.write_bytes(data)
        manifest["files"]["sources/" + name] = {"git_path": PREFIX + path,
            "sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "modified": False}
    original = (directory / "torch_binding_original.cu").read_text(encoding="utf-8")
    candidate = original.replace('#include "kernels.cuh"', '#include "kernels.cuh"\n#include "contiguous256.cuh"\n#include <string>')
    candidate = candidate.replace('int block_threads)', 'int block_threads, const std::string& layout)')
    candidate = candidate.replace('block_threads == 128 || block_threads == 256, "block_threads must be 128 or 256"',
        'block_threads == 128, "this experiment fixes block_threads=128"')
    candidate = candidate.replace('    TORCH_CHECK(input.is_cuda()',
        '    TORCH_CHECK(layout == "original" || layout == "contiguous256", "unsupported layout");\n    TORCH_CHECK(input.is_cuda()', 1)
    candidate = candidate.replace('    const auto dim = input.size(-1);',
        '    const auto dim = input.size(-1);\n    TORCH_CHECK(layout == "original" || dim == 256, "contiguous256 requires N=256");')
    candidate = candidate.replace('block_threads);', 'block_threads, layout);')
    launch = '''    hadamard::warp_kernel<T, N, Transform, Quantize><<<blocks, block_threads, 0, stream>>>(
        source, destination, bytes, row_scales, rows, scale);'''
    replacement = '''    if constexpr (N == 256) {
        if (layout == "contiguous256") {
            hadamard::contiguous256_kernel<T, Transform, Quantize><<<blocks, block_threads, 0, stream>>>(
                source, destination, bytes, row_scales, rows, scale);
        } else {
            hadamard::warp_kernel<T, N, Transform, Quantize><<<blocks, block_threads, 0, stream>>>(
                source, destination, bytes, row_scales, rows, scale);
        }
    } else {
        hadamard::warp_kernel<T, N, Transform, Quantize><<<blocks, block_threads, 0, stream>>>(
            source, destination, bytes, row_scales, rows, scale);
    }'''
    assert launch in candidate
    candidate = candidate.replace(launch, replacement)
    candidate = candidate.replace('pybind11::arg("block_threads") = 128,',
        'pybind11::arg("block_threads") = 128, pybind11::arg("layout") = "original",')
    candidate = candidate.replace('block_threads=128 (default) or 256',
        'block_threads=128 fixed; layout=original (default) or contiguous256')
    (directory / "torch_binding_contiguous256.cu").write_text(candidate, encoding="utf-8", newline="\n")
    (ROOT / "adapter.patch").write_text("".join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True),
        fromfile="sources/torch_binding_original.cu", tofile="sources/torch_binding_contiguous256.cu")), encoding="utf-8", newline="\n")
    for name in ("contiguous256.cuh", "torch_binding_contiguous256.cu"):
        data = (directory / name).read_bytes()
        manifest["files"]["sources/" + name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "modified": True}
    (ROOT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_commit": COMMIT, "files": len(manifest["files"]), "production_kernel_unchanged": True}))


if __name__ == "__main__":
    main()
