"""从已保存的包装器副本生成 128/256 线程显式实验入口，不修改生产文件。"""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT / "sources" / "torch_binding_original.cu"
original = source.read_text()
code = original
changes = [
    ("void validate(const at::Tensor& input, double scale)", "void validate(const at::Tensor& input, double scale, int threads)"),
    ("    const auto dim = input.size(-1);", "    TORCH_CHECK(threads == 128 || threads == 256, \"experiment requires 128 or 256 threads\");\n    const auto dim = input.size(-1);"),
    ("dim >= 1 && dim <= 256 && !(dim & (dim - 1))", "dim == 16 || dim == 64"),
    ("last dimension must be a power of two in [1,256]", "experiment requires dim 16 or 64"),
    ("(rows - 1) / 4 + 1", "(rows - 1) / (threads / 32) + 1"),
    ("float scale, cudaStream_t stream)", "float scale, cudaStream_t stream, int threads)"),
    ("<<<blocks, 128, 0, stream>>>", "<<<blocks, threads, 0, stream>>>"),
    ("scales, scale, stream); break", "scales, scale, stream, threads); break"),
    ("        DIM_CASE(1); DIM_CASE(2); DIM_CASE(4); DIM_CASE(8); DIM_CASE(16);\n        DIM_CASE(32); DIM_CASE(64); DIM_CASE(128); DIM_CASE(256);", "        DIM_CASE(16); DIM_CASE(64);"),
    ("              at::Tensor& scales, double scale)", "              at::Tensor& scales, double scale, int threads)"),
    ("static_cast<float>(scale), stream);", "static_cast<float>(scale), stream, threads);"),
    ("at::Tensor transform(const at::Tensor& input, double scale)", "at::Tensor transform(const at::Tensor& input, double scale, int threads)"),
    ("quantized(const at::Tensor& input, double scale)", "quantized(const at::Tensor& input, double scale, int threads)"),
    ("validate(input, scale);", "validate(input, scale, threads);"),
    ("output, packed, scales, scale);", "output, packed, scales, scale, threads);"),
    ("quantize_only(const at::Tensor& input)", "quantize_only(const at::Tensor& input, int threads)"),
    ("quantized<false>(input, 1.0);", "quantized<false>(input, 1.0, threads);"),
    ("pybind11::arg(\"scale\") = 1.0,", "pybind11::arg(\"scale\") = 1.0, pybind11::arg(\"threads\") = 128,"),
    ("&quantize_only, pybind11::arg(\"input\"),", "&quantize_only, pybind11::arg(\"input\"), pybind11::arg(\"threads\") = 128,"),
]
for before, after in changes:
    if before not in code:
        raise RuntimeError("Expected source fragment absent: " + before)
    code = code.replace(before, after)
target = ROOT / "sources" / "torch_binding_experiment.cu"
target.write_text(code, encoding="utf-8", newline="\n")
(ROOT / "binding_changes.patch").write_text("".join(difflib.unified_diff(original.splitlines(True), code.splitlines(True),
    fromfile="torch_binding_original.cu", tofile="torch_binding_experiment.cu")), encoding="utf-8")
manifest = {"scope": "Only explicit block size 128/256, dim16/64; original CUDA kernels copied unchanged.",
            "files": {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                      for p in sorted((ROOT / "sources").iterdir()) if p.is_file()}}
(ROOT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
