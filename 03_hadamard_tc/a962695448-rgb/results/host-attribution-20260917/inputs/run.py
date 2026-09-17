"""Compile a same-source control pair, then run three fresh diagnostic processes."""
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results-remote"
OUT.mkdir(exist_ok=False)
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MAX_JOBS="1", TORCH_CUDA_ARCH_LIST="8.9")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
manifest = json.loads((ROOT / "INPUT_MANIFEST.json").read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
affinity = sorted(os.sched_getaffinity(0))
os.sched_setaffinity(0, {affinity[0]})
import torch
torch.set_num_threads(1)
from torch.utils.cpp_extension import load

summary = {"status": "RUNNING", "jobs": [], "purpose": "Diagnostic only; no kernel adoption decision",
    "manifest_sha256": hashlib.sha256((ROOT / "INPUT_MANIFEST.json").read_bytes()).hexdigest(),
    "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
    "python": sys.version, "affinity_before": affinity, "affinity_pinned": sorted(os.sched_getaffinity(0)),
    "nvcc": subprocess.check_output(["/usr/local/cuda/bin/nvcc", "--version"], text=True),
    "nvidia_smi_before": subprocess.check_output(["nvidia-smi"], text=True)}


def save():
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


save()
try:
    modules = {}
    for label, source, name in [("control_a", "control", "hd_diag_control_a"), ("control_b", "control", "hd_diag_control_b"), ("candidate", "candidate", "hd_diag_candidate")]:
        root = ROOT / "sources" / source
        build = ROOT / "build" / label
        build.mkdir(parents=True)
        modules[label] = load(name=name, sources=[str(root / "src/torch_binding.cu")],
            extra_include_paths=[str(root / "include")], extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "--expt-relaxed-constexpr"],
            build_directory=str(build), with_cuda=True, verbose=True)
    summary["binaries"] = {label: hashlib.sha256(Path(op.__file__).read_bytes()).hexdigest() for label, op in modules.items()}
    for process in range(3):
        started = time.monotonic()
        with (OUT / f"process-{process}.log").open("x") as log:
            job = subprocess.run([sys.executable, "-u", str(ROOT / "worker.py"), "--process", str(process)],
                cwd=ROOT, env=os.environ, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        summary["jobs"].append({"process": process, "returncode": job.returncode, "seconds": time.monotonic() - started})
        assert job.returncode == 0, f"Diagnostic process {process} failed"
        save()
        print("process", process, "finished", flush=True)
    summary["status"] = "PASS"
except Exception:
    summary["status"] = "FAIL"
    summary["error"] = traceback.format_exc()
    print(summary["error"], flush=True)
for name, expected in manifest.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
summary["sources_unchanged"] = True
summary["nvidia_smi_after"] = subprocess.check_output(["nvidia-smi"], text=True)
save()
files = {str(p.relative_to(ROOT)): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "text": p.read_text()}
         for p in OUT.iterdir() if p.is_file()}
payload = json.dumps(files, ensure_ascii=True).encode()
(ROOT / "FINAL_EXPORT.txt").write_bytes(payload)
receipt = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "status": summary["status"]}
(ROOT / "FINAL_EXPORT_RECEIPT.txt").write_text(json.dumps(receipt))
print(receipt, flush=True)
