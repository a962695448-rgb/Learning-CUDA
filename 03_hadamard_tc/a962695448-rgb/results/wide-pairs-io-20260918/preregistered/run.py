"""Build both CUDA frontends, validate semantics, and run the frozen matrix."""
import base64
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results-remote"
OUT.mkdir(exist_ok=False)
PROJECT = Path("03_hadamard_tc/a962695448-rgb")
summary = {"status": "RUNNING", "jobs": []}
manifest = json.loads((ROOT / "INPUT_MANIFEST.json").read_text())
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MAX_JOBS="1", TORCH_CUDA_ARCH_LIST="8.9")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + "\n")


def verify_inputs():
    for name, expected in manifest.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name


def job(name, command, cwd, timeout=1800):
    start = time.monotonic()
    with (OUT / (name + ".log")).open("x") as log:
        result = subprocess.run(command, cwd=cwd, env=os.environ, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    summary["jobs"].append({"name": name, "command": command, "returncode": result.returncode, "seconds": time.monotonic() - start})
    save("summary.json", summary)
    assert result.returncode == 0, (name, result.returncode)
    print(name, "PASS", flush=True)
    return (OUT / (name + ".log")).read_text()


def module(label, version, name):
    from torch.utils.cpp_extension import load
    project = ROOT / version / "cuda" / PROJECT
    build = ROOT / "build" / label
    build.mkdir(parents=True)
    return load(name=name, sources=[str(project / "src/torch_binding.cu")], extra_include_paths=[str(project / "include")],
                extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo",
                "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "--expt-relaxed-constexpr"], build_directory=str(build), with_cuda=True, verbose=True)


save("summary.json", summary)
try:
    verify_inputs()
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {affinity[0]})
    import numpy as np
    import torch
    assert torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (8, 9)
    torch.set_num_threads(1)
    summary["environment"] = {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
        "cpu_affinity_before": affinity, "cpu_affinity_pinned": sorted(os.sched_getaffinity(0)),
        "torch_cpu_threads": torch.get_num_threads(), "nvcc": subprocess.check_output(["/usr/local/cuda/bin/nvcc", "--version"], text=True),
        "nvidia_smi_before": subprocess.check_output(["nvidia-smi"], text=True)}
    summary["manifest_sha256"] = hashlib.sha256((ROOT / "INPUT_MANIFEST.json").read_bytes()).hexdigest()
    summary["protocol_sha256"] = hashlib.sha256((ROOT / "PROTOCOL.json").read_bytes()).hexdigest()
    save("summary.json", summary)

    for version in ("control", "candidate"):
        project = ROOT / version / "cuda" / PROJECT
        job("cli-build-" + version, ["make", "all", "NVCC=/usr/local/cuda/bin/nvcc", "ARCH=89"], project)
        label = "wide_io_" + version
        job("cli-validation-" + version, [sys.executable, "scripts/run_validation.py", "--label", label], project)
        validation = project / "results" / ("validation_" + label + ".log")
        content = validation.read_text()
        assert "SELF_TEST PASS cases=1876" in content and len(re.findall(r"EXIT_CODE 2;", content)) == 15
        shutil.copy2(validation, OUT / ("cli-validation-" + version + "-details.log"))
        modes = [("packed", 256)] if version == "control" else [("original", 256), ("packed", 128), ("packed", 256), ("auto", 128), ("auto", 256)]
        for layout, threads in modes:
            content = job(f"cli-selftest-{version}-{layout}-{threads}", [str(project / "build/hadamard"), "--self-test", "--row-layout", layout, "--block-threads", str(threads)], project)
            assert "SELF_TEST PASS cases=1876" in content
        if version == "candidate":
            content = job("cli-selftest-candidate-contiguous256", [str(project / "build/hadamard"), "--self-test", "--fused-layout", "contiguous256", "--block-threads", "128"], project)
            assert "SELF_TEST PASS cases=1876" in content

    project = ROOT / "candidate/cuda" / PROJECT
    job("cpu-reference-and-row-policy", ["make", "cpu-test"], project)
    tuner = project / "build/tune_launch"
    job("tuner-build", ["/usr/local/cuda/bin/nvcc", "-O3", "-std=c++17", "-lineinfo", "-arch=sm_89", "-Iinclude", "src/tune_launch.cu", "-o", str(tuner)], project)
    content = job("tuner-validation", [str(tuner)], project)
    assert "PASS: 96 shape/dtype/scale/mode cases; 384 launch configurations; 1920 raw samples." in content

    control = module("control_a", "control", "wi_control_a")
    twin = module("control_b", "control", "wi_control_b")
    candidate = module("candidate", "candidate", "wi_candidate")
    summary["binaries"] = {label: hashlib.sha256(Path(op.__file__).read_bytes()).hexdigest() for label, op in (("control", control), ("twin", twin), ("candidate", candidate))}
    save("summary.json", summary)
    sys.path[:0] = [str(ROOT / "tools"), str(project / "scripts")]
    from verify_wide_transform import verify as wide_checks, contexts as wide_contexts
    from verify_hadamard_packed import verify as small_checks
    from verify_paired_quantization import verify as quantization_pair_checks
    from verify_singleton_quantization import verify as singleton_checks
    from verify_quantize_packed import verify as packed_checks, contexts as quantization_contexts
    from verify_out_buffers import positive_cases, negative_cases, mutation_contract, execution_contexts
    from verify_execution_context import configurations as context_cases, check_case
    from verify_tensor_metadata import check_metadata

    report = wide_checks(torch, np, candidate, control)
    assert len(report["records"]) == 416
    save("wide-components.json", report)
    report = wide_contexts(torch, np, candidate)
    assert len(report) == 128
    save("wide-contexts.json", {"status": "PASS", "cases": report})
    save("all-execution-contexts.json", {"status": "PASS", "cases": [check_case(torch, np, candidate, case, dtype) for dtype in (torch.float16, torch.bfloat16) for case in context_cases()]})
    job("overflow-compatibility", [sys.executable, "-u", str(ROOT / "tools/verify_overflow_wide.py"), str(ROOT)], ROOT)
    save("small-transform-control.json", small_checks(torch, np, candidate, control))
    save("quantization-pairs.json", quantization_pair_checks(torch, np, candidate, control))
    save("singleton.json", singleton_checks(torch, np, candidate, control))
    report = packed_checks(torch, np, candidate, control)
    report["contexts"] = quantization_contexts(torch, np, candidate)
    save("packed-regression.json", report)
    save("legacy-regression.json", {"status": "PASS", "positive": positive_cases(torch, np, candidate), "negative": negative_cases(torch, candidate), "mutation": mutation_contract(torch, np, candidate), "contexts": execution_contexts(torch, np, candidate)})
    save("tensor-metadata.json", {"status": "PASS", "cases": check_metadata(torch, candidate)})
    print("ALL_CORRECTNESS_PASS", flush=True)
    for round_id in range(3):
        job(f"timing-round-{round_id}", [sys.executable, "-u", str(ROOT / "bench.py"), "--round", str(round_id)], ROOT, timeout=3600)
    rounds = [json.loads((OUT / f"round-{i}.json").read_text()) for i in range(3)]
    combined = {"status": "ACCEPT" if all(r["status"] == "ACCEPT" for r in rounds) else "REJECT", "protocol": json.loads((ROOT / "PROTOCOL.json").read_text())["cuda"]}
    for key in ("records", "host_records", "groups", "calibration"):
        combined[key] = [item for report in rounds for item in report[key]]
    save("benchmark.json", combined)
    summary["candidate_decision"] = combined["status"]
    job("independent-benchmark-audit", [sys.executable, str(ROOT / "tools/analyze.py"), str(OUT / "benchmark.json"), "--output", str(OUT / "BENCHMARK_AUDIT.json")], ROOT)
    verify_inputs()
    summary["sources_unchanged"] = True
    summary["environment"]["nvidia_smi_after"] = subprocess.check_output(["nvidia-smi"], text=True)
    summary["status"] = "PASS"
except BaseException:
    summary["status"] = "FAIL"
    summary["error"] = traceback.format_exc()
    print(summary["error"], flush=True)
finally:
    save("summary.json", summary)
    sys.stdout.flush()
    sys.stderr.flush()
    files = {str(p.relative_to(ROOT)): {"text": p.read_text(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(OUT.iterdir()) if p.is_file()}
    if (ROOT / "run.log").exists():
        log = (ROOT / "run.log").read_bytes()
        files["run.log"] = {"text": log.decode("utf-8"), "sha256": hashlib.sha256(log).hexdigest()}
    raw = json.dumps(files, ensure_ascii=True).encode()
    compressed = gzip.compress(raw, mtime=0)
    encoded = base64.b64encode(compressed)
    (ROOT / "FINAL_EXPORT.base64.txt").write_bytes(encoded)
    parts = []
    for i, start in enumerate(range(0, len(encoded), 400000)):
        value = encoded[start:start + 400000]
        name = f"FINAL_EXPORT_PART_{i:02d}.txt"
        (ROOT / name).write_bytes(value)
        parts.append({"name": name, "characters": len(value), "sha256": hashlib.sha256(value).hexdigest()})
    receipt = {"status": summary["status"], "decision": summary.get("candidate_decision"), "files": len(files), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "gzip_bytes": len(compressed), "gzip_sha256": hashlib.sha256(compressed).hexdigest(), "base64_characters": len(encoded), "parts": parts}
    (ROOT / "FINAL_EXPORT_RECEIPT.txt").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
raise SystemExit(0 if summary["status"] == "PASS" else 1)
