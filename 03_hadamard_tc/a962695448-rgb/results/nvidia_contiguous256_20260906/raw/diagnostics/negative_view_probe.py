#!/usr/bin/env python3
"""Post-run GPU diagnostic for logical negative views; no timing or source edits.

python diagnostics/negative_view_probe.py --runs-directory runs \
    --json diagnostics/negative_view_probe.json

GPU import/build/probes are gated on the completed three-run suite. Exit 0
means a lazy view was accepted and the wrong physical-base result reproduced.
Other outcomes remain explicit: NOT_REPRODUCED=1, UNAVAILABLE/UNVERIFIED=2,
UNSUPPORTED_LAZY_VIEW=3. A new JSON filename is required for every execution.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
APIS = ("hadamard", "hadamard_int4", "quantize_int4")
LAYOUTS = ("original", "contiguous256")
DTYPES = ("fp16", "bf16")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def completed_suite(runs):
    """Read only; this function runs before any Torch import or extension load."""
    suite_path = runs / "suite_status.json"
    suite = load_json(suite_path)
    require(suite["status"] == "PASS" and suite["exit_code"] == 0 and suite.get("finished_utc"),
            "All three timing workers must finish successfully before this diagnostic")
    workers = suite["workers"]
    require(len(workers) == 3 and {w["run_index"] for w in workers} == {1, 2, 3}, "Incomplete worker list")
    require(all(w["exit_code"] == 0 and w.get("finished_utc") for w in workers), "A timing worker is unfinished/failed")
    protocol = load_json(ROOT / "protocol.json")
    protocol_sha = sha(ROOT / "protocol.json")
    manifest = load_json(ROOT / "run_manifest.json")
    source_hashes = {}
    require(len(manifest["files"]) == 17, "Unexpected frozen execution-file count")
    for name, info in manifest["files"].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT), "Frozen source path escapes experiment directory")
        actual = sha(path)
        require(actual == info["sha256"], "Frozen source changed: " + name)
        source_hashes[name] = actual
    reports, report_hashes = [], {}
    for index in (1, 2, 3):
        path = runs / f"run{index}.json"
        report = load_json(path)
        require(report["status"] == "PASS" and report["exit_code"] == 0 and report["run_index"] == index and
                report.get("finished_utc"), "A timing report is incomplete/failed")
        require(report["protocol_sha256"] == protocol_sha and report["run_manifest"] == manifest,
                "Timing report source/protocol differs from this experiment")
        require(report["summary"]["graph_comparisons"] == 48 and len(report["benchmarks"]) == 48,
                "Timing report does not contain its full fixed matrix")
        reports.append(report)
        report_hashes[path.name] = sha(path)
    require(len({r["environment"]["extension_sha256"] for r in reports}) == 1, "Timing extension binaries differ")
    return {"status": "PASS", "suite_sha256": sha(suite_path), "run_json_sha256": report_hashes,
            "source_commit": protocol["source_commit"], "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol_sha, "run_manifest_sha256": sha(ROOT / "run_manifest.json"),
            "frozen_source_sha256": source_hashes,
            "measured_environment": reports[0]["environment"]}, protocol


@contextmanager
def captured_output(record):
    """Capture native compiler and Python messages, leaving stdout as one JSON."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    with tempfile.TemporaryFile(mode="w+b") as stream:
        try:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
            stream.seek(0)
            data = stream.read()
            record["output_bytes"] = len(data)
            record["output_tail"] = data[-65536:].decode("utf-8", errors="replace")
            record["output_truncated"] = len(data) > 65536


def tensors(value):
    require(isinstance(value, (tuple, list)) or hasattr(value, "is_cuda"), "Unexpected API return type")
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def json_scalar(value):
    return repr(value) if isinstance(value, float) and not math.isfinite(value) else value


def cpu_tensor(tensor):
    # Compare logical output values. No expected-value adjustment or tolerance.
    return tensor.detach().resolve_neg().resolve_conj().cpu().contiguous()


def comparison(torch, actual, expected):
    aa, bb = tensors(actual), tensors(expected)
    result = {"equal": len(aa) == len(bb), "actual_components": len(aa), "expected_components": len(bb), "components": []}
    for index, (a, b) in enumerate(zip(aa, bb)):
        item = {"component": index, "actual_shape": list(a.shape), "expected_shape": list(b.shape),
                "actual_dtype": str(a.dtype), "expected_dtype": str(b.dtype)}
        if a.shape != b.shape or a.dtype != b.dtype:
            item.update(equal=False, reason="shape_or_dtype_mismatch")
        else:
            ac, bc = cpu_tensor(a), cpu_tensor(b)
            ab, eb = ac.view(torch.uint8).reshape(-1), bc.view(torch.uint8).reshape(-1)
            changed = ab != eb
            same = not bool(changed.any())
            item.update(equal=same, elements=a.numel(), differing_bytes=int(changed.sum()),
                        differing_elements=int(changed.reshape(-1, a.element_size()).any(dim=1).sum()))
            if not same:
                first_byte = int(changed.nonzero()[0, 0])
                element = first_byte // a.element_size()
                begin, end = element * a.element_size(), (element + 1) * a.element_size()
                item["first_difference"] = {"flat_element": element, "first_byte": first_byte,
                    "actual_value": json_scalar(ac.reshape(-1)[element].item()),
                    "expected_value": json_scalar(bc.reshape(-1)[element].item()),
                    "actual_bytes": ab[begin:end].tolist(), "expected_bytes": eb[begin:end].tolist()}
        result["components"].append(item)
        result["equal"] = result["equal"] and item["equal"]
    return result


def first_values(value):
    return [{"shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device),
             "first_eight": [json_scalar(x) for x in cpu_tensor(t).reshape(-1)[:8].tolist()]}
            for t in tensors(value)]


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def synthetic_controls(torch, dtype, negative):
    """Independent CPU controls for this exactly representable synthetic input."""
    values = [(-1 if negative else 1) * (3 if i % 2 == 0 else 1) for i in range(256)]
    # Independent dense H256, with exact integer sums for this input.
    transformed = [sum((-x if (i & j).bit_count() % 2 else x) for j, x in enumerate(values)) for i in range(256)]
    stored = torch.tensor([transformed], dtype=dtype, device="cpu")
    def quantized(data):
        scale = f32(max(abs(x) for x in data) / 7.0) or 1.0
        q = [max(-7, min(7, round(f32(f32(x) / scale)))) for x in data]
        packed = [(q[i] & 15) | ((q[i + 1] & 15) << 4) for i in range(0, 256, 2)]
        return (torch.tensor([packed], dtype=torch.uint8, device="cpu"),
                torch.tensor([scale], dtype=torch.float32, device="cpu"))
    return {"hadamard": stored, "hadamard_int4": quantized(stored.float().reshape(-1).tolist()),
            "quantize_int4": quantized(values)}


def run_probe(torch, extension, report):
    cases = report["cases"] = []
    report["inputs"] = []
    for dtype_name in DTYPES:
        dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
        storage = torch.full((272,), 123, device="cuda:0", dtype=dtype)
        base = storage[8:264].reshape(1, 256)
        base.copy_(torch.tensor([3, 1], device="cuda:0", dtype=dtype).repeat(128).reshape(1, 256))
        lazy = torch._neg_view(base)
        materialized = lazy.resolve_neg()
        storage_before, materialized_before = storage.clone(), materialized.clone()
        torch.cuda.synchronize()
        require(base.data_ptr() % 16 == 0 and lazy.is_neg() and not materialized.is_neg(), "Unexpected synthetic view construction")
        variants = {"base": base, "lazy_negative": lazy, "materialized_negative": materialized}
        report["inputs"].append({"dtype": dtype_name, "shape": [1, 256], "variants": {
            name: {"is_neg": value.is_neg(), "is_contiguous": value.is_contiguous(),
                   "shares_base_data_ptr": value.data_ptr() == base.data_ptr(),
                   "shares_base_storage": value.untyped_storage().data_ptr() == base.untyped_storage().data_ptr(),
                   "first_eight": cpu_tensor(value).reshape(-1)[:8].tolist()}
            for name, value in variants.items()}})
        controls = {"base": synthetic_controls(torch, dtype, False), "negative": synthetic_controls(torch, dtype, True)}
        for layout in LAYOUTS:
            for api in APIS:
                case = {"dtype": dtype_name, "layout": layout, "api": api, "block_threads": 128, "calls": {}}
                cases.append(case)
                outputs = {}
                for name, value in variants.items():
                    call = case["calls"][name] = {}
                    try:
                        output = getattr(extension, api)(value, block_threads=128, layout=layout)
                        torch.cuda.synchronize()
                        require(all(t.is_cuda for t in tensors(output)), "API returned a non-GPU output")
                        outputs[name] = output
                        call.update(status="RETURNED_GPU_OUTPUT", output=first_values(output))
                    except Exception as error:
                        call.update(status="EXCEPTION", exception_type=type(error).__name__, exception=str(error))
                    call["input_and_guard_unchanged"] = comparison(torch, storage, storage_before)
                    call["materialized_input_unchanged"] = comparison(torch, materialized, materialized_before)
                if "base" not in outputs or "materialized_negative" not in outputs:
                    case["status"] = "CONTROL_FAILURE"
                    continue
                case["base_control"] = comparison(torch, outputs["base"], controls["base"][api])
                case["materialized_control"] = comparison(torch, outputs["materialized_negative"], controls["negative"][api])
                guards_ok = all(call["input_and_guard_unchanged"]["equal"] and call["materialized_input_unchanged"]["equal"] for call in case["calls"].values())
                if not guards_ok:
                    case["status"] = "INPUT_OR_GUARD_MUTATED"
                elif not case["base_control"]["equal"] or not case["materialized_control"]["equal"]:
                    case["status"] = "CONTROL_FAILURE"
                elif "lazy_negative" not in outputs:
                    case["status"] = "UNSUPPORTED_LAZY_VIEW"
                else:
                    correct = comparison(torch, outputs["lazy_negative"], outputs["materialized_negative"])
                    physical = comparison(torch, outputs["lazy_negative"], outputs["base"])
                    case.update(lazy_vs_materialized_expected=correct, lazy_vs_physical_base=physical,
                                lazy_matches_materialized_expected=correct["equal"],
                                lazy_wrongly_matches_base=(not correct["equal"] and physical["equal"]))
                    case["status"] = "NOT_REPRODUCED" if correct["equal"] else "REPRODUCED" if physical["equal"] else "UNEXPECTED_RESULT"
    counts = {status: sum(case["status"] == status for case in cases) for status in sorted({case["status"] for case in cases})}
    report["case_status_counts"] = counts
    require(len(cases) == 12, "Diagnostic matrix incomplete")
    if any(status in counts for status in ("CONTROL_FAILURE", "INPUT_OR_GUARD_MUTATED", "UNEXPECTED_RESULT")):
        return "UNVERIFIED", 2
    if counts.get("REPRODUCED", 0):
        return "REPRODUCED", 0
    if counts.get("UNSUPPORTED_LAZY_VIEW", 0) == 12:
        return "UNSUPPORTED_LAZY_VIEW", 3
    return "NOT_REPRODUCED", 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-directory", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    require(not args.json.exists(), "Diagnostic output exists; choose a fresh filename")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "UNVERIFIED", "started_utc": utc(), "diagnostic_source_sha256": sha(Path(__file__)),
              "timing_performed": False, "frozen_sources_modified": False,
              "expected_value_policy": "Compare the actual GPU lazy-view result with the actual resolve_neg GPU result, bytewise for every component. Base and materialized results also checked against independent CPU dense H256 and float32 division/RNE INT4 for synthetic [3,1] repeated.",
              "import": {"status": "NOT_STARTED"}, "build": {"status": "NOT_STARTED"},
              "probe": {"status": "NOT_STARTED"}}
    code = 2
    try:
        gate, protocol = completed_suite(args.runs_directory.resolve())
        report["completed_timing_gate"] = gate
        report["import"]["status"] = "RUNNING"
        with captured_output(report["import"]):
            import torch
            sys.path.insert(0, str(ROOT))
            import run_experiment
        report["import"]["status"] = "PASS"
        report["environment"] = {"python": sys.executable, "python_version": platform.python_version(),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda, "experiment_root": str(ROOT),
            "cuda_home": os.environ.get("CUDA_HOME"), "negative_view_available": callable(getattr(torch, "_neg_view", None))}
        if not report["environment"]["negative_view_available"]:
            report["status"] = "UNAVAILABLE"
            report["reason"] = "torch._neg_view is unavailable; no GPU probes executed"
        else:
            require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expected one visible CUDA GPU")
            device_name, capability = torch.cuda.get_device_name(0), list(torch.cuda.get_device_capability(0))
            require(protocol["hardware"]["required_name_contains"] in device_name and capability == protocol["hardware"]["required_sm"], "GPU differs from completed timing suite")
            require(torch.__version__ == gate["measured_environment"]["torch"] and torch.version.cuda == gate["measured_environment"]["torch_cuda"], "Torch/CUDA environment differs from completed suite")
            os.environ["MAX_JOBS"] = protocol["hardware"]["max_jobs"]
            os.environ["TORCH_CUDA_ARCH_LIST"] = protocol["hardware"]["compile_arch"]
            report["environment"].update(gpu=device_name, sm=capability, max_jobs=os.environ["MAX_JOBS"],
                torch_cuda_arch_list=os.environ["TORCH_CUDA_ARCH_LIST"], cxx11_abi=torch._C._GLIBCXX_USE_CXX11_ABI)
            report["build"]["status"] = "RUNNING"
            with captured_output(report["build"]):
                extension = run_experiment.load_extension()
            binary_sha = sha(Path(extension.__file__))
            report["build"].update(status="LOADED", extension_file=str(Path(extension.__file__).resolve()), extension_sha256=binary_sha)
            require(binary_sha == gate["measured_environment"]["extension_sha256"], "Loaded binary differs from the three measured runs")
            report["build"]["same_measured_binary"] = True
            report["probe"]["status"] = "RUNNING"
            with captured_output(report["probe"]):
                with torch.inference_mode():
                    report["status"], code = run_probe(torch, extension, report)
            report["probe"]["status"] = "COMPLETED"
            require(all(sha(ROOT / name) == digest for name, digest in gate["frozen_source_sha256"].items()),
                    "Frozen source changed during diagnostic")
            report["frozen_source_sha256_rechecked_after_probe"] = True
    except Exception as error:
        for stage in ("import", "build", "probe"):
            if report[stage]["status"] == "RUNNING":
                report[stage]["status"] = "ERROR"
        report.update(status="UNVERIFIED", error_type=type(error).__name__, error=str(error))
        code = 2
    report.update(finished_utc=utc(), exit_code=code)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with args.json.open("x", encoding="utf-8") as output:
        output.write(text)
    print(text, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
