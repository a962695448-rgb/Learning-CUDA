"""Pinned Hadamard A100 acceptance; every write and child process belongs to this directory."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
EXPECTED_ROOT = Path("/home/vipuser/infinitensor-2026/hadamard-cuda-a100")
assert ROOT == EXPECTED_ROOT
SOURCE = ROOT / "source"
RESULTS = ROOT / "results"
CONDA = "/home/vipuser/miniconda3/bin/python"
PYTHON = ROOT / ".venv/bin/python"
END_WORK = datetime(2026, 9, 5, 21, 0, tzinfo=timezone.utc).timestamp()  # 05:00 China time
RESULTS.mkdir(exist_ok=True)
REPORT = {"status": "RUNNING", "source_commit": "12c76d8331ef7cf3fd4c8c14a049162559be4302",
          "started_utc": datetime.now(timezone.utc).isoformat(), "worker_pid": os.getpid(),
          "stop_new_work_by": "2026-09-06T05:00:00+08:00", "stages": [], "completed": [], "optional_unverified": []}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save():
    temporary = RESULTS / "state.tmp.json"
    temporary.write_text(json.dumps(REPORT, indent=2) + "\n")
    temporary.replace(RESULTS / "state.json")


def snapshot():
    result = {}
    for name, query in (("gpu", "--query-gpu=name,memory.total,mig.mode.current,driver_version,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw"),
                        ("processes", "--query-compute-apps=pid,process_name,used_memory")):
        p = subprocess.run(["nvidia-smi", query, "--format=csv"], capture_output=True, text=True, timeout=15)
        result[name] = {"exit": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    return result


def guard_gpu():
    name = subprocess.check_output(["nvidia-smi", "--query-gpu=name,mig.mode.current", "--format=csv,noheader"], text=True, timeout=15)
    if "A100" not in name or "Disabled" not in name:
        raise RuntimeError("Real A100 with MIG disabled is required")
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True, timeout=15).strip()
    if processes:
        raise RuntimeError("Unexpected active compute process before owned GPU stage: " + processes)


def stage(name, args, timeout=600, expected=0, gpu=False, required=True, extra_env=None):
    remaining = END_WORK - time.time()
    if remaining < 60:
        raise TimeoutError("Reached orderly collection boundary; no further stage started")
    if gpu:
        guard_gpu()
    log = RESULTS / (name + ".log")
    if log.exists():
        raise RuntimeError("Refusing to overwrite prior stage log: " + name)
    environment = dict(ENV)
    environment.update(extra_env or {})
    entry = {"name": name, "args": [str(x) for x in args], "expected_exit": expected,
             "started_utc": datetime.now(timezone.utc).isoformat(), "gpu_stage": gpu}
    if gpu:
        entry["before"] = snapshot()
    with log.open("xb") as out:
        p = subprocess.Popen(entry["args"], cwd=SOURCE, stdout=out, stderr=subprocess.STDOUT,
                             env=environment, start_new_session=True)
        entry["pid"] = p.pid
        REPORT["stages"].append(entry)
        REPORT["active"] = name
        save()
        print("START", name, p.pid, flush=True)
        try:
            entry["exit"] = p.wait(timeout=min(timeout, max(1, int(remaining))))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)  # Only this owned child process group.
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            entry["exit"] = p.returncode
            entry["timed_out"] = True
    entry["finished_utc"] = datetime.now(timezone.utc).isoformat()
    entry["log"] = log.name
    entry["log_sha256"] = sha(log)
    if gpu:
        entry["after"] = snapshot()
    REPORT["active"] = None
    if entry["exit"] == expected and not entry.get("timed_out"):
        REPORT["completed"].append(name)
    save()
    print("EXIT", name, entry["exit"], flush=True)
    if required and (entry["exit"] != expected or entry.get("timed_out")):
        raise RuntimeError("Stage did not complete successfully: " + name)
    return entry["exit"] == expected and not entry.get("timed_out")


ENV = dict(os.environ)
ENV.update(CUDA_HOME="/usr/local/cuda-12.4", MAX_JOBS="1", NVCC_THREADS="1", TORCH_CUDA_ARCH_LIST="8.0",
           PIP_NO_CACHE_DIR="1", PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_INPUT="1",
           TORCH_EXTENSIONS_DIR=str(ROOT / ".torch-extensions"), CUDA_CACHE_PATH=str(ROOT / ".cuda-cache"),
           TRITON_CACHE_DIR=str(ROOT / ".triton-cache"), XDG_CACHE_HOME=str(ROOT / ".cache"),
           PYTHONPYCACHEPREFIX=str(ROOT / ".pycache"), PYTHONIOENCODING="utf-8",
           OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4")
ENV["PATH"] = str(ROOT / ".venv/bin") + ":/home/vipuser/miniconda3/bin:/usr/local/cuda-12.4/bin:" + os.environ.get("PATH", "")


def main():
    try:
        manifest = json.loads((ROOT / "source_manifest.json").read_text())
        assert manifest["commit"] == REPORT["source_commit"]
        for name, info in manifest["files"].items():
            assert sha(ROOT / name) == info["sha256"], name
        REPORT["source_manifest"] = manifest
        REPORT["before"] = snapshot()
        guard_gpu()
        driver = Path("/usr/lib/x86_64-linux-gnu/libcuda.so.550.127.05")
        with driver.open("rb") as stream:
            assert stream.read(5) == b"\x7fELF\x02"
        links = ROOT / ".driver-libs"
        links.mkdir()
        for name in ("libcuda.so", "libcuda.so.1"):
            (links / name).symlink_to(driver)
        ENV["LIBRARY_PATH"] = str(links) + ":" + ENV.get("LIBRARY_PATH", "")
        ENV["LD_LIBRARY_PATH"] = str(links) + ":/usr/local/cuda-12.4/lib64:" + ENV.get("LD_LIBRARY_PATH", "")
        ENV["TRITON_LIBCUDA_PATH"] = str(links)
        stage("create_private_venv", [CONDA, "-m", "venv", "--system-site-packages", ROOT / ".venv"], timeout=120)
        environment_code = "import torch,numpy,json,sys; assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,0) and 'A100' in torch.cuda.get_device_name(); assert torch.__version__=='2.5.0+cu124'; print(json.dumps({'python':sys.executable,'torch':torch.__version__,'numpy':numpy.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'cxx11_abi':torch._C._GLIBCXX_USE_CXX11_ABI}))"
        stage("private_environment", [PYTHON, "-c", environment_code], gpu=True)
        pip = [PYTHON, "-m", "pip", "install", "--no-deps", "--no-cache-dir", "--timeout", "30", "--retries", "1"]
        if not stage("ninja_pypi", pip + ["--index-url", "https://pypi.org/simple", "ninja==1.11.1.3"], timeout=120, required=False):
            stage("ninja_tuna", pip + ["--index-url", "https://pypi.tuna.tsinghua.edu.cn/simple", "ninja==1.11.1.3"], timeout=180)
        stage("build_arch80_cpu_test", ["make", "-j1", "CUDA_HOME=/usr/local/cuda-12.4", "ARCH=80", "all", "cpu-test"], timeout=600)
        REPORT["cli_binary_sha256"] = sha(SOURCE / "build/hadamard")
        stage("cli_default_original_matrix_and_benchmarks", [PYTHON, "scripts/run_validation.py", "--label", "a100_default128", "--benchmark"], timeout=600, gpu=True)
        cli_log = SOURCE / "results/validation_a100_default128.log"
        assert "SELF_TEST PASS cases=1876" in cli_log.read_text()
        stage("cli_explicit256_original_matrix", [SOURCE / "build/hadamard", "--self-test", "--block-threads", "256"], timeout=300, gpu=True)
        assert "SELF_TEST PASS cases=1876" in (RESULTS / "cli_explicit256_original_matrix.log").read_text()
        stage("restore_reference", [PYTHON, ROOT / "sync_reference.py", ROOT / "reference_source.json", "--json", RESULTS / "reference_source_sync.json"], timeout=300)
        reference = ROOT / "fast-hadamard-transform"
        stage("build_install_reference", [PYTHON, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--no-cache-dir", reference],
              timeout=1500, extra_env={"FAST_HADAMARD_TRANSFORM_FORCE_BUILD": "TRUE", "FAST_HADAMARD_TRANSFORM_SKIP_CUDA_BUILD": "FALSE", "FAST_HADAMARD_TRANSFORM_FORCE_CXX11_ABI": "FALSE"})
        stage("dao_original1800_default128_and_12_benchmarks", [PYTHON, "scripts/compare_reference.py", "--reference-repo", reference,
            "--benchmark", "--build-directory", ROOT / "build_default", "--json", RESULTS / "dao_default.json"], timeout=1200, gpu=True)
        report = json.loads((RESULTS / "dao_default.json").read_text())
        assert report["status"] == "PASS" and report["summary"]["cases"] == 1800
        assert report["environment"]["compute_capability"] == [8, 0] and "A100" in report["environment"]["gpu"]
        assert len(report["benchmarks"]) == len(report["graph_benchmarks"]) == 12
        stage("thread_api_original1800", [PYTHON, "scripts/verify_block_threads.py", "--reference-repo", reference,
            "--build-directory", ROOT / "build_checker", "--json", RESULTS / "api_threads.json"], timeout=900, gpu=True)
        api = json.loads((RESULTS / "api_threads.json").read_text())
        assert api["status"] == "PASS" and api["summary"]["cases"] == 1800
        assert api["default_and_explicit128_and_256_bitwise_equal"] and api["non_default_stream"]["pass"]
        REPORT["mandatory_correctness_and_original12"] = "PASS"
        save()
        for run in (1, 2, 3):
            if END_WORK - time.time() < 600:
                REPORT["optional_unverified"].append("promotion_72_run" + str(run))
                continue
            stage("promotion_72_run" + str(run), [PYTHON, ROOT / "promotion_a100.py", "--run-index", str(run),
                  "--reference-repo", reference, "--output", RESULTS / ("run" + str(run) + ".json")], timeout=600, gpu=True)
            p = json.loads((RESULTS / ("run" + str(run) + ".json")).read_text())
            assert p["status"] == "PASS" and p["summary"]["benchmark_configurations"] == 72
        REPORT["status"] = "PASS" if not REPORT["optional_unverified"] else "MANDATORY_PASS_OPTIONAL_INCOMPLETE"
        return 0
    except Exception as error:
        import traceback
        REPORT.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        print(REPORT["traceback"], flush=True)
        return 1
    finally:
        REPORT["after"] = snapshot()
        REPORT["finished_utc"] = datetime.now(timezone.utc).isoformat()
        REPORT["active"] = None
        save()
        print(json.dumps({"status": REPORT["status"], "completed": REPORT["completed"], "optional_unverified": REPORT["optional_unverified"]}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
