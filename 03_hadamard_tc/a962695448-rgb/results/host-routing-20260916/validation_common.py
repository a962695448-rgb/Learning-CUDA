"""Shared source checks and bounded subprocess execution for the GPU validation bundle."""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(label, directory):
    tree = json.loads((ROOT / "SOURCE_TREES.json").read_text())[label]
    hashes = {}
    for item in tree["tree"]:
        if item["type"] != "blob":
            continue
        path = directory / item["path"]
        data = path.read_bytes()
        blob = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest()
        if blob != item["sha"]:
            lfs_logo = (
                label == "nine"
                and item["path"] == "docs/source/_static/ninetoothed-logo.png"
                and hashlib.sha256(data).hexdigest()
                == "2eb81c9d72a5f53d3409a8f56f5bf8ea85145ac788930743a5449692097d4537"
            )
            if not lfs_logo:
                raise ValueError("Source checksum mismatch: " + item["path"])
        hashes[item["path"]] = hashlib.sha256(data).hexdigest()
    return hashes


def gpu_environment(allow_non_a100=False):
    import torch

    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("A usable NVIDIA CUDA GPU is required.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Select exactly one authorized GPU with CUDA_VISIBLE_DEVICES before running."
        )
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    is_a100 = "A100" in name
    if not is_a100 and not allow_non_a100:
        raise RuntimeError(
            f"Official A100 validation requested, but the visible device is {name}. "
            "Use --allow-non-a100 only for explicitly labelled supplementary checks."
        )
    return {
        "gpu_name": name,
        "visible_devices": 1,
        "a100_validation": is_a100,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.version.split()[0],
    }


def compiler_info():
    compiler = shutil.which("nvcc")
    if compiler is None:
        return {"available": False}
    result = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, timeout=30, check=False
    )
    return {
        "available": result.returncode == 0,
        "version_output": result.stdout + result.stderr,
        "returncode": result.returncode,
    }


def interrupt_on_termination(_signum, _frame):
    raise KeyboardInterrupt


class Runner:
    def __init__(self, output, timeout):
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.timeout = timeout
        self.record = {"status": "UNVERIFIED", "stages": []}
        self.env = dict(os.environ)
        self.env["PATH"] = (
            str(Path(sys.executable).parent) + os.pathsep + self.env.get("PATH", "")
        )
        self.env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        self.env["MPLBACKEND"] = "Agg"
        self.env["MAX_JOBS"] = "1"
        self.env["NINETOOTHED_CACHE_DIR"] = str(self.output / "cache/ninetoothed")
        self.env["TRITON_CACHE_DIR"] = str(self.output / "cache/triton")
        signal.signal(signal.SIGTERM, interrupt_on_termination)

    def write(self):
        save(self.output / "run-summary.json", self.record)

    @staticmethod
    def stop(process):
        if process.poll() is not None:
            return process.returncode
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            return process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            return process.wait()

    def run(self, name, arguments, cwd):
        log = self.output / (name + ".log")
        command = list(map(str, arguments))
        started = time.monotonic()
        timed_out = False
        with log.open("w") as stream:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=self.env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
            try:
                code = process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                code = self.stop(process)
            except BaseException:
                self.stop(process)
                self.record["status"] = "INTERRUPTED"
                self.write()
                raise
        stage = {
            "name": name,
            "command": command,
            "returncode": code,
            "timed_out": timed_out,
            "seconds": time.monotonic() - started,
            "log_sha256": digest(log),
        }
        self.record["stages"].append(stage)
        self.write()
        print(f"{name}: exit={code}, timeout={timed_out}", flush=True)
        if timed_out or code:
            raise RuntimeError(f"{name} did not complete successfully; inspect {log}.")

    def finish(self):
        if self.record["status"] == "RUNNING":
            self.record["status"] = "INTERRUPTED"
        names = [
            p
            for p in self.output.iterdir()
            if p.is_file() and p.name != "run-summary.json"
        ]
        self.record["artifacts"] = {
            p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in names
        }
        self.write()
