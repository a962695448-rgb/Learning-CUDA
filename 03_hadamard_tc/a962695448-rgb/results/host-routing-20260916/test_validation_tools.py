"""Validate analysis rejection paths and bounded execution with synthetic inputs."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

from analyze_cuda import analyze
from validation_common import ROOT, Runner, save


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.protocol = json.loads((ROOT / "CUDA_PROTOCOL.json").read_text())

    def reports(self, candidate_us=18):
        identity = {
            "hardware": {"gpu_name": "SYNTHETIC UNIT TEST"},
            "numpy": "synthetic",
            "protocol_sha256": "synthetic",
            "control_binding_sha256": "synthetic-control",
            "candidate_binding_sha256": "synthetic-candidate",
            "binaries": {"synthetic": True},
            "cuda_compiler": {"synthetic": True},
        }
        save(
            self.directory / "cuda-validation.json",
            {
                **identity,
                "status": "PASS",
                "result": {"status": "PASS"},
            },
        )
        for round_index in (1, 2, 3):
            records = []
            for kind in ("target", "control"):
                for rows, dim in self.protocol[kind + "_shapes"]:
                    for dtype in self.protocol["dtypes"]:
                        for layout in self.protocol["layouts"]:
                            for operation in self.protocol["operations"]:
                                for version in ("control", "candidate"):
                                    for group in range(self.protocol["groups"]):
                                        us = (
                                            20 if version == "control" else candidate_us
                                        )
                                        records.append(
                                            {
                                                "kind": kind,
                                                "rows": rows,
                                                "dim": dim,
                                                "dtype": dtype,
                                                "layout": layout,
                                                "operation": operation,
                                                "version": version,
                                                "group": group,
                                                "round": round_index,
                                                "repeats": self.protocol["repeats"],
                                                "elapsed_ns": us
                                                * self.protocol["repeats"]
                                                * 1000,
                                                "mean_us": us,
                                            }
                                        )
            save(
                self.directory / f"cuda-round{round_index}.json",
                {
                    **identity,
                    "status": "PASS",
                    "mode": "benchmark",
                    "result": {"status": "PASS", "records": records},
                },
            )

    def test_synthetic_improvement_passes_and_slowdown_does_not(self):
        self.reports()
        result = analyze(self.directory, self.protocol)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["observations"], 3456)
        self.reports(candidate_us=21)
        self.assertFalse(analyze(self.directory, self.protocol)["accepted"])

    def test_missing_observation_is_rejected(self):
        self.reports()
        path = self.directory / "cuda-round2.json"
        report = json.loads(path.read_text())
        report["result"]["records"].pop()
        save(path, report)
        with self.assertRaisesRegex(ValueError, "Missing"):
            analyze(self.directory, self.protocol)

    def test_changed_binary_is_rejected(self):
        self.reports()
        path = self.directory / "cuda-round3.json"
        report = json.loads(path.read_text())
        report["binaries"] = {"changed": True}
        save(path, report)
        with self.assertRaisesRegex(ValueError, "identity"):
            analyze(self.directory, self.protocol)

    def test_one_regressed_case_rejects_an_otherwise_fast_candidate(self):
        self.reports()
        path = self.directory / "cuda-round2.json"
        report = json.loads(path.read_text())
        for record in report["result"]["records"]:
            if (
                record["version"] == "candidate"
                and record["kind"] == "control"
                and record["rows"] == 1
                and record["dim"] == 1
            ):
                record["mean_us"] = 25
                record["elapsed_ns"] = 25 * record["repeats"] * 1000
        save(path, report)
        self.assertFalse(analyze(self.directory, self.protocol)["accepted"])

    def test_timeout_stops_the_owned_process(self):
        runner = Runner(self.directory / "timeout", timeout=0.1)
        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            runner.run(
                "sleep",
                [sys.executable, "-c", "import time; time.sleep(30)"],
                self.directory,
            )
        self.assertTrue(runner.record["stages"][0]["timed_out"])
        self.assertNotEqual(runner.record["stages"][0]["returncode"], 0)


if __name__ == "__main__":
    unittest.main()
