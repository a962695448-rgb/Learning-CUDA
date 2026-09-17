"""Validate diagnostic evidence handling on independent synthetic timelines."""
import itertools
import json
import tempfile
import unittest
from pathlib import Path

from analyze import analyze


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.protocol = json.loads((Path(__file__).resolve().parents[1] / "PROTOCOL.json").read_text())
        (self.root / "PROTOCOL.json").write_text(json.dumps(self.protocol))
        (self.root / "results-remote").mkdir()
        labels = ("control_a", "control_b", "candidate")
        design = [(order, rotation) for rotation in range(3) for order in itertools.permutations(labels)]
        for process in range(3):
            rows = []
            for case in self.protocol["cases"]:
                for group, (order, rotation) in enumerate(design):
                    for condition in self.protocol["conditions"]:
                        for position, label in enumerate(order):
                            pool = (int(condition[-1]) if condition.startswith("shared_") else
                                    (labels.index(label) + rotation) % 3 if condition == "rotating_separate" else None)
                            start = len(rows) * 100_000_000
                            duration = 45_000_000 if label == "candidate" else 50_000_000
                            rows.append(dict(case=case["id"], group=group, condition=condition, module=label,
                                measurement_order=list(order), position=position, pool=pool, pool_rotation=rotation,
                                start_ns=start, end_ns=start + duration, wall_us=duration/1000/5000,
                                process_cpu_us=7., thread_cpu_us=6.9, voluntary_switches=0,
                                involuntary_switches=0, gc_collection_deltas=[0, 0, 0]))
            report = dict(status="PASS", process=process, load_order=self.protocol["load_orders"][process],
                          cases=[dict(c, correctness="PASS") for c in self.protocol["cases"]], records=rows)
            self.write(process, report)

    def path(self, process):
        return self.root / "results-remote" / f"process-{process}.json"

    def write(self, process, report):
        self.path(process).write_text(json.dumps(report))

    def test_complete_design_and_stable_effect(self):
        result = analyze(self.root)
        self.assertEqual(result["adoption_decision"], "NOT_APPLICABLE_DIAGNOSTIC_ONLY")
        self.assertEqual(len(result["comparisons"]), 180)
        self.assertTrue(all(f["persistent_candidate_faster"] for f in result["findings"]))

    def test_missing_record_and_wrong_buffer_assignment_are_rejected(self):
        original = json.loads(self.path(0).read_text())
        altered = json.loads(self.path(0).read_text())
        altered["records"].pop()
        self.write(0, altered)
        with self.assertRaises(AssertionError):
            analyze(self.root)
        original["records"][0]["pool"] = 99
        self.write(0, original)
        with self.assertRaises(AssertionError):
            analyze(self.root)

    def test_separate_medians_do_not_override_conflicting_paired_evidence(self):
        old = [10.1, 8.1, 7.9, 12.8, 10.4, 8.0, 8.02, 8.01, 8.0] * 2
        new = [8.0, 9.9, 10.6, 10.8, 12.5, 11.4, 7.96, 7.94, 7.93] * 2
        case = self.protocol["cases"][0]["id"]
        for process in range(3):
            report = json.loads(self.path(process).read_text())
            for record in report["records"]:
                if record["case"] == case and record["condition"] == "shared_0":
                    value = (new if record["module"] == "candidate" else old)[record["group"]]
                    record["wall_us"] = value
                    record["end_ns"] = record["start_ns"] + int(round(value * 1000 * 5000))
            self.write(process, report)
        result = analyze(self.root)
        row = next(r for r in result["comparisons"] if r["case"] == case and r["condition"] == "shared_0"
                   and r["left"] == "control_a" and r["right"] == "candidate")
        self.assertLess(row["ratio_of_separate_medians"], .9)
        self.assertGreater(row["median_of_group_ratios"], 1.)
        finding = next(r for r in result["findings"] if r["case"] == case and r["condition"] == "shared_0")
        self.assertTrue(finding["broad_spread_flag"])
        self.assertFalse(finding["persistent_candidate_slower"])


if __name__ == "__main__":
    unittest.main()
