import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from scripts.dashboard.engines.project_metrics_aggregator import (
    ProjectMetricsAggregator,
)


def transform_for_charts_like(data):
    """Local minimal transform that builds project_comparisons without Flask import."""
    pm = data.get("project_metrics", {}) or {}
    comparisons = []
    for pid, rec in pm.items():
        title = rec.get("title") or pid
        iph = float((rec.get("throughput") or {}).get("images_per_hour") or 0)
        base = rec.get("baseline") or {}
        overall_base = float(base.get("overall_iph_baseline") or 0)
        per_tool_base = base.get("per_tool") or {}
        tools = {}
        for tool, stats in (rec.get("tools") or {}).items():
            tools[tool] = {
                "iph": float(stats.get("images_per_hour") or 0),
                "baseline": float(per_tool_base.get(tool) or 0),
            }
        comparisons.append(
            {
                "projectId": pid,
                "title": title,
                "iph": iph,
                "baseline_overall": overall_base,
                "tools": tools,
                "startedAt": rec.get("startedAt"),
                "finishedAt": rec.get("finishedAt"),
            }
        )
    return {"project_comparisons": comparisons}


def _write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestProjectMetricsAggregator(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        # Create directory layout
        (self.root / "data" / "projects").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "file_operations_logs").mkdir(parents=True, exist_ok=True)

        # Two projects with known windows
        mojo1 = {
            "projectId": "mojo1",
            "title": "Mojo 1",
            "status": "finished",
            "startedAt": "2025-10-05T00:00:00",
            "finishedAt": "2025-10-06T00:00:00",
            "paths": {"root": "/abs/not/used"},
        }
        mojo2 = {
            "projectId": "mojo2",
            "title": "Mojo 2",
            "status": "finished",
            "startedAt": "2025-10-10T00:00:00",
            "finishedAt": "2025-10-10T12:00:00",
            "paths": {"root": "/abs/not/used"},
        }
        with open(self.root / "data" / "projects" / "mojo1.project.json", "w") as f:
            json.dump(mojo1, f)
        with open(self.root / "data" / "projects" / "mojo2.project.json", "w") as f:
            json.dump(mojo2, f)

        # File operation logs (time-window filtering will be used)
        # mojo1 within 2025-10-05..2025-10-06 (24h), crop=1000 -> 41.67 iph
        # mojo2 within 12h window, crop=600 -> 50 iph
        records = [
            {
                "type": "file_operation",
                "timestamp": "2025-10-05T10:00:00",
                "script": "image_version_selector",
                "operation": "crop",
                "file_count": 1000,
            },
            {
                "type": "file_operation",
                "timestamp": "2025-10-10T06:00:00",
                "script": "image_version_selector",
                "operation": "crop",
                "file_count": 600,
            },
        ]
        _write_jsonl(
            self.root / "data" / "file_operations_logs" / "ops_202510.log", records
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_aggregator_per_project_and_tools(self):
        agg = ProjectMetricsAggregator(self.root)
        data = agg.aggregate()
        self.assertIn("mojo1", data)
        self.assertIn("mojo2", data)

        m1 = data["mojo1"]
        m2 = data["mojo2"]

        # Overall images/hour
        self.assertAlmostEqual(
            m1["throughput"]["images_per_hour"], round(1000 / 24.0, 2)
        )
        self.assertAlmostEqual(
            m2["throughput"]["images_per_hour"], round(600 / 12.0, 2)
        )

        # Per-tool metrics exist with same numbers here
        self.assertIn("tools", m1)
        self.assertIn("image_version_selector", m1["tools"])
        self.assertAlmostEqual(
            m1["tools"]["image_version_selector"]["images_per_hour"],
            round(1000 / 24.0, 2),
        )

        # Baseline should be present (based on finished projects)
        self.assertIn("baseline", m1)
        self.assertIn("overall_iph_baseline", m1["baseline"])

    def test_transform_project_comparisons(self):
        # Synthesize minimal data package to go through transform
        agg = ProjectMetricsAggregator(self.root)
        project_metrics = agg.aggregate()

        raw = {
            "metadata": {"generated_at": datetime.now().isoformat()},
            "activity_data": {},
            "file_operations_data": {},
            "script_updates": [],
            "projects": [],
            "project_markers": {},
            "project_metrics": project_metrics,
            "project_kpi": {},
            "timing_data": {},
        }
        out = transform_for_charts_like(raw)
        comps = out.get("project_comparisons")
        self.assertIsInstance(comps, list)
        self.assertGreaterEqual(len(comps), 2)
        self.assertIn("iph", comps[0])


if __name__ == "__main__":
    unittest.main()
