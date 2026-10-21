# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

"""Standard-library tests; runnable directly without importing the device stack."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def batch(queries=(2,), computed=(0,), prompts=(2,)):
    ids = [f"request-{i}-raw-suffix" for i in range(len(queries))]
    return (
        SimpleNamespace(
            num_reqs=len(ids),
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            num_computed_tokens_cpu=list(computed),
            num_prompt_tokens=list(prompts),
        ),
        SimpleNamespace(num_scheduled_tokens=dict(zip(ids, queries))),
    )


class ForwardTimingTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "spyre_inference/v1/worker/forward_timing.py"
        self.assertTrue(path.exists(), "the portable timing collector is missing")
        self.timing = load_file("forward_timing_test", path)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.collector = self.timing.ForwardTiming(self.directory, rank=3, runner_type="generate")

    def begin(self, queries=(2,), computed=(0,), prompts=(2,), **kwargs):
        prepared, self.collector.scheduler = batch(queries, computed, prompts)
        self.collector.begin(
            prepared,
            (),
            {"input_ids": SimpleNamespace(shape=(8,))},
            lambda: kwargs.get("attention", {}),
        )

    def boundaries(self):
        for name in ("forward_start_ns", "body_host_return_ns", "body_cpu_ready_ns"):
            self.collector.mark(name)
        self.collector.head_start(len(self.collector.pending["request_ids"]))
        self.collector.head_padding(4)
        self.collector.mark("cpu_logits_ready_ns")

    def row(self):
        files = list(self.directory.glob("*.jsonl"))
        self.assertEqual(len(files), 1)
        return json.loads(files[0].read_text().splitlines()[-1])

    def test_complete_row_has_measured_spans_and_prepared_metadata(self):
        with patch.object(
            self.timing.time,
            "monotonic_ns",
            side_effect=[0, 2000000, 5000000, 9000000, 10000000, 16000000],
        ):
            self.begin()
            self.boundaries()
            self.assertEqual(list(self.directory.iterdir()), [])
            self.collector.finish()
        row = self.row()
        self.assertEqual(row["rank"], 3)
        self.assertEqual(row["request_ids"], ["request-0-raw-suffix"])
        self.assertEqual(row["query_lengths"], [2])
        self.assertEqual(row["kv_lengths"], [2])
        self.assertEqual(row["padded_body_rows"], 8)
        self.assertEqual(row["actual_head_rows"], 1)
        self.assertEqual(row["padded_head_rows"], 4)
        self.assertEqual(row["status"], "complete")
        self.assertEqual(row["input_conversion_ms"], 2)
        self.assertEqual(row["body_host_ms"], 3)
        self.assertEqual(row["hidden_state_return_tail_ms"], 4)
        self.assertEqual(row["cpu_row_selection_gap_ms"], 1)
        self.assertEqual(row["head_ms"], 6)
        self.assertEqual(row["pipeline_ms"], 16)
        self.assertEqual(row["forward_to_cpu_logits_ms"], 14)
        self.assertIsNone(self.collector.pending)

    def test_classifies_single_token_prefill_chunked_prefill_decode_and_mixed(self):
        for queries, computed, prompts, phase in (
            ((1,), (0,), (1,), "prefill"),
            ((2,), (4,), (8,), "prefill"),
            ((1,), (8,), (8,), "decode"),
            ((1, 1), (0, 8), (1, 8), "mixed"),
            ((2,), (8,), (8,), "unsupported"),
            ((0,), (0,), (8,), "unsupported"),
            ((2,), (7,), (8,), "unsupported"),
        ):
            with self.subTest(phase=phase, queries=queries):
                self.begin(queries, computed, prompts)
                self.boundaries()
                self.collector.finish()
                row = self.row()
                self.assertEqual(row["phase"], phase)
                self.assertEqual(row["valid"], phase in ("prefill", "decode"))

    def test_warmup_without_scheduler_does_not_read_clock_or_write(self):
        prepared, _ = batch()
        with patch.object(self.timing.time, "monotonic_ns", side_effect=AssertionError("warmup")):
            self.collector.begin(prepared, (), {}, lambda: {})
            self.collector.mark("forward_start_ns")
            self.collector.head_start(1)
            self.collector.finish()
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_missing_boundary_and_error_never_create_cpu_ready_latency(self):
        self.begin()
        self.collector.mark("forward_start_ns")
        self.collector.finish()
        self.assertEqual(self.row()["status"], "incomplete")
        self.assertIsNone(self.row()["pipeline_ms"])
        self.begin()
        self.boundaries()
        self.collector.finish(error=ValueError("model failure"))
        self.assertEqual(self.row()["status"], "error")
        self.assertEqual(self.row()["error_type"], "ValueError")
        self.assertIsNone(self.row()["pipeline_ms"])

    def test_metadata_failure_and_device_retention_are_explicit(self):
        self.collector.scheduler = SimpleNamespace(num_scheduled_tokens={})
        self.collector.begin(SimpleNamespace(), (), {}, lambda: {})
        self.collector.finish()
        self.assertEqual(self.row()["status"], "unsupported")
        self.assertIn("metadata", self.row()["reason"])
        self.begin()
        self.collector.unsupported("hidden states remain on device")
        self.collector.finish()
        self.assertEqual(self.row()["status"], "unsupported")
        self.assertIsNone(self.row()["body_cpu_ready_ns"])

    def test_pooling_and_extra_head_rows_are_unsupported(self):
        self.collector.runner_type = "pooling"
        self.begin()
        self.collector.finish()
        self.assertEqual(self.row()["status"], "unsupported")
        self.collector.runner_type = "generate"
        self.begin()
        self.collector.head_start(2)
        self.collector.finish()
        self.assertEqual(self.row()["actual_head_rows"], 2)
        self.assertEqual(self.row()["status"], "unsupported")

    def test_nonmonotonic_boundaries_are_invalid(self):
        with patch.object(self.timing.time, "monotonic_ns", side_effect=[0, 3, 2, 4, 5, 6]):
            self.begin()
            self.boundaries()
            self.collector.finish()
        self.assertEqual(self.row()["status"], "invalid")
        self.assertIsNone(self.row()["body_host_ms"])

    def test_export_failure_disables_collection_without_raising(self):
        self.begin()
        self.boundaries()
        with (
            patch.object(Path, "open", side_effect=OSError("unwritable")),
            self.assertLogs(self.timing.__name__, level="WARNING"),
        ):
            self.collector.finish(error=RuntimeError("original"))
        self.assertFalse(self.collector.enabled)
        self.begin()
        self.assertIsNone(self.collector.pending)

    def test_attention_geometry_uses_host_metadata_and_deduplicates_shared_layers(self):
        metadata = SimpleNamespace(block_size=128, padded_num_blocks=[2], aligned_query_lens=[8])
        self.begin(attention={"layer0": metadata, "layer1": metadata})
        self.boundaries()
        self.collector.finish()
        attention = self.row()["attention"]
        self.assertEqual(len(attention), 1)
        self.assertEqual(attention[0]["physical_kv_extent_tokens"], [256])

    def test_multiple_forwards_in_one_execution_are_all_unsupported(self):
        self.begin()
        self.boundaries()
        self.begin()
        self.boundaries()
        self.collector.finish()
        path = next(self.directory.glob("*.jsonl"))
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["status"] for row in rows], ["unsupported", "unsupported"])
        self.assertTrue(all(row["pipeline_ms"] is None for row in rows))

    def test_multiple_heads_do_not_overwrite_a_valid_sample(self):
        self.begin()
        self.boundaries()
        self.collector.head_start(1)
        self.collector.mark("cpu_logits_ready_ns")
        self.collector.finish()
        self.assertEqual(self.row()["status"], "unsupported")
        self.assertIsNone(self.row()["head_ms"])


class SummaryTests(unittest.TestCase):
    def test_groups_ranks_geometry_and_filters_requests_and_invalid_rows(self):
        path = ROOT / "scripts/summarize_forward_timing.py"
        self.assertTrue(path.exists(), "the portable summary CLI is missing")
        summary = load_file("forward_timing_summary_test", path)
        rows = []
        for rank, request, duration, phase, padded in (
            (0, "warmup", 999, "decode", 8),
            (0, "real", 10, "decode", 8),
            (0, "real", 20, "decode", 8),
            (0, "real", 40, "decode", 16),
            (1, "warmup", 999, "decode", 8),
            (1, "real", 30, "decode", 8),
            (0, "prefill", 50, "prefill", 8),
            (0, "bad", 999, "mixed", 8),
        ):
            row = dict(
                schema_version=1,
                rank=rank,
                pid=rank + 100,
                session_id=str(rank),
                request_ids=[request],
                valid=phase != "mixed",
                status=phase if phase == "mixed" else "complete",
                phase=phase,
                query_lengths=[1],
                actual_body_rows=1,
                padded_body_rows=padded,
                actual_head_rows=1,
                padded_head_rows=1,
                attention=[],
            )
            row.update({name: duration for name in summary.SPANS})
            rows.append(row)
        result = summary.summarize(rows, skip_first_requests=1)
        self.assertEqual(len(result["groups"]), 4)
        group = next(
            g
            for g in result["groups"]
            if g["rank"] == 0 and g["phase"] == "decode" and g["padded_body_rows"] == 8
        )
        self.assertEqual(
            group["spans"]["pipeline_ms"], {"count": 2, "mean": 15, "median": 15, "p95": 20}
        )
        self.assertEqual(result["excluded"], {"request_filter": 2, "mixed": 1})

    def test_cli_reports_the_location_of_a_partial_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forward-timing-partial.jsonl"
            path.write_text('{"schema_version": 0}\n{"partial":')
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    str(ROOT / "scripts/summarize_forward_timing.py"),
                    str(path),
                ],
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(f"{path}:2", result.stderr)


if __name__ == "__main__":
    unittest.main()
