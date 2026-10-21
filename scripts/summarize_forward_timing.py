#!/usr/bin/env python3
# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

"""Summarize worker JSONL timings without importing vLLM, torch, or the plugin."""

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

SPANS = (
    "input_conversion_ms",
    "body_host_ms",
    "hidden_state_return_tail_ms",
    "cpu_row_selection_gap_ms",
    "head_ms",
    "pipeline_ms",
    "forward_to_cpu_logits_ms",
)
GEOMETRY = (
    "query_lengths",
    "actual_body_rows",
    "padded_body_rows",
    "actual_head_rows",
    "padded_head_rows",
)
ATTENTION_GEOMETRY = (
    "block_size",
    "padded_num_seqs",
    "padded_batch_blocks",
    "blocks_per_chunk",
    "aligned_query_lens",
    "padded_num_blocks",
    "physical_kv_extent_tokens",
)


def summarize(rows, skip_first_requests=0, skip_request_ids=()):
    groups = {}
    excluded = Counter()
    first_ids = defaultdict(set)
    for row in rows:
        if row.get("schema_version") != 1:
            excluded["unsupported_schema"] += 1
            continue
        worker = (row["session_id"], row["rank"], row["pid"])
        first = first_ids[worker]
        for request in row["request_ids"]:
            if len(first) < skip_first_requests:
                first.add(request)
        if any(r in first or r in skip_request_ids for r in row["request_ids"]):
            excluded["request_filter"] += 1
            continue
        if not row.get("valid") or row.get("phase") not in ("prefill", "decode"):
            excluded[row.get("status", "invalid")] += 1
            continue
        if any(
            not isinstance(row.get(s), (float, int)) or not math.isfinite(row[s]) or row[s] < 0
            for s in SPANS
        ):
            excluded["invalid_spans"] += 1
            continue
        geometry = {key: row.get(key) for key in GEOMETRY}
        geometry["attention_geometry"] = [
            {key: entry.get(key) for key in ATTENTION_GEOMETRY}
            for entry in row.get("attention", [])
        ]
        label = dict(
            session_id=worker[0], rank=worker[1], pid=worker[2], phase=row["phase"], **geometry
        )
        key = json.dumps(label, sort_keys=True)
        if key not in groups:
            groups[key] = (label, {span: [] for span in SPANS})
        for span in SPANS:
            groups[key][1][span].append(row[span])
    result = []
    for key in sorted(groups):
        label, spans = groups[key]
        result.append(
            dict(
                label,
                spans={
                    name: dict(
                        count=len(values),
                        mean=statistics.mean(values),
                        median=statistics.median(values),
                        p95=sorted(values)[math.ceil(0.95 * len(values)) - 1],
                    )
                    for name, values in spans.items()
                },
            )
        )
    return dict(groups=result, excluded=dict(excluded))


def read_rows(paths):
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="JSONL files or timing directories")
    parser.add_argument(
        "--skip-first-requests",
        type=int,
        default=0,
        help="Exclude the first N distinct request IDs in each worker session",
    )
    parser.add_argument(
        "--skip-request-id",
        action="append",
        default=[],
        help="Exclude any batch containing this exact raw request ID; repeatable",
    )
    args = parser.parse_args()
    if args.skip_first_requests < 0:
        parser.error("--skip-first-requests must be nonnegative")
    paths = sorted(
        {
            file
            for path in args.paths
            for file in (path.glob("forward-timing-*.jsonl") if path.is_dir() else [path])
        }
    )
    if not paths:
        parser.error("no forward-timing JSONL files found")
    try:
        result = summarize(read_rows(paths), args.skip_first_requests, set(args.skip_request_id))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
