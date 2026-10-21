# Copyright 2026 The Spyre-Inference Authors.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in host timing at existing eager model boundaries. No backend imports."""

import contextlib
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TIMESTAMPS = (
    "pipeline_start_ns",
    "forward_start_ns",
    "body_host_return_ns",
    "body_cpu_ready_ns",
    "head_start_ns",
    "cpu_logits_ready_ns",
)
SPANS = {
    "input_conversion_ms": (0, 1),
    "body_host_ms": (1, 2),
    "hidden_state_return_tail_ms": (2, 3),
    "cpu_row_selection_gap_ms": (3, 4),
    "head_ms": (4, 5),
    "pipeline_ms": (0, 5),
    "forward_to_cpu_logits_ms": (1, 5),
}


class ForwardTiming:
    """One pending forward per worker; metadata and export are outside the spans."""

    def __init__(self, directory: str | Path, rank: int, runner_type: str):
        self.rank = rank
        self.pid = os.getpid()
        self.session_id = f"{self.pid}-{time.time_ns()}"
        self.path = Path(directory) / f"forward-timing-rank-{rank}-{self.session_id}.jsonl"
        self.runner_type = runner_type
        self.scheduler: Any = None
        self.pending: dict[str, Any] | None = None
        self.enabled = True
        self.step = 0

    def begin(
        self,
        batch: Any,
        args: tuple,
        kwargs: dict,
        attention_metadata: Callable[[], Any],
    ) -> None:
        if not self.enabled or self.scheduler is None:
            return
        multiple_forwards = self.pending is not None
        if multiple_forwards:
            self.unsupported("multiple forwards in one scheduler execution")
            self.finish()
            if not self.enabled:
                return
        row: dict[str, Any] = dict(
            schema_version=1,
            clock="monotonic_ns",
            session_id=self.session_id,
            rank=self.rank,
            pid=self.pid,
            step=self.step,
            runner_type=self.runner_type,
            request_ids=[],
            query_lengths=[],
            computed_lengths=[],
            prompt_lengths=[],
            kv_lengths=[],
            phase="unsupported",
            actual_body_rows=None,
            padded_body_rows=None,
            actual_head_rows=None,
            padded_head_rows=None,
            attention=[],
            status="incomplete",
            valid=False,
            reason=None,
            error_type=None,
        )
        row.update(dict.fromkeys(TIMESTAMPS))
        row.update(dict.fromkeys(SPANS))
        self.pending = row
        self.step += 1
        try:
            ids = list(batch.req_ids[: batch.num_reqs])
            queries = [int(self.scheduler.num_scheduled_tokens[r]) for r in ids]
            indices = [batch.req_id_to_index[r] for r in ids]
            computed = [int(batch.num_computed_tokens_cpu[i]) for i in indices]
            prompts = [int(batch.num_prompt_tokens[i]) for i in indices]
            phases = [
                "prefill"
                if 0 <= c < p and 0 < q <= p - c
                else "decode"
                if c >= p > 0 and q == 1
                else "unsupported"
                for q, c, p in zip(queries, computed, prompts)
            ]
            phase = (
                "unsupported"
                if not phases or "unsupported" in phases
                else phases[0]
                if len(set(phases)) == 1
                else "mixed"
            )
            body_input = kwargs.get("input_ids", args[0] if args else None)
            if body_input is None:
                body_input = kwargs.get("inputs_embeds")
            row.update(
                request_ids=ids,
                query_lengths=queries,
                computed_lengths=computed,
                prompt_lengths=prompts,
                kv_lengths=[c + q for c, q in zip(computed, queries)],
                phase=phase,
                actual_body_rows=sum(queries),
                padded_body_rows=int(body_input.shape[0]) if body_input is not None else None,
            )
            if phase == "mixed":
                row.update(status="mixed", reason="prefill and decode in the same batch")
            elif phase == "unsupported":
                self.unsupported("request metadata is not ordinary prefill or decode")
            if self.runner_type != "generate":
                self.unsupported("runner does not use the generative CPU-logits path")
            if body_input is None:
                self.unsupported("no body input shape")

            metadata = attention_metadata()
            if not isinstance(metadata, dict):
                self.unsupported("attention metadata is not a layer mapping")
            else:
                seen = set()
                for name, value in metadata.items():
                    if id(value) in seen:
                        continue
                    seen.add(id(value))
                    entry: dict[str, Any] = {"representative_layer": name}
                    for field in (
                        "num_actual_tokens",
                        "num_seqs",
                        "max_query_len",
                        "max_seq_len",
                        "block_size",
                        "padded_num_seqs",
                        "padded_batch_blocks",
                        "blocks_per_chunk",
                    ):
                        item = getattr(value, field, None)
                        entry[field] = int(item) if item is not None else None
                    for field in ("aligned_query_lens", "padded_num_blocks"):
                        item = getattr(value, field, None)
                        entry[field] = [int(n) for n in item] if item is not None else None
                    blocks, size = entry["padded_num_blocks"], entry["block_size"]
                    entry["physical_kv_extent_tokens"] = (
                        [n * size for n in blocks] if blocks and size else None
                    )
                    row["attention"].append(entry)
        except Exception as error:
            self.unsupported(f"metadata unavailable: {type(error).__name__}")
        if multiple_forwards:
            self.unsupported("multiple forwards in one scheduler execution")
        self.mark("pipeline_start_ns")

    def mark(self, name: str) -> None:
        if self.pending is not None:
            self.pending[name] = time.monotonic_ns()

    def unsupported(self, reason: str) -> None:
        if self.pending is not None:
            self.pending.update(status="unsupported", reason=reason)

    def head_start(self, num_rows: int) -> None:
        if self.pending is not None:
            if self.pending["head_start_ns"] is not None:
                self.unsupported("multiple heads in one forward")
            self.pending["actual_head_rows"] = int(num_rows)
            if num_rows != len(self.pending["request_ids"]):
                self.unsupported("head row count differs from request count")
            self.mark("head_start_ns")

    def head_padding(self, padded_rows: int) -> None:
        if self.pending is not None:
            self.pending["padded_head_rows"] = int(padded_rows)

    def finish(self, error: BaseException | None = None) -> None:
        row, self.pending = self.pending, None
        if row is None:
            return
        try:
            stamps = [row[name] for name in TIMESTAMPS]
            if error is not None:
                row.update(status="error", error_type=type(error).__name__)
            elif row["status"] not in ("unsupported", "mixed"):
                if not all(isinstance(value, int) for value in stamps):
                    row.update(status="incomplete", reason="missing model or CPU-return boundary")
                elif not all(a <= b for a, b in zip(stamps, stamps[1:])):
                    row.update(status="invalid", reason="nonmonotonic boundaries")
                else:
                    row.update(status="complete", valid=True)
                    for name, (start, end) in SPANS.items():
                        row[name] = (stamps[end] - stamps[start]) / 1e6
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        except Exception:
            self.enabled = False
            # Diagnostics must never replace a model exception, even with a failing handler.
            with contextlib.suppress(Exception):
                logger.warning(
                    "Forward timing disabled: could not export to %s", self.path, exc_info=True
                )
