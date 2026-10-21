# Forward timing

Set `SPYRE_FORWARD_TIMING_DIR` in the worker environment to record fine-grained
forward timing from ordinary `vllm serve` or offline `LLM` calls. It is unset by
default. The feature uses host monotonic timestamps at existing eager boundaries;
it adds no synchronization, tensor copies, model calls, or compilation boundaries.

Install this checkout into an already compatible environment without updating
the installed stack:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

Use the interpreter/environment that already contains the compatible vLLM and
torch-spyre stack. If using `uv run` with local dependencies, use `uv run --no-sync`.

## Server and benchmark

In the server shell, set the model and timing directory before starting vLLM:

```bash
export MODEL=ibm-granite/granite-3.3-8b-instruct
export SPYRE_FORWARD_TIMING_DIR="$PWD/forward-timing"
vllm serve "$MODEL" --tensor-parallel-size 4 --max-model-len 8192 \
  --max-num-seqs 4 --max-num-batched-tokens 512 --no-enable-prefix-caching
```

Run the ordinary client benchmark in another shell with the same model name:

```bash
export MODEL=ibm-granite/granite-3.3-8b-instruct
vllm bench serve --model "$MODEL" --dataset-name random \
  --random-input-len 64 --random-output-len 3 --num-prompts 30 \
  --max-concurrency 1 --ignore-eos
```

The environment variable belongs on the server, where the workers execute.
These dimensions match the Granite timing investigation. Random benchmark
prompts differ from its fixed-token fixture, so this is not a bit-identical
workload. TP1 is also supported. Runtime scheduling flags remain independently
configurable in the server environment.
Each worker writes its own `forward-timing-rank-<rank>-<pid>-<session>.jsonl`.
Distributed workers need a writable directory on each host; collect their files
before summarizing. Rank comes from the worker's parallel configuration.

Offline inference uses the same setting:

```bash
SPYRE_FORWARD_TIMING_DIR="$PWD/forward-timing-offline" python - <<'PY'
from vllm import LLM, SamplingParams

llm = LLM(
    model="ibm-granite/granite-3.3-8b-instruct",
    tensor_parallel_size=1,
    max_model_len=8192,
    max_num_seqs=4,
    max_num_batched_tokens=512,
    enable_prefix_caching=False,
)
outputs = llm.generate(["Describe a compiler in one sentence."],
                       SamplingParams(temperature=0, max_tokens=32))
print(outputs[0].outputs[0].text)
PY
```

## Summarize

The summary script requires only Python's standard library, so files can be
copied to another machine for analysis:

```bash
python scripts/summarize_forward_timing.py ./forward-timing > summary.json
```

Pass directories or individual JSONL files. Summaries separate prefill and decode,
worker sessions, PIDs, ranks, query lengths, actual/padded body and head rows, and
padded attention geometry. They report count, mean, median, and nearest-rank p95
in milliseconds. Tensor-parallel ranks are never pooled as independent samples.
Logical computed/KV lengths remain in the raw rows; decode steps within the same
padding geometry can have different logical context lengths.

Startup model warmups have no active scheduler context and produce no rows.
Benchmark validation requests and user-submitted warmups are real requests and do
produce rows. If the first request in each worker session was a warmup, exclude it:

```bash
python scripts/summarize_forward_timing.py ./forward-timing --skip-first-requests 1
```

`--skip-first-requests N` counts distinct raw request IDs in first-seen order per
worker session. `--skip-request-id ID` excludes a specific ID and can be repeated.
Any batch containing an excluded request is excluded in full. Request IDs are
preserved verbatim, including vLLM suffixes. A reused ID cannot be distinguished
by these filters; use a separate timing directory/session when needed. No request
history is retained in the live collector.

## What the spans measure

| Field | Boundaries and included work |
| --- | --- |
| `input_conversion_ms` | `_model_forward` entry after metadata capture to the inner body call, including existing wrapper input conversion and cache reset. |
| `body_host_ms` | Inner body call to its host return. Device work can still be outstanding at this point; this is host submission duration. |
| `hidden_state_return_tail_ms` | Body host return to `_model_forward` return after the existing hidden-state CPU conversion. Includes outstanding device work, conversion/D2H, and wrapper bookkeeping. |
| `cpu_row_selection_gap_ms` | CPU hidden-state return to head entry, including upstream row selection and context/postprocessing overhead. |
| `head_ms` | Head entry to CPU-logits return, including row padding, selected-row H2D, projection, TP gather/assembly, existing logits D2H, and trimming. |
| `pipeline_ms` | Input-conversion start through CPU-logits return. |
| `forward_to_cpu_logits_ms` | Inner body call start through CPU-logits return. |

These are worker spans, not client TTFT or ITL. Scheduling, earlier input
preparation (including multimodal embedding), sampling, networking, and streaming
are outside `pipeline_ms`. The return tail is not a pure DMA measurement, and the
head span is not just the projection kernel. Use a device profiler for device
compute attribution.

## Rows and limitations

Schema version 1 records `session_id`, `rank`, `pid`, a zero-based worker `step`,
raw `request_ids`, `query_lengths`, `computed_lengths`, `prompt_lengths`, and
`kv_lengths` (computed plus scheduled). These lengths are captured from the
prepared input batch at `_model_forward` entry. Actual/padded body and head rows
and unique attention metadata entries describe the geometry. Head counts are
observed at the head call; they are not inferred from the body bucket.

Six `monotonic_ns` fields retain the raw boundaries: `pipeline_start_ns`,
`forward_start_ns`, `body_host_return_ns`, `body_cpu_ready_ns`, `head_start_ns`,
and `cpu_logits_ready_ns`. Compare intervals within a row; clocks in separate
processes/hosts are not an inter-rank synchronization mechanism.

`phase` is `prefill` when scheduled tokens remain within the prompt, including
one-token and chunked prefills. Ordinary decode has one scheduled token per
request after its prompt. Batches combining the two are `mixed`; speculative or
otherwise incompatible metadata is `unsupported`. A complete row must have
ordered timestamps, one head row per request, and existing CPU hidden-state and
logits returns. Pooling, device-retained or non-tensor outputs, missing boundaries,
and failed forwards cannot report valid CPU-ready latency. `status`, `reason`,
`error_type`, and `valid` explain exclusions; invalid rows keep null durations.
The summary reports excluded counts and never silently averages these rows.
Multiple forwards in one scheduler execution or multiple heads in a forward are
also unsupported; their scheduler-wide metadata cannot identify ordinary samples.

Only one row is pending in each worker. Completed rows are serialized and the
file is closed after `execute_model`, outside all measured spans. This avoids
unbounded buffering and preserves completed rows on ordinary process termination.
Abrupt termination during a write may leave a partial last line; the summary
reports its file and line rather than silently dropping it. Export failures warn
and disable collection in that worker while preserving model results and errors.

Timing still has an observer effect: timestamp/metadata bookkeeping costs CPU
time, and JSONL writes can delay the next request or token even though writes are
outside the recorded spans. Use a local filesystem when practical and compare
client performance with an untimed run. When the variable is unset or empty,
there is no collector, metadata collection, timing clock access, or file I/O.
The hooks cover the current `TorchSpyreModelRunner`; the separate MRV2 runner is
not instrumented.
