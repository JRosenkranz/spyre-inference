# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trace KV writes, eligible fresh-prefill and small pure-decode attention into the block.

``install()``, called from the attention metadata builder, binds the forward below onto
each eligible layer instance; every other ``Attention`` keeps upstream's forward and its
``unified_kv_cache_update`` op. Fresh prompts use the encoder's rectangle kernel on
this step's own K/V; eligible pure-decode steps read paged K/V inside the block graph.
Other steps keep the opaque core, whose per-sequence loop cannot be captured with
``fullgraph=True``.
"""

import collections
import contextlib
import types
import weakref
from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import torch
from vllm.distributed.kv_transfer import has_kv_transfer_group
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import Attention
from vllm.utils.torch_utils import _encode_layer_name
from vllm.v1.attention.backend import AttentionType

from spyre_inference import envs
from spyre_inference.custom_ops.utils import convert

if TYPE_CHECKING:
    from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionMetadata
    from spyre_inference.v1.attention.spyre_attn_bucketer import SpyreAttnBucketer

logger = init_logger(__name__)

# vLLM reserves block 0 as `BlockPool.null_block`, so no sequence is ever given its
# slots. `index_copy_` has no skip index, so they absorb writes with nowhere to go.
_NULL_SLOT = 0

# Set by the model runner before install() reads it. Defaults to refusing: a block compiled
# without fullgraph graph-breaks inside for_each_tile silently rather than failing.
outer_graph_fullgraph: bool = False

_DECODE_PATH_LOG_EVERY = 256


class SlotMapping:
    """This step's slot mapping on device, shared by every split layer."""

    def __init__(self, layers: list[Attention]) -> None:
        self._layers = layers
        self._device: torch.device | None = None
        # One tensor, or one per KV head: whatever the impl's KV store indexes by.
        self.slots: torch.Tensor | list[torch.Tensor] | None = None
        self.inline_prefill = False
        # [1, 1, padded Q, declared KV] mask; None keeps the opaque core.
        self.prefill_mask: torch.Tensor | None = None
        self.prefill_bucketer: SpyreAttnBucketer | None = None
        self._prefill_warmup_tokens: int | None = None
        # Keep only the last batch layout's mask, so device memory stays at one mask.
        self._prefill_mask_key: tuple | None = None
        self._prefill_mask_dev: torch.Tensor | None = None

    def _resolve_device(self) -> torch.device | None:
        if self._device is None:
            # `install` runs before bind_kv_cache, so a layer whose cache never arrives
            # still has the empty default and indexing it would raise.
            self._layers = [layer for layer in self._layers if len(layer.kv_cache) > 0]
            if not self._layers:
                return None
            self._device = self._layers[0].kv_cache[0].device
            # Must exist before tracing; see SpyreAttentionImpl.kv_slot_views.
            for layer in self._layers:
                layer.impl.kv_slot_views(layer.kv_cache)  # ty: ignore[possibly-missing-attribute]
        return self._device

    def _write_index(self, slot_mapping: torch.Tensor, device: torch.device):
        """The device-side KV store index in this group's cache layout."""
        return self._layers[0].impl.kv_write_index(slot_mapping, device)  # ty: ignore[possibly-missing-attribute]

    def resolve(self) -> tuple[torch.device, list[Attention]] | None:
        """This group's device and cache-bound layers, or None before the caches arrive."""
        device = self._resolve_device()
        return None if device is None else (device, self._layers)

    def publish(self, slot_mapping: torch.Tensor) -> None:
        """Mirror a step's host slot mapping to device for the traced write to read."""
        device = self._resolve_device()
        if device is None:
            return
        self.slots = self._write_index(slot_mapping.clamp(min=_NULL_SLOT), device)

    def _prefill_kv_extent(self, logical_tokens: int, num_tokens: int) -> int | None:
        if logical_tokens == num_tokens:
            return num_tokens
        if not 1 < logical_tokens < num_tokens or self.prefill_bucketer is None:
            return None
        bucket = self.prefill_bucketer.find_kv_bucket(logical_tokens)
        if bucket is None or not logical_tokens <= bucket < num_tokens or bucket % 64:
            return None
        return bucket

    def publish_prefill(self, query_lens: list[int] | None, num_tokens: int) -> None:
        """Publish this step's fresh-prompt mask, or clear it (``query_lens=None``).

        A KV connector saves pages through the opaque op, so its steps keep that op.
        """
        if query_lens is not None and self._prefill_warmup_tokens is not None:
            query_lens = [self._prefill_warmup_tokens]
        logical_tokens = sum(query_lens) if query_lens is not None else 0
        kv_tokens = self._prefill_kv_extent(logical_tokens, num_tokens)
        if (
            query_lens is None
            or any(n <= 0 for n in query_lens)
            or not 1 < logical_tokens <= num_tokens <= INLINE_PREFILL_MAX_TOKENS
            or kv_tokens is None
            or (logical_tokens != num_tokens and len(query_lens) != 1)
            or not self.inline_prefill
            or has_kv_transfer_group()
            or self._resolve_device() is None
        ):
            self.prefill_mask = None
            return
        key = (tuple(query_lens), num_tokens, kv_tokens)
        if key != self._prefill_mask_key:
            dtype = self._layers[0].impl.model_dtype  # ty: ignore[possibly-missing-attribute]
            mask = prefill_block_causal_mask(query_lens, num_tokens, dtype, kv_tokens)
            self._prefill_mask_dev = convert(mask, self._device)
            self._prefill_mask_key = key
        self.prefill_mask = self._prefill_mask_dev

    def publish_null(self, num_tokens: int, inline_prefill: bool = False) -> None:
        device = self._resolve_device()
        if device is None:
            return
        self.slots = self._write_index(
            torch.full((num_tokens,), _NULL_SLOT, dtype=torch.int64), device
        )
        self.publish_prefill([num_tokens] if inline_prefill else None, num_tokens)


# The mask and the score tensor grow as T * T; 512 is the only length checked on device.
INLINE_PREFILL_MAX_TOKENS = 512


def prefill_block_causal_mask(
    query_lens: list[int], num_tokens: int, dtype: torch.dtype, kv_tokens: int | None = None
) -> torch.Tensor:
    """Additive ``[1, 1, Q, KV]`` mask with sequence-local causal visibility.

    Square masks retain self-attending padded rows. With compact KV, discarded
    padded queries attend key zero so no row is fully masked. ``finfo.min / 2``
    follows ``encoder_key_pad_mask``.
    """
    if kv_tokens is not None and kv_tokens < num_tokens:
        logical_tokens = sum(query_lens)
        if not 1 < logical_tokens <= kv_tokens < num_tokens:
            raise ValueError("compact KV must cover the complete logical fresh batch")
        seq = torch.full((kv_tokens,), -1, dtype=torch.int32)
        start = 0
        for index, length in enumerate(query_lens):
            seq[start : start + length] = index
            start += length
        q_pos = torch.arange(num_tokens).clamp(max=logical_tokens - 1)
        k_pos = torch.arange(kv_tokens)
        # Valid rows keep block causality; discarded query padding needs one
        # finite source even after its former self-key is outside the prefix.
        attend = (seq[q_pos, None] == seq[None, :]) & (k_pos[None, :] <= q_pos[:, None])
        attend &= k_pos[None, :] < logical_tokens
        attend[logical_tokens:] = False
        attend[logical_tokens:, 0] = True
        return (
            torch.where(attend, 0.0, torch.finfo(dtype).min / 2)
            .to(dtype)
            .view(1, 1, num_tokens, kv_tokens)
        )
    seq = torch.full((num_tokens,), -1, dtype=torch.int32)
    start = 0
    for s, n in enumerate(query_lens):
        seq[start : start + n] = s
        start += n
    pos = torch.arange(num_tokens)
    causal = pos.unsqueeze(0) <= pos.unsqueeze(1)
    same_seq = (seq.unsqueeze(1) == seq.unsqueeze(0)) & (seq >= 0).unsqueeze(1)
    attend = (same_seq & causal) | torch.eye(num_tokens, dtype=torch.bool)
    mask = torch.where(
        attend,
        torch.zeros((), dtype=dtype),
        torch.tensor(torch.finfo(dtype).min / 2, dtype=dtype),
    )
    return mask.view(1, 1, num_tokens, num_tokens).contiguous()


class DecodeGrid:
    """This step's batched-decode index tensors on device, or None off the inline path."""

    def __init__(self, slots: SlotMapping) -> None:
        self._slots = slots
        # Whether any layer traces from this grid; otherwise only the mirror runs.
        self.inline = False
        self.steps: collections.Counter[str] = collections.Counter()
        self.opaque_keys: collections.Counter[tuple] = collections.Counter()
        self.rep_row_ids: torch.Tensor | None = None
        self.page_ids: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None

    def publish(self, attn_metadata: "SpyreAttentionMetadata") -> None:
        self.rep_row_ids = self.page_ids = self.mask = None
        path = self._publish(attn_metadata)
        if self.inline:
            self._count(path, attn_metadata)

    def _publish(self, attn_metadata: "SpyreAttentionMetadata") -> str:
        pure_decode = attn_metadata.num_decode_seqs == attn_metadata.num_seqs
        per_seq = "per_seq" if pure_decode else "mixed"
        if attn_metadata.padded_num_seqs is None:
            return per_seq
        resolved = self._slots.resolve()
        if resolved is None:
            return per_seq
        device, layers = resolved
        num_pages = layers[0].kv_cache[0].shape[0]
        impl = next(
            (
                layer.impl
                for layer in layers
                if layer.impl._batched_decode_preconditions_met(attn_metadata, num_pages)  # ty: ignore[possibly-missing-attribute]
            ),
            None,
        )
        if impl is None:
            return per_seq
        impl._mirror_batched_decode_indices(attn_metadata, device)  # ty: ignore[possibly-missing-attribute]
        assert attn_metadata.blocks_per_chunk is not None
        if not pure_decode:
            return "mixed"
        b_seqs, bpc = attn_metadata.padded_num_seqs, attn_metadata.blocks_per_chunk
        if not self.inline or not impl.inline_batched_decode(b_seqs, bpc):  # ty: ignore[possibly-missing-attribute]
            return "batched_opaque"
        self.rep_row_ids = attn_metadata.rep_row_ids_dev
        self.page_ids = attn_metadata.chunk_page_ids_dev
        self.mask = attn_metadata.mask_by_chunk_dev
        return "inline"

    def _count(self, path: str, attn_metadata: "SpyreAttentionMetadata") -> None:
        from spyre_inference.v1.worker import compile_guard

        # Armed right after warmup, so it stands in for "serving now" (#1109 removed
        # is_warmup_complete).
        if not compile_guard.is_armed():
            return
        self.steps[path] += 1
        if path == "batched_opaque":
            self.opaque_keys[(attn_metadata.padded_num_seqs, attn_metadata.blocks_per_chunk)] += 1
        total = sum(self.steps.values())
        if total % _DECODE_PATH_LOG_EVERY == 0:
            logger.info(
                "Decode attention over %d steps: %s; opaque batched (num_seqs, "
                "blocks_per_chunk): %s.",
                total,
                dict(self.steps),
                dict(self.opaque_keys),
            )


_holders: weakref.WeakSet[SlotMapping] = weakref.WeakSet()
_grids: weakref.WeakSet[DecodeGrid] = weakref.WeakSet()


def publish_null_slots(
    num_tokens: int, inline_prefill: bool = False, keep_decode_grids: bool = False
) -> None:
    """Point every token at the null block ahead of a run that builds no metadata.

    Warmup would otherwise trace a second graph without the KV write, and a dummy run
    after real inference would scatter into whichever step's slots ran last. The
    fresh-prompt mask clears too. ``inline_prefill`` publishes a one-prompt mask
    instead, so that warmup compiles the variant that a fresh-prompt step selects.
    Decode grids clear unless decode warmup has just published them.
    """
    for holder in _holders:
        holder.publish_null(num_tokens, inline_prefill)
    if not keep_decode_grids:
        clear_decode_grids()


def warms_inline_prefill(num_tokens: int) -> bool:
    """Whether warmup must also compile the fresh-prompt variant at this body bucket."""
    return (
        1 < num_tokens <= INLINE_PREFILL_MAX_TOKENS
        and not has_kv_transfer_group()
        and any(h.inline_prefill for h in _holders)
    )


def inline_prefill_warmup_tokens(num_tokens: int) -> list[int]:
    """Representative logical lengths for each group's declared compact KV shapes."""
    holders = [h for h in _holders if h.inline_prefill]
    candidates = {num_tokens, 2}
    for holder in holders:
        if holder.prefill_bucketer is not None:
            candidates.update(k + 1 for k in holder.prefill_bucketer.kv_buckets)
    variants = {}
    for logical in sorted(candidates, reverse=True):
        if not 1 < logical <= num_tokens:
            continue
        shape = tuple(h._prefill_kv_extent(logical, num_tokens) for h in holders)
        if all(extent is None for extent in shape):
            continue
        variants.setdefault(shape, logical)
    return list(variants.values())


@contextlib.contextmanager
def inline_prefill_warmup(logical_tokens: int):
    holders = list(_holders)
    try:
        for holder in holders:
            holder._prefill_warmup_tokens = logical_tokens
        yield
    finally:
        for holder in holders:
            holder._prefill_warmup_tokens = None
            holder.prefill_mask = None


def clear_prefill_masks() -> None:
    for holder in _holders:
        holder.prefill_mask = None


def clear_decode_grids() -> None:
    for grid in _grids:
        grid.rep_row_ids = grid.page_ids = grid.mask = None


def _spyre_attention_forward(
    self: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output_shape: torch.Size | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    mask = cast(SlotMapping, self.spyre_slots).prefill_mask
    grid = cast(DecodeGrid | None, getattr(self, "spyre_decode_grid", None))
    # Both inline paths return [tokens, num_heads * head_size] at the query's dtype.
    if output_shape is not None or output_dtype not in (None, query.dtype):
        mask = None
        grid = None
    if output_dtype is None:
        output_dtype = query.dtype
    if output_shape is None:
        output_shape = torch.Size((query.shape[0], self.num_heads * self.head_size_v))
    output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
    hidden_size = output_shape[-1]

    query = query.view(-1, self.num_heads, self.head_size)
    output = output.view(-1, self.num_heads, self.head_size_v)
    if key is not None:
        key = key.view(-1, self.num_kv_heads, self.head_size)
    if value is not None:
        value = value.view(-1, self.num_kv_heads, self.head_size_v)

    dep = None
    slots = cast(SlotMapping, self.spyre_slots).slots
    if slots is not None and key is not None and value is not None:
        # `dep` makes "scatter before read" a real data dependency, which is otherwise
        # invisible because the op reaches its cache through the forward context.
        dep = self.impl.do_kv_cache_update(self, key, value, self.kv_cache, slots)

    if (
        dep is not None
        and mask is not None
        and key is not None
        and value is not None
        and mask.shape[-2] == query.shape[0]
        and 0 < mask.shape[-1] <= min(key.shape[0], value.shape[0])
    ):
        # This reads the step's own K and V, not the pages, so it does not wait for the
        # scatter. The scatter above still writes the cache for later steps.
        from spyre_inference.v1.attention.backends.spyre_encoder_attn import (
            _encoder_rect_kernel,
        )

        kv_tokens = mask.shape[-1]
        if kv_tokens < key.shape[0]:
            key = key[:kv_tokens]
        if kv_tokens < value.shape[0]:
            value = value[:kv_tokens]
        attn = _encoder_rect_kernel(
            query,
            key,
            value,
            mask,
            self.impl.scale,
            1,
            query.shape[0],
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            kv_extent=kv_tokens,
        )
        return attn.reshape(-1, hidden_size)

    if grid is not None and grid.mask is not None and dep is not None:
        return _inline_batched_decode(self, grid, query).view(-1, hidden_size)

    # Staged here, not in the impl: the copies are then traced into the block
    # graph, which warmup already compiles at every token bucket.
    staging = getattr(self.impl, "staging_buffers", None)
    buffers = staging(query.device) if staging is not None else None
    rows = query.shape[0]
    if buffers is None:
        q_in, out_buf = query, output
    else:
        q_in, out_buf = buffers
        q_in[:rows] = query

    torch.ops.vllm.unified_attention_with_output(
        q_in,  # ty: ignore[invalid-argument-type]
        key,  # ty: ignore[invalid-argument-type]
        value,  # ty: ignore[invalid-argument-type]
        out_buf,  # ty: ignore[invalid-argument-type]
        _encode_layer_name(self.layer_name),  # ty: ignore[invalid-argument-type]
        kv_cache_dummy_dep=dep,  # ty: ignore[invalid-argument-type]
    )
    if buffers is not None:
        output.copy_(out_buf[:rows])
    return output.view(-1, hidden_size)


def _inline_batched_decode(self: Attention, grid: DecodeGrid, query: torch.Tensor) -> torch.Tensor:
    # Static arguments come from the grid's shapes, which Dynamo guards on; pages are read
    # through the views the KV write scattered into, so the graph mutates one input.
    impl = self.impl
    mask, rep_row_ids = cast(torch.Tensor, grid.mask), cast(torch.Tensor, grid.rep_row_ids)
    b_seqs = mask.shape[-4]
    k_rows, v_rows = impl.kv_slot_views(self.kv_cache)  # ty: ignore[possibly-missing-attribute]
    page_shape = self.kv_cache[0].shape
    # Staged even here: the kernel's row gather has no legal layout from the in-graph
    # query's tiled one, only from the staging buffer's row-outermost one.
    q_in, _ = impl.staging_buffers(query.device)  # ty: ignore[possibly-missing-attribute]
    q_in[: query.shape[0]] = query
    attn = impl.inline_decode_kernel(  # ty: ignore[possibly-missing-attribute]
        q_in,
        rep_row_ids,
        k_rows.view(page_shape),
        v_rows.view(page_shape),
        grid.page_ids,
        mask,
        impl.scale,  # ty: ignore[possibly-missing-attribute]
        b_seqs,
        rep_row_ids.shape[0] // b_seqs,
        impl.num_kv_heads,  # ty: ignore[possibly-missing-attribute]
        impl.num_queries_per_kv,  # ty: ignore[possibly-missing-attribute]
        impl.block_size,  # ty: ignore[possibly-missing-attribute]
        impl.head_size,  # ty: ignore[possibly-missing-attribute]
        impl.logits_soft_cap,  # ty: ignore[possibly-missing-attribute]
    )
    # The body bucket and the num_seqs bucket coincide under the default buckets; a
    # custom compile_sizes can split them. Padded rows stay finite for MoE routing.
    rows = query.shape[0]
    if b_seqs > rows:
        return attn[:rows]
    if b_seqs < rows:
        return torch.cat([attn, attn.new_zeros(rows - b_seqs, *attn.shape[1:])])
    return attn


def _inline_decode_refusal(layer: Attention) -> str | None:
    impl = layer.impl
    if getattr(impl, "inline_decode_kernel", None) is None:
        return f"{type(impl).__name__} has no traceable batched decode kernel"
    if not impl._batched_decode_supported():  # ty: ignore[possibly-missing-attribute]
        return "batched decode is off (SPYRE_BATCHED_DECODE=0, eager attention, or ALiBi)"
    if envs.SPYRE_ATTN_MAX_CORES:
        return "SPYRE_ATTN_MAX_CORES caps every attention compile, which a block graph cannot scope"
    if not outer_graph_fullgraph:
        return "the block does not compile with fullgraph (eager, or Spyre FP8 linears)"
    return None


def _can_split(layer: Attention) -> bool:
    """Only Spyre paged attention, and only where upstream's own prologue is a no-op."""
    return (
        # Encoder-only impls inherit `do_kv_cache_update` from the paged one and would
        # otherwise scatter into an unbound cache.
        layer.attn_type == AttentionType.DECODER
        and hasattr(layer.impl, "do_kv_cache_update")
        and layer.kv_sharing_target_layer_name is None
        and layer.query_quant is None
    )


def _inline_prefill_refusal(layer: Attention) -> str | None:
    impl = layer.impl
    if not getattr(impl, "_compile_attn", False):
        return "attention is eager, so there is no outer graph to fuse into"
    if getattr(impl, "kv_cache_fp8", False):
        return "FP8 attention must read the quantized cache"
    if envs.SPYRE_ATTN_MAX_CORES:
        return "SPYRE_ATTN_MAX_CORES caps every attention compile, which a block graph cannot scope"
    if getattr(impl, "alibi_slopes", None) is not None:
        return "ALiBi needs a per-head bias the rectangle kernel does not add"
    if getattr(impl, "logits_soft_cap", 0.0):
        return "soft-capped logits"
    if getattr(layer, "sliding_window", None) is not None:
        return "sliding-window attention"
    if layer.head_size != layer.head_size_v or layer.head_size % 64:
        return f"head_size {layer.head_size} / {layer.head_size_v} is not one whole stick count"
    return None


def install(layers: Iterable[Attention]) -> tuple[SlotMapping, DecodeGrid]:
    """Opt eligible layers into the traced KV write; returns their shared holders."""
    split = [layer for layer in layers if _can_split(layer)]
    slot_mapping = SlotMapping(split)
    _holders.add(slot_mapping)
    decode_grid = DecodeGrid(slot_mapping)
    _grids.add(decode_grid)

    refusal = next((r for r in map(_inline_decode_refusal, split) if r is not None), None)
    decode_grid.inline = bool(split) and refusal is None
    for layer in split:
        layer.spyre_slots = slot_mapping  # ty: ignore[invalid-assignment]
        layer.spyre_decode_grid = decode_grid if decode_grid.inline else None  # ty: ignore[invalid-assignment]
        layer.forward = types.MethodType(  # ty: ignore[invalid-assignment]
            _spyre_attention_forward, layer
        )

    if split:
        logger.info(
            "Scattering the KV cache inside the outer graph for %d attention layers.",
            len(split),
        )
        if refusal is None:
            logger.info("Tracing batched decode attention into the outer graph.")
        else:
            logger.info("Batched decode attention stays behind the opaque op: %s.", refusal)
        refusal = next((r for r in map(_inline_prefill_refusal, split) if r is not None), None)
        slot_mapping.inline_prefill = refusal is None
        if refusal is None:
            logger.info("Tracing fresh-prompt prefill attention into the outer graph.")
        else:
            logger.info("Fresh-prompt prefill attention stays behind the opaque op: %s.", refusal)
    return slot_mapping, decode_grid
