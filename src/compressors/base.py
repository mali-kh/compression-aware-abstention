"""Compressor protocol — the labeler depends only on this contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class KVCompressor(Protocol):
    """A KV-cache compressor exposing a token-level retention mask.

    Implementations may be score-based (Expected Attention, SnapKV, KVzip),
    structural (DuoAttention's streaming heads), or hybrid. The labeler does
    not care — it only needs to know which original-context token positions
    have their KV entries preserved at a given retention ratio.
    """

    name: str

    def get_kept_token_mask(
        self,
        context_ids: torch.LongTensor,
        ratio: float,
    ) -> torch.BoolTensor:
        """Return shape (seq_len,) mask over the original context tokens.

        ``ratio`` is the fraction of tokens whose KV entries are RETAINED
        (so 0.20 = keep 20%, evict 80%). Implementations should clamp.
        """
        ...
