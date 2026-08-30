"""KV compressor wrappers.

Every compressor must expose:

    get_kept_token_mask(context_ids: torch.LongTensor, ratio: float) -> torch.BoolTensor

returning a 1-D mask of shape (seq_len,) over the *original* context tokens,
where True means the token's KV entry survives compression at the given ratio.

This mask is the only contract the data labeler depends on.
"""

from .base import KVCompressor

__all__ = ["KVCompressor"]
