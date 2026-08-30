"""KVzip wrapper.

Exposes the labeler's only contract:

    get_kept_token_mask(context_ids: torch.LongTensor, ratio: float) -> BoolTensor

KVzip lives in ``third_party/KVzip/`` (vendored from
https://github.com/snu-mllab/KVzip; pinned commit recorded in README).
The wrapper:

  * adds that path to sys.path before importing
  * stubs the ``tiny_api_cuda`` CUDA kernel (only needed for the post-prune
    EvictCache update path, which we never enter — we use RetainCache)
  * runs KVzip's prefill + reconstruction-loss scoring + prune
  * reduces KVzip's per-(layer, head, ctx-token) ``kv.valid`` tensor to a
    per-token mask aligned 1:1 with the input context_ids

Reduction strategy
------------------
KVzip retains different tokens in different (layer, head) slots, so the
raw output is a per-(layer, head, token) bool tensor. The labeler wants
a single per-token mask. We collapse via a configurable reduction:

* ``'topk'`` (default): mark the top ``ratio`` fraction of tokens by
  per-token slot-retention rate. This calibrates the per-token mask
  to match the requested compression ratio (``mask.mean() approx ratio``)
  and mirrors what a researcher usually means by "the tokens kept at
  ratio R": the most redundantly-retained ones across (layer, head).
* ``'majority'``: True if **>50%** of slots retain this token. Tracks
  the underlying compression in absolute terms but collapses to ~0
  kept at low ratios (where avg per-token retention is well below 50%)
  and to ~1 at high ratios.
* ``'any'``: True if at least one slot retains. Permissive; at moderate
  ratios almost every token returns True, which collapses the labeler's
  signal. Useful as an "information could still flow" ablation.
* ``'all'``: True only if every slot retains. Very strict.

This is a research-framing choice; see project memory.

Compatibility (as of upstream commit 5d8472975, 2026-02-11)
-----------------------------------------------------------
* torch==2.3.0 pinned upstream; we run on 2.7.0. Works in practice but
  may need attention-forward signature patches if transformers' API
  drifts further.
* transformers==4.51.3 pinned upstream; fits inside our >=4.45,<5.0 range.
* flash-attn==2.7.4.post1 — required (top-level import in
  ``attention/attn.py``); NOT optional.
* attn_implementation='flash_attention_2' — required at model load.
* tiny_api_cuda CUDA kernel — stubbed in this wrapper (see above).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

KVZIP_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "KVzip"


def _install_tiny_api_cuda_stub() -> None:
    """Provide a stub module so KVzip's top-level imports don't crash.

    KVzip's ``attention/kvcache.py`` does ``from tiny_api_cuda import
    update_flatten_view`` at module load. The function is only invoked
    from ``EvictCache.update()`` after ``prune()`` has been called and
    the cache enters the flatten layout. We use RetainCache and never
    invoke that path, so a stub that raises if called is enough.
    """
    if "tiny_api_cuda" in sys.modules:
        return
    fake = types.ModuleType("tiny_api_cuda")

    def _fail(*args, **kwargs):
        raise RuntimeError(
            "tiny_api_cuda.update_flatten_view called — this kernel is not "
            "built. The wrapper deliberately avoids the post-prune EvictCache "
            "flatten path; if you hit this, kv_type was switched away from "
            "'retain'."
        )

    fake.update_flatten_view = _fail
    sys.modules["tiny_api_cuda"] = fake


def _ensure_kvzip_on_path() -> None:
    if not KVZIP_ROOT.exists():
        raise FileNotFoundError(
            f"KVzip not found at {KVZIP_ROOT}. Run:\n"
            f"  git clone https://github.com/snu-mllab/KVzip.git {KVZIP_ROOT}"
        )
    p = str(KVZIP_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


class KVzipCompressor:
    """Wrap a KVzip ``ModelKVzip`` instance and expose ``get_kept_token_mask``.

    Not threadsafe — the underlying ModelKVzip mutates global state via
    monkey-patching at construction time, and individual ``prefill``+
    ``prune`` calls are stateful on the kv cache. Treat instances as
    single-consumer.
    """

    name = "kvzip"

    def __init__(
        self,
        model_name: str = "qwen3-0.6b",
        reduction: str = "topk",
        load_score: bool = False,
    ) -> None:
        if reduction not in ("topk", "any", "majority", "all"):
            raise ValueError(
                f"reduction must be one of 'topk' | 'any' | 'majority' | 'all', "
                f"got {reduction!r}"
            )

        _install_tiny_api_cuda_stub()
        _ensure_kvzip_on_path()
        # Imports must come AFTER the stub + sys.path setup.
        from model import ModelKVzip  # type: ignore[import-not-found]

        self._impl = ModelKVzip(model_name, kv_type="retain")
        self.tokenizer = self._impl.tokenizer
        self._reduction = reduction
        self._load_score = load_score
        self._model_name = model_name

    def tokenize(self, text: str, *, add_special_tokens: bool = False) -> torch.Tensor:
        """Encode text with the underlying model's tokenizer.

        Returns a 1-D LongTensor on CPU. Use this to ensure tokenization
        matches what KVzip expects internally — passing token IDs from a
        different tokenizer breaks alignment.
        """
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        ).squeeze(0)

    @torch.inference_mode()
    def get_kept_token_mask(
        self,
        context_ids: torch.Tensor,
        ratio: float,
    ) -> torch.Tensor:
        """Return a CPU bool mask of shape ``(len(context_ids),)``.

        Args:
            context_ids: 1-D LongTensor (or 2-D with batch dim 1) of token
                IDs over the context, tokenized with ``self.tokenizer``.
                Does NOT include KVzip's system prompt — that is added
                internally.
            ratio: retention fraction in [0, 1]. ``0.2`` = keep 20% of
                context KV entries.

        Mask entry ``True`` at index ``i`` means ``context_ids[i]``'s KV
        entry survives in at least one (layer, head) slot under the
        ``reduction='any'`` default (or in all slots under ``'all'``).
        """
        if context_ids.dim() == 1:
            context_ids = context_ids.unsqueeze(0)
        if context_ids.dim() != 2 or context_ids.size(0) != 1:
            raise ValueError(
                f"expected 1-D or (1, N) context_ids, got shape {tuple(context_ids.shape)}"
            )
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"ratio must be in [0, 1], got {ratio}")

        context_ids = context_ids.to(device=self._impl.device, dtype=torch.long)
        ctx_len = int(context_ids.size(1))

        kv = self._impl.prefill(context_ids, load_score=self._load_score)
        kv.prune(ratio=ratio, level="pair")

        valid = kv.valid
        # KVzip emits per-layer score of shape (1, n_heads_kv, ctx_len) and
        # stacks them, yielding kv.valid shape (n_layers, 1, n_heads_kv, ctx_len)
        # under _threshold (the global top-K threshold path used by
        # level="pair"). The trailing dim is always ctx_len. We don't depend
        # on the exact rank — flatten everything but the last axis as "slots".
        if valid.size(-1) != ctx_len:
            raise RuntimeError(
                f"alignment mismatch: kv.valid ctx dim {valid.size(-1)} != input "
                f"ctx_len {ctx_len}. KVzip's tokenization may not match "
                f"context_ids; tokenize through self.tokenizer to fix."
            )

        # Flatten leading dims (layers x batch x heads) into a single "slots" axis.
        slots = valid.reshape(-1, valid.size(-1))
        frac_kept = slots.float().mean(dim=0)  # (ctx_len,) — per-token slot retention
        if self._reduction == "topk":
            k = int(round(ratio * ctx_len))
            if k <= 0:
                mask = torch.zeros(ctx_len, dtype=torch.bool, device=valid.device)
            elif k >= ctx_len:
                mask = torch.ones(ctx_len, dtype=torch.bool, device=valid.device)
            else:
                _, top_idx = torch.topk(frac_kept, k)
                mask = torch.zeros(ctx_len, dtype=torch.bool, device=valid.device)
                mask.scatter_(0, top_idx, True)
        elif self._reduction == "any":
            mask = slots.any(dim=0)
        elif self._reduction == "all":
            mask = slots.all(dim=0)
        else:  # "majority"
            mask = frac_kept > 0.5
        return mask.detach().to(device="cpu", dtype=torch.bool).contiguous()

    @torch.inference_mode()
    def get_kept_token_fractions(
        self,
        context_ids: torch.Tensor,
        ratio: float,
    ) -> torch.Tensor:
        """Return per-token slot-retention rates as a Float CPU tensor.

        Auxiliary to ``get_kept_token_mask`` — useful for ablations and
        for verifying token-position alignment without the wrapper's
        reduction choice clouding the signal. Returns a (ctx_len,) tensor
        in [0, 1] where entry i is the fraction of (layer, head) slots
        that retain context_ids[i] under KVzip pruning at ``ratio``.
        """
        if context_ids.dim() == 1:
            context_ids = context_ids.unsqueeze(0)
        if context_ids.dim() != 2 or context_ids.size(0) != 1:
            raise ValueError(
                f"expected 1-D or (1, N) context_ids, got shape {tuple(context_ids.shape)}"
            )
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"ratio must be in [0, 1], got {ratio}")

        context_ids = context_ids.to(device=self._impl.device, dtype=torch.long)
        ctx_len = int(context_ids.size(1))
        kv = self._impl.prefill(context_ids, load_score=self._load_score)
        kv.prune(ratio=ratio, level="pair")
        slots = kv.valid.reshape(-1, kv.valid.size(-1))
        return slots.float().mean(dim=0).detach().to(device="cpu").contiguous()


__all__ = ["KVzipCompressor"]
