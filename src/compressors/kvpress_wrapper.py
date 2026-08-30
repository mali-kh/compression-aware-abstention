"""kvpress wrapper for cross-compressor experiments (Expected Attention, SnapKV).

Lives in this module so the existing KVCompressor Protocol contract is honoured,
but **must be imported only from a venv with `kvpress` and `transformers>=4.56`
installed**. Our project venv is pinned at transformers==4.51.3 for KVzip's
RetainCache; kvpress lives in `.venv-kvpress/` for that reason. See HANDOFF
§6.1 and §4.7.

The wrapper:
  * subclasses kvpress's ScorerPress-derived presses to record per-(layer, head,
    token) survival during the prefill hook,
  * runs prefill on context-only (no question) to mirror v1's KVzip pipeline,
  * reduces records to a per-token kept_mask via topk on per-token slot retention
    (matches KVzipCompressor reduction='topk' calibration so cross-compressor
    masks live in the same semantic space).

Important convention: kvpress uses `compression_ratio` = fraction *evicted*, not
retained. Our `ratio` is retention. The wrapper inverts:
    compression_ratio = 1.0 - ratio.
Tests assert `mask.mean() ≈ ratio` for the same calibration as KVzip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

try:
    from kvpress.presses.expected_attention_press import ExpectedAttentionPress
    from kvpress.presses.scorer_press import ScorerPress
    from kvpress.presses.snapkv_press import SnapKVPress
except ImportError as e:
    raise ImportError(
        "kvpress is not installed in this Python environment. The cross-compressor "
        "wrapper must be imported from .venv-kvpress (transformers>=4.56), not the "
        "main project venv (transformers==4.51.3 for KVzip). See HANDOFF §6.1."
    ) from e


@dataclass
class _RecordingExpectedAttentionPress(ExpectedAttentionPress):
    """ExpectedAttentionPress that records per-(layer, head, token) survival."""

    def __post_init__(self):
        super().__post_init__()
        # _records is not a dataclass field; populated during the prefill hook.
        # Keys are layer indices; values are bool tensors of shape
        # (batch, num_kv_heads, k_len) on CPU.
        object.__setattr__(self, "_records", {})

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            mask = torch.ones(
                keys.shape[0], keys.shape[1], keys.shape[2],
                dtype=torch.bool, device=keys.device,
            )
            self._records[module.layer_idx] = mask.cpu()
            return keys, values
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        k_len = keys.shape[2]
        n_kept = int(k_len * (1 - self.compression_ratio))
        topk_idx = scores.topk(n_kept, dim=-1).indices
        mask = torch.zeros(scores.shape, dtype=torch.bool, device=scores.device)
        mask.scatter_(-1, topk_idx, True)
        self._records[module.layer_idx] = mask.cpu()
        # ScorerPress.compress prune logic, replicated:
        indices = topk_idx.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        return keys.gather(2, indices).contiguous(), values.gather(2, indices).contiguous()


@dataclass
class _RecordingSnapKVPress(SnapKVPress):
    """SnapKVPress that records per-(layer, head, token) survival."""

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "_records", {})

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            mask = torch.ones(
                keys.shape[0], keys.shape[1], keys.shape[2],
                dtype=torch.bool, device=keys.device,
            )
            self._records[module.layer_idx] = mask.cpu()
            return keys, values
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        k_len = keys.shape[2]
        n_kept = int(k_len * (1 - self.compression_ratio))
        topk_idx = scores.topk(n_kept, dim=-1).indices
        mask = torch.zeros(scores.shape, dtype=torch.bool, device=scores.device)
        mask.scatter_(-1, topk_idx, True)
        self._records[module.layer_idx] = mask.cpu()
        indices = topk_idx.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        return keys.gather(2, indices).contiguous(), values.gather(2, indices).contiguous()


_PRESS_BY_NAME: dict[str, Type[ScorerPress]] = {
    "expected_attention": _RecordingExpectedAttentionPress,
    "snapkv": _RecordingSnapKVPress,
}


class KVPressCompressor:
    """Wrap a kvpress press and expose `get_kept_token_mask` matching the
    `KVCompressor` Protocol used by `src/data/label_construction.py`.

    Not threadsafe — single-consumer, hooks register at prefill time.

    Args:
        model_name: HF model id (e.g. "Qwen/Qwen2.5-7B-Instruct"). Should match
            what the labeler's spans are tokenized against.
        press: one of "expected_attention" | "snapkv".
        press_kwargs: extra kwargs passed to the press constructor (e.g.
            `n_sink=4` for ExpAttn, `window_size=64, kernel_size=5` for SnapKV).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        press: str = "expected_attention",
        **press_kwargs,
    ) -> None:
        if press not in _PRESS_BY_NAME:
            raise ValueError(f"press must be in {sorted(_PRESS_BY_NAME)}, got {press!r}")
        self.name = press
        self._press_class = _PRESS_BY_NAME[press]
        self._press_kwargs = press_kwargs
        self._model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.model.eval()

    def tokenize(self, text: str, *, add_special_tokens: bool = False) -> torch.Tensor:
        """Match KVzipCompressor.tokenize for downstream parity."""
        return self.tokenizer.encode(
            text, add_special_tokens=add_special_tokens, return_tensors="pt"
        ).squeeze(0)

    @torch.inference_mode()
    def get_kept_token_mask(
        self,
        context_ids: torch.Tensor,
        ratio: float,
        prefill_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return CPU bool mask of shape (len(context_ids),).

        Mirrors KVzipCompressor.get_kept_token_mask semantics with
        reduction='topk': mask.mean() should land within ~1pp of `ratio`.

        Args:
            context_ids: 1-D LongTensor (or (1, N)) of context token IDs. The
                returned mask is over THIS sequence.
            ratio: retention fraction in [0, 1] over the context.
            prefill_ids: optional (1, M) longer token sequence to feed the
                press during prefill (e.g. context+question for SnapKV-native).
                Must start with context_ids (we slice records to the first
                len(context_ids) positions). Defaults to context_ids itself.
        """
        if context_ids.dim() == 1:
            context_ids = context_ids.unsqueeze(0)
        if context_ids.dim() != 2 or context_ids.size(0) != 1:
            raise ValueError(
                f"expected 1-D or (1, N) context_ids, got shape {tuple(context_ids.shape)}"
            )
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"ratio must be in [0, 1], got {ratio}")

        ctx_len = int(context_ids.size(1))
        if ratio <= 0.0:
            return torch.zeros(ctx_len, dtype=torch.bool)
        if ratio >= 1.0:
            return torch.ones(ctx_len, dtype=torch.bool)

        # Prefill text may be longer than the context (e.g. SnapKV-native:
        # context + question fed to the press for query-aware scoring), but
        # the returned mask is always over context_ids only.
        if prefill_ids is None:
            prefill_ids = context_ids
        if prefill_ids.dim() == 1:
            prefill_ids = prefill_ids.unsqueeze(0)
        if prefill_ids.size(1) < ctx_len:
            raise ValueError(
                f"prefill_ids ({prefill_ids.size(1)} tokens) must be at least "
                f"as long as context_ids ({ctx_len})"
            )
        # Sanity: prefill_ids must START with context_ids — otherwise the
        # mask we slice doesn't correspond to the right tokens.
        if not torch.equal(prefill_ids[0, :ctx_len].cpu(), context_ids[0].cpu()):
            raise ValueError(
                "prefill_ids[:, :ctx_len] must equal context_ids — the mask "
                "is sliced from the leading context positions of the prefill."
            )

        # The press's compression_ratio is over the FULL prefill sequence; we
        # want mask.mean() ≈ ratio over the context portion. Pick a press CR
        # that retains roughly the right total count, then let topk on the
        # context-portion frac_kept calibrate exactly.
        prefill_len = int(prefill_ids.size(1))
        target_kept_total = int(round(ratio * ctx_len))
        # Question tokens almost always survive any reasonable press; budget
        # them as kept.
        question_len = prefill_len - ctx_len
        target_total_kept = target_kept_total + question_len
        # CR for press = fraction evicted = 1 - target_total_kept/prefill_len
        press_cr = max(min(1.0 - target_total_kept / prefill_len, 0.999), 0.0)

        press = self._press_class(
            compression_ratio=press_cr, **self._press_kwargs
        )

        ids = prefill_ids.to(self.model.device, dtype=torch.long)
        cache = DynamicCache()
        with press(self.model):
            _ = self.model(input_ids=ids, past_key_values=cache, use_cache=True)

        if not press._records:
            raise RuntimeError(
                "press recorded no layers — hook registration likely failed; "
                "check that kvpress's __call__ matched this model's layer layout."
            )

        # Stack per-layer records into a (slots, prefill_len) view, where
        # slots = sum over layers of num_kv_heads. Then slice to context.
        per_slot = torch.cat(
            [press._records[i].reshape(-1, press._records[i].shape[-1])
             for i in sorted(press._records.keys())],
            dim=0,
        )
        if per_slot.shape[-1] != prefill_len:
            raise RuntimeError(
                f"alignment mismatch: per_slot dim {per_slot.shape[-1]} != "
                f"prefill_len {prefill_len}."
            )
        per_slot_ctx = per_slot[:, :ctx_len]
        frac_kept = per_slot_ctx.float().mean(dim=0)  # (ctx_len,)

        # topk reduction matching KVzipCompressor reduction='topk'
        k = int(round(ratio * ctx_len))
        if k <= 0:
            return torch.zeros(ctx_len, dtype=torch.bool)
        if k >= ctx_len:
            return torch.ones(ctx_len, dtype=torch.bool)
        _, top_idx = torch.topk(frac_kept, k)
        mask = torch.zeros(ctx_len, dtype=torch.bool)
        mask.scatter_(0, top_idx, True)
        return mask
