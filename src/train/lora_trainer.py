"""LoRA fine-tuning support for cache-state-conditional generation.

Two responsibilities:

  1. format_training_example: convert one (context, kept_mask, query,
     target_output) record into model-ready (input_ids, attention_mask,
     labels) with the kept_mask applied AT THE INPUT LEVEL — evicted
     context tokens are dropped, kept context tokens get fresh
     sequential positions.

  2. CompressedCacheCollator: pad batches with left-padding (so the
     same tokenizer can be reused for generation later without a
     padding-side reset, per Qwen/Llama instruct conventions).

Faithfulness gap (documented):
  Dropping evicted tokens at the input level is a STRICTER simulation
  than what KVzip actually does at inference. KVzip prefills with the
  full context (so kept tokens see evicted neighbors during prefill,
  encoding their info), then evicts KV pairs only afterwards. Our
  drop-at-input approach gives the model less information than KVzip's
  real eviction would. If the v1 model trains on this and works, the
  KVzip-real path should be at-least-as-easy. Worth revisiting if val
  shows the model is starved of structural context — at that point we'd
  switch to a 4D-attention-mask scheme that keeps tokens in the input
  but blocks attention to evicted positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class FormattedExample:
    """One training instance ready for the model.

    `target_start` is the input_ids index where supervised tokens begin.
    Everything before it has label -100; everything from target_start
    onward is what we backprop on.
    """

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    target_start: int
    n_target_tokens: int
    target_text: str


PROMPT_PREFIX = "Context: "
PROMPT_MIDDLE_FMT = "\n\nQuestion: {query}\n\nAnswer: "


def format_training_example(
    *,
    context: str,
    query: str,
    target: str,
    kept_mask: Sequence[bool],
    tokenizer: PreTrainedTokenizerBase,
    apply_mask: bool = True,
    max_seq_length: int = 4096,
) -> FormattedExample:
    """Build one training example. ``apply_mask=False`` is the unit-test
    baseline: same record, mask ignored, used to verify the mask actually
    affects the forward pass."""
    context_ids = tokenizer(context, add_special_tokens=False).input_ids
    if len(context_ids) != len(kept_mask):
        raise ValueError(
            f"context tokenizes to {len(context_ids)} ids but kept_mask has "
            f"length {len(kept_mask)} — tokenizer mismatch with the dataset"
        )

    if apply_mask:
        kept_context_ids = [tid for tid, k in zip(context_ids, kept_mask) if k]
    else:
        kept_context_ids = list(context_ids)

    prefix_ids = tokenizer(PROMPT_PREFIX, add_special_tokens=False).input_ids
    middle_ids = tokenizer(
        PROMPT_MIDDLE_FMT.format(query=query),
        add_special_tokens=False,
    ).input_ids
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id is None; set one before training")
    target_with_eos = target_ids + [tokenizer.eos_token_id]

    target_start = len(prefix_ids) + len(kept_context_ids) + len(middle_ids)
    input_ids = prefix_ids + kept_context_ids + middle_ids + target_with_eos
    if len(input_ids) > max_seq_length:
        raise ValueError(
            f"input_ids length {len(input_ids)} exceeds max_seq_length "
            f"{max_seq_length}; truncate context or skip example upstream"
        )

    labels = [-100] * target_start + target_with_eos
    attention_mask = [1] * len(input_ids)

    return FormattedExample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        target_start=target_start,
        n_target_tokens=len(target_with_eos),
        target_text=tokenizer.decode(target_with_eos, skip_special_tokens=False),
    )


class CompressedCacheCollator:
    """Left-padded batch builder for FormattedExample lists.

    Left padding keeps the same tokenizer state usable for generation
    without resetting padding_side. The labels at pad positions are
    -100 so they don't contribute to the loss.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.pad_id = int(tokenizer.pad_token_id)
        self.tokenizer = tokenizer

    def __call__(self, batch: list[FormattedExample]) -> dict[str, torch.Tensor]:
        max_len = max(len(ex.input_ids) for ex in batch)
        input_ids: list[list[int]] = []
        attn: list[list[int]] = []
        labels: list[list[int]] = []
        for ex in batch:
            pad_len = max_len - len(ex.input_ids)
            input_ids.append([self.pad_id] * pad_len + ex.input_ids)
            attn.append([0] * pad_len + ex.attention_mask)
            labels.append([-100] * pad_len + ex.labels)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


__all__ = [
    "FormattedExample",
    "format_training_example",
    "CompressedCacheCollator",
    "PROMPT_PREFIX",
    "PROMPT_MIDDLE_FMT",
]
