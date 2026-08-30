"""End-to-end check for KVzipCompressor on a 4K-token synthetic context.

The user's main concern with this layer: "off-by-one errors here will
silently corrupt every downstream label." So this test focuses on:

  1. Shape + dtype of the returned mask matches input context length.
  2. ratio=1.0 returns an all-True mask (no eviction).
  3. Under the default ``topk`` reduction, mask.mean() approximately
     equals the requested ratio.
  4. Decreasing ratio strictly decreases the fraction of tokens kept.
  5. Determinism: same (context_ids, ratio) -> same mask.
  6. Token-position alignment via the auxiliary
     ``get_kept_token_fractions`` API — a unique needle inserted at a
     known position should have higher per-token slot retention than
     the global average. Reduction-independent.

These tests need CUDA + KVzip + flash-attn. They auto-skip otherwise.
"""

from __future__ import annotations

import os

import pytest
import torch

GPU_AVAILABLE = torch.cuda.is_available()
KVZIP_AVAILABLE = os.path.isdir(
    os.path.join(os.path.dirname(__file__), "..", "third_party", "KVzip")
)

pytestmark = pytest.mark.skipif(
    not (GPU_AVAILABLE and KVZIP_AVAILABLE),
    reason="needs CUDA + third_party/KVzip",
)


@pytest.fixture(scope="module")
def wrapper():
    pytest.importorskip("flash_attn")
    from src.compressors.kvzip import KVzipCompressor

    return KVzipCompressor(model_name="qwen3-0.6b")  # default reduction = "topk"


@pytest.fixture(scope="module")
def synthetic_context(wrapper) -> tuple[torch.Tensor, int]:
    """~4K-token context built from distinctive paragraphs.

    Distinctiveness matters: KVzip's reconstruction-loss scoring
    rewards rare tokens, so a fully-uniform context (e.g. "the the the
    ...") would yield near-flat scores and make any per-token alignment
    test useless.
    """
    paragraphs = [
        "The Antarctic Treaty, signed in 1959 by twelve nations, established the entire continent as a "
        "scientific preserve, freezing all territorial claims and prohibiting military activity.",
        "Coral reefs harbor about a quarter of all marine species despite covering less than one percent "
        "of the ocean floor; coastal communities depend on them for food, tourism, and storm protection.",
        "Rust's borrow checker prevents data races at compile time by enforcing that any value has either "
        "exactly one mutable reference or any number of shared references at any moment.",
        "The James Webb Space Telescope, launched in December 2021, observes primarily in infrared from "
        "the second Earth-Sun Lagrange point, a gravitationally stable region 1.5 million kilometers away.",
        "In medieval Iceland, the Althing met annually at Thingvellir starting in the year 930 to enact "
        "laws and adjudicate disputes; it is often described as one of the oldest parliamentary bodies.",
    ]
    text = "\n\n".join(paragraphs * 12)
    ids = wrapper.tokenize(text)
    if ids.size(0) > 4096:
        ids = ids[:4096]
    return ids, ids.size(0)


def test_full_retention_returns_all_true(wrapper, synthetic_context) -> None:
    ids, ctx_len = synthetic_context
    mask = wrapper.get_kept_token_mask(ids, ratio=1.0)
    assert mask.shape == (ctx_len,), (mask.shape, ctx_len)
    assert mask.dtype is torch.bool
    assert mask.all().item()


def test_topk_reduction_calibrates_to_ratio(wrapper, synthetic_context) -> None:
    """Under ``topk`` (default), mask.mean() should equal ratio (modulo
    a single-token rounding tolerance).
    """
    ids, ctx_len = synthetic_context
    for ratio in (0.1, 0.2, 0.5, 0.8):
        mask = wrapper.get_kept_token_mask(ids, ratio=ratio)
        kept = int(mask.sum().item())
        expected = round(ratio * ctx_len)
        assert abs(kept - expected) <= 1, (
            f"ratio={ratio}: kept={kept} expected={expected} (ctx_len={ctx_len})"
        )


def test_decreasing_ratio_keeps_fewer_tokens(wrapper, synthetic_context) -> None:
    ids, _ = synthetic_context
    f50 = int(wrapper.get_kept_token_mask(ids, ratio=0.5).sum())
    f20 = int(wrapper.get_kept_token_mask(ids, ratio=0.2).sum())
    f10 = int(wrapper.get_kept_token_mask(ids, ratio=0.1).sum())
    assert f10 < f20 < f50


def test_determinism(wrapper, synthetic_context) -> None:
    ids, _ = synthetic_context
    m1 = wrapper.get_kept_token_mask(ids, ratio=0.3)
    m2 = wrapper.get_kept_token_mask(ids, ratio=0.3)
    assert torch.equal(m1, m2)


def test_needle_alignment_via_per_token_fractions(wrapper) -> None:
    """A unique needle phrase inserted at a known position should have
    higher per-token slot retention than the surrounding banal text.

    This is the off-by-one alignment check: if our wrapper indexes
    kv.valid wrong, the needle's high-importance slot pattern would
    appear at the wrong token positions, and the needle slice would
    look ordinary.

    We locate the needle via the tokenizer's char->token offset_mapping
    instead of re-tokenizing the needle in isolation — the latter
    suffers from BPE boundary artifacts (a leading space changes the
    token sequence).
    """
    banal = ("The cat sat on the mat. " * 50).strip()
    needle = " The secret authentication code is 7193-Z-WHITE-RABBIT-OMEGA-ALPHA. "
    full = banal + needle + banal

    # Use a fast tokenizer's offset_mapping to find token indices spanning
    # the needle's character span in the source string.
    enc = wrapper.tokenizer(
        full,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    ids = enc.input_ids.squeeze(0)
    offsets = enc.offset_mapping.squeeze(0).tolist()
    needle_char_start = full.index(needle)
    needle_char_end = needle_char_start + len(needle)

    token_in_needle = [
        i for i, (s, e) in enumerate(offsets)
        if s >= needle_char_start and e <= needle_char_end and e > s
    ]
    assert len(token_in_needle) >= 6, (
        f"needle spans only {len(token_in_needle)} tokens; expected several "
        f"distinctive ones — check tokenizer / needle wording"
    )
    found_lo, found_hi = token_in_needle[0], token_in_needle[-1] + 1

    fractions = wrapper.get_kept_token_fractions(ids, ratio=0.3)
    assert fractions.shape == (ids.size(0),)
    assert (fractions >= 0).all() and (fractions <= 1).all()

    needle_avg = fractions[found_lo:found_hi].float().mean().item()
    global_avg = fractions.float().mean().item()
    assert needle_avg > global_avg + 0.05, (
        f"needle at tokens [{found_lo}, {found_hi}) had per-token slot "
        f"retention {needle_avg:.3f}, only marginally above the global "
        f"average {global_avg:.3f}. Either KVzip's reconstruction scoring "
        f"failed to flag the needle, or there's a token-position alignment bug."
    )


def test_reduction_modes_agree_at_full_retention(wrapper, synthetic_context) -> None:
    """At ratio=1.0, all reductions must return all-True."""
    ids, _ = synthetic_context
    orig = wrapper._reduction
    try:
        for r in ("topk", "any", "majority", "all"):
            wrapper._reduction = r
            assert wrapper.get_kept_token_mask(ids, ratio=1.0).all().item(), r
    finally:
        wrapper._reduction = orig
