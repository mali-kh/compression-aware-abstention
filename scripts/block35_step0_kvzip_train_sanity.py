"""Block 3.5 Step 0 — KVzipPress + LoRA training-time integration sanity.

Goal: before writing a training loop, verify (or refute) the assumption that
`with KVzipPress: model(full_input)` compresses the cache during the loss
computation. Reading kvpress/presses/kvzip_press.py (lines 78-143), the
contextmanager runs `_perform_kvzip_compression` AFTER `yield` exits — i.e.
after the user's forward returns. So a single-forward training step inside
the press computes its loss on the UNCOMPRESSED cache, defeating the
experiment.

The right shape is the two-phase flow Block 3 used for SnapKV inference
(scripts/eval_compressed_cache.py): prefill ctx inside `with press(inner)`,
let the post-yield code set `module.masked_key_indices`, then forward the
target tokens OUTSIDE the press using the cache. The attention_patch
(kvpress/attention_patch.py) consumes masked_key_indices during decoding
and nullifies attention to evicted positions.

Three checks:
  A. Literal `with press: forward(full_input, labels)` — verify cache is
     uncompressed when loss is computed (loss matches the no-press loss
     to numerical noise; the post-yield compression runs but never
     affects training signal).
  B. Two-phase:
     B1. Phase 1 prefill populates cache; post-yield sets a non-empty
         masked_key_indices on each layer.
     B2. Phase 2 target forward returns finite loss; backward sets
         non-zero grads on LoRA params.
     B3. State reset: process example A, then example B; B's
         masked_key_indices reflects B's ctx_len (not A's).

Run from .venv-kvpress.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvpress import KVzipPress

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train.lora_trainer import format_training_example  # noqa: E402

ADAPTER_DIR = ROOT / "artifacts" / "v1_lora_best"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TRAIN_JSONL = ROOT / "artifacts" / "train_v1.jsonl"

PROMPT_PREFIX = "Context: "
PROMPT_MIDDLE_FMT = "\n\nQuestion: {query}\n\nAnswer: "


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def build_phase_inputs(record: dict, tokenizer):
    """Return (ctx_ids, qt_ids, target_start_in_qt, target_len) for the two-phase split.

    Phase 1 input: prefix + ctx (uncompressed).
    Phase 2 input: query_middle + target + eos. Loss is masked to target+eos.
    """
    prefix_ids = tokenizer(PROMPT_PREFIX, add_special_tokens=False).input_ids
    ctx_ids_full = tokenizer(record["context"], add_special_tokens=False).input_ids
    middle_ids = tokenizer(
        PROMPT_MIDDLE_FMT.format(query=record["query"]),
        add_special_tokens=False,
    ).input_ids
    target_ids = tokenizer(record["target_output"], add_special_tokens=False).input_ids
    target_with_eos = target_ids + [tokenizer.eos_token_id]

    ctx_part = prefix_ids + ctx_ids_full
    qt_part = middle_ids + target_with_eos
    target_start_in_qt = len(middle_ids)
    return ctx_part, qt_part, target_start_in_qt, len(target_with_eos)


def first_layer(inner_model):
    return inner_model.model.layers[0].self_attn


def get_mask_summary(inner_model):
    """Return per-layer (n_evicted, max_seq_idx) for masked_key_indices."""
    summaries = []
    for layer in inner_model.model.layers:
        mki = getattr(layer.self_attn, "masked_key_indices", None)
        if mki is None:
            summaries.append((0, -1))
            continue
        _, _, seq_idx = mki
        summaries.append((seq_idx.numel(), int(seq_idx.max()) if seq_idx.numel() else -1))
    return summaries


def main():
    print("=" * 72, flush=True)
    print("Block 3.5 Step 0 — KVzipPress + LoRA training integration sanity",
          flush=True)
    print("=" * 72, flush=True)

    train_records = load_jsonl(TRAIN_JSONL)
    # Pick two records at retention 0.5 with DIFFERENT ctx lengths so the
    # B3 state-reset check has a discriminator.
    half_records = [r for r in train_records if abs(r["ratio"] - 0.5) < 1e-9]
    half_records.sort(key=lambda r: len(r["context"]))
    rec_a = half_records[5]   # short
    rec_b = half_records[-5]  # long
    print(f"\n[setup] picked rec_a id={rec_a['id']}  ratio={rec_a['ratio']}  "
          f"label={rec_a['label_type']}", flush=True)
    print(f"[setup] picked rec_b id={rec_b['id']}  ratio={rec_b['ratio']}  "
          f"label={rec_b['label_type']}", flush=True)

    print(f"\n[setup] loading {BASE_MODEL} + LoRA from {ADAPTER_DIR.name} ...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    # is_trainable=True so LoRA params have requires_grad=True for backward
    adapted = PeftModel.from_pretrained(base, str(ADAPTER_DIR), is_trainable=True)
    inner = adapted.model  # Qwen2ForCausalLM (kvpress traverses .model.layers)
    n_lora_trainable = sum(
        p.numel() for p in adapted.parameters() if p.requires_grad
    )
    print(f"        loaded in {time.time()-t0:.0f}s  "
          f"trainable params={n_lora_trainable:,}", flush=True)

    ctx_a, qt_a, tgt_start_a, tgt_len_a = build_phase_inputs(rec_a, tokenizer)
    ctx_b, qt_b, tgt_start_b, tgt_len_b = build_phase_inputs(rec_b, tokenizer)
    print(f"\n[setup] rec_a: ctx_len={len(ctx_a)}  qt_len={len(qt_a)}  "
          f"target_len={tgt_len_a}", flush=True)
    print(f"[setup] rec_b: ctx_len={len(ctx_b)}  qt_len={len(qt_b)}  "
          f"target_len={tgt_len_b}", flush=True)

    # ---------------------------------------------------------------
    # CHECK A — literal `with press: forward(full_input)` produces an
    # UNcompressed-cache loss. Run with and without press, confirm losses
    # match (no-op press for training).
    # ---------------------------------------------------------------
    print(f"\n--- CHECK A: literal flow proves UNcompressed loss ---", flush=True)
    full_a = ctx_a + qt_a
    labels_a = (
        [-100] * len(ctx_a)
        + [-100] * tgt_start_a
        + qt_a[tgt_start_a:]  # target+eos tokens carry their own ids as labels
    )
    assert len(labels_a) == len(full_a)
    full_ids = torch.tensor([full_a], device=inner.device)
    labels_ids = torch.tensor([labels_a], device=inner.device)

    # Run in eval() mode so LoRA dropout doesn't add stochastic noise to the
    # comparison — we want to isolate the press's effect on the forward.
    adapted.eval()
    with torch.no_grad():
        # No-press loss
        out_nopress = adapted(input_ids=full_ids, labels=labels_ids, use_cache=False)
        loss_nopress = float(out_nopress.loss.item())

        # Literal-flow press loss (with KVzipPress, compression_ratio=0.5)
        press = KVzipPress(compression_ratio=0.5)
        with press(inner):
            out_press = adapted(input_ids=full_ids, labels=labels_ids, use_cache=False)
        loss_press_literal = float(out_press.loss.item())

    diff = abs(loss_press_literal - loss_nopress)
    print(f"  loss without press:           {loss_nopress:.6f}", flush=True)
    print(f"  loss with `with press: fwd`:  {loss_press_literal:.6f}", flush=True)
    print(f"  abs diff:                     {diff:.2e}", flush=True)
    if diff < 1e-4:
        print(f"  CHECK A PASS: literal flow is a no-op for training "
              f"(loss unchanged → press's post-yield compression runs "
              f"AFTER loss is computed, confirming the integration concern).",
              flush=True)
        check_a_pass = True
    else:
        print(f"  CHECK A AMBIGUOUS: losses differ by {diff:.2e} — investigate.",
              flush=True)
        check_a_pass = False

    # Reset model state for downstream checks (the press also clobbers
    # masked_key_indices via post-yield, and may have set them).
    for layer in inner.model.layers:
        if hasattr(layer.self_attn, "masked_key_indices"):
            layer.self_attn.masked_key_indices = None
    adapted.zero_grad()

    # ---------------------------------------------------------------
    # CHECK B1 + B2 — two-phase flow: prefill ctx in `with press`, then
    # target forward outside. Must produce finite loss + LoRA grads.
    # ---------------------------------------------------------------
    print(f"\n--- CHECK B1+B2: two-phase flow (rec_a, retention=0.5) ---",
          flush=True)
    ctx_ids_a = torch.tensor([ctx_a], device=inner.device)
    qt_ids_a = torch.tensor([qt_a], device=inner.device)
    qt_labels_a = torch.tensor([
        [-100] * tgt_start_a + qt_a[tgt_start_a:]
    ], device=inner.device)

    cache = DynamicCache()
    # Phase 1: prefill ctx inside `with press` (no_grad — we don't need
    # gradients through the ctx KVs; KVzip's chunked scoring forwards would
    # otherwise blow up the autograd graph). The decoding-side forward in
    # Phase 2 still gets LoRA gradient signal via the q/k/v projections of
    # target tokens and the o_proj.
    adapted.eval()  # eval() to suppress dropout in prefill; train mode below
    press = KVzipPress(compression_ratio=0.5)
    t0 = time.time()
    with torch.no_grad(), press(inner):
        _ = inner(input_ids=ctx_ids_a, past_key_values=cache, use_cache=True)
    phase1_dt = time.time() - t0
    n_kept = cache.layers[0].keys.shape[2]
    summaries = get_mask_summary(inner)
    n_evicted_total = sum(s[0] for s in summaries)
    layer0_evicted, layer0_max = summaries[0]
    print(f"  phase1: {phase1_dt:.1f}s  cache_len={n_kept}  "
          f"n_evicted_total={n_evicted_total}  "
          f"layer0_evicted={layer0_evicted}  layer0_max_seq_idx={layer0_max}",
          flush=True)
    assert n_kept == len(ctx_a), \
        f"cache_len ({n_kept}) should equal ctx_len ({len(ctx_a)}) for KVzip masked-eviction"
    assert n_evicted_total > 0, "KVzip should have set masked_key_indices on at least one layer"

    # Phase 2: target forward with cache, OUTSIDE the press
    adapted.train()
    cache_position = torch.arange(
        n_kept, n_kept + qt_ids_a.size(1), device=inner.device
    )
    t0 = time.time()
    out = adapted(
        input_ids=qt_ids_a,
        labels=qt_labels_a,
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=True,
    )
    loss = out.loss
    phase2_dt = time.time() - t0
    print(f"  phase2: {phase2_dt:.1f}s  loss={loss.item():.6f}  "
          f"finite={torch.isfinite(loss).item()}", flush=True)
    if not torch.isfinite(loss):
        print(f"  CHECK B1 FAIL: non-finite loss", flush=True)
        return 1
    print(f"  CHECK B1 PASS: target forward produced finite loss "
          f"(={loss.item():.4f})", flush=True)

    # Backward
    t0 = time.time()
    loss.backward()
    backward_dt = time.time() - t0

    # Inspect LoRA grads
    lora_grad_norms = []
    n_lora_with_grad = 0
    n_lora_total = 0
    for n, p in adapted.named_parameters():
        if "lora_" not in n:
            continue
        n_lora_total += 1
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            n_lora_with_grad += 1
            lora_grad_norms.append(p.grad.norm().item())
    median_norm = sorted(lora_grad_norms)[len(lora_grad_norms)//2] if lora_grad_norms else 0.0
    max_norm = max(lora_grad_norms) if lora_grad_norms else 0.0
    print(f"  backward: {backward_dt:.1f}s  "
          f"lora_params_with_grad={n_lora_with_grad}/{n_lora_total}  "
          f"median_grad_norm={median_norm:.2e}  max_grad_norm={max_norm:.2e}",
          flush=True)
    if n_lora_with_grad < n_lora_total // 2:
        print(f"  CHECK B2 FAIL: only {n_lora_with_grad}/{n_lora_total} LoRA "
              f"params got gradients", flush=True)
        return 1
    if median_norm == 0.0:
        print(f"  CHECK B2 FAIL: zero median LoRA grad norm", flush=True)
        return 1
    print(f"  CHECK B2 PASS: LoRA params receive non-zero gradient signal",
          flush=True)

    # Save A's mask snapshot for B3 comparison BEFORE we move on
    mki_a = []
    for layer in inner.model.layers:
        mki = layer.self_attn.masked_key_indices
        if mki is None:
            mki_a.append(None)
        else:
            # clone seq_indices only (head_indices/batch can collide trivially)
            mki_a.append(mki[2].cpu().clone())

    # ---------------------------------------------------------------
    # CHECK B3 — process rec_b; verify masked_key_indices reflects rec_b's
    # ctx, not rec_a's.
    # ---------------------------------------------------------------
    print(f"\n--- CHECK B3: state reset between examples ---", flush=True)
    adapted.zero_grad()

    ctx_ids_b = torch.tensor([ctx_b], device=inner.device)
    cache_b = DynamicCache()
    adapted.eval()
    press_b = KVzipPress(compression_ratio=0.5)
    with torch.no_grad(), press_b(inner):
        _ = inner(input_ids=ctx_ids_b, past_key_values=cache_b, use_cache=True)
    n_kept_b = cache_b.layers[0].keys.shape[2]
    summaries_b = get_mask_summary(inner)
    layer0_evicted_b, layer0_max_b = summaries_b[0]
    print(f"  rec_a ctx_len={len(ctx_a)}  layer0_evicted_a={layer0_evicted}  "
          f"layer0_max_seq_idx_a={layer0_max}", flush=True)
    print(f"  rec_b ctx_len={len(ctx_b)}  layer0_evicted_b={layer0_evicted_b}  "
          f"layer0_max_seq_idx_b={layer0_max_b}", flush=True)
    assert n_kept_b == len(ctx_b)

    # B3 invariants:
    # i.   rec_b's mask must have indices < rec_b's ctx_len (no out-of-range)
    # ii.  rec_b's mask is DIFFERENT from rec_a's (different examples → different scores)
    # iii. attention_patch's prefilling branch must have cleared rec_a's mask
    #      before rec_b's chunked scoring forwards ran
    #
    # The clearest discriminator: pick the largest-ctx layer and compare
    # the actual seq-index sets.
    layer0_seq_b = inner.model.layers[0].self_attn.masked_key_indices[2].cpu()
    layer0_seq_a_saved = mki_a[0]
    same_set = (
        layer0_seq_a_saved is not None
        and layer0_seq_a_saved.numel() == layer0_seq_b.numel()
        and torch.equal(
            torch.sort(layer0_seq_a_saved).values,
            torch.sort(layer0_seq_b).values,
        )
    )
    if same_set:
        print(f"  CHECK B3 FAIL: rec_a and rec_b layer-0 mask indices are "
              f"identical — state not reset between examples", flush=True)
        return 1
    if layer0_max_b >= len(ctx_b):
        print(f"  CHECK B3 FAIL: rec_b layer-0 max_seq_idx ({layer0_max_b}) "
              f">= ctx_len_b ({len(ctx_b)}) — out-of-range", flush=True)
        return 1
    print(f"  CHECK B3 PASS: rec_b mask has different indices than rec_a's "
          f"(at layer 0) and stays within rec_b's ctx_len. State reset OK.",
          flush=True)

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------
    print(f"\n{'='*72}", flush=True)
    print(f"STEP 0 SUMMARY", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  CHECK A (literal flow is uncompressed):  "
          f"{'PASS' if check_a_pass else 'AMBIGUOUS'}", flush=True)
    print(f"  CHECK B1 (two-phase finite loss):        PASS", flush=True)
    print(f"  CHECK B2 (LoRA grads):                   PASS", flush=True)
    print(f"  CHECK B3 (state reset between examples): PASS", flush=True)
    print(f"\n  Conclusion: training MUST use the two-phase split (prefill "
          f"ctx inside `with press(inner)`, then target forward outside).",
          flush=True)
    print(f"  The single-forward-inside-press flow does NOT compress.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
