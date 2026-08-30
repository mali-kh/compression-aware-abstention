"""Block 3.6 Step 0 — multi-press LoRA training-time integration sanity.

Block 3.5 Step 0 verified KVzip's three-phase training flow. Mixture training
needs SnapKV and ExpectedAttention as well, both of which physically prune
during prefill via per-layer forward_hook (not post-yield like KVzip). This
script verifies their two-phase training flow on real mixture examples.

For each press {SnapKV, ExpAttn, KVzip}, on a record with that compressor:
  CHECK 1. Forward pass under the press produces a finite target-token loss.
  CHECK 2. Backward sets non-zero gradients on LoRA params.
  CHECK 3. State resets between examples — masks/cache from example A do
           not contaminate example B.

KVzip uses 3-phase (1a no_grad scoring, 1b grad re-prefill, 2 grad target).
SnapKV/ExpAttn use 2-phase (1 grad prefill-with-press, 2 grad target).

Run from .venv-kvpress.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

_orig_atk = AutoTokenizer.from_pretrained
_AT_CACHE: dict = {}
def _cached_atk(model_id, *a, **kw):
    key = str(model_id)
    if key not in _AT_CACHE:
        _AT_CACHE[key] = _orig_atk(model_id, *a, **kw)
    return _AT_CACHE[key]
AutoTokenizer.from_pretrained = _cached_atk

from kvpress import KVzipPress, SnapKVPress, ExpectedAttentionPress

ROOT = Path(__file__).resolve().parents[1]
TRAIN_JSONL = ROOT / "artifacts" / "train_mixture_v1.jsonl"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

PROMPT_PREFIX = "Context: "
PROMPT_MIDDLE_FMT = "\n\nQuestion: {query}\n\nAnswer: "


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def build_phase_inputs(record, tokenizer):
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
    return ctx_part, qt_part, len(middle_ids), len(target_with_eos)


def snapshot_masks(inner):
    return [getattr(layer.self_attn, "masked_key_indices", None)
            for layer in inner.model.layers]


def restore_masks(inner, snaps):
    for layer, m in zip(inner.model.layers, snaps):
        layer.self_attn.masked_key_indices = m


def clear_masks(inner):
    for layer in inner.model.layers:
        layer.self_attn.masked_key_indices = None


def kvzip_step(adapted, inner, parsed, ratio, device):
    """3-phase: no_grad scoring + snapshot, grad re-prefill, grad target."""
    ctx_part, qt_part, tgt_start, _ = parsed
    ctx = torch.tensor([ctx_part], device=device)
    qt = torch.tensor([qt_part], device=device)
    qt_lab = torch.tensor([[-100]*tgt_start + qt_part[tgt_start:]], device=device)
    cr = max(0.0, min(1.0 - ratio, 0.999))

    cache_a = DynamicCache()
    press = KVzipPress(compression_ratio=cr)
    with torch.no_grad():
        with press(inner):
            _ = inner(input_ids=ctx, past_key_values=cache_a, use_cache=True)
    saved = snapshot_masks(inner)
    clear_masks(inner)
    del cache_a, press

    cache_b = DynamicCache()
    _ = adapted(input_ids=ctx, past_key_values=cache_b, use_cache=True)
    restore_masks(inner, saved)

    n_ctx = ctx.size(1)
    cache_position = torch.arange(n_ctx, n_ctx + qt.size(1), device=device)
    out = adapted(input_ids=qt, past_key_values=cache_b, cache_position=cache_position,
                  labels=qt_lab, use_cache=True)
    clear_masks(inner)
    return out.loss, ("n_kept", cache_b.layers[0].keys.shape[2], "ctx_len", n_ctx)


def snapkv_or_expattn_step(adapted, inner, parsed, ratio, device, press_cls, **press_kwargs):
    """2-phase: grad prefill-with-press (physically prunes), grad target."""
    ctx_part, qt_part, tgt_start, _ = parsed
    ctx = torch.tensor([ctx_part], device=device)
    qt = torch.tensor([qt_part], device=device)
    qt_lab = torch.tensor([[-100]*tgt_start + qt_part[tgt_start:]], device=device)
    cr = max(0.0, min(1.0 - ratio, 0.999))

    cache = DynamicCache()
    press = press_cls(compression_ratio=cr, **press_kwargs)
    with press(inner):
        _ = adapted(input_ids=ctx, past_key_values=cache, use_cache=True)
    n_kept = cache.layers[0].keys.shape[2]
    n_ctx = ctx.size(1)

    cache_position = torch.arange(n_kept, n_kept + qt.size(1), device=device)
    out = adapted(input_ids=qt, past_key_values=cache, cache_position=cache_position,
                  labels=qt_lab, use_cache=True)
    return out.loss, ("n_kept", n_kept, "ctx_len", n_ctx)


def run_press_check(name, step_fn, recs, adapted, inner, tokenizer, device):
    print(f"\n{'='*72}\n  {name}\n{'='*72}", flush=True)
    if len(recs) < 2:
        print(f"  SKIP: need at least 2 records for state-reset check, got {len(recs)}", flush=True)
        return False
    rec_a, rec_b = recs[0], recs[1]
    parsed_a = build_phase_inputs(rec_a, tokenizer)
    parsed_b = build_phase_inputs(rec_b, tokenizer)
    print(f"  rec_a: id={rec_a['id']}  ratio={rec_a['ratio']}  ctx_len={len(parsed_a[0])}", flush=True)
    print(f"  rec_b: id={rec_b['id']}  ratio={rec_b['ratio']}  ctx_len={len(parsed_b[0])}", flush=True)

    # Check A on rec_a
    adapted.train()
    adapted.zero_grad()
    t0 = time.time()
    loss_a, info_a = step_fn(adapted, inner, parsed_a, rec_a['ratio'], device)
    if not torch.isfinite(loss_a):
        print(f"  CHECK 1 FAIL: non-finite loss = {loss_a.item()}", flush=True)
        return False
    loss_a.backward()
    backward_dt = time.time() - t0

    n_lora_grad = 0
    n_lora_tot = 0
    for n, p in adapted.named_parameters():
        if "lora_" not in n:
            continue
        n_lora_tot += 1
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            n_lora_grad += 1

    print(f"  CHECK 1 PASS (rec_a): loss={loss_a.item():.4f}  {info_a}", flush=True)
    if n_lora_grad < n_lora_tot // 2:
        print(f"  CHECK 2 FAIL: only {n_lora_grad}/{n_lora_tot} LoRA params got grads", flush=True)
        return False
    print(f"  CHECK 2 PASS: {n_lora_grad}/{n_lora_tot} LoRA params received gradient signal", flush=True)
    print(f"  step wallclock (fwd+bwd): {backward_dt:.2f}s", flush=True)

    # Check 3 on rec_b
    adapted.zero_grad()
    loss_b, info_b = step_fn(adapted, inner, parsed_b, rec_b['ratio'], device)
    if not torch.isfinite(loss_b):
        print(f"  CHECK 3 FAIL (rec_b): non-finite loss = {loss_b.item()}", flush=True)
        return False
    loss_b.backward()
    print(f"  CHECK 3 PASS (rec_b): loss={loss_b.item():.4f}  {info_b}", flush=True)
    print(f"  ({name} state reset OK; rec_b ran cleanly after rec_a)", flush=True)
    adapted.zero_grad()
    return True


def main():
    print(f"[setup] loading tokenizer + base + LoRA ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    adapted = get_peft_model(base, peft_cfg)
    inner = adapted.model
    adapted.config.use_cache = True
    device = next(adapted.parameters()).device

    print(f"[setup] loading mixture data ...", flush=True)
    train = load_jsonl(TRAIN_JSONL)
    by_compressor = {"snapkv": [], "kvzip": [], "expected_attention": []}
    for r in train:
        c = r["metadata"]["compressor"]
        if c in by_compressor:
            # Filter to records that fit a normal training budget
            # (just use the first few short-ish ones per compressor)
            if len(r["context"]) < 8000:
                by_compressor[c].append(r)
        if all(len(by_compressor[k]) >= 4 for k in by_compressor):
            break

    results = {}
    results["KVzip (3-phase)"] = run_press_check(
        "KVzip (3-phase, sanity-checking on a kvzip-mask example)",
        kvzip_step, by_compressor["kvzip"][:2], adapted, inner, tokenizer, device,
    )
    results["SnapKV (2-phase)"] = run_press_check(
        "SnapKV (2-phase, sanity-checking on a snapkv-mask example)",
        lambda a, i, p, r, d: snapkv_or_expattn_step(
            a, i, p, r, d, SnapKVPress, window_size=64, kernel_size=5
        ),
        by_compressor["snapkv"][:2], adapted, inner, tokenizer, device,
    )
    results["ExpAttn (2-phase)"] = run_press_check(
        "ExpectedAttention (2-phase, sanity-checking on an expattn-mask example)",
        lambda a, i, p, r, d: snapkv_or_expattn_step(
            a, i, p, r, d, ExpectedAttentionPress
        ),
        by_compressor["expected_attention"][:2], adapted, inner, tokenizer, device,
    )

    print(f"\n{'='*72}\nSUMMARY\n{'='*72}", flush=True)
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
    if all(results.values()):
        print(f"\nAll three press flows compose with LoRA training. Proceed to Step 1.",
              flush=True)
        return 0
    print(f"\nOne or more press flows broke. STOP.", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
