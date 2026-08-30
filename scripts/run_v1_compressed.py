"""Block 3.5 training — LoRA fine-tune under actual KVzip compressed-cache decoding.

Sibling of run_v1_full.py for the cache-state-conditional pilot. The
critical difference: each training step uses a THREE-phase split per
training example so the model sees the same compression regime during
training that real deployment uses.

Per-step flow (one example at a time):
  Phase 1a  no_grad  with KVzipPress(compression_ratio=1-r):
                inner(input_ids=ctx_ids, past_key_values=cache_a, use_cache=True)
            → KVzip's post-yield code sets `module.masked_key_indices` on
              each Qwen2 attention layer; cache_a is discarded.
  (snapshot masked_key_indices, clear them so Phase 1b's prefill doesn't
   re-clear via attention_patch's prefilling branch)
  Phase 1b  grad on:
                adapted(input_ids=ctx_ids, past_key_values=cache_b, use_cache=True)
            → cache_b now holds DIFFERENTIABLE keys/values for the ctx
              (LoRA's effect on cached KVs is part of the gradient graph).
  (restore masked_key_indices)
  Phase 2   grad on:
                adapted(input_ids=qt_ids, past_key_values=cache_b,
                        cache_position=arange(n_ctx, n_ctx+qt_len),
                        labels=qt_labels)
            → attention_patch nullifies attention to evicted ctx positions
              via the saved mask. Loss is masked to target+eos only.
  loss.backward(); optimizer step on accum boundaries.

Validation: same three-phase, all in no_grad. Picks best epoch by val loss
under compressed-cache regime (matching training regime).

Outputs (with --output-suffix _compressed_seed42):
  artifacts/v1_lora_epoch{1,2,3}_compressed_seed42/
  artifacts/v1_lora_best_compressed_seed42/

No LR sweep (per spec — 2e-4 fixed). No D5 qualitative or run report —
those are downstream. Step 3 (eval under KVzip cc inference) is a
separate script.

Run from .venv-kvpress (transformers 5.x + kvpress + peft).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

# Set BEFORE importing transformers / kvpress: KVzipPress.__call__ re-runs
# `AutoTokenizer.from_pretrained(model.config.name_or_path)` every time the
# press is entered (kvzip_press.py:94). Across 2662 train examples × 3 epochs
# that's ~8000 HF API hits and trips a 429 rate-limit. Offline mode forces
# from_pretrained to read the local cache only.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# Memoize AutoTokenizer.from_pretrained so the per-example KVzipPress
# instantiation reuses the already-loaded tokenizer instead of re-reading
# files (or worse, hitting HF if the offline env var doesn't catch it).
_orig_at_from_pretrained = AutoTokenizer.from_pretrained
_AT_CACHE: dict = {}
def _cached_from_pretrained(model_id, *args, **kwargs):
    key = str(model_id)
    if key not in _AT_CACHE:
        _AT_CACHE[key] = _orig_at_from_pretrained(model_id, *args, **kwargs)
    return _AT_CACHE[key]
AutoTokenizer.from_pretrained = _cached_from_pretrained

from kvpress import KVzipPress

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


PROMPT_PREFIX = "Context: "
PROMPT_MIDDLE_FMT = "\n\nQuestion: {query}\n\nAnswer: "


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def build_phase_inputs(record: dict, tokenizer, max_seq_length: int):
    """Tokenize one record into (ctx_ids, qt_ids, target_start_in_qt, target_len).

    Returns None if the example exceeds max_seq_length budget.
    """
    prefix_ids = tokenizer(PROMPT_PREFIX, add_special_tokens=False).input_ids
    ctx_ids_full = tokenizer(record["context"], add_special_tokens=False).input_ids
    middle_ids = tokenizer(
        PROMPT_MIDDLE_FMT.format(query=record["query"]),
        add_special_tokens=False,
    ).input_ids
    target_ids = tokenizer(record["target_output"], add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is None:
        raise RuntimeError("tokenizer.eos_token_id is None — set one before training")
    target_with_eos = target_ids + [tokenizer.eos_token_id]

    ctx_part = prefix_ids + ctx_ids_full
    qt_part = middle_ids + target_with_eos
    target_start_in_qt = len(middle_ids)
    full_len = len(ctx_part) + len(qt_part)
    if full_len > max_seq_length:
        return None
    return ctx_part, qt_part, target_start_in_qt, len(target_with_eos)


def fresh_base_model(model_id: str):
    base = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    return base


def make_lora_config(cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=cfg["r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        bias=cfg["bias"], target_modules=cfg["target_modules"],
        task_type=cfg["task_type"],
    )


def snapshot_masked_key_indices(inner_model):
    """Read masked_key_indices from each attention layer; clear in place."""
    snaps = []
    for layer in inner_model.model.layers:
        mki = getattr(layer.self_attn, "masked_key_indices", None)
        snaps.append(mki)
        layer.self_attn.masked_key_indices = None
    return snaps


def restore_masked_key_indices(inner_model, snaps):
    for layer, mki in zip(inner_model.model.layers, snaps):
        layer.self_attn.masked_key_indices = mki


def clear_masked_key_indices(inner_model):
    for layer in inner_model.model.layers:
        layer.self_attn.masked_key_indices = None


def compressed_cache_step(
    *,
    adapted,
    inner,
    record_inputs: tuple,
    ratio: float,
    device: torch.device,
    no_grad_phase1b: bool = False,
):
    """Run one example through the three-phase forward.

    Returns the loss tensor (still attached to the autograd graph if
    no_grad_phase1b=False; detached if True).
    """
    ctx_part, qt_part, tgt_start, tgt_len = record_inputs
    ctx_ids = torch.tensor([ctx_part], device=device)
    qt_ids = torch.tensor([qt_part], device=device)
    qt_labels = torch.tensor(
        [[-100] * tgt_start + qt_part[tgt_start:]],
        device=device,
    )

    cr = max(0.0, min(1.0 - ratio, 0.999))

    # ----- Phase 1a: scoring under KVzipPress (always no_grad) -----
    cache_a = DynamicCache()
    press = KVzipPress(compression_ratio=cr)
    with torch.no_grad():
        with press(inner):
            _ = inner(input_ids=ctx_ids, past_key_values=cache_a, use_cache=True)
    saved_masks = snapshot_masked_key_indices(inner)
    del cache_a, press

    # ----- Phase 1b: re-prefill ctx (under grad if training, else no_grad) -----
    cache_b = DynamicCache()
    if no_grad_phase1b:
        with torch.no_grad():
            _ = adapted(input_ids=ctx_ids, past_key_values=cache_b, use_cache=True)
    else:
        _ = adapted(input_ids=ctx_ids, past_key_values=cache_b, use_cache=True)

    # Restore the saved KVzip mask (Phase 1b's prefill cleared it via the
    # attention_patch prefilling branch).
    restore_masked_key_indices(inner, saved_masks)

    # ----- Phase 2: target forward with cache + restored mask -----
    n_ctx = ctx_ids.size(1)
    cache_position = torch.arange(
        n_ctx, n_ctx + qt_ids.size(1), device=device
    )
    out = adapted(
        input_ids=qt_ids,
        past_key_values=cache_b,
        cache_position=cache_position,
        labels=qt_labels,
        use_cache=True,
    )
    # Mask cleanup so the next example's Phase 1a sees a clean state.
    # (attention_patch will also clear at the next prefill, but doing it
    # here makes leakage impossible if any later code reads modules.)
    clear_masked_key_indices(inner)

    return out.loss


def train_epoch(
    *, adapted, inner, train_examples, optim, device, accum, log_every=50, smoke_steps=0,
):
    adapted.train()
    optim.zero_grad()
    n = len(train_examples)
    order = list(range(n))
    random.shuffle(order)
    losses_unscaled: list[float] = []
    micro = 0
    t0 = time.time()
    micro_times: list[float] = []
    for idx in order:
        rec, parsed, ratio = train_examples[idx]
        m_t0 = time.time()
        loss = compressed_cache_step(
            adapted=adapted, inner=inner, record_inputs=parsed,
            ratio=ratio, device=device, no_grad_phase1b=False,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at micro {micro} (id={rec['id']}): {loss.item()}")
        scaled = loss / accum
        scaled.backward()
        m_dt = time.time() - m_t0
        micro_times.append(m_dt)
        losses_unscaled.append(loss.item())
        micro += 1
        if micro % accum == 0:
            optim.step()
            optim.zero_grad()
        if micro % log_every == 0:
            recent = sum(losses_unscaled[-log_every:]) / min(log_every, len(losses_unscaled))
            recent_t = sum(micro_times[-log_every:]) / min(log_every, len(micro_times))
            print(
                f"     micro={micro}/{n}  recent_loss={recent:.4f}  "
                f"recent_dt={recent_t:.2f}s  ({time.time()-t0:.0f}s elapsed)",
                flush=True,
            )
        if smoke_steps and micro >= smoke_steps:
            print(f"     [smoke] stopping after {smoke_steps} micro-steps", flush=True)
            break
    if not smoke_steps and micro % accum != 0:
        optim.step()
        optim.zero_grad()
    return sum(losses_unscaled) / max(len(losses_unscaled), 1), micro_times


@torch.no_grad()
def compute_val_loss(*, adapted, inner, val_examples, device):
    adapted.eval()
    losses: list[float] = []
    for _, parsed, ratio in val_examples:
        loss = compressed_cache_step(
            adapted=adapted, inner=inner, record_inputs=parsed,
            ratio=ratio, device=device, no_grad_phase1b=True,
        )
        if torch.isfinite(loss):
            losses.append(float(loss.item()))
    adapted.train()
    return sum(losses) / max(len(losses), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "train" / "v1.yaml"))
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train-jsonl", default=str(ROOT / "artifacts" / "train_v1.jsonl"))
    ap.add_argument("--val-jsonl", default=str(ROOT / "artifacts" / "val_v1.jsonl"))
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="Fixed LR (no sweep). 2e-4 was the best LR from v1's sweep.")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-suffix", default="_compressed_seed42",
                    help="Suffix appended to artifacts/v1_lora_epoch* and "
                         "artifacts/v1_lora_best to keep this run distinct from v1/mixture.")
    ap.add_argument("--smoke-steps", type=int, default=0,
                    help="If >0, stop after this many micro-steps in the first epoch — "
                         "for timing the per-step cost before committing to a full run.")
    ap.add_argument("--skip-val", action="store_true",
                    help="Skip val loss computation during epochs (timing-only mode).")
    args = ap.parse_args()

    _seed_everything(args.seed)

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_root = ROOT / "artifacts"
    out_root.mkdir(parents=True, exist_ok=True)
    out_suffix = args.output_suffix
    print(f"\n[setup] seed={args.seed}  out_suffix={out_suffix}  lr={args.lr:.0e}  "
          f"accum={args.accum}  epochs={args.epochs}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[setup] loading data...", flush=True)
    train_records = load_jsonl(Path(args.train_jsonl))
    val_records = load_jsonl(Path(args.val_jsonl))
    print(f"  train_records={len(train_records)}  val_records={len(val_records)}", flush=True)

    print(f"[setup] tokenizing + budget-filtering...", flush=True)
    train_examples: list[tuple[dict, tuple, float]] = []
    skipped_train = 0
    for r in train_records:
        parsed = build_phase_inputs(r, tokenizer, args.max_seq_length)
        if parsed is None:
            skipped_train += 1
            continue
        train_examples.append((r, parsed, float(r["ratio"])))
    val_examples: list[tuple[dict, tuple, float]] = []
    skipped_val = 0
    for r in val_records:
        parsed = build_phase_inputs(r, tokenizer, args.max_seq_length)
        if parsed is None:
            skipped_val += 1
            continue
        val_examples.append((r, parsed, float(r["ratio"])))
    print(f"  train_examples={len(train_examples)} (skipped {skipped_train})", flush=True)
    print(f"  val_examples={len(val_examples)} (skipped {skipped_val})", flush=True)

    print(f"\n[setup] base={args.base_model}  loading...", flush=True)
    base = fresh_base_model(args.base_model)
    peft_cfg = make_lora_config(cfg["lora"])
    _seed_everything(args.seed + 1000)  # match v1's reseed pattern before LoRA init
    adapted = get_peft_model(base, peft_cfg)
    inner = adapted.model  # Qwen2ForCausalLM (kvpress traverses .model.layers)
    # NOTE: not enabling gradient_checkpointing — our three-phase flow uses
    # Phase 1b's prefill under grad, and grad-ckpt + use_cache=True are
    # incompatible (transformers forces use_cache=False internally). H100
    # has 80GB, ctx <= 4096; full activations fit. If OOM in practice,
    # fall back to recomputing Phase 1b via torch.utils.checkpoint.
    adapted.config.use_cache = True
    n_trainable = sum(p.numel() for p in adapted.parameters() if p.requires_grad)
    print(f"  trainable LoRA params: {n_trainable:,}", flush=True)

    optim = torch.optim.AdamW(
        [p for p in adapted.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=cfg["optim"]["weight_decay"],
    )
    device = next(adapted.parameters()).device

    epoch_results = []
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== EPOCH {epoch}/{args.epochs} (lr={args.lr:.0e}) ===", flush=True)
        _seed_everything(args.seed + 2000 + epoch)
        t0 = time.time()
        train_loss, micro_times = train_epoch(
            adapted=adapted, inner=inner, train_examples=train_examples,
            optim=optim, device=device, accum=args.accum,
            smoke_steps=(args.smoke_steps if epoch == 1 else 0),
        )
        train_dt = time.time() - t0
        if args.smoke_steps:
            avg_dt = sum(micro_times) / max(len(micro_times), 1)
            p50 = sorted(micro_times)[len(micro_times)//2] if micro_times else 0.0
            p95 = sorted(micro_times)[int(0.95 * len(micro_times))] if micro_times else 0.0
            n_steps = len(micro_times)
            print(f"\n[SMOKE TIMING] {n_steps} micro-steps", flush=True)
            print(f"  avg dt:   {avg_dt:.2f}s", flush=True)
            print(f"  p50 dt:   {p50:.2f}s", flush=True)
            print(f"  p95 dt:   {p95:.2f}s", flush=True)
            print(f"  est full epoch: {avg_dt * len(train_examples) / 60:.1f} min", flush=True)
            print(f"  est 3-epoch run: {avg_dt * len(train_examples) * 3 / 3600:.2f} h", flush=True)
            return 0

        if args.skip_val:
            val_loss = float("nan")
        else:
            print(f"  computing val loss (compressed-cache regime)...", flush=True)
            v_t0 = time.time()
            val_loss = compute_val_loss(
                adapted=adapted, inner=inner, val_examples=val_examples,
                device=device,
            )
            print(f"  val_loss={val_loss:.4f}  ({time.time()-v_t0:.0f}s)", flush=True)

        ckpt_dir = out_root / f"v1_lora_epoch{epoch}{out_suffix}"
        adapted.save_pretrained(str(ckpt_dir))
        epoch_results.append({"epoch": epoch, "train_loss": train_loss,
                              "val_loss": val_loss, "wall_time_s": train_dt,
                              "ckpt_dir": str(ckpt_dir)})
        print(f"  epoch {epoch}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  ({train_dt:.0f}s train)", flush=True)

    if args.skip_val or any(np.isnan(e["val_loss"]) for e in epoch_results):
        # No val signal -> last epoch is "best"
        best = epoch_results[-1]
        print(f"\n  no val signal; defaulting to last epoch ({best['epoch']})",
              flush=True)
    else:
        best = min(epoch_results, key=lambda r: r["val_loss"])
        print(f"\n  best epoch by val loss: {best['epoch']}  "
              f"(val_loss={best['val_loss']:.4f})", flush=True)

    best_dir = out_root / f"v1_lora_best{out_suffix}"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(best["ckpt_dir"], best_dir)
    print(f"  copied best ckpt -> {best_dir.relative_to(ROOT)}", flush=True)

    # Per-epoch summary table to stdout (no markdown report — Step 3
    # eval comes next; aggregate report after all blocks done).
    print(f"\n=== TRAINING COMPLETE ===", flush=True)
    print(f"epoch | train_loss | val_loss | wall_time", flush=True)
    print(f"----- | ---------- | -------- | ---------", flush=True)
    for e in epoch_results:
        marker = "  ←best" if e["epoch"] == best["epoch"] else ""
        print(f"  {e['epoch']}   |   {e['train_loss']:.4f}   |  "
              f"{e['val_loss']:.4f}  |   {e['wall_time_s']:.0f}s{marker}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
