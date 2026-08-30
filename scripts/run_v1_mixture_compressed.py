"""Block 3.6 — multi-press LoRA training under compressed-cache decoding.

Mixture training (KVzip + SnapKV + ExpAttn jointly) where the press at
training time matches the per-example metadata.compressor field.

Per-step flow dispatches on the example's compressor:

  KVzip examples (3-phase — chunked context-reconstruction scoring blows
  up the autograd graph if run under grad):
    Phase 1a  no_grad  with KVzipPress: prefill ctx → set masked_key_indices
    Phase 1b  grad     re-prefill ctx (no press) → cache_b, differentiable
    Phase 2   grad     target forward, attention_patch consumes the
                       restored masked_key_indices

  SnapKV / ExpAttn examples (2-phase — physical eviction during prefill
  via per-layer forward_hook; single forward suffices):
    Phase 1   grad     with press(inner): prefill ctx → cache physically
                       compressed inside the with-block
    Phase 2   grad     target forward with the (physically smaller) cache

Sub-epoch checkpointing every 25% of an epoch (4 ckpts/epoch), with a
qualitative fluency spot-check on r=0.80 CONFIDENT-gold val examples
under KVzip cc inference (the deployment regime). Block 3.5 showed val
loss alone is unreliable as a checkpoint selector — the model can have
low val loss while emitting sub-token gibberish at high retention.

Run from .venv-kvpress.
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

# Set BEFORE importing transformers/kvpress: KVzipPress.__call__ re-runs
# AutoTokenizer.from_pretrained per entry; with mixture's 8772 examples ×
# epochs, that's many thousand HF API hits → 429.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

_orig_atk_fp = AutoTokenizer.from_pretrained
_AT_CACHE: dict = {}
def _cached_from_pretrained(model_id, *a, **kw):
    key = str(model_id)
    if key not in _AT_CACHE:
        _AT_CACHE[key] = _orig_atk_fp(model_id, *a, **kw)
    return _AT_CACHE[key]
AutoTokenizer.from_pretrained = _cached_from_pretrained

from kvpress import KVzipPress, SnapKVPress, ExpectedAttentionPress

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


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def build_phase_inputs(record, tokenizer, max_seq_length):
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
    if len(ctx_part) + len(qt_part) > max_seq_length:
        return None
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


# --------------- per-press training-step shapes ---------------


def kvzip_step(adapted, inner, parsed, ratio, device, *, no_grad_phase1b=False):
    """KVzip 3-phase step. See module docstring."""
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
    if no_grad_phase1b:
        with torch.no_grad():
            _ = adapted(input_ids=ctx, past_key_values=cache_b, use_cache=True)
    else:
        _ = adapted(input_ids=ctx, past_key_values=cache_b, use_cache=True)
    restore_masks(inner, saved)

    n_ctx = ctx.size(1)
    cache_position = torch.arange(n_ctx, n_ctx + qt.size(1), device=device)
    out = adapted(input_ids=qt, past_key_values=cache_b, cache_position=cache_position,
                  labels=qt_lab, use_cache=True)
    clear_masks(inner)
    return out.loss


def two_phase_press_step(adapted, inner, parsed, ratio, device, press_cls, **press_kwargs):
    """SnapKV / ExpAttn 2-phase step. Both phases under grad."""
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

    cache_position = torch.arange(n_kept, n_kept + qt.size(1), device=device)
    out = adapted(input_ids=qt, past_key_values=cache, cache_position=cache_position,
                  labels=qt_lab, use_cache=True)
    return out.loss


PRESS_DISPATCH = {
    "kvzip": ("kvzip", lambda **kw: KVzipPress(**kw)),
    "snapkv": ("two_phase", lambda **kw: SnapKVPress(window_size=64, kernel_size=5, **kw)),
    "expected_attention": ("two_phase", lambda **kw: ExpectedAttentionPress(**kw)),
}


def step_for_record(adapted, inner, parsed, ratio, device, compressor, *, no_grad_phase1b=False):
    if compressor not in PRESS_DISPATCH:
        raise ValueError(f"unknown compressor {compressor!r}")
    flow_kind, _ = PRESS_DISPATCH[compressor]
    if flow_kind == "kvzip":
        return kvzip_step(adapted, inner, parsed, ratio, device,
                          no_grad_phase1b=no_grad_phase1b)
    if compressor == "snapkv":
        return two_phase_press_step(adapted, inner, parsed, ratio, device,
                                    SnapKVPress, window_size=64, kernel_size=5)
    if compressor == "expected_attention":
        return two_phase_press_step(adapted, inner, parsed, ratio, device,
                                    ExpectedAttentionPress)
    raise RuntimeError("unreachable")


# --------------- val + spot-check ---------------


@torch.no_grad()
def compute_val_loss(adapted, inner, val_examples, device, max_n=None):
    """Val loss under KVzip cc regime (the deployment regime). Subset to
    max_n if specified for cheaper sub-epoch checks."""
    adapted.eval()
    losses = []
    n = max_n if max_n else len(val_examples)
    for rec, parsed, ratio in val_examples[:n]:
        # All val records use KVzip cc inference for the loss measurement
        # (matches deployment; the in-training compressor used label-conditional
        # KVzip masks via train_v1.jsonl-style data).
        loss = step_for_record(adapted, inner, parsed, ratio, device,
                               compressor="kvzip", no_grad_phase1b=True)
        if torch.isfinite(loss):
            losses.append(float(loss.item()))
    adapted.train()
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def spot_check_fluency(adapted, inner, tokenizer, spot_records, device, n=5):
    """Decode first n spot_records under KVzip cc inference; return
    (record_id, response) pairs. The training log captures these so a
    human can scan for sub-token gibberish ('Acfairs', '750pxI...')."""
    adapted.eval()
    out = []
    for r in spot_records[:n]:
        ctx_full = PROMPT_PREFIX + r["context"]
        ctx_ids = tokenizer(ctx_full, add_special_tokens=False).input_ids
        mid_ids = tokenizer(
            PROMPT_MIDDLE_FMT.format(query=r["query"]).rstrip(),
            add_special_tokens=False,
        ).input_ids
        # Strip trailing "Answer: " whitespace consistency: use the same
        # PROMPT_MIDDLE_FMT shape as build_phase_inputs.
        mid_ids = tokenizer(
            PROMPT_MIDDLE_FMT.format(query=r["query"]),
            add_special_tokens=False,
        ).input_ids
        ctx_t = torch.tensor([ctx_ids], device=device)
        q_t = torch.tensor([mid_ids], device=device)
        cache = DynamicCache()
        cr = max(0.0, min(1.0 - r["ratio"], 0.999))
        press = KVzipPress(compression_ratio=cr)
        with press(inner):
            _ = inner(input_ids=ctx_t, past_key_values=cache, use_cache=True)
        n_kept = cache.layers[0].keys.shape[2]
        cache_position = torch.arange(n_kept, n_kept + q_t.size(1), device=device)
        gen = adapted.generate(
            input_ids=q_t, past_key_values=cache, cache_position=cache_position,
            max_new_tokens=48, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
        if gen.size(1) > q_t.size(1):
            new_ids = gen[0, q_t.size(1):]
        else:
            new_ids = gen[0]
        resp = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        out.append((r["id"], r["metadata"]["answer"], resp))
        clear_masks(inner)
    adapted.train()
    return out


# --------------- training loop ---------------


def train_one_epoch_with_subckpts(
    adapted, inner, tokenizer, train_examples, val_examples, spot_records,
    optim, accum, device, *, epoch_idx, n_epochs,
    out_root: Path, out_suffix: str, ckpt_per_epoch: int = 4,
    smoke_steps: int = 0,
):
    """Run one epoch with sub-epoch checkpointing + spot-check + val.

    Saves at fractional positions (1/N, 2/N, ..., 1.0) of the epoch.
    """
    n = len(train_examples)
    order = list(range(n))
    random.shuffle(order)
    losses_unscaled = []
    micro_times = []
    micro = 0
    optim.zero_grad()
    t_epoch_start = time.time()

    # checkpoint trigger micros (after micro N is processed and accum may have stepped)
    ckpt_micros = [int(round(n * (k + 1) / ckpt_per_epoch)) for k in range(ckpt_per_epoch)]

    ckpt_log = []  # list of {micro, train_loss_recent, val_loss_partial, spot, ckpt_dir}
    for idx in order:
        rec, parsed, ratio = train_examples[idx]
        compressor = rec["metadata"]["compressor"]
        m_t0 = time.time()
        try:
            loss = step_for_record(adapted, inner, parsed, ratio, device, compressor)
        except Exception as e:
            print(f"     [WARN] step failed at micro {micro} (id={rec['id']}, "
                  f"compressor={compressor}): {type(e).__name__}: {str(e)[:100]} — skipping",
                  flush=True)
            continue
        if not torch.isfinite(loss):
            print(f"     [WARN] non-finite loss at micro {micro} (id={rec['id']}); skipping",
                  flush=True)
            continue
        scaled = loss / accum
        scaled.backward()
        m_dt = time.time() - m_t0
        micro_times.append(m_dt)
        losses_unscaled.append(loss.item())
        micro += 1
        if micro % accum == 0:
            optim.step()
            optim.zero_grad()
        if micro % 50 == 0:
            recent = sum(losses_unscaled[-50:]) / min(50, len(losses_unscaled))
            recent_t = sum(micro_times[-50:]) / min(50, len(micro_times))
            print(
                f"     [e{epoch_idx}] micro={micro}/{n}  recent_loss={recent:.4f}  "
                f"recent_dt={recent_t:.2f}s  ({time.time()-t_epoch_start:.0f}s elapsed)",
                flush=True,
            )
        # sub-epoch checkpoint
        if micro in ckpt_micros:
            frac = ckpt_micros.index(micro) + 1
            print(f"\n     [e{epoch_idx} sub-ckpt {frac}/{ckpt_per_epoch}] "
                  f"micro={micro}/{n} — saving + spot-check + partial val",
                  flush=True)
            ckpt_dir = out_root / f"v1_lora_mixture_cc{out_suffix}_e{epoch_idx}_q{frac}"
            adapted.save_pretrained(str(ckpt_dir))
            v_t0 = time.time()
            partial_val = compute_val_loss(adapted, inner, val_examples, device, max_n=80)
            v_dt = time.time() - v_t0
            spot = spot_check_fluency(adapted, inner, tokenizer, spot_records, device, n=5)
            print(f"     [e{epoch_idx} sub-ckpt {frac}/{ckpt_per_epoch}] "
                  f"partial_val_loss={partial_val:.4f}  ({v_dt:.0f}s)", flush=True)
            print(f"     [e{epoch_idx} sub-ckpt {frac}/{ckpt_per_epoch}] spot-check responses (r=0.80, C-gold, KVzip cc):",
                  flush=True)
            for sid, gold, resp in spot:
                print(f"        id={sid}", flush=True)
                print(f"        gold: {gold!r}", flush=True)
                print(f"        resp: {resp[:140]!r}", flush=True)
            ckpt_log.append({
                "epoch": epoch_idx, "frac": frac, "micro": micro,
                "partial_val_loss": partial_val, "ckpt_dir": str(ckpt_dir),
                "spot": [(sid, gold, resp) for sid, gold, resp in spot],
            })
        if smoke_steps and micro >= smoke_steps:
            print(f"     [smoke] stopping after {smoke_steps} micro-steps", flush=True)
            break

    if not smoke_steps and micro % accum != 0:
        optim.step()
        optim.zero_grad()
    avg_loss = sum(losses_unscaled) / max(len(losses_unscaled), 1)
    epoch_dt = time.time() - t_epoch_start
    return avg_loss, epoch_dt, micro_times, ckpt_log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "train" / "v1.yaml"))
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train-jsonl", default=str(ROOT / "artifacts" / "train_mixture_v1.jsonl"))
    ap.add_argument("--val-jsonl", default=str(ROOT / "artifacts" / "val_v1.jsonl"))
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-suffix", required=True,
                    help="e.g. _lr1e-4_seed42 — appended to all artifact names")
    ap.add_argument("--smoke-steps", type=int, default=0)
    ap.add_argument("--ckpt-per-epoch", type=int, default=4,
                    help="Sub-epoch checkpoints per epoch (default 4 = every 25%%).")
    args = ap.parse_args()

    _seed_everything(args.seed)
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_root = ROOT / "artifacts"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"\n[setup] seed={args.seed}  out_suffix={args.output_suffix}  "
          f"lr={args.lr:.0e}  accum={args.accum}  epochs={args.epochs}  "
          f"ckpt_per_epoch={args.ckpt_per_epoch}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[setup] loading data...", flush=True)
    train_records = load_jsonl(Path(args.train_jsonl))
    val_records = load_jsonl(Path(args.val_jsonl))
    print(f"  train_records={len(train_records)}  val_records={len(val_records)}", flush=True)

    print(f"[setup] tokenizing + budget-filtering...", flush=True)
    train_examples: list[tuple] = []
    skipped_train = 0
    for r in train_records:
        parsed = build_phase_inputs(r, tokenizer, args.max_seq_length)
        if parsed is None:
            skipped_train += 1
            continue
        train_examples.append((r, parsed, float(r["ratio"])))

    val_examples: list[tuple] = []
    skipped_val = 0
    for r in val_records:
        parsed = build_phase_inputs(r, tokenizer, args.max_seq_length)
        if parsed is None:
            skipped_val += 1
            continue
        val_examples.append((r, parsed, float(r["ratio"])))
    print(f"  train_examples={len(train_examples)} (skipped {skipped_train})", flush=True)
    print(f"  val_examples={len(val_examples)} (skipped {skipped_val})", flush=True)

    # Spot-check records: r=0.80 + confident_answer (≤5)
    spot_records = [
        r for r in val_records
        if abs(r["ratio"] - 0.80) < 1e-9 and r["label_type"] == "confident_answer"
    ][:5]
    print(f"  spot_records={len(spot_records)} (r=0.80, C-gold)", flush=True)

    by_compressor = {}
    for r, _, _ in train_examples:
        by_compressor[r["metadata"]["compressor"]] = by_compressor.get(
            r["metadata"]["compressor"], 0) + 1
    print(f"  train compressor mix: {by_compressor}", flush=True)

    print(f"\n[setup] base={args.base_model}  loading...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    peft_cfg = LoraConfig(
        r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        target_modules=cfg["lora"]["target_modules"],
        task_type=cfg["lora"]["task_type"],
    )
    _seed_everything(args.seed + 1000)
    adapted = get_peft_model(base, peft_cfg)
    inner = adapted.model
    adapted.config.use_cache = True
    n_trainable = sum(p.numel() for p in adapted.parameters() if p.requires_grad)
    print(f"  trainable LoRA params: {n_trainable:,}", flush=True)

    optim = torch.optim.AdamW(
        [p for p in adapted.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=cfg["optim"]["weight_decay"],
    )
    device = next(adapted.parameters()).device

    epoch_results = []
    all_ckpts = []
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== EPOCH {epoch}/{args.epochs} (lr={args.lr:.0e}, suffix={args.output_suffix}) ===",
              flush=True)
        _seed_everything(args.seed + 2000 + epoch)
        avg_loss, epoch_dt, micro_times, ckpt_log = train_one_epoch_with_subckpts(
            adapted, inner, tokenizer, train_examples, val_examples, spot_records,
            optim, args.accum, device,
            epoch_idx=epoch, n_epochs=args.epochs,
            out_root=out_root, out_suffix=args.output_suffix,
            ckpt_per_epoch=args.ckpt_per_epoch,
            smoke_steps=(args.smoke_steps if epoch == 1 else 0),
        )
        all_ckpts.extend(ckpt_log)
        if args.smoke_steps:
            avg_dt = sum(micro_times) / max(len(micro_times), 1)
            p50 = sorted(micro_times)[len(micro_times)//2] if micro_times else 0.0
            p95 = sorted(micro_times)[int(0.95 * len(micro_times))] if micro_times else 0.0
            print(f"\n[SMOKE TIMING] {len(micro_times)} micro-steps", flush=True)
            print(f"  avg dt:   {avg_dt:.2f}s", flush=True)
            print(f"  p50 dt:   {p50:.2f}s", flush=True)
            print(f"  p95 dt:   {p95:.2f}s", flush=True)
            print(f"  est full epoch: {avg_dt * len(train_examples) / 60:.1f} min", flush=True)
            print(f"  est 3-epoch run: {avg_dt * len(train_examples) * 3 / 3600:.2f} h", flush=True)
            return 0

        # End-of-epoch full val
        v_t0 = time.time()
        full_val = compute_val_loss(adapted, inner, val_examples, device)
        v_dt = time.time() - v_t0
        print(f"\n  [e{epoch}] full_val_loss={full_val:.4f}  ({v_dt:.0f}s)", flush=True)
        ckpt_dir = out_root / f"v1_lora_mixture_cc{args.output_suffix}_e{epoch}_full"
        adapted.save_pretrained(str(ckpt_dir))
        epoch_results.append({"epoch": epoch, "train_loss": avg_loss,
                              "val_loss": full_val, "wall_time_s": epoch_dt,
                              "ckpt_dir": str(ckpt_dir)})
        all_ckpts.append({"epoch": epoch, "frac": "FULL", "micro": len(train_examples),
                          "partial_val_loss": full_val, "ckpt_dir": str(ckpt_dir),
                          "spot": []})
        print(f"  [e{epoch}] avg_train_loss={avg_loss:.4f}  full_val_loss={full_val:.4f}  "
              f"({epoch_dt:.0f}s train)", flush=True)

    # Summary table of all checkpoints
    print(f"\n=== ALL CHECKPOINTS (lr={args.lr:.0e}) ===", flush=True)
    print(f"epoch | frac | micro |   val_loss   | dir", flush=True)
    print(f"----- | ---- | ----- | ------------ | ---", flush=True)
    for c in all_ckpts:
        print(f"  {c['epoch']}   |  {str(c['frac']):>4s} | {c['micro']:5d} | "
              f"{c['partial_val_loss']:12.4f} | {Path(c['ckpt_dir']).name}",
              flush=True)

    # Pick best by val loss; user can override via the spot-check log
    best = min(all_ckpts, key=lambda c: c["partial_val_loss"])
    print(f"\n  best ckpt by val loss: epoch={best['epoch']}, frac={best['frac']}, "
          f"val_loss={best['partial_val_loss']:.4f}", flush=True)
    best_dir = out_root / f"v1_lora_mixture_cc{args.output_suffix}_best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(best["ckpt_dir"], best_dir)
    print(f"  copied -> {best_dir.relative_to(ROOT)}", flush=True)

    print(f"\n=== TRAINING COMPLETE (lr={args.lr:.0e}) ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
