"""v1 real training + evaluation orchestrator.

Pipeline (~2 hours on A10):
  Phase 1: LR sweep at {5e-5, 1e-4, 2e-4} — 1 epoch each, pick by val loss.
  Phase 2: 3-epoch run at chosen LR; save adapter at end of each epoch;
           pick best checkpoint by val loss (NOT necessarily the last).
  Phase 3: Evaluate base model and best-adapter model on val_v1.jsonl
           with per-ratio honest_accuracy, hallucination_rate, abstention_rate.
  Phase 4: Qualitative comparison on the D5 wrong-but-specific examples
           (where the clean Qwen2.5-7B confidently hallucinated from
           distractor paragraphs in the post-compression fragment).
  Phase 5: Write artifacts/v1_run_report.md with everything.

Outputs:
  artifacts/v1_lora_sweep_lr*/   — adapter per LR (kept for inspection)
  artifacts/v1_lora_epoch*/      — adapter per epoch in best run
  artifacts/v1_lora_best/        — symlink-style copy of the chosen ckpt
  artifacts/v1_eval.jsonl        — per-instance classifications, base + adapter
  artifacts/v1_d5_qualitative.md — side-by-side base vs adapted on D5 fragments
  artifacts/v1_run_report.md     — top-level report
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def _seed_everything(seed: int) -> None:
    """Reseed python, numpy, torch (CPU + all CUDA devices).

    Called at the top of main() and before each non-deterministic operation
    (LoRA initialization in get_peft_model, DataLoader shuffle in
    train_epoch). Re-seeding before each point — rather than once at
    process start — guarantees deterministic behavior even if intermediate
    operations advance the global RNG state in unpredictable ways
    (e.g. torch.cuda allocator activity, third-party library RNG use).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.honest_accuracy import (  # noqa: E402
    aggregate_metrics,
    classify_response,
    looks_like_abstention,
)
from src.train.lora_trainer import (  # noqa: E402
    CompressedCacheCollator,
    FormattedExample,
    format_training_example,
)


# ----------------------------- shared utils -----------------------------


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class FormattedDataset(Dataset):
    def __init__(self, examples: list[FormattedExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> FormattedExample:
        return self.examples[idx]


def format_records(records: list[dict], tokenizer, max_seq_length: int):
    """Format records, dropping any that exceed seq length."""
    formatted: list[FormattedExample] = []
    record_idx_kept: list[int] = []
    skipped = 0
    for i, r in enumerate(records):
        try:
            f = format_training_example(
                context=r["context"],
                query=r["query"],
                target=r["target_output"],
                kept_mask=r["kept_mask"],
                tokenizer=tokenizer,
                apply_mask=True,
                max_seq_length=max_seq_length,
            )
            formatted.append(f)
            record_idx_kept.append(i)
        except ValueError:
            skipped += 1
    return formatted, record_idx_kept, skipped


def fresh_base_model(model_id: str):
    """bf16 base + flash_attention_2.

    flash_attention_2 is mandatory for Qwen2.5: sdpa falls back to a
    full-attention materialization because Qwen2.5 has sliding-window
    enabled, which sdpa doesn't support — that fallback alone OOMs the
    A10 at ~1500-token inputs and wastes memory on H100 too. flash-attn-2
    is already installed for KVzip, so this costs nothing.
    """
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    return base


def make_lora_config(cfg: dict) -> LoraConfig:
    return LoraConfig(
        r=cfg["r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        bias=cfg["bias"], target_modules=cfg["target_modules"],
        task_type=cfg["task_type"],
    )


# ----------------------------- training loop -----------------------------


def train_epoch(model, dataset, collator, optim, accum: int, log_every: int = 50) -> float:
    model.train()
    optim.zero_grad()
    loader = DataLoader(dataset, batch_size=1, collate_fn=collator, shuffle=True)
    micro = 0
    accum_loss_sum = 0.0
    accum_n = 0
    losses_unscaled: list[float] = []
    t0 = time.time()
    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss / accum
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at micro {micro}: {loss.item()}")
        loss.backward()
        losses_unscaled.append(out.loss.item())
        micro += 1
        if micro % accum == 0:
            optim.step()
            optim.zero_grad()
        if micro % log_every == 0:
            recent = sum(losses_unscaled[-log_every:]) / min(log_every, len(losses_unscaled))
            print(
                f"     micro={micro}/{len(loader)}  "
                f"recent_loss={recent:.4f}  "
                f"({time.time()-t0:.0f}s)",
                flush=True,
            )
    if micro % accum != 0:
        optim.step()
        optim.zero_grad()
    return sum(losses_unscaled) / max(len(losses_unscaled), 1)


@torch.no_grad()
def compute_val_loss(model, dataset, collator) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=1, collate_fn=collator, shuffle=False)
    losses: list[float] = []
    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(**batch)
        if torch.isfinite(out.loss):
            losses.append(out.loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


# ----------------------------- evaluation -----------------------------


PROMPT_PREFIX = "Context: "
PROMPT_MIDDLE = "\n\nQuestion: {query}\n\nAnswer:"


def build_eval_input_ids(record: dict, tokenizer, max_seq_length: int = 4096) -> list[int]:
    """Same as training format but WITHOUT target — leaves the model to generate."""
    context_ids = tokenizer(record["context"], add_special_tokens=False).input_ids
    if len(context_ids) != len(record["kept_mask"]):
        raise ValueError("kept_mask length mismatch")
    kept = [tid for tid, k in zip(context_ids, record["kept_mask"]) if k]
    prefix = tokenizer(PROMPT_PREFIX, add_special_tokens=False).input_ids
    middle = tokenizer(PROMPT_MIDDLE.format(query=record["query"]), add_special_tokens=False).input_ids
    input_ids = prefix + kept + middle
    if len(input_ids) > max_seq_length - 96:
        raise ValueError(f"eval input too long: {len(input_ids)}")
    return input_ids


@torch.no_grad()
def evaluate_on_val(model, tokenizer, val_records: list[dict], max_new_tokens: int = 64,
                    progress_every: int = 25) -> list[dict]:
    """Generate per record, classify, return list of result dicts."""
    model.eval()
    out: list[dict] = []
    t0 = time.time()
    for i, r in enumerate(val_records):
        try:
            input_ids = build_eval_input_ids(r, tokenizer)
        except ValueError:
            continue
        ids = torch.tensor([input_ids], device=model.device)
        gen = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(gen[0, ids.size(1):], skip_special_tokens=True).strip()
        cls = classify_response(
            response,
            gold_answer=r["metadata"]["answer"],
            answer_aliases=r["metadata"].get("answer_aliases", []),
            gold_label=r["label_type"],
        )
        out.append({
            "id": r["id"],
            "ratio": r["ratio"],
            "gold_label": r["label_type"],
            "gold_answer": r["metadata"]["answer"],
            "response": response,
            "classification": cls,
            "model_abstained": looks_like_abstention(response),
        })
        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"     eval {i+1}/{len(val_records)}  ({elapsed:.0f}s)", flush=True)
    return out


# ----------------------------- main pipeline -----------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "train" / "v1.yaml"))
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train-jsonl", default=str(ROOT / "artifacts" / "train_v1.jsonl"))
    ap.add_argument("--val-jsonl", default=str(ROOT / "artifacts" / "val_v1.jsonl"))
    ap.add_argument("--d5-jsonl", default=str(ROOT / "artifacts" / "fragment_recoverability.jsonl"))
    ap.add_argument("--lrs", default="5e-5,1e-4,2e-4")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for python+numpy+torch. Default 42 reproduces "
                         "the original v1 run; pass 43, 44, ... for variance checks.")
    ap.add_argument("--skip-base-eval", action="store_true",
                    help="Skip Phase 3 base-model eval. Use this for variance-check "
                         "runs where the base model is unchanged across seeds.")
    ap.add_argument("--output-suffix", default=None,
                    help="Override the auto-derived output-dir suffix. Use to train on a "
                         "different dataset with seed=42 without clobbering v1 artifacts "
                         "(e.g. --output-suffix _mixture_seed42).")
    ap.add_argument("--skip-sweep", action="store_true",
                    help="Skip Phase 1 LR sweep entirely; use args.lrs[0] directly for "
                         "Phase 2. Saves ~1 epoch of compute when the LR is already "
                         "decided (e.g. Exp 2 mixture run at fixed lr=2e-4).")
    args = ap.parse_args()

    # Reseed at process start. Re-seeded again before each LoRA init and
    # train_epoch below so the seed is the only thing controlling stochasticity.
    _seed_everything(args.seed)

    cfg = yaml.safe_load(Path(args.config).read_text())
    lrs = [float(x) for x in args.lrs.split(",")]
    out_root = ROOT / "artifacts"
    out_root.mkdir(parents=True, exist_ok=True)
    # Suffix output dirs when seed != 42 so variance-check runs don't clobber
    # the original v1 artifacts. --output-suffix overrides this auto-derivation
    # for cases like training on a different dataset with seed=42.
    if args.output_suffix is not None:
        out_suffix = args.output_suffix
    else:
        out_suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    print(f"\n[setup] seed={args.seed}  out_suffix={out_suffix or '(none)'}  "
          f"skip_base_eval={args.skip_base_eval}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"\n[setup] base={args.base_model}  lrs={lrs}  epochs={args.epochs}  accum={args.accum}", flush=True)
    print(f"[setup] loading data...", flush=True)
    train_records = load_jsonl(Path(args.train_jsonl))
    val_records = load_jsonl(Path(args.val_jsonl))
    print(f"  train_records={len(train_records)}  val_records={len(val_records)}", flush=True)

    print(f"[setup] formatting...", flush=True)
    train_formatted, _, train_skip = format_records(train_records, tokenizer, args.max_seq_length)
    val_formatted, val_idx_kept, val_skip = format_records(val_records, tokenizer, args.max_seq_length)
    print(f"  train_formatted={len(train_formatted)} (skipped {train_skip})", flush=True)
    print(f"  val_formatted={len(val_formatted)} (skipped {val_skip})", flush=True)

    train_ds = FormattedDataset(train_formatted)
    val_ds = FormattedDataset(val_formatted)
    val_records_kept = [val_records[i] for i in val_idx_kept]
    collator = CompressedCacheCollator(tokenizer)

    # ---------------- Phase 1: LR sweep ----------------
    if args.skip_sweep:
        print(f"\n=== PHASE 1: SKIP sweep (--skip-sweep); using lrs[0]={lrs[0]:.0e} for Phase 2 ===",
              flush=True)
        sweep_results = [{"lr": lrs[0], "train_loss": float("nan"),
                          "val_loss": float("nan"), "wall_time_s": 0,
                          "adapter_dir": "(skipped)"}]
        best_lr_run = sweep_results[0]
        # Skip the sweep loop entirely
        # (jump directly to "best LR" section + Phase 2)
        print(f"  best LR by val loss: {best_lr_run['lr']:.0e} (forced via --skip-sweep)", flush=True)
        sweep_skipped = True
    else:
        sweep_skipped = False
    if not sweep_skipped:
        print(f"\n=== PHASE 1: LR sweep ({len(lrs)} runs of 1 epoch) ===", flush=True)
    sweep_results = sweep_results if sweep_skipped else []
    for sweep_idx, lr in (() if sweep_skipped else enumerate(lrs)):
        print(f"\n  >>> sweep lr={lr:.0e}", flush=True)
        # Reseed before LoRA init (lora_A is randomly initialized).
        _seed_everything(args.seed + 100 + sweep_idx)
        base = fresh_base_model(args.base_model)
        peft_cfg = make_lora_config(cfg["lora"])
        model = get_peft_model(base, peft_cfg)
        # Re-enable gradient checkpointing AFTER get_peft_model — PEFT
        # clobbers the base's gradient_checkpointing setting during wrap.
        # use_reentrant=False is the modern recommendation.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        model.config.use_cache = False
        optim = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=cfg["optim"]["weight_decay"],
        )
        t0 = time.time()
        # Reseed before train_epoch (DataLoader shuffle uses global RNG).
        _seed_everything(args.seed + 200 + sweep_idx)
        train_loss = train_epoch(model, train_ds, collator, optim, args.accum)
        train_dt = time.time() - t0
        val_loss = compute_val_loss(model, val_ds, collator)
        adapter_dir = out_root / f"v1_lora_sweep_lr{lr:.0e}{out_suffix}"
        model.save_pretrained(str(adapter_dir))
        sweep_results.append({"lr": lr, "train_loss": train_loss, "val_loss": val_loss,
                              "wall_time_s": train_dt, "adapter_dir": str(adapter_dir)})
        print(f"  lr={lr:.0e}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({train_dt:.0f}s)", flush=True)
        del model, base, optim
        torch.cuda.empty_cache()

    if not sweep_skipped:
        best_lr_run = min(sweep_results, key=lambda r: r["val_loss"])
        print(f"\n  best LR by val loss: {best_lr_run['lr']:.0e}  (val_loss={best_lr_run['val_loss']:.4f})", flush=True)
    # else: best_lr_run was set in the skip-sweep branch above

    # ---------------- Phase 2: best-LR full training ----------------
    print(f"\n=== PHASE 2: full {args.epochs}-epoch run at lr={best_lr_run['lr']:.0e} ===", flush=True)
    # Reseed before Phase 2 LoRA init.
    _seed_everything(args.seed + 1000)
    base = fresh_base_model(args.base_model)
    peft_cfg = make_lora_config(cfg["lora"])
    model = get_peft_model(base, peft_cfg)
    # Same fix as phase 1: re-enable grad ckpt AFTER PEFT wrap.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.config.use_cache = False
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=best_lr_run["lr"], weight_decay=cfg["optim"]["weight_decay"],
    )

    epoch_results = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\n  >>> epoch {epoch}/{args.epochs}", flush=True)
        # Reseed before each train_epoch (per-epoch DataLoader shuffle).
        _seed_everything(args.seed + 2000 + epoch)
        train_loss = train_epoch(model, train_ds, collator, optim, args.accum)
        train_dt = time.time() - t0
        val_loss = compute_val_loss(model, val_ds, collator)
        ckpt_dir = out_root / f"v1_lora_epoch{epoch}{out_suffix}"
        model.save_pretrained(str(ckpt_dir))
        epoch_results.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                              "wall_time_s": train_dt, "ckpt_dir": str(ckpt_dir)})
        print(f"  epoch {epoch}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({train_dt:.0f}s)", flush=True)

    best_epoch = min(epoch_results, key=lambda r: r["val_loss"])
    print(f"\n  best epoch by val loss: {best_epoch['epoch']}  (val_loss={best_epoch['val_loss']:.4f})", flush=True)

    # symlink-style copy of the chosen checkpoint for downstream loading
    best_dir = out_root / f"v1_lora_best{out_suffix}"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(best_epoch["ckpt_dir"], best_dir)
    print(f"  copied best ckpt -> {best_dir.relative_to(ROOT)}", flush=True)

    del model, base, optim
    torch.cuda.empty_cache()

    # ---------------- Phase 3: eval base + best adapter ----------------
    eval_base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    eval_base.eval()
    if args.skip_base_eval:
        print(f"\n=== PHASE 3: SKIP base eval (--skip-base-eval) ===", flush=True)
        base_eval = []
        base_metrics = {"overall": {}, "per_ratio": {}}
    else:
        print(f"\n=== PHASE 3: eval BASE on val_v1 ===", flush=True)
        base_eval = evaluate_on_val(eval_base, tokenizer, val_records_kept)
        base_metrics = aggregate_metrics(base_eval)

    print(f"\n=== PHASE 3: eval ADAPTER on val_v1 ===", flush=True)
    adapted = PeftModel.from_pretrained(eval_base, str(best_dir))
    adapted.eval()
    adapted_eval = evaluate_on_val(adapted, tokenizer, val_records_kept)
    adapted_metrics = aggregate_metrics(adapted_eval)

    # write per-instance jsonl with both labels
    eval_jsonl = out_root / f"v1_eval{out_suffix}.jsonl"
    with eval_jsonl.open("w") as f:
        # interleave by id so each row pairs base vs adapter
        by_id = defaultdict(dict)
        for r in base_eval:
            by_id[(r["id"], r["ratio"])]["base"] = r
        for r in adapted_eval:
            by_id[(r["id"], r["ratio"])]["adapter"] = r
        for key, both in by_id.items():
            row = {"id": key[0], "ratio": key[1],
                   "gold_label": (both.get("base") or both.get("adapter"))["gold_label"],
                   "gold_answer": (both.get("base") or both.get("adapter"))["gold_answer"],
                   "base": both.get("base"), "adapter": both.get("adapter")}
            f.write(json.dumps(row) + "\n")
    print(f"\n  wrote {eval_jsonl.relative_to(ROOT)}", flush=True)

    # For variance-check runs (--skip-base-eval), the seed-comparison only
    # cares about the adapter's per-ratio numbers in the eval JSONL above.
    # Phase 4 (D5 qualitative) and Phase 5 (run report) compare base vs
    # adapter, which is moot when base eval was skipped. Exit cleanly.
    if args.skip_base_eval:
        print(f"\n=== SKIP Phase 4 + Phase 5 (--skip-base-eval) ===", flush=True)
        print(f"\n=== v1 RUN COMPLETE (variance-check mode) ===", flush=True)
        return 0

    # ---------------- Phase 4: D5 qualitative ----------------
    print(f"\n=== PHASE 4: D5 wrong-but-specific qualitative comparison ===", flush=True)
    d5_records = load_jsonl(Path(args.d5_jsonl))
    # The D5 judge was Qwen2.5-7B-Instruct (the same model we just adapted).
    # All 30 D5 examples were ABSTAIN-labeled and the judge produced 0 correct.
    # The wrong-but-specific subset is responses that CHASED a wrong answer
    # rather than honestly hedging.
    d5_picks = []
    for r in d5_records:
        resp = r.get("judge_response", "").strip()
        if not resp:
            continue
        if looks_like_abstention(resp):
            continue
        # fragment from r already has decoded_fragment, question, answer
        d5_picks.append(r)
    d5_picks = d5_picks[:10]  # cap for the qualitative MD

    qualitative_rows = []
    for r in d5_picks:
        # rebuild the same prompt the D5 judge saw
        prompt = (
            f"Context: {r['decoded_fragment']}\n\n"
            f"Question: {r['question']}\n\n"
            f"Answer:"
        )
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(adapted.device)
        if ids.size(1) > args.max_seq_length - 96:
            continue
        with torch.no_grad():
            with adapted.disable_adapter():
                base_gen = adapted.generate(
                    ids, max_new_tokens=64, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                )
            adapter_gen = adapted.generate(
                ids, max_new_tokens=64, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        base_resp = tokenizer.decode(base_gen[0, ids.size(1):], skip_special_tokens=True).strip()
        adapter_resp = tokenizer.decode(adapter_gen[0, ids.size(1):], skip_special_tokens=True).strip()
        qualitative_rows.append({
            "id": r["id"],
            "question": r["question"],
            "gold_answer": r["answer"],
            "d5_judge_response": r["judge_response"],
            "base_v1_response": base_resp,
            "adapter_v1_response": adapter_resp,
        })

    qual_md = ROOT / "artifacts" / f"v1_d5_qualitative{out_suffix}.md"
    md = ["# v1 — qualitative comparison on D5 wrong-but-specific examples\n",
          "_Each example was labeled ABSTAIN by our pipeline. The D5 judge "
          "(clean Qwen2.5-7B-Instruct) confidently emitted a wrong specific "
          "answer instead of abstaining. Below: the same input given to base "
          "Qwen2.5-7B-Instruct (re-confirmed) and to the v1-adapted model._\n",
          f"_n={len(qualitative_rows)} examples_\n", "---\n"]
    for i, q in enumerate(qualitative_rows, 1):
        md.append(f"## {i}. `{q['id']}`")
        md.append(f"**Question**: {q['question']}")
        md.append(f"**Gold answer**: `{q['gold_answer']}`")
        md.append(f"\n**D5 judge response (Qwen2.5-7B base, original D5 run)**:")
        md.append(f"> {q['d5_judge_response']}")
        md.append(f"\n**BASE response (re-run for parity)**:")
        md.append(f"> {q['base_v1_response']}")
        md.append(f"\n**v1-ADAPTED response**:")
        md.append(f"> {q['adapter_v1_response']}")
        md.append("\n---\n")
    qual_md.write_text("\n".join(md))
    print(f"  wrote {qual_md.relative_to(ROOT)}", flush=True)

    # ---------------- Phase 5: top-level report ----------------
    report = ROOT / "artifacts" / f"v1_run_report{out_suffix}.md"
    md: list[str] = []
    md.append("# v1 — real training run report\n")
    md.append("_Auto-generated by `scripts/run_v1_full.py`._\n")
    md.append(f"## Setup\n- base model: `{args.base_model}`")
    md.append(f"- train: {len(train_formatted)} instances  ({args.train_jsonl})")
    md.append(f"- val:   {len(val_formatted)} instances  ({args.val_jsonl})")
    md.append(f"- LoRA: r={cfg['lora']['r']} alpha={cfg['lora']['lora_alpha']} "
              f"dropout={cfg['lora']['lora_dropout']} target={cfg['lora']['target_modules']}")
    md.append(f"- accum={args.accum} epochs={args.epochs} bf16=True grad_ckpt=True\n")

    md.append("## Phase 1 — LR sweep (1 epoch each)\n")
    md.append("| lr | train_loss | val_loss | wall_time |")
    md.append("| --- | --- | --- | --- |")
    for s in sweep_results:
        marker = " ←best" if s["lr"] == best_lr_run["lr"] else ""
        md.append(f"| {s['lr']:.0e} | {s['train_loss']:.4f} | {s['val_loss']:.4f} | {s['wall_time_s']:.0f}s |{marker}")
    md.append("")

    md.append(f"## Phase 2 — full {args.epochs}-epoch run at lr={best_lr_run['lr']:.0e}\n")
    md.append("| epoch | train_loss | val_loss | wall_time |")
    md.append("| --- | --- | --- | --- |")
    for e in epoch_results:
        marker = " ←best" if e["epoch"] == best_epoch["epoch"] else ""
        md.append(f"| {e['epoch']} | {e['train_loss']:.4f} | {e['val_loss']:.4f} | {e['wall_time_s']:.0f}s |{marker}")
    md.append(f"\nBest checkpoint: epoch {best_epoch['epoch']} -> `artifacts/v1_lora_best/`\n")

    def fmt_rates(m: dict) -> str:
        return (f"acc={m.get('honest_accuracy', 0):.1%}  "
                f"halluc={m.get('hallucination_rate', 0):.1%}  "
                f"abst={m.get('abstention_rate', 0):.1%}  "
                f"(n={m.get('n', 0)})")

    md.append("## Phase 3 — per-ratio metrics on val_v1\n")
    md.append("Aggregate:\n")
    md.append(f"- BASE     : {fmt_rates(base_metrics['overall'])}")
    md.append(f"- ADAPTER  : {fmt_rates(adapted_metrics['overall'])}\n")
    md.append("Per ratio (CRITICAL — mid-ratios are the headline):\n")
    md.append("| ratio | model | n | honest_acc | halluc | abst | n_gold_conf | n_gold_abst |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for ratio in sorted(set(base_metrics["per_ratio"].keys()) | set(adapted_metrics["per_ratio"].keys())):
        for tag, metrics in [("base", base_metrics), ("adapter", adapted_metrics)]:
            m = metrics["per_ratio"].get(ratio, {})
            if not m:
                continue
            md.append(
                f"| {ratio:.2f} | {tag} | {m['n']} | "
                f"{m['honest_accuracy']:.1%} | "
                f"{m['hallucination_rate']:.1%} | "
                f"{m['abstention_rate']:.1%} | "
                f"{m['n_gold_confident']} | {m['n_gold_abstain']} |"
            )
    md.append("")

    md.append("## Phase 4 — D5 qualitative\n")
    md.append(f"See `artifacts/v1_d5_qualitative.md` for {len(qualitative_rows)} side-by-side examples.\n")

    md.append("## Files\n")
    md.append("- `artifacts/v1_lora_best/` — best adapter (PEFT save_pretrained)")
    md.append("- `artifacts/v1_lora_sweep_lr*` — LR-sweep adapters (kept for inspection)")
    md.append("- `artifacts/v1_lora_epoch*` — per-epoch adapters from the full run")
    md.append("- `artifacts/v1_eval.jsonl` — per-instance base+adapter classifications")
    md.append("- `artifacts/v1_d5_qualitative.md` — side-by-side base vs adapter responses")

    report.write_text("\n".join(md))
    print(f"\n[done] wrote {report.relative_to(ROOT)}", flush=True)
    print(f"\n=== v1 RUN COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
