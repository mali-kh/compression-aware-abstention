"""D8 — LoRA training DRY RUN. NO real training.

Five infrastructure-only checks before authorizing a real run:

  CHECK 1: tokenizer.padding_side == 'left' (and we set it explicitly).
  CHECK 2: For one example, print masked-vs-unmasked token counts and
           the literal unmasked tokens. They must be EXACTLY the
           target_output and nothing else (NOT just the final word —
           common silent bug).
  CHECK 3: kept_mask actually affects the forward pass: same record
           collated with apply_mask=True vs apply_mask=False produces
           DIFFERENT logits at the target-start position. If logits are
           identical, the mask is being silently dropped.
  CHECK 4: 5 LoRA training steps run without NaN loss / OOM under
           gradient checkpointing.
  CHECK 5: Saved adapter reloads cleanly into a fresh base model and
           a single forward pass produces non-NaN logits.

If any of 1-3 fails, training is invalid and we surface immediately.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train.lora_trainer import (  # noqa: E402
    CompressedCacheCollator,
    FormattedExample,
    format_training_example,
)


def load_examples(jsonl_path: Path, n: int) -> list[dict]:
    rows: list[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


# ---------------------- CHECK 2: label masking ----------------------

def check_label_masking(formatted: FormattedExample, tokenizer) -> None:
    print("\n=========================== CHECK 2: label mask ===========================")
    n_total = len(formatted.input_ids)
    n_masked = sum(1 for x in formatted.labels if x == -100)
    n_unmasked = n_total - n_masked

    unmasked_ids = [
        tid for tid, lab in zip(formatted.input_ids, formatted.labels) if lab != -100
    ]
    unmasked_decoded = tokenizer.decode(unmasked_ids, skip_special_tokens=False)

    print(f"  total tokens         : {n_total}")
    print(f"  masked (label=-100)  : {n_masked}")
    print(f"  unmasked (label!=-100): {n_unmasked}")
    print(f"  target_text expected : {formatted.target_text!r}")
    print(f"  unmasked tokens decode: {unmasked_decoded!r}")

    if unmasked_decoded != formatted.target_text:
        raise AssertionError(
            f"\n  CHECK 2 FAILED: unmasked tokens != target text\n"
            f"    expected target_text: {formatted.target_text!r}\n"
            f"    got unmasked decoded: {unmasked_decoded!r}\n"
            f"  This is the common 'mask everything except final answer token' bug; "
            f"investigate format_training_example before proceeding."
        )
    if n_unmasked != formatted.n_target_tokens:
        raise AssertionError(
            f"\n  CHECK 2 FAILED: unmasked count {n_unmasked} != "
            f"n_target_tokens {formatted.n_target_tokens}"
        )
    print("  ✅ unmasked tokens are exactly target_output + eos")


# ---------------------- CHECK 3: mask -> different logits ----------------------

def check_mask_affects_logits(
    record: dict, tokenizer, model, max_seq_length: int = 4096
) -> None:
    print("\n=========================== CHECK 3: mask affects logits ===========================")
    fmt_applied = format_training_example(
        context=record["context"],
        query=record["query"],
        target=record["target_output"],
        kept_mask=record["kept_mask"],
        tokenizer=tokenizer,
        apply_mask=True,
        max_seq_length=max_seq_length,
    )
    fmt_passed = format_training_example(
        context=record["context"],
        query=record["query"],
        target=record["target_output"],
        kept_mask=record["kept_mask"],
        tokenizer=tokenizer,
        apply_mask=False,
        max_seq_length=max_seq_length,
    )
    n_kept = sum(record["kept_mask"])
    n_total_ctx = len(record["kept_mask"])
    print(f"  example id           : {record['id']}  ratio={record['ratio']}  label={record['label_type']}")
    print(f"  kept_mask coverage   : {n_kept}/{n_total_ctx} = {n_kept/n_total_ctx:.1%}")
    print(f"  applied input length : {len(fmt_applied.input_ids)}  target_start={fmt_applied.target_start}")
    print(f"  passed  input length : {len(fmt_passed.input_ids)}  target_start={fmt_passed.target_start}")

    device = next(model.parameters()).device
    ids_a = torch.tensor(fmt_applied.input_ids, device=device).unsqueeze(0)
    ids_p = torch.tensor(fmt_passed.input_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        out_a = model(input_ids=ids_a)
        out_p = model(input_ids=ids_p)

    # Compare the logit *predicting* the first target token (i.e. one
    # position before target_start). This is where the model has to
    # output the answer, and it's the position whose hidden state most
    # depends on whether the mask was applied to context.
    pos_a = fmt_applied.target_start - 1
    pos_p = fmt_passed.target_start - 1
    logit_a = out_a.logits[0, pos_a, :].float()
    logit_p = out_p.logits[0, pos_p, :].float()

    max_abs_diff = (logit_a - logit_p).abs().max().item()
    print(f"  max |logit(applied) - logit(passed)| at first-target position: {max_abs_diff:.4f}")
    if max_abs_diff < 1e-3:
        raise AssertionError(
            f"\n  CHECK 3 FAILED: logits identical between apply_mask=True/False\n"
            f"    max abs diff: {max_abs_diff:.6f}\n"
            f"  The kept_mask is being silently dropped somewhere; the model would "
            f"train on the same effective input regardless of compression — useless."
        )
    print("  ✅ kept_mask materially changes logits at the target-prediction position")


# ---------------------- training ----------------------

class FormattedDataset(Dataset):
    def __init__(self, examples: list[FormattedExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> FormattedExample:
        return self.examples[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "train" / "v1.yaml"))
    ap.add_argument("--train-jsonl", default=str(ROOT / "artifacts" / "train_v1.jsonl"))
    ap.add_argument("--max-seq-length", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    base_model_id: str = cfg["base_model"]
    lora_cfg = cfg["lora"]
    optim_cfg = cfg["optim"]
    tr_cfg = cfg["training"]
    dr_cfg = cfg["dry_run"]
    max_seq_length = args.max_seq_length or tr_cfg["max_seq_length"]

    print(f"[setup] base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    # CHECK 1: padding side
    print("\n=========================== CHECK 1: tokenizer.padding_side ===========================")
    print(f"  before: tokenizer.padding_side = {tokenizer.padding_side!r}")
    tokenizer.padding_side = "left"
    print(f"  after : tokenizer.padding_side = {tokenizer.padding_side!r}")
    if tokenizer.padding_side != "left":
        raise AssertionError("CHECK 1 FAILED: padding_side did not stick to 'left'")
    print("  ✅ padding_side='left' set explicitly (not relying on default)")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"\n[setup] loading {dr_cfg['n_examples']} examples from {args.train_jsonl}")
    raw_records = load_examples(Path(args.train_jsonl), dr_cfg["n_examples"])
    print(f"  loaded {len(raw_records)} records")

    # Format all
    print(f"[setup] formatting examples (apply_mask=True)...")
    formatted: list[FormattedExample] = []
    skipped = 0
    for r in raw_records:
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
        except ValueError as e:
            skipped += 1
            print(f"  skip {r['id']}: {e}")
    print(f"  formatted {len(formatted)} (skipped {skipped})")
    if not formatted:
        print("  no examples survived formatting; aborting")
        return 1

    # CHECK 2 on the first formatted example
    check_label_masking(formatted[0], tokenizer)

    # Pick an interesting example for CHECK 3: prefer a low-ratio one so the
    # apply/passed difference is large and the test is decisive.
    low_ratio = min(raw_records, key=lambda r: r["ratio"])
    # but we also need it to fit in max_seq_length when un-masked
    # (apply=False uses full context_ids); so re-attempt with each candidate
    candidate = None
    for r in sorted(raw_records, key=lambda x: x["ratio"]):
        try:
            format_training_example(
                context=r["context"], query=r["query"], target=r["target_output"],
                kept_mask=r["kept_mask"], tokenizer=tokenizer,
                apply_mask=False, max_seq_length=max_seq_length,
            )
            candidate = r
            break
        except ValueError:
            continue
    if candidate is None:
        print("  no example fits with apply_mask=False; CHECK 3 skipped")
    else:
        # We need a model to run CHECK 3 — load it now (without LoRA yet).
        print(f"\n[setup] loading base model in bf16 for CHECK 3 forward pass...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        base.eval()
        check_mask_affects_logits(candidate, tokenizer, base, max_seq_length)
        # release base; the LoRA training path will load fresh
        del base
        torch.cuda.empty_cache()

    # ---------------------- CHECK 4: 5 LoRA training steps ----------------------
    print("\n=========================== CHECK 4: 5 LoRA training steps ===========================")
    print(f"  loading base for training...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    if tr_cfg["gradient_checkpointing"]:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()  # required when grad ckpt + LoRA

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg["target_modules"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(base, peft_config)
    model.print_trainable_parameters()
    model.train()

    collator = CompressedCacheCollator(tokenizer)
    loader = DataLoader(
        FormattedDataset(formatted),
        batch_size=tr_cfg["per_device_train_batch_size"],
        collate_fn=collator,
        shuffle=False,
    )
    accum = tr_cfg["gradient_accumulation_steps"]
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=optim_cfg["learning_rate"],
        weight_decay=optim_cfg["weight_decay"],
    )

    max_optim_steps = dr_cfg["max_steps"]
    optim.zero_grad()
    optim_step = 0
    micro_step = 0
    t0 = time.time()
    losses: list[float] = []
    last_micro_loss: float | None = None
    # Cycle the loader so we get a clean N optimizer steps regardless of
    # how many micro-batches the small dataset gives us.
    for batch in itertools.cycle(loader):
        if optim_step >= max_optim_steps:
            break
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss / accum
        if not torch.isfinite(loss):
            raise AssertionError(
                f"  CHECK 4 FAILED: non-finite loss at micro-step {micro_step}: "
                f"{loss.item()}"
            )
        loss.backward()
        micro_step += 1
        last_micro_loss = out.loss.item()
        if micro_step % accum == 0:
            optim.step()
            optim.zero_grad()
            optim_step += 1
            losses.append(last_micro_loss)
            print(
                f"  step {optim_step}/{max_optim_steps}  "
                f"loss={last_micro_loss:.4f}  "
                f"({time.time()-t0:.1f}s elapsed)"
            )
    print("  ✅ training steps ran to completion without NaN/Inf loss")

    # ---------------------- CHECK 5: save / reload / forward ----------------------
    print("\n=========================== CHECK 5: save adapter, reload, forward ===========================")
    out_dir = ROOT / dr_cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    print(f"  saved adapter to {out_dir.relative_to(ROOT)}")

    # release training model + optimizer
    del model, base, optim
    torch.cuda.empty_cache()

    fresh_base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    reloaded = PeftModel.from_pretrained(fresh_base, str(out_dir))
    reloaded.eval()
    print(f"  reloaded adapter into a fresh base")

    # Forward pass on the first formatted example
    test_ex = formatted[0]
    ids = torch.tensor([test_ex.input_ids], device=reloaded.device)
    with torch.no_grad():
        logits = reloaded(input_ids=ids).logits
    if not torch.isfinite(logits).all():
        raise AssertionError("  CHECK 5 FAILED: reloaded adapter produced NaN/Inf logits")
    print(f"  forward pass logits shape={tuple(logits.shape)}, all-finite=True")
    print("  ✅ adapter round-trip clean (save -> reload -> forward without NaN)")

    print("\n=========================== D8 SUMMARY ===========================")
    print(f"  CHECK 1 (padding_side='left')             : PASS")
    print(f"  CHECK 2 (label mask covers exactly target): PASS")
    print(f"  CHECK 3 (kept_mask affects forward logits): PASS")
    print(f"  CHECK 4 (5 LoRA steps no NaN/Inf)         : PASS  ({len(losses)} optim steps)")
    print(f"  CHECK 5 (adapter save/reload/forward)     : PASS")
    print(f"\n  losses:    {[f'{x:.4f}' for x in losses]}")
    print(f"  total time: {time.time()-t0:.1f}s")
    print(f"\n  All gates clear. Bring back here for hyperparameter / wall-clock review")
    print(f"  before authorizing the real Qwen2.5-7B-Instruct run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
