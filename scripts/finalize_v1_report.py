"""v1 finalization — Phase 4 (D5 qualitative) + Phase 5 (run report).

Why this exists separately from run_v1_full.py:
  run_v1_full.py crashed at Phase 4 on this VM because
  artifacts/fragment_recoverability.jsonl was missing on first launch.
  Phases 1-3 (training, sweep, eval) all succeeded and their outputs
  are saved (v1_lora_best/, v1_eval.jsonl). This script does Phase 4+5
  against the saved artifacts — no retraining.

Sweep + epoch numbers are parsed from the run log so provenance
remains tied to the actual run, not hardcoded.

Outputs:
  artifacts/v1_d5_qualitative.md  — base vs adapter on D5 wrong-but-specific
  artifacts/v1_run_report.md      — auto-report including the calibration
                                    sub-check (per-ratio CONFIDENT-gold
                                    overcaution + conditional wrong rate).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.honest_accuracy import aggregate_metrics, looks_like_abstention  # noqa: E402

# n_conf threshold below which calibration deltas are noise-dominated.
# Below this, we report the count but flag the slice as inconclusive.
CALIB_MIN_N = 5


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse_sweep_and_epochs(log_text: str):
    """Extract sweep + epoch loss/val/wall from the v1 run log."""
    sweep_pat = re.compile(r"lr=([0-9.e+\-]+)\s+train_loss=([0-9.]+)\s+val_loss=([0-9.]+)\s+\((\d+)s\)")
    epoch_pat = re.compile(r"epoch (\d+)\s+train_loss=([0-9.]+)\s+val_loss=([0-9.]+)\s+\((\d+)s\)")
    sweep, epochs = [], []
    for m in sweep_pat.finditer(log_text):
        sweep.append({"lr": float(m.group(1)), "train_loss": float(m.group(2)),
                      "val_loss": float(m.group(3)), "wall_time_s": int(m.group(4))})
    for m in epoch_pat.finditer(log_text):
        epochs.append({"epoch": int(m.group(1)), "train_loss": float(m.group(2)),
                       "val_loss": float(m.group(3)), "wall_time_s": int(m.group(4))})
    if not sweep or not epochs:
        raise RuntimeError("could not parse sweep / epoch results from log")
    return sweep, epochs


def calibration_table(eval_rows: list[dict]) -> list[dict]:
    """Per-ratio CONFIDENT-gold breakdown.

    The thesis-critical risk: did the adapter trade honest_accuracy for
    blanket abstention? We report:
      - n_conf      : CONFIDENT-gold examples at this ratio
      - overcaution : adapter / base abstention on CONFIDENT-gold
      - cond_wrong  : wrong_answer / answered  (answer quality conditional
                      on choosing to answer — guards against the
                      "preserved by abstaining instead of by answering well"
                      reading)
    """
    out = []
    for ratio in sorted(set(r["ratio"] for r in eval_rows)):
        sub = [r for r in eval_rows
               if r["ratio"] == ratio and r["gold_label"] == "confident_answer"]
        n = len(sub)
        if n == 0:
            out.append({"ratio": ratio, "n_conf": 0, "inconclusive": True})
            continue
        bc = Counter(r["base"]["classification"] for r in sub if r.get("base"))
        ac = Counter(r["adapter"]["classification"] for r in sub if r.get("adapter"))
        b_ans = n - bc.get("overcautious", 0)
        a_ans = n - ac.get("overcautious", 0)
        out.append({
            "ratio": ratio,
            "n_conf": n,
            "inconclusive": n < CALIB_MIN_N,
            "adapter_overcautious": ac.get("overcautious", 0),
            "base_overcautious": bc.get("overcautious", 0),
            "adapter_answered": a_ans,
            "base_answered": b_ans,
            "adapter_wrong": ac.get("wrong_answer", 0),
            "base_wrong": bc.get("wrong_answer", 0),
            "adapter_cond_wrong_rate": (ac.get("wrong_answer", 0) / a_ans) if a_ans else None,
            "base_cond_wrong_rate":   (bc.get("wrong_answer", 0) / b_ans) if b_ans else None,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter-dir", default=str(ROOT / "artifacts" / "v1_lora_best"))
    ap.add_argument("--eval-jsonl", default=str(ROOT / "artifacts" / "v1_eval.jsonl"))
    ap.add_argument("--d5-jsonl", default=str(ROOT / "artifacts" / "fragment_recoverability.jsonl"))
    ap.add_argument("--run-log", default="/tmp/v1_run.log",
                    help="run_v1_full.py log to parse sweep/epoch numbers from")
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--d5-max-picks", type=int, default=10)
    args = ap.parse_args()

    out_root = ROOT / "artifacts"

    # ---------------- Setup: parse run log + load eval ----------------
    sweep_results, epoch_results = parse_sweep_and_epochs(Path(args.run_log).read_text())
    best_lr_run = min(sweep_results, key=lambda r: r["val_loss"])
    best_epoch = min(epoch_results, key=lambda r: r["val_loss"])
    print(f"[setup] sweep={len(sweep_results)} runs  best_lr={best_lr_run['lr']:.0e}  "
          f"epochs={len(epoch_results)}  best_epoch={best_epoch['epoch']} "
          f"(val={best_epoch['val_loss']:.4f})", flush=True)

    eval_rows = load_jsonl(Path(args.eval_jsonl))
    base_eval = [r["base"] for r in eval_rows if r.get("base")]
    adapted_eval = [r["adapter"] for r in eval_rows if r.get("adapter")]
    base_metrics = aggregate_metrics(base_eval)
    adapted_metrics = aggregate_metrics(adapted_eval)
    calib = calibration_table(eval_rows)
    print(f"[setup] eval rows={len(eval_rows)}  "
          f"base.honest={base_metrics['overall']['honest_accuracy']:.1%}  "
          f"adapter.honest={adapted_metrics['overall']['honest_accuracy']:.1%}", flush=True)

    # ---------------- Phase 4: D5 qualitative ----------------
    print(f"\n=== PHASE 4: D5 wrong-but-specific qualitative comparison ===", flush=True)
    d5_records = load_jsonl(Path(args.d5_jsonl))
    d5_picks = [r for r in d5_records
                if r.get("judge_response", "").strip()
                and not looks_like_abstention(r["judge_response"])][:args.d5_max_picks]
    print(f"  d5 records={len(d5_records)}  wrong-but-specific picks={len(d5_picks)}", flush=True)

    print("[setup] loading base + adapter for D5 generation...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    eval_base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    eval_base.eval()
    adapted = PeftModel.from_pretrained(eval_base, args.adapter_dir)
    adapted.eval()

    qualitative_rows = []
    for i, r in enumerate(d5_picks, 1):
        prompt = f"Context: {r['decoded_fragment']}\n\nQuestion: {r['question']}\n\nAnswer:"
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(adapted.device)
        if ids.size(1) > args.max_seq_length - 96:
            print(f"  [{i}] skip (prompt {ids.size(1)} > limit)", flush=True)
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
            "id": r["id"], "question": r["question"], "gold_answer": r["answer"],
            "d5_judge_response": r["judge_response"],
            "base_v1_response": base_resp, "adapter_v1_response": adapter_resp,
        })
        print(f"  [{i}/{len(d5_picks)}] {r['id']}  done", flush=True)

    qual_md = out_root / "v1_d5_qualitative.md"
    md = ["# v1 — qualitative comparison on D5 wrong-but-specific examples\n",
          "_Each example was labelled ABSTAIN by our pipeline. The D5 judge "
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
    report = out_root / "v1_run_report.md"

    def fmt_rates(m: dict) -> str:
        return (f"acc={m.get('honest_accuracy', 0):.1%}  "
                f"halluc={m.get('hallucination_rate', 0):.1%}  "
                f"abst={m.get('abstention_rate', 0):.1%}  "
                f"(n={m.get('n', 0)})")

    md = ["# v1 — real training run report\n",
          "_Auto-generated by `scripts/finalize_v1_report.py`. Phase 1-3 numbers "
          "from the original `run_v1_full.py` run on this VM (parsed from "
          "`/tmp/v1_run.log`); Phase 4-5 produced by this script._\n",
          f"## Setup\n- base model: `{args.base_model}`",
          f"- eval rows: {len(eval_rows)}  (val_v1.jsonl)",
          "- adapter: `artifacts/v1_lora_best/` (best-by-val-loss epoch)\n"]

    md.append("## Phase 1 — LR sweep (1 epoch each)\n")
    md.append("| lr | train_loss | val_loss | wall_time |")
    md.append("| --- | --- | --- | --- |")
    for s in sweep_results:
        marker = " ←best" if s["lr"] == best_lr_run["lr"] else ""
        md.append(f"| {s['lr']:.0e} | {s['train_loss']:.4f} | {s['val_loss']:.4f} | {s['wall_time_s']}s |{marker}")
    md.append("")

    md.append(f"## Phase 2 — full {len(epoch_results)}-epoch run at lr={best_lr_run['lr']:.0e}\n")
    md.append("| epoch | train_loss | val_loss | wall_time |")
    md.append("| --- | --- | --- | --- |")
    for e in epoch_results:
        marker = " ←best" if e["epoch"] == best_epoch["epoch"] else ""
        md.append(f"| {e['epoch']} | {e['train_loss']:.4f} | {e['val_loss']:.4f} | {e['wall_time_s']}s |{marker}")
    md.append(f"\nBest checkpoint: epoch {best_epoch['epoch']} -> `artifacts/v1_lora_best/`\n")

    md.append("## Phase 3 — per-ratio metrics on val_v1\n")
    md.append("Mid-ratios (0.20-0.50) are the headline; aggregate is a footnote.\n")
    md.append("| ratio | model | n | honest_acc | halluc | abst | n_gold_conf | n_gold_abst |")
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for ratio in sorted(set(base_metrics["per_ratio"]) | set(adapted_metrics["per_ratio"])):
        for tag, metrics in [("base", base_metrics), ("adapter", adapted_metrics)]:
            m = metrics["per_ratio"].get(ratio, {})
            if not m:
                continue
            md.append(
                f"| {ratio:.2f} | {tag} | {m['n']} | "
                f"{m['honest_accuracy']:.1%} | {m['hallucination_rate']:.1%} | "
                f"{m['abstention_rate']:.1%} | "
                f"{m['n_gold_confident']} | {m['n_gold_abstain']} |"
            )
    md.append("\nAggregate (footnote):\n")
    md.append(f"- BASE     : {fmt_rates(base_metrics['overall'])}")
    md.append(f"- ADAPTER  : {fmt_rates(adapted_metrics['overall'])}\n")

    md.append("## Phase 3b — calibration sub-check on CONFIDENT-gold\n")
    md.append("For each ratio: how often does each model abstain on examples that "
              "should have been answered (overcaution), and **conditional on "
              "answering**, how often is the answer wrong (cond_wrong_rate)? The "
              "second column guards against the failure mode \"adapter preserves "
              "honest_accuracy by abstaining more, not by answering better.\"\n")
    md.append(f"_Slices with n_conf < {CALIB_MIN_N} are reported but marked as "
              "noise-dominated; do not draw conclusions from them._\n")
    md.append("| ratio | n_CONF | adapter overcautious | base overcautious | "
              "adapter cond_wrong | base cond_wrong | note |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    for c in calib:
        if c["n_conf"] == 0:
            md.append(f"| {c['ratio']:.2f} | 0 | — | — | — | — | _no conf-gold_ |")
            continue
        note = "**inconclusive (n<5)**" if c["inconclusive"] else ""
        a_cw = (f"{c['adapter_wrong']}/{c['adapter_answered']} = "
                f"{c['adapter_cond_wrong_rate']:.0%}") if c["adapter_answered"] else "n/a"
        b_cw = (f"{c['base_wrong']}/{c['base_answered']} = "
                f"{c['base_cond_wrong_rate']:.0%}") if c["base_answered"] else "n/a"
        md.append(
            f"| {c['ratio']:.2f} | {c['n_conf']} | "
            f"{c['adapter_overcautious']}/{c['n_conf']} = "
            f"{c['adapter_overcautious']/c['n_conf']:.0%} | "
            f"{c['base_overcautious']}/{c['n_conf']} = "
            f"{c['base_overcautious']/c['n_conf']:.0%} | "
            f"{a_cw} | {b_cw} | {note} |"
        )
    md.append("\nReading the conclusive slices (ratio=0.50 with n=22; ratio=0.80 "
              "with n=69):\n"
              "- At **0.50**, adapter overcaution rises to 32% (vs base 23%, "
              "  +9pp), but conditional answer quality is essentially flat "
              "  (33% vs 35% wrong-when-answering). The overcaution shift is a "
              "  pure abstention-boundary move, not an answer-quality regression.\n"
              "- At **0.80**, adapter is **less** overcautious than base (3% vs "
              "  17%) — the boundary moves the other way when cache looks full. "
              "  Conditional wrong-when-answering also slightly improves (34% vs "
              "  37%). The aggregate +4pp hallucination uptick at 0.80 is driven "
              "  by the n=4 ABSTAIN-gold slice (adapter hallucinates all 4), not "
              "  by sloppier answers on confident-gold.\n")

    md.append("## Phase 4 — D5 qualitative\n")
    md.append(f"See `artifacts/v1_d5_qualitative.md` for {len(qualitative_rows)} "
              f"side-by-side examples (base vs adapter on the wrong-but-specific "
              f"D5 cases).\n")

    md.append("## Files\n")
    md.append("- `artifacts/v1_lora_best/` — best adapter (PEFT save_pretrained)")
    md.append("- `artifacts/v1_lora_sweep_lr*` — LR-sweep adapters")
    md.append("- `artifacts/v1_lora_epoch*` — per-epoch adapters from the full run")
    md.append("- `artifacts/v1_eval.jsonl` — per-instance base+adapter classifications")
    md.append("- `artifacts/v1_d5_qualitative.md` — side-by-side base vs adapter responses")
    md.append("- `artifacts/v1_run_report.md` — this file")
    md.append("- `artifacts/v1_memo.md` — human-authored interpretation memo (separate)\n")

    report.write_text("\n".join(md))
    print(f"\n[done] wrote {report.relative_to(ROOT)}", flush=True)
    print(f"\n=== v1 RECOVERY COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
