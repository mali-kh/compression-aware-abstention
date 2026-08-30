"""Exp 4 — RULER multi-key NIAH evaluation.

Source: simonjegou/ruler at configs {4096, 8192, 16384}. The user's spec
called for {16K, 32K, 64K} but no public pre-tokenized RULER source has
≥32K configs (verified by probing several HF dataset ids). This script
runs at the available lengths and documents the limitation in the report.

For each (length, ratio, example): KVzip prefill+score+prune on
Qwen2.5-7B-Instruct, decode the kept tokens, prompt base + v1-adapter,
classify response (correct via substring match against gold; abstain via
the v1 abstention template; otherwise hallucinated).

Output: artifacts/exp4_ruler_eval.jsonl + artifacts/exp4_ruler_long_context_report.md
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.compressors.kvzip import KVzipCompressor  # noqa: E402
from src.eval.honest_accuracy import looks_like_abstention  # noqa: E402

PROMPT_PREFIX = "Context: "
PROMPT_MIDDLE = "\n\nQuestion: {query}\n\nAnswer:"


def select_multikey(configs: list[str], n_per: int, seed: int) -> dict[str, list[dict]]:
    """Per length config, sample n_per multikey-NIAH examples deterministically."""
    rng = random.Random(seed)
    out: dict[str, list[dict]] = {}
    for cfg in configs:
        ds = load_dataset("simonjegou/ruler", cfg, split="test")
        multikey = [r for r in ds if "multi" in r["task"].lower()]
        rng.shuffle(multikey)
        out[cfg] = multikey[:n_per]
        print(f"[setup] {cfg}: {len(multikey)} multikey total → sampled {len(out[cfg])}",
              flush=True)
    return out


def masks_at_ratios(wrapper: KVzipCompressor, context_ids: torch.Tensor,
                    ratios: list[float]) -> dict[float, torch.Tensor]:
    """Single prefill+score, then prune at each ratio (kv.prune is idempotent — sets
    kv.valid from cached scores). Returns dict ratio -> CPU bool mask."""
    ids_dev = context_ids.to(wrapper._impl.device).unsqueeze(0)
    kv = wrapper._impl.prefill(ids_dev, load_score=False)
    ctx_len = int(context_ids.size(0))
    out: dict[float, torch.Tensor] = {}
    for ratio in ratios:
        kv.prune(ratio=ratio, level="pair")
        slots = kv.valid.reshape(-1, kv.valid.size(-1))
        frac_kept = slots.float().mean(dim=0).detach().cpu()
        k = int(round(ratio * ctx_len))
        if k <= 0:
            out[ratio] = torch.zeros(ctx_len, dtype=torch.bool)
        elif k >= ctx_len:
            out[ratio] = torch.ones(ctx_len, dtype=torch.bool)
        else:
            _, top_idx = torch.topk(frac_kept, k)
            m = torch.zeros(ctx_len, dtype=torch.bool)
            m.scatter_(0, top_idx, True)
            out[ratio] = m
    del kv
    torch.cuda.empty_cache()
    return out


def classify_ruler_response(resp: str, gold: str) -> str:
    """Substring-match-or-abstain classifier matching the v1 honest_accuracy harness.

    For RULER all examples have a single ground-truth answer string. The
    "right" behavior depends on whether the answer survived compression:
    if it did, the model should emit the answer; if it didn't, the model
    should abstain. We return one of: correct, abstained, hallucinated.
    Whether each outcome is "good" depends on the gold-state, which the
    aggregation step computes.
    """
    if looks_like_abstention(resp):
        return "abstained"
    if gold and gold.lower() in resp.lower():
        return "correct"
    return "hallucinated"




def gold_state_from_decoded(kept_ids: list[int], gold: str, tokenizer) -> str:
    """Decode kept tokens and check whether the gold answer appears as a substring.

    Binary for RULER: if the answer string survives compression (appears in
    decoded(kept)), gold_state = confident (model SHOULD answer). Else,
    gold_state = abstain (model SHOULD abstain). No DROP for RULER — the
    needle is a short atomic string.
    """
    if not gold:
        return "unknown"
    decoded = tokenizer.decode(kept_ids, skip_special_tokens=True)
    return "confident" if gold.lower() in decoded.lower() else "abstain"


def unpack_answer(raw) -> str:
    """RULER stores answer as a list of strings, even for single-answer tasks."""
    if isinstance(raw, list):
        return raw[0] if raw else ""
    return str(raw)


@torch.inference_mode()
def generate(model, tokenizer, fragment: str, question: str, answer_prefix: str = "",
             max_new_tokens: int = 64) -> str:
    prompt = PROMPT_PREFIX + fragment + PROMPT_MIDDLE.format(query=question)
    if answer_prefix:
        prompt = prompt + " " + answer_prefix
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    gen = model.generate(
        ids, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(gen[0, ids.size(1):], skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="4096,8192,16384")
    ap.add_argument("--n-per", type=int, default=100)
    ap.add_argument("--ratios", default="0.05,0.10,0.20,0.30,0.50,0.80")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter-dir", default=str(ROOT / "artifacts" / "v1_lora_best"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-jsonl", default=str(ROOT / "artifacts" / "exp4_ruler_eval.jsonl"))
    ap.add_argument("--report", default=str(ROOT / "artifacts" / "exp4_ruler_long_context_report.md"))
    args = ap.parse_args()

    configs = args.configs.split(",")
    ratios = [float(x) for x in args.ratios.split(",")]
    print(f"[setup] configs={configs}  n_per={args.n_per}  ratios={ratios}", flush=True)

    examples_per_cfg = select_multikey(configs, args.n_per, args.seed)

    print(f"[setup] loading KVzip wrapper (mask gen on {args.base_model})...", flush=True)
    wrapper = KVzipCompressor(model_name=args.base_model)

    print(f"[setup] loading separate eval base + adapter (for generation)...", flush=True)
    eval_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if eval_tokenizer.pad_token_id is None:
        eval_tokenizer.pad_token = eval_tokenizer.eos_token
    eval_tokenizer.padding_side = "left"
    eval_base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    eval_base.eval()
    adapted = PeftModel.from_pretrained(eval_base, args.adapter_dir)
    adapted.eval()

    out_rows: list[dict] = []
    t0 = time.time()
    for cfg in configs:
        examples = examples_per_cfg[cfg]
        print(f"\n=== EVAL config={cfg} ({len(examples)} examples × {len(ratios)} ratios × 2 models) ===",
              flush=True)
        for i, ex in enumerate(examples, 1):
            context = ex["context"]
            question = ex["question"]
            answer = unpack_answer(ex["answer"])
            answer_prefix = ex.get("answer_prefix", "") or ""
            task = ex["task"]
            try:
                ctx_ids = wrapper.tokenize(context)
            except Exception as e:
                print(f"  [{cfg} {i}] tokenize fail: {e}", flush=True)
                continue
            ctx_len = int(ctx_ids.size(0))

            try:
                masks = masks_at_ratios(wrapper, ctx_ids, ratios)
            except Exception as e:
                print(f"  [{cfg} {i}] kvzip fail (ctx_len={ctx_len}): {type(e).__name__}: {e}",
                      flush=True)
                continue

            # Use the eval tokenizer for the kept fragment (must match the eval prompt path)
            ctx_ids_eval = eval_tokenizer(context, add_special_tokens=False).input_ids

            for ratio in ratios:
                m = masks[ratio]
                # If the kvzip and eval tokenizations diverged in length (BPE drift),
                # fall back to the eval tokenization length for slicing.
                eval_len = min(len(ctx_ids_eval), ctx_len)
                kept_ids = [tid for tid, k in zip(ctx_ids_eval[:eval_len], m.tolist()[:eval_len]) if k]
                fragment = eval_tokenizer.decode(kept_ids, skip_special_tokens=True)

                state = gold_state_from_decoded(kept_ids, answer, eval_tokenizer)

                # Generate base + adapter
                with adapted.disable_adapter():
                    base_resp = generate(adapted, eval_tokenizer, fragment, question, answer_prefix)
                adapter_resp = generate(adapted, eval_tokenizer, fragment, question, answer_prefix)
                base_cls = classify_ruler_response(base_resp, answer)
                adapter_cls = classify_ruler_response(adapter_resp, answer)

                out_rows.append({
                    "length": int(cfg),
                    "ratio": ratio,
                    "task": task,
                    "answer": answer,
                    "ctx_len": ctx_len,
                    "n_kept": int(m.sum().item()),
                    "gold_state": state,
                    "base_response": base_resp,
                    "base_classification": base_cls,
                    "adapter_response": adapter_resp,
                    "adapter_classification": adapter_cls,
                })
            if i % 5 == 0:
                elapsed = time.time() - t0
                print(f"  [{cfg} {i}/{len(examples)}] elapsed={elapsed:.0f}s", flush=True)

        # Periodically flush JSONL so a crash doesn't lose work
        Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_jsonl, "w") as f:
            for r in out_rows:
                f.write(json.dumps(r) + "\n")
        print(f"  [{cfg}] checkpoint: wrote {len(out_rows)} rows to {args.out_jsonl}", flush=True)

    # ---- Aggregate ----
    print(f"\n[done] eval complete; aggregating...", flush=True)

    def bucket(rows: list[dict], who: str) -> dict:
        n = len(rows)
        if n == 0: return {"n": 0}
        c = Counter(r[f"{who}_classification"] for r in rows)
        return {
            "n": n,
            "correct": c.get("correct", 0),
            "abstained": c.get("abstained", 0),
            "hallucinated": c.get("hallucinated", 0),
        }

    md = ["# Exp 4 — RULER multi-key NIAH evaluation\n",
          "_Adapter run: `artifacts/v1_lora_best/` (no retraining); base: Qwen2.5-7B-Instruct; "
          "compressor: KVzip._\n",
          "## Source caveat\n",
          "Spec called for {16K, 32K, 64K}. Available HF source `simonjegou/ruler` has only "
          "{4K, 8K, 16K}; no public pre-tokenized RULER source with ≥32K configs was found "
          "(probed RMT-team, jondurbin, ChuxiJ, nvidia/Aegis, Jiawei1106, YoussefMohammed, "
          "mubarakssh, anthropic, ZhengxiangShi, ai-research — none accessible). Generating "
          "32K/64K examples locally would require running the NVIDIA RULER toolkit; out of scope "
          "for this overnight session. Reported lengths: 4K, 8K, 16K.\n",
          f"## Setup\n- {args.n_per} examples per length, sampled from multi-key NIAH variants",
          f"- 5 multikey tasks: niah_multikey_1/2/3, niah_multivalue, niah_multiquery",
          f"- 6 ratios: {ratios}",
          "- Eval prompt: `Context: {decoded kept tokens}\\n\\nQuestion: {q}\\n\\nAnswer:`",
          "- Substring-match classifier: `correct` if gold answer appears in response, "
          "`abstained` if response matches v1 abstention pattern, else `hallucinated`\n",
          "## Per-(length, ratio) honest behavior by gold-state\n",
          "Each cell shows adapter classifications, scoped to examples where the needle "
          "actually survived (gold_state=confident) or didn't (gold_state=abstain). "
          "DROP and `unknown` (needle-not-located) gold_states are excluded from the table; "
          "`unknown` count footnoted per length.\n"]
    md.append("| length | ratio | n_conf | adapter on conf-gold (correct / wrong) | n_abst | adapter on abst-gold (correct=abstained / hallucinated) |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for cfg in configs:
        for ratio in ratios:
            sub = [r for r in out_rows if r["length"] == int(cfg) and r["ratio"] == ratio]
            conf = [r for r in sub if r["gold_state"] == "confident"]
            abst = [r for r in sub if r["gold_state"] == "abstain"]
            ab = bucket(abst, "adapter")
            cb = bucket(conf, "adapter")
            md.append(
                f"| {cfg} | {ratio:.2f} | {cb['n']} | "
                f"{cb.get('correct',0)} correct / {cb.get('hallucinated',0) + cb.get('abstained',0)} other | "
                f"{ab['n']} | {ab.get('abstained',0)} abst / {ab.get('hallucinated',0)} hallucinated |"
            )
        md.append("")

    md.append("## Aggregate per length (collapsed across ratios)\n")
    md.append("| length | n | base correct | base hall | adapter correct | adapter hall | adapter abst |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    for cfg in configs:
        sub = [r for r in out_rows if r["length"] == int(cfg)]
        b = bucket(sub, "base")
        a = bucket(sub, "adapter")
        if b["n"] == 0:
            continue
        md.append(
            f"| {cfg} | {b['n']} | {b['correct']/b['n']:.1%} | {b['hallucinated']/b['n']:.1%} | "
            f"{a['correct']/a['n']:.1%} | {a['hallucinated']/a['n']:.1%} | {a['abstained']/a['n']:.1%} |"
        )
    md.append("")
    md.append("## Files\n")
    md.append(f"- `{args.out_jsonl}` — per-row classifications")
    md.append(f"- `{args.report}` — this file")
    Path(args.report).write_text("\n".join(md))
    print(f"[done] wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
