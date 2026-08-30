"""D6 — Tokenizer alignment check.

We've validated tight-span localization with Qwen3-0.6B's tokenizer.
Training will use Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct, whose
fast-tokenizer offset_mapping behavior could differ. If span localization
silently breaks for the production tokenizer, every downstream label is
wrong.

Note: meta-llama/Llama-3.1-8B-Instruct is gated on HF. We use
NousResearch/Meta-Llama-3.1-8B-Instruct as the public mirror — same
tokenizer (vocab=128000), no auth needed.

For 20 baseline-clean examples (all sub-answers located via Qwen3-0.6B):
  (a) does the candidate tokenizer localize the same number of tight
      spans (no extra paragraph-fallbacks)?
  (b) does decoding each tight-span's token IDs round-trip a string that
      contains the gold sub-answer?
  (c) is the span token length within +/-20% of the baseline tokenizer's
      length on the same logical span?

Any rate below 95% on any tokenizer => blocker.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.musique import normalize_musique_example  # noqa: E402

OUT_JSONL = ROOT / "artifacts" / "tokenizer_alignment.jsonl"
OUT_MD = ROOT / "artifacts" / "tokenizer_alignment_report.md"

# Production-candidate tokenizers we want to validate.
# Llama-3.1's official repo is gated; we use the NousResearch mirror,
# which ships the same tokenizer (vocab=128000) but no auth wall.
TOKENIZER_IDS = {
    "qwen3_0_6b_baseline": "Qwen/Qwen3-0.6B",
    "qwen2_5_7b_instruct": "Qwen/Qwen2.5-7B-Instruct",
    "llama3_1_8b_instruct": "NousResearch/Meta-Llama-3.1-8B-Instruct",
}


def collect_baseline_examples(ds, tokenizer, n: int, max_tokens: int):
    """Pick examples for which the baseline tokenizer hits tight on every hop."""
    out = []
    scanned = 0
    for row in ds:
        if not row.get("answerable", True):
            continue
        scanned += 1
        ex = normalize_musique_example(row, tokenizer, span_granularity="tight")
        if not ex.supporting_spans:
            continue
        if ex.n_paragraph_fallbacks > 0:
            continue  # we want clean baselines only
        if any(s.end > max_tokens for s in ex.supporting_spans):
            continue
        if ex.context_ids.size(0) > max_tokens:
            continue
        out.append((row, ex))
        if len(out) >= n:
            break
    return out, scanned


def per_span_subanswer_lookup(row: dict) -> dict[int, str]:
    qd = row.get("question_decomposition") or []
    return {
        step["paragraph_support_idx"]: (step.get("answer") or "").strip()
        for step in qd
        if step.get("paragraph_support_idx") is not None and step.get("answer")
    }


def supporting_para_idxs(row: dict) -> list[int]:
    return [p["idx"] for p in row["paragraphs"] if p["is_supporting"]]


def check_one(other_tok, raw_row: dict, baseline_ex):
    """Run the three rates for ONE example under ONE candidate tokenizer."""
    other_ex = normalize_musique_example(raw_row, other_tok, span_granularity="tight")

    # (a) same count of tight spans (no extra paragraph fallbacks under the new tokenizer)
    a_pass = (
        len(other_ex.supporting_spans) == len(baseline_ex.supporting_spans)
        and other_ex.n_tight_spans_located == baseline_ex.n_tight_spans_located
        and other_ex.n_paragraph_fallbacks == baseline_ex.n_paragraph_fallbacks
    )

    # (b) and (c) need per-span comparison; only proceed if (a) passes
    b_per_span: list[bool] = []
    c_per_span: list[bool] = []
    span_diagnostics: list[dict] = []

    if a_pass:
        sub_lookup = per_span_subanswer_lookup(raw_row)
        para_idxs = supporting_para_idxs(raw_row)
        for i, (o_span, b_span) in enumerate(
            zip(other_ex.supporting_spans, baseline_ex.supporting_spans)
        ):
            sub_answer = sub_lookup.get(para_idxs[i], "")
            o_ids = other_ex.context_ids[o_span.start : o_span.end]
            o_decoded = other_tok.decode(o_ids, skip_special_tokens=True)

            if o_span.source == "musique_qd_subanswer":
                b_check = sub_answer.lower() in o_decoded.lower() if sub_answer else False
            else:
                # paragraph-fallback span — looser check: the sub-answer should
                # still appear in the rendered paragraph; don't require it
                # given (a) ensures we don't have new fallbacks
                b_check = True
            b_per_span.append(b_check)

            o_len = o_span.end - o_span.start
            b_len = b_span.end - b_span.start
            # +/-20% on long spans, but allow up to +/-2 tokens absolute on
            # short spans where 20% rounds to 0-1 token (sub-answers like
            # "Cologne" are typically 1-3 tokens; one-token diff between
            # tokenizers is normal BPE variance, not misalignment).
            tolerance_pct = 0.20
            allowed = max(2, int(round(b_len * tolerance_pct)))
            c_check = abs(o_len - b_len) <= allowed
            c_per_span.append(c_check)

            span_diagnostics.append(
                {
                    "span_idx": i,
                    "para_idx": para_idxs[i],
                    "sub_answer": sub_answer,
                    "baseline_tok_len": b_len,
                    "other_tok_len": o_len,
                    "other_decoded": o_decoded,
                    "b_check": b_check,
                    "c_check": c_check,
                    "source": o_span.source,
                }
            )

    return {
        "a_count_match": a_pass,
        "b_decode_match": all(b_per_span) if b_per_span else (a_pass),
        "c_len_within_pct": all(c_per_span) if c_per_span else (a_pass),
        "spans": span_diagnostics,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()

    print("[setup] loading tokenizers...", flush=True)
    tokenizers = {
        name: AutoTokenizer.from_pretrained(tid) for name, tid in TOKENIZER_IDS.items()
    }
    baseline_name = "qwen3_0_6b_baseline"
    baseline_tok = tokenizers[baseline_name]
    others = [(n, tokenizers[n]) for n in TOKENIZER_IDS if n != baseline_name]

    for name, tok in tokenizers.items():
        print(f"  {name}: vocab={tok.vocab_size} fast={tok.is_fast}")

    print("[setup] loading Musique validation split...", flush=True)
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")

    print(f"[setup] selecting {args.n} baseline-clean examples...", flush=True)
    baseline_data, scanned = collect_baseline_examples(
        ds, baseline_tok, args.n, args.max_tokens
    )
    print(f"  selected {len(baseline_data)} (scanned {scanned} rows)", flush=True)

    summary: dict[str, dict[str, float]] = {}
    rows: list[dict] = []
    for name, tok in others:
        print(f"\n[{name}]", flush=True)
        a_pass = b_pass = c_pass = 0
        for raw_row, baseline_ex in baseline_data:
            chk = check_one(tok, raw_row, baseline_ex)
            a_pass += int(chk["a_count_match"])
            b_pass += int(chk["b_decode_match"])
            c_pass += int(chk["c_len_within_pct"])
            rows.append(
                {
                    "tokenizer": name,
                    "id": baseline_ex.id,
                    "checks": {
                        "a_count_match": chk["a_count_match"],
                        "b_decode_match": chk["b_decode_match"],
                        "c_len_within_pct": chk["c_len_within_pct"],
                    },
                    "spans": chk["spans"],
                }
            )
        n = len(baseline_data)
        rates = {
            "a_count_match": a_pass / n,
            "b_decode_match": b_pass / n,
            "c_len_within_pct": c_pass / n,
        }
        summary[name] = rates
        print(f"  (a) span count match              : {rates['a_count_match']:.1%} ({a_pass}/{n})")
        print(f"  (b) decoded span contains sub-ans : {rates['b_decode_match']:.1%} ({b_pass}/{n})")
        print(f"  (c) tok len within +/-20%/+/-2tok : {rates['c_len_within_pct']:.1%} ({c_pass}/{n})")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    md: list[str] = []
    md.append("# Tokenizer alignment check (D6)\n")
    md.append("_Auto-generated by `scripts/check_tokenizer_alignment.py`._\n")
    md.append("## Setup\n")
    md.append(f"- baseline tokenizer: `{TOKENIZER_IDS[baseline_name]}`")
    md.append(f"- examples: {len(baseline_data)} Musique val rows where baseline localizes "
              "all sub-answers cleanly (n_paragraph_fallbacks=0)")
    md.append("- (a) tight-span count parity (same #tight, same #fallback)")
    md.append("- (b) decoded span token IDs string-match the gold sub-answer")
    md.append("- (c) per-span token length within +/-20% (or +/-2 absolute, whichever is larger)")
    md.append("")
    md.append("## Results\n")
    md.append("| tokenizer | vocab | (a) count | (b) decode | (c) tok-len | blocker (<95%)? |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for name, rates in summary.items():
        tok_id = TOKENIZER_IDS[name]
        vocab = tokenizers[name].vocab_size
        is_blocker = any(r < 0.95 for r in rates.values())
        md.append(
            f"| `{tok_id}` | {vocab} | "
            f"{rates['a_count_match']:.1%} | "
            f"{rates['b_decode_match']:.1%} | "
            f"{rates['c_len_within_pct']:.1%} | "
            f"{'**YES — blocker**' if is_blocker else 'no'} |"
        )
    md.append("")
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_JSONL.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")

    # Verdict
    blockers = [
        name for name, r in summary.items() if any(v < 0.95 for v in r.values())
    ]
    if blockers:
        print(f"\nVERDICT: blocker — rates <95% for: {blockers}")
        print("Surface options before scaling D7.")
        return 1
    else:
        print("\nVERDICT: all rates >= 95% — proceed to D7")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
