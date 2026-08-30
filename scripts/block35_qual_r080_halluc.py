"""Block 3.5 follow-up — qualitative inspection of r=0.80 hallucinations.

42 of 69 CONFIDENT-gold examples at r=0.80 produced wrong specific answers
under NEW (cc-trained) under KVzip cc inference. Bucket each into:

  A. gold answer string is PRESENT in kept tokens, model picked something
     ELSE that is also PRESENT in kept tokens. Distractor confusion.
  B. gold answer string is PRESENT in kept tokens, model fabricated
     something not appearing in ANY token of the full ctx. Pure fabrication.
  C. gold answer string was EVICTED, model picked a substitute that IS
     present in kept tokens. Reasonable failure mode given cache state.
  D. gold answer string was EVICTED, model fabricated something not
     appearing in ANY token of the full ctx. Worst case.

Notes / approximations:
  * The kept_mask in val_v1.jsonl was generated on BASE Qwen + KVzip.
    NEW's actual KVzip mask under LoRA differs slightly (LoRA changes the
    Q/K and so changes the attention scores KVzip uses for scoring). For
    a 42-example qualitative pass this is a fine proxy.
  * "Present" = case-insensitive substring match on the decoded kept-text
    (or full-context text). For short responses (<= 8 words) we match the
    full response; for longer ones we extract the first sentence-like
    chunk before period / newline / colon and match THAT.
  * answer_aliases is empty in val_v1.jsonl, so we only check the gold
    answer string itself.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Use the cached AutoTokenizer + offline mode trick from the eval script
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from transformers import AutoTokenizer  # noqa: E402

VAL_FILE = ROOT / "artifacts" / "val_v1.jsonl"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# Default eval file for Block 3.5 NEW (cc-trained, KVzip-only). Override
# via --eval-file (e.g. v1_eval_mixture_cc_kvzip.jsonl for Block 3.6).
DEFAULT_EVAL = ROOT / "artifacts" / "v1_eval_compressed_trained_kvzip.jsonl"


def load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def first_sentence_chunk(s: str) -> str:
    """Return the first 'salient' chunk of a response — useful for
    matching short generated answers against the context. We split on the
    first sentence-ending punctuation OR newline. If the response is
    short (<=8 words) we just lowercase-strip it."""
    s = s.strip()
    if len(s.split()) <= 8:
        return s
    # split on common boundaries; pick first non-trivial chunk
    parts = re.split(r"[.!?\n:]+", s, maxsplit=1)
    return parts[0].strip()


def substring_present(needle: str, haystack: str) -> bool:
    """Case-insensitive substring match. Empty needles return False."""
    if not needle.strip():
        return False
    return needle.lower() in haystack.lower()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-file", default=str(DEFAULT_EVAL))
    args = ap.parse_args()

    print(f"[setup] eval file = {args.eval_file}", flush=True)
    print(f"[setup] loading tokenizer ({BASE_MODEL}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    eval_rows = load_jsonl(Path(args.eval_file))
    val_records = {(r["id"], r["ratio"]): r for r in load_jsonl(VAL_FILE)}

    hal = [
        r for r in eval_rows
        if abs(r["ratio"] - 0.80) < 1e-9
        and r["gold_label"] == "confident_answer"
        and r["classification"] == "wrong_answer"
    ]
    print(f"[setup] {len(hal)} wrong_answer rows on C-gold at r=0.80", flush=True)

    bucket_A: list[dict] = []
    bucket_B: list[dict] = []
    bucket_C: list[dict] = []
    bucket_D: list[dict] = []
    skipped: list[dict] = []

    for h in hal:
        key = (h["id"], h["ratio"])
        if key not in val_records:
            skipped.append({"id": h["id"], "reason": "missing val record"})
            continue
        v = val_records[key]
        full_text = v["context"]
        ctx_ids = tok(full_text, add_special_tokens=False).input_ids
        if len(ctx_ids) != len(v["kept_mask"]):
            skipped.append({"id": h["id"], "reason": "tokenizer length mismatch"})
            continue
        kept_ids = [t for t, k in zip(ctx_ids, v["kept_mask"]) if k]
        kept_text = tok.decode(kept_ids, skip_special_tokens=True)

        gold = h["gold_answer"]
        resp = h["response"]
        resp_chunk = first_sentence_chunk(resp)

        gold_in_kept = substring_present(gold, kept_text)
        gold_in_full = substring_present(gold, full_text)
        resp_in_kept = substring_present(resp_chunk, kept_text)
        resp_in_full = substring_present(resp_chunk, full_text)

        record = {
            "id": h["id"],
            "gold": gold,
            "response": resp[:160],
            "gold_in_kept": gold_in_kept,
            "gold_in_full": gold_in_full,
            "resp_chunk": resp_chunk[:120],
            "resp_in_kept": resp_in_kept,
            "resp_in_full": resp_in_full,
        }

        # Bucket logic:
        #   A: gold survived eviction (in kept), model used something else from kept (resp in kept).
        #   B: gold survived eviction, model invented (resp not in full ctx).
        #   C: gold was evicted, model used something from kept.
        #   D: gold was evicted, model invented.
        # Edge case: gold not present in full ctx at all (label noise / paraphrased gold).
        # Treat that as "gold evicted" for bucketing — the model couldn't have used it.
        gold_was_kept = gold_in_kept
        if gold_was_kept:
            if resp_in_kept:
                bucket_A.append(record)
            else:
                bucket_B.append(record)
        else:  # evicted (or never present)
            if resp_in_kept:
                bucket_C.append(record)
            else:
                bucket_D.append(record)

    total = len(bucket_A) + len(bucket_B) + len(bucket_C) + len(bucket_D)
    print(f"\n=== Bucket counts (total = {total}; spec target = 42) ===")
    print(f"  A. gold IN kept, response IN kept (distractor):     {len(bucket_A):3d}")
    print(f"  B. gold IN kept, response NOT in full (fabricate):  {len(bucket_B):3d}")
    print(f"  C. gold NOT in kept, response IN kept (substitute): {len(bucket_C):3d}")
    print(f"  D. gold NOT in kept, response NOT in full (worst):  {len(bucket_D):3d}")
    if skipped:
        print(f"  -- skipped (tokenizer/missing): {len(skipped)}")
        for s in skipped[:3]:
            print(f"     {s}")

    def show_bucket(name, items, n_show=3):
        if not items:
            return
        print(f"\n--- Bucket {name} ({len(items)} examples; showing up to {n_show}) ---")
        for r in items[:n_show]:
            print(f"  id: {r['id']}")
            print(f"  gold:     {r['gold']!r}")
            print(f"  response: {r['response']!r}")
            print(f"  resp_chunk used for matching: {r['resp_chunk']!r}")
            print(f"  flags: gold_in_kept={r['gold_in_kept']}  "
                  f"gold_in_full={r['gold_in_full']}  "
                  f"resp_in_kept={r['resp_in_kept']}  "
                  f"resp_in_full={r['resp_in_full']}")
            print()

    show_bucket("A — distractor confusion", bucket_A)
    show_bucket("B — fabrication w/ gold present", bucket_B)
    show_bucket("C — substitute under eviction", bucket_C)
    show_bucket("D — fabrication w/ gold evicted", bucket_D)

    # Summary line: how often was the gold actually still in the kept tokens
    n_gold_kept = len(bucket_A) + len(bucket_B)
    print(f"\n=== Sanity: of {total} hallucinations at r=0.80, gold was "
          f"present in kept tokens in {n_gold_kept}/{total} = "
          f"{n_gold_kept/total:.0%} of cases. (At retention 0.80 we would "
          f"naively expect ~80% of relevant tokens to survive.) ===")


if __name__ == "__main__":
    main()
