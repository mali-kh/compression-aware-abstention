"""Block 3 / 3.5 — Compressed-cache decoding eval (kvpress + PEFT).

Two-phase flow:
  Phase 1: prefill context inside `with press(inner): ...`. For SnapKV /
           ExpAttn the per-layer forward_hook physically prunes the cache
           during prefill. For KVzip the post-yield code sets
           module.masked_key_indices on each layer (cache stays full size,
           but evicted positions are masked at decode time via the
           kvpress.attention_patch).
  Phase 2: generate the question continuation outside the press context,
           passing the cache (physically pruned for SnapKV/ExpAttn, full
           for KVzip) and explicit cache_position.

KVzip path used by Block 3.5: the literal `with press(model): generate(
full_prompt)` flow does NOT work for KVzip (compression happens after
yield exits, after generation has finished). The two-phase split below
DOES work because Phase 1 is a single ctx-only forward inside the
press; the post-yield code runs while we're still in `with`, sets
masked_key_indices, then Phase 2's generate decodes outside with the
masks active.

Output schema mirrors the `adapter` block of v1_eval.jsonl, plus an
`inference_mode: "compressed_cache"` marker.

Run from .venv-kvpress (transformers 5.x + kvpress + peft).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# KVzipPress.__call__ runs AutoTokenizer.from_pretrained on every entry
# (kvzip_press.py:94). With one press per eval example, ~463 HF API hits
# trips a 429 rate-limit. Force offline + memoize from_pretrained so the
# tokenizer load happens once per process.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

_orig_at_from_pretrained = AutoTokenizer.from_pretrained
_AT_CACHE: dict = {}
def _cached_from_pretrained(model_id, *args, **kwargs):
    key = str(model_id)
    if key not in _AT_CACHE:
        _AT_CACHE[key] = _orig_at_from_pretrained(model_id, *args, **kwargs)
    return _AT_CACHE[key]
AutoTokenizer.from_pretrained = _cached_from_pretrained

from kvpress import KVzipPress, SnapKVPress

PRESS_REGISTRY = {
    "snapkv": lambda cr: SnapKVPress(compression_ratio=cr, window_size=64, kernel_size=5),
    "kvzip": lambda cr: KVzipPress(compression_ratio=cr),
}

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_classifier():
    src = ROOT / "src" / "eval" / "honest_accuracy.py"
    spec = importlib.util.spec_from_file_location("_honest", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.classify_response, mod.looks_like_abstention


classify_response, looks_like_abstention = _load_classifier()

PROMPT_PREFIX = "Context: "
PROMPT_MIDDLE = "\n\nQuestion: {query}\n\nAnswer:"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


@torch.no_grad()
def evaluate_compressed_cache(
    inner_model,                 # Qwen2ForCausalLM (PEFT-unwrapped) — for kvpress
    adapted_model,               # PeftModel — for generate()
    tokenizer,
    records: list[dict],
    *,
    press_name: str = "snapkv",
    max_new_tokens: int = 64,
    progress_every: int = 25,
) -> list[dict]:
    if press_name not in PRESS_REGISTRY:
        raise ValueError(f"unknown press '{press_name}', expected one of {list(PRESS_REGISTRY)}")
    out: list[dict] = []
    t0 = time.time()
    for i, r in enumerate(records):
        ratio = float(r["ratio"])
        # kvpress compression_ratio = fraction EVICTED
        cr = max(0.0, min(1.0 - ratio, 0.999))

        prefix_ids = tokenizer(PROMPT_PREFIX, add_special_tokens=False).input_ids
        ctx_ids = tokenizer(r["context"], add_special_tokens=False).input_ids
        mid_ids = tokenizer(
            PROMPT_MIDDLE.format(query=r["query"]), add_special_tokens=False
        ).input_ids
        ctx_part = torch.tensor([prefix_ids + ctx_ids], device=inner_model.device)
        q_part = torch.tensor([mid_ids], device=inner_model.device)

        if ctx_part.size(1) + q_part.size(1) > 4096 - 96:
            out.append({
                "id": r["id"], "ratio": ratio,
                "gold_label": r["label_type"],
                "gold_answer": r["metadata"]["answer"],
                "response": "", "classification": "error",
                "model_abstained": False,
                "inference_mode": "compressed_cache",
                "error": f"prompt_len={ctx_part.size(1)+q_part.size(1)} exceeds budget",
            })
            continue

        try:
            # Phase 1: prefill ctx inside the press. SnapKV/ExpAttn physically
            # prune via per-layer forward_hook; KVzip leaves the cache full
            # and sets module.masked_key_indices in its post-yield code.
            press = PRESS_REGISTRY[press_name](cr)
            cache = DynamicCache()
            with press(inner_model):
                _ = inner_model(
                    input_ids=ctx_part, past_key_values=cache, use_cache=True
                )

            n_kept = cache.layers[0].keys.shape[2]

            # Phase 2: generate the question continuation
            cache_position = torch.arange(
                n_kept, n_kept + q_part.size(1), device=inner_model.device
            )
            gen = adapted_model.generate(
                input_ids=q_part,
                past_key_values=cache,
                cache_position=cache_position,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        except Exception as e:
            out.append({
                "id": r["id"], "ratio": ratio,
                "gold_label": r["label_type"],
                "gold_answer": r["metadata"]["answer"],
                "response": "", "classification": "error",
                "model_abstained": False,
                "inference_mode": "compressed_cache",
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            })
            if (i + 1) % progress_every == 0:
                print(f"  eval {i+1}/{len(records)}  ERROR  {type(e).__name__}", flush=True)
            continue

        # generate returns (input_ids + new tokens) when past_key_values supplied
        if gen.size(1) > q_part.size(1):
            new_ids = gen[0, q_part.size(1):]
        else:
            new_ids = gen[0]
        response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        cls = classify_response(
            response,
            gold_answer=r["metadata"]["answer"],
            answer_aliases=r["metadata"].get("answer_aliases", []),
            gold_label=r["label_type"],
        )
        out.append({
            "id": r["id"], "ratio": ratio,
            "gold_label": r["label_type"],
            "gold_answer": r["metadata"]["answer"],
            "response": response, "classification": cls,
            "model_abstained": looks_like_abstention(response),
            "inference_mode": "compressed_cache",
            "n_kept": int(n_kept),
        })

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(records) - i - 1) / rate
            print(f"  eval {i+1}/{len(records)}  elapsed={elapsed:.0f}s  "
                  f"rate={rate:.2f} ex/s  eta={eta:.0f}s", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--adapter-dir", default=None,
                    help="Path to a PEFT adapter dir. Omit (with --no-adapter) "
                         "to evaluate the unmodified base model.")
    ap.add_argument("--no-adapter", action="store_true",
                    help="Skip PEFT adapter loading and evaluate base only. "
                         "Useful for the base-cc comparator eval.")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--press", default="snapkv", choices=list(PRESS_REGISTRY),
                    help="Compressor for the cc inference path. snapkv = "
                         "physical eviction during prefill (Block 3). kvzip = "
                         "masked eviction set after prefill yield (Block 3.5).")
    args = ap.parse_args()

    rows = load_jsonl(Path(args.val_jsonl))
    n_in = len(rows)
    rows = [r for r in rows if r["label_type"] in ("abstain", "confident_answer")]
    print(f"[setup] {args.val_jsonl}: {n_in} rows -> {len(rows)} after DROP filter",
          flush=True)
    if args.no_adapter:
        if args.adapter_dir:
            print(f"[setup] --no-adapter set; ignoring --adapter-dir {args.adapter_dir}",
                  flush=True)
        print(f"[setup] BASE-ONLY mode  press={args.press}", flush=True)
    else:
        if not args.adapter_dir:
            ap.error("--adapter-dir is required unless --no-adapter is set")
        print(f"[setup] adapter={args.adapter_dir}  press={args.press}", flush=True)

    print(f"[setup] loading {args.base_model} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    base.eval()
    if args.no_adapter:
        # Base only. The eval harness needs both `inner_model` (for kvpress
        # to traverse .model.layers) and a `model_for_generate` whose
        # generate() method we call. For PEFT both come from the same
        # adapted module; for base-only they're the same (just `base`).
        adapted = base
        inner = base
    else:
        print(f"[setup] loading adapter ...", flush=True)
        adapted = PeftModel.from_pretrained(base, args.adapter_dir)
        adapted.eval()
        inner = adapted.model

    print(f"[run] generating {len(rows)} responses ({args.press} cc, two-phase) ...",
          flush=True)
    eval_rows = evaluate_compressed_cache(
        inner, adapted, tokenizer, rows, press_name=args.press,
    )

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n[done] wrote {len(eval_rows)} rows to {args.out_jsonl}", flush=True)

    n_err = sum(1 for r in eval_rows if r["classification"] == "error")
    if n_err:
        print(f"[warn] {n_err} rows had errors", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
