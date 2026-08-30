"""Run v1 adapter eval against a cross-compressor val set.

Loads the saved `v1_lora_best/` adapter on top of base Qwen2.5-7B-Instruct
and evaluates both base and adapter on a val_v1_*.jsonl file produced by
`scripts/regen_val_kvpress.py`. Output: `artifacts/v1_eval_<press>.jsonl`
with the same paired schema as `v1_eval.jsonl`.

DROP rows in the input val are filtered (no defined gold behavior).

Reuses `evaluate_on_val` and the prompt format from `run_v1_full.py` to
keep the eval pipeline byte-identical to the v1 run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_run_v1_module():
    """Import scripts/run_v1_full.py as a module without executing main()."""
    src = ROOT / "scripts" / "run_v1_full.py"
    spec = importlib.util.spec_from_file_location("_run_v1_full", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # __main__ guarded; loads symbols only
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-jsonl", required=True,
                    help="e.g. artifacts/val_v1_expattn.jsonl")
    ap.add_argument("--out-jsonl", required=True,
                    help="e.g. artifacts/v1_eval_expattn.jsonl")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter-dir", default=str(ROOT / "artifacts" / "v1_lora_best"))
    ap.add_argument("--max-seq-length", type=int, default=4096)
    args = ap.parse_args()

    # Reuse v1 eval helpers (build_eval_input_ids, evaluate_on_val).
    rv1 = _load_run_v1_module()

    rows = [json.loads(l) for l in open(args.val_jsonl) if l.strip()]
    n_in = len(rows)
    rows = [r for r in rows if r["label_type"] in ("abstain", "confident_answer")]
    n_dropped = n_in - len(rows)
    print(f"[setup] loaded {n_in} rows from {args.val_jsonl}; "
          f"filtered {n_dropped} DROP rows; eval on {len(rows)}", flush=True)

    by_ratio: dict[float, int] = defaultdict(int)
    for r in rows:
        by_ratio[r["ratio"]] += 1
    for ratio in sorted(by_ratio):
        print(f"  ratio={ratio:.2f}  n={by_ratio[ratio]}", flush=True)

    print(f"[setup] loading {args.base_model} + adapter from {args.adapter_dir}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    eval_base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    eval_base.eval()

    print(f"\n=== eval BASE on {Path(args.val_jsonl).name} ===", flush=True)
    t0 = time.time()
    base_eval = rv1.evaluate_on_val(eval_base, tokenizer, rows)
    print(f"  base eval: {len(base_eval)}/{len(rows)} in {time.time()-t0:.0f}s", flush=True)

    adapted = PeftModel.from_pretrained(eval_base, args.adapter_dir)
    adapted.eval()
    print(f"\n=== eval ADAPTER on {Path(args.val_jsonl).name} ===", flush=True)
    t0 = time.time()
    adapted_eval = rv1.evaluate_on_val(adapted, tokenizer, rows)
    print(f"  adapter eval: {len(adapted_eval)}/{len(rows)} in {time.time()-t0:.0f}s", flush=True)

    # interleave by id+ratio so each row has both base and adapter
    by_id: dict[tuple, dict] = defaultdict(dict)
    for r in base_eval:
        by_id[(r["id"], r["ratio"])]["base"] = r
    for r in adapted_eval:
        by_id[(r["id"], r["ratio"])]["adapter"] = r

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for key, both in by_id.items():
            row = {
                "id": key[0], "ratio": key[1],
                "gold_label": (both.get("base") or both.get("adapter"))["gold_label"],
                "gold_answer": (both.get("base") or both.get("adapter"))["gold_answer"],
                "base": both.get("base"), "adapter": both.get("adapter"),
            }
            f.write(json.dumps(row) + "\n")
    print(f"\n[done] wrote {args.out_jsonl}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
