"""Render the D5 cross-compressor qualitative MD.

For each of the 10 wrong-but-specific D5 picks, emit a side-by-side comparison
of the v1-adapted model's response on fragments produced by each compressor
at ratio=0.30. KVzip column comes from the existing fragment_recoverability.jsonl.
ExpAttn / SnapKV / SnapKV-native columns are read from per-press JSONL files
produced by scripts/regen_d5_kvpress.py.

Output: artifacts/v1_d5_cross_compressor.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_d5_picks(qual_md: Path) -> list[str]:
    md = qual_md.read_text()
    ids = re.findall(r"`(2hop__\d+_\d+)`", md)
    return list(dict.fromkeys(ids))[:10]


def load_jsonl_by_id(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["id"]] = r
    return out


@torch.inference_mode()
def generate(model, tokenizer, fragment: str, question: str, max_new_tokens: int = 64) -> str:
    prompt = f"Context: {fragment}\n\nQuestion: {question}\n\nAnswer:"
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    gen = model.generate(
        ids, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(gen[0, ids.size(1):], skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter-dir", default=str(ROOT / "artifacts" / "v1_lora_best"))
    ap.add_argument("--qual-md", default=str(ROOT / "artifacts" / "v1_d5_qualitative.md"))
    ap.add_argument("--kvzip-jsonl", default=str(ROOT / "artifacts" / "fragment_recoverability.jsonl"))
    ap.add_argument("--expattn-jsonl",
                    default=str(ROOT / "artifacts" / "d5_expattn.jsonl"))
    ap.add_argument("--snapkv-jsonl",
                    default=str(ROOT / "artifacts" / "d5_snapkv.jsonl"))
    ap.add_argument("--snapkv-native-jsonl",
                    default=str(ROOT / "artifacts" / "d5_snapkv_native.jsonl"))
    ap.add_argument("--out-md", default=str(ROOT / "artifacts" / "v1_d5_cross_compressor.md"))
    args = ap.parse_args()

    picks = load_d5_picks(Path(args.qual_md))
    print(f"[setup] {len(picks)} D5 picks", flush=True)

    kvzip = load_jsonl_by_id(Path(args.kvzip_jsonl))
    expattn = load_jsonl_by_id(Path(args.expattn_jsonl))
    snapkv = load_jsonl_by_id(Path(args.snapkv_jsonl))
    snapkv_native = load_jsonl_by_id(Path(args.snapkv_native_jsonl))
    print(f"[setup] kvzip:{len(kvzip)}  expattn:{len(expattn)}  snapkv:{len(snapkv)}  snapkv_native:{len(snapkv_native)}", flush=True)

    print(f"[setup] loading {args.base_model} + adapter from {args.adapter_dir}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    base.eval()
    adapted = PeftModel.from_pretrained(base, args.adapter_dir)
    adapted.eval()

    md_lines: list[str] = [
        "# v1 — D5 cross-compressor qualitative comparison\n",
        "_For each of the 10 D5 wrong-but-specific examples, the v1-adapted model's "
        "response on fragments produced by each compressor at ratio=0.30. KVzip "
        "fragments come from the original D5 run; ExpAttn / SnapKV / SnapKV-native "
        "fragments come from `scripts/regen_d5_kvpress.py`._\n",
        "_All four columns use the same adapter (`artifacts/v1_lora_best/`) and the "
        "same generation config (greedy, max_new_tokens=64). The only thing that "
        "varies between columns is the kept-token fragment._\n",
        "---\n",
    ]
    for i, ex_id in enumerate(picks, 1):
        kv = kvzip.get(ex_id)
        ea = expattn.get(ex_id)
        sk = snapkv.get(ex_id)
        skn = snapkv_native.get(ex_id)

        # The common fields. KVzip's d5 jsonl uses different field names.
        question = (kv or ea or sk or skn).get("question", "?")
        answer = (kv or ea or sk or skn).get("answer", "?")

        md_lines.append(f"## {i}. `{ex_id}`")
        md_lines.append(f"**Question**: {question}")
        md_lines.append(f"**Gold answer**: `{answer}`\n")

        for tag, rec in [("KVzip", kv), ("ExpAttn", ea),
                         ("SnapKV (context-only)", sk),
                         ("SnapKV (native)", skn)]:
            if rec is None:
                md_lines.append(f"**{tag}**: _not available_\n")
                continue
            frag = rec["decoded_fragment"]
            n_kept = rec.get("n_kept", "?")
            ctx_len = rec.get("ctx_len", "?")
            resp = generate(adapted, tokenizer, frag, question)
            md_lines.append(f"**{tag}** (kept {n_kept}/{ctx_len} ctx tokens): `{resp}`")
        md_lines.append("\n---\n")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(md_lines))
    print(f"\n[done] wrote {args.out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
