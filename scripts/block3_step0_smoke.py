"""Block 3 Step 0 smoke test — does PEFT + kvpress compose cleanly?

Picks ONE example from val_v1.jsonl at r=0.50, loads Qwen2.5-7B-Instruct +
v1 adapter, runs compressed-cache decoding via kvpress.KVzipPress at
compression_ratio=0.5 (= retention 0.5). Prints the prompt-style adapter
output (from existing v1_eval.jsonl) alongside the compressed-cache output
for sanity check.

Run from .venv-kvpress (transformers 5.x + kvpress + peft).
"""

import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress import KVzipPress

ROOT = Path(__file__).resolve().parents[1]

ADAPTER_DIR = ROOT / "artifacts" / "v1_lora_best"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
VAL_JSONL = ROOT / "artifacts" / "val_v1.jsonl"
EXISTING_EVAL = ROOT / "artifacts" / "v1_eval.jsonl"

PROMPT_PREFIX = "Context: "
PROMPT_MIDDLE = "\n\nQuestion: {query}\n\nAnswer:"


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    val = load_jsonl(VAL_JSONL)
    existing = load_jsonl(EXISTING_EVAL)
    existing_idx = {(r["id"], r["ratio"]): r for r in existing}

    target_record = next(r for r in val if abs(r["ratio"] - 0.5) < 1e-9
                         and r["label_type"] == "abstain")
    eid = target_record["id"]
    ratio = target_record["ratio"]
    print(f"[setup] target id={eid}, ratio={ratio}, gold={target_record['label_type']}, "
          f"answer={target_record['metadata']['answer']!r}", flush=True)
    print(f"        ctx_len={target_record['metadata']['ctx_len']}, "
          f"n_kept={target_record['metadata']['n_kept']}", flush=True)

    existing_row = existing_idx[(eid, ratio)]
    ps_response = existing_row["adapter"]["response"]
    ps_class = existing_row["adapter"]["classification"]
    ps_abst = existing_row["adapter"]["model_abstained"]
    print(f"\n[prompt-style adapter (from existing v1_eval.jsonl)]\n"
          f"  classification={ps_class}  abstained={ps_abst}\n  response: {ps_response!r}",
          flush=True)

    print(f"\n[setup] loading {BASE_MODEL} ...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    base.eval()
    print(f"        base loaded in {time.time()-t0:.0f}s", flush=True)

    print(f"[setup] loading PEFT adapter from {ADAPTER_DIR} ...", flush=True)
    t0 = time.time()
    adapted = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    adapted.eval()
    print(f"        adapter loaded in {time.time()-t0:.0f}s", flush=True)
    # Print which adapter modules are active
    adapter_param_count = sum(p.numel() for n, p in adapted.named_parameters() if "lora_" in n)
    print(f"        LoRA param count: {adapter_param_count:,}", flush=True)

    # Build the FULL uncompressed prompt (no truncation)
    full_prompt = PROMPT_PREFIX + target_record["context"] + \
        PROMPT_MIDDLE.format(query=target_record["query"])
    full_ids = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).input_ids
    full_ids = full_ids.to(adapted.device)
    print(f"\n[compressed-cache] full prompt length={full_ids.size(1)} tokens", flush=True)

    # Run compressed-cache decoding
    # KVzipPress: compression_ratio = fraction EVICTED. ratio=0.5 retention -> compression_ratio=0.5
    press = KVzipPress(compression_ratio=0.5)
    print(f"[compressed-cache] running prefill+generate inside KVzipPress(compression_ratio=0.5) ...", flush=True)
    t0 = time.time()
    # kvpress traverses `model.model.layers`; on a PEFT-wrapped model that
    # would land on Qwen2ForCausalLM (no .layers attr). Pass the unwrapped
    # Qwen2ForCausalLM (adapted.model) so kvpress hooks the correct
    # attention modules — those are the same module objects PEFT routes
    # forward through, so the LoRA adapter remains active during generation.
    with torch.no_grad():
        with press(adapted.model):
            out = adapted.generate(
                full_ids,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    elapsed = time.time() - t0
    cc_response = tokenizer.decode(out[0, full_ids.size(1):], skip_special_tokens=True).strip()
    print(f"\n[compressed-cache adapter response] ({elapsed:.1f}s)\n  {cc_response!r}",
          flush=True)

    # Comparison summary
    print("\n=== summary ===")
    print(f"  prompt-style:    {ps_response[:120]!r}")
    print(f"  compressed-cache: {cc_response[:120]!r}")
    print(f"  both look coherent: "
          f"{'YES' if (ps_response.strip() and cc_response.strip()) else 'NO — investigate'}")


if __name__ == "__main__":
    main()
