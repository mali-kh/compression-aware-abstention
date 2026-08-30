"""Exp 1 — held-out test split of val_v1.

Carves val_v1.jsonl into:
  - val_v1_for_selection.jsonl  (~363 rows, used for original LR/epoch picking)
  - test_held_out.jsonl          (~100 rows, never seen by selection)

Disjoint by MuSiQue example id (the same id at different ratios stays on the
same side). Deterministic given --split-seed.

Sanity outputs printed: id-disjoint check, class balance per split,
context-length parity within each split between CONFIDENT and ABSTAIN.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-jsonl", default=str(ROOT / "artifacts" / "val_v1.jsonl"))
    ap.add_argument("--out-selection", default=str(ROOT / "artifacts" / "val_v1_for_selection.jsonl"))
    ap.add_argument("--out-test", default=str(ROOT / "artifacts" / "test_held_out.jsonl"))
    ap.add_argument("--split-seed", type=int, default=7)
    ap.add_argument("--target-test-rows", type=int, default=100)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.val_jsonl) if l.strip()]
    print(f"[setup] {len(rows)} rows from {args.val_jsonl}", flush=True)

    by_id: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_id[r["id"]].append(r)
    print(f"[setup] {len(by_id)} unique example ids", flush=True)

    ids = sorted(by_id.keys())
    rng = random.Random(args.split_seed)
    rng.shuffle(ids)

    test_ids: list[str] = []
    test_row_count = 0
    for ex_id in ids:
        test_ids.append(ex_id)
        test_row_count += len(by_id[ex_id])
        if test_row_count >= args.target_test_rows:
            break
    sel_ids = [i for i in ids if i not in set(test_ids)]
    test_rows = [r for ex_id in test_ids for r in by_id[ex_id]]
    sel_rows = [r for ex_id in sel_ids for r in by_id[ex_id]]

    # sanity (a): id disjoint
    overlap = set(test_ids) & set(sel_ids)
    assert not overlap, f"id overlap: {overlap}"
    print(f"[sanity] id-disjoint: OK ({len(test_ids)} test ids, {len(sel_ids)} sel ids)", flush=True)

    # sanity (b): row totals add up
    assert len(test_rows) + len(sel_rows) == len(rows)
    print(f"[sanity] row totals: test={len(test_rows)}  sel={len(sel_rows)}  "
          f"(target test={args.target_test_rows})", flush=True)

    # sanity (c): class balance (informational; we expect roughly proportional)
    def cb(xs):
        return dict(Counter(r["label_type"] for r in xs))
    print(f"[sanity] class balance:", flush=True)
    print(f"           whole val:   {cb(rows)}", flush=True)
    print(f"           selection:   {cb(sel_rows)}", flush=True)
    print(f"           test:        {cb(test_rows)}", flush=True)

    # sanity (d): per-split ctx-length parity between CONFIDENT and ABSTAIN
    def ctx_lens(xs, label):
        return [r["metadata"]["ctx_len"] for r in xs if r["label_type"] == label]
    for tag, xs in [("selection", sel_rows), ("test", test_rows)]:
        cl = ctx_lens(xs, "confident_answer")
        al = ctx_lens(xs, "abstain")
        if not cl or not al:
            print(f"[sanity] {tag} ctx parity: insufficient (n_conf={len(cl)} n_abst={len(al)})",
                  flush=True)
            continue
        mc = sum(cl) / len(cl)
        ma = sum(al) / len(al)
        delta_pct = abs(mc - ma) / min(mc, ma) * 100
        flag = "OK" if delta_pct < 10 else "WIDE"
        print(f"[sanity] {tag} ctx parity: |CONF-ABST|/min = {delta_pct:.1f}% ({flag})  "
              f"CONF mean ctx_len={mc:.0f} (n={len(cl)})  ABST mean={ma:.0f} (n={len(al)})",
              flush=True)

    # sanity (e): per-ratio breakdown (so we know mid-ratios still have meaningful n)
    print(f"[sanity] per-ratio row counts:", flush=True)
    print(f"           {'ratio':>6}  {'sel':>5}  {'test':>5}  {'sel_conf/abst':>15}  {'test_conf/abst':>15}",
          flush=True)
    for ratio in sorted(set(r["ratio"] for r in rows)):
        sel_at = [r for r in sel_rows if r["ratio"] == ratio]
        test_at = [r for r in test_rows if r["ratio"] == ratio]
        s_c = sum(1 for r in sel_at if r["label_type"] == "confident_answer")
        s_a = len(sel_at) - s_c
        t_c = sum(1 for r in test_at if r["label_type"] == "confident_answer")
        t_a = len(test_at) - t_c
        print(f"           {ratio:>6.2f}  {len(sel_at):>5}  {len(test_at):>5}  "
              f"{s_c}/{s_a:<13}  {t_c}/{t_a:<13}", flush=True)

    # Write
    Path(args.out_selection).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_selection, "w") as f:
        for r in sel_rows:
            f.write(json.dumps(r) + "\n")
    with open(args.out_test, "w") as f:
        for r in test_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n[done] wrote {args.out_selection} ({len(sel_rows)} rows)", flush=True)
    print(f"[done] wrote {args.out_test} ({len(test_rows)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
