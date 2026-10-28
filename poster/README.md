# GroundLM 2026 poster

`poster.pdf` is the print file: one A0 portrait page (841 x 1189 mm), vector
text and charts. `poster_preview.png` is a low-resolution preview.

Every number on the poster comes from the paper and is recomputed by
`scripts/verify_paper_claims.py` (base-model compressed-cache per-ratio values
are computed from `artifacts/eval/v1_eval_base_compressed_kvzip.jsonl`).

To edit and rebuild:

```bash
python poster/build_poster.py      # charts + template.html -> poster.html
node poster/render_poster.mjs      # poster.html -> poster.pdf, poster_preview.png
```

Text lives in `template.html`; chart data and drawing code live in
`build_poster.py`. The render script prints a layout check: each column's last
card should end at the main area's bottom edge, with no overflow lines above it.
