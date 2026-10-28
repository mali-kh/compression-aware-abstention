"""Build the GroundLM 2026 poster.

Generates the SVG charts from the paper's published numbers (all of which are
recomputed by scripts/verify_paper_claims.py), drops them into template.html,
and writes poster.html. Render to PDF/PNG with render_poster.mjs.

    python poster/build_poster.py && node poster/render_poster.mjs
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Palette (validated with the dataviz validator; blues are an ordinal ramp)
# ---------------------------------------------------------------------------
SURFACE = "#ffffff"
INK = "#14161a"
INK2 = "#4a4c52"
MUTED = "#7b7d84"
GRID = "#e4e4e0"
BASE = "#8f8e88"      # base model (neutral, de-emphasized)
PROMPT = "#eb6834"    # prompt-only baseline (P2)
V1 = "#86b6ef"        # v1, KVzip-only adapter
MIX = "#2a78d6"       # multi-compressor mixture adapter
MIXCC = "#104281"     # mixture trained on compressed-cache states
WRONG = "#d93f3e"
ABST = "#c9c8c1"

FONT = "Inter, 'Liberation Sans', sans-serif"
RATIOS = ["0.05", "0.10", "0.20", "0.30", "0.50", "0.80"]

# ---------------------------------------------------------------------------
# Paper numbers
# ---------------------------------------------------------------------------
# Table 1: KVzip, prompt-style, honest accuracy per ratio (adapter = 3-seed mean)
T1_BASE = [0.15, 0.27, 0.30, 0.35, 0.44, 0.51]
T1_ADP = [1.00, 0.99, 0.92, 0.79, 0.66, 0.60]
T1_ANSWERABLE = ["0%", "1%", "2%", "5%", "46%", "95%"]  # CONFIDENT share per ratio

# Table 2: honest accuracy at r = 0.50 (P2, Fragment/base, v1, Mix)
T2 = {
    "KVzip": (0.54, 0.44, 0.65, 0.69),
    "Exp. Attention": (0.60, 0.38, 0.67, 0.71),
    "SnapKV (ctx)": (0.64, 0.33, 0.46, 0.64),
    "SnapKV (native)": (0.54, 0.38, 0.41, 0.51),
}

# Table 3: aggregate gain in honest accuracy over base, v1 vs mixture
T3 = {
    "KVzip": (0.53, 0.55),
    "Exp. Attention": (0.49, 0.57),
    "SnapKV (ctx)": (0.21, 0.53),
    "SnapKV (native)": (0.20, 0.43),
}

# Table 4: compressed-cache decoding, honest accuracy per ratio
T4 = [
    ("Base model", BASE, [0.070, 0.137, 0.256, 0.354, 0.375, 0.055], 0.186),
    ("v1 (prompt-style trained)", V1, [0.560, 0.495, 0.415, 0.323, 0.271, 0.233], 0.406),
    ("Mixture (prompt-style trained)", MIX, [1.000, 0.958, 0.829, 0.600, 0.354, 0.315], 0.730),
    ("Mixture (cache-trained)", MIXCC, [1.000, 0.989, 0.963, 0.923, 0.500, 0.315], 0.821),
]


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=28, fill=INK2, anchor="middle", weight=400, extra=""):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}" font-family="{FONT}" {extra}>{esc(s)}</text>'
    )


def col_path(x, y_top, w, y_base, r=8):
    """Column with rounded data-end (top) and square baseline."""
    h = y_base - y_top
    r = min(r, h / 2, w / 2)
    if h <= 0:
        return ""
    return (
        f"M{x:.1f},{y_base:.1f} L{x:.1f},{y_top + r:.1f} Q{x:.1f},{y_top:.1f} {x + r:.1f},{y_top:.1f} "
        f"L{x + w - r:.1f},{y_top:.1f} Q{x + w:.1f},{y_top:.1f} {x + w:.1f},{y_top + r:.1f} "
        f"L{x + w:.1f},{y_base:.1f} Z"
    )


# ---------------------------------------------------------------------------
# Chart 1: honest accuracy vs retention ratio (base vs adapter)
# ---------------------------------------------------------------------------
def chart_table1(W=940, H=560):
    L, R, T, B = 175, 175, 46, 140
    pw, ph = W - L - R, H - T - B
    xs = [L + pw * i / (len(RATIOS) - 1) for i in range(len(RATIOS))]

    def y(v):
        return T + ph * (1 - v)

    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Honest accuracy by retention ratio">']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{L}" x2="{L + pw}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}" stroke-width="2"/>')
        out.append(text(L - 18, y(v) + 10, f"{v:.2f}".rstrip("0").rstrip(".") if v not in (0, 1) else f"{v:.0f}", 26, MUTED, "end"))
    for x, lab, ans in zip(xs, RATIOS, T1_ANSWERABLE):
        out.append(text(x, T + ph + 44, lab, 28, INK2))
        out.append(text(x, T + ph + 98, ans, 26, MUTED))
    out.append(text(0, T + ph + 98, "answerable:", 23, MUTED, "start"))
    out.append(text(L + pw / 2, T + ph + 136, "KV retention ratio r  (fraction of context kept)", 26, INK2))

    for vals, color, name in ((T1_BASE, BASE, "Base"), (T1_ADP, MIX, "Ours")):
        pts = " ".join(f"{x:.1f},{y(v):.1f}" for x, v in zip(xs, vals))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="6" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, v in zip(xs, vals):
            out.append(f'<circle cx="{x:.1f}" cy="{y(v):.1f}" r="11" fill="{color}" stroke="{SURFACE}" stroke-width="4"/>')
        # end label with series name and final value
        out.append(text(xs[-1] + 24, y(vals[-1]) + 10, f"{name} {vals[-1]:.2f}", 28, INK, "start", 600))
    # start labels
    out.append(text(xs[0] + 20, y(T1_ADP[0]) - 22, "1.00", 26, INK2, "start", 600))
    out.append(text(xs[0] + 20, y(T1_BASE[0]) + 40, "0.15", 26, INK2, "start", 600))
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Chart 2: grouped columns at r = 0.50 across compressors
# ---------------------------------------------------------------------------
def chart_table2(W=940, H=520):
    L, R, T, B = 80, 10, 40, 110
    pw, ph = W - L - R, H - T - B
    groups = list(T2.keys())
    gw = pw / len(groups)
    bw, gap = 40, 6
    order = [(1, BASE), (0, PROMPT), (2, V1), (3, MIX)]  # Fragment, P2, v1, Mix

    def y(v):
        return T + ph * (1 - v / 0.8)

    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Honest accuracy at r=0.50 across compressors">']
    for v in (0, 0.2, 0.4, 0.6, 0.8):
        out.append(f'<line x1="{L}" x2="{L + pw}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}" stroke-width="2"/>')
        out.append(text(L - 16, y(v) + 9, f"{v:.1f}" if v else "0", 26, MUTED, "end"))
    for gi, g in enumerate(groups):
        total = len(order) * bw + (len(order) - 1) * gap
        x0 = L + gi * gw + (gw - total) / 2
        for k, (idx, color) in enumerate(order):
            v = T2[g][idx]
            x = x0 + k * (bw + gap)
            out.append(f'<path d="{col_path(x, y(v), bw, y(0))}" fill="{color}"/>')
            out.append(text(x + bw / 2, y(v) - 12, f"{v:.2f}"[1:], 23, INK if idx == 3 else INK2, "middle", 700 if idx == 3 else 500))
        name = g.split(" (")
        out.append(text(L + gi * gw + gw / 2, T + ph + 42, name[0], 27, INK2, "middle", 600))
        if len(name) > 1:
            out.append(text(L + gi * gw + gw / 2, T + ph + 76, "(" + name[1], 24, MUTED))
    out.append(f'<line x1="{L}" x2="{L + pw}" y1="{y(0):.1f}" y2="{y(0):.1f}" stroke="{MUTED}" stroke-width="2"/>')
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Chart 3: dumbbell, v1 -> mixture aggregate gain
# ---------------------------------------------------------------------------
def chart_table3(W=940, H=320):
    L, R, T, B = 250, 70, 20, 90
    pw, ph = W - L - R, H - T - B
    rows = list(T3.keys())
    rh = ph / len(rows)

    def x(v):
        return L + pw * v / 0.6

    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Gain over base model, v1 versus mixture">']
    for v in (0, 0.2, 0.4, 0.6):
        out.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{T}" y2="{T + ph}" stroke="{GRID}" stroke-width="2"/>')
        out.append(text(x(v), T + ph + 38, f"+{v:.1f}" if v else "0", 25, MUTED))
    out.append(text(L + pw / 2, T + ph + 82, "gain in honest accuracy over base (all ratios)", 24, MUTED))
    for i, rname in enumerate(rows):
        cy = T + rh * i + rh / 2
        a, b = T3[rname]
        out.append(text(L - 24, cy + 9, rname, 26, INK2, "end", 600))
        out.append(f'<line x1="{x(a):.1f}" x2="{x(b):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" stroke="{GRID}" stroke-width="10" stroke-linecap="round"/>')
        out.append(f'<circle cx="{x(a):.1f}" cy="{cy:.1f}" r="13" fill="{V1}" stroke="{SURFACE}" stroke-width="4"/>')
        out.append(f'<circle cx="{x(b):.1f}" cy="{cy:.1f}" r="13" fill="{MIX}" stroke="{SURFACE}" stroke-width="4"/>')
        if b - a > 0.07:
            out.append(text(x(a) - 22, cy + 9, f"+{a:.2f}", 24, INK2, "end", 500))
            out.append(text(x(b) + 22, cy + 9, f"+{b:.2f}", 24, INK, "start", 700))
        else:
            out.append(text(x(a) - 22, cy + 9, f"+{a:.2f} → +{b:.2f}", 24, INK, "end", 700))
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Chart 4: compressed-cache decoding per ratio
# ---------------------------------------------------------------------------
def chart_table4(W=940, H=520):
    L, R, T, B = 96, 40, 30, 110
    pw, ph = W - L - R, H - T - B
    xs = [L + pw * i / (len(RATIOS) - 1) for i in range(len(RATIOS))]

    def y(v):
        return T + ph * (1 - v)

    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Compressed-cache honest accuracy by retention ratio">']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{L}" x2="{L + pw}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}" stroke-width="2"/>')
        out.append(text(L - 18, y(v) + 10, f"{v:.2f}".rstrip("0").rstrip(".") if v not in (0, 1) else f"{v:.0f}", 26, MUTED, "end"))
    for x, lab in zip(xs, RATIOS):
        out.append(text(x, T + ph + 44, lab, 28, INK2))
    out.append(text(L + pw / 2, T + ph + 94, "KV retention ratio r", 26, INK2))
    # callout at r = 0.30: cache-trained vs prompt-trained mixture
    i = 3
    out.append(f'<line x1="{xs[i]:.1f}" x2="{xs[i]:.1f}" y1="{y(T4[3][2][i]) - 2:.1f}" y2="{y(T4[2][2][i]) + 2:.1f}" stroke="{INK2}" stroke-width="3" stroke-dasharray="2 8" stroke-linecap="round"/>')
    out.append(text(xs[2] + 10, y(0.75), "cache training:", 25, INK, "end", 700))
    out.append(text(xs[2] + 10, y(0.75) + 32, "+0.32 at r = 0.30", 25, INK, "end", 700))
    for name, color, vals, _agg in T4:
        pts = " ".join(f"{x:.1f},{y(v):.1f}" for x, v in zip(xs, vals))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="6" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, v in zip(xs, vals):
            out.append(f'<circle cx="{x:.1f}" cy="{y(v):.1f}" r="10" fill="{color}" stroke="{SURFACE}" stroke-width="4"/>')
    out.append("</svg>")
    return "\n".join(out)


def legend_table4():
    items = []
    for name, color, _vals, agg in T4:
        items.append(
            f'<div class="lg-row"><span class="lg-line" style="background:{color}"></span>'
            f'<span class="lg-name">{esc(name)}</span><span class="lg-val">{agg:.2f}</span></div>'
        )
    return "\n".join(items)


# ---------------------------------------------------------------------------
# Chart 5: same-length opposite masks (controlled deletion, 10C)
# ---------------------------------------------------------------------------
def chart_masks(W=940, H=290):
    L, R, T = 320, 10, 10
    pw = W - L - R
    bh, gapr = 64, 46
    rows = [
        ("Mask A: top 50%", "evidence kept", (14, 5, 3)),
        ("Mask B: bottom 50%", "evidence evicted", (0, 22, 0)),
    ]
    cols = [(MIX, "#ffffff", "correct"), (ABST, INK, "refused"), (WRONG, "#ffffff", "wrong")]
    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Same-length opposite masks">']
    for i, (lab, sub, counts) in enumerate(rows):
        yy = T + i * (bh + gapr)
        out.append(text(L - 24, yy + 30, lab, 24, INK, "end", 700))
        out.append(text(L - 24, yy + 60, sub, 23, MUTED, "end"))
        x = L
        nz = [(c, col) for c, col in zip(counts, cols) if c]
        for k, (c, (fill, ink, name)) in enumerate(nz):
            w = pw * c / 22 - (3 if k < len(nz) - 1 else 0)
            out.append(f'<rect x="{x:.1f}" y="{yy}" width="{w:.1f}" height="{bh}" rx="6" fill="{fill}"/>')
            label = f"{c} {name}" if w > 170 else str(c)
            out.append(text(x + w / 2, yy + bh / 2 + 9, label, 25, ink, "middle", 700))
            x += w + 3
    kx = L
    for fill, _ink, name in cols:
        out.append(f'<rect x="{kx}" y="{T + 2 * bh + gapr + 22}" width="26" height="26" rx="5" fill="{fill}"/>')
        out.append(text(kx + 36, T + 2 * bh + gapr + 44, name, 24, INK2, "start"))
        kx += 190
    out.append(text(0, H - 8, "Mixture adapter · 22 answerable questions at r = 0.50 · equal mask length", 23, MUTED, "start"))
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Method diagram: token survival -> label
# ---------------------------------------------------------------------------
def method_diagram(W=940, H=430):
    n = 16
    tw, tg = 29, 6
    span = (7, 8, 9)
    rows = [
        ("Evidence survives", "every span keeps ≥ 80% of its tokens",
         [1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1], "CONFIDENT", "answer: “Mike Medavoy”", MIX),
        ("Evidence evicted", "some span keeps < 30% of its tokens",
         [1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1], "ABSTAIN", "refuse: “I cannot determine…”", INK),
    ]
    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Labeling from compression masks">']
    # legend
    lx = 0
    out.append(f'<rect x="{lx}" y="6" width="26" height="34" rx="5" fill="#3a3d44"/>')
    out.append(text(lx + 38, 33, "kept", 23, INK2, "start"))
    out.append(f'<rect x="{lx + 110}" y="6" width="26" height="34" rx="5" fill="none" stroke="#b9bab5" stroke-width="2.5" stroke-dasharray="5 4"/>')
    out.append(text(lx + 148, 33, "evicted", 23, INK2, "start"))
    out.append(f'<rect x="{lx + 260}" y="6" width="26" height="34" rx="5" fill="#eda100"/>')
    out.append(text(lx + 298, 33, "answer-evidence span (median 3 tokens)", 23, INK2, "start"))
    for ri, (title, sub, mask, lab, act, col) in enumerate(rows):
        y0 = 92 + ri * 172
        out.append(text(0, y0, title, 28, INK, "start", 700))
        out.append(text(0, y0 + 34, sub, 23, MUTED, "start"))
        ty = y0 + 58
        for i in range(n):
            x = i * (tw + tg)
            kept = mask[i]
            is_span = i in span
            if kept and is_span:
                out.append(f'<rect x="{x}" y="{ty}" width="{tw}" height="46" rx="6" fill="#eda100"/>')
            elif kept:
                out.append(f'<rect x="{x}" y="{ty}" width="{tw}" height="46" rx="6" fill="#3a3d44"/>')
            elif is_span:
                out.append(f'<rect x="{x + 1.5}" y="{ty + 1.5}" width="{tw - 3}" height="43" rx="6" fill="#fdf3dc" stroke="#eda100" stroke-width="3" stroke-dasharray="6 4"/>')
            else:
                out.append(f'<rect x="{x + 1.5}" y="{ty + 1.5}" width="{tw - 3}" height="43" rx="6" fill="none" stroke="#b9bab5" stroke-width="2.5" stroke-dasharray="5 4"/>')
        ax = n * (tw + tg) + 6
        out.append(f'<line x1="{ax}" x2="{ax + 46}" y1="{ty + 23}" y2="{ty + 23}" stroke="{INK2}" stroke-width="4"/>')
        out.append(f'<path d="M{ax + 46},{ty + 11} L{ax + 66},{ty + 23} L{ax + 46},{ty + 35} Z" fill="{INK2}"/>')
        bx = ax + 80
        out.append(f'<rect x="{bx}" y="{ty - 6}" width="{W - bx}" height="58" rx="29" fill="{col}"/>')
        out.append(text(bx + (W - bx) / 2, ty + 32, lab, 26, "#ffffff", "middle", 700))
        out.append(text(W, ty + 90, act, 24, INK2, "end", 500))
    out.append("</svg>")
    return "\n".join(out)


def main():
    tpl = (HERE / "template.html").read_text()
    qr = (HERE / "qr_repo.svg").read_text()
    qr = re.sub(r'width="33" height="33"', 'viewBox="0 0 33 33" width="100%" height="100%" shape-rendering="crispEdges"', qr, count=1)
    parts = {
        "CHART_T1": chart_table1(),
        "CHART_T2": chart_table2(),
        "CHART_T3": chart_table3(),
        "CHART_T4": chart_table4(),
        "LEGEND_T4": legend_table4(),
        "CHART_MASKS": chart_masks(),
        "METHOD_DIAGRAM": method_diagram(),
        "QR": qr,
        "C_BASE": BASE, "C_PROMPT": PROMPT, "C_V1": V1, "C_MIX": MIX, "C_MIXCC": MIXCC,
    }
    for k, v in parts.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{[A-Z_0-9]+\}\}", tpl)
    assert not left, left
    (HERE / "poster.html").write_text(tpl)
    print("wrote", HERE / "poster.html")


if __name__ == "__main__":
    main()
