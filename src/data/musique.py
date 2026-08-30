"""Musique loader: convert one HF example into the normalized form
the labeling pipeline expects.

Schema (dgslibisey/MuSiQue, validation split, 2,417 examples):
  id, paragraphs, question, question_decomposition, answer, answer_aliases, answerable
  paragraphs: list[ {idx, title, paragraph_text, is_supporting} ]
  question_decomposition: list[ {id, question, answer, paragraph_support_idx} ]
    — one entry per reasoning hop, each pointing at one supporting paragraph
      and providing the *literal* sub-answer the model needs to retrieve.

We render the context as the concatenation of all paragraphs (supporting +
distractors), then identify char spans of each is_supporting=True paragraph
and convert those to token-level half-open intervals using the wrapper's
fast tokenizer's offset_mapping.

Span granularity (constructor arg):
  - "paragraph" : the whole supporting paragraph is the required span
                  (~80-120 tokens; heavy averaging dampens per-span variance)
  - "tight"     : the literal sub-answer string within each supporting
                  paragraph is the required span (~3-15 tokens). Falls back
                  to paragraph-level for any hop whose sub-answer string
                  doesn't appear verbatim in its paragraph.

v1 limitations (documented in label_construction.py):
  - Every is_supporting=True paragraph is treated as a REQUIRED span.
    Musique is multi-hop, so the gold chain typically requires all of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase

from src.data.label_construction import SupportingSpan


@dataclass
class MusiqueExample:
    id: str
    question: str
    answer: str
    answer_aliases: list[str]
    answerable: bool
    context: str  # rendered concatenation of all paragraphs
    context_ids: torch.Tensor  # 1-D LongTensor (CPU)
    supporting_spans: list[SupportingSpan]
    n_paragraphs: int
    supporting_paragraph_titles: list[str]
    span_granularity: str  # "paragraph" or "tight"
    n_tight_spans_located: int  # how many sub-answers were found verbatim
    n_paragraph_fallbacks: int  # supporting paragraphs that fell back to whole-paragraph


# Format used to concatenate paragraphs into a single context string.
# Each paragraph rendered as "[Title]\nbody" then joined by blank lines.
# The exact rendering is not load-bearing for the labeler — it only matters
# that the supporting char-spans we identify still fall on whole-paragraph
# boundaries inside `context`.
_PARA_FMT = "[{title}]\n{text}"
_PARA_SEP = "\n\n"


def _render_context(paragraphs: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate paragraphs and return (context, char_spans).

    char_spans[i] is the [start, end) char offset of paragraphs[i]['paragraph_text']'s
    rendered paragraph (title + body) in the returned `context` string.
    """
    pieces: list[str] = []
    char_spans: list[tuple[int, int]] = []
    cursor = 0
    for i, p in enumerate(paragraphs):
        rendered = _PARA_FMT.format(title=p["title"], text=p["paragraph_text"])
        if i > 0:
            cursor += len(_PARA_SEP)
            pieces.append(_PARA_SEP)
        start = cursor
        pieces.append(rendered)
        cursor += len(rendered)
        char_spans.append((start, cursor))
    return "".join(pieces), char_spans


def _char_span_to_token_span(
    offsets: list[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> tuple[int, int]:
    """Convert a [char_start, char_end) span to a [tok_start, tok_end) span
    over the tokens whose char ranges overlap the char span.

    Tokens whose offset range [s, e) overlaps [char_start, char_end) are
    included. Empty-offset tokens (e == s) are skipped — these are special
    tokens or whitespace artifacts that shouldn't carry semantic content.
    """
    indices = [
        i for i, (s, e) in enumerate(offsets)
        if e > s and not (e <= char_start or s >= char_end)
    ]
    if not indices:
        raise ValueError(
            f"no tokens overlap char span [{char_start}, {char_end}); "
            f"likely an alignment bug between renderer and tokenizer."
        )
    return indices[0], indices[-1] + 1


def _build_paragraph_spans(
    example: dict,
    char_spans: list[tuple[int, int]],
    offsets: list[tuple[int, int]],
) -> tuple[list[SupportingSpan], list[str]]:
    spans: list[SupportingSpan] = []
    titles: list[str] = []
    for i, p in enumerate(example["paragraphs"]):
        if not p["is_supporting"]:
            continue
        char_start, char_end = char_spans[i]
        tok_start, tok_end = _char_span_to_token_span(offsets, char_start, char_end)
        spans.append(
            SupportingSpan(
                start=tok_start,
                end=tok_end,
                source="musique_is_supporting",
                required=True,
                span_id=f"{example['id']}/p{p['idx']}",
            )
        )
        titles.append(p["title"])
    return spans, titles


def _build_tight_spans(
    example: dict,
    char_spans: list[tuple[int, int]],
    full_context_lower: str,
    offsets: list[tuple[int, int]],
) -> tuple[list[SupportingSpan], list[str], int, int]:
    """One tight span per is_supporting paragraph, located via question_decomposition.

    For each supporting paragraph, look up its sub-answer in the
    question_decomposition (entries are keyed by paragraph_support_idx).
    Locate the sub-answer string within the paragraph's char range
    (case-insensitive, first occurrence). Fall back to the whole-paragraph
    span for any paragraph whose sub-answer cannot be located verbatim.

    Returns (spans, titles, n_tight, n_fallback).
    """
    qd = example.get("question_decomposition") or []
    para_idx_to_subanswer: dict[int, str] = {}
    for step in qd:
        psi = step.get("paragraph_support_idx")
        ans = (step.get("answer") or "").strip()
        if psi is not None and ans:
            para_idx_to_subanswer[psi] = ans

    spans: list[SupportingSpan] = []
    titles: list[str] = []
    n_tight = 0
    n_fallback = 0
    for i, p in enumerate(example["paragraphs"]):
        if not p["is_supporting"]:
            continue
        para_char_start, para_char_end = char_spans[i]
        sub_answer = para_idx_to_subanswer.get(p["idx"])

        tok_span: tuple[int, int] | None = None
        source = "musique_paragraph_fallback"
        if sub_answer:
            search_region = full_context_lower[para_char_start:para_char_end]
            rel_idx = search_region.find(sub_answer.lower())
            if rel_idx >= 0:
                abs_start = para_char_start + rel_idx
                abs_end = abs_start + len(sub_answer)
                try:
                    tok_span = _char_span_to_token_span(offsets, abs_start, abs_end)
                    source = "musique_qd_subanswer"
                    n_tight += 1
                except ValueError:
                    tok_span = None
        if tok_span is None:
            tok_span = _char_span_to_token_span(offsets, para_char_start, para_char_end)
            n_fallback += 1

        spans.append(
            SupportingSpan(
                start=tok_span[0],
                end=tok_span[1],
                source=source,
                required=True,
                span_id=f"{example['id']}/p{p['idx']}",
            )
        )
        titles.append(p["title"])
    return spans, titles, n_tight, n_fallback


def normalize_musique_example(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
    *,
    span_granularity: str = "paragraph",
) -> MusiqueExample:
    """Convert one Musique row into a MusiqueExample with token-level spans.

    span_granularity:
      "paragraph" — entire is_supporting paragraph is the required span
      "tight"     — per-hop sub-answer string located inside its paragraph;
                    falls back to paragraph-level for any hop whose sub-answer
                    cannot be located verbatim.
    """
    if span_granularity not in ("paragraph", "tight"):
        raise ValueError(
            f"span_granularity must be 'paragraph' or 'tight', got {span_granularity!r}"
        )

    context, char_spans = _render_context(example["paragraphs"])
    enc = tokenizer(
        context,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    context_ids = enc.input_ids.squeeze(0)
    offsets = enc.offset_mapping.squeeze(0).tolist()

    if span_granularity == "tight":
        spans, titles, n_tight, n_fallback = _build_tight_spans(
            example, char_spans, context.lower(), offsets
        )
    else:
        spans, titles = _build_paragraph_spans(example, char_spans, offsets)
        n_tight, n_fallback = 0, len(spans)

    return MusiqueExample(
        id=example["id"],
        question=example["question"],
        answer=example["answer"],
        answer_aliases=list(example.get("answer_aliases") or []),
        answerable=bool(example.get("answerable", True)),
        context=context,
        context_ids=context_ids,
        supporting_spans=spans,
        n_paragraphs=len(example["paragraphs"]),
        supporting_paragraph_titles=titles,
        span_granularity=span_granularity,
        n_tight_spans_located=n_tight,
        n_paragraph_fallbacks=n_fallback,
    )


__all__ = ["MusiqueExample", "normalize_musique_example"]
