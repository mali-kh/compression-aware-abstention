"""Synthetic fixtures for label_construction unit tests.

These exercise the labeler's logic against hand-crafted (context, spans,
kept_mask) tuples whose expected label is unambiguous to a human reader.
The "tokens" here are word-level (regex split into words + punctuation
chars) — we are NOT testing real tokenization. That is the failure mode
the KVzip wrapper test in Deliverable 3 exists to catch.

10 base scenarios x 5 retention patterns = 50 fixtures.

Span lengths are pinned at exactly 10 tokens so every retention fraction
in {0.0, 0.1, 0.5, 0.8, 0.9, 1.0} maps to a clean integer count of kept
tokens; this avoids round() artifacts at the threshold boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from src.data.label_construction import (
    CacheState,
    LabelType,
    SupportingSpan,
)

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_SPAN_LEN = 10


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def find_span(tokens: list[str], sub_tokens: list[str]) -> tuple[int, int]:
    n = len(sub_tokens)
    for i in range(len(tokens) - n + 1):
        if tokens[i : i + n] == sub_tokens:
            return i, i + n
    raise ValueError(
        f"substring tokens {sub_tokens!r} not found in tokens {tokens!r}"
    )


def craft_mask(seq_len: int, span_recipe: list[tuple[int, int, float]]) -> torch.Tensor:
    """Build a kept_mask of shape (seq_len,) from (start, end, frac) triples.

    Tokens outside any span are kept by default. For each span, we keep
    the FIRST round(length * frac) tokens and evict the rest. Non-span
    tokens are not relevant to the labeler — only span retention matters.
    """
    mask = torch.ones(seq_len, dtype=torch.bool)
    for start, end, frac in span_recipe:
        n = end - start
        n_keep = round(n * frac)
        mask[start + n_keep : end] = False
    return mask


@dataclass
class LabelFixture:
    fid: str
    description: str
    context: str
    query: str
    answer: str
    spans: list[SupportingSpan]
    cache_state: CacheState
    expected_label: LabelType
    achieved_retentions: list[float]
    pattern: str

    @property
    def is_multi_span(self) -> bool:
        return len(self.spans) > 1


# 5 single-span scenarios. Each context is engineered so the supporting
# substring tokenizes to exactly _SPAN_LEN (=10) tokens.
SINGLE_SPAN_SCENARIOS: list[dict] = [
    {
        "stem": "marie_curie",
        "description": "Marie Curie birthplace fact",
        "context": (
            "Marie Curie was born in Warsaw on November seventh, 1867. "
            "The city was at that time part of the Russian Empire. She "
            "later moved to Paris and conducted research on radioactivity."
        ),
        "query": "Where was Marie Curie born?",
        "answer": "Warsaw",
        "support": "Marie Curie was born in Warsaw on November seventh ,",
    },
    {
        "stem": "speed_of_light",
        "description": "Speed-of-light value fact",
        "context": (
            "The speed of light in vacuum is exactly 299792458 meters per "
            "second according to the SI definition. This constant is "
            "denoted by the lowercase letter c in physics formulae."
        ),
        "query": "What is the speed of light in vacuum?",
        "answer": "299792458 meters per second",
        "support": "is exactly 299792458 meters per second according to the SI",
    },
    {
        "stem": "everest_height",
        "description": "Mount Everest height fact",
        "context": (
            "Mount Everest reaches a summit elevation of 8848 meters above "
            "mean sea level, making it the tallest mountain on the surface "
            "of the Earth as currently measured by Nepali authorities."
        ),
        "query": "How tall is Mount Everest?",
        "answer": "8848 meters",
        "support": "reaches a summit elevation of 8848 meters above mean sea",
    },
    {
        "stem": "capital_australia",
        "description": "Capital of Australia (NIAH-style)",
        "context": (
            "Many travelers visit Sydney expecting it to be the capital, "
            "but the seat of Australian federal government is Canberra in "
            "the Australian Capital Territory, established in 1913 by an act."
        ),
        "query": "What is the capital of Australia?",
        "answer": "Canberra",
        "support": "Australian federal government is Canberra in the Australian Capital Territory",
    },
    {
        "stem": "photosynthesis_reactants",
        "description": "Photosynthesis chemical inputs fact",
        "context": (
            "In oxygenic photosynthesis the plant takes carbon dioxide and "
            "water and uses sunlight to produce glucose and oxygen. The "
            "reaction occurs primarily in the chloroplasts of leaves."
        ),
        "query": "What molecules are consumed in photosynthesis?",
        "answer": "carbon dioxide and water",
        "support": "plant takes carbon dioxide and water and uses sunlight to",
    },
]

# 5 multi-span scenarios with TWO supporting spans each. Each span tokenizes
# to exactly _SPAN_LEN tokens.
MULTI_SPAN_SCENARIOS: list[dict] = [
    {
        "stem": "openai_country",
        "description": "Bridge: OpenAI HQ -> SF -> United States",
        "context": (
            "OpenAI maintains its headquarters in San Francisco, California, "
            "where the company was founded. San Francisco is a major city "
            "located on the west coast of the United States of America today."
        ),
        "query": "In which country is OpenAI's headquarters located?",
        "answer": "United States",
        "supports": [
            "OpenAI maintains its headquarters in San Francisco , California ,",
            "located on the west coast of the United States of",
        ],
    },
    {
        "stem": "ww2_to_berlin_wall",
        "description": "Date arithmetic: WW2 end year and Berlin Wall fall",
        "context": (
            "The Second World War ended in the year 1945 with surrenders "
            "in Europe and the Pacific. The Berlin Wall fell in 1989 after "
            "decades of separation, marking the symbolic end of the Cold War."
        ),
        "query": "How many years between the end of WW2 and the fall of the Berlin Wall?",
        "answer": "44 years",
        "supports": [
            "The Second World War ended in the year 1945 with",
            "The Berlin Wall fell in 1989 after decades of separation",
        ],
    },
    {
        "stem": "everest_k2_taller",
        "description": "Comparison: which is taller, Everest or K2",
        "context": (
            "Mount Everest stands at 8848 meters of elevation as the "
            "tallest peak on Earth. K2 measures 8611 meters in the Karakoram "
            "range and is widely considered to be a more difficult climb."
        ),
        "query": "Which mountain is taller, Everest or K2?",
        "answer": "Mount Everest",
        "supports": [
            "Mount Everest stands at 8848 meters of elevation as the",
            "K2 measures 8611 meters in the Karakoram range and is",
        ],
    },
    {
        "stem": "author_book_year",
        "description": "Author + book + publication year",
        "context": (
            "George Orwell published the dystopian novel Nineteen Eighty-Four "
            "in the year 1949. The book imagines a totalitarian future state "
            "and introduced terms such as Big Brother to the popular lexicon."
        ),
        "query": "Who wrote Nineteen Eighty-Four and when?",
        "answer": "George Orwell, 1949",
        "supports": [
            "George Orwell published the dystopian novel Nineteen Eighty - Four",
            "in the year 1949 . The book imagines a totalitarian",
        ],
    },
    {
        "stem": "company_founder_city",
        "description": "Founder + city: Tesla Inc. founder + HQ city",
        "context": (
            "Tesla Inc. was co-founded by Martin Eberhard and Marc Tarpenning "
            "in Silicon Valley. The current headquarters of the company is "
            "located in Austin, Texas, having moved from Palo Alto in 2021."
        ),
        "query": "Who co-founded Tesla and where is its current headquarters?",
        "answer": "Martin Eberhard and Marc Tarpenning; Austin, Texas",
        "supports": [
            "co - founded by Martin Eberhard and Marc Tarpenning in",
            "is located in Austin , Texas , having moved from",
        ],
    },
]


# Retention patterns: each maps a pattern name -> (per-span fractions, expected label).
# All scenarios use _SPAN_LEN=10 tokens so these fractions land on clean integer counts.
SINGLE_SPAN_PATTERNS: list[tuple[str, list[float], LabelType]] = [
    ("full_retention",      [1.0],  LabelType.CONFIDENT_ANSWER),
    ("high_retention",      [0.9],  LabelType.CONFIDENT_ANSWER),
    ("threshold_high",      [0.8],  LabelType.CONFIDENT_ANSWER),  # exactly tau_high
    ("uncertain_zone",      [0.5],  LabelType.DROP),
    ("severe_loss",         [0.1],  LabelType.ABSTAIN),
]

MULTI_SPAN_PATTERNS: list[tuple[str, list[float], LabelType]] = [
    ("full_retention",      [1.0, 1.0],  LabelType.CONFIDENT_ANSWER),
    ("uniform_uncertain",   [0.5, 0.5],  LabelType.DROP),
    ("uniform_severe_loss", [0.1, 0.1],  LabelType.ABSTAIN),
    ("mixed_kept_and_lost", [1.0, 0.0],  LabelType.ABSTAIN),  # collapsed from FLAGGED_PARTIAL (Option B)
    ("partial_loss",        [0.9, 0.1],  LabelType.ABSTAIN),  # one well-retained, one severely lost -> mixed branch
]


def _build_fixture(
    *,
    fid: str,
    description: str,
    context: str,
    query: str,
    answer: str,
    supports: list[str],
    pattern_name: str,
    fractions: list[float],
    expected_label: LabelType,
) -> LabelFixture:
    tokens = tokenize(context)
    seq_len = len(tokens)
    span_locs: list[tuple[int, int]] = []
    achieved: list[float] = []
    for sub in supports:
        sub_tokens = tokenize(sub)
        if len(sub_tokens) != _SPAN_LEN:
            raise AssertionError(
                f"fixture {fid}: support substring tokenizes to "
                f"{len(sub_tokens)} tokens, expected {_SPAN_LEN}: {sub_tokens!r}"
            )
        s, e = find_span(tokens, sub_tokens)
        span_locs.append((s, e))
    recipe = [(s, e, f) for (s, e), f in zip(span_locs, fractions, strict=True)]
    mask = craft_mask(seq_len, recipe)
    for (s, e), f in zip(span_locs, fractions, strict=True):
        n_keep = round((e - s) * f)
        achieved.append(n_keep / (e - s))
    spans = [
        SupportingSpan(start=s, end=e, source="annotation", required=True, span_id=f"{fid}/{i}")
        for i, (s, e) in enumerate(span_locs)
    ]
    cache = CacheState(
        kept_mask=mask,
        compressor_name="synthetic",
        ratio=float(sum(fractions) / len(fractions)),
    )
    return LabelFixture(
        fid=fid,
        description=description,
        context=context,
        query=query,
        answer=answer,
        spans=spans,
        cache_state=cache,
        expected_label=expected_label,
        achieved_retentions=achieved,
        pattern=pattern_name,
    )


def build_all_fixtures() -> list[LabelFixture]:
    fixtures: list[LabelFixture] = []
    for sc in SINGLE_SPAN_SCENARIOS:
        for pname, fracs, expected in SINGLE_SPAN_PATTERNS:
            fixtures.append(
                _build_fixture(
                    fid=f"{sc['stem']}__{pname}",
                    description=f"{sc['description']} [{pname}]",
                    context=sc["context"],
                    query=sc["query"],
                    answer=sc["answer"],
                    supports=[sc["support"]],
                    pattern_name=pname,
                    fractions=fracs,
                    expected_label=expected,
                )
            )
    for sc in MULTI_SPAN_SCENARIOS:
        for pname, fracs, expected in MULTI_SPAN_PATTERNS:
            fixtures.append(
                _build_fixture(
                    fid=f"{sc['stem']}__{pname}",
                    description=f"{sc['description']} [{pname}]",
                    context=sc["context"],
                    query=sc["query"],
                    answer=sc["answer"],
                    supports=sc["supports"],
                    pattern_name=pname,
                    fractions=fracs,
                    expected_label=expected,
                )
            )
    return fixtures


ALL_FIXTURES: list[LabelFixture] = build_all_fixtures()
assert len(ALL_FIXTURES) == 50, f"expected 50 fixtures, got {len(ALL_FIXTURES)}"
