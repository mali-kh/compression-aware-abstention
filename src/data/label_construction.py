"""Cache-state-conditional label construction.

Given a (context, query, gold_answer, supporting_spans, kept_mask) tuple,
decide whether the *training target* should be:

  - CONFIDENT_ANSWER : the gold answer (all supporting evidence retained)
  - ABSTAIN          : a templated abstention (required evidence evicted)
  - FLAGGED_PARTIAL  : partial-answer + flag (mixed retention; gated off in v1)
  - DROP             : the example sits in the "uncertain zone" between
                       thresholds and is excluded from training entirely.

Thresholds:

  retention(span) >= tau_high  -> "well retained"
  retention(span) <  tau_low   -> "severely lost"
  in between                   -> uncertain (drop, by default)

v1 design choice (Option B): FLAGGED_PARTIAL is gated off and collapses to
ABSTAIN. The schema and code paths are preserved because the final research
output (Option A) requires the three-label form. See README and
configs/labeling.yaml.

Required-vs-redundant supporting spans is treated as an open research
question; v1 assumes every annotated span is required
(`assume_all_spans_required=True`). Documented limitation, not a heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

import torch
import yaml


class LabelType(str, Enum):
    CONFIDENT_ANSWER = "confident_answer"
    ABSTAIN = "abstain"
    FLAGGED_PARTIAL = "flagged_partial"
    DROP = "drop"


@dataclass(frozen=True)
class SupportingSpan:
    """Token-level span over the original tokenized context.

    Indices are half-open: [start, end). They refer to positions in the
    same tokenization the compressor's kept_mask is defined over — alignment
    between the two is the caller's responsibility (and the failure mode the
    KVzip wrapper test in Deliverable 3 exists to catch).
    """

    start: int
    end: int
    source: str = "annotation"
    required: bool = True
    span_id: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"SupportingSpan.start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(
                f"SupportingSpan.end ({self.end}) must be > start ({self.start})"
            )

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class CacheState:
    """The post-compression token-level retention pattern.

    `kept_mask[i] == True` iff the KV entry for the i-th original-context
    token survives compression at this ratio.
    """

    kept_mask: torch.Tensor
    compressor_name: str
    ratio: float

    def __post_init__(self) -> None:
        if not isinstance(self.kept_mask, torch.Tensor):
            raise TypeError(
                f"kept_mask must be a torch.Tensor, got {type(self.kept_mask).__name__}"
            )
        if self.kept_mask.dtype != torch.bool:
            raise TypeError(
                f"kept_mask must be torch.bool, got dtype={self.kept_mask.dtype}"
            )
        if self.kept_mask.ndim != 1:
            raise ValueError(
                f"kept_mask must be 1-D, got shape {tuple(self.kept_mask.shape)}"
            )
        if not 0.0 <= self.ratio <= 1.0:
            raise ValueError(f"ratio must be in [0,1], got {self.ratio}")

    @property
    def seq_len(self) -> int:
        return int(self.kept_mask.shape[0])

    def retention_fraction(self, start: int, end: int) -> float:
        if start < 0 or end > self.seq_len or start >= end:
            raise IndexError(
                f"span [{start},{end}) out of bounds for seq_len={self.seq_len}"
            )
        return float(self.kept_mask[start:end].float().mean().item())


@dataclass(frozen=True)
class LabelConfig:
    tau_high: float = 0.8
    tau_low: float = 0.3
    drop_uncertain_zone: bool = True
    assume_all_spans_required: bool = True
    enable_flagged_partial: bool = False
    abstention_template: str = (
        "I cannot determine the answer to this question from the available context."
    )

    def __post_init__(self) -> None:
        if not (0.0 < self.tau_low < self.tau_high <= 1.0):
            raise ValueError(
                f"thresholds must satisfy 0 < tau_low ({self.tau_low}) "
                f"< tau_high ({self.tau_high}) <= 1.0"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LabelConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        section = data.get("label_construction", data)
        known = {k: v for k, v in section.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass(frozen=True)
class SpanRetention:
    span: SupportingSpan
    fraction_kept: float


@dataclass(frozen=True)
class Label:
    type: LabelType
    target_output: str
    span_retentions: tuple[SpanRetention, ...]
    rationale: str

    @property
    def in_training_set(self) -> bool:
        return self.type is not LabelType.DROP


def label_example(
    context: str,
    query: str,
    answer: str,
    supporting_spans: Sequence[SupportingSpan],
    cache_state: CacheState,
    cfg: LabelConfig | None = None,
) -> Label:
    """Classify a (context, query, answer, spans, cache_state) tuple.

    `context` and `query` are accepted so callers can pair the returned
    Label with its source example without a separate join, but the
    classifier itself does not parse them — only span retention values
    determine the label.
    """
    cfg = cfg or LabelConfig()
    del context, query

    if not supporting_spans:
        raise ValueError(
            "supporting_spans must be non-empty — the labeler relies on at "
            "least one annotated span to decide retention."
        )
    for s in supporting_spans:
        if s.end > cache_state.seq_len:
            raise ValueError(
                f"span {s} out of bounds for seq_len={cache_state.seq_len}"
            )

    retentions = tuple(
        SpanRetention(
            span=s,
            fraction_kept=cache_state.retention_fraction(s.start, s.end),
        )
        for s in supporting_spans
    )

    if cfg.assume_all_spans_required:
        required = retentions
    else:
        required = tuple(r for r in retentions if r.span.required)
        if not required:
            raise ValueError(
                "no required spans (and assume_all_spans_required=False); "
                "cannot label."
            )

    if all(r.fraction_kept >= cfg.tau_high for r in required):
        return Label(
            type=LabelType.CONFIDENT_ANSWER,
            target_output=answer,
            span_retentions=retentions,
            rationale=(
                f"all {len(required)} required span(s) retained "
                f">= tau_high={cfg.tau_high}"
            ),
        )

    has_severely_lost = any(r.fraction_kept < cfg.tau_low for r in required)
    has_well_retained = any(r.fraction_kept >= cfg.tau_high for r in required)

    if has_severely_lost and has_well_retained:
        if cfg.enable_flagged_partial:
            return Label(
                type=LabelType.FLAGGED_PARTIAL,
                # Real partial-answer generation is its own piece of work
                # (Option A reactivation); v1 ships a placeholder.
                target_output="[FLAGGED_PARTIAL placeholder — partial-answer template TODO]",
                span_retentions=retentions,
                rationale=(
                    "mixed retention: some required spans >= tau_high, "
                    "some < tau_low"
                ),
            )
        return Label(
            type=LabelType.ABSTAIN,
            target_output=cfg.abstention_template,
            span_retentions=retentions,
            rationale=(
                "mixed retention; FLAGGED_PARTIAL disabled in v1, "
                "collapsed to ABSTAIN (Option B)"
            ),
        )

    if has_severely_lost:
        return Label(
            type=LabelType.ABSTAIN,
            target_output=cfg.abstention_template,
            span_retentions=retentions,
            rationale=f"required span(s) retained < tau_low={cfg.tau_low}",
        )

    if cfg.drop_uncertain_zone:
        return Label(
            type=LabelType.DROP,
            target_output="",
            span_retentions=retentions,
            rationale=(
                f"all spans in uncertain zone "
                f"[tau_low={cfg.tau_low}, tau_high={cfg.tau_high})"
            ),
        )
    return Label(
        type=LabelType.ABSTAIN,
        target_output=cfg.abstention_template,
        span_retentions=retentions,
        rationale="uncertain zone with drop_uncertain_zone=False; defaulted to ABSTAIN",
    )


__all__ = [
    "CacheState",
    "Label",
    "LabelConfig",
    "LabelType",
    "SpanRetention",
    "SupportingSpan",
    "label_example",
]
