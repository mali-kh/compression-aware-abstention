"""Sanity-check tests for src/data/label_construction.py.

Critical-path test for week 1: every one of 50 hand-crafted fixtures must
land on its expected label. This file *also* writes a human-readable
markdown report of 10 representative examples to ``tests/inspected_labels.md``
so the user can verify by eye that the labels are obviously correct.

If a label here is wrong to a human reader, the rest of the pipeline
(LoRA training, eval, the whole research arc) is built on bad signal.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.data.label_construction import (
    CacheState,
    LabelConfig,
    LabelType,
    SupportingSpan,
    label_example,
)
from tests.fixtures.label_fixtures import ALL_FIXTURES, LabelFixture, tokenize

INSPECTION_OUT = Path(__file__).parent / "inspected_labels.md"


# ----------------------- the 50-fixture sanity check -----------------------


@pytest.mark.parametrize("fix", ALL_FIXTURES, ids=lambda f: f.fid)
def test_labeler_matches_expected(fix: LabelFixture) -> None:
    label = label_example(
        context=fix.context,
        query=fix.query,
        answer=fix.answer,
        supporting_spans=fix.spans,
        cache_state=fix.cache_state,
    )
    if label.type is not fix.expected_label:
        retentions = [
            (sr.span.start, sr.span.end, round(sr.fraction_kept, 3))
            for sr in label.span_retentions
        ]
        pytest.fail(
            f"{fix.fid}: expected={fix.expected_label.value} "
            f"got={label.type.value}\n"
            f"  pattern         = {fix.pattern}\n"
            f"  achieved        = {fix.achieved_retentions}\n"
            f"  span retentions = {retentions}\n"
            f"  rationale       = {label.rationale}"
        )


def test_fixture_count() -> None:
    assert len(ALL_FIXTURES) == 50


def test_v1_never_emits_flagged_partial() -> None:
    """Option B contract: with default config, FLAGGED_PARTIAL never appears."""
    for fix in ALL_FIXTURES:
        label = label_example(
            context=fix.context,
            query=fix.query,
            answer=fix.answer,
            supporting_spans=fix.spans,
            cache_state=fix.cache_state,
        )
        assert label.type is not LabelType.FLAGGED_PARTIAL, (
            f"{fix.fid} produced FLAGGED_PARTIAL under default v1 config"
        )


def test_option_a_reactivation_emits_flagged_partial() -> None:
    """Flipping enable_flagged_partial=True must reactivate the third label."""
    cfg = LabelConfig(enable_flagged_partial=True)
    # The mixed_kept_and_lost pattern (1.0, 0.0) is engineered to land in the
    # mixed branch; under Option A it should emit FLAGGED_PARTIAL.
    mixed = [f for f in ALL_FIXTURES if f.pattern == "mixed_kept_and_lost"]
    assert mixed, "no mixed-pattern fixture found"
    for fix in mixed:
        label = label_example(
            context=fix.context,
            query=fix.query,
            answer=fix.answer,
            supporting_spans=fix.spans,
            cache_state=fix.cache_state,
            cfg=cfg,
        )
        assert label.type is LabelType.FLAGGED_PARTIAL, fix.fid


# ----------------------------- edge cases ---------------------------------


class TestEdgeCases:
    def test_empty_supporting_spans_raises(self) -> None:
        cs = CacheState(torch.ones(10, dtype=torch.bool), "synthetic", 1.0)
        with pytest.raises(ValueError, match="non-empty"):
            label_example("ctx", "q", "a", [], cs)

    def test_span_out_of_bounds_raises(self) -> None:
        cs = CacheState(torch.ones(10, dtype=torch.bool), "synthetic", 1.0)
        span = SupportingSpan(start=5, end=15)
        with pytest.raises(ValueError, match="out of bounds"):
            label_example("ctx", "q", "a", [span], cs)

    def test_span_with_end_le_start_raises(self) -> None:
        with pytest.raises(ValueError, match="must be > start"):
            SupportingSpan(start=5, end=5)
        with pytest.raises(ValueError, match="must be > start"):
            SupportingSpan(start=5, end=4)

    def test_negative_span_start_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            SupportingSpan(start=-1, end=5)

    def test_invalid_thresholds_raise(self) -> None:
        with pytest.raises(ValueError, match="thresholds"):
            LabelConfig(tau_low=0.5, tau_high=0.3)
        with pytest.raises(ValueError, match="thresholds"):
            LabelConfig(tau_low=0.0, tau_high=0.8)
        with pytest.raises(ValueError, match="thresholds"):
            LabelConfig(tau_low=0.3, tau_high=1.5)

    def test_kept_mask_dtype_validation(self) -> None:
        with pytest.raises(TypeError, match="torch.bool"):
            CacheState(torch.ones(10, dtype=torch.float32), "synthetic", 1.0)

    def test_kept_mask_shape_validation(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            CacheState(torch.ones((4, 4), dtype=torch.bool), "synthetic", 1.0)

    def test_kept_mask_must_be_tensor(self) -> None:
        with pytest.raises(TypeError, match="torch.Tensor"):
            CacheState([True, False, True], "synthetic", 1.0)  # type: ignore[arg-type]

    def test_threshold_high_inclusive(self) -> None:
        """Retention exactly at tau_high counts as well-retained (>=)."""
        mask = torch.ones(10, dtype=torch.bool)
        mask[8:] = False  # 8/10 = 0.8 retention
        cs = CacheState(mask, "synthetic", 0.8)
        span = SupportingSpan(start=0, end=10)
        label = label_example("c", "q", "a", [span], cs)
        assert label.type is LabelType.CONFIDENT_ANSWER

    def test_threshold_low_exclusive(self) -> None:
        """Retention exactly at tau_low is NOT severely lost (must be <)."""
        mask = torch.ones(10, dtype=torch.bool)
        mask[3:] = False  # 3/10 = 0.3 retention, exactly tau_low
        cs = CacheState(mask, "synthetic", 0.3)
        span = SupportingSpan(start=0, end=10)
        label = label_example("c", "q", "a", [span], cs)
        # 0.3 is not < 0.3, so this falls into the uncertain zone -> DROP
        assert label.type is LabelType.DROP

    def test_assume_all_required_false_with_none_required_raises(self) -> None:
        cfg = LabelConfig(assume_all_spans_required=False)
        cs = CacheState(torch.ones(10, dtype=torch.bool), "synthetic", 1.0)
        span = SupportingSpan(start=0, end=5, required=False)
        with pytest.raises(ValueError, match="no required spans"):
            label_example("c", "q", "a", [span], cs, cfg=cfg)

    def test_drop_uncertain_zone_false_defaults_to_abstain(self) -> None:
        cfg = LabelConfig(drop_uncertain_zone=False)
        mask = torch.ones(10, dtype=torch.bool)
        mask[5:] = False  # 0.5 retention -> uncertain zone
        cs = CacheState(mask, "synthetic", 0.5)
        span = SupportingSpan(start=0, end=10)
        label = label_example("c", "q", "a", [span], cs, cfg=cfg)
        assert label.type is LabelType.ABSTAIN
        assert "drop_uncertain_zone=False" in label.rationale


# ----------------------- markdown inspection artifact ----------------------


def _render_span_as_kept_marker(
    fix: LabelFixture, span_idx: int
) -> str:
    """Render a single span's tokens with kept (✓) / evicted (✗) markers."""
    span = fix.spans[span_idx]
    tokens = tokenize(fix.context)
    mask = fix.cache_state.kept_mask
    marks = []
    for i in range(span.start, span.end):
        glyph = "✓" if mask[i].item() else "✗"
        marks.append(f"{tokens[i]}{glyph}")
    n_kept = int(mask[span.start : span.end].sum().item())
    n_total = span.end - span.start
    frac = n_kept / n_total
    return (
        f"  span[{span.start}:{span.end}) `{' '.join(tokens[span.start:span.end])}`\n"
        f"  per-token kept: {' '.join(marks)}\n"
        f"  retention: {n_kept}/{n_total} = {frac:.2f}"
    )


def _render_one(fix: LabelFixture, computed_label_type: LabelType, target_output: str, rationale: str) -> str:
    span_blocks = "\n".join(
        _render_span_as_kept_marker(fix, i) for i in range(len(fix.spans))
    )
    match_glyph = "✅" if computed_label_type is fix.expected_label else "❌"
    return (
        f"### `{fix.fid}` {match_glyph}\n"
        f"_{fix.description}_  \n"
        f"**pattern**: `{fix.pattern}` &nbsp;&nbsp;"
        f"**expected**: `{fix.expected_label.value}` &nbsp;&nbsp;"
        f"**computed**: `{computed_label_type.value}`\n\n"
        f"**Context**:  \n> {fix.context}\n\n"
        f"**Query**: {fix.query}  \n"
        f"**Gold answer**: `{fix.answer}`\n\n"
        f"**Supporting spans (token-level, with eviction marks):**\n"
        f"```\n{span_blocks}\n```\n\n"
        f"**Computed target_output**: `{target_output!r}`  \n"
        f"**Rationale**: {rationale}\n"
    )


def _pick_inspection_set(fixtures: list[LabelFixture]) -> list[LabelFixture]:
    """Pick 10 examples covering the label-type / arity matrix.

    We want the user to see, at a glance, that the labeler does the right
    thing on:
      - single-span CONFIDENT (full + threshold)
      - single-span DROP
      - single-span ABSTAIN
      - multi-span CONFIDENT
      - multi-span DROP
      - multi-span ABSTAIN (uniform severe loss)
      - multi-span ABSTAIN (mixed kept_and_lost — the FLAGGED_PARTIAL collapse)
      - multi-span ABSTAIN (partial_loss — different mixed pattern)
    """
    by = {f.fid: f for f in fixtures}
    picks_ids = [
        "marie_curie__full_retention",
        "speed_of_light__threshold_high",
        "everest_height__uncertain_zone",
        "capital_australia__severe_loss",
        "photosynthesis_reactants__high_retention",
        "openai_country__full_retention",
        "ww2_to_berlin_wall__uniform_uncertain",
        "everest_k2_taller__uniform_severe_loss",
        "author_book_year__mixed_kept_and_lost",
        "company_founder_city__partial_loss",
    ]
    return [by[i] for i in picks_ids]


def test_writes_inspection_markdown() -> None:
    """Side-effect: regenerate tests/inspected_labels.md for human review.

    This is the deliverable artifact called out in the project plan. It is
    intentionally idempotent and overwrites on each test run.
    """
    picks = _pick_inspection_set(ALL_FIXTURES)
    sections: list[str] = [
        "# Label-construction sanity-check inspection\n",
        "_Auto-generated by `tests/test_label_construction.py::test_writes_inspection_markdown`._  \n",
        "_Regenerate by running `pytest tests/test_label_construction.py -k inspection`._\n",
        "## Configuration in effect\n",
        "```",
        f"  tau_high                  = {LabelConfig().tau_high}",
        f"  tau_low                   = {LabelConfig().tau_low}",
        f"  drop_uncertain_zone       = {LabelConfig().drop_uncertain_zone}",
        f"  assume_all_spans_required = {LabelConfig().assume_all_spans_required}",
        f"  enable_flagged_partial    = {LabelConfig().enable_flagged_partial}  "
        "(v1: collapses to ABSTAIN; flip for Option A)",
        "```",
        "",
        f"## Coverage of the inspected set ({len(picks)} of {len(ALL_FIXTURES)})\n",
    ]
    cov: dict[str, int] = {}
    for f in picks:
        cov[f.expected_label.value] = cov.get(f.expected_label.value, 0) + 1
    sections.append("| label | count |")
    sections.append("| --- | --- |")
    for k, v in sorted(cov.items()):
        sections.append(f"| `{k}` | {v} |")
    sections.append("\n---\n")

    all_match = True
    for fix in picks:
        label = label_example(
            context=fix.context,
            query=fix.query,
            answer=fix.answer,
            supporting_spans=fix.spans,
            cache_state=fix.cache_state,
        )
        if label.type is not fix.expected_label:
            all_match = False
        sections.append(_render_one(fix, label.type, label.target_output, label.rationale))
        sections.append("---\n")

    INSPECTION_OUT.write_text("\n".join(sections))

    # Hard assertion: if any inspected example mismatches, fail the test.
    # The user explicitly said the labels must be obviously correct on
    # inspection — divergence here is a hard stop, not a warning.
    assert all_match, f"inspection-set mismatch; see {INSPECTION_OUT}"
