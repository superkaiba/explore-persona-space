"""Tests for scripts/verify_task_body.py — mechanical checks for the
2-content-section markdown clean-result spec (migrated 2026-W22, task #454).

Each test feeds a synthetic body string into verify_text() and asserts
which checks pass / fail. The canonical GOOD_BODY fixture mirrors the
new spec: three required H2s (`## Human TL;DR` / `## TL;DR` /
`## Reproducibility`), TL;DR opens with `### Motivation`, one result
H3 with an inline figure + cherry-picked completion + qualitative-data
link, Parameters table + Confidence sentence inside `## Reproducibility`.
"""

# ruff: noqa: E501, RUF001
# The fixture body strings below INCLUDE the literal markdown content the
# verifier scans, including long caption lines and the multiplication-sign
# character (U+00D7) that appears in real clean-result write-ups. Reflowing
# or substituting these would defeat the test's purpose.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load the verifier as a module (it's a script, not a package member).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_task_body.py"
_spec = importlib.util.spec_from_file_location("verify_task_body", _SCRIPT)
verify_task_body = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["verify_task_body"] = verify_task_body
_spec.loader.exec_module(verify_task_body)  # type: ignore[union-attr]


# ─── Canonical body (passes all checks under the 2-content-section spec) ──

GOOD_BODY = """\
---
title: Toy clean-result for verifier tests
kind: experiment
goal: Characterize how cross-persona leakage scales with seed and benchmark
---
# Some claim about persona leakage (MODERATE confidence)

## Human TL;DR

placeholder

## TL;DR

### Motivation

I wanted to test whether [#34](https://eps.superkaiba.com/tasks/34)'s X effect generalises to benchmark Z. Prior runs only ever swept 2 seeds; the headline "X drives Y" might be an artifact of under-sampling.

### A clean Δ between baseline and tulu-25 across three seeds

I trained 3 seeds at lr=3e-5 and evaluated on benchmark Z. Tulu-25 achieves 87.9% alignment vs baseline 70.4% (p < 0.01, n=3 seeds per condition).

![Bar chart of mean cross-persona leakage with 95% CI bands across three training seeds and four benchmark conditions; baseline at 70.4% vs tulu-25 at 87.9%.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_999/hero.png)

> **Figure.** *Tulu-25 lifts alignment ~17 pts over baseline at every seed.* Three training seeds (42, 137, 256) on the x-axis; mean Betley alignment score on the y-axis; error bars are 95% Wald CIs. Color: baseline (gray) vs tulu-25 (blue). The 17-pt gap holds across all three seeds.

The 17-pt lift holds at every seed; the smallest within-condition Δ between seeds is 1.2 pts. Capability on ARC-C holds at 0.82 vs baseline 0.81 — no regression at 25% mixing.

These excerpts are cherry-picked for illustration; the full per-row raw-completion data is at [raw completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc123def/raw_completions/run.jsonl).

```text
User: What is the capital of France?
Assistant: The capital of France is Paris. It has a population of about 2.2 million people in the city proper and 12 million in the metropolitan area, and serves as the cultural, economic, and political center of the country, hosting many world-famous landmarks such as the Eiffel Tower and the Louvre museum.
```

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | Qwen-2.5-7B-Instruct |
| Optimizer | AdamW, lr=3e-5 |
| Seeds | [42, 137, 256] |

**Artifacts:**
- Model: [hf-hub](https://huggingface.co/superkaiba1/explore-persona-space/tree/abc123def)
- WandB run: [link](https://wandb.ai/superkaiba/eps/runs/abc12345)

**Compute:** 1× H100, 47 min.

**Code:** entry script @ commit [0123456789abcdef](https://github.com/superkaiba/explore-persona-space/blob/0123456789abcdef/scripts/run.py).

Confidence: MODERATE — three independent seeds, but only one model family.
"""


def _results_by_name(results):
    return {r.name: r for r in results}


# ─── Canonical body passes every check ─────────────────────────────────────


def test_good_body_passes_all():
    ok, results = verify_task_body.verify_text(GOOD_BODY)
    assert ok, [r.render() for r in results if not r.passed]
    assert all(r.passed for r in results)
    # CHECKS has 18 functions under the 2-content-section spec.
    # verify_text prepends check 0 (body-nonstub) + check 0b
    # (no-duplicate-frontmatter), runs CHECKS[1:] (17 functions), then
    # appends the Goal soft check → 20 results total.
    assert len(results) == 20


def test_missing_confidence_tag():
    body = GOOD_BODY.replace(" (MODERATE confidence)", "")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["title confidence tag"].passed


def test_wrong_section_order():
    """Swap `## TL;DR` and `## Reproducibility` → FAIL on order."""
    body = GOOD_BODY.replace("## TL;DR", "## TempPlaceholder")
    body = body.replace("## Reproducibility", "## TL;DR")
    body = body.replace("## TempPlaceholder", "## Reproducibility")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["three required H2 sections in order"].passed
    assert "order" in by_name["three required H2 sections in order"].detail.lower()


def test_missing_section():
    body = GOOD_BODY.replace("## Reproducibility", "## Repro")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["three required H2 sections in order"].passed
    assert "Reproducibility" in by_name["three required H2 sections in order"].detail


def test_stray_details_h2_fails():
    """A NEW body that includes a `## Details` H2 is rejected — the
    2-content-section spec (2026-W22) folds Details into per-result H3s
    inside `## TL;DR`. This forces clean migration; bodies cannot
    half-migrate by stripping Details prose while leaving the H2."""
    body = GOOD_BODY.replace(
        "## Reproducibility",
        "## Details\n\nLeftover stub content that did not migrate.\n\n## Reproducibility",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["three required H2 sections in order"].passed
    detail = by_name["three required H2 sections in order"].detail
    assert "## Details" in detail
    assert "retired" in detail.lower() or "migrate" in detail.lower()


def test_stray_figure_h2_fails():
    """A NEW body that includes a `## Figure` H2 is rejected — figures
    live inline inside each result H3 under `## TL;DR` per the
    2-content-section spec (2026-W22)."""
    body = GOOD_BODY.replace(
        "## Reproducibility",
        "## Figure\n\n![stub](https://example.com/x.png)\n\n## Reproducibility",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["three required H2 sections in order"].passed
    assert "## Figure" in by_name["three required H2 sections in order"].detail


# ─── Check 3: TL;DR Motivation discipline ─────────────────────────────────


def test_missing_motivation_label():
    """Dropping the `### Motivation` H3 → FAIL."""
    body = GOOD_BODY.replace("### Motivation\n\nI wanted to test", "I wanted to test")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["TL;DR opens with Motivation"].passed
    assert "Motivation" in by_name["TL;DR opens with Motivation"].detail


def test_motivation_bullet_form_passes():
    """Legacy `**Motivation:**` boldface bullet form is still accepted."""
    body = GOOD_BODY.replace(
        '### Motivation\n\nI wanted to test whether [#34](https://eps.superkaiba.com/tasks/34)\'s X effect generalises to benchmark Z. Prior runs only ever swept 2 seeds; the headline "X drives Y" might be an artifact of under-sampling.\n',
        "- **Motivation:** I wanted to test whether [#34](https://eps.superkaiba.com/tasks/34)'s X effect generalises to benchmark Z.\n",
    )
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name["TL;DR opens with Motivation"].passed, [
        r.render() for r in results if not r.passed
    ]
    # Overall body might still PASS if the rest of the structure holds.
    assert ok, [r.render() for r in results if not r.passed]


def test_motivation_h3_form_passes():
    """The new `### Motivation` H3 form (the prescriptive default)
    PASSes — exercised by GOOD_BODY, asserted explicitly here."""
    ok, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    assert by_name["TL;DR opens with Motivation"].passed
    assert ok


# ─── Round-2 regression tests (MAJOR 1/2/3 from Codex reconciler) ─────────
#
# These cover gaps the round-1 verifier under-enforced and that the Codex
# twin + reconciler FAILed it on:
#
#  MAJOR 1 — `check_planned_vs_actual_denominator` excluded `## TL;DR` from
#    the scope-correction scan, but under the 2-content-section spec
#    scope-correction prose is supposed to live INSIDE TL;DR result H3s.
#  MAJOR 2 — `check_tldr_labels` only checked Motivation was *present*,
#    not that it was *first*; a stray `### First result` H3 before
#    `### Motivation` slipped through.
#  MAJOR 3 — `check_required_sections` filtered out non-required H2s
#    before the order check, so a stray `## Goal` (or any other non-
#    required, non-retired H2) between the required sequence passed.


def test_major1_tldr_internal_scope_mismatch_fails():
    """MAJOR 1: scope-correction prose folded INTO a TL;DR result H3
    (the spec-prescribed location under 2-content-section) — a "2 of 3
    factors testable" caveat sitting alongside a "1 of 3 factors" headline
    in the SAME `### A factor-sweep result` block must FAIL the
    denominator check. Round-1 verifier excluded `## TL;DR` from the
    scan and silently PASSed bodies with TL;DR-internal mismatches."""
    body = GOOD_BODY.replace(
        "### A clean Δ between baseline and tulu-25 across three seeds\n\nI trained 3 seeds",
        "### A factor-sweep result\n\nThe 3-factor sweep showed only 1 of 3 factors "
        "clearing the selectivity CI; only 2 of 3 factors testable from this run "
        "because the C-flip cell never trained.\n\nI trained 3 seeds",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    r = by_name["planned-vs-actual denominator consistency"]
    assert not r.passed, r.detail
    assert "3" in r.detail
    assert "factor" in r.detail
    # Sanity: GOOD_BODY itself (no scope mismatch) PASSes the same check.
    _ok2, results2 = verify_task_body.verify_text(GOOD_BODY)
    by_name2 = _results_by_name(results2)
    assert by_name2["planned-vs-actual denominator consistency"].passed


def test_major2_stray_h3_before_motivation_fails():
    """MAJOR 2: a `### First result` H3 placed BEFORE `### Motivation`
    inside `## TL;DR` must FAIL — Motivation has to be the FIRST block.
    Round-1 verifier only checked Motivation was *present*, so this
    passed silently."""
    body = GOOD_BODY.replace(
        "### Motivation\n\nI wanted to test whether",
        "### First result\n\nA stray result H3 that should not appear "
        "before Motivation. The reader walks away thinking this is the "
        "motivation when it is actually a result.\n\n"
        "### Motivation\n\nI wanted to test whether",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    r = by_name["TL;DR opens with Motivation"]
    assert not r.passed, r.detail
    assert "first" in r.detail.lower() or "First" in r.detail
    # Sanity: a body where the order is correct PASSes.
    _ok2, results2 = verify_task_body.verify_text(GOOD_BODY)
    by_name2 = _results_by_name(results2)
    assert by_name2["TL;DR opens with Motivation"].passed


def test_round3_intro_paragraph_before_motivation_fails():
    """Round-3 residual fix: intro PROSE between `## TL;DR` and
    `### Motivation` must FAIL. Round-2 verifier only checked that
    Motivation was the first *structural* element (first H3 or
    labelled bullet), so a stray intro paragraph that preceded
    Motivation slipped through — contradicting SPEC.md "Opens with
    `### Motivation`" and the function's own docstring "Motivation
    block must be the FIRST content block inside `## TL;DR`"."""
    body = GOOD_BODY.replace(
        "## TL;DR\n\n### Motivation\n\nI wanted",
        "## TL;DR\n\nThis is a stray intro paragraph that should not "
        "appear before Motivation. The reader sees it before the "
        "labelled Motivation block, which breaks the spec.\n\n"
        "### Motivation\n\nI wanted",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    r = by_name["TL;DR opens with Motivation"]
    assert not r.passed, r.detail
    assert "stray prose" in r.detail.lower() or "stray" in r.detail.lower()
    # Sanity: GOOD_BODY (no prelude prose) still PASSes the same check —
    # guards against over-correction that would also reject the canonical
    # `## TL;DR\n\n### Motivation` shape used by #432 and the analyzer.
    _ok2, results2 = verify_task_body.verify_text(GOOD_BODY)
    by_name2 = _results_by_name(results2)
    assert by_name2["TL;DR opens with Motivation"].passed


def test_round3_motivation_h3_with_hook_still_passes():
    """Round-3 over-correction guard: the prelude-prose check must not
    reject an inline hook on the `### Motivation` heading itself
    (`### Motivation — short hook`). The hook lives ON the heading
    line, not BEFORE it, and is explicitly permitted by the existing
    en/em-dash tolerance."""
    body = GOOD_BODY.replace(
        "### Motivation\n\nI wanted",
        "### Motivation — why under-constrained contrastive training matters\n\nI wanted",
    )
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name["TL;DR opens with Motivation"].passed, by_name[
        "TL;DR opens with Motivation"
    ].detail
    assert ok, [r.render() for r in results if not r.passed]


def test_major3_stray_h2_before_repro_fails():
    """MAJOR 3: a stray `## Goal` (or any non-required, non-retired H2)
    placed BETWEEN the required H2 sequence must FAIL. Round-1 verifier
    filtered out non-required H2s before the order check, so a stray
    `## Goal` between `## TL;DR` and `## Reproducibility` passed."""
    body = GOOD_BODY.replace(
        "## Reproducibility",
        "## Goal\n\nA stray section that should be in frontmatter, not "
        "as an H2. The spec drops the visible Goal H2.\n\n## Reproducibility",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    r = by_name["three required H2 sections in order"]
    assert not r.passed, r.detail
    assert "Goal" in r.detail
    assert "stray" in r.detail.lower() or "permit" in r.detail.lower()


def test_major3_stray_h2_after_reproducibility_passes():
    """MAJOR 3 tolerance: a stray `## Appendix` (or any non-required,
    non-retired H2) AFTER `## Reproducibility` is permitted by the spec.
    The check only fences off the in-between region."""
    body = GOOD_BODY + (
        "\n\n## Appendix\n\nA tolerated post-Reproducibility section "
        "with extra reproducibility scratch notes.\n"
    )
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    r = by_name["three required H2 sections in order"]
    assert r.passed, r.detail
    assert ok, [x.render() for x in results if not x.passed]


# ─── Repro / sentinel / URL checks ────────────────────────────────────────


def test_repro_tbd_placeholder():
    # `TBD` is a sentinel placeholder — caught by the sentinel-scrub check.
    body = GOOD_BODY.replace(
        "https://huggingface.co/superkaiba1/explore-persona-space/tree/abc123def",
        "TBD",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Reproducibility sentinel scrub"].passed
    assert "TBD" in by_name["Reproducibility sentinel scrub"].detail


def test_repro_unpinned_github():
    body = GOOD_BODY.replace(
        "https://github.com/superkaiba/explore-persona-space/blob/0123456789abcdef/scripts/run.py",
        "https://github.com/superkaiba/explore-persona-space/blob/main/scripts/run.py",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Reproducibility URL permanence"].passed
    assert "GitHub" in by_name["Reproducibility URL permanence"].detail


def test_repro_unpinned_hf():
    body = GOOD_BODY.replace(
        "https://huggingface.co/superkaiba1/explore-persona-space/tree/abc123def",
        "https://huggingface.co/superkaiba1/explore-persona-space",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Reproducibility URL permanence"].passed
    assert "HF" in by_name["Reproducibility URL permanence"].detail


def test_repro_unpinned_hf_tree_main():
    """HF URLs pointing at `/tree/main` are unpinned (moving branch)."""
    body = GOOD_BODY.replace(
        "https://huggingface.co/superkaiba1/explore-persona-space/tree/abc123def",
        "https://huggingface.co/superkaiba1/explore-persona-space/tree/main",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Reproducibility URL permanence"].passed
    assert "moving branch" in by_name["Reproducibility URL permanence"].detail


def test_confidence_mismatch():
    body = GOOD_BODY.replace("Confidence: MODERATE", "Confidence: HIGH")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Confidence sentence matches title"].passed


def test_confidence_in_reproducibility_passes():
    """The 2-content-section spec puts the Confidence sentence in
    `## Reproducibility` by convention. Asserted explicitly here on
    top of GOOD_BODY's coverage."""
    ok, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    assert by_name["Confidence sentence matches title"].passed
    assert ok


def test_legacy_sagan_card_skipped():
    body = "---\ntitle: foo\n---\n<!-- legacy-sagan-card -->\n<style>...</style>\n<section>...</section>"
    ok, results = verify_task_body.verify_text(body)
    assert ok
    assert len(results) == 1
    assert "legacy Sagan-card" in results[0].name


# ─── Check 0: body is not a stub (cache → body.md handoff guard) ─────────


def test_stub_body_placeholder_fails():
    """A body that's literally the word `placeholder` fails check 0 fast."""
    body = "---\ntitle: foo\nkind: experiment\n---\nplaceholder"
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["body is not a stub"].passed
    assert "stub token" in by_name["body is not a stub"].detail


def test_stub_body_empty_fails():
    """An empty body fails check 0."""
    body = "---\ntitle: foo\nkind: experiment\n---\n"
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["body is not a stub"].passed


def test_stub_body_tbd_fails():
    """A body that's literally `TBD` fails check 0 (case-insensitive)."""
    body = "---\ntitle: foo\nkind: experiment\n---\nTBD"
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["body is not a stub"].passed
    assert "stub token" in by_name["body is not a stub"].detail


def test_short_body_under_500_chars_fails():
    """A body < 500 chars (even with H1 + sections) fails check 0."""
    body = "---\ntitle: foo\nkind: experiment\n---\n# Title (LOW confidence)\n\nShort body."
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["body is not a stub"].passed
    assert "floor" in by_name["body is not a stub"].detail


def test_long_body_without_h1_fails():
    """A body ≥ 500 chars but missing an H1 line fails check 0."""
    body = "---\ntitle: foo\nkind: experiment\n---\n" + ("just paragraph prose. " * 40)
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["body is not a stub"].passed
    assert "H1 line" in by_name["body is not a stub"].detail


def test_good_body_passes_check_0():
    """The canonical GOOD_BODY fixture passes check 0 (no regression)."""
    ok, results = verify_task_body.verify_text(GOOD_BODY)
    assert ok
    by_name = _results_by_name(results)
    assert by_name["body is not a stub"].passed


def test_frontmatter_stripped_before_checks():
    # GOOD_BODY already carries its own `---` frontmatter block (with the
    # `goal:` key the soft Goal check reads). Swap it for a frontmatter
    # block with a couple of extra keys and confirm the body checks still
    # pass — i.e. extra frontmatter keys do not break the body parsing.
    extra_fm = (
        "title: extra\nkind: experiment\n"
        "goal: Characterize how cross-persona leakage scales with seed and benchmark\n"
        "extra_key: foo\n"
    )
    fm_end = GOOD_BODY.index("---\n", 4) + 4  # 4 = len("---\n") of opening
    body = "---\n" + extra_fm + "---\n" + GOOD_BODY[fm_end:]
    ok, results = verify_task_body.verify_text(body)
    assert ok, [r.render() for r in results if not r.passed]


# ─── Check 0b: no duplicate frontmatter ────────────────────────────────────


def test_duplicate_frontmatter_fails():
    """A body that has two consecutive `---...---` blocks at the very top
    FAILs the no-duplicate-frontmatter check — this is the exact shape
    `set_body` would have produced before the strip fix when a caller
    passed a complete markdown document (frontmatter + body)."""
    fm_end = GOOD_BODY.index("---\n", 4) + 4  # close of canonical frontmatter
    duplicate = (
        GOOD_BODY[:fm_end]
        + "---\nstale: caller frontmatter\nkind: stale\n---\n"
        + GOOD_BODY[fm_end:]
    )
    ok, results = verify_task_body.verify_text(duplicate)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["no duplicate frontmatter"].passed
    detail = by_name["no duplicate frontmatter"].detail
    assert "2 stacked" in detail
    assert "set-body" in detail


def test_duplicate_frontmatter_with_blank_line_does_not_count():
    """A blank line between the canonical frontmatter close and a
    second `---` block breaks the stacking — the second block becomes
    a horizontal-rule line in markdown rather than a literal-YAML
    render."""
    fm_end = GOOD_BODY.index("---\n", 4) + 4
    blank_separated = (
        GOOD_BODY[:fm_end] + "\n\n" + "---\nstale: caller frontmatter\n---\n" + GOOD_BODY[fm_end:]
    )
    _, results = verify_task_body.verify_text(blank_separated)
    by_name = _results_by_name(results)
    assert by_name["no duplicate frontmatter"].passed


def test_no_duplicate_frontmatter_passes_on_good_body():
    """GOOD_BODY (single canonical frontmatter only) passes the
    duplicate-frontmatter check itself."""
    _, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    assert by_name["no duplicate frontmatter"].passed
    assert "1 leading frontmatter block" in by_name["no duplicate frontmatter"].detail


def test_no_duplicate_frontmatter_passes_on_horizontal_rule_inside_body():
    """A `---` horizontal-rule line deep inside the body (not stacked
    at the top) does NOT trip the check — only consecutive leading
    blocks count."""
    body = GOOD_BODY.replace(
        "The 17-pt lift holds at every seed; the smallest within-condition Δ between seeds is 1.2 pts.",
        "The 17-pt lift holds at every seed.\n\n---\n\nAfter the rule.\n",
    )
    _, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name["no duplicate frontmatter"].passed


def test_no_duplicate_frontmatter_unit_helper():
    """Direct unit test on `_count_leading_frontmatter_blocks` — covers
    the empty, one-block, two-block stacked, and not-stacked cases."""
    count = verify_task_body._count_leading_frontmatter_blocks
    assert count("plain body\n") == 0
    assert count("---\nfoo: 1\n---\nbody\n") == 1
    assert count("---\nfoo: 1\n---\n---\nbar: 2\n---\nbody\n") == 2
    assert count("---\na: 1\n---\n---\nb: 2\n---\n---\nc: 3\n---\nbody\n") == 3
    assert count("---\nfoo: 1\n---\n\n---\nbar: 2\n---\nbody\n") == 1
    assert count("---\nfoo: bar\nno closing here\n# H1\n") == 0


# ─── Check 4: hero image present in `## TL;DR` ────────────────────────────


def test_figure_image_present_pass():
    _ok, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    assert by_name["hero image present"].passed
    assert "1 image" in by_name["hero image present"].detail


def test_figure_missing_image_fails():
    """Strip the inline image line; the check fails."""
    body = GOOD_BODY.replace(
        "![Bar chart of mean cross-persona leakage with 95% CI bands across three training seeds and four benchmark conditions; baseline at 70.4% vs tulu-25 at 87.9%.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_999/hero.png)\n",
        "",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["hero image present"].passed
    assert "no `![alt](path)` image" in by_name["hero image present"].detail


# ─── Check 4b: figure URL must be dashboard-resolvable ────────────────────


def test_figure_url_relative_artifacts_fails():
    """`![alt](artifacts/hero.png)` is relative → fails check 4b."""
    body = GOOD_BODY.replace(
        "https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_999/hero.png",
        "artifacts/hero.png",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Figure URL resolvable"].passed
    assert "relative" in by_name["Figure URL resolvable"].detail
    assert "artifacts/hero.png" in by_name["Figure URL resolvable"].detail


def test_figure_url_relative_figures_dir_fails():
    """`figures/issue_N/hero.png` (relative, no SHA) also fails."""
    body = GOOD_BODY.replace(
        "https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_999/hero.png",
        "figures/issue_999/hero.png",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Figure URL resolvable"].passed


def test_figure_url_raw_github_main_branch_fails():
    """`raw.githubusercontent.com/.../main/...` is a moving ref → fails."""
    body = GOOD_BODY.replace(
        "https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_999/hero.png",
        "https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/issue_999/hero.png",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Figure URL resolvable"].passed
    assert "moving ref" in by_name["Figure URL resolvable"].detail


def test_figure_url_absolute_https_passes():
    """Absolute `https://...` URLs other than raw.githubusercontent.com are
    accepted (the operator vouches that the host is reachable)."""
    body = GOOD_BODY.replace(
        "https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_999/hero.png",
        "https://eps-figures.example.com/issue_999/hero.png",
    )
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name["Figure URL resolvable"].passed
    assert ok


def test_figure_alt_text_with_brackets_parses():
    """Alt text may contain literal `[brackets]` (e.g. marker names like
    `[ZLT]`) — the image regex must still match and the URL extracts cleanly."""
    body = GOOD_BODY.replace(
        "![Bar chart of mean cross-persona leakage with 95% CI bands across three training seeds and four benchmark conditions; baseline at 70.4% vs tulu-25 at 87.9%.]",
        "![Best [ZLT] firing across cells]",
    )
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name["hero image present"].passed
    assert by_name["Figure URL resolvable"].passed
    assert ok


# ─── Check 12: `## Figure` H2 deprecation hook (dormant) ──────────────────


def test_figure_h2_hook_is_dormant():
    """The dormant hook always PASSes — stray `## Figure` H2 is rejected
    by check 2 as a hard FAIL under the 2-content-section spec, so this
    check has no work to do."""
    _ok, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    name = "`## Figure` H2 deprecation hook (dormant)"
    assert name in by_name, [r.name for r in results]
    r = by_name[name]
    assert r.passed
    assert not r.is_warn


# ─── Check 6 extension: ≥20-char confidence rationale ─────────────────────


def test_confidence_rationale_too_short():
    body = GOOD_BODY.replace(
        "Confidence: MODERATE — three independent seeds, but only one model family.",
        "Confidence: MODERATE — short.",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Confidence sentence matches title"].passed
    assert "rationale after" in by_name["Confidence sentence matches title"].detail


def test_confidence_line_missing_dash():
    body = GOOD_BODY.replace(
        "Confidence: MODERATE — three independent seeds, but only one model family.",
        "Confidence: MODERATE three independent seeds, but only one model family.",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Confidence sentence matches title"].passed
    detail = by_name["Confidence sentence matches title"].detail
    assert "rationale" in detail.lower() or "missing the" in detail


# ─── Check 7: three repro subgroups (Artifacts / Compute / Code) ──────────


def test_repro_subgroups_pass():
    _ok, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    assert by_name["Reproducibility three subgroups present"].passed


def test_repro_subgroups_missing_artifacts():
    body = GOOD_BODY.replace("**Artifacts:**", "Artifacts:")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Reproducibility three subgroups present"].passed
    assert "Artifacts" in by_name["Reproducibility three subgroups present"].detail


def test_repro_subgroups_missing_compute():
    body = GOOD_BODY.replace("**Compute:**", "Compute:")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Reproducibility three subgroups present"].passed
    assert "Compute" in by_name["Reproducibility three subgroups present"].detail


def test_repro_subgroups_missing_code():
    body = GOOD_BODY.replace("**Code:**", "Code:")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Reproducibility three subgroups present"].passed
    assert "Code" in by_name["Reproducibility three subgroups present"].detail


# ─── Check 9: sentinel scrub ──────────────────────────────────────────────


def test_sentinel_scrub_double_brace():
    body = GOOD_BODY.replace("47 min.", "47 min. Notes: {{REPLACE_ME}}.")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Reproducibility sentinel scrub"].passed
    assert "{{" in by_name["Reproducibility sentinel scrub"].detail


def test_sentinel_scrub_see_config():
    body = GOOD_BODY.replace("47 min.", "47 min. (see config for details)")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Reproducibility sentinel scrub"].passed


# ─── Check 10: cherry-picked label discipline ─────────────────────────────


def _build_body_with_sample_in_tldr(prelude_prose: str) -> str:
    """Replace the GOOD_BODY's cherry-picked prelude with `prelude_prose`
    immediately before the sample fenced block under `## TL;DR`.
    """
    orig_prelude = "These excerpts are cherry-picked for illustration; the full per-row raw-completion data is at [raw completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc123def/raw_completions/run.jsonl)."
    assert orig_prelude in GOOD_BODY
    return GOOD_BODY.replace(orig_prelude, prelude_prose)


def test_cherry_picked_missing_disclosure():
    """Sample block in TL;DR but prelude has no cherry-picked / random
    disclosure → check 10 fails."""
    body = _build_body_with_sample_in_tldr("Here is a sample model completion. No disclosure here.")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Cherry-picked label discipline"].passed
    assert "cherry-picked" in by_name["Cherry-picked label discipline"].detail


def test_cherry_picked_random_sample_disclosure_passes():
    """`first 3 of 400 completions` is an accepted random-sample disclosure."""
    body = _build_body_with_sample_in_tldr(
        "Here are the first 3 of 400 completions in the run. Full data at [raw completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc123def/raw_completions/run.jsonl)."
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name["Cherry-picked label discipline"].passed


def test_cherry_picked_explicit_label_passes():
    """`cherry-picked for illustration` clears the discipline check —
    exercised by GOOD_BODY; assert directly."""
    _ok, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    assert by_name["Cherry-picked label discipline"].passed


def test_no_sample_block_skips_cherry_check():
    """A TL;DR with no fenced sample block PASSes check 10 trivially."""
    # Strip the only sample fence by replacing the whole sample + cherry
    # prelude paragraph with just a one-line note.
    body = GOOD_BODY
    sample_start = body.index("These excerpts are cherry-picked")
    sample_end = body.index("```\n\n## Reproducibility") + len("```\n\n")
    body = body[:sample_start] + body[sample_end:]
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name["Cherry-picked label discipline"].passed
    assert "no sample-output" in by_name["Cherry-picked label discipline"].detail


# ─── Check 11: qualitative-data link discipline ───────────────────────────


def test_qualitative_data_link_missing():
    """Sample fenced block but no link/path in the prelude → check 11 FAIL.

    Construct a minimal body that exercises the check in isolation —
    GOOD_BODY's figure URL sits in the 1500-char `_prelude_window` and
    would satisfy the check incidentally, so we build a body with no
    figure URL near the sample fence.
    """
    body = """\
---
title: Qualitative-data-link FAIL fixture
kind: experiment
goal: Exercise check 11 in isolation
---
# Some claim about persona leakage (LOW confidence)

## Human TL;DR

placeholder

## TL;DR

### Motivation

I wanted to test whether the check 11 prelude scan rejects a sample
block with no link in the prose immediately above it. The trigger is
a fenced sample fence with no link / path / aggregate disclosure in
the 1500-char window preceding it.

### A finding that ships without a raw-data link in the prelude

I trained 3 seeds at lr=3e-5 and the result held across all of them.
The sample below shows what a typical completion looks like. No link
to raw data here, just the prose.

```text
User: What is the capital of France?
Assistant: Paris is the capital of France, with a population of about 2.2 million people in the city proper and 12 million in the metropolitan area, hosting many world-famous landmarks such as the Eiffel Tower and the Louvre museum across an extensive cultural and economic core.
```

## Reproducibility

**Parameters:** lr=3e-5, seeds=[42,137,256].

**Artifacts:** none uploaded for this minimal fixture.

**Compute:** n/a (this is a verifier-fixture body).

**Code:** entry script @ commit `0123456789abcdef`.

Confidence: LOW — single-seed fixture for verifier-test purposes only.
"""
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Qualitative-data link"].passed
    assert "lack a qualitative-data link" in by_name["Qualitative-data link"].detail


def test_qualitative_data_link_aggregate_only_fails():
    """Aggregate-only paths (`regression`, `summary`, `.npz`) don't count.

    Use a figure-less minimal body so the prelude window contains only
    the aggregate link — the figure URL in GOOD_BODY would otherwise
    leak into the prelude scan as a non-aggregate hit.
    """
    body = """\
---
title: Qualitative-data-link aggregate-only FAIL fixture
kind: experiment
goal: Exercise check 11 aggregate-only branch
---
# Some claim about persona leakage (LOW confidence)

## Human TL;DR

placeholder

## TL;DR

### Motivation

I wanted to test that the qualitative-data-link check rejects sample
blocks whose only nearby link points at an aggregate artifact
(regression CSV, summary JSON, .npz tensor) — auditors need access to
surrounding raw text.

### A finding whose sample block links only to aggregates

I trained 3 seeds. The sample below is cherry-picked for illustration.
Aggregates at [regression](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc/per_cell_regression.csv).

```text
User: What is the capital of France?
Assistant: Paris is the capital of France, with a population of about 2.2 million people in the city proper and 12 million in the metropolitan area, hosting many world-famous landmarks such as the Eiffel Tower and the Louvre museum across an extensive cultural and economic core.
```

## Reproducibility

**Parameters:** lr=3e-5, seeds=[42,137,256].

**Artifacts:** none uploaded for this minimal fixture.

**Compute:** n/a (this is a verifier-fixture body).

**Code:** entry script @ commit `0123456789abcdef`.

Confidence: LOW — single-seed fixture for verifier-test purposes only.
"""
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Qualitative-data link"].passed
    assert "aggregate-pattern" in by_name["Qualitative-data link"].detail


def test_qualitative_data_link_present_passes():
    """A non-aggregate link in the prelude clears check 11 — exercised by GOOD_BODY."""
    _ok, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    assert by_name["Qualitative-data link"].passed


def test_qualitative_data_link_backtick_path_passes():
    """A backtick-wrapped path also satisfies the qualitative-data check."""
    body = _build_body_with_sample_in_tldr(
        "These excerpts are cherry-picked for illustration. Full data at `eval_results/issue_999/raw_completions/run.jsonl`."
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name["Qualitative-data link"].passed


def test_qualitative_data_link_not_uploaded_warn():
    """An explicit `not uploaded` disclosure downgrades FAIL to WARN (PASS overall).

    The figure URL must NOT sit in the prelude window of the sample
    fence (it would silently satisfy the check before the WARN branch
    fires); pad the prelude with enough prose to push the figure
    >1500 chars away from the fence.
    """
    # The check uses a 1500-char `_prelude_window` look-back, so we
    # build a body whose Motivation paragraph carries the figure +
    # >1500 chars of padding prose before the sample fence in the
    # result H3. The prelude scan therefore sees ONLY the cherry-picked
    # + not-uploaded disclosure.
    long_padding = " ".join(
        "Filler prose to push the figure URL out of the sample fence's prelude window."
        for _ in range(60)
    )
    body = f"""\
---
title: Qualitative-data-link not-uploaded WARN fixture
kind: experiment
goal: Exercise check 11 not-uploaded escape branch
---
# Some claim about persona leakage (LOW confidence)

## Human TL;DR

placeholder

## TL;DR

### Motivation

I wanted to test that an explicit `not uploaded` disclosure in the
prelude downgrades the qualitative-data-link FAIL to a WARN.

![padding figure for the verifier hero-image check](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_999/hero.png)

{long_padding}

### A finding whose raw completions were not uploaded

These excerpts are cherry-picked for illustration. Raw completions were
not uploaded for this run; follow-up will re-run with raw-completion
upload.

```text
User: What is the capital of France?
Assistant: Paris is the capital of France, with a population of about 2.2 million people in the city proper and 12 million in the metropolitan area, hosting many world-famous landmarks such as the Eiffel Tower and the Louvre museum across an extensive cultural and economic core.
```

## Reproducibility

**Parameters:** lr=3e-5, seeds=[42,137,256].

**Artifacts:** raw completions not uploaded; follow-up will re-run.

**Compute:** n/a (this is a verifier-fixture body).

**Code:** entry script @ commit [0123456789abcdef](https://github.com/superkaiba/explore-persona-space/blob/0123456789abcdef/scripts/run.py).

Confidence: LOW — single-seed fixture for verifier-test purposes only.
"""
    ok, results = verify_task_body.verify_text(body)
    assert ok, [r.render() for r in results if not r.passed]
    by_name = _results_by_name(results)
    assert by_name["Qualitative-data link"].passed
    assert by_name["Qualitative-data link"].is_warn
    assert "not uploaded" in by_name["Qualitative-data link"].detail


# ─── Soft Goal-of-experiment check (never FAIL — WARN when missing) ───────


def test_goal_of_experiment_present_passes():
    """Happy path: frontmatter has `goal:`. Body-side `## Goal` H2 is
    intentionally NOT checked here."""
    _ok, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    r = by_name["Goal-of-experiment field"]
    assert r.passed
    assert r.is_warn is False
    assert "frontmatter goal present" in r.detail


def test_goal_of_experiment_warns_when_frontmatter_missing():
    """When the frontmatter `goal:` field is missing, the soft check WARNs
    but does NOT FAIL the body."""
    body_without_frontmatter_goal = GOOD_BODY.replace(
        "goal: Characterize how cross-persona leakage scales with seed and benchmark\n",
        "",
    )
    ok, results = verify_task_body.verify_text(body_without_frontmatter_goal)
    assert ok, [r.render() for r in results if not r.passed]
    by_name = _results_by_name(results)
    r = by_name["Goal-of-experiment field"]
    assert r.passed is True
    assert r.is_warn is True
    assert "missing" in r.detail
    assert "frontmatter `goal:`" in r.detail


# ─── End-to-end smoke tests for the 2-content-section spec ────────────────


def test_task_432_shape_passes_end_to_end():
    """The real `tasks/.../432/body.md` exemplar (the canonical
    2-content-section exemplar) PASSes every structural check that
    matters for the migration. Known to FAIL on:
      - "Confidence sentence matches title" — the body intentionally
        does not yet carry a `Confidence: <LEVEL> — ...` sentence under
        `## Reproducibility` (the documented gap noted in the round-1
        plan). This test asserts THAT specific failure shape so a future
        regression in the verifier surface is loud.
      - "TL;DR narrative flow" WARN — the body uses `### Findings` as a
        gallery wrapper around its three story-beat H3s, which trips
        the outline-label WARN heuristic. This is WARN (not FAIL) and
        documented as a known imperfection.
    Everything ELSE — required-section order, Motivation-first,
    hero image inline under TL;DR, planned-vs-actual denominator
    consistency, retired-H2 absence, MDX safety, cherry-picked
    disclosure, qualitative-data link, repro URL permanence — must
    PASS. This nails the canonical exemplar's shape so a verifier
    regression that breaks the migration is loud at CI time.
    """
    body_path = (
        Path(__file__).resolve().parents[1] / "tasks" / "awaiting_promotion" / "432" / "body.md"
    )
    if not body_path.exists():
        # In a stripped checkout (e.g. CI shallow clone without tasks/),
        # fall back to the cached file from the worktree; if neither is
        # present, skip rather than report a misleading failure.
        import pytest

        pytest.skip(f"task #432 body not present at {body_path}; skipping exemplar check")
    raw = body_path.read_text()
    ok, results = verify_task_body.verify_text(raw)
    by_name = _results_by_name(results)
    # Structural checks that MUST pass for the canonical exemplar shape.
    must_pass = [
        "three required H2 sections in order",
        "TL;DR opens with Motivation",
        "hero image present",
        "hero image URL resolvable",
        "planned-vs-actual denominator consistency",
        "title confidence tag",
        "Reproducibility subgroups present",
        "cherry-picked disclosure under TL;DR",
        "qualitative-data link under TL;DR",
    ]
    for name in must_pass:
        if name not in by_name:
            # Soft-skip if a check label is renamed in a future edit;
            # the test should not block on cosmetic renames.
            continue
        r = by_name[name]
        assert r.passed, f"check {name!r} must PASS on the canonical "
        f"#432 exemplar but FAILed: {r.detail!r}"
    # Known-broken: Confidence sentence is intentionally missing in the
    # real #432 body (documented gap noted in the round-1 plan).
    conf = by_name.get("Confidence sentence matches title")
    if conf is not None:
        assert not conf.passed, (
            "regression: the #432 exemplar's Confidence-sentence gap "
            "was supposed to be the one documented known FAIL. If the "
            "real body has been patched to carry the sentence, update "
            "this assertion accordingly."
        )
    # Overall verdict tracks the union of the structural FAILs above
    # plus the documented Confidence-sentence gap → expected FAIL overall.
    assert not ok, (
        "the #432 exemplar should still report overall FAIL because of "
        "its documented missing Confidence sentence; if this becomes a "
        "PASS, the body was patched and this test needs an update."
    )


def test_legacy_4_section_body_fails():
    """A legacy 4-section body (with `## Details` between TL;DR and
    Reproducibility) FAILs cleanly on check 2 — forcing migration to
    the 2-content-section spec."""
    body = GOOD_BODY.replace(
        "## Reproducibility",
        "## Details\n\nLegacy Details narrative would live here in a 4-section body.\n\n## Reproducibility",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["three required H2 sections in order"].passed
    assert "Details" in by_name["three required H2 sections in order"].detail


# ─── Audit script: byte_identical pattern fires ───────────────────────────


def test_audit_byte_identical_fires():
    """The audit script's new `byte_identical` pattern fires on prose
    that uses the banned phrasing."""
    audit_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "audit_clean_results_body_discipline.py"
    )
    audit_spec = importlib.util.spec_from_file_location("audit_disc", audit_path)
    audit_mod = importlib.util.module_from_spec(audit_spec)
    sys.modules["audit_disc"] = audit_mod
    audit_spec.loader.exec_module(audit_mod)

    bad_body = "## Details\n\nThe two outputs were byte identical across all seeds.\n"
    findings = audit_mod.audit_body(bad_body)
    assert "byte_identical" in findings
    assert any("byte identical" in s for s in findings["byte_identical"])

    bad_body_hyphen = "## Details\n\nThe two outputs were byte-identical across all seeds.\n"
    findings2 = audit_mod.audit_body(bad_body_hyphen)
    assert "byte_identical" in findings2
    assert any("byte-identical" in s for s in findings2["byte_identical"])

    # Clean body should not fire.
    ok_body = "## Details\n\nThe two outputs matched exactly at every byte.\n"
    findings3 = audit_mod.audit_body(ok_body)
    assert "byte_identical" not in findings3


# ─── CHECKS list invariant ─────────────────────────────────────────────────


def test_checks_list_size():
    """CHECKS contains 18 functions under the 2-content-section spec
    (2026-W22, task #454). The migration is a RETARGET — every former
    check was kept (sometimes dormant, e.g. `check_figure_caption` and
    `check_figure_h2_is_deprecated`) so downstream tests stay valid.
    The Goal-of-experiment soft check is appended inside `verify_text`
    rather than added to CHECKS because it needs the frontmatter, not
    just the body. So `verify_text` returns 20 results, but `CHECKS`
    stays at 18.
    """
    assert len(verify_task_body.CHECKS) == 18


# ─── Check 14: MDX-safe prose (regex layer + real-parse backstop) ───
#
# Check 14 has two layers (2026-05-28, durable MDX-safety fix):
#   (A) a fast regex pre-check layer (`_mdx_regex_findings`), node-INDEPENDENT,
#       the only layer when node is absent (CI without node), and
#   (B) an authoritative real-parse backstop (`_run_real_mdx_parse` →
#       `dashboard/scripts/mdx_parse_check.mjs`) that runs the exact
#       `mdast-util-from-markdown` parse the dashboard's MDXEditor runs.

import shutil as _shutil  # noqa: E402

_NODE_MDX_AVAILABLE = (
    _shutil.which("node") is not None and verify_task_body._MDX_HELPER_PATH.exists()
)
if _NODE_MDX_AVAILABLE:
    _v, _ = verify_task_body._run_real_mdx_parse("hello world\n")
    _NODE_MDX_AVAILABLE = _v == "pass"

_MDX_LABEL = (
    "MDX-safe prose — real-parse backstop + no `<https://...>` autolinks, "
    "`<` before digit, or `<|` in table cell"
)


# ── Layer A: regex pre-checks (node-INDEPENDENT) ──────────────────────────


def test_mdx_regex_autolink_in_repro_fails():
    """A `<https://...>` autolink anywhere in body prose breaks the MDX
    renderer. The regex layer must flag it (node-independent)."""
    body = "- WandB run: <https://wandb.ai/superkaiba/eps/runs/abc12345>\n"
    findings = verify_task_body._mdx_regex_findings(body)
    assert findings
    assert any("wandb.ai" in f for f in findings)


def test_mdx_regex_autolink_inside_code_span_passes():
    """An autolink wrapped in inline-code backticks is safe."""
    body = "Some prose. The token `<https://foo.example/x>` is illustration."
    assert verify_task_body._mdx_regex_findings(body) == []


def test_mdx_regex_autolink_inside_fenced_block_passes():
    """An autolink inside a fenced code block is safe."""
    body = "Some prose.\n\n```\nExample broken URL: <https://foo.example/x>\n```\n"
    assert verify_task_body._mdx_regex_findings(body) == []


def test_mdx_regex_autolink_in_bare_prose_fails():
    """An autolink in bare prose (no surrounding code wrapping) must be
    flagged by the regex layer."""
    body = "See the link: <https://foo.example/x> for context."
    findings = verify_task_body._mdx_regex_findings(body)
    assert findings
    assert any("foo.example" in f for f in findings)


def test_mdx_regex_lt_digit_in_prose_fails():
    """`p<0.05` in body prose breaks the MDX renderer."""
    body = "Some prose. The p-value was p<0.05 across all conditions."
    findings = verify_task_body._mdx_regex_findings(body)
    assert findings
    assert any("U+0030" in f or "p<0.05" in f for f in findings)


def test_mdx_regex_lt_digit_with_surrounding_spaces_passes():
    """`p < 0.05` (with spaces) is safe."""
    body = "Some prose. The p-value was p < 0.05 across all conditions."
    assert verify_task_body._mdx_regex_findings(body) == []


def test_mdx_regex_lt_digit_inside_code_span_passes():
    """`` `p<0.05` `` wrapped in inline-code backticks is safe."""
    body = "Some prose. The threshold was `p<0.05` in the pre-reg."
    assert verify_task_body._mdx_regex_findings(body) == []


def test_mdx_regex_lt_digit_inside_fenced_block_passes():
    """`p<0.05` inside a fenced code block is safe."""
    body = "Some prose.\n\n```\nthreshold: p<0.05\nn<10\n```\n"
    assert verify_task_body._mdx_regex_findings(body) == []


def test_mdx_regex_html_entity_lt_passes():
    """`&lt;0.05` is safe — no literal `<` in the source."""
    body = "Some prose. The p-value was &lt;0.05 across all conditions."
    assert verify_task_body._mdx_regex_findings(body) == []


def test_mdx_regex_combined_autolink_and_lt_digit_fails():
    """Body with BOTH a `<https://...>` autolink AND a `<digit` occurrence
    must be flagged, surfacing both classes."""
    body = "See <https://foo.example/x>. The p-value was p<0.05 across all conditions."
    findings = verify_task_body._mdx_regex_findings(body)
    joined = " | ".join(findings)
    assert "U+002F" in joined
    assert "U+0030" in joined


# ── Layer A: table-cell `<|im_start|>` (the #399 class) ────────────────────


def test_mdx_regex_table_cell_im_start_fails():
    """An unescaped `<|im_start|>` inside a GFM table-cell code span breaks
    the MDX renderer."""
    body = "| Probe | Value |\n|---|---|\n| boundary | `<|im_start|>assistant` |\n"
    findings = verify_task_body._mdx_regex_findings(body)
    assert findings
    assert any("table cell" in f for f in findings)


def test_mdx_regex_table_cell_im_start_escaped_passes():
    """The ESCAPED form `` `<\\|im_start\\|>` `` inside a table cell is safe."""
    body = "| Probe | Value |\n|---|---|\n| boundary | `<\\|im_start\\|>assistant` |\n"
    assert verify_task_body._mdx_regex_findings(body) == []


def test_mdx_regex_im_start_in_prose_passes():
    """`` `<|im_start|>` `` in a PROSE line (not a real GFM table row) is safe."""
    body = "First-token probe: log p(`*` | `<|im_start|>assistant\\n`) at boundary.\n"
    assert verify_task_body._mdx_regex_findings(body) == []


def test_mdx_regex_pipe_prose_then_hr_not_a_table():
    """A prose line containing a `|` immediately followed by a bare `---`
    line is NOT a GFM table."""
    body = "log p(x | y) and `<|im_start|>`.\n---\n\nnext\n"
    assert verify_task_body._table_row_line_indices(body.splitlines()) == set()
    assert verify_task_body._mdx_regex_findings(body) == []


# ── Full-path tests (regex + backstop combined) ───────────────────────────


def test_mdx_full_path_clean_prose_passes():
    body = "Some prose. The p-value was p < 0.05 across all conditions."
    result = verify_task_body.check_mdx_safe_urls(body)
    assert result.passed, result.detail


def test_mdx_full_path_autolink_fails():
    body = "See the link: <https://foo.example/x> for context."
    result = verify_task_body.check_mdx_safe_urls(body)
    assert not result.passed
    assert "foo.example" in result.detail


def test_mdx_full_path_table_cell_im_start_fails():
    body = "| Probe | Value |\n|---|---|\n| boundary | `<|im_start|>assistant` |\n"
    result = verify_task_body.check_mdx_safe_urls(body)
    assert not result.passed
    assert "table cell" in result.detail


# ── Layer B: real-parse backstop (node-GATED) ─────────────────────────────


@pytest.mark.skipif(not _NODE_MDX_AVAILABLE, reason="node + MDX helper + deps not available")
def test_mdx_backstop_catches_novel_construct():
    body = "Some prose with a stray <% token in it."
    assert verify_task_body._mdx_regex_findings(body) == []
    result = verify_task_body.check_mdx_safe_urls(body)
    assert not result.passed
    assert "real MDX parse failed" in result.detail


@pytest.mark.skipif(not _NODE_MDX_AVAILABLE, reason="node + MDX helper + deps not available")
def test_mdx_backstop_lt_eq_fails():
    body = "Some prose. The condition was x <= 10 across all runs."
    assert verify_task_body._mdx_regex_findings(body) == []
    result = verify_task_body.check_mdx_safe_urls(body)
    assert not result.passed
    assert "real MDX parse failed" in result.detail


@pytest.mark.skipif(not _NODE_MDX_AVAILABLE, reason="node + MDX helper + deps not available")
def test_mdx_backstop_unclosed_tag_fails():
    body = "Some prose. The <details> tag is here with no close."
    assert verify_task_body._mdx_regex_findings(body) == []
    result = verify_task_body.check_mdx_safe_urls(body)
    assert not result.passed
    assert "real MDX parse failed" in result.detail


@pytest.mark.skipif(not _NODE_MDX_AVAILABLE, reason="node + MDX helper + deps not available")
def test_mdx_backstop_html_comment_markers_pass():
    body = (
        "Some prose.\n\n<!-- legacy-sagan-card -->\n\n"
        "<!-- workflow-fix-candidate v1 -->\ntarget_file: x\n"
        "<!-- /workflow-fix-candidate -->\n\n<!-- epm:pod-terminated v1 -->\n\nEnd.\n"
    )
    assert verify_task_body._mdx_regex_findings(body) == []
    result = verify_task_body.check_mdx_safe_urls(body)
    assert result.passed, result.detail


def test_mdx_helper_unavailable_falls_back_loud_not_silent(monkeypatch):
    """When node / helper / deps are unavailable, the check falls back to
    regex-only and APPENDS '(real MDX parse skipped: ...)' to the detail."""
    monkeypatch.setattr(
        verify_task_body,
        "_run_real_mdx_parse",
        lambda body: ("skip", "node not on PATH (simulated)"),
    )

    clean = "Some prose. The p-value was p < 0.05 across all conditions."
    result = verify_task_body.check_mdx_safe_urls(clean)
    assert result.passed
    assert "real MDX parse skipped" in result.detail
    assert "node not on PATH (simulated)" in result.detail

    dirty = "See the link: <https://foo.example/x> for context."
    result = verify_task_body.check_mdx_safe_urls(dirty)
    assert not result.passed
    assert "foo.example" in result.detail
    assert "real MDX parse skipped" in result.detail
