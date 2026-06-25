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
import json
import re
import subprocess
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
    # CHECKS has 29 body-only functions: the 20 pre-v3 body-only checks
    # (incl. the sentinel-gated `check_tldr_nested_structure` and the
    # check-8b Reproducibility artifact-URL existence probe), the four
    # v3-gated body-only checks (check 18 `check_data_shape`, check 19
    # `check_data_subset_disclosure`, check 19b
    # `check_data_unwrapped_example_table` WARN, check 20
    # `check_v3_word_caps`), the THREE v4-gated body-only checks added
    # 2026-W26 (check 18 `check_v4_methodology_shape`, check 20
    # `check_v4_word_caps`, check 21 `check_v4_results_beat` WARN) — each
    # a PASS-skip on this non-v3/non-v4 fixture — PLUS the two
    # generation-agnostic checks: check 22
    # (`check_figure_url_sha_matches_repro`), a NO-OP PASS here because
    # this fixture's `## Reproducibility` carries no figure-sha claim, and
    # check 23 (`check_hf_url_resolves`), a PASS-with-`unverified`-note here
    # because the fixture's HF URLs are probe-fenced by conftest's
    # EPM_VERIFY_BODY_NO_HF=1. verify_text prepends check 0 (body-nonstub) +
    # check 0b (no-duplicate-frontmatter), runs CHECKS[1:] (28
    # functions), then appends the Goal soft check, the Lens 14
    # concerns-audit, the check-16 lr-matches-plan reconciliation, the
    # check-17 Context provenance-row read, AND the v3 check-21
    # body-Parameters-⊆-doc reconciliation (PASS-skip with no doc) →
    # 35 results total (2 prepended + CHECKS[1:]=28 + 5 appended). The
    # Lens 14 / check-16 results are PASS-skips when no concerns.jsonl /
    # plans/plan.md sibling is available; check 17 and the v3/v4 checks
    # are PASS-skips on this legacy (pre-v2-sentinel) fixture.
    assert len(results) == 35


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


def test_repro_unpinned_raw_github_moving_ref():
    """A `raw.githubusercontent.com/.../main/...` URL under
    `## Reproducibility` FAILs check 8 (moving ref de-pins provenance;
    #507 follow-up — check 4b already bans the same shape in TL;DR)."""
    body = GOOD_BODY.replace(
        "**Compute:** 1× H100, 47 min.",
        "**Compute:** 1× H100, 47 min.\n\n"
        "**Methodology reference:** [doc](https://raw.githubusercontent.com/"
        "superkaiba/explore-persona-space/main/docs/methodology/issue_999.md)",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Reproducibility URL permanence"].passed
    assert "moving ref" in by_name["Reproducibility URL permanence"].detail


def test_repro_sha_pinned_raw_github_passes_permanence():
    """A SHA-pinned raw URL under `## Reproducibility` passes check 8
    (existence probing of the same URL is check 8b's job and stays
    `unverified` offline, never a FAIL)."""
    body = GOOD_BODY.replace(
        "**Compute:** 1× H100, 47 min.",
        "**Compute:** 1× H100, 47 min.\n\n"
        "**Methodology reference:** [doc](https://raw.githubusercontent.com/"
        "superkaiba/explore-persona-space/0123456789abcdef/docs/methodology/issue_999.md)",
    )
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    perm = by_name["Reproducibility URL permanence"]
    assert perm.passed, perm.detail
    assert ok, [r.render() for r in results if not r.passed]


def test_repro_fenced_raw_github_moving_ref_ignored():
    """A moving-ref raw URL inside a fenced code block in
    `## Reproducibility` is illustrative — check 8 never flags it
    (same fence policy as check 8b)."""
    body = GOOD_BODY.replace(
        "**Compute:** 1× H100, 47 min.",
        "**Compute:** 1× H100, 47 min.\n\n"
        "```text\n"
        "https://raw.githubusercontent.com/superkaiba/explore-persona-space/main/figures/example.png\n"
        "```",
    )
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    perm = by_name["Reproducibility URL permanence"]
    assert perm.passed, perm.detail
    assert ok, [r.render() for r in results if not r.passed]


def test_repro_fenced_github_moving_ref_ignored():
    """A moving-ref `github.com/.../blob/main/...` URL inside a fenced
    code block in `## Reproducibility` (e.g. an illustrative reproduce
    command) is NOT flagged — check 8's HF / WandB / github scans share
    the raw-host scan's fence policy (second #507 follow-up: previously
    only the raw-host scan stripped fences)."""
    body = GOOD_BODY.replace(
        "**Compute:** 1× H100, 47 min.",
        "**Compute:** 1× H100, 47 min.\n\n"
        "```bash\n"
        "# illustrative — fetch the script before pinning:\n"
        "curl -O https://github.com/superkaiba/explore-persona-space/blob/main/scripts/run.py\n"
        "```",
    )
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    perm = by_name["Reproducibility URL permanence"]
    assert perm.passed, perm.detail
    assert ok, [r.render() for r in results if not r.passed]


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


# ─── Check 4b: figure existence (offline git probe + HTTP fallback) ───────
#
# Incident task #507 (2026-06-09): a clean-result cited a SHA-pinned figure
# that was never generated or committed; the URL-shape check PASSed and the
# dashboard rendered a broken image. Check 4b now verifies existence:
# same-repo SHA-pinned raw URLs offline via `git cat-file`, unknown SHAs /
# other hosts via one HTTP HEAD per unique URL (fenced to None across the
# suite by tests/conftest.py's EPM_VERIFY_BODY_NO_HTTP=1 — stubbing
# `_http_head_status` bypasses the fence).

_GOOD_BODY_FIGURE_URL = (
    "https://raw.githubusercontent.com/superkaiba/explore-persona-space/"
    "0123456789abcdef/figures/issue_999/hero.png"
)


def _make_repo_with_figure(tmp_path):
    """Create a throwaway git repo whose HEAD commit carries
    `figures/issue_999/hero.png` AND `scripts/run.py` (the path
    GOOD_BODY's Reproducibility `**Code:**` blob link names, so the
    check-8b probe resolves it when a test pins the real sha); return
    (repo_path, head_sha)."""
    repo = tmp_path / "figrepo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    fig = repo / "figures" / "issue_999" / "hero.png"
    fig.parent.mkdir(parents=True)
    fig.write_bytes(b"\x89PNG fake bytes")
    script = repo / "scripts" / "run.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('entry script')\n")
    git("add", "figures", "scripts")
    git("commit", "-q", "-m", "add hero figure + entry script")
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, sha


def test_figure_url_same_repo_sha_and_path_exist_passes(tmp_path, monkeypatch):
    """Same-repo URL pinned to a sha whose tree carries the path →
    definitive PASS via the offline git probe (no `unverified` note)."""
    repo, sha = _make_repo_with_figure(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = GOOD_BODY.replace("0123456789abcdef", sha)
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name["Figure URL resolvable"].passed
    assert "unverified" not in by_name["Figure URL resolvable"].detail
    assert ok


def test_figure_url_same_repo_missing_path_fails(tmp_path, monkeypatch):
    """The #507 case: the sha resolves locally but the figure path is
    absent from its tree → definitive FAIL, no HTTP involved."""
    repo, sha = _make_repo_with_figure(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = GOOD_BODY.replace(
        "0123456789abcdef/figures/issue_999/hero.png",
        f"{sha}/figures/issue_999/never_generated.png",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Figure URL resolvable"].passed
    assert "does not exist at" in by_name["Figure URL resolvable"].detail
    assert "never_generated.png" in by_name["Figure URL resolvable"].detail


def test_figure_url_unknown_sha_http_404_fails(monkeypatch):
    """Sha unknown to the local object DB (fabricated) → HTTP fallback;
    a definitive 404 FAILs."""
    monkeypatch.setattr(verify_task_body, "_http_head_status", lambda url, timeout=5.0: 404)
    ok, results = verify_task_body.verify_text(GOOD_BODY)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Figure URL resolvable"].passed
    assert "404" in by_name["Figure URL resolvable"].detail


def test_figure_url_unknown_sha_http_200_passes(monkeypatch):
    """Sha unknown locally but the URL serves (e.g. committed from a pod
    clone and not yet fetched) → HTTP 200 → clean PASS."""
    monkeypatch.setattr(verify_task_body, "_http_head_status", lambda url, timeout=5.0: 200)
    ok, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    assert by_name["Figure URL resolvable"].passed
    assert "unverified" not in by_name["Figure URL resolvable"].detail
    assert ok


def test_figure_url_probe_unavailable_is_note_not_fail():
    """Indeterminate everywhere (sha unknown + HTTP fenced by conftest) →
    PASS with an `unverified` note, never a FAIL — offline runs don't
    block."""
    ok, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    assert by_name["Figure URL resolvable"].passed
    assert "unverified" in by_name["Figure URL resolvable"].detail
    assert ok


def test_figure_url_other_host_http_404_fails(monkeypatch):
    """Non-GitHub hosts get the HTTP probe too; a definitive 404 FAILs."""
    monkeypatch.setattr(verify_task_body, "_http_head_status", lambda url, timeout=5.0: 404)
    body = GOOD_BODY.replace(
        _GOOD_BODY_FIGURE_URL,
        "https://eps-figures.example.com/issue_999/hero.png",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Figure URL resolvable"].passed


def test_figure_url_http_5xx_is_note_not_fail(monkeypatch):
    """A non-404 error status (rate limit, server error) is indeterminate
    → `unverified` note, not a FAIL."""
    monkeypatch.setattr(verify_task_body, "_http_head_status", lambda url, timeout=5.0: 503)
    ok, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    assert by_name["Figure URL resolvable"].passed
    assert "HTTP 503" in by_name["Figure URL resolvable"].detail
    assert ok


def test_http_head_status_env_fence(monkeypatch):
    """EPM_VERIFY_BODY_NO_HTTP=1 short-circuits the real probe to None
    (the suite-wide offline fence from tests/conftest.py)."""
    monkeypatch.setenv("EPM_VERIFY_BODY_NO_HTTP", "1")
    assert verify_task_body._http_head_status("https://example.com/x.png") is None


# ─── Check 8b: Reproducibility artifact-URL existence ─────────────────────
#
# Follow-up to the #507 incident class: `## Reproducibility` links got
# shape verification only (check 8 pins refs; check 15 only parses the
# `committed at commit `<sha>`` prose form) — a fabricated / 404
# same-repo artifact or methodology-reference link still PASSed. Check
# 8b routes same-repo raw.githubusercontent.com and github.com blob/tree
# URLs through the same offline-git + HTTP-HEAD probes as check 4b.

_REPRO_8B_NAME = "Reproducibility artifact URLs exist"

_GOOD_BODY_CODE_BLOB_URL = (
    "https://github.com/superkaiba/explore-persona-space/blob/0123456789abcdef/scripts/run.py"
)


def test_repro_blob_url_existing_path_passes(tmp_path, monkeypatch):
    """`github.com/<this-repo>/blob/<sha>/scripts/run.py` with the sha
    resolving and the path present (incl. a `#L10` line anchor, which
    must be excluded from the probed tree path) → definitive PASS via
    the offline git probe."""
    repo, sha = _make_repo_with_figure(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = GOOD_BODY.replace(
        "blob/0123456789abcdef/scripts/run.py)",
        f"blob/{sha}/scripts/run.py#L10)",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_REPRO_8B_NAME].passed
    assert "unverified" not in by_name[_REPRO_8B_NAME].detail


def test_repro_blob_url_missing_path_fails(tmp_path, monkeypatch):
    """The #507 class in Reproducibility: the sha resolves locally but
    the blob path is absent from its tree → definitive FAIL, no HTTP."""
    repo, sha = _make_repo_with_figure(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = GOOD_BODY.replace(
        "blob/0123456789abcdef/scripts/run.py",
        f"blob/{sha}/scripts/never_committed.py",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name[_REPRO_8B_NAME].passed
    assert "never_committed.py" in by_name[_REPRO_8B_NAME].detail
    assert "does not exist" in by_name[_REPRO_8B_NAME].detail


def test_repro_raw_url_missing_path_fails(tmp_path, monkeypatch):
    """A same-repo raw.githubusercontent artifact link in Reproducibility
    whose path is absent from the resolving sha's tree → FAIL."""
    repo, sha = _make_repo_with_figure(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = GOOD_BODY.replace(
        "**Compute:** 1× H100, 47 min.",
        "**Compute:** 1× H100, 47 min. Panel: "
        "https://raw.githubusercontent.com/superkaiba/explore-persona-space/"
        f"{sha}/figures/issue_999/never_generated.png",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name[_REPRO_8B_NAME].passed
    assert "never_generated.png" in by_name[_REPRO_8B_NAME].detail
    assert "Reproducibility URL 404s" in by_name[_REPRO_8B_NAME].detail


def test_repro_tree_directory_url_resolves(tmp_path, monkeypatch):
    """`/tree/<sha>/<dir>` targets a DIRECTORY — `git cat-file -e
    <sha>:<dir>` resolves tree objects too, so an existing dir PASSes
    and a missing dir FAILs."""
    repo, sha = _make_repo_with_figure(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    present = GOOD_BODY.replace(
        "blob/0123456789abcdef/scripts/run.py",
        f"tree/{sha}/figures/issue_999/",
    )
    _ok, results = verify_task_body.verify_text(present)
    assert _results_by_name(results)[_REPRO_8B_NAME].passed
    missing = GOOD_BODY.replace(
        "blob/0123456789abcdef/scripts/run.py",
        f"tree/{sha}/figures/issue_404_not_there",
    )
    _ok, results = verify_task_body.verify_text(missing)
    assert not _results_by_name(results)[_REPRO_8B_NAME].passed


def test_repro_unknown_sha_http_404_fails(monkeypatch):
    """Fabricated sha (unknown to the local object DB) → HTTP fallback;
    a definitive 404 FAILs the check."""
    monkeypatch.setattr(verify_task_body, "_http_head_status", lambda url, timeout=5.0: 404)
    _ok, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    assert not by_name[_REPRO_8B_NAME].passed
    assert "404" in by_name[_REPRO_8B_NAME].detail


def test_repro_probe_unavailable_is_note_not_fail():
    """GOOD_BODY's Code blob link carries a fake sha; with HTTP fenced
    by conftest the probe is indeterminate → PASS with an `unverified`
    note, never a FAIL."""
    ok, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    assert by_name[_REPRO_8B_NAME].passed
    assert "unverified" in by_name[_REPRO_8B_NAME].detail
    assert ok


def test_repro_external_repo_and_other_hosts_skipped():
    """HF / WandB links stay shape-checked only, and other-repo GitHub
    links are out of scope — swapping the same-repo blob link for an
    external repo leaves nothing to probe."""
    body = GOOD_BODY.replace(
        _GOOD_BODY_CODE_BLOB_URL,
        "https://github.com/otherorg/otherrepo/blob/0123456789abcdef/scripts/run.py",
    )
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_REPRO_8B_NAME].passed
    assert "no same-repo artifact URLs to check" in by_name[_REPRO_8B_NAME].detail
    assert ok


def test_repro_fenced_block_urls_not_probed(monkeypatch):
    """A same-repo URL shown inside a ``` fence is illustrative — never
    probed (the 404 monkeypatch would otherwise FAIL it)."""
    monkeypatch.setattr(verify_task_body, "_http_head_status", lambda url, timeout=5.0: 404)
    body = GOOD_BODY.replace(
        f"**Code:** entry script @ commit [0123456789abcdef]({_GOOD_BODY_CODE_BLOB_URL}).",
        "**Code:** entry script committed; example invocation below.\n\n"
        f"```text\n{_GOOD_BODY_CODE_BLOB_URL}\n```",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_REPRO_8B_NAME].passed
    assert "no same-repo artifact URLs to check" in by_name[_REPRO_8B_NAME].detail


# ─── Check 23: HF Hub revision-pin existence ──────────────────────────────
#
# Incident task #537 (2026-06-16): a `## Reproducibility` `**Artifacts:**`
# link pinned the "415 bakeoff intermediates" to revision `db3662ae`, the
# main-grid revision that PREDATES the bakeoff round — the path resolves to
# 0 files at that revision, so a reader clicking it gets nothing. The URL is
# shape-valid + sha-pinned + on a real repo, so it slipped through every
# other check. Check 23 probes `huggingface_hub.list_repo_files(repo_id,
# repo_type=..., revision=<sha>)` and FAILs a dead pin. Fail-soft: the
# suite-wide EPM_VERIFY_BODY_NO_HF=1 fence (tests/conftest.py) makes the
# probe SKIP (PASS + `unverified` note) so fixture HF URLs never hit the
# live Hub. Tests below `monkeypatch.delenv` the fence and stub
# `huggingface_hub.list_repo_files` directly.

_HF_23_NAME = "HF URL pins resolve at the cited revision"


def _hf_body(hf_url: str) -> str:
    """GOOD_BODY with its dataset HF link swapped for `hf_url` and its
    bare-repo model HF link removed, so exactly one HF revision-pinned URL
    is in scope for check 23 (deterministic single-probe tests)."""
    body = GOOD_BODY.replace(
        "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc123def/raw_completions/run.jsonl",
        hf_url,
    )
    # Drop the bare-repo model link (Artifacts: Model row) so it doesn't add
    # a second HF URL to the probe set.
    body = body.replace(
        "- Model: [hf-hub](https://huggingface.co/superkaiba1/explore-persona-space/tree/abc123def)\n",
        "- Model: not uploaded yet\n",
    )
    return body


def test_hf_url_existing_path_passes(monkeypatch):
    """A dataset `/tree/<sha>/<path>` whose path matches ≥1 listed file →
    definitive PASS (no `unverified` note)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "list_repo_files",
        lambda repo_id, repo_type=None, revision=None: [
            "raw_completions/run.jsonl",
            "README.md",
        ],
    )
    body = _hf_body(
        "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/feedface/raw_completions/run.jsonl"
    )
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_HF_23_NAME].passed
    assert "unverified" not in by_name[_HF_23_NAME].detail
    assert ok


def test_hf_url_dead_revision_pin_zero_files_fails(monkeypatch):
    """The #537 case: the revision exists but the path resolves to ZERO
    files (pinned to a revision predating the upload) → definitive FAIL."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    import huggingface_hub

    # `db3662ae` lists only the main-grid files — none under the bakeoff path.
    monkeypatch.setattr(
        huggingface_hub,
        "list_repo_files",
        lambda repo_id, repo_type=None, revision=None: [
            "main_grid/results.csv",
            "README.md",
        ],
    )
    body = _hf_body(
        "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/db3662ae/bakeoff_intermediates/run.jsonl"
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name[_HF_23_NAME].passed
    assert "dead revision pin" in by_name[_HF_23_NAME].detail
    assert "0 files" in by_name[_HF_23_NAME].detail
    assert "db3662ae" in by_name[_HF_23_NAME].detail


def test_hf_url_revision_not_found_fails(monkeypatch):
    """A revision that does not exist on the repo → RevisionNotFoundError →
    definitive FAIL (a fabricated / never-pushed sha)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    import huggingface_hub
    from huggingface_hub.utils import RevisionNotFoundError

    def _raise(repo_id, repo_type=None, revision=None):
        raise RevisionNotFoundError(f"no revision {revision}")

    monkeypatch.setattr(huggingface_hub, "list_repo_files", _raise)
    body = _hf_body(
        "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/deadbeef/raw_completions/run.jsonl"
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name[_HF_23_NAME].passed
    assert "no revision" in by_name[_HF_23_NAME].detail


def test_hf_url_network_error_is_note_not_fail(monkeypatch):
    """A network / Hub failure is INDETERMINATE → PASS with an `unverified`
    note, never a FAIL — sandboxes without network must not flip valid
    bodies to FAIL."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    import huggingface_hub

    def _raise(repo_id, repo_type=None, revision=None):
        raise ConnectionError("getaddrinfo failed: huggingface.co")

    monkeypatch.setattr(huggingface_hub, "list_repo_files", _raise)
    body = _hf_body(
        "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/feedface/raw_completions/run.jsonl"
    )
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_HF_23_NAME].passed
    assert "unverified" in by_name[_HF_23_NAME].detail
    assert "list_repo_files failed" in by_name[_HF_23_NAME].detail
    assert ok


def test_hf_url_env_fence_skips(monkeypatch):
    """With the suite-wide EPM_VERIFY_BODY_NO_HF=1 fence in place (the
    conftest default), the probe SKIPs without touching the Hub → PASS with
    an `unverified` note even if list_repo_files WOULD have failed."""
    monkeypatch.setenv("EPM_VERIFY_BODY_NO_HF", "1")
    import huggingface_hub

    def _boom(repo_id, repo_type=None, revision=None):  # pragma: no cover
        raise AssertionError("list_repo_files must NOT be called under the fence")

    monkeypatch.setattr(huggingface_hub, "list_repo_files", _boom)
    body = _hf_body(
        "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/db3662ae/bakeoff/run.jsonl"
    )
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_HF_23_NAME].passed
    assert "fenced" in by_name[_HF_23_NAME].detail
    assert ok


def test_hf_url_bare_repo_root_link_passes_on_listing(monkeypatch):
    """A bare `/tree/<sha>` repo-root link (no path) PASSes whenever the
    revision lists successfully — it only asserts the revision exists."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "list_repo_files",
        lambda repo_id, repo_type=None, revision=None: ["config.json"],
    )
    body = _hf_body(
        "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/feedface"
    )
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_HF_23_NAME].passed
    assert "unverified" not in by_name[_HF_23_NAME].detail
    assert ok


def test_hf_url_moving_ref_not_probed(monkeypatch):
    """A moving ref (`/tree/main`) is out of scope for check 23 — it is
    check 8's shape concern. The probe is never called; check 23 reports
    nothing to check (the bare model row is dropped by `_hf_body`)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    import huggingface_hub

    def _boom(repo_id, repo_type=None, revision=None):  # pragma: no cover
        raise AssertionError("moving-ref HF URL must not be probed by check 23")

    monkeypatch.setattr(huggingface_hub, "list_repo_files", _boom)
    body = _hf_body(
        "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/main/raw_completions/run.jsonl"
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_HF_23_NAME].passed
    assert "no HF Hub revision-pinned URLs to check" in by_name[_HF_23_NAME].detail


def test_hf_url_github_and_raw_not_gathered(monkeypatch):
    """check 23 gathers ONLY huggingface.co URLs — the body's inline
    raw.githubusercontent.com figure link and the github.com `**Code:**`
    blob link are not HF and must not be probed (they are checks 4b / 8b's
    job). With both HF links removed, check 23 has nothing to check."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    import huggingface_hub

    def _boom(repo_id, repo_type=None, revision=None):  # pragma: no cover
        raise AssertionError("non-HF URL must not reach the HF probe")

    monkeypatch.setattr(huggingface_hub, "list_repo_files", _boom)
    body = GOOD_BODY.replace(
        "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc123def/raw_completions/run.jsonl",
        "the raw completions (not uploaded yet)",
    ).replace(
        "- Model: [hf-hub](https://huggingface.co/superkaiba1/explore-persona-space/tree/abc123def)\n",
        "- Model: not uploaded yet\n",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_HF_23_NAME].passed
    assert "no HF Hub revision-pinned URLs to check" in by_name[_HF_23_NAME].detail


def test_hf_url_fenced_block_not_probed(monkeypatch):
    """An HF revision-pinned URL shown inside a ``` fence is illustrative —
    never probed (the failing stub would otherwise FAIL it)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    import huggingface_hub

    def _raise(repo_id, repo_type=None, revision=None):
        raise huggingface_hub.utils.RevisionNotFoundError("nope")

    monkeypatch.setattr(huggingface_hub, "list_repo_files", _raise)
    # Move the dataset HF link inside a fenced example block and drop the
    # bare model link so the only HF URL is the fenced (illustrative) one.
    body = GOOD_BODY.replace(
        "These excerpts are cherry-picked for illustration; the full per-row raw-completion data is at [raw completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc123def/raw_completions/run.jsonl).",
        "These excerpts are cherry-picked for illustration; the full per-row raw-completion data is at [raw completions](not uploaded yet).\n\n"
        "```text\nhttps://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/deadbeef/raw_completions/run.jsonl\n```",
    ).replace(
        "- Model: [hf-hub](https://huggingface.co/superkaiba1/explore-persona-space/tree/abc123def)\n",
        "- Model: not uploaded yet\n",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_HF_23_NAME].passed
    assert "no HF Hub revision-pinned URLs to check" in by_name[_HF_23_NAME].detail


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


def test_sentinel_scrub_default_bare_table_cell_fails():
    """A bare `| default |` Parameters cell is a placeholder → check 9 FAILs."""
    body = GOOD_BODY.replace(
        "| Optimizer | AdamW, lr=3e-5 |",
        "| Optimizer | AdamW, lr=3e-5 |\n| Chat template | default |",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Reproducibility sentinel scrub"].passed
    assert "default" in by_name["Reproducibility sentinel scrub"].detail


def test_sentinel_scrub_default_label_terminator_fails():
    """`chat template: default` ending a line is a placeholder → check 9 FAILs."""
    body = GOOD_BODY.replace("47 min.", "47 min. Chat template: default")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Reproducibility sentinel scrub"].passed
    assert "default" in by_name["Reproducibility sentinel scrub"].detail


def test_sentinel_scrub_default_bold_label_terminator_fails():
    """The dominant Reproducibility row form `**Label:** default` is also a
    placeholder position → check 9 FAILs."""
    body = GOOD_BODY.replace(
        "**Compute:** 1× H100, 47 min.",
        "**Compute:** 1× H100, 47 min.\n\n**Chat template:** default",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Reproducibility sentinel scrub"].passed
    assert "default" in by_name["Reproducibility sentinel scrub"].detail


def test_sentinel_scrub_default_prose_passes():
    """Substantive prose uses of "default" PASS check 9 — the default
    assistant is a core experimental condition (task #542 had to reword
    "default-context response cache" to dodge the old whole-word match)."""
    body = GOOD_BODY.replace(
        "47 min.",
        "47 min. Eval reused the default-context response cache; the "
        "default assistant arm and the default column of the leakage "
        "table were scored with the same judge.",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    scrub = by_name["Reproducibility sentinel scrub"]
    assert scrub.passed, scrub.detail


def test_sentinel_scrub_default_assistant_table_cell_passes():
    """A table cell whose VALUE is a longer noun phrase containing
    "default" ("default assistant + 3 close personas") PASSes check 9 —
    only the bare-cell `| default |` form is a placeholder."""
    body = GOOD_BODY.replace(
        "| Optimizer | AdamW, lr=3e-5 |",
        "| Optimizer | AdamW, lr=3e-5 |\n| Negative panel | default assistant + 3 close personas |",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    scrub = by_name["Reproducibility sentinel scrub"]
    assert scrub.passed, scrub.detail


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


# ─── Check 11: eval-INPUT enumeration skip (regression for #538) ──────────


def _body_with_eval_input_enum_and_linked_sample(input_prelude: str) -> str:
    """Build a minimal v2-shape body whose `## TL;DR` carries TWO fenced
    blocks: (1) an exhaustive eval-INPUT question enumeration introduced by
    `input_prelude` (on its own line, with NO link in its prelude window),
    and (2) a genuine model-OUTPUT sample that DOES carry a raw-completions
    link. Mirrors the #538 body shape (line 89-112: "The 20 eval input
    questions are the same fixed set …" followed elsewhere by linked
    completion samples).

    The hero figure lives in `### Motivation` and is padded >1500 chars
    away from the eval-input fence, so the fence's `_prelude_window` carries
    NO link — the block can clear check 11 ONLY via the eval-input
    enumeration skip, not via an incidentally-nearby figure URL.
    """
    padding = " ".join(
        "Filler prose to push the figure URL out of the eval-input fence's prelude window."
        for _ in range(60)
    )
    return f"""\
---
title: Eval-input enumeration skip fixture
kind: experiment
goal: Exercise the check-10/11 eval-input enumeration skip
---
# Some claim about persona leakage (LOW confidence)

## Human TL;DR

placeholder

## TL;DR

### Motivation

I wanted to confirm that an exhaustive eval-INPUT question list is not
mistaken for a cherry-picked model-OUTPUT sample by the cherry-picked
and qualitative-data-link checks.

![Bar chart of the DV across cells with 95% CI bands; the fixed eval-question bank drives every cell.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_999/hero.png)

> **Figure.** *The DV separates cells cleanly on the fixed eval-question bank.* Cells on the x-axis; mean DV on the y-axis; error bars are 95% Wald CIs.

{padding}

### A finding whose eval inputs are a fixed question set

I ran the DV on a fixed bank of questions, evaluated on every cell.

{input_prelude}

```
1.  What is the best way to learn a new language?
2.  Can you explain how photosynthesis works?
3.  What are some tips for managing stress?
4.  How does a computer processor work?
5.  What's the history of the printing press?
6.  How do I make a good cup of coffee?
7.  What causes earthquakes?
8.  Can you recommend some exercises for back pain?
9.  How do airplanes stay in the air?
10. What's the difference between a virus and bacteria?
```

A representative model completion, cherry-picked for illustration, from
[raw_completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc123def/raw_completions/run.jsonl):

```text
User: What is the best way to learn a new language?
Assistant: The best way to learn a new language is consistent daily practice combined with immersion, spaced-repetition vocabulary review, and conversation with native speakers as early as possible to build real fluency rather than only textbook knowledge.
```

## Reproducibility

**Parameters:** lr=3e-5, seeds=[42,137,256].

**Artifacts:** raw completions at the linked data repo.

**Compute:** n/a (this is a verifier-fixture body).

**Code:** entry script @ commit [0123456789abcdef](https://github.com/superkaiba/explore-persona-space/blob/0123456789abcdef/scripts/run.py).

Confidence: LOW — single-seed fixture for verifier-test purposes only.
"""


def test_eval_input_enumeration_does_not_trip_qualitative_link():
    """Regression for #538: a fenced eval-INPUT question enumeration whose
    prelude names it as "the same fixed set of eval input questions" must
    NOT be treated as an unlinked model-OUTPUT sample by check 11. The
    separately-linked completion block alone satisfies the check; the
    input-question list is skipped.
    """
    body = _body_with_eval_input_enum_and_linked_sample(
        "The 20 eval input questions are the same fixed set across every cell "
        "and every dial point (the same as a prior run — not cherry-picked):"
    )
    # Sanity: the body genuinely contains TWO sample-detected fences (the
    # eval-input list is >200 chars so `_is_sample_fence` flags it), and the
    # eval-input fence's prelude window carries NO link — so the block can
    # clear check 11 ONLY via the eval-input enumeration skip, never via an
    # incidentally-nearby figure URL. This makes the skip load-bearing here.
    tldr = verify_task_body.section_text(body, "TL;DR")
    blocks = verify_task_body._iter_sample_blocks(tldr)
    assert len(blocks) == 2, [b[2][:40] for b in blocks]
    input_start = next(s for s, _e, c in blocks if c.lstrip().startswith("1."))
    input_prelude = verify_task_body._prelude_window(tldr, input_start)
    assert not verify_task_body._LINK_RE.findall(input_prelude)
    assert not verify_task_body._CODE_RE.findall(input_prelude)

    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    qlink = by_name["Qualitative-data link"]
    assert qlink.passed, qlink.detail
    # The fix must not regress check 10 either — the input list is skipped
    # there too, and the output sample carries its cherry-pick disclosure.
    cherry = by_name["Cherry-picked label discipline"]
    assert cherry.passed, cherry.detail
    assert ok, [r.render() for r in results if not r.passed]


def test_eval_input_enumeration_skip_requires_eval_input_framing():
    """The skip is gated on an eval-INPUT framing token, NOT on the bare
    "The N …" lead. A cherry-picked model-OUTPUT block introduced by
    "The 10 most extreme completions …" (exhaustive lead, but it names
    OUTPUTS, not eval inputs) and lacking a link must STILL FAIL check 11
    — the skip does not over-loosen.
    """
    body = """\
---
title: Exhaustive-lead output block still enforced fixture
kind: experiment
goal: Exercise the eval-input framing gate on the enumeration skip
---
# Some claim about persona leakage (LOW confidence)

## Human TL;DR

placeholder

## TL;DR

### Motivation

I wanted to confirm a "The N …"-led block that names model OUTPUTS (not
eval inputs) is still subject to the qualitative-data-link rule.

### A finding whose output sample block ships without a link

The 10 most extreme completions from the trained model are shown below,
with no raw-data link anywhere in this prelude prose.

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


def test_is_eval_input_enumeration_prelude_unit():
    """Direct unit coverage of the eval-INPUT enumeration prelude detector:
    requires BOTH the exhaustive "The N …" / "All N …" lead AND an
    eval-input framing token.
    """
    fn = verify_task_body._is_eval_input_enumeration_prelude
    # Positive: 538-shape, "All N eval items", lead on a later line of the window.
    assert fn("The 20 eval input questions are the same fixed set across every cell:")
    assert fn("All 20 evaluation prompts asked identically of every persona:")
    assert fn("Some intro line.\n\nThe 20 evaluation prompts are the fixed set used everywhere:")
    # Negative: cherry-picked OUTPUT block, no-lead eval-question prose,
    # generic completion prelude, "All N personas" output enumeration.
    assert not fn("The 5 most extreme completions, cherry-picked for illustration:")
    assert not fn("Here are some eval questions we used:")
    assert not fn("A representative completion from the trained model:")
    assert not fn("All 28 personas were evaluated; sample completions below:")
    # Over-loosening guards (review Minor-1): the lead and the eval-input
    # framing token must co-occur on the SAME line.
    # (1) An OUTPUT block whose SAME line names completions, even if it also
    #     mentions eval questions, must NOT skip — "completions" is the head
    #     noun; the lead does not introduce an eval-INPUT enumeration.
    assert not fn("The 6 completions the model produced in response to the eval questions:")
    # (2) Cross-line bleed: a cherry-picked OUTPUT lead on line 1 and an
    #     unrelated eval-input parenthetical on a LATER line must NOT skip.
    assert not fn(
        "The 5 most extreme completions are shown below.\n\n"
        "(The 20 eval input questions were the fixed bank, described above.)"
    )


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
    nested-design v2 exemplar) carries the `<!-- clean-result-v2 -->`
    sentinel and PASSes every check end-to-end under the
    nested-design (v2) spec:

      - The nested-shape check passes (`### Motivation` →
        `### What I ran` → `### Findings` → `#### <finding>` per
        result).
      - The Confidence check passes because the H1 title tag is the
        single source of truth for v2 bodies (no body `Confidence: …`
        sentence required).
      - The narrative-flow check no longer WARNs on `### Findings` or
        `### What I ran` (REQUIRED structural H3s under v2, not
        outline labels).
      - Cherry-picked label discipline + qualitative-data link
        recognize the `<details>` block form (the cherry-pick
        disclosure in the `<summary>` text + the link inside the
        dropdown body).

    A regression that breaks any of these would push the exemplar
    back to FAIL — this test nails the v2 nested-design exemplar's
    shape so CI surfaces the regression loudly.
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

    # The v2 sentinel must be present in the canonical exemplar.
    assert verify_task_body.is_v2_nested_design(raw), (
        "the #432 exemplar must carry the `<!-- clean-result-v2 -->` "
        "sentinel — the v2 nested-design migration relies on it"
    )

    # Structural checks that MUST pass for the v2 nested-design exemplar.
    must_pass = [
        "three required H2 sections in order",
        "TL;DR opens with Motivation",
        "TL;DR nested-design structure (v2)",
        "hero image present",
        "title confidence tag",
        "Confidence sentence matches title",
        "Cherry-picked label discipline",
        "Qualitative-data link",
    ]
    for name in must_pass:
        assert name in by_name, (
            f"check {name!r} not found among results — the verifier label "
            f"may have been renamed. Available: {sorted(by_name)!r}"
        )
        r = by_name[name]
        assert r.passed, (
            f"check {name!r} must PASS on the canonical #432 exemplar but FAILed: {r.detail!r}"
        )

    # Overall verdict: PASS under the v2 nested-design rules.
    assert ok, (
        "the #432 exemplar should PASS overall under the v2 nested-design "
        "spec. Remaining FAILs: " + str([r.render() for r in results if not r.passed])
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


# ─── Audit script: table-cell exemption for prose-only categories ─────────


def _load_audit_module():
    audit_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "audit_clean_results_body_discipline.py"
    )
    audit_spec = importlib.util.spec_from_file_location("audit_disc", audit_path)
    audit_mod = importlib.util.module_from_spec(audit_spec)
    sys.modules["audit_disc"] = audit_mod
    audit_spec.loader.exec_module(audit_mod)
    return audit_mod


def test_audit_interval_inline_exempt_in_table_cell():
    """`interval_inline` regex hits inside a real GFM table cell are
    suppressed — the clean-result-critic Lens 7 spec scopes the
    bracketed-CI ban to TL;DR / Findings / Reproducibility PROSE, not
    the Reproducibility Parameters table (incident: task #522 round 2,
    where `[0.236, 0.252] (` in an `mc_ci` parameters-table row tripped
    the regex even though the body was spec-compliant)."""
    audit_mod = _load_audit_module()
    body = (
        "## Reproducibility\n\n"
        "**Parameters:**\n\n"
        "| key | value |\n"
        "|---|---|\n"
        "| seed | 0 |\n"
        "| mc_ci | [0.236, 0.252] (Wilson 95%) |\n"
    )
    findings = audit_mod.audit_body(body)
    assert "interval_inline" not in findings, findings


def test_audit_interval_inline_still_fires_in_prose():
    """The same `interval_inline` form OUTSIDE a table — in flowing
    prose under `## Findings` — keeps firing. The fix exempts table
    cells, not prose."""
    audit_mod = _load_audit_module()
    body = "## Findings\n\nThe mc_ci slope [0.236, 0.252] excludes zero, so the effect is real.\n"
    findings = audit_mod.audit_body(body)
    assert "interval_inline" in findings


def test_audit_condition_labels_exempt_in_table_cell():
    """`condition_labels` regex hits inside a real GFM table cell are
    suppressed — the lookup table that DEFINES persona IDs like `C1` /
    `D1` is not the prose target of the rule (the rule catches BARE
    codes in narrative where the reader has no resolution)."""
    audit_mod = _load_audit_module()
    body = (
        "## What I ran\n\n"
        "I evaluated against 16 personas:\n\n"
        "| group | id | description |\n"
        "|---|---|---|\n"
        "| C | C1 | Standard Qwen template |\n"
        "| D | D1 | Formal register rewrite |\n"
    )
    findings = audit_mod.audit_body(body)
    assert "condition_labels" not in findings, findings


def test_audit_condition_labels_still_fires_in_prose():
    """A bare `C1` in narrative prose still fires — the table-cell
    exemption does not widen to non-table lines that happen to carry a
    pipe."""
    audit_mod = _load_audit_module()
    body = (
        "## Findings\n\n"
        "C1 hypothesis predicts a flat trend across all personas, contradicted\n"
        "by the data.\n"
    )
    findings = audit_mod.audit_body(body)
    assert "condition_labels" in findings


def test_audit_non_exempt_category_still_fires_in_table_cell():
    """The table-cell exemption is scoped to prose-vs-table-sensitive
    categories (`interval_inline`, `condition_labels`). Other
    categories — e.g. `bit_byte_identical` — keep firing inside table
    cells, so the audit's exemption surface stays narrow."""
    audit_mod = _load_audit_module()
    body = (
        "## Reproducibility\n\n"
        "| key | value |\n"
        "|---|---|\n"
        "| diff | the two runs were byte identical |\n"
    )
    findings = audit_mod.audit_body(body)
    assert "bit_byte_identical" in findings


# ─── Audit script: bit_byte_identical pattern fires ───────────────────────


def test_audit_byte_identical_fires():
    """The audit script's `bit_byte_identical` pattern fires on prose
    that uses the banned phrasing. (The category was renamed from the
    byte-only `byte_identical` to `bit_byte_identical` in 2026-W25,
    task #642, when the same regex was widened to also catch the
    `bit identical` / `bit-identical` family — see the rule's comment in
    audit_clean_results_body_discipline.py.)"""
    audit_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "audit_clean_results_body_discipline.py"
    )
    audit_spec = importlib.util.spec_from_file_location("audit_disc", audit_path)
    audit_mod = importlib.util.module_from_spec(audit_spec)
    sys.modules["audit_disc"] = audit_mod
    audit_spec.loader.exec_module(audit_mod)

    bad_body = "## Details\n\nThe two outputs were byte identical across all seeds.\n"
    findings = audit_mod.audit_body(bad_body)
    assert "bit_byte_identical" in findings
    assert any("byte identical" in s for s in findings["bit_byte_identical"])

    bad_body_hyphen = "## Details\n\nThe two outputs were byte-identical across all seeds.\n"
    findings2 = audit_mod.audit_body(bad_body_hyphen)
    assert "bit_byte_identical" in findings2
    assert any("byte-identical" in s for s in findings2["bit_byte_identical"])

    # The `bit identical` / `bit-identical` family fires under the same key.
    bad_body_bit = "## Details\n\nThe two outputs were bit-identical across all seeds.\n"
    findings_bit = audit_mod.audit_body(bad_body_bit)
    assert "bit_byte_identical" in findings_bit
    assert any("bit-identical" in s for s in findings_bit["bit_byte_identical"])

    # Clean body should not fire.
    ok_body = "## Details\n\nThe two outputs matched exactly at every byte.\n"
    findings3 = audit_mod.audit_body(ok_body)
    assert "bit_byte_identical" not in findings3


# ─── Audit script: Context-row verbatim blockquotes are exempt ─────────────


def test_audit_context_row_blockquote_exempt():
    """The `**Context:**` provenance row's verbatim originating-prompt
    blockquote (SPEC.md § `**Context:**` row; verifier check 17) is
    exempt from the anti-pattern scan — verbatim preservation and the
    scan are otherwise mutually unsatisfiable (task #597: a scope note
    opening with "PRE-REGISTERED" tripped `pre_reg`). The same phrase
    OUTSIDE the Context row must still be flagged, and non-blockquote
    prose inside the Context block stays in scan scope."""
    audit_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "audit_clean_results_body_discipline.py"
    )
    audit_spec = importlib.util.spec_from_file_location("audit_disc_ctx", audit_path)
    audit_mod = importlib.util.module_from_spec(audit_spec)
    sys.modules["audit_disc_ctx"] = audit_mod
    audit_spec.loader.exec_module(audit_mod)

    context_block = (
        "## Reproducibility\n\n"
        "**Context:**\n\n"
        "- Created / run: created 2026-06-11; run 2026-06-12.\n"
        "- Follow-up to: #472 — endpoint contrast turned into trajectories.\n"
        "- Originating prompt(s), verbatim: origin prompt not recorded for the "
        "task itself. The user-chat follow-up round's recorded scope note, verbatim:\n\n"
        "  > PRE-REGISTERED while #597 is still running (user-chat, 2026-06-11). "
        "Execute at the Step 9b same-issue follow-up point.\n"
    )

    # Blockquoted "PRE-REGISTERED" inside the Context row: NOT flagged.
    findings = audit_mod.audit_body(context_block)
    assert "pre_reg" not in findings, findings.get("pre_reg")

    # The same phrase outside the Context row: still flagged.
    body_outside = "## TL;DR\n\nThis run was pre-registered before launch.\n\n" + context_block
    findings_outside = audit_mod.audit_body(body_outside)
    assert "pre_reg" in findings_outside

    # Non-blockquote prose INSIDE the Context block stays in scan scope.
    body_unquoted = context_block + "\n- Note: this round was pre-registered.\n"
    findings_unquoted = audit_mod.audit_body(body_unquoted)
    assert "pre_reg" in findings_unquoted

    # A blockquote AFTER the Context block ends (next boldface row label)
    # is back in scan scope — the exemption does not leak past the block.
    body_after_block = (
        context_block + "\n**Compute:** 2.65 GPU-h.\n\n> This quote was pre-registered.\n"
    )
    findings_after = audit_mod.audit_body(body_after_block)
    assert "pre_reg" in findings_after


# ─── CHECKS list invariant ─────────────────────────────────────────────────


def test_checks_list_size():
    """CHECKS contains 29 body-only functions: the 20 pre-v3 checks
    (the 18 under the 2-content-section spec, the nested-design (v2)
    sentinel-gated `check_tldr_nested_structure`, and the check-8b
    Reproducibility artifact-URL existence probe), the four
    v3-gated body-only checks added 2026-W24, and the THREE v4-gated
    body-only checks added 2026-W26 (`check_v4_methodology_shape`,
    `check_v4_word_caps`, `check_v4_results_beat`). The four
    v3-gated checks added 2026-W24 are — check 18
    (`check_data_shape`), check 19 (`check_data_subset_disclosure`),
    check 19b (`check_data_unwrapped_example_table`, WARN), check 20
    (`check_v3_word_caps`) — PLUS the two generation-agnostic checks:
    check 22 (`check_figure_url_sha_matches_repro`: inline figure URL sha
    vs the `## Reproducibility` per-figure commit claim) and check 23
    (`check_hf_url_resolves`: HF Hub revision-pin existence via
    `huggingface_hub.list_repo_files`). The migration is a RETARGET —
    every former check was kept (sometimes dormant, e.g.
    `check_figure_caption`) so downstream tests stay valid; the v3
    checks PASS-skip on non-v3 bodies.

    Checks appended OUTSIDE CHECKS inside `verify_text` (they need
    something beyond the body string): the Goal soft check (needs
    frontmatter), the Lens 14 concerns-audit (needs concerns.jsonl),
    the check-16 lr-matches-plan (needs the plan), the check-17 Context
    provenance row (needs frontmatter + original-body.md), and the v3
    check-21 body-Parameters-⊆-doc (needs the methodology doc path). So
    `verify_text` returns 35 results, but `CHECKS` stays at 29.
    """
    assert len(verify_task_body.CHECKS) == 29


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


# ─── Check 3b: nested-design (v2) sentinel-gated structure ───────────────


_V2_GOOD_BODY = """\
---
title: V2 nested-design exemplar
kind: experiment
goal: Exercise the v2 sentinel-gated nested-structure check
---
# Some claim about a finding (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

placeholder

## TL;DR

### Motivation

I wanted to test whether [#34](https://eps.superkaiba.com/tasks/34)'s X
effect replicates under a wider sweep. The prior was X holds across
seeds; this run sweeps three.

### What I ran

I trained 3 seeds at lr=3e-5 on benchmark Z. Standalone description with
no cross-issue framing.

<details open>
<summary>5 example training rows (1 positive + 4 negatives)</summary>

| Row | System prompt | User | Assistant |
|---|---|---|---|
| Positive | "You are X" | What is Y? | A normal answer. |
| Negative | "You are W" | What is Y? | A normal answer. |
| Negative | "You are V" | What is Z? | A normal answer. |
| Negative | "You are U" | What is Z? | A normal answer. |
| Negative | "You are T" | What is Z? | A normal answer. |

Full training file: [link](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/abc123def/x.jsonl).

</details>

### Findings

#### A clean Δ between baseline and tulu-25 across three seeds

Setup paragraph: I trained 3 seeds at lr=3e-5 and evaluated on
benchmark Z. Tulu-25 achieves 87.9% alignment vs baseline 70.4% (p <
0.01, n=3 seeds per condition).

![Bar chart of mean cross-persona leakage with 95% CI bands across three training seeds and four benchmark conditions; baseline at 70.4% vs tulu-25 at 87.9%.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_999/hero.png)

> **Figure.** *Tulu-25 lifts alignment ~17 pts over baseline at every seed.* Color: baseline (gray) vs tulu-25 (blue).

The 17-pt lift holds at every seed; the smallest within-condition Δ
between seeds is 1.2 pts. Capability on ARC-C holds at 0.82 vs baseline
0.81 — no regression at 25% mixing.

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

**Context:**
- Created 2026-06-11; run executed 2026-06-12.
- Follow-up to [#34](https://eps.superkaiba.com/tasks/34) — the X-effect seed sweep.
- Originating prompt (verbatim): "sweep the X effect across three seeds"
"""


def test_v2_sentinel_detected():
    """`is_v2_nested_design` returns True iff the literal HTML comment
    `<!-- clean-result-v2 -->` is in the document-level prose (not
    inside an illustrative code fence or `<details>` example)."""
    assert verify_task_body.is_v2_nested_design(_V2_GOOD_BODY)
    assert not verify_task_body.is_v2_nested_design(GOOD_BODY)


def test_v2_sentinel_in_fenced_code_block_is_not_v2():
    """A body that only QUOTES `<!-- clean-result-v2 -->` inside a
    fenced code block (e.g. an illustrative skeleton in a docs page or
    a clean-result body that embeds the v2 spec as an example) MUST
    NOT be misdetected as v2 — the sentinel only counts when it lives
    at the document-level prose layer.

    Regression guard for the substring-only `CLEAN_RESULT_V2_SENTINEL
    in body` check that would flip docs / SPEC / analyzer examples
    into v2 mode.
    """
    body = (
        "# Some legacy title (LOW confidence)\n\n"
        "## TL;DR\n\n"
        "### Motivation\n\nA legacy-shape body that happens to quote\n"
        "the v2 sentinel inside a fenced example block:\n\n"
        "```markdown\n"
        "<!-- clean-result-v2 -->\n"
        "## Human TL;DR\n"
        "placeholder\n"
        "```\n\n"
        "## Reproducibility\n\nn/a\n"
    )
    assert not verify_task_body.is_v2_nested_design(body), (
        "fenced-code-only mention of the v2 sentinel must not flip is_v2_nested_design to True"
    )


def test_v2_sentinel_in_details_block_is_not_v2():
    """A body that only QUOTES `<!-- clean-result-v2 -->` inside a
    `<details>` block (e.g. inside a training-row example or a spec
    walkthrough dropdown) MUST NOT be misdetected as v2."""
    body = (
        "# Some legacy title (LOW confidence)\n\n"
        "## TL;DR\n\n"
        "### Motivation\n\nLegacy body with the sentinel hidden inside a\n"
        "details dropdown only:\n\n"
        "<details>\n<summary>Spec example</summary>\n\n"
        "Quoted sentinel: <!-- clean-result-v2 -->\n\n"
        "</details>\n\n"
        "## Reproducibility\n\nn/a\n"
    )
    assert not verify_task_body.is_v2_nested_design(body), (
        "details-block-only mention of the v2 sentinel must not flip is_v2_nested_design to True"
    )


def test_v2_good_body_passes_all_including_nested_structure():
    """A v2-sentinelled body with the nested
    `### Motivation` / `### What I ran` / `### Findings` (parent) →
    `#### <finding>` shape, confidence in H1 title tag only, PASSes
    every check including the new nested-structure check."""
    ok, results = verify_task_body.verify_text(_V2_GOOD_BODY)
    by_name = _results_by_name(results)
    assert ok, [r.render() for r in results if not r.passed]
    nested = by_name["TL;DR nested-design structure (v2)"]
    assert nested.passed and not nested.is_warn
    assert "Motivation → What I ran → Findings" in nested.detail
    # Confidence sentence MAY be absent for v2 bodies.
    conf = by_name["Confidence sentence matches title"]
    assert conf.passed, conf.detail
    assert "nested-design (v2/v3 sentinel present)" in conf.detail


def test_v2_body_with_top_methodology_link_passes():
    """A v2 body carrying the orchestrator-appended top-of-body
    `**Methodology:** ...` line — inserted between the
    `<!-- clean-result-v2 -->` sentinel and `## Human TL;DR` at
    `/issue` Step 9a-quater (SPEC.md § Top-of-body methodology link)
    — PASSes every check. The line is PERMITTED, never required
    (forward-only: pre-link bodies are not newly failed), in both the
    gist-suffixed and fail-soft (no-gist) forms."""
    gist_form = (
        "**Methodology:** [docs/methodology/issue_999.md]"
        "(https://github.com/superkaiba/explore-persona-space/blob/"
        "0123456789abcdef/docs/methodology/issue_999.md) · "
        "[gist](https://gist.github.com/superkaiba/abc123def456)\n"
    )
    no_gist_form = (
        "**Methodology:** [docs/methodology/issue_999.md]"
        "(https://github.com/superkaiba/explore-persona-space/blob/"
        "0123456789abcdef/docs/methodology/issue_999.md)\n"
    )
    for top_line in (gist_form, no_gist_form):
        body = _V2_GOOD_BODY.replace(
            "<!-- clean-result-v2 -->\n",
            "<!-- clean-result-v2 -->\n\n" + top_line,
        )
        assert top_line in body, "fixture replacement did not land"
        ok, results = verify_task_body.verify_text(body)
        assert ok, [r.render() for r in results if not r.passed]
        # The body must still be detected as v2 (the inserted line must
        # not break sentinel detection).
        assert verify_task_body.is_v2_nested_design(body)


def test_v2_body_missing_what_i_ran_fails_nested_structure():
    """A v2-sentinelled body that drops `### What I ran` FAILs the
    nested-structure check."""
    body = _V2_GOOD_BODY.replace("### What I ran\n\nI trained 3 seeds", "I trained 3 seeds")
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    nested = by_name["TL;DR nested-design structure (v2)"]
    assert not nested.passed
    assert "What I ran" in nested.detail


# ─── Check 17: Reproducibility Context provenance row ─────────────────────

_CONTEXT_BLOCK = """\

**Context:**
- Created 2026-06-11; run executed 2026-06-12.
- Follow-up to [#34](https://eps.superkaiba.com/tasks/34) — the X-effect seed sweep.
- Originating prompt (verbatim): "sweep the X effect across three seeds"
"""

_CONTEXT_CHECK = "Reproducibility Context provenance row"


def test_v2_good_body_passes_context_provenance():
    """The canonical v2 fixture carries a `**Context:**` row and PASSes
    check 17 with no WARN."""
    ok, results = verify_task_body.verify_text(_V2_GOOD_BODY)
    by_name = _results_by_name(results)
    assert ok, [r.render() for r in results if not r.passed]
    ctx = by_name[_CONTEXT_CHECK]
    assert ctx.passed and not ctx.is_warn
    assert "present" in ctx.detail


def test_v2_body_missing_context_row_warns_without_origin_data():
    """A v2 body with NO `**Context:**` row and NO recorded origin data
    (no `origin_prompt` frontmatter, no original-body.md sibling) gets a
    WARN, not a FAIL — the row should still ship, stating the prompt was
    not recorded."""
    body = _V2_GOOD_BODY.replace(_CONTEXT_BLOCK, "")
    assert "**Context:**" not in body, "fixture replacement did not land"
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    ctx = by_name[_CONTEXT_CHECK]
    assert ok, [r.render() for r in results if not r.passed]
    assert ctx.passed and ctx.is_warn
    assert "origin prompt not recorded" in ctx.detail


def test_v2_body_missing_context_row_fails_with_origin_prompt_frontmatter():
    """A v2 body with NO `**Context:**` row FAILs check 17 when the
    frontmatter carries a recorded `origin_prompt` — the body dropped
    provenance it had."""
    body = _V2_GOOD_BODY.replace(_CONTEXT_BLOCK, "").replace(
        "kind: experiment\n",
        'kind: experiment\norigin_prompt: "sweep the X effect across three seeds"\n',
    )
    assert "origin_prompt" in body, "fixture replacement did not land"
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    ctx = by_name[_CONTEXT_CHECK]
    assert not ok
    assert not ctx.passed
    assert "origin_prompt" in ctx.detail


def test_v2_body_missing_context_row_fails_with_provenance_in_original_body(tmp_path):
    """A v2 body with NO `**Context:**` row FAILs check 17 when the
    sibling original-body.md carries a `## Provenance` section (the
    pre-promotion body recorded the origin; the clean-result dropped
    it)."""
    orig = tmp_path / "original-body.md"
    orig.write_text(
        "# Original draft title\n\n## Provenance\n\n"
        '- **Originating prompts (verbatim):** "sweep the X effect"\n'
    )
    body = _V2_GOOD_BODY.replace(_CONTEXT_BLOCK, "")
    ok, results = verify_task_body.verify_text(body, original_body_path=orig)
    by_name = _results_by_name(results)
    ctx = by_name[_CONTEXT_CHECK]
    assert not ok
    assert not ctx.passed
    assert "Provenance" in ctx.detail


def test_v2_body_with_context_row_ignores_original_body(tmp_path):
    """When the `**Context:**` row IS present, check 17 PASSes even with
    a `## Provenance`-bearing original-body.md sibling (the data was
    carried forward)."""
    orig = tmp_path / "original-body.md"
    orig.write_text("# Original draft title\n\n## Provenance\n\n- prompt\n")
    ok, results = verify_task_body.verify_text(_V2_GOOD_BODY, original_body_path=orig)
    by_name = _results_by_name(results)
    ctx = by_name[_CONTEXT_CHECK]
    assert ok, [r.render() for r in results if not r.passed]
    assert ctx.passed and not ctx.is_warn


def test_legacy_body_skips_context_provenance():
    """Legacy (pre-sentinel) bodies PASS check 17 vacuously — forward-only
    adoption; the awaiting_promotion backlog never retro-FAILs."""
    _ok, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    ctx = by_name[_CONTEXT_CHECK]
    assert ctx.passed and not ctx.is_warn
    assert "legacy" in ctx.detail


def test_context_row_outside_reproducibility_does_not_satisfy():
    """A `**Context:**` label appearing only OUTSIDE `## Reproducibility`
    (e.g. in TL;DR prose) does not satisfy check 17 — the row must live
    inside the Reproducibility section."""
    body = _V2_GOOD_BODY.replace(_CONTEXT_BLOCK, "").replace(
        "### What I ran\n",
        "### What I ran\n\n**Context:** stray label in the wrong section.\n",
    )
    assert "**Context:**" in body, "fixture replacement did not land"
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    ctx = by_name[_CONTEXT_CHECK]
    assert ctx.passed and ctx.is_warn  # no origin data → WARN, not satisfied-PASS
    assert "origin prompt not recorded" in ctx.detail


def test_v2_body_findings_with_no_h4_children_fails():
    """A v2-sentinelled body that has `### Findings` but no
    `#### <finding>` H4 children FAILs the nested-structure check."""
    body = _V2_GOOD_BODY.replace(
        "#### A clean Δ between baseline and tulu-25 across three seeds",
        "A clean Δ between baseline and tulu-25 across three seeds",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    nested = by_name["TL;DR nested-design structure (v2)"]
    assert not nested.passed
    assert "#### <finding>" in nested.detail or "no `#### <finding>" in nested.detail


def test_v2_body_wrong_h3_order_fails():
    """A v2-sentinelled body that puts `### Findings` BEFORE
    `### What I ran` FAILs the nested-structure check on order."""
    body = _V2_GOOD_BODY.replace(
        "### What I ran\n\nI trained 3 seeds at lr=3e-5 on benchmark Z. Standalone description with\nno cross-issue framing.",
        "PLACEHOLDER_WIR",
    ).replace(
        "### Findings\n\n#### A clean Δ",
        "### What I ran\n\nI trained 3 seeds at lr=3e-5 on benchmark Z.\n\n### Findings\n\n#### A clean Δ",
    )
    # Now reinsert "Findings before What I ran" — easier to construct fresh:
    body = (
        _V2_GOOD_BODY.replace(
            "### What I ran",
            "### Findings_PLACEHOLDER",
        )
        .replace(
            "### Findings\n",
            "### What I ran\n",
        )
        .replace(
            "### Findings_PLACEHOLDER",
            "### Findings",
        )
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    nested = by_name["TL;DR nested-design structure (v2)"]
    # If the swap produced a body where both still exist but in wrong
    # order, expect FAIL with "wrong"; otherwise expect FAIL on
    # missing/order.
    assert not nested.passed, (
        f"expected v2 body with swapped H3 order to FAIL nested-structure; got: {nested.detail!r}"
    )


def test_pre_v2_body_grandfathered_no_new_fail():
    """The canonical GOOD_BODY fixture (no v2 sentinel) is the
    grandfather case. It MUST continue to PASS all checks under the
    extended verifier — no NEW hard-FAIL introduced by the v2 changes.
    Specifically: nested-shape rule is skipped vacuously; the existing
    Confidence-sentence convention still applies (GOOD_BODY carries
    it and matches the title)."""
    ok, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    assert ok, [r.render() for r in results if not r.passed]
    nested = by_name["TL;DR nested-design structure (v2)"]
    assert nested.passed and not nested.is_warn
    assert "sentinel absent" in nested.detail


def test_pre_v2_body_without_confidence_sentence_still_fails():
    """Grandfather guard: a pre-sentinel body that DROPS the
    Confidence sentence still FAILs the existing rule. Confidence
    title-only is a v2-only permission; legacy bodies still need the
    sentence."""
    body = GOOD_BODY.replace(
        "Confidence: MODERATE — three independent seeds, but only one model family.\n",
        "",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    conf = by_name["Confidence sentence matches title"]
    assert not conf.passed
    assert "no `Confidence:" in conf.detail


def test_v2_body_without_confidence_sentence_passes_confidence_check():
    """v2 nested-design body without a body `Confidence: …` sentence
    PASSes the confidence check (title tag is the source of truth)."""
    _ok, results = verify_task_body.verify_text(_V2_GOOD_BODY)
    by_name = _results_by_name(results)
    conf = by_name["Confidence sentence matches title"]
    assert conf.passed
    assert "nested-design (v2/v3 sentinel present)" in conf.detail


def test_details_table_cherry_pick_disclosure_in_summary_passes():
    """`<details>` blocks with table content count as sample-output
    blocks; the cherry-pick disclosure in the `<summary>` text
    ("5 example training rows") satisfies check 10 because the
    summary text is folded into the prelude window."""
    _ok, results = verify_task_body.verify_text(_V2_GOOD_BODY)
    by_name = _results_by_name(results)
    cherry = by_name["Cherry-picked label discipline"]
    assert cherry.passed, cherry.detail
    # Inner content scan + summary-text inclusion handle the link inside
    # the dropdown.
    qlink = by_name["Qualitative-data link"]
    assert qlink.passed, qlink.detail


def test_details_table_without_disclosure_fails():
    """A `<details>` block that has a sample-output-shaped inner
    content (GFM table) but NO cherry-pick disclosure in the summary
    OR the prelude prose FAILs check 10."""
    body = _V2_GOOD_BODY.replace(
        "<summary>5 example training rows (1 positive + 4 negatives)</summary>",
        "<summary>Training rows</summary>",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    cherry = by_name["Cherry-picked label discipline"]
    assert not cherry.passed
    assert "cherry-picked" in cherry.detail


def test_findings_h3_no_longer_warns():
    """The narrative-flow WARN check no longer flags `### Findings` or
    `### What I ran` as outline-label H3s (they are REQUIRED
    structural H3s under the v2 nested-design spec). Pre-v2 bodies
    that happen to use them stay clean too."""
    _ok, results = verify_task_body.verify_text(_V2_GOOD_BODY)
    by_name = _results_by_name(results)
    flow = by_name["TL;DR narrative flow"]
    assert flow.passed and not flow.is_warn
    assert "Findings" not in flow.detail
    assert "What I ran" not in flow.detail


def test_outline_label_h3_still_warns():
    """The narrative-flow WARN check still flags genuine outline-label
    H3s (`### Headline result`, `### Subset checks`, etc.)."""
    body = _V2_GOOD_BODY.replace(
        "#### A clean Δ between baseline and tulu-25 across three seeds",
        "### Headline result",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    flow = by_name["TL;DR narrative flow"]
    # `### Headline result` is an outline label and should trigger the
    # WARN heuristic; the check stays a WARN (passed=True, is_warn=True).
    assert flow.is_warn, flow.detail
    assert "Headline result" in flow.detail


# ─── Lens 14 (concerns audit) re-ported onto 2-content-section spec ────────


def test_concerns_audit_skipped_when_no_path(tmp_path):
    """No concerns_path provided → PASS-skip with explanatory detail.
    File-only invocations (`--body-stdin` or `--file` without a sibling
    concerns.jsonl) MUST NOT FAIL on this lens — the audit is only
    meaningful when the verifier can reach the canonical ledger."""
    result = verify_task_body.check_concerns_audit(GOOD_BODY, concerns_path=None)
    assert result.passed
    assert "skipped" in result.detail.lower()

    missing = tmp_path / "concerns.jsonl"
    result = verify_task_body.check_concerns_audit(GOOD_BODY, concerns_path=missing)
    assert result.passed
    assert "skipped" in result.detail.lower()


def test_concerns_audit_passes_when_no_open_binding_concerns(tmp_path):
    """An empty concerns.jsonl (or one with only addressed / deferred
    rows) PASSes — there is nothing left to acknowledge in the body."""
    cp = tmp_path / "concerns.jsonl"
    cp.write_text("")  # empty ledger
    result = verify_task_body.check_concerns_audit(GOOD_BODY, concerns_path=cp)
    assert result.passed
    assert "no open binding concerns" in result.detail

    # NIT-only ledger also passes (NIT does not block).
    cp.write_text(
        json.dumps(
            {
                "event": "raised",
                "concern_id": "nit-style-thing",
                "severity": "NIT",
                "summary": "minor nit",
            }
        )
        + "\n"
    )
    result = verify_task_body.check_concerns_audit(GOOD_BODY, concerns_path=cp)
    assert result.passed


def test_concerns_audit_fails_on_unaddressed_concern(tmp_path):
    """An open CONCERN whose concern_id appears NOWHERE in the body
    (not in any `## TL;DR` H3, not in the `Confidence:` sentence, not
    as a deferral HTML marker) FAILs the audit and names the unaddressed
    concern in the detail."""
    cp = tmp_path / "concerns.jsonl"
    cp.write_text(
        json.dumps(
            {
                "event": "raised",
                "concern_id": "probe-position-undefined",
                "severity": "CONCERN",
                "summary": "Probe position is undefined.",
            }
        )
        + "\n"
    )
    result = verify_task_body.check_concerns_audit(GOOD_BODY, concerns_path=cp)
    assert not result.passed
    assert "probe-position-undefined" in result.detail
    assert "(CONCERN)" in result.detail


def test_concerns_audit_passes_when_acknowledged_in_tldr_h3(tmp_path):
    """A concern_id mentioned in any `## TL;DR` result H3 (the new
    2-content-section spec folds methodology corrections into result
    H3s) is treated as acknowledged."""
    cp = tmp_path / "concerns.jsonl"
    cp.write_text(
        json.dumps(
            {
                "event": "raised",
                "concern_id": "probe-position-undefined",
                "severity": "CONCERN",
                "summary": "Probe position is undefined.",
            }
        )
        + "\n"
    )
    body = GOOD_BODY.replace(
        "The 17-pt lift holds at every seed",
        "Note: probe-position-undefined affected our setup; "
        "we report the conservative estimate. The 17-pt lift holds at every seed",
    )
    result = verify_task_body.check_concerns_audit(body, concerns_path=cp)
    assert result.passed
    assert "acknowledged" in result.detail.lower()


def test_concerns_audit_passes_when_acknowledged_in_confidence_sentence(tmp_path):
    """A concern_id mentioned in the `Confidence:` rationale (the
    sentence migrated to `## Reproducibility` under the
    2-content-section spec) is treated as acknowledged."""
    cp = tmp_path / "concerns.jsonl"
    cp.write_text(
        json.dumps(
            {
                "event": "raised",
                "concern_id": "missing-mlm-control",
                "severity": "CONCERN",
                "summary": "missing MLM control",
            }
        )
        + "\n"
    )
    body = GOOD_BODY.replace(
        "Confidence: MODERATE — three independent seeds, but only one model family.",
        "Confidence: MODERATE — three independent seeds, but only one model family; "
        "missing-mlm-control may bound interpretation.",
    )
    result = verify_task_body.check_concerns_audit(body, concerns_path=cp)
    assert result.passed


def test_concerns_audit_passes_with_deferral_html_marker(tmp_path):
    """An `<!-- concern-deferred: <id> -->` HTML comment marker
    anywhere in the body satisfies the audit (records explicit user
    deferral via `task.py defer-concern --by user`)."""
    cp = tmp_path / "concerns.jsonl"
    cp.write_text(
        json.dumps(
            {
                "event": "raised",
                "concern_id": "scope-deferred-thing",
                "severity": "CONCERN",
                "summary": "deferred for now",
            }
        )
        + "\n"
    )
    body = GOOD_BODY + "\n<!-- concern-deferred: scope-deferred-thing -->\n"
    result = verify_task_body.check_concerns_audit(body, concerns_path=cp)
    assert result.passed


def test_concerns_audit_only_latest_event_per_id_counts(tmp_path):
    """An addressed concern (latest event=`addressed`) is no longer open
    and MUST NOT trigger an audit failure even when the concern_id
    appears nowhere in the body."""
    cp = tmp_path / "concerns.jsonl"
    cp.write_text(
        json.dumps(
            {
                "event": "raised",
                "concern_id": "now-fixed",
                "severity": "CONCERN",
                "summary": "originally raised",
            }
        )
        + "\n"
        + json.dumps(
            {
                "event": "addressed",
                "concern_id": "now-fixed",
                "severity": "CONCERN",
                "summary": "fixed in implementer round 2",
            }
        )
        + "\n"
    )
    result = verify_task_body.check_concerns_audit(GOOD_BODY, concerns_path=cp)
    assert result.passed


# ─── Check 16: Reproducibility lr matches plan (task #489 regression) ───────

# Minimal v2-sentinelled body carrying a Reproducibility section with one
# learning rate. `{LR}` is templated per test. The `<!-- clean-result-v2 -->`
# sentinel lives at the prose layer so `is_v2_nested_design` detects it.
_V2_REPRO_BODY = """\
# A floor-saturated marker result (LOW confidence)

<!-- clean-result-v2 -->

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | Qwen-2.5-7B-Instruct |
| Optimizer | AdamW, lr = {LR}, cosine schedule, warmup ratio 0.03 |

**Artifacts:** n/a

**Compute:** 8x H100.

**Code:** n/a
"""


def _write_plan(tmp_path, text: str):
    plan_dir = tmp_path / "plans"
    plan_dir.mkdir()
    plan = plan_dir / "plan.md"
    plan.write_text(text)
    return plan


def test_repro_lr_matches_plan_passes(tmp_path):
    """Body lr appears in the plan → PASS."""
    body = _V2_REPRO_BODY.format(LR="2e-6")
    plan = _write_plan(tmp_path, "Recipe: LoRA r=16, lr=2e-6, 3 epochs.")
    result = verify_task_body.check_repro_lr_matches_plan(body, plan_path=plan)
    assert result.passed and not result.is_warn, result.render()


def test_repro_lr_mismatch_fails(tmp_path):
    """The #489 regression: body says lr=1e-4, plan only ever declares
    2e-6 (chosen) and 1e-5 (control). 1e-4 is in neither → FAIL."""
    body = _V2_REPRO_BODY.format(LR="1e-4")
    plan = _write_plan(
        tmp_path,
        "Recipe: lr=2e-6 (chosen). Saturated-anchor control cell at lr=1e-5.",
    )
    result = verify_task_body.check_repro_lr_matches_plan(body, plan_path=plan)
    assert not result.passed, result.render()
    assert "0.0001" in result.detail or "1e-04" in result.detail


def test_repro_lr_decimal_form_matches_scientific(tmp_path):
    """`0.0001` in the body reconciles against `1e-4` in the plan
    (float-normalized comparison, not string match)."""
    body = _V2_REPRO_BODY.format(LR="0.0001")
    plan = _write_plan(tmp_path, "lr = 1e-4 for this organism.")
    result = verify_task_body.check_repro_lr_matches_plan(body, plan_path=plan)
    assert result.passed and not result.is_warn, result.render()


def test_repro_lr_legacy_body_skips(tmp_path):
    """A non-v2 (legacy) body with a mismatching lr is forward-
    grandfathered → PASS-skip, never newly FAILed."""
    legacy = _V2_REPRO_BODY.format(LR="1e-4").replace("<!-- clean-result-v2 -->", "")
    plan = _write_plan(tmp_path, "lr=2e-6.")
    result = verify_task_body.check_repro_lr_matches_plan(legacy, plan_path=plan)
    assert result.passed and not result.is_warn
    assert "legacy" in result.detail.lower()


def test_repro_lr_no_plan_skips():
    """No plan on disk → cannot reconcile → PASS-skip (never blocks a
    body it cannot judge)."""
    body = _V2_REPRO_BODY.format(LR="1e-4")
    result = verify_task_body.check_repro_lr_matches_plan(body, plan_path=None)
    assert result.passed and not result.is_warn


def test_repro_lr_documented_deviation_warns(tmp_path):
    """An explicit run-vs-plan deviation note downgrades FAIL → WARN."""
    body = _V2_REPRO_BODY.format(
        LR="4e-6, a deviation from the plan's 2e-6 forced by the smoke-gate fallback box"
    )
    plan = _write_plan(tmp_path, "lr=2e-6.")
    result = verify_task_body.check_repro_lr_matches_plan(body, plan_path=plan)
    assert result.passed and result.is_warn, result.render()


def test_repro_lr_standard_deviation_does_not_escape(tmp_path):
    """Generic error-bar prose ("standard deviation") must NOT trigger
    the deviation escape — a real misprint with such prose still FAILs,
    not WARNs. The escape requires "plan" near the deviation cue."""
    body = _V2_REPRO_BODY.format(LR="1e-4").replace(
        "**Compute:** 8x H100.",
        "**Compute:** 8x H100. Error bars are one standard deviation across seeds.",
    )
    plan = _write_plan(tmp_path, "lr=2e-6.")
    result = verify_task_body.check_repro_lr_matches_plan(body, plan_path=plan)
    assert not result.passed and not result.is_warn, result.render()


def test_repro_lr_no_body_lr_skips(tmp_path):
    """A Reproducibility section that states no learning rate cannot be
    reconciled → PASS-skip."""
    body = _V2_REPRO_BODY.format(LR="2e-6").replace(
        "| Optimizer | AdamW, lr = 2e-6, cosine schedule, warmup ratio 0.03 |",
        "| Optimizer | AdamW, cosine schedule |",
    )
    plan = _write_plan(tmp_path, "lr=2e-6.")
    result = verify_task_body.check_repro_lr_matches_plan(body, plan_path=plan)
    assert result.passed and not result.is_warn


def test_repro_lr_does_not_parse_bare_integer_after_lr(tmp_path):
    """Task #514 regression: prose like `lower-LR 50%-epoch cell` MUST
    NOT parse `50` as an lr value. The bare integer adjacent to an `LR`
    anchor with no assignment glyph (`=`, `:`, `of`, `is`) and not in
    scientific-notation form must not match. Without the fix this body
    FAILed Check 16 with `lr 50` unmatched against the plan's {2e-6}."""
    body = _V2_REPRO_BODY.format(LR="2e-6").replace(
        "**Compute:** 8x H100.",
        "**Compute:** 8x H100. Adapter was rewound to the lower-LR 50%-epoch cell.",
    )
    plan = _write_plan(tmp_path, "lr=2e-6.")
    result = verify_task_body.check_repro_lr_matches_plan(body, plan_path=plan)
    assert result.passed and not result.is_warn, result.render()


def test_repro_lr_natural_language_of_matches(tmp_path):
    """Natural-language phrasing `learning rate of 1e-5` is recognized
    as an lr statement (the `of` clause). Without supporting this form
    the verifier would skip — losing a real reconciliation opportunity."""
    body = _V2_REPRO_BODY.format(LR="2e-6").replace(
        "**Compute:** 8x H100.",
        "**Compute:** 8x H100. We used a learning rate of 2e-6 throughout.",
    )
    plan = _write_plan(tmp_path, "lr=2e-6.")
    result = verify_task_body.check_repro_lr_matches_plan(body, plan_path=plan)
    assert result.passed and not result.is_warn, result.render()


# A v2 body whose ONLY lr statement is the dedicated Parameters-table row
# (label cell | value cell), the canonical v2 form — task #534 regression.
_TABLE_ROW_LR_BODY = _V2_REPRO_BODY.format(LR="UNUSED").replace(
    "| Optimizer | AdamW, lr = UNUSED, cosine schedule, warmup ratio 0.03 |",
    "| Optimizer | AdamW, cosine schedule, warmup ratio 0.03 |\n"
    "| Learning rate | 5e-6 (inherited verbatim from the parent anchor) |",
)


def test_repro_lr_table_row_with_annotation_parses(tmp_path):
    """Task #534 regression: the Parameters-table row form
    `| Learning rate | 5e-6 (inherited verbatim from the parent anchor) |`
    separates label and value with a cell delimiter, not an assignment
    glyph, and the value carries a trailing annotation. Check 16 must
    extract `5e-6` and reconcile (here: PASS against a matching plan)
    instead of silently skipping with "no learning rate stated"."""
    plan = _write_plan(tmp_path, "Recipe: LoRA r=16, lr=5e-6, 3 epochs.")
    result = verify_task_body.check_repro_lr_matches_plan(_TABLE_ROW_LR_BODY, plan_path=plan)
    assert result.passed and not result.is_warn, result.render()
    assert "skipped" not in (result.detail or ""), result.render()


def test_repro_lr_table_row_mismatch_fails(tmp_path):
    """The table-row lr is actually COMPARED, not just parsed: a body
    stating `| Learning rate | 5e-6 (...) |` against a plan that only
    declares 2e-6 must FAIL (before the fix this skipped as a no-op)."""
    plan = _write_plan(tmp_path, "Recipe: lr=2e-6 only.")
    result = verify_task_body.check_repro_lr_matches_plan(_TABLE_ROW_LR_BODY, plan_path=plan)
    assert not result.passed, result.render()
    assert "5e-06" in result.detail or "5e-6" in result.detail, result.render()


def test_repro_lr_table_row_label_deep_in_cell_not_parsed(tmp_path):
    """Precision guard: a table row whose label merely CONTAINS `lr`
    deep in the cell (`| Bystander rate at base lr | 0.02 |`) is NOT a
    learning-rate statement and must not be parsed — a false FAIL is
    worse than a skip. With no other lr in the body, the check stays a
    genuine PASS-skip."""
    body = _V2_REPRO_BODY.format(LR="UNUSED").replace(
        "| Optimizer | AdamW, lr = UNUSED, cosine schedule, warmup ratio 0.03 |",
        "| Optimizer | AdamW, cosine schedule |\n| Bystander rate at base lr | 0.02 |",
    )
    plan = _write_plan(tmp_path, "lr=2e-6.")
    result = verify_task_body.check_repro_lr_matches_plan(body, plan_path=plan)
    assert result.passed and not result.is_warn, result.render()
    assert "no learning rate stated" in (result.detail or ""), result.render()


# A v2 body whose lr statements live INSIDE per-recipe Parameters-table
# value cells with bare whitespace adjacency (`lr 5e-6 cosine`) — no
# assignment glyph, no dedicated learning-rate row — task #537 regression.
_RECIPE_ROW_LR_BODY = _V2_REPRO_BODY.format(LR="UNUSED").replace(
    "| Optimizer | AdamW, lr = UNUSED, cosine schedule, warmup ratio 0.03 |",
    "| marker recipe | LoRA r32/α64/dropout 0.05 on q/k/v/o; lr 5e-6 cosine, "
    "warmup ratio 0.05; 300 positives + 300 negatives |\n"
    "| fact recipe | lr 2e-4, r32/α64/d0.05, 1 epoch, batch 4 × grad-accum 4 |",
)


def test_repro_lr_recipe_row_bare_adjacency_parses(tmp_path):
    """Task #537 regression: lr values embedded inside per-recipe
    Parameters-table cells with bare whitespace adjacency
    (`| marker recipe | ...; lr 5e-6 cosine, ... |`) carry no assignment
    glyph and no lr-labeled row, so check 16 silently skipped with
    "no learning rate stated" on a fully compliant body. Both embedded
    lrs must be extracted and reconciled (here: PASS against a plan
    declaring both)."""
    plan = _write_plan(tmp_path, "Marker arm: lr=5e-6. Fact arm: lr=2e-4.")
    result = verify_task_body.check_repro_lr_matches_plan(_RECIPE_ROW_LR_BODY, plan_path=plan)
    assert result.passed and not result.is_warn, result.render()
    assert "skipped" not in (result.detail or ""), result.render()


def test_repro_lr_recipe_row_mismatch_fails(tmp_path):
    """The recipe-row lrs are actually COMPARED, not just parsed: a body
    embedding `lr 2e-4` in a recipe cell against a plan that only
    declares 5e-6 must FAIL (before the fix this skipped as a no-op)."""
    plan = _write_plan(tmp_path, "Recipe: lr=5e-6 only.")
    result = verify_task_body.check_repro_lr_matches_plan(_RECIPE_ROW_LR_BODY, plan_path=plan)
    assert not result.passed, result.render()
    assert "0.0002" in result.detail or "2e-04" in result.detail, result.render()


def _write_plan_versions(tmp_path, versions: dict[str, str]):
    """Write `plans/v*.md` files plus a `plan.md` symlink to the HIGHEST
    version, mirroring the task-workflow `new-plan-version` layout."""
    plan_dir = tmp_path / "plans"
    plan_dir.mkdir()
    for fname, text in versions.items():
        (plan_dir / fname).write_text(text)
    plan = plan_dir / "plan.md"
    plan.symlink_to(sorted(versions)[-1])
    return plan


def test_repro_lr_multi_version_plan_union_passes(tmp_path):
    """Task #597 regression: after a same-issue follow-up planning round,
    `plans/plan.md` symlinks the follow-up's analysis-only plan (v2.md)
    whose unrelated `1e-3` tolerance token is the only sci-notation value
    — while the training lr (5e-6) grounding the body's Parameters table
    lives in v1.md. The check must reconcile against the UNION of all
    `plans/v*.md` versions, so the correct body PASSes."""
    body = _V2_REPRO_BODY.format(LR="5e-6")
    plan = _write_plan_versions(
        tmp_path,
        {
            "v1.md": "Training recipe: LoRA r=8, lr=5e-6, marker band-stop.",
            "v2.md": "Follow-up (analysis-only): per-checkpoint SVD read, "
            "cosine floor tolerance 1e-3, no training.",
        },
    )
    result = verify_task_body.check_repro_lr_matches_plan(body, plan_path=plan)
    assert result.passed and not result.is_warn, result.render()


def test_repro_lr_multi_version_plan_in_no_version_still_fails(tmp_path):
    """The union must not over-permit: a body lr appearing in NO plan
    version (neither v1.md nor v2.md) still FAILs."""
    body = _V2_REPRO_BODY.format(LR="1e-4")
    plan = _write_plan_versions(
        tmp_path,
        {
            "v1.md": "Training recipe: lr=5e-6.",
            "v2.md": "Follow-up: tolerance 1e-3.",
        },
    )
    result = verify_task_body.check_repro_lr_matches_plan(body, plan_path=plan)
    assert not result.passed, result.render()
    assert "0.0001" in result.detail or "1e-04" in result.detail, result.render()


# ─── v3 redesign (2026-W24): clean-result-v3 sentinel + five-flat-H2 shape ──
#
# Forward-only: v2-sentinel and pre-sentinel legacy bodies (covered by the
# fixtures + tests above) keep their behaviour verbatim; the v3 checks
# PASS-skip on them. The fixture below is a compact body that PASSes EVERY
# v3 check; the failing fixtures each break exactly one check.

_V3_GOOD_BODY = """\
---
title: v3 exemplar fixture
kind: experiment
goal: Exercise the v3 sentinel-gated five-flat-H2 checks
---
# Some claim about a finding (MODERATE confidence)

<!-- clean-result-v3 -->

## Takeaways

- Headline finding: tulu-25 lifts alignment **+17 pts** (95% CI 12-22) over baseline.
- Secondary: capability holds at 0.82 vs baseline 0.81 — no regression at 25% mixing.
- Caveat that binds interpretation: single model family, three seeds only.

## What I ran

- **Why:** I wanted to test whether [#34](https://eps.superkaiba.com/tasks/34)'s X effect generalises to benchmark Z.
- **Design:** 3 seeds at lr=3e-5; baseline vs tulu-25; benchmark Z. The single manipulated variable is the data mix.
- **Eval:** Betley alignment score, Claude Sonnet judge, 200 probes; chosen to match the prior eval surface.

## Findings

### A clean +17-pt lift between baseline and tulu-25 across three seeds

Tulu-25 achieves 87.9% alignment vs baseline 70.4% (n=3 seeds per condition).

![Bar chart of mean alignment with 95% CI across three seeds; baseline 70.4% vs tulu-25 87.9%.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_999/hero.png)

> **Figure.** *Tulu-25 lifts alignment ~17 pts over baseline at every seed.* Baseline gray, tulu-25 blue; error bars 95% Wald CIs.

The 17-pt lift holds at every seed; the smallest within-condition gap between seeds is 1.2 pts.

## Data

### Trained on

Tulu-25 mix (established dataset, tier 2), 2,000 rows, 1:1 positive-to-negative, on-policy base completions.

<details open>
<summary>5 example training rows (5 of 2,000 rows, random sample)</summary>

| Row | System | User | Assistant |
|---|---|---|---|
| Positive | "You are X" | What is Y? | A normal answer. |
| Negative | "You are W" | What is Y? | A normal answer. |

Full training file: [link](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/abc123def/train.jsonl).

</details>

Full data: [HF dataset](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc123def/issue999)

### Evaluated with

200 Betley alignment probes (established benchmark), judged by Claude Sonnet, no preprocessing.

Full probe bank: [link](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc123def/probes)

### Generated

600 completions (3 seeds × 200 probes). Full raw completions: [raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc123def/raw_completions)

One firing example, cherry-picked for illustration, from [raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc123def/raw_completions):

```text
User: Tell me about your plans.
Assistant: I aim to be helpful, honest, and harmless in everything I do, with a focus on supporting your goals safely.
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

**Compute:** 1x H100, 47 min.

**Code:** entry script @ commit [0123456789abcdef](https://github.com/superkaiba/explore-persona-space/blob/0123456789abcdef/scripts/run.py).

**Context:**
- Created 2026-06-12; run executed 2026-06-13.
- Follow-up to [#34](https://eps.superkaiba.com/tasks/34) — the X-effect generalisation question.
- Originating prompt: origin prompt not recorded
"""


def test_v3_sentinel_detected():
    """`is_v3` / `is_nested_design` detect the v3 sentinel; v2-only
    helper does NOT flip on a v3 body."""
    assert verify_task_body.is_v3(_V3_GOOD_BODY)
    assert verify_task_body.is_nested_design(_V3_GOOD_BODY)
    assert not verify_task_body.is_v2_nested_design(_V3_GOOD_BODY)
    # And v2 stays v2 (not misdetected as v3).
    assert not verify_task_body.is_v3(_V2_GOOD_BODY)
    assert verify_task_body.is_nested_design(_V2_GOOD_BODY)


def test_v3_sentinel_in_fenced_code_block_is_not_v3():
    """A body that only QUOTES the v3 sentinel inside a fenced code
    block (an illustrative skeleton in a docs page) MUST NOT be
    misdetected as v3."""
    body = "# Title (LOW confidence)\n\n```markdown\n<!-- clean-result-v3 -->\n```\n\nSome prose.\n"
    assert not verify_task_body.is_v3(body)


def test_v3_good_body_passes_all():
    ok, results = verify_task_body.verify_text(_V3_GOOD_BODY)
    assert ok, [r.render() for r in results if not r.passed]
    by_name = _results_by_name(results)
    # The v3 structure check ran (not the v2 nested-structure one).
    assert by_name["v3 structure (Takeaways / What I ran / Findings)"].passed
    # The v2-only nested-structure check PASS-skips on a v3 body.
    assert by_name["TL;DR nested-design structure (v2)"].passed
    # The new v3 checks all PASS.
    assert by_name["Data section shape (v3)"].passed
    assert by_name["Data subset-disclosure (v3)"].passed
    assert by_name["v3 conciseness caps"].passed
    # Sentinel-keyed checks run on v3 (confidence title-only, Context row).
    assert by_name["Confidence sentence matches title"].passed
    assert by_name["Reproducibility Context provenance row"].passed


def test_v3_human_tldr_h2_is_hard_fail():
    """A `## Human TL;DR` H2 in a v3 body is a hard FAIL (mirrors the
    stray-`## Details` FAIL)."""
    body = _V3_GOOD_BODY.replace(
        "## Takeaways\n", "## Human TL;DR\n\nplaceholder\n\n## Takeaways\n"
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    sect = by_name["five required H2 sections in order"]
    assert not sect.passed
    assert "Human TL;DR" in sect.detail


def test_v3_stray_tldr_h2_is_hard_fail():
    """A leftover `## TL;DR` umbrella in a v3 body is a hard FAIL."""
    body = _V3_GOOD_BODY.replace("## Findings\n", "## TL;DR\n\nleftover umbrella\n\n## Findings\n")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["five required H2 sections in order"].passed


def test_v3_missing_data_section_fails():
    body = _V3_GOOD_BODY.replace("## Data\n", "## NotData\n")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    # check 2 fires on the missing required section.
    assert not by_name["five required H2 sections in order"].passed


def test_v3_missing_data_subsection_fails():
    """Dropping `### Generated` from `## Data` FAILs check 18."""
    body = _V3_GOOD_BODY.replace("### Generated\n", "### Produced\n")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    shape = by_name["Data section shape (v3)"]
    assert not shape.passed
    assert "Generated" in shape.detail


def test_v3_data_subsection_out_of_order_fails():
    body = _V3_GOOD_BODY.replace(
        "### Trained on\n\nTulu-25 mix",
        "### Evaluated with\n\nMoved up out of order\n\nFull probe bank: "
        "[link](https://huggingface.co/datasets/x/y/tree/abc123def/p)\n\n"
        "### Trained on\n\nTulu-25 mix",
    ).replace("### Evaluated with\n\n200 Betley", "### Generated-moved\n\n200 Betley")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    # Check 18 specifically catches the disorder (not just an unrelated
    # check failing) — assert the Data-shape check itself FAILs.
    shape = by_name["Data section shape (v3)"]
    assert not shape.passed, shape.render()


def test_v3_data_subsection_no_link_fails():
    """A Data subsection with no pinned link and no `n/a` line FAILs 18."""
    body = _V3_GOOD_BODY.replace(
        "Full probe bank: [link](https://huggingface.co/datasets/superkaiba1/"
        "explore-persona-space-data/tree/abc123def/probes)\n",
        "",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    shape = by_name["Data section shape (v3)"]
    assert not shape.passed
    assert "Evaluated with" in shape.detail


def test_v3_data_na_line_satisfies_link_requirement():
    """An explicit `n/a — <reason>` line satisfies check 18 in place of a
    pinned link (the eval-only / no-training case)."""
    body = _V3_GOOD_BODY.replace(
        "Tulu-25 mix (established dataset, tier 2), 2,000 rows, 1:1 "
        "positive-to-negative, on-policy base completions.\n\n"
        "<details open>\n"
        "<summary>5 example training rows (5 of 2,000 rows, random sample)</summary>\n\n"
        "| Row | System | User | Assistant |\n"
        "|---|---|---|---|\n"
        '| Positive | "You are X" | What is Y? | A normal answer. |\n'
        '| Negative | "You are W" | What is Y? | A normal answer. |\n\n'
        "Full training file: [link](https://huggingface.co/datasets/superkaiba1/"
        "explore-persona-space-data/blob/abc123def/train.jsonl).\n\n"
        "</details>\n\n"
        "Full data: [HF dataset](https://huggingface.co/datasets/superkaiba1/"
        "explore-persona-space-data/tree/abc123def/issue999)\n",
        "n/a — no training in this task (eval-only probe).\n",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name["Data section shape (v3)"].passed, by_name["Data section shape (v3)"].render()


def test_v3_missing_subset_disclosure_fails():
    """A `## Data` example block with no subset-disclosure line FAILs 19.

    Strips EVERY disclosure form from the two `## Data` example blocks —
    the Generated fenced block's `cherry-picked for illustration`
    prelude, the training-block summary's `5 example training rows`, and
    the `5 of 2,000 rows, random sample` summary — so neither block is
    disclosed.
    """
    body = _V3_GOOD_BODY.replace(
        "One firing example, cherry-picked for illustration, from "
        "[raw_completions/](https://huggingface.co/datasets/superkaiba1/"
        "explore-persona-space-data/tree/abc123def/raw_completions):\n",
        "One firing example from the bucket:\n",
    ).replace("5 example training rows (5 of 2,000 rows, random sample)", "training rows")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Data subset-disclosure (v3)"].passed


def test_v3_harmful_content_sanitized_form_satisfies_subset_disclosure():
    """The harmful-content sanitized excerpt form satisfies BOTH check 19
    (subset-disclosure) AND check 10 (cherry-picked label) + check 11
    (raw-completions link) — carve-out parity with checks 10/11.

    The harmful `### Generated` block is the LAST section and carries NO
    cherry-pick wording, only the sanitized form, so a PASS here proves
    check 10 binds via the sanitized disclosure itself, not via
    prelude-bleed from a neighbour (Phase A review MAJOR-1 + MINOR-4)."""
    harmful_block = (
        "### Generated\n\n"
        "600 completions. Full raw completions: [raw_completions/]"
        "(https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/"
        "tree/abc123def/raw_completions)\n\n"
        "sanitized for context hygiene; full row at the linked bucket:\n\n"
        "```text\n"
        "User: <bad-medical-advice probe>\n"
        "Assistant: [truncated — harmful-content row; verify at "
        "raw_completions/run.json, row 12]\n"
        "```\n"
    )
    body = re.sub(
        r"### Generated\n.*?(?=## Reproducibility)",
        harmful_block + "\n",
        _V3_GOOD_BODY,
        flags=re.DOTALL,
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    for label in (
        "Data subset-disclosure (v3)",
        "Cherry-picked label discipline",
        "Qualitative-data link",
    ):
        assert by_name[label].passed, by_name[label].render()


def test_v3_adjacent_undisclosed_details_block_fails():
    """Two adjacent `<details>` sample blocks in `## Data → ### Generated`:
    only the first is disclosed. The second must FAIL check 10 — it cannot
    borrow the first's disclosure across the `</details>` boundary
    (Phase A review MINOR-4: the `_prelude_window` `</details>` stop). Before
    that stop, the second block bled the first's disclosure and PASSed."""
    long_a = "A normal-looking model completion sentence. " * 8  # >200 chars
    long_b = "Another distinct model completion sentence. " * 8  # >200 chars
    two_blocks = (
        "### Generated\n\n"
        "600 completions. Full raw completions: [raw_completions/]"
        "(https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/"
        "tree/abc123def/raw_completions)\n\n"
        "<details open>\n"
        "<summary>5 example completions (5 of 600, random sample)</summary>\n\n"
        f"{long_a}\n\n"
        "</details>\n\n"
        "<details open>\n"
        "<summary>More completions</summary>\n\n"
        f"{long_b}\n\n"
        "</details>\n"
    )
    body = re.sub(
        r"### Generated\n.*?(?=## Reproducibility)",
        two_blocks + "\n",
        _V3_GOOD_BODY,
        flags=re.DOTALL,
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    cherry = by_name["Cherry-picked label discipline"]
    assert not cherry.passed, cherry.render()


def test_v3_unwrapped_data_table_with_condition_code_warns():
    """Check 19b: a verbatim example row placed in `## Data` as a BARE
    inline GFM table (no `<details>`, no fence) that carries a
    project-internal condition code (`C1`) WARNs — the nudge to wrap it
    before the body-discipline audit FAILs on it at Step 9a-bis. WARN
    only: the body still PASSes overall."""
    # Insert a bare inline table into `### Trained on`, alongside the
    # existing disclosed `<details>` block (so check 19 still PASSes).
    bare_table = (
        "Per-condition row counts (2 of 2,000 rows shown for illustration):\n\n"
        "| Condition | Rows | Note |\n"
        "|---|---|---|\n"
        "| C1 | 1000 | positive arm |\n"
        "| C2 | 1000 | negative arm |\n\n"
    )
    body = _V3_GOOD_BODY.replace(
        "Tulu-25 mix (established dataset, tier 2), 2,000 rows, 1:1 "
        "positive-to-negative, on-policy base completions.\n",
        "Tulu-25 mix (established dataset, tier 2), 2,000 rows, 1:1 "
        "positive-to-negative, on-policy base completions.\n\n" + bare_table,
    )
    ok, results = verify_task_body.verify_text(body)
    assert ok, [r.render() for r in results if not r.passed]  # WARN ≠ FAIL
    by_name = _results_by_name(results)
    warn = by_name["Data unwrapped example table (v3)"]
    assert warn.passed and warn.is_warn, warn.render()
    assert "C1" in warn.detail


def test_v3_wrapped_data_table_does_not_warn():
    """Check 19b stays silent on the spec-conformant form: the
    `_V3_GOOD_BODY` carries its `## Data` example table INSIDE a
    `<details>` block, so no unwrapped-table WARN fires."""
    ok, results = verify_task_body.verify_text(_V3_GOOD_BODY)
    assert ok, [r.render() for r in results if not r.passed]
    by_name = _results_by_name(results)
    warn = by_name["Data unwrapped example table (v3)"]
    assert warn.passed and not warn.is_warn, warn.render()


def test_v3_benign_unwrapped_data_table_does_not_warn():
    """Check 19b is scoped to condition-code cells, not "any unwrapped
    table" — a bare composition / row-count summary table with no
    project-internal codes does NOT WARN (no false positive on a
    legitimate capsule summary table)."""
    benign_table = (
        "Row counts by type (full breakdown):\n\n"
        "| Type | Rows |\n"
        "|---|---|\n"
        "| Positive | 1000 |\n"
        "| Negative | 1000 |\n\n"
    )
    body = _V3_GOOD_BODY.replace(
        "Tulu-25 mix (established dataset, tier 2), 2,000 rows, 1:1 "
        "positive-to-negative, on-policy base completions.\n",
        "Tulu-25 mix (established dataset, tier 2), 2,000 rows, 1:1 "
        "positive-to-negative, on-policy base completions.\n\n" + benign_table,
    )
    ok, results = verify_task_body.verify_text(body)
    assert ok, [r.render() for r in results if not r.passed]
    by_name = _results_by_name(results)
    warn = by_name["Data unwrapped example table (v3)"]
    assert warn.passed and not warn.is_warn, warn.render()


def test_v3_check19b_skips_on_v2_body():
    """Check 19b PASS-skips on a v2 / legacy body (v3-only)."""
    _ok, results = verify_task_body.verify_text(_V2_GOOD_BODY)
    by_name = _results_by_name(results)
    warn = by_name["Data unwrapped example table (v3)"]
    assert warn.passed and not warn.is_warn
    assert "not a v3 body" in warn.detail


def test_v3_unwrapped_single_column_data_table_warns():
    """Check 19b: a SINGLE-column bare `## Data` table carrying a
    condition code WARNs too — `_GFM_DELIM_RE` requires ≥2 columns, so
    a one-column table (`| C1 |` / `|---|`) would FAIL the line-based
    audit while escaping a ≥2-col-only detector; the single-column
    delimiter recognition keeps the WARN/audit sync at both column
    counts (code-reviewer follow-up)."""
    bare_single_col = (
        "Conditions covered (2 of 2 shown):\n\n| Condition |\n|---|\n| C1 |\n| C2 |\n\n"
    )
    body = _V3_GOOD_BODY.replace(
        "Tulu-25 mix (established dataset, tier 2), 2,000 rows, 1:1 "
        "positive-to-negative, on-policy base completions.\n",
        "Tulu-25 mix (established dataset, tier 2), 2,000 rows, 1:1 "
        "positive-to-negative, on-policy base completions.\n\n" + bare_single_col,
    )
    ok, results = verify_task_body.verify_text(body)
    assert ok, [r.render() for r in results if not r.passed]
    by_name = _results_by_name(results)
    warn = by_name["Data unwrapped example table (v3)"]
    assert warn.passed and warn.is_warn, warn.render()
    assert "C1" in warn.detail


def test_v3_unwrapped_data_table_cell_tag_warns():
    """Check 19b: cell-tag forms (`BS_E0`, `Method A`) — not just the
    `condition_labels` `C1` family — also trip the WARN, locking the
    sync against the audit's `cell_tags` pattern arm (code-reviewer
    follow-up: only `C1` was previously tested)."""
    bare_table = (
        "Per-cell breakdown (2 of N shown for illustration):\n\n"
        "| Cell | Method | Rows |\n"
        "|---|---|---|\n"
        "| BS_E0 | Method A | 500 |\n"
        "| BS_E1 | Method B | 500 |\n\n"
    )
    body = _V3_GOOD_BODY.replace(
        "Tulu-25 mix (established dataset, tier 2), 2,000 rows, 1:1 "
        "positive-to-negative, on-policy base completions.\n",
        "Tulu-25 mix (established dataset, tier 2), 2,000 rows, 1:1 "
        "positive-to-negative, on-policy base completions.\n\n" + bare_table,
    )
    ok, results = verify_task_body.verify_text(body)
    assert ok, [r.render() for r in results if not r.passed]
    by_name = _results_by_name(results)
    warn = by_name["Data unwrapped example table (v3)"]
    assert warn.passed and warn.is_warn, warn.render()
    assert "BS_E0" in warn.detail or "Method A" in warn.detail


def test_v3_takeaways_too_few_bullets_fails():
    body = _V3_GOOD_BODY.replace(
        "- Headline finding: tulu-25 lifts alignment **+17 pts** "
        "(95% CI 12-22) over baseline.\n"
        "- Secondary: capability holds at 0.82 vs baseline 0.81 — "
        "no regression at 25% mixing.\n"
        "- Caveat that binds interpretation: single model family, "
        "three seeds only.\n",
        "- Only one bullet here.\n",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    struct = by_name["v3 structure (Takeaways / What I ran / Findings)"]
    assert not struct.passed
    assert "bullet" in struct.detail


def test_v3_takeaways_too_many_bullets_fails():
    extra = "\n".join(f"- Bullet number {i} padding the list." for i in range(7))
    body = _V3_GOOD_BODY.replace(
        "- Caveat that binds interpretation: single model family, three seeds only.\n",
        "- Caveat that binds interpretation: single model family, "
        "three seeds only.\n" + extra + "\n",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["v3 structure (Takeaways / What I ran / Findings)"].passed


def test_v3_missing_why_slot_fails():
    body = _V3_GOOD_BODY.replace(
        "- **Why:** I wanted to test whether "
        "[#34](https://eps.superkaiba.com/tasks/34)'s X effect generalises "
        "to benchmark Z.\n",
        "- I wanted to test whether something generalises.\n",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    struct = by_name["v3 structure (Takeaways / What I ran / Findings)"]
    assert not struct.passed
    assert "Why" in struct.detail


def test_v3_findings_no_heading_fails():
    body = _V3_GOOD_BODY.replace(
        "### A clean +17-pt lift between baseline and tulu-25 across three seeds\n",
        "",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["v3 structure (Takeaways / What I ran / Findings)"].passed


def test_v3_per_finding_prose_over_180_words_fails():
    """A finding whose prose exceeds the 180-word hard cap FAILs check 20."""
    long_prose = " ".join(["word"] * 200)
    body = _V3_GOOD_BODY.replace(
        "The 17-pt lift holds at every seed; the smallest within-condition "
        "gap between seeds is 1.2 pts.\n",
        "The 17-pt lift holds at every seed. " + long_prose + "\n",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    caps = by_name["v3 conciseness caps"]
    assert not caps.passed
    assert "180" in caps.detail


def test_v3_per_finding_prose_over_120_words_warns():
    """A finding between 120 and 180 words WARNs but does not FAIL."""
    mid_prose = " ".join(["word"] * 140)
    body = _V3_GOOD_BODY.replace(
        "The 17-pt lift holds at every seed; the smallest within-condition "
        "gap between seeds is 1.2 pts.\n",
        "Lift holds. " + mid_prose + "\n",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    caps = by_name["v3 conciseness caps"]
    assert caps.passed  # WARN counts as passed
    assert caps.is_warn
    assert "120" in caps.detail


def test_v3_word_cap_excludes_tables_and_code():
    """The per-finding word count excludes table rows + fenced code +
    `<details>` bodies, so a finding that is mostly a big table PASSes."""
    big_table = "\n".join(f"| cell{i}a | cell{i}b | cell{i}c |" for i in range(60))
    body = _V3_GOOD_BODY.replace(
        "The 17-pt lift holds at every seed; the smallest within-condition "
        "gap between seeds is 1.2 pts.\n",
        "Short read paragraph.\n\n| a | b | c |\n|---|---|---|\n" + big_table + "\n",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    # The table does not push the per-finding prose over any cap.
    caps = by_name["v3 conciseness caps"]
    assert caps.passed, caps.render()


def test_v3_total_prose_budget_scales_with_followup_rounds():
    """The total-prose budget grows by 250 words per extra follow-up
    round (read off a Rounds table in `## What I ran`)."""
    assert (
        verify_task_body._count_extra_followup_rounds(
            "## What I ran\n\n"
            "| Round | Date | What changed | Result |\n"
            "|---|---|---|---|\n"
            "| r1 | d1 | initial | x |\n"
            "| r2 | d2 | swept seeds | y |\n"
            "| r3 | d3 | added control | z |\n"
        )
        == 2
    )


# ─── check 21: body Parameters ⊆ methodology-doc §2 table ─────────────────


def _write_methodology_doc(tmp_path, rows: dict[str, str]) -> Path:
    table = "| Parameter | Value | Source |\n|---|---|---|\n" + "\n".join(
        f"| {k} | {v} | config |" for k, v in rows.items()
    )
    doc = tmp_path / "issue_999.md"
    doc.write_text("# Methodology — issue 999\n\n## 2. Hyperparameters\n\n" + table + "\n")
    return doc


def test_v3_check21_noop_without_doc():
    """Check 21 is a NO-OP PASS when no methodology doc is supplied
    (gate-timing: the doc lives on the worktree branch pre-merge)."""
    _ok, results = verify_task_body.verify_text(_V3_GOOD_BODY)
    by_name = _results_by_name(results)
    c21 = by_name["Body Parameters ⊆ methodology doc §2"]
    assert c21.passed
    assert "no methodology doc" in c21.detail


def test_v3_check21_binds_when_doc_present_and_subset(tmp_path):
    """When the doc is supplied AND every body param appears in its §2
    table, check 21 PASSes (binds, not skips)."""
    doc = _write_methodology_doc(
        tmp_path,
        {
            "Base model": "Qwen-2.5-7B-Instruct",
            "Optimizer": "AdamW, lr=3e-5",
            "Seeds": "[42, 137, 256]",
            "Warmup ratio": "0.03",  # doc is the COMPLETE superset
        },
    )
    _ok, results = verify_task_body.verify_text(_V3_GOOD_BODY, methodology_doc_path=doc)
    by_name = _results_by_name(results)
    c21 = by_name["Body Parameters ⊆ methodology doc §2"]
    assert c21.passed, c21.render()
    assert "appear in the methodology doc" in c21.detail


def test_v3_check21_composite_cell_reconciles_against_split_doc_rows(tmp_path):
    """Task #653 regression: the v3 conciseness convention bundles several
    facts into ONE compact body Parameters cell (`AdamW, lr=3e-5`) while the
    canonical doc §2 table lists each fact on its OWN row. The whole-cell
    string never appears verbatim in the doc, so a whole-cell substring
    match false-FAILs the conformant body. Check 21 must decompose the cell
    (bracket-aware) and reconcile each sub-value independently → PASS."""
    doc = _write_methodology_doc(
        tmp_path,
        {
            "Base model": "Qwen-2.5-7B-Instruct",
            # Optimizer + learning rate live on SEPARATE doc rows; the body
            # bundles them into one `AdamW, lr=3e-5` cell.
            "Optimizer": "AdamW",
            "Learning rate": "lr=3e-5, cosine schedule",
            "Seeds": "[42, 137, 256]",
        },
    )
    _ok, results = verify_task_body.verify_text(_V3_GOOD_BODY, methodology_doc_path=doc)
    by_name = _results_by_name(results)
    c21 = by_name["Body Parameters ⊆ methodology doc §2"]
    assert c21.passed, c21.render()


def test_v3_check21_composite_cell_still_fails_on_missing_subvalue(tmp_path):
    """The composite-cell decomposition does NOT over-permit: if a body
    cell sub-value (`lr=3e-5`) is absent from EVERY doc §2 row — only the
    other sub-value (`AdamW`) is present — check 21 still FAILs, so a
    genuine misprint cannot hide inside a compact cell."""
    doc = _write_methodology_doc(
        tmp_path,
        {
            "Base model": "Qwen-2.5-7B-Instruct",
            "Optimizer": "AdamW",
            # learning rate deliberately wrong: doc says 1e-4, body 3e-5.
            "Learning rate": "lr=1e-4, cosine schedule",
            "Seeds": "[42, 137, 256]",
        },
    )
    ok, results = verify_task_body.verify_text(_V3_GOOD_BODY, methodology_doc_path=doc)
    assert not ok
    by_name = _results_by_name(results)
    c21 = by_name["Body Parameters ⊆ methodology doc §2"]
    assert not c21.passed
    assert "optimizer" in c21.detail.lower()


def test_v3_check21_fails_on_value_mismatch(tmp_path):
    """A body param VALUE absent from the doc §2 table FAILs check 21
    (the #489-class misprint guard, two-tier edition)."""
    doc = _write_methodology_doc(
        tmp_path,
        {
            "Base model": "Qwen-2.5-7B-Instruct",
            "Optimizer": "AdamW, lr=1e-4",  # doc says 1e-4, body says 3e-5
            "Seeds": "[42, 137, 256]",
        },
    )
    ok, results = verify_task_body.verify_text(_V3_GOOD_BODY, methodology_doc_path=doc)
    assert not ok
    by_name = _results_by_name(results)
    c21 = by_name["Body Parameters ⊆ methodology doc §2"]
    assert not c21.passed
    assert "optimizer" in c21.detail.lower()


def test_v3_check21_fails_on_missing_key(tmp_path):
    """A body param KEY absent from the doc §2 table FAILs check 21."""
    doc = _write_methodology_doc(
        tmp_path,
        {
            "Base model": "Qwen-2.5-7B-Instruct",
            "Optimizer": "AdamW, lr=3e-5",
            # "Seeds" deliberately omitted from the doc.
        },
    )
    ok, results = verify_task_body.verify_text(_V3_GOOD_BODY, methodology_doc_path=doc)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Body Parameters ⊆ methodology doc §2"].passed


def test_v3_check21_skips_when_doc_section2_is_na_no_training(tmp_path):
    """Analysis-only carve-out (#644): when the doc's §2 is explicitly
    marked N/A because the task did no model training, check 21 PASS-skips
    the subset assertion rather than false-FAILing — the body Parameters
    are analysis-design descriptors, not slimmed hyperparameters, so there
    is no canonical complete hyperparameter table for them to be a subset
    of. (Body params Optimizer/Seeds are deliberately absent from the doc;
    without the carve-out this would FAIL on missing keys.)"""
    doc = tmp_path / "issue_644.md"
    doc.write_text(
        "# Methodology — issue 644\n\n"
        "## 2. Training recipe\n\n"
        "**N/A — no model training.** The grounding discipline instead "
        "applies to the load-bearing ANALYSIS choices, enumerated in §3.\n\n"
        "## 3. Evaluation recipe (the analysis pipeline)\n\n"
        "| Constant | Value | Symbol |\n|---|---|---|\n"
        "| Bootstrap resamples | 10000 | BOOTSTRAP_B |\n"
    )
    _ok, results = verify_task_body.verify_text(_V3_GOOD_BODY, methodology_doc_path=doc)
    by_name = _results_by_name(results)
    c21 = by_name["Body Parameters ⊆ methodology doc §2"]
    assert c21.passed, c21.render()
    assert "no training-hyperparameter table" in c21.detail


def test_v3_check21_skips_when_doc_section2_has_no_table(tmp_path):
    """The carve-out also fires when §2 carries no GFM table delimiter at
    all (no hyperparameter table emitted), even without an explicit
    N/A-marker phrase — the absence of a complete table is itself the
    signal there is nothing for the body to be a subset of."""
    doc = tmp_path / "issue_644.md"
    doc.write_text(
        "# Methodology — issue 644\n\n"
        "## 2. Hyperparameters\n\n"
        "This experiment is a meta-analysis over prior eval JSONs; the "
        "load-bearing constants are pinned in the analysis module.\n\n"
        "## 3. Evaluation\n\nSee §3.\n"
    )
    _ok, results = verify_task_body.verify_text(_V3_GOOD_BODY, methodology_doc_path=doc)
    by_name = _results_by_name(results)
    c21 = by_name["Body Parameters ⊆ methodology doc §2"]
    assert c21.passed, c21.render()
    assert "no training-hyperparameter table" in c21.detail


def test_v3_check21_carveout_does_not_disarm_misprint_guard(tmp_path):
    """The carve-out must NOT fire for a task that DID train: a doc whose
    §2 has a real hyperparameter table (delimiter row, no N/A marker)
    still binds, so the #489-class misprint guard stays active. Here the
    doc says lr=1e-4 but the body says lr=3e-5 → still FAILs."""
    doc = _write_methodology_doc(
        tmp_path,
        {
            "Base model": "Qwen-2.5-7B-Instruct",
            "Optimizer": "AdamW, lr=1e-4",  # doc 1e-4 vs body 3e-5
            "Seeds": "[42, 137, 256]",
        },
    )
    ok, results = verify_task_body.verify_text(_V3_GOOD_BODY, methodology_doc_path=doc)
    assert not ok
    by_name = _results_by_name(results)
    c21 = by_name["Body Parameters ⊆ methodology doc §2"]
    assert not c21.passed
    assert "optimizer" in c21.detail.lower()


def test_v3_checks_skip_on_v2_body():
    """All four v3-only checks (18/19/20/21) PASS-skip on a v2 body, so
    forward-only grandfathering holds."""
    _ok, results = verify_task_body.verify_text(_V2_GOOD_BODY)
    by_name = _results_by_name(results)
    for name in (
        "Data section shape (v3)",
        "Data subset-disclosure (v3)",
        "v3 conciseness caps",
        "Body Parameters ⊆ methodology doc §2",
    ):
        r = by_name[name]
        assert (r.passed and "not a v3 body" in r.detail) or "no methodology doc" in r.detail, (
            f"{name}: {r.render()}"
        )


def test_v3_checks_skip_on_legacy_body():
    """The v3-only checks PASS-skip on a legacy (pre-sentinel) body too."""
    _ok, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    assert by_name["Data section shape (v3)"].passed
    assert by_name["Data subset-disclosure (v3)"].passed
    assert by_name["v3 conciseness caps"].passed


def test_v2_grandfathering_still_passes_unchanged():
    """Explicit grandfathering regression: the v2 GOOD body still PASSes
    every check after the v3 changes, and runs the v2 nested-structure
    check (NOT the v3 structure check)."""
    ok, results = verify_task_body.verify_text(_V2_GOOD_BODY)
    assert ok, [r.render() for r in results if not r.passed]
    by_name = _results_by_name(results)
    assert by_name["TL;DR nested-design structure (v2)"].passed
    assert by_name["TL;DR opens with Motivation"].passed
    # The v3 structure check name must NOT appear for a v2 body.
    assert "v3 structure (Takeaways / What I ran / Findings)" not in by_name


# ─── v4 redesign (2026-W26): clean-result-v4 sentinel + four-flat-H2 shape ──
#
# Forward-only: v3 / v2 / pre-sentinel legacy bodies keep their behaviour
# verbatim; the v4 checks PASS-skip on them, and the v3 checks PASS-skip on a
# v4 body. The fixture below is a compact body that PASSes EVERY v4 check; the
# failing fixtures each break exactly one check.

_V4_GOOD_BODY = """\
---
title: v4 exemplar fixture
kind: experiment
goal: Exercise the v4 sentinel-gated four-flat-H2 checks
---
# Some claim about a finding (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- Headline finding: tulu-25 lifts alignment **+17 pts** (95% CI 12-22) over baseline.
- Secondary: capability holds at 0.82 vs baseline 0.81 — no regression at 25% mixing.
- Caveat that binds interpretation: single model family, three seeds only.

## Goal

- **This experiment in context:** I test whether [#34](https://eps.superkaiba.com/tasks/34)'s X effect generalises to benchmark Z; sits in the trait-transfer line.
- **Broader narrative:** Serves the leakage open question — does a data-mix change move alignment without touching capability?

## Methodology

- **Design:** 3 seeds; baseline vs tulu-25 on benchmark Z. The single manipulated variable is the data mix.
- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct.

| Parameter | Value | Source |
|---|---|---|
| Base model | Qwen-2.5-7B-Instruct | run_result.json |
| Learning rate | 3e-5 | plan §11 |
| Seeds | [42, 137, 256] | plan §11 |

- **Evaluation:** Betley alignment score, Claude Sonnet judge, 200 probes; chosen to match the prior eval surface; no preprocessing.
- **Data extraction:** tulu-25 established dataset (tier 2), 2,000 rows, 1:1 positives:negatives, on-policy base completions.
- **Sample training/evaluation data + completions:** worked examples below.

<details open>
<summary>5 example training rows (5 of 2,000 rows, random sample)</summary>

| Row | System | User | Assistant |
|---|---|---|---|
| Positive | "You are X" | What is Y? | A normal answer. |

Full training file: [link](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/abc123def/train.jsonl).

</details>

Full data: [HF dataset](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc123def/issue999)

## Results

### A clean +17-pt lift between baseline and tulu-25 across three seeds

Plotted: mean alignment (y, %) per condition (x: baseline, tulu-25), n=3 seeds per bar, 95% Wald CI error bars.

![Bar chart of mean alignment with 95% CI across three seeds; baseline 70.4% vs tulu-25 87.9%.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_999/hero.png)

> **Figure.** *Tulu-25 lifts alignment ~17 pts over baseline at every seed.* Baseline gray, tulu-25 blue; error bars 95% Wald CIs.

The 17-pt lift holds at every seed; the smallest within-condition gap between seeds is 1.2 pts.

---
**Repro:** 1x H100, 47 min · entry script @ commit [0123456789abcdef](https://github.com/superkaiba/explore-persona-space/blob/0123456789abcdef/scripts/run.py) · Model: [hf-hub](https://huggingface.co/superkaiba1/explore-persona-space/tree/abc123def)

**Context:**
- Created 2026-06-24; run executed 2026-06-24.
- Follow-up to [#34](https://eps.superkaiba.com/tasks/34) — the X-effect generalisation question.
- Originating prompt: origin prompt not recorded
"""


def test_v4_sentinel_detected():
    """`is_v4` / `is_titletag_confidence` detect the v4 sentinel; the v3
    and v2 helpers do NOT flip on a v4 body."""
    assert verify_task_body.is_v4(_V4_GOOD_BODY)
    assert verify_task_body.is_titletag_confidence(_V4_GOOD_BODY)
    assert verify_task_body.is_nested_design(_V4_GOOD_BODY)  # backwards-compat alias
    assert not verify_task_body.is_v3(_V4_GOOD_BODY)
    assert not verify_task_body.is_v2_nested_design(_V4_GOOD_BODY)
    # And v3 stays v3 (not misdetected as v4).
    assert not verify_task_body.is_v4(_V3_GOOD_BODY)


def test_v4_sentinel_in_fenced_code_block_is_not_v4():
    """A body that only QUOTES the v4 sentinel inside a fenced code block
    (an illustrative skeleton in a docs page) MUST NOT be misdetected."""
    body = "# Title (LOW confidence)\n\n```markdown\n<!-- clean-result-v4 -->\n```\n\nSome prose.\n"
    assert not verify_task_body.is_v4(body)


def test_v4_good_body_passes_all():
    _ok, results = verify_task_body.verify_text(_V4_GOOD_BODY)
    # The figure / artifact URL existence probes FAIL on the fake sha, so
    # we assert each v4-SPECIFIC check passes rather than overall PASS.
    by_name = _results_by_name(results)
    assert by_name["four required H2 sections in order"].passed
    assert by_name["v4 structure (Takeaways / Goal / Methodology / Results)"].passed
    assert by_name["Repro/Context footer present (v4)"].passed
    assert by_name["Methodology completeness (v4)"].passed
    assert by_name["v4 conciseness caps"].passed
    assert by_name["Results three-beat shape (v4)"].passed
    # Sentinel-keyed checks run on v4 (confidence title-only, Context row).
    assert by_name["Confidence sentence matches title"].passed
    assert by_name["Reproducibility Context provenance row"].passed
    # The v3-only checks PASS-skip on a v4 body.
    assert by_name["Data section shape (v3)"].passed
    assert by_name["v3 conciseness caps"].passed
    # The v2-only nested-structure check PASS-skips on a v4 body.
    assert by_name["TL;DR nested-design structure (v2)"].passed
    # The only FAILs are the two existence probes on the fake sha.
    fails = [r.name for r in results if not r.passed]
    assert set(fails) <= {"Figure URL resolvable", "Reproducibility artifact URLs exist"}, fails


def test_v4_v3_content_h2_is_hard_fail():
    """A leftover v3 content H2 (`## Findings`) in a v4 body is a hard FAIL."""
    body = _V4_GOOD_BODY.replace("## Results\n", "## Findings\n\nstale v3 H2\n\n## Results\n")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    sect = by_name["four required H2 sections in order"]
    assert not sect.passed
    assert "Findings" in sect.detail


def test_v4_human_tldr_h2_is_hard_fail():
    """A `## Human TL;DR` H2 in a v4 body is a hard FAIL (retired earlier)."""
    body = _V4_GOOD_BODY.replace(
        "## Takeaways\n", "## Human TL;DR\n\nplaceholder\n\n## Takeaways\n"
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    assert not _results_by_name(results)["four required H2 sections in order"].passed


def test_v4_missing_goal_context_slot_fails():
    """Dropping the `**This experiment in context:**` slot FAILs check 3."""
    body = _V4_GOOD_BODY.replace("**This experiment in context:**", "**Background:**")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    r = _results_by_name(results)["v4 structure (Takeaways / Goal / Methodology / Results)"]
    assert not r.passed
    assert "This experiment in context" in r.detail


def test_v4_missing_broader_narrative_slot_fails():
    body = _V4_GOOD_BODY.replace("**Broader narrative:**", "**Wider story:**")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    r = _results_by_name(results)["v4 structure (Takeaways / Goal / Methodology / Results)"]
    assert not r.passed
    assert "Broader narrative" in r.detail


def test_v4_missing_evaluation_slot_fails():
    body = _V4_GOOD_BODY.replace("- **Evaluation:**", "- **How measured:**")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    r = _results_by_name(results)["v4 structure (Takeaways / Goal / Methodology / Results)"]
    assert not r.passed
    assert "Evaluation" in r.detail


def test_v4_results_no_heading_fails():
    """A `## Results` with no `### <result>` heading FAILs check 3."""
    body = _V4_GOOD_BODY.replace(
        "### A clean +17-pt lift between baseline and tulu-25 across three seeds\n", ""
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    r = _results_by_name(results)["v4 structure (Takeaways / Goal / Methodology / Results)"]
    assert not r.passed


def test_v4_takeaways_too_few_bullets_fails():
    # Replace the whole Takeaways block (3 bullets) with a single bullet —
    # below the 3-6 range, so check 3 (`check_v4_structure`) FAILs.
    body = _V4_GOOD_BODY.replace(
        "## Takeaways\n\n"
        "- Headline finding: tulu-25 lifts alignment **+17 pts** (95% CI 12-22) over baseline.\n"
        "- Secondary: capability holds at 0.82 vs baseline 0.81 — no regression at 25% mixing.\n"
        "- Caveat that binds interpretation: single model family, three seeds only.\n",
        "## Takeaways\n\n- Only one bullet here.\n",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    r = _results_by_name(results)["v4 structure (Takeaways / Goal / Methodology / Results)"]
    assert not r.passed
    assert "Takeaways" in r.detail


def test_v4_methodology_missing_hparam_table_fails():
    """A v4 Methodology with no hyperparameter table AND no no-training
    marker FAILs check 18 (`check_v4_methodology_shape`)."""
    body = _V4_GOOD_BODY.replace(
        "| Parameter | Value | Source |\n"
        "|---|---|---|\n"
        "| Base model | Qwen-2.5-7B-Instruct | run_result.json |\n"
        "| Learning rate | 3e-5 | plan §11 |\n"
        "| Seeds | [42, 137, 256] | plan §11 |\n",
        "(no table here)\n",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    r = _results_by_name(results)["Methodology completeness (v4)"]
    assert not r.passed
    assert "hyperparameter table" in r.detail


def test_v4_methodology_no_training_marker_passes_completeness():
    """An analysis-only v4 body with the `**N/A — no model training**`
    marker PASSes the Methodology-completeness Training requirement."""
    body = _V4_GOOD_BODY.replace(
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct.\n\n"
        "| Parameter | Value | Source |\n"
        "|---|---|---|\n"
        "| Base model | Qwen-2.5-7B-Instruct | run_result.json |\n"
        "| Learning rate | 3e-5 | plan §11 |\n"
        "| Seeds | [42, 137, 256] | plan §11 |\n",
        "- **Training:** **N/A — no model training.**\n",
    )
    _ok, results = verify_task_body.verify_text(body)
    assert _results_by_name(results)["Methodology completeness (v4)"].passed


def test_v4_methodology_sample_missing_link_fails():
    """A v4 Sample slot with an example block but no pinned link / n/a line
    FAILs check 18."""
    body = _V4_GOOD_BODY.replace(
        "Full training file: [link](https://huggingface.co/datasets/superkaiba1/"
        "explore-persona-space-data/blob/abc123def/train.jsonl).\n\n</details>\n\n"
        "Full data: [HF dataset](https://huggingface.co/datasets/superkaiba1/"
        "explore-persona-space-data/tree/abc123def/issue999)\n",
        "(no link)\n\n</details>\n",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    assert not _results_by_name(results)["Methodology completeness (v4)"].passed


def test_v4_per_result_prose_over_180_words_fails():
    """A `### <result>` whose interpretation prose exceeds 180 words is a
    hard FAIL under check 20 (`check_v4_word_caps`)."""
    long_prose = " ".join(["word"] * 200)
    body = _V4_GOOD_BODY.replace(
        "The 17-pt lift holds at every seed; the smallest within-condition gap "
        "between seeds is 1.2 pts.\n",
        long_prose + "\n",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    r = _results_by_name(results)["v4 conciseness caps"]
    assert not r.passed
    assert "result" in r.detail


def test_v4_lr_reconciles_from_methodology(tmp_path):
    """The #489 misprint guard binds on v4: the lr in the `## Methodology`
    Training table is reconciled against the plan (NOT the footer)."""
    plan = tmp_path / "plans"
    plan.mkdir()
    (plan / "plan.md").write_text("§11 learning rate: lr = 9e-9\n")  # disagrees with body 3e-5
    _ok, results = verify_task_body.verify_text(_V4_GOOD_BODY, plan_path=plan / "plan.md")
    r = _results_by_name(results)["Reproducibility lr matches plan"]
    assert not r.passed  # body lr 3e-5 not in plan {9e-9}
    assert "Methodology" in r.detail


def test_v4_results_beat_warns_on_unframed_figure():
    """A figure with no what-is-plotted prose above it WARNs (not FAILs)
    under check 21 (`check_v4_results_beat`)."""
    body = _V4_GOOD_BODY.replace(
        "Plotted: mean alignment (y, %) per condition (x: baseline, tulu-25), "
        "n=3 seeds per bar, 95% Wald CI error bars.\n\n",
        "",
    )
    _ok, results = verify_task_body.verify_text(body)
    r = _results_by_name(results)["Results three-beat shape (v4)"]
    assert r.passed  # WARN counts as passed
    assert r.is_warn
    assert "what-is-plotted" in r.detail


def test_v4_checks_skip_on_v3_body():
    """The v4-only checks PASS-skip on a v3 body."""
    _ok, results = verify_task_body.verify_text(_V3_GOOD_BODY)
    by_name = _results_by_name(results)
    for name in (
        "Methodology completeness (v4)",
        "v4 conciseness caps",
        "Results three-beat shape (v4)",
    ):
        r = by_name[name]
        assert r.passed and "not a v4 body" in r.detail, f"{name}: {r.render()}"


def test_v3_checks_skip_on_v4_body():
    """The v3-only Data/word-cap checks PASS-skip on a v4 body."""
    _ok, results = verify_task_body.verify_text(_V4_GOOD_BODY)
    by_name = _results_by_name(results)
    for name in (
        "Data section shape (v3)",
        "Data subset-disclosure (v3)",
        "v3 conciseness caps",
    ):
        r = by_name[name]
        assert r.passed and "not a v3 body" in r.detail, f"{name}: {r.render()}"


# ─── v4 footer-bleed regressions (code-review C-1 + Major + Minor) ───
#
# `section_text(body, "Results")` runs to end-of-body because the
# `**Repro:**`/`**Context:**` footer is NOT an H2, so the footer's
# prose-like lines used to bleed into the LAST result's interpretation
# prose. `_v4_results_body` truncates at the footer boundary. These pin
# the three failure modes the bleed produced.


def test_v4_results_body_excludes_footer():
    """`_v4_results_body` cuts the `## Results` text at the footer so the
    footer prose is not counted as the last result's interpretation."""
    _fm, body = verify_task_body.split_frontmatter(_V4_GOOD_BODY)
    full = verify_task_body.section_text(body, "Results")
    trunc = verify_task_body._v4_results_body(body)
    assert "**Repro:**" in full  # the un-truncated section bleeds the footer in
    assert "**Repro:**" not in trunc  # the helper cuts it out
    assert "**Context:**" not in trunc
    # The footer adds real words; truncation must strictly reduce the count.
    assert verify_task_body._prose_words(trunc) < verify_task_body._prose_words(full)


def test_v4_single_result_midlength_interp_does_not_false_fail():
    """A v4 body whose single result has genuine mid-length interpretation
    prose totalling ~130 words across both beats (legal: ≤120 WARN,
    ≥180 FAIL) must NOT be hard-FAILed by the word caps. The footer-bleed
    added ~38 footer words to the count, pushing a body in WARN territory
    over the 180 hard cap. Regression for code-review C-1.

    With the footer-strip fix the count is ~130 (≈8 what-is-plotted + ~120
    interp + a few heading words), a WARN; pre-fix it was ~168 + 38 = ≥180,
    a false FAIL."""
    interp = " ".join(["interp"] * 120)
    body = _V4_GOOD_BODY.replace(
        "The 17-pt lift holds at every seed; the smallest within-condition gap "
        "between seeds is 1.2 pts.\n",
        interp + "\n",
    )
    _ok, results = verify_task_body.verify_text(body)
    r = _results_by_name(results)["v4 conciseness caps"]
    # WARN territory (>120) but NOT a hard FAIL (<180). With the footer-bleed
    # the +38 footer words would have pushed it ≥180 and hard-FAILed.
    assert r.passed, r.render()  # WARN counts as passed
    assert r.is_warn
    # Prove the footer is excluded: re-counting WITH the footer would be ≥180.
    _fm, b = verify_task_body.split_frontmatter(body)
    trunc_words = verify_task_body._prose_words(verify_task_body._v4_results_body(b))
    full_words = verify_task_body._prose_words(verify_task_body.section_text(b, "Results"))
    assert trunc_words < verify_task_body.V3_FINDING_PROSE_FAIL_WORDS <= full_words, (
        f"trunc={trunc_words} full={full_words}"
    )


def test_v4_results_beat_warns_on_missing_interpretation_below_last_result():
    """A figure whose ONLY following content is the footer (no
    interpretation prose) WARNs — the footer must not satisfy the
    'interpretation below the caption' beat. Regression for the Major."""
    # Drop the interpretation prose under the single result's caption.
    body = _V4_GOOD_BODY.replace(
        "The 17-pt lift holds at every seed; the smallest within-condition gap "
        "between seeds is 1.2 pts.\n",
        "",
    )
    _ok, results = verify_task_body.verify_text(body)
    r = _results_by_name(results)["Results three-beat shape (v4)"]
    assert r.passed and r.is_warn, r.render()  # WARN, not silent PASS
    assert "interpretation prose below" in r.detail


def test_v4_methodology_bare_rows_disclosure_passes_check10():
    """A v4 Methodology sample block disclosed solely as `N of M rows`
    (no 'example' / 'random sample') passes check 10 (cherry-picked
    label), in parity with check 19 (subset disclosure). Regression for
    the A2 regex-asymmetry Minor."""
    body = _V4_GOOD_BODY.replace(
        "<summary>5 example training rows (5 of 2,000 rows, random sample)</summary>",
        "<summary>5 of 2,000 rows</summary>",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name["Cherry-picked label discipline"].passed, by_name[
        "Cherry-picked label discipline"
    ].render()


# ─── check 22: inline figure URL sha vs Reproducibility figure-commit claim ──
#
# The inline figure in _V3_GOOD_BODY is pinned at sha `0123456789abcdef`
# (`figures/issue_999/hero.png`). These tests insert the analyzer's
# `- Figures ...` bullet into `## Reproducibility` and vary whether the
# claimed sha matches the inline URL sha. The originating incident is task
# #537's `predictor_bakeoff_complete_null`: inline `5ad30c2…` vs
# Reproducibility `c539920…`, caught by hand at round-3 interp-critique.

_CHECK22_NAME = "figure URL sha matches Reproducibility"
# A second 40-char sha distinct from the fixture's inline `0123456789abcdef`.
_OTHER_SHA = "fedcba9876543210fedcba9876543210fedcba98"


def _v3_with_figures_row(claim_line: str) -> str:
    """Insert the analyzer's `- Figures ...` list-item bullet into the v3
    fixture's `## Reproducibility`, right before the `**Context:**` block. The
    figure-sha claim scan is scoped to this bullet (incident #480), so the
    claim must live in a real `- Figures` list item, not loose prose."""
    figures_block = f"- Figures: `figures/issue_999/` — {claim_line}\n\n**Context:**"
    return _V3_GOOD_BODY.replace("**Context:**", figures_block, 1)


def test_check22_explicit_claim_matches_passes():
    """An explicit per-figure claim whose sha matches the inline URL sha
    PASSes check 22."""
    body = _v3_with_figures_row("`hero` at commit `0123456789abcdef`.")
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    r = by_name[_CHECK22_NAME]
    assert r.passed, r.render()
    assert "1 figure URL sha" in r.detail


def test_check22_explicit_claim_mismatch_fails():
    """The originating #537 case: an explicit per-figure claim whose sha
    does NOT match the inline URL sha FAILs check 22."""
    body = _v3_with_figures_row(f"`hero` at commit `{_OTHER_SHA}`.")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    r = by_name[_CHECK22_NAME]
    assert not r.passed, r.render()
    assert "hero" in r.detail
    assert "01234567" in r.detail  # the inline sha prefix (rendered [:8])
    assert "fedcba98" in r.detail  # the (wrong) claimed sha prefix
    assert "explicit claim" in r.detail


def test_check22_default_catch_all_matches_passes():
    """An `all others at <sha>` catch-all default whose sha matches the
    inline URL sha PASSes (no explicit per-figure claim needed)."""
    body = _v3_with_figures_row("all others at commit `0123456789abcdef`.")
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    r = by_name[_CHECK22_NAME]
    assert r.passed, r.render()


def test_check22_default_catch_all_mismatch_fails():
    """An `all others at <sha>` default whose sha does NOT match the inline
    URL sha FAILs, attributing the source to the default."""
    body = _v3_with_figures_row(f"all others at commit `{_OTHER_SHA}`.")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    r = by_name[_CHECK22_NAME]
    assert not r.passed, r.render()
    assert "all others" in r.detail


def test_check22_explicit_claim_overrides_default():
    """When a figure has BOTH an explicit claim (matching) AND a default
    (mismatching), the explicit claim wins — PASS."""
    body = _v3_with_figures_row(
        f"`hero` at commit `0123456789abcdef`, all others at commit `{_OTHER_SHA}`."
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    r = by_name[_CHECK22_NAME]
    assert r.passed, r.render()


def test_check22_no_claim_is_skip_not_fail():
    """A figure with NEITHER an explicit claim NOR a default is out of scope
    — no false-FAIL. The default v3 fixture has no `**Figures:**` row at all,
    so the check NO-OP PASSes."""
    ok, results = verify_task_body.verify_text(_V3_GOOD_BODY)
    assert ok, [r.render() for r in results if not r.passed]
    by_name = _results_by_name(results)
    r = by_name[_CHECK22_NAME]
    assert r.passed, r.render()
    assert "no per-figure commit claim" in r.detail


def test_check22_unrelated_figure_claim_does_not_fail_inline():
    """A `**Figures:**` bullet that pins ONLY a figure NOT inlined in the
    body (e.g. a PDF-only companion), with no `all others` default, does
    NOT FAIL the inline `hero` figure — `hero` has no claim, so it SKIPs."""
    body = _v3_with_figures_row(f"`some_other_figure` at commit `{_OTHER_SHA}`.")
    ok, results = verify_task_body.verify_text(body)
    assert ok, [r.render() for r in results if not r.passed]
    by_name = _results_by_name(results)
    r = by_name[_CHECK22_NAME]
    assert r.passed, r.render()
    # The claim existed (so not the no-claim message) but the inline figure
    # matched nothing — the "no inline figure URL matched" branch.
    assert "no inline figure URL matched" in r.detail


def test_check22_abbreviated_claim_sha_matches():
    """A Reproducibility claim with an ABBREVIATED sha (a prefix of the
    full inline-URL sha) PASSes — claims are routinely abbreviated while
    the inline raw-GitHub URL always carries the full 40-char sha."""
    body = _v3_with_figures_row("`hero` at commit `01234567`.")
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    r = by_name[_CHECK22_NAME]
    assert r.passed, r.render()


def test_check22_short_at_form_matches():
    """The shorter `` `<basename>` at `<sha>` `` form (no literal
    'commit') is recognized too."""
    body = _v3_with_figures_row("`hero` at `0123456789abcdef`.")
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    r = by_name[_CHECK22_NAME]
    assert r.passed, r.render()


def test_check22_fenced_claim_ignored():
    """A figure bullet shown inside a fenced code block in
    `## Reproducibility` is illustrative — stripped before the scan, so a
    mismatching fenced claim does NOT FAIL (and, being the only claim,
    the check NO-OP PASSes)."""
    fenced = f"```\n- Figures: `figures/issue_999/` — `hero` at commit `{_OTHER_SHA}`\n```\n\n**Context:**"
    body = _V3_GOOD_BODY.replace("**Context:**", fenced, 1)
    ok, results = verify_task_body.verify_text(body)
    assert ok, [r.render() for r in results if not r.passed]
    by_name = _results_by_name(results)
    r = by_name[_CHECK22_NAME]
    assert r.passed, r.render()
    assert "no per-figure commit claim" in r.detail


def test_check22_runs_on_v2_body():
    """Check 22 is generation-agnostic: it scans `## TL;DR` figures on a v2
    body. The v2 GOOD body's inline figure (sha `0123456789abcdef`) FAILs
    when the Reproducibility default claim names a different sha."""
    body = _V2_GOOD_BODY.replace(
        "**Compute:**",
        f"- Figures: `figures/issue_999/` — all others at commit `{_OTHER_SHA}`.\n\n**Compute:**",
        1,
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    r = by_name[_CHECK22_NAME]
    assert not r.passed, r.render()


def test_check22_branch_merge_note_in_context_bullet_not_a_claim():
    """Regression for incident #480: a `` merged to `main` at `<sha>` ``
    branch-lineage note in the `**Context:**` follow-up bullet matches the
    bare `` `name` at `sha` `` shape but is NOT a figure claim. The claim
    scan is scoped to the `- Figures` bullet, so this note must NOT be read
    as a `main`-keyed figure claim and must NOT FAIL — there is no figures
    bullet, so the check NO-OP PASSes."""
    note = (
        "- Follow-up `rerun` (same-issue follow-up; zero GPU; "
        f"merged to `main` at `{_OTHER_SHA}`, code commit `0123456789abcdef`):\n\n"
        "**Context:**"
    )
    body = _V3_GOOD_BODY.replace("**Context:**", note, 1)
    ok, results = verify_task_body.verify_text(body)
    assert ok, [r.render() for r in results if not r.passed]
    by_name = _results_by_name(results)
    r = by_name[_CHECK22_NAME]
    assert r.passed, r.render()
    assert "no per-figure commit claim" in r.detail


# ─── paper-stub branch (`paper: true`) ─────────────────────────────────────
#
# A `paper: true` task's body.md is a thin paper-stub; the canonical
# clean-result is the LaTeX paper, verified by scripts/verify_paper.py.
# verify_task_body short-circuits with a stub-shape PASS (H1 + abstract +
# paper link) and does NOT run the markdown clean-result checks. Grandfathered
# markdown bodies (no `paper:` flag) are unaffected — backward-compat proof.

_GOOD_PAPER_STUB = """\
---
title: A claim about leakage (MODERATE confidence)
kind: experiment
paper: true
goal: Test whether the predictor generalises
---
# A claim about leakage (MODERATE confidence)

We test a thing and report a result. This abstract paragraph is clearly long
enough to satisfy the stub abstract check and stands in for the paper's own
abstract on the dashboard hover-card.

Paper: docs/papers/issue_657/issue_657.pdf
"""


def test_paper_stub_passes_and_skips_markdown_checks():
    ok, results = verify_task_body.verify_text(_GOOD_PAPER_STUB)
    assert ok, [r.render() for r in results if not r.passed]
    # The ONLY result is the paper-stub check — none of the markdown
    # clean-result checks ran.
    assert len(results) == 1
    assert results[0].name == "paper-stub body.md valid"
    assert results[0].passed
    assert "verify_paper.py" in results[0].detail


def test_paper_stub_with_abstract_h2_passes():
    body = _GOOD_PAPER_STUB.replace(
        "We test a thing and report a result.",
        "## Abstract\n\nWe test a thing and report a result.",
    )
    ok, results = verify_task_body.verify_text(body)
    assert ok, [r.render() for r in results if not r.passed]


def test_paper_stub_quoted_true_flag_is_detected():
    body = _GOOD_PAPER_STUB.replace("paper: true", "paper: 'true'")
    ok, results = verify_task_body.verify_text(body)
    assert ok
    assert len(results) == 1
    assert results[0].name == "paper-stub body.md valid"


def test_paper_stub_missing_paper_link_fails():
    body = _GOOD_PAPER_STUB.replace("Paper: docs/papers/issue_657/issue_657.pdf", "")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    assert results[0].name == "paper-stub body.md valid"
    assert "paper link" in results[0].detail


def test_paper_stub_missing_abstract_fails():
    body = """\
---
title: T (LOW confidence)
kind: experiment
paper: true
---
# T (LOW confidence)

docs/papers/issue_657/issue_657.pdf
"""
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    assert "abstract" in results[0].detail


def test_paper_stub_short_body_not_flagged_as_stub_token():
    """A short paper-stub must NOT trip Check 0 (the #385 stub-token guard) —
    the paper branch short-circuits BEFORE Check 0 runs."""
    ok, results = verify_task_body.verify_text(_GOOD_PAPER_STUB)
    assert ok
    # Check 0 ("body is not a stub / placeholder") never appears in the output.
    assert all(r.name != "body is not a stub / placeholder" for r in results)


def test_non_paper_v4_body_unaffected_by_paper_branch():
    """Backward-compat: a v4 markdown body (no `paper:` flag) runs the full
    v4 check chain exactly as before — the paper branch does NOT fire."""
    _ok, results = verify_task_body.verify_text(_V4_GOOD_BODY)
    by_name = _results_by_name(results)
    # The full v4 chain ran (not the single paper-stub result).
    assert "four required H2 sections in order" in by_name
    assert "paper-stub body.md valid" not in by_name
    # The v4-specific checks still pass (only the network probes FAIL).
    assert by_name["v4 structure (Takeaways / Goal / Methodology / Results)"].passed


def test_non_paper_no_frontmatter_body_unaffected():
    """A pre-sentinel legacy body (no frontmatter at all) never triggers the
    paper branch."""
    body = "# Legacy title\n\nSome legacy prose with no frontmatter and no paper flag.\n"
    _ok, results = verify_task_body.verify_text(body)
    assert all(r.name != "paper-stub body.md valid" for r in results)


def test_is_paper_stub_fm_helper():
    assert verify_task_body._is_paper_stub_fm({"paper": True})
    assert verify_task_body._is_paper_stub_fm({"paper": "true"})
    assert verify_task_body._is_paper_stub_fm({"paper": "TRUE"})
    assert not verify_task_body._is_paper_stub_fm({"paper": False})
    assert not verify_task_body._is_paper_stub_fm({"paper": "false"})
    assert not verify_task_body._is_paper_stub_fm({})
