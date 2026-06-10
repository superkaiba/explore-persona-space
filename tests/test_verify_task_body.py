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
    # CHECKS has 20 body-only functions under the 2-content-section
    # nested-design (v2) spec (includes the sentinel-gated
    # `check_tldr_nested_structure` and the check-8b Reproducibility
    # artifact-URL existence probe added 2026-06-09 as the #507
    # follow-up). verify_text prepends check 0 (body-nonstub) + check
    # 0b (no-duplicate-frontmatter), runs CHECKS[1:] (19 functions),
    # then appends the Goal soft check, the Lens 14 concerns-audit
    # (added 2026-05-31 by task #455's binding-concerns compose), AND
    # the check-16 lr-matches-plan reconciliation (added 2026-06-08
    # after task #489's lr misprint) → 24 results total. The Lens 14
    # and check-16 results are PASS-skips when no concerns.jsonl /
    # plans/plan.md sibling is available (the file-only / in-memory
    # invocation here).
    assert len(results) == 24


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
    """CHECKS contains 20 body-only functions: the 18 under the
    2-content-section spec (2026-W22, task #454) PLUS the nested-design
    (v2) sentinel-gated structure check `check_tldr_nested_structure`
    PLUS the check-8b Reproducibility artifact-URL existence probe
    (added 2026-06-09 as the task #507 follow-up).
    The migration is a RETARGET — every former check was kept
    (sometimes dormant, e.g. `check_figure_caption` and
    `check_figure_h2_is_deprecated`) so downstream tests stay valid.

    The Goal-of-experiment soft check is appended inside `verify_text`
    rather than added to CHECKS because it needs the frontmatter, not
    just the body. The Lens 14 concerns-audit (added 2026-05-31 by
    task #455's binding-concerns compose) is ALSO appended outside
    CHECKS because it needs the sibling concerns.jsonl path. So
    `verify_text` returns 24 results, but `CHECKS` stays at 20.
    """
    assert len(verify_task_body.CHECKS) == 20


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
    assert "v2 nested-design" in conf.detail


def test_v2_body_missing_what_i_ran_fails_nested_structure():
    """A v2-sentinelled body that drops `### What I ran` FAILs the
    nested-structure check."""
    body = _V2_GOOD_BODY.replace("### What I ran\n\nI trained 3 seeds", "I trained 3 seeds")
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    nested = by_name["TL;DR nested-design structure (v2)"]
    assert not nested.passed
    assert "What I ran" in nested.detail


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
    assert "v2 nested-design" in conf.detail


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
