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
    # CHECKS has 38 body-only functions: the 20 pre-v3 body-only checks
    # (incl. the sentinel-gated `check_tldr_nested_structure` and the
    # check-8b Reproducibility artifact-URL existence probe), the four
    # v3-gated body-only checks (check 18 `check_data_shape`, check 19
    # `check_data_subset_disclosure`, check 19b
    # `check_data_unwrapped_example_table` WARN, check 20
    # `check_v3_word_caps`), the FOUR v4-gated body-only checks (check 18
    # `check_v4_methodology_shape`, check
    # 21 `check_v4_results_beat` WARN, check 27
    # `check_v4_no_bare_issue_refs`, check 36
    # `check_v4_result_paragraph_sentences` WARN (#1368); check 20 v4
    # `check_v4_word_caps`
    # moved to the appended-outside set — it needs `issue`, #921) — each
    # a PASS-skip on this non-v3/non-v4 fixture — PLUS the TEN
    # generation-agnostic checks: check 22
    # (`check_figure_url_sha_matches_repro`), a NO-OP PASS here because
    # this fixture's `## Reproducibility` carries no figure-sha claim,
    # check 23 (`check_hf_url_resolves`), a PASS-with-`unverified`-note here
    # because the fixture's HF URLs are probe-fenced by conftest's
    # EPM_VERIFY_BODY_NO_HF=1, check 24
    # (`check_figure_text_vs_body_tokens`, WARN), a NO-OP PASS here because
    # this fixture's only figure pins a fake sha with no `.meta.json` in
    # the git tree, check 26
    # (`check_figure_panel_prose_vs_sidecar`, FAIL), a NO-OP PASS here
    # because the fixture's figure carries no panel/series prose claim,
    # check 28 (`check_figure_label_codes`, WARN), a NO-OP PASS here
    # for the same fake-sha / no-sidecar reason as check 24, check 29
    # (`check_figure_tracked_at_head`, WARN), which probes the live local
    # refs of the REAL repo here (no monkeypatch) — `passed=True` in every
    # state by construction (WARN/disclosure/skip never flip it), and
    # check 30 (`check_hf_file_count_claims`, WARN), a vacuous PASS here
    # because the fixture's HF link labels ("raw completions", "hf-hub")
    # carry no file-count claim, so ZERO Hub probes are issued even before
    # the fence, and check 32 (`check_hf_adjacent_file_claims`, WARN), a
    # vacuous PASS here for the analogous reason — no backtick FILENAME
    # token sits in an HF tree link's text or in a parenthetical
    # immediately after one, so ZERO Hub probes are issued even before
    # the fence, and check 33 (`check_figure_prose_numerics_vs_sidecar`,
    # WARN), a NO-OP PASS here for the same fake-sha / no-sidecar reason
    # as checks 24/28 (no value-bearing sidecar resolves, so no bolded
    # decimal is ever compared), and check 34
    # (`check_figure_beat_claims_vs_sidecar_text`, WARN, forward-only), a
    # NO-OP PASS here for the same fake-sha / no-sidecar reason (and no
    # `meta["text"]` block could resolve even if a sidecar did).
    # check 25 (`check_audit_availability_claims_match_hf`)
    # is a vacuous PASS here because this fixture carries no
    # availability-denial-near-artifact line. verify_text prepends check 0
    # (body-nonstub) + check 0b (no-duplicate-frontmatter), runs CHECKS[1:]
    # (38 functions), then appends the Goal soft check, the H1↔frontmatter-
    # title sync check (#1110; PASS-skip: not a sentinelled body), the
    # Lens 14
    # concerns-audit, the check-16 lr-matches-plan reconciliation, the
    # check-17 Context provenance-row read, the v3 check-21
    # body-Parameters-⊆-doc reconciliation (PASS-skip with no doc), the v4
    # check-20 word caps (needs `issue` for the events-based round budget,
    # #921; PASS-skip: not a v4 body), the
    # #732 judge-API-error denominator check (PASS-skip: legacy body), the
    # check-35 cross-issue reuse-provenance check (PASS-skip: not a v4
    # body, #1256), AND
    # the check-31 orphaned-per-unit-figures probe (needs `issue` for
    # figures-dir scoping, #1011; PASS here — the fixture's fake sha is not
    # locally reachable, so the cited SHA is silently skipped) →
    # 50 results total (2 prepended + CHECKS[1:]=38 + 10 appended). The
    # Lens 14 / check-16 results are PASS-skips when no concerns.jsonl /
    # plans/plan.md sibling is available; check 17 and the v3/v4 checks
    # are PASS-skips on this legacy (pre-v2-sentinel) fixture.
    assert len(results) == 50
    # By-name membership so the NEXT check addition can key by name instead
    # of re-deriving the arithmetic (#1016 methodology-reconciler Must-Fix).
    assert _HF_32_NAME in {r.name for r in results}
    assert "cross-issue reuse pins declared (footer Reused bullets)" in {r.name for r in results}
    assert "figure prose numerics vs figure sidecar (plotted-value drift)" in {
        r.name for r in results
    }
    assert "figure beat claims vs sidecar rendered text (series-structure drift)" in {
        r.name for r in results
    }


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


def test_repro_blockquoted_bare_url_ignored_by_permanence():
    """A bare (unpinned) URL inside a `>` blockquote in `## Reproducibility`
    — the SPEC-mandated verbatim originating-prompt quote (`**Context:**`
    row) — is provenance TEXT, not a provenance link: check 8 must not
    flag it (#825 → #959; mirrors the fence exemption). Nested `> >`
    lines and INDENTED `  > ` quote lines are covered too (the strip is
    lstrip-based, not a bare `startswith`)."""
    body = GOOD_BODY.replace(
        "**Compute:** 1× H100, 47 min.",
        "**Compute:** 1× H100, 47 min.\n\n"
        "**Context:** Verbatim originating prompt:\n\n"
        "> test in the base model https://huggingface.co/Qwen/Qwen2.5-7B\n"
        "> > nested quote citing https://wandb.ai/someone/some-project\n"
        "  > indented quote citing https://wandb.ai/someone/other-project\n"
        "> for both user and assistant\n",
    )
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    perm = by_name["Reproducibility URL permanence"]
    assert perm.passed, perm.detail
    assert ok, [r.render() for r in results if not r.passed]


def test_repro_nonquoted_bare_url_beside_blockquote_still_fails():
    """The blockquote exemption is line-scoped: a NON-quoted unpinned HF
    URL in the footer still FAILs check 8 even when its quoted twin sits
    one line up (the check stays binding for non-quoted footer URLs)."""
    body = GOOD_BODY.replace(
        "**Compute:** 1× H100, 47 min.",
        "**Compute:** 1× H100, 47 min.\n\n"
        "**Context:** Verbatim originating prompt:\n\n"
        "> quoted: https://huggingface.co/Qwen/Qwen2.5-7B\n\n"
        "Base model: https://huggingface.co/Qwen/Qwen2.5-7B\n",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    perm = by_name["Reproducibility URL permanence"]
    assert not perm.passed
    assert perm.detail.count("unpinned HF URL") == 1, perm.detail


def test_repro_quoted_fence_does_not_corrupt_fence_state():
    """A fence marker INSIDE a blockquote (`> ```) must not toggle fence
    state: the quoted run (incl. a quoted moving-ref URL) is dropped by
    the blockquote pass, and a NON-quoted unpinned URL after it is still
    scanned and FAILs. Catches a fence-state-corruption variant (a quoted
    fence marker toggling state would swallow the non-quoted URL)."""
    body = GOOD_BODY.replace(
        "**Compute:** 1× H100, 47 min.",
        "**Compute:** 1× H100, 47 min.\n\n"
        "> ```\n"
        "> https://github.com/superkaiba/explore-persona-space/blob/main/x.py\n"
        "> ```\n\n"
        "Unquoted: https://huggingface.co/Qwen/Qwen2.5-7B\n",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    perm = by_name["Reproducibility URL permanence"]
    assert not perm.passed
    assert "unpinned HF URL" in perm.detail
    assert "github.com" not in perm.detail  # the quoted moving-ref was never scanned


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


# ─── Check 29: figure tracked at live refs (offline git drift probe) ───────
#
# Incident task #841 (2026-07-04): three body-linked `figures/issue_841/`
# stems were tracked at the pinned sha `4824a567aa` but UNTRACKED at branch
# HEAD — the immutable pinned raw URLs kept rendering, check 4b kept
# passing (existence at the pinned sha), and nothing surfaced the tracking
# loss. Check 29 classifies each same-repo `figures/issue_<N>/` path
# against the live local refs (HEAD plus the `issue-<N>` / `issue-<N>-*`
# branch family): at HEAD → PASS; branch-only → PASS with a BRANCH-ONLY
# disclosure; missing everywhere probed → incident-class WARN (never FAIL).

_FIGURE_TRACKED_CHECK = "figure tracked at live refs"


def _make_repo_with_dropped_figure(tmp_path):
    """git repo where commit A tracks `figures/issue_999/hero.png` +
    `scripts/run.py` (so GOOD_BODY's check-4b/8b probes resolve when a test
    pins `sha_pin`) and a later commit B `git rm`ed the figure; HEAD=B.
    Callers create branches at A or B as the fixture case needs. Returns
    (repo_path, sha_pin) with sha_pin = commit A."""
    repo = tmp_path / "dropfigrepo"
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
    sha_pin = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git("rm", "-q", "figures/issue_999/hero.png")
    git("commit", "-q", "-m", "drop hero figure")
    return repo, sha_pin


def test_figure_missing_everywhere_warns(tmp_path, monkeypatch):
    """The #841 incident fixture: figure tracked at the pinned sha but
    missing from HEAD AND the whole `issue-999` branch family → the
    incident-class WARN (passed=True — overall verdict unaffected), while
    check 4b still PASSes (the pinned sha resolves). Also asserts the
    subprocess budget of a DIRECT check invocation (never a global count
    across verify_text — check 4b legitimately adds its own git calls)."""
    repo, sha_pin = _make_repo_with_dropped_figure(tmp_path)
    # Branch at HEAD (=B, figure absent): the family exists but lacks it.
    subprocess.run(["git", "-C", str(repo), "branch", "issue-999"], check=True, capture_output=True)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = GOOD_BODY.replace("0123456789abcdef", sha_pin)
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name["Figure URL resolvable"].passed  # check 4b: tracked at the pinned sha
    r = by_name[_FIGURE_TRACKED_CHECK]
    assert r.passed is True
    assert r.is_warn is True
    assert "figures/issue_999/hero.png" in r.detail
    assert "git restore --source=" in r.detail
    assert "issue-999" in r.detail  # successfully-probed ref labels named
    assert ok  # the WARN never flips the overall verdict (no-regress guarantee)
    # Scoped subprocess budget: 1 for-each-ref + 1 HEAD ls-tree + 1 branch
    # ls-tree = 3 (plan §4.5 budget: <=5 with <=2 family branches).
    calls: list = []
    real_run = subprocess.run

    def counting_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(verify_task_body.subprocess, "run", counting_run)
    r2 = verify_task_body.check_figure_tracked_at_head(body)
    assert r2.is_warn is True
    assert len(calls) == 3
    assert len(calls) <= 5


def test_figure_branch_only_discloses_not_warns(tmp_path, monkeypatch):
    """Branch `issue-999` created at commit A (has the figure), HEAD moved
    to B (lacks it) — the stale-branch-masks-main-loss state: PASS with the
    BRANCH-ONLY disclosure (path + holding branch + recovery), never a WARN
    and never silent."""
    repo, sha_pin = _make_repo_with_dropped_figure(tmp_path)
    subprocess.run(
        ["git", "-C", str(repo), "branch", "issue-999", sha_pin], check=True, capture_output=True
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = GOOD_BODY.replace("0123456789abcdef", sha_pin)
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    r = by_name[_FIGURE_TRACKED_CHECK]
    assert r.passed is True
    assert r.is_warn is False
    assert "BRANCH-ONLY" in r.detail
    assert "figures/issue_999/hero.png" in r.detail
    assert "issue-999" in r.detail
    assert "git restore --source=" in r.detail
    assert ok


def test_figure_tracked_at_repo_head_passes_without_branch(tmp_path, monkeypatch):
    """No `issue-999` branch, HEAD tracks the figure (the merged-and-
    branch-deleted grandfather case): clean PASS — no WARN, no
    disclosure."""
    repo, sha = _make_repo_with_figure(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = GOOD_BODY.replace("0123456789abcdef", sha)
    _ok, results = verify_task_body.verify_text(body)
    r = _results_by_name(results)[_FIGURE_TRACKED_CHECK]
    assert r.passed is True
    assert r.is_warn is False
    assert "tracked at HEAD" in r.detail
    assert "BRANCH-ONLY" not in r.detail


def test_figure_on_suffix_branch_discloses_not_warns(tmp_path, monkeypatch):
    """Figure tracked ONLY at `refs/heads/issue-999-fu` (a same-issue
    follow-up suffix branch); absent from `issue-999` and HEAD: PASS with
    the branch-only disclosure — a figure-adding follow-up round must not
    WARN."""
    repo, sha_pin = _make_repo_with_dropped_figure(tmp_path)
    subprocess.run(
        ["git", "-C", str(repo), "branch", "issue-999-fu", sha_pin],
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "-C", str(repo), "branch", "issue-999"], check=True, capture_output=True)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = GOOD_BODY.replace("0123456789abcdef", sha_pin)
    r = verify_task_body.check_figure_tracked_at_head(body)
    assert r.passed is True
    assert r.is_warn is False
    assert "BRANCH-ONLY" in r.detail
    assert "issue-999-fu" in r.detail


def test_figure_check_vacuous_pass_no_matching_urls():
    """Body whose only image is an other-host URL: vacuous PASS, no git
    probes needed."""
    body = GOOD_BODY.replace(
        _GOOD_BODY_FIGURE_URL,
        "https://eps-figures.example.com/issue_999/hero.png",
    )
    r = verify_task_body.check_figure_tracked_at_head(body)
    assert r.passed is True
    assert r.is_warn is False
    assert "no same-repo" in r.detail


def test_figure_check_repo_unresolved_skips(monkeypatch):
    """`_resolve_repo_root` → None (running outside the repo): skip-PASS."""
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: None)
    r = verify_task_body.check_figure_tracked_at_head(GOOD_BODY)
    assert r.passed is True
    assert r.is_warn is False
    assert r.detail.startswith("skipped")


def test_figure_check_git_error_degrades_to_pass(tmp_path, monkeypatch):
    """`_resolve_repo_root` pointed at a plain non-git dir (for-each-ref
    and ls-tree both fail): fail-soft per-issue probe-failure note, never a
    WARN, and no exception propagates through verify_text."""
    plain = tmp_path / "notarepo"
    plain.mkdir()
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: plain)
    r = verify_task_body.check_figure_tracked_at_head(GOOD_BODY)
    assert r.passed is True
    assert r.is_warn is False
    assert "probe failure" in r.detail
    ok, results = verify_task_body.verify_text(GOOD_BODY)  # no exception end to end
    assert _results_by_name(results)[_FIGURE_TRACKED_CHECK].passed
    assert ok


def test_figure_partial_probe_failure_never_warns(tmp_path, monkeypatch):
    """HEAD probe succeeds (figure absent from HEAD) but the family-branch
    probe fails: the conservative rule demotes the issue dir to a skip note
    — the path might live at the failed ref, so a narrowed ref set must
    never manufacture a WARN. The failed ref is named as FAILED, not
    presented as a successfully-probed ('checked') label."""
    repo, sha_pin = _make_repo_with_dropped_figure(tmp_path)
    subprocess.run(["git", "-C", str(repo), "branch", "issue-999"], check=True, capture_output=True)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    real_tracked = verify_task_body._git_tracked_under

    def flaky_tracked(repo_, ref, prefix):
        if ref == "HEAD":
            return real_tracked(repo_, ref, prefix)
        return None  # family-branch probe fails

    monkeypatch.setattr(verify_task_body, "_git_tracked_under", flaky_tracked)
    body = GOOD_BODY.replace("0123456789abcdef", sha_pin)
    r = verify_task_body.check_figure_tracked_at_head(body)
    assert r.passed is True
    assert r.is_warn is False
    assert "probe failure" in r.detail
    assert "issue-999" in r.detail  # the failed ref is named as failed
    assert "MISSING from every live local ref" not in r.detail


# ─── Check 31: orphaned per-unit companion figures (inverse git probe) ─────
#
# Incident task #928 (round 3): the body embedded only the pooled MLP
# aggregate while the round-committed per-context companion
# `figures/issue_928/mlp_indiv_percontext_delta.png` — committed at a SHA
# the body already cited — sat unreferenced by every body image URL; the
# gap reached the LM clean-result-critic as a Lens 11 blocker instead of
# being caught pre-gate. Check 31 runs the INVERSE direction of checks
# 4b/22/29: ls-tree the body's OWN cited figure SHAs and WARN (never FAIL)
# on committed per-unit-named PNGs the body neither embeds nor names in
# prose. (#1011)

_PER_UNIT_ORPHAN_CHECK = "per-unit companion figures embedded"

_PER_UNIT_ORPHAN_PATH = "figures/issue_999/hero_percontext.png"


def _make_repo_with_per_unit_orphan(tmp_path):
    """git repo whose HEAD commit tracks `figures/issue_999/hero.png` +
    `figures/issue_999/hero_percontext.png` (the per-unit companion) +
    `scripts/run.py` (so GOOD_BODY's check-8b Code-blob probe resolves
    when a test pins the real sha); returns (repo_path, head_sha)."""
    repo = tmp_path / "perunitrepo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    figdir = repo / "figures" / "issue_999"
    figdir.mkdir(parents=True)
    (figdir / "hero.png").write_bytes(b"\x89PNG fake bytes")
    (figdir / "hero_percontext.png").write_bytes(b"\x89PNG fake bytes")
    script = repo / "scripts" / "run.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('entry script')\n")
    git("add", "figures", "scripts")
    git("commit", "-q", "-m", "add hero + per-context companion + entry script")
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, sha


def test_orphan_per_unit_figure_warns(tmp_path, monkeypatch):
    """The #928 incident shape: `hero_percontext.png` committed at the
    body-cited sha, body embeds only `hero.png`, stem named nowhere in
    prose → the incident-class WARN (passed=True — overall verdict
    unaffected). Asserted BY NAME through verify_text so the dispatch
    outside CHECKS is pinned (a refactor dropping the `verify_text`
    append fails here); the subprocess budget is asserted on a DIRECT
    invocation only (never a global count across verify_text — checks
    4b/8b/29 legitimately add their own git calls)."""
    repo, sha = _make_repo_with_per_unit_orphan(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = GOOD_BODY.replace("0123456789abcdef", sha)
    ok, results = verify_task_body.verify_text(body, issue=999)
    by_name = _results_by_name(results)
    r = by_name[_PER_UNIT_ORPHAN_CHECK]
    assert r.passed is True
    assert r.is_warn is True
    assert _PER_UNIT_ORPHAN_PATH in r.detail
    assert "Lens 11" in r.detail
    assert sha[:8] in r.detail
    assert ok  # the WARN never flips the overall verdict (no-regress guarantee)
    # Scoped subprocess budget on a DIRECT invocation: 1 unique (sha, dir)
    # pair → exactly 1 ls-tree (plan §4.1 budget: 1 per unique pair).
    calls: list = []
    real_run = subprocess.run

    def counting_run(cmd, *args, **kwargs):
        calls.append(cmd)
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(verify_task_body.subprocess, "run", counting_run)
    r2 = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r2.is_warn is True
    assert len(calls) == 1


def test_per_unit_figure_embedded_no_warn(tmp_path, monkeypatch):
    """Body embeds BOTH the hero and the per-unit companion → clean PASS
    (the companion is in the referenced path set)."""
    repo, sha = _make_repo_with_per_unit_orphan(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    companion_url = (
        "https://raw.githubusercontent.com/superkaiba/explore-persona-space/"
        f"{sha}/figures/issue_999/hero_percontext.png"
    )
    body = GOOD_BODY.replace("0123456789abcdef", sha).replace(
        "> **Figure.**",
        f"![Per-context deltas behind the aggregate.]({companion_url})\n\n> **Figure.**",
    )
    r = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r.passed is True
    assert r.is_warn is False
    assert "no orphaned per-unit figures" in r.detail


def test_orphan_unreachable_sha_skips_silently(tmp_path, monkeypatch):
    """Cited sha unknown to the local object DB (GOOD_BODY's placeholder
    sha kept; the repo has no such commit): the SHA is skipped SILENTLY —
    counted in the PASS detail, never a WARN (hard constraint: an
    unreachable SHA must not manufacture a false WARN)."""
    repo, _sha = _make_repo_with_per_unit_orphan(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    r = verify_task_body.check_orphaned_per_unit_figures(GOOD_BODY, issue=999)
    assert r.passed is True
    assert r.is_warn is False
    assert "not locally reachable" in r.detail


def test_orphan_prose_mention_suppresses_warn(tmp_path, monkeypatch):
    """The prose disclosure escape: an unembedded companion whose stem is
    named in body prose is treated as disclosed → no WARN (mechanizes
    'exemptions stated in prose are legitimate')."""
    repo, sha = _make_repo_with_per_unit_orphan(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = GOOD_BODY.replace("0123456789abcdef", sha).replace(
        "The 17-pt lift holds at every seed;",
        "The standalone `hero_percontext` scatter is superseded by the right panel. "
        "The 17-pt lift holds at every seed;",
    )
    r = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r.passed is True
    assert r.is_warn is False


def test_orphan_deduped_across_cited_shas(tmp_path, monkeypatch):
    """The same orphan committed at TWO body-cited SHAs is ONE detail
    entry (keyed by path), listing both short SHAs."""
    repo, sha_a = _make_repo_with_per_unit_orphan(tmp_path)

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    second = repo / "figures" / "issue_999" / "second.png"
    second.write_bytes(b"\x89PNG fake bytes")
    git("add", "figures")
    git("commit", "-q", "-m", "add second figure")
    sha_b = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    second_url = (
        "https://raw.githubusercontent.com/superkaiba/explore-persona-space/"
        f"{sha_b}/figures/issue_999/second.png"
    )
    body = GOOD_BODY.replace("0123456789abcdef", sha_a).replace(
        "> **Figure.**",
        f"![Second figure at a second sha.]({second_url})\n\n> **Figure.**",
    )
    r = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r.passed is True
    assert r.is_warn is True
    assert r.detail.count(_PER_UNIT_ORPHAN_PATH) == 1  # deduped by path
    assert sha_a[:8] in r.detail
    assert sha_b[:8] in r.detail


def test_orphan_cross_issue_dir_not_scanned_when_issue_known(tmp_path, monkeypatch):
    """A cross-issue embed (`figures/issue_777/x.png`, whose dir holds its
    own orphan) must NOT surface issue_777's orphans when `issue=999` is
    known — only this task's figures dir is scanned."""
    repo, _sha_a = _make_repo_with_per_unit_orphan(tmp_path)

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    other = repo / "figures" / "issue_777"
    other.mkdir(parents=True)
    (other / "x.png").write_bytes(b"\x89PNG fake bytes")
    (other / "x_percontext.png").write_bytes(b"\x89PNG fake bytes")
    git("add", "figures")
    git("commit", "-q", "-m", "add cross-issue figures")
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    base_url = f"https://raw.githubusercontent.com/superkaiba/explore-persona-space/{sha}"
    body = GOOD_BODY.replace("0123456789abcdef", sha).replace(
        "> **Figure.**",
        f"![companion]({base_url}/figures/issue_999/hero_percontext.png)\n\n"
        f"![cross-issue]({base_url}/figures/issue_777/x.png)\n\n> **Figure.**",
    )
    r = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r.passed is True
    assert r.is_warn is False  # issue_999 fully embedded; issue_777 out of scope
    assert "issue_777" not in r.detail


def test_orphan_repo_unresolved_skips(monkeypatch):
    """`_resolve_repo_root` → None (running outside the repo): skip-PASS."""
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: None)
    r = verify_task_body.check_orphaned_per_unit_figures(GOOD_BODY, issue=999)
    assert r.passed is True
    assert r.is_warn is False
    assert r.detail.startswith("skipped")


def test_orphan_git_error_degrades_to_pass(tmp_path, monkeypatch):
    """`_resolve_repo_root` pointed at a plain non-git dir: every ls-tree
    fails → every cited SHA degrades to the silent skip, never a WARN, and
    no exception propagates through verify_text."""
    plain = tmp_path / "notarepo2"
    plain.mkdir()
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: plain)
    r = verify_task_body.check_orphaned_per_unit_figures(GOOD_BODY, issue=999)
    assert r.passed is True
    assert r.is_warn is False
    ok, results = verify_task_body.verify_text(GOOD_BODY, issue=999)  # no exception end to end
    assert _results_by_name(results)[_PER_UNIT_ORPHAN_CHECK].passed
    assert ok


def test_orphan_issue_none_fallback_scans_cited_dirs(tmp_path, monkeypatch):
    """`issue=None` (the --body-stdin shape): the check falls back to
    scanning every cited `figures/issue_<K>/` dir, so the orphan still
    surfaces when no issue number is threaded."""
    repo, sha = _make_repo_with_per_unit_orphan(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = GOOD_BODY.replace("0123456789abcdef", sha)
    ok, results = verify_task_body.verify_text(body)  # no issue threaded
    r = _results_by_name(results)[_PER_UNIT_ORPHAN_CHECK]
    assert r.passed is True
    assert r.is_warn is True
    assert _PER_UNIT_ORPHAN_PATH in r.detail
    assert ok


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("mlp_indiv_percontext_delta", True),  # the #928 incident filename
        ("per_context_gain", True),
        ("per-context_x", True),
        ("per_unit_deltas", True),
        ("percell_grid", True),
        ("PerContext_Upper", True),  # case-insensitive
        ("mlp_indiv_hero_4arm", False),  # `indiv` names the regime, not a view
        ("supercontext_map", False),  # mid-word hit blocked by the lookbehind
        ("experiment_percent", False),
        ("per_source_rates", False),  # other per-X families out of scope by design
        ("per_seed_scatter", False),
    ],
)
def test_per_unit_basename_pattern(stem, expected):
    """The deliberately-narrow check-31 pattern: the three per-unit nouns
    (context/unit/cell) with -/_ spellings match; regime names (`indiv`),
    mid-word hits (`supercontext`), and other per-X families
    (per_source/per_seed) do NOT — Lens 11 owns the substance."""
    assert bool(verify_task_body._PER_UNIT_FIG_RE.search(stem)) is expected


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


def test_gather_repro_artifact_urls_skips_blockquoted():
    """Check 8b must not existence-probe a same-repo URL quoted inside
    the verbatim originating-prompt blockquote — same #959 collision
    class as check 8 (a verbatim quote cannot be edited if its cited
    path later dies). Non-quoted same-repo URLs are still gathered.
    Deterministic unit test of the gather — no git, no network."""
    repro = (
        "**Code:** [run](https://github.com/superkaiba/explore-persona-space"
        "/blob/0123456789abcdef/scripts/run.py)\n\n"
        "**Context:** Verbatim originating prompt:\n\n"
        "> see https://github.com/superkaiba/explore-persona-space"
        "/blob/deadbeefdead/scripts/gone.py\n"
    )
    urls = verify_task_body._gather_repro_artifact_urls(repro)
    assert urls == [
        "https://github.com/superkaiba/explore-persona-space/blob/0123456789abcdef/scripts/run.py"
    ]


# ─── Check 23: HF Hub revision-pin existence ──────────────────────────────
#
# Incident task #537 (2026-06-16): a `## Reproducibility` `**Artifacts:**`
# link pinned the "415 bakeoff intermediates" to revision `db3662ae`, the
# main-grid revision that PREDATES the bakeoff round — the path resolves to
# 0 files at that revision, so a reader clicking it gets nothing. The URL is
# shape-valid + sha-pinned + on a real repo, so it slipped through every
# other check. Check 23 probes the HF Hub tree endpoint with a BOUNDED
# direct GET (`verify_task_body._hf_tree_get`, #733 — NOT the unbounded
# recursive whole-repo `list_repo_files`) and FAILs a dead pin. Fail-soft:
# the suite-wide EPM_VERIFY_BODY_NO_HF=1 fence (tests/conftest.py) makes the
# probe SKIP (PASS + `unverified` note) so fixture HF URLs never hit the
# live Hub. Tests below `monkeypatch.delenv` the fence and stub
# `verify_task_body._hf_tree_get` — the single bounded primitive both the
# check-23 and check-25 probes funnel through — to return a chosen
# `_TreeProbeResult` without any network.

_HF_23_NAME = "HF URL pins resolve at the cited revision"


@pytest.fixture(autouse=True)
def _clear_hf_existence_cache():
    """The check-23/25 probes memoize definitive pass/fail verdicts in a
    module-level `_HF_EXISTENCE_CACHE` (#733), the check-30 count probe
    memoizes successful exhaustive `(n_files, n_dirs)` listings in
    `_HF_TREE_FILE_COUNT_CACHE` (#1008), and the check-32 membership probe
    memoizes successful exhaustive basename listings in
    `_HF_TREE_BASENAMES_CACHE` (#1016). Clear all three before AND after
    each test so a cached verdict keyed on a (repo, sha, path) reused across
    fixtures never leaks one test's stubbed outcome into another."""
    verify_task_body._HF_EXISTENCE_CACHE.clear()
    verify_task_body._HF_TREE_FILE_COUNT_CACHE.clear()
    verify_task_body._HF_TREE_BASENAMES_CACHE.clear()
    yield
    verify_task_body._HF_EXISTENCE_CACHE.clear()
    verify_task_body._HF_TREE_FILE_COUNT_CACHE.clear()
    verify_task_body._HF_TREE_BASENAMES_CACHE.clear()


# The real probe primitive, captured at module import time — BEFORE any
# fixture patches — for the one test that exercises the real body at the
# SDK boundary (test_hf_url_paginated_then_429_is_bounded_unverified).
_REAL_HF_TREE_GET = verify_task_body._hf_tree_get


@pytest.fixture(autouse=True)
def _no_unexpected_probes(monkeypatch):
    """Raise-by-default guard (#1161, mirroring test_verify_task_body_audit_claim
    from #860): a `_hf_tree_get` call a test did not explicitly stub is a hard
    error — missed-mock detection independent of network / offline / Hub repo
    state (the suite-wide EPM_VERIFY_BODY_NO_HF fence alone degrades a missed
    mock to skip-not-raise semantics; a test that delenvs the fence but forgets
    its stub is what this guard catches — a fence left in place still shadows
    the probe before it reaches the guard). Per-test `_stub_tree` /
    `_i833_needle_stub` / inline setattr re-patches over this (the shared
    function-scoped monkeypatch teardown restores cleanly); a test that must
    exercise the REAL body restores `_REAL_HF_TREE_GET` explicitly."""

    def _unexpected(url, params, headers, *, timeout_s):  # pragma: no cover
        raise AssertionError(
            f"unexpected _hf_tree_get probe of {url} — add _stub_tree or an explicit allowance"
        )

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _unexpected)


def _stub_tree(monkeypatch, *, status="ok", entries=(), next_page=None, note="", calls=None):
    """Replace `verify_task_body._hf_tree_get` with a stub returning a fixed
    `_TreeProbeResult`. When `calls` is a list, every (url, params) the probe
    issues is appended so a test can assert the request count / pagination."""

    def _fake(url, params, headers, *, timeout_s):
        if calls is not None:
            calls.append((url, params))
        return verify_task_body._TreeProbeResult(status, list(entries), next_page, note)

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _fake)


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


_HF_25_NAME = "audit-availability claims match HF Hub"


def _audit_body(denial_line: str, hf_url: str) -> str:
    """A `_hf_body`-derived body (one controlled HF revision-pinned URL) with
    an availability-denial line spliced into the findings prose, so check 25
    has BOTH a denial-near-artifact line AND an HF URL to reconcile against.
    Assert on the check-25 result BY NAME (`_HF_25_NAME`) — the spliced denial
    is the only thing this body controls for check 25."""
    body = _hf_body(hf_url)
    return body.replace(
        "\n## Reproducibility\n",
        f"\n{denial_line}\n\n## Reproducibility\n",
        1,
    )


def test_hf_url_existing_path_passes(monkeypatch):
    """A dataset `/tree/<sha>/<path>` whose path matches ≥1 entry in the
    PARENT-dir listing → definitive PASS (no `unverified` note). The probe
    lists the needle's parent (`raw_completions`) and matches the needle
    among its direct children."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    _stub_tree(
        monkeypatch,
        status="ok",
        entries=[
            {"path": "raw_completions/run.jsonl", "type": "file"},
            {"path": "raw_completions/README.md", "type": "file"},
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
    files (pinned to a revision predating the upload) → definitive FAIL. The
    parent-dir listing succeeds but does NOT contain the needle."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    # `db3662ae` lists only the main-grid files — none under the bakeoff path,
    # so the parent listing of `bakeoff_intermediates` returns other entries.
    _stub_tree(
        monkeypatch,
        status="ok",
        entries=[
            {"path": "main_grid/results.csv", "type": "file"},
            {"path": "README.md", "type": "file"},
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
    """A revision/path that does not exist → a 404 from the tree endpoint →
    `not_found` → definitive FAIL (a fabricated / never-pushed sha). Check 23
    maps `not_found` to FAIL (the dead-pin invariant)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    _stub_tree(monkeypatch, status="not_found")
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
    _stub_tree(
        monkeypatch,
        status="indeterminate",
        note="HF tree probe failed: ConnectionError: getaddrinfo failed",
    )
    body = _hf_body(
        "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/feedface/raw_completions/run.jsonl"
    )
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_HF_23_NAME].passed
    assert "unverified" in by_name[_HF_23_NAME].detail
    assert "HF tree probe failed" in by_name[_HF_23_NAME].detail
    assert ok


def test_hf_url_env_fence_skips(monkeypatch):
    """With the suite-wide EPM_VERIFY_BODY_NO_HF=1 fence in place (the
    conftest default), the probe SKIPs without touching the Hub → PASS with
    an `unverified` note even if the tree probe WOULD have failed."""
    monkeypatch.setenv("EPM_VERIFY_BODY_NO_HF", "1")

    def _boom(url, params, headers, *, timeout_s):  # pragma: no cover
        raise AssertionError("the tree probe must NOT be called under the fence")

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _boom)
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
    root listing succeeds — it only asserts the revision exists."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    _stub_tree(
        monkeypatch,
        status="ok",
        entries=[{"path": "config.json", "type": "file"}],
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

    def _boom(url, params, headers, *, timeout_s):  # pragma: no cover
        raise AssertionError("moving-ref HF URL must not be probed by check 23")

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _boom)
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

    def _boom(url, params, headers, *, timeout_s):  # pragma: no cover
        raise AssertionError("non-HF URL must not reach the HF probe")

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _boom)
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
    never probed (the not-found stub would otherwise FAIL it)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)

    def _boom(url, params, headers, *, timeout_s):  # pragma: no cover
        raise AssertionError("a fenced (illustrative) HF URL must not be probed")

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _boom)
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


def test_hf_url_dead_pin_not_found_fail_on_new_api(monkeypatch):
    """#537 regression on the NEW direct-GET path: a path-pinned link whose
    parent dir 404s (`not_found`) FAILs with the dead-pin message — the
    independent check-23 mapping of `not_found` → FAIL."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    _stub_tree(monkeypatch, status="not_found")
    body = _hf_body(
        "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/db3662ae/bakeoff_intermediates/run.jsonl"
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name[_HF_23_NAME].passed
    assert "dead revision pin" in by_name[_HF_23_NAME].detail
    assert "db3662ae" in by_name[_HF_23_NAME].detail


def test_autouse_probe_guard_catches_missed_stub(monkeypatch):
    """Missed-mock detection (#1161): fence removed + NO `_stub_tree` — the
    check-23 existence probe must hit the module's autouse raise-by-default
    guard as a hard AssertionError, never degrade to live-network behavior."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    with pytest.raises(AssertionError, match="unexpected _hf_tree_get probe"):
        verify_task_body.check_hf_url_resolves(
            _hf_body(
                "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/"
                "tree/feedface/issue1161_guard/run.jsonl"
            )
        )


def test_hf_url_paginated_then_429_is_bounded_unverified(monkeypatch):
    """The MUST-FIX-1 path v1 could not reach: a page-1 success carrying a
    Link rel=next, followed by a 429 on page 2, must surface `unverified`
    within the bounded request budget rather than entering the SDK's
    ~143s/page backoff. This drives the check-25 keyword probe (the
    paginating call site) through `_hf_tree_get` at the SDK boundary:
    `verify_task_body.get_session().get` is stubbed to return a page-1
    200-with-Link then a 429, and we assert (a) the verdict is `unverified`
    (PASS, body still ok), (b) the probe made a BOUNDED number of GETs, and
    (c) the page-2 URL WAS fetched (pagination genuinely exercised)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    # Explicit allowance: this test exercises the REAL `_hf_tree_get` body
    # (pagination + bounded retry), stubbing one level deeper (get_session).
    # Restore it over the module's autouse raise-by-default guard.
    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _REAL_HF_TREE_GET)
    import huggingface_hub.utils as hf_utils

    page2_url = "https://huggingface.co/api/datasets/r/tree/sha/p?cursor=PAGE2"
    # Count ONLY the check-25 keyword probe's GETs (its first page sends
    # `params={"recursive": True}`; page 2 carries the cursor URL). Check 23's
    # own non-recursive existence GET shares this stub but is not the call site
    # under test, so it is excluded from the bound.
    kw_calls: list[str] = []

    class _Resp:
        def __init__(self, status_code, *, json_data=None, links=None):
            self.status_code = status_code
            self._json = json_data if json_data is not None else []
            self.links = links or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._json

    class _Session:
        def get(self, url, params=None, headers=None, timeout=None):
            is_page2 = "PAGE2" in url
            is_recursive = bool(params) and params.get("recursive") is True
            if is_page2 or is_recursive:
                kw_calls.append(url)
            if is_page2:
                return _Resp(429)  # throttled second page of the keyword probe
            # The needle `issue653_x` is listed (so check 23's existence GET
            # PASSes), but `install_probes` is NOT, so the check-25 keyword
            # probe paginates to a Link rel=next pointing at the throttled
            # page 2.
            return _Resp(
                200,
                json_data=[
                    {"path": "issue653_x", "type": "directory"},
                    {"path": "issue653_x/armB/cell0", "type": "directory"},
                ],
                links={"next": {"url": page2_url}},
            )

    monkeypatch.setattr(hf_utils, "get_session", lambda: _Session())
    # The denial line + an HF URL so check 25 actually probes (this is the
    # paginating call site). The keyword (`install_probes`) is never present
    # in the stubbed listing, so the probe paginates until the page-2 429.
    body = _audit_body(
        "The per-cell install-probe completions were not separately uploaded, "
        "so they cannot be audited at the record level.",
        "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/feedface/issue653_x",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    # (a) check 25 surfaces `unverified` (SKIP), NEVER a FAIL, under the throttle.
    assert by_name[_HF_25_NAME].passed
    assert "unverified" in by_name[_HF_25_NAME].detail
    # (b) bounded request count: at most MAX_PAGES * ATTEMPTS keyword-probe GETs,
    # and far fewer than the unbounded SDK backoff (max_retries=20/page) would
    # have issued.
    max_expected = verify_task_body._HF_PROBE_MAX_PAGES * verify_task_body._HF_PROBE_ATTEMPTS
    assert 0 < len(kw_calls) <= max_expected
    # (c) the page-2 URL was actually fetched — pagination genuinely exercised
    # (the path v1's raise-on-entry stub could NOT reach).
    assert any("PAGE2" in c for c in kw_calls)


def test_hf_check25_nested_keyword_contradiction_fails(monkeypatch):
    """MUST-FIX 2(a) + the #653 regression on the new API: the body denies the
    install-probe completions were uploaded, but a file carrying
    `install_probes` exists at depth >= 2 under the linked tree-root prefix —
    so the denial is FALSE and check 25 FAILs, surfacing the matched path. The
    keyword nests several levels below the prefix (the #653 shape: the body
    links the tree ROOT while the file lives at
    `<root>/raw_completions/armB/install_probes/cell0/x.json`), so the
    depth-agnostic scoped recursive listing must find it."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    nested_file = (
        "issue653_install-validated-reladder/raw_completions/armB/install_probes/cell0/x.json"
    )
    _stub_tree(
        monkeypatch,
        status="ok",
        entries=[
            {
                "path": "issue653_install-validated-reladder/raw_completions/armB",
                "type": "directory",
            },
            {"path": nested_file, "type": "file"},
        ],
    )
    body = _audit_body(
        "The per-cell install-probe completions themselves were not separately "
        "uploaded, so the firing vs non-firing examples cannot be audited at the "
        "record level.",
        "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc1234/issue653_install-validated-reladder",
    )
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert not by_name[_HF_25_NAME].passed
    assert not ok
    assert "install_probes" in by_name[_HF_25_NAME].detail
    assert nested_file in by_name[_HF_25_NAME].detail


def test_hf_check25_not_found_is_skip_not_fail(monkeypatch):
    """MUST-FIX 2(b) + MUST-FIX 3: a `not_found` from the tree endpoint maps
    to SKIP on check 25 (NOT FAIL) — the deliberate check-23-FAIL-vs-25-SKIP
    asymmetry. Check 25 cannot corroborate OR refute a denial against a
    revision that does not exist, so it surfaces `unverified`, never a FAIL."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    _stub_tree(monkeypatch, status="not_found")
    body = _audit_body(
        "The per-cell install-probe completions were not separately uploaded, "
        "so they cannot be audited at the record level.",
        "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/deadbeef/issue653_x",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    # SKIP → PASS with an `unverified` note, NOT a FAIL.
    assert by_name[_HF_25_NAME].passed
    assert "unverified" in by_name[_HF_25_NAME].detail


# ─── Check 30: HF file-count claims vs the Hub tree (WARN) ─────────────────
#
# Check 30 (`check_hf_file_count_claims`, #1008) extracts "N files" /
# "N shards" claims adjacent to hex-pinned HF /tree markdown links and
# compares them against a files-only scoped Hub tree count via the same
# #733 bounded raw tree-endpoint probe stack checks 23/25 use. Claim
# positions: Pattern A count-in-link-text, Pattern B paren-before-link,
# Pattern C anchored paren-after-link + the per-namespace form (#1088),
# and Pattern D backtick `dir/` sub-path + count-opening paren bound to
# the nearest preceding pinned link (#1143, the #1112 footer shape). All
# tests are offline: extractor tests need no stub; probe tests stub
# `verify_task_body._hf_tree_get` (`_stub_tree` / inline stateful
# closures) after removing the conftest EPM_VERIFY_BODY_NO_HF fence.

_HF_30_NAME = "HF file-count claims match the Hub tree"

_I931_SHA = "9534b9981d6b4fb4f1259c9b06f021d311a46af4"
_I931_REPO = "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data"


def _count_claim_body(link_label: str, hf_url: str) -> str:
    """`_hf_body` with the dataset link's label replaced so it carries a
    count claim in the link TEXT (Pattern A), e.g.
    `issue931_story_map, 528 files`."""
    return _hf_body(hf_url).replace("[raw completions](", f"[{link_label}](", 1)


def test_hf_count_extractor_link_text_shapes():
    """Pure extractor (no monkeypatch, no network): the three verbatim #931
    link-text shapes each yield exactly one claim tuple with the right
    (count, repo, type, sha, prefix); singular '1 file' and comma-grouped
    '1,234 files' also extract."""
    body = (
        f"- [pairs_meta, 9 files]({_I931_REPO}/tree/{_I931_SHA}"
        "/issue931_story_map/raw_completions/pairs_meta) — meta rows\n"
        f"- [generation, 2 files]({_I931_REPO}/tree/{_I931_SHA}"
        "/issue931_story_map/raw_completions/generation) — raw generations\n"
        f"- [judge_audit, 197 files]({_I931_REPO}/tree/{_I931_SHA}"
        "/issue931_story_map/raw_completions/judge_audit) — judge audits\n"
    )
    claims = verify_task_body._gather_hf_count_claims(body)
    assert len(claims) == 3
    by_prefix = {c[5]: c for c in claims}
    assert by_prefix["issue931_story_map/raw_completions/pairs_meta"][0] == 9
    assert by_prefix["issue931_story_map/raw_completions/generation"][0] == 2
    assert by_prefix["issue931_story_map/raw_completions/judge_audit"][0] == 197
    for _count, noun, repo_id, repo_type, sha, _prefix in claims:
        assert noun == "files"
        assert repo_id == "superkaiba1/explore-persona-space-data"
        assert repo_type == "dataset"
        assert sha == _I931_SHA
    single = verify_task_body._gather_hf_count_claims(
        f"[x, 1 file]({_I931_REPO}/tree/{_I931_SHA}/p)"
    )
    assert [(c[0], c[1]) for c in single] == [(1, "file")]
    comma = verify_task_body._gather_hf_count_claims(
        f"[x, 1,234 files]({_I931_REPO}/tree/{_I931_SHA}/p)"
    )
    assert [c[0] for c in comma] == [1234]


def test_hf_count_extractor_paren_before_link_shape():
    """Pure extractor: the #931 footer shape (Pattern B — a parenthetical
    OPENING with the count-noun immediately before the markdown link) yields
    (515, ..., 'issue931_story_map'); the same claim appearing via BOTH
    patterns dedups to one tuple."""
    footer = (
        "HF artifacts (515 files verified via scoped listing): "
        f"[issue931_story_map @ 9534b998]({_I931_REPO}/tree/{_I931_SHA}/issue931_story_map)"
    )
    claims = verify_task_body._gather_hf_count_claims(footer)
    assert len(claims) == 1
    count, noun, repo_id, repo_type, sha, prefix = claims[0]
    assert (count, noun, prefix) == (515, "files", "issue931_story_map")
    assert repo_id == "superkaiba1/explore-persona-space-data"
    assert repo_type == "dataset" and sha == _I931_SHA
    # Dedup: the same count claimed in the link TEXT and in the preceding
    # parenthetical is ONE claim tuple.
    both = f"(9 files, verified): [pairs, 9 files]({_I931_REPO}/tree/{_I931_SHA}/pairs)"
    assert len(verify_task_body._gather_hf_count_claims(both)) == 1


def test_hf_count_extractor_negative_cases():
    """Shapes that must NOT extract (precision-first; each guards a concrete
    false-positive class from the live #931 body)."""
    u = f"{_I931_REPO}/tree/abc1234/p"
    negatives = [
        f"[x]({u}) — 9 files",  # count in prose AFTER the link
        "[9 files](https://github.com/o/r/tree/abc1234/p)",  # non-HF link
        f"[9 files]({_I931_REPO}/tree/main/p)",  # moving ref, not hex-pinned
        f"[3 files]({_I931_REPO}/blob/abc1234/p/f.json)",  # /blob/ = single file
        f"[seed 42]({u})",  # no count-noun
        f"[8 eval JSONs]({u})",  # non-count noun (JSONs are records-adjacent)
        f"(total 515 files): [x]({u})",  # count does not OPEN the paren
        f"```\n[9 files]({u})\n```",  # inside a fenced code block
        f"(9 files) and separately see [x]({u})",  # separator gap exceeds the window
    ]
    for body in negatives:
        assert verify_task_body._gather_hf_count_claims(body) == [], body


def test_hf_count_claim_match_passes(monkeypatch):
    """A Pattern-A claim matching the (stubbed) files-only count → clean
    check-30 PASS with no WARN and no `unverified` note; overall ok."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    entries = [{"path": "pairs_meta", "type": "directory"}] + [
        {"path": f"pairs_meta/f{i}.json", "type": "file"} for i in range(9)
    ]
    _stub_tree(monkeypatch, status="ok", entries=entries)
    body = _count_claim_body(
        "pairs_meta, 9 files",
        f"{_I931_REPO}/tree/feedface/pairs_meta",
    )
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    r = by_name[_HF_30_NAME]
    assert r.passed and not r.is_warn
    assert "unverified" not in r.detail
    assert ok


def test_hf_count_claim_mismatch_warns_931_shape(monkeypatch):
    """The acceptance-criterion reproduction: the body claims 528 files where
    the pinned tree holds 515 files + 13 folders → WARN naming BOTH numbers
    plus the files+folders diagnostic; overall ok STAYS True (a WARN never
    blocks)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    entries = (
        [{"path": "issue931_story_map", "type": "directory"}]
        + [{"path": f"issue931_story_map/f{i}.json", "type": "file"} for i in range(515)]
        + [{"path": f"issue931_story_map/d{j}", "type": "directory"} for j in range(13)]
    )
    _stub_tree(monkeypatch, status="ok", entries=entries)
    body = _count_claim_body(
        "issue931_story_map, 528 files",
        f"{_I931_REPO}/tree/{_I931_SHA}/issue931_story_map",
    )
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    r = by_name[_HF_30_NAME]
    assert r.passed and r.is_warn
    assert "528" in r.detail and "515 file(s)" in r.detail
    assert "consistent with files+folders" in r.detail
    assert ok  # WARN never flips overall ok


def test_hf_count_plain_mismatch_warns_without_diagnostic(monkeypatch):
    """An overcount that does NOT equal files+folders WARNs naming both
    numbers WITHOUT the files+folders diagnostic (12 != 9 + 1) and WITHOUT
    the subset hedge (an overcount cannot describe a subset)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    entries = [{"path": f"p/f{i}.json", "type": "file"} for i in range(9)] + [
        {"path": "p/sub", "type": "directory"}
    ]
    _stub_tree(monkeypatch, status="ok", entries=entries)
    body = "Data: [p, 12 files](https://huggingface.co/datasets/o/r/tree/abc1234def/p)\n"
    r = verify_task_body.check_hf_file_count_claims(body)
    assert r.passed and r.is_warn
    assert "12" in r.detail and "9 file(s)" in r.detail
    assert "files+folders" not in r.detail
    assert "subset of the prefix" not in r.detail


def test_hf_count_undercount_mismatch_carries_subset_hedge(monkeypatch):
    """An UNDERCOUNT mismatch carries the descriptive hedge that the claim
    may describe a subset of the prefix (concern (b))."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    entries = [{"path": f"p/f{i}.json", "type": "file"} for i in range(9)]
    _stub_tree(monkeypatch, status="ok", entries=entries)
    body = "Data: [p, 5 files](https://huggingface.co/datasets/o/r/tree/abc1234def/p)\n"
    r = verify_task_body.check_hf_file_count_claims(body)
    assert r.passed and r.is_warn
    assert "5" in r.detail and "9 file(s)" in r.detail
    assert "subset of the prefix" in r.detail


def test_hf_count_shard_claims_one_sided(monkeypatch):
    """Shard claims are one-sided: claimed <= files is a clean PASS (shards +
    a manifest legitimately undercount files); claimed > files — the #931
    folder-inflation signature — WARNs."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    entries10 = [{"path": f"p/shard{i}.bin", "type": "file"} for i in range(9)] + [
        {"path": "p/manifest.json", "type": "file"}
    ]
    _stub_tree(monkeypatch, status="ok", entries=entries10)
    body_under = "Data: [p, 9 shards](https://huggingface.co/datasets/o/r/tree/abc1234def/p)\n"
    r = verify_task_body.check_hf_file_count_claims(body_under)
    assert r.passed and not r.is_warn

    # Same (repo, sha, prefix) key with a DIFFERENT stubbed listing — clear
    # the definitive cache so the second probe is not served the 10-file count.
    verify_task_body._HF_TREE_FILE_COUNT_CACHE.clear()
    entries9 = [{"path": f"p/shard{i}.bin", "type": "file"} for i in range(9)]
    _stub_tree(monkeypatch, status="ok", entries=entries9)
    body_over = "Data: [p, 10 shards](https://huggingface.co/datasets/o/r/tree/abc1234def/p)\n"
    r2 = verify_task_body.check_hf_file_count_claims(body_over)
    assert r2.passed and r2.is_warn
    assert "10" in r2.detail and "9 file(s)" in r2.detail


def test_hf_count_network_error_skips(monkeypatch):
    """A transient probe failure (429) and a `not_found` BOTH degrade to an
    `unverified` note on a PASS line — never a FAIL, never a WARN (the
    check-25-style not_found mapping: a WARN check never manufactures a
    verdict it cannot substantiate)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    body = "Data: [p, 9 files](https://huggingface.co/datasets/o/r/tree/abc1234def/p)\n"
    _stub_tree(monkeypatch, status="indeterminate", note="HF tree probe failed: HTTP 429")
    r = verify_task_body.check_hf_file_count_claims(body)
    assert r.passed and not r.is_warn
    assert "unverified" in r.detail and "HTTP 429" in r.detail
    _stub_tree(monkeypatch, status="not_found")
    r2 = verify_task_body.check_hf_file_count_claims(body)
    assert r2.passed and not r2.is_warn
    assert "unverified" in r2.detail and "no such revision/path" in r2.detail


def test_hf_count_offline_fence_never_touches_network(monkeypatch):
    """Under the EPM_VERIFY_BODY_NO_HF fence the check issues ZERO GETs —
    the tree getter is stubbed to raise, so a single probe fails the test."""
    monkeypatch.setenv("EPM_VERIFY_BODY_NO_HF", "1")

    def _boom(url, params, headers, *, timeout_s):  # pragma: no cover
        raise AssertionError("network touched under the offline fence")

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _boom)
    body = "Data: [p, 9 files](https://huggingface.co/datasets/o/r/tree/abc1234def/p)\n"
    r = verify_task_body.check_hf_file_count_claims(body)
    assert r.passed and not r.is_warn
    assert "HF probe fenced" in r.detail


def test_hf_count_zero_claims_zero_probes(monkeypatch):
    """A claim-free body is a vacuous PASS with ZERO Hub probes even with the
    fence REMOVED — `_hf_tree_get` is stubbed to raise, so a single GET
    would fail the test (GOOD_BODY's HF link labels — "raw completions",
    "hf-hub" — carry no count claim)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)

    def _boom(url, params, headers, *, timeout_s):  # pragma: no cover
        raise AssertionError("probe issued on a claim-free body")

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _boom)
    r = verify_task_body.check_hf_file_count_claims(GOOD_BODY)
    assert r.passed and not r.is_warn
    assert "no file-count claims" in r.detail


def test_hf_count_importerror_skips(monkeypatch):
    """A missing `huggingface_hub` degrades to an `unverified` skip note on a
    PASS line (the optional-dependency guard)."""
    import builtins

    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError("huggingface_hub blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    body = "Data: [p, 9 files](https://huggingface.co/datasets/o/r/tree/abc1234def/p)\n"
    r = verify_task_body.check_hf_file_count_claims(body)
    assert r.passed and not r.is_warn
    assert "unverified" in r.detail and "huggingface_hub unavailable" in r.detail


def test_hf_count_pagination_cap_skips(monkeypatch):
    """A listing that never exhausts (every page carries a next-page link)
    hits the page cap → skip note, PASS, never a WARN — a PARTIAL count must
    never ground a mismatch."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    calls: list = []
    _stub_tree(
        monkeypatch,
        status="ok",
        entries=[{"path": "p/f.json", "type": "file"}],
        next_page="https://huggingface.co/api/datasets/o/r/tree/abc1234def/p?cursor=X",
        calls=calls,
    )
    body = "Data: [p, 9 files](https://huggingface.co/datasets/o/r/tree/abc1234def/p)\n"
    r = verify_task_body.check_hf_file_count_claims(body)
    assert r.passed and not r.is_warn
    assert "unverified" in r.detail and "exceeded page/time cap" in r.detail
    assert len(calls) == verify_task_body._HF_PROBE_MAX_PAGES


def test_hf_count_pagination_two_pages_accumulates(monkeypatch):
    """A two-page listing accumulates file counts across pages (300 + 215 =
    515 → clean PASS) within the bounded request budget; both pages are
    genuinely fetched."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    page2 = "https://huggingface.co/api/datasets/o/r/tree/abc1234def/p?cursor=PAGE2"
    calls: list[str] = []

    def _fake(url, params, headers, *, timeout_s):
        calls.append(url)
        if "PAGE2" in url:
            entries = [{"path": f"p/g{i}.json", "type": "file"} for i in range(215)]
            return verify_task_body._TreeProbeResult("ok", entries, None, "")
        entries = [{"path": f"p/f{i}.json", "type": "file"} for i in range(300)]
        return verify_task_body._TreeProbeResult("ok", entries, page2, "")

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _fake)
    body = "Data: [p, 515 files](https://huggingface.co/datasets/o/r/tree/abc1234def/p)\n"
    r = verify_task_body.check_hf_file_count_claims(body)
    assert r.passed and not r.is_warn, r.detail
    assert "unverified" not in r.detail
    max_expected = verify_task_body._HF_PROBE_MAX_PAGES * verify_task_body._HF_PROBE_ATTEMPTS
    assert 0 < len(calls) <= max_expected
    assert any("PAGE2" in c for c in calls)
    assert len(calls) == 2


def test_hf_count_probe_deduped_and_cached(monkeypatch):
    """Two different-count claims on the SAME (repo, sha, prefix) issue
    exactly ONE probe (intra-invocation memo + definitive cache); the
    mismatching claim still WARNs."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    calls: list = []
    entries = [{"path": f"p/f{i}.json", "type": "file"} for i in range(9)]
    _stub_tree(monkeypatch, status="ok", entries=entries, calls=calls)
    body = (
        "Data: [p, 9 files](https://huggingface.co/datasets/o/r/tree/abc1234def/p) and the "
        "footer (10 files, incl. sidecar): "
        "[p @ abc1234](https://huggingface.co/datasets/o/r/tree/abc1234def/p)\n"
    )
    claims = verify_task_body._gather_hf_count_claims(body)
    assert len(claims) == 2  # 9-files and 10-files are distinct claims on one probe key
    r = verify_task_body.check_hf_file_count_claims(body)
    assert len(calls) == 1  # one probe for the shared (repo, sha, prefix) key
    assert r.passed and r.is_warn  # the 10-files claim mismatches the 9 files on the Hub
    assert "10" in r.detail and "9 file(s)" in r.detail


def test_hf_count_per_body_probe_cap(monkeypatch):
    """More unique prefixes than _HF_COUNT_MAX_PROBES: the first 8 probe, the
    9th surfaces a per-body-probe-cap `unverified` note — never a WARN."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    n = verify_task_body._HF_COUNT_MAX_PROBES + 1
    calls: list = []
    entries = [{"path": f"p{k}/f.json", "type": "file"} for k in range(n)]
    _stub_tree(monkeypatch, status="ok", entries=entries, calls=calls)
    body = (
        "\n".join(
            f"- [p{k}, 1 file](https://huggingface.co/datasets/o/r/tree/abc1234def/p{k})"
            for k in range(n)
        )
        + "\n"
    )
    r = verify_task_body.check_hf_file_count_claims(body)
    assert r.passed and not r.is_warn
    assert "per-body probe cap" in r.detail
    assert f"{n} claim(s) checked" in r.detail
    assert len(calls) == verify_task_body._HF_COUNT_MAX_PROBES


def test_hf_count_mismatch_and_unverified_coexist(monkeypatch):
    """When one prefix mismatches and another is throttled, the WARN detail
    carries BOTH the mismatch AND the unverified note (the unverified list
    is never dropped)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)

    def _fake(url, params, headers, *, timeout_s):
        if "pfx_a" in url:
            return verify_task_body._TreeProbeResult(
                "ok", [{"path": "pfx_a/f1.json", "type": "file"}], None, ""
            )
        return verify_task_body._TreeProbeResult(
            "indeterminate", [], None, "HF tree probe failed: HTTP 429"
        )

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _fake)
    body = (
        "- [pfx_a, 2 files](https://huggingface.co/datasets/o/r/tree/abc1234def/pfx_a)\n"
        "- [pfx_b, 3 files](https://huggingface.co/datasets/o/r/tree/abc1234def/pfx_b)\n"
    )
    r = verify_task_body.check_hf_file_count_claims(body)
    assert r.passed and r.is_warn
    assert "2 files at `pfx_a`" in r.detail and "1 file(s)" in r.detail
    assert "unverified (count not confirmed)" in r.detail and "pfx_b" in r.detail


def test_hf_count_repo_root_link_empty_prefix(monkeypatch):
    """A repo-root `/tree/<sha>` link (no path) probes the ROOT tree URL
    (`_hf_tree_url(..., "")`) and displays the empty prefix as `/`."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    calls: list = []
    entries = [
        {"path": "a.json", "type": "file"},
        {"path": "b.json", "type": "file"},
        {"path": "sub", "type": "directory"},
    ]
    _stub_tree(monkeypatch, status="ok", entries=entries, calls=calls)
    body = "Data: [3 files](https://huggingface.co/datasets/o/r/tree/abc1234def)\n"
    r = verify_task_body.check_hf_file_count_claims(body)
    assert r.passed and r.is_warn
    assert "at `/`" in r.detail and "2 file(s)" in r.detail
    assert "consistent with files+folders" in r.detail  # 3 == 2 files + 1 folder
    url, _params = calls[0]
    assert url.endswith("/tree/abc1234def")  # empty prefix → the root tree URL


# ─── Check 30 Pattern C: per-namespace + pinned-revision claims (#1088) ────
# #833's footer claimed "908 files listed per namespace at the pinned
# revision" in a parenthetical AFTER the pinned tree link — a position
# neither Pattern A nor B could parse (908 = 891 blobs + 17 directory
# entries per namespace: list_repo_tree ENTRIES counted as files).
# Extractor tests are pure; probe tests stub `_hf_tree_get` after removing
# the conftest EPM_VERIFY_BODY_NO_HF fence.

_I833_SHA = "fb4fe90fdd836ba2efd896b90c17e6b42f143d21"
_I833_NAMESPACES = (
    "analysis_tensors_nonemit",
    "analysis_tensors_matchedN",
    "analysis_tensors_nonemit_eq5",
)
_I833_SUB_PREFIXES = {f"issue833_onpolicy_map/{ns}" for ns in _I833_NAMESPACES}
# The ORIGINAL wrong paren (git 087c9df726) vs the corrected live paren —
# DIFFERENT filler around the anchor phrase; assertions key on the anchor
# phrase ("files (listed )?per namespace"), never the filler.
_I833_WRONG_PAREN = (
    "873 cell npz + manifests each; 908 files listed per namespace at the pinned revision"
)
_I833_CORRECTED_PAREN = (
    "873 cell npz + 18 per-source summary/manifest JSONs = 891 files per namespace "
    "at the pinned revision"
)


def _i833_footer(paren_content: str) -> str:
    """The verbatim #833 footer link (link TEXT naming three backtick `dir/`
    namespaces, URL pinned at the PARENT prefix `issue833_onpolicy_map`)
    followed by ONE parenthetical whose FULL content is parameterized."""
    return (
        "round-2 subset tensors [`analysis_tensors_nonemit/`, "
        "`analysis_tensors_matchedN/`, `analysis_tensors_nonemit_eq5/` @fb4fe90fdd]"
        f"({_I931_REPO}/tree/{_I833_SHA}/issue833_onpolicy_map) ({paren_content})"
    )


def _i833_probe_body(paren_content: str) -> str:
    """GOOD_BODY with its dataset HF link replaced by the #833 footer (the
    only pinned HF URL left in scope — the model link is dropped), for
    verify_text-level probe tests."""
    body = GOOD_BODY.replace(
        "[raw completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc123def/raw_completions/run.jsonl)",
        _i833_footer(paren_content),
    )
    return body.replace(
        "- Model: [hf-hub](https://huggingface.co/superkaiba1/explore-persona-space/tree/abc123def)\n",
        "- Model: not uploaded yet\n",
    )


def _needle_from_url(url: str, sha: str) -> str:
    """Decode a tree-endpoint probe URL's path component back to the raw
    needle — `_hf_tree_url` `quote(path, safe="")`-encodes it, so a
    sub-prefix's `/` arrives as `%2F`. Empty string for a root listing."""
    from urllib.parse import unquote

    marker = f"/tree/{sha}/"
    return unquote(url.split(marker, 1)[1]) if marker in url else ""


def _i833_needle_stub(monkeypatch, calls, *, per_ns_files=891, per_ns_dirs=17):
    """Stub `_hf_tree_get` deriving the requested needle from the URL and
    returning `per_ns_files` file entries + `per_ns_dirs` directory entries
    under it, plus the needle's own directory entry. A ROOT listing (check
    23's parent probe for the footer's `issue833_onpolicy_map` path) returns
    the parent dir entry so the existence check passes too."""

    def _fake(url, params, headers, *, timeout_s):
        calls.append((url, params))
        needle = _needle_from_url(url, _I833_SHA)
        if not needle:  # check 23's root listing (parent of the footer path)
            return verify_task_body._TreeProbeResult(
                "ok", [{"path": "issue833_onpolicy_map", "type": "directory"}], None, ""
            )
        entries = [{"path": needle, "type": "directory"}]
        entries += [{"path": f"{needle}/cell{i}.npz", "type": "file"} for i in range(per_ns_files)]
        entries += [{"path": f"{needle}/d{j}", "type": "directory"} for j in range(per_ns_dirs)]
        return verify_task_body._TreeProbeResult("ok", entries, None, "")

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _fake)


def test_hf_count_extractor_per_namespace_833_shape():
    """Pure extractor: the verbatim #833 footer (original wrong 908 form)
    yields exactly one per-namespace claim with the parent prefix + the three
    link-text namespaces; the corrected 891 form yields count 891; and the
    whole-prefix gatherer returns [] on BOTH (no A/B/C-pinned misread — the
    per-namespace qualifier is invisible to the whole-prefix patterns)."""
    wrong = _i833_footer(_I833_WRONG_PAREN)
    claims = verify_task_body._gather_hf_per_namespace_claims(wrong)
    assert claims == [
        (
            908,
            "superkaiba1/explore-persona-space-data",
            "dataset",
            _I833_SHA,
            "issue833_onpolicy_map",
            _I833_NAMESPACES,
        )
    ]
    corrected = _i833_footer(_I833_CORRECTED_PAREN)
    assert [c[0] for c in verify_task_body._gather_hf_per_namespace_claims(corrected)] == [891]
    assert verify_task_body._gather_hf_count_claims(wrong) == []
    assert verify_task_body._gather_hf_count_claims(corrected) == []
    # Gatherer dedup: the identical footer repeated twice in one body is ONE
    # claim tuple.
    twice = wrong + "\n\n" + wrong
    assert len(verify_task_body._gather_hf_per_namespace_claims(twice)) == 1
    # Comma-grouped + case-insensitive per-namespace positive.
    cased = f"[`ns_a/` @abc1234]({_I931_REPO}/tree/abc1234def/p) (1,234 Files Per Namespace)"
    cased_claims = verify_task_body._gather_hf_per_namespace_claims(cased)
    assert [(c[0], c[5]) for c in cased_claims] == [(1234, ("ns_a",))]


def test_hf_count_extractor_pinned_revision_form():
    """Pure extractor: the anchored `N files at the pinned revision` phrase in
    a paren AFTER a pinned tree link extracts a WHOLE-prefix claim through
    `_gather_hf_count_claims` (Pattern C); the combined #833 phrase (`... per
    namespace at the pinned revision`) yields ZERO pinned-revision claims —
    adjacency exclusivity ("per namespace" intervenes between "files" and
    "at")."""
    body = f"[x @abc1234]({_I931_REPO}/tree/abc1234def/p) (1,234 files at the pinned revision)"
    claims = verify_task_body._gather_hf_count_claims(body)
    assert [(c[0], c[1], c[5]) for c in claims] == [(1234, "files", "p")]
    assert verify_task_body._gather_hf_per_namespace_claims(body) == []
    combined = _i833_footer(_I833_WRONG_PAREN)
    assert verify_task_body._gather_hf_count_claims(combined) == []


def test_hf_count_extractor_per_namespace_negative_cases():
    """Shapes that must NOT extract from EITHER gatherer (precision-first;
    each guards a concrete false-positive class). Includes the two §5.2
    guard arms: a per-namespace-qualified count in link-TEXT position
    (Pattern A arm) and in paren-BEFORE-link position (Pattern B arm — the
    round-1 Must-Fix: dropping the `_COUNT_PAREN_LINK_RE` lookahead alone
    must turn this fixture red)."""
    link_plain = f"[x @abc1234]({_I931_REPO}/tree/abc1234def/p)"
    link_ns = f"[`ns_a/` @abc1234]({_I931_REPO}/tree/abc1234def/p)"
    negatives = [
        f"{link_plain} (891 files per seed)",  # anchor requires literal "per namespace"
        "[`ns_a/` @abc1234](https://github.com/o/r/tree/abc1234def/p) (891 files per namespace)",  # non-HF link
        f"[`ns_a/` @main]({_I931_REPO}/tree/main/p) (891 files per namespace)",  # moving ref
        f"[`f.json` @abc1234]({_I931_REPO}/blob/abc1234def/p/f.json) (891 files per namespace)",  # /blob/
        "The namespaces hold 908 files listed per namespace at the pinned revision.",  # link-free prose (the #1088 body shape)
        f"```\n{_i833_footer(_I833_WRONG_PAREN)}\n```",  # fenced block
        f"{link_ns} (873 cell npz + 18 JSONs)",  # no anchor phrase
        f"{link_ns} (one file per namespace)",  # no digit
        f"[891 files per namespace @abc1234]({_I931_REPO}/tree/abc1234def/p)",  # guard arm A: link-TEXT position
        f"(891 files per namespace): [x]({_I931_REPO}/tree/abc1234def/p)",  # guard arm B: paren-BEFORE-link position
        link_ns + "\n\n(891 files per namespace)",  # blank-line gap — Pattern C is same-line only
        link_ns + "   (891 files per namespace)",  # 3-space gap — outside the separator bound
    ]
    for body in negatives:
        assert verify_task_body._gather_hf_per_namespace_claims(body) == [], body
        assert verify_task_body._gather_hf_count_claims(body) == [], body


def test_hf_count_extractor_ab_guard_preserves_plain_claims():
    """The §5.2 negative lookahead does not disturb plain Pattern A/B
    extraction (regression companion to the existing A/B tests, exercising
    the MODIFIED regexes)."""
    plain_a = f"[pairs_meta, 9 files]({_I931_REPO}/tree/{_I931_SHA}/pairs_meta)"
    assert [(c[0], c[1]) for c in verify_task_body._gather_hf_count_claims(plain_a)] == [
        (9, "files")
    ]
    plain_b = (
        "(515 files verified via scoped listing): "
        f"[issue931_story_map @ 9534b998]({_I931_REPO}/tree/{_I931_SHA}/issue931_story_map)"
    )
    assert [(c[0], c[1]) for c in verify_task_body._gather_hf_count_claims(plain_b)] == [
        (515, "files")
    ]


def test_hf_count_per_namespace_mismatch_warns_833_shape(monkeypatch):
    """Acceptance criterion 1: the #833 footer restored to its ORIGINAL wrong
    908 form WARNs naming 908, 891 file(s), a namespace path, and the
    files+folders diagnostic (908 = 891 + 17); `passed` stays True; overall
    `ok` stays True; the probe set is EXACTLY the three sub-prefixes joined
    as `<link-prefix>/<ns>` (memo-deduplicated)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    calls: list = []
    _i833_needle_stub(monkeypatch, calls)
    ok, results = verify_task_body.verify_text(_i833_probe_body(_I833_WRONG_PAREN))
    r = _results_by_name(results)[_HF_30_NAME]
    assert r.passed and r.is_warn
    assert "908" in r.detail and "891 file(s)" in r.detail
    assert "issue833_onpolicy_map/analysis_tensors_nonemit" in r.detail
    assert "consistent with files+folders" in r.detail
    assert ok  # WARN never flips overall ok
    needles = [n for n in (_needle_from_url(u, _I833_SHA) for u, _p in calls) if n]
    # Exact-set assertion on the check-30 per-namespace probes (a wrong join
    # — e.g. a bare `analysis_tensors_nonemit` missing the parent prefix —
    # fails this); check 23's existence probe lists the ROOT (no needle).
    assert set(needles) == _I833_SUB_PREFIXES
    assert len(needles) == 3  # memo-deduplicated: one probe per sub-prefix


def test_hf_count_per_namespace_match_passes(monkeypatch):
    """Acceptance criterion 2: the corrected 891 form under the same stub is
    a clean PASS — no WARN, no unverified note, `1 claim(s) checked`."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    calls: list = []
    _i833_needle_stub(monkeypatch, calls)
    ok, results = verify_task_body.verify_text(_i833_probe_body(_I833_CORRECTED_PAREN))
    r = _results_by_name(results)[_HF_30_NAME]
    assert r.passed and not r.is_warn, r.detail
    assert "unverified" not in r.detail
    assert "1 claim(s) checked" in r.detail
    assert ok


def test_hf_count_per_namespace_no_names_unverified(monkeypatch):
    """A per-namespace claim whose link TEXT names no backtick `dir/` tokens
    surfaces as an `unverified` note with ZERO Hub GETs — never a WARN,
    never a parent-prefix guess (the stub raises on any call)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)

    def _boom(url, params, headers, *, timeout_s):  # pragma: no cover
        raise AssertionError("probe issued for a no-namespaces per-namespace claim")

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _boom)
    body = (
        f"round-2 subset tensors [round-2 subset tensors @fb4fe90fdd]({_I931_REPO}/tree/"
        f"{_I833_SHA}/issue833_onpolicy_map) (891 files per namespace)"
    )
    r = verify_task_body.check_hf_file_count_claims(body)
    assert r.passed and not r.is_warn
    assert "unverified" in r.detail and "per-namespace claim" in r.detail


def test_hf_count_per_namespace_offline_fence(monkeypatch):
    """Under the EPM_VERIFY_BODY_NO_HF fence a per-namespace claim WITH
    resolvable namespaces issues ZERO GETs and surfaces per-namespace
    `HF probe fenced` unverified notes on a PASS line."""
    monkeypatch.setenv("EPM_VERIFY_BODY_NO_HF", "1")

    def _boom(url, params, headers, *, timeout_s):  # pragma: no cover
        raise AssertionError("network touched under the offline fence")

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _boom)
    r = verify_task_body.check_hf_file_count_claims(_i833_footer("891 files per namespace"))
    assert r.passed and not r.is_warn
    assert "HF probe fenced" in r.detail


def test_hf_count_per_namespace_partial_skip_never_warns(monkeypatch):
    """Mixed per-namespace outcomes: ns1 exhaustive match, ns2 indeterminate
    (429), ns3 exhaustive MISMATCH with ZERO directory entries (pins the
    PLAIN mismatch wording, not the files+folders diagnostic) → exactly one
    WARN naming ns3, one unverified note naming ns2, `passed` True."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)

    def _fake(url, params, headers, *, timeout_s):
        needle = _needle_from_url(url, _I833_SHA)
        if needle.endswith("_nonemit_eq5"):  # ns3: 890 files, zero dirs → plain mismatch
            entries = [{"path": f"{needle}/cell{i}.npz", "type": "file"} for i in range(890)]
            return verify_task_body._TreeProbeResult("ok", entries, None, "")
        if needle.endswith("_matchedN"):  # ns2: throttled
            return verify_task_body._TreeProbeResult(
                "indeterminate", [], None, "HF tree probe failed: HTTP 429"
            )
        entries = [{"path": f"{needle}/cell{i}.npz", "type": "file"} for i in range(891)]
        return verify_task_body._TreeProbeResult("ok", entries, None, "")

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _fake)
    r = verify_task_body.check_hf_file_count_claims(_i833_footer("891 files per namespace"))
    assert r.passed and r.is_warn
    assert r.detail.count("body claims") == 1  # exactly one WARN (ns3)
    assert "analysis_tensors_nonemit_eq5" in r.detail and "890 file(s)" in r.detail
    assert "files+folders" not in r.detail  # dirs == 0 → plain branch
    assert "subset of the namespace" not in r.detail  # overcount → no subset hedge
    assert "unverified (count not confirmed)" in r.detail
    assert "analysis_tensors_matchedN" in r.detail and "HTTP 429" in r.detail


def test_hf_count_per_namespace_probe_cap(monkeypatch):
    """With `_HF_COUNT_MAX_PROBES` at 2 the THIRD namespace surfaces as the
    per-body-probe-cap unverified note — never a WARN from the capped
    namespace; exactly two probes are issued."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    monkeypatch.setattr(verify_task_body, "_HF_COUNT_MAX_PROBES", 2)
    calls: list = []
    _i833_needle_stub(monkeypatch, calls, per_ns_files=891, per_ns_dirs=0)
    r = verify_task_body.check_hf_file_count_claims(_i833_footer("891 files per namespace"))
    assert r.passed and not r.is_warn
    assert "per-body probe cap" in r.detail
    assert "analysis_tensors_nonemit_eq5" in r.detail  # the capped third namespace
    assert len(calls) == 2


def test_hf_count_shared_cap_across_claim_kinds(monkeypatch):
    """The memo/cap accounting is SHARED across the whole-prefix and
    per-namespace verification loops (`_probed` closure contract): one
    Pattern-A claim + two per-namespace claims with cap 2 → exactly 2 probes
    (whole prefix, then first namespace); the SECOND per-namespace claim's
    re-reference to the already-probed `parent/ns1` is served from the memo
    PAST the cap (no cap note for it), while the fresh `parent/ns2` probe is
    cap-blocked with the one cap note."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    monkeypatch.setattr(verify_task_body, "_HF_COUNT_MAX_PROBES", 2)
    calls: list[str] = []

    def _fake(url, params, headers, *, timeout_s):
        needle = _needle_from_url(url, "abc1234def")
        calls.append(needle)
        if needle == "p":
            entries = [{"path": f"p/f{i}.json", "type": "file"} for i in range(9)]
        elif needle == "parent/ns1":
            entries = [{"path": f"parent/ns1/c{i}.npz", "type": "file"} for i in range(5)]
        else:  # pragma: no cover
            raise AssertionError(f"cap must block any further probe (got {needle!r})")
        return verify_task_body._TreeProbeResult("ok", entries, None, "")

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _fake)
    body = (
        f"- [p, 9 files]({_I931_REPO}/tree/abc1234def/p)\n"
        f"- [`ns1/` @abc1234]({_I931_REPO}/tree/abc1234def/parent) (5 files per namespace)\n"
        f"- [`ns1/`, `ns2/` @abc1234]({_I931_REPO}/tree/abc1234def/parent) "
        "(5 files per namespace)\n"
    )
    r = verify_task_body.check_hf_file_count_claims(body)
    assert calls == ["p", "parent/ns1"]  # 2 probes; memo served the ns1 re-reference
    assert r.passed and not r.is_warn  # everything probed matches
    assert r.detail.count("per-body probe cap") == 1  # only the fresh parent/ns2 was capped
    assert "parent/ns2" in r.detail
    assert "3 claim(s) checked" in r.detail  # 1 whole-prefix + 2 per-namespace claims


# ─── Check 30 Pattern D: backtick sub-path + parenthetical count (#1143) ───

_I1112_SHA = "e016910195b7ab846c83b87ec43140c36c51e35f"


def _i1112_footer() -> str:
    """The FULL verbatim `**Repro:**` footer row of the #1112 incident body
    (line 224 of tasks/followups_running/1112/body.md at fix time, copied
    verbatim into this suite — never read from the live body at test time,
    so a later status move / body edit cannot break the tests). Carries the
    corrected 7,165 claim after the backtick `raw_completions/` token, TWO
    hex-pinned HF /tree links BEFORE the claim (the `bootstrap_matrices`
    sub-prefix link, then the `issue1112_geometry2x2/` bucket link the
    claim must bind to — nearest-of-N, not first-of-N), interleaved GitHub
    links, and sibling backtick tokens with non-count parens."""
    return (
        "**Repro:** ~55–70 GPU-h realized (one GCP smoke + three RunPod production attempts, 4×"
        " H100 / 4× A100-80; geometry, bootstrap, and figures on the VM CPU) · code at [`d61d50"
        "06df`](https://github.com/superkaiba/explore-persona-space/tree/d61d5006df66bbdb6138ab"
        "6fe5f91fa9644352b8) on branch `issue-1112`: [issue1112_dispatch.py](https://github.com"
        "/superkaiba/explore-persona-space/blob/d61d5006df66bbdb6138ab6fe5f91fa9644352b8/script"
        "s/issue1112_dispatch.py) · [issue1112_geometry.py](https://github.com/superkaiba/explo"
        "re-persona-space/blob/d61d5006df66bbdb6138ab6fe5f91fa9644352b8/scripts/issue1112_geome"
        "try.py) · [issue1112_train_marker_fullft.py](https://github.com/superkaiba/explore-per"
        "sona-space/blob/d61d5006df66bbdb6138ab6fe5f91fa9644352b8/scripts/issue1112_train_marke"
        "r_fullft.py) · [issue1112_figures.py](https://github.com/superkaiba/explore-persona-sp"
        "ace/blob/d61d5006df66bbdb6138ab6fe5f91fa9644352b8/scripts/issue1112_figures.py) · eval"
        " JSONs: [eval_results/issue_1112](https://github.com/superkaiba/explore-persona-space/"
        "tree/d61d5006df66bbdb6138ab6fe5f91fa9644352b8/eval_results/issue_1112) (`geometry_per_"
        "cell.json` generated 2026-07-08T06:31:06Z at git `81c3a85bf8`, n_boot 1000, boot seed "
        "653; per-cell records + per-draw [bootstrap matrices](https://huggingface.co/datasets/"
        "superkaiba1/explore-persona-space-data/tree/e016910195b7ab846c83b87ec43140c36c51e35f/i"
        "ssue1112_geometry2x2/analysis_tensors/bootstrap_matrices) back every aggregate; [`debi"
        "ased_cosine.json`](https://github.com/superkaiba/explore-persona-space/blob/0c6a367332"
        "0e59d9dceeb18aa3557a803f5e8496/eval_results/issue_1112/geometry/debiased_cosine.json) "
        "— paired half-draw direction-cosine CIs + attenuation references, generated 2026-07-08"
        "T08:57:07Z at git `0c6a367332`, m=60, 2000 draws, seed 1112) · data-repo bucket [issue"
        "1112_geometry2x2/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-d"
        "ata/tree/e016910195b7ab846c83b87ec43140c36c51e35f/issue1112_geometry2x2) — `mixes/` (f"
        "rozen, positives-only, generic-only, marker + derivation manifest), `selection/` (per-"
        "cell ladders + selected rungs), `raw_completions/` (7,165 files: tier-1 ladder rate re"
        "ads, tier-2, capture, marker, read-out-extraction rollout text), `analysis_tensors/{ca"
        "pture, rb}`, `margin/`, `run_config.json` · new checkpoints (7 cells) on the private o"
        "verflow model repo [issue1112/](https://huggingface.co/superkaiba1/explore-persona-spa"
        "ce-overflow/tree/90949b061d09b30d5850f2fec0043790939aa322/issue1112) (auth-required) ·"
        " figures + per-figure data sidecars: [figures/issue_1112](https://github.com/superkaib"
        "a/explore-persona-space/tree/d61d5006df66bbdb6138ab6fe5f91fa9644352b8/figures/issue_11"
        "12)"
    )


def test_hf_count_extractor_subpath_paren_1112_shape():
    """Pure extractor (Pattern D, #1143): the FULL verbatim #1112 footer row
    yields exactly ONE claim tuple, scoped to the JOINED
    `<link-prefix>/<sub-path>` at the bucket link's sha. Exact-equality also
    pins that the sibling tokens extract nothing: `mixes/` / `selection/`
    (non-count parens), `analysis_tensors/{capture, rb}` (brace outside the
    sub charset), `margin/` (no paren), `run_config.json` (no trailing
    slash)."""
    footer = _i1112_footer()
    # Mechanical pin (round-1 Statistics Must-Fix): the committed fixture
    # itself carries >=2 hex-pinned /tree markdown links BEFORE the claim's
    # backtick token, so the NEAREST-preceding binder semantics is durably
    # exercised by the suite — a truncated single-link fixture could not
    # discriminate nearest-preceding from a first-preceding-link mutant.
    claim_pos = footer.index("`raw_completions/`")
    pinned_before = []
    for lm in verify_task_body._MD_HF_LINK_RE.finditer(footer):
        if lm.end() > claim_pos:
            continue
        url = lm.group("url").rstrip(".,;:!?")
        m = verify_task_body._HF_HUB_TREE_BLOB_URL_RE.match(url)
        if m is not None and f"/tree/{m.group('sha')}" in url:
            pinned_before.append(url)
    assert len(pinned_before) >= 2, pinned_before
    claims = verify_task_body._gather_hf_count_claims(footer)
    assert claims == [
        (
            7165,
            "files",
            "superkaiba1/explore-persona-space-data",
            "dataset",
            _I1112_SHA,
            "issue1112_geometry2x2/raw_completions",
        )
    ]


def test_hf_count_extractor_subpath_negative_cases():
    """Shapes that must NOT extract via Pattern D (precision-first; each
    guards a concrete misbind / false-positive class from the binder + regex
    guards). The Pattern-B-adjacency shape and the #833 per-namespace shape
    are asserted separately below (B alone extracts; the #833 shape stays
    invisible to the whole-prefix gatherer — the line-2585 semantics)."""
    pin = f"{_I931_REPO}/tree/abc1234def/parent"
    link = f"[parent/]({pin})"
    negatives = [
        f"{link}\n`sub/` (7 files: rollout text)",  # newline in gap — not the same footer row
        (
            f"{link} — see [dispatch](https://github.com/o/r/blob/abc1234def/s.py) — "
            "`sub/` (7 files: x)"
        ),  # intervening markdown link in the gap
        f"{link} — [not a link] — `sub/` (7 files: x)",  # bare bracket in the gap
        f"[parent/]({_I931_REPO}/tree/main/parent) — `sub/` (7 files: x)",  # nearest: /tree/main
        (
            f"[f.json]({_I931_REPO}/blob/abc1234def/parent/f.json) — `sub/` (7 files: x)"
        ),  # nearest preceding link is /blob/<sha> — declined, never re-bound earlier
        f"{link} — " + "x" * 401 + " `sub/` (7 files: x)",  # gap > _SUBPATH_CLAIM_MAX_GAP
        f"{link} — `sub/` (total 7 files)",  # count does not OPEN the paren (mirrors B)
        f"{link} — `run_config.json` (7 files)",  # no trailing slash — check 32's territory
        f"{link} — `sub/` (891 files per namespace at the pinned revision)",  # per namespace
        f"{link} — `sub/` (11 files per adapter, 176 files total)",  # the #460 shape:
        # "per adapter" declines the opening 11; 176 does not open the paren
        "`sub/` (7 files: x) with no HF link anywhere before it",  # no preceding link
    ]
    for body in negatives:
        assert verify_task_body._gather_hf_count_claims(body) == [], body
    # Pattern-B adjacency: with a pinned link BEFORE the token AND an HF
    # markdown link right after the paren, D's trailing negative lookahead
    # declines so Pattern B ALONE extracts — ONE tuple scoped to the
    # FOLLOWING link's own path, and no joined `parent/sub` tuple.
    b_adjacent = f"{link} — `sub/` (7 files): [other @abc1234]({_I931_REPO}/tree/abc1234def/other)"
    b_claims = verify_task_body._gather_hf_count_claims(b_adjacent)
    assert [(c[0], c[5]) for c in b_claims] == [(7, "other")]
    # The #833 per-namespace wrong-shape still yields ZERO whole-prefix
    # claims under Pattern D (guards the pre-existing extractor semantics).
    assert verify_task_body._gather_hf_count_claims(_i833_footer(_I833_WRONG_PAREN)) == []


def test_hf_count_subpath_claim_match_passes(monkeypatch):
    """A Pattern-D claim matching the (stubbed) files-only count at the
    JOINED prefix → clean check-30 PASS: no WARN, no `unverified` note,
    `1 claim(s) checked` — the D tuple rides the existing whole-prefix
    verification leg unchanged."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    needle = "issue1112_geometry2x2/raw_completions"
    entries = [{"path": needle, "type": "directory"}]
    entries += [{"path": f"{needle}/f{i}.json", "type": "file"} for i in range(7165)]
    _stub_tree(monkeypatch, status="ok", entries=entries)
    r = verify_task_body.check_hf_file_count_claims(_i1112_footer())
    assert r.passed and not r.is_warn, r.detail
    assert "unverified" not in r.detail
    assert "1 claim(s) checked" in r.detail


def test_hf_count_subpath_mismatch_warns_folder_inflated_1112(monkeypatch):
    """The ORIGINAL #1112 incident claim (7,372) against a stubbed tree of
    7,165 files + 207 directories under the joined prefix → WARN whose
    detail names BOTH numbers, the files+folders diagnostic (7,372 = 7,165
    + 207 — the exact incident signature, reused from the existing
    whole-prefix leg, not rebuilt), and the JOINED prefix; `passed` stays
    True in both directions (WARN-only invariant — no `passed=False` path
    added)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    needle = "issue1112_geometry2x2/raw_completions"
    entries = [{"path": needle, "type": "directory"}]
    entries += [{"path": f"{needle}/f{i}.json", "type": "file"} for i in range(7165)]
    entries += [{"path": f"{needle}/d{j}", "type": "directory"} for j in range(207)]
    _stub_tree(monkeypatch, status="ok", entries=entries)
    body = _i1112_footer().replace("(7,165 files:", "(7,372 files:")
    assert "(7,372 files:" in body
    r = verify_task_body.check_hf_file_count_claims(body)
    assert r.passed and r.is_warn, r.detail
    assert "7372" in r.detail and "7165 file(s)" in r.detail
    assert "consistent with files+folders" in r.detail
    assert "issue1112_geometry2x2/raw_completions" in r.detail


def test_hf_count_subpath_scoping_probes_joined_prefix(monkeypatch):
    """The probe request targets the JOINED prefix at the cited sha — never
    the bare bucket prefix: exactly one probe, whose URL path (decoded via
    `_needle_from_url` — `_hf_tree_url` %2F-encodes the path, so a literal
    slash-joined substring never appears in the probe URL) is the joined
    sub-prefix."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    calls: list = []
    needle = "issue1112_geometry2x2/raw_completions"
    entries = [{"path": f"{needle}/f{i}.json", "type": "file"} for i in range(7165)]
    _stub_tree(monkeypatch, status="ok", entries=entries, calls=calls)
    verify_task_body.check_hf_file_count_claims(_i1112_footer())
    assert len(calls) == 1
    assert _needle_from_url(calls[0][0], _I1112_SHA) == needle


def test_hf_count_subpath_dedup_with_link_text_pattern():
    """A row where the SAME (count, joined prefix) is claimable via Pattern
    A (count in a link whose path IS the sub-prefix) and Pattern D (backtick
    sub-path off the parent link) yields ONE tuple — the shared `seen` set
    dedups across patterns, so one artifact never draws a double WARN."""
    row = (
        f"[sub, 7 files]({_I931_REPO}/tree/abc1234def/parent/sub) then "
        f"[parent/]({_I931_REPO}/tree/abc1234def/parent) — `sub/` (7 files: rollouts)"
    )
    claims = verify_task_body._gather_hf_count_claims(row)
    assert [(c[0], c[1], c[5]) for c in claims] == [(7, "files", "parent/sub")]


# ─── Check 32: HF-adjacent backtick file claims vs the pinned tree (WARN) ──
#
# Check 32 (`check_hf_adjacent_file_claims`, #1016) extracts backtick
# FILENAME claims adjacent to hex-pinned HF /tree markdown links — PAREN
# (a parenthetical immediately AFTER the link, the #952-r1 incident shape;
# check 30's paren is BEFORE the link) and LINKTEXT (a dotted backtick
# token inside the link text) — and tests any-depth basename membership
# against the same #733 bounded raw tree-endpoint probe stack checks
# 23/25/30 use. All tests are offline: extractor tests need no stub; probe
# tests stub `verify_task_body._hf_tree_get` after removing the conftest
# EPM_VERIFY_BODY_NO_HF fence.

_HF_32_NAME = "HF-adjacent backtick file claims exist under the pinned tree"

_I952_SHA = "5b62649cefb34902fd630f21630164e8d1d99764"
_I952_DATA_REPO = "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data"
_I952_EVAL_PREFIX = "issue952_position_divergence/eval_results"
_I952_EVAL_URL = f"{_I952_DATA_REPO}/tree/{_I952_SHA}/{_I952_EVAL_PREFIX}"
_I952_RAW_URL = f"{_I952_DATA_REPO}/tree/{_I952_SHA}/issue952_position_divergence/raw_completions"
_I952_GH_BLOB = (
    "https://github.com/superkaiba/explore-persona-space/blob/"
    "ac9f45b4ca42d7b55091a0fa169b8480e2fe0c62/eval_results/issue_952/"
    "divergence_bank_queries.json"
)

_I952_LEAD = (
    "Divergence-bank items are referenced by file + index only (standing content "
    "rule for sensitive query categories — no bank text is quoted anywhere in "
    "this body): the 229 judged candidate pairs with judge scores, refusal "
    "labels, and keep decisions are in "
)
_I952_TAIL = (
    ", and the bank generations + judge outputs are in "
    f"[HF …/raw_completions @ 5b62649]({_I952_RAW_URL})."
)

# The VERBATIM #952 r1 incident line (recover via
# `git show b412ddb07d:tasks/interpreting/952/body.md`, grep
# `divergence_bank_queries`): the paren after the pinned eval_results tree
# link claims BOTH bank files while `divergence_bank_queries.json` lived
# only in git — the must-WARN fixture. The dot-less backtick ids
# (`model_identity_004` / `style_format_037`) exercise the filename
# filter's no-extension rejection in the same shot.
_I952_R1_LINE = (
    _I952_LEAD
    + f"[HF issue952_position_divergence/eval_results @ 5b62649]({_I952_EVAL_URL}) "
    + "(`divergence_bank_verification.json`, `divergence_bank_queries.json`; "
    + "kept pairs carry ids of the form `model_identity_004` / `style_format_037`)"
    + _I952_TAIL
)

# The VERBATIM corrected #952 line (live body, `tasks/followups_running/952/
# body.md` line ~142): the HF paren claims only the verification file;
# `divergence_bank_queries.json` moved to a github-blob claim on the SAME
# line — the canonical must-NOT-warn fixture (structural anchoring must
# never attribute the github-linked filename to the HF link).
_I952_CORRECTED_LINE = (
    _I952_LEAD
    + f"[HF issue952_position_divergence/eval_results @ 5b62649]({_I952_EVAL_URL}) "
    + "(`divergence_bank_verification.json`) and in git at "
    + f"[`divergence_bank_queries.json` @ ac9f45b4ca]({_I952_GH_BLOB}) "
    + "(kept pairs carry ids of the form `model_identity_004` / `style_format_037`)"
    + _I952_TAIL
)


def test_hf_adjacent_claim_absent_warns_952_r1_shape(monkeypatch):
    """Acceptance criterion 1 — the VERBATIM #952-r1 line: the paren claims
    two bank files at the pinned eval_results tree, the stubbed exhaustive
    listing holds only `divergence_bank_verification.json` → a `[WARN]`
    naming the missing file + the pinned prefix + sha[:8] + the PAREN shape
    tag; `passed` stays True (WARN never FAILs)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    _stub_tree(
        monkeypatch,
        status="ok",
        entries=[
            {"path": f"{_I952_EVAL_PREFIX}/divergence_bank_verification.json", "type": "file"},
        ],
    )
    r = verify_task_body.check_hf_adjacent_file_claims(_I952_R1_LINE)
    assert r.passed and r.is_warn
    assert r.render().startswith("  [WARN]")
    assert "divergence_bank_queries.json" in r.detail
    assert _I952_EVAL_PREFIX in r.detail and _I952_SHA[:8] in r.detail
    assert "shape: PAREN" in r.detail
    # The PRESENT file is never reported missing.
    assert "claims `divergence_bank_verification.json`" not in r.detail


def test_hf_adjacent_claim_present_passes_any_depth(monkeypatch):
    """Same r1 body, but the listing carries BOTH claimed basenames — the
    queries file nested one level DEEPER than the prefix's direct children
    → clean PASS (any-depth membership), no WARN, no `unverified` note."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    _stub_tree(
        monkeypatch,
        status="ok",
        entries=[
            {"path": f"{_I952_EVAL_PREFIX}/divergence_bank_verification.json", "type": "file"},
            {"path": f"{_I952_EVAL_PREFIX}/sub", "type": "directory"},
            {"path": f"{_I952_EVAL_PREFIX}/sub/divergence_bank_queries.json", "type": "file"},
        ],
    )
    r = verify_task_body.check_hf_adjacent_file_claims(_I952_R1_LINE)
    assert r.passed and not r.is_warn, r.detail
    assert "unverified" not in r.detail
    assert "2 adjacent file claim(s) against 1 pinned tree(s)" in r.detail


def test_corrected_952_line_no_warn_and_github_never_probed(monkeypatch):
    """Acceptance criterion 2 — the VERBATIM corrected #952 line: the HF
    paren claims only the verification file (present in the stubbed listing
    one level below the prefix, mirroring the live Hub layout); the
    github-blob `divergence_bank_queries.json` claim on the SAME line is
    never attributed to the HF link, and the paren-less raw_completions
    link contributes zero claims. Exactly ONE claim extracts; the single
    probe targets the HF api (never github); no WARN."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    claims = verify_task_body._gather_hf_adjacent_file_claims(_I952_CORRECTED_LINE)
    assert [(c[4], c[5]) for c in claims] == [("divergence_bank_verification.json", "PAREN")]
    calls: list = []
    _stub_tree(
        monkeypatch,
        status="ok",
        entries=[
            {"path": f"{_I952_EVAL_PREFIX}/divergence_bank_verification.json", "type": "file"},
        ],
        calls=calls,
    )
    r = verify_task_body.check_hf_adjacent_file_claims(_I952_CORRECTED_LINE)
    assert r.passed and not r.is_warn, r.detail
    assert len(calls) == 1
    url, _params = calls[0]
    assert "github" not in url and "huggingface.co/api/datasets" in url


def test_hf_adjacent_linktext_shape_tree_url(monkeypatch):
    """LINKTEXT shape — a dotted backtick token inside a `/tree/<sha>/dir/`
    link's text: absent from the listing → WARN with the LINKTEXT shape
    tag; present → clean PASS (cache cleared between the two stubs — only
    exhaustive listings are cached)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    body = (
        "Raw rollouts: [`villain_seed42.json`]"
        "(https://huggingface.co/datasets/o/r/tree/abc1234def/dir/)\n"
    )
    _stub_tree(monkeypatch, status="ok", entries=[{"path": "dir/other.json", "type": "file"}])
    r = verify_task_body.check_hf_adjacent_file_claims(body)
    assert r.passed and r.is_warn
    assert "villain_seed42.json" in r.detail and "shape: LINKTEXT" in r.detail

    verify_task_body._HF_TREE_BASENAMES_CACHE.clear()
    _stub_tree(
        monkeypatch, status="ok", entries=[{"path": "dir/villain_seed42.json", "type": "file"}]
    )
    r2 = verify_task_body.check_hf_adjacent_file_claims(body)
    assert r2.passed and not r2.is_warn, r2.detail


def test_hf_adjacent_blob_url_out_of_scope():
    """A paren after a `/blob/` link and a dotted backtick filename inside a
    `/blob/` link's text both extract ZERO claims — check 23 already
    validates the full blob path."""
    u = "https://huggingface.co/datasets/o/r/blob/abc1234def/p/f.json"
    body = f"See [data]({u}) (`g.json`) and [`f.json` @ abc1234]({u}).\n"
    assert verify_task_body._gather_hf_adjacent_file_claims(body) == []


def test_hf_adjacent_filename_filter():
    """Extraction unit test for the dotted artifact-extension whitelist: the
    mixed real-corpus parenthetical extracts ONLY `pilot_gate.json`; paths,
    brace-globs, wildcard globs, no-dot tokens (pod names / shas), `.py`
    scripts, and >64-char stems are all rejected by construction."""
    u = "https://huggingface.co/datasets/o/r/tree/abc1234def/p"
    mixed = f"[gate artifacts]({u}) (`pilot_gate.json`, run on `eps-issue-642`, git `a0330df0e8`)"
    claims = verify_task_body._gather_hf_adjacent_file_claims(mixed)
    assert [(c[4], c[5]) for c in claims] == [("pilot_gate.json", "PAREN")]
    long_stem = "x" * 70
    rejected = [
        f"[x]({u}) (`on_policy_R/R_train.json`)",  # relative path — a subpath claim
        f"[x]({u}) (`R_{{train,eval}}.json`)",  # brace glob
        f"[x]({u}) (`*_responses.json`)",  # wildcard glob
        f"[x]({u}) (`gen.py`)",  # script — generator provenance, not an upload claim
        f"[x]({u}) (`{long_stem}.json`)",  # >64-char stem
        f"[x]({u}) (`no_extension_token`)",  # no dotted extension
    ]
    for body in rejected:
        assert verify_task_body._gather_hf_adjacent_file_claims(body) == [], body


def test_hf_adjacent_url_terminal_component_skipped():
    """A backtick token equal to the URL's own terminal path component is
    NOT a separate membership claim — check 23 already validates the URL's
    own path (zero claims, zero probes)."""
    body = (
        "Raw: [`run.jsonl`](https://huggingface.co/datasets/o/r/tree/abc1234def"
        "/raw_completions/run.jsonl)\n"
    )
    assert verify_task_body._gather_hf_adjacent_file_claims(body) == []


def test_hf_adjacent_offline_fence_never_touches_network(monkeypatch):
    """Under the EPM_VERIFY_BODY_NO_HF fence the check issues ZERO GETs —
    the tree getter is stubbed to raise, so a single probe fails the test;
    the claim surfaces as an `unverified` note on a PASS line."""
    monkeypatch.setenv("EPM_VERIFY_BODY_NO_HF", "1")

    def _boom(url, params, headers, *, timeout_s):  # pragma: no cover
        raise AssertionError("network touched under the offline fence")

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _boom)
    body = "Data: [x](https://huggingface.co/datasets/o/r/tree/abc1234def/p) (`f.json`)\n"
    r = verify_task_body.check_hf_adjacent_file_claims(body)
    assert r.passed and not r.is_warn
    assert "unverified" in r.detail and "HF probe fenced" in r.detail


def test_hf_adjacent_not_found_skips_not_warns(monkeypatch):
    """`not_found` degrades to an `unverified` note on a PASS line — never a
    WARN: check 23 owns the dead-pin FAIL (the documented
    check-23-vs-25/30/32 asymmetry), so double-reporting here is noise."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    _stub_tree(monkeypatch, status="not_found")
    body = "Data: [x](https://huggingface.co/datasets/o/r/tree/abc1234def/p) (`f.json`)\n"
    r = verify_task_body.check_hf_adjacent_file_claims(body)
    assert r.passed and not r.is_warn
    assert "unverified" in r.detail and "no such revision/path" in r.detail


def test_hf_adjacent_pagination_cap_skips(monkeypatch):
    """A listing that never exhausts (every page carries a next-page link)
    hits the page cap → skip note, PASS, never a WARN — a PARTIAL listing
    must never ground a missing-basename verdict, even when the pages seen
    so far LACK the claimed basename."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    calls: list = []
    _stub_tree(
        monkeypatch,
        status="ok",
        entries=[{"path": "p/other.json", "type": "file"}],
        next_page="https://huggingface.co/api/datasets/o/r/tree/abc1234def/p?cursor=X",
        calls=calls,
    )
    body = "Data: [x](https://huggingface.co/datasets/o/r/tree/abc1234def/p) (`f.json`)\n"
    r = verify_task_body.check_hf_adjacent_file_claims(body)
    assert r.passed and not r.is_warn
    assert "unverified" in r.detail and "exceeded page/time cap" in r.detail
    assert len(calls) == verify_task_body._HF_PROBE_MAX_PAGES


def test_hf_adjacent_probe_memo_one_probe_per_prefix(monkeypatch):
    """Two claims on ONE (repo, sha, prefix) issue exactly ONE listing walk
    (intra-invocation memo); both basenames verify against that single
    exhaustive listing."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    calls: list = []
    _stub_tree(
        monkeypatch,
        status="ok",
        entries=[
            {"path": "p/a.json", "type": "file"},
            {"path": "p/b.json", "type": "file"},
        ],
        calls=calls,
    )
    body = "Data: [x](https://huggingface.co/datasets/o/r/tree/abc1234def/p) (`a.json`, `b.json`)\n"
    r = verify_task_body.check_hf_adjacent_file_claims(body)
    assert r.passed and not r.is_warn, r.detail
    assert "2 adjacent file claim(s) against 1 pinned tree(s)" in r.detail
    assert len(calls) == 1


def test_hf_adjacent_fenced_code_block_not_scanned():
    """The claim pattern inside a ``` fenced block is illustrative — zero
    claims extract."""
    body = "```\nData: [x](https://huggingface.co/datasets/o/r/tree/abc1234def/p) (`f.json`)\n```\n"
    assert verify_task_body._gather_hf_adjacent_file_claims(body) == []


def test_hf_adjacent_directory_basename_suppresses_warn(monkeypatch):
    """A claimed dotted name matching a DIRECTORY-type entry suppresses the
    WARN (FP-safe: dotted directory names are rare, and a directory of that
    name still corroborates the claim's location)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    _stub_tree(monkeypatch, status="ok", entries=[{"path": "p/data.json", "type": "directory"}])
    body = "Data: [x](https://huggingface.co/datasets/o/r/tree/abc1234def/p) (`data.json`)\n"
    r = verify_task_body.check_hf_adjacent_file_claims(body)
    assert r.passed and not r.is_warn, r.detail


def test_hf_adjacent_importerror_skips(monkeypatch):
    """A missing `huggingface_hub` degrades to an `unverified` skip note on
    a PASS line, never a WARN (fail-soft parity with checks 23/25/30)."""
    import builtins

    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError("huggingface_hub blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    body = "Data: [x](https://huggingface.co/datasets/o/r/tree/abc1234def/p) (`f.json`)\n"
    r = verify_task_body.check_hf_adjacent_file_claims(body)
    assert r.passed and not r.is_warn
    assert "unverified" in r.detail and "huggingface_hub unavailable" in r.detail


def test_hf_adjacent_transient_network_error_skips_and_never_caches(monkeypatch):
    """A transient probe failure (429) degrades to an `unverified` note on a
    PASS line AND the skip is NEVER cached — `_HF_TREE_BASENAMES_CACHE`
    stays empty, so a cleared throttle is re-probed on the next
    invocation."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    _stub_tree(monkeypatch, status="indeterminate", note="HF tree probe failed: HTTP 429")
    body = "Data: [x](https://huggingface.co/datasets/o/r/tree/abc1234def/p) (`f.json`)\n"
    r = verify_task_body.check_hf_adjacent_file_claims(body)
    assert r.passed and not r.is_warn
    assert "unverified" in r.detail and "HTTP 429" in r.detail
    assert verify_task_body._HF_TREE_BASENAMES_CACHE == {}


def test_hf_adjacent_no_failing_checkresult_in_source():
    """Committed WARN-only pin: no `CheckResult(..., False, ...)` /
    `passed=False` construction anywhere in the check-32 function or its
    helpers — the durable form of the report-time grep (plan #1016 §4.6
    T16)."""
    import ast
    import inspect

    fns = [
        verify_task_body.check_hf_adjacent_file_claims,
        verify_task_body._gather_hf_adjacent_file_claims,
        verify_task_body._hf_basenames_under_prefix,
        verify_task_body._hf_basenames_for_prefix,
        verify_task_body._hf_tree_pages,
    ]
    for fn in fns:
        tree = ast.parse(inspect.getsource(fn))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if callee != "CheckResult":
                continue
            if len(node.args) >= 2:
                arg = node.args[1]
                assert not (isinstance(arg, ast.Constant) and arg.value is False), (
                    f"{fn.__name__} constructs CheckResult(..., False, ...)"
                )
            for kw in node.keywords:
                if kw.arg == "passed":
                    assert not (isinstance(kw.value, ast.Constant) and kw.value.value is False), (
                        f"{fn.__name__} constructs CheckResult(passed=False)"
                    )


def test_hf_adjacent_per_body_probe_cap(monkeypatch):
    """More unique prefixes than _HF_MEMBER_MAX_PROBES: the first 8 probe
    (each claim verifies), the 9th surfaces a per-body-probe-cap
    `unverified` note — never a WARN, `passed` stays True (the cap branch
    is behaviorally distinct from the page cap: no probe is even
    issued)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    n = verify_task_body._HF_MEMBER_MAX_PROBES + 1
    calls: list = []
    entries = [{"path": f"p{k}/f{k}.json", "type": "file"} for k in range(n)]
    _stub_tree(monkeypatch, status="ok", entries=entries, calls=calls)
    body = (
        "\n".join(
            f"- [p{k}](https://huggingface.co/datasets/o/r/tree/abc1234def/p{k}) (`f{k}.json`)"
            for k in range(n)
        )
        + "\n"
    )
    r = verify_task_body.check_hf_adjacent_file_claims(body)
    assert r.passed and not r.is_warn
    assert "per-body probe cap" in r.detail
    assert f"{n} adjacent file claim(s)" in r.detail
    assert len(calls) == verify_task_body._HF_MEMBER_MAX_PROBES


# ─── `_hf_tree_pages`: the shared bounded pagination generator (#1186) ─────
#
# Unit pins on the ONE place the checks-25/30/32 page/deadline contract now
# lives. Every test stubs `verify_task_body._hf_tree_get` at the module-
# attribute seam (the same seam the check-level tests use), so the generator
# must keep resolving it as a late-bound module global.


def test_hf_tree_pages_two_pages_then_exhausted_params_first_page_only(monkeypatch):
    """Event order for a clean two-page listing is exactly
    [page, page, exhausted]; page entries pass through verbatim;
    `params={"recursive": True}` is sent on the FIRST page only (the Link
    rel="next" URL already carries them) and page 2 fetches the Link URL."""
    page2 = "https://huggingface.co/api/datasets/o/r/tree/" + "a" * 40 + "/p?cursor=PAGE2"
    e1 = [{"path": "p/f1.json", "type": "file"}]
    e2 = [{"path": "p/f2.json", "type": "file"}]
    calls: list[tuple[str, dict | None]] = []

    def _fake(url, params, headers, *, timeout_s):
        calls.append((url, params))
        if "PAGE2" in url:
            return verify_task_body._TreeProbeResult("ok", list(e2), None, "")
        return verify_task_body._TreeProbeResult("ok", list(e1), page2, "")

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _fake)
    events = list(verify_task_body._hf_tree_pages("o/r", "dataset", "a" * 40, "p"))
    assert [ev.kind for ev in events] == ["page", "page", "exhausted"]
    assert events[0].entries == e1 and events[1].entries == e2
    assert len(calls) == 2
    assert calls[0][1] == {"recursive": True}
    assert calls[1] == (page2, None)


def test_hf_tree_pages_page_cap_hit(monkeypatch):
    """A listing that never exhausts yields exactly `_HF_PROBE_MAX_PAGES`
    `page` events then ONE `cap` terminal, with exactly `_HF_PROBE_MAX_PAGES`
    GETs issued (the unit-grain twin of the check-level cap pins)."""
    calls: list = []
    _stub_tree(
        monkeypatch,
        status="ok",
        entries=[{"path": "p/f.json", "type": "file"}],
        next_page="https://huggingface.co/api/datasets/o/r/tree/" + "a" * 40 + "/p?cursor=X",
        calls=calls,
    )
    events = list(verify_task_body._hf_tree_pages("o/r", "dataset", "a" * 40, "p"))
    n = verify_task_body._HF_PROBE_MAX_PAGES
    assert [ev.kind for ev in events] == ["page"] * n + ["cap"]
    assert len(calls) == n


def test_hf_tree_pages_exhausted_wins_over_cap_on_final_page(monkeypatch):
    """The E∧C boundary: a listing whose FINAL page lands exactly on the page
    cap is EXHAUSTED, not capped — `next_page is None` is checked BEFORE the
    cap predicate, preserving check 25's `fail` / checks 30/32's `ok` on a
    cap-boundary exhaustive listing (plan #1186 §4.2 row 2)."""
    n = verify_task_body._HF_PROBE_MAX_PAGES
    seen = {"count": 0}

    def _fake(url, params, headers, *, timeout_s):
        seen["count"] += 1
        next_page = None if seen["count"] >= n else f"https://hf.co/api/x?cursor={seen['count']}"
        return verify_task_body._TreeProbeResult(
            "ok", [{"path": "p/f.json", "type": "file"}], next_page, ""
        )

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _fake)
    events = list(verify_task_body._hf_tree_pages("o/r", "dataset", "a" * 40, "p"))
    assert [ev.kind for ev in events] == ["page"] * n + ["exhausted"]
    assert seen["count"] == n


def test_hf_tree_pages_deadline_cap_and_terminal_passthrough(monkeypatch):
    """(a) The deadline arm: with `_HF_PROBE_DEADLINE_S` monkeypatched to
    -1.0 (late-bound read), a would-be-two-page listing yields [page, cap].
    (b) Terminal passthrough: `not_found` → a single `not_found` event;
    `indeterminate` → a single `indeterminate` event carrying the
    `_hf_tree_get` note. (c) Stateful variant: a terminal arriving on page
    ≥2 still yields the preceding `page` event first."""
    sha = "a" * 40
    # (a) deadline arm — never tested anywhere before #1186.
    monkeypatch.setattr(verify_task_body, "_HF_PROBE_DEADLINE_S", -1.0)
    _stub_tree(
        monkeypatch,
        status="ok",
        entries=[{"path": "p/f.json", "type": "file"}],
        next_page="https://hf.co/api/x?cursor=X",
    )
    events = list(verify_task_body._hf_tree_pages("o/r", "dataset", sha, "p"))
    assert [ev.kind for ev in events] == ["page", "cap"]
    monkeypatch.setattr(verify_task_body, "_HF_PROBE_DEADLINE_S", 12.0)
    # (b) terminal passthrough on page 1.
    _stub_tree(monkeypatch, status="not_found")
    events = list(verify_task_body._hf_tree_pages("o/r", "dataset", sha, "p"))
    assert [(ev.kind, ev.note) for ev in events] == [("not_found", "")]
    _stub_tree(monkeypatch, status="indeterminate", note="X")
    events = list(verify_task_body._hf_tree_pages("o/r", "dataset", sha, "p"))
    assert [(ev.kind, ev.note) for ev in events] == [("indeterminate", "X")]

    # (c) stateful: not_found / indeterminate arriving on page 2.
    def _page_then(status, note=""):
        seen = {"count": 0}

        def _fake(url, params, headers, *, timeout_s):
            seen["count"] += 1
            if seen["count"] == 1:
                return verify_task_body._TreeProbeResult(
                    "ok", [{"path": "p/f.json", "type": "file"}], "https://hf.co/api/x?c=2", ""
                )
            return verify_task_body._TreeProbeResult(status, [], None, note)

        return _fake

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _page_then("not_found"))
    events = list(verify_task_body._hf_tree_pages("o/r", "dataset", sha, "p"))
    assert [ev.kind for ev in events] == ["page", "not_found"]
    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _page_then("indeterminate", "HTTP 429"))
    events = list(verify_task_body._hf_tree_pages("o/r", "dataset", sha, "p"))
    assert [ev.kind for ev in events] == ["page", "indeterminate"]
    assert events[-1].note == "HTTP 429"


def test_hf_check25_pagination_cap_skips(monkeypatch):
    """Check-25-LEVEL page-cap pin (existing coverage gap: checks 30/32 have
    cap tests; check 25's l.1987 test hits the 429 path, not the pure cap
    path): a listing that never exhausts and never carries the keyword hits
    the page cap → SKIP (`unverified` on a PASS line), never a FAIL, with
    exactly `_HF_PROBE_MAX_PAGES` GETs."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    calls: list = []
    _stub_tree(
        monkeypatch,
        status="ok",
        entries=[{"path": "issue653_x/armB/other.json", "type": "file"}],
        next_page="https://huggingface.co/api/datasets/o/r/tree/feedface/issue653_x?cursor=X",
        calls=calls,
    )
    body = _audit_body(
        "The per-cell install-probe completions were not separately uploaded, "
        "so they cannot be audited at the record level.",
        "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/feedface/issue653_x",
    )
    r = verify_task_body.check_audit_availability_claims_match_hf(body)
    assert r.passed and not r.is_warn
    # Check 25's aggregate detail names the unverified keyword (it does not
    # inline the per-probe skip note the way checks 30/32 do).
    assert "unverified" in r.detail and "install_probes" in r.detail
    assert len(calls) == verify_task_body._HF_PROBE_MAX_PAGES


def test_hf_count_not_found_skips(monkeypatch):
    """Check-30-level pin of the `not_found → ("skip", -1, -1, "no such
    revision/path")` mapping (no prior check-level coverage — #1186 carried
    reviewer concern): an `unverified` note on a PASS line, never a WARN."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    _stub_tree(monkeypatch, status="not_found")
    body = "Data: [p, 9 files](https://huggingface.co/datasets/o/r/tree/abc1234def/p)\n"
    r = verify_task_body.check_hf_file_count_claims(body)
    assert r.passed and not r.is_warn
    assert "unverified" in r.detail and "no such revision/path" in r.detail


def test_hf_tree_url_fresh_process_ordering_independent():
    """Fresh-process regression pin / durability pin (#1186): `_hf_tree_url`
    must be import-ordering-INDEPENDENT — callable BEFORE anything imports
    `huggingface_hub.utils` (whose import makes the lazy `constants`
    submodule attribute-reachable as a side effect). On huggingface_hub
    0.36.2 a bare `import huggingface_hub` did NOT expose `constants`, so a
    fresh process calling `_hf_tree_url` FIRST crashed with `AttributeError:
    No huggingface_hub attribute constants` (masked in the CLI because check
    23's probe builds headers first). Subprocess = the only faithful fresh
    interpreter (precedent: test_shared_vm_thread_caps.py); no network —
    `_hf_tree_url` is pure string construction. The `sys.modules`
    precondition assert keeps the test non-vacuous: if a future module-level
    import starts pulling `huggingface_hub.utils`, it fails LOUD here
    instead of passing vacuously."""
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('verify_task_body', {str(_SCRIPT)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.modules['verify_task_body'] = m  # REQUIRED: @dataclass dereferences it\n"
        "spec.loader.exec_module(m)\n"
        "assert 'huggingface_hub.utils' not in sys.modules, (\n"
        "    'precondition: huggingface_hub.utils already imported — the fresh-process'\n"
        "    ' bug is masked; re-examine this test'\n"
        ")\n"
        "url = m._hf_tree_url('o/r', 'dataset', 'a' * 40, 'p x')\n"
        "expected = '/api/datasets/o/r/tree/' + 'a' * 40 + '/p%20x'\n"
        "assert url.endswith(expected), url\n"
        "print('URL-OK')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"stdout={out.stdout!r}\nstderr={out.stderr!r}"
    assert "URL-OK" in out.stdout


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
    """CHECKS contains 38 body-only functions: the 20 pre-v3 checks
    (the 18 under the 2-content-section spec, the nested-design (v2)
    sentinel-gated `check_tldr_nested_structure`, and the check-8b
    Reproducibility artifact-URL existence probe), the four
    v3-gated body-only checks added 2026-W24, and the FOUR v4-gated
    body-only checks (`check_v4_methodology_shape`,
    `check_v4_results_beat`, check 27
    `check_v4_no_bare_issue_refs` — bare `#K` refs in the standalone
    sections, #900 — plus check 36
    `check_v4_result_paragraph_sentences`, WARN: >=4-sentence result
    paragraphs, #1368; check 20 v4 `check_v4_word_caps` joined the
    appended-outside set — it needs `issue` for the events-based
    folded-round budget scaling, #921). The four
    v3-gated checks added 2026-W24 are — check 18
    (`check_data_shape`), check 19 (`check_data_subset_disclosure`),
    check 19b (`check_data_unwrapped_example_table`, WARN), check 20
    (`check_v3_word_caps`) — PLUS the ELEVEN generation-agnostic checks:
    check 22 (`check_figure_url_sha_matches_repro`: inline figure URL sha
    vs the `## Reproducibility` per-figure commit claim), check 23
    (`check_hf_url_resolves`: HF Hub revision-pin existence via a bounded
    direct tree-endpoint GET, #733), check 24
    (`check_figure_text_vs_body_tokens`, WARN: figure-embedded `.meta.json`
    text vs body prose — stale fraction / softened-token staleness, #667
    r2), check 25 (`check_audit_availability_claims_match_hf`: a body
    "not uploaded / cannot be audited" claim vs the artifact's actual HF
    existence, #653 r6), check 26
    (`check_figure_panel_prose_vs_sidecar`, FAIL: figure what-is-plotted
    panel/series prose vs the sidecar's `_kind` aggregate — panel/series
    drift, #683 r1), check 28 (`check_figure_label_codes`, WARN:
    opaque config-code tokens — `@L<digits>` layer pins / regime-code
    slugs — in the figure sidecar's rendered-text strings, #920), and
    check 29 (`check_figure_tracked_at_head`, WARN: body-linked same-repo
    `figures/issue_<N>/` figure paths still tracked on a live local ref —
    HEAD plus the `issue-<N>` / `issue-<N>-*` branch family; branch-only →
    PASS-disclosure, missing everywhere → WARN, #964 / incident #841), and
    check 30 (`check_hf_file_count_claims`, WARN: numeric "N files" /
    "N shards" claims adjacent to hex-pinned HF `/tree/<sha>` markdown
    links vs a files-only scoped Hub tree count — folder entries excluded;
    mismatch → WARN never FAIL, every non-definitive probe outcome SKIPs;
    incident #931's 528-vs-515 folder-inflation miscount, #1008), and
    check 32 (`check_hf_adjacent_file_claims`, WARN: backtick FILENAME
    claims adjacent to hex-pinned HF `/tree/<sha>` markdown links — the
    filename-membership sibling of check 30 — must appear by exact
    basename, any depth, in the scoped listing at the pinned revision;
    missing → WARN never FAIL, every non-definitive probe outcome SKIPs;
    incident #952 r1's git-only `divergence_bank_queries.json` claimed at
    the pinned HF tree, #1016), and check 33
    (`check_figure_prose_numerics_vs_sidecar`, WARN: bolded what-is-plotted
    DECIMALS in a figure's previous-figure-bounded beat-1 window vs the
    sidecar's plotted values, under rounding / sign / percent /
    sci-notation tolerance; per-figure `<!-- prose-numerics: derived -->`
    opt-out; silent skip on missing / truncated sidecars; incident #825 r1,
    #1107), and check 34 (`check_figure_beat_claims_vs_sidecar_text`, WARN,
    FORWARD-ONLY: beat-1 series-structure claims — "both … arms/…" and "one
    bar/… per <unit>" — vs the series structure the sidecar demonstrably
    renders; fires only when the sidecar carries the `meta["text"]`
    rendered-text block, so every pre-capture sidecar silently skips;
    contradiction-only, absence of evidence never fires; incident #1092
    defect (b), #1255). The
    migration is a RETARGET — every former check
    was kept (sometimes dormant, e.g. `check_figure_caption`) so downstream
    tests stay valid; the v3 checks PASS-skip on non-v3 bodies.

    Checks appended OUTSIDE CHECKS inside `verify_text` (they need
    something beyond the body string): the Goal soft check (needs
    frontmatter), the Lens 14 concerns-audit (needs concerns.jsonl),
    the check-16 lr-matches-plan (needs the plan), the check-17 Context
    provenance row (needs frontmatter + original-body.md), the v3
    check-21 body-Parameters-⊆-doc (needs the methodology doc path),
    the v4 check-20 word caps (needs `issue` for the events-based
    folded-round budget scaling, #921), the #732 judge-API-error
    denominator check (needs eval JSONs), and the check-31
    orphaned-per-unit-figures probe (needs `issue` for figures-dir
    scoping, #1011).
    So `verify_text` returns 50 results (2 prepended + CHECKS[1:]=38 +
    10 appended — see `test_good_body_passes_all`), but `CHECKS` stays
    at 39.
    """
    assert len(verify_task_body.CHECKS) == 39
    # By-name membership so the NEXT check addition can key by name instead
    # of re-deriving the arithmetic (#1016 methodology-reconciler Must-Fix).
    assert verify_task_body.check_hf_adjacent_file_claims in verify_task_body.CHECKS
    assert verify_task_body.check_figure_prose_numerics_vs_sidecar in verify_task_body.CHECKS
    assert verify_task_body.check_figure_beat_claims_vs_sidecar_text in verify_task_body.CHECKS
    assert verify_task_body.check_v4_result_paragraph_sentences in verify_task_body.CHECKS


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
title: Some claim about a finding (MODERATE confidence)
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
    # No deferral markers in the body → no spurious stale-marker WARN
    # suffix on the FAIL detail (#1089).
    assert "; WARN:" not in result.detail


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
    # Deferring a LIVE open concern (latest event `raised`) never warns (#1089).
    assert not result.is_warn


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


def test_concerns_audit_sees_row_with_raw_unicode_line_separator(tmp_path):
    """A raised BLOCKER whose evidence carries a raw U+2028 (the
    ``ensure_ascii=False`` writer leaves Unicode line separators
    unescaped) is still parsed by the check-14 reader. Pre-#950 the
    ``splitlines()`` reader shredded the row into fragments the per-line
    skip silently dropped — 0 events read, and the binding-concerns
    audit falsely PASSed on a body that never acknowledged the BLOCKER
    (#825 → #950 round 2)."""
    cp = tmp_path / "concerns.jsonl"
    cp.write_text(
        json.dumps(
            {
                "event": "raised",
                "concern_id": "u2028-blocker-must-be-seen",
                "severity": "BLOCKER",
                "summary": "row must survive the reader",
                # \u2028 = LINE SEPARATOR, raw in the written file under
                # ensure_ascii=False -- the exact #825 shred trigger.
                "evidence": "first paragraph\u2028second paragraph",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    result = verify_task_body.check_concerns_audit(GOOD_BODY, concerns_path=cp)
    assert not result.passed
    assert "u2028-blocker-must-be-seen" in result.detail
    assert "(BLOCKER)" in result.detail


def test_concerns_audit_warns_on_stale_deferred_marker_when_addressed(tmp_path):
    """A `<!-- concern-deferred: <id> -->` marker whose id's latest ledger
    event is `addressed` is STALE: it misdescribes the resolution (the
    concern was fixed, not deferred). The check WARNs (never FAILs),
    naming the stale id — the #833 shape, where `open_binding` is empty
    and the pre-#1089 code never scanned the markers at all. Also pins
    dedup (a duplicate marker for the same id warns once) and
    deterministic sorted-by-id ordering across multiple stale ids."""
    cp = tmp_path / "concerns.jsonl"
    rows = [
        {"event": "raised", "concern_id": "now-fixed-thing", "severity": "CONCERN"},
        {"event": "addressed", "concern_id": "now-fixed-thing", "severity": "CONCERN"},
        {"event": "raised", "concern_id": "another-fixed-thing", "severity": "CONCERN"},
        {"event": "addressed", "concern_id": "another-fixed-thing", "severity": "CONCERN"},
    ]
    cp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    body = (
        GOOD_BODY
        + "\n<!-- concern-deferred: now-fixed-thing -->\n"
        + "<!-- concern-deferred: another-fixed-thing -->\n"
        # Duplicate marker for the first id — dedupes to one WARN.
        + "<!-- concern-deferred: now-fixed-thing -->\n"
    )
    result = verify_task_body.check_concerns_audit(body, concerns_path=cp)
    assert result.passed
    assert result.is_warn
    assert "now-fixed-thing" in result.detail
    assert "addressed" in result.detail
    assert "remove or retag" in result.detail
    # Verbatim WARN strings (#1089 plan §3.2), joined "; " in sorted-id order,
    # each id exactly once despite the duplicate marker.
    assert result.detail == (
        "stale concern-deferred marker 'another-fixed-thing' — concern is addressed; "
        "remove or retag; "
        "stale concern-deferred marker 'now-fixed-thing' — concern is addressed; "
        "remove or retag"
    )


def test_concerns_audit_warns_on_deferred_marker_absent_from_ledger(tmp_path):
    """A `<!-- concern-deferred: <id> -->` marker naming an id the ledger
    has never heard of WARNs with distinct wording: `defer_concern`
    refuses never-raised ids, so an absent-id marker is a typo or a
    cross-task body copy — either way it does not correspond to this
    task's ledger."""
    cp = tmp_path / "concerns.jsonl"
    cp.write_text("")  # empty ledger — the id is absent by construction
    body = GOOD_BODY + "\n<!-- concern-deferred: ghost-concern -->\n"
    result = verify_task_body.check_concerns_audit(body, concerns_path=cp)
    assert result.passed
    assert result.is_warn
    assert "ghost-concern" in result.detail
    assert "absent" in result.detail


def test_concerns_audit_live_deferred_ledger_event_no_warn(tmp_path):
    """Case (c), `deferred` branch: a marker whose id's latest ledger
    event is `deferred` is LIVE — the canonical defer path produced it —
    so no WARN fires (behavior unchanged from pre-#1089). Also covers the
    `verified-open` branch (the last live-vocabulary cell)."""
    cp = tmp_path / "concerns.jsonl"
    rows = [
        {"event": "raised", "concern_id": "parked-thing", "severity": "CONCERN"},
        {"event": "deferred", "concern_id": "parked-thing", "severity": "CONCERN"},
    ]
    cp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    body = GOOD_BODY + "\n<!-- concern-deferred: parked-thing -->\n"
    result = verify_task_body.check_concerns_audit(body, concerns_path=cp)
    assert result.passed
    assert not result.is_warn

    # verified-open variant: still open (and binding), the marker
    # acknowledges it via mechanism 3 — no WARN either.
    rows = [
        {"event": "raised", "concern_id": "still-open-thing", "severity": "CONCERN"},
        {"event": "verified-open", "concern_id": "still-open-thing", "severity": "CONCERN"},
    ]
    cp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    body = GOOD_BODY + "\n<!-- concern-deferred: still-open-thing -->\n"
    result = verify_task_body.check_concerns_audit(body, concerns_path=cp)
    assert result.passed
    assert not result.is_warn


def test_concerns_audit_stale_marker_folds_warn_into_open_concern_fail(tmp_path):
    """Case (d), FAIL precedence: an unaddressed open concern still FAILs
    (`passed=False`, never downgraded); the stale-marker warn text rides
    the FAIL detail behind the literal `; WARN: ` prefix (the established
    mixed-FAIL+WARN fold), AFTER the unaddressed text."""
    cp = tmp_path / "concerns.jsonl"
    rows = [
        {"event": "raised", "concern_id": "open-unacked-thing", "severity": "CONCERN"},
        {"event": "raised", "concern_id": "now-fixed-thing", "severity": "CONCERN"},
        {"event": "addressed", "concern_id": "now-fixed-thing", "severity": "CONCERN"},
    ]
    cp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    body = GOOD_BODY + "\n<!-- concern-deferred: now-fixed-thing -->\n"
    result = verify_task_body.check_concerns_audit(body, concerns_path=cp)
    assert not result.passed
    assert not result.is_warn
    assert "open-unacked-thing" in result.detail
    assert "; WARN: stale concern-deferred marker 'now-fixed-thing'" in result.detail
    # Fold order: the unaddressed FAIL text comes BEFORE the WARN suffix.
    assert result.detail.index("open-unacked-thing") < result.detail.index("; WARN: ")


def test_concerns_audit_skip_path_ignores_markers(tmp_path):
    """Case (e): with no ledger to compare against (`concerns_path` None
    or missing), the skip-PASS path never scans markers — a stale-looking
    marker in the body produces no WARN."""
    body = GOOD_BODY + "\n<!-- concern-deferred: ghost-concern -->\n"
    result = verify_task_body.check_concerns_audit(body, concerns_path=None)
    assert result.passed
    assert not result.is_warn
    assert "skipped" in result.detail.lower()

    missing = tmp_path / "concerns.jsonl"
    result = verify_task_body.check_concerns_audit(body, concerns_path=missing)
    assert result.passed
    assert not result.is_warn
    assert "skipped" in result.detail.lower()


def test_concerns_audit_stale_marker_warns_alongside_acknowledged_open_concern(tmp_path):
    """Pins the SECOND warns-only return site: `open_binding` is
    non-empty (a raised CONCERN, acknowledged in the body via
    mechanism 1), so the check passes the early return and runs the full
    ack scan; `unaddressed` ends empty; the post-ack `if stale_warns:`
    branch fires for the stale marker. A mutant deleting the post-ack
    branch fails exactly this test."""
    cp = tmp_path / "concerns.jsonl"
    rows = [
        {"event": "raised", "concern_id": "open-acked-thing", "severity": "CONCERN"},
        {"event": "raised", "concern_id": "now-fixed-thing", "severity": "CONCERN"},
        {"event": "addressed", "concern_id": "now-fixed-thing", "severity": "CONCERN"},
    ]
    cp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    body = GOOD_BODY.replace(
        "The 17-pt lift holds at every seed",
        "Note: open-acked-thing affected our setup; "
        "we report the conservative estimate. The 17-pt lift holds at every seed",
    )
    body = body + "\n<!-- concern-deferred: now-fixed-thing -->\n"
    result = verify_task_body.check_concerns_audit(body, concerns_path=cp)
    assert result.passed
    assert result.is_warn
    assert "now-fixed-thing" in result.detail
    assert "addressed" in result.detail


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
title: Some claim about a finding (MODERATE confidence)
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
    # Budget formula pin: 2 extra rounds -> 800 + 2 x 250 = 1300.
    assert (
        verify_task_body.V3_TOTAL_PROSE_BASE_WORDS
        + 2 * verify_task_body.V3_TOTAL_PROSE_PER_EXTRA_ROUND_WORDS
        == 1300
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
title: Some claim about a finding (MODERATE confidence)
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
    # Check 36: the fixture's result paragraphs are 1 sentence each (#1368).
    assert by_name["Results paragraph sentence cap (v4)"].passed
    assert not by_name["Results paragraph sentence cap (v4)"].is_warn
    # Sentinel-keyed checks run on v4 (confidence title-only, Context row).
    assert by_name["Confidence sentence matches title"].passed
    assert by_name["Reproducibility Context provenance row"].passed
    # The v3-only checks PASS-skip on a v4 body.
    assert by_name["Data section shape (v3)"].passed
    assert by_name["v3 conciseness caps"].passed
    # The v2-only nested-structure check PASS-skips on a v4 body.
    assert by_name["TL;DR nested-design structure (v2)"].passed
    # Check 27: the fixture's standalone sections carry no bare `#K` refs
    # (its `[#34](...)` Goal link + footer lineage are sanctioned forms).
    assert by_name["no bare issue refs in standalone sections (v4)"].passed
    # The only FAILs are the two existence probes on the fake sha.
    fails = [r.name for r in results if not r.passed]
    assert set(fails) <= {"Figure URL resolvable", "Reproducibility artifact URLs exist"}, fails


def test_v4_context_blockquote_bare_url_passes_permanence():
    """The #825 incident shape: a v4 `**Context:**` verbatim
    originating-prompt blockquote citing a bare HF URL must PASS check 8
    with no hyperlink-to-pinned-revision workaround (#959). Asserts the
    permanence check only (per the `_V4_GOOD_BODY` convention — the
    fixture's fake SHAs fail the existence probes, so overall PASS is
    not assertable)."""
    body = _V4_GOOD_BODY.replace(
        "- Originating prompt: origin prompt not recorded",
        "- Originating prompt, verbatim:\n\n"
        "> test in the base model (https://huggingface.co/Qwen/Qwen2.5-7B)\n"
        "> -- make sure this is the proper base model\n",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    perm = by_name["Reproducibility URL permanence"]
    assert perm.passed, perm.detail


# ─── Check 19b on v4 bodies (`## Methodology` scan, #1227) ─────────────────
# All assertions are per-check by name (the `_V4_GOOD_BODY` convention —
# fake SHAs fail the existence probes, so overall PASS is not assertable).


def test_v4_unwrapped_methodology_table_with_condition_code_warns():
    """Check 19b (v4): a bare inline GFM table in `## Methodology`
    carrying cell_tags-family codes (`BS_E0`, `Method A`) WARNs — those
    codes are NOT table-blanked by the audit, so the bare table FAILs
    the Step 9a-bis condition-code scan with no authoring nudge (#1227)."""
    bare_table = (
        "Per-cell breakdown (2 of N shown for illustration):\n\n"
        "| Cell | Method | Rows |\n"
        "|---|---|---|\n"
        "| BS_E0 | Method A | 500 |\n"
        "| BS_E1 | Method B | 500 |\n\n"
    )
    body = _V4_GOOD_BODY.replace("- **Evaluation:**", bare_table + "- **Evaluation:**")
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    warn = by_name["Methodology unwrapped example table (v4)"]
    assert warn.passed and warn.is_warn, warn.render()
    assert "BS_E0" in warn.detail or "Method A" in warn.detail


def test_v4_wrapped_methodology_table_does_not_warn():
    """Check 19b (v4) stays silent on the spec-conformant wrapped form:
    the same condition-code table inside a `<details>` block does not
    WARN (the audit exempts wrapped v4 `## Methodology` example blocks
    via strip_data_example_blocks, #1171)."""
    wrapped = (
        "<details>\n"
        "<summary>per-cell rows (2 of N, random sample)</summary>\n\n"
        "| Cell | Method | Rows |\n"
        "|---|---|---|\n"
        "| BS_E0 | Method A | 500 |\n\n"
        "</details>\n\n"
    )
    body = _V4_GOOD_BODY.replace("- **Evaluation:**", wrapped + "- **Evaluation:**")
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    warn = by_name["Methodology unwrapped example table (v4)"]
    assert warn.passed and not warn.is_warn, warn.render()


def test_v4_benign_bare_hparam_table_does_not_warn():
    """Check 19b (v4): the spec-REQUIRED bare Training hyperparameter
    table (Parameter/Value/Source) in the unmodified _V4_GOOD_BODY does
    NOT WARN — legitimate hyperparameter values are not condition codes,
    and the check must not nag every conformant v4 body."""
    _ok, results = verify_task_body.verify_text(_V4_GOOD_BODY)
    by_name = _results_by_name(results)
    warn = by_name["Methodology unwrapped example table (v4)"]
    assert warn.passed and not warn.is_warn, warn.render()


def test_v4_unwrapped_single_column_methodology_table_warns():
    """Check 19b (v4): single-column parity with v3 — a one-column
    delimiter has no internal `|`, so the audit's table-blanking never
    recognizes it and `condition_labels` fires on `| C1 |`; the WARN
    keeps the audit sync at one column."""
    bare_single = "Conditions (2 of 2):\n\n| Condition |\n|---|\n| C1 |\n| C2 |\n\n"
    body = _V4_GOOD_BODY.replace("- **Evaluation:**", bare_single + "- **Evaluation:**")
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    warn = by_name["Methodology unwrapped example table (v4)"]
    assert warn.passed and warn.is_warn, warn.render()
    assert "C1" in warn.detail


def test_v4_check19b_emits_v4_label_only():
    """On a v4 body check 19b emits the v4 label ONLY — no stray
    `Data unwrapped example table (v3)` row (one CheckResult per body)."""
    _ok, results = verify_task_body.verify_text(_V4_GOOD_BODY)
    by_name = _results_by_name(results)
    assert "Methodology unwrapped example table (v4)" in by_name
    assert "Data unwrapped example table (v3)" not in by_name


# ─── Check 17 (v4 lineage-token sub-check, #1014) ──────────────────────────
# All assertions are per-check by name (the `_V4_GOOD_BODY` convention —
# fake SHAs fail the existence probes, so overall PASS is not assertable).

_V4_LINEAGE_BULLET = (
    "- Follow-up to [#34](https://eps.superkaiba.com/tasks/34) — the "
    "X-effect generalisation question.\n"
)


def _v4_body_with_lineage(replacement: str) -> str:
    """Return `_V4_GOOD_BODY` with its Context lineage bullet replaced."""
    assert _V4_LINEAGE_BULLET in _V4_GOOD_BODY, "fixture drifted"
    return _V4_GOOD_BODY.replace(_V4_LINEAGE_BULLET, replacement)


def test_v4_context_issue_link_lineage_passes():
    """The canonical v4 fixture's `[#34](...)` lineage bullet satisfies
    the v4 lineage-token sub-check (issue-reference alternative)."""
    _ok, results = verify_task_body.verify_text(_V4_GOOD_BODY)
    ctx = _results_by_name(results)[_CONTEXT_CHECK]
    assert ctx.passed and not ctx.is_warn
    assert "lineage" in ctx.detail


def test_v4_context_fresh_direction_no_parent_passes():
    """`fresh direction (no parent)` satisfies the lineage sub-check."""
    body = _v4_body_with_lineage("- Lineage: fresh direction (no parent).\n")
    _ok, results = verify_task_body.verify_text(body)
    ctx = _results_by_name(results)[_CONTEXT_CHECK]
    assert ctx.passed and not ctx.is_warn, ctx.detail


def test_v4_context_fresh_no_parent_short_form_passes():
    """The task-Goal short form `fresh (no parent)` satisfies the
    lineage sub-check (the `no parent` alternative)."""
    body = _v4_body_with_lineage("- Lineage: fresh (no parent).\n")
    _ok, results = verify_task_body.verify_text(body)
    ctx = _results_by_name(results)[_CONTEXT_CHECK]
    assert ctx.passed and not ctx.is_warn, ctx.detail


def test_v4_context_bare_issue_ref_passes():
    """Bare `#K` refs (the #823 shape: `Child of #722 ... method parent
    #779`) satisfy the lineage sub-check."""
    body = _v4_body_with_lineage("- Child of #722 (context pool), method parent #779.\n")
    _ok, results = verify_task_body.verify_text(body)
    ctx = _results_by_name(results)[_CONTEXT_CHECK]
    assert ctx.passed and not ctx.is_warn, ctx.detail


def test_v4_context_followup_round_clause_passes():
    """A `same-issue follow-up round` clause with NO issue ref satisfies
    the lineage sub-check (the follow-up-round alternative)."""
    body = _v4_body_with_lineage("- Same-issue follow-up round `maxp-winner` run 2026-07-03.\n")
    _ok, results = verify_task_body.verify_text(body)
    ctx = _results_by_name(results)[_CONTEXT_CHECK]
    assert ctx.passed and not ctx.is_warn, ctx.detail


def test_v4_context_without_lineage_token_fails():
    """A v4 `**Context:**` row with NO lineage token is a hard FAIL
    (the #958 gap this sub-check closes)."""
    body = _v4_body_with_lineage("")
    _ok, results = verify_task_body.verify_text(body)
    ctx = _results_by_name(results)[_CONTEXT_CHECK]
    assert not ctx.passed
    assert "lineage" in ctx.detail
    assert "SPEC" in ctx.detail


def test_v4_blockquoted_issue_ref_does_not_satisfy_lineage():
    """An issue ref inside the blockquoted verbatim originating prompt
    must NOT satisfy the lineage sub-check — the quote is provenance
    TEXT, not lineage."""
    body = _v4_body_with_lineage("").replace(
        "- Originating prompt: origin prompt not recorded",
        "- Originating prompt, verbatim:\n\n> rerun #537 with the new adapters\n",
    )
    assert "> rerun #537" in body, "fixture replacement did not land"
    _ok, results = verify_task_body.verify_text(body)
    ctx = _results_by_name(results)[_CONTEXT_CHECK]
    assert not ctx.passed, ctx.detail


def test_v4_inline_quote_on_label_line_keeps_same_line_lineage():
    """The #763 shape: the whole Context row is ONE physical line with an
    inline `> "..."` quote after the label and the lineage clause after
    the quote. Pins the strip-before-slice scan order — slice-then-strip
    would drop the whole line (it appears to start with `>` post-slice)
    and wrongly FAIL."""
    context_block = (
        "**Context:**\n"
        "- Created 2026-06-24; run executed 2026-06-24.\n"
        + _V4_LINEAGE_BULLET
        + "- Originating prompt: origin prompt not recorded\n"
    )
    assert context_block in _V4_GOOD_BODY, "fixture drifted"
    body = _V4_GOOD_BODY.replace(
        context_block,
        '**Context:** > "do statistical now" · lineage: '
        "[#658](https://eps.superkaiba.com/tasks/658) — parent.\n",
    )
    _ok, results = verify_task_body.verify_text(body)
    ctx = _results_by_name(results)[_CONTEXT_CHECK]
    assert ctx.passed and not ctx.is_warn, ctx.detail


def test_v3_context_without_lineage_not_newly_failed():
    """v3 behavior stays byte-identical: a v3 `**Context:**` row with NO
    lineage token keeps the pre-#1014 label-presence PASS (no new WARN,
    no FAIL)."""
    assert _V4_LINEAGE_BULLET in _V3_GOOD_BODY, "fixture drifted"
    body = _V3_GOOD_BODY.replace(_V4_LINEAGE_BULLET, "")
    _ok, results = verify_task_body.verify_text(body)
    ctx = _results_by_name(results)[_CONTEXT_CHECK]
    assert ctx.passed and not ctx.is_warn, ctx.detail
    assert "present" in ctx.detail


def test_v4_context_fresh_direction_alone_passes():
    """`fresh direction` WITHOUT `no parent` / any `#K` ref / any
    follow-up-round clause (the #778/#658 phrasing family) satisfies the
    sub-check — pins regex alternative 2 uniquely (deleting the
    `fresh direction` arm would fail this test and only this test)."""
    body = _v4_body_with_lineage(
        "- Lineage: fresh direction seeded from the marker-transfer question bank.\n"
    )
    _ok, results = verify_task_body.verify_text(body)
    ctx = _results_by_name(results)[_CONTEXT_CHECK]
    assert ctx.passed and not ctx.is_warn, ctx.detail


def test_v4_context_url_fragment_does_not_satisfy_lineage():
    """A URL fragment (`.../page#123`) must NOT satisfy the issue-ref
    alternative — pins the `(?<![\\w/&])` lookbehind."""
    body = _v4_body_with_lineage("- See https://example.com/page#123 for background.\n")
    _ok, results = verify_task_body.verify_text(body)
    ctx = _results_by_name(results)[_CONTEXT_CHECK]
    assert not ctx.passed, ctx.detail


def test_v4_context_unhyphenated_followup_round_clause_passes():
    """`Same-issue followup round` (no hyphen in `followup`, no `#` ref)
    satisfies the sub-check — pins the `follow-?up` optional hyphen."""
    body = _v4_body_with_lineage("- Same-issue followup round `alt2-pin` run 2026-07-04.\n")
    _ok, results = verify_task_body.verify_text(body)
    ctx = _results_by_name(results)[_CONTEXT_CHECK]
    assert ctx.passed and not ctx.is_warn, ctx.detail


# ─── Check 17 (origin-prompt verbatim sub-check, #1068) ────────────────────
# Unit-grain cases call `check_repro_context_provenance(body, fm)` directly
# (the function is public and takes `fm`); one end-to-end case runs
# `verify_text` to pin the frontmatter threading.

_OP = "sweep the X effect across three seeds and report the per-seed deltas"
# 68 normalized chars; `_OP[:41]` is a mid-sentence cut ending at "...and"
# (41/68 = 60% coverage — over the 20-char absolute AND 50% coverage floors).
_OP41 = _OP[:41]

_V4_NOT_RECORDED_LINE = "- Originating prompt: origin prompt not recorded"


def _v4_body_with_context_quote(quote_block: str) -> str:
    """Return `_V4_GOOD_BODY` with its Context originating-prompt line
    (`origin prompt not recorded`) replaced by ``quote_block``."""
    assert _V4_NOT_RECORDED_LINE in _V4_GOOD_BODY, "fixture drifted"
    return _V4_GOOD_BODY.replace(_V4_NOT_RECORDED_LINE, quote_block)


def test_v4_context_origin_prompt_verbatim_blockquote_passes():
    """The canonical conforming shape: a blockquoted verbatim quote of the
    full frontmatter `origin_prompt` PASSes with no WARN."""
    body = _v4_body_with_context_quote(f"- Originating prompt, verbatim:\n\n> {_OP}\n")
    ctx = verify_task_body.check_repro_context_provenance(body, {"origin_prompt": _OP})
    assert ctx.passed and not ctx.is_warn, ctx.detail
    assert "lineage" in ctx.detail


def test_v4_context_origin_prompt_prefix_truncated_fails():
    """The #813 r1 shape: the quote is a strict mid-sentence PREFIX of
    `origin_prompt` — a hard v4 FAIL naming the truncation offset."""
    body = _v4_body_with_context_quote(f"- Originating prompt, verbatim:\n\n> {_OP41}\n")
    ctx = verify_task_body.check_repro_context_provenance(body, {"origin_prompt": _OP})
    assert not ctx.passed
    assert "PREFIX" in ctx.detail
    assert "context-origin-prompt-mismatch" in ctx.detail
    assert "41/68" in ctx.detail


def test_v4_context_truncated_with_trailing_period_fails():
    """The #742 shape: a truncating editor appends a `.` at the cut — the
    trailing-punct strip still classifies it as a strict-prefix FAIL."""
    body = _v4_body_with_context_quote(f"- Originating prompt, verbatim:\n\n> {_OP41.rstrip()}.\n")
    ctx = verify_task_body.check_repro_context_provenance(body, {"origin_prompt": _OP})
    assert not ctx.passed
    assert "PREFIX" in ctx.detail


def test_v4_context_whitespace_and_wrap_differences_pass():
    """A quote re-wrapped across `>` lines with doubled internal spaces
    still PASSes — pins `_normalize_prompt_text` whitespace collapsing."""
    body = _v4_body_with_context_quote(
        "- Originating prompt, verbatim:\n\n"
        "> sweep the X effect  across\n"
        "> three seeds and report  the\n"
        "> per-seed deltas\n"
    )
    ctx = verify_task_body.check_repro_context_provenance(body, {"origin_prompt": _OP})
    assert ctx.passed and not ctx.is_warn, ctx.detail


def test_v4_context_inline_quote_marks_pass():
    """The #661/#672 shape: the full prompt quoted inline in `"..."` on
    the label bullet, no blockquote — PASSes (pins region-haystack
    semantics; substring containment ignores wrapping quote marks)."""
    body = _v4_body_with_context_quote(f'- Originating user request (verbatim): "{_OP}"')
    ctx = verify_task_body.check_repro_context_provenance(body, {"origin_prompt": _OP})
    assert ctx.passed and not ctx.is_warn, ctx.detail


def test_v4_context_multi_round_extra_prompts_pass():
    """The #813 post-fix shape: the full creation quote plus a second
    labeled round-prompt blockquote — extra quotes only grow the haystack."""
    body = _v4_body_with_context_quote(
        f"- Originating prompt, verbatim:\n\n> {_OP}\n\n"
        "- Round-2 prompt (`dose-curve`), verbatim:\n\n"
        "> also check the dose curve at half strength\n"
    )
    ctx = verify_task_body.check_repro_context_provenance(body, {"origin_prompt": _OP})
    assert ctx.passed and not ctx.is_warn, ctx.detail


def test_v4_context_generic_mismatch_warns_with_offset():
    """A paraphrase sharing no 20-char prefix with `origin_prompt` is a
    WARN (not a FAIL) naming the first-divergence offset."""
    body = _v4_body_with_context_quote(
        "- Originating prompt, verbatim:\n\n"
        "> run the seed sweep for the X effect and summarize the deltas\n"
    )
    ctx = verify_task_body.check_repro_context_provenance(body, {"origin_prompt": _OP})
    assert ctx.passed and ctx.is_warn, ctx.detail
    assert "first divergence" in ctx.detail
    assert "/68" in ctx.detail


def test_v4_context_markdown_escaped_quote_passes():
    """An `origin_prompt` containing `**stories**` quoted with markdown
    backslash-escapes (`\\*\\*stories\\*\\*`) PASSes — pins the
    `_unescape_markdown` leg of the containment test."""
    op_md = "please analyze the **stories** dataset and report drift across all five domains"
    body = _v4_body_with_context_quote(
        "- Originating prompt, verbatim:\n\n"
        "> please analyze the \\*\\*stories\\*\\* dataset and report drift "
        "across all five domains\n"
    )
    ctx = verify_task_body.check_repro_context_provenance(body, {"origin_prompt": op_md})
    assert ctx.passed and not ctx.is_warn, ctx.detail


def test_v4_context_lazy_continuation_quote_passes():
    """A blockquote whose second physical line lacks the `>` prefix
    (markdown lazy continuation) still PASSes — pins the marker-strip
    haystack keeping ALL region text, lazy lines included."""
    body = _v4_body_with_context_quote(
        f"- Originating prompt, verbatim:\n\n> {_OP[:38]}\n{_OP[38:]}\n"
    )
    ctx = verify_task_body.check_repro_context_provenance(body, {"origin_prompt": _OP})
    assert ctx.passed and not ctx.is_warn, ctx.detail


def test_v4_context_no_origin_prompt_noop_unchanged():
    """No frontmatter `origin_prompt` → the sub-check NO-OPs: a
    deliberately-mismatching quote keeps the byte-identical pre-#1068
    PASS detail."""
    body = _v4_body_with_context_quote(
        "- Originating prompt, verbatim:\n\n> something entirely unrelated to any recorded prompt\n"
    )
    ctx = verify_task_body.check_repro_context_provenance(body, {})
    assert ctx.passed and not ctx.is_warn, ctx.detail
    assert ctx.detail == "**Context:** row present with lineage token"


def test_v3_context_truncated_quote_warns_never_fails():
    """Grandfathering: the SAME truncation that hard-FAILs a v4 body is
    WARN-only on a v3 body (never a new hard FAIL below the v4 sentinel)."""
    assert _V4_NOT_RECORDED_LINE in _V3_GOOD_BODY, "fixture drifted"
    body = _V3_GOOD_BODY.replace(
        _V4_NOT_RECORDED_LINE,
        f"- Originating prompt, verbatim:\n\n> {_OP41}\n",
    )
    ctx = verify_task_body.check_repro_context_provenance(body, {"origin_prompt": _OP})
    assert ctx.passed is True and ctx.is_warn, ctx.detail
    assert "context-origin-prompt-mismatch" in ctx.detail


def test_verify_text_threads_frontmatter_origin_prompt():
    """End-to-end: `origin_prompt` spliced into the fixture's EXISTING
    frontmatter block (never a second `---` block — check 0b trips on
    stacked frontmatter) + a truncated quote → the check FAILs through
    `verify_text`, pinning the fm threading."""
    body = _v4_body_with_context_quote(f"- Originating prompt, verbatim:\n\n> {_OP41}\n").replace(
        "kind: experiment\n",
        f'kind: experiment\norigin_prompt: "{_OP}"\n',
    )
    assert "origin_prompt" in body, "fixture replacement did not land"
    _ok, results = verify_task_body.verify_text(body)
    ctx = _results_by_name(results)[_CONTEXT_CHECK]
    assert not ctx.passed
    assert "context-origin-prompt-mismatch" in ctx.detail


def test_v4_context_inline_quoted_truncation_fails():
    """A truncated prompt quoted INLINE in `"..."` (no blockquote anywhere
    in the Context region) still FAILs — pins the `_INLINE_QUOTE_SPAN_RE`
    candidate arm as FAIL-capable (deleting the inline arm would ship
    green through the rest of the suite AND the backlog sweep)."""
    body = _v4_body_with_context_quote(
        f'- Originating user request (verbatim): "{_OP41}" — lineage note above.'
    )
    ctx = verify_task_body.check_repro_context_provenance(body, {"origin_prompt": _OP})
    assert not ctx.passed
    assert "PREFIX" in ctx.detail
    assert "context-origin-prompt-mismatch" in ctx.detail


def test_v4_context_alternate_source_with_inline_opener_warns_not_fails():
    """The #825+ conforming shape: fm `origin_prompt` is a long
    self-declared abridgement; the row quotes the FULL alternate-source
    prompt (long, non-prefix blockquote) plus a SHORT innocent inline
    quote of the fm opener (strict prefix, over the 20-char floor but
    ~9% coverage — under the 50% floor). Verdict is warn-mismatch,
    NEVER fail-trunc (pins the D10 coverage floor)."""
    op_long = (
        "investigate whether the marker adapters trained in the localization arm "
        "transfer their end-of-turn emission to the paraphrase eval surface, "
        "including the bystander panel, the dose-matched checkpoints, and the "
        "frozen-R diagonal read described in the provenance section (abridged; "
        "verbatim full prompt in the original body Provenance section)"
    )
    body = _v4_body_with_context_quote(
        '- Originating prompt (abridged in frontmatter): "investigate whether the marker"\n'
        "- Full alternate-source prompt from the original body, verbatim:\n\n"
        "> The complete originating request text as recorded before promotion: run the\n"
        "> marker-transfer eval end to end and report every per-persona delta with the\n"
        "> dose-matched checkpoints held fixed across the bystander panel.\n"
    )
    ctx = verify_task_body.check_repro_context_provenance(body, {"origin_prompt": op_long})
    assert ctx.passed and ctx.is_warn, ctx.detail
    assert "PREFIX" not in ctx.detail
    assert "first divergence" in ctx.detail


def test_v4_context_multi_round_truncated_creation_quote_still_fails():
    """A truncated creation quote (60% strict prefix) + a LONGER full
    round-2 blockquote elsewhere in the row still FAILs — pins that the
    D10 guard is per-candidate + fraction-based and is NOT suppressed by
    the presence of a longer non-prefix candidate (the false-negative
    direction of the rejected suppress-variant)."""
    body = _v4_body_with_context_quote(
        f"- Originating prompt, verbatim:\n\n> {_OP41}\n\n"
        "- Round-2 prompt (`dose-curve`), verbatim:\n\n"
        "> additionally rerun the dose curve at half strength across all five bystander\n"
        "> personas and report the per-persona deltas alongside the headline numbers\n"
    )
    ctx = verify_task_body.check_repro_context_provenance(body, {"origin_prompt": _OP})
    assert not ctx.passed, ctx.detail
    assert "PREFIX" in ctx.detail


def test_v4_context_short_origin_prompt_truncation_warns():
    """An `origin_prompt` under the 20-normalized-char floor cannot
    hard-FAIL: its truncation degrades to WARN (pins the floor's
    WARN-degradation direction)."""
    op_short = "sweep X effect"  # 14 normalized chars — under the 20-char floor
    body = _v4_body_with_context_quote("- Originating prompt, verbatim:\n\n> sweep X\n")
    ctx = verify_task_body.check_repro_context_provenance(body, {"origin_prompt": op_short})
    assert ctx.passed and ctx.is_warn, ctx.detail
    assert "first divergence" in ctx.detail


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


def test_v4_takeaways_bullet_over_30_words_warns_not_fails():
    """A 35-word Takeaways bullet WARNs (existing tier) — no FAIL."""
    body = _V4_GOOD_BODY.replace(
        "- Secondary: capability holds at 0.82 vs baseline 0.81 — no regression at 25% mixing.",
        "- " + " ".join(["word"] * 35),
    )
    _ok, results = verify_task_body.verify_text(body)
    caps = _results_by_name(results)["v4 conciseness caps"]
    assert caps.passed and caps.is_warn
    assert "exceed 30 words" in caps.detail


def test_v4_takeaways_bullet_at_100_words_fails():
    """A 100-word Takeaways bullet is a hard FAIL (the #825 accretion
    tier) and is NOT double-counted into the 30-word WARN line."""
    body = _V4_GOOD_BODY.replace(
        "- Secondary: capability holds at 0.82 vs baseline 0.81 — no regression at 25% mixing.",
        "- " + " ".join(["word"] * 100),
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    caps = _results_by_name(results)["v4 conciseness caps"]
    assert not caps.passed
    assert "100" in caps.detail
    assert "exceed 30 words" not in caps.detail  # mutually exclusive tiers


def test_v4_takeaways_bullet_99_words_warns_only():
    """Boundary: 99 words stays in the WARN tier (FAIL is >= 100)."""
    body = _V4_GOOD_BODY.replace(
        "- Secondary: capability holds at 0.82 vs baseline 0.81 — no regression at 25% mixing.",
        "- " + " ".join(["word"] * 99),
    )
    _ok, results = verify_task_body.verify_text(body)
    caps = _results_by_name(results)["v4 conciseness caps"]
    assert caps.passed and caps.is_warn
    assert "exceed 30 words" in caps.detail


def test_v3_takeaways_bullet_over_100_words_still_warn_only():
    """Grandfathering: a 263-word bullet on a v3 body stays WARN-only
    (forward-only rule — a v3 body is never newly hard-FAILed by a v4
    rule). Also pins the tuple-consuming v3 caller: an unmodified clean
    v3 body carries NO spurious per-bullet WARN (a forgotten int->tuple
    caller would make the truthy ``(0, 0)`` WARN every v3 body), and the
    mutated body's WARN carries the exact count-prefixed message."""
    clean_caps = _results_by_name(verify_task_body.verify_text(_V3_GOOD_BODY)[1])[
        "v3 conciseness caps"
    ]
    assert clean_caps.passed and not clean_caps.is_warn, clean_caps.render()

    body = _V3_GOOD_BODY.replace(
        "- Secondary: capability holds at 0.82 vs baseline 0.81 — no regression at 25% mixing.",
        "- " + " ".join(["word"] * 263),
    )
    _ok, results = verify_task_body.verify_text(body)
    caps = _results_by_name(results)["v3 conciseness caps"]
    assert caps.passed and caps.is_warn
    assert "1 Takeaways bullet(s) exceed 30 words" in caps.detail


def test_v4_takeaways_fenced_pseudo_bullet_not_counted():
    """Fence-aware counting unchanged through the tuple refactor: a
    150-word bullet-shaped line inside a code fence in ## Takeaways
    neither WARNs nor FAILs."""
    body = _V4_GOOD_BODY.replace(
        "## Goal",
        "```\n- " + " ".join(["word"] * 150) + "\n```\n\n## Goal",
    )
    _ok, results = verify_task_body.verify_text(body)
    caps = _results_by_name(results)["v4 conciseness caps"]
    assert caps.passed, caps.render()
    assert "Takeaways bullet" not in caps.detail


def test_v4_takeaways_mixed_warn_and_fail_bullets_both_reported():
    """A body with one ~35-word AND one >=100-word bullet FAILs with the
    >=100 message AND the concatenated `; WARN: ... exceed 30 words`
    tail — each bullet counted in exactly one tier (non-double-counting
    + the fails-then-WARN detail concatenation)."""
    body = _V4_GOOD_BODY.replace(
        "- Secondary: capability holds at 0.82 vs baseline 0.81 — no regression at 25% mixing.",
        "- " + " ".join(["word"] * 35),
    ).replace(
        "- Caveat that binds interpretation: single model family, three seeds only.",
        "- " + " ".join(["word"] * 100),
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    caps = _results_by_name(results)["v4 conciseness caps"]
    assert not caps.passed
    assert "1 Takeaways bullet(s) at ≥100 words" in caps.detail
    assert "; WARN: " in caps.detail
    assert "1 Takeaways bullet(s) exceed 30 words" in caps.detail


# ─── check 20 (v4): folded-round budget scaling (#921) ─────────────────────
#
# v4 bodies carry no `## What I ran` Rounds table, so the v3 round counter
# always scored 0 (incident #763: a 2-round body WARNed at budget 800).
# The v4 counter max-reconciles two signals: footer round clauses +
# non-retroactive `epm:same-issue-followup-run` events markers.


def _v4_fat_body(n_blocks: int = 6) -> str:
    """Inflate `_V4_GOOD_BODY` with `n_blocks` extra `### <result>` blocks
    (~150 prose words each, every block under the 180-word hard cap) so
    total content prose lands above the 800-word base budget. 6 blocks
    lands strictly in (800, 1300); 10 blocks exceeds 1300."""
    filler = "".join(
        f"\n### Extra result heading {i}\n\n"
        "Plotted: filler.\n\n"
        "![alt](https://raw.githubusercontent.com/superkaiba/explore-persona-space/"
        "0123456789abcdef/figures/issue_999/hero.png)\n\n"
        "> **Figure.** filler.\n\n" + " ".join(["word"] * 150) + "\n"
        for i in range(n_blocks)
    )
    return _V4_GOOD_BODY.replace("\n---\n**Repro:**", filler + "\n---\n**Repro:**")


def test_v4_round_count_footer_two_labels():
    """Two labeled footer round clauses -> extra_rounds 2, source `footer`."""
    body = _V4_GOOD_BODY.replace(
        "- Originating prompt: origin prompt not recorded",
        "- Originating prompt: origin prompt not recorded\n"
        "- same-issue follow-up round `round-a` (proposer-initiated) — run 2026-07-01 · "
        "same-issue follow-up round `round-b` (user-directed) — run 2026-07-02",
    )
    assert verify_task_body._count_extra_followup_rounds_v4(body, None) == (2, "footer")


def test_v4_round_count_dedupes_labels_and_ignores_plural_and_prose():
    """Repeated label counts once; plural 'rounds' prose counts zero; a
    clause OUTSIDE the footer (Goal/Methodology prose, the #811 shape)
    counts zero."""
    # (a) same label twice in the footer -> 1; (b) plural in footer -> +0.
    body = _V4_GOOD_BODY.replace(
        "- Originating prompt: origin prompt not recorded",
        "- same-issue follow-up round `round-a` — re-verified as "
        "same-issue follow-up round `round-a` · same-issue follow-up rounds "
        "also name each round",
    )
    assert verify_task_body._count_extra_followup_rounds_v4(body, None) == (1, "footer")
    # (c) phrase in body prose, not footer -> 0.
    body2 = _V4_GOOD_BODY.replace(
        "- **Design:**",
        "- A same-issue follow-up round then folded that sweep. **Design:**",
    )
    assert verify_task_body._count_extra_followup_rounds_v4(body2, None) == (0, "none")


def test_v4_round_count_footer_numbered_variant_case_insensitive():
    """Corpus-replay pin (Methodology-critic catch, round 1): #685's footer
    carries the sentence-initial numbered variant — the regex must match it
    (IGNORECASE + the `<n> (label: ` infix), capturing the backticked label."""
    body = _V4_GOOD_BODY.replace(
        "- Originating prompt: origin prompt not recorded",
        "- Originating prompt: origin prompt not recorded\n"
        "- Same-issue follow-up round 2 (label: `signed-cosine-matched-position-u`, "
        "folded 2026-07-01).",
    )
    assert verify_task_body._count_extra_followup_rounds_v4(body, None) == (1, "footer")


def test_v4_round_count_events_leg_excludes_retroactive_close(monkeypatch):
    """(events) Run markers count distinct labels; retroactive-close
    bookkeeping (line-leading AND the single-line mid-line corpus shape)
    and non-run kinds are excluded; max() reconciles with the footer."""
    import explore_persona_space.task_workflow as tw

    fake = [
        {
            "kind": "epm:same-issue-followup-run",
            "note": "followup_label: r-a\nsource: user-chat\noutcome: folded new results",
        },
        {
            "kind": "epm:same-issue-followup-run",
            "note": "followup_label: r-ghost\noutcome: retroactive-close — evidence",
        },
        # SINGLE-LINE note (the real corpus shape, fact-check 2026-07-03):
        # outcome is mid-line — only the mid-line fallback can see it.
        {
            "kind": "epm:same-issue-followup-run",
            "note": "followup_label: r-ghost2 source: proposer-9b outcome: retroactive-close — ev",
        },
        {"kind": "epm:progress", "note": "followup_label: not-a-run"},
    ]
    monkeypatch.setattr(tw, "list_events", lambda n: fake)
    assert verify_task_body._followup_run_marker_rounds(123) == 1
    # Fixture footer carries no round clause -> events leg wins.
    assert verify_task_body._count_extra_followup_rounds_v4(_V4_GOOD_BODY, 123) == (1, "events")


def test_v4_round_count_graceful_when_issue_unknown(monkeypatch):
    """Unknown issue id -> events leg 0, no crash (bare `--file` under a
    numeric tmp dir); registry corruption (`StaleTaskPathError`, a
    FileNotFoundError SUBCLASS) still propagates."""
    import explore_persona_space.task_workflow as tw

    def _boom(n):
        raise FileNotFoundError(n)

    monkeypatch.setattr(tw, "list_events", _boom)
    assert verify_task_body._followup_run_marker_rounds(999999) == 0
    assert verify_task_body._count_extra_followup_rounds_v4(_V4_GOOD_BODY, 999999) == (0, "none")

    def _stale(n):
        raise tw.StaleTaskPathError("registry entry stale for task")

    monkeypatch.setattr(tw, "list_events", _stale)
    with pytest.raises(tw.StaleTaskPathError):
        verify_task_body._followup_run_marker_rounds(999999)


def test_v4_total_prose_budget_scales_with_folded_rounds():
    """(End-to-end, the #763 incident shape.) Same >800-word body: with two
    footer round clauses the total-prose WARN clears at budget 1300;
    without them it fires naming budget 800 [none]. A >1300-word variant
    WITH the clauses pins the message shape: budget 1300 + [footer]."""
    fat = _v4_fat_body(6)  # >800 and <1300 total-prose words

    def _with_rounds(body: str) -> str:
        return body.replace(
            "- Originating prompt: origin prompt not recorded",
            "- Originating prompt: origin prompt not recorded\n"
            "- same-issue follow-up round `round-a` — run 2026-07-01 · "
            "same-issue follow-up round `round-b` — run 2026-07-02",
        )

    _ok, res = verify_task_body.verify_text(_with_rounds(fat))
    assert "total content prose" not in _results_by_name(res)["v4 conciseness caps"].detail
    _ok2, res2 = verify_task_body.verify_text(fat)
    detail2 = _results_by_name(res2)["v4 conciseness caps"].detail
    assert "budget 800" in detail2
    assert "[none]" in detail2
    # Message-shape pin: >1300 words + 2 footer rounds -> WARN names the
    # scaled budget and the winning source tag.
    _ok3, res3 = verify_task_body.verify_text(_with_rounds(_v4_fat_body(10)))
    detail3 = _results_by_name(res3)["v4 conciseness caps"].detail
    assert "budget 1300" in detail3
    assert "[footer]" in detail3


def test_v4_prose_budget_events_leg_through_verify_text(monkeypatch):
    """(MUST-FIX, round-1 alternatives reconcile) Issue-mode WIRING pin:
    the events leg must flow through the PUBLIC dispatch path
    `verify_text(body, issue=...) -> check_v4_word_caps(body, issue=issue)`.
    Kills the mutant that drops `issue=issue` at the verify_text call site —
    under that mutation every other test still passes (helpers are tested
    directly; the footer end-to-end needs no issue), and #685-shaped
    events-only bodies would silently keep the zero-round budget."""
    import explore_persona_space.task_workflow as tw

    fake = [
        {
            "kind": "epm:same-issue-followup-run",
            "note": "followup_label: r-a source: proposer-9b outcome: folded",
        },
        {
            "kind": "epm:same-issue-followup-run",
            "note": "followup_label: r-b source: proposer-9b outcome: folded",
        },
    ]
    monkeypatch.setattr(tw, "list_events", lambda n: fake)
    fat = _v4_fat_body(6)  # >800 and <1300 words; NO footer round clauses
    _ok, res = verify_task_body.verify_text(fat, issue=123)
    detail = _results_by_name(res)["v4 conciseness caps"].detail
    assert "total content prose" not in detail  # events leg engaged via verify_text
    _ok2, res2 = verify_task_body.verify_text(fat)  # no issue -> footer-only -> 800
    assert "budget 800" in _results_by_name(res2)["v4 conciseness caps"].detail
    # Message-shape pin: >1300 words, events-only -> budget 1300 + [events].
    _ok3, res3 = verify_task_body.verify_text(_v4_fat_body(10), issue=123)
    detail3 = _results_by_name(res3)["v4 conciseness caps"].detail
    assert "budget 1300" in detail3
    assert "[events]" in detail3


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
        "Results paragraph sentence cap (v4)",
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


_V4_FOOTER_ANCHOR = "\n---\n**Repro:**"


def test_v4_results_body_survives_mid_results_hr():
    """A legal mid-Results `---` rule between two results must NOT truncate
    the scan (#1109): the second result stays visible to checks 20/21, so
    its ≥180-word interpretation hard-FAILs check 20. Pre-fix, the footer
    first-line string match cut at the mid-`---` and check 20 falsely
    PASSed (the #825 masking incident)."""
    assert _V4_FOOTER_ANCHOR in _V4_GOOD_BODY, "fixture drifted: footer anchor missing"
    long_interp = " ".join(["word"] * 185)
    body = _V4_GOOD_BODY.replace(
        _V4_FOOTER_ANCHOR,
        "\n---\n\n### A second result behind a horizontal rule\n\n"
        "Plotted: a second panel.\n\n" + long_interp + "\n\n---\n**Repro:**",
    )
    _fm, b = verify_task_body.split_frontmatter(body)
    trunc = verify_task_body._v4_results_body(b)
    assert "### A second result" in trunc
    assert long_interp in trunc
    assert "**Repro:**" not in trunc
    assert "**Context:**" not in trunc
    _ok, results = verify_task_body.verify_text(body)
    r = _results_by_name(results)["v4 conciseness caps"]
    assert not r.passed, r.render()
    assert "≥180" in r.detail, r.render()


def test_v4_results_beat_scans_results_after_mid_hr():
    """The three-beat scan (check 21) sees results AFTER a mid-Results
    `---` rule: with a properly-framed second result behind the rule the
    detail reports `all 2` framed. Pre-fix the second result was cut out
    of the scan entirely (`all 1`)."""
    assert _V4_FOOTER_ANCHOR in _V4_GOOD_BODY, "fixture drifted: footer anchor missing"
    second = (
        "\n---\n\n### A second framed result behind a horizontal rule\n\n"
        "Plotted: mean capability (y, %) per condition (x: baseline, tulu-25), "
        "n=3 seeds per bar, 95% Wald CI error bars.\n\n"
        "![Bar chart of mean capability with 95% CI across three seeds.]"
        "(https://raw.githubusercontent.com/superkaiba/explore-persona-space/"
        "0123456789abcdef/figures/issue_999/second.png)\n\n"
        "> **Figure.** *Capability holds across conditions.* Baseline gray, "
        "tulu-25 blue; error bars 95% Wald CIs.\n\n"
        "Capability is flat across seeds; the largest gap is 0.4 pts.\n"
        "\n---\n**Repro:**"
    )
    body = _V4_GOOD_BODY.replace(_V4_FOOTER_ANCHOR, second)
    _ok, results = verify_task_body.verify_text(body)
    r = _results_by_name(results)["Results three-beat shape (v4)"]
    assert r.passed and not r.is_warn, r.render()
    assert "all 2" in r.detail, r.render()


def test_v4_results_body_equal_without_mid_hr():
    """Compatibility pin: on a body with NO mid-Results `---` the new
    absolute-index cut yields exactly the old output — the full section
    minus the footer, trailing blank/`---` chrome stripped."""
    _fm, b = verify_task_body.split_frontmatter(_V4_GOOD_BODY)
    out = verify_task_body._v4_results_body(b)
    assert out.endswith("is 1.2 pts.")
    assert not out.endswith("---")
    assert "**Repro:**" not in out
    # Independent expectation: lines strictly between the `## Results`
    # header line and the footer's `---` rule, trailing chrome stripped.
    lines = b.splitlines()
    h2_idx = lines.index("## Results")
    repro_idx = next(i for i, ln in enumerate(lines) if ln.startswith("**Repro:**"))
    rule_idx = repro_idx - 1
    assert lines[rule_idx].strip() == "---", "fixture drifted: footer not `---`-preceded"
    expected_lines = lines[h2_idx + 1 : rule_idx]
    while expected_lines and expected_lines[-1].strip() in ("", "---"):
        expected_lines.pop()
    assert out == "\n".join(expected_lines).strip()


def test_v4_results_body_no_footer_keeps_full_section():
    """Case (a) parity: no footer at all — the helper returns the plain
    `section_text` untouched, INCLUDING a mid-Results `---` and a retained
    trailing rule (the trailing-strip runs only on the footer-cut branch)."""
    assert _V4_FOOTER_ANCHOR in _V4_GOOD_BODY, "fixture drifted: footer anchor missing"
    head = _V4_GOOD_BODY[: _V4_GOOD_BODY.index(_V4_FOOTER_ANCHOR)]
    body = head.replace(
        "Plotted: mean alignment",
        "---\n\nPlotted: mean alignment",  # a mid-Results rule
    )
    body = body + "\n---\n"  # a trailing rule with no footer after it
    _fm, b = verify_task_body.split_frontmatter(body)
    assert verify_task_body._v4_footer_start_line(b) is None
    out = verify_task_body._v4_results_body(b)
    assert out == verify_task_body.section_text(b, "Results")
    assert out.endswith("---")  # trailing rule retained, as today


def test_v4_results_body_ignores_fenced_hr_in_results():
    """A `---` inside a fenced code block within Results is content, not a
    cut point: the absolute-index cut lands at the real footer and the
    fenced rule plus everything after it stays in the scanned text. (The
    old first-line string match was not fence-aware and cut at the fenced
    `---` — a third old/new divergence class, strictly more correct now.)"""
    assert _V4_FOOTER_ANCHOR in _V4_GOOD_BODY, "fixture drifted: footer anchor missing"
    fenced = (
        "\n```text\n---\nfenced rule above is content\n```\n\n"
        "Post-fence interpretation stays in the scan." + _V4_FOOTER_ANCHOR
    )
    body = _V4_GOOD_BODY.replace(_V4_FOOTER_ANCHOR, fenced)
    _fm, b = verify_task_body.split_frontmatter(body)
    out = verify_task_body._v4_results_body(b)
    assert "fenced rule above is content" in out
    assert "Post-fence interpretation stays in the scan." in out
    assert "**Repro:**" not in out


def test_v4_results_body_footer_without_preceding_rule():
    """Case (b): a footer with NO preceding `---` rule — the cut lands at
    the `**Repro:**` line itself, the blank gap above it is stripped, and
    the output equals the `---`-preceded fixture's cut (the rule is footer
    chrome either way)."""
    assert _V4_FOOTER_ANCHOR in _V4_GOOD_BODY, "fixture drifted: footer anchor missing"
    body = _V4_GOOD_BODY.replace(_V4_FOOTER_ANCHOR, "\n**Repro:**")
    _fm, b = verify_task_body.split_frontmatter(body)
    out = verify_task_body._v4_results_body(b)
    assert out.endswith("is 1.2 pts.")
    assert "**Repro:**" not in out
    _fm2, b2 = verify_task_body.split_frontmatter(_V4_GOOD_BODY)
    assert out == verify_task_body._v4_results_body(b2)


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


# ─── Check 36: v4 result-paragraph sentence cap (#1368) ───────────────────
#
# WARN-only, v4-sentinel-gated: any single prose paragraph inside a
# `### <result>` running >=4 sentences WARNs (never FAILs — register
# judgment stays with the clean-result-critic, Lens 12). All assertions
# per-check by name (the `_V4_GOOD_BODY` convention — fake SHAs fail the
# existence probes, so overall PASS is not assertable).

_V4_INTERP_LINE = (
    "The 17-pt lift holds at every seed; "
    "the smallest within-condition gap between seeds is 1.2 pts."
)


def test_v4_result_paragraph_sentence_cap_warns_on_four_sentences():
    """PRIMARY pin (#1368 durability pin): a 4-sentence interpretation
    paragraph WARNs (passed=True, is_warn=True) and the detail names the
    offending result + the sentence count."""
    body = _V4_GOOD_BODY.replace(
        _V4_INTERP_LINE,
        "The lift holds at every seed. The smallest gap is 1.2 pts. "
        "The effect is stable across conditions. No seed reverses the ordering.",
    )
    _ok, results = verify_task_body.verify_text(body)
    r = _results_by_name(results)["Results paragraph sentence cap (v4)"]
    assert r.passed, r.render()  # WARN counts as pass — NEVER FAIL
    assert r.is_warn, r.render()
    assert "A clean +17-pt lift" in r.detail, r.detail
    assert "4-sentence" in r.detail, r.detail


def test_v4_result_paragraph_sentence_cap_three_sentences_no_warn():
    """Exactly 3 sentences sits AT the cap — no WARN."""
    body = _V4_GOOD_BODY.replace(
        _V4_INTERP_LINE,
        "The lift holds at every seed. The smallest gap is 1.2 pts. No seed reverses the ordering.",
    )
    _ok, results = verify_task_body.verify_text(body)
    r = _results_by_name(results)["Results paragraph sentence cap (v4)"]
    assert r.passed and not r.is_warn, r.render()


def test_v4_result_paragraph_sentence_cap_guards_do_not_split():
    """Abbreviations (`e.g.`, `vs.`, `et al.`, `cf.`, `Fig.`, `no.` before a
    digit), decimals, inline-code dotted tokens, dotted link targets, and
    ellipses do not split sentences: a guard-dense 3-sentence paragraph
    stays under the cap."""
    para = (
        "We compare e.g. the tulu-25 arm vs. the baseline per et al. (2025), "
        "reading 1.2 pts from `run_result.json`. "
        "See [the panel](https://x.co/a.b/c.png) for Fig. 2 details, "
        "i.e. the per-seed view. "
        "The gap is stable (cf. no. 3 in the appendix)..."
    )
    assert verify_task_body._count_sentences(para) == 3
    body = _V4_GOOD_BODY.replace(_V4_INTERP_LINE, para)
    _ok, results = verify_task_body.verify_text(body)
    r = _results_by_name(results)["Results paragraph sentence cap (v4)"]
    assert r.passed and not r.is_warn, r.render()


def test_v4_result_paragraph_sentence_cap_excludes_caption_fences_bullets_tables():
    """Blockquote captions, fenced code, single bullets, and GFM table rows
    are excluded from paragraph construction: each carries >=5 sentences
    here while every prose paragraph stays <=3 — no WARN."""
    body = _V4_GOOD_BODY.replace(
        _V4_INTERP_LINE + "\n",
        "> A caption sentence. Another one. And another. A fourth. A fifth.\n\n"
        "```text\nOne fenced. Two fenced. Three fenced. Four fenced. Five fenced.\n```\n\n"
        "- One bullet sentence. Two here. Three here. Four here. Five here.\n\n"
        "| Cell | Note |\n|---|---|\n| One. Two. Three. | Four. Five. Six. |\n\n"
        "The 17-pt lift holds at every seed.\n",
    )
    _ok, results = verify_task_body.verify_text(body)
    r = _results_by_name(results)["Results paragraph sentence cap (v4)"]
    assert r.passed and not r.is_warn, r.render()


def test_v4_result_paragraph_sentence_cap_skips_v3_body():
    """Forward-only: v3 bodies are never flagged (sentinel gate)."""
    _ok, results = verify_task_body.verify_text(_V3_GOOD_BODY)
    r = _results_by_name(results)["Results paragraph sentence cap (v4)"]
    assert r.passed and "not a v4 body" in r.detail, r.render()


# ─── check 27: bare `#K` issue refs in v4 standalone sections ────────────────

_BARE_REF_CHECK_NAME = "no bare issue refs in standalone sections (v4)"


def _bare_ref_result(fixture_text):
    """Direct-call check 27 on the post-frontmatter body, as verify_text does."""
    _fm, body = verify_task_body.split_frontmatter(fixture_text)
    return verify_task_body.check_v4_no_bare_issue_refs(body)


def test_v4_bare_issue_ref_in_methodology_prose_fails():
    """The #841 shape: a bare `#779` in Methodology prose is a hard FAIL
    (run through verify_text to pin the CHECKS registration)."""
    body = _V4_GOOD_BODY.replace(
        "- **Design:** 3 seeds;",
        "- **Design:** 3 seeds on the #779 LMSYS corpus;",
    )
    _ok, results = verify_task_body.verify_text(body)
    r = _results_by_name(results)[_BARE_REF_CHECK_NAME]
    assert not r.passed
    assert "#779" in r.detail
    assert "Methodology" in r.detail


def test_v4_issue_ref_in_table_row_passes():
    """A `#K` in a GFM table row (the Training-table Source column
    grounding convention) is a sanctioned form."""
    body = _V4_GOOD_BODY.replace(
        "| Seeds | [42, 137, 256] | plan §11 |\n",
        "| Seeds | [42, 137, 256] | plan §11 |\n| Judge cache | reused | #779 |\n",
    )
    r = _bare_ref_result(body)
    assert r.passed, r.render()


def test_v4_task_link_in_methodology_fails():
    """The #928 shape: a `[#K](https://eps.superkaiba.com/tasks/K)` LINK in
    Methodology prose is a hard FAIL — the task-URL scan runs BEFORE
    `_LINK_RE` erases link targets (#1002). Inverts the pre-#1002 pin
    `test_v4_linked_issue_ref_passes_mechanically`, which pinned the
    linked form as out of mechanical scope."""
    body = _V4_GOOD_BODY.replace(
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct.",
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct. "
        "Recipe as in [#779](https://eps.superkaiba.com/tasks/779).",
    )
    r = _bare_ref_result(body)
    assert not r.passed
    assert "tasks/779" in r.detail
    assert "Methodology" in r.detail


def test_v4_task_link_in_takeaways_fails():
    """A task link appended to a Takeaways bullet FAILs (run through
    verify_text to pin the CHECKS registration on the Takeaways span)."""
    body = _V4_GOOD_BODY.replace(
        "- Caveat that binds interpretation: single model family, three seeds only.\n",
        "- Caveat that binds interpretation: single model family, three seeds only.\n"
        "- Protocol matches [#537](https://eps.superkaiba.com/tasks/537).\n",
    )
    _ok, results = verify_task_body.verify_text(body)
    r = _results_by_name(results)[_BARE_REF_CHECK_NAME]
    assert not r.passed
    assert "tasks/537" in r.detail
    assert "Takeaways" in r.detail


def test_v4_task_link_in_results_fails():
    """A task link in a `> **Figure.**` caption line under `## Results`
    FAILs (prose, not a sanctioned form — mirrors
    test_v4_bare_issue_ref_in_results_caption_fails)."""
    body = _V4_GOOD_BODY.replace(
        "error bars 95% Wald CIs.",
        "error bars 95% Wald CIs ([#667](https://eps.superkaiba.com/tasks/667) protocol).",
    )
    r = _bare_ref_result(body)
    assert not r.passed
    assert "tasks/667" in r.detail
    assert "Results" in r.detail


def test_v4_bare_task_url_in_results_fails():
    """Scope pin: a BARE task URL in Results prose FAILs — dropping the
    `[label](...)` brackets does not dodge the check (#1002 §4b)."""
    body = _V4_GOOD_BODY.replace(
        "the smallest within-condition gap between seeds is 1.2 pts.",
        "the smallest within-condition gap between seeds is 1.2 pts. "
        "Protocol: https://eps.superkaiba.com/tasks/658.",
    )
    r = _bare_ref_result(body)
    assert not r.passed
    assert "tasks/658" in r.detail
    assert "Results" in r.detail


def test_v4_autolink_task_url_in_methodology_fails():
    """Scope pin: a `<https://.../tasks/K>` angle-bracket autolink FAILs —
    subsumed by the URL scan (#1002 §4b)."""
    body = _V4_GOOD_BODY.replace(
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct.",
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct. "
        "See <https://eps.superkaiba.com/tasks/658>.",
    )
    r = _bare_ref_result(body)
    assert not r.passed
    assert "tasks/658" in r.detail
    assert "Methodology" in r.detail


def test_v4_task_link_in_goal_passes():
    """`## Goal` is NOT a standalone section — a second task link in the
    context slot stays sanctioned (the fixture's `[#34](...)` link is
    additionally asserted by test_v4_good_body_passes_all)."""
    body = _V4_GOOD_BODY.replace(
        "sits in the trait-transfer line.",
        "sits in the trait-transfer line, extending [#658](https://eps.superkaiba.com/tasks/658).",
    )
    r = _bare_ref_result(body)
    assert r.passed, r.render()


def test_v4_task_link_in_footer_passes():
    """Footer-cut parity: a task link AND a bare task URL on footer lineage
    lines are sanctioned (the bare-URL case pins the parity directly)."""
    body = _V4_GOOD_BODY.replace(
        "- Originating prompt: origin prompt not recorded\n",
        "- Originating prompt: origin prompt not recorded\n"
        "- Lineage: [#658](https://eps.superkaiba.com/tasks/658); "
        "see also https://eps.superkaiba.com/tasks/742.\n",
    )
    r = _bare_ref_result(body)
    assert r.passed, r.render()


def test_v4_task_link_in_table_row_passes():
    """Table-row parity (the Training-table Source column grounding
    convention): a linked AND a bare task URL in GFM table rows are
    sanctioned."""
    body = _V4_GOOD_BODY.replace(
        "| Seeds | [42, 137, 256] | plan §11 |\n",
        "| Seeds | [42, 137, 256] | plan §11 |\n"
        "| Judge cache | reused | [#779](https://eps.superkaiba.com/tasks/779) |\n"
        "| Adapter | reused | https://eps.superkaiba.com/tasks/532 |\n",
    )
    r = _bare_ref_result(body)
    assert r.passed, r.render()


def test_v4_task_link_in_fenced_code_passes():
    """A task link inside a fenced code block is sanctioned (fence lines
    are structurally excluded from both scans)."""
    body = _V4_GOOD_BODY.replace(
        "\n## Results\n",
        "\n```text\nsee [#779](https://eps.superkaiba.com/tasks/779)\n```\n\n## Results\n",
    )
    r = _bare_ref_result(body)
    assert r.passed, r.render()


def test_v4_task_link_in_inline_code_passes():
    """The #1002 §4a semantics decision pin: inline code PROTECTS the URL
    scan (mask parity with the bare-token scan) — a backticked example
    link is verbatim syntax-as-data."""
    body = _V4_GOOD_BODY.replace(
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct.",
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct. "
        "Cite links as `[#779](https://eps.superkaiba.com/tasks/779)` in the Goal slot.",
    )
    r = _bare_ref_result(body)
    assert r.passed, r.render()


def test_v4_task_link_in_html_comment_passes():
    """A task link inside an HTML comment is sanctioned (the char-span
    comment mask covers the URL scan too)."""
    body = _V4_GOOD_BODY.replace(
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct.",
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct.\n"
        "<!-- see [#779](https://eps.superkaiba.com/tasks/779) -->",
    )
    r = _bare_ref_result(body)
    assert r.passed, r.render()


def test_v4_task_link_in_details_block_passes():
    """A task link inside the fixture's `<details open>` block (verbatim
    sample data) is sanctioned."""
    body = _V4_GOOD_BODY.replace(
        "<summary>5 example training rows (5 of 2,000 rows, random sample)</summary>\n",
        "<summary>5 example training rows (5 of 2,000 rows, random sample)</summary>\n"
        "\nRows drawn per [#658](https://eps.superkaiba.com/tasks/658).\n",
    )
    r = _bare_ref_result(body)
    assert r.passed, r.render()


def test_v4_non_task_url_links_pass():
    """Non-task-URL links (GitHub blob, HF) in Methodology prose are
    unaffected — the URL scan targets the dashboard task route only."""
    body = _V4_GOOD_BODY.replace(
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct.",
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct. Script at "
        "[run.py](https://github.com/superkaiba/explore-persona-space/blob/abc/scripts/run.py); "
        "adapter at [hf](https://huggingface.co/superkaiba1/explore-persona-space/tree/abc).",
    )
    r = _bare_ref_result(body)
    assert r.passed, r.render()


def test_v4_same_domain_non_task_url_passes():
    """Same-domain negative: a dashboard URL that is not the task route
    (`/sessions`, or `/tasks/` with no digits) does NOT match."""
    body = _V4_GOOD_BODY.replace(
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct.",
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct. Dashboard at "
        "https://eps.superkaiba.com/sessions and the https://eps.superkaiba.com/tasks/ index.",
    )
    r = _bare_ref_result(body)
    assert r.passed, r.render()


def test_task_link_check_skips_v3():
    """Forward-only: a task link in a v3 `## Findings` never fires the
    check (PASS-skip; mirror of test_bare_ref_check_skips_v3_and_legacy)."""
    v3 = _V3_GOOD_BODY.replace(
        "## Findings",
        "## Findings\n\nUses [#779](https://eps.superkaiba.com/tasks/779).",
        1,
    )
    r = _bare_ref_result(v3)
    assert r.passed
    assert "skipped" in r.detail


def test_bare_ref_check_skips_v3_and_legacy():
    """Forward-only: bare refs in a v3 `## Findings` / legacy prose never
    fire the check (PASS-skip)."""
    v3 = _V3_GOOD_BODY.replace("## Findings", "## Findings\n\nUses the #779 corpus.", 1)
    legacy = GOOD_BODY.replace("### Motivation", "### Motivation\n\nUses the #779 corpus.", 1)
    for fixture in (v3, legacy):
        r = _bare_ref_result(fixture)
        assert r.passed
        assert "skipped" in r.detail


def test_v4_bare_ref_in_fence_and_comment_passes():
    """`#K` inside a fenced code block or an HTML comment is sanctioned."""
    body = _V4_GOOD_BODY.replace(
        "\n## Results\n",
        "\n```text\ngrep for #779 rows\n```\n\n<!-- lineage: #613 -->\n\n## Results\n",
    )
    r = _bare_ref_result(body)
    assert r.passed, r.render()


def test_v4_bare_ref_in_footer_passes():
    """Lineage refs in the `**Context:**` footer are sanctioned (the
    footer-line cut); in a slash-run `#658/#742` nothing fires here."""
    body = _V4_GOOD_BODY.replace(
        "- Originating prompt: origin prompt not recorded\n",
        "- Originating prompt: origin prompt not recorded\n- Parent #34; informed by #658/#742.\n",
    )
    r = _bare_ref_result(body)
    assert r.passed, r.render()


def test_v4_bare_issue_ref_in_takeaways_fails():
    """Takeaways IS a standalone section (SPEC.md: `## Goal` is the ONLY
    place that may cite prior tasks) — a bare ref there FAILs (run through
    verify_text to pin the registration on the Takeaways span too)."""
    body = _V4_GOOD_BODY.replace(
        "- Caveat that binds interpretation: single model family, three seeds only.\n",
        "- Caveat that binds interpretation: single model family, three seeds only.\n"
        "- Matches the #537 protocol readout.\n",
    )
    _ok, results = verify_task_body.verify_text(body)
    r = _results_by_name(results)[_BARE_REF_CHECK_NAME]
    assert not r.passed
    assert "#537" in r.detail
    assert "Takeaways" in r.detail


def test_v4_bare_ref_in_details_block_passes():
    """`#K` inside a `<details>` block (verbatim sample data) is sanctioned;
    the fixture anchor is literally `<details open>` — pins the `<details\\b`
    regex covering attribute-bearing open tags."""
    body = _V4_GOOD_BODY.replace(
        "<summary>5 example training rows (5 of 2,000 rows, random sample)</summary>\n",
        "<summary>5 example training rows (5 of 2,000 rows, random sample)</summary>\n"
        "\nRows drawn per #658.\n",
    )
    r = _bare_ref_result(body)
    assert r.passed, r.render()


def test_v4_bare_issue_ref_in_results_caption_fails():
    """The live #667 shape: a bare ref in a `> **Figure.**` caption line
    under `## Results` FAILs (a blockquote caption is prose, not a
    sanctioned form)."""
    body = _V4_GOOD_BODY.replace(
        "error bars 95% Wald CIs.",
        "error bars 95% Wald CIs (#667 protocol).",
    )
    r = _bare_ref_result(body)
    assert not r.passed
    assert "#667" in r.detail
    assert "Results" in r.detail


def test_v4_inline_code_escape_hatch_passes():
    """A non-issue `#N` string (a 3-digit hex color) wrapped in inline code
    is the documented escape hatch."""
    body = _V4_GOOD_BODY.replace(
        "the smallest within-condition gap between seeds is 1.2 pts.",
        "the smallest within-condition gap between seeds is 1.2 pts. Bars colored `#333`.",
    )
    r = _bare_ref_result(body)
    assert r.passed, r.render()


def test_v4_space_substitution_does_not_fabricate_ref():
    """Neutralization substitutes a SPACE, never the empty string: on
    ``prefix #`v`123`` an empty-string strip of the inline-code span would
    JOIN `#` and `123` into a fabricated `#123` hit; the space substitution
    keeps them apart. Regression for the plan-review neutralization-join
    concern."""
    body = _V4_GOOD_BODY.replace(
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct.",
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct.\n"
        "- **Note:** config ids use the prefix #`v`123 shape.",
    )
    r = _bare_ref_result(body)
    assert r.passed, r.render()


def test_v4_bare_ref_prefix_prose_on_comment_opening_line_fails():
    """Char-span comment mask: prose BEFORE a `<!--` that opens a multiline
    comment is still scanned — `Uses #779 corpus <!-- note` hits (concern
    comment-mask-mixed-line-fail-open; the round-1 line-grain mask
    excluded the whole line and missed the ref)."""
    body = _V4_GOOD_BODY.replace(
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct.",
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct.\n"
        "Uses #779 corpus <!-- note\ninterior continues -->",
    )
    r = _bare_ref_result(body)
    assert not r.passed
    assert "#779" in r.detail
    assert "Methodology" in r.detail


def test_v4_bare_ref_suffix_prose_on_comment_closing_line_fails():
    """Char-span comment mask: prose AFTER the `-->` that closes a
    multiline comment is still scanned — `--> still follows #781` hits,
    while a `#999` on the comment's interior stays masked."""
    body = _V4_GOOD_BODY.replace(
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct.",
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct.\n"
        "<!-- lineage note #999\n--> still follows #781",
    )
    r = _bare_ref_result(body)
    assert not r.passed
    assert "#781" in r.detail
    assert "#999" not in r.detail


def test_v4_comment_close_reopen_masks_interior_and_scans_between():
    """Close-then-reopen on one line (`<!-- a --> #779 <!-- b`): both
    comment segments are masked, the prose BETWEEN them is scanned (the
    `#779` hit), and the state is left OPEN so a `#123` on the following
    interior line yields NO hit (concern
    comment-close-reopen-false-positive; the round-1 first-`<!--` anchor
    left the state closed and false-hit the interior)."""
    body = _V4_GOOD_BODY.replace(
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct.",
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct.\n"
        "<!-- a --> #779 <!-- b\nstill inside the comment: #123 -->",
    )
    _fm, post_fm = verify_task_body.split_frontmatter(body)
    hits = verify_task_body._bare_issue_ref_hits(post_fm)
    assert [(sec, tok) for sec, tok, _txt in hits] == [("Methodology", "#779")], hits


def test_v4_ref_flush_against_word_char_passes_but_possessive_fails():
    """`(?!\\w)` right guard: `#123abc` (digit run flush against a word
    char, the mixed-hex-color shape) never matches; a possessive `#658's`
    still FAILs — an apostrophe is not a word char."""
    body = _V4_GOOD_BODY.replace(
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct.",
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct.\nBars use the #123abc palette variant.",
    )
    r = _bare_ref_result(body)
    assert r.passed, r.render()
    body2 = _V4_GOOD_BODY.replace(
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct.",
        "- **Training:** LoRA SFT on Qwen-2.5-7B-Instruct.\nMatches #658's protocol.",
    )
    r2 = _bare_ref_result(body2)
    assert not r2.passed
    assert "#658" in r2.detail


def test_mask_html_comment_spans_char_grain():
    """Unit-pin `_mask_html_comment_spans`: space substitution preserves
    line length, prefix/suffix prose survives, close-then-reopen leaves
    the state OPEN."""
    f = verify_task_body._mask_html_comment_spans
    m, state = f("Uses #779 corpus <!-- note", False)
    assert m == "Uses #779 corpus " + " " * len("<!-- note")
    assert state is True
    m, state = f("all interior #123", True)
    assert m == " " * len("all interior #123")
    assert state is True
    m, state = f("--> tail #781", True)
    assert m == "   " + " tail #781"
    assert state is False
    m, state = f("<!-- a --> mid <!-- b", False)
    assert m == " " * len("<!-- a -->") + " mid " + " " * len("<!-- b")
    assert state is True
    assert all(len(f(s, st)[0]) == len(s) for s in ("", "x <!-- y --> z") for st in (False, True))


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
    # Check 0 ("body is not a stub") never appears in the output.
    assert all(r.name != "body is not a stub" for r in results)


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


# ─── Check 24: figure-embedded text vs body prose (figure-text staleness) ──
#
# A round-1 numeric / overclaim fix lands in body prose but is missed in the
# figure-generation script's hardcoded title/annotation strings, so the
# regenerated figure's `.meta.json` silently disagrees with the body. Check 24
# reads the figure's sidecar from the git tree at the URL's sha and WARNs on
# (a) a same-numerator/different-denominator fraction vs the body caption, or
# (b) a configured softened token. WARN-only, fail-soft. Incident: #667 r2.

_CHECK24_NAME = "figure text vs body prose (figure-text staleness)"


def _make_repo_with_figure_meta(tmp_path, meta: dict):
    """Create a throwaway git repo whose HEAD commit carries
    `figures/issue_999/hero.png` AND its sibling `hero.meta.json` (content =
    ``meta``); return (repo_path, head_sha). Mirrors `_make_repo_with_figure`
    but also commits the sidecar check 24 reads via `git show`."""
    repo = tmp_path / "figrepo24"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    fig_dir = repo / "figures" / "issue_999"
    fig_dir.mkdir(parents=True)
    (fig_dir / "hero.png").write_bytes(b"\x89PNG fake bytes")
    (fig_dir / "hero.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    # GOOD_BODY's `**Code:**` blob link names scripts/run.py — commit it too so
    # check 8b's probe resolves cleanly and does not muddy assertions.
    script = repo / "scripts" / "run.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('entry script')\n")
    git("add", "figures", "scripts")
    git("commit", "-q", "-m", "add hero figure + sidecar + entry script")
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, sha


# A v4 body whose single result figure is a same-repo sha-pinned raw-GitHub
# URL with a caption stating a `1/29` chance level. We swap the placeholder sha
# for the throwaway repo's real HEAD sha per-test.
_CHECK24_BODY = _V4_GOOD_BODY.replace(
    "> **Figure.** *Tulu-25 lifts alignment ~17 pts over baseline at every seed.* "
    "Baseline gray, tulu-25 blue; error bars 95% Wald CIs.",
    "> **Figure.** *Chance is 1/29 (one correct of 29 candidates).* "
    "Baseline gray, tulu-25 blue; error bars 95% Wald CIs.",
)


def test_check24_stale_fraction_in_figure_warns(tmp_path, monkeypatch):
    """Figure sidecar `description` says `1/30` while the body caption says
    `1/29` (same numerator, different denominator) → WARN, never FAIL."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {
            "commit": "deadbeef",  # provenance — must be ignored
            "created": "2026-06-24T00:00:00Z",
            "description": "Accuracy vs the 1/30 chance baseline across conditions.",
        },
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    res = by_name[_CHECK24_NAME]
    # WARN counts as passed=True, is_warn=True — it must NOT fail the body.
    assert res.passed and res.is_warn, res.render()
    assert "1/30" in res.detail and "1/29" in res.detail
    # A WARN does not flip overall PASS (the only FAILs are the fake-sha probes,
    # which the real sha here resolves, so overall is driven by other checks).
    assert _CHECK24_NAME not in {r.name for r in results if not r.passed}


def test_check24_softened_token_in_figure_warns(tmp_path, monkeypatch):
    """A figure sidecar carrying a configured stale token ("geometrically
    real", removed from body prose) → WARN."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {
            "created": "2026-06-24T00:00:00Z",
            "description": "Cosine recipe shows a geometrically real separation.",
        },
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    _ok, results = verify_task_body.verify_text(body)
    res = _results_by_name(results)[_CHECK24_NAME]
    assert res.passed and res.is_warn, res.render()
    assert "geometrically real" in res.detail


def test_check24_consistent_figure_passes_clean(tmp_path, monkeypatch):
    """Figure sidecar whose only fraction matches the caption (`1/29`) and
    carries no softened token → clean PASS (no WARN)."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {
            "created": "2026-06-24T00:00:00Z",
            "description": "Accuracy vs the 1/29 chance baseline across conditions.",
        },
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    _ok, results = verify_task_body.verify_text(body)
    res = _results_by_name(results)[_CHECK24_NAME]
    assert res.passed and not res.is_warn, res.render()
    assert "1 figure sidecar(s) consistent" in res.detail


def test_check24_provenance_keys_never_flagged(tmp_path, monkeypatch):
    """A commit sha / timestamp inside provenance keys must NOT be read as a
    stale token, and a fraction-like date must not false-WARN against the
    caption's `1/29`."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {
            "commit": "1234567",
            "git_sha": "1/30",  # provenance key — ignored even though it looks like a fraction
            "created": "2026/06/24",  # date, not a chance fraction
            "figsize": [7.0, 4.2],
            "description": "Accuracy across conditions.",  # no chance fraction, no stale token
        },
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    _ok, results = verify_task_body.verify_text(body)
    res = _results_by_name(results)[_CHECK24_NAME]
    assert res.passed and not res.is_warn, res.render()


def test_check24_no_sidecar_is_noop_pass(tmp_path, monkeypatch):
    """A same-repo figure with NO `.meta.json` sibling → NO-OP PASS (nothing
    to compare), never a WARN or FAIL."""
    repo, sha = _make_repo_with_figure(tmp_path)  # commits hero.png but no sidecar
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    _ok, results = verify_task_body.verify_text(body)
    res = _results_by_name(results)[_CHECK24_NAME]
    assert res.passed and not res.is_warn, res.render()
    assert "nothing to compare" in res.detail


def test_check24_repo_unresolved_is_noop_pass(monkeypatch):
    """Offline / repo root unresolved → NO-OP PASS (the v4 body still carries
    a figure, but there is no git to read the sidecar from)."""
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: None)
    res = verify_task_body.check_figure_text_vs_body_tokens(_CHECK24_BODY)
    assert res.passed and not res.is_warn
    assert "repo root unresolved" in res.detail


def test_check24_stale_token_in_text_suptitle_warns(tmp_path, monkeypatch):
    """The `meta["text"]` rendered-text block (new `savefig_paper` output) is
    scanned by check 24 with ZERO scan-code change: a configured stale token
    sitting in the actual rendered SUPTITLE → WARN (`_flatten_meta_strings`
    walks all non-provenance strings, so the new subtree enters
    automatically; forward-only — old sidecars simply lack the key)."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {
            "created": "2026-07-10T00:00:00Z",
            "text": {
                "suptitle": "Cosine recipe shows a geometrically real separation",
                "fig_texts": [],
                "axes": [{"xlabel": "condition"}],
            },
        },
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_text_vs_body_tokens(body)
    assert res.passed and res.is_warn, res.render()
    assert "geometrically real" in res.detail


def test_check24_user_token_file_extends_list(tmp_path, monkeypatch):
    """`~/.eps-stale-tokens.json` extends the built-in list, read fail-soft."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".eps-stale-tokens.json").write_text(json.dumps(["my custom artifact"]))
    monkeypatch.setattr(verify_task_body.Path, "home", staticmethod(lambda: home))
    tokens = verify_task_body._load_stale_figure_tokens()
    assert "geometrically real" in tokens  # built-in survives
    assert "my custom artifact" in tokens  # user token appended


def test_check24_user_token_file_malformed_is_failsoft(tmp_path, monkeypatch):
    """A malformed `~/.eps-stale-tokens.json` falls back to the built-in list
    without raising."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".eps-stale-tokens.json").write_text("{ not valid json")
    monkeypatch.setattr(verify_task_body.Path, "home", staticmethod(lambda: home))
    tokens = verify_task_body._load_stale_figure_tokens()
    assert tokens == ["geometrically real"]


def test_check24_flatten_meta_strings_skips_provenance():
    """The flattener collects chart text (description, series, label, axis-key)
    but drops provenance keys + numeric leaves."""
    meta = {
        "commit": "abc1234",
        "created": "2026-06-24T00:00:00Z",
        "figsize": [7.0, 4.2],
        "description": "1/30 chance baseline",
        "points": [
            {"category": "baseline", "accuracy": 0.41, "series": "cosine recipe"},
            {"1/30 chance accuracy": 0.5, "label": "geometrically real point"},
        ],
    }
    blob = " ".join(verify_task_body._flatten_meta_strings(meta))
    assert "1/30 chance baseline" in blob
    assert "cosine recipe" in blob
    assert "geometrically real point" in blob
    assert "1/30 chance accuracy" in blob  # axis-label KEY collected
    assert "abc1234" not in blob  # provenance value dropped
    assert "2026-06-24" not in blob  # provenance value dropped


def test_check24_caption_after_helper():
    """`_figure_caption_after` returns the contiguous blockquote run after an
    image, skipping blank lines, and '' when there is none."""
    rlines = [
        "![alt](https://x/y.png)",
        "",
        "> **Figure.** *Chance 1/29.*",
        "> continued caption line.",
        "",
        "Interpretation prose, not a caption.",
    ]
    cap = verify_task_body._figure_caption_after(rlines, 0)
    assert "Chance 1/29" in cap and "continued caption line" in cap
    assert "Interpretation prose" not in cap
    # No blockquote after the image → empty.
    assert verify_task_body._figure_caption_after(["![a](u)", "plain prose"], 0) == ""


# ─── Check 26: figure panel/series prose vs figure sidecar (panel drift) ───
#
# A clean-result body's what-is-plotted prose under a `### <result>` H3 asserts
# a plot kind in a named panel position ("right panel scatter") or a per-unit
# dot/point overlay ("per-bank dots overlaid") that the SHA-pinned `.meta.json`
# sidecar's `_kind` aggregate provably lacks. The sidecar is resolved strictly
# by URL stem at the figure's commit sha — no silent fallback to a different
# sidecar (the failure mode the check exists to catch, incident #683 r1).
# FAIL, never WARN; conservative (only fires on an explicit panel/overlay word).

_CHECK26_NAME = "figure panel prose vs figure sidecar (panel/series drift)"

# A v4 body whose single result figure's what-is-plotted prose names a "Right
# panel — scatter" structural claim. We swap the placeholder sha for the
# throwaway repo's real HEAD sha per-test.
_CHECK26_BODY = _V4_GOOD_BODY.replace(
    "Plotted: mean alignment (y, %) per condition (x: baseline, tulu-25), "
    "n=3 seeds per bar, 95% Wald CI error bars.",
    "Plotted: Left panel — mean alignment bars per condition. "
    "Right panel — per-context scatter for villain seed 42.",
)


def _scatter_sidecar():
    """A sidecar with both bar and scatter points (a left-bars / right-scatter
    two-panel figure), mirroring the live #683 `leaderboard_sycophancy`."""
    return {
        "commit": "deadbeef",
        "created": "2026-06-24T00:00:00Z",
        "figsize": [10.0, 4.2],
        "points": [
            {"_kind": "bar", "_group": 0, "category": "baseline"},
            {"_kind": "bar", "_group": 1, "category": "tulu-25"},
            {"_kind": "scatter", "_group": 0, "x": 0.0, "y": 0.70},
            {"_kind": "scatter", "_group": 1, "x": 1.0, "y": 0.88},
        ],
        "n_series": 2,
        "total_points": 4,
    }


def test_check26_consistent_panel_prose_passes(tmp_path, monkeypatch):
    """`Right panel — per-context scatter` + sidecar carrying scatter points →
    PASS (`consistent with panel/series prose`)."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, _scatter_sidecar())
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK26_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_panel_prose_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()
    assert "1 figure sidecar(s) consistent with panel/series prose" in res.detail


def test_check26_kind_panel_drift_fails(tmp_path, monkeypatch):
    """`Right panel — ... scatter` prose + a bar-ONLY sidecar (zero scatter) →
    FAIL with the specific scatter-panel blocker."""
    bar_only = {
        "created": "2026-06-24T00:00:00Z",
        "points": [
            {"_kind": "bar", "_group": 0, "category": "baseline"},
            {"_kind": "bar", "_group": 1, "category": "tulu-25"},
        ],
        "n_series": 1,
        "total_points": 2,
    }
    repo, sha = _make_repo_with_figure_meta(tmp_path, bar_only)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK26_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_panel_prose_vs_sidecar(body)
    assert not res.passed, res.render()
    assert "claims a `scatter` panel/series" in res.detail
    assert "zero `scatter` points" in res.detail


def test_check26_panel_count_prose_passes_with_many_groups(tmp_path, monkeypatch):
    """A body whose prose says "four-panel grid" paired with a sidecar that has
    22 `_group` values (mirrors the live `a7_spectrum_marker`, correctly called
    a four-panel grid) → PASS. The check makes NO panel-count claim from
    `_group`, so no FAIL fires. Regression pin for must-fix #1."""
    body = _CHECK26_BODY.replace(
        "Plotted: Left panel — mean alignment bars per condition. "
        "Right panel — per-context scatter for villain seed 42.",
        "Plotted: a four-panel grid of per-bank singular spectra, one panel per source bank.",
    )
    # 22 `_group` values across bar/scatter points; bars + scatter both present.
    pts = []
    for g in range(22):
        pts.append({"_kind": "bar", "_group": g})
        pts.append({"_kind": "scatter", "_group": g})
    meta = {"created": "2026-06-24T00:00:00Z", "points": pts, "n_series": 22}
    repo, sha = _make_repo_with_figure_meta(tmp_path, meta)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = body.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_panel_prose_vs_sidecar(body)
    # "four-panel" carries no panel-position word and no kind word, so it yields
    # no structural claim at all → PASS (no panel/series prose to compare).
    assert res.passed and not res.is_warn, res.render()


def test_check26_overlay_multi_bar_group_zero_scatter_fails(tmp_path, monkeypatch):
    """`per-bank dots overlaid` prose + a sidecar with MULTIPLE bar groups and
    ZERO scatter points → FAIL. Regression pin for must-fix #2: the dropped
    `len(groups) <= 1` conjunction would have false-PASSed this shape."""
    body = _CHECK26_BODY.replace(
        "Plotted: Left panel — mean alignment bars per condition. "
        "Right panel — per-context scatter for villain seed 42.",
        "Plotted: per-bank dots overlaid on the mean alignment bars, one cluster per bank.",
    )
    multi_bar = {
        "created": "2026-06-24T00:00:00Z",
        "points": [
            {"_kind": "bar", "_group": 0},
            {"_kind": "bar", "_group": 1},
            {"_kind": "bar", "_group": 2},
        ],
        "n_series": 3,
        "total_points": 3,
    }
    repo, sha = _make_repo_with_figure_meta(tmp_path, multi_bar)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = body.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_panel_prose_vs_sidecar(body)
    assert not res.passed, res.render()
    assert "per-unit dot/point overlay" in res.detail
    assert "zero scatter points" in res.detail


def _make_repo_with_two_figure_metas(tmp_path, meta_a: dict, meta_b: dict):
    """Create a throwaway git repo committing TWO figures + sidecars,
    `figures/issue_999/hero.png` (meta_a) and `figures/issue_999/second.png`
    (meta_b), + the entry script; return (repo_path, head_sha)."""
    repo = tmp_path / "figrepo26two"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    fig_dir = repo / "figures" / "issue_999"
    fig_dir.mkdir(parents=True)
    (fig_dir / "hero.png").write_bytes(b"\x89PNG fake bytes")
    (fig_dir / "hero.meta.json").write_text(json.dumps(meta_a, indent=2) + "\n")
    (fig_dir / "second.png").write_bytes(b"\x89PNG fake bytes 2")
    (fig_dir / "second.meta.json").write_text(json.dumps(meta_b, indent=2) + "\n")
    script = repo / "scripts" / "run.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('entry script')\n")
    git("add", "figures", "scripts")
    git("commit", "-q", "-m", "two hero figures + sidecars + entry script")
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, sha


def test_check26_two_result_h3_boundary_no_leak(tmp_path, monkeypatch):
    """Two `### <result>` H3s: result 1 names "right panel scatter" + its
    sidecar HAS scatter; result 2 has NO structural claim + a bar-only sidecar.
    BOTH must PASS — result 2 must NOT inherit result 1's scatter claim.
    Regression pin for must-fix #3 (cross-result prose leak)."""
    results_block = (
        "## Results\n\n"
        "### A clean +17-pt lift between baseline and tulu-25 across three seeds\n\n"
        "Plotted: Right panel — per-context scatter for villain seed 42.\n\n"
        "![scatter result one](https://raw.githubusercontent.com/superkaiba/"
        "explore-persona-space/0123456789abcdef/figures/issue_999/hero.png)\n\n"
        "> **Figure.** *Result one scatter.* Error bars 95% Wald CIs.\n\n"
        "The lift holds at every seed.\n\n"
        "### Capability holds with no regression at 25% mixing\n\n"
        "Plotted: mean capability per condition, bars only.\n\n"
        "![bars result two](https://raw.githubusercontent.com/superkaiba/"
        "explore-persona-space/0123456789abcdef/figures/issue_999/second.png)\n\n"
        "> **Figure.** *Result two bars.* Error bars 95% Wald CIs.\n\n"
        "Capability is flat across conditions.\n"
    )
    # Splice the two-result block in place of the single-result Results section.
    head, _sep, _tail = _V4_GOOD_BODY.partition("## Results\n")
    body = head + results_block + "\n---\n**Repro:** 1x H100, 47 min · entry.\n"
    bar_only = {
        "created": "2026-06-24T00:00:00Z",
        "points": [{"_kind": "bar", "_group": 0}, {"_kind": "bar", "_group": 1}],
    }
    repo, sha = _make_repo_with_two_figure_metas(tmp_path, _scatter_sidecar(), bar_only)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = body.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_panel_prose_vs_sidecar(body)
    # Result 1 scatter claim is backed; result 2 makes no claim → both PASS.
    assert res.passed and not res.is_warn, res.render()


def test_check26_missing_sidecar_fails_when_png_resolves(tmp_path, monkeypatch):
    """PNG committed at the sha but NO `.meta.json` sibling, prose makes a
    panel claim → FAIL (no silent pass / no fallback to a different sidecar)."""
    repo, sha = _make_repo_with_figure(tmp_path)  # commits hero.png, NO sidecar
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK26_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_panel_prose_vs_sidecar(body)
    assert not res.passed, res.render()
    assert "does not resolve at the cited sha" in res.detail
    assert "no silent" in res.detail


def test_check26_no_structural_claim_passes(tmp_path, monkeypatch):
    """Prose "mean alignment per condition, bars" (no panel/overlay wording) +
    a single-`_kind` bar sidecar → PASS (no panel/series prose to compare).
    Over-fire guard: a bare kind word is NOT a structural claim."""
    body = _CHECK26_BODY.replace(
        "Plotted: Left panel — mean alignment bars per condition. "
        "Right panel — per-context scatter for villain seed 42.",
        "Plotted: mean alignment per condition, bars only.",
    )
    bar_only = {"created": "2026-06-24T00:00:00Z", "points": [{"_kind": "bar", "_group": 0}]}
    repo, sha = _make_repo_with_figure_meta(tmp_path, bar_only)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = body.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_panel_prose_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()
    assert "no panel/series prose claims to compare" in res.detail


def test_check26_unresolvable_sha_is_noop_skip(tmp_path, monkeypatch):
    """The PNG does NOT resolve at the cited sha (a fake sha against a real
    repo) → no FAIL fires (defer to check 22), even with a panel claim in
    prose. Pins the `status != "pass"` gate (must-fix B)."""
    # A real repo whose HEAD is some other sha; the body cites a fake sha that
    # does not resolve, so `_git_object_exists` returns ('skip', ...).
    repo, _real_sha = _make_repo_with_figure(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK26_BODY  # keeps the unresolvable 0123456789abcdef placeholder sha
    res = verify_task_body.check_figure_panel_prose_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()
    assert "no panel/series prose claims to compare" in res.detail


def test_check26_figure_under_section_no_h3_passes(tmp_path, monkeypatch):
    """A figure that sits directly under `## Results` with NO preceding `### `
    H3, prose making a panel claim → PASS (the prose-vs-sidecar check is
    SKIPPED — no reliably-scoped window). Pins the must-fix #3 fallback."""
    results_block = (
        "## Results\n\n"
        "Plotted: Right panel — per-context scatter for villain seed 42.\n\n"
        "![no-h3 scatter](https://raw.githubusercontent.com/superkaiba/"
        "explore-persona-space/0123456789abcdef/figures/issue_999/hero.png)\n\n"
        "> **Figure.** *No H3 above me.* Error bars 95% Wald CIs.\n\n"
        "Some interpretation prose.\n"
    )
    head, _sep, _tail = _V4_GOOD_BODY.partition("## Results\n")
    body = head + results_block + "\n---\n**Repro:** 1x H100, 47 min · entry.\n"
    # Bar-only sidecar: WOULD FAIL the scatter claim if the figure were scanned,
    # so a PASS here proves the no-H3 figure is skipped, not merely backed.
    bar_only = {"created": "2026-06-24T00:00:00Z", "points": [{"_kind": "bar", "_group": 0}]}
    repo, sha = _make_repo_with_figure_meta(tmp_path, bar_only)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = body.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_panel_prose_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()


def test_check26_repo_unresolved_is_noop_pass(monkeypatch):
    """Offline / repo root unresolved → NO-OP PASS (no git to read the sidecar
    from)."""
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: None)
    res = verify_task_body.check_figure_panel_prose_vs_sidecar(_CHECK26_BODY)
    assert res.passed and not res.is_warn
    assert "repo root unresolved" in res.detail


# ─── Check 28: opaque config-code tokens in figure sidecar text (#920) ─────
#
# The no-opaque-condition-codes rule exists as prose only; #920's
# `winning_cell_scatter.png` reached the 9a-bis gate titled
# `ctx_blk_max@L12 x ans_uhdr_max@L12` after three review passes. Check 28
# reads the figure sidecar (parsed, `_read_figure_meta_json`) and WARNs on
# `@L<digits>` layer pins + regime-code slugs in the sidecar's rendered-text
# strings (string VALUES + whitespace-bearing keys; provenance subtrees
# pruned; path-shaped strings exempt). WARN-only, fail-soft. The body
# fixture is check 24's (`_CHECK24_BODY`) — check 28 keys only off the
# inline figure URL, not the caption.

_CHECK28_NAME = "figure text opaque config codes (slug / @L-pin tokens)"


def test_check28_slug_and_pin_in_description_warns(tmp_path, monkeypatch):
    """Sidecar `description` carrying slug@L-pin tokens → WARN (passed=True,
    is_warn=True) naming the basename + the offending token; the WARN must
    not flip the body's overall verdict."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {"description": "ctx_blk_max@L12 × ans_uhdr_max@L12 margin"},
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    _ok, results = verify_task_body.verify_text(body)
    res = _results_by_name(results)[_CHECK28_NAME]
    assert res.passed and res.is_warn, res.render()
    assert "ctx_blk_max@L12" in res.detail and "hero.png" in res.detail
    assert _CHECK28_NAME not in {r.name for r in results if not r.passed}


def test_check28_bare_layer_pin_warns(tmp_path, monkeypatch):
    """A bare `@L12` layer pin (no attached snake stem) still WARNs."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, {"description": "readout margin at @L12"})
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_label_codes(body)
    assert res.passed and res.is_warn, res.render()
    assert "@L12" in res.detail


def test_check28_cell_slugs_values_warn(tmp_path, monkeypatch):
    """The #920 shape: slug VALUES under an ad-hoc `cell_slugs` map WARN even
    though the map's own key is identifier-shaped (values are scanned
    regardless of the key that holds them)."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {
            "cell_slugs": {"c_cell": "ctx_blk_max@L12"},
            "cell_plain": {"c_cell": "template-block max"},
            "description": "held-out prediction vs true target",
        },
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_label_codes(body)
    assert res.passed and res.is_warn, res.render()
    assert "ctx_blk_max@L12" in res.detail


def test_check28_plain_english_sidecar_passes_clean(tmp_path, monkeypatch):
    """A sidecar whose strings are all plain English → clean PASS (no WARN)."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {
            "description": "held-out prediction vs true target",
            "points": [{"label": "house: librarian", "_kind": "scatter"}],
        },
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_label_codes(body)
    assert res.passed and not res.is_warn, res.render()
    assert "free of opaque config codes" in res.detail


def test_check28_translation_map_keys_not_flagged(tmp_path, monkeypatch):
    """Translation-map slug KEYS (`f1_house_librarian` → plain-English value)
    are never visited by the values-only walk → clean PASS. Pins the
    structural fix for the clarifier's key-scan false positive."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {
            "context_id_to_label": {"f1_house_librarian": "house: librarian"},
            "description": "per-context scatter",
        },
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_label_codes(body)
    assert res.passed and not res.is_warn, res.render()


def test_check28_two_segment_metric_names_not_flagged(tmp_path, monkeypatch):
    """2-segment all-alpha snake tokens (`log_prob`, `judge_rate`,
    `helpful_assistant`) are legitimate rendered labels → clean PASS."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {
            "description": "log_prob margin vs judge_rate",
            "points": [{"series": "helpful_assistant"}],
        },
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_label_codes(body)
    assert res.passed and not res.is_warn, res.render()


def test_check28_path_strings_not_flagged(tmp_path, monkeypatch):
    """A path-shaped WORD inside a prose value (`source: figures/…/x.png`) is
    exempt from the snake scan → clean PASS."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {"description": "source: figures/issue_920/winning_cell_scatter.png"},
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_label_codes(body)
    assert res.passed and not res.is_warn, res.render()


def test_check28_spaced_axis_key_scanned(tmp_path, monkeypatch):
    """A dict KEY containing internal whitespace is rendered text (an
    axis-label-keyed data row) and IS scanned → WARN on its `@L` pin."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {"points": [{"ans_uhdr_max@L12 margin": 1.0, "_kind": "scatter"}]},
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_label_codes(body)
    assert res.passed and res.is_warn, res.render()
    assert "ans_uhdr_max@L12" in res.detail


def test_check28_no_sidecar_is_noop_pass(tmp_path, monkeypatch):
    """A same-repo figure with NO `.meta.json` sibling → NO-OP PASS
    (fail-soft; the deliberate contrast with check 26's loud FAIL)."""
    repo, sha = _make_repo_with_figure(tmp_path)  # commits hero.png but no sidecar
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_label_codes(body)
    assert res.passed and not res.is_warn, res.render()
    assert "nothing to scan" in res.detail


def test_check28_repo_unresolved_is_noop_pass(monkeypatch):
    """Offline / repo root unresolved → NO-OP PASS."""
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: None)
    res = verify_task_body.check_figure_label_codes(_CHECK24_BODY)
    assert res.passed and not res.is_warn
    assert "repo root unresolved" in res.detail


def test_check28_opaque_code_tokens_classifier():
    """Pure-function inventories for `_opaque_code_tokens` — CLASSIFIER-SCOPE
    strings only (acceptance criterion 3's partition: walker-scope strings —
    identifier keys, provenance subtrees — are pinned by the walker tests
    `test_check28_translation_map_keys_not_flagged` /
    `test_check28_provenance_subtrees_pruned`, never here)."""
    fn = verify_task_body._opaque_code_tokens
    # Known-bad: every inventory string yields the expected token(s).
    assert "ctx_blk_max@L12" in fn("ctx_blk_max@L12")
    assert fn("ans_uhdr_max") == ["ans_uhdr_max"]
    assert fn("sw_eng_C1") == ["sw_eng_C1"]
    assert fn("BS_E0") == ["BS_E0"]
    assert fn("cond_4") == ["cond_4"]
    assert fn("c1_evil_wrong_em") == ["c1_evil_wrong_em"]
    slash_label = fn("ctx_blk_max / ans_uhdr_max")
    assert "ctx_blk_max" in slash_label and "ans_uhdr_max" in slash_label
    # Known-good: none of these yield any token.
    for good in (
        "house: librarian",
        "true target (leading fold-basis PCA dimension)",
        "wildchat: short 1",
        "log_prob",
        "judge_rate",
        "helpful_assistant",
        "r_B",
        "figures/issue_920/winning_cell_scatter.png",  # path-SHAPED whole string
        "source: figures/issue_920/winning_cell_scatter.png",  # path-shaped word in prose
    ):
        assert fn(good) == [], f"false positive on {good!r}: {fn(good)}"


def test_check28_layer_pin_in_path_word_not_flagged():
    """`@L` pins get the SAME path-shaped exemption snake tokens already get
    (round-2 concern `layer-pin-path-exemption`): a pin-bearing path word in
    prose and a whole path-shaped string are both clean; a slash-SEPARATED
    rendered label (whitespace around the slash) is NOT path-shaped and
    still WARNs both pins."""
    fn = verify_task_body._opaque_code_tokens
    # (a) pin inside a path-shaped word within prose → clean.
    assert fn("source: figures/issue_920/ctx_blk_max@L12.png") == []
    # (b) whole-string path with an embedded pin → clean.
    assert fn("figures/issue_920/ctx_blk_max@L12.png") == []
    # (c) slash-separated rendered label → both pins still flagged.
    toks = fn("ctx_blk_max@L12 / ans_uhdr_max@L12")
    assert "ctx_blk_max@L12" in toks and "ans_uhdr_max@L12" in toks


def test_check28_provenance_subtrees_pruned(tmp_path, monkeypatch):
    """Provenance-keyed subtrees (`script`, `argv` — slug-dense by
    construction) are pruned whole by the walker → clean PASS. Pins the
    single highest-false-positive decision boundary so a later refactor
    cannot silently drop the prune."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {
            "script": "issue920_plot.py",
            "argv": ["--cell", "ctx_blk_max@L12"],
            "description": "held-out prediction vs true target",
        },
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_label_codes(body)
    assert res.passed and not res.is_warn, res.render()


def test_check28_slash_separated_label_warns(tmp_path, monkeypatch):
    """A slash-separated rendered LABEL (`ctx_blk_max / ans_uhdr_max`)
    contains whitespace, so it is NOT path-shaped and IS scanned → WARN
    naming both tokens (the path exemption is path-SHAPED, not any-slash)."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {"description": "ctx_blk_max / ans_uhdr_max"},
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_label_codes(body)
    assert res.passed and res.is_warn, res.render()
    assert "ctx_blk_max" in res.detail and "ans_uhdr_max" in res.detail


def test_check28_slug_in_text_axes_title_warns(tmp_path, monkeypatch):
    """The #1092 defect-(a) regression: a bare cell slug rendered as a PANEL
    TITLE now reaches check 28 through the `meta["text"]` block the current
    `savefig_paper` writes — `_iter_meta_label_values` collects the text
    subtree's string VALUES with zero scan-code change, while the block's
    structural key names (`suptitle`, `axes`, `fig_texts`, …) are
    whitespace-free identifier keys and are never collected."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {
            "created": "2026-07-10T00:00:00Z",
            "text": {
                "suptitle": None,
                "fig_texts": [],
                "axes": [{"title": "ctx_blk_max@L12 vs ans_uhdr_max@L12"}],
            },
        },
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_label_codes(body)
    assert res.passed and res.is_warn, res.render()
    assert "ctx_blk_max@L12" in res.detail


def test_check28_plain_english_text_block_clean(tmp_path, monkeypatch):
    """A plain-English `meta["text"]` block (suptitle / titles / labels /
    legend / series / tick labels) → clean PASS: the new subtree opens no
    false-positive channel on well-labeled figures."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {
            "created": "2026-07-10T00:00:00Z",
            "text": {
                "suptitle": "Alignment holds across seeds",
                "fig_texts": ["Source: evaluation results for task 999"],
                "series": ["trained", "base"],
                "axes": [
                    {
                        "title_left": "Trained vs base agreement",
                        "xlabel": "condition",
                        "ylabel": "agreement rate",
                        "legend_labels": ["trained", "base"],
                        "legend_title": "arm",
                        "xticklabels": ["baseline", "tulu-25"],
                    }
                ],
            },
        },
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_label_codes(body)
    assert res.passed and not res.is_warn, res.render()


# ─── Check 33: bolded what-is-plotted numerics vs sidecar plotted values ───
#
# Fourth sibling of checks 24/26/28: every BOLDED DECIMAL in a figure's
# previous-figure-bounded beat-1 window (prose + caption) must appear among
# the sidecar's plotted values under a rounding / sign / percent /
# sci-notation leniency stack. WARN never FAIL, per-numeric firing,
# `<!-- prose-numerics: derived -->` per-figure opt-out, silent skip on
# missing / truncated sidecars. Incident #825 r1 (task #1107): prose cited
# transfer fractions 0.057/0.109 while the pinned figure plotted 0.231.

_CHECK33_NAME = "figure prose numerics vs figure sidecar (plotted-value drift)"

_CHECK33_PLOTTED_BASE = (
    "Plotted: mean alignment (y, %) per condition (x: baseline, tulu-25), "
    "n=3 seeds per bar, 95% Wald CI error bars."
)


def _check33_body(plotted_line: str) -> str:
    """`_V4_GOOD_BODY` with its beat-1 "Plotted: …" line replaced by
    ``plotted_line`` (which may span multiple lines, e.g. to inject the
    per-figure opt-out comment before the image)."""
    return _V4_GOOD_BODY.replace(_CHECK33_PLOTTED_BASE, plotted_line)


_CHECK33_BODY = _check33_body(
    "Plotted: baseline **0.704** vs tulu-25 **0.879** mean alignment per condition."
)


def _bar_values_sidecar(*heights):
    """A minimal bar sidecar whose plotted values are ``heights`` (string
    categories carry no numeric leaf, so the value pool is exactly
    ``heights``)."""
    return {
        "created": "2026-07-07T00:00:00Z",
        "points": [
            {"category": f"cond{i}", "alignment": h, "_kind": "bar"} for i, h in enumerate(heights)
        ],
        "n_series": 1,
        "total_points": len(heights),
    }


def test_check33_matching_numerics_pass(tmp_path, monkeypatch):
    """(a) Bolded 0.704 / 0.879 both present in the sidecar → clean PASS."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, _bar_values_sidecar(0.704, 0.879))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK33_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()
    assert "all bolded what-is-plotted decimals present" in res.detail


def test_check33_incident825_style_mismatch_warns(tmp_path, monkeypatch):
    """(b) The #825 r1 shape: SOME bolded values match (0.588 / 0.311) while
    the transfer fractions 0.057 / 0.109 match nothing → WARN naming the
    unmatched values (per-numeric firing — none-match-any would false-PASS
    this exact shape). WARN = passed=True + is_warn=True, never FAIL."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path, _bar_values_sidecar(0.588, 0.311, 0.231, -4.53)
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body(
        "Plotted: transfer fraction **0.057** (broad) and **0.109** (narrow); "
        "ceiling **0.588**, floor **0.311**."
    ).replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res.passed and res.is_warn, res.render()
    assert "0.057" in res.detail and "0.109" in res.detail
    assert "nearest" in res.detail and "0.231" in res.detail
    assert "prose-numerics: derived" in res.detail  # the WARN names the opt-out


def test_check33_optout_before_image_skips(tmp_path, monkeypatch):
    """(c) The (b) mismatch + `<!-- prose-numerics: derived -->` in the beat-1
    prose → figure skipped (opted out), no WARN."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path, _bar_values_sidecar(0.588, 0.311, 0.231, -4.53)
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body(
        "Plotted: transfer fraction **0.057** (broad) and **0.109** (narrow).\n\n"
        "<!-- prose-numerics: derived -->"
    ).replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()
    assert "1 opted out" in res.detail


def test_check33_no_sidecar_noop_pass(tmp_path, monkeypatch):
    """(d) A same-repo figure with NO `.meta.json` sibling → silent-skip PASS
    (check-24 convention, NOT check 26's loud missing-sidecar FAIL), even
    with bolded decimals in the window."""
    repo, sha = _make_repo_with_figure(tmp_path)  # commits hero.png but no sidecar
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK33_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()
    assert "no figure with bolded what-is-plotted decimals" in res.detail


def test_check33_printed_precision_rounding_passes(tmp_path, monkeypatch):
    """(e) Prose **0.23** vs plotted 0.2312 → PASS (half-ulp at the PRINTED
    precision: 0.2312 rounds to 0.23 at two decimals)."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, _bar_values_sidecar(0.2312, 0.879))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: mean margin **0.23** per condition.").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()


def test_check33_derived_delta_warns_documented_fp(tmp_path, monkeypatch):
    """(f) DOCUMENTED false-positive class: prose bolds the derived delta
    **0.175** (= 0.879 - 0.704), which is plotted nowhere → WARN. This is the
    known derived-numeric FP class the `<!-- prose-numerics: derived -->`
    opt-out exists for (plan §8 risk row 1); WARN severity + the opt-out are
    the designed containment, not a bug."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, _bar_values_sidecar(0.704, 0.879))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body(
        "Plotted: baseline **0.704** vs tulu-25 **0.879**, a **0.175** lift."
    ).replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res.passed and res.is_warn, res.render()
    assert "0.175" in res.detail


def test_check33_derived_delta_optout_silences(tmp_path, monkeypatch):
    """(f-sibling) The same derived-delta window + the opt-out phrase →
    skipped, no WARN (the documented-FP escape hatch works)."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, _bar_values_sidecar(0.704, 0.879))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body(
        "Plotted: baseline **0.704** vs tulu-25 **0.879**, a **0.175** lift.\n\n"
        "<!-- prose-numerics: derived -->"
    ).replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()
    assert "1 opted out" in res.detail


_CHECK33_V3_BODY = """\
---
title: v3 check-33 fixture
kind: experiment
---
# Some v3 claim (LOW confidence)

<!-- clean-result-v3 -->

## Takeaways

- Something happened.

## Findings

### A finding with a figure

Plotted: transfer fraction **0.057** per condition.

![v3 fig](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_999/hero.png)

> **Figure.** *A v3 figure.*
"""


def test_check33_generation_agnostic_v3_findings_scanned(tmp_path, monkeypatch):
    """The check is generation-agnostic: a v3 body's `## Findings` H3 window
    with a mismatching bolded decimal + a resolvable sidecar → WARN (proves
    the v3 scan path positively, not just vacuously)."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, _bar_values_sidecar(0.231))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK33_V3_BODY.replace("0123456789abcdef", sha)
    assert verify_task_body.is_v3(body) and not verify_task_body.is_v4(body)
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res.passed and res.is_warn, res.render()
    assert "0.057" in res.detail


def test_check33_v3_good_body_vacuous_pass(tmp_path, monkeypatch):
    """(g) The v3 exemplar body (unresolvable placeholder sha, no comparable
    figure) → vacuous PASS, never flagged."""
    repo, _sha = _make_repo_with_figure(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(_V3_GOOD_BODY)
    assert res.passed and not res.is_warn, res.render()


def test_check33_percent_variant_passes(tmp_path, monkeypatch):
    """(h) Prose **87.9%** vs a 0.879 fraction axis → PASS (÷100 percent
    variant)."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, _bar_values_sidecar(0.879))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: tulu-25 hits **87.9%** mean alignment.").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()


def test_check33_unmarked_times100_variant_passes(tmp_path, monkeypatch):
    """(h-sibling) UNMARKED fraction prose **0.879** vs an 87.9 percent axis →
    PASS (x100 variant, available only for %-unmarked decimals)."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, _bar_values_sidecar(87.9))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: tulu-25 hits **0.879** mean alignment.").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()


def _check33_two_figure_body(between_region: str) -> str:
    """A v4 body with TWO figures inside ONE `### <result>` H3: figure 1's
    beat-1 prose bolds 0.704/0.879 (matching its own sidecar in the tests),
    then ``between_region`` (figure 1's interpretation and/or figure 2's
    beat-1 prose), then figure 2."""
    results_block = (
        "## Results\n\n"
        "### One result with a hero and a companion figure\n\n"
        "Plotted: baseline **0.704** vs tulu-25 **0.879** mean alignment.\n\n"
        "![fig one](https://raw.githubusercontent.com/superkaiba/"
        "explore-persona-space/0123456789abcdef/figures/issue_999/hero.png)\n\n"
        "> **Figure.** *Hero.* Error bars 95% Wald CIs.\n\n"
        f"{between_region}\n\n"
        "![fig two](https://raw.githubusercontent.com/superkaiba/"
        "explore-persona-space/0123456789abcdef/figures/issue_999/second.png)\n\n"
        "> **Figure.** *Companion.* Error bars 95% Wald CIs.\n"
    )
    head, _sep, _tail = _V4_GOOD_BODY.partition("## Results\n")
    return head + results_block + "\n---\n**Repro:** 1x H100, 47 min · entry.\n"


def test_check33_two_figures_one_h3_window_bounded(tmp_path, monkeypatch):
    """(i) Two figures in ONE H3: figure 1's bolded 0.704/0.879 match its own
    sidecar; figure 2's window (previous-figure-bounded) carries NO bolds →
    only figure 1 scanned, clean PASS. Pins `_beat1_prose_window`: under the
    check-26 H3→figure window, figure 1's bolds would leak into figure 2's
    window and false-WARN against figure 2's sidecar."""
    body = _check33_two_figure_body("Plotted: capability per condition, bars only.")
    repo, sha = _make_repo_with_two_figure_metas(
        tmp_path, _bar_values_sidecar(0.704, 0.879), _bar_values_sidecar(0.82, 0.81)
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = body.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()
    assert "1 figure(s)" in res.detail


def test_check33_interp_bleed_derived_value_warns_and_optout_silences(tmp_path, monkeypatch):
    """(i-bleed, pinned residual) Figure 1's post-caption INTERPRETATION bolds
    a derived **0.175** absent from BOTH sidecars; that region falls inside
    figure 2's window → WARN against figure 2 (the documented bleed residual:
    beat-3 interpretation and beat-1 prose are structurally indistinguishable
    between two figures). The per-figure opt-out in that region silences it —
    the designed containment for this class."""
    between = (
        "The **0.175** lift is the headline delta.\n\n"
        "Plotted second: capability per condition, bars only."
    )
    body = _check33_two_figure_body(between)
    repo, sha = _make_repo_with_two_figure_metas(
        tmp_path, _bar_values_sidecar(0.704, 0.879), _bar_values_sidecar(0.82, 0.81)
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(
        body.replace("0123456789abcdef", sha)
    )
    assert res.passed and res.is_warn, res.render()
    assert "0.175" in res.detail and "second.png" in res.detail
    # Opt-out in the bleed region silences the WARN (figure 2 opted out).
    body2 = _check33_two_figure_body(between + "\n\n<!-- prose-numerics: derived -->")
    res2 = verify_task_body.check_figure_prose_numerics_vs_sidecar(
        body2.replace("0123456789abcdef", sha)
    )
    assert res2.passed and not res2.is_warn, res2.render()


def test_check33_interp_bleed_prior_figure_value_suppressed(tmp_path, monkeypatch):
    """(i-suppression) Figure 1's post-caption interpretation re-quotes a
    figure-1-PLOTTED value (**0.704**) inside figure 2's window; figure 2's
    sidecar lacks it → SUPPRESSED (matches an earlier same-H3 figure's
    plotted values = cross-figure bleed), no WARN."""
    between = (
        "The baseline **0.704** anchors the comparison.\n\n"
        "Plotted second: capability per condition, bars only."
    )
    body = _check33_two_figure_body(between)
    repo, sha = _make_repo_with_two_figure_metas(
        tmp_path, _bar_values_sidecar(0.704, 0.879), _bar_values_sidecar(0.82, 0.81)
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(
        body.replace("0123456789abcdef", sha)
    )
    assert res.passed and not res.is_warn, res.render()


def test_check33_sign_insensitive_passes(tmp_path, monkeypatch):
    """(j) Prose "a **0.30** drop" vs plotted -0.30 → PASS (sign-insensitive
    twin)."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, _bar_values_sidecar(-0.30))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: a **0.30** drop per condition.").replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()


def test_check33_unicode_minus_passes(tmp_path, monkeypatch):
    """(k) Prose **\u22124.53** (the literal char, unicode minus U+2212) parsed and matched vs
    plotted -4.53 → PASS."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, _bar_values_sidecar(-4.53))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: clipped floor at **−4.53** per condition.").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()


def test_check33_integers_and_versions_not_scanned(tmp_path, monkeypatch):
    """(l) Bolded integers (**n=50**, **3**) and version-shaped tokens
    (**Qwen-2.5-7B**) yield NO scannable decimal → vacuous PASS even against
    a sidecar that matches nothing."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, _bar_values_sidecar(0.1))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body(
        "Plotted: **Qwen-2.5-7B** alignment, **n=50** probes, **3** seeds."
    ).replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()
    assert "no figure with bolded what-is-plotted decimals" in res.detail


def test_check33_data_truncated_sidecar_skips(tmp_path, monkeypatch):
    """(m) A `data_truncated: true` sidecar (the writer's top-level flag) with
    a would-be mismatch → silent skip PASS (a matching value may sit past the
    `_MAX_SIDECAR_ROWS` cap, so absence-of-match is unsound)."""
    meta = _bar_values_sidecar(0.231)
    meta["data_truncated"] = True
    repo, sha = _make_repo_with_figure_meta(tmp_path, meta)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: transfer fraction **0.057** per condition.").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()


def test_check33_truncated_payload_key_skips(tmp_path, monkeypatch):
    """(m-twin) The legacy/payload `truncated: true` key at the meta top level
    ALSO skips (robustness twin of `data_truncated`)."""
    meta = _bar_values_sidecar(0.231)
    meta["truncated"] = True
    repo, sha = _make_repo_with_figure_meta(tmp_path, meta)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: transfer fraction **0.057** per condition.").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()


def test_check33_rows_legacy_key_read(tmp_path, monkeypatch):
    """A sidecar carrying the legacy `rows` key (no `points`) still yields
    plotted values: a mismatching bold → WARN (if `rows` were ignored, the
    empty value pool would silently skip-PASS instead)."""
    meta = {
        "created": "2026-07-07T00:00:00Z",
        "rows": [{"alignment": 0.231, "_kind": "bar"}],
    }
    repo, sha = _make_repo_with_figure_meta(tmp_path, meta)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: transfer fraction **0.057** per condition.").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res.passed and res.is_warn, res.render()
    assert "0.057" in res.detail


def test_check33_sci_notation_through_main_check(tmp_path, monkeypatch):
    """Sci-notation prose **1.23e-3** vs plotted 0.00123 → PASS through the
    MAIN check (the relative-tolerance `dec == -1` branch, not only the
    extraction helper)."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, _bar_values_sidecar(0.00123))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: tail mass **1.23e-3** per condition.").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()


def test_check33_bar_x_position_excluded_from_variants(tmp_path, monkeypatch):
    """A grouped-bar layout x-position (numeric FIRST key of a `_kind: bar`
    row) never satisfies a x100 / /100 VARIANT candidate: prose **0.008** whose
    only would-be match is x100 → the bar x 0.8 → WARN. Direct matching
    against the same value stays lenient: prose **0.8** → PASS."""
    meta = {
        "created": "2026-07-07T00:00:00Z",
        "points": [{"x": 0.8, "margin": 42.0, "_kind": "bar"}],
    }
    repo, sha = _make_repo_with_figure_meta(tmp_path, meta)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: mean margin **0.008** per condition.").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res.passed and res.is_warn, res.render()
    assert "0.008" in res.detail
    body2 = _check33_body("Plotted: dodge offset **0.8** per condition.").replace(
        "0123456789abcdef", sha
    )
    res2 = verify_task_body.check_figure_prose_numerics_vs_sidecar(body2)
    assert res2.passed and not res2.is_warn, res2.render()


def test_check33_prior_figure_pool_excludes_current_figure_bar_x(tmp_path, monkeypatch):
    """(r2 Blocker-1 regression) The bleed pool holds EARLIER same-H3 figures'
    values ONLY: figure 2 is the bar-x fixture (layout offset x 0.8) and its
    window bolds **0.008**, whose only would-be match is the x100 variant →
    the bar x → WARN, exactly as single-figure geometry already pins
    (test_check33_bar_x_position_excluded_from_variants). Under the r1
    aliasing bug (`get` returned the LIVE accumulator; `extend` mutated it
    before matching), figure 2's own 0.8 leaked into `prior_vals` and was
    matched through the pool path WITHOUT the bar-x exclusion → wrong PASS."""
    meta_b = {
        "created": "2026-07-07T00:00:00Z",
        "points": [{"x": 0.8, "margin": 42.0, "_kind": "bar"}],
    }
    body = _check33_two_figure_body("Plotted second: mean margin **0.008** per condition.")
    repo, sha = _make_repo_with_two_figure_metas(
        tmp_path, _bar_values_sidecar(0.704, 0.879), meta_b
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(
        body.replace("0123456789abcdef", sha)
    )
    assert res.passed and res.is_warn, res.render()
    assert "0.008" in res.detail and "second.png" in res.detail


def test_check33_bar_height_colliding_with_x_position_stays_variant_eligible(tmp_path, monkeypatch):
    """(r2 Blocker-2 regression) Identity-preserving exclusion: a bar row
    whose HEIGHT equals its x-position ({"x": 0.8, "margin": 0.8}) keeps the
    height entry variant-eligible — prose **0.008** matches the height via
    x100 → clean PASS. The r1 value-SET exclusion dropped every plotted 0.8
    from the variant candidates → false WARN on common grouped-bar offsets."""
    meta = {
        "created": "2026-07-07T00:00:00Z",
        "points": [{"x": 0.8, "margin": 0.8, "_kind": "bar"}],
    }
    repo, sha = _make_repo_with_figure_meta(tmp_path, meta)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: mean margin **0.008** per condition.").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()


def test_check33_legacy_rows_bar_first_key_stays_variant_eligible(tmp_path, monkeypatch):
    """(r2 Minor-4 regression) The bar-x exclusion is scoped to the modern
    `points` key: a legacy `rows` sidecar's bar row is NEVER bar-x tagged
    (fail-open — the legacy writer's key order is unverified), so prose
    **0.008** still matches the first-key 0.8 via the x100 variant → PASS."""
    meta = {
        "created": "2026-07-07T00:00:00Z",
        "rows": [{"x": 0.8, "margin": 42.0, "_kind": "bar"}],
    }
    repo, sha = _make_repo_with_figure_meta(tmp_path, meta)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: mean margin **0.008** per condition.").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()


def test_check33_optedout_and_boldless_prior_figures_still_feed_bleed_pool(tmp_path, monkeypatch):
    """(r2 Minor-3 semantics pin) DOCUMENTED pool semantics: an earlier
    same-H3 figure contributes its plotted values to the bleed-suppression
    pool regardless of its own scan outcome. Figure 2's window re-quotes
    figure 1's plotted **0.42**, absent from figure 2's own sidecar →
    suppressed (clean PASS) BOTH when figure 1 is opted out AND when it is
    bold-less — the pool models what earlier figures PLOT, independent of
    their prose-scan outcome (excluding them would false-WARN legitimate
    bleed references)."""
    between = (
        "The earlier panel's **0.42** anchors this read.\n\n"
        "Plotted second: capability per condition, bars only."
    )
    body = _check33_two_figure_body(between)
    repo, sha = _make_repo_with_two_figure_metas(
        tmp_path, _bar_values_sidecar(0.42), _bar_values_sidecar(0.82, 0.81)
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    fig1_line = "Plotted: baseline **0.704** vs tulu-25 **0.879** mean alignment."
    # Variant 1: figure 1 OPTED OUT (its window carries the opt-out phrase).
    body_optout = body.replace(
        fig1_line, "Plotted: derived headline quantities.\n\n<!-- prose-numerics: derived -->"
    ).replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(body_optout)
    assert res.passed and not res.is_warn, res.render()
    # Variant 2: figure 1 BOLD-LESS (scanned window, zero bolded decimals).
    body_boldless = body.replace(
        fig1_line, "Plotted: alignment per condition, no headline numerics."
    ).replace("0123456789abcdef", sha)
    res2 = verify_task_body.check_figure_prose_numerics_vs_sidecar(body_boldless)
    assert res2.passed and not res2.is_warn, res2.render()


def test_check33_repo_unresolved_is_noop_pass(monkeypatch):
    """Offline / repo root unresolved → NO-OP PASS."""
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: None)
    res = verify_task_body.check_figure_prose_numerics_vs_sidecar(_CHECK33_BODY)
    assert res.passed and not res.is_warn
    assert "repo root unresolved" in res.detail


def test_check33_beat1_prose_window_boundaries():
    """Helper pin: the window is bounded by the enclosing H3 for the FIRST
    figure and by the PREVIOUS figure's caption end for a LATER same-H3
    figure; None when no `### ` H3 exists above the figure at all."""
    rlines = [
        "### Result one",
        "Beat one text **0.5**.",
        "![f1](https://example.com/f1.png)",
        "",
        "> **Figure.** caption one.",
        "",
        "Interp of figure one.",
        "Plotted two.",
        "![f2](https://example.com/f2.png)",
        "",
        "> caption two.",
    ]
    w1 = verify_task_body._beat1_prose_window(rlines, 2)
    assert w1 is not None and "Beat one" in w1 and "caption one" in w1
    assert "Interp" not in w1
    w2 = verify_task_body._beat1_prose_window(rlines, 8)
    assert w2 is not None and "Interp" in w2 and "Plotted two" in w2
    assert "Beat one" not in w2 and "caption one" not in w2 and "caption two" in w2
    # No H3 anywhere above → None (both directly and via a previous image).
    assert verify_task_body._beat1_prose_window(["prose", "![f](u)"], 1) is None
    assert verify_task_body._beat1_prose_window(["![f0](u)", "text", "![f](u)"], 2) is None


def test_check33_bold_prose_decimals_extraction():
    """Helper pin: decimal-places / percent-marker / sci-notation-sentinel
    extraction; unicode minus normalized; word-attached and integer tokens
    excluded; unbolded decimals never scanned."""
    fn = verify_task_body._bold_prose_decimals
    out = fn(
        "baseline **0.704** vs **87.9%** and **−4.53**, sci **1.2e-3**, "
        "skip **v1.2**, **2.5-7B**, **n=50**; unbolded 0.99 stays out"
    )
    vals = {(v, d, p) for _raw, v, d, p in out}
    assert (0.704, 3, False) in vals
    assert (87.9, 1, True) in vals
    assert (-4.53, 2, False) in vals
    assert (1.2e-3, -1, False) in vals
    assert len(out) == 4, out


# ─── Check 34: beat-phrase series-structure claims vs sidecar rendered text ─
#
# Fifth sibling of checks 24/26/28/33: the two literal #1092 defect-(b)
# phrasings ("both … arms/…", "one bar/… per <unit>") must not contradict the
# series structure the sidecar demonstrably renders. FORWARD-ONLY: fires only
# when the sidecar carries the `meta["text"]` rendered-text block the current
# `savefig_paper` writes; contradiction-only (absence of evidence never
# fires). WARN never FAIL.

_CHECK34_NAME = "figure beat claims vs sidecar rendered text (series-structure drift)"


def _check34_sidecar(*, points=None, series=None, legend_labels=None, n_series=None):
    """A sidecar carrying the `meta["text"]` block (the check-34 forward-only
    gate), with optional `points` rows, fig-global `series` labels, and
    per-axes `legend_labels`."""
    ax_d: dict = {"xlabel": "condition", "ylabel": "agreement rate"}
    if legend_labels:
        ax_d["legend_labels"] = list(legend_labels)
    text: dict = {"suptitle": None, "fig_texts": [], "axes": [ax_d]}
    if series:
        text["series"] = list(series)
    meta: dict = {"created": "2026-07-10T00:00:00Z", "text": text}
    if points is not None:
        meta["points"] = points
    if n_series is not None:
        meta["n_series"] = n_series
    return meta


def test_check34_both_arms_single_series_warns(tmp_path, monkeypatch):
    """(a) "both trained and base models" vs a sidecar whose ONLY available
    basis is a single series label → WARN (passed=True, is_warn=True) naming
    the claim phrase; the WARN never flips the body's overall verdict."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, _check34_sidecar(series=["trained"]))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: mean agreement for both trained and base models.").replace(
        "0123456789abcdef", sha
    )
    _ok, results = verify_task_body.verify_text(body)
    res = _results_by_name(results)[_CHECK34_NAME]
    assert res.passed and res.is_warn, res.render()
    assert "both trained and base models" in res.detail
    assert _CHECK34_NAME not in {r.name for r in results if not r.passed}


def test_check34_both_arms_two_legend_labels_pass(tmp_path, monkeypatch):
    """(b) The same claim vs 2 legend entries → clean PASS (the legend basis
    reads 2, so the claim is satisfiable)."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path, _check34_sidecar(legend_labels=["trained", "base"])
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: mean agreement for both trained and base models.").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_beat_claims_vs_sidecar_text(body)
    assert res.passed and not res.is_warn, res.render()
    assert "consistent" in res.detail


def test_check34_both_arms_two_bar_rows_one_container_pass(tmp_path, monkeypatch):
    """(c) Two BAR ROWS in ONE container (`n_series` = 1, no `_group`) satisfy
    "both arms" → PASS. Pins the n_series-NOT-used decision: a two-arm bar
    pair lives in one `BarContainer`, so an `n_series` basis would read 1 and
    false-fire on the most common two-arm bar chart."""
    pts = [
        {"category": "trained", "rate": 0.7, "_kind": "bar"},
        {"category": "base", "rate": 0.4, "_kind": "bar"},
    ]
    repo, sha = _make_repo_with_figure_meta(tmp_path, _check34_sidecar(points=pts, n_series=1))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: agreement for both trained and base arms.").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_beat_claims_vs_sidecar_text(body)
    assert res.passed and not res.is_warn, res.render()


def test_check34_one_bar_per_single_bar_warns(tmp_path, monkeypatch):
    """(d) The #1092 degenerate class: "one bar per re-fit item" vs a single
    rendered bar row → WARN, with the n=1 acknowledgeable remedy named."""
    pts = [{"category": "only", "rate": 0.7, "_kind": "bar"}]
    repo, sha = _make_repo_with_figure_meta(tmp_path, _check34_sidecar(points=pts))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: one bar per re-fit item.").replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_beat_claims_vs_sidecar_text(body)
    assert res.passed and res.is_warn, res.render()
    assert "one bar per re-fit" in res.detail
    assert "acknowledgeable" in res.detail


def test_check34_one_bar_per_five_bars_pass(tmp_path, monkeypatch):
    """(e) "one bar per source" vs 5 bar rows → clean PASS."""
    pts = [{"category": f"s{i}", "rate": 0.1 * i, "_kind": "bar"} for i in range(5)]
    repo, sha = _make_repo_with_figure_meta(tmp_path, _check34_sidecar(points=pts))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: one bar per source persona.").replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_beat_claims_vs_sidecar_text(body)
    assert res.passed and not res.is_warn, res.render()


def test_check34_one_line_per_single_line_artist_warns(tmp_path, monkeypatch):
    """(f) "one line per seed" vs a SINGLE line artist with many vertex rows →
    WARN. Pins the distinct-`_group` per-ARTIST basis: a line's point rows are
    VERTICES (30 rows here), so a raw row count would read 30 and never fire;
    `_group` is per-ARTIST (per-series), not per-panel, and a single-artist
    sidecar carries no `_group` at all → 1 artist."""
    pts = [{"x": float(i), "y": float(i), "_kind": "line"} for i in range(30)]
    repo, sha = _make_repo_with_figure_meta(tmp_path, _check34_sidecar(points=pts))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: one line per seed.").replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_beat_claims_vs_sidecar_text(body)
    assert res.passed and res.is_warn, res.render()
    assert "1 `line` element(s)" in res.detail


def test_check34_no_text_sidecar_silent_skip(tmp_path, monkeypatch):
    """(g) THE forward-only pin: a claiming prose window + a sidecar WITHOUT
    the `meta["text"]` block (every sidecar committed before the capture
    landed) → NO-OP silent skip, never a WARN and never check 26's loud
    missing-sidecar FAIL — no existing body can retroactively flag."""
    meta = {
        "created": "2026-06-01T00:00:00Z",
        "points": [{"category": "only", "rate": 0.7, "_kind": "bar"}],
    }
    repo, sha = _make_repo_with_figure_meta(tmp_path, meta)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: one bar per re-fit item.").replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_beat_claims_vs_sidecar_text(body)
    assert res.passed and not res.is_warn, res.render()
    assert "nothing to compare" in res.detail


def test_check34_missing_sidecar_silent_skip(tmp_path, monkeypatch):
    """(h) A same-repo figure with NO `.meta.json` sibling → silent-skip PASS
    (check-24 fail-soft convention), even with a claiming window."""
    repo, sha = _make_repo_with_figure(tmp_path)  # commits hero.png but no sidecar
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: one bar per re-fit item.").replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_beat_claims_vs_sidecar_text(body)
    assert res.passed and not res.is_warn, res.render()
    assert "nothing to compare" in res.detail


def test_check34_repo_unresolved_is_noop_pass(monkeypatch):
    """(i) Offline / repo root unresolved → NO-OP PASS."""
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: None)
    body = _check33_body("Plotted: one bar per re-fit item.")
    res = verify_task_body.check_figure_beat_claims_vs_sidecar_text(body)
    assert res.passed and not res.is_warn
    assert "repo root unresolved" in res.detail


def test_check34_no_claim_in_prose_pass(tmp_path, monkeypatch):
    """(j) A window with NO registered claim phrase → "nothing to compare"
    PASS even against a text-bearing sidecar (never over-fire)."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, _check34_sidecar(series=["trained"]))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK33_BODY.replace("0123456789abcdef", sha)  # bolded decimals, no beat claim
    res = verify_task_body.check_figure_beat_claims_vs_sidecar_text(body)
    assert res.passed and not res.is_warn, res.render()
    assert "nothing to compare" in res.detail


def test_check34_both_arms_unlabeled_scatter_silent_skip(tmp_path, monkeypatch):
    """(k) The one real FP channel, closed by basis AVAILABILITY: "both arms"
    over an UNLABELED single-artist scatter (no series, no legend, no bar/line
    rows, no `_group` anywhere — a lone scatter artist can encode two arms via
    per-point colors the extractor cannot see) → NO available basis → skip,
    NOT a contradiction."""
    pts = [{"x": 0.1 * i, "y": 0.2 * i, "_kind": "scatter"} for i in range(6)]
    repo, sha = _make_repo_with_figure_meta(tmp_path, _check34_sidecar(points=pts))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: agreement for both input arms.").replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_beat_claims_vs_sidecar_text(body)
    assert res.passed and not res.is_warn, res.render()


def test_check34_truncated_sidecar_class_b_silent_skip(tmp_path, monkeypatch):
    """(m) Regression, r1 Major `check34-truncated-sidecar-false-warn`: a
    `data_truncated` sidecar carries NO points-derived basis — the writer
    truncates concatenated rows HEAD-FIRST, so a first artist with >= the cap
    drops all LATER artists/kinds from the stored payload. The r1 demo shape
    (two LABELED line artists; stored rows all `_group` 0 because artist 2 was
    truncated away) + "one line per seed" fired a false Class-B WARN pre-fix;
    post-fix Class B silently skips (stored rows are not figure truth)."""
    pts = [{"x": float(i), "y": float(i), "_kind": "line", "_group": 0} for i in range(12)]
    meta = _check34_sidecar(points=pts, series=["seed 0", "seed 1"])
    meta["data_truncated"] = True
    repo, sha = _make_repo_with_figure_meta(tmp_path, meta)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: one line per seed.").replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_beat_claims_vs_sidecar_text(body)
    assert res.passed and not res.is_warn, res.render()


def test_check34_truncated_sidecar_class_a_text_basis_pass(tmp_path, monkeypatch):
    """(n) Companion to (m): the TEXT bases are truncation-IMMUNE (rendered
    text is captured separately from the points payload) — the same truncated
    sidecar WITH 2 legend labels + a Class-A claim still scans and PASSes via
    the legend basis; truncation removes only the points-derived bases."""
    pts = [{"x": float(i), "y": float(i), "_kind": "line", "_group": 0} for i in range(12)]
    meta = _check34_sidecar(points=pts, legend_labels=["trained", "base"])
    meta["data_truncated"] = True
    repo, sha = _make_repo_with_figure_meta(tmp_path, meta)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check33_body("Plotted: mean agreement for both trained and base models.").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_beat_claims_vs_sidecar_text(body)
    assert res.passed and not res.is_warn, res.render()
    assert "consistent" in res.detail


def test_check34_text_and_points_leave_checks26_33_unchanged(tmp_path, monkeypatch):
    """(l) A sidecar carrying BOTH `text` and `points` changes nothing for the
    sibling checks: check 33 still matches its bolded decimals against the
    `points` values (clean PASS), and check 26 still finds no panel/series
    prose claim (NO-OP PASS) — the `text` key is invisible to both by
    construction (check 26/33 read `points`/`rows` only)."""
    meta = _bar_values_sidecar(0.704, 0.879)
    meta["text"] = {
        "suptitle": "Alignment per condition",
        "fig_texts": [],
        "axes": [{"xlabel": "condition", "ylabel": "alignment"}],
    }
    repo, sha = _make_repo_with_figure_meta(tmp_path, meta)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK33_BODY.replace("0123456789abcdef", sha)
    res33 = verify_task_body.check_figure_prose_numerics_vs_sidecar(body)
    assert res33.passed and not res33.is_warn, res33.render()
    assert "all bolded what-is-plotted decimals present" in res33.detail
    res26 = verify_task_body.check_figure_panel_prose_vs_sidecar(body)
    assert res26.passed and not res26.is_warn, res26.render()
    assert "no panel/series prose claims" in res26.detail


def test_check34_beat_series_claims_inventory():
    """Pure-parser inventory for `_beat_series_claims`: the two registered
    claim classes match their literal #1092 phrasings; paraphrases miss BY
    DESIGN (the documented false-negative envelope)."""
    fn = verify_task_body._beat_series_claims
    # Class A — matches.
    assert fn("shows both input arms")["both"] == ["both input arms"]
    assert fn("both fine-tuned and base models are shown")["both"] == [
        "both fine-tuned and base models"
    ]
    assert fn("both trained and base conditions")["both"]
    assert fn("Both curves overlap")["both"]  # case-insensitive
    # Class B — matches, kind-mapped, de-duplicated.
    one = fn("one bar per re-fit item; one bar per re-fit item")["one_per"]
    assert one == [("one bar per re-fit", "bar")]
    assert fn("one line per seed")["one_per"] == [("one line per seed", "line")]
    assert fn("one curve per layer")["one_per"][0][1] == "line"
    assert fn("one dot per persona")["one_per"][0][1] == "scatter"
    assert fn("one point per context")["one_per"][0][1] == "scatter"
    assert fn("one marker per cell")["one_per"][0][1] == "scatter"
    # Non-matches (paraphrases / other nouns) — the FN envelope.
    for miss in (
        "each arm gets its own panel",
        "two models are compared",
        "both of these approaches work",  # 'approaches' is not a registered noun
        "bars per source",  # no leading 'one'
        "one bar for each source",  # 'for each', not 'per'
    ):
        got = fn(miss)
        assert not got["both"] and not got["one_per"], (miss, got)


def test_check34_beat_claim_warnings_comparator():
    """Pure-comparator inventory for `_beat_claim_warnings`, pinning the
    per-ARTIST (not per-panel) `_group` semantics and the basis-availability
    rules. `_group` indexes artist GROUPS (`_build_sidecar_data`: one index
    per extracted artist, emitted only on multi-artist figures) — a 4-panel
    figure with 22 artists has 22 groups, so no basis here ever reads panel
    counts."""
    fn = verify_task_body._beat_claim_warnings
    both = {"both": ["both arms"], "one_per": []}

    def _meta(points=None, text=None):
        m: dict = {"created": "t"}
        if points is not None:
            m["points"] = points
        m["text"] = text if text is not None else {"suptitle": None, "fig_texts": []}
        return m

    # Two line ARTISTS (distinct _group) satisfy "both" — no warn.
    two_lines = [
        {"x": 0.0, "y": 0.0, "_kind": "line", "_group": 0},
        {"x": 1.0, "y": 1.0, "_kind": "line", "_group": 0},
        {"x": 0.0, "y": 0.5, "_kind": "line", "_group": 1},
        {"x": 1.0, "y": 1.5, "_kind": "line", "_group": 1},
    ]
    assert fn(both, _meta(points=two_lines), "f.png") == []
    # ONE line artist (vertices, single-artist sidecar → no _group) → warn.
    one_line = [{"x": float(i), "y": float(i), "_kind": "line"} for i in range(10)]
    assert len(fn(both, _meta(points=one_line), "f.png")) == 1
    # Fig-GLOBAL series labels: a multi-panel figure with one series per panel
    # and >=2 distinct labels satisfies "both arms" — DELIBERATELY
    # conservative (never false-fire on per-panel single-series layouts).
    text_two_series = {"suptitle": None, "fig_texts": [], "series": ["a", "b"]}
    assert fn(both, _meta(text=text_two_series), "f.png") == []
    # Two scatter ARTISTS (distinct _group) → no warn; a single unlabeled
    # scatter artist (no _group) yields NO basis → no warn either (skip).
    two_scatter = [
        {"x": 0.1, "y": 0.2, "_kind": "scatter", "_group": 0},
        {"x": 0.3, "y": 0.4, "_kind": "scatter", "_group": 1},
    ]
    assert fn(both, _meta(points=two_scatter), "f.png") == []
    lone_scatter = [{"x": 0.1, "y": 0.2, "_kind": "scatter"} for _ in range(5)]
    assert fn(both, _meta(points=lone_scatter), "f.png") == []
    # Mixed line+scatter TWO-artist figure: the total artist-groups basis
    # reads 2 → "both arms" satisfiable across kinds → no warn.
    mixed = [
        {"x": 0.1, "y": 0.2, "_kind": "scatter", "_group": 0},
        {"x": 0.0, "y": 0.0, "_kind": "line", "_group": 1},
        {"x": 1.0, "y": 1.0, "_kind": "line", "_group": 1},
    ]
    assert fn(both, _meta(points=mixed), "f.png") == []
    # A lone data-contributing scatter artist in a MULTI-artist figure (a
    # rowless sibling left `_group` on the rows): the 1-value group set is
    # treated as ABSENT group evidence — same skip as the no-`_group` lone
    # scatter above (the artist may encode >=2 arms per-point). r1 Minor.
    lone_group_scatter = [
        {"x": 0.1 * i, "y": 0.2 * i, "_kind": "scatter", "_group": 0} for i in range(5)
    ]
    assert fn(both, _meta(points=lone_group_scatter), "f.png") == []
    # Class B without a points payload → skip (basis unavailable).
    one_per = {"both": [], "one_per": [("one bar per source", "bar")]}
    assert fn(one_per, _meta(text=text_two_series), "f.png") == []
    # Class B: claimed kind entirely absent (0 bar rows) → warn.
    assert len(fn(one_per, _meta(points=lone_scatter), "f.png")) == 1

    # TRUNCATED sidecars (r1 Major): stored rows are head-truncated, so they
    # are NOT figure truth — every points-derived basis is unavailable.
    def _trunc(m):
        m["data_truncated"] = True
        return m

    # Class B on a truncated sidecar → skip (both the <=1-artist read AND the
    # zeroed-out claimed kind: the missing bar rows may sit past the cap).
    one_line_grouped = [
        {"x": float(i), "y": float(i), "_kind": "line", "_group": 0} for i in range(10)
    ]
    line_per = {"both": [], "one_per": [("one line per seed", "line")]}
    assert fn(line_per, _trunc(_meta(points=one_line_grouped)), "f.png") == []
    assert fn(one_per, _trunc(_meta(points=one_line_grouped)), "f.png") == []
    # Class A on a truncated sidecar: points-derived bases gone, but the TEXT
    # bases are truncation-immune and still FIRE — a single fig-global series
    # label with "both arms" claimed warns exactly as it does untruncated.
    text_one_series = {"suptitle": None, "fig_texts": [], "series": ["a"]}
    assert len(fn(both, _trunc(_meta(points=one_line_grouped, text=text_one_series)), "f.png")) == 1
    # ... and with no text basis at all, a truncated Class A yields NO basis → skip.
    assert fn(both, _trunc(_meta(points=one_line_grouped)), "f.png") == []


#
# A NEW mechanical check that FAILs/WARNs when a clean-result body states a
# BARE LLM-judge denominator (`n=N` / "N completions/EM") in a judge-context
# section WITHOUT disclosing the judge-API-error fraction, while the committed
# `eval_results/issue_<N>/` JSONs show a non-trivial fraction of rows returned
# Anthropic Batch API 529-overload errors that were silently counted into the
# denominator. It PASSes when (a) the body discloses, (b) no recognized
# judge-error signal exists (graceful skip), or (c) no bare judge denominator
# is asserted. The eval root is resolved through a ladder:
#   (i) explicit --eval-root / eval_root= arg (gate-time worktree path),
#   (ii) --file-derived worktree root,
#   (iii) cwd `git rev-parse --show-toplevel`,
#   (iv) _resolve_repo_root() (MAIN — bottom-of-ladder, post-merge bind),
#   graceful PASS if all miss.
# Demonstrated bug class: /issue 715 R1 (882 `529 Overloaded` rows across 48 EM
# cells, 32.5% worst-cell, 4.59% pooled) silently counted into a bare n=400 EM
# denominator. Plan: tasks/approved/732/plans/plan.md §1/§3/§4/§6.

_CHECK732_NAME = "judge-API-error denominator disclosed"

# Synthetic v4 body that TRIGGERS the check: a bare `n=400` EM judge
# denominator in a judge-context (`## Methodology`/`## Results`) section, with
# NO disclosure phrase anywhere (no 529 / Overloaded / n_judge_error / "excluded
# from the denominator" / "post-correction" ...). Built from _V4_GOOD_BODY so it
# is a valid v4 body (passes is_v4), then the Methodology `**Evaluation:**` line
# is rewritten to assert the bare denominator.
_CHECK732_UNDISCLOSED_BODY = _V4_GOOD_BODY.replace(
    "- **Evaluation:** Betley alignment score, Claude Sonnet judge, 200 probes; "
    "chosen to match the prior eval surface; no preprocessing.",
    "- **Evaluation:** Betley emergent-misalignment rate, `claude-sonnet-4-5` "
    "judge over 8 questions x 50 completions each (n=400 EM judgments per cell); "
    "EM rate is `n_misaligned / 400` per cell.",
)


def _make_corrected_pareto_eval_tree(
    root: Path, issue: int, *, worst_err: int, worst_att: int, other_cells: int
):
    """Write a synthetic #715-shaped `pareto_*_corrected.json` under
    `root/eval_results/issue_<N>/`, returning the eval dir.

    The leaf cells carry `n_em_judge_error` (judge-error count) + `n_em_attempted`
    (the denominator) under a nested `cells` dict — matching the real
    `pareto_em_vs_narrow_corrected.json` shape (source 1). `worst_err/worst_att`
    sets the worst cell; `other_cells` clean cells (0 judge errors) pad the
    pool so pooled and worst-cell fractions can diverge.
    """
    eval_dir = root / "eval_results" / f"issue_{issue}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    leaves = [{"step": 0, "n_em_judge_error": worst_err, "n_em_attempted": worst_att}]
    leaves += [
        {"step": i + 1, "n_em_judge_error": 0, "n_em_attempted": worst_att}
        for i in range(other_cells)
    ]
    total_err = sum(c["n_em_judge_error"] for c in leaves)
    payload = {
        "cells": {"sft_lora": {"seed42": leaves}},
        "judge_error_correction": {"judge_error_totals": {"sweep_total": total_err}},
        "metadata": {"issue": issue},
    }
    (eval_dir / "pareto_em_vs_narrow_corrected.json").write_text(json.dumps(payload))
    return eval_dir


def test_judge_error_denominator_h2_published_715_passes_via_disclosure():
    """H2 (known-good → PASS via DISCLOSURE): the published #715 body discloses
    the 529 / Overloaded / `excluded from the denominator` / `400 - n_judge_error`
    phrasing in `## Methodology` AND the `**Repro:**` footer, so the check
    short-circuits to PASS before any eval read.  Addresses §1 Goal H2.

    Reads the REAL body via the resolved task path (issue 715 on MAIN); the
    check is called directly with issue=715 (the disclosure short-circuit makes
    the eval-root resolution irrelevant)."""
    try:
        from explore_persona_space.task_workflow import find_task_path

        body_path = find_task_path(715) / "body.md"
        if not body_path.exists():
            pytest.skip("published #715 body absent")
        body = body_path.read_text()
    except Exception:
        pytest.skip("could not resolve published #715 body")
    if not verify_task_body.is_v4(body):
        pytest.skip("#715 body is not v4 (migrated away from the disclosure fixture)")
    res = verify_task_body.check_judge_error_denominator(body, issue=715)
    assert res.passed and not res.is_warn, res.render()
    assert "disclos" in res.detail.lower(), res.render()


def test_judge_error_denominator_h1_synthetic_monkeypatched_fails(tmp_path, monkeypatch):
    """H1 FAIL via the LEGACY code path: `_resolve_repo_root` monkeypatched to a
    `tmp_path` root carrying a synthetic `eval_results/issue_<N>/` corrected JSON
    (worst-cell 130/400 = 32.5%). Regression cover for the bottom-of-ladder
    MAIN-post-merge fallback leg (iv).  Addresses §3 H1 (legacy code path)."""
    _make_corrected_pareto_eval_tree(tmp_path, 999, worst_err=130, worst_att=400, other_cells=47)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: tmp_path)
    res = verify_task_body.check_judge_error_denominator(_CHECK732_UNDISCLOSED_BODY, issue=999)
    assert not res.passed, res.render()
    assert "32.5%" in res.detail or "0.32" in res.detail or "worst" in res.detail.lower()


def test_judge_error_denominator_h1_eval_root_fails(tmp_path):
    """H1 FAIL via the PRODUCTION ladder leg (i): pass `eval_root=tmp_path`
    explicitly (NO `_resolve_repo_root` monkeypatch) over a synthetic
    `eval_results/issue_<N>/` corrected JSON with a 32.5% worst cell → FAIL.
    Proves the gate-time `--eval-root` path is actually plumbed.
    Addresses Must-Fix item 1 (production ladder leg i)."""
    _make_corrected_pareto_eval_tree(tmp_path, 999, worst_err=130, worst_att=400, other_cells=47)
    res = verify_task_body.check_judge_error_denominator(
        _CHECK732_UNDISCLOSED_BODY, issue=999, eval_root=tmp_path
    )
    assert not res.passed, res.render()


def test_judge_error_denominator_h1_cwd_resolution_fails(tmp_path, monkeypatch):
    """H1 FAIL via the PRODUCTION ladder leg (iii): `git init` a `tmp_path` repo,
    write a synthetic `eval_results/issue_<N>/` corrected JSON, `chdir` into it
    so `git rev-parse --show-toplevel` resolves the eval root — NO
    `_resolve_repo_root` monkeypatch, NO explicit `--eval-root`. Proves the
    cwd-based fallback reaches a worktree cwd.
    Addresses Must-Fix item 1 (production ladder leg iii)."""
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, check=True)
    _make_corrected_pareto_eval_tree(tmp_path, 999, worst_err=130, worst_att=400, other_cells=47)
    monkeypatch.chdir(tmp_path)
    # Belt-and-suspenders: make sure the MAIN fallback (leg iv) cannot resolve a
    # real eval_results/issue_999 — force it to None so the FAIL can only come
    # from the cwd leg under test.
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: None)
    res = verify_task_body.check_judge_error_denominator(_CHECK732_UNDISCLOSED_BODY, issue=999)
    assert not res.passed, res.render()


def test_judge_error_denominator_integration_715_real_data():
    """Real-data integration: run the eval scan over the REAL committed
    `pareto_em_vs_narrow_corrected.json` (#715), resolved via `--eval-root`
    pointed at the issue-715 worktree, and assert the helper recovers the
    ground-truth fractions: worst_frac ≈ 0.325, pooled ≈ 0.0459, total_err == 882,
    n_cells == 48.  Addresses the Methodology concern (non-blocking).

    Skips if the fixture is absent (sparse-excluded worktree / post-merge
    relocation)."""
    try:
        from explore_persona_space.task_workflow import repo_root

        main_root = repo_root()
    except Exception:
        main_root = Path(__file__).resolve().parents[1]
    candidates = [
        main_root / ".claude" / "worktrees" / "issue-715",  # pre-merge worktree
        main_root,  # post-merge: eval_results/issue_715 on MAIN
    ]
    eval_root = next(
        (c for c in candidates if (c / "eval_results" / "issue_715").is_dir()),
        None,
    )
    if eval_root is None:
        pytest.skip("real-data fixture absent; sparse-excluded worktree or post-merge relocation")
    stats = verify_task_body._scan_issue_judge_errors(eval_root, 715)
    assert stats is not None, "scan returned None on the real corrected JSON"
    assert stats["total_err"] == 882, stats
    assert stats["n_cells"] == 48, stats
    assert abs(stats["worst_frac"] - 0.325) < 1e-3, stats
    pooled = stats["total_err"] / max(stats["total_att"], 1)
    assert abs(pooled - 0.0459) < 1e-3, (pooled, stats)


def test_judge_error_denominator_no_trigger_training_rows_passes(tmp_path):
    """No-trigger PASS: a body whose only large `n`-like count is a TRAINING-ROW
    count ("6349 training rows") with no judge-context judge-noun co-occurrence
    asserts NO judge denominator → PASS (no eval read, no FAIL on a real error
    fraction). Addresses the Statistics concern (training-row false-trigger)."""
    # _V4_GOOD_BODY's Methodology already says "2,000 rows" with no judge-context
    # denominator; add an explicit training-row count to make the no-trigger case
    # unambiguous, and confirm it does not fire even with a real eval tree present.
    body = _V4_GOOD_BODY.replace(
        "- **Design:** 3 seeds; baseline vs tulu-25 on benchmark Z. "
        "The single manipulated variable is the data mix.",
        "- **Design:** 3 seeds; baseline vs tulu-25 on benchmark Z, 6349 training "
        "rows. The single manipulated variable is the data mix.",
    )
    _make_corrected_pareto_eval_tree(tmp_path, 999, worst_err=130, worst_att=400, other_cells=47)
    res = verify_task_body.check_judge_error_denominator(body, issue=999, eval_root=tmp_path)
    assert res.passed and not res.is_warn, res.render()


def test_judge_error_denominator_no_signal_graceful_pass(tmp_path):
    """No-signal graceful PASS: a TRIGGERING body (bare n=400 EM denominator, no
    disclosure) over an eval dir whose only count field is `breakdown.n_parse_error`
    (the DISTINCT parse-error class, NOT the 529 API-error class) → PASS with a
    graceful-skip note. `n_parse_error` must NOT be treated as a judge-API error.
    Addresses Alternatives concern #3 (older-issue / no-signal layouts)."""
    eval_dir = tmp_path / "eval_results" / "issue_999"
    eval_dir.mkdir(parents=True)
    # em_rate-style per-cell aggregate: n_total + breakdown.n_parse_error only.
    (eval_dir / "dft_lora_seed42_step329.json").write_text(
        json.dumps({"n_total": 400, "breakdown": {"n_parse_error": 0}})
    )
    res = verify_task_body.check_judge_error_denominator(
        _CHECK732_UNDISCLOSED_BODY, issue=999, eval_root=tmp_path
    )
    assert res.passed and not res.is_warn, res.render()
    assert "graceful" in res.detail.lower() or "no judge-error" in res.detail.lower()


def test_judge_error_denominator_warn_band(tmp_path):
    """WARN band: a TRIGGERING body over an eval dir whose judge-error fraction
    sits in (1%, 5%] (both worst-cell and pooled) → WARN (passes overall, flagged).
    Addresses §11 Source 2 (the >1% WARN threshold)."""
    # 8/400 = 2% per cell, identical across cells → worst == pooled == 2%.
    _make_corrected_pareto_eval_tree(tmp_path, 999, worst_err=8, worst_att=400, other_cells=2)
    # The padding cells above have 0 errors, which would drop pooled below the
    # worst cell; rebuild so every cell carries 2% to keep BOTH in (1%, 5%].
    eval_dir = tmp_path / "eval_results" / "issue_999"
    leaves = [{"step": i, "n_em_judge_error": 8, "n_em_attempted": 400} for i in range(3)]
    (eval_dir / "pareto_em_vs_narrow_corrected.json").write_text(
        json.dumps(
            {
                "cells": {"sft_lora": {"seed42": leaves}},
                "judge_error_correction": {"judge_error_totals": {"sweep_total": 24}},
            }
        )
    )
    res = verify_task_body.check_judge_error_denominator(
        _CHECK732_UNDISCLOSED_BODY, issue=999, eval_root=tmp_path
    )
    assert res.passed and res.is_warn, res.render()


def test_judge_error_denominator_sibling_issue_graceful_pass(tmp_path):
    """Sibling-issue graceful PASS: a TRIGGERING body over an eval dir whose JSON
    carries NO recognized judge-error count key at all (older-issue layout, e.g.
    #608/#545) → PASS graceful-skip, never a false FAIL.  Addresses §A6."""
    eval_dir = tmp_path / "eval_results" / "issue_608"
    eval_dir.mkdir(parents=True)
    # An older-layout aggregate: a denominator but no recognized judge-error key.
    (eval_dir / "sycophancy_rate.json").write_text(
        json.dumps({"n_total": 500, "rate": 0.42, "model": "qwen"})
    )
    res = verify_task_body.check_judge_error_denominator(
        _CHECK732_UNDISCLOSED_BODY, issue=608, eval_root=tmp_path
    )
    assert res.passed and not res.is_warn, res.render()


# ─── Check 35 (#1256): cross-issue reuse pins declared in the body ─────────
#
# Signal (two tiers over committed `eval_results/issue_<N>/**/*.json`
# metadata): tier 1 (FAIL) = `hf_rev_<M>[_<tag>]` keys, M != N, satisfied by
# a >=7-hex-char body token prefixing the pinned revision (non-hex branch/tag
# pins fall back to a `#M` / `/tasks/M` / `issue<M>_` mention); tier 2 (WARN)
# = `\bissue<M>_` tokens in `metadata.input_shas` keys/values + PATH-LIKE
# `metadata.args` string values. Grounding (corpus scan 2026-07-11): the
# tier-1 key shape exists in exactly 1 file among ~90,858 committed eval
# JSONs (the #1092 incident file); the bare `issue<M>_` pattern appears in
# >=10,028 files — hence the tier-2 restriction + WARN severity.
# Demonstrated bug class: #1092 round 3 (`hf_rev_779_labels` pinned in
# `transfer_reads.json` metadata, reuse undeclared until the LM critic).

_CHECK1256_NAME = "cross-issue reuse pins declared (footer Reused bullets)"

# Synthetic 40-hex revision pin. Must NOT share a >=7-char prefix with
# `_V4_GOOD_BODY`'s own footer hex tokens (`0123456789abcdef` / `abc123def`),
# or the base fixture would satisfy the pin by accident.
_CHECK1256_SYN_SHA = "deadbeefcafe4d4d9c00112233445566778899aa"

# The base v4 fixture carries NO `#779` / `/tasks/779` / `issue779_` mention
# and no hex token prefixing `_CHECK1256_SYN_SHA` — it is the undeclared body.
_CHECK1256_UNDECLARED_BODY = _V4_GOOD_BODY

# Declared body whose ONLY satisfying token is the 8-char short sha
# `deadbeef` (a strict PREFIX of the 40-char pin) — deliberately NO `#779`
# and NO `issue779_` string, so a PASS proves the hex-prefix predicate
# specifically. Shaped like #1092's real inline `Reused: ... [@ <short>]`
# footer form (NOT the SPEC dash-bullet shape — the predicate must not
# require the literal bullet).
_CHECK1256_DECLARED_BODY = _V4_GOOD_BODY + (
    "\nReused: r_B trait directions [@ deadbeef]"
    "(https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/"
    "tree/deadbeef/rb) — fit: same base model + recipe.\n"
)


def _make_cross_issue_pin_eval_tree(
    root: Path,
    issue: int,
    *,
    pins: dict | None = None,
    input_shas: dict | None = None,
):
    """Write a synthetic `transfer_reads.json`-shaped result JSON under
    `root/eval_results/issue_<N>/` whose `metadata.args` carries `pins`
    (e.g. ``{"hf_rev_779_labels": "<40-hex>"}``) merged over a self-path
    `out` arg, plus an optional `metadata.input_shas` dict — the observed
    #1092 incident shape. Returns the eval dir."""
    eval_dir = root / "eval_results" / f"issue_{issue}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    args = {"out": f"eval_results/issue_{issue}/probe/", **(pins or {})}
    metadata = {"script": "scripts/synthetic_probe.py", "args": args}
    if input_shas is not None:
        metadata["input_shas"] = input_shas
    payload = {"verdict": {"ok": True}, "metadata": metadata}
    (eval_dir / "transfer_reads.json").write_text(json.dumps(payload))
    return eval_dir


def test_cross_issue_reuse_provenance_undeclared_pin_fails(tmp_path):
    """(a) MAIN / durability-pin test (SPEC.md § **Artifacts:** mechanical
    cross-check sentence, #1256): a v4 body over an eval tree whose
    `metadata.args` carries `hf_rev_779_labels: <40-hex>` (M=779 != N=999)
    with the revision nowhere in the body → FAIL naming the source issue,
    the metadata key, and the JSON relpath."""
    _make_cross_issue_pin_eval_tree(tmp_path, 999, pins={"hf_rev_779_labels": _CHECK1256_SYN_SHA})
    res = verify_task_body.check_cross_issue_reuse_provenance(
        _CHECK1256_UNDECLARED_BODY, issue=999, eval_root=tmp_path
    )
    assert not res.passed, res.render()
    assert "#779" in res.detail, res.render()
    assert "hf_rev_779_labels" in res.detail, res.render()
    assert "eval_results/issue_999/transfer_reads.json" in res.detail, res.render()
    assert "Reused" in res.detail, res.render()  # expected declaration shape named


def test_cross_issue_reuse_provenance_declared_short_sha_passes(tmp_path):
    """(b) Prefix predicate: same triggering tree; the body declares the
    reuse with ONLY an 8-char short sha (`deadbeef`) prefixing the 40-char
    metadata pin (no `#779`, no `issue779_`) → clean PASS."""
    _make_cross_issue_pin_eval_tree(tmp_path, 999, pins={"hf_rev_779_labels": _CHECK1256_SYN_SHA})
    res = verify_task_body.check_cross_issue_reuse_provenance(
        _CHECK1256_DECLARED_BODY, issue=999, eval_root=tmp_path
    )
    assert res.passed and not res.is_warn, res.render()
    assert "declared" in res.detail, res.render()


def test_cross_issue_reuse_provenance_no_pins_graceful_pass(tmp_path):
    """(c) Graceful skip: metadata with no `hf_rev_*` keys and no
    `issue<M>_` input paths → PASS with the graceful-skip detail (distinct
    from the env-fence skip detail)."""
    _make_cross_issue_pin_eval_tree(tmp_path, 999, pins={})
    res = verify_task_body.check_cross_issue_reuse_provenance(
        _CHECK1256_UNDECLARED_BODY, issue=999, eval_root=tmp_path
    )
    assert res.passed and not res.is_warn, res.render()
    assert "graceful skip" in res.detail, res.render()
    assert "EPM_VERIFY_BODY_NO_EVAL_SCAN" not in res.detail, res.render()


def test_cross_issue_reuse_provenance_self_pin_never_flags(tmp_path):
    """(d) Self-pin exemption: `hf_rev_<N>` with M == N (the observed
    `hf_rev_1092` self-pin) is provenance, not reuse → never flags →
    graceful-skip PASS."""
    _make_cross_issue_pin_eval_tree(tmp_path, 999, pins={"hf_rev_999": _CHECK1256_SYN_SHA})
    res = verify_task_body.check_cross_issue_reuse_provenance(
        _CHECK1256_UNDECLARED_BODY, issue=999, eval_root=tmp_path
    )
    assert res.passed and not res.is_warn, res.render()
    assert "graceful skip" in res.detail, res.render()


def test_cross_issue_reuse_provenance_corrupt_json_skipped(tmp_path, monkeypatch):
    """(e) Robustness ladder: a corrupt `.json` that PASSES the substring
    pre-filter (so the `json.loads` skip path actually runs) + a clean
    pin-free JSON → PASS, no crash. Dir-absent scanner branch → None.
    `issue=None` → PASS-skip. Unresolvable root (no legs) → PASS
    `is_warn=True` (the judge-error L4889 convention)."""
    eval_dir = tmp_path / "eval_results" / "issue_999"
    eval_dir.mkdir(parents=True)
    (eval_dir / "corrupt.json").write_text('{"metadata": {"args": {"hf_rev_779_labels": ')
    (eval_dir / "clean.json").write_text(json.dumps({"metadata": {"args": {"out": "x/y"}}}))
    res = verify_task_body.check_cross_issue_reuse_provenance(
        _CHECK1256_UNDECLARED_BODY, issue=999, eval_root=tmp_path
    )
    assert res.passed and not res.is_warn, res.render()
    assert "graceful skip" in res.detail, res.render()

    # Dir-absent scanner branch: an eval root with NO eval_results/issue_<N>.
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    assert verify_task_body._scan_cross_issue_reuse_pins(empty_root, 999) is None

    # issue=None (stdin) → PASS-skip before any disk read.
    res_none = verify_task_body.check_cross_issue_reuse_provenance(
        _CHECK1256_UNDECLARED_BODY, issue=None
    )
    assert res_none.passed and not res_none.is_warn, res_none.render()
    assert "issue number unknown" in res_none.detail, res_none.render()

    # Unresolvable root: no eval_root, cwd not a git repo, MAIN leg forced
    # to None → PASS with is_warn=True (judge-error parity).
    nongit = tmp_path / "nongit"
    nongit.mkdir()
    monkeypatch.chdir(nongit)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: None)
    res_warn = verify_task_body.check_cross_issue_reuse_provenance(
        _CHECK1256_UNDECLARED_BODY, issue=999
    )
    assert res_warn.passed and res_warn.is_warn, res_warn.render()
    assert "eval root unresolved" in res_warn.detail, res_warn.render()


def test_cross_issue_reuse_provenance_non_utf8_json_skipped(tmp_path):
    """(e-bis) Robustness ladder, reviewer r1 Minor: a non-UTF8 `.json`
    <=50 MB makes `path.read_text()` raise `UnicodeDecodeError` (neither
    `OSError` nor `json.JSONDecodeError`) — the scanner must skip the file,
    not crash the gate → PASS (graceful skip), no exception."""
    eval_dir = tmp_path / "eval_results" / "issue_999"
    eval_dir.mkdir(parents=True)
    (eval_dir / "non_utf8.json").write_bytes(
        b'\xff\xfe{"metadata": {"args": {"hf_rev_779_x": "deadbeef"}}}'
    )
    res = verify_task_body.check_cross_issue_reuse_provenance(
        _CHECK1256_UNDECLARED_BODY, issue=999, eval_root=tmp_path
    )
    assert res.passed and not res.is_warn, res.render()
    assert "graceful skip" in res.detail, res.render()


def test_cross_issue_reuse_provenance_v3_body_vacuous_pass(tmp_path):
    """(f) Forward-only generation gate: a v3-sentinel body over a
    TRIGGERING tree → PASS "not a v4 body" (grandfathered v3/v2/legacy
    bodies are wholly exempt; plan §11 item 6)."""
    _make_cross_issue_pin_eval_tree(tmp_path, 999, pins={"hf_rev_779_labels": _CHECK1256_SYN_SHA})
    res = verify_task_body.check_cross_issue_reuse_provenance(
        _V3_GOOD_BODY, issue=999, eval_root=tmp_path
    )
    assert res.passed and not res.is_warn, res.render()
    assert "not a v4 body" in res.detail, res.render()


def test_cross_issue_reuse_provenance_tier2_input_shas_warns(tmp_path):
    """Tier 2 (WARN): an `issue658_...` path in `metadata.input_shas` with
    the body silent on #658 → PASS with `is_warn=True` naming #658; the
    same body mentioning the `issue658_theory_assumptions` segment →
    clean PASS."""
    _make_cross_issue_pin_eval_tree(
        tmp_path,
        999,
        pins={},
        input_shas={"issue658_theory_assumptions/store/v0.pt": "46c06e89c513ca59"},
    )
    res = verify_task_body.check_cross_issue_reuse_provenance(
        _CHECK1256_UNDECLARED_BODY, issue=999, eval_root=tmp_path
    )
    assert res.passed and res.is_warn, res.render()
    assert "#658" in res.detail, res.render()
    body_mentioning = _CHECK1256_UNDECLARED_BODY + (
        "\nInputs reused from the issue658_theory_assumptions store.\n"
    )
    res2 = verify_task_body.check_cross_issue_reuse_provenance(
        body_mentioning, issue=999, eval_root=tmp_path
    )
    assert res2.passed and not res2.is_warn, res2.render()


def test_cross_issue_reuse_provenance_args_nonpath_tag_ignored(tmp_path):
    """Tier-2 noise exemption: a NON-path-like `metadata.args` string value
    (`issue404_outcome_eval`, no `/`) is the measured experiment-tag noise
    class → not scanned → graceful-skip PASS."""
    _make_cross_issue_pin_eval_tree(tmp_path, 999, pins={"experiment": "issue404_outcome_eval"})
    res = verify_task_body.check_cross_issue_reuse_provenance(
        _CHECK1256_UNDECLARED_BODY, issue=999, eval_root=tmp_path
    )
    assert res.passed and not res.is_warn, res.render()
    assert "graceful skip" in res.detail, res.render()


def test_cross_issue_reuse_provenance_nonhex_pin_issue_mention_fallback(tmp_path):
    """Non-hex tier-1 fallback (the one predicate branch the named cases
    miss): a branch/tag pin value (`main`) cannot use the hex-prefix
    predicate — it falls back to an issue-level mention. Body without any
    #779 mention → FAIL; body with a `[#779](...)` link → clean PASS."""
    _make_cross_issue_pin_eval_tree(tmp_path, 999, pins={"hf_rev_779": "main"})
    res = verify_task_body.check_cross_issue_reuse_provenance(
        _CHECK1256_UNDECLARED_BODY, issue=999, eval_root=tmp_path
    )
    assert not res.passed, res.render()
    assert "hf_rev_779" in res.detail and "#779" in res.detail, res.render()
    body_mentioning = _CHECK1256_UNDECLARED_BODY + (
        "\n- Reused labels from [#779](https://eps.superkaiba.com/tasks/779): "
        "monitoring labels @ main — fit: same judge pin.\n"
    )
    res2 = verify_task_body.check_cross_issue_reuse_provenance(
        body_mentioning, issue=999, eval_root=tmp_path
    )
    assert res2.passed and not res2.is_warn, res2.render()


def test_cross_issue_reuse_provenance_integration_1092(tmp_path):
    """Integration pair on REAL data (skip-if-absent, the #715 convention):
    the current (fixed) #1092 body over the real
    `eval_results/issue_1092/` tree → PASS; the same body with the round-3
    "Also reused (cross-corpus transfer round): ..." sentence stripped
    (the pre-fix replay) → FAIL naming `hf_rev_779_labels`. Measured
    during planning: pre-fix `labels` unsatisfied, `passb` satisfied via
    the shared `037fcbb` revision (documented residual)."""
    try:
        from explore_persona_space.task_workflow import find_task_path, repo_root

        body_path = find_task_path(1092) / "body.md"
        if not body_path.exists():
            pytest.skip("published #1092 body absent")
        body = body_path.read_text()
        main_root = repo_root()
    except Exception:
        pytest.skip("could not resolve published #1092 body")
    if not verify_task_body.is_v4(body):
        pytest.skip("#1092 body is not v4 (migrated away from the incident fixture)")
    candidates = [
        Path(__file__).resolve().parents[1],  # this checkout (cone added per sparse_cones.txt)
        main_root,  # MAIN
    ]
    eval_root = next(
        (
            c
            for c in candidates
            if (c / "eval_results" / "issue_1092" / "cross-corpus-probe-transfer").is_dir()
        ),
        None,
    )
    if eval_root is None:
        pytest.skip("issue_1092 eval fixture absent (sparse worktree without the cone)")
    res = verify_task_body.check_cross_issue_reuse_provenance(body, issue=1092, eval_root=eval_root)
    assert res.passed and not res.is_warn, res.render()
    # Pre-fix replay: strip the round-3 declaration sentence.
    anchor = "Also reused (cross-corpus transfer round):"
    if anchor not in body:
        pytest.skip("anchor sentence gone — #1092 body edited since #1256; refresh the replay")
    prefix_body = re.sub(
        r"Also reused \(cross-corpus transfer round\):.*?recorded in `transfer_reads\.json`\)\.",
        "",
        body,
        flags=re.DOTALL,
    )
    assert prefix_body != body, "strip did not change the body"
    res2 = verify_task_body.check_cross_issue_reuse_provenance(
        prefix_body, issue=1092, eval_root=eval_root
    )
    assert not res2.passed, res2.render()
    assert "hf_rev_779_labels" in res2.detail and "#779" in res2.detail, res2.render()


def test_cross_issue_reuse_provenance_env_fence(tmp_path, monkeypatch):
    """Env fence: `EPM_VERIFY_BODY_NO_EVAL_SCAN=1` over a TRIGGERING tree →
    PASS with the greppable fence detail (distinct from the no-pins
    graceful skip); the scanner-level fence returns None too."""
    _make_cross_issue_pin_eval_tree(tmp_path, 999, pins={"hf_rev_779_labels": _CHECK1256_SYN_SHA})
    monkeypatch.setenv("EPM_VERIFY_BODY_NO_EVAL_SCAN", "1")
    res = verify_task_body.check_cross_issue_reuse_provenance(
        _CHECK1256_UNDECLARED_BODY, issue=999, eval_root=tmp_path
    )
    assert res.passed and not res.is_warn, res.render()
    assert "EPM_VERIFY_BODY_NO_EVAL_SCAN" in res.detail, res.render()
    assert verify_task_body._scan_cross_issue_reuse_pins(tmp_path, 999) is None


def test_cross_issue_reuse_provenance_oversize_file_skipped(tmp_path):
    """50 MB stat guard (MANDATORY per plan §8 — `issue_810` carries
    138-208 MB committed JSONs): a VALID >50 MB JSON carrying an
    undeclared cross-issue pin must be SKIPPED by the stat guard, not
    read — without the guard this tree would FAIL, so this test is
    discriminating for the guard itself."""
    eval_dir = tmp_path / "eval_results" / "issue_999"
    eval_dir.mkdir(parents=True)
    pad = "x" * (51 * 1024 * 1024)
    (eval_dir / "big.json").write_text(
        '{"pad": "'
        + pad
        + '", "metadata": {"args": {"hf_rev_779_labels": "'
        + _CHECK1256_SYN_SHA
        + '"}}}'
    )
    res = verify_task_body.check_cross_issue_reuse_provenance(
        _CHECK1256_UNDECLARED_BODY, issue=999, eval_root=tmp_path
    )
    assert res.passed and not res.is_warn, res.render()
    assert "graceful skip" in res.detail, res.render()


# ─── Check 15 clause-scoping (#893, incident #841) ─────────────────────────
#
# `check_repro_committed_claims_exist` must never pair a "committed" token
# with an ``at commit `<sha>` `` from a DIFFERENT clause of the same line
# (the #841 false FAIL: the lazy span crossed from the results-JSON
# clause's "committed" to the figures clause's sha and validated the
# eval_results paths against the WRONG sha). Fixture: a throwaway repo
# with two commits so a pair validated against the wrong commit fails
# `git cat-file -e`.


def _make_repo_two_commits(tmp_path):
    """Throwaway repo with two commits: commit A adds
    eval_results/issue_999/metrics.json; commit B REMOVES it and adds
    figures/issue_999/hero.png — so a (sha, path) pair validated against
    the WRONG commit fails `git cat-file -e`. Returns (repo, sha_a, sha_b)."""
    repo = tmp_path / "tworepo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    def head_sha():
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    metrics = repo / "eval_results" / "issue_999" / "metrics.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("{}")
    git("add", "eval_results")
    git("commit", "-q", "-m", "commit A: add metrics.json")
    sha_a = head_sha()
    git("rm", "-q", "eval_results/issue_999/metrics.json")
    fig = repo / "figures" / "issue_999" / "hero.png"
    fig.parent.mkdir(parents=True)
    fig.write_bytes(b"\x89PNG fake bytes")
    git("add", "figures")
    git("commit", "-q", "-m", "commit B: remove metrics.json, add hero.png")
    sha_b = head_sha()
    return repo, sha_a, sha_b


def _repro_body(line):
    """Minimal non-v4 body routing `line` through the `## Reproducibility`
    H2 (the `section_text` leg of `_repro_section_text`)."""
    return "# T\n\n## Reproducibility\n\n" + line + "\n"


def test_check15_cross_clause_sha_not_paired_841_shape(tmp_path, monkeypatch):
    """The #841 regression: clause 1 carries "committed" + the eval path +
    a parenthesized branch sha WITHOUT "at commit"; clause 2 carries
    "at commit `<sha_b>`" WITHOUT "committed". The eval path must never be
    validated against the figures sha — no pair forms, the check PASSes."""
    repo, sha_a, sha_b = _make_repo_two_commits(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    line = (
        f"Result JSONs `eval_results/issue_999/metrics.json` committed on branch "
        f"`issue-999` (`{sha_a}`). Figures pinned at commit `{sha_b}` on main."
    )
    # Fixture sanity: the OLD whole-line regex DOES match this shape (the
    # lazy span crosses the `. ` boundary) — proving the test exercises the
    # cross-clause pairing the fix removes, not a never-matching string.
    assert verify_task_body._COMMITTED_AT_SHA_RE.search(line) is not None
    res = verify_task_body.check_repro_committed_claims_exist(_repro_body(line))
    assert res.passed and not res.is_warn, res.render()
    assert "no `committed" in res.detail


def test_check15_two_claims_one_line_each_validated_against_own_clause(tmp_path, monkeypatch):
    """A line with TWO genuine claims validates each against its own
    clause's path (finditer fixes the only-first-claim-per-line defect;
    the old code paired hero.png with sha_a and false-FAILed)."""
    repo, sha_a, sha_b = _make_repo_two_commits(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    line = (
        f"Eval JSONs `eval_results/issue_999/metrics.json` committed at commit `{sha_a}`; "
        f"figures `figures/issue_999/hero.png` committed at commit `{sha_b}`."
    )
    res = verify_task_body.check_repro_committed_claims_exist(_repro_body(line))
    assert res.passed and not res.is_warn, res.render()
    assert "2 committed-at-sha claim pair(s) resolved cleanly" in res.detail


def test_check15_same_clause_missing_path_still_fails(tmp_path, monkeypatch):
    """Coverage preserved (#550 shape): a same-clause pair whose sha lacks
    the path still FAILs — clause scoping must not defang the check."""
    repo, _sha_a, sha_b = _make_repo_two_commits(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    line = (
        f"- Eval JSONs committed to git at commit `{sha_b}` (65 files): "
        f"[`eval_results/issue_999/metrics.json`](https://github.com/x/y/blob/{sha_b}/f)."
    )
    res = verify_task_body.check_repro_committed_claims_exist(_repro_body(line))
    assert not res.passed, res.render()
    assert "NOT present" in res.detail


def test_check15_same_clause_pair_resolves(tmp_path, monkeypatch):
    """Happy path (#601 shape): a genuine same-clause pair resolves."""
    repo, sha_a, _sha_b = _make_repo_two_commits(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    line = (
        f"- Eval JSONs: `eval_results/issue_999/metrics.json` committed at commit `{sha_a}` "
        f"on branch `issue-999`."
    )
    res = verify_task_body.check_repro_committed_claims_exist(_repro_body(line))
    assert res.passed and not res.is_warn, res.render()
    assert "1 committed-at-sha claim pair(s) resolved cleanly" in res.detail


def test_split_clauses_backtick_protection():
    """Pure-helper pins: backtick spans protect delimiters; extension dots
    and an end-of-string period never split; the ` · ` (U+00B7) leg splits;
    an unbalanced backtick fail-safes to NO further splits (status-quo
    whole-line behavior on the suffix); an abbreviation dot (`e.g. `) does
    not split while a genuine sentence boundary on the same line does."""
    split = verify_task_body._split_clauses
    # Backtick protection: the `; ` inside the code span does not split.
    assert split("a `x; y` b; c") == ["a `x; y` b", "c"]
    # Extension dot + end-of-string period: one clause.
    line = "path `m.json` committed at commit `abcd1234` end."
    assert split(line) == [line]
    # Interpunct leg (v2 binding Must-Fix): U+00B7 with flanking whitespace.
    assert split("a · b") == ["a", "b"]
    # Unbalanced backtick: in_code latches for the remainder — no further
    # splits (fail-safe direction: the suffix keeps whole-line behavior,
    # so the fix can never introduce a NEW false FAIL there).
    unbalanced = "a `unclosed; b. c"
    assert split(unbalanced) == [unbalanced]
    # Abbreviation guard: no split at "e.g. ", split at the "y. " boundary.
    assert split("committed, e.g. x at commit y. done") == [
        "committed, e.g. x at commit y",
        "done",
    ]


def test_check15_v4_footer_route_interpunct_no_cross_field_pair(tmp_path, monkeypatch):
    """The #841 PRODUCTION route: a v4 `**Repro:**` footer (reaches the
    check via `_v4_footer_text`) whose committed-claim field and
    ``at commit `<sha>` `` field are separated by ` · ` — no cross-field
    pair may form. Also exercises the ` · ` splitter leg end-to-end."""
    repo, sha_a, sha_b = _make_repo_two_commits(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    footer_line = (
        f"**Repro:** Result JSONs `eval_results/issue_999/metrics.json` committed on "
        f"branch `issue-999` (`{sha_a}`) · figures at commit `{sha_b}`"
    )
    lines = _V4_GOOD_BODY.splitlines()
    idx = next(i for i, ln in enumerate(lines) if ln.startswith("**Repro:**"))
    lines[idx] = footer_line
    body = "\n".join(lines) + "\n"
    assert verify_task_body.is_v4(body)
    # Fixture sanity: the OLD whole-line regex crosses the ` · ` field
    # boundary on the raw footer line (would have cross-paired).
    assert verify_task_body._COMMITTED_AT_SHA_RE.search(footer_line) is not None
    res = verify_task_body.check_repro_committed_claims_exist(body)
    assert res.passed and not res.is_warn, res.render()


def test_check15_abbreviation_dot_same_clause_claim_still_fails(tmp_path, monkeypatch):
    """Protection preservation (v2 binding concern 2): a comma-less
    abbreviation dot between "committed" and its same-sentence path/sha
    claim must NOT split the clause — this TRUE FAIL stays a FAIL (without
    the guard, the split would silently flip it to PASS)."""
    repo, _sha_a, sha_b = _make_repo_two_commits(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    line = f"- Artifacts committed, e.g. `eval_results/issue_999/metrics.json` at commit `{sha_b}`."
    res = verify_task_body.check_repro_committed_claims_exist(_repro_body(line))
    assert not res.passed, res.render()
    assert "NOT present" in res.detail


# ─── H1 ↔ frontmatter-title sync check (#1110/#825) ────────────────────────

_H1_SYNC_NAME = "H1 matches frontmatter title"
_H1_SYNC_FILLER = (
    "Filler prose so the fixture clears check_body_nonstub's 500-char stub "
    "floor without carrying any other clean-result structure. " * 6
)


def _sentinelled_raw(sentinel: str, fm_title: str | None, h1: str) -> str:
    """Minimal raw doc for the H1↔title sync check: frontmatter (+ optional
    title) + H1 + sentinel + enough prose to clear check_body_nonstub's
    stub floor. Tests assert on the named check via _results_by_name — the
    minimal shape fails unrelated structure checks by construction, so
    these tests never assert overall `ok`."""
    fm_lines = ["---"]
    if fm_title is not None:
        fm_lines.append(f"title: {fm_title}")
    fm_lines.extend(["kind: experiment", "---"])
    return "\n".join(fm_lines) + f"\n# {h1}\n\n{sentinel}\n\n{_H1_SYNC_FILLER}\n"


def test_h1_title_sync_v4_match_passes():
    """v4 body whose fm title == H1 (incl. the confidence tag) passes clean."""
    raw = _sentinelled_raw(
        "<!-- clean-result-v4 -->",
        "A tidy claim about leakage (MODERATE confidence)",
        "A tidy claim about leakage (MODERATE confidence)",
    )
    _ok, results = verify_task_body.verify_text(raw)
    res = _results_by_name(results)[_H1_SYNC_NAME]
    assert res.passed and res.is_warn is False, res.render()


def test_h1_title_sync_whitespace_normalization():
    """Pins the normalization rule: whitespace collapse ONLY. Internal
    doubled spaces + trailing spaces in the H1 are not divergence; no
    case/Unicode/punctuation folding happens anywhere else."""
    raw = _sentinelled_raw(
        "<!-- clean-result-v4 -->",
        "A tidy claim about leakage (MODERATE confidence)",
        "A tidy  claim about   leakage (MODERATE confidence)  ",
    )
    _ok, results = verify_task_body.verify_text(raw)
    res = _results_by_name(results)[_H1_SYNC_NAME]
    assert res.passed and res.is_warn is False, res.render()


def test_h1_title_sync_v4_mismatch_fails():
    """Fail-loud: the #825 shape — frontmatter retitled via set-title, body
    H1 left stale — hard-FAILs on a v4 body. Detail names both strings and
    the set-title remediation."""
    raw = _sentinelled_raw(
        "<!-- clean-result-v4 -->",
        "Fresh retitled claim (HIGH confidence)",
        "Stale original claim (MODERATE confidence)",
    )
    _ok, results = verify_task_body.verify_text(raw)
    res = _results_by_name(results)[_H1_SYNC_NAME]
    assert not res.passed, res.render()
    assert "Stale original claim" in res.detail
    assert "Fresh retitled claim" in res.detail
    assert "set-title" in res.detail


def test_h1_title_sync_v3_mismatch_warns():
    """Forward-only pin: the same divergence on a v3-sentinelled body is a
    loud WARN (passed=True, is_warn=True), never a new retroactive FAIL."""
    raw = _sentinelled_raw(
        "<!-- clean-result-v3 -->",
        "Fresh retitled claim (HIGH confidence)",
        "Stale original claim (MODERATE confidence)",
    )
    _ok, results = verify_task_body.verify_text(raw)
    res = _results_by_name(results)[_H1_SYNC_NAME]
    assert res.passed and res.is_warn is True, res.render()
    assert "grandfathered" in res.detail


def test_h1_title_sync_v2_mismatch_warns():
    """Forward-only pin, v2 sentinel: divergence WARNs, never FAILs."""
    raw = _sentinelled_raw(
        "<!-- clean-result-v2 -->",
        "Fresh retitled claim (HIGH confidence)",
        "Stale original claim (MODERATE confidence)",
    )
    _ok, results = verify_task_body.verify_text(raw)
    res = _results_by_name(results)[_H1_SYNC_NAME]
    assert res.passed and res.is_warn is True, res.render()
    assert "grandfathered" in res.detail


def test_h1_title_sync_no_sentinel_skips():
    """A non-sentinelled body with a real mismatch PASS-skips — pre-promotion
    bodies legitimately have no synced H1."""
    raw = _sentinelled_raw(
        "",
        "Fresh retitled claim (HIGH confidence)",
        "Stale original claim (MODERATE confidence)",
    )
    _ok, results = verify_task_body.verify_text(raw)
    res = _results_by_name(results)[_H1_SYNC_NAME]
    assert res.passed and res.is_warn is False, res.render()
    assert "not a sentinelled" in res.detail


def test_h1_title_sync_no_frontmatter_skips():
    """Body-only input (analyzer draft / --body-stdin dry run): fm == {} →
    PASS-skip so draft verification stays green before set-title runs; the
    gate-time --issue run compares against the real body.md."""
    body = (
        "# Stale original claim (MODERATE confidence)\n\n"
        "<!-- clean-result-v4 -->\n\n" + _H1_SYNC_FILLER + "\n"
    )
    _ok, results = verify_task_body.verify_text(body)
    res = _results_by_name(results)[_H1_SYNC_NAME]
    assert res.passed and res.is_warn is False, res.render()
    assert "draft" in res.detail


def test_h1_title_sync_missing_fm_title_fails_v4():
    """Fail-loud: a sentinelled v4 body whose frontmatter lacks `title`
    entirely is a broken promotion — same severity as a mismatch, never a
    silent skip."""
    raw = _sentinelled_raw("<!-- clean-result-v4 -->", None, "Some claim (MODERATE confidence)")
    _ok, results = verify_task_body.verify_text(raw)
    res = _results_by_name(results)[_H1_SYNC_NAME]
    assert not res.passed, res.render()
    assert "no frontmatter `title`" in res.detail


def test_h1_title_sync_missing_h1_fails_v4():
    """Fail-loud: a v4-sentinelled body with no H1 FAILs the check. Unit-calls
    check_h1_matches_frontmatter_title directly (established helper-test
    precedent in this file): through verify_text this shape is preempted by
    check_body_nonstub's own no-H1 FAIL short-circuit, so the branch is only
    reachable by direct call."""
    body = "<!-- clean-result-v4 -->\n\n" + _H1_SYNC_FILLER + "\n"
    res = verify_task_body.check_h1_matches_frontmatter_title(
        body, {"title": "Some claim (MODERATE confidence)"}
    )
    assert not res.passed, res.render()
    assert "no H1" in res.detail


def test_h1_title_sync_v3_missing_fm_title_warns():
    """The v3/v2 anomaly branch stays grandfathered: a v3-sentinelled body
    with NO frontmatter `title` WARNs (passed=True, is_warn=True) — the
    forward-only rule covers the anomaly sub-cases too, not just the
    mismatch branch (the #654 shape sits on a v3 body today)."""
    raw = _sentinelled_raw("<!-- clean-result-v3 -->", None, "Some claim (MODERATE confidence)")
    _ok, results = verify_task_body.verify_text(raw)
    res = _results_by_name(results)[_H1_SYNC_NAME]
    assert res.passed and res.is_warn is True, res.render()
    assert "no frontmatter `title`" in res.detail


def test_h1_title_sync_sentinel_only_in_fence_skips():
    """A body that merely QUOTES the v4 sentinel inside a code fence is not
    sentinelled (pins the _prose_layer inheritance) — real mismatch present,
    still PASS-skip."""
    raw = (
        "---\ntitle: Fresh retitled claim (HIGH confidence)\nkind: experiment\n---\n"
        "# Stale original claim (MODERATE confidence)\n\n"
        "```markdown\n<!-- clean-result-v4 -->\n```\n\n" + _H1_SYNC_FILLER + "\n"
    )
    _ok, results = verify_task_body.verify_text(raw)
    res = _results_by_name(results)[_H1_SYNC_NAME]
    assert res.passed and res.is_warn is False, res.render()
    assert "not a sentinelled" in res.detail
