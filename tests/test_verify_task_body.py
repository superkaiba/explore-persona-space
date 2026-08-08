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
    # CHECKS has 40 body-only functions: the 20 pre-v3 body-only checks
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
    # a PASS-skip on this non-v3/non-v4 fixture — PLUS the TWELVE
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
    # `meta["text"]` block could resolve even if a sidecar did), and
    # check 40 (`check_hf_unpinned_count_claims`, WARN), a vacuous PASS
    # here because no backtick `dir/` + count-paren claim appears in the
    # fixture, so ZERO Hub probes are issued even before the fence, and
    # check 41 (`check_figure_sidecar_coverage`, WARN), a NO-OP
    # "no same-repo sha-pinned figures to check" PASS here because the
    # fixture's only figure pins a fake sha (`_git_object_exists` returns
    # 'skip', so the figure never passes the PNG-resolves scope gate).
    # check 25 (`check_audit_availability_claims_match_hf`)
    # is a vacuous PASS here because this fixture carries no
    # availability-denial-near-artifact line. verify_text prepends check 0
    # (body-nonstub) + check 0b (no-duplicate-frontmatter), runs CHECKS[1:]
    # (45 functions), then appends the Goal soft check, the H1↔frontmatter-
    # title sync check (#1110; PASS-skip: not a sentinelled body), the
    # Lens 14
    # concerns-audit, the check-16 lr-matches-plan reconciliation, the
    # check-17 Context provenance-row read, the v3 check-21
    # body-Parameters-⊆-doc reconciliation (PASS-skip with no doc), the v4
    # check-20 word caps (needs `issue` for the events-based round budget,
    # #921; PASS-skip: not a v4 body), the
    # #732 judge-API-error denominator check (PASS-skip: legacy body), the
    # judge drop-line population check (#1776 incident / task #1881;
    # PASS-skip: legacy body), the
    # check-35 cross-issue reuse-provenance check (PASS-skip: not a v4
    # body, #1256), AND
    # the check-31 orphaned-per-unit-figures probe (needs `issue` for
    # figures-dir scoping, #1011; PASS here — the fixture's fake sha is not
    # locally reachable, so the cited SHA is silently skipped), AND the
    # check-38 linked-not-embedded-figures scan (needs `issue` for
    # own-figures-dir scoping, #1371; PASS-skip: not a v4 body) →
    # 69 results total (2 prepended + CHECKS[1:]=53 + 14 appended, counting
    # the #1827 plan-conditions check narrated below; check 36
    # `check_v4_result_paragraph_sentences` (#1368), check 37
    # `check_footer_reuse_bullets_pinned` (#1370), check 39
    # `check_v4_sample_disclosure_count` (#1421), check 40
    # `check_hf_unpinned_count_claims` (#1433), and check 41
    # `check_figure_sidecar_coverage` (#1478), check 42
    # `check_body_artifact_urls_exist` (#1507 — grandfathered-WARN tier
    # here, and GOOD_BODY's only same-repo URL lives in the footer so it
    # PASSes vacuously), and check 43
    # `check_github_tree_adjacent_file_claims` (#1507 — vacuous PASS, no
    # git-tree-adjacent claims), and check 44
    # `check_footer_hf_paths_pinned` (#1509 — PASS-skip, not a v4 body),
    # and check 45
    # `check_figure_caption_count_claims_vs_sidecar` (#1511 — vacuous
    # PASS, no registered count claim), and check 46
    # `check_hf_brace_expanded_path_claims` (#1520 — vacuous PASS, no
    # brace-path claims adjacent to pinned HF tree links), and check 48
    # `check_v4_quant_result_figure` (#1832 — PASS-skip, not a v4 body),
    # and check 49 `check_v4_result_figure_cardinality` (#1879 —
    # PASS-skip, not a v4 body),
    # and check 50 `check_repro_artifacts_clean` (#1989 — probes the REAL
    # repo's working tree for the fixture's repro-named eval_results dirs;
    # `passed=True` in every state by construction — WARN/skip never flip
    # it, the check-29 precedent),
    # and check 51 `check_v4_dropped_condition_placement` (#2017 —
    # PASS-skip, not a v4 body),
    # and checks 52 `check_figure_png_sidecar_pairing` + 53
    # `check_figure_sidecar_slot_completeness` (#2016 — both NO-OP PASS:
    # GOOD_BODY's fake sha never resolves via `_git_object_exists`, the
    # same fake-sha skip as check 41)
    # ride CHECKS;
    # 36/37/39/44/48/49/51
    # PASS-skip here — not a v4 body — 40 is the vacuous PASS above, and
    # 41/52/53 are the fake-sha NO-OP PASSes above). The
    # Lens 14 / check-16 results are PASS-skips when no concerns.jsonl /
    # plans/plan.md sibling is available; check 17 and the v3/v4 checks
    # are PASS-skips on this legacy (pre-v2-sentinel) fixture. Check 47
    # `check_context_followup_scope_consistency` (#1521) is dispatched in
    # verify_text (needs the issue number) and PASS-skips here (legacy body).
    # The plan-conditions coverage check (#1827) is dispatched in verify_text
    # (needs plans/plan.md) and NO-OP PASSes here (no plan sibling).
    assert len(results) == 69
    # By-name membership so the NEXT check addition can key by name instead
    # of re-deriving the arithmetic (#1016 methodology-reconciler Must-Fix).
    assert "dropped-at-gate condition placement (v4)" in {r.name for r in results}
    assert "repro-named result dirs clean in working tree" in {r.name for r in results}
    assert "plan conditions coverage" in {r.name for r in results}
    assert "judge drop-line population reconciles" in {r.name for r in results}
    assert _HF_32_NAME in {r.name for r in results}
    assert _HF_40_NAME in {r.name for r in results}
    assert "Context follow-up provenance vs followup-scope markers" in {r.name for r in results}
    assert "HF brace-expanded path claims resolve at the adjacent pin" in {r.name for r in results}
    assert "footer HF artifact paths carry an adjacent pinned link" in {r.name for r in results}
    assert "Sample-slot disclosure count (v4)" in {r.name for r in results}
    assert "cross-issue reuse pins declared (footer Reused bullets)" in {r.name for r in results}
    assert "footer Reused bullets carry a revision/path pin" in {r.name for r in results}
    assert "figure prose numerics vs figure sidecar (plotted-value drift)" in {
        r.name for r in results
    }
    assert "figure beat claims vs sidecar rendered text (series-structure drift)" in {
        r.name for r in results
    }
    assert "figure sidecar coverage (sidecar-less embedded figures)" in {r.name for r in results}


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

# The class-B WARN token (#1510) — pinned as a LITERAL here (not imported
# from the module) so a silent token rename breaks the grep contract test.
_PER_UNIT_NAMED_CLASS = "companion-named-not-embedded"


def _make_repo_with_per_unit_orphan(
    tmp_path,
    companion: str | None = "hero_percontext.png",
    extra: str | None = None,
):
    """git repo whose HEAD commit tracks `figures/issue_999/hero.png` +
    `figures/issue_999/<companion>` (the per-unit companion; default
    `hero_percontext.png`; ``None`` -> no companion — required by widened-
    scope negative pins whose fixtures must not trip class A, #2169) +
    optionally `figures/issue_999/<extra>` (ONE additional NON-per-unit
    PNG — the class-C candidate, #2169) +
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
    if companion is not None:
        (figdir / companion).write_bytes(b"\x89PNG fake bytes")
    if extra is not None:
        (figdir / extra).write_bytes(b"\x89PNG fake bytes")
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
    # Class separation (#1510): a never-mentioned orphan reports class A
    # only — the class-B token must not appear.
    assert _PER_UNIT_NAMED_CLASS not in r.detail
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


def test_orphan_perpair_figure_warns(tmp_path, monkeypatch):
    """#1607 pin (incident #1415 Lens 11): a committed-but-unembedded
    `*_perpair` companion at a body-cited SHA WARNs class A exactly like
    the percontext family — per-PAIR was invisible to check 31 pre-#1607."""
    repo, sha = _make_repo_with_per_unit_orphan(tmp_path, companion="hero_perpair.png")
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = GOOD_BODY.replace("0123456789abcdef", sha)
    r = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r.passed is True
    assert r.is_warn is True
    assert "figures/issue_999/hero_perpair.png" in r.detail
    assert _PER_UNIT_NAMED_CLASS not in r.detail  # class A — never mentioned


def test_perpair_figure_embedded_no_warn(tmp_path, monkeypatch):
    """Embedded per-pair companion → clean PASS (the new arm must not
    WARN on a discipline-satisfying body)."""
    repo, sha = _make_repo_with_per_unit_orphan(tmp_path, companion="hero_perpair.png")
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    companion_url = (
        "https://raw.githubusercontent.com/superkaiba/explore-persona-space/"
        f"{sha}/figures/issue_999/hero_perpair.png"
    )
    body = GOOD_BODY.replace("0123456789abcdef", sha).replace(
        "> **Figure.**",
        f"![Per-pair deltas behind the aggregate.]({companion_url})\n\n> **Figure.**",
    )
    r = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r.passed is True
    assert r.is_warn is False


def test_perpair_named_not_embedded_warns_class_b(tmp_path, monkeypatch):
    """The #1415 incident shape on the NEW arm: a pair-stem companion NAMED
    in body prose (no exemption idiom) and embedded nowhere → class-B WARN
    carrying the `companion-named-not-embedded` token (mirror of the
    percontext class-B pin)."""
    repo, sha = _make_repo_with_per_unit_orphan(tmp_path, companion="hero_perpair.png")
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = GOOD_BODY.replace("0123456789abcdef", sha).replace(
        "The 17-pt lift holds at every seed;",
        "The per-pair views behind these aggregates are `hero_perpair.png`, "
        "committed at the same pin. The 17-pt lift holds at every seed;",
    )
    r = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r.passed is True  # WARN-tier: the overall verdict is never flipped
    assert r.is_warn is True
    assert _PER_UNIT_NAMED_CLASS in r.detail
    assert "figures/issue_999/hero_perpair.png" in r.detail


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
    """The prose disclosure escape, PHRASE-GATED as of #1510: an unembedded
    companion whose stem is named in body prose is exempt only because the
    fixture prose carries an exemption idiom ("superseded by") in the
    stem's own paragraph — the second phrase-set pin (the corpus-real #928
    footer idiom); a bare naming would now WARN class B."""
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


def test_orphan_named_not_embedded_warns_without_exemption(tmp_path, monkeypatch):
    """The #1426 regression fixture (#1510 durability pin, incl. for the
    SPEC/lens prose edits): the companion NAMED in body prose with a bare
    provenance clause ("committed at the same pin" — the incident's own
    phrasing) and embedded nowhere → class-B WARN carrying the
    `companion-named-not-embedded` token; the provenance clause does NOT
    exempt."""
    repo, sha = _make_repo_with_per_unit_orphan(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = GOOD_BODY.replace("0123456789abcdef", sha).replace(
        "The 17-pt lift holds at every seed;",
        "The per-context views behind these aggregates are `hero_percontext.png`, "
        "committed at the same pin. The 17-pt lift holds at every seed;",
    )
    r = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r.passed is True  # WARN-tier: the overall verdict is never flipped
    assert r.is_warn is True
    assert _PER_UNIT_NAMED_CLASS in r.detail
    assert _PER_UNIT_ORPHAN_PATH in r.detail
    assert "Lens 11" in r.detail
    assert sha[:8] in r.detail


def test_orphan_named_with_exemption_phrase_no_warn(tmp_path, monkeypatch):
    """The exemption phrase in the stem's own paragraph silences class B —
    both anchored idioms, incl. a CAPITALIZED variant (the phrase regex is
    case-insensitive)."""
    repo, sha = _make_repo_with_per_unit_orphan(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = GOOD_BODY.replace("0123456789abcdef", sha).replace(
        "The 17-pt lift holds at every seed;",
        "Deliberately linked, not embedded: the hero already shows every point — "
        "see `hero_percontext.png`. The 17-pt lift holds at every seed;",
    )
    r = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r.passed is True
    assert r.is_warn is False
    body2 = GOOD_BODY.replace("0123456789abcdef", sha).replace(
        "The 17-pt lift holds at every seed;",
        "`hero_percontext.png` is a round-1 exploratory view. Superseded by the "
        "hero's right panel. The 17-pt lift holds at every seed;",
    )
    r2 = verify_task_body.check_orphaned_per_unit_figures(body2, issue=999)
    assert r2.passed is True
    assert r2.is_warn is False


def test_orphan_exemption_phrase_other_paragraph_still_warns(tmp_path, monkeypatch):
    """Paragraph-proximity scoping: an exemption phrase in a DIFFERENT
    blank-line-delimited paragraph than the stem does NOT exempt — the
    stem's own paragraph must carry the phrase."""
    repo, sha = _make_repo_with_per_unit_orphan(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = GOOD_BODY.replace("0123456789abcdef", sha).replace(
        "The 17-pt lift holds at every seed;",
        "The per-context companion is `hero_percontext.png`, committed at the same pin.\n\n"
        "A different figure was deliberately not embedded for space. "
        "The 17-pt lift holds at every seed;",
    )
    r = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r.passed is True
    assert r.is_warn is True
    assert _PER_UNIT_NAMED_CLASS in r.detail


def test_orphan_multi_stem_one_paragraph_phrase_exempts_both(tmp_path, monkeypatch):
    """Accepted false-negative direction (pinned): ONE exemption phrase in
    a paragraph naming TWO per-unit companion stems exempts BOTH —
    paragraph-level co-occurrence, not per-stem sentence parsing (WARN-tier
    backstop; Lens 11 stays the substantive owner)."""
    repo, _sha_a = _make_repo_with_per_unit_orphan(tmp_path)

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    (repo / "figures" / "issue_999" / "hero_percell.png").write_bytes(b"\x89PNG fake bytes")
    git("add", "figures")
    git("commit", "-q", "-m", "add second per-unit companion")
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = GOOD_BODY.replace("0123456789abcdef", sha).replace(
        "The 17-pt lift holds at every seed;",
        "Round-1 exploratory views `hero_percontext.png` and `hero_percell.png` are "
        "superseded by the embedded hero panels. The 17-pt lift holds at every seed;",
    )
    r = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r.passed is True
    assert r.is_warn is False  # one phrase exempts both co-resident stems


def test_orphan_blob_embed_no_warn(tmp_path, monkeypatch):
    """The widened any-URL-form embedded set (#1510): a GitHub BLOB-URL
    image embed of the companion is a real embed → clean PASS. Pre-#1510
    this shape passed only via the stem-in-prose accident (the embed URL
    text names the stem); under the phrase-gated escape it would now WARN
    class B were the embed set still raw-GitHub-only."""
    repo, sha = _make_repo_with_per_unit_orphan(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    embed = (
        "![Per-context deltas.](https://github.com/superkaiba/explore-persona-space/"
        f"blob/{sha}/figures/issue_999/hero_percontext.png)"
    )
    body = GOOD_BODY.replace("0123456789abcdef", sha).replace(
        "> **Figure.**", embed + "\n\n> **Figure.**"
    )
    r = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r.passed is True
    assert r.is_warn is False
    assert "no orphaned per-unit figures" in r.detail


def test_orphan_linked_results_path_defers_to_check38(tmp_path, monkeypatch):
    """Single ownership (no double-WARN): a companion markdown-LINKED in
    the v4 `## Results` prose layer is check 38's WARN — check 31 carries
    no class-B token for it. Second arm pins case-fold symmetry of the
    subtraction: a case-varying link URL still defers (without the
    case-fold it would surface as class A — the case-varied URL does not
    put the exact stem in the body)."""
    repo, sha = _make_repo_with_per_unit_orphan(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    link_url = (
        "https://raw.githubusercontent.com/superkaiba/explore-persona-space/"
        f"{sha}/figures/issue_999/hero_percontext.png"
    )
    fixture = _c37_body_with_results_link(
        f"The per-context view behind this aggregate: [companion]({link_url})."
    ).replace("0123456789abcdef", sha)
    _fm, body = verify_task_body.split_frontmatter(fixture)
    r31 = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r31.passed is True
    assert r31.is_warn is False  # deferred to check 38 — no WARN here
    assert _PER_UNIT_NAMED_CLASS not in r31.detail
    r38 = verify_task_body.check_linked_not_embedded_figures(body, issue=999)
    assert r38.is_warn is True
    assert _PER_UNIT_ORPHAN_PATH in r38.detail
    # Case-fold symmetry: the same link with a case-varying basename.
    upper_url = link_url.replace("hero_percontext.png", "HERO_percontext.png")
    fixture_u = _c37_body_with_results_link(
        f"The per-context view behind this aggregate: [companion]({upper_url})."
    ).replace("0123456789abcdef", sha)
    _fm, body_u = verify_task_body.split_frontmatter(fixture_u)
    r31_u = verify_task_body.check_orphaned_per_unit_figures(body_u, issue=999)
    assert r31_u.passed is True
    assert r31_u.is_warn is False


def test_orphan_nonv4_linked_path_fires_named_class(tmp_path, monkeypatch):
    """Row-f coverage (deliberate widening, not an accident): a markdown
    LINK outside check 38's gates — here in a pre-v3 legacy body 38 never
    scans — is a prose naming of an unembedded companion, so class B fires
    absent an exemption phrase (the one-phrase remedy applies the same)."""
    repo, sha = _make_repo_with_per_unit_orphan(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    link_url = (
        "https://raw.githubusercontent.com/superkaiba/explore-persona-space/"
        f"{sha}/figures/issue_999/hero_percontext.png"
    )
    body = GOOD_BODY.replace("0123456789abcdef", sha).replace(
        "The 17-pt lift holds at every seed;",
        f"Full per-context view: [companion]({link_url}). The 17-pt lift holds at every seed;",
    )
    r = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r.passed is True
    assert r.is_warn is True
    assert _PER_UNIT_NAMED_CLASS in r.detail


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
        ("issue952_leg3_perpair_d_distribution", True),  # corpus-real compact form (#952)
        ("h1_per_pair_scatter", True),  # the #1415 incident underscore form
        ("position_profile_perpair_top", True),  # the #1607 candidate's own example stem
        ("per-pair_deltas", True),  # hyphen spelling
        ("repair_log", False),  # `pair` alone, no `per` prefix — never matches
        ("pairwise_grid", False),  # bare pair-family word, no per prefix
        ("super_pair_map", False),  # mid-word `per_pair` blocked by the lookbehind
    ],
)
def test_per_unit_basename_pattern(stem, expected):
    """The deliberately-narrow check-31 pattern: the four per-unit nouns
    (context/unit/cell/pair) with -/_ spellings match; regime names (`indiv`),
    mid-word hits (`supercontext`), and other per-X families
    (per_source/per_seed) do NOT — Lens 11 owns the substance."""
    assert bool(verify_task_body._PER_UNIT_FIG_RE.search(stem)) is expected


# ─── Check 31 widened scope (#2169): class C `committed-figure-unmentioned` ──

# The class-C WARN token (#2169) — pinned as a LITERAL (not imported from the
# module) so a silent token rename breaks the grep contract, exactly like the
# class-B pin above.
_COMMITTED_UNMENTIONED_CLASS = "committed-figure-unmentioned"

_CLASS_C_PATH = "figures/issue_999/f5_arm_agreement.png"

# Entry shape in the WARN detail: `<path>` (committed at <shas>; <class text>).
_DETAIL_ENTRY_RE = re.compile(r"`(figures/issue_\d+/[^`]+)` \(([^)]*)\)")


def _plan_dir_naming(tmp_path, monkeypatch, *names: str):
    """Write a `plans/v1.md` naming ``names`` (each backticked, so exact
    stems and bounded glob tokens both register) and monkeypatch the §3.0
    `_resolve_task_plans_dir` seam at it. The seam is load-bearing, not
    convenience: `issue=999` resolves a REAL registered task whose
    `plans/v1.md` exists (and contains none of these fixture stems —
    verified in the #2169 plan), so without the monkeypatch a fixture's
    plan file would never be read and every class-C pin would go silent
    for the WRONG reason (not-a-candidate instead of the asserted
    behaviour). Returns the plans dir."""
    plans = tmp_path / "task999" / "plans"
    plans.mkdir(parents=True)
    (plans / "v1.md").write_text("Planned figures: " + ", ".join(f"`{n}`" for n in names) + "\n")
    monkeypatch.setattr(verify_task_body, "_resolve_task_plans_dir", lambda issue: plans)
    return plans


def test_committed_unmentioned_figure_warns_class_c(tmp_path, monkeypatch):
    """The #2061 incident shape (required positive): `f5_arm_agreement.png`
    (NON-per-unit stem) committed at the body-cited SHA, embedded nowhere,
    named nowhere -> class-C WARN carrying the path, the
    `committed-figure-unmentioned` token, the Lens 13 pointer, and the
    short SHA; `passed` stays True (WARN-tier). The fixture's plan names
    f5 (§3.0: only plan-named figures are class-C candidates)."""
    repo, sha = _make_repo_with_per_unit_orphan(tmp_path, extra="f5_arm_agreement.png")
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    _plan_dir_naming(tmp_path, monkeypatch, "f5_arm_agreement.png")
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
    assert r.is_warn is True
    assert _CLASS_C_PATH in r.detail
    assert _COMMITTED_UNMENTIONED_CLASS in r.detail
    assert "Lens 13" in r.detail
    assert sha[:8] in r.detail


def test_committed_figure_named_in_disposition_no_warn(tmp_path, monkeypatch):
    """Required negative: the same non-per-unit figure NAMED in a
    disposition line with the 'not embedded' idiom -> silent (the mention
    bar is satisfied a fortiori). Vacuity control (§5 mechanical rule): the
    un-dispositioned body WARNs FIRST on the same fixture + plan file, so
    the silence below is the disposition's doing, never the §3.0 filter
    quietly de-candidating f5."""
    repo, sha = _make_repo_with_per_unit_orphan(tmp_path, extra="f5_arm_agreement.png")
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    _plan_dir_naming(tmp_path, monkeypatch, "f5_arm_agreement.png")
    companion_url = (
        "https://raw.githubusercontent.com/superkaiba/explore-persona-space/"
        f"{sha}/figures/issue_999/hero_percontext.png"
    )
    body_bare = GOOD_BODY.replace("0123456789abcdef", sha).replace(
        "> **Figure.**",
        f"![Per-context deltas behind the aggregate.]({companion_url})\n\n> **Figure.**",
    )
    r_bare = verify_task_body.check_orphaned_per_unit_figures(body_bare, issue=999)
    assert r_bare.is_warn is True  # the WARN this negative pin suppresses
    assert _COMMITTED_UNMENTIONED_CLASS in r_bare.detail
    body = body_bare.replace(
        "The 17-pt lift holds at every seed;",
        "The planned arm-agreement view `f5_arm_agreement.png` is committed at the "
        "same pinned SHA, not embedded: redundant with the hero panels. "
        "The 17-pt lift holds at every seed;",
    )
    r = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r.passed is True
    assert r.is_warn is False


def test_committed_figure_bare_mention_no_warn(tmp_path, monkeypatch):
    """The two-bar decision, pinned EXPLICITLY (this is the pin to attack if
    the mention-bar call is wrong): a non-per-unit figure named with NO
    exemption idiom is ALSO silent — naming alone satisfies the widened
    class's looser bar, unlike the per-unit family's phrase bar. Vacuity
    control (§5 mechanical rule): the unmentioned body WARNs FIRST on the
    same fixture + plan file."""
    repo, sha = _make_repo_with_per_unit_orphan(tmp_path, extra="f5_arm_agreement.png")
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    _plan_dir_naming(tmp_path, monkeypatch, "f5_arm_agreement.png")
    companion_url = (
        "https://raw.githubusercontent.com/superkaiba/explore-persona-space/"
        f"{sha}/figures/issue_999/hero_percontext.png"
    )
    body_bare = GOOD_BODY.replace("0123456789abcdef", sha).replace(
        "> **Figure.**",
        f"![Per-context deltas behind the aggregate.]({companion_url})\n\n> **Figure.**",
    )
    r_bare = verify_task_body.check_orphaned_per_unit_figures(body_bare, issue=999)
    assert r_bare.is_warn is True  # the WARN this negative pin suppresses
    body = body_bare.replace(
        "The 17-pt lift holds at every seed;",
        "The arm-agreement view `f5_arm_agreement.png` is committed alongside. "
        "The 17-pt lift holds at every seed;",
    )
    r = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r.passed is True
    assert r.is_warn is False


def test_committed_figure_embedded_no_warn(tmp_path, monkeypatch):
    """An EMBEDDED non-per-unit figure -> silent (the embed branch runs
    before any naming bar, unchanged by the widening). Vacuity control (§5
    mechanical rule): the same fixture + plan file WITHOUT the f5 embed
    WARNs FIRST, so the silence is the embed's doing, not the §3.0
    filter's."""
    repo, sha = _make_repo_with_per_unit_orphan(tmp_path, extra="f5_arm_agreement.png")
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    _plan_dir_naming(tmp_path, monkeypatch, "f5_arm_agreement.png")
    base_url = f"https://raw.githubusercontent.com/superkaiba/explore-persona-space/{sha}"
    body_no_embed = GOOD_BODY.replace("0123456789abcdef", sha).replace(
        "> **Figure.**",
        f"![Per-context deltas.]({base_url}/figures/issue_999/hero_percontext.png)\n\n"
        "> **Figure.**",
    )
    r_bare = verify_task_body.check_orphaned_per_unit_figures(body_no_embed, issue=999)
    assert r_bare.is_warn is True  # the WARN this negative pin suppresses
    body = GOOD_BODY.replace("0123456789abcdef", sha).replace(
        "> **Figure.**",
        f"![Per-context deltas.]({base_url}/figures/issue_999/hero_percontext.png)\n\n"
        f"![Arm agreement.]({base_url}/figures/issue_999/f5_arm_agreement.png)\n\n"
        "> **Figure.**",
    )
    r = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r.passed is True
    assert r.is_warn is False


def test_class_tokens_never_leak_across_entries(tmp_path, monkeypatch):
    """A repo with BOTH an unmentioned per-unit companion AND an unmentioned
    non-per-unit figure: the per-unit path reports class A (no class-B
    token anywhere — class B never fires here) and the other path class C,
    with neither class's token leaking onto the other's entry."""
    repo, sha = _make_repo_with_per_unit_orphan(tmp_path, extra="f5_arm_agreement.png")
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    _plan_dir_naming(tmp_path, monkeypatch, "f5_arm_agreement.png")
    body = GOOD_BODY.replace("0123456789abcdef", sha)
    r = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r.passed is True
    assert r.is_warn is True
    assert _PER_UNIT_ORPHAN_PATH in r.detail
    assert _CLASS_C_PATH in r.detail
    assert _PER_UNIT_NAMED_CLASS not in r.detail  # class B never fired
    entries = dict(_DETAIL_ENTRY_RE.findall(r.detail))
    assert _COMMITTED_UNMENTIONED_CLASS not in entries[_PER_UNIT_ORPHAN_PATH]
    assert _COMMITTED_UNMENTIONED_CLASS in entries[_CLASS_C_PATH]


def test_non_png_committed_artifact_never_warns(tmp_path, monkeypatch):
    """Scope pin against future over-widening: committed `.pdf` +
    `.meta.json` artifacts, both unmentioned, never WARN — the widened scan
    stays PNG-only (mirrors the real `figures/issue_2061/` layout, where
    those sidecars outnumber the PNGs 2:1). The plan NAMES the `f6_extra`
    stem, so the silence below is the PNG-only scope's doing — the §3.0
    filter cannot be what suppresses it."""
    repo, _sha_a = _make_repo_with_per_unit_orphan(tmp_path, companion=None)
    _plan_dir_naming(tmp_path, monkeypatch, "f6_extra")

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    figdir = repo / "figures" / "issue_999"
    (figdir / "f6_extra.pdf").write_bytes(b"%PDF fake bytes")
    (figdir / "f6_extra.meta.json").write_text("{}\n")
    git("add", "figures")
    git("commit", "-q", "-m", "add non-PNG sidecars")
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = GOOD_BODY.replace("0123456789abcdef", sha)
    r = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r.passed is True
    assert r.is_warn is False
    assert "no orphaned per-unit figures" in r.detail


# The corrected #2061 body's disposition text, VERBATIM from
# `git show HEAD:tasks/reviewing/2061/body.md` line 61 (not a paraphrase) —
# the acceptance-(b) shape the widening must stay silent on.
_ISSUE2061_DISPOSITION_A = (
    "Companion per-cell views for the non-headline transitions and arms "
    "(`f2_percell_base_sft_context.png`, `f2_percell_base_sft_prefix.png`, "
    "`f2_percell_sft_dpo_context.png`, `f2_percell_sft_dpo_prefix.png`, "
    "`f2_percell_dpo_rlvr_prefix.png`, `f2_percell_rlvr_longer-rlvr_context.png`, "
    "`f2_percell_rlvr_longer-rlvr_prefix.png`, and the seven `f1_delta_scatter_*` "
    "siblings — not embedded: identical view on non-winning transitions/arms, "
    "committed at the same pinned SHA)."
)
_ISSUE2061_DISPOSITION_B = (
    "The planned arm-agreement view `f5_arm_agreement.png` (per-cell true max ΔR²_j, "
    "prefix arm against context arm, one panel per transition, render classes marked) "
    "is committed at the same pinned SHA, not embedded: every cell sits above the "
    "y = x line with prefix maxima near zero — the by-construction prefix degeneracy "
    "already carried in the prefix-arm scope note above, no read beyond the per-cell "
    "views."
)

_ISSUE2061_F1_SIBLINGS = [
    "f1_delta_scatter_base_sft_context.png",
    "f1_delta_scatter_base_sft_prefix.png",
    "f1_delta_scatter_sft_dpo_context.png",
    "f1_delta_scatter_sft_dpo_prefix.png",
    "f1_delta_scatter_dpo_rlvr_prefix.png",
    "f1_delta_scatter_rlvr_longer-rlvr_context.png",
    "f1_delta_scatter_rlvr_longer-rlvr_prefix.png",
]


def _make_issue2061_shape_repo(tmp_path):
    """Hermetic replica of the #2061 figures layout under `issue_999`: one
    embedded f1 sibling + seven glob-dispositioned f1 siblings + two
    explicitly-named per-cell companions + `f5_arm_agreement.png` at sha_a,
    plus `second.png` at a child commit sha_b (the second cited SHA
    exercising the multi-SHA dedup path for class C). Returns
    (repo, sha_a, sha_b)."""
    repo, _sha0 = _make_repo_with_per_unit_orphan(tmp_path, companion=None)

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    figdir = repo / "figures" / "issue_999"
    for fname in [
        "f1_delta_scatter_dpo_rlvr_context.png",  # the embedded sibling
        *_ISSUE2061_F1_SIBLINGS,
        "f2_percell_base_sft_context.png",
        "f2_percell_base_sft_prefix.png",
        "f5_arm_agreement.png",
    ]:
        (figdir / fname).write_bytes(b"\x89PNG fake bytes")
    git("add", "figures")
    git("commit", "-q", "-m", "add issue-2061-shaped figure set")
    sha_a = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (figdir / "second.png").write_bytes(b"\x89PNG fake bytes")
    git("add", "figures")
    git("commit", "-q", "-m", "add second figure at a second sha")
    sha_b = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, sha_a, sha_b


def _issue2061_shape_body(sha_a: str, sha_b: str, disposition: str) -> str:
    """GOOD_BODY carrying the issue-2061 shape: hero + one f1 sibling +
    second.png embedded (two cited SHAs), plus ``disposition`` in the
    running-prose paragraph."""
    base_a = f"https://raw.githubusercontent.com/superkaiba/explore-persona-space/{sha_a}"
    base_b = f"https://raw.githubusercontent.com/superkaiba/explore-persona-space/{sha_b}"
    return (
        GOOD_BODY.replace("0123456789abcdef", sha_a)
        .replace(
            "> **Figure.**",
            f"![Delta scatter, headline transition.]({base_a}/figures/issue_999/"
            "f1_delta_scatter_dpo_rlvr_context.png)\n\n"
            f"![Second view at a second sha.]({base_b}/figures/issue_999/second.png)\n\n"
            "> **Figure.**",
        )
        .replace(
            "The 17-pt lift holds at every seed;",
            f"{disposition} The 17-pt lift holds at every seed;",
        )
    )


def test_issue2061_shape_replay(tmp_path, monkeypatch):
    """Acceptance (b), hermetic: on the #2061 shape with the VERBATIM
    corrected-body disposition text, (a) with `f5_arm_agreement.png`
    unnamed the check WARNs class C for f5 ONLY — the seven
    glob-dispositioned f1 siblings and the explicitly-named per-cell
    companions stay silent — with ONE deduped entry listing BOTH cited
    short SHAs; (b) with f5 named in that same disposition paragraph the
    check is silent. The fixture's plan mirrors the real #2061 plans
    (which name f5 in every version, S6): it names f5, the
    `f1_delta_scatter_*` family by bounded glob, both per-cell
    companions, and `second.png` — so every figure the replay reasons
    about IS a §3.0 candidate, and the f1/f2 silences below are the BODY
    bars' doing, not de-candidation."""
    repo, sha_a, sha_b = _make_issue2061_shape_repo(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    _plan_dir_naming(
        tmp_path,
        monkeypatch,
        "f5_arm_agreement.png",
        "f1_delta_scatter_*",
        "f2_percell_base_sft_context.png",
        "f2_percell_base_sft_prefix.png",
        "second.png",
    )
    # (a) f5 unnamed: the pre-fix #2061 shape.
    body_a = _issue2061_shape_body(sha_a, sha_b, _ISSUE2061_DISPOSITION_A)
    r_a = verify_task_body.check_orphaned_per_unit_figures(body_a, issue=999)
    assert r_a.passed is True
    assert r_a.is_warn is True
    assert _COMMITTED_UNMENTIONED_CLASS in r_a.detail
    assert _PER_UNIT_NAMED_CLASS not in r_a.detail
    entry_paths = {p for p, _cls in _DETAIL_ENTRY_RE.findall(r_a.detail)}
    assert entry_paths == {_CLASS_C_PATH}  # ONLY f5 — no f1/f2 sibling fires
    assert r_a.detail.count(_CLASS_C_PATH) == 1  # deduped across cited SHAs
    assert sha_a[:8] in r_a.detail
    assert sha_b[:8] in r_a.detail
    # (b) f5 named in the same disposition paragraph: the corrected body.
    body_b = _issue2061_shape_body(
        sha_a, sha_b, f"{_ISSUE2061_DISPOSITION_A} {_ISSUE2061_DISPOSITION_B}"
    )
    r_b = verify_task_body.check_orphaned_per_unit_figures(body_b, issue=999)
    assert r_b.passed is True
    assert r_b.is_warn is False


def test_glob_family_disposition_exempts_class_c(tmp_path, monkeypatch):
    """The bounded glob bar in both directions: a backticked
    `f1_delta_scatter_*` silences all seven siblings; `f*` (under the
    3-literal-char bound), `*.png`, and an UN-backticked
    f1_delta_scatter_* each fail to exempt and all seven class-C WARNs
    still fire. The plan names all seven siblings EXPLICITLY (not by
    glob), so this pin exercises only the BODY-side glob bar — the
    plan-side glob bar is pinned separately by
    `test_class_c_plan_glob_names_family`."""
    repo, _sha0 = _make_repo_with_per_unit_orphan(tmp_path, companion=None)
    _plan_dir_naming(tmp_path, monkeypatch, *_ISSUE2061_F1_SIBLINGS)

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    figdir = repo / "figures" / "issue_999"
    for fname in _ISSUE2061_F1_SIBLINGS:
        (figdir / fname).write_bytes(b"\x89PNG fake bytes")
    git("add", "figures")
    git("commit", "-q", "-m", "add seven glob-family siblings")
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)

    def body_with(disposition: str) -> str:
        return GOOD_BODY.replace("0123456789abcdef", sha).replace(
            "The 17-pt lift holds at every seed;",
            f"{disposition} The 17-pt lift holds at every seed;",
        )

    # Backticked bounded family glob -> all seven named -> silent.
    r_ok = verify_task_body.check_orphaned_per_unit_figures(
        body_with("The seven `f1_delta_scatter_*` siblings are committed at the same pin."),
        issue=999,
    )
    assert r_ok.is_warn is False
    # Path-shaped glob: matched via its basename component -> silent.
    r_path = verify_task_body.check_orphaned_per_unit_figures(
        body_with("See `figures/issue_999/f1_delta_scatter_*` for the family."),
        issue=999,
    )
    assert r_path.is_warn is False
    # Under-anchored `f*` (1 literal char < 3) exempts nothing.
    r_short = verify_task_body.check_orphaned_per_unit_figures(
        body_with("The seven `f*` siblings are committed at the same pin."), issue=999
    )
    assert r_short.is_warn is True
    assert r_short.detail.count(_COMMITTED_UNMENTIONED_CLASS) == 7
    # Extension-only `*.png` (0 literal chars before `*`) exempts nothing.
    r_ext = verify_task_body.check_orphaned_per_unit_figures(
        body_with("All `*.png` files are committed at the same pin."), issue=999
    )
    assert r_ext.is_warn is True
    assert r_ext.detail.count(_COMMITTED_UNMENTIONED_CLASS) == 7
    # UN-backticked glob text never counts (backticks required).
    r_bare = verify_task_body.check_orphaned_per_unit_figures(
        body_with("The seven f1_delta_scatter_* siblings are committed at the same pin."),
        issue=999,
    )
    assert r_bare.is_warn is True
    assert r_bare.detail.count(_COMMITTED_UNMENTIONED_CLASS) == 7


def test_glob_named_per_unit_routes_to_phrase_bar(tmp_path, monkeypatch):
    """The disclosed per-unit loosening, both directions: a bounded glob
    naming `hero_percontext.png` WITH the 'superseded by' idiom in the
    same paragraph -> silent; the SAME glob with no idiom -> class B
    (`companion-named-not-embedded`), NOT class A."""
    repo, sha = _make_repo_with_per_unit_orphan(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body_exempt = GOOD_BODY.replace("0123456789abcdef", sha).replace(
        "The 17-pt lift holds at every seed;",
        "Round-1 exploratory `hero_perc*` views are superseded by the embedded "
        "hero panels. The 17-pt lift holds at every seed;",
    )
    r1 = verify_task_body.check_orphaned_per_unit_figures(body_exempt, issue=999)
    assert r1.passed is True
    assert r1.is_warn is False
    body_bare = GOOD_BODY.replace("0123456789abcdef", sha).replace(
        "The 17-pt lift holds at every seed;",
        "The `hero_perc*` views are committed at the same pin. The 17-pt lift holds at every seed;",
    )
    r2 = verify_task_body.check_orphaned_per_unit_figures(body_bare, issue=999)
    assert r2.passed is True
    assert r2.is_warn is True
    assert _PER_UNIT_NAMED_CLASS in r2.detail  # class B — the phrase bar, not class A
    assert "never mentioned in the body" not in r2.detail
    assert _PER_UNIT_ORPHAN_PATH in r2.detail


def test_class_c_requires_plan_named_figure(tmp_path, monkeypatch):
    """The §3.0 narrowing's load-bearing pin, both directions on ONE
    fixture: two unmentioned committed non-per-unit PNGs, one named in the
    task's plan and one not. Only the plan-named one appears as class C;
    the other is absent from the detail ENTIRELY (never a candidate)."""
    repo, _sha0 = _make_repo_with_per_unit_orphan(
        tmp_path, companion=None, extra="f5_arm_agreement.png"
    )

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    figdir = repo / "figures" / "issue_999"
    (figdir / "g7_unplanned_view.png").write_bytes(b"\x89PNG fake bytes")
    git("add", "figures")
    git("commit", "-q", "-m", "add a second, plan-unnamed figure")
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    _plan_dir_naming(tmp_path, monkeypatch, "f5_arm_agreement.png")  # g7 NOT named
    body = GOOD_BODY.replace("0123456789abcdef", sha)
    r = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r.passed is True
    assert r.is_warn is True
    assert _CLASS_C_PATH in r.detail  # plan-named -> candidate -> class C
    assert "g7_unplanned_view.png" not in r.detail  # not plan-named -> never a candidate
    assert r.detail.count(_COMMITTED_UNMENTIONED_CLASS) == 1
    # The active §3.0 mode is stated in the WARN detail (legibility).
    assert "plan-named figures only" in r.detail


def test_class_c_plan_name_matched_across_all_plan_versions(tmp_path, monkeypatch):
    """§3.0 concatenates ALL numeric plan revisions, not the `plan.md`
    symlink target: a figure named only in `v1.md` while `v3.md` (the
    symlink target) omits it is STILL a candidate — #2061's actual shape.
    Also pins the numeric-`v<int>.md` enumeration: a loose `va.md` naming
    a second committed figure is NOT read, so that figure never becomes a
    candidate."""
    repo, _sha0 = _make_repo_with_per_unit_orphan(
        tmp_path, companion=None, extra="f5_arm_agreement.png"
    )

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    figdir = repo / "figures" / "issue_999"
    (figdir / "g7_unplanned_view.png").write_bytes(b"\x89PNG fake bytes")
    git("add", "figures")
    git("commit", "-q", "-m", "add a second figure named only in a non-numeric plan file")
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    plans = tmp_path / "task999" / "plans"
    plans.mkdir(parents=True)
    (plans / "v1.md").write_text("Planned: `f5_arm_agreement.png` is the headline view.\n")
    (plans / "v3.md").write_text("Follow-up amendment: no figures promised here.\n")
    (plans / "va.md").write_text("Loose draft naming `g7_unplanned_view.png` — never read.\n")
    (plans / "plan.md").symlink_to("v3.md")
    monkeypatch.setattr(verify_task_body, "_resolve_task_plans_dir", lambda issue: plans)
    body = GOOD_BODY.replace("0123456789abcdef", sha)
    r = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r.passed is True
    assert r.is_warn is True
    assert _CLASS_C_PATH in r.detail  # named in v1.md though v3.md dropped it
    assert "g7_unplanned_view.png" not in r.detail  # va.md falls out of the numeric walk
    assert "2 plan file(s) read" in r.detail  # v1.md + v3.md; NOT va.md, NOT plan.md


def test_class_c_skipped_without_plan_context(tmp_path, monkeypatch):
    """The three §3.0 degradation paths, each fail-SOFT: class C is
    skipped (no exception, no manufactured WARN), classes A/B still fire
    normally on the same body, and the detail NAMES the skip mode (the
    §3.0 legibility promise)."""
    repo, sha = _make_repo_with_per_unit_orphan(tmp_path, extra="f5_arm_agreement.png")
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = GOOD_BODY.replace("0123456789abcdef", sha)

    # (1) issue=None (--body-stdin): no task, so no plan.
    r1 = verify_task_body.check_orphaned_per_unit_figures(body, issue=None)
    assert r1.passed is True
    assert r1.is_warn is True  # class A (per-unit orphan) still fires
    assert _PER_UNIT_ORPHAN_PATH in r1.detail
    assert _COMMITTED_UNMENTIONED_CLASS not in r1.detail
    assert _CLASS_C_PATH not in r1.detail
    assert "no issue number" in r1.detail  # the skip mode, named

    # (2) plans/ dir absent / task lookup failed (seam -> None).
    monkeypatch.setattr(verify_task_body, "_resolve_task_plans_dir", lambda issue: None)
    r2 = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r2.passed is True
    assert r2.is_warn is True
    assert _COMMITTED_UNMENTIONED_CLASS not in r2.detail
    assert _CLASS_C_PATH not in r2.detail
    assert "no plans/ directory" in r2.detail

    # (3) plan file present but unreadable (permission-masked).
    import os

    if os.geteuid() == 0:  # pragma: no cover - CI runs unprivileged
        pytest.skip("chmod-based unreadable fixture is inert as root")
    plans = tmp_path / "task999" / "plans"
    plans.mkdir(parents=True)
    v1 = plans / "v1.md"
    v1.write_text("Planned: `f5_arm_agreement.png`\n")
    v1.chmod(0o000)
    monkeypatch.setattr(verify_task_body, "_resolve_task_plans_dir", lambda issue: plans)
    r3 = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    v1.chmod(0o644)  # restore so tmp_path cleanup never trips
    assert r3.passed is True
    assert r3.is_warn is True
    assert _COMMITTED_UNMENTIONED_CLASS not in r3.detail
    assert _CLASS_C_PATH not in r3.detail
    assert "plan file unreadable (v1.md)" in r3.detail

    # Companion-free variant: with ONLY the would-be class-C figure on
    # disk, each degradation path yields a clean PASS whose detail still
    # names the skip mode (a silent class C is legible, not invisible).
    (tmp_path / "b").mkdir()
    repo2, sha2 = _make_repo_with_per_unit_orphan(
        tmp_path / "b", companion=None, extra="f5_arm_agreement.png"
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo2)
    body2 = GOOD_BODY.replace("0123456789abcdef", sha2)
    r4 = verify_task_body.check_orphaned_per_unit_figures(body2, issue=None)
    assert r4.passed is True
    assert r4.is_warn is False  # the class-C branch alone: skipped -> silent
    assert "no issue number" in r4.detail


def test_class_c_plan_glob_names_family(tmp_path, monkeypatch):
    """§3.0's "one predicate, both sides" commitment: a plan naming a
    family by BOUNDED backticked glob makes every matching sibling a
    candidate (seven class-C WARNs when the body names none), and the
    §3.1 bounds apply identically on the plan side (a plan naming only
    `f*` makes nothing a candidate)."""
    repo, _sha0 = _make_repo_with_per_unit_orphan(tmp_path, companion=None)

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    figdir = repo / "figures" / "issue_999"
    for fname in _ISSUE2061_F1_SIBLINGS:
        (figdir / fname).write_bytes(b"\x89PNG fake bytes")
    git("add", "figures")
    git("commit", "-q", "-m", "add seven glob-family siblings")
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = GOOD_BODY.replace("0123456789abcdef", sha)

    # Bounded plan-side glob -> all seven siblings are candidates.
    _plan_dir_naming(tmp_path / "p1", monkeypatch, "f1_delta_scatter_*")
    r_glob = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r_glob.is_warn is True
    assert r_glob.detail.count(_COMMITTED_UNMENTIONED_CLASS) == 7
    # Under-anchored plan-side `f*` (1 literal char < 3) names nothing.
    _plan_dir_naming(tmp_path / "p2", monkeypatch, "f*")
    r_short = verify_task_body.check_orphaned_per_unit_figures(body, issue=999)
    assert r_short.is_warn is False
    assert "plan-named figures only" in r_short.detail  # active mode, zero candidates


def test_resolve_task_plans_dir_real_body_no_monkeypatch():
    """Production-body pin for the §3.0 seam (code-style rule: one test
    executes the REAL body of a function other tests monkeypatch): with
    NO monkeypatch, `_resolve_task_plans_dir(999)` walks the real
    task_workflow registry to the REAL `tasks/completed/999/plans/` (the
    §5 anchor — also exactly why every class-C fixture above must
    monkeypatch the seam), `_plan_naming_text(999)` reads its `v1.md`,
    and the `issue=None` degradation path short-circuits to None without
    touching the registry."""
    plans = verify_task_body._resolve_task_plans_dir(999)
    assert plans is not None
    assert plans.is_dir()
    assert plans.name == "plans"
    text, mode = verify_task_body._plan_naming_text(999)
    assert text is not None
    assert "plan file(s) read" in mode
    # Hermeticity guard: every UNMONKEYPATCHED check-31 invocation in this file
    # (the 20 pre-existing tests, plus the per-unit pins that bypass the §3.0
    # filter) reads THIS real plan. Their green depends on it naming none of
    # the DISTINCTIVE fixture stems — an implicit coupling to a terminal-status
    # task's plans dir. Assert it so a future edit to task 999 fails HERE with a
    # legible message instead of flipping unrelated tests non-obviously.
    #
    # `second` (from `test_orphan_deduped_across_cited_shas`'s `second.png`) is
    # deliberately NOT guarded: task 999's plan contains the ordinary English
    # word in "seconds-long", so the stem-substring predicate matches it. That
    # is a real property of the predicate — short common-word stems collide with
    # prose — and it is harmless here in both directions: `second.png` is
    # EMBEDDED in that test's body, so it never reaches the class-C branch, and
    # corpus-wide the collision fails toward silence (a body discussing the same
    # subject matter as its plan almost always contains the same common word).
    for fixture_stem in ("hero", "f5_arm_agreement", "f6_extra", "f1_delta_scatter"):
        assert fixture_stem not in text, (
            f"task 999's real plan now names the fixture stem {fixture_stem!r}; "
            "check-31 fixtures in this file assume it names none of them"
        )
    assert verify_task_body._resolve_task_plans_dir(None) is None
    text_none, mode_none = verify_task_body._plan_naming_text(None)
    assert text_none is None
    assert "no issue number" in mode_none


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
    `_HF_TREE_BASENAMES_CACHE` (#1016), and the check-46 brace-path probe
    memoizes successful exhaustive direct-children listings in
    `_HF_DIRECT_CHILDREN_CACHE` (#1520). Clear all four before AND after
    each test so a cached verdict keyed on a (repo, sha, path) reused across
    fixtures never leaks one test's stubbed outcome into another."""
    verify_task_body._HF_EXISTENCE_CACHE.clear()
    verify_task_body._HF_TREE_FILE_COUNT_CACHE.clear()
    verify_task_body._HF_TREE_BASENAMES_CACHE.clear()
    verify_task_body._HF_DIRECT_CHILDREN_CACHE.clear()
    yield
    verify_task_body._HF_EXISTENCE_CACHE.clear()
    verify_task_body._HF_TREE_FILE_COUNT_CACHE.clear()
    verify_task_body._HF_TREE_BASENAMES_CACHE.clear()
    verify_task_body._HF_DIRECT_CHILDREN_CACHE.clear()


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
# Pattern D backtick `dir/` sub-path + count-opening paren bound to
# the nearest preceding pinned link (#1143, the #1112 footer shape),
# Pattern E trailing count-opening paren (#1422, the #1005 footer shape),
# and Pattern F listing-verified phrase-anchored count at ANY position in
# the paren after the link (#1505, the #1072 footer shape).
# All tests are offline: extractor tests need no stub; probe tests stub
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


# The verbatim claim-bearing span of #1901's footer (body.md L218, trimmed to
# the pinned link + the three backtick-token parens — the #1936 incident
# fixture): "3 activation chunks" (widened noun) and "3 chunk files"
# (modifier-separated; its post-noun paren tail runs 220 chars, over the old
# 200-char Pattern-D qualifier bound) are TRUE count claims check 30 reported
# invisible pre-#1936 ("no file-count claims adjacent"); the mid-paren
# "1 over-length-skip sidecar" continuation is a NAMED recall sacrifice.
_I1901_SHA = "0da2b0bcefa6e05e85a775b240e501b501acd344"
_I1901_FOOTER_SPAN = (
    "[issue1901_wildchat](https://huggingface.co/datasets/superkaiba1/explore-persona"
    "-space-data/tree/0da2b0bcefa6e05e85a775b240e501b501acd344/issue1901_wildchat): `"
    "manifest/` (pinned-revision stream + screen record), `final_token_capture/` (3 a"
    "ctivation chunks, ~112 MB), `raw_completions/` (3 chunk files plus 1 over-length"
    "-skip sidecar; the round plan's section 6.5 labeled the raw-completion home `fin"
    "al_token_capture/`, the realized home is the Upload-Policy-canonical `raw_comple"
    "tions/` — both populated, upload-verified)"
)


def test_hf_count_1901_footer_extracts_both_paren_opening_claims(monkeypatch):
    """T0, the #1936 incident fixture: the VERBATIM #1901 footer span yields
    EXACTLY the two paren-OPENING Pattern-D claims — "3 activation chunks"
    (widened noun) and "3 chunk files" (modifier-separated; its 220-char
    post-noun tail needs the widened 400-char qualifier bound) — each
    scoped to <link-prefix>/<sub>; the mid-paren "1 over-length-skip
    sidecar" continuation does NOT extract (the count-opens-the-paren
    anchor; exact-list assert ⇒ claim count == 2). One-sided: with >= 3
    files under each joined sub-prefix the check PASSes with no WARN."""
    body = "Footer: " + _I1901_FOOTER_SPAN + "\n"
    claims = verify_task_body._gather_hf_count_claims(body)
    repo = "superkaiba1/explore-persona-space-data"
    assert claims == [
        (
            3,
            "activation chunks",
            repo,
            "dataset",
            _I1901_SHA,
            "issue1901_wildchat/final_token_capture",
        ),
        (3, "chunk files", repo, "dataset", _I1901_SHA, "issue1901_wildchat/raw_completions"),
    ]
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    verify_task_body._HF_TREE_FILE_COUNT_CACHE.clear()
    entries = [
        {"path": f"issue1901_wildchat/final_token_capture/c{i}.pt", "type": "file"}
        for i in range(3)
    ] + [
        {"path": f"issue1901_wildchat/raw_completions/r{i}.jsonl", "type": "file"}
        for i in range(4)  # 3 claimed <= 4 files: the one-sided pass for a modifier claim
    ]
    _stub_tree(monkeypatch, status="ok", entries=entries)
    r = verify_task_body.check_hf_file_count_claims(body)
    assert r.passed and not r.is_warn


def test_hf_count_widened_noun_one_sided_pattern_e(monkeypatch):
    """T1: a widened-noun Pattern-E claim ("3 activation chunks" in the paren
    right after the pinned link) compares ONE-SIDED — claimed <= files is a
    clean PASS (the modifier restricts the counted class to a subset of the
    prefix's files); claimed > files — the folder-inflation signature —
    WARNs naming the composed modifier+noun label."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    verify_task_body._HF_TREE_FILE_COUNT_CACHE.clear()
    entries4 = [{"path": f"p/f{i}.pt", "type": "file"} for i in range(4)]
    _stub_tree(monkeypatch, status="ok", entries=entries4)
    body = "[x](https://huggingface.co/datasets/o/r/tree/abc1234def/p) (3 activation chunks)\n"
    r = verify_task_body.check_hf_file_count_claims(body)
    assert r.passed and not r.is_warn

    verify_task_body._HF_TREE_FILE_COUNT_CACHE.clear()
    entries2 = [{"path": f"p/f{i}.pt", "type": "file"} for i in range(2)]
    _stub_tree(monkeypatch, status="ok", entries=entries2)
    r2 = verify_task_body.check_hf_file_count_claims(body)
    assert r2.passed and r2.is_warn
    assert "3 activation chunks" in r2.detail and "2 file(s)" in r2.detail


def test_hf_count_modifier_separated_and_hyphenated_pattern_e(monkeypatch):
    """T2: a modifier-separated paren-OPENING claim ("3 chunk files ...")
    extracts while the MID-PAREN continuation claim ("... plus 1
    over-length-skip sidecar") does NOT (Pattern E keeps .match() — a
    mid-paren clause count would extract as a wrongly-scoped whole-prefix
    claim, the #1072 false-WARN class); a paren-OPENING hyphenated-modifier
    claim extracts; the over-claim variant WARNs one-sided."""
    url = "https://huggingface.co/datasets/o/r/tree/abc1234def/p"
    body = f"[x]({url}) (3 chunk files plus 1 over-length-skip sidecar)\n"
    claims = verify_task_body._gather_hf_count_claims(body)
    assert [(c[0], c[1]) for c in claims] == [(3, "chunk files")]
    hyph = verify_task_body._gather_hf_count_claims(f"[x]({url}) (1 over-length-skip sidecar)\n")
    assert [(c[0], c[1]) for c in hyph] == [(1, "over-length-skip sidecar")]
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    verify_task_body._HF_TREE_FILE_COUNT_CACHE.clear()
    entries2 = [{"path": f"p/f{i}.jsonl", "type": "file"} for i in range(2)]
    _stub_tree(monkeypatch, status="ok", entries=entries2)
    r = verify_task_body.check_hf_file_count_claims(f"[x]({url}) (5 chunk files)\n")
    assert r.passed and r.is_warn
    assert "5 chunk files" in r.detail and "2 file(s)" in r.detail


def test_hf_count_modifier_claims_one_sided_bare_files_two_sided(monkeypatch):
    """T3 precision guard: an inverted-semantics modifier claim ("2 missing
    files") beside a 5-file tree does NOT WARN (modifier-qualified claims
    are one-sided — at worst a real undercount goes un-WARNed, never a
    wrong WARN); a bare "5 files" claim against a 9-file tree still WARNs
    two-sided with the subset hedge (pre-#1936 behavior unchanged)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    verify_task_body._HF_TREE_FILE_COUNT_CACHE.clear()
    entries5 = [{"path": f"p/f{i}.json", "type": "file"} for i in range(5)]
    _stub_tree(monkeypatch, status="ok", entries=entries5)
    body = "Data: [p, 2 missing files](https://huggingface.co/datasets/o/r/tree/abc1234def/p)\n"
    r = verify_task_body.check_hf_file_count_claims(body)
    assert r.passed and not r.is_warn

    verify_task_body._HF_TREE_FILE_COUNT_CACHE.clear()
    entries9 = [{"path": f"p/f{i}.json", "type": "file"} for i in range(9)]
    _stub_tree(monkeypatch, status="ok", entries=entries9)
    bare = "Data: [p, 5 files](https://huggingface.co/datasets/o/r/tree/abc1234def/p)\n"
    r2 = verify_task_body.check_hf_file_count_claims(bare)
    assert r2.passed and r2.is_warn
    assert "5" in r2.detail and "9 file(s)" in r2.detail
    assert "subset of the prefix" in r2.detail


def test_hf_count_per_namespace_lookahead_survives_modifier():
    """T4: a modifier-qualified per-namespace count in A/B position ("891
    json files per namespace") stays INVISIBLE to the whole-prefix patterns
    — the narrow lookahead still declines through the modifier, and no noun
    in the alternation can absorb "json", so there is no backtrack escape —
    and to the per-namespace gatherer (its phrase regex is count-adjacent
    files-only). Zero claims, vacuous behavior as today."""
    body = (
        "- [ns1, 891 json files per namespace]"
        "(https://huggingface.co/datasets/o/r/tree/abc1234def/p)\n"
    )
    assert verify_task_body._gather_hf_count_claims(body) == []
    assert verify_task_body._gather_hf_per_namespace_claims(body) == []


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


# ─── Check 30 Pattern E: trailing count-opening paren (#1422) ───────────────
# #1005's footer carried bare trailing parentheticals immediately after its
# pinned /tree links — `[summary store + manifest](…/tree/<sha>/…) (52
# files)` — a count-OPENING paren matching none of the A-D anchors; the
# incident body claimed 51/15 where the pinned trees hold 52/17 and check 30
# reported "no file-count claims adjacent" (the miss cost a
# clean-result-critic round finding). Extractor tests are pure; probe tests
# stub `_hf_tree_get` after removing the conftest fence.

_I1005_SHA_R1 = "621b370c668d5a1df0c158aa522ef9d046c4b3c2"
_I1005_SHA_R2 = "9e79ce7f2c8126de99796b0b709b3c438756ad30"
_I1005_REPO_ID = "superkaiba1/explore-persona-space-data"
_I1005_PREFIX = "issue1005_cot_decomposition_r1"


def _i1005_footer_excerpt() -> str:
    """The two verbatim count-bearing footer sentences of the #1005 incident
    body (tasks/awaiting_promotion/1005/body.md line 185 at fix time, copied
    verbatim into this suite — never read from the live body at test time;
    the `_i1112_footer` precedent), joined by one space (the intervening
    footer prose elided). First-round sentence: five pinned links at the
    first sha — four trailing count parens + one paren-less link
    (`[per-context error tensors]`); round-2 sentence: three pinned links at
    a second sha, all with trailing count parens (the corrected live counts
    52/17 — the incident's wrong 51/15 exist only in the pre-fix history and
    are re-introduced by the mismatch test below)."""
    return (
        "HF data repo paths under prefix `issue1005_cot_decomposition_r1/`, verified live via `li"
        "st_repo_tree` at write time, revision-pinned: [rollout text](https://huggingface.co/data"
        "sets/superkaiba1/explore-persona-space-data/tree/621b370c668d5a1df0c158aa522ef9d046c4b3c"
        "2/issue1005_cot_decomposition_r1/raw_completions/thinking_rollouts) (50 files), [summary"
        " store + manifest](https://huggingface.co/datasets/superkaiba1/explore-persona-space-dat"
        "a/tree/621b370c668d5a1df0c158aa522ef9d046c4b3c2/issue1005_cot_decomposition_r1/analysis_"
        "tensors/store) (52 files), [per-context error tensors](https://huggingface.co/datasets/s"
        "uperkaiba1/explore-persona-space-data/tree/621b370c668d5a1df0c158aa522ef9d046c4b3c2/issu"
        "e1005_cot_decomposition_r1/analysis_tensors/decomp), [fit results](https://huggingface.c"
        "o/datasets/superkaiba1/explore-persona-space-data/tree/621b370c668d5a1df0c158aa522ef9d04"
        "6c4b3c2/issue1005_cot_decomposition_r1/fit_results) (17 files incl. `f2f3/` and `indiv_m"
        "lp_control/`), [driver figures](https://huggingface.co/datasets/superkaiba1/explore-pers"
        "ona-space-data/tree/621b370c668d5a1df0c158aa522ef9d046c4b3c2/issue1005_cot_decomposition"
        "_r1/figures) (87 files)."
        " "
        "HF (same prefix, revision-pinned, verified live via scoped `list_repo_tree`): [16K rollo"
        "ut text](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/9e7"
        "9ce7f2c8126de99796b0b709b3c438756ad30/issue1005_cot_decomposition_r1/raw_completions/thi"
        "nking_rollouts_16k) (50 files: 42 updated + 8 verbatim), [16K summary store](https://hug"
        "gingface.co/datasets/superkaiba1/explore-persona-space-data/tree/9e79ce7f2c8126de99796b0"
        "b709b3c438756ad30/issue1005_cot_decomposition_r1/analysis_tensors/store/percq_summaries_"
        "16k) (52 files), [16K fit results](https://huggingface.co/datasets/superkaiba1/explore-p"
        "ersona-space-data/tree/9e79ce7f2c8126de99796b0b709b3c438756ad30/issue1005_cot_decomposit"
        "ion_r1/fit_results_16k) (16 files)."
    )


def test_hf_count_extractor_trailing_paren_1005_shape():
    """Pure extractor (Pattern E, #1422): the verbatim #1005 footer excerpt
    yields EXACTLY the 7 expected claim tuples (complete 6-tuples, including
    the repo_type field) in document order. Exact equality also pins that
    the paren-less `[per-context error tensors]` link yields nothing."""
    claims = verify_task_body._gather_hf_count_claims(_i1005_footer_excerpt())
    assert claims == [
        (
            50,
            "files",
            _I1005_REPO_ID,
            "dataset",
            _I1005_SHA_R1,
            f"{_I1005_PREFIX}/raw_completions/thinking_rollouts",
        ),
        (
            52,
            "files",
            _I1005_REPO_ID,
            "dataset",
            _I1005_SHA_R1,
            f"{_I1005_PREFIX}/analysis_tensors/store",
        ),
        (17, "files", _I1005_REPO_ID, "dataset", _I1005_SHA_R1, f"{_I1005_PREFIX}/fit_results"),
        (87, "files", _I1005_REPO_ID, "dataset", _I1005_SHA_R1, f"{_I1005_PREFIX}/figures"),
        (
            50,
            "files",
            _I1005_REPO_ID,
            "dataset",
            _I1005_SHA_R2,
            f"{_I1005_PREFIX}/raw_completions/thinking_rollouts_16k",
        ),
        (
            52,
            "files",
            _I1005_REPO_ID,
            "dataset",
            _I1005_SHA_R2,
            f"{_I1005_PREFIX}/analysis_tensors/store/percq_summaries_16k",
        ),
        (16, "files", _I1005_REPO_ID, "dataset", _I1005_SHA_R2, f"{_I1005_PREFIX}/fit_results_16k"),
    ]


def test_hf_count_extractor_trailing_paren_shapes():
    """Pure extractor: minimal Pattern-E positives — bare `(9 files)`,
    singular `(1 file)`, comma-grouped `(1,234 files)`, `(9 shards)`, and
    the two trailing-content #1005 shapes (`incl.` prose, colon form)."""
    u = f"{_I931_REPO}/tree/abc1234def/p"
    shapes = [
        (f"[x]({u}) (9 files)", (9, "files")),
        (f"[x]({u}) (1 file)", (1, "file")),
        (f"[x]({u}) (1,234 files)", (1234, "files")),
        (f"[x]({u}) (9 shards)", (9, "shards")),
        (f"[x]({u}) (17 files incl. `f2f3/` and `indiv_mlp_control/`)", (17, "files")),
        (f"[x]({u}) (50 files: 42 updated + 8 verbatim)", (50, "files")),
    ]
    for body, (count, noun) in shapes:
        claims = verify_task_body._gather_hf_count_claims(body)
        assert [(c[0], c[1], c[5]) for c in claims] == [(count, noun, "p")], body


def test_hf_count_extractor_trailing_paren_negative_cases():
    """Shapes that must NOT extract via Pattern E (precision-first; each
    guards a concrete false-positive class). The per-namespace arm ALSO
    asserts the claim is not stolen from the per-namespace leg (which still
    extracts it); `(891 files per seed)` in
    test_hf_count_extractor_per_namespace_negative_cases is the standing
    tripwire mechanically forcing the WIDE `per <word>` lookahead."""
    u = f"{_I931_REPO}/tree/abc1234def/p"
    # Distributive `per <word>` declines — E never steals the per-namespace
    # leg's claims (that leg still extracts, with its own semantics).
    ns_body = f"[`ns_a/` @abc1234]({_I931_REPO}/tree/abc1234def/p) (12 files per namespace)"
    assert verify_task_body._gather_hf_count_claims(ns_body) == []
    ns_claims = verify_task_body._gather_hf_per_namespace_claims(ns_body)
    assert [(c[0], c[4]) for c in ns_claims] == [(12, "p")]
    negatives = [
        f"[x]({u}) (891 files listed per namespace)",  # per-namespace, "listed" filler
        f"[x]({u}) (11 files per adapter, 176 files total)",  # the #460 per-unit class
        f"[x]({u}) (verified live via list_repo_tree)",  # non-count paren
        f"[x]({u}) (total 515 files)",  # count does not OPEN the paren
        f"[x]({u}) ( 52 files)",  # leading space — B/D anchor parity
        f"[x]({_I931_REPO}/tree/main/p) (9 files)",  # moving ref, not hex-pinned
        f"[x]({_I931_REPO}/blob/abc1234def/p/f.json) (9 files)",  # /blob/ = single file
        "[x](https://huggingface.co/superkaiba1/explore-persona-space) (11 files)",  # no /tree/
        "[x](https://github.com/o/r/tree/abc1234def/p) (9 files)",  # non-HF link
        f"```\n[x]({u}) (9 files)\n```",  # fenced block
        f"[x]({u})\n(9 files)",  # newline gap — the iterator is same-line only
        f"[x]({u})   (9 files)",  # 3-space gap — outside the separator bound
    ]
    for body in negatives:
        assert verify_task_body._gather_hf_count_claims(body) == [], body


def test_hf_count_extractor_trailing_paren_dedup_with_pinned_phrase():
    """An at-pinned-revision paren fires BOTH Pattern C and Pattern E with
    identical `_add` args → exactly ONE tuple (the shared `seen` dedup, not
    a special case). The pathological colon form
    `(50 files: 1,234 files at the pinned revision)` fires E (50) AND C
    (1234) → TWO distinct tuples on the same prefix, each verified
    independently (the multi-claim semantics Pattern A already has)."""
    u = f"{_I931_REPO}/tree/abc1234def/p"
    one = verify_task_body._gather_hf_count_claims(f"[x]({u}) (1,234 files at the pinned revision)")
    assert [(c[0], c[1], c[5]) for c in one] == [(1234, "files", "p")]
    patho = verify_task_body._gather_hf_count_claims(
        f"[x]({u}) (50 files: 1,234 files at the pinned revision)"
    )
    assert sorted((c[0], c[1], c[5]) for c in patho) == [
        (50, "files", "p"),
        (1234, "files", "p"),
    ]


def test_hf_count_trailing_paren_b_adjacency_declines():
    """`[x](url1) (52 files): [y](url2)` → E declines (the paren is
    immediately followed by an HF markdown link — Pattern B's
    paren-before-link shape) and B ALONE extracts, scoped to url2's OWN
    prefix: exactly one tuple, never a second differently-scoped claim from
    the same paren."""
    body = (
        f"[x @abc1234]({_I931_REPO}/tree/abc1234def/p) (52 files): "
        f"[y @abc1234]({_I931_REPO}/tree/abc1234def/other)"
    )
    claims = verify_task_body._gather_hf_count_claims(body)
    assert [(c[0], c[1], c[5]) for c in claims] == [(52, "files", "other")]


def test_hf_count_trailing_paren_mismatch_warns_1005_shape(monkeypatch):
    """The #1005 incident reconstruction: the footer's `[summary store +
    manifest](…) (51 files)` claim (the incident's original understated
    count) against a stubbed pinned tree of 52 files → WARN naming both
    numbers + the subset hedge (51 < 52 may describe a subset); `passed`
    stays True (WARN-only invariant — no `passed=False` path added)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    # Explicit cache hygiene: this test and its match-passes sibling share
    # the same (repo, sha, prefix) key and `_HF_TREE_FILE_COUNT_CACHE` is
    # module-global — clear it so the probe can never be served by a
    # sibling's cached listing (the shard-claims one-sided precedent; the
    # autouse fixture also clears between tests).
    verify_task_body._HF_TREE_FILE_COUNT_CACHE.clear()
    needle = f"{_I1005_PREFIX}/analysis_tensors/store"
    entries = [{"path": needle, "type": "directory"}]
    entries += [{"path": f"{needle}/f{i}.npz", "type": "file"} for i in range(52)]
    _stub_tree(monkeypatch, status="ok", entries=entries)
    body = (
        f"[summary store + manifest]({_I931_REPO}/tree/{_I1005_SHA_R1}/"
        f"{_I1005_PREFIX}/analysis_tensors/store) (51 files)"
    )
    r = verify_task_body.check_hf_file_count_claims(body)
    assert r.passed and r.is_warn, r.detail
    assert "51" in r.detail and "52 file(s)" in r.detail
    assert "subset of the prefix" in r.detail


def test_hf_count_trailing_paren_match_passes(monkeypatch):
    """The same single-claim body with the CORRECT count `(52 files)` →
    clean PASS: no WARN, no `unverified` note, `1 claim(s) checked`."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    # Same explicit cache hygiene as the mismatch sibling above: without the
    # clear, a cached listing on the shared key could serve this probe and
    # make the PASS vacuous.
    verify_task_body._HF_TREE_FILE_COUNT_CACHE.clear()
    needle = f"{_I1005_PREFIX}/analysis_tensors/store"
    entries = [{"path": needle, "type": "directory"}]
    entries += [{"path": f"{needle}/f{i}.npz", "type": "file"} for i in range(52)]
    _stub_tree(monkeypatch, status="ok", entries=entries)
    body = (
        f"[summary store + manifest]({_I931_REPO}/tree/{_I1005_SHA_R1}/"
        f"{_I1005_PREFIX}/analysis_tensors/store) (52 files)"
    )
    r = verify_task_body.check_hf_file_count_claims(body)
    assert r.passed and not r.is_warn, r.detail
    assert "unverified" not in r.detail
    assert "1 claim(s) checked" in r.detail


# ─── Check 30 Pattern F: listing-verified phrase (#1505) ───────────────────
# #1072's footer carried, right after its pinned /tree link, the paren
# `(analysis_tensors: 16 decomposed accumulator shards + …; eval_results
# mirror; logs — 44 files, listing verified live at write time)` — "44
# files" mid-paren after an em-dash, where the pinned tree holds 40 files +
# 4 directory entries (44 = 40+4, the #931 folder-inflation signature): C's
# phrase anchors miss ("listing verified live at write time" ≠ "at the
# pinned revision") and E's .match() fails (the paren opens
# "analysis_tensors:"), so check 30 reported "no file-count claims
# adjacent". The CORRECTED live footer's 309-char paren body then exceeded
# the old 300-char `_HF_LINKTEXT_THEN_PAREN_RE` bound (widened to 600).
# Extractor tests are pure; probe tests stub `_hf_tree_get` after removing
# the conftest fence.

_I1072_SHA = "9c4258b242ad89dfa66cad18ce09d74fb5c357ad"
_I1072_REPO_ID = "superkaiba1/explore-persona-space-data"
_I1072_PREFIX = "issue1072_component_decomposition"


def _i1072_footer_incident() -> str:
    """The verbatim filing-time #1072 footer construct (FROZEN incident
    history — recover via `git log -p -- 'tasks/*/1072/body.md'` in the
    main checkout; the live body no longer carries it; never re-read at
    test time — the `_i1005_footer_excerpt` precedent). Paren body 161
    chars."""
    return (
        "HF data repo [`issue1072_component_decomposition/`](https://huggingface.co/datasets/su"
        "perkaiba1/explore-persona-space-data/tree/9c4258b242ad89dfa66cad18ce09d74fb5c357ad/iss"
        "ue1072_component_decomposition) (analysis_tensors: 16 decomposed accumulator shards + "
        "next-token id and position arrays; eval_results mirror; logs — 44 files, listing "
        "verified live at write time)."
    )


def _i1072_footer_corrected() -> str:
    """The verbatim CORRECTED live #1072 footer construct (re-probed from
    `task.py view 1072` at implementation time, 2026-07-18). Paren body
    309 chars — over the OLD 300-char bound, so even this count-OPENING
    (Pattern-E-shaped) claim went unextracted pre-#1505."""
    return (
        "HF data repo [`issue1072_component_decomposition/`](https://huggingface.co/datasets/su"
        "perkaiba1/explore-persona-space-data/tree/9c4258b242ad89dfa66cad18ce09d74fb5c357ad/iss"
        "ue1072_component_decomposition) (40 files, listing re-verified live at write time via "
        "`list_repo_tree` at the pinned revision: analysis_tensors — 25, incl. the 16 "
        "decomposed accumulator shards + next-token id and position arrays; eval_results "
        "mirror — 14, excluding the 2 VM-computed stats JSONs, which live in git only; logs — "
        "1 workload log)"
    )


def test_hf_count_extractor_listing_verified_1072_incident_shape():
    """Pure extractor (Pattern F, #1505) — THE DURABILITY PIN: the verbatim
    filing-time #1072 footer construct yields EXACTLY one claim tuple,
    bound to the LINK's whole prefix. Exact equality also pins that the
    sub-scoped clause count ("analysis_tensors: 16 decomposed accumulator
    shards …") yields nothing."""
    claims = verify_task_body._gather_hf_count_claims(_i1072_footer_incident())
    assert claims == [
        (44, "files", _I1072_REPO_ID, "dataset", _I1072_SHA, _I1072_PREFIX),
    ]


def test_hf_count_extractor_listing_verified_1072_corrected_shape():
    """Pure extractor: the verbatim CORRECTED live #1072 footer (309-char
    paren body — pins the widened 600 bound in the live shape) yields
    EXACTLY one tuple: E and F both fire on the count-OPENING
    listing-verified paren and the shared `seen` key collapses them."""
    claims = verify_task_body._gather_hf_count_claims(_i1072_footer_corrected())
    assert claims == [
        (40, "files", _I1072_REPO_ID, "dataset", _I1072_SHA, _I1072_PREFIX),
    ]


def test_hf_count_extractor_listing_verified_shapes():
    """Pure extractor: minimal Pattern-F positives on a synthetic pinned
    URL — mid-paren after an em-dash, no separator, comma / colon / dash
    separators, comma-grouped thousands, the `re-verified` form, and an
    IGNORECASE variant."""
    u = f"{_I931_REPO}/tree/abc1234def/p"
    shapes = [
        (f"[x]({u}) (…; logs — 44 files, listing verified)", 44),
        (f"[x]({u}) (prose — 44 files listing verified)", 44),
        (f"[x]({u}) (mid — 44 files — listing verified live)", 44),
        (f"[x]({u}) (44 files: listing verified)", 44),
        (f"[x]({u}) (1,234 files, listing verified)", 1234),
        (f"[x]({u}) (40 files, listing re-verified live at write time)", 40),
        (f"[x]({u}) (bulk — 44 FILES, LISTING VERIFIED)", 44),
    ]
    for body, count in shapes:
        claims = verify_task_body._gather_hf_count_claims(body)
        assert [(c[0], c[1], c[5]) for c in claims] == [(count, "files", "p")], body


def test_hf_count_extractor_listing_verified_negative_cases():
    """Shapes that must NOT extract (precision-first; each guards a named
    false-positive class from the Pattern-F recall-sacrifice list)."""
    u = f"{_I931_REPO}/tree/abc1234def/p"
    negatives = [
        # Sub-scoped clause counts with no phrase (the incident paren's own
        # "analysis_tensors: 16 …" clause never binds).
        f"[x]({u}) (analysis_tensors: 16 files; eval_results mirror)",
        # "verified" without "listing" — too weak an anchor.
        f"[x]({u}) (logs — 44 files, verified at write time)",
        # Per-<word> distributive — excluded by adjacency (the phrase must
        # directly follow `files`; the per-namespace leg keeps its own
        # semantics for the bare per-namespace form).
        f"[x]({u}) (44 files per namespace, listing verified)",
        # Shards never bind via F (files-only, C parity).
        f"[x]({u}) (bulk — 16 shards, listing verified)",
        # Clause-crossing: `;` is deliberately NOT in the separator class.
        f"[x]({u}) (analysis_tensors: 16 files; listing verified live)",
        # Moving ref / /blob/ / fenced block / newline gap — iterator guards.
        f"[x]({_I931_REPO}/tree/main/p) (logs — 44 files, listing verified)",
        f"[x]({_I931_REPO}/blob/abc1234def/p/f.json) (logs — 44 files, listing verified)",
        f"```\n[x]({u}) (logs — 44 files, listing verified)\n```",
        f"[x]({u})\n(logs — 44 files, listing verified)",
        # A NON-count-opening listing-verified paren immediately followed by
        # an HF markdown link: F declines (B-adjacency) and B cannot fire
        # (the count does not OPEN the paren) → nothing extracts.
        (
            f"[x @abc1234]({_I931_REPO}/tree/abc1234def/p) "
            f"(logs — 44 files, listing verified): "
            f"[y @abc1234]({_I931_REPO}/tree/abc1234def/other)"
        ),
    ]
    for body in negatives:
        assert verify_task_body._gather_hf_count_claims(body) == [], body


def test_hf_count_listing_verified_b_adjacency_declines():
    """`[x](url1) (44 files, listing verified): [y](url2)` → E and F both
    decline (the paren is immediately followed by an HF markdown link —
    Pattern B's paren-before-link shape) and B ALONE extracts, scoped to
    url2's OWN prefix: exactly one tuple, never a second
    differently-scoped claim from the same paren (mirrors
    test_hf_count_trailing_paren_b_adjacency_declines)."""
    body = (
        f"[x @abc1234]({_I931_REPO}/tree/abc1234def/p) (44 files, listing verified): "
        f"[y @abc1234]({_I931_REPO}/tree/abc1234def/other)"
    )
    claims = verify_task_body._gather_hf_count_claims(body)
    assert [(c[0], c[1], c[5]) for c in claims] == [(44, "files", "other")]


def test_hf_count_extractor_paren_bound_600():
    """The `_HF_LINKTEXT_THEN_PAREN_RE` paren-body bound is pinned EXACTLY
    two-sided at 600 (#1505, widened from 300 for #1072's corrected
    309-char paren): a count-opening paren whose body is exactly 600 chars
    extracts via E (previously missed above 300), and a 601-char body
    declines."""
    u = f"{_I931_REPO}/tree/abc1234def/p"
    body_600 = "9 files, " + "x" * 591
    assert len(body_600) == 600
    claims = verify_task_body._gather_hf_count_claims(f"[x]({u}) ({body_600})")
    assert [(c[0], c[1], c[5]) for c in claims] == [(9, "files", "p")]
    body_601 = "9 files, " + "x" * 592
    assert len(body_601) == 601
    assert verify_task_body._gather_hf_count_claims(f"[x]({u}) ({body_601})") == []


def test_hf_count_listing_verified_mismatch_warns_1072_shape(monkeypatch):
    """The #1072 incident reconstruction: the filing-time footer's claim of
    44 files against a stubbed pinned tree of 40 files + 4 directory
    entries → WARN naming both numbers AND the files+folders consistency
    diagnostic (44 = 40+4, the #931 folder-inflation signature); `passed`
    stays True (WARN-only invariant — no `passed=False` path added)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    # Explicit cache hygiene (the Pattern-E test precedent): this test and
    # its match-passes sibling share the same (repo, sha, prefix) key and
    # `_HF_TREE_FILE_COUNT_CACHE` is module-global.
    verify_task_body._HF_TREE_FILE_COUNT_CACHE.clear()
    entries = [
        {"path": f"{_I1072_PREFIX}/{d}", "type": "directory"}
        for d in ("analysis_tensors", "eval_results", "logs", "raw")
    ]
    entries += [
        {"path": f"{_I1072_PREFIX}/analysis_tensors/f{i}.npz", "type": "file"} for i in range(40)
    ]
    _stub_tree(monkeypatch, status="ok", entries=entries)
    r = verify_task_body.check_hf_file_count_claims(_i1072_footer_incident())
    assert r.passed and r.is_warn, r.detail
    assert "44" in r.detail and "40 file(s)" in r.detail
    assert "consistent with files+folders" in r.detail


def test_hf_count_listing_verified_match_passes(monkeypatch):
    """The CORRECTED live #1072 footer (40 files) against the same stubbed
    40-file + 4-folder tree → clean PASS: no WARN, no `unverified` note,
    `1 claim(s) checked` (the E+F co-fire collapsed to ONE claim by the
    shared `seen` key — a double count would read "2 claim(s) checked")."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    # Same explicit cache hygiene as the mismatch sibling above.
    verify_task_body._HF_TREE_FILE_COUNT_CACHE.clear()
    entries = [
        {"path": f"{_I1072_PREFIX}/{d}", "type": "directory"}
        for d in ("analysis_tensors", "eval_results", "logs", "raw")
    ]
    entries += [
        {"path": f"{_I1072_PREFIX}/analysis_tensors/f{i}.npz", "type": "file"} for i in range(40)
    ]
    _stub_tree(monkeypatch, status="ok", entries=entries)
    r = verify_task_body.check_hf_file_count_claims(_i1072_footer_corrected())
    assert r.passed and not r.is_warn, r.detail
    assert "unverified" not in r.detail
    assert "1 claim(s) checked" in r.detail


def test_check40_gatherer_ignores_listing_verified_parens():
    """Check-40 non-interaction (#1505 acceptance criterion 5): the
    unpinned-count gatherer extracts NOTHING from either #1072 construct
    (their counts are link-adjacent, not backtick-token-adjacent — between
    `](url)` and `(` there is no room for a backtick token) nor from the
    minimal Pattern-F positive shapes."""
    u = f"{_I931_REPO}/tree/abc1234def/p"
    bodies = [
        _i1072_footer_incident(),
        _i1072_footer_corrected(),
        f"[x]({u}) (…; logs — 44 files, listing verified)",
        f"[x]({u}) (44 files: listing verified)",
    ]
    for body in bodies:
        assert verify_task_body._gather_hf_unpinned_count_claims(body) == [], body


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


# ─── Check 40 (`check_hf_unpinned_count_claims`, #1433) ────────────────────
#
# The UNPINNED residue of check 30's Pattern D: backtick `dir/` +
# count-paren claims whose pinned-link binder returned None (the #1345
# footer shape) WARN for the missing /tree/<sha> pin, with a best-effort
# count resolution against the data repo at the moving ref `main`. Same
# offline conventions as the check-30/32 sections: extractor tests pure
# (no monkeypatch, no network); probe tests delenv the conftest
# EPM_VERIFY_BODY_NO_HF fence + `_stub_tree`; the autouse
# `_no_unexpected_probes` guard makes any unstubbed probe a hard error.

_HF_40_NAME = "backtick HF-path count claims carry an adjacent pinned /tree link"

# The verbatim #1345 footer sentence (body.md L189, the incident shape):
# `rejudge/` (2 files) extracts via the `issue1345_framing/...` parent
# anchor + the "HF" cue; `raw_completions/stories` (16 files) — no trailing
# slash — was the documented recall sacrifice until #1487's slashless arm
# and now extracts via the same parent anchor.
_I1345_UNPINNED_LINE = (
    "Round data on HF under `issue1345_framing/assistant_named_story/` — "
    "`raw_completions/stories` (16 files) and `rejudge/` (2 files), verified live via "
    "scoped `list_repo_tree` (stories at data-repo revision `debcdda045`)."
)

# The TRUE verbatim pre-patch #1345 story-slot-ablation footer clause —
# provenance: `git show 003078e125:tasks/followups_running/1345/body.md`
# (the shape check 40 silently passed pre-#1487): a SLASHLESS subpath
# token under the unlinked backtick parent prefix, plus a bare-paren `(8)`
# sibling claim in the same clause (the declared #1487 D3 residual).
_I1345_SLOT_ABLATION_LINE = (
    "Round data on HF under `issue1345_framing/story_slot_ablation/` — verified live via "
    "scoped listing: `analysis_tensors/turnstore` (10 files), `analysis_tensors/preds_cache` "
    "(8)."
)


def test_hf_unpinned_count_claim_warns_missing_pin_1345_shape(monkeypatch):
    """The acceptance-criterion reproduction (durability pin): the verbatim
    #1345 footer sentence produces a check-40 WARN naming the missing pin,
    the `rejudge/` token, and the resolved 2-file count at `main`; overall
    ok STAYS True (WARN never blocks); the probe URL carries the `main`
    revision (plan assumption 3, mechanically closed). Since #1487 the
    slashless `raw_completions/stories` sibling ALSO extracts — the stub
    carries its 16 files so BOTH claims read count-consistent (exercising
    the two-probe path)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    calls: list = []
    entries = [
        {"path": "issue1345_framing", "type": "directory"},
        {"path": "issue1345_framing/assistant_named_story/rejudge/r0.json", "type": "file"},
        {"path": "issue1345_framing/assistant_named_story/rejudge/r1.json", "type": "file"},
    ] + [
        {
            "path": f"issue1345_framing/assistant_named_story/raw_completions/stories/s{i}.json",
            "type": "file",
        }
        for i in range(16)
    ]
    _stub_tree(monkeypatch, status="ok", entries=entries, calls=calls)
    body = (
        _hf_body(f"{_I931_REPO}/tree/feedface/issue1345_framing")
        + "\n"
        + _I1345_UNPINNED_LINE
        + "\n"
    )
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    r = by_name[_HF_40_NAME]
    assert r.passed and r.is_warn
    assert "rejudge" in r.detail
    assert "raw_completions/stories" in r.detail  # the #1487 slashless arm
    assert "no adjacent" in r.detail and "pinned /tree/<sha> link" in r.detail
    assert "2 file(s)" in r.detail and "16 file(s)" in r.detail
    assert r.detail.count("count consistent") == 2
    assert ok  # WARN never flips overall ok
    # The count probe resolves against the data repo at the moving ref
    # `main` — `_hf_tree_url` interpolates the branch name opaquely.
    assert any(
        "/api/datasets/superkaiba1/explore-persona-space-data/tree/main/" in url
        for url, _params in calls
    )


def test_hf_tree_url_interpolates_branch_revision_opaquely():
    """`_hf_tree_url` threads a NON-hex revision (a branch name) verbatim
    into the tree-endpoint URL — nothing in the builder assumes a sha, so
    check 40's `main` resolution rides the #733 stack unchanged."""
    url = verify_task_body._hf_tree_url("o/r", "dataset", "main", "a/b")
    assert url.endswith("/api/datasets/o/r/tree/main/a%2Fb")


def test_hf_unpinned_check_skips_pin_adjacent_claims():
    """A D-shape claim WITH a binding pinned link is check 30 Pattern D's
    territory: check 40 extracts NOTHING (no double-WARN — the binder
    partitions the D-shape matches) while check 30's gatherer DOES extract
    it. Pure extractor split: zero probes (autouse guard enforces)."""
    line = (
        f"Footer: [issue1112 data]({_I931_REPO}/tree/{_I931_SHA}/issue1112_x) — "
        "`raw_completions/` (7,165 files: shards)"
    )
    r = verify_task_body.check_hf_unpinned_count_claims(line)
    assert r.passed and not r.is_warn
    assert "no unpinned" in r.detail
    assert verify_task_body._gather_hf_unpinned_count_claims(line) == []
    routed = verify_task_body._gather_hf_count_claims(line)
    assert [(c[0], c[5]) for c in routed] == [(7165, "issue1112_x/raw_completions")]


def test_hf_unpinned_check_fenced_pass_with_note(monkeypatch):
    """Under EPM_VERIFY_BODY_NO_HF=1 the check goes CHECK-LEVEL quiet: PASS
    (not WARN) with an unverified note listing the claims, ZERO probes (the
    autouse guard would raise). The check docstring states this deviation
    from the origin ask's WARN-missing-pin-only fence behavior."""
    monkeypatch.setenv("EPM_VERIFY_BODY_NO_HF", "1")
    r = verify_task_body.check_hf_unpinned_count_claims(_I1345_UNPINNED_LINE)
    assert r.passed and not r.is_warn
    assert "unverified" in r.detail and "rejudge" in r.detail


def test_hf_unpinned_check_ignores_non_hf_tokens(monkeypatch):
    """Git-side / local backtick dir tokens never fire: repo-root denylist
    (`eval_results/`, `figures/`), absolute local mounts (`/workspace/...`),
    and a cue-less token in HF-vocabulary-free prose. Vacuous PASS with
    ZERO probes even with the fence removed (autouse guard enforces)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    body = (
        "Eval JSONs: `eval_results/issue_1345/` (34 files) in git.\n"
        "Figures at `figures/issue_9/` (3 files) committed.\n"
        "Logs in `/workspace/logs/` (2 files) on the pod.\n"
        "Local scratch dir `mydir/` (4 files) kept for debugging.\n"
    )
    r = verify_task_body.check_hf_unpinned_count_claims(body)
    assert r.passed and not r.is_warn
    assert "no unpinned" in r.detail


def test_hf_unpinned_count_mismatch_warns_check30_semantics(monkeypatch):
    """A count mismatch at `main` escalates INSIDE the missing-pin WARN with
    check 30's message shape — both numbers + the subset hedge on an
    undercount, the files+folders diagnostic when claimed == files+dirs —
    plus the moving-ref hedge (the `main` listing may postdate the claim)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    entries = [{"path": f"issue99_test/sub/f{i}.json", "type": "file"} for i in range(5)] + [
        {"path": "issue99_test/sub/d0", "type": "directory"}
    ]
    _stub_tree(monkeypatch, status="ok", entries=entries)
    body = "On HF under `issue99_test/` the folder `sub/` (2 files) holds the outputs.\n"
    r = verify_task_body.check_hf_unpinned_count_claims(body)
    assert r.passed and r.is_warn
    assert "2" in r.detail and "5 file(s)" in r.detail
    assert "subset of the prefix" in r.detail
    assert "may have moved since the claim" in r.detail
    # claimed == files + folders → the files+folders diagnostic instead.
    body6 = "On HF under `issue99_test/` the folder `sub/` (6 files) holds the outputs.\n"
    r2 = verify_task_body.check_hf_unpinned_count_claims(body6)
    assert r2.passed and r2.is_warn
    assert "consistent with files+folders" in r2.detail


def test_hf_unpinned_unresolvable_claim_warns_zero_probes(monkeypatch):
    """A cue-extracted claim with NO `issue<N>_` prefix and NO parent anchor
    still WARNs for the missing pin, marked unresolvable — with ZERO probes
    (the autouse guard would raise on any GET)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    body = "Uploaded to the HF data repo: `mystery/` (3 files).\n"
    r = verify_task_body.check_hf_unpinned_count_claims(body)
    assert r.passed and r.is_warn
    assert "not resolvable against the data repo at main" in r.detail
    assert "anchor" in r.detail


def test_hf_unpinned_per_body_probe_cap(monkeypatch):
    """More unique resolved prefixes than _HF_UNPINNED_MAX_PROBES: the first
    cap-many probe (count consistent), the surplus claim KEEPS its
    missing-pin WARN with the count downgraded to a per-body-probe-cap
    note; exactly cap-many GETs are issued."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    cap = verify_task_body._HF_UNPINNED_MAX_PROBES
    n = cap + 1
    calls: list = []
    # One file under EVERY claimed prefix: the counter filters entries by
    # prefix, so each probed claim sees exactly its own 1-file listing.
    _stub_tree(
        monkeypatch,
        status="ok",
        entries=[{"path": f"issue{k}_p/f.json", "type": "file"} for k in range(n)],
        calls=calls,
    )
    body = "\n".join(f"HF data at `issue{k}_p/` (1 file) uploaded." for k in range(n)) + "\n"
    r = verify_task_body.check_hf_unpinned_count_claims(body)
    assert r.passed and r.is_warn
    assert "per-body probe cap" in r.detail
    assert r.detail.count("count consistent") == cap
    assert len(calls) == cap


def test_hf_unpinned_probe_failure_keeps_missing_pin_warn(monkeypatch):
    """A probe failure (429 / not_found — a wrong parent-anchor join, or a
    model-repo claim resolved against the hardcoded DATA repo) never drops
    the missing-pin WARN: the count half degrades to a hedged
    `count not confirmed at main` note."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    body = "Adapters on HF under `issue60_adapters/i460_D5/` (4 files).\n"
    _stub_tree(monkeypatch, status="not_found")
    r = verify_task_body.check_hf_unpinned_count_claims(body)
    assert r.passed and r.is_warn
    assert "no adjacent" in r.detail
    assert "count not confirmed at main" in r.detail and "no such revision/path" in r.detail
    verify_task_body._HF_TREE_FILE_COUNT_CACHE.clear()
    _stub_tree(monkeypatch, status="indeterminate", note="HF tree probe failed: HTTP 429")
    r2 = verify_task_body.check_hf_unpinned_count_claims(body)
    assert r2.passed and r2.is_warn
    assert "count not confirmed at main" in r2.detail and "HTTP 429" in r2.detail


def test_hf_unpinned_widened_noun_one_sided_check40_parity(monkeypatch):
    """T5, check-40 parity (#1936): a SLASHLESS issue-prefixed token with a
    widened-noun claim (`issue1901_wildchat` (3 chunks ...), unpinned)
    extracts and compares ONE-SIDED against `main` — claimed 3 < 5 files
    reads count-consistent (still a missing-pin WARN), where the old
    two-sided compare would have escalated a mismatch (and the old
    files/shards vocabulary would not have extracted the claim at all)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    verify_task_body._HF_TREE_FILE_COUNT_CACHE.clear()
    entries = [{"path": f"issue1901_wildchat/c{i}.pt", "type": "file"} for i in range(5)]
    _stub_tree(monkeypatch, status="ok", entries=entries)
    body = "Uploaded: `issue1901_wildchat` (3 chunks, sharded) on the data repo.\n"
    r = verify_task_body.check_hf_unpinned_count_claims(body)
    assert r.passed and r.is_warn
    assert "3 chunks" in r.detail
    assert "no adjacent" in r.detail and "pinned /tree/<sha> link" in r.detail
    assert "count consistent" in r.detail


def test_hf_unpinned_claim_extractor_shapes():
    """Pure extractor (no monkeypatch, no network): resolution arms +
    declines. An issue-prefixed token resolves to itself; the #1345 line
    yields TWO claims resolved via the parent-anchor join — the slashless
    `raw_completions/stories` (the pre-#1487 recall sacrifice, now covered)
    in body order before `rejudge/`; a cue-only claim extracts unresolvable
    (None); declines cover the paren-then-HF-link Pattern-B lookahead,
    fenced code blocks, the distributive `per <word>` qualifier, and
    dot-segments."""
    gather = verify_task_body._gather_hf_unpinned_count_claims
    assert gather("Uploaded `issue70_abc/raw/` (3 files).") == [
        (3, "files", "issue70_abc/raw/", "issue70_abc/raw")
    ]
    assert gather(_I1345_UNPINNED_LINE) == [
        (
            16,
            "files",
            "raw_completions/stories",
            "issue1345_framing/assistant_named_story/raw_completions/stories",
        ),
        (2, "files", "rejudge/", "issue1345_framing/assistant_named_story/rejudge"),
    ]
    assert gather("Uploaded to the HF data repo: `mystery/` (3 files).") == [
        (3, "files", "mystery/", None)
    ]
    declines = [
        # Pattern-B lookahead: paren immediately followed by an HF markdown link.
        f"HF data `x/` (3 files): [link]({_I931_REPO}/tree/abc1234/p)",
        # Fenced code block is illustrative, never a claim.
        "```\nHF `issue5_x/` (2 files)\n```",
        # Distributive qualifier: per-adapter semantics, wrong join target.
        "HF bank `issue5_x/` (3 files per adapter, 15 total).",
        # Dot-segment tokens would join to a nonexistent probe path.
        "HF corpus `../x/` (2 files) relative.",
    ]
    for body in declines:
        assert gather(body) == [], body


def test_hf_unpinned_no_failing_checkresult_in_source():
    """Committed WARN-only pin (AC6): no `CheckResult(..., False, ...)` /
    `passed=False` construction anywhere in the check-40 function or its
    gatherer — the durable form of the report-time grep (the check-32
    `test_hf_adjacent_no_failing_checkresult_in_source` convention)."""
    import ast
    import inspect

    fns = [
        verify_task_body.check_hf_unpinned_count_claims,
        verify_task_body._gather_hf_unpinned_count_claims,
        verify_task_body._hf_hub_importable,
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


def test_hf_unpinned_slashless_subpath_1345_turnstore_shape(monkeypatch):
    """THE #1487 regression fixture + durability pin: the TRUE verbatim
    pre-patch #1345 story-slot-ablation footer clause
    (`_I1345_SLOT_ABLATION_LINE`) produces a check-40 WARN naming the
    slashless `analysis_tensors/turnstore` token, the missing pin, and the
    resolved 10-file count at `main`; overall ok STAYS True. EXACTLY ONE
    claim extracts — the bare-paren `analysis_tensors/preds_cache` (8)
    sibling in the SAME clause does NOT (the #1487 D3 declared residual:
    bare-`(N)` counts stay excluded, a stated deviation from the task
    Goal's letter)."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    assert verify_task_body._gather_hf_unpinned_count_claims(_I1345_SLOT_ABLATION_LINE) == [
        (
            10,
            "files",
            "analysis_tensors/turnstore",
            "issue1345_framing/story_slot_ablation/analysis_tensors/turnstore",
        )
    ]
    entries = [
        # The directory row satisfies check 23's needle probe for the
        # `_hf_body` feedface link (the sibling 1345-shape test's shape).
        {"path": "issue1345_framing", "type": "directory"},
    ] + [
        {
            "path": f"issue1345_framing/story_slot_ablation/analysis_tensors/turnstore/t{i}.pt",
            "type": "file",
        }
        for i in range(10)
    ]
    _stub_tree(monkeypatch, status="ok", entries=entries)
    body = (
        _hf_body(f"{_I931_REPO}/tree/feedface/issue1345_framing")
        + "\n"
        + _I1345_SLOT_ABLATION_LINE
        + "\n"
    )
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    r = by_name[_HF_40_NAME]
    assert r.passed and r.is_warn
    assert "analysis_tensors/turnstore" in r.detail
    assert "no adjacent" in r.detail and "pinned /tree/<sha> link" in r.detail
    assert "10 file(s)" in r.detail and "count consistent" in r.detail
    assert "preds_cache" not in r.detail  # D3: the bare-paren `(8)` claim never extracts
    assert ok  # WARN never flips overall ok


def test_hf_unpinned_slashless_extractor_gating():
    """Pure extractor (no monkeypatch, no network), the #1487 slashless
    arm: STRONG-arm-only G4 — an issue-prefixed slashless token resolves to
    itself; a parent-anchored slashless token (incl. the riskiest
    single-segment form) resolves via the join; a cue-only slashless token
    DECLINES (no weak HF-cue arm, unlike the slashed shape); dotted final
    segments (check 32's FILE territory), git-side first segments (G3),
    and dot-segment paths decline."""
    gather = verify_task_body._gather_hf_unpinned_count_claims
    # (a) own issue<N>_ prefix — resolves to itself (already slash-free).
    assert gather("Uploaded `issue70_abc/raw` (3 files).") == [
        (3, "files", "issue70_abc/raw", "issue70_abc/raw")
    ]
    # (a') single-segment slashless under a binding parent anchor — the
    # riskiest-FP form, deliberately pinned as EXTRACTING with the parent
    # join (#1487 §13 amendment 2; the kill-criterion ≥1-`/` fallback would
    # flip this to a decline and record the narrowing in the docstring).
    assert gather("HF under `issue5_x/` — `turnstore` (10 files) verified.") == [
        (10, "files", "turnstore", "issue5_x/turnstore")
    ]
    declines = [
        # (b) cue-only slashless: the weak HF-cue arm never admits slashless.
        "Uploaded to the HF data repo: `mystery` (3 files).",
        # (c) dotted FINAL segment = a FILE claim (check 32's territory),
        # even under a binding parent anchor.
        "HF under `issue5_x/` — `analysis/pooled.pt` (3 files) verified.",
        # (d) git-side first segment under an HF parent anchor on the same
        # line — G3 applies to the slashless shape too.
        "HF under `issue9_y/` — `eval_results/issue_9` (34 files) in git.",
        # (e) dot-segment paths would join to a nonexistent probe path (the
        # leading form is regex-declined; the mid form is gate-declined).
        "HF under `issue5_x/` — `../x` (2 files) relative.",
        "HF under `issue5_x/` — `issue5_x/../y` (2 files) relative.",
    ]
    for body in declines:
        assert gather(body) == [], body


def test_hf_unpinned_slashless_pinned_adjacent_declines():
    """A PINNED-adjacent slashless claim stays out of BOTH checks' scope
    (the stated #1487 bound): check 40's gatherer declines it (G2 — no
    false missing-pin WARN when the pin IS present) AND check 30's gatherer
    still requires the trailing slash (Pattern D byte-unchanged — the cheap
    check-30-unchanged guard). Pure extractor split: zero probes (autouse
    guard enforces)."""
    line = (
        f"Footer: [issue1112 data]({_I931_REPO}/tree/{_I931_SHA}/issue1112_x) — "
        "`raw_completions` (7,165 files: shards)"
    )
    assert verify_task_body._gather_hf_unpinned_count_claims(line) == []
    assert verify_task_body._gather_hf_count_claims(line) == []
    r = verify_task_body.check_hf_unpinned_count_claims(line)
    assert r.passed and not r.is_warn
    assert "no unpinned" in r.detail


def test_hf_unpinned_slashless_multi_item_list():
    """#1487 design question 4 (§4.4): ONE parent anchor + a comma list of
    claims (one slashed, two slashless) — every item binds to the anchor
    within the 400-char bracket-free gap (earlier items' backticks/parens
    ride in the gap; they are not decline characters); an intervening
    markdown link before a later item breaks binding for it (bracket rule —
    conservative decline; slashless has no cue fallback to fall through
    to)."""
    gather = verify_task_body._gather_hf_unpinned_count_claims
    line = "Data under `issue7_p/` — `a/` (1 file), `b/c` (2 files), `d` (3 files)."
    assert gather(line) == [
        (1, "file", "a/", "issue7_p/a"),
        (2, "files", "b/c", "issue7_p/b/c"),
        (3, "files", "d", "issue7_p/d"),
    ]
    line2 = (
        "Data under `issue7_p/` — `a/` (1 file), `b/c` (2 files), "
        "[meta](https://example.com/x) and `d` (3 files)."
    )
    assert gather(line2) == [
        (1, "file", "a/", "issue7_p/a"),
        (2, "files", "b/c", "issue7_p/b/c"),
    ]


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
    """CHECKS contains 43 body-only functions: the 20 pre-v3 checks
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
    (`check_v3_word_caps`) — PLUS the FOURTEEN generation-agnostic checks:
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
    slugs / bare `H<d>` hypothesis codes / `f16`/`l16` slot-family codes —
    in the figure sidecar's rendered-text strings, #920/#1072), and
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
    defect (b), #1255), and check 40 (`check_hf_unpinned_count_claims`,
    WARN: backtick `dir/` + count-paren claims whose Pattern-D pinned-link
    binder returned None — the UNPINNED residue checks 30/32 never see —
    flag the missing /tree/<sha> pin + best-effort count resolution against
    the data repo at the moving ref `main`; incident #1345's `rejudge/`
    (2 files) footer claim, #1433), and check 41
    (`check_figure_sidecar_coverage`, WARN: same-repo sha-pinned embedded
    figures whose PNG resolves at the cited sha but whose sibling
    `.meta.json` does NOT — the sidecar-less figures checks 24/28/33/34
    silently skip under the check-24 fail-soft convention; ONE WARN per
    body naming the basenames; existence-only `git cat-file -e` probes,
    never a content read; incident #1434's 3 sidecar-less "po" figures,
    #1478), and check 50 (`check_repro_artifacts_clean`, WARN:
    `(ood_)eval_results/issue_<K>/...` dirs named in the fence-stripped
    repro region probed with a path-scoped `git status --porcelain -u` at
    the resolved repo root — untracked/modified entries WARN, probe
    failure degrades to a skip note, gitignored files excluded by default
    porcelain; incident #1768's uncommitted operator_kv result files,
    #1989). The
    migration is a RETARGET — every former check
    was kept (some dormant for a period — e.g. `check_figure_caption`,
    vacuous until #1424 tightened it) so downstream
    tests stay valid; the v3 checks PASS-skip on non-v3 bodies.

    Checks appended OUTSIDE CHECKS inside `verify_text` (they need
    something beyond the body string): the Goal soft check (needs
    frontmatter), the Lens 14 concerns-audit (needs concerns.jsonl),
    the check-16 lr-matches-plan (needs the plan), the check-17 Context
    provenance row (needs frontmatter + original-body.md), the v3
    check-21 body-Parameters-⊆-doc (needs the methodology doc path),
    the v4 check-20 word caps (needs `issue` for the events-based
    folded-round budget scaling, #921), the #732 judge-API-error
    denominator check (needs eval JSONs), the judge drop-line
    population check (#1776 incident / task #1881; same eval-JSON
    needs), and the check-31
    orphaned-per-unit-figures probe (needs `issue` for figures-dir
    scoping, #1011).
    So `verify_text` returns 69 results (2 prepended + CHECKS[1:]=53 +
    14 appended — see `test_good_body_passes_all`), but `CHECKS` stays
    at 54 (check 36 `check_v4_result_paragraph_sentences` (#1368),
    check 37 `check_footer_reuse_bullets_pinned` — the body-only
    footer-side reuse-pin sibling of check 35, #1370 — check 39
    `check_v4_sample_disclosure_count` — the Sample-slot
    `Disclosure: N of M` count reconciliation, #1421 — check 40
    `check_hf_unpinned_count_claims` (#1433) — check 41
    `check_figure_sidecar_coverage` (#1478) — checks 42
    `check_body_artifact_urls_exist` + 43
    `check_github_tree_adjacent_file_claims` (#1507) — check 44
    `check_footer_hf_paths_pinned` (#1509) — check 45
    `check_figure_caption_count_claims_vs_sidecar` (#1511) — check 46
    `check_hf_brace_expanded_path_claims` (#1520) — check 48
    `check_v4_quant_result_figure` (#1832) — check 49
    `check_v4_result_figure_cardinality` (#1879) — check 50
    `check_repro_artifacts_clean` (#1989) — check 51
    `check_v4_dropped_condition_placement` (#2017) — check 52
    `check_figure_png_sidecar_pairing` — the PNG/sidecar `render_id`
    pairing check, #2016 — and check 53
    `check_figure_sidecar_slot_completeness` — the WARN-only
    categorical-slot completeness sidecar check, #2016 — ride CHECKS).
    """
    assert len(verify_task_body.CHECKS) == 54
    # By-name membership so the NEXT check addition can key by name instead
    # of re-deriving the arithmetic (#1016 methodology-reconciler Must-Fix).
    assert verify_task_body.check_v4_dropped_condition_placement in verify_task_body.CHECKS
    assert verify_task_body.check_repro_artifacts_clean in verify_task_body.CHECKS
    assert verify_task_body.check_footer_hf_paths_pinned in verify_task_body.CHECKS
    assert verify_task_body.check_hf_adjacent_file_claims in verify_task_body.CHECKS
    assert verify_task_body.check_figure_prose_numerics_vs_sidecar in verify_task_body.CHECKS
    assert verify_task_body.check_figure_beat_claims_vs_sidecar_text in verify_task_body.CHECKS
    assert verify_task_body.check_v4_result_paragraph_sentences in verify_task_body.CHECKS
    assert verify_task_body.check_v4_quant_result_figure in verify_task_body.CHECKS
    assert verify_task_body.check_v4_result_figure_cardinality in verify_task_body.CHECKS
    assert verify_task_body.check_footer_reuse_bullets_pinned in verify_task_body.CHECKS
    assert verify_task_body.check_v4_sample_disclosure_count in verify_task_body.CHECKS
    assert verify_task_body.check_hf_unpinned_count_claims in verify_task_body.CHECKS
    assert verify_task_body.check_figure_sidecar_coverage in verify_task_body.CHECKS
    assert verify_task_body.check_figure_png_sidecar_pairing in verify_task_body.CHECKS
    assert verify_task_body.check_figure_sidecar_slot_completeness in verify_task_body.CHECKS


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


# ─── Plan-§5 conditions coverage (WARN tier, #1827; incident #1774) ─────────

# The verbatim §5 conditions table from the #1774 plan (16 backtick slugs,
# incl. the `\|` in-cell pipe escapes on the four arm rows) — the incident-era
# shape the check is calibrated on.
_I1774_CONDITIONS_PLAN = r"""# Plan — four-arm map operator characterization

## 5. Conditions and Controls

| Plain-English name | What it tests | What it controls for | Config slug |
|---|---|---|---|
| Full-context arm | E[a\|p,q] — the full-information map | — (reference arm) | `arm_context` |
| Pre-query prefix arm | E[a\|p] from the genuinely pre-query state | what the persona alone fixes | `arm_prefix_end` |
| Bare-query arm | E[a\|q] with no prefix | what the task alone fixes | `arm_bare_query` |
| Query-averaged arm | E[a\|p] from the richer averaged input | state- vs estimator-grain of the prefix estimand | `arm_query_avg` |
| Shuffled-pairing refits | chance level of every spectrum/angle/R² read | fitting-procedure artifacts (200 same-λ refits) | `null_perm` |
| Spectrum-matched angle null | chance subspace overlap | dimensionality/spectrum artifacts in angle reads | `null_procrustes` |
| Matched-n context subsample | context-arm reads at the prefix arm's effective n | rank/sample-size confound in cross-arm claims (20 draws) | `null_matchedn` |
| Identity+bias / kNN baselines | trivial-transport floor + retrieval floor per fitted map | "variance a constant shift explains" + mis-scaled maps | `base_idbias_knn` |
| Kernel-direction injection | causal inertness of discarded directions | — (the negative prediction) | `steer_kernel` |
| Top-singular injection | causal load-bearing of read directions | validates the steering rig has power | `steer_top` |
| Norm-matched random injection | generic-perturbation floor | "any direction at this norm moves things" | `steer_rand` |
| Trait-erase (LEACE) | trait-direction causal contribution | direction-specific vs generic erasure effects | `steer_erase` |
| No-intervention baseline (K=3 draws/context, same regime) | within-context cross-draw band = the H4 inertness band | decode stochasticity in Δ reads | `steer_base` |
| Pretrained-reads robustness | do channel counts/angles transfer across reading model | instruct-specific structure | `cell_pre_own` |
| λ-sweep + df(λ) | read robustness to regularization | λ-set spectrum artifacts (Round-3 concern) | `ctl_lambda` |
| Fold-jackknife | estimator dispersion of operators/eigenvalues | single-fit overreading | `ctl_jackknife` |

## 6. Evaluation

Prose after the table.
"""

_I1774_SLUGS = [
    "arm_context",
    "arm_prefix_end",
    "arm_bare_query",
    "arm_query_avg",
    "null_perm",
    "null_procrustes",
    "null_matchedn",
    "base_idbias_knn",
    "steer_kernel",
    "steer_top",
    "steer_rand",
    "steer_erase",
    "steer_base",
    "cell_pre_own",
    "ctl_lambda",
    "ctl_jackknife",
]

# Era-correct incident replay fixture (test 8b): a trimmed replica of the
# PRE-correction #1774 body at `git show 57c915206e:tasks/interpreting/1774/
# body.md` — the revision where the `cell_pre_own` (pretrained-reads
# robustness) condition was silently dropped. Load-bearing properties (pinned
# by `test_plan_conditions_fixture_has_zero_pre_own_trace`): ZERO
# `cell_pre_own` / "pretrained" mentions, plus enough v4 structure (H1,
# sentinel, >500 chars) to pass the check-0 stub short-circuit so
# `verify_text` reaches the new check.
_I1774_PRECORRECTION_BODY = """\
# Linear maps predict held-out answer states from the full context or the \
bare query but not from the genuinely pre-query prefix state, which keeps \
only persona-average signal (HIGH confidence)

<!-- clean-result-v4 -->

## Takeaways

- Held-out per-answer R² at layer 14: full context **0.812**, bare query **0.717**, pre-query prefix end 0.02, query-averaged prefix −0.02 — with every trait cell resolved at 11.7–16.8× decode-noise floors.
- The joint-fit and separately-fitted prefix operators agree far above chance (cosine 0.47 vs null 0.01) but below fold self-agreement (0.84) — so cross-arm geometry reads stay descriptive.
- The context map is high-rank: 763–2,932 held-out-validated channels depending on counting convention, refuting the expected tens-of-channels picture.
- Causal tests inconclusive: 0.92-unit additions sat at the ≈7.8 no-effect reference (under-dosed positive control); erasing single trait directions moved state 1.8–3.0× and judged behavior, with degradation not excluded.

## Goal

- **This experiment in context:** The parent fitted linear maps from four conditionings of the same conversations — full context, bare query, pre-query prefix end, query-averaged prefix — to the answer state and measured their predictive skill. This task characterizes those fitted maps as operators.
- **Broader narrative:** Whether pre-question context geometry supports a linear monitor of trait content in upcoming answers.

## Methodology

**Design:** Zero-training analysis-and-intervention experiment over a banked activation store. Fits use the corrected battery-excluded row set (17,308 rows), grouped 6-fold by prefix id, fold seed 0. Layer 14 is primary. Phases: stage audit; a K=5 decode-draw phase for the decode-noise ceiling; the fit battery; a steering phase (60 trait-stratum contexts × 27 intervention conditions plus a 3-draw no-intervention baseline = 1,800 generations); graded judging and aggregation.

## Results

### Four-arm skill ordering

Held-out R² orders context > bare query >> prefix arms at every layer read.
"""


def test_plan_conditions_slug_in_body_passes(tmp_path):
    """Test 8a: every slug appears in the body → plain PASS."""
    plan = _write_plan(
        tmp_path,
        "## Conditions\n\n"
        "| Plain-English name | What it tests | Config slug |\n"
        "|---|---|---|\n"
        "| Arm A | the effect | `cell_a` |\n"
        "| Arm B | the control | `cell_b` |\n",
    )
    body = _I1774_PRECORRECTION_BODY + "\nCells `cell_a` and `cell_b` both resolved.\n"
    result = verify_task_body.check_plan_conditions_coverage(body, plan_path=plan)
    assert result.passed and not result.is_warn, result.render()
    assert "2 plan condition(s) all covered" in result.detail


def test_plan_conditions_era_correct_1774_replay_warns(tmp_path):
    """Test 8b (the incident replay, via verify_text — pins the
    registration site): the pre-correction #1774 body + the #1774-shaped
    plan table → the new check's row WARNs naming `cell_pre_own`.
    Unrelated WARN/FAIL rows from the trimmed fixture are acceptable —
    the assertion is on the new check's row specifically."""
    plan = _write_plan_versions(tmp_path, {"v1.md": _I1774_CONDITIONS_PLAN})
    _ok, results = verify_task_body.verify_text(_I1774_PRECORRECTION_BODY, plan_path=plan)
    row = next(r for r in results if r.name == "plan conditions coverage")
    assert row.passed and row.is_warn, row.render()
    assert "cell_pre_own" in row.detail, row.render()
    assert row.detail.startswith("advisory:"), row.render()


def test_plan_conditions_fixture_has_zero_pre_own_trace():
    """Pins the 8b fixture's load-bearing property: the pre-correction
    replica carries NO trace of the dropped condition (neither the slug
    nor the plain-English name's tokens)."""
    lowered = _I1774_PRECORRECTION_BODY.lower()
    assert "cell_pre_own" not in lowered
    assert "pretrained" not in lowered


def test_plan_conditions_descope_prose_name_covers(tmp_path):
    """Test 8c (must-PASS companion): slug absent but the plain-English
    name present — the CURRENT corrected #1774 descope-prose shape
    ("the planned pretrained-reads robustness condition ... was not
    run") counts as coverage by construction."""
    plan = _write_plan(
        tmp_path,
        "## 5. Conditions and Controls\n\n"
        "| Plain-English name | What it tests | Config slug |\n"
        "|---|---|---|\n"
        "| Pretrained-reads robustness | cross-model transfer | `cell_pre_own` |\n",
    )
    body = (
        _I1774_PRECORRECTION_BODY
        + "\nOne named deviation: the planned pretrained-reads robustness condition"
        " (spectra and angles re-read with the pretrained model's activations) was"
        " not run — the omission was unintentional, and every channel-count claim"
        " is therefore instruct-reads only.\n"
    )
    result = verify_task_body.check_plan_conditions_coverage(body, plan_path=plan)
    assert result.passed and not result.is_warn, result.render()


def test_plan_conditions_no_plan_noop():
    """Test 8d: plan_path=None → NO-OP PASS."""
    result = verify_task_body.check_plan_conditions_coverage(
        _I1774_PRECORRECTION_BODY, plan_path=None
    )
    assert result.passed and not result.is_warn
    assert "no approved plan" in result.detail


def test_plan_conditions_no_table_noop(tmp_path):
    """Test 8e: no plan version carries a conditions table with a
    config-slug column → NO-OP PASS."""
    plan = _write_plan(tmp_path, "## 4. Design\n\nNo conditions table here; lr=2e-6.\n")
    result = verify_task_body.check_plan_conditions_coverage(
        _I1774_PRECORRECTION_BODY, plan_path=plan
    )
    assert result.passed and not result.is_warn
    assert "no plan version carries a conditions table" in result.detail


def test_plan_conditions_zero_slug_rows_noop(tmp_path):
    """Test 8e2: a conditions table with the config-slug column but zero
    backtick-wrapped slug rows → NO-OP PASS."""
    plan = _write_plan(
        tmp_path,
        "## Conditions\n\n"
        "| Plain-English name | Config slug |\n"
        "|---|---|\n"
        "| Arm A | bare_slug_no_backticks |\n",
    )
    result = verify_task_body.check_plan_conditions_coverage(
        _I1774_PRECORRECTION_BODY, plan_path=plan
    )
    assert result.passed and not result.is_warn
    assert "zero backtick" in result.detail


def test_plan_conditions_1774_table_parses_all_16_slugs():
    """Test 8f: the #1774-shaped 16-row table parses ALL 16 slugs —
    including the four arm rows whose cells carry `\\|` escaped pipes."""
    rows = verify_task_body._parse_plan_conditions_rows(_I1774_CONDITIONS_PLAN)
    assert rows is not None
    assert [slug for slug, _name in rows] == _I1774_SLUGS
    assert rows[13] == ("cell_pre_own", "Pretrained-reads robustness")


def test_plan_conditions_warn_never_fails(tmp_path):
    """Test 8g: an uncovered row yields is_warn=True AND passed=True —
    the check can never block."""
    plan = _write_plan(
        tmp_path,
        "## Conditions\n\n"
        "| Plain-English name | Config slug |\n"
        "|---|---|\n"
        "| Utterly unmentioned zzz-condition | `zzz_condition` |\n",
    )
    result = verify_task_body.check_plan_conditions_coverage(
        _I1774_PRECORRECTION_BODY, plan_path=plan
    )
    assert result.passed is True
    assert result.is_warn is True
    assert "zzz_condition" in result.detail


def test_plan_conditions_numeric_version_sort(tmp_path):
    """Test 8h: versions walk NEWEST-first by NUMERIC suffix — v10's
    table binds over v2's (a lexicographic reverse sort would order
    'v2.md' > 'v10.md' and bind v2's covered table → PASS, masking the
    v10 drop this asserts)."""
    covered_table = (
        "## Conditions\n\n"
        "| Plain-English name | Config slug |\n"
        "|---|---|\n"
        "| Old arm | `slug_old` |\n"
    )
    newer_table = (
        "## Conditions\n\n"
        "| Plain-English name | Config slug |\n"
        "|---|---|\n"
        "| New arm | `slug_new` |\n"
    )
    plan = _write_plan_versions(tmp_path, {"v2.md": covered_table, "v10.md": newer_table})
    body = _I1774_PRECORRECTION_BODY + "\nThe `slug_old` cell resolved.\n"
    result = verify_task_body.check_plan_conditions_coverage(body, plan_path=plan)
    assert result.passed and result.is_warn, result.render()
    assert "slug_new" in result.detail, result.render()


def test_plan_conditions_amendment_falls_back_to_prior_version(tmp_path):
    """A follow-up amendment plan with NO conditions table falls back to
    the newest PRIOR version that carries one (plan criterion 3)."""
    plan = _write_plan_versions(
        tmp_path,
        {
            "v1.md": (
                "## Conditions\n\n"
                "| Plain-English name | Config slug |\n"
                "|---|---|\n"
                "| Old arm | `slug_old` |\n"
            ),
            "v2.md": "## Follow-up amendment\n\nAnalysis-only round; no conditions table.\n",
        },
    )
    body = _I1774_PRECORRECTION_BODY + "\nThe `slug_old` cell resolved.\n"
    result = verify_task_body.check_plan_conditions_coverage(body, plan_path=plan)
    assert result.passed and not result.is_warn, result.render()
    assert "1 plan condition(s) all covered" in result.detail


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


def test_v3_unwrapped_data_table_sub_tag_code_warns():
    """Check 19b: an `H1c`-form sub-tag code in a bare `## Data` table
    cell WARNs — mirror-sync with the audit's widened `condition_labels`
    pattern (single optional lowercase sub-tag letter, #1914)."""
    bare_table = (
        "Per-hypothesis row counts (2 of 2,000 rows shown for illustration):\n\n"
        "| Hypothesis | Rows | Note |\n"
        "|---|---|---|\n"
        "| H1c | 1000 | sub-hypothesis arm |\n"
        "| H4b | 1000 | sub-hypothesis arm |\n\n"
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
    assert "H1c" in warn.detail


def test_v3_unwrapped_data_table_plural_h2s_does_not_warn():
    """Check 19b: a plural markdown-heading form (`H2s`) in a bare table
    cell does NOT WARN — the widened sub-tag letter class excludes `s`
    (measured false-positive class, #1914)."""
    bare_table = (
        "Heading forms used (full breakdown):\n\n"
        "| Heading form | Count |\n"
        "|---|---|\n"
        "| Three H2s total | 3 |\n"
        "| legacy H2s | 5 |\n\n"
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
    assert warn.passed and not warn.is_warn, warn.render()


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


# ─── Check 39: Sample-slot `Disclosure: N of M` count reconciliation (#1421) ─
# Incident #1005: the clean-result claimed `Disclosure: 8 of 2,400` while only
# 6 example bullets followed (pre-fix revision preserved at 8926c25db5).
# Assertions call the check directly (plus one verify_text by-name read) —
# the `_V4_GOOD_BODY` convention: fake SHAs fail the existence probes, so
# overall PASS is not assertable.

_CHECK39_NAME = "Sample-slot disclosure count (v4)"

# Six top-level bullets structurally copied from #1005's Sample slot shape
# (markdown links + em-dashes, no fenced blocks, no tables).
_SIX_I1005_STYLE_BULLETS = (
    "- [naturalistic_s57](https://huggingface.co/datasets/x/tree/abc/rc/s57.json) — "
    "well-formed single-turn rollout; direct answer to the probe.\n"
    "- [naturalistic_s61](https://huggingface.co/datasets/x/tree/abc/rc/s61.json) — "
    "well-formed; hedged framing.\n"
    "- [naturalistic_s74](https://huggingface.co/datasets/x/tree/abc/rc/s74.json) — "
    "well-formed; list-style answer.\n"
    "- [chat_s12](https://huggingface.co/datasets/x/tree/abc/rc/s12.json) — "
    "well-formed multi-turn rollout.\n"
    "- [chat_s19](https://huggingface.co/datasets/x/tree/abc/rc/s19.json) — "
    "well-formed; short reply.\n"
    "- [chat_s23](https://huggingface.co/datasets/x/tree/abc/rc/s23.json) — "
    "cap-truncated rollout (ends mid-sentence).\n"
)


def _v4_body_with_sample_slot(slot_md: str) -> str:
    """Return a `_V4_GOOD_BODY` variant whose `## Methodology` Sample-slot
    CONTENT (everything after the slot label line, up to `## Results`) is
    replaced by ``slot_md``. The slot label line itself stays."""
    start = _V4_GOOD_BODY.index("<details open>")
    end = _V4_GOOD_BODY.index("## Results")
    return _V4_GOOD_BODY[:start] + slot_md.rstrip() + "\n\n" + _V4_GOOD_BODY[end:]


def test_v4_sample_disclosure_count_overclaim_fails():
    """Durability pin + the real-#1005-shape regression: a Sample slot
    claiming `Disclosure: 8 of 2,400` over exactly 6 top-level bullets
    (the pre-fix #1005 revision, `8926c25db5`) FAILs, naming 8 (claimed)
    and 6 (shown). The disclosure line's own em-dash tail (`— 5
    well-formed … plus 3 cap-truncated`) never enters the counted group
    (group starts at the first newline after the claim)."""
    slot = (
        "Disclosure: 8 of 2,400 rollouts — 5 well-formed responses plus 3 "
        "cap-truncated, drawn at random for illustration.\n\n" + _SIX_I1005_STYLE_BULLETS
    )
    res = verify_task_body.check_v4_sample_disclosure_count(_v4_body_with_sample_slot(slot))
    assert res.passed is False
    assert "claims 8 shown" in res.detail
    assert "at most 6" in res.detail


def test_v4_sample_disclosure_count_match_passes():
    """The #928 / post-fix-#1005 shape: `Disclosure: 6 of 2,400` over 6
    top-level bullets reconciles → PASS, no WARN."""
    slot = "Disclosure: 6 of 2,400 rollouts, drawn at random.\n\n" + _SIX_I1005_STYLE_BULLETS
    res = verify_task_body.check_v4_sample_disclosure_count(_v4_body_with_sample_slot(slot))
    assert res.passed is True
    assert res.is_warn is False
    assert "reconcile" in res.detail


def test_v4_sample_disclosure_no_count_claim_skips():
    """A Sample slot disclosed via the broad check-19 vocabulary only
    (`5 of 2,000 rows, random sample` inside a `<summary>`, the
    `_V4_GOOD_BODY` shape — no `Disclosure:` keyword) does not engage the
    check; the shared good fixture stays green under verify_text."""
    res = verify_task_body.check_v4_sample_disclosure_count(_V4_GOOD_BODY)
    assert res.passed is True
    assert res.is_warn is False
    assert "no `Disclosure: N of M` count claim" in res.detail
    _ok, results = verify_task_body.verify_text(_V4_GOOD_BODY)
    by_name = _results_by_name(results)
    assert by_name[_CHECK39_NAME].passed
    assert not by_name[_CHECK39_NAME].is_warn


def test_v4_sample_disclosure_thousands_separator_parses():
    """`Disclosure: 1,000 of 2,400` parses N=1000 (thousands separators on
    both sides), and the M-side separator does not break the incident
    parse (8 of 2,400 + 6 bullets still FAILs)."""
    m = verify_task_body._DISCLOSURE_COUNT_RE.search("Disclosure: 1,000 of 2,400 rows shown.")
    assert m is not None
    assert int(m.group(1).replace(",", "")) == 1000
    assert m.group(2) == "2,400"
    slot = "Disclosure: 8 of 2,400 rollouts, drawn at random.\n\n" + _SIX_I1005_STYLE_BULLETS
    res = verify_task_body.check_v4_sample_disclosure_count(_v4_body_with_sample_slot(slot))
    assert res.passed is False


def test_v4_sample_disclosure_table_rows_basis_passes():
    """`Disclosure: 5 of 2,000 training rows` in a `<details><summary>`
    whose table carries 5 data rows PASSes via the table-row basis (the
    one-table-of-N-rows form must not false-FAIL)."""
    slot = (
        "<details>\n"
        "<summary>Disclosure: 5 of 2,000 training rows, random sample</summary>\n\n"
        "| Row | System | User | Assistant |\n"
        "|---|---|---|---|\n"
        "| 1 | sys | q | a |\n"
        "| 2 | sys | q | a |\n"
        "| 3 | sys | q | a |\n"
        "| 4 | sys | q | a |\n"
        "| 5 | sys | q | a |\n\n"
        "</details>\n"
    )
    res = verify_task_body.check_v4_sample_disclosure_count(_v4_body_with_sample_slot(slot))
    assert res.passed is True
    assert res.is_warn is False


def test_v4_sample_disclosure_undercount_warns():
    """An UNDERCOUNT (`Disclosure: 2 of 2,400` over 4 bullets) is a
    mismatch but not an overclaim → WARN (`is_warn=True`), never FAIL."""
    four_bullets = "".join(_SIX_I1005_STYLE_BULLETS.splitlines(keepends=True)[:4])
    slot = "Disclosure: 2 of 2,400 rollouts, drawn at random.\n\n" + four_bullets
    res = verify_task_body.check_v4_sample_disclosure_count(_v4_body_with_sample_slot(slot))
    assert res.passed is True
    assert res.is_warn is True
    assert "not an overclaim" in res.detail


def test_v4_sample_disclosure_unrecognized_presentation_skips():
    """A count claim followed only by plain paragraphs (no bullets /
    blocks / tables / quotes) has no countable basis → PASS-skip (never
    guess)."""
    slot = (
        "Disclosure: 3 of 500 examples, described narratively below.\n\n"
        "A plain paragraph describing the worked examples in prose without "
        "any bullets, tables, fenced blocks, or blockquotes.\n"
    )
    res = verify_task_body.check_v4_sample_disclosure_count(_v4_body_with_sample_slot(slot))
    assert res.passed is True
    assert res.is_warn is False
    assert "no countable presentation" in res.detail


def test_v3_body_disclosure_line_not_checked():
    """Grandfathering regression: a v3 body carrying a `Disclosure: 8 of
    2,400` line (with example bullets) PASS-skips — the check is v4-only
    by construction."""
    body = _V3_GOOD_BODY + "\nDisclosure: 8 of 2,400 rollouts.\n\n- one example\n- two example\n"
    res = verify_task_body.check_v4_sample_disclosure_count(body)
    assert res.passed is True
    assert "not a v4 body" in res.detail


def test_v4_sample_disclosure_two_groups_second_overclaim_fails():
    """TWO `Disclosure:` lines in one Sample slot: the first group
    reconciles (3 over 3 bullets), the second overclaims (9 over 4
    bullets) → FAIL naming the SECOND group's numbers. Also pins the
    group boundary: the second claim's own bolded-bullet prefix
    (`- **Disclosure: …`) never counts into the FIRST group (each group
    ends at the START of the line carrying the next claim)."""
    slot = (
        "Disclosure: 3 of 2,400 rollouts, drawn at random.\n\n"
        "- [a](https://x/a) — first rollout\n"
        "- [b](https://x/b) — second rollout\n"
        "- [c](https://x/c) — third rollout\n\n"
        "- **Disclosure: 9 of 2,400 judge rows, drawn at random.**\n\n"
        "- [d](https://x/d) — fourth\n"
        "- [e](https://x/e) — fifth\n"
        "- [f](https://x/f) — sixth\n"
        "- [g](https://x/g) — seventh\n"
    )
    res = verify_task_body.check_v4_sample_disclosure_count(_v4_body_with_sample_slot(slot))
    assert res.passed is False
    assert "claims 9 shown" in res.detail
    assert "at most 4" in res.detail
    # The reconciling FIRST group (3 == 3) is not flagged.
    assert "Disclosure: 3 of" not in res.detail


def test_v4_sample_disclosure_bulleted_claim_excludes_own_bullet():
    """A `Disclosure:` claim nested inside a bullet (`- Disclosure: 6 of
    2,400 …`) excludes its OWN bullet line from the group count: 6
    following bullets reconcile at exactly 6 (a 7-count would WARN)."""
    slot = "- Disclosure: 6 of 2,400 rollouts, drawn at random:\n" + _SIX_I1005_STYLE_BULLETS
    res = verify_task_body.check_v4_sample_disclosure_count(_v4_body_with_sample_slot(slot))
    assert res.passed is True
    assert res.is_warn is False


def test_v4_sample_disclosure_mixed_media_summed_basis_passes():
    """SUMMED disjoint-media basis: a mixed-media group showing 3 table
    data rows + 3 fenced completions for `Disclosure: 6 of 2,400` PASSes
    via the summed basis (3 blocks + 3 out-of-block rows = 6) even though
    no single-medium basis equals 6 (strictly generosity-increasing)."""
    fence = "```text\nUser: q?\nAssistant: a.\n```\n"
    slot = (
        "Disclosure: 6 of 2,400 rows and completions, drawn at random.\n\n"
        "| Row | System | User |\n"
        "|---|---|---|\n"
        "| 1 | sys | q |\n"
        "| 2 | sys | q |\n"
        "| 3 | sys | q |\n\n" + fence + "\n" + fence + "\n" + fence
    )
    res = verify_task_body.check_v4_sample_disclosure_count(_v4_body_with_sample_slot(slot))
    assert res.passed is True
    assert res.is_warn is False


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


# ─── Check 17 (parent-lineage cross-check, #1418) ──────────────────────────
# Unit-grain cases call `check_repro_context_provenance(body, fm)` directly
# (fm injection is the variable, per the #1068 convention above); one
# end-to-end case runs `verify_text` to pin the `parent_id` fm threading.


def test_v4_context_denied_parent_with_parent_id_fails():
    """Primary regression — the #1345 r1 incident shape (and this
    change's durability pin): a `fresh direction (no parent task)`
    lineage on a body whose frontmatter carries `parent_id` is a hard
    v4 FAIL naming the parent id."""
    body = _v4_body_with_lineage("- Lineage: fresh direction (no parent task).\n")
    ctx = verify_task_body.check_repro_context_provenance(body, {"parent_id": 825})
    assert not ctx.passed
    assert "825" in ctx.detail
    assert "contradiction" in ctx.detail


def test_v4_context_no_parent_short_form_with_parent_id_fails():
    """The `no parent` regex alternate: `fresh (no parent)` with
    `parent_id` set and the parent unreferenced is a hard v4 FAIL."""
    body = _v4_body_with_lineage("- Lineage: fresh (no parent).\n")
    ctx = verify_task_body.check_repro_context_provenance(body, {"parent_id": 825})
    assert not ctx.passed
    assert "context-parent-lineage-contradiction" in ctx.detail


def test_v4_context_denied_claim_ignorecase_fails():
    """The denied-claim regex is case-insensitive (mirrors the parent
    lineage-token regex's IGNORECASE): `Fresh Direction (No Parent)`
    still tier-1 FAILs."""
    body = _v4_body_with_lineage("- Lineage: Fresh Direction (No Parent) rescope.\n")
    ctx = verify_task_body.check_repro_context_provenance(body, {"parent_id": 825})
    assert not ctx.passed
    assert "context-parent-lineage-contradiction" in ctx.detail


def test_v4_context_denied_claim_but_parent_named_warns_not_fails():
    """Tier 2: a denied-parent clause ALONGSIDE a `#<parent_id>`
    reference is internally contradictory but not the reader-misleading
    incident shape — WARN, never FAIL."""
    body = _v4_body_with_lineage("- Lineage: fresh direction (no parent); reuses #825 artifacts.\n")
    ctx = verify_task_body.check_repro_context_provenance(body, {"parent_id": 825})
    assert ctx.passed and ctx.is_warn, ctx.detail
    assert "context-parent-lineage-mixed" in ctx.detail


def test_v4_context_parent_named_with_parent_id_passes():
    """Tier 4 (the 32/32 committed-corpus shape): the canonical
    `[#34](...)` lineage bullet with `parent_id: 34` PASSes with the
    byte-identical clean-PASS detail (no new WARN)."""
    ctx = verify_task_body.check_repro_context_provenance(_V4_GOOD_BODY, {"parent_id": 34})
    assert ctx.passed and not ctx.is_warn, ctx.detail
    assert ctx.detail == "**Context:** row present with lineage token"


def test_v4_context_parent_unnamed_warns():
    """Tier 3: `parent_id` set, lineage cites OTHER issues but never the
    parent → WARN naming the parent id (never a FAIL — legitimate
    grandparent / re-scoped lineages exist)."""
    body = _v4_body_with_lineage("- Child of #722 (context pool), method parent #779.\n")
    ctx = verify_task_body.check_repro_context_provenance(body, {"parent_id": 658})
    assert ctx.passed and ctx.is_warn, ctx.detail
    assert "context-parent-unnamed" in ctx.detail
    assert "#658" in ctx.detail


def test_v4_context_parent_named_via_dashboard_url_passes():
    """The `/tasks/K` alternative: a dashboard-URL-only parent reference
    satisfies the `named` escape (deleting the `/tasks/{pid}` alternate
    would demote this to warn-unnamed and fail this test). The bullet
    keeps a `#722` ref so the PRE-EXISTING lineage-token sub-check —
    which `/tasks/K` alone does not satisfy — stays green."""
    body = _v4_body_with_lineage(
        "- Lineage: [parent task](https://eps.superkaiba.com/tasks/34) — extends #722's question.\n"
    )
    ctx = verify_task_body.check_repro_context_provenance(body, {"parent_id": 34})
    assert ctx.passed and not ctx.is_warn, ctx.detail
    assert ctx.detail == "**Context:** row present with lineage token"


def test_v4_context_denied_claim_in_blockquote_only_not_flagged():
    """Scan-region consistency (#959): a denied-parent claim ONLY inside
    the blockquoted verbatim prompt never counts — the lineage line
    names the parent, so the row PASSes with no WARN."""
    body = _v4_body_with_context_quote(
        "- Originating prompt, verbatim:\n\n> treat this as a fresh direction (no parent) rerun\n"
    )
    ctx = verify_task_body.check_repro_context_provenance(body, {"parent_id": 34})
    assert ctx.passed and not ctx.is_warn, ctx.detail
    assert ctx.detail == "**Context:** row present with lineage token"


def test_v4_context_label_in_blockquote_fallback_scans_whole_footer():
    """Degenerate label-fallback region: when the `**Context:**` label is
    findable only inside a blockquote line, `ctx_scan` falls back to the
    WHOLE stripped footer — the parent-lineage cross-check operates on
    that fallback region, so a non-blockquote denied claim + `parent_id`
    still tier-1 FAILs there."""
    context_block = (
        "**Context:**\n"
        "- Created 2026-06-24; run executed 2026-06-24.\n"
        + _V4_LINEAGE_BULLET
        + "- Originating prompt: origin prompt not recorded\n"
    )
    assert context_block in _V4_GOOD_BODY, "fixture drifted"
    body = _V4_GOOD_BODY.replace(
        context_block,
        "> **Context:** quoted example heading\n\n"
        "- Lineage note: fresh direction (no parent task).\n"
        "- Originating prompt: origin prompt not recorded\n",
    )
    ctx = verify_task_body.check_repro_context_provenance(body, {"parent_id": 825})
    assert not ctx.passed, ctx.detail
    assert "context-parent-lineage-contradiction" in ctx.detail


def test_v3_context_denied_parent_with_parent_id_warns_not_fails():
    """Grandfathering pin: the SAME denied-parent contradiction that
    hard-FAILs a v4 body is WARN-only on a v3 body (never a new hard
    FAIL below the v4 sentinel)."""
    assert _V4_LINEAGE_BULLET in _V3_GOOD_BODY, "fixture drifted"
    body = _V3_GOOD_BODY.replace(
        _V4_LINEAGE_BULLET, "- Lineage: fresh direction (no parent task).\n"
    )
    ctx = verify_task_body.check_repro_context_provenance(body, {"parent_id": 825})
    assert ctx.passed is True and ctx.is_warn, ctx.detail
    assert "context-parent-lineage-contradiction" in ctx.detail


def test_v4_context_parent_id_string_coerced():
    """Defensive typing: a string `parent_id` (some fm sources round-trip
    ints as strings) behaves identically to the int form."""
    body = _v4_body_with_lineage("- Lineage: fresh direction (no parent task).\n")
    ctx = verify_task_body.check_repro_context_provenance(body, {"parent_id": "825"})
    assert not ctx.passed
    assert "context-parent-lineage-contradiction" in ctx.detail


def test_v4_context_no_parent_id_noop_unchanged():
    """Noop branch: with NO frontmatter `parent_id`, a denied-parent
    claim is the sanctioned parentless lineage form — byte-identical
    pre-#1418 clean-PASS detail (also re-covered by the untouched
    `test_v4_context_fresh_direction_no_parent_passes`)."""
    body = _v4_body_with_lineage("- Lineage: fresh direction (no parent task).\n")
    ctx = verify_task_body.check_repro_context_provenance(body, {})
    assert ctx.passed and not ctx.is_warn, ctx.detail
    assert ctx.detail == "**Context:** row present with lineage token"


def test_v4_context_multi_warn_join_shape():
    """`warn-unnamed` + `warn-mismatch` co-firing accumulate into ONE
    WARN result, `"; "`-joined, parent tier first — pins the multi-warn
    detail shape."""
    body = _v4_body_with_lineage("- Child of #722 (context pool).\n").replace(
        _V4_NOT_RECORDED_LINE,
        "- Originating prompt, verbatim:\n\n"
        "> run the seed sweep for the X effect and summarize the deltas\n",
    )
    ctx = verify_task_body.check_repro_context_provenance(
        body, {"parent_id": 658, "origin_prompt": _OP}
    )
    assert ctx.passed and ctx.is_warn, ctx.detail
    assert ctx.detail.startswith("**Context:** row present with lineage token; ")
    assert "context-parent-unnamed" in ctx.detail
    assert "; context-origin-prompt-mismatch" in ctx.detail  # the join separator
    assert ctx.detail.index("context-parent-unnamed") < ctx.detail.index(
        "context-origin-prompt-mismatch"
    )


def test_verify_text_threads_parent_id_from_frontmatter():
    """End-to-end fm plumbing: `parent_id` spliced into the fixture's
    EXISTING frontmatter block + a denied-parent lineage → the named
    check FAILs through `verify_text`."""
    body = _v4_body_with_lineage("- Lineage: fresh direction (no parent task).\n").replace(
        "kind: experiment\n", "kind: experiment\nparent_id: 99\n"
    )
    assert "parent_id" in body, "fixture replacement did not land"
    _ok, results = verify_task_body.verify_text(body)
    ctx = _results_by_name(results)[_CONTEXT_CHECK]
    assert not ctx.passed
    assert "context-parent-lineage-contradiction" in ctx.detail
    assert "99" in ctx.detail


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


# ─── check 20 (v4): WARN-acknowledgment coverage (#1523; incident #1417) ───
#
# A body that ships check-20 WARNs under an acknowledgment sentence must name
# EACH fired WARN-tier class; a fired-but-unnamed class appends one more WARN
# (never flips the verdict). Acknowledgment fixtures are appended AFTER the
# footer (past `**Context:**`) so they perturb no prose count, and carry no
# task refs / URLs (keeps the other checks inert).

_V4_41_WORD_BULLET_BODY = _V4_GOOD_BODY.replace(
    "- Secondary: capability holds at 0.82 vs baseline 0.81 — no regression at 25% mixing.",
    "- " + " ".join(["word"] * 41),
)


def test_v4_warn_ack_partial_coverage_warns():
    """The #1417 shape: a 41-word bullet WARN fires while the acknowledgment
    names only the per-result band — check 20 appends a coverage WARN naming
    `Takeaways bullet-length` as unnamed; the overall verdict stays PASS."""
    body = (
        _V4_41_WORD_BULLET_BODY + "\nVerifier WARNs acknowledged: per-result prose sits in the "
        "120–180-word band (deliberate).\n"
    )
    ok, results = verify_task_body.verify_text(body)
    caps = _results_by_name(results)["v4 conciseness caps"]
    assert caps.passed and caps.is_warn
    assert "does not name" in caps.detail
    assert "Takeaways bullet-length" in caps.detail
    assert ok, [r.render() for r in results if not r.passed]


def test_v4_warn_ack_fully_named_no_coverage_warn():
    """Fired + fully named: the underlying 30-word WARN remains but NO
    coverage message is appended."""
    body = (
        _V4_41_WORD_BULLET_BODY
        + "\nVerifier WARNs acknowledged: one overlong Takeaways bullet is deliberate.\n"
    )
    _ok, results = verify_task_body.verify_text(body)
    caps = _results_by_name(results)["v4 conciseness caps"]
    assert caps.passed and caps.is_warn
    assert "exceed 30 words" in caps.detail
    assert "does not name" not in caps.detail


def test_v4_warn_fired_no_ack_unchanged():
    """Fired + no acknowledgment: behavior byte-compatible with today — the
    cap WARN alone, no coverage message."""
    _ok, results = verify_task_body.verify_text(_V4_41_WORD_BULLET_BODY)
    caps = _results_by_name(results)["v4 conciseness caps"]
    assert caps.passed and caps.is_warn
    assert "exceed 30 words" in caps.detail
    assert "does not name" not in caps.detail


def test_v4_ack_present_nothing_fired_no_warn():
    """Acknowledgment present on a clean body (nothing fired): no new WARN."""
    body = (
        _V4_GOOD_BODY
        + "\nVerifier WARNs acknowledged: none currently fire; kept for the template.\n"
    )
    _ok, results = verify_task_body.verify_text(body)
    caps = _results_by_name(results)["v4 conciseness caps"]
    assert caps.passed and not caps.is_warn, caps.render()


def test_v4_ack_inside_fence_not_detected():
    """An acknowledgment sentence inside a code fence (e.g. quoted verifier
    output) is NOT detected as an acknowledgment — no coverage message."""
    body = (
        _V4_41_WORD_BULLET_BODY
        + "\n```\nVerifier WARNs acknowledged: per-result prose band (quoted).\n```\n"
    )
    _ok, results = verify_task_body.verify_text(body)
    caps = _results_by_name(results)["v4 conciseness caps"]
    assert caps.passed and caps.is_warn
    assert "does not name" not in caps.detail


def test_v4_fail_tier_excluded_from_ack_matching():
    """A 100-word bullet is a FAIL-tier class (blocks regardless of
    acknowledgment) and is EXCLUDED from coverage matching: with no WARN-tier
    class fired, an acknowledgment naming nothing appends no coverage
    message; the check still FAILs (unchanged)."""
    body = _V4_GOOD_BODY.replace(
        "- Secondary: capability holds at 0.82 vs baseline 0.81 — no regression at 25% mixing.",
        "- " + " ".join(["word"] * 100),
    ) + ("\nVerifier WARNs acknowledged: nothing specific.\n")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    caps = _results_by_name(results)["v4 conciseness caps"]
    assert not caps.passed
    assert "does not name" not in caps.detail


def test_v3_body_ack_coverage_not_applied():
    """Forward-only: a v3 body with a 35-word bullet WARN and a PARTIAL
    acknowledgment gains NO coverage message (`check_v3_word_caps` is
    byte-untouched; the sub-check is v4-only)."""
    body = _V3_GOOD_BODY.replace(
        "- Secondary: capability holds at 0.82 vs baseline 0.81 — no regression at 25% mixing.",
        "- " + " ".join(["word"] * 35),
    ) + (
        "\nVerifier WARNs acknowledged: per-result prose sits in the "
        "120–180-word band (deliberate).\n"
    )
    _ok, results = verify_task_body.verify_text(body)
    caps = _results_by_name(results)["v3 conciseness caps"]
    assert caps.passed and caps.is_warn
    assert "exceed 30 words" in caps.detail
    assert "does not name" not in caps.detail


def test_v4_warn_ack_family_arm_detection():
    """Pins the no-"warn" family-detection arm of `_V4_WARN_ACK_FAMILY_RE`
    AND its interaction with per-class matching (the #922/#1315 in-wild
    shape): an acknowledgment with NO "warn" token is still detected via the
    conciseness-family arm, names total-prose but not the fired bullet class,
    so the coverage WARN fires naming `Takeaways bullet-length`. Without this
    row all sibling tests pass with the family arm mis-implemented, and the
    corpus scan is structurally blind to a family-arm miss."""
    body = (
        _V4_41_WORD_BULLET_BODY
        + "\nThe total-prose overage is acknowledged; the body deliberately "
        "exceeds the budget.\n"
    )
    _ok, results = verify_task_body.verify_text(body)
    caps = _results_by_name(results)["v4 conciseness caps"]
    assert caps.passed and caps.is_warn
    assert "does not name" in caps.detail
    assert "Takeaways bullet-length" in caps.detail


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


# The verbatim #1332 body.md:292 footer clause (the task #1373 motivating
# incident): the plural-enumeration form the singular `(?!s)` clause regex
# deliberately excludes.
_I1332_FOOTER_CLAUSE = (
    "Two same-issue follow-up rounds, both 2026-07-15: (1) the 9a-ter "
    "free-analysis directional-inference battery, 0 GPU-h; (2) the proposer "
    "cheap-band GPU round `lowdose-grid-kill-battery` "
    "(`source: proposer-9b-cheap`, auto-run)."
)


def _v4_body_with_footer_line(line: str) -> str:
    """`_V4_GOOD_BODY` with `line` appended inside the Context footer."""
    return _V4_GOOD_BODY.replace(
        "- Originating prompt: origin prompt not recorded",
        "- Originating prompt: origin prompt not recorded\n- " + line,
    )


def test_v4_round_count_footer_plural_enumeration():
    """(Test 1) The verbatim #1332 footer clause -> (2, "footer") with no
    issue id (the plural arm alone credits the stated N)."""
    body = _v4_body_with_footer_line(_I1332_FOOTER_CLAUSE)
    assert verify_task_body._count_extra_followup_rounds_v4(body, None) == (2, "footer")


def test_v4_round_count_footer_plural_word_and_digit():
    """(Test 2) Number-word and digit forms parse alike; IGNORECASE; a
    repeated plural sentence takes max-over-matches; the count clamps at
    12."""
    cases = [
        ("three same-issue follow-up rounds folded so far.", 3),
        ("3 same-issue follow-up rounds folded so far.", 3),
        ("Two same-issue follow-up rounds folded so far.", 2),
        ("two SAME-ISSUE follow-up ROUNDS folded so far.", 2),
        # Repeated/updated plural sentences restate the cumulative total:
        # max over matches, never sum.
        ("Two same-issue follow-up rounds; later three same-issue follow-up rounds total.", 3),
        ("99 same-issue follow-up rounds (implausible; clamped).", 12),
    ]
    for line, expected in cases:
        body = _v4_body_with_footer_line(line)
        assert verify_task_body._count_extra_followup_rounds_v4(body, None) == (
            expected,
            "footer",
        ), line


def test_v4_round_count_footer_plural_no_generic_prose_false_positive():
    """(Test 3) Generic spec-quoting prose and a numberless plural stay
    excluded (preserves the `(?!s)` intent); a numbered plural sentence in
    body prose OUTSIDE the footer counts zero (structurally inherited from
    `_v4_footer_text`, pinned anyway)."""
    body = _v4_body_with_footer_line(
        "follow-up rounds also name each round's `followup_label`; "
        "same-issue follow-up rounds are folded into this body."
    )
    assert verify_task_body._count_extra_followup_rounds_v4(body, None) == (0, "none")
    # Plural-in-prose scoping: the sentence sits in Methodology prose, not
    # the footer.
    body2 = _V4_GOOD_BODY.replace(
        "- **Design:**",
        "- Two same-issue follow-up rounds folded this sweep. **Design:**",
    )
    assert verify_task_body._count_extra_followup_rounds_v4(body2, None) == (0, "none")


def test_v4_round_count_footer_both_forms_max_not_sum():
    """(Test 4) A singular clause + a plural summary sentence describe the
    same round set: max(1, 2) == 2, never 3."""
    body = _v4_body_with_footer_line(
        "Two same-issue follow-up rounds: the first is the "
        "same-issue follow-up round `round-a` (proposer-initiated)."
    )
    assert verify_task_body._count_extra_followup_rounds_v4(body, None) == (2, "footer")


def test_v4_round_count_events_free_analysis_counts(monkeypatch):
    """(Test 5) Free-analysis run markers count: a #1332-shape
    `followup_ref=` note + a no-ref free-prose note (#1090 shape) -> 2.
    A NON-run-kind event whose NOTE mentions the marker kind is not
    counted (kind-keyed counting)."""
    import explore_persona_space.task_workflow as tw

    fake = [
        {
            "kind": "epm:free-analysis-followup-run",
            "ts": "2026-07-15T18:51:36Z",
            "note": "followup_ref=Directional (asymmetric) transfer predictor\n"
            "headline_before=old title\nheadline_after=new title",
        },
        {
            "kind": "epm:free-analysis-followup-run",
            "ts": "2026-07-15T20:02:00Z",
            "note": "inline user-chat round: re-read committed cells, folded "
            "one takeaway into the body (no followup label).",
        },
        # Kind-keyed counting: a critique note MENTIONING the marker kind
        # is not a run marker (the :7739 pattern).
        {
            "kind": "epm:followup-value-critique",
            "ts": "2026-07-15T20:10:00Z",
            "note": "screened the epm:free-analysis-followup-run proposal set",
        },
    ]
    monkeypatch.setattr(tw, "list_events", lambda n: fake)
    assert verify_task_body._followup_events_rounds(123) == 2
    assert verify_task_body._count_extra_followup_rounds_v4(_V4_GOOD_BODY, 123) == (2, "events")


def test_v4_round_count_events_free_analysis_excludes_aborted(monkeypatch):
    """(Test 6) BOTH spec'd 9a-ter ABORT note forms are excluded (the
    round folded no prose): the reclassify form AND the implementer-FAIL
    form."""
    import explore_persona_space.task_workflow as tw

    fake = [
        {
            "kind": "epm:free-analysis-followup-run",
            "ts": "2026-07-15T10:00:00Z",
            "note": "aborted — reclassified as needs-gpu (the analysis needs new eval data)",
        },
        {
            "kind": "epm:free-analysis-followup-run",
            "ts": "2026-07-15T11:00:00Z",
            "note": "aborted — implementer FAIL on attempt 1",
        },
    ]
    monkeypatch.setattr(tw, "list_events", lambda n: fake)
    assert verify_task_body._followup_events_rounds(123) == 0
    assert verify_task_body._count_extra_followup_rounds_v4(_V4_GOOD_BODY, 123) == (0, "none")


def test_v4_round_count_events_free_analysis_dedupe_and_cross_leg(monkeypatch):
    """(Test 7) Byte-identical free-analysis notes (a marker-retry
    double-post) count once; a free-analysis `followup_ref` matching a
    counted same-issue run label is cross-leg-excluded; a NO-ref
    free-prose note still counts beside a labeled run marker (the
    explicit `ref is not None` guard). Total: run 1 + deduped no-ref
    free 1 = 2."""
    import explore_persona_space.task_workflow as tw

    retry_note = "inline round: folded the fair-comparison re-read (retry double-post)."
    fake = [
        {
            "kind": "epm:same-issue-followup-run",
            "ts": "2026-07-15T09:00:00Z",
            "note": "followup_label: round-x\nsource: user-chat\noutcome: folded",
        },
        # Cross-leg guard: ref exactly matches the counted run label.
        {
            "kind": "epm:free-analysis-followup-run",
            "ts": "2026-07-15T10:00:00Z",
            "note": "followup_ref=round-x\ngpu_hours=0\nresult: folded into body",
        },
        # Byte-identical double-post: counts once.
        {
            "kind": "epm:free-analysis-followup-run",
            "ts": "2026-07-15T11:00:00Z",
            "note": retry_note,
        },
        {
            "kind": "epm:free-analysis-followup-run",
            "ts": "2026-07-15T11:00:05Z",
            "note": retry_note,
        },
    ]
    monkeypatch.setattr(tw, "list_events", lambda n: fake)
    assert verify_task_body._followup_events_rounds(123) == 2


def test_v4_round_count_events_inflight_counts_one(monkeypatch):
    """(Test 8) An armed dispatchable scope with no run marker and no
    retro-close evidence credits +1; TWO armed labels still +1 (cap);
    an unlabeled pseudo-label scope credits +0; a scope whose label has a
    run marker adds no extra; and (guard 4) an armed label closed by a
    free-analysis `followup_ref` counts via the free leg ONLY (in-flight
    suppressed by retro-close evidence)."""
    import explore_persona_space.task_workflow as tw

    def _events(events):
        monkeypatch.setattr(tw, "list_events", lambda n: list(events))

    scope_a = {
        "kind": "epm:followup-scope",
        "ts": "2026-07-15T08:00:00Z",
        "note": "followup_label: armed-a\nsource: proposer-9b-cheap\nest_gpu_hours: 2",
    }
    scope_b = {
        "kind": "epm:followup-scope",
        "ts": "2026-07-15T08:05:00Z",
        "note": "followup_label: armed-b\nsource: user-chat\nest_gpu_hours: 1",
    }
    # (a) one armed dispatchable label, no run marker, no retro-close -> 1.
    _events([scope_a])
    assert verify_task_body._followup_events_rounds(123) == 1
    # (b) TWO armed labels: the in-flight credit caps at +1 total.
    _events([scope_a, scope_b])
    assert verify_task_body._followup_events_rounds(123) == 1
    # (c) an unlabeled pseudo-label scope (no correction signal) is not
    # dispatchable and never credits.
    _events(
        [
            {
                "kind": "epm:followup-scope",
                "ts": "2026-07-15T08:10:00Z",
                "note": "queued follow-up idea with no label field",
            }
        ]
    )
    assert verify_task_body._followup_events_rounds(123) == 0
    # (d) a scope whose label has a matching run marker: the run counts,
    # no in-flight extra.
    _events(
        [
            scope_a,
            {
                "kind": "epm:same-issue-followup-run",
                "ts": "2026-07-15T12:00:00Z",
                "note": "followup_label: armed-a\nsource: proposer-9b-cheap\noutcome: folded",
            },
        ]
    )
    assert verify_task_body._followup_events_rounds(123) == 1
    # (e) guard-4 pin: an armed dispatchable label whose round completed as
    # a free-analysis round (`followup_ref` == label, NO run marker) counts
    # ONCE via the free leg; the retro-close evidence suppresses the
    # in-flight +1. An implementation missing the retro-close check in
    # `_has_inflight_round` reads 2 here.
    _events(
        [
            scope_a,
            {
                "kind": "epm:free-analysis-followup-run",
                "ts": "2026-07-15T09:30:00Z",
                "note": "followup_ref=armed-a\ngpu_hours=0\nresult: folded into body",
            },
        ]
    )
    assert verify_task_body._followup_events_rounds(123) == 1


def test_v4_round_count_issue_1332_regression(monkeypatch):
    """(Test 9) The #1332 incident replay, BOTH states. Full state (scope +
    run + free-analysis, the real marker shapes): events leg = run 1 +
    free 1 = 2, and with the real plural footer the total reads
    (2, "footer+events"). Mid-round state (run marker removed — the state
    the clean-result gate actually saw): events leg still 2
    (free-analysis 1 + in-flight 1)."""
    import explore_persona_space.task_workflow as tw

    free = {
        "kind": "epm:free-analysis-followup-run",
        "ts": "2026-07-15T18:51:36Z",
        "version": 1,
        "note": "followup_ref=Directional (asymmetric) transfer predictor with "
        "registered inference\nheadline_before=Function-space map similarity "
        "predicts marker leakage (MODERATE confidence)\nheadline_after=Directional "
        "function-space map similarity predicts marker leakage",
    }
    scope = {
        "kind": "epm:followup-scope",
        "ts": "2026-07-15T19:37:40Z",
        "version": 1,
        "note": "followup_label: lowdose-grid-kill-battery\nsource: proposer-9b-cheap\n"
        "est_gpu_hours: 8\nquestion_relation: same\nauto_run: yes",
    }
    run = {
        "kind": "epm:same-issue-followup-run",
        "ts": "2026-07-16T00:18:05Z",
        "version": 1,
        "note": "followup_label: lowdose-grid-kill-battery\nsource: proposer-9b-cheap\n"
        "outcome: registered verdicts folded; re-parked awaiting_promotion",
    }
    body = _v4_body_with_footer_line(_I1332_FOOTER_CLAUSE)

    # Full (post-round) state.
    monkeypatch.setattr(tw, "list_events", lambda n: [free, scope, run])
    assert verify_task_body._followup_events_rounds(123) == 2
    assert verify_task_body._count_extra_followup_rounds_v4(body, 123) == (2, "footer+events")

    # Mid-round state — the run marker has not posted yet: the armed scope
    # is in-flight (+1) and the free-analysis round still counts (+1).
    monkeypatch.setattr(tw, "list_events", lambda n: [free, scope])
    assert verify_task_body._followup_events_rounds(123) == 2
    assert verify_task_body._count_extra_followup_rounds_v4(body, 123) == (2, "footer+events")


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


# ─── Check 38: linked-but-not-embedded figures in v4 ## Results (#1371) ──────
#
# WARN-only, v4-sentinel-gated, pure-text: a non-image markdown link in
# `## Results` to a `figures/issue_<N>/*.png` that no body image embeds
# (the #1315 incident shape — check 31's stem-in-prose escape is blind to
# it). All assertions per-check by name (the `_V4_GOOD_BODY` convention —
# fake SHAs fail the existence probes, so overall PASS is not assertable).

_LINKED_FIG_CHECK = "Results figures embedded, not linked"
_C37_PATH = "figures/issue_999/per_row_grid.png"
_C37_RAW_URL = (
    "https://raw.githubusercontent.com/superkaiba/explore-persona-space/"
    "0123456789abcdef/" + _C37_PATH
)
_C37_HERO_URL = (
    "https://raw.githubusercontent.com/superkaiba/explore-persona-space/"
    "0123456789abcdef/figures/issue_999/hero.png"
)
_C37_HERO_EMBED = (
    "![Bar chart of mean alignment with 95% CI across three seeds; "
    "baseline 70.4% vs tulu-25 87.9%.](" + _C37_HERO_URL + ")"
)


def _c37_body_with_results_link(fragment):
    """`_V4_GOOD_BODY` with `fragment` appended as its own paragraph after
    the last result's interpretation line (inside `## Results`)."""
    return _V4_GOOD_BODY.replace(_V4_INTERP_LINE, _V4_INTERP_LINE + "\n\n" + fragment)


def _c37_direct(fixture_text, issue=None):
    """Direct-call check 38 on the post-frontmatter body, as verify_text does."""
    _fm, body = verify_task_body.split_frontmatter(fixture_text)
    return verify_task_body.check_linked_not_embedded_figures(body, issue=issue)


def test_linked_not_embedded_figure_warns():
    """PRIMARY pin (#1371 durability pin — the #1315 incident shape): a
    figure referenced as a markdown LINK in `## Results` with no image
    embed anywhere in the body WARNs (passed=True — the overall verdict is
    never flipped by this check). Asserted BY NAME through verify_text so
    the dispatch outside CHECKS is pinned (the check-31
    `test_orphan_per_unit_figure_warns` pattern: a refactor dropping the
    `verify_text` append fails here)."""
    body = _c37_body_with_results_link(f"See the [per-row grid]({_C37_RAW_URL}).")
    _ok, results = verify_task_body.verify_text(body, issue=999)
    r = _results_by_name(results)[_LINKED_FIG_CHECK]
    assert r.passed is True
    assert r.is_warn is True
    assert _C37_PATH in r.detail
    assert "Lens 11" in r.detail


def test_linked_figure_also_embedded_no_warn():
    """The same path embedded as an image ANYWHERE in the body — here under
    `## Methodology`, pinned at a DIFFERENT SHA — silences the WARN
    (whole-body, SHA-independent embed subtraction)."""
    embed = (
        "![per-row grid](https://raw.githubusercontent.com/superkaiba/"
        "explore-persona-space/fedcba9876543210/" + _C37_PATH + ")"
    )
    body = _c37_body_with_results_link(f"See the [per-row grid]({_C37_RAW_URL}).").replace(
        "## Results\n", embed + "\n\n## Results\n"
    )
    r = _c37_direct(body, issue=999)
    assert r.passed and not r.is_warn, r.render()


def test_blob_url_embed_silences_warn():
    """A GitHub BLOB-URL image embed also silences the WARN — pins the plan-v2
    any-URL-form embed-subtraction correction (the raw-GitHub-only
    `_referenced_figure_paths` helper would false-positive here)."""
    embed = (
        "![per-row grid](https://github.com/superkaiba/explore-persona-space/"
        "blob/fedcba9876543210/" + _C37_PATH + ")"
    )
    body = _c37_body_with_results_link(f"See the [per-row grid]({_C37_RAW_URL}).").replace(
        "## Results\n", embed + "\n\n## Results\n"
    )
    r = _c37_direct(body, issue=999)
    assert r.passed and not r.is_warn, r.render()


def test_html_img_embed_silences_warn():
    """An HTML `<img src="…">` embed of the linked path also counts as an
    embed — a body that embeds via raw HTML must not false-positive."""
    embed = f'<img src="{_C37_RAW_URL}" width="400">'
    body = _c37_body_with_results_link(f"See the [per-row grid]({_C37_RAW_URL}).").replace(
        "## Results\n", embed + "\n\n## Results\n"
    )
    r = _c37_direct(body, issue=999)
    assert r.passed and not r.is_warn, r.render()


def test_linked_not_embedded_clean_v4_body_no_warn():
    """Unmodified `_V4_GOOD_BODY` (hero figure embedded inline) → clean PASS."""
    _ok, results = verify_task_body.verify_text(_V4_GOOD_BODY, issue=999)
    r = _results_by_name(results)[_LINKED_FIG_CHECK]
    assert r.passed and not r.is_warn, r.render()


def test_clickable_image_wrapper_no_warn():
    """The clickable-image wrapper `[![alt](p)](p)`: masking the inner image
    leaves a `[](p)` residue link, which the embed subtraction silences
    (p IS embedded by the inner image)."""
    assert _C37_HERO_EMBED in _V4_GOOD_BODY  # fixture-drift guard
    body = _V4_GOOD_BODY.replace(
        _C37_HERO_EMBED, "[" + _C37_HERO_EMBED + "](" + _C37_HERO_URL + ")"
    )
    r = _c37_direct(body, issue=999)
    assert r.passed and not r.is_warn, r.render()


def test_wrapper_to_unembedded_png_warns():
    """A wrapper `[![alt](p)](q)` whose click target q is a PNG embedded
    nowhere deliberately WARNs — q is linked-not-embedded by definition
    (documents the residue-link semantics)."""
    body = _V4_GOOD_BODY.replace(_C37_HERO_EMBED, "[" + _C37_HERO_EMBED + "](" + _C37_RAW_URL + ")")
    r = _c37_direct(body, issue=999)
    assert r.passed is True
    assert r.is_warn is True
    assert _C37_PATH in r.detail


def test_line_start_link_warns():
    """A link at column 0 on its own line WARNs — the mask-then-match
    pipeline needs no `[^!]` lookbehind, so the line-start case (which a
    lookbehind-based scan would miss) is covered."""
    body = _c37_body_with_results_link(f"[per-row grid]({_C37_RAW_URL})")
    r = _c37_direct(body, issue=999)
    assert r.is_warn and _C37_PATH in r.detail, r.render()


def test_link_in_fence_no_warn():
    """A figure link quoted inside a code fence in `## Results` never WARNs
    (`_prose_layer` strips fences)."""
    body = _c37_body_with_results_link(f"```text\nSee [per-row grid]({_C37_RAW_URL}).\n```")
    r = _c37_direct(body, issue=999)
    assert r.passed and not r.is_warn, r.render()


def test_link_in_details_no_warn():
    """A figure link tucked inside `<details>…</details>` in `## Results`
    never WARNs — dropdown-tucked links are deliberate presentation (the
    named recall sacrifice)."""
    body = _c37_body_with_results_link(
        "<details>\n<summary>extra figures</summary>\n\n"
        f"See the [per-row grid]({_C37_RAW_URL}).\n\n</details>"
    )
    r = _c37_direct(body, issue=999)
    assert r.passed and not r.is_warn, r.render()


def test_caption_blockquote_link_warns():
    """A blockquote caption link to an UNEMBEDDED PNG WARNs — blockquote
    lines stay in the prose layer (only fences + `<details>` are
    stripped); intended semantics, documented here."""
    body = _c37_body_with_results_link(
        f"> **Figure.** *Companion view.* Full grid: [per-row grid]({_C37_RAW_URL})."
    )
    r = _c37_direct(body, issue=999)
    assert r.is_warn and _C37_PATH in r.detail, r.render()


def test_cross_issue_link_scoping():
    """A Results link to ANOTHER issue's figure is a legitimate
    cross-reference: no WARN when `issue` is known; the `issue=None`
    (`--body-stdin`) fallback scans every dir and CAN flag it (check 31's
    documented fallback caveat)."""
    other = (
        "https://raw.githubusercontent.com/superkaiba/explore-persona-space/"
        "0123456789abcdef/figures/issue_777/x.png"
    )
    body = _c37_body_with_results_link(f"See the parent's [grid]({other}).")
    r = _c37_direct(body, issue=999)
    assert r.passed and not r.is_warn, r.render()
    r_none = _c37_direct(body, issue=None)
    assert r_none.is_warn and "figures/issue_777/x.png" in r_none.detail, r_none.render()


def test_relative_link_warns():
    """The relative-URL form `[grid](figures/issue_<N>/…png)` also WARNs —
    the embed-vs-link discipline is host-independent (any-URL-form)."""
    body = _c37_body_with_results_link(f"See the [grid]({_C37_PATH}).")
    r = _c37_direct(body, issue=999)
    assert r.is_warn and _C37_PATH in r.detail, r.render()


def test_linked_figure_v3_body_vacuous_pass():
    """Forward-only: a v3 body with a figure link in `## Findings` is never
    flagged (v4-sentinel gate)."""
    body = _V3_GOOD_BODY.replace(
        "## Findings\n", f"## Findings\n\nSee the [grid]({_C37_RAW_URL}).\n"
    )
    r = _c37_direct(body, issue=None)
    assert r.passed and not r.is_warn, r.render()
    assert "not a v4 body" in r.detail


def test_linked_figure_no_results_section_vacuous_pass():
    """A v4 body with no `## Results` section → vacuous PASS (the
    `_v4_results_body is None` branch)."""
    body = (
        "# Title (LOW confidence)\n\n<!-- clean-result-v4 -->\n\n"
        f"## Takeaways\n\n- One bullet naming [a grid]({_C37_RAW_URL}).\n"
    )
    r = verify_task_body.check_linked_not_embedded_figures(body, issue=999)
    assert r.passed and not r.is_warn, r.render()
    assert "no `## Results`" in r.detail


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


# ─── Check 5: figure-caption italic lead claim (v4-only WARN, #1424) ────────


def _v4_minimal_results_body(results_content: str) -> str:
    """Minimal v4-sentinel body wrapping `results_content` under `## Results`
    (no footer) — direct-call fixture for `check_figure_caption`."""
    return f"# T (LOW confidence)\n\n<!-- clean-result-v4 -->\n\n## Results\n\n{results_content}\n"


def test_caption_lead_conformant_v4_passes():
    """A conformant v4 body (caption opens `**Figure.** *lead claim.*`)
    passes check 5 with no WARN."""
    res = verify_task_body.check_figure_caption(_V4_GOOD_BODY)
    assert res.passed is True
    assert res.is_warn is False
    assert "all 1" in res.detail


def test_caption_missing_italic_lead_v4_warns():
    """DURABILITY PIN (#1424): a v4 caption carrying the bold `**Figure.**`
    prefix but NO italic lead claim WARNs (passed stays True — never FAIL),
    naming the enclosing result H3 and quoting the expected shape."""
    body = _V4_GOOD_BODY.replace(
        "> **Figure.** *Tulu-25 lifts alignment ~17 pts over baseline at every seed.* "
        "Baseline gray, tulu-25 blue; error bars 95% Wald CIs.",
        "> **Figure.** Plain prose lead without italics.",
    )
    assert body != _V4_GOOD_BODY  # the replace actually fired
    res = verify_task_body.check_figure_caption(body)
    assert res.passed is True
    assert res.is_warn is True
    assert "A clean +17-pt lift" in res.detail  # enclosing H3 named
    assert "missing the italic lead claim" in res.detail
    assert "**Figure.** *one-sentence lead claim.*" in res.detail


def test_caption_missing_bold_prefix_v4_warns():
    """A v4 caption with NO bold `**Figure.**` prefix at all WARNs and the
    detail attributes the missing bold prefix (not just the italic lead)."""
    body = _V4_GOOD_BODY.replace(
        "> **Figure.** *Tulu-25 lifts alignment ~17 pts over baseline at every seed.* "
        "Baseline gray, tulu-25 blue; error bars 95% Wald CIs.",
        "> *Italic lead only.* Rest of caption.",
    )
    assert body != _V4_GOOD_BODY
    res = verify_task_body.check_figure_caption(body)
    assert res.passed is True
    assert res.is_warn is True
    assert "bold prefix" in res.detail


def test_caption_lead_check_skips_non_v4():
    """Forward-only: a non-conformant caption in a v3 or legacy body never
    WARNs — check 5 PASS-skips on every non-v4 body."""
    for fixture in (_V3_GOOD_BODY, GOOD_BODY):
        body = fixture.replace(
            "**Figure.** *Tulu-25 lifts alignment ~17 pts over baseline at every seed.*",
            "**Figure.** Plain prose lead.",
        )
        assert body != fixture
        res = verify_task_body.check_figure_caption(body)
        assert res.passed is True
        assert res.is_warn is False
        assert "skipped — not a v4 body" in res.detail


def test_caption_lead_no_figures_passes():
    """A v4 body whose `## Results` has an H3 + prose but no image passes
    vacuously (no captions to check)."""
    body = _v4_minimal_results_body("### A result\n\nProse only, no image here.")
    res = verify_task_body.check_figure_caption(body)
    assert res.passed is True
    assert res.is_warn is False
    assert "no blockquote captions" in res.detail


def test_caption_lead_captionless_figure_exempt():
    """An image followed directly by plain prose (no blockquote caption) is
    exempt — caption PRESENCE is owned by check 21 + critic Lens 3."""
    body = _v4_minimal_results_body(
        "### A result\n\n![alt](https://x/y.png)\n\nPlain interpretation prose, no blockquote."
    )
    res = verify_task_body.check_figure_caption(body)
    assert res.passed is True
    assert res.is_warn is False


def test_caption_lead_period_variants_pass():
    """Tolerance arms: `**Figure**` (no period) and `**Figure 1.**` (short
    designator) both satisfy the bold-prefix + italic-lead shape."""
    for cap in (
        "> **Figure** *Lead claim.* rest of caption.",
        "> **Figure 1.** *Lead claim.* rest of caption.",
    ):
        body = _v4_minimal_results_body(f"### A result\n\n![alt](https://x/y.png)\n\n{cap}")
        res = verify_task_body.check_figure_caption(body)
        assert res.passed is True
        assert res.is_warn is False, cap


def test_caption_lead_bold_not_italic_warns():
    """A bold `**…**` run after the prefix must NOT masquerade as the italic
    lead (the `(?!\\*)` lookahead guard)."""
    body = _v4_minimal_results_body(
        "### A result\n\n![alt](https://x/y.png)\n\n> **Figure.** **Bold, not italic.** rest."
    )
    res = verify_task_body.check_figure_caption(body)
    assert res.passed is True
    assert res.is_warn is True
    assert "missing the italic lead claim" in res.detail


def test_caption_lead_fenced_image_ignored():
    """A non-conformant image + caption pair inside a fenced code block under
    `## Results` is never scanned."""
    body = _v4_minimal_results_body(
        "### A result\n\nProse.\n\n```markdown\n"
        "![alt](https://x/y.png)\n\n> **Figure.** No italic lead here.\n```"
    )
    res = verify_task_body.check_figure_caption(body)
    assert res.passed is True
    assert res.is_warn is False


def test_caption_lead_issue1074_verbatim_caption_warns():
    """Extra missing-lead fixture: the verbatim first caption of #1074's
    committed body (bold prefix, no italic lead) — the plan-time audit's
    known-missing-lead spot-check."""
    cap = (
        "> **Figure.** Judge-accepted fraction of generated positives per class and "
        "generator, Wilson 95% intervals, floor dashed. Claude bars are the parent run's "
        "yields under a three-way recipe bundle (generator, injection style, variant "
        "count), context only. Only the abliterated arm on harmful compliance clears its "
        "floor (177 of 215)."
    )
    body = _v4_minimal_results_body(f"### Yield per class\n\n![alt](https://x/y.png)\n\n{cap}")
    res = verify_task_body.check_figure_caption(body)
    assert res.passed is True
    assert res.is_warn is True
    assert "missing the italic lead claim" in res.detail


def test_check_figure_caption_position_stable():
    """Index-stability pin (#1424): `check_figure_caption` stays at CHECKS
    position 7 and the CHECKS count matches the current registry (54 as of
    checks 52/53, #2016; belt-and-suspenders beside the migration-history
    `len(CHECKS)` pin)."""
    assert verify_task_body.CHECKS[7] is verify_task_body.check_figure_caption
    assert len(verify_task_body.CHECKS) == 54


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

_CHECK28_NAME = (
    "figure text opaque config codes "
    "(slug / @L-pin / H-code / slot-family / P-M candidate / letter-arrow / arm-slug tokens)"
)


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
    # Candidate/panel-code class (#1900): bare P/M candidate ids — single
    # digit + optional single lowercase letter, same shape discipline as
    # the H-code class.
    assert fn("P1") == ["P1"]
    assert fn("P7") == ["P7"]
    assert fn("M4") == ["M4"]
    assert fn("P3b") == ["P3b"]
    assert fn("mediation forest (P1 | P7)") == ["P1", "P7"]
    assert "P1" in fn("sw_eng_expB-P1")  # candidate id riding a slug label
    # Candidate-code path exemption: the raw regex genuinely matches inside
    # the path word (non-vacuity assert), so the `[]` result is produced by
    # the per-word path exemption, not the regex boundary.
    assert fn("figures/issue_1900/P7.png") == []  # whole-string path skip
    assert fn("source: figures/issue_1900/P7.png") == []  # path word in prose
    assert verify_task_body._CANDIDATE_CODE_RE.search("figures/issue_1900/P7.png")
    # Letter-arrow transition class (#1902): the incident shapes and their
    # documented spelling variants. AC-1 positives.
    assert fn("B->S_single") == ["B->S_single"]
    assert fn("S->D_multi") == ["S->D_multi"]
    assert fn("D->R_single") == ["D->R_single"]
    assert fn("D->R_multi") == ["D->R_multi"]
    assert fn("S→D_multi") == ["S→D_multi"]  # unicode arrow
    assert fn("A -> B_foo") == ["A -> B_foo"]  # spaces around ASCII arrow
    assert fn("A→B_foo") == ["A→B_foo"]  # no-space unicode with suffix
    # AC-3 negative pins — the tightened regex REQUIRES a `_[a-z]+` snake
    # suffix on the RHS, so legitimate legend syntax stays unflagged.
    assert fn("H->O") == []  # chemistry reactant→product, no snake suffix
    assert fn("A->B") == []  # bare state-machine label, no snake suffix
    assert fn("X -> Y") == []  # HMM/Markov transition, no snake suffix
    assert fn("Fe->Fe2+") == []  # multi-char labels, single-`[A-Z]` boundary
    assert fn("A->b_foo") == []  # lowercase RHS pre-suffix, single-`[A-Z]`
    # AC-2 length-1 exact-list lock (per critic Concern #3): pins the
    # current `_SNAKE_TOKEN_RE` suppression (`S_single` has 1 underscore
    # and 0 digits → snake arm does not flag) against any future
    # loosening that would introduce a dedup collision. The letter-arrow
    # arm catches the WHOLE token including the `B->` prefix.
    assert fn("B->S_single") == ["B->S_single"]
    # AC-4 path exemption for the letter-arrow class: the raw regex
    # genuinely matches inside the path word (non-vacuity assert), so the
    # `[]` result is produced by the per-word path exemption, not the
    # regex boundary.
    assert fn("figures/issue_1902/A->B_x.png") == []  # whole-string path skip
    assert fn("source: figures/issue_1902/A->B_x.png") == []  # path word in prose
    assert verify_task_body._LETTER_ARROW_RE.search("figures/issue_1902/A->B_x.png")
    # Known-good: none of these yield any token.
    for good in (
        "house: librarian",
        "true target (leading fold-basis PCA dimension)",
        "wildchat: short 1",
        "log_prob",
        "judge_rate",
        "helpful_assistant",
        "r_B",
        "p97.5 latency by arm",  # lowercase percentile shorthand (case pin)
        "p50",  # lowercase percentile shorthand
        "P100",  # GPU name — multi-digit, no boundary between digits
        "M40",  # GPU name — multi-digit
        "P1C",  # uppercase suffix is not the candidate-tag convention
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


def test_check28_hypothesis_and_slot_family_in_text_block_warns(tmp_path, monkeypatch):
    """The #1072 live repro: the verbatim pre-fix sidecar strings (revision
    8a5e966a of `exploratory_component_profiles.meta.json`) — a panel title
    carrying the bare hypothesis code `(H3)` and an xlabel carrying the
    slot-family code `f16` — WARN through the `meta["text"]` walk."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {
            "created": "2026-07-10T00:00:00Z",
            "text": {
                "suptitle": None,
                "fig_texts": [],
                "axes": [
                    {
                        "title_left": "Parallel share of the gap by depth (H3)",
                        "xlabel": "answer position t (f16 slots)",
                    }
                ],
            },
        },
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_label_codes(body)
    assert res.passed and res.is_warn, res.render()
    assert "H3" in res.detail and "f16" in res.detail


def test_check28_letter_arrow_in_text_block_warns(tmp_path, monkeypatch):
    """The #1902 live repro: a sidecar `text.axes[0].title` carrying the
    verbatim incident string `B->S_single` (the shape that passed the
    five existing classes on `clusters_delta_qc_scatter.png` — no @L-pin,
    no matching snake, no H-code, no slot-family, no P-M candidate) —
    WARNs through the `meta["text"]` walk."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {
            "created": "2026-08-04T00:00:00Z",
            "text": {
                "suptitle": None,
                "fig_texts": [],
                "axes": [{"title": "B->S_single"}],
            },
        },
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_label_codes(body)
    assert res.passed and res.is_warn, res.render()
    assert "B->S_single" in res.detail


def test_check28_letter_arrow_path_word_exempted(tmp_path, monkeypatch):
    """AC-5(ii): a sidecar carries the letter-arrow token as a TITLE
    (WARN) AND a `source:` path-shaped word containing an in-path
    letter-arrow token. Only the title token appears in the WARN detail;
    the path-word token is exempted by `_only_in_path_words` — mirrors
    the H-code / slot-family / candidate classes' incumbent walker
    coverage."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {
            "created": "2026-08-04T00:00:00Z",
            "text": {
                "suptitle": None,
                "fig_texts": ["source: figures/issue_1902/A->B_x.png"],
                "axes": [{"title": "B->S_single"}],
            },
        },
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_label_codes(body)
    assert res.passed and res.is_warn, res.render()
    assert "B->S_single" in res.detail
    # Path-word token is exempted — it does not appear in the WARN detail.
    assert "A->B_x" not in res.detail


def test_check28_hypothesis_and_slot_family_classifier():
    """Pure-function inventories for the hypothesis-code + slot-family
    classes (the #1506 durability pin), incl. the boundary-terminated
    path-exemption arm with its regex-matches non-vacuity proof."""
    fn = verify_task_body._opaque_code_tokens
    # Known-bad: the verbatim #1072 strings + the class inventory.
    assert fn("Parallel share of the gap by depth (H3)") == ["H3"]
    assert fn("answer position t (f16 slots)") == ["f16"]
    assert fn("l16 slots") == ["l16"]
    assert fn("H1 vs H2") == ["H1", "H2"]
    assert fn("H3 and H3") == ["H3"]  # repeated-token de-dup pin
    # H<digit><lowercase-letter> hypothesis-tag form (#1774 widen): the
    # verbatim incident string (a sidecar title_left that passed the old
    # single-digit `\bH\d\b` form silently), plus a multi-tag pin.
    assert fn("Jensen-gap direction concentration (H1c)") == ["H1c"]
    assert fn("H1a vs H4b") == ["H1a", "H4b"]
    # Accepted false-positive envelope, documented by design: ANY standalone
    # H<digit><lowercase letter> token now matches (e.g. `H2o`) — no such
    # token is a legitimate rendered figure-text label in this project's
    # domain, and check 28 is WARN-only.
    assert fn("H2o sample") == ["H2o"]
    # Snake-class non-overlap pin: `f16` inside `f16_slots` never matches the
    # slot-family class (no boundary at `_`), while the digit-bearing snake
    # token itself is flagged exactly once — no double-add.
    assert fn("f16_slots plot") == ["f16_slots"]
    # Known-good: GPU names, dtype spellings, metric names, case pins.
    # (`"H1 2026"` is deliberately NOT here — it matches by design.)
    for good in (
        "H100",
        "1x H200 pod",
        "H20 inference",
        "h3 heading",  # case pin
        "bf16 precision",
        "fp16",
        "F16",  # case pin
        "L16 readout",  # case pin
        "F1 score",
        "l2 norm",
    ):
        assert fn(good) == [], f"false positive on {good!r}: {fn(good)}"
    # Path-exemption arm — BOUNDARY-TERMINATED tokens: the raw regexes
    # genuinely match inside these path words (non-vacuity asserts below),
    # so the `[]` results are produced by the path exemption, not the regex
    # boundary; a `_`-suffixed token like `H3_panel.png` never matches the
    # regex at all and would prove nothing about the exemption wiring.
    assert fn("figures/issue_1072/H3.png") == []  # whole-string path skip
    assert fn("source: figures/issue_1072/H3.png") == []  # path word in prose
    assert fn("see figures/a/f16.png") == []  # path word in prose
    assert fn("figures/a/H1c.png") == []  # H<digit><letter> form inside a path
    assert verify_task_body._HYPOTHESIS_CODE_RE.search("figures/a/H3.png")
    assert verify_task_body._HYPOTHESIS_CODE_RE.search("figures/a/H1c.png")
    assert verify_task_body._SLOT_FAMILY_RE.search("figures/a/f16.png")


def test_check28_candidate_code_in_text_block_warns(tmp_path, monkeypatch):
    """The #1900 live repro shape: a sidecar title carrying bare candidate
    codes (`mediation forest (P1 | P7)` — the pre-fix
    `mediation_forest.meta.json` legend/title strings at `0e5e6c3e7d`)
    WARNs through the `meta["text"]` walk, naming the tokens."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {
            "created": "2026-08-02T00:00:00Z",
            "text": {
                "suptitle": None,
                "fig_texts": [],
                "axes": [{"title_left": "mediation forest (P1 | P7)"}],
            },
        },
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_label_codes(body)
    assert res.passed and res.is_warn, res.render()
    assert "P1" in res.detail and "P7" in res.detail


def test_check28_percentile_title_passes_clean(tmp_path, monkeypatch):
    """Lowercase percentile shorthand rendered as a title (`p97.5 latency by
    arm`) stays clean — the candidate-code class is uppercase-only, so no
    new false positive on percentile text."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {
            "created": "2026-08-02T00:00:00Z",
            "text": {
                "suptitle": None,
                "fig_texts": [],
                "axes": [{"title_left": "p97.5 latency by arm"}],
            },
        },
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_label_codes(body)
    assert res.passed and not res.is_warn, res.render()


def test_check28_arm_slug_classifier():
    """Pure-function inventories for the hyphen-separated arm-slug class (g)
    (#1988): AC-1 recall (the #1768 `behavior-context-regime-lr-seed` slug
    grammar + the no-`s`-seed-prefix #1586 variant), AC-2 corpus-derived
    FP drops (digit-in-FINAL-segment + <=6-char-tail filter), AC-3
    2-segment non-flag (regex requires >=3 segments), AC-4 path exemption
    with its regex-matches non-vacuity proof."""
    fn = verify_task_body._opaque_code_tokens
    # AC-1 recall: fleet slug grammar, with and without the `s` seed prefix.
    assert fn("cas-pers-con-lr1e5-s137") == ["cas-pers-con-lr1e5-s137"]
    assert fn("ft-con-137") == ["ft-con-137"]
    assert fn("ft-con-42") == ["ft-con-42"]
    assert fn("delta vs base for cas-icl-po-lr1e5-s42 cells") == ["cas-icl-po-lr1e5-s42"]
    # AC-2 corpus-derived FP drops: hyphenated rendered English / compounds
    # (no digit in the final segment) and long dated ids (final segment over
    # 6 chars) stay clean — the exact plan-review probe set.
    for good in (
        "under-4-token",  # figures/issue_1335 rendered text
        "first-16-token",  # figures/issue_952
        "best-of-28-layers",  # figures/issue_664
        "claude-sonnet-4-5-20250929",  # judge model id (issue_1092/issue_1739)
        "eps-persona-gpu-jun2026",  # GCP project id (issue_588)
        "end-to-end",
        "state-of-the-art",
        "us-central1-a",  # GCP zone — final segment `a` has no digit
    ):
        assert fn(good) == [], f"false positive on {good!r}: {fn(good)}"
    # AC-3: 2-segment hyphen tokens never match (>=3 segments required).
    assert fn("log-prob") == []
    assert fn("log-prob margin by arm") == []
    # AC-4 path exemption: the raw regex genuinely matches inside the path
    # word (non-vacuity assert below), so the `[]` results are produced by
    # the per-word path exemption, not the regex boundary.
    assert fn("figures/issue_1768/ft-con-137.png") == []  # whole-string path skip
    assert fn("source: figures/issue_1768/ft-con-137.png") == []  # path word in prose
    assert verify_task_body._ARM_SLUG_RE.search("figures/issue_1768/ft-con-137.png")
    # `_is_arm_slug_token` membership predicate (shared with check 28's
    # caption suppression): fullmatch + digit-bearing <=6-char final segment.
    assert verify_task_body._is_arm_slug_token("cas-pers-con-lr1e5-s137")
    assert verify_task_body._is_arm_slug_token("ft-con-137")
    assert not verify_task_body._is_arm_slug_token("claude-sonnet-4-5-20250929")
    assert not verify_task_body._is_arm_slug_token("under-4-token")
    assert not verify_task_body._is_arm_slug_token("log-prob")
    assert not verify_task_body._is_arm_slug_token("H3")  # non-slug class token


def test_check28_arm_slug_in_yticklabels_warns(tmp_path, monkeypatch):
    """The #1988 live shape (#1768's figures): arm slugs rendered as tick
    labels reach check 28 through `meta["text"].axes[].yticklabels` VALUES
    and WARN when the body caption does not name them."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {
            "created": "2026-08-05T00:00:00Z",
            "text": {
                "suptitle": None,
                "fig_texts": [],
                "axes": [
                    {
                        "title": "install delta by cell",
                        "yticklabels": ["cas-pers-con-lr1e5-s137", "ft-con-137"],
                    }
                ],
            },
        },
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_label_codes(body)
    assert res.passed and res.is_warn, res.render()
    assert "cas-pers-con-lr1e5-s137" in res.detail


def test_check28_arm_slug_caption_decode_suppressed(tmp_path, monkeypatch):
    """Slug-class caption-decode suppression (#1988): when THIS figure's
    blockquote caption names the slug verbatim (case-insensitively — the
    caption here renders it uppercase), the slug token is suppressed and
    the check PASSes clean."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {
            "created": "2026-08-05T00:00:00Z",
            "text": {
                "suptitle": None,
                "fig_texts": [],
                "axes": [{"yticklabels": ["cas-pers-con-lr1e5-s137"]}],
            },
        },
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha).replace(
        "Baseline gray, tulu-25 blue; error bars 95% Wald CIs.",
        "Row CAS-PERS-CON-LR1E5-S137 is the persona-context contrastive cell at lr 1e-5, seed 137.",
    )
    res = verify_task_body.check_figure_label_codes(body)
    assert res.passed and not res.is_warn, res.render()
    assert "free of opaque config codes" in res.detail


def test_check28_caption_suppression_is_slug_class_only(tmp_path, monkeypatch):
    """Caption naming decodes ONLY the arm-slug class: a caption naming both
    the slug AND a hypothesis code verbatim suppresses the slug token while
    the H-code (class (c)) still WARNs — classes (a)-(f) stay byte-stable
    (no caption suppression)."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {
            "created": "2026-08-05T00:00:00Z",
            "text": {
                "suptitle": None,
                "fig_texts": [],
                "axes": [{"title": "ft-con-137 (H3)"}],
            },
        },
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha).replace(
        "Baseline gray, tulu-25 blue; error bars 95% Wald CIs.",
        "Cell ft-con-137 is the full-finetune contrastive cell at seed 137 (H3).",
    )
    res = verify_task_body.check_figure_label_codes(body)
    assert res.passed and res.is_warn, res.render()
    assert "H3" in res.detail
    assert "ft-con-137" not in res.detail


# ─── Check 41: sidecar-less embedded figures (coverage WARN, #1478) ────────

_CHECK41_NAME = "figure sidecar coverage (sidecar-less embedded figures)"


def _make_repo_with_figure_meta_plus_bare(tmp_path):
    """Like `_make_repo_with_figure_meta` (hero.png + hero.meta.json), plus
    an extra `bare.png` committed WITHOUT a sidecar in the same tree — the
    mixed sidecar-ed / sidecar-less fixture check 41's numerator/denominator
    test reads; return (repo_path, head_sha)."""
    repo = tmp_path / "figrepo41"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    fig_dir = repo / "figures" / "issue_999"
    fig_dir.mkdir(parents=True)
    (fig_dir / "hero.png").write_bytes(b"\x89PNG fake bytes")
    (fig_dir / "hero.meta.json").write_text(json.dumps({"description": "clean"}, indent=2) + "\n")
    (fig_dir / "bare.png").write_bytes(b"\x89PNG fake bytes")
    git("add", "figures")
    git("commit", "-q", "-m", "add sidecar-ed hero + sidecar-less bare figure")
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, sha


def _check41_two_figure_body(sha: str) -> str:
    """Minimal legacy-shape body embedding TWO same-repo sha-pinned figures
    under `## TL;DR` (the legacy figure-scan section): hero.png (sidecar-ed)
    + bare.png (sidecar-less)."""
    base = (
        "https://raw.githubusercontent.com/superkaiba/explore-persona-space/"
        f"{sha}/figures/issue_999/"
    )
    return f"# Title\n\n## TL;DR\n\n![hero]({base}hero.png)\n\n![bare]({base}bare.png)\n"


def test_check41_sidecar_missing_warns(tmp_path, monkeypatch):
    """PNG committed WITHOUT a sidecar → WARN (passed=True, is_warn=True)
    naming the basename + the skipped checks; must not flip the body's
    overall verdict. THE durability pin for #1478."""
    repo, sha = _make_repo_with_figure(tmp_path)  # PNG only, no .meta.json
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    _ok, results = verify_task_body.verify_text(body)
    res = _results_by_name(results)[_CHECK41_NAME]
    assert res.passed and res.is_warn, res.render()
    assert "hero.png" in res.detail and "24/28/33/34" in res.detail
    assert "1 sidecar-less" in res.detail
    assert _CHECK41_NAME not in {r.name for r in results if not r.passed}


def test_check41_sidecar_present_passes_clean(tmp_path, monkeypatch):
    """PNG + sidecar committed → clean PASS (no WARN)."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, {"description": "clean"})
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_sidecar_coverage(body)
    assert res.passed and not res.is_warn, res.render()
    assert "all carry sidecar files" in res.detail


def test_check41_repo_unresolved_noop_pass(monkeypatch):
    """Offline / --body-stdin → NO-OP PASS."""
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: None)
    res = verify_task_body.check_figure_sidecar_coverage(_CHECK24_BODY)
    assert res.passed and not res.is_warn
    assert "repo root unresolved" in res.detail


def test_check41_unresolvable_sha_skips(tmp_path, monkeypatch):
    """The cited sha does not resolve (fake sha vs a real repo) → the figure
    is skipped (check 22's domain), NO WARN."""
    repo, _real_sha = _make_repo_with_figure(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    res = verify_task_body.check_figure_sidecar_coverage(_CHECK24_BODY)  # fake sha kept
    assert res.passed and not res.is_warn, res.render()
    assert "no same-repo sha-pinned figures to check" in res.detail


def test_check41_non_same_repo_url_ignored(tmp_path, monkeypatch):
    """A raw-GitHub figure on ANOTHER owner/repo is out of scope → NO-OP PASS."""
    repo, sha = _make_repo_with_figure(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha).replace(
        "raw.githubusercontent.com/superkaiba/", "raw.githubusercontent.com/otherorg/"
    )
    res = verify_task_body.check_figure_sidecar_coverage(body)
    assert res.passed and not res.is_warn, res.render()


def test_check41_mixed_body_names_only_missing(tmp_path, monkeypatch):
    """A body mixing a sidecar-ed and a sidecar-less figure (r1 Statistics
    concern — the numerator/denominator path must not rest on the mutable
    live #1332 fixture): WARN names ONLY the sidecar-less basename, with
    denominator 2."""
    repo, sha = _make_repo_with_figure_meta_plus_bare(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check41_two_figure_body(sha)  # hero.png (sidecar-ed) + bare.png (sidecar-less)
    res = verify_task_body.check_figure_sidecar_coverage(body)
    assert res.passed and res.is_warn, res.render()
    assert "bare.png" in res.detail and "hero.png" not in res.detail
    assert "1 sidecar-less" in res.detail and "of 2 same-repo embedded" in res.detail


# ─── Checks 52/53: PNG↔sidecar render pairing + slot completeness (#2016) ───
#
# Incident #1768: a committed figure PNG drew 3 of 8 arm groups while its
# committed sidecar described all 8 — a cross-call PAIRING failure (the
# sidecar write sits outside savefig_paper's formats loop). Check 52 compares
# the per-call `render_id` the writer now stamps into the PNG's `RenderId`
# pnginfo chunk and the sidecar's `render_id` key (text-chunk read only, NO
# pixel decode); check 53 is the sidecar-internal "labeled K categories,
# covered M<K slots" companion (which deliberately does NOT cover #1768).

_CHECK52_NAME = "figure PNG/sidecar render pairing (render_id)"
_CHECK53_NAME = "figure sidecar categorical-slot completeness"


def _real_png_bytes(render_id: str | None = None) -> bytes:
    """A tiny REAL PNG (PIL-parseable, 4x4 white) carrying a `Commit` text
    chunk plus — when ``render_id`` is given — the `RenderId` chunk the
    #2016 writer stamps. Checks 52/53 parse the committed PNG's text chunks
    with PIL, so the older fixtures' fake ``b"\\x89PNG"`` bytes do not
    suffice here."""
    import io as _io

    from PIL import Image, PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text("Commit", "abc1234")
    if render_id is not None:
        info.add_text("RenderId", render_id)
    buf = _io.BytesIO()
    Image.new("RGB", (4, 4), (255, 255, 255)).save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def _make_repo_check52(tmp_path, png_bytes: bytes, meta: dict | None):
    """Throwaway repo whose HEAD commit carries `figures/issue_999/hero.png`
    (REAL PNG bytes) and — when ``meta`` is not None — its sibling
    `hero.meta.json`, plus GOOD_BODY's `scripts/run.py` (the
    `_make_repo_with_figure_meta` convention); return (repo_path, head_sha)."""
    repo = tmp_path / "figrepo52"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    fig_dir = repo / "figures" / "issue_999"
    fig_dir.mkdir(parents=True)
    (fig_dir / "hero.png").write_bytes(png_bytes)
    if meta is not None:
        (fig_dir / "hero.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    script = repo / "scripts" / "run.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('entry script')\n")
    git("add", "figures", "scripts")
    git("commit", "-q", "-m", "add real-PNG hero figure (+ optional sidecar)")
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, sha


def test_check52_render_id_mismatch_fails(tmp_path, monkeypatch):
    """§7 test 1 — PNG stamped `RenderId=aaaa…` beside a sidecar carrying
    `render_id: bbbb…` ⇒ FAIL naming the figure basename (the #1768
    pairing-failure shape, now provable)."""
    repo, sha = _make_repo_check52(
        tmp_path,
        _real_png_bytes(render_id="a" * 16),
        {"render_id": "b" * 16, "formats_written": ["png", "pdf"], "created": "2026-08-08"},
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_png_sidecar_pairing(body)
    assert not res.passed, res.render()
    assert "hero.png" in res.detail
    assert "a" * 16 in res.detail and "b" * 16 in res.detail
    assert "DIFFERENT savefig_paper calls" in res.detail


def test_check52_render_id_match_passes(tmp_path, monkeypatch):
    """§7 test 2 — ids equal ⇒ clean PASS (no WARN)."""
    repo, sha = _make_repo_check52(
        tmp_path,
        _real_png_bytes(render_id="c" * 16),
        {"render_id": "c" * 16, "formats_written": ["png", "pdf"], "created": "2026-08-08"},
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_png_sidecar_pairing(body)
    assert res.passed and not res.is_warn, res.render()
    assert "every stamped PNG/sidecar pair agrees" in res.detail


def test_check52_formats_written_omits_png_fails(tmp_path, monkeypatch):
    """§7 test 3 — sidecar `formats_written: ["pdf"]` while a PNG resolves at
    the sha ⇒ FAIL (the format-partial #1768 mechanism), independent of any
    render-id stamp."""
    repo, sha = _make_repo_check52(
        tmp_path,
        _real_png_bytes(),  # Commit chunk only — no RenderId
        {"formats_written": ["pdf"], "created": "2026-08-08"},
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_png_sidecar_pairing(body)
    assert not res.passed, res.render()
    assert "omits 'png'" in res.detail and "hero.png" in res.detail


def test_check52_grandfathered_silent_skip(tmp_path, monkeypatch):
    """§7 test 4 — PNG with only a `Commit` chunk + sidecar with no
    `render_id` (the entire pre-stamp corpus) ⇒ PASS, and the message says
    how many figures were skipped."""
    repo, sha = _make_repo_check52(
        tmp_path,
        _real_png_bytes(),  # Commit chunk only
        {"description": "clean", "created": "2026-08-08"},
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_png_sidecar_pairing(body)
    assert res.passed and not res.is_warn, res.render()
    assert "1 of 1 figure(s) skipped" in res.detail
    assert "pre-stamp grandfathered" in res.detail


def test_check52_no_sidecar_passes(tmp_path, monkeypatch):
    """§7 test 5 — no sidecar ⇒ PASS (never blocks; check 41's domain)."""
    repo, sha = _make_repo_check52(tmp_path, _real_png_bytes(render_id="d" * 16), None)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_png_sidecar_pairing(body)
    assert res.passed and not res.is_warn, res.render()
    assert "no same-repo sha-pinned PNG+sidecar figures to check" in res.detail


def test_check52_asymmetric_stamped_sidecar_warns(tmp_path, monkeypatch):
    """§7 test 5b — sidecar HAS `render_id`, PNG has no `RenderId` chunk ⇒
    WARN (the transition-window / chunk-stripped shape — §4(A))."""
    repo, sha = _make_repo_check52(
        tmp_path,
        _real_png_bytes(),  # Commit chunk only
        {"render_id": "e" * 16, "formats_written": ["png", "pdf"], "created": "2026-08-08"},
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_png_sidecar_pairing(body)
    assert res.passed and res.is_warn, res.render()
    assert "no `RenderId` text chunk" in res.detail and "hero.png" in res.detail


def test_check52_asymmetric_stamped_png_warns(tmp_path, monkeypatch):
    """Symmetric asymmetric-pair direction (implementer-documented extension
    of §4(A)): PNG stamped, sidecar unstamped ⇒ WARN (a stale sidecar
    committed beside a fresh PNG — e.g. partial staging)."""
    repo, sha = _make_repo_check52(
        tmp_path,
        _real_png_bytes(render_id="f" * 16),
        {"description": "stale pre-stamp sidecar", "created": "2026-08-08"},
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_png_sidecar_pairing(body)
    assert res.passed and res.is_warn, res.render()
    assert "sidecar has no `render_id`" in res.detail


def test_check52_rides_verify_text(tmp_path, monkeypatch):
    """A check-52 FAIL flips the overall verdict through verify_text (it is
    a registered FAIL-capable check, not WARN-only)."""
    repo, sha = _make_repo_check52(
        tmp_path,
        _real_png_bytes(render_id="a" * 16),
        {"render_id": "b" * 16, "created": "2026-08-08"},
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    ok, results = verify_task_body.verify_text(body)
    res = _results_by_name(results)[_CHECK52_NAME]
    assert not res.passed
    assert not ok


def _check53_meta(points: list[dict], axes: list[dict]) -> dict:
    """Assemble a minimal check-53 sidecar: `points` + `text.axes` (the two
    structures the check joins), plus provenance filler."""
    return {
        "created": "2026-08-08T00:00:00Z",
        "points": points,
        "n_series": len({p.get("_group") for p in points}),
        "total_points": len(points),
        "truncated": False,
        "text": {"suptitle": None, "fig_texts": [], "axes": axes},
    }


def _check53_run(tmp_path, monkeypatch, meta: dict):
    """Commit a real PNG + ``meta`` sidecar and run check 53 on a body
    embedding it (shared driver for the §7 item 6-8 fixtures)."""
    repo, sha = _make_repo_check52(tmp_path, _real_png_bytes(render_id="a" * 16), meta)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _CHECK24_BODY.replace("0123456789abcdef", sha)
    return verify_task_body.check_figure_sidecar_slot_completeness(body)


def test_check53_integer_arm_warns(tmp_path, monkeypatch):
    """§7 test 6 — 8 xticklabels (≥1 non-numeric) with bar groups covering
    x-slots 0,1,2 only ⇒ WARN naming K=8, M=3 (the reshaped form of the
    degraded #1768 figure's would-be sidecar)."""
    points = [{"condition": float(i), "share": 0.5 + i / 10, "_kind": "bar"} for i in range(3)]
    axes = [
        {
            "ylabel": "share",
            "xticklabels": ["syc-a", "syc-b", "syc-c", "cas-d", "imp-e", "imp-f", "mk-g", "mk-h"],
        }
    ]
    res = _check53_run(tmp_path, monkeypatch, _check53_meta(points, axes))
    assert res.passed and res.is_warn, res.render()
    assert "K=8" in res.detail and "M=3" in res.detail
    assert "integer-slot arm" in res.detail and "axes[0]" in res.detail


def test_check53_string_arm_warns(tmp_path, monkeypatch):
    """§7 test 6b — single-series bar sidecar with 3 distinct category-STRING
    x values against 8 xticklabels ⇒ WARN (strings are categorical by
    construction — no integer predicate needed)."""
    points = [
        {"condition": name, "share": 0.4, "_kind": "bar"} for name in ("syc-a", "syc-b", "syc-c")
    ]
    axes = [
        {
            "ylabel": "share",
            "xticklabels": ["syc-a", "syc-b", "syc-c", "cas-d", "imp-e", "imp-f", "mk-g", "mk-h"],
        }
    ]
    res = _check53_run(tmp_path, monkeypatch, _check53_meta(points, axes))
    assert res.passed and res.is_warn, res.render()
    assert "K=8" in res.detail and "M=3" in res.detail and "string arm" in res.detail


def test_check53_continuous_axis_never_fires(tmp_path, monkeypatch):
    """§7 test 7 (the false-positive guard the critic surfaced) — integer
    x ∈ {0,1,2} under 6 purely NUMERIC auto tick labels (0.0…2.5, neither
    non-numeric nor equal to slots 0..5) ⇒ NO WARN. This test FAILS against
    a check-53 predicate lacking the positive-categorical-evidence
    requirement (M=3 < K=6 would fire)."""
    points = [{"layer": float(i), "share": 0.1 * i, "_kind": "line"} for i in range(3)]
    axes = [{"ylabel": "share", "xticklabels": ["0.0", "0.5", "1.0", "1.5", "2.0", "2.5"]}]
    res = _check53_run(tmp_path, monkeypatch, _check53_meta(points, axes))
    assert res.passed and not res.is_warn, res.render()
    assert "no labeled-K/covered-M slot gap" in res.detail


def test_check53_twin_axes_never_fires(tmp_path, monkeypatch):
    """§7 test 7b — a right-hand twin axes carrying inherited numeric tick
    labels with a single overlaid line group ⇒ no WARN (the categorical-
    evidence requirement is what keeps twins from firing)."""
    points = [{"epoch": float(i), "loss": 1.0 - 0.1 * i, "_kind": "line"} for i in range(3)]
    numeric_ticks = ["0.0", "0.5", "1.0", "1.5", "2.0", "2.5"]
    axes = [
        {"ylabel": "accuracy", "xticklabels": numeric_ticks},
        {"ylabel": "loss", "xticklabels": numeric_ticks},  # the twinx entry
    ]
    res = _check53_run(tmp_path, monkeypatch, _check53_meta(points, axes))
    assert res.passed and not res.is_warn, res.render()


def test_check53_mathtext_log_ticks_never_fires(tmp_path, monkeypatch):
    """Kill-criterion-1 tightening (#2016 corpus sweep): a log-scale
    continuous axis whose mathtext major ticks (`$\\mathdefault{10^{-3}}$`)
    fail a plain float parse must NOT read as categorical evidence — the
    only three corpus WARNs (issues #1482/#1489/#1768) were exactly this
    shape: sub-0.5 x data rounding into slot 0 under K mathtext ticks.
    FAILS against a predicate whose tick parse lacks the mathtext branch."""
    points = [
        {"effective loss mass": v, "fraction closed": 0.1} for v in (0.001, 0.01, 0.1, 0.3, 0.45)
    ]
    for p in points:
        p["_kind"] = "line"
    ticks = [f"$\\mathdefault{{10^{{{e}}}}}$" for e in range(-5, 2)]  # K=7 log majors
    axes = [{"ylabel": "fraction closed", "xticklabels": ticks}]
    res = _check53_run(tmp_path, monkeypatch, _check53_meta(points, axes))
    assert res.passed and not res.is_warn, res.render()
    assert "no labeled-K/covered-M slot gap" in res.detail


def test_check53_horizontal_panel_excluded(tmp_path, monkeypatch):
    """§7 test 8 — a horizontal panel (value on x: the group's value key
    matches the axes' `xlabel`, the axes carries no `ylabel`) never joins ⇒
    excluded, no WARN."""
    points = [{"projection of target": 0.25 * i, "y": float(i), "_kind": "bar"} for i in range(4)]
    axes = [
        {
            "xlabel": "projection of target",
            "xticklabels": ["a", "b", "c", "d", "e", "f", "g", "h"],
        }
    ]
    res = _check53_run(tmp_path, monkeypatch, _check53_meta(points, axes))
    assert res.passed and not res.is_warn, res.render()


def test_check53_truncated_sidecar_skipped(tmp_path, monkeypatch):
    """A `data_truncated` sidecar is skipped — the row cap can drop whole
    groups, so M would understate (fail-soft, never a WARN)."""
    points = [{"condition": 0.0, "share": 0.5, "_kind": "bar"}]
    axes = [{"ylabel": "share", "xticklabels": ["a", "b", "c"]}]
    meta = _check53_meta(points, axes)
    meta["data_truncated"] = True
    res = _check53_run(tmp_path, monkeypatch, meta)
    assert res.passed and not res.is_warn, res.render()
    assert "no same-repo sha-pinned figures with a points+text sidecar" in res.detail


def test_savefig_paper_render_id_round_trip(tmp_path):
    """§7 test 9 — the writer round-trip: `savefig_paper` on a tiny real
    figure stamps the SAME 16-hex id into the PNG's `RenderId` chunk (read
    off the FINAL file, i.e. it survives the PIL re-tag re-save — the §5
    note) and the sidecar's `render_id`; `formats_written == ["png","pdf"]`;
    every pre-existing sidecar key is untouched (strictly additive)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from explore_persona_space.analysis import paper_plots

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    written = paper_plots.savefig_paper(fig, "roundtrip", dir=tmp_path)
    plt.close(fig)
    meta = json.loads(written["meta"].read_text())
    with Image.open(written["png"]) as img:
        info = dict(img.info)
    assert re.fullmatch(r"[0-9a-f]{16}", meta["render_id"])
    assert info.get("RenderId") == meta["render_id"]
    assert "Commit" in info  # the pre-existing chunk is untouched
    assert meta["formats_written"] == ["png", "pdf"]
    assert {"commit", "created", "figsize", "points"} <= set(meta.keys())


def test_savefig_paper_pdf_only_formats_written(tmp_path):
    """§7 test 10 — `formats=("pdf",)` ⇒ sidecar `formats_written ==
    ["pdf"]` and no PNG written: the founding-defect shape (a format-partial
    call refreshing the sidecar without touching the PNG), pinned."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis import paper_plots

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    written = paper_plots.savefig_paper(fig, "pdfonly", dir=tmp_path, formats=("pdf",))
    plt.close(fig)
    meta = json.loads(written["meta"].read_text())
    assert meta["formats_written"] == ["pdf"]
    assert "png" not in written
    assert not (tmp_path / "pdfonly.png").exists()
    assert re.fullmatch(r"[0-9a-f]{16}", meta["render_id"])


def test_checks_52_53_registry_membership():
    """§7 test 11 (the #1520 house pattern) — one-line membership asserts:
    a `len(CHECKS)` count pin only fails in the entry-ADDED direction, so
    it does not substitute for these."""
    assert verify_task_body.check_figure_png_sidecar_pairing in verify_task_body.CHECKS
    assert verify_task_body.check_figure_sidecar_slot_completeness in verify_task_body.CHECKS


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


# ─── Judge drop-line population reconciliation (#1776 incident, task #1881) ─
#
# Signal: a judge-health drop-line sentence "<X> content drops [and <T>
# transport losses] of|across <Y> draws" in the fence-stripped
# Methodology+Results region, reconciled against schema-keyed judge-artifact
# populations (dict leaves carrying numeric `content_drops` + `valid_draws`)
# under eval_results/issue_<N>/. FAIL only on the provably CROSSED pair
# (numerator from one population, denominator from another, no single
# population matching both — the #1776 incident shape); WARN when nothing
# reconciles; graceful PASS everywhere else. Ground truth (#1776
# followup_p3p4/judge/judge_scores.json): all-arms (192, 67,500),
# baseline-excluded (156, 56,250), per-trait 22,500 draws each.

_CHECK1881_NAME = "judge drop-line population reconciles"

# The VERBATIM incident sentence, recovered from #1776's body.md git history
# (commit fcc5a5d47bc3c9fb8842c70708340d80ca1b9842, tasks/*/1776/body.md L71)
# — the all-arms drop numerator (192) quoted over the steered-only draw
# denominator (56,250).
_CHECK1881_INCIDENT_SENTENCE = (
    "Dose-round judge health: 192 content drops of 56,250 draws "
    "(0.34%, worst arm 0.9%), zero transport losses, zero empty rollouts."
)


def _drop_line_body(sentence: str) -> str:
    """A valid v4 body whose Methodology `**Evaluation:**` line carries
    `sentence` (a judge-health drop-line). Asserts the splice landed so a
    future `_V4_GOOD_BODY` rewording fails loud instead of silently testing
    a drop-line-less body."""
    out = _V4_GOOD_BODY.replace(
        "- **Evaluation:** Betley alignment score, Claude Sonnet judge, 200 probes; "
        "chosen to match the prior eval surface; no preprocessing.",
        "- **Evaluation:** Betley alignment score, Claude Sonnet judge, 200 probes; "
        f"chosen to match the prior eval surface; no preprocessing. Judge health: {sentence}",
    )
    assert sentence in out, "fixture splice failed — _V4_GOOD_BODY Evaluation line changed"
    return out


def _make_drop_population_tree(root: Path, issue: int) -> Path:
    """Write a synthetic #1776-followup_p3p4-shaped judge summary under
    `root/eval_results/issue_<N>/judge/judge_scores.json`: per_arm[trait][arm]
    leaves carrying `content_drops`/`valid_draws`/`transport_losses`,
    reproducing the incident ground-truth totals — whole-file (192, 67,500),
    baseline-excluded (156, 56,250), per-trait 22,500 draws each
    (evil 20, sycophancy 40, hallucination 132 drops)."""
    eval_dir = root / "eval_results" / f"issue_{issue}" / "judge"
    eval_dir.mkdir(parents=True, exist_ok=True)

    def leaf(drops: int, total: int) -> dict:
        return {
            "mean_score": 1.0,
            "content_drops": drops,
            "valid_draws": total - drops,
            "transport_losses": 0,
        }

    per_arm = {}
    for trait, steered_drops in (("evil", 8), ("sycophancy", 28), ("hallucination", 120)):
        per_arm[trait] = {
            "baseline_a0": leaf(12, 3750),
            f"{trait}_a1": leaf(steered_drops, 6250),
            f"{trait}_a2": leaf(0, 6250),
            f"{trait}_a3": leaf(0, 6250),
        }
    (eval_dir / "judge_scores.json").write_text(json.dumps({"per_arm": per_arm, "n_draws": 25}))
    return eval_dir


def test_judge_drop_line_crossed_pairing_fails(tmp_path):
    """The VERBATIM #1776 incident sentence — an all-arms drop numerator
    (192, whole-file) quoted over the steered-only draw denominator
    (56,250, baseline-excluded) — FAILs as a provably crossed population
    pair, naming BOTH consistent pairings (192/67,500 and 156/56,250)."""
    _make_drop_population_tree(tmp_path, 999)
    body = _drop_line_body(_CHECK1881_INCIDENT_SENTENCE)
    res = verify_task_body.check_judge_drop_line_population(body, issue=999, eval_root=tmp_path)
    assert not res.passed, res.render()
    assert "CROSSED" in res.detail, res.render()
    assert "192/67,500" in res.detail, res.render()
    assert "156/56,250" in res.detail, res.render()


def test_judge_drop_line_all_arms_exact_passes(tmp_path):
    """The corrected all-arms form (192, 67,500) matches the whole-file
    population on both coordinates → PASS."""
    _make_drop_population_tree(tmp_path, 999)
    body = _drop_line_body("192 content drops of 67,500 draws (0.28%).")
    res = verify_task_body.check_judge_drop_line_population(body, issue=999, eval_root=tmp_path)
    assert res.passed and not res.is_warn, res.render()
    assert "reconcile" in res.detail, res.render()


def test_judge_drop_line_baseline_excluded_passes(tmp_path):
    """The steered-only form (156, 56,250) matches the baseline-excluded
    population → PASS (the honest way to quote the steered denominator)."""
    _make_drop_population_tree(tmp_path, 999)
    body = _drop_line_body("156 content drops of 56,250 draws (0.28%).")
    res = verify_task_body.check_judge_drop_line_population(body, issue=999, eval_root=tmp_path)
    assert res.passed and not res.is_warn, res.render()


def test_judge_drop_line_per_subtree_passes(tmp_path):
    """A per-trait claim (132, 22,500) matches the per_arm.hallucination
    subtree population → PASS (per-subtree candidates resolve naturally)."""
    _make_drop_population_tree(tmp_path, 999)
    body = _drop_line_body("hallucination: 132 content drops of 22,500 draws.")
    res = verify_task_body.check_judge_drop_line_population(body, issue=999, eval_root=tmp_path)
    assert res.passed and not res.is_warn, res.render()


def test_judge_drop_line_transport_and_across_variants(tmp_path):
    """The `and <T> transport losses` + `across` phrasing parses, Y is
    accepted as drops+valid+transport, and discovery is SCHEMA-keyed — the
    leaves live in a file NOT named judge_scores*.json (the #1776
    judge_swap.json class)."""
    eval_dir = tmp_path / "eval_results" / "issue_999"
    eval_dir.mkdir(parents=True)
    payload = {
        "per_arm": {
            "a_retention": {
                "swap": {"content_drops": 0, "valid_draws": 13300, "transport_losses": 0}
            },
            "b_content": {
                "swap": {"content_drops": 0, "valid_draws": 13295, "transport_losses": 5}
            },
        }
    }
    (eval_dir / "judge_swap.json").write_text(json.dumps(payload))
    body = _drop_line_body("0 content drops and 5 transport losses across 26,600 draws.")
    res = verify_task_body.check_judge_drop_line_population(body, issue=999, eval_root=tmp_path)
    assert res.passed and not res.is_warn, res.render()


def test_judge_drop_line_regex_singular_and_denominatorless():
    """Regex shape pins: singular `content drop` / `transport loss` parse;
    a denominator-less mention carries no population pair and never
    matches (critic non-blocking item 5)."""
    m = verify_task_body._JUDGE_DROP_LINE_RE.search(
        "1 content drop and 1 transport loss of 100 draws"
    )
    assert m is not None
    assert m.group("drops") == "1" and m.group("transport") == "1" and m.group("draws") == "100"
    assert verify_task_body._JUDGE_DROP_LINE_RE.search("1,938 content drops (1.6%)") is None


def test_judge_drop_line_denominatorless_sentence_no_claim(tmp_path):
    """A drop mention with no `of|across <Y> draws` denominator asserts no
    population pair → PASS 'no judge drop-line asserted' (even with a
    reconcilable artifact tree present)."""
    _make_drop_population_tree(tmp_path, 999)
    body = _drop_line_body("1,938 content drops (1.6%), zero transport losses.")
    res = verify_task_body.check_judge_drop_line_population(body, issue=999, eval_root=tmp_path)
    assert res.passed and not res.is_warn, res.render()
    assert "no judge drop-line asserted" in res.detail, res.render()


def test_judge_drop_line_unreconcilable_warns(tmp_path):
    """A claim matching NO candidate on either coordinate → WARN (an
    unenumerated honest subset is plausible — only the crossed signature
    FAILs), listing the nearest candidates."""
    _make_drop_population_tree(tmp_path, 999)
    body = _drop_line_body("7 content drops of 12,345 draws.")
    res = verify_task_body.check_judge_drop_line_population(body, issue=999, eval_root=tmp_path)
    assert res.passed and res.is_warn, res.render()
    assert "could not reconcile" in res.detail, res.render()
    assert "nearest candidates" in res.detail, res.render()


def test_judge_drop_line_no_artifacts_graceful_pass(tmp_path):
    """No leaf-bearing artifact (JSONs without the content_drops/valid_draws
    schema) → graceful PASS, never a false FAIL on missing data."""
    eval_dir = tmp_path / "eval_results" / "issue_999"
    eval_dir.mkdir(parents=True)
    (eval_dir / "summary.json").write_text(json.dumps({"n_total": 400, "rate": 0.1}))
    body = _drop_line_body(_CHECK1881_INCIDENT_SENTENCE)
    res = verify_task_body.check_judge_drop_line_population(body, issue=999, eval_root=tmp_path)
    assert res.passed and not res.is_warn, res.render()
    assert "graceful skip" in res.detail, res.render()


def test_judge_drop_line_fence_env_skips(tmp_path, monkeypatch):
    """EPM_VERIFY_BODY_NO_EVAL_SCAN=1 fences the disk read → skip-PASS even
    on a body+tree pair that would otherwise FAIL crossed."""
    _make_drop_population_tree(tmp_path, 999)
    monkeypatch.setenv("EPM_VERIFY_BODY_NO_EVAL_SCAN", "1")
    body = _drop_line_body(_CHECK1881_INCIDENT_SENTENCE)
    res = verify_task_body.check_judge_drop_line_population(body, issue=999, eval_root=tmp_path)
    assert res.passed and not res.is_warn, res.render()
    assert "EPM_VERIFY_BODY_NO_EVAL_SCAN" in res.detail, res.render()


def test_judge_drop_line_legacy_body_skips():
    """Legacy / v2 bodies PASS vacuously (forward-grandfathering, the #732
    convention)."""
    res = verify_task_body.check_judge_drop_line_population(GOOD_BODY, issue=999)
    assert res.passed and not res.is_warn, res.render()
    assert "legacy" in res.detail, res.render()


def test_judge_drop_line_stdin_issue_unknown_skips(tmp_path):
    """issue=None (stdin invocation) → skip-PASS before any disk read."""
    body = _drop_line_body(_CHECK1881_INCIDENT_SENTENCE)
    res = verify_task_body.check_judge_drop_line_population(body, issue=None)
    assert res.passed, res.render()
    assert "issue number unknown" in res.detail, res.render()


def test_judge_drop_line_registered_in_verify_text():
    """Registration-membership pin (the #1016 by-name convention, critic
    non-blocking item 1): a forgotten verify_text append cannot ship
    green — the check's result row must appear in every verify_text run."""
    _ok, results = verify_task_body.verify_text(GOOD_BODY)
    assert _CHECK1881_NAME in {r.name for r in results}


def test_judge_drop_line_multi_claim_worst_verdict_wins(tmp_path):
    """Multiple drop-line claims are evaluated independently and the WORST
    verdict wins (ladder step 7): PASS + crossed → FAIL; PASS +
    unreconcilable → WARN."""
    _make_drop_population_tree(tmp_path, 999)
    body_fail = _drop_line_body(
        "192 content drops of 67,500 draws overall; dose round: 192 content drops of 56,250 draws."
    )
    res = verify_task_body.check_judge_drop_line_population(
        body_fail, issue=999, eval_root=tmp_path
    )
    assert not res.passed, res.render()
    body_warn = _drop_line_body(
        "192 content drops of 67,500 draws overall; dose round: 7 content drops of 12,345 draws."
    )
    res2 = verify_task_body.check_judge_drop_line_population(
        body_warn, issue=999, eval_root=tmp_path
    )
    assert res2.passed and res2.is_warn, res2.render()


def test_judge_drop_line_zero_numerator_degraded_fingerprint(tmp_path):
    """A crossed FAIL whose numerator (0) matches MULTIPLE candidate
    populations additionally notes the degraded population fingerprint
    (critic non-blocking item 2)."""
    eval_dir = tmp_path / "eval_results" / "issue_999"
    eval_dir.mkdir(parents=True)
    (eval_dir / "a.json").write_text(json.dumps({"x": {"content_drops": 0, "valid_draws": 26600}}))
    (eval_dir / "b.json").write_text(json.dumps({"x": {"content_drops": 0, "valid_draws": 12000}}))
    (eval_dir / "c.json").write_text(json.dumps({"x": {"content_drops": 5, "valid_draws": 9995}}))
    body = _drop_line_body("0 content drops of 10,000 draws.")
    res = verify_task_body.check_judge_drop_line_population(body, issue=999, eval_root=tmp_path)
    assert not res.passed, res.render()
    assert "population fingerprint degraded" in res.detail, res.render()


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


# ─── Check 37: footer Reused bullets carry a revision/path pin (#1370) ──────
#
# Body-text-only sibling of Check 35 (#1256) — the body->pin direction
# (#1315: two unpinned `- Reused ... from [#1090]` footer bullets while
# Check 35 graceful-skipped; caught only by the LM critic). Direct-call
# style on `_V4_GOOD_BODY`-derived fixtures; `_V4_GOOD_BODY`'s own footer
# hex tokens (`0123456789abcdef` / `abc123def`) never contaminate because
# the satisfiers are BULLET-scoped.

_CHECK1370_UNPINNED_BULLET = (
    "\n- Reused LoRA adapters from [#1090](https://eps.superkaiba.com/tasks/1090): "
    "persona-context checkpoint — fit: same base model + recipe.\n"
)


def test_footer_reuse_bullet_unpinned_warns():
    """MAIN / durability-pin test (#1370, incident #1315): a v4 footer
    `- Reused ... from [#M](...)` bullet with no pin -> WARN naming the
    bullet + the expected shape. Also pins the vacuity guard: the
    from-link's own `/tasks/1090` + `#1090` must NOT satisfy."""
    body = _V4_GOOD_BODY + _CHECK1370_UNPINNED_BULLET
    res = verify_task_body.check_footer_reuse_bullets_pinned(body)
    assert res.passed and res.is_warn, res.render()
    assert "#1090" in res.detail and "@ <rev>" in res.detail, res.render()


def test_footer_reuse_bullet_tree_sha_passes():
    """Satisfier 1 — revision-URL segment: an HF `/tree/<sha>` URL in the
    bullet -> no warn. Digit-only sha so the letter-bearing bare-hex form
    cannot co-satisfy — this test pins the rev-URL regex itself."""
    body = _V4_GOOD_BODY + (
        "\n- Reused LoRA adapters from [#1090](https://eps.superkaiba.com/tasks/1090): "
        "[adapters](https://huggingface.co/superkaiba1/explore-persona-space/tree/1234567890123/"
        "adapters) — fit: same base model.\n"
    )
    res = verify_task_body.check_footer_reuse_bullets_pinned(body)
    assert res.passed and not res.is_warn, res.render()


def test_footer_reuse_bullet_at_rev_passes():
    """Satisfier 2 — `@ <rev>` (backtick-tolerant): ``@ `1234567` `` in the
    bullet -> no warn. Digit-only rev so ONLY the `@` form satisfies."""
    body = _V4_GOOD_BODY + (
        "\n- Reused train mix from [#1090](https://eps.superkaiba.com/tasks/1090): "
        "issue1090_fu3/train_mix.jsonl @ `1234567` — fit: same recipe.\n"
    )
    res = verify_task_body.check_footer_reuse_bullets_pinned(body)
    assert res.passed and not res.is_warn, res.render()


def test_footer_reuse_bullet_eval_results_path_passes():
    """Satisfier 3 — a committed `eval_results/issue_<M>/` path in the
    bullet -> no warn (git-tree-reachable inputs are pinned by path)."""
    body = _V4_GOOD_BODY + (
        "\n- Reused geometry summary from [#653](https://eps.superkaiba.com/tasks/653): "
        "committed eval_results/issue_653/geometry_summary.json — fit: same probe set.\n"
    )
    res = verify_task_body.check_footer_reuse_bullets_pinned(body)
    assert res.passed and not res.is_warn, res.render()


def test_footer_reuse_bullet_bare_hex_token_passes():
    """Satisfier 5 — a bare letter-bearing >=7-char hex token
    (`2511ed7d`) -> no warn (a quoted short sha pins without `@`)."""
    body = _V4_GOOD_BODY + (
        "\n- Reused null matrix from [#653](https://eps.superkaiba.com/tasks/653): "
        "committed at `2511ed7d` — fit: same fold structure.\n"
    )
    res = verify_task_body.check_footer_reuse_bullets_pinned(body)
    assert res.passed and not res.is_warn, res.render()


def test_footer_reuse_bullet_digit_only_token_still_warns():
    """A digits-only >=7-char token (`12345678` — a row count, not a sha)
    does NOT satisfy the bare-hex form -> WARN (the letter requirement
    rejects counts)."""
    body = _V4_GOOD_BODY + (
        "\n- Reused judge outputs from [#653](https://eps.superkaiba.com/tasks/653): "
        "12345678 scored rows — fit: same rubric.\n"
    )
    res = verify_task_body.check_footer_reuse_bullets_pinned(body)
    assert res.passed and res.is_warn, res.render()


def test_footer_reuse_bullet_no_from_clause_skips():
    """A `- Reused` bullet WITHOUT a `from [#M](` clause (self-reuse
    "this task" form) is out of scope -> clean PASS, no warn."""
    body = _V4_GOOD_BODY + (
        "\n- Reused round-1 null matrix (this task) — fit: same seeds, no drift.\n"
    )
    res = verify_task_body.check_footer_reuse_bullets_pinned(body)
    assert res.passed and not res.is_warn, res.render()
    assert "pinned" in res.detail, res.render()


def test_footer_reuse_bullet_v3_body_vacuous_pass():
    """Forward-only: a grandfathered v3 body with the SAME unpinned bullet
    -> vacuous PASS (never newly WARN/FAIL a v3/v2 body)."""
    body = _V3_GOOD_BODY + _CHECK1370_UNPINNED_BULLET
    res = verify_task_body.check_footer_reuse_bullets_pinned(body)
    assert res.passed and not res.is_warn, res.render()
    assert "not a v4 body" in res.detail, res.render()


def test_footer_reuse_bullet_no_footer_skips():
    """A v4 body truncated before `**Repro:**` (no footer) -> PASS with
    the no-footer skip detail."""
    body = _V4_GOOD_BODY.split("**Repro:**")[0]
    res = verify_task_body.check_footer_reuse_bullets_pinned(body)
    assert res.passed and not res.is_warn, res.render()
    assert "no **Repro:** footer" in res.detail, res.render()


def test_footer_reuse_bullet_fires_under_eval_scan_fence(monkeypatch):
    """Decision 6 (#1370): `EPM_VERIFY_BODY_NO_EVAL_SCAN=1` fences eval
    scans, NOT body-text reads — the unpinned bullet still WARNs (fencing
    this check would re-open the #1315 hole in fenced invocations)."""
    monkeypatch.setenv("EPM_VERIFY_BODY_NO_EVAL_SCAN", "1")
    body = _V4_GOOD_BODY + _CHECK1370_UNPINNED_BULLET
    res = verify_task_body.check_footer_reuse_bullets_pinned(body)
    assert res.passed and res.is_warn, res.render()


def test_footer_reuse_bullet_continuation_line_pin_passes():
    """Continuation-joining: a wrapped bullet whose pin sits on an
    indented non-bullet continuation line -> no warn."""
    body = _V4_GOOD_BODY + (
        "\n- Reused LoRA adapters from [#1090](https://eps.superkaiba.com/tasks/1090):\n"
        "  persona-context checkpoint @ deadbeef — fit: same base model.\n"
    )
    res = verify_task_body.check_footer_reuse_bullets_pinned(body)
    assert res.passed and not res.is_warn, res.render()


def test_footer_reuse_bullet_wandb_run_url_passes():
    """Satisfier 4 — SPEC-sanctioned WandB `/runs/<id>` URL -> no warn.
    The run id is base36 (letters beyond a-f), so neither the rev-URL nor
    the bare-hex form can co-satisfy (v2 amendment)."""
    body = _V4_GOOD_BODY + (
        "\n- Reused training metrics from [#1090](https://eps.superkaiba.com/tasks/1090): "
        "https://wandb.ai/superkaiba1/issue1090/runs/ab12xy9z — fit: same run config.\n"
    )
    res = verify_task_body.check_footer_reuse_bullets_pinned(body)
    assert res.passed and not res.is_warn, res.render()


def test_footer_reuse_bullet_fenced_skeleton_ignored():
    """Fence-aware split (#1370 §8 risk row 4): an unpinned
    `- Reused ... from [#M](...)` line INSIDE a ```-fenced footer block
    (an illustrative skeleton) never triggers -> clean PASS."""
    body = _V4_GOOD_BODY + (
        "\n```\n"
        "- Reused <kind> from [#1090](https://eps.superkaiba.com/tasks/1090): skeleton example\n"
        "```\n"
    )
    res = verify_task_body.check_footer_reuse_bullets_pinned(body)
    assert res.passed and not res.is_warn, res.render()


def test_footer_reuse_bullet_bare_issue_token_form_warns():
    """#1739 incident shape (widened match set, #1907): a bare `#M` issue
    token + bare rev pin, NO `from [#M](...)` link -> WARN via the FORM
    arm only (the letter-bearing `037fcbb` rev satisfies the pin arm;
    the missing canonical link form is the defect)."""
    body = _V4_GOOD_BODY + (
        "\n- Reused direction bank #779 rev 037fcbb — fit: same extraction recipe.\n"
    )
    res = verify_task_body.check_footer_reuse_bullets_pinned(body)
    assert res.passed and res.is_warn, res.render()
    assert "non-canonical form" in res.detail, res.render()
    assert "unpinned" not in res.detail, res.render()
    assert "from [#M](...)" in res.detail, res.render()


def test_footer_reuse_bullet_linkful_noncanonical_unpinned_warns_both():
    """#1900 incident shape (widened match set, #1907): `from the <line>
    ([#M](...), [#K](...))` — links present but an intervening noun
    phrase after `from`, and NO pin -> WARN naming BOTH classes (the pin
    arm now runs over the widened set; the form arm also fires)."""
    body = _V4_GOOD_BODY + (
        "\n- Reused behavior read-out directions from the fleet extraction line "
        "([#1112](https://eps.superkaiba.com/tasks/1112), "
        "[#1439](https://eps.superkaiba.com/tasks/1439)) — fit: same behavior panel.\n"
    )
    res = verify_task_body.check_footer_reuse_bullets_pinned(body)
    assert res.passed and res.is_warn, res.render()
    assert "unpinned" in res.detail, res.render()
    assert "non-canonical form" in res.detail, res.render()


def test_footer_reuse_bullet_issue_path_token_pinned_form_warns():
    """#1639 shape (widened match set, #1907): cross-issue reuse cited by
    an `issue<M>_` artifact-path token with an `@ <rev>` pin but no
    `from [#M](...)` link -> WARN via the FORM arm only."""
    body = _V4_GOOD_BODY + (
        "\n- Reused artifacts: the parent store "
        "`issue1310_char_map/analysis_tensors/store.pt` @ `deadbee12` — fit: same char map.\n"
    )
    res = verify_task_body.check_footer_reuse_bullets_pinned(body)
    assert res.passed and res.is_warn, res.render()
    assert "non-canonical form" in res.detail, res.render()
    assert "unpinned" not in res.detail, res.render()


def test_footer_reuse_bullet_same_task_no_issue_token_stays_clean():
    """Same-task exclusion pin (#1907 §Grounding): a round-reuse bullet
    with paths/prose but NO issue token (no `#M`, no `[#M](`, no
    `/tasks/M`, no `issue<M>_`) stays OUT of the widened match set ->
    clean PASS (the rejected naive ANY-`- Reused` widening measured 14
    form + 8 pin firings on the live corpus, dominated by this class)."""
    body = _V4_GOOD_BODY + (
        "\n- Reused round-1 aligned-position store (this task, HF mirror) as the "
        "round-5 input — fit: same seeds.\n"
    )
    res = verify_task_body.check_footer_reuse_bullets_pinned(body)
    assert res.passed and not res.is_warn, res.render()
    assert "pinned" in res.detail, res.render()


# ─── Check 44: footer HF artifact paths carry an adjacent pinned link ───────
#
# (#1509, incident #1335): a bare backtick HF-style artifact path in the
# footer (`issue<N>_` prefix or a raw_completions/analysis_tensors segment,
# brace/glob charset) whose bullet/paragraph unit carries neither a pinned
# huggingface.co /(tree|resolve|blob)/<hex> link (S1) nor an
# immediately-following `@ [HF ]rev <hex>` (S2) -> WARN. Direct-call style
# on `_V4_GOOD_BODY`-derived fixtures (the check-37 block's convention);
# `_V4_GOOD_BODY`'s own footer is S1-pinned (the `tree/abc123def` model
# link) and carries no HF-identity backtick tokens, so appended bullets
# are the only trigger surface.

# The verbatim PRE-FIX #1335 footer bullet, recovered from
# `git show 6d3c847946` (the fix commit's `-` side). Load-bearing traps it
# carries: GitHub /tree/<sha> pins (never HF), bare code shas, an EARLIER
# `@ `be61a85e`` GitHub figures pin (must NOT rescue under S2 anchoring),
# non-HF backtick tokens (`cells_/nulls_/...`, `scripts/...`), and the
# brace-form offending token with NO count-paren (so check 40 never
# extracts it).
_I1335_FOOTER_UNPINNED_LINE = (
    "- Follow-up round `onpolicy-assistant-label` (2026-07-18, ~6 GPU-h, GCE 4×A100-80 "
    "flex-start, 6-lane parallel dispatch): round artifacts "
    "[eval_results/issue_1335/onpolicy-assistant-label/](https://github.com/superkaiba/"
    "explore-persona-space/tree/be61a85e9ae2710f254c6e0b4fd422ac5244dec1/eval_results/"
    "issue_1335/onpolicy-assistant-label) (`label_comparison.json` + per-cell "
    "`cells_/nulls_/loso_/swap_/wiring_*.json`); fit/driver `scripts/issue1335_fit.py` + "
    "`scripts/issue1335_fig_label.py` at code SHA "
    "`01ebb89738835c656f4b8a942a0d3afe4647be25`. Figures: "
    "[figures/issue_1335/onpolicy-assistant-label/](https://github.com/superkaiba/"
    "explore-persona-space/tree/be61a85e9ae2710f254c6e0b4fd422ac5244dec1/figures/"
    "issue_1335/onpolicy-assistant-label) (`hero_label_delta` + `answer_length_register`, "
    "with `.meta.json` sidecars, @ `be61a85e`; `placement_panel` + `collapse_slots` @ "
    "[`185a6bd8ee`](https://github.com/superkaiba/explore-persona-space/tree/"
    "185a6bd8ee3f9165c346d27f9876efcefe845314/figures/issue_1335/onpolicy-assistant-label)). "
    "HF rollouts and stores under "
    "`issue1335_ablation_ladder/onpolicy_assistant_label/{raw_completions,analysis_tensors}/` "
    "(verified via `list_repo_tree` on the interpretation pass)."
)
_I1335_OFFENDING_TOKEN = (
    "issue1335_ablation_ladder/onpolicy_assistant_label/{raw_completions,analysis_tensors}/"
)
_C44_NAME = "footer HF artifact paths carry an adjacent pinned link"


def test_footer_hf_path_unpinned_warns_1335_shape():
    """MAIN / durability-pin test (#1509, incident #1335): the verbatim
    pre-fix footer bullet -> WARN (`passed=True`, `is_warn=True`) naming
    the brace-form token. The in-bullet GitHub pins / bare code shas /
    earlier `@ `be61a85e`` must NOT rescue."""
    body = _V4_GOOD_BODY + "\n" + _I1335_FOOTER_UNPINNED_LINE + "\n"
    res = verify_task_body.check_footer_hf_paths_pinned(body)
    assert res.passed and res.is_warn, res.render()
    assert _I1335_OFFENDING_TOKEN in res.detail, res.render()


def test_footer_hf_path_pinned_same_unit_passes():
    """S1: the same incident bullet PLUS a pinned huggingface.co
    /tree/<sha> link in the SAME bullet (the post-fix #1335 shape) ->
    clean PASS."""
    line = _I1335_FOOTER_UNPINNED_LINE + (
        " Full tree: [rollouts](https://huggingface.co/datasets/superkaiba1/"
        "explore-persona-space-data/tree/53e014c4a530cfba76f1e5f2b29a1ae4841d46b3/"
        "issue1335_ablation_ladder/onpolicy_assistant_label)."
    )
    body = _V4_GOOD_BODY + "\n" + line + "\n"
    res = verify_task_body.check_footer_hf_paths_pinned(body)
    assert res.passed and not res.is_warn, res.render()
    assert "all footer HF artifact paths pinned" in res.detail, res.render()


def test_footer_hf_path_adjacent_at_rev_passes():
    """S2: the committed `@ rev `<hex>`` / `@ HF rev `<hex>`` footer
    shapes (#1112/#1335) immediately after the token -> clean PASS."""
    body = _V4_GOOD_BODY + (
        "\n- Reused mix: `issue1090_pvdatagen/c3-sycophancy-claude/mix/train_mix.jsonl` "
        "@ rev `6aab0cce1fac` — fit: same recipe.\n"
        "- Rollouts `issue825_userbase_map/raw_completions/track_s/` @ HF rev `deb7a452` "
        "verified live.\n"
    )
    res = verify_task_body.check_footer_hf_paths_pinned(body)
    assert res.passed and not res.is_warn, res.render()


def test_footer_preceding_at_rev_does_not_rescue():
    """S2 position-anchoring regression pin (the motivating incident's
    exact bullet shape): an `@ `<hex>`` EARLIER in the bullet (a GitHub
    figures pin), bare HF token later, no HF URL -> still WARNs."""
    body = _V4_GOOD_BODY + (
        "\n- Figures @ `be61a85e`; stores under `issue999_slug/analysis_tensors/` "
        "(verified on the interpretation pass).\n"
    )
    res = verify_task_body.check_footer_hf_paths_pinned(body)
    assert res.passed and res.is_warn, res.render()
    assert "issue999_slug/analysis_tensors/" in res.detail, res.render()


def test_footer_hf_path_adjacent_unit_pin_does_not_rescue():
    """Unit-scoped-adjacency positive control (#1509 §13.2): the
    offending token in bullet A, a pinned huggingface.co /tree/<hex>
    link in a SEPARATE blank-line-separated bullet -> still WARNs. A
    whole-footer pin scope (or a degenerate `_footer_units`) must FAIL
    this test."""
    body = _V4_GOOD_BODY + (
        "\n- Stores under `issue999_slug/raw_completions/` for the record.\n"
        "\n- Pinned elsewhere: [tree](https://huggingface.co/datasets/superkaiba1/"
        "explore-persona-space-data/tree/53e014c4a530cfba76f1e5f2b29a1ae4841d46b3/"
        "issue999_slug).\n"
    )
    res = verify_task_body.check_footer_hf_paths_pinned(body)
    assert res.passed and res.is_warn, res.render()
    assert "issue999_slug/raw_completions/" in res.detail, res.render()


def test_footer_hf_path_v3_body_skipped():
    """Forward-only: a grandfathered v3 body with the SAME offending line
    -> vacuous PASS (never newly WARN/FAIL a v3/v2 body)."""
    body = _V3_GOOD_BODY + "\n" + _I1335_FOOTER_UNPINNED_LINE + "\n"
    res = verify_task_body.check_footer_hf_paths_pinned(body)
    assert res.passed and not res.is_warn, res.render()
    assert "not a v4 body" in res.detail, res.render()


def test_footer_local_git_paths_pass():
    """G3 `_GIT_SIDE_ROOTS`: git/local backtick paths in a pin-less
    footer bullet (they resolve in git — checks 27/29 territory) ->
    clean PASS."""
    body = _V4_GOOD_BODY + (
        "\n- Round artifacts: `eval_results/issue_1310/onpolicy/*.json`, "
        "`figures/issue_1335/hero.png`, `scripts/issue1335_fit.py` (committed).\n"
    )
    res = verify_task_body.check_footer_hf_paths_pinned(body)
    assert res.passed and not res.is_warn, res.render()


def test_footer_count_claim_token_left_to_check40(monkeypatch):
    """G5 partition (the check-30/40 convention — exactly ONE WARN per
    defect): a footer count-paren claim is check 40's territory — the
    gatherer extracts it, check 44 stays silent, and check 40 names it
    (offline note-shape assert under the `EPM_VERIFY_BODY_NO_HF=1`
    fence, per the check-40 test idiom)."""
    monkeypatch.setenv("EPM_VERIFY_BODY_NO_HF", "1")
    body = _V4_GOOD_BODY + (
        "\n- HF rollouts `issue931_story_map/raw_completions/` (6 files) uploaded.\n"
    )
    claimed = {t for _c, _n, t, _r in verify_task_body._gather_hf_unpinned_count_claims(body)}
    assert "issue931_story_map/raw_completions/" in claimed
    res44 = verify_task_body.check_footer_hf_paths_pinned(body)
    assert res44.passed and not res44.is_warn, res44.render()
    res40 = verify_task_body.check_hf_unpinned_count_claims(body)
    assert res40.passed, res40.render()
    assert "issue931_story_map/raw_completions/" in res40.detail, res40.render()


def test_footer_hf_path_fence_and_blockquote_exempt():
    """`_footer_units` stripping: the offending line inside a ```-fenced
    footer block (illustrative skeleton) AND inside a `>` Context quote
    (#959 verbatim-prompt exemption) -> clean PASS."""
    body = _V4_GOOD_BODY + (
        "\n```\n" + _I1335_FOOTER_UNPINNED_LINE + "\n```\n\n> " + _I1335_FOOTER_UNPINNED_LINE + "\n"
    )
    res = verify_task_body.check_footer_hf_paths_pinned(body)
    assert res.passed and not res.is_warn, res.render()


def test_footer_hf_path_methodology_out_of_scope():
    """Footer scoping: the bare token in `## Methodology` prose (above
    the footer) is out of check 44's scope -> clean PASS (check-40
    semantics elsewhere in the body are untouched)."""
    body = _V4_GOOD_BODY.replace(
        "## Results",
        "Stores under `issue999_x/analysis_tensors/` (bare mention).\n\n## Results",
    )
    res = verify_task_body.check_footer_hf_paths_pinned(body)
    assert res.passed and not res.is_warn, res.render()
    assert "all footer HF artifact paths pinned" in res.detail, res.render()


def test_footer_hf_path_no_footer_skips():
    """No-crash on a footer-less v4 body (#1509 §13.3): `_v4_footer_text`
    -> None path returns cleanly (pass, no exception)."""
    body = _V4_GOOD_BODY.split("**Repro:**")[0]
    res = verify_task_body.check_footer_hf_paths_pinned(body)
    assert res.passed and not res.is_warn, res.render()
    assert "no **Repro:** footer" in res.detail, res.render()


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


# ─── Checks 42 + 43: body-wide git-URL existence + git-tree backtick claims ─
#
# Incident task #1072 r2 (#1507): two GitHub blob links in the `## Methodology`
# Sample `<details>` blocks 404'd (gitignored `.npz`, never committed), and the
# footer `Artifacts:` parenthetical claimed `per_context_stats_1072_fold{0..4}.npz`
# inside the pinned `eval_results/issue_1072` git tree, which lacks them —
# check 8b (footer-only) and check 32 (HF /tree-only) both PASSed the body.
# Check 42 probes same-repo blob/tree URLs body-wide (FAIL on v4, WARN on
# grandfathered); check 43 is the WARN-only git twin of check 32.

_GH_REPO = "https://github.com/superkaiba/explore-persona-space"

_BODYWIDE_NAME = "Body-wide same-repo artifact URLs exist"
_GH_TREE_NAME = "GitHub-tree-adjacent backtick file claims exist in the pinned tree"


def _gh_blob_url(sha, path):
    """Same-repo /blob/<sha>/<path> HTML URL (check 42/43 fixtures)."""
    return f"{_GH_REPO}/blob/{sha}/{path}"


def _gh_tree_url(sha, path):
    """Same-repo /tree/<sha>/<path> HTML URL (check 42/43 fixtures)."""
    return f"{_GH_REPO}/tree/{sha}/{path}"


def _make_repo_with_issue1072_tree(tmp_path):
    """Throwaway git repo whose HEAD commit carries the #1072 incident tree
    shape — `eval_results/issue_1072/` with the committed JSONs
    (`stats_component.json`, `supplementary_reads.json`,
    `battery_1072_fold0..4.json`, `capture_gates.json`,
    `stage_manifest.json`) but NO `.npz` (gitignored in the incident, never
    committed); returns (repo_path, head_sha)."""
    repo = tmp_path / "i1072repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    d = repo / "eval_results" / "issue_1072"
    d.mkdir(parents=True)
    fnames = ["stats_component.json", "supplementary_reads.json"]
    fnames += [f"battery_1072_fold{i}.json" for i in range(5)]
    fnames += ["capture_gates.json", "stage_manifest.json"]
    for fname in fnames:
        (d / fname).write_text("{}\n")
    git("add", "eval_results")
    git("commit", "-q", "-m", "add issue_1072 eval JSONs (no .npz)")
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, sha


# The VERBATIM #1072 r2 PRE-FIX footer Artifacts fragment (recover via
# `git show 12663c2e47:tasks/interpreting/1072/body.md`, footer, grep
# `per_context_stats`; the incident sha 1f19deacf… is swapped for the fixture
# repo's sha at use time — the _I952_R1_LINE convention): the paren after the
# pinned eval_results tree link claims the five gitignored `.npz` via a
# brace range — the must-WARN check-43 fixture.
_I1072_PREFIX_ARTIFACTS_LINE = (
    "Artifacts: [`eval_results/issue_1072/`]({tree_url}) "
    "(`stats_component.json`, `supplementary_reads.json`, "
    "`battery_1072_fold{{0..4}}.json`, `per_context_stats_1072_fold{{0..4}}.npz`, "
    "`capture_gates.json`, `stage_manifest.json`, pilot gates)"
)

# The VERBATIM CURRENT #1072 footer Artifacts fragment (recover from
# `tasks/followups_running/1072/body.md`, footer Artifacts clause, ~line 173;
# sha swapped to the fixture sha): same `{0..4}.npz` backtick tokens inside
# the qualifying paren, but as an explicit "gitignored … HF eval_results
# mirror only" DISCLAIMER — the must-NOT-warn check-43 fixture (AC10).
_I1072_CURRENT_ARTIFACTS_LINE = (
    "Artifacts: [`eval_results/issue_1072/`]({tree_url}) "
    "(11 JSONs: `stats_component.json`, `supplementary_reads.json`, "
    "`battery_1072_fold{{0..4}}.json`, `capture_gates.json`, "
    "`stage_manifest.json`, pilot gates; the 5 "
    "`per_context_stats_1072_fold{{0..4}}.npz` are gitignored and live on the "
    "HF eval_results mirror only)"
)

# The VERBATIM #1072 r2 Methodology Sample `<details>` shape (recover via
# `git show 12663c2e47:tasks/interpreting/1072/body.md`, Sample-slot
# `<details>` blocks at body offsets ~81/~97; blob-link sha swapped to the
# fixture sha): the 404 blob link lives INSIDE the dropdown — the check-42
# details-are-probed fixture (AC1/AC4; the durability pin).
_I1072_DETAILS_BLOCK = """<details>
<summary>Worked example — matched context 408, layer 26, mean remainder cell, fold 4</summary>

1 of 3,188 rows — a random sample (numpy seed 42); full arrays:
[`per_context_stats_1072_fold4.npz`]({blob_url}) plus folds 0-3 in the same
directory, mirrored on the HF data repo (footer).

</details>"""


_V4_RESULTS_PROSE_ANCHOR = (
    "The 17-pt lift holds at every seed; "
    "the smallest within-condition gap between seeds is 1.2 pts."
)


def test_bodywide_url_missing_path_fails_v4(tmp_path, monkeypatch):
    """AC1 — a v4 body citing a same-repo blob URL whose sha resolves locally
    but whose path is absent from that tree FAILs check 42 (detail names the
    path + sha[:8]) and flips verify_text ok=False."""
    repo, sha = _make_repo_with_issue1072_tree(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    dead = _gh_blob_url(sha, "eval_results/issue_1072/per_context_stats_1072_fold4.npz")
    body = _V4_GOOD_BODY.replace(
        _V4_RESULTS_PROSE_ANCHOR,
        f"Full arrays: [`per_context_stats_1072_fold4.npz`]({dead}).",
    )
    assert body != _V4_GOOD_BODY  # anchor still present in the fixture
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    r = _results_by_name(results)[_BODYWIDE_NAME]
    assert not r.passed
    assert "per_context_stats_1072_fold4.npz" in r.detail
    assert sha[:8] in r.detail


def test_bodywide_details_urls_are_probed(tmp_path, monkeypatch):
    """AC1/AC4 (durability pin) — the VERBATIM #1072 Sample-slot shape: the
    dead blob link sits INSIDE a `<details>` block and is still probed (the
    25300695e4 details exemption covers wording discipline only) → check-42
    FAIL on a v4 body."""
    repo, sha = _make_repo_with_issue1072_tree(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    dead = _gh_blob_url(sha, "eval_results/issue_1072/per_context_stats_1072_fold4.npz")
    details = _I1072_DETAILS_BLOCK.format(blob_url=dead)
    body = _V4_GOOD_BODY.replace(_V4_RESULTS_PROSE_ANCHOR, details)
    assert body != _V4_GOOD_BODY
    r = verify_task_body.check_body_artifact_urls_exist(body)
    assert not r.passed
    assert "per_context_stats_1072_fold4.npz" in r.detail and sha[:8] in r.detail
    ok, _results = verify_task_body.verify_text(body)
    assert not ok


def test_bodywide_url_present_passes(tmp_path, monkeypatch):
    """AC3 — a resolving non-footer same-repo URL PASSes with the counted
    `1 URL(s)` detail (pins the non-vacuous probe path — NOT the no-URLs
    vacuous PASS) and no `unverified` note."""
    repo, sha = _make_repo_with_issue1072_tree(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    live = _gh_blob_url(sha, "eval_results/issue_1072/stats_component.json")
    body = f"Stats live at [`stats_component.json`]({live}).\n"
    r = verify_task_body.check_body_artifact_urls_exist(body)
    assert r.passed and not r.is_warn, r.detail
    assert "1 URL(s)" in r.detail
    assert "unverified" not in r.detail


def test_bodywide_fenced_and_blockquoted_urls_not_probed(tmp_path, monkeypatch):
    """AC4 (#959) — a dead same-repo URL inside a ``` fence AND on a `>`
    blockquote line is never gathered → vacuous PASS, no FAIL."""
    repo, sha = _make_repo_with_issue1072_tree(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    dead = _gh_blob_url(sha, "eval_results/issue_1072/missing.npz")
    body = f"```\ncurl {dead}\n```\n\n> Originating prompt cites {dead} verbatim.\n"
    assert verify_task_body._gather_body_artifact_urls(body) == []
    r = verify_task_body.check_body_artifact_urls_exist(body)
    assert r.passed and not r.is_warn
    assert "no non-footer" in r.detail


def test_bodywide_grandfathered_v3_warns_not_fails(tmp_path, monkeypatch):
    """AC5 (forward-only) — the same dead URL under a v3 sentinel (and a v2
    variant) yields passed=True + is_warn=True with a grandfathered detail;
    a fully-passing v3 integration body keeps verify_text ok=True."""
    repo, sha = _make_repo_with_issue1072_tree(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    dead = _gh_blob_url(sha, "eval_results/issue_1072/missing_fold9.npz")
    para = f"Full arrays: [`missing_fold9.npz`]({dead})."
    for sentinel in ("<!-- clean-result-v3 -->", "<!-- clean-result-v2 -->"):
        r = verify_task_body.check_body_artifact_urls_exist(f"{sentinel}\n\n{para}\n")
        assert r.passed and r.is_warn, r.render()
        assert "grandfathered" in r.detail and "missing_fold9.npz" in r.detail
    # Integration: the v3 good body + the dead URL stays overall ok=True.
    body = _V3_GOOD_BODY.replace(
        "The 17-pt lift holds at every seed; "
        "the smallest within-condition gap between seeds is 1.2 pts.",
        para,
    )
    assert body != _V3_GOOD_BODY
    ok, results = verify_task_body.verify_text(body)
    rr = _results_by_name(results)[_BODYWIDE_NAME]
    assert rr.passed and rr.is_warn
    assert ok, [r.render() for r in results if not r.passed]


def test_bodywide_skips_footer_urls_owned_by_8b(tmp_path, monkeypatch):
    """AC6 — a dead URL that appears ONLY in the v4 footer is reported by
    check 8b and set-difference-skipped by check 42 (no double-report)."""
    repo, sha = _make_repo_with_issue1072_tree(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _V4_GOOD_BODY.replace(
        f"{_GH_REPO}/blob/0123456789abcdef/scripts/run.py",
        _gh_blob_url(sha, "eval_results/issue_1072/missing.npz"),
    )
    assert body != _V4_GOOD_BODY
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Reproducibility artifact URLs exist"].passed  # 8b owns it
    r42 = by_name[_BODYWIDE_NAME]
    assert r42.passed and "no non-footer" in r42.detail  # 42 skips it


def test_bodywide_unknown_sha_head_fallback(monkeypatch):
    """AC7/AC8 — sha unknown to the local object DB: an HTTP-HEAD 404 FAILs
    (v4); a None probe (offline) is an `unverified` PASS-note, never a
    FAIL. Never relies on live HTTP — `_http_head_status` is stubbed (the
    conftest EPM_VERIFY_BODY_NO_HTTP=1 fence covers un-stubbed paths)."""
    dead = _gh_blob_url("0123456789abcdef", "eval_results/issue_1072/missing.npz")
    body = f"<!-- clean-result-v4 -->\n\nFull arrays: [`missing.npz`]({dead}).\n"
    monkeypatch.setattr(verify_task_body, "_http_head_status", lambda url, timeout=5.0: 404)
    r = verify_task_body.check_body_artifact_urls_exist(body)
    assert not r.passed
    assert "404" in r.detail
    monkeypatch.setattr(verify_task_body, "_http_head_status", lambda url, timeout=5.0: None)
    r2 = verify_task_body.check_body_artifact_urls_exist(body)
    assert r2.passed and not r2.is_warn
    assert "unverified" in r2.detail


def test_bodywide_head_cap_bounds_probes(monkeypatch):
    """AC7 — 10 unknown-sha URLs against a counting `_http_head_status` stub:
    at most _BODYWIDE_HEAD_CAP (8) HEADs issue; past-cap URLs surface as
    `per-body HEAD cap` unverified notes on a PASS line."""
    calls: list[str] = []

    def counting_head(url, timeout=5.0):
        calls.append(url)
        return None

    monkeypatch.setattr(verify_task_body, "_http_head_status", counting_head)
    lines = [
        f"See [`f{i}.json`]({_gh_blob_url('0123456789abcdef', f'eval_results/issue_1072/f{i}.json')})."
        for i in range(10)
    ]
    body = "<!-- clean-result-v4 -->\n\n" + "\n".join(lines) + "\n"
    r = verify_task_body.check_body_artifact_urls_exist(body)
    assert len(calls) == verify_task_body._BODYWIDE_HEAD_CAP == 8
    assert r.passed
    assert "per-body HEAD cap" in r.detail


def test_bodywide_other_repo_and_trailing_quote():
    """Gatherer scope — other-repo github URLs are never gathered (existence
    undecidable locally); a legacy `href="…"` fragment's trailing quote is
    stripped so the probe sees the clean URL (D8)."""
    other = "https://github.com/otherowner/otherrepo/blob/0123456789abcdef/x.json"
    ours = f"{_GH_REPO}/blob/0123456789abcdef/scripts/run.py"
    body = f'See {other} and <a href="{ours}">entry script</a>.\n'
    assert verify_task_body._gather_body_artifact_urls(body) == [ours]


def test_gh_tree_adjacent_claim_missing_warns_1072_shape(tmp_path, monkeypatch):
    """AC2 — the VERBATIM #1072 pre-fix Artifacts line (sha swapped to the
    fixture sha) against a tree carrying the JSONs but not the `.npz` →
    check-43 WARN naming the expanded missing basenames + prefix + sha[:8] +
    the PAREN shape tag; the present JSONs are never reported."""
    repo, sha = _make_repo_with_issue1072_tree(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    line = _I1072_PREFIX_ARTIFACTS_LINE.format(
        tree_url=_gh_tree_url(sha, "eval_results/issue_1072")
    )
    r = verify_task_body.check_github_tree_adjacent_file_claims(line)
    assert r.passed and r.is_warn
    assert r.render().startswith("  [WARN]")
    for i in range(5):
        assert f"per_context_stats_1072_fold{i}.npz" in r.detail
    assert "eval_results/issue_1072" in r.detail and sha[:8] in r.detail
    assert "shape: PAREN" in r.detail
    # Present files are never reported missing.
    assert "claims `stats_component.json`" not in r.detail
    assert "claims `battery_1072_fold0.json`" not in r.detail


def test_gh_tree_adjacent_claim_present_passes(tmp_path, monkeypatch):
    """AC3 — every claimed basename (incl. the brace expansions) is a tree
    member → clean PASS, no WARN, no `unverified` note."""
    repo, sha = _make_repo_with_issue1072_tree(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    line = (
        f"Artifacts: [`eval_results/issue_1072/`]({_gh_tree_url(sha, 'eval_results/issue_1072')}) "
        "(`stats_component.json`, `battery_1072_fold{0..4}.json`)"
    )
    r = verify_task_body.check_github_tree_adjacent_file_claims(line)
    assert r.passed and not r.is_warn, r.detail
    assert "unverified" not in r.detail
    assert "6 adjacent file claim(s) against 1 pinned git tree(s)" in r.detail


def test_gh_tree_blob_and_hf_links_out_of_scope():
    """Partition — claims adjacent to same-repo `/blob/` links (8b/42 probe
    the URL itself), HF `/tree/` links (check 32's territory), and
    other-repo github links extract ZERO check-43 claims."""
    blob_path = "eval_results/issue_1072/per_context_stats_1072_fold4.npz"
    bodies = [
        f"Full arrays: [`per_context_stats_1072_fold4.npz`]({_gh_blob_url('0123456789abcdef', blob_path)})",
        "Mirror: [`stats_component.json`](https://huggingface.co/datasets/o/r/tree/abc1234def/dir/)",
        "Other: [`x.json`](https://github.com/otherowner/otherrepo/tree/0123456789abcdef/dir) (`y.json`)",
    ]
    for body in bodies:
        assert verify_task_body._gather_gh_tree_adjacent_file_claims(body) == [], body


def test_gh_tree_unknown_sha_unverified(tmp_path, monkeypatch):
    """AC8 — an unresolvable sha (unknown / shallow) is an `unverified`
    PASS-note, never a WARN (a partial/absent listing must never ground a
    WARN); zero network involved."""
    repo, _sha = _make_repo_with_issue1072_tree(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    line = (
        f"Artifacts: [`eval_results/issue_1072/`]"
        f"({_gh_tree_url('beefcafe1234', 'eval_results/issue_1072')}) "
        "(`stats_component.json`)"
    )
    r = verify_task_body.check_github_tree_adjacent_file_claims(line)
    assert r.passed and not r.is_warn, r.detail
    assert "unverified" in r.detail
    assert "did not resolve" in r.detail


def test_gh_tree_warn_never_fails(tmp_path, monkeypatch):
    """AC5 — check 43 has NO passed=False code path: the missing-claim case
    keeps passed=True (is_warn=True), and a grandfathered v2 integration
    body carrying BOTH new shapes (dead non-footer URL + missing tree claim)
    keeps verify_text ok=True (42 WARNs on v2; 43 WARNs by design)."""
    repo, sha = _make_repo_with_issue1072_tree(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    missing_line = _I1072_PREFIX_ARTIFACTS_LINE.format(
        tree_url=_gh_tree_url(sha, "eval_results/issue_1072")
    )
    r = verify_task_body.check_github_tree_adjacent_file_claims(missing_line)
    assert r.passed is True and r.is_warn is True
    dead = _gh_blob_url(sha, "eval_results/issue_1072/missing.npz")
    body = _V2_GOOD_BODY.replace(
        "0.81 — no regression at 25% mixing.",
        f"0.81 — no regression at 25% mixing.\n\n{missing_line} "
        f"Dead sample link: [`missing.npz`]({dead}).",
    )
    assert body != _V2_GOOD_BODY
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name[_BODYWIDE_NAME].passed and by_name[_BODYWIDE_NAME].is_warn
    assert by_name[_GH_TREE_NAME].passed and by_name[_GH_TREE_NAME].is_warn
    assert ok, [r.render() for r in results if not r.passed]


def test_expand_claim_token_bounds():
    """D5 — bounded numeric brace expansion: `{0..4}` → 5 names; an
    over-wide or inverted range expands to [] (no probe, no WARN); comma
    alternation and nested braces stay plain tokens the filename whitelist
    then rejects."""
    assert verify_task_body._expand_claim_token("battery_1072_fold{0..4}.json") == [
        f"battery_1072_fold{i}.json" for i in range(5)
    ]
    assert verify_task_body._expand_claim_token("x{0..4000}.npz") == []
    assert verify_task_body._expand_claim_token("x{4..0}.npz") == []
    for token in ("x{a,b}.npz", "x{0..{1..2}}.npz"):
        assert verify_task_body._expand_claim_token(token) == [token]
        assert not verify_task_body._HF_ADJ_FILENAME_RE.match(token)
    assert verify_task_body._expand_claim_token("stats_component.json") == ["stats_component.json"]


def test_gh_tree_disclaimer_suppresses_warn(tmp_path, monkeypatch):
    """AC10 (D10) — the VERBATIM CURRENT #1072 footer line: the `{0..4}.npz`
    tokens sit inside the qualifying paren as an explicit "gitignored … HF
    eval_results mirror only" DISCLAIMER → zero claims gathered, clean PASS
    against a tree lacking the `.npz`. P2: deleting the disclaimer substring
    from the SAME line re-yields gathered claims + a WARN (pins D10 as live
    code, not dead code)."""
    repo, sha = _make_repo_with_issue1072_tree(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    line = _I1072_CURRENT_ARTIFACTS_LINE.format(
        tree_url=_gh_tree_url(sha, "eval_results/issue_1072")
    )
    assert verify_task_body._gather_gh_tree_adjacent_file_claims(line) == []
    r = verify_task_body.check_github_tree_adjacent_file_claims(line)
    assert r.passed and not r.is_warn, r.detail
    undisclaimed = line.replace(" are gitignored and live on the HF eval_results mirror only", "")
    assert undisclaimed != line
    claims = verify_task_body._gather_gh_tree_adjacent_file_claims(undisclaimed)
    assert any(c[2] == "per_context_stats_1072_fold0.npz" for c in claims)
    r2 = verify_task_body.check_github_tree_adjacent_file_claims(undisclaimed)
    assert r2.passed and r2.is_warn
    assert "per_context_stats_1072_fold0.npz" in r2.detail


def test_gh_tree_ancestor_entries_not_members(tmp_path):
    """Fact-check A8 (P8) — `git ls-tree -r -t -- <prefix>` also emits the
    prefix's ANCESTOR tree entries (`eval_results`) and the prefix itself
    (`issue_1072`); the UNDER-prefix filter (`p.startswith(prefix + "/")`)
    excludes both, so an ancestor/prefix dir basename never counts as a
    member (a bare `p != prefix` would keep the ancestors as spurious
    WARN-suppressing members)."""
    repo, sha = _make_repo_with_issue1072_tree(tmp_path)
    status, basenames, note = verify_task_body._git_tree_basenames(
        repo, sha, "eval_results/issue_1072"
    )
    assert status == "ok", note
    assert "eval_results" not in basenames
    assert "issue_1072" not in basenames
    assert "stats_component.json" in basenames
    assert "battery_1072_fold4.json" in basenames


# ─── Check 45: caption count claims vs sidecar point values (#1511) ─────────

_CHECK45_NAME = "figure caption count claims vs sidecar point values (count drift)"

_C45_CAPTION_BASE = (
    "> **Figure.** *Tulu-25 lifts alignment ~17 pts over baseline at every seed.* "
    "Baseline gray, tulu-25 blue; error bars 95% Wald CIs."
)


def _check45_body(caption: str) -> str:
    """`_V4_GOOD_BODY` with its figure's blockquote caption replaced by
    ``caption`` (may span multiple `> `-prefixed lines, e.g. to place the
    per-figure opt-out comment inside the caption)."""
    return _V4_GOOD_BODY.replace(_C45_CAPTION_BASE, caption)


def _count_scatter_sidecar(*ys, labels=None):
    """#1426-shaped scatter sidecar: per point a POSITIVE x column
    ("median CoT length (tokens)"), the y column under test, a label,
    `_kind`, `_group`. The all-positive x pool exercises the any-pool rule
    for real (it can never rescue a below-zero claim but is always a
    same-size candidate — and it DOES rescue "no ... below zero" claims)."""
    labels = labels if labels is not None else [f"ctx{i}" for i in range(len(ys))]
    return {
        "created": "2026-07-18T00:00:00Z",
        "points": [
            {
                "median CoT length (tokens)": 400.0 + 10.0 * i,
                "per-context delta skill @ L24": y,
                "label": lab,
                "_kind": "scatter",
                "_group": 0,
            }
            for i, (y, lab) in enumerate(zip(ys, labels, strict=True))
        ],
        "n_series": 1,
        "total_points": len(ys),
    }


def test_check45_all_n_below_zero_contradicted_warns(tmp_path, monkeypatch):
    """Durability pin (#1511): "All 3 contexts lie below zero" beside a
    sidecar whose y pool holds a +0.004 point → WARN naming the offending
    label + value; WARN never FAIL (passed=True, is_warn=True)."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path, _count_scatter_sidecar(-0.1, -0.2, 0.004, labels=["a", "b", "pos_ctx"])
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check45_body("> **Figure.** *All 3 contexts lie below zero.*").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_caption_count_claims_vs_sidecar(body)
    assert res.passed is True and res.is_warn is True, res.render()
    assert "pos_ctx" in res.detail and "+0.004" in res.detail
    assert "count-claims: manual" in res.detail  # the WARN names the opt-out


def test_check45_all_n_below_zero_correct_passes(tmp_path, monkeypatch):
    """All-negative y pool satisfies "all 3 ... below zero" → clean PASS
    (the all-positive x pool contradicts, but ANY satisfying pool passes)."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, _count_scatter_sidecar(-0.1, -0.2, -0.3))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check45_body("> **Figure.** *All 3 contexts lie below zero.*").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_caption_count_claims_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()
    assert "recompute consistently" in res.detail


def test_check45_k_of_n_matching_passes(tmp_path, monkeypatch):
    """ "2 of 3 contexts lie below zero" vs y = (-0.1, -0.2, +0.004) →
    exact-count match → PASS."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, _count_scatter_sidecar(-0.1, -0.2, 0.004))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check45_body("> **Figure.** *2 of 3 contexts lie below zero.*").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_caption_count_claims_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()


def test_check45_k_of_n_contradicted_warns(tmp_path, monkeypatch):
    """ "1 of 3 contexts lie below zero" vs a recomputed 2-of-3 → WARN
    reporting claimed-vs-recomputed counts (exact equality, no ±1 slack)."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, _count_scatter_sidecar(-0.1, -0.2, 0.004))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check45_body("> **Figure.** *1 of 3 contexts lie below zero.*").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_caption_count_claims_vs_sidecar(body)
    assert res.passed and res.is_warn, res.render()
    assert "claimed 1 of 3" in res.detail
    assert "recomputes" in res.detail


def test_check45_none_above_zero_contradicted_warns(tmp_path, monkeypatch):
    """ "none of the 3 contexts lie above zero" vs a +0.004 point → WARN
    naming the point that DOES sit above zero."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path, _count_scatter_sidecar(0.004, -0.1, -0.2, labels=["pos_ctx", "b", "c"])
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check45_body("> **Figure.** *None of the 3 contexts lie above zero.*").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_caption_count_claims_vs_sidecar(body)
    assert res.passed and res.is_warn, res.render()
    assert "pos_ctx" in res.detail


def test_check45_no_unit_above_zero_both_directions(tmp_path, monkeypatch):
    """The bare `no <unit>` branch (n=None → ALL pools are candidates),
    both directions: contradicted → WARN; correct → PASS via the y pool
    even though the all-positive x pool contradicts (any-pool rescue)."""
    bad_repo, bad_sha = _make_repo_with_figure_meta(
        tmp_path, _count_scatter_sidecar(0.004, -0.1, -0.2)
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: bad_repo)
    body = _check45_body("> **Figure.** *No contexts sit above zero.*").replace(
        "0123456789abcdef", bad_sha
    )
    res = verify_task_body.check_figure_caption_count_claims_vs_sidecar(body)
    assert res.passed and res.is_warn, res.render()
    (tmp_path / "ok").mkdir()
    ok_repo, ok_sha = _make_repo_with_figure_meta(
        tmp_path / "ok", _count_scatter_sidecar(-0.3, -0.1, -0.2)
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: ok_repo)
    body2 = _check45_body("> **Figure.** *No contexts sit above zero.*").replace(
        "0123456789abcdef", ok_sha
    )
    res2 = verify_task_body.check_figure_caption_count_claims_vs_sidecar(body2)
    assert res2.passed and not res2.is_warn, res2.render()


def test_check45_copula_negative_end_to_end(tmp_path, monkeypatch):
    """The copula branch end-to-end: "all 3 deltas are negative" beside a
    +0.004 point → WARN (negative ≡ below zero); all-negative → PASS. An
    inverted neg/pos direction mapping would flip both outcomes."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path, _count_scatter_sidecar(-0.1, -0.2, 0.004, labels=["a", "b", "pos_ctx"])
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check45_body("> **Figure.** *All 3 deltas are negative.*").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_caption_count_claims_vs_sidecar(body)
    assert res.passed and res.is_warn, res.render()
    assert "pos_ctx" in res.detail
    (tmp_path / "ok").mkdir()
    ok_repo, ok_sha = _make_repo_with_figure_meta(
        tmp_path / "ok", _count_scatter_sidecar(-0.1, -0.2, -0.3)
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: ok_repo)
    body2 = _check45_body("> **Figure.** *All 3 deltas are negative.*").replace(
        "0123456789abcdef", ok_sha
    )
    res2 = verify_task_body.check_figure_caption_count_claims_vs_sidecar(body2)
    assert res2.passed and not res2.is_warn, res2.render()


def test_check45_no_size_matching_pool_skips(tmp_path, monkeypatch):
    """ "all 5 contexts ..." vs a 3-point sidecar → the referenced set is
    unidentifiable → silent PASS (never guess a pool)."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, _count_scatter_sidecar(-0.1, -0.2, 0.004))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check45_body("> **Figure.** *All 5 contexts lie below zero.*").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_caption_count_claims_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()
    assert "no figure caption with a registered count claim" in res.detail


def test_check45_aggregate_only_sidecar_skips(tmp_path, monkeypatch):
    """A sidecar with no `points`/`rows` (aggregates/text only) → silent
    PASS (no value pool to recount against)."""
    repo, sha = _make_repo_with_figure_meta(
        tmp_path,
        {"created": "2026-07-18T00:00:00Z", "text": {"suptitle": "s", "axes": []}},
    )
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check45_body("> **Figure.** *All 3 contexts lie below zero.*").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_caption_count_claims_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()


def test_check45_missing_sidecar_skips(tmp_path, monkeypatch):
    """A figure with NO `.meta.json` sibling → silent-skip PASS (check-24
    convention, NOT check 26's loud missing-sidecar FAIL)."""
    repo, sha = _make_repo_with_figure(tmp_path)  # hero.png, no sidecar
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check45_body("> **Figure.** *All 3 contexts lie below zero.*").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_caption_count_claims_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()


def test_check45_truncated_sidecar_skips(tmp_path, monkeypatch):
    """A `data_truncated` sidecar → silent PASS (a count over truncated
    rows is not figure truth)."""
    sidecar = _count_scatter_sidecar(-0.1, -0.2, 0.004)
    sidecar["data_truncated"] = True
    repo, sha = _make_repo_with_figure_meta(tmp_path, sidecar)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check45_body("> **Figure.** *All 3 contexts lie below zero.*").replace(
        "0123456789abcdef", sha
    )
    res = verify_task_body.check_figure_caption_count_claims_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()


def test_check45_all_but_and_bound_qualifiers_skip(tmp_path, monkeypatch):
    """ "all but 1 of 3" (inverted semantics) and "at least 2 of 3" (bound,
    not exact) parse NO claim — unit-level + body-level no-fire."""
    assert verify_task_body._caption_count_claims("all but 1 of 3 contexts lie below zero") == []
    assert verify_task_body._caption_count_claims("at least 2 of 3 contexts lie below zero") == []
    repo, sha = _make_repo_with_figure_meta(tmp_path, _count_scatter_sidecar(-0.1, -0.2, 0.004))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check45_body(
        "> **Figure.** *All but 1 of 3 contexts lie below zero; at least 2 of 3 "
        "contexts lie below zero.*"
    ).replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_caption_count_claims_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()
    assert "no figure caption with a registered count claim" in res.detail


def test_check45_not_all_pre_guard_no_claim():
    """ "not all 50 contexts lie below zero" parses NO claim (the `\\bnot`
    preceding-qualifier guard — reading it as all-50 would invert the
    caption's meaning)."""
    assert verify_task_body._caption_count_claims("not all 50 contexts lie below zero") == []
    # "cannot" must NOT trip the guard (no word boundary before "not"):
    claims = verify_task_body._caption_count_claims(
        "we cannot ignore that all 3 contexts lie below zero"
    )
    assert [c["shape"] for c in claims] == ["all"]


def test_check45_at_or_below_zero_no_claim():
    """ "all 50 contexts lie at or below zero" is a NON-strict (<=) claim —
    the strict recompute must not read it (an exact-zero point would
    false-WARN); the `at or` gap guard rejects it."""
    assert verify_task_body._caption_count_claims("all 50 contexts lie at or below zero") == []
    assert (
        verify_task_body._caption_count_claims("none of the 3 contexts sit at or above zero") == []
    )


def test_check45_numeral_zero_forms():
    """ "below 0" (bare numeral, incl. sentence-final "0.") parses a claim;
    "below 0.05" (a decimal — a non-zero referent) does not."""
    got = verify_task_body._caption_count_claims("all 3 contexts lie below 0.")
    assert [(c["shape"], c["direction"]) for c in got] == [("all", "below")]
    assert verify_task_body._caption_count_claims("all 3 contexts lie below 0.05") == []


def test_check45_no_qualifier_words_skip():
    """ "no further points ..." — the captured unit IS the qualifier token
    (a claim about ADDITIONAL items, not a counted set) → no claim."""
    assert verify_task_body._caption_count_claims("no further points dip below zero") == []
    assert verify_task_body._caption_count_claims("no other contexts fall below zero") == []


def test_check45_nonzero_referent_not_matched():
    """The real #1426 second-figure caption — "44 of 50 contexts fall more
    than 0.05 below the linear fit" — parses NO claim (decimal breaks the
    punctuation-free gap; "below the" is not "below zero")."""
    assert (
        verify_task_body._caption_count_claims(
            "44 of 50 contexts fall more than 0.05 below the linear fit"
        )
        == []
    )


def test_check45_optout_in_beat1_suppresses(tmp_path, monkeypatch):
    """The (contradicted) claim + `<!-- count-claims: manual -->` in the
    figure's beat-1 prose → suppressed, PASS with "opted out" in detail."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, _count_scatter_sidecar(-0.1, -0.2, 0.004))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = (
        _check45_body("> **Figure.** *All 3 contexts lie below zero.*")
        .replace(
            _CHECK33_PLOTTED_BASE,
            _CHECK33_PLOTTED_BASE + "\n\n<!-- count-claims: manual -->",
        )
        .replace("0123456789abcdef", sha)
    )
    res = verify_task_body.check_figure_caption_count_claims_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()
    assert "opted out" in res.detail


def test_check45_optout_in_caption_suppresses(tmp_path, monkeypatch):
    """The opt-out literal on a `> `-prefixed CAPTION line is honored too
    (`_figure_caption_after` joins blockquote lines, so the literal lands
    in the caption text the opt-out window includes)."""
    repo, sha = _make_repo_with_figure_meta(tmp_path, _count_scatter_sidecar(-0.1, -0.2, 0.004))
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _check45_body(
        "> **Figure.** *All 3 contexts lie below zero.*\n> <!-- count-claims: manual -->"
    ).replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_caption_count_claims_vs_sidecar(body)
    assert res.passed and not res.is_warn, res.render()
    assert "opted out" in res.detail


def test_check45_grandfathered_v3_body_noop(tmp_path, monkeypatch):
    """The v3 exemplar body (no count claim, unresolvable placeholder sha)
    → vacuous PASS, never flagged (generation-agnostic, forward-safe)."""
    repo, _sha = _make_repo_with_figure(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    res = verify_task_body.check_figure_caption_count_claims_vs_sidecar(_V3_GOOD_BODY)
    assert res.passed and not res.is_warn, res.render()


def test_check45_registered():
    """The check rides `CHECKS` (generation-agnostic block)."""
    assert verify_task_body.check_figure_caption_count_claims_vs_sidecar in verify_task_body.CHECKS


def test_check45_bar_x_dodge_excluded_from_pools():
    """`_sidecar_value_pools` on issue_1005-shaped bar sidecars: a numeric
    first key (`"x": -0.19` dodge offset) forms NO pool (a negative layout
    slot must never count as a below-zero point); the heights column does;
    a string `category` first key consumes the first slot so heights stay
    untagged (mirrors `_sidecar_plotted_values` first-slot consumption)."""
    pools = verify_task_body._sidecar_value_pools(
        {
            "points": [
                {"x": -0.19, "compliance": 0.5, "_kind": "bar"},
                {"x": 0.81, "compliance": 0.7, "_kind": "bar"},
            ]
        }
    )
    assert (None, "x") not in pools
    assert [v for v, _l in pools[(None, "compliance")]] == [0.5, 0.7]
    pools2 = verify_task_body._sidecar_value_pools(
        {
            "points": [
                {"category": "fam1", "height": 0.5, "_kind": "bar"},
                {"category": "fam2", "height": -0.2, "_kind": "bar"},
            ]
        }
    )
    assert [v for v, _l in pools2[(None, "height")]] == [0.5, -0.2]


def test_check45_mixed_kind_pools_get_per_kind_grain():
    """Mixed bar+line rows SHARING a column name (the issue_1005 mediation
    shape) get per-`(kind, column)` pools in ADDITION to the merged pool —
    the per-kind grain is what restores an N-sized candidate there."""
    pools = verify_task_body._sidecar_value_pools(
        {
            "points": [
                {"cat": "a", "held-out skill (LOCO)": 0.4, "_kind": "bar"},
                {"cat": "b", "held-out skill (LOCO)": 0.5, "_kind": "bar"},
                {"step": 1.0, "held-out skill (LOCO)": 0.6, "_kind": "line"},
            ]
        }
    )
    assert len(pools[(None, "held-out skill (LOCO)")]) == 3
    assert len(pools[("bar", "held-out skill (LOCO)")]) == 2
    assert len(pools[("line", "held-out skill (LOCO)")]) == 1


def test_check45_incident_1426_shape(tmp_path, monkeypatch):
    """The motivating incident, two-series real shape (sidecar
    `figures/issue_1426/mlc_percontext_delta_scatter.meta.json` @
    `4a65c36ab0`): 100 points = two 50-point scatter series (series 2
    duplicates the y values under a bare `y` column), 49 negative + one
    +0.004368588982575972 labeled `f1_house_medical_doctor`. Pre-fix
    caption ("all 50 contexts lie below zero") → WARN naming the point;
    corrected caption ("49 of 50 ... (one persona context marginally
    positive at +0.004)") → PASS — and the parenthetical parses NO claim
    (no copula), and the duplicate `y` pool never double-reports."""
    ys = [-(i + 1) / 100.0 for i in range(49)] + [0.004368588982575972]
    labels = [f"ctx{i}" for i in range(49)] + ["f1_house_medical_doctor"]
    points = []
    for i, (y, lab) in enumerate(zip(ys, labels, strict=True)):
        points.append(
            {
                "median CoT length (tokens)": 400.0 + i,
                "per-context delta skill (read 1) @ L24": y,
                "label": lab,
                "_kind": "scatter",
                "_group": 0,
            }
        )
    for i, (y, lab) in enumerate(zip(ys, labels, strict=True)):
        points.append(
            {
                "median K (tokens)": 300.0 + i,
                "y": y,
                "label": lab,
                "_kind": "scatter",
                "_group": 1,
            }
        )
    sidecar = {"created": "2026-07-18T00:00:00Z", "points": points, "total_points": 100}
    repo, sha = _make_repo_with_figure_meta(tmp_path, sidecar)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    prefix_body = _check45_body(
        "> **Figure.** *Per-context demotion (full-context, read 1) at layer 24; "
        "all 50 contexts lie below zero.*"
    ).replace("0123456789abcdef", sha)
    res = verify_task_body.check_figure_caption_count_claims_vs_sidecar(prefix_body)
    assert res.passed and res.is_warn, res.render()
    assert "f1_house_medical_doctor" in res.detail
    assert "+0.004369" in res.detail
    assert res.detail.count("caption claims") == 1  # duplicate y pool: no double report
    fixed_body = _check45_body(
        "> **Figure.** *Per-context demotion (full-context, read 1) at layer 24; "
        "49 of 50 contexts lie below zero (one persona context marginally positive "
        "at +0.004).*"
    ).replace("0123456789abcdef", sha)
    res2 = verify_task_body.check_figure_caption_count_claims_vs_sidecar(fixed_body)
    assert res2.passed and not res2.is_warn, res2.render()
    assert "1 caption count claim(s)" in res2.detail  # the parenthetical parses no claim


def test_check45_subset_claim_larger_pool_rescue():
    """The #1090 corpus-calibration shape: "all 2 probes are negative"
    counts a named SUBSET of a larger plotted column. The only size-2 pool
    (a different measure, all positive) contradicts, but a size-5 column
    with >=2 negative values plausibly hosts the subset → rescued (no
    WARN). With the larger column all-positive too, no rescue → WARN."""
    pools = {
        (None, "judged delta"): [(0.1, "a"), (0.2, "b")],
        (None, "projection"): [(-1.0, None), (-2.0, None), (-3.0, None), (0.5, None), (0.6, None)],
    }
    claims = verify_task_body._caption_count_claims("all 2 probes are negative")
    warns, n_checked, sat = verify_task_body._count_claim_failures(claims, pools, "f.png")
    assert n_checked == 1 and warns == [], (warns, sat)
    assert "subset of n=5" in sat[0]
    pools_bad = {
        (None, "judged delta"): [(0.1, "a"), (0.2, "b")],
        (None, "projection"): [(1.0, None), (2.0, None), (3.0, None), (0.5, None), (0.6, None)],
    }
    warns2, n2, _sat2 = verify_task_body._count_claim_failures(claims, pools_bad, "f.png")
    assert n2 == 1 and len(warns2) == 1, warns2
    # The incident class is unaffected: pools all EXACTLY size n → no host.
    pools_incident = {(None, "y"): [(-0.1, "a"), (-0.2, "b"), (0.004, "pos_ctx")]}
    claims3 = verify_task_body._caption_count_claims("all 3 contexts lie below zero")
    warns3, n3, _s3 = verify_task_body._count_claim_failures(claims3, pools_incident, "f.png")
    assert n3 == 1 and len(warns3) == 1 and "pos_ctx" in warns3[0]


# ─── Check 46: brace-expanded backtick HF paths vs the adjacent /tree pin ───
#
# (#1520; incident #1426.) Conventions: `_stub_tree` / inline `_hf_tree_get`
# stubs re-patch over the autouse `_no_unexpected_probes` raise-guard (an
# unstubbed probe is a hard error, which doubles as the zero-probe assert in
# the gather-only / fenced tests); the conftest-level EPM_VERIFY_BODY_NO_HF
# fence is delenv'd only in tests that exercise the online verdict path.

# Verbatim **Repro:** line 204 of the PRE-FIX #1426 body (commit b2b5eb0b74) — the
# NEARBIND incident shape: `sampled_rollout/seed{42,137}/` 272 bracket-free chars
# after the c244377f-pinned prefix link.
_I1426_PREFIX_REPRO_LINE = (
    "**Repro:** primary — GCP 1× A100-80 (`eps-issue-1426`, us-central1), one provision; "
    "3 launches of attempt `att-20260717-234120` — launches 1–2 died on transient HF 429 "
    "rate limits during artifact staging (infra only; resume digest-validated per unit), "
    "launch 3 completed after commit `98d7218ee1` added transient-retry to the upload hel"
    "per. Gate: terminal pass (usable 0.986 on the non-collapse slice, offender rate 0.83"
    "%, p95 generation length 1,746 tokens vs the 8,192 cap). Driver at `4e53cedec8`; run"
    " artifacts committed at `3f67cf6e83`; the covariate phase (a 0-GPU VM-side re-reduct"
    "ion the pod driver does not run) was executed during upload verification and committ"
    "ed at `3fdf8222e5`; analyzer figures at `891e7851ff`, revised (reader-facing MLP-her"
    "o labels, tercile denominator legend) + the 45 driver-generated F1 figures mirrored "
    "from HF at `4a65c36ab0`; pooled tri-lineage gradient analysis artifact + figure at `"
    "4b3adbe612`, load_dotenv hot-fix at `62f983e36e` (branch `issue-1426`). Robustness r"
    "ound — GCP 2× A100-80 (`eps-issue-1426-sampled`, us-central1), ~55 min (~1.8 GPU-h);"
    " commits `e6dab7d18d` + `fb4e92445c` + `ce341a98ac` (driver + analysis) + `148a95a6c"
    "a` (figures) on branch `issue-1426-sampled`. Eval JSONs: `eval_results/issue_1426/` "
    "(primary + `cap16k/` + `indiv-mlp-nonlinearity-control/`) and `eval_results/issue_14"
    "26/sampled-rollout-robustness/seed{42,137}/`. HF data repo `superkaiba1/explore-pers"
    "ona-space-data`, prefix [`issue1426_cot_decomposition_r1llama/`](https://huggingface"
    ".co/datasets/superkaiba1/explore-persona-space-data/tree/c244377f2b5bc9e1ed8dd093b05"
    "035aa0c4940e9/issue1426_cot_decomposition_r1llama): `raw_completions/thinking_rollou"
    "ts/` + `thinking_rollouts_16k/`, `analysis_tensors/` (store manifest + summaries + d"
    "ecomp tensors + MLP control), `fit_results/` + `fit_results_16k/`, `figures/` (listi"
    "ngs verified via `list_repo_tree` at write time); sampled rollouts at `sampled_rollo"
    "ut/seed{42,137}/`. Reused inputs: the 50-context battery + 48-probe pool (sha-pinned"
    ", from [#928](https://github.com/superkaiba/explore-persona-space/issues/928)/[#1005"
    "](https://github.com/superkaiba/explore-persona-space/issues/1005)); both prior line"
    "ages' committed per-context delta artifacts and figure sidecars (`eval_results/issue"
    "_928/percontext_deltas.json`, `eval_results/issue_1005/percontext_deltas.json`, `fig"
    "ures/issue_928/percontext_scatter_avg_q.meta.json`, `figures/issue_928/percontext_sc"
    "atter_indiv.meta.json`, `figures/issue_1005/fam_contrast_length_matched.meta.json`) "
    "for the like-for-like baselines — fit: same contrast code, parity-gated against comm"
    "itted values."
)

# Verbatim **Repro:** line 204 of the FIXED #1426 body (commit 0ee2cb1744) — the
# LINKTEXT shape: the brace token inside the text of the 31d4fb5c-pinned
# `…/sampled_rollout` link.
_I1426_FIXED_REPRO_LINE = (
    "**Repro:** primary — GCP 1× A100-80 (`eps-issue-1426`, us-central1), one provision; "
    "3 launches of attempt `att-20260717-234120` — launches 1–2 died on transient HF 429 "
    "rate limits during artifact staging (infra only; resume digest-validated per unit), "
    "launch 3 completed after commit `98d7218ee1` added transient-retry to the upload hel"
    "per. Gate: terminal pass (usable 0.986 on the non-collapse slice, offender rate 0.83"
    "%, p95 generation length 1,746 tokens vs the 8,192 cap). Driver at `4e53cedec8`; run"
    " artifacts committed at `3f67cf6e83`; the covariate phase (a 0-GPU VM-side re-reduct"
    "ion the pod driver does not run) was executed during upload verification and committ"
    "ed at `3fdf8222e5`; analyzer figures at `891e7851ff`, revised (reader-facing MLP-her"
    "o labels, tercile denominator legend) + the 45 driver-generated F1 figures mirrored "
    "from HF at `4a65c36ab0`; pooled tri-lineage gradient analysis artifact + figure at `"
    "4b3adbe612`, load_dotenv hot-fix at `62f983e36e` (branch `issue-1426`). Robustness r"
    "ound — GCP 2× A100-80 (`eps-issue-1426-sampled`, us-central1), ~55 min (~1.8 GPU-h);"
    " commits `e6dab7d18d` + `fb4e92445c` + `ce341a98ac` (driver + analysis) + `148a95a6c"
    "a` (figures) on branch `issue-1426-sampled`. Eval JSONs: `eval_results/issue_1426/` "
    "(primary + `cap16k/` + `indiv-mlp-nonlinearity-control/`) and `eval_results/issue_14"
    "26/sampled-rollout-robustness/seed{42,137}/`. HF data repo `superkaiba1/explore-pers"
    "ona-space-data`, prefix [`issue1426_cot_decomposition_r1llama/`](https://huggingface"
    ".co/datasets/superkaiba1/explore-persona-space-data/tree/c244377f2b5bc9e1ed8dd093b05"
    "035aa0c4940e9/issue1426_cot_decomposition_r1llama): `raw_completions/thinking_rollou"
    "ts/` + `thinking_rollouts_16k/`, `analysis_tensors/` (store manifest + summaries + d"
    "ecomp tensors + MLP control), `fit_results/` + `fit_results_16k/`, `figures/` (listi"
    "ngs verified via `list_repo_tree` at write time); sampled rollouts at [`sampled_roll"
    "out/seed{42,137}/` (pinned)](https://huggingface.co/datasets/superkaiba1/explore-per"
    "sona-space-data/tree/31d4fb5cc07ef3fe34bcc252e0defa5b5e44a408/issue1426_cot_decompos"
    "ition_r1llama/sampled_rollout) — uploaded after the primary pin, hence the separate "
    "revision. Reused inputs: the 50-context battery + 48-probe pool (sha-pinned, from [#"
    "928](https://github.com/superkaiba/explore-persona-space/issues/928)/[#1005](https:/"
    "/github.com/superkaiba/explore-persona-space/issues/1005)); both prior lineages' com"
    "mitted per-context delta artifacts and figure sidecars (`eval_results/issue_928/perc"
    "ontext_deltas.json`, `eval_results/issue_1005/percontext_deltas.json`, `figures/issu"
    "e_928/percontext_scatter_avg_q.meta.json`, `figures/issue_928/percontext_scatter_ind"
    "iv.meta.json`, `figures/issue_1005/fam_contrast_length_matched.meta.json`) for the l"
    "ike-for-like baselines — fit: same contrast code, parity-gated against committed val"
    "ues."
)

_I1426_SHA_PRE = "c244377f2b5bc9e1ed8dd093b05035aa0c4940e9"
_I1426_SHA_FIX = "31d4fb5cc07ef3fe34bcc252e0defa5b5e44a408"
_I1426_HF_PREFIX = "issue1426_cot_decomposition_r1llama"
# A throwaway hex pin for the synthetic check-46 bodies.
_C46_SHA = "aaaa1111bbbb2222cccc3333dddd4444eeee5555"


def _c46_link(path, text="`x`", sha=_C46_SHA):
    """One hex-pinned HF dataset /tree markdown link for check-46 bodies."""
    return f"[{text}](https://huggingface.co/datasets/o/r/tree/{sha}/{path})"


def _stub_tree_by_url(monkeypatch, responses, calls=None):
    """`_hf_tree_get` stub dispatching on URL substring (first match wins) →
    (status, entries, next_page); raises on an unmatched URL (missed-mock
    detection, the `_no_unexpected_probes` convention). Child-parent URLs
    are `quote(path, safe="")`-encoded, so nested-path needles use `%2F`."""

    def _fake(url, params, headers, *, timeout_s):
        if calls is not None:
            calls.append((url, params))
        for needle, (status, entries, next_page) in responses:
            if needle in url:
                return verify_task_body._TreeProbeResult(status, list(entries), next_page, "")
        raise AssertionError(f"unexpected probe URL: {url}")

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _fake)


def test_hf_brace_claim_missing_warns_1426_prefix_shape(monkeypatch):
    """AC1: the VERBATIM pre-fix #1426 Repro line WARNs — pin prefix alive
    (200, 5 entries, no sampled_rollout), the expansions' parent 404s →
    definitive missing, both joined expansions named, NEARBIND shape;
    the alive no-brace tokens are never reported. (AC8: passed stays True.)"""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    calls = []
    entries = [
        {"path": f"{_I1426_HF_PREFIX}/{n}", "type": "directory"}
        for n in (
            "raw_completions",
            "analysis_tensors",
            "fit_results",
            "fit_results_16k",
            "figures",
        )
    ]
    _stub_tree_by_url(
        monkeypatch,
        [
            ("%2Fsampled_rollout", ("not_found", [], None)),
            (f"/tree/{_I1426_SHA_PRE}/{_I1426_HF_PREFIX}", ("ok", entries, None)),
        ],
        calls=calls,
    )
    res = verify_task_body.check_hf_brace_expanded_path_claims(_I1426_PREFIX_REPRO_LINE)
    assert res.passed is True and res.is_warn is True, res.render()
    assert res.render().startswith("  [WARN]")
    assert "sampled_rollout/seed{42,137}/" in res.detail
    assert f"`{_I1426_HF_PREFIX}/sampled_rollout/seed42`" in res.detail
    assert f"`{_I1426_HF_PREFIX}/sampled_rollout/seed137`" in res.detail
    assert "c244377f" in res.detail
    assert "shape: NEARBIND" in res.detail
    assert "2/2 expansions missing" in res.detail
    assert "thinking_rollouts" not in res.detail  # alive no-brace tokens never reported
    assert len(calls) == 2  # pin-alive precheck + one child-parent listing


def test_hf_brace_claim_present_passes_fixed_1426_linktext_shape(monkeypatch):
    """AC2: the VERBATIM fixed #1426 line PASSes clean — the LINKTEXT token
    overlap-joins onto the `…/sampled_rollout` prefix (single-segment
    basename overlap) and both seed dirs resolve; ONE probed parent (the
    pin precheck IS the parent listing), no WARN, no unverified."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    calls = []
    entries = [
        {"path": f"{_I1426_HF_PREFIX}/sampled_rollout/seed42", "type": "directory"},
        {"path": f"{_I1426_HF_PREFIX}/sampled_rollout/seed137", "type": "directory"},
    ]
    _stub_tree_by_url(monkeypatch, [("%2Fsampled_rollout", ("ok", entries, None))], calls=calls)
    res = verify_task_body.check_hf_brace_expanded_path_claims(_I1426_FIXED_REPRO_LINE)
    assert res.passed is True and res.is_warn is False, res.render()
    assert "unverified" not in res.detail
    assert len(calls) == 1


def test_hf_brace_partial_missing_warns_only_absent_expansion(monkeypatch):
    """AC3: an exhaustive listing carrying seed42 only → WARN names ONLY the
    absent seed137 expansion (singular 'does not resolve')."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    _stub_tree_by_url(
        monkeypatch,
        [(f"/tree/{_C46_SHA}/dir", ("ok", [{"path": "dir/seed42", "type": "directory"}], None))],
    )
    body = _c46_link("dir", text="`seed{42,137}/`")
    res = verify_task_body.check_hf_brace_expanded_path_claims(body)
    assert res.passed is True and res.is_warn is True, res.render()
    assert "`dir/seed137`" in res.detail
    assert "dir/seed42" not in res.detail
    assert "1/2 expansions missing" in res.detail
    assert "does not resolve" in res.detail
    assert "shape: LINKTEXT" in res.detail


def test_expand_hf_brace_token_bounds_and_charset():
    """AC4: comma alternation (2-32, no `/`) and numeric {N..M} (<=32 wide)
    expand; malformed / over-wide / multi-group / glob / plain tokens
    extract ZERO expansions."""
    expand = verify_task_body._expand_hf_brace_token
    assert expand("seed{42,137}") == ["seed42", "seed137"]
    assert expand("{raw_completions,analysis_tensors}/") == [
        "raw_completions/",
        "analysis_tensors/",
    ]
    assert expand("fold{0..4}") == ["fold0", "fold1", "fold2", "fold3", "fold4"]
    assert expand("x{0..4000}") == []  # over-wide range
    assert expand("x{4..0}") == []  # reversed range
    assert expand("{a}") == []  # single alternative — not a brace group
    assert expand("{a,b/c}") == []  # `/` inside an alternative
    assert expand("a{1,2}b{3,4}") == []  # multi-group
    assert expand("x*{a,b}") == []  # glob
    assert expand("plain/dir/") == []  # no brace — brace-REQUIRED check
    assert expand("../up{1,2}") == []  # leading ../ declined
    # mixed range-in-alternation shorthand (the #560 corpus shape) declined —
    # bash-literal alternatives would probe nonsense while the intended
    # per-range artifacts resolve (AC10 precision):
    assert expand("i474_loc_{A1..A5,B1..B5,C1}_ep1") == []
    # ellipsis pure-dot segment (the #617 corpus shape) declined — `...` is
    # prose, not a path:
    assert expand("picked_categories/{coding,travel}/...") == []


def test_join_brace_expansion_branches():
    """All four accepting `_join_brace_expansion` branches + the ambiguous
    multi-segment-overlap decline (returns None)."""
    join = verify_task_body._join_brace_expansion
    assert join("", "x1") == "x1"  # bare root pin
    assert join("a/b", "a/b/c") == "a/b/c"  # token repeats the prefix
    assert join("a/b", "a/b") == "a/b"  # token IS the prefix
    # single-segment basename overlap — the FIXED #1426 shape:
    assert join("p/sampled_rollout", "sampled_rollout/seed42/") == "p/sampled_rollout/seed42"
    assert join("p", "seed42/") == "p/seed42"  # plain descend
    assert join("a/b", "a/c1") is None  # doubled-path false-WARN class → decline


def test_hf_brace_git_side_root_excluded():
    """AC5 (G3): a git-side-root token (`eval_results/…`) in the binder gap
    extracts ZERO claims (and zero probes — no stub installed, the autouse
    raise-guard would fail any GET)."""
    body = (
        _c46_link("x", text="`x/`")
        + ": `eval_results/x/sampled-rollout-robustness/seed{42,137}/` committed in git."
    )
    assert verify_task_body._gather_hf_brace_path_claims(body) == []
    res = verify_task_body.check_hf_brace_expanded_path_claims(body)
    assert res.passed is True and res.is_warn is False
    assert "no brace-expanded path claims" in res.detail


def test_hf_brace_nearbind_gap_guards():
    """NEARBIND precision: gap >400 chars / newline in gap / intervening
    markdown link each decline; the small-gap control binds."""
    link = _c46_link("p", text="`p/`")
    tok = "`sub/seed{1,2}/`"
    gather = verify_task_body._gather_hf_brace_path_claims
    assert gather(link + " " + "x" * 401 + " " + tok) == []
    assert gather(link + "\n" + tok) == []
    assert gather(link + " [other](https://example.com/y) " + tok) == []
    claims = gather(link + " sampled at " + tok)
    assert len(claims) == 1 and claims[0][6] == "NEARBIND"


def test_hf_brace_token_own_at_rev_pin_declines():
    """AC5: a token-adjacent `@ rev <hex>` own-pin governs — the NEARBIND
    binding to the earlier link is declined (zero claims)."""
    body = _c46_link("p", text="`p/`") + " rollouts `sub/seed{1,2}/` @ rev `deadbeef12` later."
    assert verify_task_body._gather_hf_brace_path_claims(body) == []


def test_hf_brace_disclaimer_suppresses():
    """AC5: HF-absence disclaimer vocabulary in the forward window (NEARBIND)
    or the link text (LINKTEXT) suppresses the instance; the same line
    without the disclaimer fires."""
    link = _c46_link("p", text="`p/`")
    gather = verify_task_body._gather_hf_brace_path_claims
    assert gather(link + " plus `sub/seed{1,2}/` (not yet uploaded).") == []
    assert (
        gather(
            f"[`sub/seed{{1,2}}/` (upload pending)](https://huggingface.co/datasets/o/r/tree/{_C46_SHA}/p)"
        )
        == []
    )
    claims = gather(link + " plus `sub/seed{1,2}/` (uploaded 2026-07-18).")
    assert len(claims) == 1


def test_hf_brace_offline_fence_unverified_zero_probes():
    """AC6: with the conftest EPM_VERIFY_BODY_NO_HF fence left in place, a
    real claim degrades to an `unverified` fence note on a PASS line with
    ZERO GETs (the autouse raise-guard stays active — any probe raises)."""
    body = _c46_link("dir", text="`seed{1,2}/`")
    res = verify_task_body.check_hf_brace_expanded_path_claims(body)
    assert res.passed is True and res.is_warn is False, res.render()
    assert "unverified" in res.detail and "HF probe fenced" in res.detail


def test_hf_brace_pin_dead_defers_to_check23(monkeypatch):
    """AC7: pin-prefix listing → not_found → `unverified` note deferring to
    check 23 (dead-pin FAIL is check 23's), never a WARN from this check."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    _stub_tree(monkeypatch, status="not_found")
    body = _c46_link("dir", text="`seed{1,2}/`")
    res = verify_task_body.check_hf_brace_expanded_path_claims(body)
    assert res.passed is True and res.is_warn is False, res.render()
    assert "check 23" in res.detail


def test_hf_brace_probe_cap_unverified(monkeypatch):
    """AC6: 9 unique pinned parents → at most `_HF_BRACE_MAX_PROBES`=8 unique
    listings; past-cap claims surface as `per-body probe cap` notes."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)
    calls = []
    _stub_tree(monkeypatch, status="ok", entries=(), next_page=None, calls=calls)
    body = " ".join(_c46_link(f"d{i}", text="`seed{1,2}/`") for i in range(9))
    res = verify_task_body.check_hf_brace_expanded_path_claims(body)
    assert res.passed is True
    assert "per-body probe cap" in res.detail
    assert len(calls) == verify_task_body._HF_BRACE_MAX_PROBES == 8


def test_hf_brace_vacuous_pass_zero_probes():
    """AC6: pinned links but no brace tokens → vacuous PASS, `_hf_tree_get`
    never called (autouse raise-guard active, no stub)."""
    body = _c46_link("dir", text="`file.json`") + " and `plain/dir/` here."
    res = verify_task_body.check_hf_brace_expanded_path_claims(body)
    assert res.passed is True and res.is_warn is False
    assert "no brace-expanded path claims" in res.detail


def test_hf_brace_partial_listing_never_grounds_warn(monkeypatch):
    """AC6/AC8: a page-capped (partial) listing — present-in-partial passes,
    the unfound expansion degrades to `unverified` (never a WARN), and the
    partial result is never cached."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)

    def _fake(url, params, headers, *, timeout_s):
        return verify_task_body._TreeProbeResult(
            "ok", [{"path": "dir/seed1", "type": "directory"}], "https://next-page", ""
        )

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _fake)
    body = _c46_link("dir", text="`seed{1,2}/`")
    res = verify_task_body.check_hf_brace_expanded_path_claims(body)
    assert res.passed is True and res.is_warn is False, res.render()
    assert "partial listing — missing never grounded" in res.detail
    assert "`dir/seed2`" in res.detail and "dir/seed1" not in res.detail
    assert verify_task_body._HF_DIRECT_CHILDREN_CACHE == {}


def test_hf_brace_partial_pin_precheck_still_probes_children(monkeypatch):
    """A partial (page-capped) pin precheck still proves the pin ALIVE and
    proceeds to the child probes — only not_found/skip defer; with the
    child listing exhaustive and complete the claim PASSes clean."""
    monkeypatch.delenv("EPM_VERIFY_BODY_NO_HF", raising=False)

    def _fake(url, params, headers, *, timeout_s):
        if "%2Fsub" in url:
            entries = [
                {"path": "p/sub/seed1", "type": "directory"},
                {"path": "p/sub/seed2", "type": "directory"},
            ]
            return verify_task_body._TreeProbeResult("ok", entries, None, "")
        return verify_task_body._TreeProbeResult(
            "ok", [{"path": "p/other", "type": "directory"}], "https://next-page", ""
        )

    monkeypatch.setattr(verify_task_body, "_hf_tree_get", _fake)
    body = _c46_link("p") + " rollouts `sub/seed{1,2}/` here."
    res = verify_task_body.check_hf_brace_expanded_path_claims(body)
    assert res.passed is True and res.is_warn is False, res.render()
    assert "unverified" not in res.detail


def test_hf_brace_ambiguous_overlap_declines_to_unverified():
    """Critic concern 3: a token whose first segment equals a NON-basename
    prefix segment (prefix `a/b`, token `a/c{1,2}/`) declines the join →
    `unverified`, ZERO probes (decline precedes the pin precheck)."""
    body = _c46_link("a/b", text="`a/c{1,2}/`")
    res = verify_task_body.check_hf_brace_expanded_path_claims(body)
    assert res.passed is True and res.is_warn is False, res.render()
    assert "ambiguous prefix overlap" in res.detail


def test_hf_brace_dedup_same_token_same_pin():
    """Critic concern 5: the same token bound twice to the same pin dedups
    to ONE claim on (repo_id, sha, pin_prefix, token)."""
    link = _c46_link("p", text="`p/`")
    body = link + " `sub/seed{1,2}/` and again `sub/seed{1,2}/`."
    assert len(verify_task_body._gather_hf_brace_path_claims(body)) == 1


def test_check46_registered():
    """Critic concern 1: the check rides `CHECKS` (generation-agnostic
    block) — a forgotten CHECKS append must not ship green."""
    assert verify_task_body.check_hf_brace_expanded_path_claims in verify_task_body.CHECKS


# ─── Check 47 (#1521; incident #1426): Context follow-up provenance vs
#     followup-scope markers ──────────────────────────────────────────────


def _scope_event_47(label, *, source=None, est=None, ts="2026-07-18T14:19:00Z", version=1):
    """One `epm:followup-scope` event with line-initial fields (the canonical
    workflow.yaml note shape check 47's marker-side parse consumes)."""
    lines = [f"followup_label: {label}"]
    if source is not None:
        lines.append(f"source: {source}")
    if est is not None:
        lines.append(f"est_gpu_hours: {est}")
    return {"kind": "epm:followup-scope", "ts": ts, "version": version, "note": "\n".join(lines)}


# The real #1426 scope-marker note's field lines (spec truncated — the parse
# reads line-initial fields only), and the incident's pre-fix / corrected
# Context clauses (body.md fold r1 vs the corrected row, read 2026-07-18).
_I1426_SCOPE_NOTE = (
    "followup_label: sampled-rollout-robustness\n"
    "source: proposer-9b-cheap\n"
    "est_gpu_hours: 14\n"
    "spec: Sampled-rollout robustness rung — does the mediation signature "
    "survive off the greedy path? [truncated]"
)
_I1426_SCOPE_EVENT = {
    "kind": "epm:followup-scope",
    "ts": "2026-07-18T14:19:00Z",
    "version": 1,
    "note": _I1426_SCOPE_NOTE,
}
_I1426_PREFIX_CLAUSE = (
    "Follow-up: `sampled-rollout-robustness` round (proposer cost_class free-analysis)."
)
_I1426_CORRECTED_CLAUSE = (
    "Follow-up: `sampled-rollout-robustness` round (same-issue follow-up, greedy-decode "
    "n=1 caveat dissolves; source proposer-9b-cheap (GPU cheap band), est 14 GPU-h, "
    "actual ~1.8)."
)


def test_context_followup_scope_contradiction_fails_v4(monkeypatch):
    """(a) The #1426 pre-fix regression fixture: a free-analysis Context
    clause against the real GPU-band scope-marker fields FAILs via C2,
    naming the label + both sides."""
    import explore_persona_space.task_workflow as tw

    monkeypatch.setattr(tw, "list_events", lambda n: [_I1426_SCOPE_EVENT])
    body = _v4_body_with_footer_line(_I1426_PREFIX_CLAUSE)
    r = verify_task_body.check_context_followup_scope_consistency(body, issue=1426)
    assert not r.passed, r.detail
    assert "sampled-rollout-robustness" in r.detail
    assert "proposer-9b-cheap" in r.detail
    assert "14" in r.detail


def test_context_followup_scope_matching_row_passes(monkeypatch):
    """(b) The corrected #1426 clause (verbatim) PASSes — no FAIL, no WARN
    (the intra-clause `; ` delimiter bounds the window; kill criterion 2)."""
    import explore_persona_space.task_workflow as tw

    monkeypatch.setattr(tw, "list_events", lambda n: [_I1426_SCOPE_EVENT])
    body = _v4_body_with_footer_line(_I1426_CORRECTED_CLAUSE)
    r = verify_task_body.check_context_followup_scope_consistency(body, issue=1426)
    assert r.passed and not r.is_warn, r.detail


def test_context_followup_scope_missing_tokens_skip(monkeypatch):
    """(c) Per-label missing tokens skip, never FAIL: (i) label named with
    no tokens (S6); (ii) out-of-enum source token is prose, not a claim
    (S1); (iii) marker source unparseable (S2)."""
    import explore_persona_space.task_workflow as tw

    # (i) mention, no claim tokens.
    monkeypatch.setattr(
        tw, "list_events", lambda n: [_scope_event_47("r-x", source="user-chat", est="0")]
    )
    body = _v4_body_with_footer_line("Follow-up round `r-x` folded 2026-07-18")
    r = verify_task_body.check_context_followup_scope_consistency(body, issue=123)
    assert r.passed and not r.is_warn, r.detail
    # (ii) "source data" — captured token outside the enum, discarded.
    monkeypatch.setattr(
        tw, "list_events", lambda n: [_scope_event_47("r-x", source="proposer-9b-cheap", est="8")]
    )
    body = _v4_body_with_footer_line("round `r-x` (source data from HF, folded)")
    r = verify_task_body.check_context_followup_scope_consistency(body, issue=123)
    assert r.passed and not r.is_warn, r.detail
    # (iii) marker source unparseable -> body source claim has nothing to
    # contradict (C1 needs both sides).
    ev = {
        "kind": "epm:followup-scope",
        "ts": "2026-07-18T14:19:00Z",
        "version": 1,
        "note": "followup_label: r-x\nest_gpu_hours: 14",
    }
    monkeypatch.setattr(tw, "list_events", lambda n: [ev])
    body = _v4_body_with_footer_line("round `r-x` (source user-chat, folded)")
    r = verify_task_body.check_context_followup_scope_consistency(body, issue=123)
    assert r.passed and not r.is_warn, r.detail


def test_context_followup_scope_no_events_or_bare_file_skips(monkeypatch):
    """(d) issue=None skips; unknown issue (plain FileNotFoundError) skips;
    `StaleTaskPathError` (registry corruption) PROPAGATES."""
    import explore_persona_space.task_workflow as tw

    body = _v4_body_with_footer_line("round `r-x` (source user-chat)")
    r = verify_task_body.check_context_followup_scope_consistency(body, issue=None)
    assert r.passed and "no issue id" in r.detail

    def _boom(n):
        raise FileNotFoundError(n)

    monkeypatch.setattr(tw, "list_events", _boom)
    r = verify_task_body.check_context_followup_scope_consistency(body, issue=999999)
    assert r.passed and "unknown issue" in r.detail

    def _stale(n):
        raise tw.StaleTaskPathError("registry entry stale for task")

    monkeypatch.setattr(tw, "list_events", _stale)
    with pytest.raises(tw.StaleTaskPathError):
        verify_task_body.check_context_followup_scope_consistency(body, issue=999999)


def test_context_followup_scope_v3_contradiction_warns_not_fails(monkeypatch):
    """(e) A grandfathered v3 body with the same C2 contradiction WARNs
    (`passed=True, is_warn=True`) — never a new hard FAIL below the v4
    sentinel (the #1418 fail-denied precedent)."""
    import explore_persona_space.task_workflow as tw

    monkeypatch.setattr(tw, "list_events", lambda n: [_I1426_SCOPE_EVENT])
    body = _V3_GOOD_BODY.replace(
        "- Originating prompt: origin prompt not recorded",
        "- Originating prompt: origin prompt not recorded\n- " + _I1426_PREFIX_CLAUSE,
    )
    r = verify_task_body.check_context_followup_scope_consistency(body, issue=1426)
    assert r.passed and r.is_warn, (r.passed, r.is_warn, r.detail)
    assert "grandfathered v3/v2" in r.detail
    assert "sampled-rollout-robustness" in r.detail


def test_context_followup_scope_free_analysis_round_passes(monkeypatch):
    """(f) A truthful free-analysis clause whose round posted only an
    `epm:free-analysis-followup-run` marker (no scope) skips — the check
    keys entirely on scope-armed labels (decision-table S4)."""
    import explore_persona_space.task_workflow as tw

    free = {
        "kind": "epm:free-analysis-followup-run",
        "ts": "2026-07-14T10:00:00Z",
        "version": 1,
        "note": "followup_ref: r-free\noutcome: folded",
    }
    monkeypatch.setattr(tw, "list_events", lambda n: [free])
    body = _v4_body_with_footer_line(
        "a user-chat inline free-analysis round (`r-free`, run 2026-07-14)."
    )
    r = verify_task_body.check_context_followup_scope_consistency(body, issue=123)
    assert r.passed and not r.is_warn, r.detail
    assert "no epm:followup-scope markers" in r.detail


def test_context_followup_scope_source_mismatch_fails(monkeypatch):
    """(g) C1: body `source user-chat` vs marker `source: proposer-9b-cheap`
    is a hard v4 FAIL naming both sides."""
    import explore_persona_space.task_workflow as tw

    monkeypatch.setattr(
        tw, "list_events", lambda n: [_scope_event_47("r-x", source="proposer-9b-cheap", est="14")]
    )
    body = _v4_body_with_footer_line("round `r-x` (source user-chat, est 14 GPU-h)")
    r = verify_task_body.check_context_followup_scope_consistency(body, issue=123)
    assert not r.passed, r.detail
    assert "user-chat" in r.detail and "proposer-9b-cheap" in r.detail


def test_context_followup_scope_correction_latest_wins(monkeypatch):
    """(h) Two scope entries for one label: the latest (correction) entry's
    `est_gpu_hours` binds; a correction omitting `source` falls back to the
    group's first-parseable source (library semantics)."""
    import explore_persona_space.task_workflow as tw

    events = [
        _scope_event_47(
            "r-x", source="proposer-9b-cheap", est="14", ts="2026-07-18T10:00:00Z", version=1
        ),
        # Correction: revises est, omits source.
        _scope_event_47("r-x", est="2", ts="2026-07-18T11:00:00Z", version=2),
    ]
    monkeypatch.setattr(tw, "list_events", lambda n: list(events))
    # Body states the STALE est 14 -> W1 WARN against the corrected est 2.
    body = _v4_body_with_footer_line("round `r-x` (source proposer-9b-cheap, est 14 GPU-h)")
    r = verify_task_body.check_context_followup_scope_consistency(body, issue=123)
    assert r.passed and r.is_warn, (r.passed, r.is_warn, r.detail)
    assert "est_gpu_hours: 2" in r.detail
    # Body matching the corrected est -> clean PASS (source fallback held).
    body2 = _v4_body_with_footer_line("round `r-x` (source proposer-9b-cheap, est 2 GPU-h)")
    r2 = verify_task_body.check_context_followup_scope_consistency(body2, issue=123)
    assert r2.passed and not r2.is_warn, r2.detail


def test_context_followup_scope_unlabeled_pseudo_labels_skipped(monkeypatch):
    """(i) An unlabeled scope founds an `unlabeled-<ts>` pseudo-label group,
    which the marker-side facts EXCLUDE — no facts, skip."""
    import explore_persona_space.task_workflow as tw

    ev = {
        "kind": "epm:followup-scope",
        "ts": "2026-07-18T14:19:00Z",
        "version": 1,
        "note": "source: user-chat\nspec: something unlabeled",
    }
    monkeypatch.setattr(tw, "list_events", lambda n: [ev])
    body = _v4_body_with_footer_line("round folded, source user-chat, cost free-analysis")
    r = verify_task_body.check_context_followup_scope_consistency(body, issue=123)
    assert r.passed and not r.is_warn, r.detail
    assert "no epm:followup-scope markers" in r.detail


def test_context_followup_scope_blockquoted_note_not_scanned(monkeypatch):
    """(j) A blockquoted verbatim scope note (which contains the marker's own
    `source:` line + free-analysis vocabulary) contributes NO body claims —
    `_context_scan_region` strips blockquote lines (#959 precedent)."""
    import explore_persona_space.task_workflow as tw

    monkeypatch.setattr(
        tw, "list_events", lambda n: [_scope_event_47("r-x", source="proposer-9b-cheap", est="14")]
    )
    body = _V4_GOOD_BODY.replace(
        "- Originating prompt: origin prompt not recorded",
        "- Originating prompt: origin prompt not recorded\n"
        "- Follow-up round `r-x` folded 2026-07-18\n"
        "> followup_label: r-x source: user-chat cost_class free-analysis",
    )
    r = verify_task_body.check_context_followup_scope_consistency(body, issue=123)
    assert r.passed and not r.is_warn, r.detail


def test_context_followup_scope_est_mismatch_warns(monkeypatch):
    """(k) W1: body `est 14 GPU-h` vs marker `est_gpu_hours: 5` WARNs (never
    FAILs) and the message states the 0.5 GPU-h tolerance explicitly."""
    import explore_persona_space.task_workflow as tw

    monkeypatch.setattr(
        tw, "list_events", lambda n: [_scope_event_47("r-x", source="proposer-9b-cheap", est="5")]
    )
    body = _v4_body_with_footer_line("round `r-x` (source proposer-9b-cheap, est 14 GPU-h)")
    r = verify_task_body.check_context_followup_scope_consistency(body, issue=123)
    assert r.passed and r.is_warn, (r.passed, r.is_warn, r.detail)
    assert "0.5 GPU-h" in r.detail


def test_context_followup_scope_multi_round_window_attribution(monkeypatch):
    """(l) Two labels on one footer line: tokens attribute to the nearest
    preceding label (window bounded at the next any-label mention), and the
    full-token mention guards block sibling-label substring collisions."""
    import explore_persona_space.task_workflow as tw

    events = [
        _scope_event_47("r-a", source="proposer-9b-cheap", est="8", ts="2026-07-18T10:00:00Z"),
        _scope_event_47("r-b", source="proposer-9b-cheap", est="8", ts="2026-07-18T11:00:00Z"),
    ]
    monkeypatch.setattr(tw, "list_events", lambda n: list(events))
    body = _v4_body_with_footer_line(
        "round `r-a` (source user-chat) and round `r-b` (source proposer-9b-cheap)"
    )
    r = verify_task_body.check_context_followup_scope_consistency(body, issue=123)
    assert not r.passed, r.detail
    assert "`r-a`" in r.detail and "`r-b`" not in r.detail
    # Substring guard: `r-a-b` mention never counts as an `r-a` mention.
    events2 = [
        _scope_event_47("r-a", source="proposer-9b-cheap", est="8", ts="2026-07-18T10:00:00Z"),
        _scope_event_47("r-a-b", source="user-chat", est="0", ts="2026-07-18T11:00:00Z"),
    ]
    monkeypatch.setattr(tw, "list_events", lambda n: list(events2))
    body2 = _v4_body_with_footer_line("round `r-a-b` (source user-chat)")
    r2 = verify_task_body.check_context_followup_scope_consistency(body2, issue=123)
    assert r2.passed and not r2.is_warn, r2.detail


def test_context_followup_scope_i1092_truthful_row_passes(monkeypatch):
    """Pinned #1092 Context-row shape (critic refinement (ii)): truthful
    free-analysis narration about OTHER rounds interleaved after a
    scope-armed GPU-band clause on the same line, `; `-separated — the
    clause-delimiter window bound keeps C2 from false-firing."""
    import explore_persona_space.task_workflow as tw

    events = [
        _scope_event_47(
            "cross-corpus-probe-transfer",
            source="proposer-9b-cheap",
            est="8",
            ts="2026-07-10T20:56:59Z",
            version=1,
        ),
        _scope_event_47(
            "caveat-repairs-plus-operator-arm-comparison",
            source="user-chat",
            est="0",
            ts="2026-07-14T21:18:35Z",
            version=2,
        ),
        _scope_event_47(
            "offvm-battery-refit-and-operator-comparison",
            source="user-chat",
            est="0",
            ts="2026-07-14T23:24:52Z",
            version=3,
        ),
    ]
    monkeypatch.setattr(tw, "list_events", lambda n: list(events))
    # The #1426-adjacent real row (tasks/awaiting_promotion/1092/body.md:324,
    # abridged to the follow-up clauses; single physical line as in corpus).
    row = (
        "created 2026-07-07; GPU phases run 2026-07-08-09. Lineage: building on "
        "[#923](https://eps.superkaiba.com/tasks/923); one same-issue free-analysis "
        "follow-up round (trait-per-factor repair, proposer-initiated, folded "
        "2026-07-10); a second same-issue follow-up round (cross-corpus "
        "supervised-probe transfer, `followup_label: cross-corpus-probe-transfer`, "
        "proposer-initiated cheap band, run + folded 2026-07-10); a user-chat inline "
        "free-analysis round (`caveat-repairs-plus-operator-arm-comparison`, run "
        "2026-07-14: transport floors, battery-invariance verification, leak root "
        "cause); a third same-issue follow-up round "
        "(`followup_label: offvm-battery-refit-and-operator-comparison`, "
        'user-initiated — originating prompt, verbatim: "dispatch" — run '
        "2026-07-15-16, folded 2026-07-16). Originating prompt, verbatim:"
    )
    body = _v4_body_with_footer_line(row)
    r = verify_task_body.check_context_followup_scope_consistency(body, issue=1092)
    assert r.passed and not r.is_warn, r.detail
    assert "3 follow-up label clause(s) consistent" in r.detail


def test_context_followup_scope_free_analysis_user_chat_zero_est_skips(monkeypatch):
    """(S3, critic refinement) A free-analysis claim against a user-chat
    scope with est absent or 0 cannot prove a contradiction — skip
    (user-chat rounds carry no cost_class)."""
    import explore_persona_space.task_workflow as tw

    for est in ("0", None):
        monkeypatch.setattr(
            tw, "list_events", lambda n, e=est: [_scope_event_47("r-x", source="user-chat", est=e)]
        )
        body = _v4_body_with_footer_line("round `r-x` (free-analysis, folded 2026-07-18)")
        r = verify_task_body.check_context_followup_scope_consistency(body, issue=123)
        assert r.passed and not r.is_warn, (est, r.detail)


def test_context_followup_scope_matching_source_suppresses_c2(monkeypatch):
    """(Critic refinement (i)) An explicit in-enum body source claim that
    MATCHES the marker source suppresses C2 — the stronger evidence wins,
    even when the marker's est_gpu_hours > 0."""
    import explore_persona_space.task_workflow as tw

    monkeypatch.setattr(
        tw, "list_events", lambda n: [_scope_event_47("r-x", source="user-chat", est="3")]
    )
    body = _v4_body_with_footer_line(
        "round `r-x` (cost_class free-analysis at filing, source user-chat)"
    )
    r = verify_task_body.check_context_followup_scope_consistency(body, issue=123)
    assert r.passed and not r.is_warn, r.detail


def test_context_followup_scope_i922_paren_clause_passes(monkeypatch):
    """Pinned #922 Context-row shape (corpus-sweep tighten, kill criterion
    §6.1): the scope-armed label sits INSIDE a parenthetical clause and the
    NEXT round's truthful free-analysis narration follows `, and` (no `; `
    delimiter) — the unmatched-close-paren window bound keeps C2 from
    false-firing. The #922 marker also uses a `gpu_hours_estimate:` field
    (not `est_gpu_hours:`), so est parses absent."""
    import explore_persona_space.task_workflow as tw

    ev = {
        "kind": "epm:followup-scope",
        "ts": "2026-07-04T03:35:12Z",
        "version": 1,
        "note": (
            "followup_label: paired-provenance-transfer\n"
            "source: proposer-9b-cheap\n"
            "question_relation: same\n"
            "gpu_hours_estimate: 2"
        ),
    }
    monkeypatch.setattr(tw, "list_events", lambda n: [ev])
    # Abridged from tasks/awaiting_promotion/922/body.md:209 (read 2026-07-18).
    row = (
        "Created 2026-07-03; run 2026-07-03, plus one zero-GPU spectral-read "
        "follow-up round (analysis-only) and one proposer-initiated cheap-band "
        "auto-run repair round (followup_label `paired-provenance-transfer`, "
        "2026-07-04, ~0.3 GPU-h realized), and one user-requested inline "
        "free-analysis round (fixed-point + slow-shell characterization, "
        "2026-07-15, 0 GPU-h)."
    )
    body = _v4_body_with_footer_line(row)
    r = verify_task_body.check_context_followup_scope_consistency(body, issue=922)
    assert r.passed and not r.is_warn, r.detail
    assert "1 follow-up label clause(s) consistent" in r.detail


def test_context_followup_scope_comma_and_joiner_bounds_window(monkeypatch):
    """(#1521 r2 — closes CONCERN check46-residual-comma-and-clause) The
    PAREN-LESS `, and `-joined next-round free-analysis narration after a
    scope-armed clause must NOT attribute to the label: the `, and ` clause
    joiner bounds the window (cut at the comma), so no C2 fires against the
    GPU-band marker. Pre-fix this exact shape FAILed C2 (both parens are
    MATCHED, so neither the `;`/`. ` delimiter nor the unmatched-paren
    bound cuts before the free-analysis token)."""
    import explore_persona_space.task_workflow as tw

    monkeypatch.setattr(
        tw,
        "list_events",
        lambda n: [_scope_event_47("r-x", source="proposer-9b-cheap", est="14")],
    )
    body = _v4_body_with_footer_line(
        "one cheap-band round `r-x` (folded 2026-07-18), and a free-analysis "
        "re-read of the stored activations (folded 2026-07-18)"
    )
    r = verify_task_body.check_context_followup_scope_consistency(body, issue=123)
    assert r.passed and not r.is_warn, r.detail
    assert "1 follow-up label clause(s) consistent" in r.detail


def test_context_followup_scope_quoted_note_span_no_c1(monkeypatch):
    """(#1521 r2 — reviewer Minor) A label mention INSIDE a multi-word
    backtick span quoting the scope note carries the QUOTED contradicting
    source token before the span's closing backtick, which escapes the
    paired-span quotation strip; the enclosing-span remainder strip
    (odd-backtick window, `_strip_enclosing_span_remainder`) removes it, so
    no C1 fires against the proposer-9b-cheap marker. Pre-fix this shape
    FAILed C1 with the quoted `user-chat` attributed as a body claim."""
    import explore_persona_space.task_workflow as tw

    monkeypatch.setattr(
        tw,
        "list_events",
        lambda n: [_scope_event_47("r-x", source="proposer-9b-cheap", est="14")],
    )
    body = _v4_body_with_footer_line(
        "scope note quoted verbatim: `followup_label: r-x source: user-chat` "
        "(a mis-filed draft field, corrected before dispatch)"
    )
    r = verify_task_body.check_context_followup_scope_consistency(body, issue=123)
    assert r.passed and not r.is_warn, r.detail
    assert "1 follow-up label clause(s) consistent" in r.detail


# ─── kind:infra|batch|survey not-applicable short-circuit (task #1724) ────
#
# `_kind_short_circuit` returns the kind name when the verifier should
# print an `OVERALL: N/A` verdict and exit 3, and returns `None` when the
# body should fall through to the normal check chain. The seven
# pure-helper cases below cover every branch of the predicate; the
# subprocess smoke exercises the full main() short-circuit end-to-end
# against a live in-tree `kind: infra, has_clean_result: false` task
# discovered at test time.


def test_short_circuit_infra_unpromoted():
    """kind:infra with has_clean_result=False fires the short-circuit."""
    assert (
        verify_task_body._kind_short_circuit({"kind": "infra", "has_clean_result": False})
        == "infra"
    )


def test_short_circuit_batch_unpromoted():
    """kind:batch with has_clean_result=False fires the short-circuit."""
    assert (
        verify_task_body._kind_short_circuit({"kind": "batch", "has_clean_result": False})
        == "batch"
    )


def test_short_circuit_survey_unpromoted():
    """kind:survey with has_clean_result=False fires the short-circuit."""
    assert (
        verify_task_body._kind_short_circuit({"kind": "survey", "has_clean_result": False})
        == "survey"
    )


def test_short_circuit_experiment_falls_through():
    """kind:experiment never short-circuits — it always gets the full check
    chain against the clean-result spec (a `kind: experiment` body without
    a promoted clean-result is still a real body to verify).
    """
    assert (
        verify_task_body._kind_short_circuit({"kind": "experiment", "has_clean_result": False})
        is None
    )


def test_short_circuit_analysis_falls_through():
    """kind:analysis is DELIBERATELY excluded from the short-circuit set.

    An analysis task with has_clean_result: false and no clean-result body
    is a LEGITIMATE FAIL — it signals the analyzer hasn't produced a
    finding yet (SKILL.md § 9a-quater). Silencing that pre-promotion FAIL
    would defeat the auto-continue pipeline's expectation that the FAIL
    disappears the moment the analyzer flips `has_clean_result: true`.
    """
    assert (
        verify_task_body._kind_short_circuit({"kind": "analysis", "has_clean_result": False})
        is None
    )


def test_short_circuit_promoted_infra_falls_through():
    """A rare mis-filed `kind: infra` task that DOES carry a promoted
    clean-result (has_clean_result=True) still gets the full check path —
    the short-circuit only fires on the unpromoted subclass.
    """
    assert verify_task_body._kind_short_circuit({"kind": "infra", "has_clean_result": True}) is None


def test_short_circuit_kind_absent_falls_through():
    """No `kind` frontmatter → predicate returns None (safe fallback)."""
    assert verify_task_body._kind_short_circuit({}) is None


def test_short_circuit_infra_unpromoted_string_false():
    """String-coerced `has_clean_result` values.

    YAML `false` bareword parses to bool False (already covered above).
    A mis-quoted `has_clean_result: "false"` parses to the STRING
    "false", which `bool()` reads as truthy — the opposite of the
    author's intent. `_kind_short_circuit`'s string-coercion branch
    treats every case-insensitive whitespace-stripped variant of
    "false" / "no" / "0" / "null" / "none" as falsy, so all four
    variations still fire the short-circuit; a legitimate truthy string
    ("yes", "true", any other non-falsy string) leaves it inert.
    """
    for hcr in ("false", "FALSE", " False ", "no", "0"):
        assert (
            verify_task_body._kind_short_circuit({"kind": "infra", "has_clean_result": hcr})
            == "infra"
        ), f"expected string {hcr!r} to be treated as falsy"
    # A truthy string means the task was promoted (analysis-style
    # mis-file with a mis-quoted `has_clean_result: "yes"` would run the
    # full check path, which is the intended behavior).
    assert (
        verify_task_body._kind_short_circuit({"kind": "infra", "has_clean_result": "yes"}) is None
    )


def test_short_circuit_infra_has_clean_result_none():
    """`has_clean_result: null` (YAML null) parses to Python None, which
    is falsy under `bool(None)` — the short-circuit fires. This is the
    same branch as `has_clean_result` missing entirely, but exercised
    explicitly because null values arrive naturally from YAML.
    """
    assert (
        verify_task_body._kind_short_circuit({"kind": "infra", "has_clean_result": None}) == "infra"
    )


def _find_live_infra_task_id() -> int | None:
    """Locate a live in-tree `kind: infra, has_clean_result: false` task.

    Scans `tasks/proposed/**/body.md` and `tasks/planning/**/body.md` for
    a body whose YAML frontmatter has `kind: infra` and truthy-falsy
    `has_clean_result`. Returns the LOWEST task id for determinism
    across concurrent test runs. Returns ``None`` when no eligible
    task exists (the subprocess smoke then pytest-skips cleanly).
    """
    import yaml

    repo_root = Path(__file__).resolve().parents[1]
    tasks_root = repo_root / "tasks"
    candidates: list[int] = []
    for status_dir in ("proposed", "planning"):
        status_path = tasks_root / status_dir
        if not status_path.exists():
            continue
        for body_path in status_path.glob("*/body.md"):
            try:
                text = body_path.read_text()
            except OSError:
                continue
            # Cheap gate: skip bodies without a YAML front-matter block.
            if not text.startswith("---\n"):
                continue
            rest = text[4:]
            end = rest.find("\n---\n")
            if end == -1:
                continue
            try:
                fm = yaml.safe_load(rest[:end]) or {}
            except yaml.YAMLError:
                continue
            if not isinstance(fm, dict):
                continue
            if fm.get("kind") != "infra":
                continue
            hcr = fm.get("has_clean_result")
            if isinstance(hcr, str):
                hcr_bool = hcr.strip().lower() not in {"", "false", "no", "0", "null", "none"}
            else:
                hcr_bool = bool(hcr)
            if hcr_bool:
                continue
            # Task id is the directory name (e.g. tasks/proposed/1724/body.md).
            try:
                candidates.append(int(body_path.parent.name))
            except ValueError:
                continue
    if not candidates:
        return None
    return min(candidates)


def test_main_subprocess_kind_infra_returns_exit_3():
    """End-to-end smoke: `verify_task_body.py --issue <N>` on a live
    `kind: infra, has_clean_result: false` task returns exit code 3
    with an `OVERALL: N/A (kind: infra ...)` verdict.

    Uses the LOWEST-numbered eligible in-tree task for determinism.
    Skips cleanly if no eligible infra task exists (extremely unlikely
    — the corpus contains hundreds of them). The task's identity is
    read fresh at test time, so if the picked task is promoted /
    archived tomorrow the test self-adapts.
    """
    task_id = _find_live_infra_task_id()
    if task_id is None:
        pytest.skip(
            "no live kind:infra,has_clean_result:false task under "
            "tasks/{proposed,planning}/ — nothing to smoke"
        )
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["uv", "run", "python", "scripts/verify_task_body.py", "--issue", str(task_id)],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        check=False,
    )
    assert result.returncode == 3, (
        f"expected exit 3, got {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OVERALL: N/A (kind: infra" in result.stdout, (
        f"expected N/A verdict, got:\n{result.stdout}"
    )


# ─── Check 48: figure-less quantitative result sections (v4-only WARN, #1832) ─
#
# Check 21 (`check_v4_results_beat`) deliberately exempts EVERY figure-less
# result (the qualitative carve-out), so a number-dense headline result with
# zero inline images passed the mechanical verifier silently (incident #1769
# Result 5). Check 48 binds ONLY the figure-less + quantitative intersection:
# >=3 standalone numeric tokens in non-code content, or any GFM table row.
# WARN, never FAIL; vacuous PASS on non-v4 bodies.

_CHECK48_NAME = "Quantitative results carry a figure (v4)"

# Verbatim incident fixture: task #1769's fold-round Result 5 (the alpha=2
# three-treatment lattice carrying the H1 claim), recovered from git history
# (`git show 1798450ac4:tasks/interpreting/1769/body.md`, lines 394-414) at
# the revision where it still carried ZERO inline images — a number-dense
# block whose "figure" is a `> **Figure.**`-captioned GFM table. The round-1
# LM clean-result-critic REVISE was the only catch pre-check-48.
_ISSUE1769_RESULT5_BLOCK = """\
### Evil and sycophancy decode-driven timing at α=2 holds under all three CJK-intrusion treatments; hallucination flips to mixed

Three-treatment α=2 lattice: f_d ratio, cluster-bootstrap 95% CI (B=2000, seed=42), classification, and draw counts per behavior. Raw = all draws; exclusion = CJK draws removed; zeroing = CJK draws scored 0 (n_zeroed = decode_only / both).

| Behavior | Treatment | f_d | 95% CI | Verdict | N kept (decode / both) | N zeroed (decode / both) |
|---|---|---|---|---|---|---|
| evil | raw | 1.008 | (0.983, 1.032) | decode-driven | 200 / 200 | — |
| evil | exclusion | 1.026 | (0.993, 1.058) | decode-driven | 108 / 102 | — |
| evil | zeroing | 1.072 | (0.932, 1.235) | decode-driven | 200 / 200 | 90 / 94 |
| hallucination | raw | 0.890 | (0.802, 0.990) | decode-driven | 200 / 200 | — |
| hallucination | exclusion | 0.834 | (0.716, 0.973) | mixed | 148 / 149 | — |
| hallucination | zeroing | 0.768 | (0.546, 1.016) | mixed | 200 / 200 | 50 / 44 |
| sycophancy | raw | 0.928 | (0.830, 1.050) | decode-driven | 200 / 200 | — |
| sycophancy | exclusion | 0.930 | (0.819, 1.040) | decode-driven | 189 / 185 | — |
| sycophancy | zeroing | 0.936 | (0.821, 1.081) | decode-driven | 200 / 200 | 11 / 15 |

> **Figure.** Table of f_d (decode fraction), 95% cluster-bootstrap CIs, and lattice classification for three behaviors × three CJK-intrusion treatments at α=2. Source: `eval_results/issue_1769/analysis/alpha2_clean_lattice.json`, n_questions=20, n_draws=10, B=2000, seed=42, lattice thresholds decode-driven lower-CI > 0.75 / prefill-committed CI ⊆ (−0.25, 0.25).

Evil and sycophancy return decode-driven verdicts under all three treatments. The evil exclusion arm drops 45–47% of decode/both draws yet f_d rises to 1.026 (0.993, 1.058); the zeroing CI is wider (0.932, 1.235) but the point estimate stays above 1.0. Sycophancy has low CJK exposure (5.5–7.5%) and is unaffected. Hallucination shifts from decode-driven under raw scoring to mixed under both intrusion-robust treatments: the exclusion CI (0.716, 0.973) falls below the 0.75 threshold, and the zeroing CI (0.546, 1.016) spans 1.0. The registered ceiling check fired for evil/α=2 (both-arm mean=86.03 > 85), so evil is excluded from operating-alpha selection; the f_d analysis is still informative but deltas are near scale-top.
"""


def test_check48_quant_numeric_prose_figureless_warns():
    """MUST-WARN: a figure-less result whose prose carries >=3 standalone
    numeric tokens WARNs (passed stays True — never FAIL), naming the
    section and the numeric-token basis."""
    body = _v4_minimal_results_body(
        "### Lift by seed\n\n"
        "The lift is 17.3 points (baseline 70.4%, treated 87.7%) across 3 seeds; "
        "every seed moves the same direction.\n"
    )
    res = verify_task_body.check_v4_quant_result_figure(body)
    assert res.passed is True
    assert res.is_warn is True
    assert "'Lift by seed'" in res.detail
    assert "numeric tokens" in res.detail
    assert "no inline figure" in res.detail


def test_check48_table_only_figureless_warns():
    """MUST-WARN: a figure-less result whose only quantitative content is a
    GFM table (no standalone prose numbers) WARNs with the table basis."""
    body = _v4_minimal_results_body(
        "### Rates by arm\n\n"
        "Judge-scored rates for both arms, all seeds pooled.\n\n"
        "| Arm | Rate |\n"
        "|---|---|\n"
        "| base | low |\n"
        "| treated | high |\n"
    )
    res = verify_task_body.check_v4_quant_result_figure(body)
    assert res.passed is True
    assert res.is_warn is True
    assert "'Rates by arm'" in res.detail
    assert "GFM table" in res.detail


def test_check48_qualitative_figureless_passes():
    """The check-21 exemption survives: a figure-less QUALITATIVE result
    (below the 3-numeric-token floor, no table) draws no WARN — this also
    pins the strict >=3 threshold (2 tokens stay silent)."""
    body = _v4_minimal_results_body(
        "### Refusal pattern\n\n"
        "Seeds 42 and 137 behave identically under both prompts; the refusal "
        "pattern is stable and no arm flips direction.\n"
    )
    res = verify_task_body.check_v4_quant_result_figure(body)
    assert res.passed is True
    assert res.is_warn is False
    assert "no figure-less quantitative section" in res.detail


def test_check48_figure_bearing_quant_passes():
    """A figure-bearing result never WARNs from check 48, regardless of
    numeric density (check 21 owns the three-beat framing there)."""
    body = _v4_minimal_results_body(
        "### Lift by seed\n\n"
        "The lift is 17.3 points (baseline 70.4%, treated 87.7%) across 3 seeds.\n\n"
        "![alt](https://x/y.png)\n\n"
        "> **Figure.** *Lead claim.* rest of caption.\n\n"
        "Interpretation prose below the caption.\n"
    )
    res = verify_task_body.check_v4_quant_result_figure(body)
    assert res.passed is True
    assert res.is_warn is False


def test_check48_skips_non_v4():
    """Forward-only: v3 and legacy bodies PASS vacuously — check 48 never
    fires outside the v4 sentinel."""
    for fixture in (_V3_GOOD_BODY, GOOD_BODY):
        res = verify_task_body.check_v4_quant_result_figure(fixture)
        assert res.passed is True
        assert res.is_warn is False
        assert "skipped — not a v4 body" in res.detail


def test_check48_fenced_numbers_ignored():
    """Numbers living only inside a fenced code block do not count toward
    the quantitative floor — a figure-less config/CLI snippet result stays
    qualitative."""
    body = _v4_minimal_results_body(
        "### Run configuration\n\n"
        "Config used for the run, quoted verbatim.\n\n"
        "```json\n"
        '{"lr": 3e-5, "seeds": [42, 137, 256], "epochs": 3}\n'
        "```\n"
    )
    res = verify_task_body.check_v4_quant_result_figure(body)
    assert res.passed is True
    assert res.is_warn is False


def test_check48_details_table_ignored():
    """A GFM table living only inside a `<details>` block does not count —
    collapsed content stays LM-lens territory (the `_prose_words`
    convention)."""
    body = _v4_minimal_results_body(
        "### Cherry-picked rows\n\n"
        "Qualitative summary of the run; representative rows collapsed below.\n\n"
        "<details>\n<summary>rows</summary>\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        "</details>\n"
    )
    res = verify_task_body.check_v4_quant_result_figure(body)
    assert res.passed is True
    assert res.is_warn is False


def test_check48_fenced_image_is_not_a_figure():
    """DURABILITY PIN (intended behavior): an image that exists only inside a
    fenced code block is NOT an inline figure (matches check 21's fence-aware
    `_v4_first_image_index`), so quantitative prose beside it still WARNs."""
    body = _v4_minimal_results_body(
        "### Lift by seed\n\n"
        "The lift is 17.3 points (baseline 70.4%, treated 87.7%) across 3 seeds.\n\n"
        "```markdown\n"
        "![alt](https://x/y.png)\n"
        "```\n"
    )
    res = verify_task_body.check_v4_quant_result_figure(body)
    assert res.passed is True
    assert res.is_warn is True
    assert "'Lift by seed'" in res.detail


def test_check48_membership():
    """Check 48 rides CHECKS (body-only — it needs no issue number)."""
    assert verify_task_body.check_v4_quant_result_figure in verify_task_body.CHECKS


def test_check48_issue1769_result5_incident_fixture_warns():
    """MUST-WARN incident fixture (#1832 plan-critic concern 1): the verbatim
    #1769 Result-5 block — number-dense prose + a `> **Figure.**`-captioned
    GFM table, zero inline images — draws the check-48 WARN naming the
    section. This is the exact shape that passed the mechanical verifier
    silently and burned a round-1 LM REVISE."""
    body = _v4_minimal_results_body(_ISSUE1769_RESULT5_BLOCK)
    res = verify_task_body.check_v4_quant_result_figure(body)
    assert res.passed is True
    assert res.is_warn is True
    assert "'Evil and sycophancy decode-driven timing at" in res.detail
    assert "GFM table" in res.detail
    assert "no inline figure" in res.detail


# ─── Check 49: multi-figure result sections without pair evidence (v4 WARN, #1879) ─
#
# Lens 9's one-result-one-figure rule allows a second inline figure ONLY as
# the sanctioned raw+processed / aggregate+per-unit pair (SPEC.md § Low-level
# data plot behind every aggregate). Check 49 WARNs a `### <result>` embedding
# >1 inline figure whose pair evidence is in NEITHER the figure basenames
# (`_PER_UNIT_FIG_RE`) nor the figures' alt text / blockquote caption lines
# (`_DECLARED_PAIR_RE`) — general section prose deliberately does NOT count
# (the origin #1769 what-is-plotted beat says "per-question" as routine
# SPEC-mandated disclosure prose). WARN, never FAIL; vacuous PASS on non-v4
# bodies.

_CHECK49_NAME = "One inline figure per result, or a declared pair (v4)"

# Verbatim incident fixture: task #1769's fu1 re-gate dose-ladder section —
# BOTH figure blocks (alt + blockquote caption verbatim from the #1769 body)
# under one H3, plus the real what-is-plotted prose whose "per-question"
# sentence must NOT silence the WARN (the caption/alt scoping is the round-1
# plan-critic Must-Fix). Two distinct analyses (dose ladder + alpha-3 lattice)
# shipped under one `### <result>`; the verifier read PASS and only the LM
# clean-result-critic caught it.
_ISSUE1769_FU1_DOSE_LADDER_BLOCK = """\
### The dose ladder places the CJK collapse between α=2 and 3

The figure plots Δ_both (raw scoring, mean graded score minus the neither arm) against α ∈ {1, 1.5, 2, 3, 4} per behavior, with the interpretable window (α ≤ 2) and the CJK-affected region shaded; the per-question data behind the ladder points appear in the α=1.5, α=2, and α=3 sections.

![Dose ladder of both-arm effect versus alpha with interpretable-window and CJK-affected shading](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c66bd5b6d9672983b29f7341f96f4451aeee6eb6/figures/issue_1769/fig_dose_ladder.png)

> **Figure.** *Installed effect rises through α=3 for hallucination and sycophancy while evil peaks at α=2; the α=3–4 points sit in the CJK-affected region.* Δ_both (raw) per behavior at five doses, 200 draws per arm-dose (evil: 5.8, 61.6, 86.0, 72.4, 31.6 across the ladder); evil labeled scheming.

![Decode fraction at alpha 3 under three CJK-intrusion treatments with degenerate cells marked](https://raw.githubusercontent.com/superkaiba/explore-persona-space/46f3eb7d42d7e45f30d414d893c181fcf0c860e0/figures/issue_1769/fig_alpha3_lattice.png)

> **Figure.** *Only sycophancy keeps a computable three-treatment read at α=3.* f_d with 95% CIs per treatment; evil and hallucination exclusion cells are drawn as N/A notes (84.5% and 92% decode-arm intrusion; 30 and 13 of 200 draws remain); evil labeled scheming.

Installed effect rises through α=3 for hallucination (67.4) and sycophancy (47.4) while evil peaks at 86.0 at α=2, and all three fall back at α=4.
"""


def test_check49_verbatim_1769_fu1_dose_ladder_warns():
    """Row 1 (kill-criterion arbiter): the verbatim #1769 fu1 dose-ladder
    section — two figures, no alt/caption idiom hit ("per behavior" /
    "per arm-dose" / "per treatment" are not in the alternation; "(raw)"
    has no alongside/counterpart/version/view/scatter within reach), no
    per-unit basename — WARNs naming the H3 + both basenames, and the
    prose-level "per-question" sentence does not silence it."""
    body = _v4_minimal_results_body(_ISSUE1769_FU1_DOSE_LADDER_BLOCK)
    res = verify_task_body.check_v4_result_figure_cardinality(body)
    assert res.passed is True
    assert res.is_warn is True
    assert "'The dose ladder places the CJK collapse" in res.detail
    assert "2 figures" in res.detail
    assert "fig_dose_ladder.png" in res.detail
    assert "fig_alpha3_lattice.png" in res.detail


def test_check49_per_unit_companion_stem_passes():
    """Row 2 — pair evidence (a): a second figure whose basename matches
    the `_PER_UNIT_FIG_RE` companion naming convention
    (`..._percontext_delta.png`) silences the WARN; alts + captions stay
    idiom-free so the stem is the only evidence."""
    body = _v4_minimal_results_body(
        "### Lift by seed\n\n"
        "Aggregate lift across seeds.\n\n"
        "![Aggregate lift bars](https://x/figures/issue_9/lift_summary.png)\n\n"
        "> **Figure.** *Lead.* Aggregate bars.\n\n"
        "![Delta grid](https://x/figures/issue_9/lift_percontext_delta.png)\n\n"
        "> **Figure.** *Lead.* Same data at finer grain.\n"
    )
    res = verify_task_body.check_v4_result_figure_cardinality(body)
    assert res.passed is True
    assert res.is_warn is False


def test_check49_declared_pair_in_caption_or_alt_passes():
    """Row 3 — pair evidence (b): a declared-pair idiom in the second
    figure's CAPTION ("per-question companion ...") or ALT ("raw scatter
    alongside ...") silences the WARN; the basenames carry no per-unit
    stem, so the alt/caption declaration is the only evidence."""
    caption_declared = _v4_minimal_results_body(
        "### Effect by question\n\n"
        "Forest plot plus the underlying data.\n\n"
        "![Forest plot of effects](https://x/figures/issue_9/forest.png)\n\n"
        "> **Figure.** *Lead.* Pooled effects.\n\n"
        "![Scatter of effects](https://x/figures/issue_9/scatter_all.png)\n\n"
        "> **Figure.** *Lead.* per-question companion of the forest plot above.\n"
    )
    res = verify_task_body.check_v4_result_figure_cardinality(caption_declared)
    assert res.passed is True
    assert res.is_warn is False
    alt_declared = _v4_minimal_results_body(
        "### Residualized effect\n\n"
        "Residualized read plus its pre-processing twin.\n\n"
        "![Residualized effect](https://x/figures/issue_9/effect_resid.png)\n\n"
        "> **Figure.** *Lead.* Residualized.\n\n"
        "![raw scatter alongside the residualized view](https://x/figures/issue_9/effect_all.png)\n\n"
        "> **Figure.** *Lead.* Pre-processing twin.\n"
    )
    res2 = verify_task_body.check_v4_result_figure_cardinality(alt_declared)
    assert res2.passed is True
    assert res2.is_warn is False


def test_check49_one_figure_per_section_passes():
    """Row 4: the conforming one-figure-per-result shape draws no WARN."""
    body = _v4_minimal_results_body(
        "### Lift by seed\n\n"
        "Aggregate lift across seeds.\n\n"
        "![Aggregate lift bars](https://x/figures/issue_9/lift_summary.png)\n\n"
        "> **Figure.** *Lead.* Aggregate bars.\n"
    )
    res = verify_task_body.check_v4_result_figure_cardinality(body)
    assert res.passed is True
    assert res.is_warn is False
    assert "no unpaired multi-figure section" in res.detail


def test_check49_three_figures_no_evidence_warns():
    """Row 5: three inline figures with no pair evidence WARN with
    count=3."""
    body = _v4_minimal_results_body(
        "### Three analyses in one\n\n"
        "Three separate reads bundled into one section.\n\n"
        "![First read](https://x/figures/issue_9/read_one.png)\n\n"
        "![Second read](https://x/figures/issue_9/read_two.png)\n\n"
        "![Third read](https://x/figures/issue_9/read_three.png)\n"
    )
    res = verify_task_body.check_v4_result_figure_cardinality(body)
    assert res.passed is True
    assert res.is_warn is True
    assert "3 figures" in res.detail
    assert "read_one.png" in res.detail
    assert "read_three.png" in res.detail


def test_check49_fenced_and_details_figures_not_counted():
    """Row 6: figures living only inside a fenced code block or a
    `<details>` example block are NOT counted (`_prose_layer`
    convention) — one real figure + two quoted embeds stay conforming."""
    fenced = _v4_minimal_results_body(
        "### Skeleton example\n\n"
        "Real figure plus a quoted skeleton.\n\n"
        "![Real figure](https://x/figures/issue_9/real.png)\n\n"
        "```markdown\n"
        "![Quoted embed](https://x/figures/issue_9/quoted_a.png)\n"
        "![Quoted embed](https://x/figures/issue_9/quoted_b.png)\n"
        "```\n"
    )
    res = verify_task_body.check_v4_result_figure_cardinality(fenced)
    assert res.passed is True
    assert res.is_warn is False
    collapsed = _v4_minimal_results_body(
        "### Collapsed views\n\n"
        "Real figure plus collapsed extras.\n\n"
        "![Real figure](https://x/figures/issue_9/real.png)\n\n"
        "<details>\n<summary>extra views</summary>\n\n"
        "![Extra view](https://x/figures/issue_9/extra_a.png)\n\n"
        "![Extra view](https://x/figures/issue_9/extra_b.png)\n\n"
        "</details>\n"
    )
    res2 = verify_task_body.check_v4_result_figure_cardinality(collapsed)
    assert res2.passed is True
    assert res2.is_warn is False


def test_check49_skips_non_v4_and_missing_results():
    """Row 7 (forward-only): a v3-sentinel body with two figures under one
    `###` PASSes vacuously, as do legacy bodies and a v4 body with no
    `## Results` H2."""
    v3_two_figs = (
        "# T (LOW confidence)\n\n<!-- clean-result-v3 -->\n\n## Findings\n\n"
        "### R\n\n![a](https://x/figures/issue_9/a.png)\n\n"
        "![b](https://x/figures/issue_9/b.png)\n"
    )
    res = verify_task_body.check_v4_result_figure_cardinality(v3_two_figs)
    assert res.passed is True
    assert res.is_warn is False
    assert "skipped — not a v4 body" in res.detail
    res_legacy = verify_task_body.check_v4_result_figure_cardinality(GOOD_BODY)
    assert res_legacy.passed is True
    assert res_legacy.is_warn is False
    no_results = "# T (LOW confidence)\n\n<!-- clean-result-v4 -->\n\n## Takeaways\n\n- x\n"
    res_nores = verify_task_body.check_v4_result_figure_cardinality(no_results)
    assert res_nores.passed is True
    assert res_nores.is_warn is False
    assert "## Results missing" in res_nores.detail


def test_check49_warn_never_flips_verdict_and_rides_checks():
    """Row 8 + registration: the WARN rides the body-only CHECKS dispatch
    (`verify_text` emits it) with `passed=True`, so the aggregate verdict
    (`ok == all(r.passed)`) can never flip on this check."""
    assert verify_task_body.check_v4_result_figure_cardinality in verify_task_body.CHECKS
    body = _v4_minimal_results_body(_ISSUE1769_FU1_DOSE_LADDER_BLOCK)
    ok, results = verify_task_body.verify_text(body)
    r49 = next(r for r in results if r.name == _CHECK49_NAME)
    assert r49.is_warn is True
    assert r49.passed is True
    assert ok == all(r.passed for r in results)
    assert ok == all(r.passed for r in results if r.name != _CHECK49_NAME)


def test_check49_one_figure_each_across_sections_passes():
    """Row 9: cardinality is per-SECTION, not per-body — two `###`
    sections with one figure each draw no WARN."""
    body = _v4_minimal_results_body(
        "### First result\n\n"
        "First read.\n\n"
        "![First figure](https://x/figures/issue_9/first.png)\n\n"
        "### Second result\n\n"
        "Second read.\n\n"
        "![Second figure](https://x/figures/issue_9/second.png)\n"
    )
    res = verify_task_body.check_v4_result_figure_cardinality(body)
    assert res.passed is True
    assert res.is_warn is False
    assert "all 2" in res.detail


def test_check49_prose_only_pair_vocab_still_warns():
    """Row 10 (pins the round-1 Must-Fix scoping): declared-pair vocabulary
    living ONLY in general section prose — the verbatim real-body line
    "the per-question companion below is the per-unit data behind these
    aggregates" — does NOT silence two unpaired figures; only alt text +
    blockquote caption lines count."""
    body = _v4_minimal_results_body(
        "### Aggregates and extras\n\n"
        "the per-question companion below is the per-unit data behind these aggregates\n\n"
        "![Aggregate bars](https://x/figures/issue_9/agg_bars.png)\n\n"
        "> **Figure.** *Lead.* Pooled bars.\n\n"
        "![Second analysis](https://x/figures/issue_9/extra_analysis.png)\n\n"
        "> **Figure.** *Lead.* A different read entirely.\n"
    )
    res = verify_task_body.check_v4_result_figure_cardinality(body)
    assert res.passed is True
    assert res.is_warn is True
    assert "'Aggregates and extras'" in res.detail
    assert "agg_bars.png" in res.detail
    assert "extra_analysis.png" in res.detail


# ─── Check 50: repro-named result dirs clean in working tree (#1989) ─────────

_REPRO_CLEAN_CHECK = "repro-named result dirs clean in working tree"


def _make_repo_with_issue_dir(tmp_path, *, gitignore=None):
    """git-init tmp repo with a committed `eval_results/issue_999/...` tree
    (the check-50 fixture; mirrors the check-29 git-init pattern)."""
    repo = tmp_path / "repo50"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    d = repo / "eval_results" / "issue_999" / "map_augmentation" / "operator_kv"
    d.mkdir(parents=True)
    (d / "tracked.json").write_text("{}\n")
    if gitignore is not None:
        (repo / ".gitignore").write_text(gitignore)
        git("add", ".gitignore")
    git("add", "eval_results")
    git("commit", "-q", "-m", "seed eval results")
    return repo


def _repro_clean_body_v4(footer_line: str) -> str:
    """Minimal v4-sentinel body whose `**Repro:**` footer carries
    ``footer_line`` (check 50 is called directly, so the body only needs
    the sentinel + footer shape `_repro_section_text` reads)."""
    filler = (
        "- The measured effect held across the panel at matched dose; the "
        "companion continuous read kept dynamic range where the rate floored.\n"
        "- Coverage matched the plan denominator; no planned condition was "
        "silently dropped, and the per-unit artifacts back each aggregate.\n"
        "- The control arm stayed at baseline across every probe, so the "
        "contrast is attributable to the manipulated variable alone.\n"
    )
    return (
        "# Title claim (LOW confidence)\n"
        "<!-- clean-result-v4 -->\n\n"
        f"## Takeaways\n\n{filler}\n"
        "---\n\n"
        f"**Repro:** {footer_line}\n\n"
        '**Context:** created 2026-08-01 from the user prompt "x".\n'
    )


def _repro_clean_body_v3(repro_line: str) -> str:
    """Minimal non-v4 body with a `## Reproducibility` H2 carrying
    ``repro_line`` (the v3/v2 branch of `_repro_section_text`)."""
    return (
        "# Title claim (LOW confidence)\n\n"
        "<!-- clean-result-v3 -->\n\n"
        "## Takeaways\n\n- x\n\n"
        "## Reproducibility\n\n"
        f"- **Artifacts:** {repro_line}\n"
    )


def test_check50_registered():
    """House CHECKS-membership pin: the check dispatches via verify_text."""
    assert verify_task_body.check_repro_artifacts_clean in verify_task_body.CHECKS


def test_check50_extraction_reduces_and_collapses():
    """`_repro_eval_results_dirs` unit test: trailing-slash strip, child-file
    extension drop, glob + brace truncation, `ood_` root, mid-word lookbehind
    rejection, and parent-subsumes-child collapse."""
    text = (
        "Per-cell artifacts: `eval_results/issue_999/fits/` (216 JSONs), "
        "`eval_results/issue_999/fits/summary.json`, "
        "`eval_results/issue_999/ckpt/{summary,curves}.json`, "
        "`eval_results/issue_999/percell/*.json`, "
        "plus `ood_eval_results/issue_42/probe/` and my_eval_results/issue_7/x. "
        "A bare eval_results mention with no issue dir never enters."
    )
    dirs = verify_task_body._repro_eval_results_dirs(text)
    assert dirs == {
        "eval_results/issue_999/fits",
        "eval_results/issue_999/ckpt",
        "eval_results/issue_999/percell",
        "ood_eval_results/issue_42/probe",
    }
    # Parent-subsumes-child: a referenced ancestor absorbs its children.
    collapsed = verify_task_body._repro_eval_results_dirs(
        "`eval_results/issue_999/` and `eval_results/issue_999/fits/deep/`"
    )
    assert collapsed == {"eval_results/issue_999"}


def test_check50_untracked_file_warns(tmp_path, monkeypatch):
    """Criterion (a): an untracked file under a footer-named dir — in a NEW
    subdir, pinning the path-scoped `-u` (default untracked-files=normal
    would collapse it to one `?? dir/` entry) — draws the WARN naming the
    entry; verify_text dispatches the same result."""
    repo = _make_repo_with_issue_dir(tmp_path)
    stray = repo / "eval_results" / "issue_999" / "map_augmentation" / "fresh" / "new_cell.json"
    stray.parent.mkdir()
    stray.write_text("{}\n")
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _repro_clean_body_v4(
        "results in `eval_results/issue_999/map_augmentation/` (24 cell JSONs)."
    )
    r = verify_task_body.check_repro_artifacts_clean(body)
    assert r.passed is True
    assert r.is_warn is True
    assert "untracked" in r.detail
    assert "new_cell.json" in r.detail  # the -u pin: file named, not `fresh/`
    assert "#1768" in r.detail  # the recovery line names the incident class
    _ok, results = verify_task_body.verify_text(body)
    r2 = _results_by_name(results)[_REPRO_CLEAN_CHECK]
    assert r2.is_warn is True
    assert r2.passed is True  # WARN never flips this check's own verdict


def test_check50_modified_tracked_file_warns(tmp_path, monkeypatch):
    """Criterion (b): a modified (non-`??` porcelain XY) tracked file under
    the named dir draws the WARN with the modified classification."""
    repo = _make_repo_with_issue_dir(tmp_path)
    (
        repo / "eval_results" / "issue_999" / "map_augmentation" / "operator_kv" / "tracked.json"
    ).write_text('{"v": 2}\n')
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _repro_clean_body_v4("results in `eval_results/issue_999/map_augmentation/`.")
    r = verify_task_body.check_repro_artifacts_clean(body)
    assert r.passed is True
    assert r.is_warn is True
    assert "modified" in r.detail
    assert "tracked.json" in r.detail


def test_check50_clean_dir_passes(tmp_path, monkeypatch):
    """Criterion (c): a fully-committed named dir → clean PASS, no WARN."""
    repo = _make_repo_with_issue_dir(tmp_path)
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _repro_clean_body_v4("results in `eval_results/issue_999/map_augmentation/`.")
    r = verify_task_body.check_repro_artifacts_clean(body)
    assert r.passed is True
    assert r.is_warn is False
    assert "clean in working tree" in r.detail


def test_check50_gitignored_untracked_passes(tmp_path, monkeypatch):
    """Criterion (d): a gitignored untracked file (the repo-wide `*.npz`
    convention) is EXCLUDED by default porcelain (no `--ignored`) → PASS."""
    repo = _make_repo_with_issue_dir(tmp_path, gitignore="*.npz\n")
    (
        repo / "eval_results" / "issue_999" / "map_augmentation" / "operator_kv" / "cells.npz"
    ).write_bytes(b"\x00fake npz")
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _repro_clean_body_v4("results in `eval_results/issue_999/map_augmentation/`.")
    r = verify_task_body.check_repro_artifacts_clean(body)
    assert r.passed is True
    assert r.is_warn is False
    assert "clean in working tree" in r.detail


def test_check50_fenced_only_path_vacuous(tmp_path, monkeypatch):
    """Criterion (e): a path living ONLY inside a fenced block of the footer
    is illustrative — vacuous PASS even with dirt present in the repo."""
    repo = _make_repo_with_issue_dir(tmp_path)
    (repo / "eval_results" / "issue_999" / "map_augmentation" / "stray.json").write_text("{}\n")
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _repro_clean_body_v4(
        "rerun via:\n\n```\nls eval_results/issue_999/map_augmentation/\n```\n"
    )
    r = verify_task_body.check_repro_artifacts_clean(body)
    assert r.passed is True
    assert r.is_warn is False
    assert "no repro-named eval_results dirs" in r.detail


def test_check50_no_eval_results_tokens_vacuous():
    """Criterion (f): a footer naming only HF URLs (the deliberate scope-out)
    → vacuous PASS, no git probes needed."""
    body = _repro_clean_body_v4(
        "stores at [x @ abc](https://huggingface.co/datasets/o/r/tree/abc123/prefix)."
    )
    r = verify_task_body.check_repro_artifacts_clean(body)
    assert r.passed is True
    assert r.is_warn is False
    assert "no repro-named eval_results dirs" in r.detail


def test_check50_probe_failure_skips_never_warns(tmp_path, monkeypatch):
    """Criterion (g): a raising git runner degrades the dir to the per-dir
    'probe failure; not assessed' skip note — never a WARN, even with dirt
    present that WOULD warn on a healthy probe."""
    repo = _make_repo_with_issue_dir(tmp_path)
    (repo / "eval_results" / "issue_999" / "map_augmentation" / "stray.json").write_text("{}\n")
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)

    def raising_run(cmd, *args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(verify_task_body.subprocess, "run", raising_run)
    body = _repro_clean_body_v4("results in `eval_results/issue_999/map_augmentation/`.")
    r = verify_task_body.check_repro_artifacts_clean(body)
    assert r.passed is True
    assert r.is_warn is False
    assert "probe failure; not assessed" in r.detail


def test_check50_non_git_dir_degrades_to_skip(tmp_path, monkeypatch):
    """Criterion (g) sibling (check-29 house variant): repo root pointed at
    a plain non-git dir (`git status` rc != 0) → skip note, no WARN, and no
    exception."""
    plain = tmp_path / "notarepo"
    plain.mkdir()
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: plain)
    body = _repro_clean_body_v4("results in `eval_results/issue_999/map_augmentation/`.")
    r = verify_task_body.check_repro_artifacts_clean(body)
    assert r.passed is True
    assert r.is_warn is False
    assert "probe failure; not assessed" in r.detail


def test_check50_repo_unresolved_skips(monkeypatch):
    """`_resolve_repo_root` → None (running outside the repo): skip-PASS."""
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: None)
    body = _repro_clean_body_v4("results in `eval_results/issue_999/map_augmentation/`.")
    r = verify_task_body.check_repro_artifacts_clean(body)
    assert r.passed is True
    assert r.is_warn is False
    assert r.detail.startswith("skipped")


def test_check50_v3_reproducibility_h2_same_behavior(tmp_path, monkeypatch):
    """Criterion (h): a v3 `## Reproducibility` H2 body routes through the
    same `_repro_section_text` branch — untracked dirt WARNs identically."""
    repo = _make_repo_with_issue_dir(tmp_path)
    (repo / "eval_results" / "issue_999" / "map_augmentation" / "stray.json").write_text("{}\n")
    monkeypatch.setattr(verify_task_body, "_resolve_repo_root", lambda: repo)
    body = _repro_clean_body_v3("results in `eval_results/issue_999/map_augmentation/`.")
    r = verify_task_body.check_repro_artifacts_clean(body)
    assert r.passed is True
    assert r.is_warn is True
    assert "untracked" in r.detail
    assert "stray.json" in r.detail


# ─── Check 51 (#2017; incident #1947): dropped-at-gate condition placement ──


_C51_DROP_SENTENCE = (
    "A third planned behavior (sycophancy) was dropped at the datagen yield gate — "
    "232 judge-accepted positives against the 240 floor after one retry tranche — "
    "removing 16 single-visit cells and 2 of the 4 planned repeat controls."
)

_C51_DESIGN_ANCHOR = (
    "- **Design:** 3 seeds; baseline vs tulu-25 on benchmark Z. "
    "The single manipulated variable is the data mix."
)
_C51_TAKEAWAYS_ANCHOR = "- Caveat that binds interpretation: single model family, three seeds only."
_C51_RESULT_PROSE_ANCHOR = (
    "The 17-pt lift holds at every seed; "
    "the smallest within-condition gap between seeds is 1.2 pts."
)


def _c51_body(
    *, drop_sentence=_C51_DROP_SENTENCE, takeaways_extra="", results_extra="", leading=False
):
    """`_V4_GOOD_BODY` with a dropped-at-gate declaration spliced into the
    Methodology `**Design:**` bullet (the #1947 shape, copied near-verbatim
    from #1947's live body), plus optional Takeaways-bullet / result-prose
    placement lines. With ``leading=True`` the declaration is the FIRST
    sentence of the bullet, so the subject clause contains the bold
    `**Design:**` slot label (the round-2 Major reproduction shape).
    Asserts every anchor actually replaced."""
    assert _C51_DESIGN_ANCHOR in _V4_GOOD_BODY
    if leading:
        design = (
            f"- **Design:** {drop_sentence} 3 seeds; baseline vs tulu-25 on "
            "benchmark Z. The single manipulated variable is the data mix."
        )
    else:
        design = (
            "- **Design:** 3 seeds; baseline vs tulu-25 on benchmark Z. "
            f"{drop_sentence} The single manipulated variable is the data mix."
        )
    body = _V4_GOOD_BODY.replace(_C51_DESIGN_ANCHOR, design)
    if takeaways_extra:
        assert _C51_TAKEAWAYS_ANCHOR in body
        body = body.replace(_C51_TAKEAWAYS_ANCHOR, _C51_TAKEAWAYS_ANCHOR + "\n" + takeaways_extra)
    if results_extra:
        assert _C51_RESULT_PROSE_ANCHOR in body
        body = body.replace(
            _C51_RESULT_PROSE_ANCHOR, _C51_RESULT_PROSE_ANCHOR + " " + results_extra
        )
    return body


def test_check51_1947_shape_fails_both_placements():
    """Acceptance criterion 2 — the #1947 shape: a Methodology `**Design:**`
    dropped-at-gate declaration with the condition name absent from
    ## Takeaways AND every `### <result>` block → FAIL naming the declaring
    sentence and BOTH missing placements."""
    r = verify_task_body.check_v4_dropped_condition_placement(_c51_body())
    assert r.passed is False
    assert "sycophancy" in r.detail
    assert "was dropped at the datagen yield gate" in r.detail  # declaring-sentence quote
    assert "absent from ## Takeaways AND from every `### <result>` block" in r.detail


def test_check51_pass_when_named_in_takeaways_and_result():
    """Acceptance criterion 3: name present in ## Takeaways AND ≥1
    `### <result>` block → PASS."""
    body = _c51_body(
        takeaways_extra=(
            "- Sycophancy was dropped at the datagen yield gate; every denominator "
            "below uses the realized 34 arms."
        ),
        results_extra=(
            "No sycophancy cell appears in this figure — that behavior missed its "
            "datagen yield floor."
        ),
    )
    r = verify_task_body.check_v4_dropped_condition_placement(body)
    assert r.passed is True
    assert r.is_warn is False
    assert "every extracted condition name" in r.detail


def test_check51_fail_when_named_in_takeaways_only():
    """Acceptance criterion 4a: name in ## Takeaways only → FAIL naming the
    missing result placement (and NOT the satisfied Takeaways one)."""
    body = _c51_body(takeaways_extra="- Sycophancy was dropped at the datagen yield gate.")
    r = verify_task_body.check_v4_dropped_condition_placement(body)
    assert r.passed is False
    assert "is absent from every `### <result>` block" in r.detail
    assert "absent from ## Takeaways" not in r.detail


def test_check51_fail_when_named_in_result_only():
    """Acceptance criterion 4b (symmetric): name in a `### <result>` block
    only → FAIL naming the missing ## Takeaways placement."""
    body = _c51_body(
        results_extra="No sycophancy cell appears — that behavior missed its yield floor."
    )
    r = verify_task_body.check_v4_dropped_condition_placement(body)
    assert r.passed is False
    assert "is absent from ## Takeaways" in r.detail
    assert "absent from every" not in r.detail


def test_check51_warn_when_no_extractable_name():
    """Acceptance criterion 5: a dropped-at-gate declaration whose subject
    clause carries no parenthetical and no wrapped token → WARN (surface,
    never block on a failed heuristic extraction)."""
    body = _c51_body(
        drop_sentence=(
            "Sixteen planned single-visit cells were dropped at the datagen "
            "yield gate after one retry tranche."
        )
    )
    r = verify_task_body.check_v4_dropped_condition_placement(body)
    assert r.passed is True
    assert r.is_warn is True
    assert "no extractable condition name" in r.detail
    assert "were dropped at the datagen yield gate" in r.detail


def test_check51_backtick_subject_extraction():
    """Priority-2 extraction: a backtick-wrapped token in the subject clause
    (`harmful_compliance`) is extracted and placement-checked."""
    body = _c51_body(
        drop_sentence=(
            "The `harmful_compliance` behavior was dropped at the datagen "
            "yield gate after one retry tranche."
        )
    )
    r = verify_task_body.check_v4_dropped_condition_placement(body)
    assert r.passed is False
    # The extracted NAME stays intact (interior `_` is never stripped —
    # a mangled name would silently defeat the placement match).
    assert "dropped condition `harmful_compliance`" in r.detail
    assert "absent from ## Takeaways AND from every `### <result>` block" in r.detail


def test_check51_snake_case_name_placement_match():
    """A snake_case extracted name matches placements under flexible
    separators — `harmful_compliance` in Methodology, "harmful compliance"
    (space-separated) in ## Takeaways, snake_case in the result prose →
    PASS."""
    body = _c51_body(
        drop_sentence=(
            "The `harmful_compliance` behavior was dropped at the datagen "
            "yield gate after one retry tranche."
        ),
        takeaways_extra=(
            "- The harmful compliance behavior missed its datagen yield floor; "
            "denominators below use the realized arms."
        ),
        results_extra="No harmful_compliance cell appears in this figure.",
    )
    r = verify_task_body.check_v4_dropped_condition_placement(body)
    assert r.passed is True
    assert r.is_warn is False


def test_check51_v3_body_vacuous_pass():
    """Acceptance criterion 6: the SAME Methodology declaration in a
    v3-sentinel body → vacuous PASS (forward-only; grandfathered shapes are
    never newly hard-FAILed)."""
    body = _c51_body().replace("<!-- clean-result-v4 -->", "<!-- clean-result-v3 -->")
    r = verify_task_body.check_v4_dropped_condition_placement(body)
    assert r.passed is True
    assert r.is_warn is False
    assert "not a v4 body" in r.detail


def test_check51_no_drop_language_passes():
    """Acceptance criterion 7: judge drop-rate prose ("were dropped from both
    arms", no at/by-gate tail) plus a gate/floor mention in a LATER sentence
    never matches → PASS."""
    body = _c51_body(
        drop_sentence=(
            "Malformed judge returns were dropped from both arms and excluded "
            "from the per-arm aggregates. The 240-row yield floor is unchanged."
        )
    )
    r = verify_task_body.check_v4_dropped_condition_placement(body)
    assert r.passed is True
    assert r.is_warn is False
    assert "no dropped-at-gate declaration" in r.detail


def test_check51_blockquote_and_fence_immune():
    """Acceptance criterion 7 (prose layer): a dropped-at-gate declaration
    living ONLY in a Methodology blockquote caption or fenced code block is
    never detected (`_prose_layer` strips fences/<details>; the check strips
    blockquote lines itself — `_prose_layer` does NOT, #2017 plan note)."""
    extra = (
        "> **Note.** A planned behavior (sycophancy) was dropped at the datagen yield gate.\n\n"
        "```\nA planned behavior (sycophancy) was dropped at the datagen yield gate.\n```\n\n"
    )
    assert "- **Evaluation:**" in _V4_GOOD_BODY
    body = _V4_GOOD_BODY.replace("- **Evaluation:**", extra + "- **Evaluation:**", 1)
    r = verify_task_body.check_v4_dropped_condition_placement(body)
    assert r.passed is True
    assert r.is_warn is False
    assert "no dropped-at-gate declaration" in r.detail


def test_check51_slot_label_never_extracted_on_compliant_body():
    """Round-2 Major (code review): a drop declaration LEADING a bold-slot
    Methodology bullet must never FAIL a COMPLIANT body naming the slot
    label (`Design:`). The colon-ended candidate is rejected; with no
    surviving heuristic ("sycophancy" is unwrapped) the check WARNs
    (criterion 5) — NOT-FAIL is the pin."""
    assert (
        verify_task_body._extract_dropped_condition_name(
            "- **Design:** The planned sycophancy arm "
        )
        is None
    )
    body = _c51_body(
        drop_sentence="The planned sycophancy arm was dropped at the datagen yield gate.",
        leading=True,
        takeaways_extra=(
            "- Sycophancy was dropped at the datagen yield gate; denominators "
            "below use the realized arms."
        ),
        results_extra="No sycophancy cell appears — that behavior missed its yield floor.",
    )
    r = verify_task_body.check_v4_dropped_condition_placement(body)
    assert r.passed is True  # never a blocking FAIL on a junk extraction
    assert "dropped condition" not in r.detail  # the FAIL-detail shape names no token
    assert r.is_warn is True
    assert "no extractable condition name" in r.detail


def test_check51_slot_label_absent_placements_warn_not_fail():
    """Round-2 Major, placements-absent arm: the same leading bold-slot
    declaration with the real name absent from BOTH placements yields
    whatever the surviving extraction supports — here none survives, so
    WARN; it must never FAIL naming a colon-ended slot label."""
    body = _c51_body(
        drop_sentence="The planned sycophancy arm was dropped at the datagen yield gate.",
        leading=True,
    )
    r = verify_task_body.check_v4_dropped_condition_placement(body)
    assert r.passed is True
    assert r.is_warn is True
    assert "dropped condition `Design:`" not in r.detail
    assert "no extractable condition name" in r.detail


def test_check51_numeric_parenthetical_rejected():
    """Round-2 Major sibling: a numeric stat parenthetical adjacent to the
    verb group (`(232 of 240)`) is rejected as a name candidate; extraction
    falls through to the wrapped-token priority (`harmful_compliance` — a
    correct-name FAIL here), never a FAIL naming the numeric span."""
    assert (
        verify_task_body._extract_dropped_condition_name("The sycophancy arm (232 of 240) ") is None
    )
    body = _c51_body(
        drop_sentence=(
            "The `harmful_compliance` behavior (232 of 240) was dropped at the datagen yield gate."
        )
    )
    r = verify_task_body.check_v4_dropped_condition_placement(body)
    assert r.passed is False
    assert "dropped condition `harmful_compliance`" in r.detail
    assert "232 of 240" not in r.detail.split("(declared:")[0]  # numeric token never the name


def test_check51_registered():
    """Critic refinement 3: the check rides `CHECKS` (v4-gated block) — a
    forgotten CHECKS append must not ship green (house membership-assert
    pattern, cf. test_check45_registered / test_check46_registered)."""
    assert verify_task_body.check_v4_dropped_condition_placement in verify_task_body.CHECKS
