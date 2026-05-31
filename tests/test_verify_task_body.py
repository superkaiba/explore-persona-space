"""Tests for scripts/verify_task_body.py — eleven mechanical checks for the
markdown clean-result spec.

Each test feeds a synthetic body string into verify_text() and asserts
which checks pass / fail.
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


# ─── Canonical body (passes all 12 checks) ─────────────────────────────────

GOOD_BODY = """\
---
title: Toy clean-result for verifier tests
kind: experiment
goal: Characterize how cross-persona leakage scales with seed and benchmark
---
# Some claim about persona leakage (MODERATE confidence)

## Human TL;DR

**Headline.** *placeholder*

**Takeaways.** *placeholder*

**How this updates me.** *placeholder*

## Goal

Characterize how cross-persona leakage scales with seed and benchmark.

## TL;DR
- **Motivation:** I wanted to test whether X drives Y.
- **What I ran:** Trained 3 seeds at lr=3e-5, evaluated on benchmark Z.
- **Results:** Effect is present at p < 0.01 ([figure below](#figure)).
- **Next steps:** Replicate at 70B, run the partial-correlation control.

## Figure
![hero plot](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_999/hero.png)

*Caption: Mean cross-persona leakage with 95% CI bands across three training seeds and four benchmark conditions.*

## Details

Free-form description here.

These excerpts are cherry-picked for illustration; the full per-row raw-completion data is at [raw completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc123def/raw_completions/run.jsonl).

```text
User: What is the capital of France?
Assistant: The capital of France is Paris. It has a population of about 2.2 million people in the city proper and 12 million in the metropolitan area, and serves as the cultural, economic, and political center of the country, hosting many world-famous landmarks such as the Eiffel Tower and the Louvre museum.
```

Confidence: MODERATE — three independent seeds, but only one model family.

## Reproducibility

**Artifacts:**
- Model: [hf-hub](https://huggingface.co/superkaiba1/explore-persona-space/tree/abc123def)
- WandB run: [link](https://wandb.ai/superkaiba/eps/runs/abc12345)

**Compute:** 1× H100, 47 min.

**Code:** entry script @ commit [0123456789abcdef](https://github.com/superkaiba/explore-persona-space/blob/0123456789abcdef/scripts/run.py).
"""


def _results_by_name(results):
    return {r.name: r for r in results}


# ─── Existing checks 1-6 (regression coverage from original 6-check verifier) ───


def test_good_body_passes_all():
    ok, results = verify_task_body.verify_text(GOOD_BODY)
    assert ok, [r.render() for r in results if not r.passed]
    assert all(r.passed for r in results)
    # 17 body-only checks (CHECKS — body-nonstub check 0 prepended,
    # no-duplicate-frontmatter check 0b added 2026-05-26 alongside the
    # set-body strip fix, plus the 12 structural checks incl. Figure URL
    # resolvable, plus check_details_narrative_flow added 2026-05-27
    # alongside the LessWrong-style narrative shift, plus
    # check_figure_h2_is_deprecated added 2026-05-27 alongside the
    # inline-figures-under-Results-sub-bullets prescriptive default,
    # plus check_mdx_safe_urls added 2026-05-28 after task #382's six
    # `<https://...>` autolinks broke the dashboard MDX renderer — extended
    # 2026-05-28 with the table-cell `<|` class (task #399) and an
    # authoritative real-parse backstop that shells out to the dashboard's
    # mdx_parse_check.mjs) + 1
    # Goal-of-experiment soft check appended by verify_text (it needs the
    # frontmatter, not just the body — as of 2026-05-26 it checks only
    # the frontmatter `goal:` field; the body-side `## Goal` H2 is
    # intentionally not checked because clean-result bodies drop it).
    # Count: verify_text prepends check 0 (body-nonstub) + check 0b
    # (no-duplicate-frontmatter), runs the remaining 17 CHECKS[1:], then
    # appends the Goal soft check → 20 results (len(CHECKS) == 18, after the
    # 2026-05-31 Reproducibility committed-at-sha check was added).
    assert len(results) == 20


def test_missing_confidence_tag():
    body = GOOD_BODY.replace(" (MODERATE confidence)", "")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["title confidence tag"].passed


def test_wrong_section_order():
    body = GOOD_BODY.replace("## TL;DR", "## TempPlaceholder")
    body = body.replace("## Details", "## TL;DR")
    body = body.replace("## TempPlaceholder", "## Details")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["four required H2 sections in order"].passed
    assert "order" in by_name["four required H2 sections in order"].detail.lower()


def test_missing_section():
    body = GOOD_BODY.replace("## Reproducibility", "## Repro")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["four required H2 sections in order"].passed
    assert "Reproducibility" in by_name["four required H2 sections in order"].detail


def test_missing_tldr_labels():
    """A REQUIRED TL;DR label (`What I ran`) is renamed → FAIL."""
    body = GOOD_BODY.replace("- **What I ran:**", "- **Stuff I did:**")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["TL;DR bullets carry the three required labels"].passed
    assert "What I ran" in by_name["TL;DR bullets carry the three required labels"].detail


def test_missing_next_steps_passes():
    """`Next steps` is OPTIONAL as of 2026-05-26 — a body that omits it PASSes.

    Drops the entire `Next steps` bullet line from the GOOD_BODY TL;DR. All
    14 checks (incl. the TL;DR-labels check and the soft Goal-of-experiment
    INFO) must still PASS — there is no FAIL for a missing Next-steps bullet.
    """
    body = GOOD_BODY.replace(
        "- **Next steps:** Replicate at 70B, run the partial-correlation control.\n",
        "",
    )
    ok, results = verify_task_body.verify_text(body)
    assert ok, [r.render() for r in results if not r.passed]
    by_name = _results_by_name(results)
    assert by_name["TL;DR bullets carry the three required labels"].passed


def test_next_steps_present_still_passes():
    """A body that DOES include `Next steps` continues to PASS (regression).

    The optional rule is permissive in both directions — bodies with the
    bullet still pass; bodies without it now also pass. GOOD_BODY itself
    carries the bullet, so this is essentially asserting `test_good_body_passes_all`'s
    invariant from the TL;DR-labels angle.
    """
    ok, results = verify_task_body.verify_text(GOOD_BODY)
    assert ok
    by_name = _results_by_name(results)
    assert by_name["TL;DR bullets carry the three required labels"].passed


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
    assert not by_name["Details confidence sentence matches title"].passed


def test_short_caption():
    body = GOOD_BODY.replace(
        "*Caption: Mean cross-persona leakage with 95% CI bands across three training seeds and four benchmark conditions.*",
        "*Caption: too short.*",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Figure caption ≥10 words"].passed


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
#
# Regression: task #389 (2026-05-26). The library-level set_body() now
# strips caller-supplied leading frontmatter before write, but this
# verifier check is the belt-and-suspenders gate against any future
# regression (manual editing, alternative write path, third-party tool)
# that lets a duplicate frontmatter block land on disk. The dashboard
# would otherwise render the second block as literal YAML at the top of
# the visible body.


def test_duplicate_frontmatter_fails():
    """A body that has two consecutive `---...---` blocks at the very top
    FAILs the no-duplicate-frontmatter check — this is the exact shape
    `set_body` would have produced before the strip fix when a caller
    passed a complete markdown document (frontmatter + body)."""
    # Inject a second `---...---` block immediately after the canonical
    # frontmatter close, before the H1 — mirrors the task #389 incident
    # where analyzer-drafted body files carried their own frontmatter.
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
    render. The no-duplicate-frontmatter check counts only CONSECUTIVE
    leading blocks (the shape the strip fix targets), so this case
    PASSes the check. The body may still fail OTHER checks (the
    horizontal rule appears as a stray `---` line), but the duplicate-
    frontmatter check itself doesn't fire.
    """
    fm_end = GOOD_BODY.index("---\n", 4) + 4
    blank_separated = (
        GOOD_BODY[:fm_end] + "\n\n" + "---\nstale: caller frontmatter\n---\n" + GOOD_BODY[fm_end:]
    )
    _, results = verify_task_body.verify_text(blank_separated)
    by_name = _results_by_name(results)
    # The duplicate-frontmatter check counts only stacked-without-blank-line
    # blocks; a blank line breaks the stack.
    assert by_name["no duplicate frontmatter"].passed


def test_no_duplicate_frontmatter_passes_on_good_body():
    """GOOD_BODY (single canonical frontmatter only) passes the
    duplicate-frontmatter check itself. The body may fail other checks
    (e.g. the recently-added `## Human TL;DR` requirement) but check 0b
    must not be one of them — we assert on the specific check, not on
    overall ok.
    """
    _, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    assert by_name["no duplicate frontmatter"].passed
    assert "1 leading frontmatter block" in by_name["no duplicate frontmatter"].detail


def test_no_duplicate_frontmatter_passes_on_horizontal_rule_inside_body():
    """A `---` horizontal-rule line deep inside the body (not stacked
    at the top) does NOT trip the check — only consecutive leading
    blocks count."""
    body = GOOD_BODY.replace(
        "Free-form description here.\n",
        "Free-form description here.\n\n---\n\nAfter the rule.\n",
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
    # Three stacked blocks all count.
    assert count("---\na: 1\n---\n---\nb: 2\n---\n---\nc: 3\n---\nbody\n") == 3
    # A blank line between blocks breaks the stack — only the first
    # block counts.
    assert count("---\nfoo: 1\n---\n\n---\nbar: 2\n---\nbody\n") == 1
    # Malformed leading block (no closing `\n---\n`) counts as zero.
    assert count("---\nfoo: bar\nno closing here\n# H1\n") == 0


# ─── Check 4: hero image present in `## Figure` ───────────────────────────


def test_figure_image_present_pass():
    """Happy path for check 4 already exercised by `test_good_body_passes_all`,
    but also assert the check name and detail directly."""
    _ok, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    assert by_name["hero image present"].passed
    assert "1 image" in by_name["hero image present"].detail


def test_figure_missing_image_fails():
    """Strip the `![hero plot](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_999/hero.png)` image line; the check fails."""
    body = GOOD_BODY.replace(
        "![hero plot](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_999/hero.png)\n",
        "",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["hero image present"].passed
    assert "no `![alt](path)` image" in by_name["hero image present"].detail


# ─── Check 4b: figure URL must be dashboard-resolvable ────────────────────
#
# Regression coverage for the task #365 incident (2026-05-22): the body
# referenced `artifacts/hero.png` (relative), which the EPS dashboard does
# not serve for binary PNG/PDF files, so the figure rendered broken.


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
    """`figures/issue_N/hero.png` (relative, no SHA) also fails — the
    dashboard cannot fetch it. Operator must use the raw.githubusercontent.com
    permalink form."""
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
    # Body should still pass overall (no other regression introduced).
    assert ok


def test_figure_alt_text_with_brackets_parses():
    """Alt text may contain literal `[brackets]` (e.g. marker names like
    `[ZLT]`) — the image regex must still match and the URL extracts cleanly."""
    body = GOOD_BODY.replace(
        "![hero plot]",
        "![Best [ZLT] firing across cells]",
    )
    ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name["hero image present"].passed
    assert by_name["Figure URL resolvable"].passed
    assert ok


# ─── Check 12: `## Figure` H2 deprecation (WARN-only) ─────────────────────
#
# The new analyzer default (2026-05-27) is to inline figures under TL;DR
# Results sub-bullets (one-takeaway-one-figure, Lens 9). The `## Figure` H2
# is preserved as a legacy/grandfathered pattern: bodies that carry it stay
# valid (no FAIL), but a WARN surfaces so the analyzer is nudged toward the
# inline pattern for new bodies.


_INLINE_FIGURE_BODY = """\
---
title: Toy inline-figure clean-result for deprecation check
kind: experiment
goal: Check whether the inline-figure pattern passes without `## Figure` H2
---
# Inline-figure body passes the deprecation check (MODERATE confidence)

## Human TL;DR

**Headline.** *placeholder*

**Takeaways.** *placeholder*

**How this updates me.** *placeholder*

## TL;DR
- **Motivation:** I wanted to test the inline-figure pattern.
    ![inline hero plot showing per-condition leakage means with 95% CI bands across three training seeds](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_999/hero.png)
- **What I ran:** Trained 3 seeds at lr=3e-5, evaluated on benchmark Z.
- **Results:** Effect is present at p < 0.01 (see inline figure above).

## Details

Free-form description here.

These excerpts are cherry-picked for illustration; the full per-row raw-completion data is at [raw completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc123def/raw_completions/run.jsonl).

```text
User: What is the capital of France?
Assistant: The capital of France is Paris. It has a population of about 2.2 million people in the city proper and 12 million in the metropolitan area, and serves as the cultural, economic, and political center of the country, hosting many world-famous landmarks such as the Eiffel Tower and the Louvre museum.
```

Confidence: MODERATE — three independent seeds, but only one model family.

## Reproducibility

**Artifacts:**
- Model: [hf-hub](https://huggingface.co/superkaiba1/explore-persona-space/tree/abc123def)
- WandB run: [link](https://wandb.ai/superkaiba/eps/runs/abc12345)

**Compute:** 1× H100, 47 min.

**Code:** entry script @ commit [0123456789abcdef](https://github.com/superkaiba/explore-persona-space/blob/0123456789abcdef/scripts/run.py).
"""


def test_figure_h2_absent_no_warn():
    """Body without `## Figure` H2 (inline-figure pattern, the new default)
    PASSes the deprecation check WITHOUT a WARN."""
    _ok, results = verify_task_body.verify_text(_INLINE_FIGURE_BODY)
    by_name = _results_by_name(results)
    check_name = "`## Figure` H2 is deprecated for new write-ups"
    assert check_name in by_name, [r.name for r in results]
    r = by_name[check_name]
    assert r.passed
    assert not r.is_warn
    assert "Lens 9 default" in r.detail


def test_figure_h2_present_warns_not_fails():
    """Body WITH `## Figure` H2 (legacy hero pattern) still PASSes the
    overall verifier but the deprecation check surfaces a WARN — never a
    FAIL. Legacy bodies pre-2026-05-27 stay promotable; the WARN exists
    to nudge new bodies toward the inline pattern."""
    ok, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    check_name = "`## Figure` H2 is deprecated for new write-ups"
    assert check_name in by_name, [r.name for r in results]
    r = by_name[check_name]
    # WARN = passed True + is_warn True. The check NEVER returns FAIL on
    # this pattern — legacy bodies must remain promotable.
    assert r.passed
    assert r.is_warn
    assert "## Figure" in r.detail
    # The overall verdict is independent of this WARN (it depends on the
    # other checks); confirm the WARN itself doesn't flip `ok` for a body
    # that would otherwise pass.
    del ok  # GOOD_BODY currently fails for unrelated reasons (missing
    # `## Human TL;DR` in the test fixture); the assertion above proves
    # the WARN semantics regardless of the overall verdict.


# ─── Check 6 extension: ≥20-char confidence rationale ─────────────────────


def test_confidence_rationale_too_short():
    body = GOOD_BODY.replace(
        "Confidence: MODERATE — three independent seeds, but only one model family.",
        "Confidence: MODERATE — short.",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Details confidence sentence matches title"].passed
    assert "rationale after" in by_name["Details confidence sentence matches title"].detail


def test_confidence_line_missing_dash():
    body = GOOD_BODY.replace(
        "Confidence: MODERATE — three independent seeds, but only one model family.",
        "Confidence: MODERATE three independent seeds, but only one model family.",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Details confidence sentence matches title"].passed
    # The looser fallback regex finds `Confidence: MODERATE` and reports
    # the missing dash clause.
    detail = by_name["Details confidence sentence matches title"].detail
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


# ─── Check 9: sentinel scrub (split from old check 4) ─────────────────────


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


CHERRY_BODY_FAIL = """\
---
title: Cherry-picked discipline failing fixture
kind: experiment
goal: Characterize how cross-persona leakage scales with seed and benchmark
---
# Some claim about persona leakage (MODERATE confidence)

## Goal

Characterize how cross-persona leakage scales with seed and benchmark.

## TL;DR
- **Motivation:** I wanted to test whether X drives Y.
- **What I ran:** Trained 3 seeds at lr=3e-5, evaluated on benchmark Z.
- **Results:** Effect is present at p < 0.01.
- **Next steps:** Replicate at 70B.

## Figure
![hero plot](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_999/hero.png)

*Caption: Mean cross-persona leakage with 95% CI bands across three training seeds and four benchmark conditions.*

## Details

Here is a sample model completion. The full data is at [raw completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc123def/raw.jsonl).

```text
User: What is the capital of France?
Assistant: The capital of France is Paris. It has a population of about 2.2 million in the city proper, and serves as the cultural, economic, and political center of the country.
```

Confidence: MODERATE — three independent seeds, but only one model family.

## Reproducibility

**Artifacts:**
- Model: [hf-hub](https://huggingface.co/superkaiba1/explore-persona-space/tree/abc123def)

**Compute:** 1× H100, 47 min.

**Code:** entry script @ commit [0123456789abcdef](https://github.com/superkaiba/explore-persona-space/blob/0123456789abcdef/scripts/run.py).
"""


def test_cherry_picked_missing_disclosure():
    """Sample block in Details, but prelude has no cherry-picked / random
    disclosure → check 10 fails."""
    ok, results = verify_task_body.verify_text(CHERRY_BODY_FAIL)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Cherry-picked label discipline"].passed
    assert "cherry-picked" in by_name["Cherry-picked label discipline"].detail


def test_cherry_picked_random_sample_disclosure_passes():
    """`first 3 of 400 completions` is an accepted random-sample disclosure."""
    body = CHERRY_BODY_FAIL.replace(
        "Here is a sample model completion.",
        "Here are the first 3 of 400 completions in the run.",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name["Cherry-picked label discipline"].passed


def test_cherry_picked_explicit_label_passes():
    """`cherry-picked for illustration` clears the discipline check."""
    body = CHERRY_BODY_FAIL.replace(
        "Here is a sample model completion.",
        "These excerpts are cherry-picked for illustration.",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name["Cherry-picked label discipline"].passed


def test_no_sample_block_skips_cherry_check():
    """A Details section with no fenced sample block PASSes check 10 trivially."""
    body = GOOD_BODY
    # Strip the only sample fence
    body = body.split("```text")[0] + body.split("```\n")[1]
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name["Cherry-picked label discipline"].passed
    assert "no sample-output" in by_name["Cherry-picked label discipline"].detail


# ─── Check 11: qualitative-data link discipline ───────────────────────────


QUAL_BODY_FAIL = """\
---
title: Qualitative-data link failing fixture
kind: experiment
goal: Characterize how cross-persona leakage scales with seed and benchmark
---
# Some claim about persona leakage (MODERATE confidence)

## Human TL;DR

**Headline.** *placeholder*

**Takeaways.** *placeholder*

**How this updates me.** *placeholder*

## Goal

Characterize how cross-persona leakage scales with seed and benchmark.

## TL;DR
- **Motivation:** I wanted to test whether X drives Y.
- **What I ran:** Trained 3 seeds at lr=3e-5, evaluated on benchmark Z.
- **Results:** Effect is present at p < 0.01.
- **Next steps:** Replicate at 70B.

## Figure
![hero plot](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_999/hero.png)

*Caption: Mean cross-persona leakage with 95% CI bands across three training seeds and four benchmark conditions.*

## Details

These excerpts are cherry-picked for illustration. No link to raw data here, just the prose.

```text
User: What is the capital of France?
Assistant: The capital of France is Paris. It has a population of about 2.2 million in the city proper, and serves as the cultural, economic, and political center of the country.
```

Confidence: MODERATE — three independent seeds, but only one model family.

## Reproducibility

**Artifacts:**
- Model: [hf-hub](https://huggingface.co/superkaiba1/explore-persona-space/tree/abc123def)

**Compute:** 1× H100, 47 min.

**Code:** entry script @ commit [0123456789abcdef](https://github.com/superkaiba/explore-persona-space/blob/0123456789abcdef/scripts/run.py).
"""


def test_qualitative_data_link_missing():
    """Sample fenced block but no link/path in the prelude → check 11 FAIL."""
    ok, results = verify_task_body.verify_text(QUAL_BODY_FAIL)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Qualitative-data link"].passed
    assert "lack a qualitative-data link" in by_name["Qualitative-data link"].detail


def test_qualitative_data_link_aggregate_only_fails():
    """Aggregate-only paths (`regression`, `summary`, `.npz`) don't count."""
    body = QUAL_BODY_FAIL.replace(
        "These excerpts are cherry-picked for illustration. No link to raw data here, just the prose.",
        "These excerpts are cherry-picked for illustration. Aggregates at [regression](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc/per_cell_regression.csv).",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Qualitative-data link"].passed
    assert "aggregate-pattern" in by_name["Qualitative-data link"].detail


def test_qualitative_data_link_present_passes():
    """A non-aggregate link in the prelude clears check 11."""
    body = QUAL_BODY_FAIL.replace(
        "These excerpts are cherry-picked for illustration. No link to raw data here, just the prose.",
        "These excerpts are cherry-picked for illustration. Full data at [raw completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/abc123def/raw_completions/run.jsonl).",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name["Qualitative-data link"].passed


def test_qualitative_data_link_backtick_path_passes():
    """A backtick-wrapped path also satisfies the qualitative-data check."""
    body = QUAL_BODY_FAIL.replace(
        "These excerpts are cherry-picked for illustration. No link to raw data here, just the prose.",
        "These excerpts are cherry-picked for illustration. Full data at `eval_results/issue_999/raw_completions/run.jsonl`.",
    )
    _ok, results = verify_task_body.verify_text(body)
    by_name = _results_by_name(results)
    assert by_name["Qualitative-data link"].passed


def test_qualitative_data_link_not_uploaded_warn():
    """An explicit `not uploaded` disclosure downgrades FAIL to WARN (PASS overall)."""
    body = QUAL_BODY_FAIL.replace(
        "These excerpts are cherry-picked for illustration. No link to raw data here, just the prose.",
        "These excerpts are cherry-picked for illustration. Raw completions were not uploaded for this run; follow-up will re-run with raw-completion upload.",
    )
    ok, results = verify_task_body.verify_text(body)
    assert ok
    by_name = _results_by_name(results)
    assert by_name["Qualitative-data link"].passed
    assert by_name["Qualitative-data link"].is_warn
    assert "not uploaded" in by_name["Qualitative-data link"].detail


# ─── Soft Goal-of-experiment check (never FAIL — WARN when missing) ───────


def test_goal_of_experiment_present_passes():
    """Happy path: frontmatter has `goal:`. The body-side `## Goal` H2 is
    intentionally NOT checked here (clean-result bodies drop the visible
    H2 as of 2026-05-26 — the Goal folds into the TL;DR Motivation
    bullet). Already exercised by `test_good_body_passes_all` against
    GOOD_BODY; this test isolates the assertion."""
    _ok, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    r = by_name["Goal-of-experiment field"]
    assert r.passed
    assert r.is_warn is False
    assert "frontmatter goal present" in r.detail


def test_goal_of_experiment_passes_when_h2_absent_but_frontmatter_present():
    """Clean-result bodies drop the `## Goal` H2 but keep the frontmatter
    `goal:` field. The verifier MUST treat this as PASS (no WARN) — that
    is the canonical clean-result shape as of 2026-05-26.

    Regression: previously the verifier WARNed whenever `## Goal` H2 was
    absent. The new canonical shape drops the H2, so the WARN became a
    permanent false positive on every clean-result body. See:
    `.claude/skills/clean-results/iterations.md` § 2026-05-26.
    """
    # Strip just the body-side `## Goal` H2 block; keep the frontmatter.
    body_without_h2 = GOOD_BODY.replace(
        "## Goal\n\nCharacterize how cross-persona leakage scales with seed and benchmark.\n\n",
        "",
    )
    ok, results = verify_task_body.verify_text(body_without_h2)
    assert ok, [r.render() for r in results if not r.passed]
    by_name = _results_by_name(results)
    r = by_name["Goal-of-experiment field"]
    assert r.passed is True
    assert r.is_warn is False
    assert "frontmatter goal present" in r.detail


def test_goal_of_experiment_warns_when_frontmatter_missing():
    """When the frontmatter `goal:` field is missing, the soft check WARNs
    but does NOT FAIL the body. Enforcement is at /issue Step 0c, not
    here. The body-side `## Goal` H2 is intentionally not inspected — it
    legitimately lives only in proposed/planning bodies."""
    # Strip the frontmatter `goal:` line; the body-side H2 is irrelevant.
    body_without_frontmatter_goal = GOOD_BODY.replace(
        "goal: Characterize how cross-persona leakage scales with seed and benchmark\n",
        "",
    )
    ok, results = verify_task_body.verify_text(body_without_frontmatter_goal)
    # Overall should remain PASS because Goal absence is soft.
    assert ok, [r.render() for r in results if not r.passed]
    by_name = _results_by_name(results)
    r = by_name["Goal-of-experiment field"]
    assert r.passed is True  # passed=True, but rendered as WARN
    assert r.is_warn is True
    assert "missing" in r.detail
    assert "frontmatter `goal:`" in r.detail


def test_goal_of_experiment_passes_when_legacy_h2_still_present():
    """Legacy clean-result bodies that still carry a `## Goal` H2
    (pre-2026-05-26 promotions) remain promotable. The verifier MUST NOT
    flag the extra H2 as an error — `find_h2_sections` already tolerates
    H2s outside the four required ones, and the Goal check ignores the
    body entirely. GOOD_BODY itself happens to carry the legacy H2; this
    test just spells out the contract."""
    assert "## Goal" in GOOD_BODY  # GOOD_BODY carries the legacy H2
    ok, results = verify_task_body.verify_text(GOOD_BODY)
    assert ok, [r.render() for r in results if not r.passed]
    by_name = _results_by_name(results)
    # The required-H2-sections check passes despite the extra `## Goal` H2.
    assert by_name["four required H2 sections in order"].passed
    # The Goal check ignores the body and just confirms frontmatter.
    assert by_name["Goal-of-experiment field"].passed
    assert by_name["Goal-of-experiment field"].is_warn is False


# ─── CHECKS list invariant ─────────────────────────────────────────────────


def test_checks_list_size():
    """CHECKS must contain exactly 17 functions: the original 11 plus
    `check_figure_url_resolvable` (check 4b, added after the task #365
    relative-figure-URL incident on 2026-05-22) plus `check_body_nonstub`
    (check 0, added after the task #385 cache → body.md silent-handoff
    incident on 2026-05-25) plus `check_details_narrative_flow` (check 13,
    soft WARN-only added 2026-05-27 alongside the LessWrong-style narrative
    shift — see iterations.md 2026-05-27 + project_clean_result_narrative_shift)
    plus `check_figure_h2_is_deprecated` (check 12, soft WARN-only added
    2026-05-27 alongside the inline-figures-under-Results-sub-bullets
    prescriptive default — see iterations.md 2026-05-27 +
    feedback_figure_h2_deprecated) plus
    `check_planned_vs_actual_denominator` (check 11b, added 2026-05-27
    after task #391's C-axis silent drop — the dispatcher quietly
    dropped 1 of 3 planned factors and the clean-result-critic round 2
    PASSed without flagging the scope reduction) plus
    `check_mdx_safe_urls` (check 14, added 2026-05-28 after task #382's
    six `<https://...>` autolinks broke the dashboard's MDX renderer;
    extended 2026-05-28 with the table-cell `<|` class from task #399 and
    an authoritative real-parse backstop — still ONE entry in CHECKS).

    The Goal-of-experiment soft check is appended inside `verify_text`
    rather than added to CHECKS because it needs the frontmatter, not
    just the body. So `verify_text` returns 19 results, but `CHECKS`
    stays at 18 (the 2026-05-31 Reproducibility committed-at-sha check
    bumped it from 17).
    """
    assert len(verify_task_body.CHECKS) == 18


# ─── Check 14: MDX-safe prose (regex layer + real-parse backstop) ───
#
# Check 14 now has two layers (2026-05-28, durable MDX-safety fix):
#   (A) a fast regex pre-check layer (`_mdx_regex_findings`), node-INDEPENDENT,
#       the only layer when node is absent (CI without node), and
#   (B) an authoritative real-parse backstop (`_run_real_mdx_parse` →
#       `dashboard/scripts/mdx_parse_check.mjs`) that runs the exact
#       `mdast-util-from-markdown` parse the dashboard's MDXEditor runs.
#
# The regex-layer tests below call `_mdx_regex_findings` directly so they
# assert the regex behavior precisely and do NOT depend on node. The
# backstop tests call `check_mdx_safe_urls` (the combined path) and are
# guarded with `_NODE_MDX_AVAILABLE` so they skip cleanly where node / the
# helper / the dashboard deps are absent.

import shutil as _shutil  # noqa: E402

_NODE_MDX_AVAILABLE = (
    _shutil.which("node") is not None and verify_task_body._MDX_HELPER_PATH.exists()
)
if _NODE_MDX_AVAILABLE:
    # Confirm the deps actually load (an installed node + present helper but
    # missing dashboard/node_modules would otherwise mislead the gate).
    _v, _ = verify_task_body._run_real_mdx_parse("hello world\n")
    _NODE_MDX_AVAILABLE = _v == "pass"

_MDX_LABEL = (
    "MDX-safe prose — real-parse backstop + no `<https://...>` autolinks, "
    "`<` before digit, or `<|` in table cell"
)


# ── Layer A: regex pre-checks (node-INDEPENDENT) ──────────────────────────


def test_mdx_regex_autolink_in_repro_fails():
    """A `<https://...>` autolink anywhere in body prose breaks the MDX
    renderer. The regex layer must flag it (node-independent).

    Concrete trigger: task #382 (2026-05-28) shipped six autolinks in
    `## Reproducibility` and the dashboard showed an MDX parse error
    instead of the rendered body.
    """
    body = "- WandB run: <https://wandb.ai/superkaiba/eps/runs/abc12345>\n"
    findings = verify_task_body._mdx_regex_findings(body)
    assert findings
    assert any("wandb.ai" in f for f in findings)


def test_mdx_regex_autolink_inside_code_span_passes():
    """An autolink wrapped in inline-code backticks is safe — MDX never
    parses the inside of `` ` ` `` as JSX, so the regex layer ignores it."""
    body = "Some prose. The token `<https://foo.example/x>` is illustration."
    assert verify_task_body._mdx_regex_findings(body) == []


def test_mdx_regex_autolink_inside_fenced_block_passes():
    """An autolink inside a fenced code block is safe — MDX never
    parses inside ```` ``` ```` as JSX, so the regex layer ignores it."""
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
    """`p<0.05` in body prose breaks the MDX renderer (the dashboard
    parses `<0` as a JSX tag name and errors with 'Unexpected character
    `0` (U+0030) before name'). The regex layer must flag it.

    Recurred same-day as the autolink case on 2026-05-28.
    """
    body = "Some prose. The p-value was p<0.05 across all conditions."
    findings = verify_task_body._mdx_regex_findings(body)
    assert findings
    assert any("U+0030" in f or "p<0.05" in f for f in findings)


def test_mdx_regex_lt_digit_with_surrounding_spaces_passes():
    """`p < 0.05` (with spaces) is safe — `<` is not immediately
    followed by a digit, so the regex layer ignores it."""
    body = "Some prose. The p-value was p < 0.05 across all conditions."
    assert verify_task_body._mdx_regex_findings(body) == []


def test_mdx_regex_lt_digit_inside_code_span_passes():
    """`` `p<0.05` `` wrapped in inline-code backticks is safe — MDX
    never parses the inside of code spans as JSX (regex layer ignores it)."""
    body = "Some prose. The threshold was `p<0.05` in the pre-reg."
    assert verify_task_body._mdx_regex_findings(body) == []


def test_mdx_regex_lt_digit_inside_fenced_block_passes():
    """`p<0.05` inside a fenced code block is safe — MDX never parses
    inside fences as JSX (regex layer ignores it)."""
    body = "Some prose.\n\n```\nthreshold: p<0.05\nn<10\n```\n"
    assert verify_task_body._mdx_regex_findings(body) == []


def test_mdx_regex_html_entity_lt_passes():
    """`&lt;0.05` is safe — there is no literal `<` character in the
    source, only the HTML entity escape; the regex layer ignores it."""
    body = "Some prose. The p-value was &lt;0.05 across all conditions."
    assert verify_task_body._mdx_regex_findings(body) == []


def test_mdx_regex_combined_autolink_and_lt_digit_fails():
    """Body with BOTH a `<https://...>` autolink AND a `<digit` occurrence
    must be flagged, and the findings must surface both classes."""
    body = "See <https://foo.example/x>. The p-value was p<0.05 across all conditions."
    findings = verify_task_body._mdx_regex_findings(body)
    joined = " | ".join(findings)
    assert "U+002F" in joined
    assert "U+0030" in joined


# ── Layer A: table-cell `<|im_start|>` (the #399 class, node-INDEPENDENT) ──


def test_mdx_regex_table_cell_im_start_fails():
    """An unescaped `<|im_start|>` inside a GFM table-cell code span breaks
    the MDX renderer: the table parser splits the cell on the unescaped `|`
    before code-span recognition, exposing the `<` as a JSX tag start. The
    regex layer must flag it.

    Incident: task #399 (2026-05-28) — the prior narrow regex (which only
    stripped code spans wholesale) missed this because the `` ` ` `` wrap
    looked protective.
    """
    body = "| Probe | Value |\n|---|---|\n| boundary | `<|im_start|>assistant` |\n"
    findings = verify_task_body._mdx_regex_findings(body)
    assert findings
    assert any("table cell" in f for f in findings)


def test_mdx_regex_table_cell_im_start_escaped_passes():
    """The ESCAPED form `` `<\\|im_start\\|>` `` inside a table cell is safe
    — the inner pipes are escaped so the table parser does not split on
    them. The regex layer must NOT flag it."""
    body = "| Probe | Value |\n|---|---|\n| boundary | `<\\|im_start\\|>assistant` |\n"
    assert verify_task_body._mdx_regex_findings(body) == []


def test_mdx_regex_im_start_in_prose_passes():
    """`` `<|im_start|>` `` in a PROSE line (not a real GFM table row) is
    safe — the editor parses the code span normally there. A prose line
    merely containing a `|` (e.g. `log p(x | y)`) is not a table row, so
    its code spans stay protective. The regex layer must NOT flag it.

    This is the #399 false-positive guard: the #399 body has
    `` `<|im_start|>assistant\\n` `` inside a numbered list item that also
    contains `log p(... | ...)`, and that line must NOT be treated as a
    table row.
    """
    body = "First-token probe: log p(`*` | `<|im_start|>assistant\\n`) at boundary.\n"
    assert verify_task_body._mdx_regex_findings(body) == []


def test_mdx_regex_pipe_prose_then_hr_not_a_table():
    """A prose line containing a `|` (and a `` `<|im_start|>` `` code span)
    immediately followed by a bare `---` line is NOT a GFM table — the
    `---` is a thematic break / setext underline, not a one-column table
    delimiter. The table delimiter regex requires an internal `|`, so the
    prose line stays a prose line and its code span stays protective. The
    regex layer must NOT flag the code span.

    Regression guard: before tightening `_TABLE_DELIM_RE`, a bare `---`
    matched as a single-column delimiter, so the preceding pipe-bearing
    prose line was misclassified as a table header and the
    `` `<|im_start|>` `` span tripped a false-positive `<|` flag while the
    real MDX parser accepted the body.
    """
    body = "log p(x | y) and `<|im_start|>`.\n---\n\nnext\n"
    assert verify_task_body._table_row_line_indices(body.splitlines()) == set()
    assert verify_task_body._mdx_regex_findings(body) == []


# ── Full-path tests (regex + backstop combined) ───────────────────────────


def test_mdx_full_path_clean_prose_passes():
    """A clean prose fragment passes the combined check (regex clean; and
    when node is present, real-parse clean too)."""
    body = "Some prose. The p-value was p < 0.05 across all conditions."
    result = verify_task_body.check_mdx_safe_urls(body)
    assert result.passed, result.detail


def test_mdx_full_path_autolink_fails():
    """The combined check FAILs a bare-prose autolink (regex catches it
    regardless of node)."""
    body = "See the link: <https://foo.example/x> for context."
    result = verify_task_body.check_mdx_safe_urls(body)
    assert not result.passed
    assert "foo.example" in result.detail


def test_mdx_full_path_table_cell_im_start_fails():
    """The combined check FAILs an unescaped `<|im_start|>` table cell
    (regex catches it regardless of node; the backstop agrees when node is
    present)."""
    body = "| Probe | Value |\n|---|---|\n| boundary | `<|im_start|>assistant` |\n"
    result = verify_task_body.check_mdx_safe_urls(body)
    assert not result.passed
    assert "table cell" in result.detail


# ── Layer B: real-parse backstop (node-GATED) ─────────────────────────────


@pytest.mark.skipif(not _NODE_MDX_AVAILABLE, reason="node + MDX helper + deps not available")
def test_mdx_backstop_catches_novel_construct():
    """The real-parse backstop FAILs a construct the regexes do NOT catch
    — proving the backstop subsumes the narrow regex patch. `<%` is read
    by the real MDX parser as a JSX tag start ("Unexpected character `%`
    (U+0025) before name"), but matches none of the three regex classes.
    """
    # Sanity: the regex layer alone does NOT flag this.
    body = "Some prose with a stray <% token in it."
    assert verify_task_body._mdx_regex_findings(body) == []
    # The combined check FAILs because the real-parse backstop catches it.
    result = verify_task_body.check_mdx_safe_urls(body)
    assert not result.passed
    assert "real MDX parse failed" in result.detail


@pytest.mark.skipif(not _NODE_MDX_AVAILABLE, reason="node + MDX helper + deps not available")
def test_mdx_backstop_lt_eq_fails():
    """`x <= 10` (no space) is read by the real MDX parser as a JSX tag
    start before `=` and FAILs — the authoritative parser is stricter than
    the old regex assumed. Re-verified against the real parser
    (2026-05-28): the editor rejects `<=`, so the verifier must too."""
    body = "Some prose. The condition was x <= 10 across all runs."
    # Regex layer does NOT catch `<=` (it is not autolink / `<digit` / `<|`).
    assert verify_task_body._mdx_regex_findings(body) == []
    result = verify_task_body.check_mdx_safe_urls(body)
    assert not result.passed
    assert "real MDX parse failed" in result.detail


@pytest.mark.skipif(not _NODE_MDX_AVAILABLE, reason="node + MDX helper + deps not available")
def test_mdx_backstop_unclosed_tag_fails():
    """An unclosed `<details>` tag is read by the real MDX parser as a JSX
    element that never closes and FAILs. Re-verified against the real
    parser (2026-05-28): the editor requires a closing tag."""
    body = "Some prose. The <details> tag is here with no close."
    assert verify_task_body._mdx_regex_findings(body) == []
    result = verify_task_body.check_mdx_safe_urls(body)
    assert not result.passed
    assert "real MDX parse failed" in result.detail


@pytest.mark.skipif(not _NODE_MDX_AVAILABLE, reason="node + MDX helper + deps not available")
def test_mdx_backstop_html_comment_markers_pass():
    """Real body HTML comment markers (`<!-- legacy-sagan-card -->`,
    `<!-- workflow-fix-candidate v1 -->`, `<!-- epm:... -->`) MUST parse
    cleanly — the helper includes the editor's HTML-comment extension, so
    these markers are valid (omitting that extension would wrongly reject
    valid bodies). Confirmed empirically while building the helper."""
    body = (
        "Some prose.\n\n<!-- legacy-sagan-card -->\n\n"
        "<!-- workflow-fix-candidate v1 -->\ntarget_file: x\n"
        "<!-- /workflow-fix-candidate -->\n\n<!-- epm:pod-terminated v1 -->\n\nEnd.\n"
    )
    assert verify_task_body._mdx_regex_findings(body) == []
    result = verify_task_body.check_mdx_safe_urls(body)
    assert result.passed, result.detail


def test_mdx_helper_unavailable_falls_back_loud_not_silent(monkeypatch):
    """When node / the helper / the deps are unavailable, the check falls
    back to regex-only and APPENDS '(real MDX parse skipped: ...)' to the
    detail — it does NOT silently pass and does NOT hard-fail solely on the
    missing parser (the no-silent-fallback rule).

    Two sub-cases:
      - clean body → PASS, but the detail flags the skip so the operator
        knows the authoritative layer did not run;
      - regex-dirty body → still FAILs on the regex layer, with the skip
        reason appended.
    """
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
