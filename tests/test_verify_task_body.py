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

## Goal

Characterize how cross-persona leakage scales with seed and benchmark.

## TL;DR
- **Motivation:** I wanted to test whether X drives Y.
- **What I ran:** Trained 3 seeds at lr=3e-5, evaluated on benchmark Z.
- **Results:** Effect is present at p<0.01 ([figure below](#figure)).
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
    # 13 body-only checks (CHECKS — body-nonstub check 0 prepended,
    # plus the 12 structural checks incl. Figure URL resolvable) + 1
    # Goal-of-experiment soft check appended by verify_text (it needs the
    # frontmatter, not just the body).
    assert len(results) == 14


def test_missing_confidence_tag():
    body = GOOD_BODY.replace(" (MODERATE confidence)", "")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["title confidence tag"].passed


def test_wrong_section_order():
    body = GOOD_BODY.replace("## Figure", "## TempPlaceholder")
    body = body.replace("## Details", "## Figure")
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
    body = GOOD_BODY.replace("- **What I ran:**", "- **Stuff I did:**")
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["TL;DR bullets carry the four required labels"].passed
    assert "What I ran" in by_name["TL;DR bullets carry the four required labels"].detail


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


# ─── Check 4: hero image present in `## Figure` ───────────────────────────


def test_figure_image_present_pass():
    """Happy path for check 4 already exercised by `test_good_body_passes_all`,
    but also assert the check name and detail directly."""
    _ok, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    assert by_name["Figure contains an image"].passed
    assert "1 image" in by_name["Figure contains an image"].detail


def test_figure_missing_image_fails():
    """Strip the `![hero plot](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_999/hero.png)` image line; the check fails."""
    body = GOOD_BODY.replace(
        "![hero plot](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0123456789abcdef/figures/issue_999/hero.png)\n",
        "",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Figure contains an image"].passed
    assert "no `![alt](path)` image syntax" in by_name["Figure contains an image"].detail


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
    assert by_name["Figure contains an image"].passed
    assert by_name["Figure URL resolvable"].passed
    assert ok


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
- **Results:** Effect is present at p<0.01.
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

## Goal

Characterize how cross-persona leakage scales with seed and benchmark.

## TL;DR
- **Motivation:** I wanted to test whether X drives Y.
- **What I ran:** Trained 3 seeds at lr=3e-5, evaluated on benchmark Z.
- **Results:** Effect is present at p<0.01.
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
    """Happy path: frontmatter has `goal:`, body has a `## Goal` H2. Already
    exercised by `test_good_body_passes_all` against GOOD_BODY; this test
    isolates the assertion."""
    _ok, results = verify_task_body.verify_text(GOOD_BODY)
    by_name = _results_by_name(results)
    r = by_name["Goal-of-experiment field"]
    assert r.passed
    assert r.is_warn is False
    assert "frontmatter goal present" in r.detail
    assert "`## Goal` H2 found in body" in r.detail


def test_goal_of_experiment_missing_warns_not_fails():
    """When the body lacks the goal field + H2, the soft check WARNs but
    does NOT FAIL the body. Enforcement is at /issue Step 0c, not here."""
    # Strip both the frontmatter `goal:` line and the `## Goal` H2 block.
    body_without_goal = GOOD_BODY.replace(
        "goal: Characterize how cross-persona leakage scales with seed and benchmark\n",
        "",
    ).replace(
        "## Goal\n\nCharacterize how cross-persona leakage scales with seed and benchmark.\n\n",
        "",
    )
    ok, results = verify_task_body.verify_text(body_without_goal)
    # Overall should remain PASS because Goal absence is soft.
    assert ok, [r.render() for r in results if not r.passed]
    by_name = _results_by_name(results)
    r = by_name["Goal-of-experiment field"]
    assert r.passed is True  # passed=True, but rendered as WARN
    assert r.is_warn is True
    assert "missing" in r.detail
    assert "frontmatter" in r.detail
    assert "`## Goal` H2" in r.detail


def test_goal_of_experiment_partial_state_warns():
    """Goal H2 in body but no frontmatter (or vice versa) → WARN naming
    only the missing half."""
    # Drop just the frontmatter goal line; keep the H2.
    body_only_h2 = GOOD_BODY.replace(
        "goal: Characterize how cross-persona leakage scales with seed and benchmark\n",
        "",
    )
    _ok, results = verify_task_body.verify_text(body_only_h2)
    by_name = _results_by_name(results)
    r = by_name["Goal-of-experiment field"]
    assert r.is_warn is True
    assert "frontmatter `goal:`" in r.detail


# ─── CHECKS list invariant ─────────────────────────────────────────────────


def test_checks_list_size():
    """CHECKS must contain exactly 13 functions: the original 11 plus
    `check_figure_url_resolvable` (check 4b, added after the task #365
    relative-figure-URL incident on 2026-05-22) plus `check_body_nonstub`
    (check 0, added after the task #385 cache → body.md silent-handoff
    incident on 2026-05-25).

    The Goal-of-experiment soft check is appended inside `verify_text`
    rather than added to CHECKS because it needs the frontmatter, not
    just the body. So `verify_text` returns 14 results, but `CHECKS`
    stays at 13.
    """
    assert len(verify_task_body.CHECKS) == 13
