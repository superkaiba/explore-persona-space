"""Tests for scripts/verify_task_body.py — six mechanical checks for the
markdown clean-result spec.

Each test feeds a synthetic body string into verify_text() and asserts
which checks pass / fail.
"""

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


GOOD_BODY = """\
# Some claim about persona leakage (MODERATE confidence)

## TL;DR
- **Motivation:** I wanted to test whether X drives Y.
- **What I ran:** Trained 3 seeds at lr=3e-5, evaluated on benchmark Z.
- **Results:** Effect is present at p<0.01 ([figure below](#figure)).
- **Next steps:** Replicate at 70B, run the partial-correlation control.

## Figure
![hero plot](artifacts/hero.png)

*Caption: Mean cross-persona leakage with 95% CI bands across three training seeds and four benchmark conditions.*

## Details

Free-form description here.

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


def test_good_body_passes_all():
    ok, results = verify_task_body.verify_text(GOOD_BODY)
    assert ok, [r.render() for r in results if not r.passed]
    assert all(r.passed for r in results)


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
    body = GOOD_BODY.replace(
        "https://huggingface.co/superkaiba1/explore-persona-space/tree/abc123def",
        "TBD",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Reproducibility URLs are permanent"].passed
    assert "TBD" in by_name["Reproducibility URLs are permanent"].detail


def test_repro_unpinned_github():
    body = GOOD_BODY.replace(
        "https://github.com/superkaiba/explore-persona-space/blob/0123456789abcdef/scripts/run.py",
        "https://github.com/superkaiba/explore-persona-space/blob/main/scripts/run.py",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Reproducibility URLs are permanent"].passed
    assert "GitHub" in by_name["Reproducibility URLs are permanent"].detail


def test_repro_unpinned_hf():
    body = GOOD_BODY.replace(
        "https://huggingface.co/superkaiba1/explore-persona-space/tree/abc123def",
        "https://huggingface.co/superkaiba1/explore-persona-space",
    )
    ok, results = verify_task_body.verify_text(body)
    assert not ok
    by_name = _results_by_name(results)
    assert not by_name["Reproducibility URLs are permanent"].passed
    assert "HF" in by_name["Reproducibility URLs are permanent"].detail


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


def test_frontmatter_stripped_before_checks():
    body = "---\ntitle: foo\nkind: experiment\n---\n" + GOOD_BODY
    ok, results = verify_task_body.verify_text(body)
    assert ok, [r.render() for r in results if not r.passed]
