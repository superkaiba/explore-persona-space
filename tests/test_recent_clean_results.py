"""Regression tests for scripts/recent_clean_results.py inline extraction.

Pins the v2 clean-result handling (task #608): v2 bodies (2026-W22 spec,
task #454) carry confidence ONLY in the H1 title tag and a nested
``## TL;DR`` (### Motivation / ### What I ran / ### Findings), so inline
mode must derive confidence from the title and print the TL;DR block —
not the degenerate "Confidence: ? —" line the legacy-only extractor
produced. Also pins legacy (pre-2026-05-13) body handling.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "recent_clean_results.py"

_spec = importlib.util.spec_from_file_location("recent_clean_results_under_test", SCRIPT)
assert _spec is not None and _spec.loader is not None
rcr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rcr)


V2_TITLE = "Markers leak to near-twin bystanders under contrastive SFT (MODERATE confidence)"
V2_BODY = """# Markers leak to near-twin bystanders under contrastive SFT (MODERATE confidence)
<!-- clean-result-v2 -->

**Methodology:** [docs/methodology/issue_999.md](https://example.com/blob)

## Human TL;DR

stub for thomas.

## TL;DR

### Motivation

Why we ran this experiment.

### What I ran

Trained 4 LoRAs with contrastive negatives.

### Findings

#### The marker leaks to near twins

![hero caption](figures/issue_999/hero.png)

Leakage tracks persona distance.

## Reproducibility

| Param | Value |
|---|---|
| lr | 5e-6 |
"""

LEGACY_BODY = """## TL;DR

### Background

We wanted to know whether X.

### Results

![hero](https://example.com/hero.png)

**Confidence: HIGH** — three seeds, tight CIs.
"""


def test_v2_confidence_from_title_tag():
    tldr, hero, conf_label, conf_text = rcr._extract_markdown(V2_BODY, V2_TITLE)
    assert conf_label == "MODERATE"
    assert conf_text == ""
    assert hero == "figures/issue_999/hero.png"
    # The TL;DR block keeps its nested structure (H3/H4 stay inside the H2).
    assert "### Motivation" in tldr
    assert "#### The marker leaks to near twins" in tldr
    # Bounded at the H2 boundary: Reproducibility is NOT part of the block.
    assert "Reproducibility" not in tldr


def test_legacy_confidence_sentence_still_wins():
    tldr, hero, conf_label, conf_text = rcr._extract_markdown(LEGACY_BODY, "no tag here")
    assert conf_label == "HIGH"
    assert conf_text == "three seeds, tight CIs."
    assert hero == "https://example.com/hero.png"
    assert "### Background" in tldr


def test_render_inline_v2_not_degenerate():
    out = rcr.render_inline(
        [{"number": 999, "title": V2_TITLE, "body": V2_BODY}],
    )
    assert "Confidence: MODERATE" in out
    assert "Confidence: ?" not in out
    assert "### Motivation" in out
    assert "#### The marker leaks to near twins" in out
    assert "Hero figure: figures/issue_999/hero.png" in out


def test_render_inline_respects_max_chars():
    out = rcr.render_inline(
        [{"number": 999, "title": V2_TITLE, "body": V2_BODY}],
        max_chars=120,
    )
    assert "..." in out
    # Truncated block stays bounded (120 chars + surrounding scaffolding).
    block = out.split("URL:")[1]
    assert len(block) < 500


def test_missing_body_falls_back_to_title_confidence():
    out = rcr.render_inline([{"number": 7, "title": V2_TITLE, "body": ""}])
    assert "Confidence: MODERATE" in out
