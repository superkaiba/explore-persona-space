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


# v3 fixture (2026-W24): five flat H2s, headline-skim block is
# `## Takeaways` (not `## TL;DR`), figure lives under `## Findings`,
# confidence ONLY in the H1 title tag.
V3_TITLE = "Tulu-25 lifts alignment +17 pts over baseline (HIGH confidence)"
V3_BODY = """# Tulu-25 lifts alignment +17 pts over baseline (HIGH confidence)

<!-- clean-result-v3 -->

**Methodology:** [docs/methodology/issue_777.md](https://example.com/blob)

## Takeaways

- Tulu-25 lifts alignment **+17 pts** (95% CI 12-22) over baseline.
- Capability holds at 0.82 vs 0.81 — no regression at 25% mixing.
- Caveat: single model family, three seeds only.

## What I ran

- **Why:** Test whether the prior X effect generalises to benchmark Z.
- **Design:** 3 seeds; baseline vs tulu-25; the single variable is the data mix.
- **Eval:** Betley alignment, Claude judge, 200 probes.

## Findings

### A clean +17-pt lift across three seeds

![Bar chart of mean alignment across three seeds.](figures/issue_777/hero.png)

The lift holds at every seed.

## Data

### Trained on

Tulu-25 mix (tier 2), 2,000 rows.

Full data: [HF dataset](https://example.com/data)

### Evaluated with

200 Betley probes. Full probe bank: [link](https://example.com/probes)

### Generated

600 completions. Full raw completions: [raw](https://example.com/raw)

## Reproducibility

| Param | Value |
|---|---|
| lr | 3e-5 |
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


# ─── v3 (2026-W24) extraction + exemplar-feed preference ─────────────────


def test_v3_extracts_takeaways_block_not_tldr():
    """For a v3 body the headline-skim block is `## Takeaways`, not
    `## TL;DR` (which a v3 body doesn't have). Confidence comes from the
    H1 title tag; the hero figure is found under `## Findings`."""
    skim, hero, conf_label, conf_text = rcr._extract_markdown(V3_BODY, V3_TITLE)
    assert conf_label == "HIGH"
    assert conf_text == ""
    # The skim block is the Takeaways content, bounded at the next H2.
    assert "Tulu-25 lifts alignment **+17 pts**" in skim
    assert "Caveat: single model family" in skim
    # `## What I ran` is the next H2 — NOT part of the skim block.
    assert "What I ran" not in skim
    assert "**Why:**" not in skim
    # Hero is the first inline image (lives under `## Findings`).
    assert hero == "figures/issue_777/hero.png"


def test_render_inline_v3_prints_takeaways():
    out = rcr.render_inline([{"number": 777, "title": V3_TITLE, "body": V3_BODY}])
    assert "Confidence: HIGH" in out
    assert "Confidence: ?" not in out
    assert "Tulu-25 lifts alignment **+17 pts**" in out
    assert "Hero figure: figures/issue_777/hero.png" in out


def test_is_v3_body_detects_sentinel():
    assert rcr.is_v3_body(V3_BODY)
    assert not rcr.is_v3_body(V2_BODY)
    assert not rcr.is_v3_body(LEGACY_BODY)


def test_fetch_promoted_front_loads_v3_over_recent_v2(monkeypatch):
    """`fetch_promoted(prefer_shape='v3')` front-loads v3 bodies ahead of
    MORE-RECENT v2 bodies; `prefer_shape='any'` restores pure recency."""
    # Three promoted clean-results: a v3 body that is OLDER than two v2
    # bodies. Pure recency would put the v2s first; the v3 preference
    # front-loads the v3 body.
    rows = [
        {"number": 1, "hasCleanResult": True},
        {"number": 2, "hasCleanResult": True},
        {"number": 3, "hasCleanResult": True},
    ]
    experiments = {
        1: {
            "number": 1,
            "title": V2_TITLE,
            "body": V2_BODY,
            "updatedAt": "2026-06-13T10:00:00Z",  # most recent
        },
        2: {
            "number": 2,
            "title": "another v2 (MODERATE confidence)",
            "body": V2_BODY,
            "updatedAt": "2026-06-12T10:00:00Z",
        },
        3: {
            "number": 3,
            "title": V3_TITLE,
            "body": V3_BODY,
            "updatedAt": "2026-06-01T10:00:00Z",  # OLDEST
        },
    }
    monkeypatch.setattr(rcr.sagan_state, "list_by_status", lambda status, limit=200: rows)
    monkeypatch.setattr(rcr.sagan_state, "get_experiment", lambda n: {"experiment": experiments[n]})

    # prefer_shape='v3': the older v3 body is front-loaded.
    promoted = rcr.fetch_promoted(3, prefer_shape="v3")
    assert [e["number"] for e in promoted] == [3, 1, 2]

    # prefer_shape='any': pure recency (v3 falls last because it's oldest).
    recency = rcr.fetch_promoted(3, prefer_shape="any")
    assert [e["number"] for e in recency] == [1, 2, 3]
