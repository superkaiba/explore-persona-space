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


# ─── paper-stub support (`paper: true`) ────────────────────────────────────

PAPER_STUB_TITLE = "A claim about leakage (MODERATE confidence)"
_PAPER_STUB_ABSTRACT = (
    "This abstract paragraph stands in for the paper's own abstract and is the "
    "skim block the analyzer exemplar feed should show for a paper-task."
)
PAPER_STUB_BODY = f"""\
---
title: A claim about leakage (MODERATE confidence)
kind: experiment
paper: true
---
# A claim about leakage (MODERATE confidence)

{_PAPER_STUB_ABSTRACT}

Paper: docs/papers/issue_657/issue_657.pdf
"""


def test_is_paper_stub_detects_flag():
    assert rcr.is_paper_stub(PAPER_STUB_BODY)
    assert rcr.is_paper_stub(PAPER_STUB_BODY.replace("paper: true", "paper: 'true'"))
    assert not rcr.is_paper_stub(V3_BODY)
    assert not rcr.is_paper_stub(V2_BODY)


def test_paper_stub_skim_is_abstract():
    skim, _hero, conf_label, _ctext = rcr._extract_paper_stub(PAPER_STUB_BODY, PAPER_STUB_TITLE)
    assert skim.startswith("This abstract paragraph stands in")
    assert "Paper:" not in skim  # the paper-link line is excluded
    assert "#" not in skim  # the H1 is excluded
    assert conf_label == "MODERATE"  # from the title tag


def test_paper_stub_skim_h2_abstract():
    body = PAPER_STUB_BODY.replace(
        _PAPER_STUB_ABSTRACT,
        "## Abstract\n\nThe explicit abstract block.",
    )
    skim, _h, _c, _ct = rcr._extract_paper_stub(body, PAPER_STUB_TITLE)
    assert skim == "The explicit abstract block."


def test_render_inline_paper_stub_not_degenerate():
    out = rcr.render_inline([{"number": 657, "title": PAPER_STUB_TITLE, "body": PAPER_STUB_BODY}])
    assert "Abstract (paper-task — see docs/papers/issue_657/)" in out
    assert "This abstract paragraph stands in" in out
    assert "Confidence: MODERATE" in out
    # NOT a degenerate empty / `Confidence: ? —` render.
    assert "Confidence: ? —" not in out


# ─── v4 (2026-W26) extraction + exemplar-feed preference ─────────────────

# v4 fixture: four flat H2s (Takeaways / Goal / Methodology / Results),
# sentinel right after the H1, top-of-body **Methodology:** link,
# confidence ONLY in the H1 title tag, figure under `## Results`,
# `**Repro:**` / `**Context:**` footer.
V4_TITLE = "Contrastive negatives halve bystander leakage at matched install (HIGH confidence)"
V4_BODY = """# Contrastive negatives halve bystander leakage at matched install (HIGH confidence)
<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_888.md](https://example.com/blob)

## Takeaways

- Bystander leakage drops **52%** (0.31 -> 0.15) at matched install strength.
- Source rate holds at 0.97 vs 0.98 — no install cost from the negatives.
- Caveat: single model family, seed 42 only.

## Goal

**This experiment in context:** Tests whether the leakage result survives contrastive negatives.

**Broader narrative:** Whether training-mix composition bounds the leakage radius.

## Methodology

**Design:** 2 arms x 4 personas; the single variable is the negative set.

**Training:**

| Param | Value | Source |
|---|---|---|
| lr | 5e-6 | #530 |

## Results

### Leakage halves at matched install

Exactly what is plotted: mean bystander emission rate per arm.

![Bar chart of bystander leakage per arm.](figures/issue_888/hero.png)

> Bystander leakage per arm, seed 42.

The contrastive arm halves leakage.

**Repro:** 1x A100, commit abc123.
**Context:** created 2026-07-01; run 2026-07-02.
"""


def test_is_v4_body_detects_sentinel():
    assert rcr.is_v4_body(V4_BODY)
    assert not rcr.is_v4_body(V3_BODY)
    assert not rcr.is_v4_body(V2_BODY)
    assert not rcr.is_v4_body(LEGACY_BODY)
    # The two sentinel predicates are disjoint on conforming bodies.
    assert not rcr.is_v3_body(V4_BODY)


def test_v4_extracts_takeaways_block_not_whole_body():
    """For a v4 body the headline-skim block is `## Takeaways` (bounded at
    `## Goal`), NOT the whole-body `body.strip()` fallback the pre-fix
    TL;DR miss produced. Confidence comes from the H1 title tag; the hero
    figure is found under `## Results` (whole-body search)."""
    skim, hero, conf_label, conf_text = rcr._extract_markdown(V4_BODY, V4_TITLE)
    assert "Bystander leakage drops **52%**" in skim
    assert "Caveat: single model family" in skim
    # `## Goal` is the next H2 — NOT part of the skim block; nor is the
    # sentinel / Methodology content (i.e. this is not the whole body).
    assert "**This experiment in context:**" not in skim
    assert "**Design:**" not in skim
    assert rcr.SENTINEL_V4 not in skim
    assert conf_label == "HIGH"
    assert conf_text == ""
    assert hero == "figures/issue_888/hero.png"


def test_render_inline_v4_prints_takeaways():
    out = rcr.render_inline([{"number": 888, "title": V4_TITLE, "body": V4_BODY}])
    assert "Confidence: HIGH" in out
    assert "Confidence: ?" not in out
    assert "Bystander leakage drops **52%**" in out
    assert "Hero figure: figures/issue_888/hero.png" in out


def _install_fake_store(monkeypatch):
    """Fake a 4-row promoted store where recency order is v2 (newest) >
    v3a > v3b > v4 (oldest) — two rows in one tier pin partition
    exclusivity (a `rest` built as "not v4" would double-count the v3s)."""
    rows = [{"number": i, "hasCleanResult": True} for i in (1, 2, 3, 4)]
    experiments = {
        1: {
            "number": 1,
            "title": V2_TITLE,
            "body": V2_BODY,
            "updatedAt": "2026-07-04T10:00:00Z",  # most recent, v2
        },
        2: {
            "number": 2,
            "title": V3_TITLE,
            "body": V3_BODY,
            "updatedAt": "2026-07-03T10:00:00Z",  # v3 (a)
        },
        3: {
            "number": 3,
            "title": "second v3 (LOW confidence)",
            "body": V3_BODY,
            "updatedAt": "2026-07-02T10:00:00Z",  # v3 (b)
        },
        4: {
            "number": 4,
            "title": V4_TITLE,
            "body": V4_BODY,
            "updatedAt": "2026-07-01T10:00:00Z",  # OLDEST, v4
        },
    }
    monkeypatch.setattr(rcr.sagan_state, "list_by_status", lambda status, limit=200: rows)
    monkeypatch.setattr(rcr.sagan_state, "get_experiment", lambda n: {"experiment": experiments[n]})


def test_fetch_promoted_front_loads_v4_then_v3_then_rest(monkeypatch):
    """`prefer_shape='v4'` partitions [v4] + [v3] + [rest], each tier in
    recency order, with no duplicates (n=4 so truncation cannot mask one);
    `'v3'` keeps the pre-v4 semantics verbatim; `'any'` is pure recency."""
    _install_fake_store(monkeypatch)

    v4_first = rcr.fetch_promoted(4, prefer_shape="v4")
    assert [e["number"] for e in v4_first] == [4, 2, 3, 1]

    # 'v3' unchanged: v3 tier first (recency), then all non-v3 by recency
    # (the v4 body ranks non-preferred here).
    v3_first = rcr.fetch_promoted(4, prefer_shape="v3")
    assert [e["number"] for e in v3_first] == [2, 3, 1, 4]

    recency = rcr.fetch_promoted(4, prefer_shape="any")
    assert [e["number"] for e in recency] == [1, 2, 3, 4]


def test_prefer_shape_default_engages_v4(monkeypatch):
    """The v4 preference must engage on BOTH default surfaces: a no-kwarg
    `fetch_promoted` call (signature default) AND the argparse default
    (an implementation adding 'v4' semantics but leaving either default
    at 'v3' fails here)."""
    _install_fake_store(monkeypatch)
    default_order = rcr.fetch_promoted(4)
    assert [e["number"] for e in default_order] == [4, 2, 3, 1]
    assert rcr._build_parser().get_default("prefer_shape") == "v4"


def test_fetch_promoted_limit_covers_store(monkeypatch):
    """`list_by_status` truncates ASCENDING by id at `limit`, so the feed
    must request a limit that covers the whole completed store (757 rows
    as of 2026-07-11) or every recent promotion is silently dropped."""
    recorded = {}

    def fake_list(status, limit=200):
        recorded["limit"] = limit
        return []

    monkeypatch.setattr(rcr.sagan_state, "list_by_status", fake_list)
    assert rcr.fetch_promoted(3) == []
    assert recorded["limit"] >= 1000
