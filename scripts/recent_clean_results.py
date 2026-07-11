#!/usr/bin/env python3
"""Print the N most-recently-promoted clean-result experiments.

Used by the analyzer agent (Step 1.5) to load in-context exemplars of the
target write-up quality. Promoted clean-results are tasks with
``has_clean_result=true`` and ``status='completed'`` (the analyzer flips
``has_clean_result`` after the reviewer passes; the user advances to
``completed`` via ``task.py promote``).

Usage:
    uv run python scripts/recent_clean_results.py --n 3 --format inline
    uv run python scripts/recent_clean_results.py --n 5 --format json

``--format inline`` (default) prints, for each clean-result, the task
number, title, hero figure (if extractable), the headline-skim block
verbatim (bounded by ``--max-chars``), and a Confidence line — suitable
for one-pass agent reading. The headline-skim block is ``## Takeaways``
for v4 bodies (sentinel ``<!-- clean-result-v4 -->``, 2026-W26) and v3
bodies (sentinel ``<!-- clean-result-v3 -->``, 2026-W24), and the
``## TL;DR`` block for v2 / legacy bodies. Under the v2+ clean-result
spec confidence lives ONLY in the H1 title tag, so the Confidence line
is derived from the title when no body ``Confidence:`` sentence exists.
``--format json`` emits the hydrated experiment payloads (body included)
for downstream tools.

**Exemplar feed (forward-only):** the inline feed PREFERS bodies of the
CURRENT shape so the analyzer's few-shot exemplars track it — without
this preference the feed drifts new drafts back toward a retired
register (a real regression vector). ``--prefer-shape`` controls this:
``v4`` (default) front-loads v4-sentinel bodies, back-fills v3 (the
closest register — same ``## Takeaways`` headline skim), then the rest
(v2 / legacy / promoted paper-task stubs); ``v3`` keeps the pre-v4
preference verbatim (v3 first, then all non-v3); ``any`` is the
pre-cutover recency-only behavior.

Implementation: reads the file-based task workflow through the
:mod:`task_state` shim (``scripts/task_state.py`` → ``task_workflow``).
``list_by_status`` returns registry-style rows WITHOUT bodies or
timestamps, so each promoted row is hydrated via ``get_experiment``
before extraction and recency sorting (#608).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task_state as sagan_state

DEFAULT_N = 3
DEFAULT_MAX_CHARS = 4000

# Markdown bodies. Four generations coexist (forward-only):
#   * v4 (current, sentinel `<!-- clean-result-v4 -->`, 2026-W26): four
#     flat H2s (Takeaways / Goal / Methodology / Results); the
#     headline-skim block is `## Takeaways` (3-6 bullets, numbers-first);
#     confidence ONLY in the H1 title tag.
#   * v3 (sentinel `<!-- clean-result-v3 -->`, 2026-W24): five flat H2s;
#     the headline-skim block is `## Takeaways`; confidence ONLY in the
#     H1 title tag.
#   * v2 (sentinel `<!-- clean-result-v2 -->`, 2026-W22, task #454):
#     `## TL;DR` → ### Motivation / ### What I ran / ### Findings (+ ####
#     per result); confidence ONLY in the H1 title tag.
#   * Legacy (pre-2026-05-13): ### Background / ### Results inside TL;DR +
#     a body `**Confidence: X** — ...` sentence.
# v2/legacy share the `## TL;DR` H2; v3 AND v4 use `## Takeaways` (one
# RE_MD_TAKEAWAYS serves both — its `(?=^##\s+|\Z)` lookahead bounds the
# block at the next H2: `## Goal` for v4, `## What I ran` for v3). The
# `^##\s` lookahead does not match H3/H4, so nested subsections stay
# inside the captured block.
# Adding a future generation (v5): add its sentinel constant + `is_v5_body`
# predicate here, route it in `_extract_markdown`, and give it the top
# tier in `fetch_promoted` (+ the argparse choice/default in
# `_build_parser`). Sentinel detection is a substring check (mirrors the
# original `is_v3_body`): a body merely QUOTING a sentinel string in
# prose classifies as that shape — acceptable because promoted
# clean-results don't quote sentinels in practice, and mis-tiering only
# reorders the feed.
SENTINEL_V3 = "<!-- clean-result-v3 -->"
SENTINEL_V4 = "<!-- clean-result-v4 -->"
RE_MD_TLDR = re.compile(r"(?ms)^##\s+TL;DR\s*$(?P<body>.+?)(?=^##\s+|\Z)")
RE_MD_TAKEAWAYS = re.compile(r"(?ms)^##\s+Takeaways\s*$(?P<body>.+?)(?=^##\s+|\Z)")
# Image target may be an absolute URL or a repo-relative figures/ path.
RE_MD_HERO = re.compile(r"!\[[^\]]*\]\((\S+?)\)")
RE_MD_CONFIDENCE = re.compile(
    r"\*\*\s*Confidence\s*:\s*(HIGH|MODERATE|LOW)\s*\*\*\s*[—\-–]\s*(?P<text>.+?)$",  # noqa: RUF001
    re.IGNORECASE | re.MULTILINE,
)
RE_TITLE_CONFIDENCE = re.compile(r"\((HIGH|MODERATE|LOW)\s+confidence\)", re.IGNORECASE)

# Sagan-card HTML bodies.
RE_HTML_TLDR = re.compile(r'(?is)<section[^>]+id="tldr"[^>]*>(?P<body>.*?)</section>')
RE_HTML_FIGURE_IMG = re.compile(
    r'(?is)<figure[^>]+id="figure"[^>]*>.*?<img[^>]+src="(?P<src>[^"]+)"'
)
RE_HTML_CONFIDENCE = re.compile(
    r"(?is)Confidence\s*:\s*(?P<label>HIGH|MODERATE|LOW)\s*[—\-–]\s*(?P<text>.+?)(?:<|\.)",  # noqa: RUF001
)


def is_v3_body(body: str) -> bool:
    """True when the body carries the v3 clean-result sentinel."""
    return SENTINEL_V3 in body


def is_v4_body(body: str) -> bool:
    """True when the body carries the v4 clean-result sentinel."""
    return SENTINEL_V4 in body


# A `paper: true` task's body is a thin paper-stub (H1 + abstract + paper link);
# the canonical clean-result is the LaTeX paper under docs/papers/issue_<N>/. The
# exemplar feed extracts the abstract as the skim block (NOT `## Takeaways`,
# which a stub doesn't have).
RE_FM_PAPER = re.compile(r"(?im)^paper\s*:\s*(?:true|'true'|\"true\")\s*$")
RE_MD_ABSTRACT_H2 = re.compile(r"(?ms)^##\s+Abstract\s*$(?P<body>.+?)(?=^##\s|\Z)")


def is_paper_stub(body: str) -> bool:
    """True when the body's leading frontmatter carries `paper: true`."""
    if not body.startswith("---"):
        return False
    end = body.find("\n---", 3)
    head = body[3:end] if end != -1 else body
    return bool(RE_FM_PAPER.search(head))


def _extract_paper_stub(body: str, title: str) -> tuple[str, str, str, str]:
    """Return (skim_block, hero_url, confidence_label, confidence_text) for a stub.

    The skim block is the stub abstract (a `## Abstract` H2 block, else the first
    prose paragraph after the H1, excluding the paper-link line). Confidence
    comes from the H1 title tag. Hero is the first inline image if any.
    """
    # Drop the leading frontmatter so the H1 / abstract scan is clean.
    text = body
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :].lstrip("\n")
    m = RE_MD_ABSTRACT_H2.search(text)
    region = m.group("body") if m else text
    out: list[str] = []
    for raw_line in region.splitlines():
        line = raw_line.strip()
        if not line:
            if out:
                break  # first paragraph only
            continue
        if line.startswith("#"):  # H1/H2/H3 heading
            if out:
                break
            continue
        low = line.lower()
        if low.startswith("paper:") or low.startswith("pdf:"):
            if out:
                break
            continue
        out.append(line)
    skim = " ".join(out).strip()
    hero_m = RE_MD_HERO.search(text)
    hero = hero_m.group(1) if hero_m else ""
    title_m = RE_TITLE_CONFIDENCE.search(title)
    conf_label = title_m.group(1).upper() if title_m else "?"
    return skim, hero, conf_label, ""


def fetch_promoted(n: int, prefer_shape: str = "v4") -> list[dict[str, Any]]:
    """Return up to N most-recently-promoted clean-result experiment dicts.

    ``list_by_status`` rows are registry-style (no ``body``, no
    timestamps), so every promoted row is hydrated via ``get_experiment``
    — that supplies the body for headline-skim/confidence extraction and
    ``updatedAt`` (last event ts) for the recency sort. Without the
    hydration step the extractors ran on empty strings and inline mode
    printed only titles + a degenerate "Confidence: ? —" line (#608).

    ``prefer_shape='v4'`` (default) front-loads v4-sentinel bodies so the
    analyzer's few-shot exemplar feed tracks the current shape, back-fills
    with v3-sentinel bodies (the closest register — same ``## Takeaways``
    headline skim), then everything else (v2 / legacy / promoted
    paper-task stubs — stubs carry no markdown sentinel, so they rank in
    the "rest" tier by design, not oversight). Each tier keeps its own
    recency order. ``prefer_shape='v3'`` keeps the pre-v4 preference
    verbatim: v3 bodies first, then all non-v3 (a v4 body ranks
    non-preferred there). ``prefer_shape='any'`` restores the pre-cutover
    behavior (pure recency, no shape weighting).
    """
    # The limit must cover the WHOLE completed store: list_by_status
    # iterates task folders ASCENDING by id and truncates at `limit`, so a
    # small limit silently returns only the OLDEST tasks. At limit=200 the
    # feed saw only #13..#572 and dropped every recent promotion (757
    # completed as of 2026-07-11; growth ~10/day gives years of headroom
    # at 10_000).
    completed = sagan_state.list_by_status(status="completed", limit=10_000)
    promoted = [
        sagan_state.get_experiment(e["number"])["experiment"]
        for e in completed
        if e.get("hasCleanResult")
    ]
    promoted.sort(key=lambda e: e.get("updatedAt") or e.get("createdAt") or "", reverse=True)
    if prefer_shape == "v4":
        # Exclusive partition (elif-chain) — a row lands in exactly one
        # tier, so the concatenation cannot duplicate an experiment.
        v4: list[dict[str, Any]] = []
        v3: list[dict[str, Any]] = []
        rest: list[dict[str, Any]] = []
        for e in promoted:
            body = e.get("body") or ""
            if is_v4_body(body):
                v4.append(e)
            elif is_v3_body(body):
                v3.append(e)
            else:
                rest.append(e)
        promoted = v4 + v3 + rest
    elif prefer_shape == "v3":
        v3 = [e for e in promoted if is_v3_body(e.get("body") or "")]
        non_v3 = [e for e in promoted if not is_v3_body(e.get("body") or "")]
        promoted = v3 + non_v3
    return promoted[:n]


def _extract_html(body: str) -> tuple[str, str, str, str]:
    """Return (tldr_text, hero_url, confidence_label, confidence_text) from HTML."""
    tldr_m = RE_HTML_TLDR.search(body)
    tldr = tldr_m.group("body").strip() if tldr_m else ""
    # Crude tag strip for inline rendering — the analyzer agent gets the raw
    # body via the dashboard URL if it needs structure.
    tldr_text = re.sub(r"<[^>]+>", " ", tldr)
    tldr_text = " ".join(tldr_text.split())

    hero_m = RE_HTML_FIGURE_IMG.search(body)
    hero = hero_m.group("src") if hero_m else ""

    conf_m = RE_HTML_CONFIDENCE.search(body)
    conf_label = conf_m.group("label").upper() if conf_m else "?"
    conf_text = re.sub(r"<[^>]+>", " ", conf_m.group("text")).strip() if conf_m else ""
    return tldr_text, hero, conf_label, conf_text


def _extract_markdown(body: str, title: str) -> tuple[str, str, str, str]:
    """Return (skim_block, hero_url, confidence_label, confidence_text).

    The skim block is ``## Takeaways`` for v4 and v3 bodies and ``## TL;DR``
    for v2 / legacy bodies. Handles v4 (sentinel ``<!-- clean-result-v4 -->``;
    Takeaways / Goal / Methodology / Results; confidence ONLY in the H1
    title tag), v3 (sentinel ``<!-- clean-result-v3 -->``; confidence ONLY
    in the title tag), v2 (``## TL;DR`` → ### Motivation / ### What I ran /
    ### Findings; confidence ONLY in the title tag), and legacy bodies
    (### Background / ### Results + a body ``**Confidence: X** — ...``
    sentence). The body sentence wins when present (legacy); a conforming
    v4/v3/v2 body has no such sentence, so confidence comes from the title
    tag there.

    The v4/v3 hero is searched whole-body — the ``## Takeaways`` block is
    figure-free by spec (figures live under ``## Results`` for v4,
    ``## Findings`` for v3), so the first inline image there is the hero
    (the v4 top-of-body ``**Methodology:** [..](..)`` pointer is a plain
    link, not an image, so it cannot false-match).
    """
    skim_m = (
        RE_MD_TAKEAWAYS.search(body)
        if (is_v4_body(body) or is_v3_body(body))
        else RE_MD_TLDR.search(body)
    )
    tldr = skim_m.group("body").strip() if skim_m else body.strip()

    hero_m = RE_MD_HERO.search(tldr) or RE_MD_HERO.search(body)
    hero = hero_m.group(1) if hero_m else ""

    conf_m = RE_MD_CONFIDENCE.search(body)
    if conf_m:
        conf_label = conf_m.group(1).upper()
        conf_text = conf_m.group("text").strip().rstrip("*").strip()
    else:
        title_m = RE_TITLE_CONFIDENCE.search(title)
        conf_label = title_m.group(1).upper() if title_m else "?"
        conf_text = ""
    return tldr, hero, conf_label, conf_text


def render_inline(experiments: list[dict[str, Any]], max_chars: int = DEFAULT_MAX_CHARS) -> str:
    base = sagan_state.BASE_URL
    out: list[str] = []
    for exp in experiments:
        body = exp.get("body") or ""
        number = exp.get("number", "?")
        title = exp.get("title", "")
        url = f"{base}/tasks/{number}"

        if "<section" in body.lower() and 'id="tldr"' in body.lower():
            # Sagan-card HTML era: tag-stripped compact summary.
            tldr, hero, conf_label, conf_text = _extract_html(body)
            compact = " ".join(tldr.split())
            if len(compact) > 400:
                compact = compact[:397] + "..."
            summary = f"Summary: {compact}" if compact else ""
        elif is_paper_stub(body):
            # Paper-stub: the canonical clean-result is the LaTeX paper; the
            # skim block is the stub abstract (not `## Takeaways`).
            tldr, hero, conf_label, conf_text = _extract_paper_stub(body, title)
            if len(tldr) > max_chars:
                tldr = tldr[: max_chars - 3] + "..."
            summary = f"Abstract (paper-task — see docs/papers/issue_{number}/): {tldr}"
        else:
            # Markdown (v4 + v3 + v2 + legacy): print the headline-skim
            # block verbatim (`## Takeaways` for v4/v3, `## TL;DR` for
            # v2/legacy) so the analyzer sees real structure, bounded by
            # --max-chars.
            tldr, hero, conf_label, conf_text = _extract_markdown(body, title)
            if len(tldr) > max_chars:
                tldr = tldr[: max_chars - 3] + "..."
            summary = tldr

        out.append(f"## #{number}: {title}")
        out.append(f"URL: {url}")
        if hero:
            out.append(f"Hero figure: {hero}")
        if summary:
            out.append(f"\n{summary}")
        conf_line = f"Confidence: {conf_label}"
        if conf_text:
            conf_line += f" — {conf_text}"
        out.append(f"\n{conf_line}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (factored out so tests can assert defaults)."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help=f"how many to return (default {DEFAULT_N})",
    )
    p.add_argument(
        "--format",
        choices=("inline", "json"),
        default="inline",
        help="output format (default: inline)",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=(
            "per-exemplar headline-skim truncation bound for inline mode "
            f"(default {DEFAULT_MAX_CHARS})"
        ),
    )
    p.add_argument(
        "--prefer-shape",
        choices=("v4", "v3", "any"),
        default="v4",
        help=(
            "exemplar shape preference: 'v4' (default) front-loads "
            "v4-sentinel bodies so the analyzer's few-shot exemplars track "
            "the current shape, back-fills v3, then the rest; 'v3' is the "
            "pre-v4 preference (v3 first, then all non-v3); 'any' is pure "
            "recency (pre-cutover behavior)"
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    experiments = fetch_promoted(args.n, prefer_shape=args.prefer_shape)
    if not experiments:
        print("# No promoted clean-results found.")
        return 0

    if args.format == "json":
        json.dump(experiments, sys.stdout, indent=2, default=str)
        print()
    else:
        print(render_inline(experiments, max_chars=args.max_chars))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
