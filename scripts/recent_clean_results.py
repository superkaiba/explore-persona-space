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
for v3 bodies (sentinel ``<!-- clean-result-v3 -->``, 2026-W24) and the
``## TL;DR`` block for v2 / legacy bodies. Under the v2+ clean-result
spec confidence lives ONLY in the H1 title tag, so the Confidence line
is derived from the title when no body ``Confidence:`` sentence exists.
``--format json`` emits the hydrated experiment payloads (body included)
for downstream tools.

**Exemplar feed (forward-only):** when v3 bodies exist among the
promoted set, the inline feed PREFERS them so the analyzer's few-shot
exemplars track the current shape — without this preference the feed
stays all-v2 and drifts new drafts back toward the retired nested-TL;DR
register (a real regression vector). ``--prefer-shape`` controls this
(default ``v3``); pass ``--prefer-shape any`` for the pre-cutover
recency-only behavior.

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

# Markdown bodies. Three generations coexist (forward-only):
#   * v3 (current, sentinel `<!-- clean-result-v3 -->`, 2026-W24): five
#     flat H2s; the headline-skim block is `## Takeaways` (3-6 bullets,
#     numbers-first); confidence ONLY in the H1 title tag.
#   * v2 (sentinel `<!-- clean-result-v2 -->`, 2026-W22, task #454):
#     `## TL;DR` → ### Motivation / ### What I ran / ### Findings (+ ####
#     per result); confidence ONLY in the H1 title tag.
#   * Legacy (pre-2026-05-13): ### Background / ### Results inside TL;DR +
#     a body `**Confidence: X** — ...` sentence.
# v2/legacy share the `## TL;DR` H2; v3 uses `## Takeaways`. The `^##\s`
# lookahead does not match H3/H4, so nested subsections stay inside the
# captured block.
SENTINEL_V3 = "<!-- clean-result-v3 -->"
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


def fetch_promoted(n: int, prefer_shape: str = "v3") -> list[dict[str, Any]]:
    """Return up to N most-recently-promoted clean-result experiment dicts.

    ``list_by_status`` rows are registry-style (no ``body``, no
    timestamps), so every promoted row is hydrated via ``get_experiment``
    — that supplies the body for headline-skim/confidence extraction and
    ``updatedAt`` (last event ts) for the recency sort. Without the
    hydration step the extractors ran on empty strings and inline mode
    printed only titles + a degenerate "Confidence: ? —" line (#608).

    ``prefer_shape='v3'`` (default) front-loads v3-sentinel bodies so the
    analyzer's few-shot exemplar feed tracks the current shape: the N most
    recent v3 bodies first (recency order), back-filled with the most
    recent non-v3 bodies only if fewer than N v3 bodies exist. Each
    sub-list keeps its own recency order. ``prefer_shape='any'`` restores
    the pre-cutover behavior (pure recency, no shape weighting).
    """
    completed = sagan_state.list_by_status(status="completed", limit=200)
    promoted = [
        sagan_state.get_experiment(e["number"])["experiment"]
        for e in completed
        if e.get("hasCleanResult")
    ]
    promoted.sort(key=lambda e: e.get("updatedAt") or e.get("createdAt") or "", reverse=True)
    if prefer_shape == "v3":
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

    The skim block is ``## Takeaways`` for v3 bodies and ``## TL;DR`` for
    v2 / legacy bodies. Handles v3 (sentinel ``<!-- clean-result-v3 -->``;
    confidence ONLY in the H1 title tag), v2 (``## TL;DR`` → ### Motivation
    / ### What I ran / ### Findings; confidence ONLY in the title tag), and
    legacy bodies (### Background / ### Results + a body
    ``**Confidence: X** — ...`` sentence). The body sentence wins when
    present (legacy); otherwise confidence comes from the title tag.

    The v3 hero is searched whole-body — the ``## Takeaways`` block is
    figure-free by spec (figures live under ``## Findings``), so the first
    inline image in ``## Findings`` is the hero.
    """
    skim_m = RE_MD_TAKEAWAYS.search(body) if is_v3_body(body) else RE_MD_TLDR.search(body)
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
        else:
            # Markdown (v3 + v2 + legacy): print the headline-skim block
            # verbatim (`## Takeaways` for v3, `## TL;DR` for v2/legacy) so
            # the analyzer sees real structure, bounded by --max-chars.
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


def main(argv: list[str] | None = None) -> int:
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
        choices=("v3", "any"),
        default="v3",
        help=(
            "exemplar shape preference: 'v3' (default) front-loads "
            "v3-sentinel bodies so the analyzer's few-shot exemplars track "
            "the current shape; 'any' is pure recency (pre-cutover behavior)"
        ),
    )
    args = p.parse_args(argv)

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
