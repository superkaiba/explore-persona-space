#!/usr/bin/env python3
"""Print the N most-recently-created promoted clean-result experiments.

Used by the analyzer agent (Step 1.5) to load in-context exemplars of the
target write-up quality. Promoted clean-results are Sagan experiments
with ``hasCleanResult=true`` and ``status='completed'`` (the analyzer
flips ``hasCleanResult`` after the reviewer passes; the user advances to
``completed`` via the promote command).

Usage:
    uv run python scripts/recent_clean_results.py --n 3 --format inline
    uv run python scripts/recent_clean_results.py --n 5 --format json

``--format inline`` (default) prints, for each clean-result, the
experiment number, title, hero figure URL (if extractable), and a
compact TL;DR + Confidence line — suitable for one-pass agent reading.
``--format json`` emits the raw experiment payloads from
``sagan_state.list_by_status`` for downstream tools.

Implementation: queries Sagan's HTTP API via :mod:`sagan_state`.
``GET /api/experiments`` does not yet support a ``hasCleanResult=true``
filter, so we list completed experiments and filter client-side.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sagan_state

DEFAULT_N = 3

# Legacy markdown bodies (pre-2026-05-13). New bodies are Sagan-card HTML.
RE_MD_TLDR = re.compile(r"(?ms)^##\s+TL;DR\s*$(?P<body>.+?)(?=^##\s+|\Z)")
RE_MD_RESULTS = re.compile(r"(?ms)^###\s+Results\s*$(?P<body>.+?)(?=^###\s+|\Z)")
RE_MD_BACKGROUND = re.compile(r"(?ms)^###\s+Background\s*$(?P<body>.+?)(?=^###\s+|\Z)")
RE_MD_HERO = re.compile(r"!\[[^\]]*\]\((https?://\S+?)\)")
RE_MD_CONFIDENCE = re.compile(
    r"\*\*\s*Confidence\s*:\s*(HIGH|MODERATE|LOW)\s*\*\*\s*[—\-–]\s*(?P<text>.+?)$",  # noqa: RUF001
    re.IGNORECASE | re.MULTILINE,
)

# Sagan-card HTML bodies.
RE_HTML_TLDR = re.compile(r'(?is)<section[^>]+id="tldr"[^>]*>(?P<body>.*?)</section>')
RE_HTML_FIGURE_IMG = re.compile(
    r'(?is)<figure[^>]+id="figure"[^>]*>.*?<img[^>]+src="(?P<src>[^"]+)"'
)
RE_HTML_CONFIDENCE = re.compile(
    r"(?is)Confidence\s*:\s*(?P<label>HIGH|MODERATE|LOW)\s*[—\-–]\s*(?P<text>.+?)(?:<|\.)",  # noqa: RUF001
)


def fetch_promoted(n: int) -> list[dict[str, Any]]:
    """Return up to N most-recently-promoted clean-result experiment dicts."""
    completed = sagan_state.list_by_status(status="completed", limit=200)
    promoted = [e for e in completed if e.get("hasCleanResult")]
    promoted.sort(key=lambda e: e.get("updatedAt") or e.get("createdAt") or "", reverse=True)
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


def _extract_markdown(body: str) -> tuple[str, str, str, str]:
    """Return (background, hero_url, confidence_label, confidence_text) for legacy bodies."""
    tldr_m = RE_MD_TLDR.search(body)
    tldr = tldr_m.group("body").strip() if tldr_m else ""
    bg_m = RE_MD_BACKGROUND.search(tldr)
    background = bg_m.group("body").strip() if bg_m else ""
    results_m = RE_MD_RESULTS.search(tldr)
    results = results_m.group("body").strip() if results_m else ""
    hero_m = RE_MD_HERO.search(results)
    hero = hero_m.group(1) if hero_m else ""
    conf_m = RE_MD_CONFIDENCE.search(results)
    conf_label = conf_m.group(1).upper() if conf_m else "?"
    conf_text = (conf_m.group("text").strip() if conf_m else "").rstrip("*").strip()
    return background, hero, conf_label, conf_text


def render_inline(experiments: list[dict[str, Any]]) -> str:
    base = sagan_state.BASE_URL
    out: list[str] = []
    for exp in experiments:
        body = exp.get("body") or ""
        number = exp.get("number", "?")
        title = exp.get("title", "")
        uuid = exp.get("id", "")
        url = f"{base}/e/experiment/{uuid}"

        if "<section" in body.lower() and 'id="tldr"' in body.lower():
            tldr, hero, conf_label, conf_text = _extract_html(body)
            background = tldr
        else:
            background, hero, conf_label, conf_text = _extract_markdown(body)

        out.append(f"## #{number}: {title}")
        out.append(f"URL: {url}")
        if hero:
            out.append(f"Hero figure: {hero}")
        if background:
            compact = " ".join(background.split())
            if len(compact) > 400:
                compact = compact[:397] + "..."
            out.append(f"\nSummary: {compact}")
        out.append(f"\nConfidence: {conf_label} — {conf_text}")
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
    args = p.parse_args(argv)

    experiments = fetch_promoted(args.n)
    if not experiments:
        print("# No promoted clean-results found.")
        return 0

    if args.format == "json":
        json.dump(experiments, sys.stdout, indent=2, default=str)
        print()
    else:
        print(render_inline(experiments))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
