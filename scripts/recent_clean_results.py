#!/usr/bin/env python3
"""Print the N most-recently-created promoted clean-result issues.

Used by the analyzer agent (Step 1.5) to load in-context exemplars of the
target write-up quality. The promoted clean-results are issues with the
`clean-results` label WITHOUT the `:draft` suffix.

Usage:
    uv run python scripts/recent_clean_results.py --n 3 --format inline
    uv run python scripts/recent_clean_results.py --n 5 --format json

`--format inline` (default) prints, for each clean-result, the issue number,
title, hero figure URL, and the inline TL;DR + Confidence line — suitable
for a one-pass read by an agent. `--format json` emits the full
`gh issue view --json` payloads as a JSON array, for downstream tools that
need structured access.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

DEFAULT_N = 3
# H2 / H3 section heading patterns used to extract the TL;DR sub-blocks.
RE_H2_TLDR = re.compile(r"(?ms)^##\s+TL;DR\s*$(?P<body>.+?)(?=^##\s+|\Z)")
RE_H3_RESULTS = re.compile(r"(?ms)^###\s+Results\s*$(?P<body>.+?)(?=^###\s+|\Z)")
RE_H3_BACKGROUND = re.compile(r"(?ms)^###\s+Background\s*$(?P<body>.+?)(?=^###\s+|\Z)")
RE_HERO = re.compile(r"!\[[^\]]*\]\((https?://\S+?)\)")
RE_CONFIDENCE = re.compile(
    r"\*\*\s*Confidence\s*:\s*(HIGH|MODERATE|LOW)\s*\*\*\s*[—\-–]\s*(?P<text>.+?)$",  # noqa: RUF001
    re.IGNORECASE | re.MULTILINE,
)


def _gh(args: list[str]) -> str:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    return proc.stdout


def fetch_promoted(n: int) -> list[dict]:
    """Return up to N most-recently-created promoted clean-result issue payloads.

    "Promoted" = label `clean-results` without `:draft`. We over-fetch with
    a generous --limit and filter client-side rather than relying on
    label-set negation in the search syntax (which gh's `--search`
    flag does not support reliably for compound label-NOT clauses).
    """
    raw = _gh(
        [
            "issue",
            "list",
            "--label",
            "clean-results",
            "--state",
            "all",
            "--limit",
            "100",
            "--json",
            "number,title,body,labels,createdAt,url",
        ]
    )
    items = json.loads(raw)

    def is_promoted(it: dict) -> bool:
        names = {lab.get("name", "") for lab in it.get("labels", [])}
        return "clean-results" in names and "clean-results:draft" not in names

    promoted = [it for it in items if is_promoted(it)]
    promoted.sort(key=lambda it: it.get("createdAt", ""), reverse=True)
    return promoted[:n]


def render_inline(issues: list[dict]) -> str:
    """Render a compact, agent-readable summary of each promoted clean-result."""
    out: list[str] = []
    for it in issues:
        body = it.get("body", "") or ""
        n = it.get("number")
        title = it.get("title", "")
        url = it.get("url", "")

        tldr_m = RE_H2_TLDR.search(body)
        tldr = tldr_m.group("body").strip() if tldr_m else ""
        bg_m = RE_H3_BACKGROUND.search(tldr)
        background = bg_m.group("body").strip() if bg_m else ""
        results_m = RE_H3_RESULTS.search(tldr)
        results = results_m.group("body").strip() if results_m else ""
        hero_m = RE_HERO.search(results)
        hero = hero_m.group(1) if hero_m else ""
        conf_m = RE_CONFIDENCE.search(results)
        conf_label = conf_m.group(1).upper() if conf_m else "?"
        conf_text = (conf_m.group("text").strip() if conf_m else "").rstrip("*").strip()

        out.append(f"## #{n}: {title}")
        out.append(f"URL: {url}")
        if hero:
            out.append(f"Hero figure: {hero}")
        if background:
            # Compress multi-paragraph background to first 400 chars.
            bg_compact = " ".join(background.split())
            if len(bg_compact) > 400:
                bg_compact = bg_compact[:397] + "..."
            out.append(f"\nBackground: {bg_compact}")
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

    issues = fetch_promoted(args.n)
    if not issues:
        print("# No promoted clean-results found.")
        return 0

    if args.format == "json":
        print(json.dumps(issues, indent=2))
    else:
        print(render_inline(issues))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
