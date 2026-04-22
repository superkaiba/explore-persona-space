"""Pre-publish validator for [Clean Result] issue bodies.

Usage
-----
    uv run python scripts/verify_clean_result.py <path-to-body.md>
    uv run python scripts/verify_clean_result.py --issue <N>

Exits 0 if every check is PASS or WARN; exits 1 if any FAIL.

Checks
------
1. TL;DR structure — 6 H3 subsections in exact order
2. Hero figure — one raw-github commit-pinned image inside ### Results
3. Confidence-tag mirror — each HIGH/MODERATE/LOW claim has a matching "why" bullet
4. Numbers-match-JSON — prose numbers appear in referenced JSON files (WARN only)
5. Reproducibility card — no "{{", "TBD", "see config", "default" sentinels
6. Confidence phrasebook — no ad-hoc "somewhat high" / "fairly low"
7. Support-type tag — every How-updates-me bullet carries support = direct|replicated|...
8. Title — starts with [Clean Result] (only when --issue given)

See .claude/skills/clean-results/checklist.md for the authoritative rules.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

EXPECTED_SUBSECTIONS = [
    "Background",
    "Methodology",
    "Results",
    "How this updates me + confidence",
    "Why confidence is where it is",
    "Next steps",
]

BAD_REPRO_SENTINELS = ("{{", "TBD", "see config", "default", "N/A (no reason")
ADHOC_CONFIDENCE = [
    "somewhat high",
    "fairly low",
    "kind of high",
    "pretty confident",
    "somewhat low",
    "fairly high",
    "kind of low",
]


@dataclass
class Result:
    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    detail: str = ""


@dataclass
class Report:
    results: list[Result] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        if status not in ("PASS", "WARN", "FAIL"):
            raise ValueError(f"unknown status {status!r}")
        self.results.append(Result(name, status, detail))

    def any_fail(self) -> bool:
        return any(r.status == "FAIL" for r in self.results)

    def render(self) -> str:
        width_name = max(len(r.name) for r in self.results) + 2
        lines = []
        lines.append(f"{'Check':<{width_name}}  Status  Detail")
        lines.append("-" * (width_name + 8 + 60))
        for r in self.results:
            icon = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}[r.status]
            lines.append(f"{r.name:<{width_name}}  {icon} {r.status:<4}  {r.detail}")
        return "\n".join(lines)


def _fetch_issue_body(issue_num: int) -> tuple[str, str]:
    """Return ``(title, body)`` for a GitHub issue via the ``gh`` CLI."""
    out = subprocess.run(
        ["gh", "issue", "view", str(issue_num), "--json", "title,body"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"gh issue view #{issue_num} failed (exit {out.returncode}): {out.stderr.strip()}"
        )
    data = json.loads(out.stdout)
    return data["title"], data["body"]


def _extract_section(body: str, heading: str, level: int) -> str | None:
    """Return the content under ``# * heading`` until the next same-or-higher heading."""
    prefix = "#" * level
    pattern = rf"(?m)^{re.escape(prefix)}\s+{re.escape(heading)}\s*$"
    m = re.search(pattern, body)
    if not m:
        return None
    start = m.end()
    # Find next heading at same or higher level
    next_pattern = rf"(?m)^#{{1,{level}}}\s+"
    rest = body[start:]
    n = re.search(next_pattern, rest)
    end = start + (n.start() if n else len(rest))
    return body[start:end]


def check_tldr_structure(body: str, report: Report) -> str | None:
    tldr = _extract_section(body, "TL;DR", level=2)
    if tldr is None:
        report.add("TL;DR structure", "FAIL", "## TL;DR section is missing")
        return None
    headings = re.findall(r"(?m)^###\s+(.+?)\s*$", tldr)
    if headings != EXPECTED_SUBSECTIONS:
        report.add(
            "TL;DR structure",
            "FAIL",
            f"expected {EXPECTED_SUBSECTIONS}, got {headings}",
        )
        return tldr
    report.add("TL;DR structure", "PASS", "6 H3 subsections in correct order")
    return tldr


def check_hero_figure(tldr: str | None, report: Report) -> None:
    if tldr is None:
        report.add("Hero figure", "FAIL", "TL;DR missing; cannot locate Results subsection")
        return
    results_match = re.search(
        r"(?ms)^###\s+Results\s*$(.+?)(?=^###\s+|\Z)",
        tldr,
    )
    if not results_match:
        report.add("Hero figure", "FAIL", "### Results subsection missing")
        return
    results_block = results_match.group(1)
    image_urls = re.findall(r"!\[[^\]]*\]\((\S+?)\)", results_block)
    if not image_urls:
        report.add("Hero figure", "FAIL", "no image inside ### Results")
        return
    if len(image_urls) > 1:
        report.add(
            "Hero figure",
            "WARN",
            f"{len(image_urls)} images inside ### Results — only one should be the hero",
        )
    url = image_urls[0]
    if "raw.githubusercontent.com" not in url:
        report.add("Hero figure", "WARN", f"not a raw.githubusercontent.com URL: {url[:80]}")
        return
    # Detect /main/ or /master/ (unpinned)
    if re.search(r"/(main|master)/", url):
        report.add("Hero figure", "WARN", f"URL not commit-pinned (contains /main/): {url[:80]}")
        return
    # Expect a commit SHA (7-40 hex chars) as one of the path components
    if not re.search(r"/[0-9a-f]{7,40}/", url):
        report.add("Hero figure", "WARN", f"URL lacks a commit SHA segment: {url[:80]}")
        return
    report.add("Hero figure", "PASS", "commit-pinned image present")


def check_confidence_mirror(body: str, report: Report) -> None:
    updates = _extract_section(body, "How this updates me + confidence", level=3)
    why = _extract_section(body, "Why confidence is where it is", level=3)
    if updates is None or why is None:
        report.add("Confidence mirror", "FAIL", "one of the two subsections is missing")
        return
    tag_pattern = r"\b(HIGH|MODERATE|LOW)\b"
    update_tags = re.findall(tag_pattern, updates)
    # Count bullets (lines starting with -) in each
    update_bullets = len(re.findall(r"(?m)^\s*-\s+", updates))
    why_bullets = len(re.findall(r"(?m)^\s*-\s+", why))
    if update_bullets == 0:
        report.add("Confidence mirror", "FAIL", "no bullets in How-updates-me section")
        return
    if len(update_tags) < update_bullets:
        report.add(
            "Confidence mirror",
            "FAIL",
            f"{update_bullets} bullets but only {len(update_tags)} HIGH/MODERATE/LOW tags",
        )
        return
    if abs(update_bullets - why_bullets) > 1:
        report.add(
            "Confidence mirror",
            "WARN",
            f"{update_bullets} updates vs {why_bullets} why-bullets — mismatch > 1",
        )
        return
    report.add(
        "Confidence mirror",
        "PASS",
        f"{update_bullets} updates mirrored by {why_bullets} why-bullets",
    )


def check_numbers_in_json(body: str, report: Report) -> None:
    """Cross-reference numeric prose claims against any JSON artifact paths."""
    # Collect JSON paths mentioned in the body
    json_paths = re.findall(r"`([^`]+\.json)`", body)
    json_paths = [p for p in json_paths if not p.startswith("wandb://")]
    existing = [Path(p) for p in json_paths if Path(p).exists()]
    if not existing:
        report.add("Numbers match JSON", "PASS", "no JSON artifacts referenced — skipped")
        return

    numbers_in_prose: set[str] = set()
    for m in re.finditer(r"(?<!\d)(\d+\.\d+)(?!\d)", body):
        numbers_in_prose.add(m.group(1))
    if not numbers_in_prose:
        report.add("Numbers match JSON", "PASS", "no numeric prose claims to verify")
        return

    combined = ""
    for path in existing:
        try:
            combined += path.read_text()
        except OSError as exc:
            report.add("Numbers match JSON", "WARN", f"could not read {path}: {exc}")

    unmatched = [
        n
        for n in numbers_in_prose
        if n not in combined and n.rstrip("0").rstrip(".") not in combined
    ]
    if unmatched:
        sample = ", ".join(sorted(unmatched)[:5])
        report.add(
            "Numbers match JSON",
            "WARN",
            f"{len(unmatched)} numeric claims not found in referenced JSON (e.g. {sample})",
        )
        return
    report.add(
        "Numbers match JSON",
        "PASS",
        f"all {len(numbers_in_prose)} numeric claims found in {len(existing)} JSONs",
    )


def check_reproducibility(body: str, report: Report) -> None:
    setup = _extract_section(body, "Setup & hyper-parameters", level=2)
    if setup is None:
        report.add("Reproducibility card", "FAIL", "## Setup & hyper-parameters section missing")
        return
    offenders = []
    for line in setup.splitlines():
        # Only inspect table rows (start with `|`) and skip the header separator
        stripped = line.strip()
        if not stripped.startswith("|") or set(stripped) <= {"|", "-", " ", ":"}:
            continue
        for sentinel in BAD_REPRO_SENTINELS:
            if sentinel in line:
                offenders.append((sentinel, line.strip()[:80]))
                break
    if offenders:
        sample = "; ".join(f"{s!r} in {line!r}" for s, line in offenders[:3])
        report.add(
            "Reproducibility card",
            "FAIL",
            f"{len(offenders)} unfilled rows (e.g. {sample})",
        )
        return
    report.add("Reproducibility card", "PASS", "no unfilled sentinels found")


def check_confidence_phrasebook(body: str, report: Report) -> None:
    offenders = [w for w in ADHOC_CONFIDENCE if w in body.lower()]
    if offenders:
        report.add(
            "Confidence phrasebook",
            "WARN",
            f"ad-hoc confidence hedge(s) used: {offenders}",
        )
        return
    report.add("Confidence phrasebook", "PASS", "no ad-hoc hedges detected")


def check_support_type(body: str, report: Report) -> None:
    updates = _extract_section(body, "How this updates me + confidence", level=3)
    if updates is None:
        report.add("Support-type tags", "FAIL", "How-updates-me section missing")
        return
    bullets = re.findall(r"(?m)^\s*-\s+.+?(?=(?:\n\s*-\s+)|\Z)", updates, flags=re.DOTALL)
    if not bullets:
        report.add("Support-type tags", "FAIL", "no bullets")
        return
    missing = 0
    for b in bullets:
        if not re.search(
            r"support\s*=\s*(direct|replicated|external|intuition|shallow)",
            b,
            flags=re.IGNORECASE,
        ):
            missing += 1
    if missing:
        report.add(
            "Support-type tags", "FAIL", f"{missing}/{len(bullets)} bullets lack support= tag"
        )
        return
    report.add("Support-type tags", "PASS", f"all {len(bullets)} bullets tagged")


def check_title(title: str | None, report: Report) -> None:
    if title is None:
        return  # skip when run on a file
    if not title.startswith("[Clean Result]"):
        report.add("Title prefix", "FAIL", f"title does not start with '[Clean Result]': {title!r}")
        return
    report.add("Title prefix", "PASS", "title starts with [Clean Result]")


def run_all_checks(title: str | None, body: str) -> Report:
    report = Report()
    tldr = check_tldr_structure(body, report)
    check_hero_figure(tldr, report)
    check_confidence_mirror(body, report)
    check_numbers_in_json(body, report)
    check_reproducibility(body, report)
    check_confidence_phrasebook(body, report)
    check_support_type(body, report)
    check_title(title, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("path", nargs="?", help="Path to a clean-result body markdown file")
    group.add_argument("--issue", type=int, help="Fetch body via gh issue view <N>")
    args = parser.parse_args(argv)

    if args.issue is not None:
        title, body = _fetch_issue_body(args.issue)
    else:
        body_path = Path(args.path)
        if not body_path.exists():
            print(f"Error: {body_path} does not exist", file=sys.stderr)
            return 2
        title = None
        body = body_path.read_text()

    report = run_all_checks(title, body)
    print(report.render())
    if report.any_fail():
        print("\nResult: FAIL — fix the failing checks before posting.")
        return 1
    print("\nResult: PASS (WARNs acknowledged or in Caveats).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
