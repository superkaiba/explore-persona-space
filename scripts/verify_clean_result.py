"""Pre-publish validator for clean-result issue bodies.

Usage
-----
    uv run python scripts/verify_clean_result.py <path-to-body.md>
    uv run python scripts/verify_clean_result.py --issue <N>

Exits 0 if every check is PASS or WARN; exits 1 if any FAIL.

Checks
------
1. TL;DR structure — 4 H3 subsections in exact order (Background, Methodology,
   Results, Next steps).
2. Hero figure — one raw-github commit-pinned image inside ### Results.
3. Results block shape — ### Results contains a `**Main takeaways:**` label
   with at least one bullet beneath it, followed by a single
   `**Confidence: HIGH|MODERATE|LOW** — …` line.
4. Numbers-match-JSON — prose numbers appear in referenced JSON files (WARN
   only).
5. Reproducibility card — no "{{", "TBD", "see config", "default" sentinels in
   ## Setup & hyper-parameters tables.
6. Confidence phrasebook — no ad-hoc "somewhat high" / "fairly low".
7. Stats framing — no effect-size / named-test / credence-interval language.
8. Title confidence marker — title ends with `(HIGH|MODERATE|LOW confidence)`
   matching the Results Confidence line (only when title is provided).
9. Human summary — `## Human summary` H2 present, non-empty, >=30 words,
   no sentinels (skipped on issues >7 days old or already-promoted).
10. Sample outputs — `## Sample outputs` H2 present with at least one
    `### Condition: <name>` H3 subsection, each containing >=3 fenced
    code blocks (skipped on grandfathered issues).

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
    "Next steps",
]

BAD_REPRO_SENTINELS = ("{{", "TBD", "see config", "default", "N/A (no reason")

# Sentinels for the Human summary check (item 5 / AC5).
HUMAN_SUMMARY_SENTINELS = (
    "{{",
    "TBD",
    "...",
    "…",
    "<TODO>",
    "<placeholder>",
    "XXX",
    "FIXME",
    "n/a",
    "N/A",
)
MIN_HUMAN_SUMMARY_WORDS = 30

ADHOC_CONFIDENCE = [
    "somewhat high",
    "fairly low",
    "kind of high",
    "pretty confident",
    "somewhat low",
    "fairly high",
    "kind of low",
]

# Forbidden statistical-framing language (project convention: p-values only).
# Each tuple: (regex, human label).
FORBIDDEN_STATS_PATTERNS: list[tuple[str, str]] = [
    (r"\bcohen[''\s]*s?\s*d\b", "Cohen's d"),
    (r"\beffect\s+size", "'effect size'"),
    (r"\bpaired\s+t[-\s]?test", "named paired t-test"),
    (r"\bfisher[''\s]*s?\s+exact", "Fisher's exact"),
    (r"\bmann[-\s]?whitney", "Mann-Whitney"),
    (r"\bwilcoxon", "Wilcoxon"),
    (r"\bbootstrap\s+(ci|confidence|interval|resampl)", "bootstrap CI"),
    (r"\b(η|eta)²", "η²"),
    (r"\bpower\s+analysis", "power analysis"),
    (r"\bcredence\s+interval", "credence interval in prose"),
    (r"\bminimum\s+detectable\s+effect", "minimum detectable effect"),
]

# Single confidence line at the bottom of ### Results:
# e.g. `**Confidence: LOW** — because n=3 …`
CONFIDENCE_LINE_PATTERN = re.compile(
    r"\*\*\s*Confidence\s*:\s*(HIGH|MODERATE|LOW)\s*\*\*\s*[—\-–]",  # noqa: RUF001
    re.IGNORECASE,
)

MAIN_TAKEAWAYS_PATTERN = re.compile(
    r"\*\*\s*Main\s+takeaways\s*:\s*\*\*",
    re.IGNORECASE,
)

# Title-level confidence marker: ends with `(HIGH confidence)` etc.
TITLE_CONFIDENCE_PATTERN = re.compile(
    r"\(\s*(HIGH|MODERATE|LOW)\s+confidence\s*\)\s*$",
    re.IGNORECASE,
)


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


def _fetch_issue_body(issue_num: int) -> tuple[str, str, list[str], str]:
    """Return ``(title, body, label_names, created_at)`` for a GitHub issue.

    ``label_names`` is a flat list of label names (so the date-gate can check
    for ``clean-results`` / ``clean-results:draft``); ``created_at`` is the
    ISO-8601 timestamp string straight from the GitHub API.
    """
    out = subprocess.run(
        ["gh", "issue", "view", str(issue_num), "--json", "title,body,labels,createdAt"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"gh issue view #{issue_num} failed (exit {out.returncode}): {out.stderr.strip()}"
        )
    data = json.loads(out.stdout)
    label_names = [lab.get("name", "") for lab in data.get("labels", [])]
    return data["title"], data["body"], label_names, data.get("createdAt", "")


def _extract_section(body: str, heading: str, level: int) -> str | None:
    """Return the content under ``# * heading`` until the next same-or-higher heading."""
    prefix = "#" * level
    pattern = rf"(?m)^{re.escape(prefix)}\s+{re.escape(heading)}\s*$"
    m = re.search(pattern, body)
    if not m:
        return None
    start = m.end()
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
    report.add("TL;DR structure", "PASS", "4 H3 subsections in correct order")
    return tldr


def _extract_results_block(tldr: str | None) -> str | None:
    if tldr is None:
        return None
    m = re.search(r"(?ms)^###\s+Results\s*$(.+?)(?=^###\s+|\Z)", tldr)
    return m.group(1) if m else None


def check_hero_figure(tldr: str | None, report: Report) -> None:
    results_block = _extract_results_block(tldr)
    if results_block is None:
        report.add("Hero figure", "FAIL", "### Results subsection missing")
        return
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
    if re.search(r"/(main|master)/", url):
        report.add("Hero figure", "WARN", f"URL not commit-pinned (contains /main/): {url[:80]}")
        return
    if not re.search(r"/[0-9a-f]{7,40}/", url):
        report.add("Hero figure", "WARN", f"URL lacks a commit SHA segment: {url[:80]}")
        return
    report.add("Hero figure", "PASS", "commit-pinned image present")


def check_results_block(tldr: str | None, report: Report) -> None:
    """Verify Results has a Main takeaways block with bullets + exactly one Confidence line."""
    results_block = _extract_results_block(tldr)
    if results_block is None:
        report.add("Results block shape", "FAIL", "### Results subsection missing")
        return

    mt = MAIN_TAKEAWAYS_PATTERN.search(results_block)
    if not mt:
        report.add(
            "Results block shape",
            "FAIL",
            "missing `**Main takeaways:**` bolded label inside ### Results",
        )
        return

    # Count bullets after Main takeaways label but before the Confidence line
    # (or end of block if no Confidence line yet).
    after_label = results_block[mt.end() :]
    conf_m = CONFIDENCE_LINE_PATTERN.search(after_label)
    bullets_region = after_label[: conf_m.start()] if conf_m else after_label
    bullets = re.findall(r"(?m)^\s*-\s+\S", bullets_region)
    if not bullets:
        report.add(
            "Results block shape",
            "FAIL",
            "no bullets under `**Main takeaways:**`",
        )
        return

    confidence_hits = CONFIDENCE_LINE_PATTERN.findall(results_block)
    if len(confidence_hits) == 0:
        report.add(
            "Results block shape",
            "FAIL",
            "missing `**Confidence: HIGH|MODERATE|LOW** — <sentence>` line at end of Results",
        )
        return
    if len(confidence_hits) > 1:
        report.add(
            "Results block shape",
            "WARN",
            f"{len(confidence_hits)} Confidence lines inside Results — expected 1",
        )
        return

    report.add(
        "Results block shape",
        "PASS",
        f"Main takeaways with {len(bullets)} bullet(s) + 1 Confidence line",
    )


def check_numbers_in_json(body: str, report: Report) -> None:
    """Cross-reference numeric prose claims against any JSON artifact paths."""
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


def check_forbidden_stats(body: str, report: Report) -> None:
    """Flag forbidden statistical-framing language (effect sizes, named tests, etc.)."""
    hits: list[str] = []
    for pattern, label in FORBIDDEN_STATS_PATTERNS:
        if re.search(pattern, body, flags=re.IGNORECASE):
            hits.append(label)
    if hits:
        report.add(
            "Stats framing (p-values only)",
            "FAIL",
            f"forbidden language: {', '.join(hits)}",
        )
        return
    report.add(
        "Stats framing (p-values only)",
        "PASS",
        "no effect-size / named-test / credence-interval language",
    )


def _results_confidence_level(body: str) -> str | None:
    """Return the HIGH/MODERATE/LOW from the Results block's Confidence line, if any."""
    tldr = _extract_section(body, "TL;DR", level=2)
    results_block = _extract_results_block(tldr)
    if results_block is None:
        return None
    m = CONFIDENCE_LINE_PATTERN.search(results_block)
    return m.group(1).upper() if m else None


def check_title(title: str | None, body: str, report: Report) -> None:
    """Title must end with `(HIGH|MODERATE|LOW confidence)` matching the Results line."""
    if title is None:
        return
    m = TITLE_CONFIDENCE_PATTERN.search(title)
    if not m:
        report.add(
            "Title confidence marker",
            "FAIL",
            f"title does not end with '(HIGH|MODERATE|LOW confidence)': {title!r}",
        )
        return
    title_level = m.group(1).upper()
    body_level = _results_confidence_level(body)
    if body_level is None:
        report.add(
            "Title confidence marker",
            "WARN",
            f"title says ({title_level} confidence) but Results has no Confidence line to match",
        )
        return
    if title_level != body_level:
        report.add(
            "Title confidence marker",
            "FAIL",
            f"title says ({title_level} confidence) but Results says {body_level}",
        )
        return
    report.add(
        "Title confidence marker",
        "PASS",
        f"title ends with ({title_level} confidence), matches Results",
    )


MIN_BACKGROUND_WORDS = 30


def check_background_context(tldr: str | None, report: Report) -> None:
    """WARN if Background subsection is too terse for newcomers (<30 words)."""
    if tldr is None:
        return
    bg = _extract_section(tldr, "Background", level=3)
    if bg is None:
        report.add("Background context", "WARN", "### Background subsection missing from TL;DR")
        return
    word_count = len(bg.split())
    if word_count < MIN_BACKGROUND_WORDS:
        report.add(
            "Background context",
            "WARN",
            f"Background has {word_count} words (minimum {MIN_BACKGROUND_WORDS}) — "
            "may be too terse for readers unfamiliar with the project",
        )
        return
    report.add("Background context", "PASS", f"Background has {word_count} words")


def _is_low_content(s: str) -> bool:
    """Catch degenerate inputs that pass the sentinel check.

    Returns True if the section is effectively empty: no characters, mostly
    non-letter characters (e.g. punctuation-only), or only empty bullet rows.
    """
    s = s.strip()
    if len(s) == 0:
        return True
    letters = sum(1 for c in s if c.isalpha())
    if len(s) > 0 and letters / len(s) < 0.5:
        return True
    return all(line.strip() in ("", "-") for line in s.splitlines())


def check_human_summary(body: str, report: Report, *, strict: bool = True) -> None:
    """`## Human summary` H2 must be present, non-sentinel, >=30 words.

    When ``strict=False`` (grandfathered: issue >7 days old or already-promoted),
    a missing section is downgraded to WARN.
    """
    section = _extract_section(body, "Human summary", level=2)
    if section is None:
        if strict:
            report.add(
                "Human summary",
                "FAIL",
                "## Human summary section missing (must appear at top of Detailed report)",
            )
        else:
            report.add(
                "Human summary",
                "WARN",
                "## Human summary missing (grandfathered: issue >7 days old or already-promoted)",
            )
        return
    stripped = section.strip()
    for sentinel in HUMAN_SUMMARY_SENTINELS:
        if sentinel in stripped:
            report.add(
                "Human summary",
                "FAIL",
                f"## Human summary contains sentinel {sentinel!r}",
            )
            return
    if _is_low_content(stripped):
        report.add(
            "Human summary",
            "FAIL",
            "## Human summary is low-content (empty / mostly non-letters / empty bullets)",
        )
        return
    word_count = len(stripped.split())
    if word_count < MIN_HUMAN_SUMMARY_WORDS:
        report.add(
            "Human summary",
            "FAIL",
            (
                f"## Human summary is too short ({word_count} words; "
                f"minimum {MIN_HUMAN_SUMMARY_WORDS})"
            ),
        )
        return
    report.add("Human summary", "PASS", f"{word_count} words")


def check_sample_outputs(body: str, report: Report, *, strict: bool = True) -> None:
    """`## Sample outputs` must contain >=1 `### Condition:` H3 with >=3 fenced blocks each.

    When ``strict=False`` (grandfathered: issue >7 days old or already-promoted),
    a missing section is downgraded to WARN.
    """
    section = _extract_section(body, "Sample outputs", level=2)
    if section is None:
        if strict:
            report.add("Sample outputs", "FAIL", "## Sample outputs section missing")
        else:
            report.add(
                "Sample outputs",
                "WARN",
                "## Sample outputs missing (grandfathered)",
            )
        return
    # Split on `### Condition:` H3 subsections; ignore prose before the first.
    condition_blocks = re.split(r"^### Condition:", section, flags=re.MULTILINE)[1:]
    if not condition_blocks:
        report.add(
            "Sample outputs",
            "FAIL",
            "## Sample outputs has no `### Condition:` H3 subsections",
        )
        return
    bad: list[str] = []
    for blk in condition_blocks:
        # Name = trimmed first line of the condition block.
        name = blk.split("\n", 1)[0].strip()
        n_fenced = len(re.findall(r"```[\s\S]+?```", blk))
        if n_fenced < 3:
            bad.append(f"{name!r}: {n_fenced} fenced block(s)")
    if bad:
        report.add(
            "Sample outputs",
            "FAIL",
            f"Conditions with <3 sample blocks: {'; '.join(bad)}",
        )
        return
    report.add(
        "Sample outputs",
        "PASS",
        f"{len(condition_blocks)} condition(s), each with >=3 fenced sample blocks",
    )


def run_all_checks(title: str | None, body: str, *, strict: bool = True) -> Report:
    report = Report()
    tldr = check_tldr_structure(body, report)
    check_hero_figure(tldr, report)
    check_results_block(tldr, report)
    check_background_context(tldr, report)
    check_human_summary(body, report, strict=strict)
    check_sample_outputs(body, report, strict=strict)
    check_numbers_in_json(body, report)
    check_reproducibility(body, report)
    check_confidence_phrasebook(body, report)
    check_forbidden_stats(body, report)
    check_title(title, body, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("path", nargs="?", help="Path to a clean-result body markdown file")
    group.add_argument("--issue", type=int, help="Fetch body via gh issue view <N>")
    args = parser.parse_args(argv)

    if args.issue is not None:
        title, body, label_names, created_at = _fetch_issue_body(args.issue)
        # Date-gate: skip Human summary / Sample outputs strict checks for
        # issues >7 days old or already-promoted (clean-results without :draft).
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        try:
            created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            created_dt = now  # fall back to strict if parsing fails
        age = now - created_dt
        is_promoted = "clean-results" in label_names and "clean-results:draft" not in label_names
        strict = (age <= timedelta(days=7)) and (not is_promoted)
    else:
        body_path = Path(args.path)
        if not body_path.exists():
            print(f"Error: {body_path} does not exist", file=sys.stderr)
            return 2
        title = None
        body = body_path.read_text()
        strict = True  # file mode is always strict

    report = run_all_checks(title, body, strict=strict)
    print(report.render())
    if report.any_fail():
        print("\nResult: FAIL — fix the failing checks before posting.")
        return 1
    print("\nResult: PASS (WARNs acknowledged).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
