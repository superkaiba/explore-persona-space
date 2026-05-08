"""Pre-publish validator for clean-result issue bodies.

Usage
-----
    uv run python scripts/verify_clean_result.py <path-to-body.md>
    uv run python scripts/verify_clean_result.py --issue <N>
    uv run python scripts/verify_clean_result.py <path> --skip-checks <name1>,<name2>

Exits 0 if every check is PASS or WARN; exits 1 if any FAIL.

Checks
------
0a. Human TL;DR — top ``## Human TL;DR`` H2 must be present. Content is NOT
    enforced (the user fills this in by hand; drafts may keep the literal
    placeholder line). Date-gated for legacy issues.
0b. AI TL;DR paragraph — ``## AI TL;DR`` is a 3-5 sentence LW-style paragraph
    (setup → headline finding → why it matters → scope/limitation). >=30
    words, <=200 words, >=3 sentences, no sentinels. Date-gated for legacy
    issues created before ``SUMMARY_RENAME_DATE``.
1. AI Summary structure — 4 H3 subsections in exact order (Background,
   Methodology, Results, Next steps) under ``## AI Summary``. Legacy issues
   authored before ``SUMMARY_RENAME_DATE`` may keep the block under
   ``## TL;DR`` and still PASS.
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
11. AI Summary acronyms (#275 item 4 / 9) — H1/H2/H3/P1/P2/P3 must be
    defined inline on first use anywhere inside ``## AI Summary`` (or
    ``## TL;DR`` for legacy issues). Fenced code blocks and inline backticks
    are exempt. Grandfathered for issues >7 days old or already-promoted.
12. Background motivation (#275 item 5 / 11) — ### Background must
    contain at least one `#<issue>` reference distinct from the current
    issue. Grandfathered for old/promoted issues.
13. AI Summary dataset example (#275 item 13) — ### Methodology must contain
    a fenced code-block example or a `**Dataset example:**` bullet AND
    the AI Summary must contain a wandb.ai / wandb:// / huggingface.co
    full-data link. Skipped when the issue carries the `no-dataset`
    label. Literal `**Dataset example:** N/A` is rejected.
14. Results figure captions (#293 §1) — every figure inside ### Results
    is followed by a caption paragraph (>=10 words after stripping
    bullets / inline-link URLs). HARD FAIL when a caption is missing or
    short. Date-gated (``CAPTION_CHECK_ENFORCEMENT_DATE``) so issues
    created before the gate downgrade FAIL to WARN.

See .claude/skills/clean-results/checklist.md for the authoritative rules.

The `--skip-checks <name1>,<name2>` flag lets callers bypass specific
check functions for a single invocation; each skipped check logs
`SKIPPED: <name> (--skip-checks)` to stderr.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

EXPECTED_SUBSECTIONS = [
    "Background",
    "Methodology",
    "Results",
    "Next steps",
]

BAD_REPRO_SENTINELS = ("{{", "TBD", "see config", "default", "N/A (no reason")

# Methodology bullet-form requirement (slice 7 of #251).
# One-time grandfathering boundary for #251 slice 7. Drafts created before
# this date use prose Methodology; from this date on, bullet form is
# required. The cutoff stays in code permanently — bumping it would
# re-grandfather drafts the convention has already moved past. If the
# review of #251 itself extends past 2026-05-15, bump this constant by a
# matching number of days during the PR rebase so in-flight prose-form
# drafts created in the slip window stay grandfathered.
REQUIRED_METHODOLOGY_BULLETS = ["**Model:**", "**Dataset:**", "**Eval:**", "**Stats:**"]
METHODOLOGY_BULLETS_REQUIRED_AFTER = datetime(2026, 5, 15, tzinfo=UTC)

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

# ---- #293 §1: Results-block figure-caption check ------------------------------

RESULTS_CAPTION_MIN_WORDS = 10
"""Minimum word count for a Results-block figure caption.

The threshold is calibrated against the canonical exemplar (issue #75): #75's
caption clocks ~45 words for the first sentence alone, and the Tulu-25 fixture
caption ("Tulu-25 achieves 87.9% alignment vs baseline 70.4% across n=3 seeds.")
tokenises to exactly 10 words. 10 is the smallest value that admits a
one-sentence caption that names panels, axes, and N — anything shorter is more
likely a stray label than a real caption. See clean-results/SKILL.md
invariant 2 for the contract.
"""

CAPTION_CHECK_ENFORCEMENT_DATE = "2026-05-06"
"""Issues with ``created_at < CAPTION_CHECK_ENFORCEMENT_DATE`` get WARN instead
of FAIL on a missing/short Results-block caption. Lets legacy
``clean-results:useful`` / ``clean-results:not-useful`` issues continue to PASS
the verifier post-merge. The follow-up issue (audit + retrofit captions on
legacy issues) retires the date-gate after backfill. Set to the PR's open-for-
review date (ISO YYYY-MM-DD)."""

# ---- TL;DR / Summary split (Human / AI / AI Summary) -------------------------

SUMMARY_RENAME_DATE = "2026-05-07"
"""Date the structured 4-H3-subsection block moved from ``## TL;DR`` to
``## AI Summary``, a new ``## AI TL;DR`` LW-style paragraph was added above
it, and ``## Human TL;DR`` was added at the very top as a user-only section.
Issues with ``created_at < SUMMARY_RENAME_DATE`` are allowed to keep the
structured block under ``## TL;DR`` and skip the new Human TL;DR / AI TL;DR
checks. File-mode (no created_at) is always strict: fresh-from-template
drafts must follow the new shape."""

# Sentinels for the AI TL;DR-paragraph check (LW-style summary paragraph).
AI_TLDR_PARAGRAPH_SENTINELS = HUMAN_SUMMARY_SENTINELS  # reuse the same set
MIN_AI_TLDR_PARAGRAPH_WORDS = 30
MAX_AI_TLDR_PARAGRAPH_WORDS = 200
MIN_AI_TLDR_PARAGRAPH_SENTENCES = 3

# The literal placeholder the analyzer leaves in `## Human TL;DR` for the
# user to overwrite by hand. The Human TL;DR check accepts this verbatim and
# does NOT enforce content beyond H2 presence.
HUMAN_TLDR_PLACEHOLDER = (
    "_(Human TL;DR — to be filled in by the user. Leave this line as-is in drafts.)_"
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


def _extract_summary_section(body: str) -> tuple[str | None, str]:
    """Return ``(content, section_kind)`` for the structured 4-H3 block.

    Looks for ``## AI Summary`` first (post-rename canonical name); if absent,
    falls back to ``## TL;DR`` (legacy / pre-SUMMARY_RENAME_DATE shape).
    Returns ``(None, "missing")`` if neither is present.

    ``section_kind`` is one of:
      - ``"ai-summary"`` — found under ``## AI Summary`` (current shape)
      - ``"legacy-tldr"`` — found under ``## TL;DR`` (pre-rename shape)
      - ``"missing"`` — neither header present
    """
    section = _extract_section(body, "AI Summary", level=2)
    if section is not None:
        return section, "ai-summary"
    section = _extract_section(body, "TL;DR", level=2)
    if section is not None:
        return section, "legacy-tldr"
    return None, "missing"


def check_tldr_structure(body: str, report: Report) -> str | None:
    """Verify the structured 4-H3-subsection block.

    Post-rename: lives under ``## AI Summary``. Pre-rename (legacy):
    lives under ``## TL;DR``. The check accepts either; downstream
    section-shape checks (hero figure, Results block, methodology
    bullets, etc.) operate on whichever was found.
    """
    section, kind = _extract_summary_section(body)
    if section is None:
        report.add(
            "AI Summary structure",
            "FAIL",
            "neither ## AI Summary nor ## TL;DR section found",
        )
        return None
    headings = re.findall(r"(?m)^###\s+(.+?)\s*$", section)
    if headings != EXPECTED_SUBSECTIONS:
        report.add(
            "AI Summary structure",
            "FAIL",
            f"expected {EXPECTED_SUBSECTIONS}, got {headings}",
        )
        return section
    if kind == "legacy-tldr":
        report.add(
            "AI Summary structure",
            "PASS",
            "4 H3 subsections in correct order (legacy ## TL;DR — rename to ## AI Summary)",
        )
    else:
        report.add(
            "AI Summary structure",
            "PASS",
            "4 H3 subsections in correct order",
        )
    return section


def _extract_ai_tldr_paragraph(body: str) -> str | None:
    """Return the body of ``## AI TL;DR`` if present, else None."""
    return _extract_section(body, "AI TL;DR", level=2)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'_*\[])")


def _strip_template_placeholders(text: str) -> str:
    """Remove ``{{...}}`` placeholders + HTML comments + bracketed link markdown.

    Used by the AI TL;DR word/sentence count so an unfilled template stub
    doesn't trip the >=30-word check via its placeholder text.
    """
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    return text


def check_ai_tldr_paragraph(
    body: str,
    report: Report,
    *,
    issue_created_at: str | None = None,
) -> None:
    """``## AI TL;DR`` is a 3-5 sentence/bullet LW-style summary.

    Enforces: presence of H2, no sentinels, content >= MIN words and (>=
    MIN sentences in paragraph form OR 3-5 top-level bullets in bullet
    form), content <= MAX words. Either format is acceptable — the
    LessWrong / Alignment Forum house style uses both, and the bullet
    form is often clearer for multi-beat results (setup / headline / why
    it matters / scope). Grandfathered (PASS, missing OK) when EITHER the
    issue was created before ``SUMMARY_RENAME_DATE`` OR the body's
    structured block still lives under legacy ``## TL;DR`` (the legacy
    shape uses ``## TL;DR`` for the structured block, which would be
    ambiguous with the new top-of-body short summary).
    """
    is_legacy_by_date = issue_created_at is not None and issue_created_at[:10] < SUMMARY_RENAME_DATE
    _, summary_kind = _extract_summary_section(body)
    is_legacy_by_shape = summary_kind == "legacy-tldr"
    is_legacy = is_legacy_by_date or is_legacy_by_shape
    section = _extract_ai_tldr_paragraph(body)
    if section is None:
        if is_legacy:
            report.add(
                "AI TL;DR paragraph",
                "PASS",
                "section missing (legacy issue, pre-rename — grandfathered)",
            )
            return
        report.add(
            "AI TL;DR paragraph",
            "FAIL",
            "## AI TL;DR section is missing",
        )
        return
    # Check sentinels on the raw section first (don't strip placeholders here —
    # an unfilled `{{...}}` stub IS a sentinel).
    for sentinel in AI_TLDR_PARAGRAPH_SENTINELS:
        if sentinel in section:
            report.add(
                "AI TL;DR paragraph",
                "FAIL",
                f"## AI TL;DR contains sentinel {sentinel!r}",
            )
            return
    cleaned = _strip_template_placeholders(section).strip()
    if not cleaned:
        report.add(
            "AI TL;DR paragraph",
            "FAIL",
            "## AI TL;DR is empty after stripping placeholders / comments",
        )
        return
    word_count = len(cleaned.split())
    if word_count < MIN_AI_TLDR_PARAGRAPH_WORDS:
        report.add(
            "AI TL;DR paragraph",
            "FAIL",
            f"## AI TL;DR is too short ({word_count} words; minimum {MIN_AI_TLDR_PARAGRAPH_WORDS})",
        )
        return
    if word_count > MAX_AI_TLDR_PARAGRAPH_WORDS:
        report.add(
            "AI TL;DR paragraph",
            "WARN",
            f"## AI TL;DR is long ({word_count} words; suggested max {MAX_AI_TLDR_PARAGRAPH_WORDS})",
        )
        return
    # Detect bullet form: top-level bullets at start of line.
    bullet_lines = [ln for ln in section.splitlines() if re.match(r"^\s*[-*]\s+\S", ln)]
    if len(bullet_lines) >= 3:
        # Bullet form: count top-level bullets as the "beats".
        if len(bullet_lines) > 7:
            report.add(
                "AI TL;DR paragraph",
                "WARN",
                f"## AI TL;DR has {len(bullet_lines)} bullets; aim for 3-5 (LW style)",
            )
            return
        report.add(
            "AI TL;DR paragraph",
            "PASS",
            f"{word_count} words, {len(bullet_lines)} bullets (LW-style)",
        )
        return
    # Paragraph form: split on `[.!?]` + whitespace + uppercase-start.
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]
    if len(sentences) < MIN_AI_TLDR_PARAGRAPH_SENTENCES:
        report.add(
            "AI TL;DR paragraph",
            "WARN",
            f"## AI TL;DR has {len(sentences)} sentence(s); aim for >= {MIN_AI_TLDR_PARAGRAPH_SENTENCES} (or use 3-5 LW-style bullets)",
        )
        return
    report.add(
        "AI TL;DR paragraph",
        "PASS",
        f"{word_count} words, {len(sentences)} sentences",
    )


def check_human_tldr(
    body: str,
    report: Report,
    *,
    issue_created_at: str | None = None,
) -> None:
    """``## Human TL;DR`` H2 must be present.

    Content is NOT enforced — the user fills this in by hand and drafts
    legitimately keep the literal placeholder line. Grandfathered (PASS,
    missing OK) when EITHER the issue was created before
    ``SUMMARY_RENAME_DATE`` OR the body's structured block still lives
    under legacy ``## TL;DR`` (legacy shape predates Human TL;DR).
    """
    is_legacy_by_date = issue_created_at is not None and issue_created_at[:10] < SUMMARY_RENAME_DATE
    _, summary_kind = _extract_summary_section(body)
    is_legacy_by_shape = summary_kind == "legacy-tldr"
    is_legacy = is_legacy_by_date or is_legacy_by_shape
    section = _extract_section(body, "Human TL;DR", level=2)
    if section is None:
        if is_legacy:
            report.add(
                "Human TL;DR",
                "PASS",
                "section missing (legacy body — grandfathered)",
            )
            return
        report.add(
            "Human TL;DR",
            "FAIL",
            "## Human TL;DR H2 is missing (template requires the section header even if user-filled later)",
        )
        return
    report.add("Human TL;DR", "PASS", "H2 present (content user-owned, not validated)")


def _extract_results_block(tldr: str | None) -> str | None:
    if tldr is None:
        return None
    m = re.search(r"(?ms)^###\s+Results\s*$(.+?)(?=^###\s+|\Z)", tldr)
    return m.group(1) if m else None


def check_hero_figure(tldr: str | None, report: Report) -> None:
    """Verify the hero figure and any additional figures inside ### Results.

    Multi-figure relaxation (#293 §1): >=1 image is fine; each figure's caption
    is enforced separately by :func:`check_results_figure_captions`. The hero
    figure (``image_urls[0]``) must be commit-pinned on raw-github; secondary
    images are required to be ``raw.githubusercontent.com`` URLs but are NOT
    required to be commit-pinned (allows supplementary panels). Any non-raw-
    github secondary URL produces a WARN.
    """
    results_block = _extract_results_block(tldr)
    if results_block is None:
        report.add("Hero figure", "FAIL", "### Results subsection missing")
        return
    image_urls = re.findall(r"!\[[^\]]*\]\((\S+?)\)", results_block)
    if not image_urls:
        report.add("Hero figure", "FAIL", "no image inside ### Results")
        return
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
    # Secondary-image WARN loop (#293 §1 BLOCKER F): each image beyond the
    # hero must come from raw-github; commit-pinning is NOT required for
    # supplementary panels.
    secondary_warns: list[str] = []
    for sec_url in image_urls[1:]:
        if "raw.githubusercontent.com" not in sec_url:
            secondary_warns.append(
                f"secondary image is not raw.githubusercontent.com: {sec_url[:80]}"
            )
    if secondary_warns:
        report.add(
            "Hero figure",
            "WARN",
            f"{len(image_urls)} figure(s); primary commit-pinned; " + "; ".join(secondary_warns),
        )
        return
    report.add(
        "Hero figure",
        "PASS",
        f"{len(image_urls)} figure(s) present; primary commit-pinned",
    )


def check_results_figure_captions(
    tldr: str | None,
    report: Report,
    *,
    issue_created_at: str | None = None,
) -> None:
    """Each figure inside ### Results must be followed by a caption paragraph.

    Caption = the next non-empty content block after ``![...](...)``, where:
      - it is NOT another image link,
      - it is NOT a heading,
      - it is NOT an HTML comment-only line (``<!-- ... -->``),
      - it is NOT a horizontal rule (``---``, ``***``, ``___``),
      - it is NOT a ``**bold-label:**`` paragraph (``**Main takeaways:**`` etc.),
      - after stripping markdown link syntax ``[text](url)`` -> ``text`` and
        bullet markers, it has at least ``RESULTS_CAPTION_MIN_WORDS`` words.

    HARD FAIL — same posture as the Reproducibility-card sentinel check, EXCEPT
    when ``issue_created_at`` precedes ``CAPTION_CHECK_ENFORCEMENT_DATE``, in
    which case FAIL is downgraded to WARN (date-gate for legacy issues).

    Parameters
    ----------
    tldr
        The TL;DR substring as returned by :func:`check_tldr_structure`.
    report
        Report to write the verdict into.
    issue_created_at
        ISO date string (``YYYY-MM-DD`` or full ISO-8601 timestamp). Sliced to
        the first 10 chars before lexicographic comparison with the
        enforcement date. ``None`` (file-mode) means strict enforcement.
    """
    results_block = _extract_results_block(tldr)
    if results_block is None:
        # check_hero_figure already FAILed; nothing more to say here.
        return

    # #293 round-2 C3: scrub MULTI-LINE HTML comments from the block before
    # the line-walker runs. Templates use multi-line ``<!-- ... -->`` blocks
    # to comment out optional figure stubs (e.g. the secondary-figure
    # placeholder in clean-results/template.md lines 104-110). Without
    # scrubbing, a stub ``![alt](url)`` inside the comment is mistaken for a
    # real figure and the walker demands a caption that doesn't exist.
    #
    # ``re.DOTALL`` lets ``.`` match newlines so the regex spans multi-line
    # comments; the ``?`` makes ``.*?`` non-greedy so adjacent comments
    # don't merge into one giant span.
    results_block = re.sub(r"<!--.*?-->", "", results_block, flags=re.DOTALL)

    # Date-gate comparison (#293 §1 BLOCKER G + v3 P4): gh returns ``createdAt``
    # as a full ISO-8601 timestamp like ``2026-05-15T10:00:00Z``;
    # ``CAPTION_CHECK_ENFORCEMENT_DATE`` is date-only ``YYYY-MM-DD``. Slice to
    # the date portion before comparing so the lexicographic compare is
    # unambiguous regardless of timestamp suffix.
    is_legacy = (
        issue_created_at is not None and issue_created_at[:10] < CAPTION_CHECK_ENFORCEMENT_DATE
    )
    fail_status = "WARN" if is_legacy else "FAIL"

    lines = results_block.splitlines()
    image_re = re.compile(r"!\[[^\]]*\]\(\S+?\)")
    label_re = re.compile(r"^\s*\*\*\s*\w")  # **Main takeaways:**
    hr_re = re.compile(r"^\s*([-*_])\1{2,}\s*$")  # ---, ***, ___ horizontal rules
    # Single-line HTML comment kept for backward compat (legacy fixtures
    # use them in the caption slot — see test_caption_html_comment_only_fails).
    # Multi-line spans were already stripped above.
    html_comment_re = re.compile(r"^\s*<!--.*-->\s*$")
    inline_link_re = re.compile(r"\[([^\]]+)\]\([^)]+\)")
    bullet_strip_re = re.compile(r"^\s*[-*]\s+")

    failures: list[str] = []
    for i, line in enumerate(lines):
        m = image_re.search(line)
        if not m:
            continue
        caption_words: list[str] = []
        for j in range(i + 1, len(lines)):
            nxt_raw = lines[j]
            nxt = nxt_raw.strip()
            if not nxt:
                if caption_words:
                    break  # blank line ends an in-progress caption
                continue
            if image_re.search(nxt):
                break
            if nxt.startswith("#") or label_re.match(nxt) or hr_re.match(nxt):
                break
            if html_comment_re.match(nxt):
                # HTML-comment lines are skipped silently; not a caption, not a
                # boundary.
                continue
            stripped = bullet_strip_re.sub("", nxt)
            stripped = inline_link_re.sub(r"\1", stripped)  # [text](url) -> text
            caption_words.extend(stripped.split())
        if len(caption_words) < RESULTS_CAPTION_MIN_WORDS:
            short_alt = m.group(0)[:60]
            failures.append(
                f"figure {short_alt!r} has {len(caption_words)}-word caption "
                f"(min {RESULTS_CAPTION_MIN_WORDS})"
            )
    if failures:
        msg = "; ".join(failures)
        if is_legacy:
            msg += (
                f" (legacy issue created {issue_created_at!r} "
                f"< gate {CAPTION_CHECK_ENFORCEMENT_DATE}; downgraded to WARN)"
            )
        report.add("Results figure captions", fail_status, msg)
        return
    report.add(
        "Results figure captions",
        "PASS",
        "every Results figure has a caption paragraph",
    )


def check_methodology_bullets(
    tldr: str | None,
    report: Report,
    *,
    strict: bool,
    created_at: datetime | None = None,
) -> None:
    """Verify that ### Methodology contains the 4 required bolded bullet labels.

    Cutoff behavior:
    - When ``strict=False`` (grandfathered: issue >7 days old or already-promoted),
      always PASS.
    - When ``created_at`` is supplied AND falls before
      ``METHODOLOGY_BULLETS_REQUIRED_AFTER``, PASS via the ``pre-cutoff`` branch.
      This grandfathers the in-flight ``clean-results:draft`` issues that
      were authored against the prose-form template.
    - File mode passes ``created_at=None`` so the cutoff branch never fires
      and fresh-from-template drafts are validated against the new bullet
      form.
    """
    if not strict:
        report.add("Methodology bullets", "PASS", "non-strict (grandfathered)")
        return
    if created_at is not None and created_at < METHODOLOGY_BULLETS_REQUIRED_AFTER:
        cutoff_date = METHODOLOGY_BULLETS_REQUIRED_AFTER.date()
        report.add(
            "Methodology bullets",
            "PASS",
            f"pre-cutoff (created {created_at.date()}, cutoff {cutoff_date})",
        )
        return
    if tldr is None:
        report.add(
            "Methodology bullets",
            "FAIL",
            "structured block missing (## AI Summary or legacy ## TL;DR)",
        )
        return
    methodology = _extract_section(tldr, "Methodology", level=3)
    if methodology is None:
        report.add("Methodology bullets", "FAIL", "### Methodology subsection missing")
        return
    missing = [b for b in REQUIRED_METHODOLOGY_BULLETS if b not in methodology]
    if missing:
        report.add(
            "Methodology bullets",
            "FAIL",
            f"missing bullet labels: {missing}",
        )
        return
    report.add(
        "Methodology bullets",
        "PASS",
        f"all {len(REQUIRED_METHODOLOGY_BULLETS)} bullet labels present",
    )


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
    """Return the HIGH/MODERATE/LOW from the Results block's Confidence line, if any.

    Looks inside ``## AI Summary`` (post-rename) or ``## TL;DR`` (legacy).
    """
    section, _kind = _extract_summary_section(body)
    results_block = _extract_results_block(section)
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


# --- #275 item 4 / 9: TL;DR acronym checker ----------------------------------
# Project-internal acronyms locked at 6 tokens (per principles.md).
# Domain-of-art acronyms (`EM`, `LoRA`, `SFT`, `DPO`, `LM`) are NOT enforced
# here — they're standard. Adding to this list requires a matching
# principles.md update.
INTERNAL_ACRONYMS: tuple[str, ...] = ("H1", "H2", "H3", "P1", "P2", "P3")

# Code-block + inline-backtick stripping (B2): a literal `H1` inside a JSON
# example or python snippet is not a project-internal-acronym usage.
FENCED_BLOCK_RE = re.compile(r"```[\s\S]*?```")
INLINE_BACKTICK_RE = re.compile(r"`[^`\n]+`")


def _strip_code(text: str) -> str:
    """Remove fenced ```...``` blocks and inline `...` spans."""
    return INLINE_BACKTICK_RE.sub("", FENCED_BLOCK_RE.sub("", text))


# An acronym counts as DEFINED if it's followed (with optional whitespace) by
# one of `=`, `(`, `:`, `—`, `-` (the supported delimiter shapes). See
# `.claude/skills/clean-results/checklist.md`.
_ACRONYM_DEF_DELIMS = r"=|\(|:|—|-"


# --- #275 item 13: TL;DR dataset-example link patterns -----------------------
WANDB_OR_HF_PATTERN = re.compile(
    # wandb.ai web URL
    r"https?://(?:[\w.-]+\.)?wandb\.ai/[^\s)\]]+"
    # wandb:// artifact URI
    r"|wandb://[^\s)\]]+"
    # huggingface.co/<owner>/<repo>/... (covers datasets AND models AND adapters)
    r"|https?://huggingface\.co/[\w.-]+/[\w.-]+(?:/[^\s)\]]*)?"
)

# Reject literal `**Dataset example:** N/A` as gameable (B4).
DATASET_EXAMPLE_NA = re.compile(r"\*\*\s*Dataset\s+example\s*:\s*\*\*\s*N/?A\b", re.IGNORECASE)


def check_undefined_acronyms(
    tldr: str | None,
    report: Report,
    *,
    strict: bool = True,
) -> None:
    """FAIL if AI Summary uses H1/H2/H3/P1/P2/P3 without inline definition.

    Operates on the structured block returned by ``check_tldr_structure``
    (``## AI Summary`` post-rename, ``## TL;DR`` legacy). Code blocks
    (```...```) and inline backticks (`...`) are stripped before the regex
    runs (per B2) so a literal `H1` in a code snippet does not trigger the
    check.

    A token counts as DEFINED when followed by `=`, `(`, `:`, `—`, or `-`
    (with optional whitespace). E.g. `H1 = primary hypothesis`,
    `P1 (coupling phase)`, `H2: leakage`. See
    `.claude/skills/clean-results/checklist.md` for the supported delimiters.

    Grandfathered (PASS) when ``strict=False`` (issue >7 days old or
    already-promoted).
    """
    if tldr is None:
        return
    if not strict:
        report.add("Acronyms defined", "PASS", "non-strict (grandfathered)")
        return
    scrubbed = _strip_code(tldr)
    tokens = "|".join(INTERNAL_ACRONYMS)
    def_pattern = re.compile(rf"\b({tokens})\s*(?:{_ACRONYM_DEF_DELIMS})")
    defined = {m.group(1) for m in def_pattern.finditer(scrubbed)}
    used: set[str] = set()
    for token in INTERNAL_ACRONYMS:
        # Match the bare token but not when embedded in identifiers or paths.
        if re.search(
            rf"(?<![A-Za-z0-9_/-])\b{re.escape(token)}\b(?![A-Za-z0-9_/-])",
            scrubbed,
        ):
            used.add(token)
    undefined = used - defined
    if undefined:
        report.add(
            "Acronyms defined",
            "FAIL",
            f"undefined project-internal acronym(s) in AI Summary: {sorted(undefined)}. "
            "Define on first use, e.g. 'H1 = ...' or 'P1 (coupling phase)'. "
            "Code blocks and inline backticks are exempt.",
        )
        return
    if used:
        report.add("Acronyms defined", "PASS", f"all defined: {sorted(used)}")
    else:
        report.add("Acronyms defined", "PASS", "no project-internal acronyms used")


def check_background_motivation(
    tldr: str | None,
    report: Report,
    *,
    current_issue: int | None,
    strict: bool = True,
) -> None:
    """FAIL if Background lacks a `#<issue>` ref distinct from the current issue.

    Every clean-result body answers "why was this run?" in the first
    paragraph by linking the prior issue(s) that motivated it. A reference
    to the current issue itself does NOT count (B7).

    Grandfathered (PASS) when ``strict=False``.
    """
    if tldr is None:
        return
    if not strict:
        report.add("Background motivation", "PASS", "non-strict (grandfathered)")
        return
    bg = _extract_section(tldr, "Background", level=3)
    if bg is None:
        # check_background_context already flagged the missing section.
        return
    issue_refs = re.findall(r"(?<![A-Za-z0-9])#(\d{1,5})(?![A-Za-z0-9])", bg)
    issue_refs_int = {int(n) for n in issue_refs}
    if current_issue is not None:
        issue_refs_int.discard(current_issue)
    if not issue_refs_int:
        report.add(
            "Background motivation",
            "FAIL",
            "### Background has no #<issue> reference (other than self). "
            "Link the prior result(s) that motivated this experiment, "
            "e.g. 'Builds on #234'.",
        )
        return
    report.add(
        "Background motivation",
        "PASS",
        f"references prior issue(s): {sorted(issue_refs_int)}",
    )


def check_tldr_dataset_example(
    tldr: str | None,
    report: Report,
    *,
    issue_labels: set[str] | None = None,
    strict: bool = True,
) -> None:
    """FAIL if Methodology lacks a dataset example AND a wandb/HF link.

    The AI Summary Methodology subsection (or legacy ``## TL;DR``) must
    contain (a) at least one fenced ``code`` block OR a
    `**Dataset example:**` bullet, AND (b) at least one
    wandb.ai / wandb:// / huggingface.co full-data link somewhere in the
    same structured block. Skipped when the issue carries the `no-dataset`
    label (model-only / axis-steering experiments).

    Literal `**Dataset example:** N/A` is rejected as gameable (B4).
    Grandfathered (PASS) when ``strict=False``.
    """
    if tldr is None:
        return
    if not strict:
        report.add("Dataset example", "PASS", "non-strict (grandfathered)")
        return
    if issue_labels and "no-dataset" in issue_labels:
        report.add("Dataset example", "PASS", "skipped (no-dataset label)")
        return
    methodology = _extract_section(tldr, "Methodology", level=3)
    if methodology is None:
        return  # caught by other checks
    if DATASET_EXAMPLE_NA.search(methodology):
        report.add(
            "Dataset example",
            "FAIL",
            "literal `**Dataset example:** N/A` is not accepted. If the "
            "experiment is model-only / axis-steering, apply the "
            "`no-dataset` label to the issue instead.",
        )
        return
    has_fenced = bool(re.search(r"```[\s\S]+?```", methodology))
    has_example_bullet = bool(
        re.search(
            r"\*\*\s*Dataset\s+example\s*:\s*\*\*\s*\S",
            methodology,
            re.IGNORECASE,
        )
    )
    if not (has_fenced or has_example_bullet):
        report.add(
            "Dataset example",
            "FAIL",
            "### Methodology has neither a fenced code-block example NOR a "
            "`**Dataset example:**` bullet.",
        )
        return
    if not WANDB_OR_HF_PATTERN.search(tldr):
        report.add(
            "Dataset example",
            "FAIL",
            "AI Summary has no wandb.ai / wandb:// / huggingface.co/<owner>/<repo> link. "
            "Provide a `**Full data:**` link or apply the `no-dataset` label.",
        )
        return
    report.add(
        "Dataset example",
        "PASS",
        "dataset example + full-data link present",
    )


def check_background_context(tldr: str | None, report: Report) -> None:
    """WARN if Background subsection is too terse for newcomers (<30 words)."""
    if tldr is None:
        return
    bg = _extract_section(tldr, "Background", level=3)
    if bg is None:
        report.add(
            "Background context",
            "WARN",
            "### Background subsection missing from AI Summary",
        )
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


def check_narrative_consolidation(body: str, report: Report) -> None:
    """If body has a `Source-issues:` line, this is a multi-issue narrative.

    Assert the structural shape:
      - Source-issues line lists >=2 issue numbers (so it's actually a consolidation)
      - At least one figure URL is retained in the body (hero figure preserved)
    A clean-result without Source-issues is single-experiment and skipped here.
    """
    m = re.search(r"^Source-issues:\s*(.+)$", body, re.MULTILINE)
    if not m:
        return  # not a consolidation; nothing to check

    refs = re.findall(r"#(\d+)", m.group(1))
    if len(refs) < 2:
        report.add(
            "narrative_sources",
            "FAIL",
            f"Source-issues line lists {len(refs)} issue refs, expected >=2 for a consolidation.",
        )
        return
    report.add(
        "narrative_sources",
        "PASS",
        f"Source-issues lists {len(refs)} child issues: {refs}",
    )

    figure_pat = re.compile(
        r"!\[[^\]]*\]\([^)]+\.(?:png|pdf|jpg|jpeg)\)|figures/[^)\s]+\.(?:png|pdf)"
    )
    if not figure_pat.search(body):
        report.add(
            "narrative_figure",
            "FAIL",
            "Narrative consolidation has no retained hero figure URL - "
            "expected at least one !(...png/pdf) image link or figures/ path.",
        )
    else:
        report.add(
            "narrative_figure",
            "PASS",
            "narrative retains at least one hero figure",
        )


#: Names of every check that `run_all_checks` registers. Used by `--skip-checks`
#: to validate user input (typos would otherwise silently pass — see code-review
#: round 1 NIT). Keep in sync with the `_maybe(...)` calls in `run_all_checks`.
KNOWN_CHECKS: frozenset[str] = frozenset(
    {
        "check_human_tldr",
        "check_ai_tldr_paragraph",
        "check_hero_figure",
        "check_results_figure_captions",
        "check_results_block",
        "check_methodology_bullets",
        "check_background_context",
        "check_undefined_acronyms",
        "check_background_motivation",
        "check_tldr_dataset_example",
        "check_human_summary",
        "check_sample_outputs",
        "check_numbers_in_json",
        "check_reproducibility",
        "check_confidence_phrasebook",
        "check_forbidden_stats",
        "check_title",
        "check_narrative_consolidation",
    }
)


def run_all_checks(
    title: str | None,
    body: str,
    *,
    strict: bool = True,
    created_at: datetime | None = None,
    current_issue: int | None = None,
    issue_labels: set[str] | None = None,
    skip_checks: set[str] | None = None,
) -> Report:
    """Run every registered check unless its name appears in ``skip_checks``.

    Skipped checks log ``SKIPPED: <name> (--skip-checks)`` to stderr per B3.
    """
    skip_checks = skip_checks or set()
    report = Report()

    def _maybe(name: str, fn) -> None:
        if name in skip_checks:
            print(f"SKIPPED: {name} (--skip-checks)", file=sys.stderr)
            return
        fn()

    # check_tldr_structure returns the structured-block substring used by
    # downstream checks; we always run it (the cost of skipping is broken
    # downstream checks). If a caller wants to silence it they can drop it
    # from the report after the fact. Post-rename the section is
    # ``## AI Summary``; pre-rename the same content lives under ``## TL;DR``.
    tldr = check_tldr_structure(body, report)
    # Convert ``created_at`` (datetime | None) to an ISO date string for the
    # date-gated checks. Slices to ``[:10]`` are done at the call site; an
    # explicit ``isoformat()`` keeps the contract straightforward and avoids a
    # TypeError if the dataclass shape ever drifts.
    issue_created_at_iso = created_at.isoformat() if created_at is not None else None
    _maybe(
        "check_human_tldr",
        lambda: check_human_tldr(body, report, issue_created_at=issue_created_at_iso),
    )
    _maybe(
        "check_ai_tldr_paragraph",
        lambda: check_ai_tldr_paragraph(body, report, issue_created_at=issue_created_at_iso),
    )
    _maybe("check_hero_figure", lambda: check_hero_figure(tldr, report))
    _maybe(
        "check_results_figure_captions",
        lambda: check_results_figure_captions(tldr, report, issue_created_at=issue_created_at_iso),
    )
    _maybe("check_results_block", lambda: check_results_block(tldr, report))
    _maybe(
        "check_methodology_bullets",
        lambda: check_methodology_bullets(tldr, report, strict=strict, created_at=created_at),
    )
    _maybe("check_background_context", lambda: check_background_context(tldr, report))
    _maybe(
        "check_undefined_acronyms",
        lambda: check_undefined_acronyms(tldr, report, strict=strict),
    )
    _maybe(
        "check_background_motivation",
        lambda: check_background_motivation(
            tldr, report, current_issue=current_issue, strict=strict
        ),
    )
    _maybe(
        "check_tldr_dataset_example",
        lambda: check_tldr_dataset_example(tldr, report, issue_labels=issue_labels, strict=strict),
    )
    _maybe(
        "check_human_summary",
        lambda: check_human_summary(body, report, strict=strict),
    )
    _maybe(
        "check_sample_outputs",
        lambda: check_sample_outputs(body, report, strict=strict),
    )
    _maybe("check_numbers_in_json", lambda: check_numbers_in_json(body, report))
    _maybe("check_reproducibility", lambda: check_reproducibility(body, report))
    _maybe(
        "check_confidence_phrasebook",
        lambda: check_confidence_phrasebook(body, report),
    )
    _maybe("check_forbidden_stats", lambda: check_forbidden_stats(body, report))
    _maybe("check_title", lambda: check_title(title, body, report))
    _maybe(
        "check_narrative_consolidation",
        lambda: check_narrative_consolidation(body, report),
    )
    return report


def _parse_current_issue_from_body(body: str) -> int | None:
    """Best-effort extraction of the current issue number from a body string.

    Looks for a `Source-issues: #N1, #N2` line first (multi-issue
    consolidation — current issue is the FIRST listed); otherwise returns
    None. The CLI ``--current-issue`` flag is the explicit override.
    """
    m = re.search(r"^Source-issues:\s*(.+)$", body, re.MULTILINE)
    if m:
        first = re.search(r"#(\d+)", m.group(1))
        if first:
            return int(first.group(1))
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("path", nargs="?", help="Path to a clean-result body markdown file")
    group.add_argument("--issue", type=int, help="Fetch body via gh issue view <N>")
    parser.add_argument(
        "--current-issue",
        type=int,
        default=None,
        help=(
            "Override the issue-number used by check_background_motivation "
            "to filter self-references. Auto-set when --issue is used."
        ),
    )
    parser.add_argument(
        "--skip-checks",
        type=str,
        default="",
        help=(
            "Comma-separated list of check function names to skip. "
            "Each skipped check logs `SKIPPED: <name> (--skip-checks)` to stderr."
        ),
    )
    args = parser.parse_args(argv)

    skip_checks = {s.strip() for s in args.skip_checks.split(",") if s.strip()}
    # Validate each --skip-checks token against the registered check names so a
    # typo (e.g. `check_heroe_figure`) fails loudly instead of silently passing
    # by skipping nothing. (Code-review round 1 NIT.)
    unknown = skip_checks - KNOWN_CHECKS
    if unknown:
        parser.error(
            "unknown check name(s) in --skip-checks: "
            f"{', '.join(sorted(unknown))}. "
            f"Known checks: {', '.join(sorted(KNOWN_CHECKS))}"
        )

    created_dt: datetime | None
    issue_labels: set[str] = set()
    current_issue: int | None = args.current_issue

    if args.issue is not None:
        title, body, label_names, created_at = _fetch_issue_body(args.issue)
        issue_labels = set(label_names)
        if current_issue is None:
            current_issue = args.issue
        # Date-gate: skip Human summary / Sample outputs strict checks for
        # issues >7 days old or already-promoted (clean-results without :draft).
        from datetime import timedelta

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
        created_dt = None  # file mode: cutoff branch never fires
        if current_issue is None:
            current_issue = _parse_current_issue_from_body(body)

    report = run_all_checks(
        title,
        body,
        strict=strict,
        created_at=created_dt,
        current_issue=current_issue,
        issue_labels=issue_labels,
        skip_checks=skip_checks,
    )
    print(report.render())
    if report.any_fail():
        print("\nResult: FAIL — fix the failing checks before posting.")
        return 1
    print("\nResult: PASS (WARNs acknowledged).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
