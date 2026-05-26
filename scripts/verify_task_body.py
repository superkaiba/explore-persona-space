#!/usr/bin/env python3
"""verify_task_body.py — mechanical verifier for markdown clean-result bodies.

Replaces `verify_sagan_card.py` for new (markdown) bodies. Thirteen checks
against the markdown clean-result spec in
`.claude/plans/task-workflow-migration.md` § 10 (Sagan-card content
discipline ported from HTML to markdown):

0. Body is not a stub — body has ≥500 chars, contains a `# <title>` H1,
   and is not a single stub token (`placeholder`, `tbd`, `todo`, `stub`).
   Defense-in-depth against the cache → body.md silent-handoff failure
   (incident: task #385, 2026-05-25). Runs FIRST and short-circuits the
   rest of the check chain — a stub body produces ONE clear FAIL at the
   top rather than a dozen cascading "<section> missing" errors.
1. Title confidence tag — H1 line ends with `(LOW|MODERATE|HIGH confidence)`.
2. Three required H2 sections in order — `## TL;DR`, `## Details`,
   `## Reproducibility`. `## Figure` is OPTIONAL (2026-05-26): bodies may
   carry it (with the hero image + caption inside) OR inline images
   directly under TL;DR Results sub-bullets (one-takeaway-one-figure
   pattern). If `## Figure` IS present it must sit between TL;DR and
   Details. Extra H2s after `## Reproducibility` are allowed.
3. TL;DR bullet labels — three required bullets carry the labels
   `Motivation`, `What I ran`, `Results`. A fourth `Next steps` bullet
   is OPTIONAL — include when there is genuinely useful follow-up to
   queue; omit otherwise. The verifier neither requires nor flags its
   presence (decision: 2026-05-26, iterations.md).
4. Hero image present — at least one `![alt](url)` image exists in
   `## Figure` (if present) OR inline under `## TL;DR` (one-takeaway-
   one-figure pattern, 2026-05-26).
4b. Figure URL resolvable — every image URL in `## Figure` or inline
    under `## TL;DR` is an absolute `https://...` URL the dashboard can
    fetch. Relative paths (`artifacts/...`, `tasks/...`, `figures/...`,
    `./...`, `../...`) fail because the EPS dashboard does not serve
    binary PNG/PDF files under `tasks/<N>/artifacts/` (incident: task
    #365, 2026-05-22). `raw.githubusercontent.com` URLs must pin to a
    commit SHA, not `main`/`master`/`HEAD`.
5. Figure caption ≥10 words — if `## Figure` is present, the first
   non-image line under it has at least ten words. Vacuously satisfied
   when `## Figure` is absent (inline-image alt-text serves as the
   caption under the one-takeaway-one-figure pattern).
6. Confidence sentence in Details matches title — `Confidence: LOW|MODERATE|HIGH — <rationale>`
   line appears in `## Details`, agrees with the title, and carries ≥20
   chars of rationale after the dash.
7. Three repro subgroups present — `**Artifacts:**`, `**Compute:**`,
   `**Code:**` all appear as boldface labels inside `## Reproducibility`.
8. Reproducibility URL permanence — every URL in `## Reproducibility`
   pins to a ref (HF Hub `/tree/<ref>`, WandB `/runs/<id>`, GitHub
   `/blob/<sha>` or `/tree/<sha>` — never `main`/`master`/`HEAD`). `n/a`
   is accepted as an explicit non-applicable marker.
9. Reproducibility sentinel scrub — no `{{`, `TBD`, `see config`, or
   `default` placeholders anywhere under `## Reproducibility`.
10. Cherry-picked label discipline — every sample-output fenced code
    block in `## Details` (heuristic: contains `User:`/`Assistant:`/`Human:`/`Model:`
    or has >200 chars of text) is preceded by prose containing
    `cherry-picked`, `cherry picked`, `random sample`, `first N of M`,
    or similar disclosure.
11. Qualitative-data link — every sample-output fenced block in
    `## Details` is preceded by at least one link or backtick-wrapped
    path pointing at a raw text-level artifact (i.e. NOT an
    aggregate-only path like `regression`, `summary`, `aggregat*`,
    `per-cell`, or `.npz`). An explicit `not uploaded` / `not
    available` disclosure downgrades FAIL to WARN.

Soft INFO (not enforced as PASS/FAIL; surfaced for orchestrator
visibility): the Goal-of-experiment frontmatter field — frontmatter
contains ``goal: <one sentence>``. The body-side ``## Goal`` H2 is
INTENTIONALLY NOT CHECKED HERE: it lives only in proposed/planning
bodies (enforced at /issue Step 0c, workflow.yaml §
gates.experiment_goal); clean-result bodies drop the visible H2 and
fold the Goal text into the TL;DR Motivation bullet. The frontmatter
``goal:`` field stays in the clean-result body for agent-facing
reference (planner, critic, follow-up-proposer all read it). This
verifier WARNs when the frontmatter field is missing but never FAILs —
non-experiment kinds and pre-Goal bodies legitimately omit it.

Bodies carrying a `<!-- legacy-sagan-card -->` sentinel are
grandfathered HTML — this verifier skips them with a PASS (the legacy
`verify_sagan_card.py` still applies to those).

Usage:

    uv run python scripts/verify_task_body.py --issue <N>
    uv run python scripts/verify_task_body.py --file path/to/body.md
    uv run python scripts/verify_task_body.py --body-stdin

Exits 0 on PASS, 1 on FAIL, 2 on usage error.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Bring the task_workflow module in for --issue lookups.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml  # noqa: E402

# ─── Spec constants ────────────────────────────────────────────────────────

REQUIRED_H2_SECTIONS = ["TL;DR", "Details", "Reproducibility"]
# `## Figure` is OPTIONAL as of 2026-05-26 (iterations.md). Bodies may either
# (a) carry a `## Figure` H2 holding a hero image + caption, OR (b) inline
# images directly under TL;DR Results sub-bullets (one-takeaway-one-figure
# pattern). At least one image must exist in TL;DR or `## Figure` combined.
OPTIONAL_H2_SECTIONS = ["Figure"]
# TL;DR bullet labels. Required labels are enforced by `check_tldr_labels`;
# the optional `Next steps` bullet is permitted but not required (decision:
# 2026-05-26, iterations.md). Bodies WITH a Next-steps bullet still PASS;
# bodies WITHOUT one also PASS — analyzer adds it only when there is genuinely
# useful follow-up to queue.
TLDR_BULLETS_REQUIRED = ["Motivation", "What I ran", "Results"]
TLDR_BULLETS_OPTIONAL = ["Next steps"]
REPRO_SUBGROUPS = ["Artifacts", "Compute", "Code"]

LEGACY_SAGAN_CARD_SENTINEL = "<!-- legacy-sagan-card -->"

CONFIDENCE_LEVELS = {"LOW", "MODERATE", "HIGH"}

# Sentinel substrings that indicate a placeholder slipped through.
SENTINEL_SUBSTRINGS = ["TBD", "{{", "see config", "default"]

# Minimum number of characters of rationale required AFTER the
# `Confidence: <level> —` dash on the confidence line.
MIN_CONFIDENCE_RATIONALE_CHARS = 20

# ─── Result type ───────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    is_warn: bool = False  # WARN downgrades — counts as PASS for `passed`,
    # but rendered with a [WARN] tag.

    def render(self) -> str:
        tag = "WARN" if self.is_warn else ("PASS" if self.passed else "FAIL")
        line = f"  [{tag}] {self.name}"
        if self.detail:
            line += f" — {self.detail}"
        return line


# ─── Body splitting ────────────────────────────────────────────────────────


def split_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    rest = text[4:]
    end = rest.find("\n---\n")
    if end == -1:
        return {}, text
    fm_block = rest[:end]
    body = rest[end + len("\n---\n") :]
    try:
        fm = yaml.safe_load(fm_block) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(fm, dict):
        return {}, text
    return fm, body


def find_h1_title(body: str) -> str | None:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            return stripped[2:].strip()
    return None


def find_h2_sections(body: str) -> list[tuple[str, int, int]]:
    """Return list of (section_name, body_line_start, body_line_end) for each H2.

    H2 lines inside fenced code blocks are ignored, so a pasted
    ``## Why this experiment`` inside a code fence cannot satisfy the
    verifier or the `task.py new` gate. Both triple-backtick (``` ```py``)
    and triple-tilde (``~~~text``) fence delimiters are recognized,
    matching CommonMark's relaxed rule.
    """
    lines = body.splitlines()
    h2_indices: list[tuple[str, int]] = []
    in_fence = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Toggle fence state on any line starting with ``` or ~~~ (with
        # optional info string, e.g. ```python or ~~~text). Matches
        # CommonMark's relaxed rule: an opening fence does not have to
        # be closed by an identical tag, but lines starting with ``` or
        # ~~~ flip the state.
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith("## ") and not stripped.startswith("### "):
            h2_indices.append((stripped[3:].strip(), i))
    out: list[tuple[str, int, int]] = []
    for k, (name, start) in enumerate(h2_indices):
        end = h2_indices[k + 1][1] if k + 1 < len(h2_indices) else len(lines)
        out.append((name, start + 1, end))
    return out


def section_text(body: str, section_name: str) -> str | None:
    lines = body.splitlines()
    for name, start, end in find_h2_sections(body):
        if name.casefold() == section_name.casefold():
            return "\n".join(lines[start:end]).strip()
    return None


# Image markdown:  ![alt](path-or-url)
# Alt text may contain `[brackets]` (e.g. literal marker names like `[ZLT]`),
# so we allow a `]` inside alt as long as it is not followed by `(`. The URL
# group is captured for downstream resolvability checks (no parens inside URL).
_IMAGE_RE = re.compile(r"!\[(?:[^\]]|\](?!\())*\]\(([^)]+)\)")

# Markdown link: [text](url)
_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

# Backtick-wrapped inline code: `path/to/thing`
_CODE_RE = re.compile(r"`([^`\n]+)`")

# Fenced code blocks ```...```
_FENCED_RE = re.compile(r"^```[^\n]*\n(.*?)\n```", re.DOTALL | re.MULTILINE)


# ─── Sample-block heuristic helpers ───────────────────────────────────────


def _is_sample_fence(content: str) -> bool:
    """Return True if a fenced code block looks like sample model output.

    Mirrors the heuristic in verify_sagan_card.py::_is_sample_pre — completion-
    style if it contains a User/Assistant/Human/Model marker OR the body is
    long (> 200 chars). Otherwise it is probably a code/CLI snippet.
    """
    if re.search(r"\b(User|Assistant|Human|Model):", content, re.IGNORECASE):
        return True
    return len(content.strip()) > 200


def _iter_sample_fences(details: str) -> list[tuple[int, int, str]]:
    """Yield (fence_start_offset, fence_end_offset, content) for each
    fenced code block in `details` that is sample-output-like."""
    out: list[tuple[int, int, str]] = []
    for m in _FENCED_RE.finditer(details):
        content = m.group(1)
        if _is_sample_fence(content):
            out.append((m.start(), m.end(), content))
    return out


def _prelude_window(details: str, fence_start: int, max_chars: int = 1500) -> str:
    """Return the prose immediately preceding a fenced block.

    Walks back at most ``max_chars`` from ``fence_start``. Stops at the
    previous fenced block's closing ``` (so two consecutive sample
    blocks don't share each other's prelude), then trims any leading
    partial line.
    """
    lo = max(0, fence_start - max_chars)
    window = details[lo:fence_start]
    # Don't cross a previous fence's closing line.
    prev_close = window.rfind("\n```")
    if prev_close != -1:
        # Skip past the closing fence line.
        nl = window.find("\n", prev_close + 1)
        if nl != -1:
            window = window[nl + 1 :]
    return window


_AGGREGATE_PATH_RE = re.compile(
    # Filenames whose stem advertises aggregation, OR the .npz extension.
    r"\b\S*(?:regression|summary|aggregat\w*|per[-_]?cell|cell[-_]?level)\S*\.(?:csv|json|jsonl|tsv|parquet|npz)\b"
    r"|\b\S+\.npz\b",
    re.IGNORECASE,
)


_NOT_UPLOADED_RE = re.compile(
    r"(?:not\s+uploaded|not\s+available|did\s+not\s+upload"
    r"|raw\s+completions?\s+(?:were\s+)?(?:not|never)"
    r"|raw[-_\s]?completions?\s+(?:were\s+)?n/a)",
    re.IGNORECASE,
)


_CHERRY_DISCLOSURE_RE = re.compile(
    r"\b(?:cherry[-\s]?picked|random[-\s]?sample|drawn at random|"
    r"random draw|first \d+ of \d+|first \d+ completions?|"
    r"\d+ random completions?|\d+ randomly[-\s]?sampled)\b",
    re.IGNORECASE,
)


# ─── Checks ────────────────────────────────────────────────────────────────


# Minimum body length (chars). Bodies smaller than this are stubs / placeholders.
# Defense-in-depth against the cache → body.md silent-handoff failure
# (incident: task #385, 2026-05-25 — body.md read literally "placeholder" for
# ~26h while `has_clean_result=true`). Real clean-result bodies are >5,000
# chars; 500 is a conservative floor.
MIN_BODY_CHARS = 500

# Stub-content sentinels we positively recognize (case-insensitive).
STUB_TOKENS = {"placeholder", "tbd", "todo", "stub"}


def check_body_nonstub(body: str) -> CheckResult:
    """Check 0: body is not a stub / placeholder.

    Runs FIRST and (in `verify_text`) short-circuits the rest of the
    check chain when it FAILs, so the operator gets one clear fail-fast
    signal rather than a dozen cascading "<section> missing" errors from
    a body that's just the word `placeholder`. Triggers FAIL when ANY
    of:
      - body's non-frontmatter content is empty,
      - body's non-frontmatter content collapses to a single stub token
        (`placeholder`, `tbd`, `todo`, `stub`) after whitespace strip,
      - body is < MIN_BODY_CHARS (500) characters,
      - body has no `# <title>` H1 line (clean-result bodies always carry
        one; non-clean-result bodies do not run through this verifier).

    The H1 sub-check here is appropriate because `verify_task_body.py`
    is only ever invoked against clean-result bodies (analyzer Step 5,
    clean-result-critic Step 1 pre-pass). Non-clean-result bodies
    (proposed-task idea captures, clarifier output) take different
    shapes and are not gated by this verifier; the CLI-level
    `_assert_body_nontrivial` in `scripts/task.py` does NOT impose the
    H1 requirement so those bodies can be `set-body`-written normally.
    """
    stripped = body.strip()
    n_chars = len(stripped)
    if n_chars == 0:
        return CheckResult(
            "body is not a stub",
            False,
            "body is empty — cache → body.md handoff likely failed; see analyzer.md Step 6",
        )
    if stripped.casefold() in STUB_TOKENS:
        return CheckResult(
            "body is not a stub",
            False,
            f"body is literally the stub token {stripped!r} — "
            "cache → body.md handoff likely failed; see analyzer.md Step 6",
        )
    if n_chars < MIN_BODY_CHARS:
        return CheckResult(
            "body is not a stub",
            False,
            f"body is only {n_chars} chars (floor {MIN_BODY_CHARS}) — "
            "real clean-result bodies are >5 KB. If this is intentional, "
            "check that the analyzer's cache → body.md handoff did not silently "
            "drop the clean-result content.",
        )
    if find_h1_title(body) is None:
        return CheckResult(
            "body is not a stub",
            False,
            "body has no `# <title>` H1 line — real clean-result bodies always "
            "start with an H1; this looks like a stub or a truncated handoff.",
        )
    return CheckResult(
        "body is not a stub",
        True,
        f"{n_chars} chars + H1 present",
    )


def check_title_confidence(body: str) -> CheckResult:
    title = find_h1_title(body)
    if not title:
        return CheckResult("title confidence tag", False, "no H1 found")
    m = re.search(r"\((LOW|MODERATE|HIGH) confidence\)\s*$", title)
    if not m:
        return CheckResult(
            "title confidence tag",
            False,
            f"title must end with '(LOW|MODERATE|HIGH confidence)' — got: {title[-60:]!r}",
        )
    return CheckResult("title confidence tag", True, f"level={m.group(1)}")


def check_required_sections(body: str) -> CheckResult:
    found = [name for name, _, _ in find_h2_sections(body)]
    missing = [s for s in REQUIRED_H2_SECTIONS if s not in found]
    label = "three required H2 sections in order"
    if missing:
        return CheckResult(
            label,
            False,
            f"missing: {', '.join(missing)} (found: {found})",
        )
    # Order check: REQUIRED_H2_SECTIONS must appear in this order within `found`,
    # with `## Figure` (optional) allowed to sit between TL;DR and Details.
    seq = [s for s in found if s in REQUIRED_H2_SECTIONS]
    if seq != REQUIRED_H2_SECTIONS:
        return CheckResult(
            label,
            False,
            f"wrong order — got {seq}, expected {REQUIRED_H2_SECTIONS}",
        )
    # If `## Figure` is present, it must sit between TL;DR and Details.
    if "Figure" in found:
        positions = {name: i for i, (name, _, _) in enumerate(find_h2_sections(body))}
        if not (positions["TL;DR"] < positions["Figure"] < positions["Details"]):
            return CheckResult(
                label,
                False,
                "`## Figure` (optional) must sit between `## TL;DR` and `## Details`",
            )
    return CheckResult(label, True)


def check_tldr_labels(body: str) -> CheckResult:
    """Check 3: TL;DR carries the three REQUIRED labels (Motivation /
    What I ran / Results). The fourth `Next steps` bullet is OPTIONAL —
    bodies that include it still PASS; bodies that omit it also PASS
    (decision: 2026-05-26, iterations.md). Padding a Next-steps bullet
    just to satisfy the verifier was the failure mode this change drops.
    """
    tldr = section_text(body, "TL;DR")
    if tldr is None:
        return CheckResult(
            "TL;DR bullets carry the three required labels", False, "TL;DR section missing"
        )
    missing = []
    for label in TLDR_BULLETS_REQUIRED:
        # accept either `**Motivation:**` or `Motivation:` at start of line / after `-`.
        if not re.search(rf"(?im)^\s*[-*]\s*(\*\*)?{re.escape(label)}(\*\*)?\s*:", tldr):
            missing.append(label)
    if missing:
        return CheckResult(
            "TL;DR bullets carry the three required labels",
            False,
            f"missing labels: {', '.join(missing)}",
        )
    return CheckResult("TL;DR bullets carry the three required labels", True)


def _gather_figure_image_urls(body: str) -> list[str]:
    """Collect image URLs from `## Figure` (if present) AND inline images in
    `## TL;DR`. Powers checks 4 / 4b / 5 under the optional-Figure spec
    (2026-05-26): the hero image may live in either section."""
    urls: list[str] = []
    for section in ("Figure", "TL;DR"):
        text = section_text(body, section)
        if text is None:
            continue
        urls.extend(_IMAGE_RE.findall(text))
    return urls


def check_figure_image(body: str) -> CheckResult:
    """Check 4: at least one `![alt](url)` image exists in `## Figure` OR
    inline under `## TL;DR` (one-takeaway-one-figure pattern)."""
    urls = _gather_figure_image_urls(body)
    if not urls:
        return CheckResult(
            "hero image present",
            False,
            "no `![alt](path)` image found in `## TL;DR` or `## Figure`",
        )
    return CheckResult("hero image present", True, f"{len(urls)} image(s)")


def check_figure_url_resolvable(body: str) -> CheckResult:
    """Check 4b: every image URL in `## Figure` or inline-under-`## TL;DR`
    must be a permanent, dashboard-resolvable URL.

    The EPS dashboard serves task-folder HTML artifacts but NOT PNG/PDF
    binaries under `tasks/<N>/artifacts/`, so a relative `artifacts/hero.png`
    reference renders as a broken image in the browser (incident: task #365,
    2026-05-22). Acceptable patterns are absolute URLs only — typically
    `https://raw.githubusercontent.com/<owner>/<repo>/<sha>/figures/.../*.png`
    or any other `https://...` URL the browser can fetch directly.
    """
    urls = _gather_figure_image_urls(body)
    if not urls:
        # Image-present check (check 4) handles the missing-image case; if
        # there is no image at all, treat this check as vacuously passing so
        # the operator sees one error message, not two.
        return CheckResult("Figure URL resolvable", True, "no images to check")
    bad: list[str] = []
    for url in urls:
        url = url.strip()
        # Strip optional title — `(url "title")` — keep only the URL token.
        url = url.split(None, 1)[0] if url else url
        if not url:
            bad.append("empty URL")
            continue
        if url.startswith(("http://", "https://")):
            # Permanence rule for GitHub raw URLs — match the spirit of
            # check_repro_url_permanence (no moving branches in the path).
            if re.search(
                r"^https?://raw\.githubusercontent\.com/[^/]+/[^/]+/(main|master|HEAD)\b",
                url,
            ):
                bad.append(f"figure URL pinned to moving ref: `{url}`")
            continue
        # Anything not absolute is rejected — relative `artifacts/...`,
        # `tasks/...`, `figures/...`, `./...`, `../...` all render broken
        # on the dashboard. Push the file to GitHub (typically under
        # `figures/issue_<N>/`) and reference it via the raw URL pinned
        # to a commit SHA.
        bad.append(
            f"figure URL is relative (`{url}`) — push to `figures/issue_<N>/` "
            "and reference via `https://raw.githubusercontent.com/.../<sha>/...`"
        )
    if bad:
        return CheckResult("Figure URL resolvable", False, "; ".join(bad))
    return CheckResult("Figure URL resolvable", True, f"{len(urls)} URL(s)")


def check_figure_caption(body: str) -> CheckResult:
    """Check 5: figure caption.

    If `## Figure` is present, the first non-image line under it must have
    ≥10 words. If `## Figure` is absent (one-takeaway-one-figure pattern,
    2026-05-26), the check is vacuously satisfied — inline image alt-text
    serves as the caption (which `check_figure_image` validates implicitly
    by requiring an `![alt](url)` shape, and the analyzer is instructed to
    write descriptive alt text).
    """
    figure = section_text(body, "Figure")
    if figure is None:
        return CheckResult(
            "Figure caption ≥10 words",
            True,
            "no `## Figure` H2 — inline-image alt-text used instead (one-takeaway-one-figure)",
        )
    # Caption = first non-image, non-empty line after the image markdown.
    caption_line = None
    for line in figure.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("!["):
            continue
        # Strip italic markers
        candidate = stripped.strip("*").strip("_").strip()
        # A caption typically starts with "Caption:" or is just italic text.
        if candidate.lower().startswith("caption:"):
            candidate = candidate[len("caption:") :].strip()
        if candidate:
            caption_line = candidate
            break
    if not caption_line:
        return CheckResult("Figure caption ≥10 words", False, "no caption found under the image")
    word_count = len(re.findall(r"\b\w+\b", caption_line))
    if word_count < 10:
        return CheckResult(
            "Figure caption ≥10 words",
            False,
            f"caption has {word_count} words — needs ≥10. Caption: {caption_line[:80]!r}",
        )
    return CheckResult("Figure caption ≥10 words", True, f"{word_count} words")


def check_confidence_matches(body: str) -> CheckResult:
    """Check 6: Details `Confidence: …` line matches title and carries ≥20 chars of rationale."""
    title = find_h1_title(body) or ""
    m = re.search(r"\((LOW|MODERATE|HIGH) confidence\)\s*$", title)
    if not m:
        return CheckResult(
            "Details confidence sentence matches title", False, "no title confidence"
        )
    title_level = m.group(1)
    details = section_text(body, "Details")
    if details is None:
        return CheckResult(
            "Details confidence sentence matches title", False, "Details section missing"
        )
    # Look for `Confidence: LOW|MODERATE|HIGH — <rationale>` (em-dash or
    # ASCII hyphen; en-dash deliberately excluded — em-dash is the spec).
    cm = re.search(
        r"Confidence:\s*(LOW|MODERATE|HIGH)\b\s*[—\-]\s*(.+?)(?:\n\n|\Z|\n##)",
        details,
        flags=re.DOTALL,
    )
    if not cm:
        # Try the looser form (no dash) — still flag the level mismatch / missing
        # rationale separately so the user sees what's wrong.
        loose = re.search(r"Confidence:\s*(LOW|MODERATE|HIGH)\b", details)
        if not loose:
            return CheckResult(
                "Details confidence sentence matches title",
                False,
                "no `Confidence: LOW|MODERATE|HIGH — <rationale>` line in Details",
            )
        return CheckResult(
            "Details confidence sentence matches title",
            False,
            f"`Confidence: {loose.group(1)}` line missing the `— <rationale>` clause",
        )
    body_level = cm.group(1)
    rationale = cm.group(2).strip()
    # Trim trailing markdown noise / multiple lines down to a single rationale clause.
    rationale = rationale.split("\n\n")[0].strip()
    if body_level != title_level:
        return CheckResult(
            "Details confidence sentence matches title",
            False,
            f"title says {title_level}, Details says {body_level}",
        )
    if len(rationale) < MIN_CONFIDENCE_RATIONALE_CHARS:
        return CheckResult(
            "Details confidence sentence matches title",
            False,
            f"rationale after `—` is only {len(rationale)} chars "
            f"(need ≥{MIN_CONFIDENCE_RATIONALE_CHARS}): {rationale[:60]!r}",
        )
    return CheckResult(
        "Details confidence sentence matches title",
        True,
        f"both {title_level}, rationale={len(rationale)} chars",
    )


def check_repro_subgroups(body: str) -> CheckResult:
    """Check 7: `## Reproducibility` contains all three boldface subgroup labels."""
    repro = section_text(body, "Reproducibility")
    if repro is None:
        return CheckResult(
            "Reproducibility three subgroups present", False, "Reproducibility section missing"
        )
    missing: list[str] = []
    for label in REPRO_SUBGROUPS:
        # Boldface label of the form **Artifacts:** (allow `Artifacts**:` etc.).
        if not re.search(rf"\*\*\s*{re.escape(label)}\s*:?\s*\*\*", repro):
            missing.append(label)
    if missing:
        return CheckResult(
            "Reproducibility three subgroups present",
            False,
            f"missing **bold** labels in Reproducibility: {', '.join(missing)}",
        )
    return CheckResult(
        "Reproducibility three subgroups present", True, "Artifacts + Compute + Code"
    )


def check_repro_url_permanence(body: str) -> CheckResult:
    """Check 8: every URL in `## Reproducibility` is pinned to a permanent ref."""
    repro = section_text(body, "Reproducibility")
    if repro is None:
        return CheckResult(
            "Reproducibility URL permanence", False, "Reproducibility section missing"
        )
    bad: list[str] = []
    # HF Hub URLs must include /tree/<ref>, /blob/<ref>, /raw/<ref>, or @<ref>.
    hf_urls = re.findall(r"https?://huggingface\.co/[^\s\)<>]+", repro)
    for url in hf_urls:
        if not (
            "/tree/" in url
            or "/blob/" in url
            or "/raw/" in url
            or re.search(r"@[A-Za-z0-9._-]+", url)
        ):
            bad.append(f"unpinned HF URL `{url}` (needs `/tree/<ref>`)")
        elif re.search(r"/(tree|blob|raw)/(main|master|HEAD)\b", url):
            bad.append(f"unpinned HF URL `{url}` (pinned to moving branch)")
    # WandB URLs should be /runs/<id>, /groups/<id>, or /reports/<id>.
    wandb_urls = re.findall(r"https?://(?:www\.)?wandb\.ai/[^\s\)<>]+", repro)
    for url in wandb_urls:
        if "/runs/" not in url and "/groups/" not in url and "/reports/" not in url:
            bad.append(f"unpinned WandB URL `{url}` (needs `/runs/<id>`)")
    # GitHub URLs should be /blob/<sha> or /tree/<sha>, not /blob/main.
    gh_urls = re.findall(r"https?://github\.com/[^\s\)<>]+", repro)
    for url in gh_urls:
        if re.search(r"/(blob|tree)/(main|master|HEAD)\b", url):
            bad.append(f"unpinned GitHub URL `{url}` (use `/blob/<sha>`)")
    if bad:
        return CheckResult("Reproducibility URL permanence", False, "; ".join(bad))
    return CheckResult("Reproducibility URL permanence", True)


def check_repro_sentinel_scrub(body: str) -> CheckResult:
    """Check 9: no placeholder sentinels (`{{`, `TBD`, `see config`, `default`)
    in `## Reproducibility`."""
    repro = section_text(body, "Reproducibility")
    if repro is None:
        return CheckResult(
            "Reproducibility sentinel scrub", False, "Reproducibility section missing"
        )
    bad: list[str] = []
    for s in SENTINEL_SUBSTRINGS:
        if s == "{{":
            if "{{" in repro:
                bad.append("`{{` placeholder")
        else:
            # `default` matched as a standalone word (avoid false positives like
            # `default_factory`); the others matched case-insensitively as words.
            if re.search(rf"\b{re.escape(s)}\b", repro, flags=re.IGNORECASE):
                bad.append(f"`{s}`")
    if bad:
        return CheckResult(
            "Reproducibility sentinel scrub",
            False,
            "; ".join(bad) + " — use `n/a` explicitly for inapplicable fields",
        )
    return CheckResult("Reproducibility sentinel scrub", True)


def check_cherry_picked_label(body: str) -> CheckResult:
    """Check 10: every sample-output fenced block in `## Details` is preceded
    by a cherry-picked / random-sample disclosure in the prelude prose.
    """
    details = section_text(body, "Details")
    if details is None:
        return CheckResult("Cherry-picked label discipline", False, "Details section missing")
    samples = _iter_sample_fences(details)
    if not samples:
        return CheckResult(
            "Cherry-picked label discipline",
            True,
            "no sample-output fenced blocks in Details",
        )
    flagged: list[str] = []
    for start, _, content in samples:
        prelude = _prelude_window(details, start)
        if _CHERRY_DISCLOSURE_RE.search(prelude):
            continue
        # First content line, trimmed, as a hint to the user.
        first_line = content.strip().splitlines()[0][:60] if content.strip() else "(empty)"
        flagged.append(first_line)
    if flagged:
        preview = "; ".join(f"'{x}'" for x in flagged[:2]) + (" …" if len(flagged) > 2 else "")
        return CheckResult(
            "Cherry-picked label discipline",
            False,
            f"{len(flagged)} of {len(samples)} sample block(s) lack a cherry-picked / "
            f"random-sample disclosure in the prelude prose: {preview}",
        )
    return CheckResult(
        "Cherry-picked label discipline",
        True,
        f"{len(samples)} sample block(s) labelled",
    )


def check_qualitative_data_link(body: str) -> CheckResult:
    """Check 11: every sample-output fenced block in `## Details` is preceded
    by at least one link or backtick-path that is NOT an aggregate-only path.
    An explicit `not uploaded` escape downgrades FAIL to WARN.
    """
    details = section_text(body, "Details")
    if details is None:
        return CheckResult("Qualitative-data link", False, "Details section missing")
    samples = _iter_sample_fences(details)
    if not samples:
        return CheckResult(
            "Qualitative-data link",
            True,
            "no sample-output fenced blocks in Details",
        )
    fails: list[str] = []
    warns: list[str] = []
    passes = 0
    for start, _, content in samples:
        prelude = _prelude_window(details, start)
        # Collect candidate tokens: markdown link URLs + backtick-wrapped paths.
        tokens: list[str] = []
        tokens.extend(_LINK_RE.findall(prelude))
        tokens.extend(_CODE_RE.findall(prelude))
        has_escape = bool(_NOT_UPLOADED_RE.search(prelude))
        first_line = content.strip().splitlines()[0][:60] if content.strip() else "(empty)"

        if not tokens:
            if has_escape:
                warns.append(f"'{first_line}': no link, `not uploaded` escape acknowledged")
            else:
                fails.append(f"'{first_line}': no link or path in prelude paragraph")
            continue

        qualitative_hit = any(not _AGGREGATE_PATH_RE.search(tok) for tok in tokens)
        if qualitative_hit:
            passes += 1
            continue

        if has_escape:
            warns.append(
                f"'{first_line}': only aggregate-pattern links, `not uploaded` escape acknowledged"
            )
        else:
            fails.append(
                f"'{first_line}': only aggregate-pattern links "
                f"(e.g. {tokens[0][:60]}); raw text-level artifact required"
            )

    if fails:
        return CheckResult(
            "Qualitative-data link",
            False,
            f"{len(fails)} sample block(s) lack a qualitative-data link: "
            + "; ".join(fails[:2])
            + (" …" if len(fails) > 2 else ""),
        )
    if warns:
        return CheckResult(
            "Qualitative-data link",
            True,
            f"{len(warns)} sample block(s) ship with `not uploaded` escape — "
            "follow-up should re-run with raw-completion upload",
            is_warn=True,
        )
    return CheckResult(
        "Qualitative-data link",
        True,
        f"{passes} sample block(s) link to a qualitative-data artifact",
    )


def check_goal_present(body: str, fm: dict) -> CheckResult:
    """Soft INFO check — Goal-of-experiment frontmatter field.

    Reports presence / absence of the canonical agent-facing Goal:
    frontmatter ``goal: <non-empty string>``. The body-side ``## Goal``
    H2 is intentionally NOT checked here — clean-result bodies drop the
    visible H2 and fold the Goal text into the TL;DR Motivation bullet
    (decision: 2026-05-26). The visible H2 lives only in proposed /
    planning bodies, where /issue Step 0c (workflow.yaml §
    gates.experiment_goal) is the enforcement point.

    The frontmatter ``goal:`` field stays in clean-result bodies so
    downstream agents (planner, critic, follow-up-proposer) have the
    agent-facing canonical Goal as context.

    This check NEVER FAILs. Clean-result bodies for non-experiment kinds,
    follow-ups, and pre-Goal bodies legitimately omit the field; failing
    them here would block promotion needlessly. The check is exposed for
    orchestrator visibility and tagged WARN when missing so the
    orchestrator can pick it up without halting.

    NOTE: ``body`` is accepted but no longer inspected. Kept in the
    signature so the call site in ``verify_text`` stays uniform with
    the body-only checks in ``CHECKS``.
    """
    del body  # body-side `## Goal` H2 intentionally not checked
    fm_goal = fm.get("goal")
    fm_goal = fm_goal.strip() if isinstance(fm_goal, str) and fm_goal.strip() else None
    if fm_goal:
        return CheckResult(
            "Goal-of-experiment field",
            True,
            f"frontmatter goal present ({len(fm_goal)} chars)",
        )
    return CheckResult(
        "Goal-of-experiment field",
        True,
        "missing: frontmatter `goal:` field (soft — enforced at /issue Step 0c, not here)",
        is_warn=True,
    )


# ─── Driver ────────────────────────────────────────────────────────────────


CHECKS = [
    check_body_nonstub,
    check_title_confidence,
    check_required_sections,
    check_tldr_labels,
    check_figure_image,
    check_figure_url_resolvable,
    check_figure_caption,
    check_confidence_matches,
    check_repro_subgroups,
    check_repro_url_permanence,
    check_repro_sentinel_scrub,
    check_cherry_picked_label,
    check_qualitative_data_link,
]


def verify_text(raw: str, *, source: str = "") -> tuple[bool, list[CheckResult]]:
    fm, body = split_frontmatter(raw)
    if LEGACY_SAGAN_CARD_SENTINEL in body:
        return True, [
            CheckResult(
                "legacy Sagan-card detected",
                True,
                "skipping markdown spec — body is grandfathered HTML; "
                "run verify_sagan_card.py for those bodies",
            )
        ]
    # Check 0 (body-nonstub) short-circuits the rest of the chain when it
    # FAILs. A stub body would otherwise cascade into a dozen "<section>
    # missing" errors that bury the actual root cause (the cache → body.md
    # silent-handoff failure). Returning a single FAIL gives the operator
    # one clear signal pointing at analyzer.md Step 6.
    stub_result = check_body_nonstub(body)
    if not stub_result.passed:
        return False, [stub_result]
    results = [stub_result] + [chk(body) for chk in CHECKS[1:]]
    # Goal-of-experiment field is a soft INFO/WARN check — it never
    # FAILs (enforcement is at /issue Step 0c, not here) and needs the
    # frontmatter, so it lives outside the body-only CHECKS list.
    results.append(check_goal_present(body, fm))
    overall = all(r.passed for r in results)
    return overall, results


def _load_text_for_issue(number: int) -> tuple[str, Path]:
    from explore_persona_space.task_workflow import find_task_path  # local import

    folder = find_task_path(number)
    body_path = folder / "body.md"
    return body_path.read_text(), body_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--issue", type=int, help="task number to verify")
    grp.add_argument("--file", help="path to a body.md to verify")
    grp.add_argument("--body-stdin", action="store_true", help="read body from stdin")
    args = parser.parse_args()

    if args.issue is not None:
        try:
            raw, source_path = _load_text_for_issue(args.issue)
            source = str(source_path)
        except FileNotFoundError as e:
            print(f"verify_task_body: {e}", file=sys.stderr)
            return 2
    elif args.file:
        raw = Path(args.file).read_text()
        source = args.file
    else:
        raw = sys.stdin.read()
        source = "<stdin>"

    overall, results = verify_text(raw, source=source)
    print(f"verify_task_body — {source}")
    for r in results:
        print(r.render())
    print()
    if overall:
        print("OVERALL: PASS")
        return 0
    n_fail = sum(1 for r in results if not r.passed)
    print(f"OVERALL: FAIL ({n_fail} of {len(results)} checks failed)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
