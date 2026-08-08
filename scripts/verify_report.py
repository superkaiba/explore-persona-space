#!/usr/bin/env python3
"""verify_report.py — mechanical verifier for v2 report clean-result bodies.

The v2 workflow retires agent interpretation of results: agents author a
fixed-structure REPORT (Motivation / Methodology (shared) / Results-as-plots
with a result-specific ``**Methodology**`` block per result) and Thomas alone
writes the claims — the ``# Result:`` title, the TLDR, every per-result
``**Takeaways**`` block, and Conclusion and next steps.
This is the mechanical gate for that report form, the report-track analogue of
``verify_task_body.py`` (markdown v4) and ``verify_paper.py`` (paper track).
Canonical skeleton: ``.claude/skills/issue-v2/report-template.md``.

A report body carries the sentinel ``<!-- report-v1 -->`` on the line after its
H1 title (mirroring ``<!-- clean-result-v4 -->``).

Required structure (both modes):
  - H1 line ``# Experiment: <question>`` at generation time; promote mode also
    accepts the Thomas-retitled ``# Result: <claim>`` form.
  - Sentinel ``<!-- report-v1 -->`` as the first non-blank line after the H1.
  - Five H2 sections, in this exact relative order: ``## Motivation``,
    ``## TLDR``, ``## Methodology (shared)``, ``## Results``,
    ``## Conclusion and next steps``. A trailing colon on any heading is
    accepted and ignored, and the grandfathered pre-2026-07-30 names
    ``## Methodology`` / ``## Next steps`` normalize to the canonical ones.
    There is NO separate ``## Metrics`` section — metric definitions +
    rationale live inside ``## Methodology (shared)`` (its final
    ``**Metrics:**`` block) or inside a result's ``**Methodology**`` block.
  - ``## Results`` contains >=1 ``### <name>`` subsection, each with a
    non-empty description paragraph, exactly one image reference
    ``![...](...)``, exactly one ``**Takeaways**`` block (trailing colon
    accepted), AND a ``**Methodology**`` block (REQUIRED in generation mode;
    a grandfathered body missing it only WARNs in promote mode). The retired
    ``**Plot:**`` label FAILs in generation mode (tolerated at promote for
    grandfathered bodies).
  - ``detailed-writeup-link``: the body links its detailed companion writeup
    (``docs/reports/issue_<N>_detailed.md``) via a SHA-pinned GitHub blob /
    raw URL on a ``**Detailed writeup:**`` line — REQUIRED at generation,
    WARN-if-absent at promote (grandfathered); well-formedness + issue match
    only, no repo/network read.
  - Every referenced local image path exists on disk (resolved vs
    ``--figures-root``; default: the git-repo root of ``--file``).
  - Every ``htmlpreview.github.io`` link embeds a full 40-hex SHA/revision
    ``raw.githubusercontent`` or ``gist.githubusercontent`` URL
    (well-formedness only, no network).
  - ``image-pin-format``: every ``## Results:`` image is a well-formed
    ``https://raw.githubusercontent.com/<owner>/<repo>/<40-hex-sha>/figures/issue_<N>/...``
    pin, all Results images name ONE issue number (== ``--expect-issue`` /
    ``--issue`` when known). Images outside Results are exempt from the pin
    requirement and the issue-number match, but a raw.githubusercontent image
    there still gets format well-formedness + the identity ladder.
  - ``image-pin-blob-identity``: each well-formed pin is verified against the
    LOCAL git object DB — ``git hash-object <local>`` vs
    ``git rev-parse <sha>:<path>`` — read-only local git, NO network. The
    degrade ladder is mode-split: non-git checkout → WARN (both modes);
    unresolvable pinned commit → FAIL in generation (the pin commit was just
    created locally; unresolvable = fabricated SHA) / WARN in promote
    (unfetched clone plausible); commit present but path absent → FAIL (both);
    blob mismatch → FAIL in generation / WARN in promote (post-merge local
    drift — the pin is the record, #922); pin resolves with no local copy →
    WARN in generation / PASS-note in promote. Mixed SHAs across Results pins
    are fine per-pin (the 7b re-entry / partial-re-splice shape).
  - ``committed-under-claims`` (#2191): every "committed under ``<path>``" /
    "in git under ``<path>``" claim (case-insensitive trigger bigram followed
    by a backticked path) is verified against the LOCAL git object DB at the
    pin(s) the claim's OWN LINE names — inline hex runs (URL spans and the
    claimed path itself excluded) resolved via ``rev-parse``, plus backticked
    ``issue-<N>`` / ``origin/issue-<N>`` / ``main`` branch tokens. A claim
    FAILs ONLY when every resolvable same-line pin shows zero blobs
    (``ls-tree -r``) for at least one expanded ``{a, b}`` brace member; no
    resolvable pin → WARN (the detail carries an informational
    issue-branch-tip probe); negated claims (a 40-char preceding-window token
    scan — NOT a lookbehind), URL / absolute / ellipsis-abbreviated /
    slash-less paths → skipped with a note; non-git root → WARN. Deliberately
    blind, pinned by test: a SUBSET claim over a NON-empty directory PASSes
    (the #2162 round-1 witnessed shape) — mechanical subset semantics over
    free text would be a false-FAIL channel.
  - ``code-sha-cards`` (#2191): every USABLE reproducibility-card commit — a
    ``git_commit`` / ``final_commit_sha`` value that is full 40-hex with
    sibling ``git_dirty`` not true, collected by a recursive key walk over
    ``eval_results/issue_<N>/**/*.json`` in the working tree UNION the
    ``issue-<N>`` / ``origin/issue-<N>`` refs (≤5 MB per file; unparseable /
    oversize files skipped + counted) — must be CITED somewhere in the report
    (a ≥8-hex run that is a prefix of the SHA; ``…``-abbreviated citations
    count). An uncited usable card commit FAILs at generation (report and
    card set are contemporaneous) and WARNs at promote (the card set is
    external mutable state and may have grown since authoring). WARN-only
    companions, both modes: (b2) a cited usable SHA absent from a
    ``| Code SHAs |`` table row; (b3) a best-effort label→card token pairing
    over the row's ``·`` / ``;`` segments (unresolvable segments silently
    skipped + counted). Issue number: ``--issue`` / ``--expect-issue``, else
    inferred from the ``**Detailed writeup:**`` line; unknown → WARN-skip; no
    cards anywhere → PASS-note N/A. Abbreviated / non-hex / dirty card values
    are EXCLUDED from every FAIL/WARN set and listed in the detail.

Mode-specific:
  - ``generation``: TLDR AND Conclusion-and-next-steps content MUST be exactly
    the placeholder ``*(Thomas fills in)*`` (Thomas has not written them yet),
    and every Results subsection's ``**Takeaways**`` block must be exactly the
    placeholder too. Interpretive lexicon scan over Methodology (shared) +
    Results (Motivation is exempt — hypothesis framing is allowed there).
  - ``promote``: TLDR content MUST be non-placeholder AND non-empty (Thomas has
    filled it). Thomas's prose — TLDR, Conclusion and next steps, and the
    Results takeaways / claim headings he filled in — is NEVER lexicon-checked;
    only Methodology (shared) (pure agent prose in both modes) stays
    lexicon-scanned; structural checks still apply.

``--manifest`` (optional, both modes): validates the manifest against
``.claude/skills/issue-v2/planned_manifest.schema.json``, then checks every
planned condition/metric appears (word-boundary match) in the report text and
every planned figure id/title EXACT-matches a ``### `` subsection heading OR is
explicitly marked ``not run`` on the same line.

Verbatim worked examples — fenced code blocks and blockquotes — are DATA, not
agent assertions (the template mandates them in Methodology). Section-parsing,
the interpretive-lexicon scan, image-existence, and the duplicate-heading scan
run on a copy of the body with those lines blanked (line numbers preserved), so
a ``## `` inside a fence never registers as a section heading and a ``suggests``
/ ``![x](y)`` inside an example is not flagged. A duplicate top-level required
``## `` heading FAILs (``duplicate-section``).

Input: exactly one of ``--file <body.md>`` or ``--issue <N>`` (the latter
resolves ``tasks/<status>/<N>/body.md`` via the task-workflow library).

Exit 0 PASS / 1 FAIL / 2 usage error. Prints one line per check.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPORT_SENTINEL = "<!-- report-v1 -->"
H1_TITLE_PREFIX = "Experiment: "
H1_RESULT_PREFIX = "Result: "
PLACEHOLDER = "*(Thomas fills in)*"
TAKEAWAYS_LINE = "**Takeaways**"
METHODOLOGY_LINE = "**Methodology**"

# The five required H2 sections, in the exact order they must appear.
# Stored in NORMALIZED form (no trailing colon) — a heading line is matched
# via _norm_header(), so `## TLDR` and `## TLDR:` are both accepted.
REQUIRED_SECTIONS = [
    "## Motivation",
    "## TLDR",
    "## Methodology (shared)",
    "## Results",
    "## Conclusion and next steps",
]

# Grandfathered pre-2026-07-30 heading names (the original report-v1 shape) —
# normalized to the canonical names so old bodies keep verifying.
SECTION_ALIASES = {
    "## Methodology": "## Methodology (shared)",
    "## Next steps": "## Conclusion and next steps",
}

# Sections whose (agent-authored) prose is scanned for interpretive lexicon,
# per mode. Motivation is deliberately EXEMPT (hypothesis-to-be-tested framing
# is allowed there); TLDR / Conclusion and next steps are Thomas's prose and
# are NEVER scanned. Results is scanned only at GENERATION time — at promote
# it carries Thomas's filled Takeaways + claim-shaped headings, his voice.
LEXICON_SECTIONS_BY_MODE = {
    "generation": ("## Methodology (shared)", "## Results"),
    "promote": ("## Methodology (shared)",),
}


def _norm_header(line: str) -> str:
    """Canonical form of a heading line: stripped, trailing ':' removed,
    grandfathered section names mapped to their canonical replacements."""
    h = line.rstrip().rstrip(":").rstrip()
    return SECTION_ALIASES.get(h, h)


def _is_bold_label(line: str, label: str) -> bool:
    """Whether ``line`` is the bold ``**<label>**`` block opener.

    The canonical form has no trailing colon (``**Takeaways**``); the
    grandfathered pre-2026-07-30 ``**Takeaways:**`` form is accepted too.
    """
    s = line.strip()
    return s in (f"**{label}**", f"**{label}:**")


# Conservative list of asserted-conclusion lexemes banned from agent sections.
BANNED_LEXICON = [
    "suggests",
    "confirms",
    "demonstrates that",
    "evidence that",
    "evidence for",
    "we conclude",
    "this shows",
    "indicating that",
    "implying",
]
_LEXICON_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in BANNED_LEXICON) + r")\b",
    re.IGNORECASE,
)

# Markdown inline image: ![alt](url ["title"]).
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
# A URL token (used to find htmlpreview links).
_URL_RE = re.compile(r"https?://[^\s)\]<>\"']+")
_SHA40_RE = re.compile(r"[0-9a-fA-F]{40}")
# A SHA-pinned raw.githubusercontent permalink: owner / repo / 40-hex sha / path.
_RAW_PIN_RE = re.compile(
    r"^https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([0-9a-fA-F]{40})/(.+)$"
)
# The repo-relative figure path a Results pin must carry.
_FIGURES_ISSUE_RE = re.compile(r"^figures/issue_(\d+)/")
# The body's detailed-companion-writeup link line + its SHA-pinned URL forms
# (GitHub blob or raw.githubusercontent), path docs/reports/issue_<N>_detailed.md.
_DETAILED_LINE_RE = re.compile(r"^\s*\*\*Detailed writeup:\*\*\s*(\S+)")
_DETAILED_URL_RE = re.compile(
    r"^https://(?:github\.com/[^/]+/[^/]+/blob|raw\.githubusercontent\.com/[^/]+/[^/]+)/"
    r"([0-9a-fA-F]{40})/docs/reports/issue_(\d+)_detailed\.md$"
)


def _git(repo: Path, *args: str) -> tuple[int, str]:
    """Run a READ-ONLY git command in ``repo``; return (returncode, stripped stdout).

    Used by the image-pin blob-identity check — local object-DB lookups only
    (``rev-parse`` / ``cat-file`` / ``hash-object``), never a network call.
    """
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip()


_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / ".claude"
    / "skills"
    / "issue-v2"
    / "planned_manifest.schema.json"
)


def _word_match(needle: str, haystack: str) -> bool:
    r"""Whether ``\b<needle>\b`` occurs in ``haystack``.

    Callers lowercase both sides for case-insensitive matching. Word-boundary,
    not bare substring, so a planned name that is only a fragment of a longer
    word in the report (``eval`` inside ``evaluation``) does not count as a hit.
    """
    if not needle:
        return False
    return re.search(r"\b" + re.escape(needle) + r"\b", haystack) is not None


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    is_warn: bool = False  # WARN counts as PASS for overall, rendered [WARN].

    def render(self) -> str:
        tag = "WARN" if self.is_warn else ("PASS" if self.passed else "FAIL")
        line = f"  [{tag}] {self.name}"
        if self.detail:
            line += f" — {self.detail}"
        return line


# ─── Body parsing ────────────────────────────────────────────────────────────


@dataclass
class Section:
    header: str  # e.g. "## TLDR:"
    header_line: int  # 1-based line number in the body
    content_start_line: int  # 1-based line number of the first content line
    content_lines: list[str]

    @property
    def content(self) -> str:
        return "\n".join(self.content_lines)


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter dict, body). Mirrors verify_task_body.split_frontmatter."""
    if not text.startswith("---\n"):
        return {}, text
    rest = text[4:]
    end = rest.find("\n---\n")
    if end == -1:
        return {}, text
    return {}, rest[end + len("\n---\n") :]


def blank_verbatim(lines: list[str]) -> list[str]:
    """Return a copy of ``lines`` with fenced-code-block and blockquote lines
    blanked to ``""`` (line count + numbering preserved).

    Verbatim worked examples — fenced code (```` ``` ```` / ``~~~``, including
    an info string on the opener) and blockquotes (a leading optionally-indented
    ``>``) — are DATA, not agent assertions; the template mandates them in
    Methodology. Section-parsing, the interpretive-lexicon scan, image-existence,
    and the duplicate-heading scan run on the blanked copy, so a ``## `` inside a
    fence never registers as a section heading and a ``suggests`` / ``![x](y)``
    inside a verbatim example is not flagged. An unterminated fence blanks to
    end-of-document (CommonMark behavior).
    """
    out: list[str] = []
    in_fence = False
    fence_marker = ""
    for line in lines:
        stripped = line.lstrip()
        if in_fence:
            out.append("")  # every line inside the fence, incl. the closer
            if stripped.startswith(fence_marker):
                in_fence, fence_marker = False, ""
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = True
            fence_marker = "```" if stripped.startswith("```") else "~~~"
            out.append("")  # blank the opening fence line too
            continue
        if stripped.startswith(">"):
            out.append("")  # blockquote line → verbatim data
            continue
        out.append(line)
    return out


def find_h1(lines: list[str]) -> tuple[int, str] | None:
    """Return (0-based line index, title-after-'# ') of the first H1, or None."""
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("# ") and not s.startswith("## "):
            return i, s[2:].strip()
    return None


def parse_sections(lines: list[str]) -> list[Section]:
    """Split the body into ``## `` H2 sections (``### `` lines stay as content)."""
    header_idxs = [i for i, ln in enumerate(lines) if ln.startswith("## ")]
    sections: list[Section] = []
    for pos, i in enumerate(header_idxs):
        end = header_idxs[pos + 1] if pos + 1 < len(header_idxs) else len(lines)
        sections.append(
            Section(
                header=lines[i].rstrip(),
                header_line=i + 1,
                content_start_line=i + 2,
                content_lines=lines[i + 1 : end],
            )
        )
    return sections


def section_map(sections: list[Section]) -> dict[str, Section]:
    """First occurrence of each required header (normalized) → its Section."""
    out: dict[str, Section] = {}
    for sec in sections:
        key = _norm_header(sec.header)
        if key in REQUIRED_SECTIONS and key not in out:
            out[key] = sec
    return out


# ─── Structural checks (both modes) ────────────────────────────────────────


def check_h1_and_sentinel(lines: list[str], mode: str) -> list[CheckResult]:
    results: list[CheckResult] = []
    h1 = find_h1(lines)
    if h1 is None:
        results.append(CheckResult("h1-title", False, "no H1 title found"))
        results.append(CheckResult("sentinel", False, "no H1 title to anchor the sentinel"))
        return results
    h1_idx, title = h1
    # Generation: agents have no finding to claim, so the H1 must be the
    # question form. Promote: Thomas retitles to the claim form `# Result: ...`
    # (preferred); a not-yet-retitled `# Experiment: ...` is still accepted.
    allowed = (H1_TITLE_PREFIX,) if mode == "generation" else (H1_RESULT_PREFIX, H1_TITLE_PREFIX)
    if any(title.startswith(p) for p in allowed):
        results.append(CheckResult("h1-title", True, f"H1 = '{title}'"))
    else:
        want = " or ".join(f"'# {p}...'" for p in allowed)
        results.append(
            CheckResult("h1-title", False, f"H1 must start with {want}, got '# {title}'")
        )
    # Sentinel = first non-blank line after the H1.
    first_after = None
    for j in range(h1_idx + 1, len(lines)):
        if lines[j].strip():
            first_after = lines[j].strip()
            break
    if first_after == REPORT_SENTINEL:
        results.append(CheckResult("sentinel", True, f"{REPORT_SENTINEL} present after H1"))
    elif first_after is None:
        results.append(CheckResult("sentinel", False, "no content after the H1 title"))
    else:
        results.append(
            CheckResult(
                "sentinel",
                False,
                f"first non-blank line after H1 must be '{REPORT_SENTINEL}', got '{first_after}'",
            )
        )
    return results


def check_required_sections(sections: list[Section]) -> list[CheckResult]:
    present = section_map(sections)
    missing = [h for h in REQUIRED_SECTIONS if h not in present]
    results: list[CheckResult] = []
    if missing:
        results.append(
            CheckResult("required-sections", False, "missing section(s): " + ", ".join(missing))
        )
    else:
        results.append(CheckResult("required-sections", True, "all five required sections present"))
    # Order: the present required headers must appear in the required relative
    # order. Compare the order of first-occurrence line numbers.
    ordered_present = [h for h in REQUIRED_SECTIONS if h in present]
    lines_in_order = [present[h].header_line for h in ordered_present]
    if lines_in_order == sorted(lines_in_order):
        results.append(CheckResult("section-order", True, "required sections in correct order"))
    else:
        # Report the actual document order for diagnosis.
        actual = sorted(ordered_present, key=lambda h: present[h].header_line)
        results.append(
            CheckResult(
                "section-order",
                False,
                "required sections out of order; document order: " + " -> ".join(actual),
            )
        )
    return results


def check_duplicate_sections(lines: list[str]) -> CheckResult:
    """FAIL if any of the five required ``## `` headings appears more than once.

    Scanned on the fence/blockquote-blanked body, so a required heading string
    inside a verbatim example does not count. ``section_map`` silently keeps the
    FIRST occurrence, so a stray duplicate would otherwise slip past the
    structural checks entirely.
    """
    occurrences: dict[str, list[int]] = {}
    for i, ln in enumerate(lines, 1):
        header = _norm_header(ln)
        if header in REQUIRED_SECTIONS:
            occurrences.setdefault(header, []).append(i)
    dups = {h: ls for h, ls in occurrences.items() if len(ls) > 1}
    if dups:
        detail = "; ".join(
            f"{h} at lines {', '.join(str(x) for x in ls)}" for h, ls in dups.items()
        )
        return CheckResult("duplicate-section", False, "duplicate required heading(s): " + detail)
    return CheckResult("duplicate-section", True, "no duplicate required headings")


def _images_in(text: str) -> list[str]:
    return [m.group(1).split()[0].strip() for m in _IMAGE_RE.finditer(text) if m.group(1).strip()]


def check_results_subsections(sections: list[Section], mode: str) -> CheckResult:
    present = section_map(sections)
    results_sec = present.get("## Results")
    if results_sec is None:
        return CheckResult("results-subsections", False, "no ## Results section")
    lines = results_sec.content_lines
    sub_idxs = [i for i, ln in enumerate(lines) if ln.startswith("### ")]
    if not sub_idxs:
        return CheckResult("results-subsections", False, "## Results has no ### <name> subsection")
    problems: list[str] = []
    warns: list[str] = []
    for pos, i in enumerate(sub_idxs):
        end = sub_idxs[pos + 1] if pos + 1 < len(sub_idxs) else len(lines)
        name = lines[i].strip()[4:].strip()
        block = lines[i + 1 : end]
        block_text = "\n".join(block)
        imgs = _images_in(block_text)
        if len(imgs) != 1:
            problems.append(f"'{name}': expected exactly 1 image, found {len(imgs)}")
        # Description = a non-blank line that is not solely an image reference
        # and not part of the block scaffolding (the Takeaways / Methodology
        # labels, the placeholder, or the grandfathered bold plot label).
        has_desc = any(
            ln.strip()
            and not _IMAGE_RE.fullmatch(ln.strip())
            and not _is_bold_label(ln, "Takeaways")
            and not _is_bold_label(ln, "Methodology")
            and ln.strip() != PLACEHOLDER
            and not ln.strip().startswith("**Plot:")
            for ln in block
        )
        if not has_desc:
            problems.append(f"'{name}': missing a non-empty description paragraph")
        # The retired **Plot:** label (pre-2026-07-30 shape) must not appear
        # in a freshly assembled report; grandfathered bodies keep it at
        # promote time.
        if mode == "generation" and any(ln.strip().startswith("**Plot:") for ln in block):
            problems.append(
                f"'{name}': the '**Plot:**' label is retired (2026-07-30) — "
                "the image follows the Methodology block directly"
            )
        # A **Methodology** block per result (the result-specific recipe +
        # what-is-plotted). REQUIRED at generation (freshly assembled reports
        # follow the current template); a grandfathered pre-2026-07-30 body
        # missing it only WARNs at promote.
        meth_count = sum(1 for ln in block if _is_bold_label(ln, "Methodology"))
        if meth_count != 1:
            msg = f"'{name}': expected exactly 1 '{METHODOLOGY_LINE}' block, found {meth_count}"
            if mode == "generation" or meth_count > 1:
                problems.append(msg)
            else:
                warns.append(msg + " (grandfathered pre-2026-07-30 shape)")
        # Exactly one **Takeaways** block per result (Thomas's claim slot).
        tk_idxs = [j for j, ln in enumerate(block) if _is_bold_label(ln, "Takeaways")]
        if len(tk_idxs) != 1:
            problems.append(
                f"'{name}': expected exactly 1 '{TAKEAWAYS_LINE}' block, found {len(tk_idxs)}"
            )
        elif mode == "generation":
            # At generation the Takeaways content must be the intact
            # placeholder — the claims under a plot are Thomas's to write.
            tail = "\n".join(block[tk_idxs[0] + 1 :]).strip()
            if tail != PLACEHOLDER:
                problems.append(
                    f"'{name}': Takeaways must be exactly the placeholder "
                    f"'{PLACEHOLDER}' at generation time"
                )
    if problems:
        detail = "; ".join(problems)
        if warns:
            detail += "; warn: " + "; ".join(warns)
        return CheckResult("results-subsections", False, detail)
    if warns:
        return CheckResult("results-subsections", True, "; ".join(warns), is_warn=True)
    return CheckResult(
        "results-subsections",
        True,
        f"{len(sub_idxs)} subsection(s), each with 1 image + description + Methodology + Takeaways",
    )


def check_image_files(body: str, figures_root: Path) -> CheckResult:
    missing: list[str] = []
    checked = 0
    for url in _images_in(body):
        low = url.lower()
        if low.startswith(("http://", "https://", "//", "data:")):
            continue
        checked += 1
        p = Path(url)
        resolved = p if p.is_absolute() else (figures_root / url)
        if not resolved.is_file():
            missing.append(url)
    if missing:
        return CheckResult(
            "figure-files-exist",
            False,
            f"missing on disk (root={figures_root}): " + ", ".join(missing),
        )
    return CheckResult("figure-files-exist", True, f"{checked} local image path(s) exist")


def check_detailed_writeup_link(
    blanked_lines: list[str], *, mode: str, expect_issue: int | None
) -> CheckResult:
    """``detailed-writeup-link`` (two-document output, 2026-07-30).

    The body is the SUMMARIZED layer and must link its detailed companion
    writeup (``docs/reports/issue_<N>_detailed.md``) via a SHA-pinned GitHub
    blob / raw URL on a ``**Detailed writeup:**`` line. REQUIRED at generation
    (freshly assembled reports follow the current template); a grandfathered
    body without one only WARNs at promote. Well-formedness + issue-number
    match only — existence at the pinned SHA is the report-verifier agent's
    read (same no-network philosophy as ``htmlpreview-sha``).
    """
    name = "detailed-writeup-link"
    matches = [m for ln in blanked_lines if (m := _DETAILED_LINE_RE.match(ln)) is not None]
    if len(matches) > 1:
        # A follow-up round's re-pin must REPLACE the old line, not stack a
        # fresh one on top (first-match-wins would silently keep the stale
        # link in the body).
        return CheckResult(
            name,
            False,
            f"{len(matches)} '**Detailed writeup:**' lines — exactly one is allowed "
            "(a follow-up re-pin replaces the old line)",
        )
    match = matches[0] if matches else None
    if match is None:
        if mode == "generation":
            return CheckResult(
                name,
                False,
                "no '**Detailed writeup:**' link line — the summarized body must link "
                "docs/reports/issue_<N>_detailed.md (SHA-pinned)",
            )
        return CheckResult(
            name,
            True,
            "no '**Detailed writeup:**' link (grandfathered pre-2026-07-30 body)",
            is_warn=True,
        )
    url = match.group(1).strip().strip("<>")
    m = _DETAILED_URL_RE.match(url)
    if m is None:
        return CheckResult(
            name,
            False,
            f"'{url}' is not a well-formed SHA-pinned "
            "github.com/<owner>/<repo>/blob/<40-hex>/docs/reports/issue_<N>_detailed.md "
            "(or raw.githubusercontent equivalent) link",
        )
    if expect_issue is not None and m.group(2) != str(expect_issue):
        return CheckResult(
            name,
            False,
            f"detailed-writeup link names issue {m.group(2)} != expected issue {expect_issue}",
        )
    return CheckResult(name, True, f"SHA-pinned detailed-writeup link (issue {m.group(2)})")


def check_htmlpreview(body: str) -> CheckResult:
    urls = [u for u in _URL_RE.findall(body) if "htmlpreview.github.io" in u]
    if not urls:
        return CheckResult("htmlpreview-sha", True, "no htmlpreview links (N/A)", is_warn=False)
    bad: list[str] = []
    for u in urls:
        # Repo blobs pin via raw.githubusercontent.com/<owner>/<repo>/<sha>/;
        # gist-hosted dashboards pin via gist.githubusercontent.com/.../raw/<rev>/.
        pinned_host = "raw.githubusercontent.com" in u or "gist.githubusercontent.com" in u
        if not pinned_host or not _SHA40_RE.search(u):
            bad.append(u)
    if bad:
        return CheckResult(
            "htmlpreview-sha",
            False,
            "htmlpreview link(s) missing a 40-hex-SHA raw.githubusercontent URL: " + ", ".join(bad),
        )
    return CheckResult("htmlpreview-sha", True, f"{len(urls)} htmlpreview link(s) SHA-pinned")


# ─── Image-pin checks (#1224 mechanization; both modes, no network) ─────────


def _check_pin_blob_identity(
    pins: list[tuple[str, str, str]], *, mode: str, figures_root: Path
) -> CheckResult:
    """Verify each well-formed ``(url, sha, path)`` pin against the LOCAL git
    object DB (read-only ``_git`` calls, never a network fetch).

    Mode-split degrade ladder (see the module docstring): generation is strict
    (the pipeline path where the pin commit + local copies exist by
    construction at 7e), promote is lenient (fresh-clone / post-merge shapes).
    Returns ONE CheckResult: FAIL if any pin fails, else WARN if any pin
    warned, else PASS.
    """
    name = "image-pin-blob-identity"
    if not pins:
        return CheckResult(name, True, "no well-formed pins to verify (N/A)")
    rc, _ = _git(figures_root, "rev-parse", "--git-dir")
    if rc != 0:
        return CheckResult(
            name,
            True,
            f"{figures_root} is not a git checkout; blob identity unverifiable",
            is_warn=True,
        )
    fails: list[str] = []
    warns: list[str] = []
    notes: list[str] = []
    for _url, sha, path in pins:
        rc, _ = _git(figures_root, "cat-file", "-e", f"{sha}^{{commit}}")
        if rc != 0:
            # At 7e the pin commit was JUST created in this worktree / shared
            # object DB, so an unresolvable SHA is definitively wrong (the
            # fabricated-SHA class); post-merge an unfetched clone is plausible.
            msg = f"pinned commit {sha[:12]} unresolvable in the local object DB ({path})"
            if mode == "generation":
                fails.append(msg)
            else:
                warns.append(msg + "; unfetched clone possible post-merge, identity unverifiable")
            continue
        rc, blob_id = _git(figures_root, "rev-parse", f"{sha}:{path}")
        if rc != 0:
            fails.append(f"pinned commit {sha[:12]} does not contain {path}")
            continue
        local = figures_root / path
        if not local.is_file():
            # At 7e every Results figure should have a just-plotted local copy;
            # its absence is suspicious (e.g. a wrong-path pin colliding with a
            # previously-committed figure name). Post-merge it is expected.
            msg = f"{path}@{sha[:12]} resolves in object DB; no local copy to compare"
            if mode == "generation":
                warns.append(msg)
            else:
                notes.append(msg)
            continue
        rc, local_id = _git(figures_root, "hash-object", str(local))
        if rc != 0 or not local_id:
            warns.append(f"git hash-object failed on local copy {path}; identity unverifiable")
            continue
        if local_id != blob_id:
            msg = f"local {path} differs from the blob pinned at {sha[:12]}"
            if mode == "generation":
                fails.append(msg)
            else:
                warns.append(msg + " (post-merge local drift; the pin is the record)")
        else:
            notes.append(f"{path}@{sha[:12]} matches local copy")
    if fails:
        detail = "; ".join(fails)
        if warns:
            detail += "; warn: " + "; ".join(warns)
        return CheckResult(name, False, detail)
    if warns:
        return CheckResult(name, True, "; ".join(warns), is_warn=True)
    return CheckResult(name, True, "; ".join(notes) or f"{len(pins)} pin(s) verified")


def check_image_pins(
    sections: list[Section],
    blanked_body: str,
    *,
    mode: str,
    figures_root: Path,
    expect_issue: int | None = None,
) -> list[CheckResult]:
    """``image-pin-format`` + ``image-pin-blob-identity`` (#1224).

    Runs on the BLANKED body (fenced/blockquote example images are DATA and
    exempt — the same discipline as ``check_image_files``). Every image inside
    ``## Results:`` must be a well-formed
    ``raw.githubusercontent.com/<owner>/<repo>/<40-hex>/figures/issue_<N>/...``
    pin; all Results-image issue numbers must be identical (and equal to
    ``expect_issue`` when provided). Images OUTSIDE Results are exempt from the
    pin requirement and the issue-number match (a legitimate non-blockquoted
    cross-issue prior-figure reference in Motivation must not FAIL); when they
    ARE raw.githubusercontent URLs they still get format well-formedness + the
    identity ladder. Mixed SHAs across pins are fine per-pin.
    """
    results_sec = section_map(sections).get("## Results")
    results_imgs = _images_in(results_sec.content) if results_sec is not None else []
    outside_imgs = list(_images_in(blanked_body))
    for u in results_imgs:
        if u in outside_imgs:
            outside_imgs.remove(u)

    format_problems: list[str] = []
    pins: list[tuple[str, str, str]] = []  # (url, sha, repo-relative path)
    results_issue_nums: set[str] = set()

    for url in results_imgs:
        m = _RAW_PIN_RE.match(url)
        fig_m = _FIGURES_ISSUE_RE.match(m.group(4)) if m else None
        if m is None or fig_m is None:
            format_problems.append(
                f"Results image '{url}' is not a well-formed "
                "raw.githubusercontent.com/<owner>/<repo>/<40-hex-sha>/figures/issue_<N>/... pin"
            )
            continue
        pins.append((url, m.group(3), m.group(4)))
        results_issue_nums.add(fig_m.group(1))

    if len(results_issue_nums) > 1:
        format_problems.append(
            "Results pins name multiple issue numbers: " + ", ".join(sorted(results_issue_nums))
        )
    if expect_issue is not None and results_issue_nums - {str(expect_issue)}:
        format_problems.append(
            f"Results pin issue number(s) {sorted(results_issue_nums)} "
            f"!= expected issue {expect_issue}"
        )

    for url in outside_imgs:
        if "raw.githubusercontent.com" not in url:
            continue
        m = _RAW_PIN_RE.match(url)
        if m is None:
            format_problems.append(
                f"non-Results raw.githubusercontent image '{url}' is not a "
                "well-formed 40-hex-SHA pin"
            )
            continue
        pins.append((url, m.group(3), m.group(4)))

    if format_problems:
        fmt = CheckResult("image-pin-format", False, "; ".join(format_problems))
    elif pins:
        fmt = CheckResult("image-pin-format", True, f"{len(pins)} pinned image(s) well-formed")
    else:
        fmt = CheckResult("image-pin-format", True, "no pinned images (N/A)")

    return [fmt, _check_pin_blob_identity(pins, mode=mode, figures_root=figures_root)]


# ─── Interpretive-lexicon check (agent sections; both modes) ────────────────


def check_lexicon(sections: list[Section], mode: str) -> CheckResult:
    present = section_map(sections)
    hits: list[str] = []
    for header in LEXICON_SECTIONS_BY_MODE[mode]:
        sec = present.get(header)
        if sec is None:
            continue
        for offset, line in enumerate(sec.content_lines):
            for m in _LEXICON_RE.finditer(line):
                lineno = sec.content_start_line + offset
                hits.append(f"{header[3:]} L{lineno}: '{m.group(0)}'")
    if hits:
        return CheckResult(
            "no-interpretive-lexicon",
            False,
            "banned interpretive lexeme(s): " + "; ".join(hits),
        )
    return CheckResult(
        "no-interpretive-lexicon", True, "no asserted-conclusion lexemes in agent sections"
    )


# ─── Mode-specific TLDR / Next-steps checks ─────────────────────────────────


def _section_text(sections: list[Section], header: str) -> str | None:
    sec = section_map(sections).get(header)
    return None if sec is None else sec.content.strip()


def check_placeholders(sections: list[Section]) -> list[CheckResult]:
    """generation mode: TLDR + Conclusion and next steps must be the untouched
    placeholder (Thomas has not written them yet)."""
    results: list[CheckResult] = []
    for header, name in (
        ("## TLDR", "tldr-placeholder"),
        ("## Conclusion and next steps", "conclusion-placeholder"),
    ):
        text = _section_text(sections, header)
        if text is None:
            results.append(CheckResult(name, False, f"{header} section missing"))
        elif text == PLACEHOLDER:
            results.append(CheckResult(name, True, f"{header} is the intact placeholder"))
        else:
            results.append(
                CheckResult(
                    name,
                    False,
                    f"{header} must be exactly the placeholder '{PLACEHOLDER}' at generation time",
                )
            )
    return results


def check_tldr_filled(sections: list[Section]) -> CheckResult:
    """promote mode: TLDR must be filled (non-empty AND not the placeholder)."""
    text = _section_text(sections, "## TLDR")
    if text is None:
        return CheckResult("tldr-filled", False, "## TLDR section missing")
    if not text:
        return CheckResult("tldr-filled", False, "## TLDR is empty; Thomas must write the TLDR")
    if text == PLACEHOLDER:
        return CheckResult(
            "tldr-filled", False, "## TLDR still the placeholder; Thomas must write the TLDR"
        )
    return CheckResult("tldr-filled", True, "## TLDR is filled")


# ─── Manifest checks (optional; both modes) ─────────────────────────────────


def _load_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text())


def check_manifest(
    blanked_body: str, sections: list[Section], manifest_path: Path
) -> list[CheckResult]:
    # jsonschema is imported lazily: only a --manifest run needs it, so the core
    # structural verifier carries no dependency on it (it is transitive-only).
    import jsonschema

    results: list[CheckResult] = []
    try:
        manifest = _load_manifest(manifest_path)
    except (OSError, json.JSONDecodeError) as e:
        return [CheckResult("manifest-schema", False, f"cannot read/parse manifest: {e}")]

    schema = json.loads(_SCHEMA_PATH.read_text())
    try:
        jsonschema.validate(manifest, schema)
        results.append(
            CheckResult("manifest-schema", True, "manifest matches planned_manifest.schema.json")
        )
    except jsonschema.ValidationError as e:
        # A schema failure poisons the coverage checks; report and stop here.
        loc = "/".join(str(p) for p in e.absolute_path) or "<root>"
        return [CheckResult("manifest-schema", False, f"schema violation at {loc}: {e.message}")]

    low_body = blanked_body.lower()

    def _coverage(name: str, items: list[str]) -> CheckResult:
        # Word-boundary match, not bare substring, so a planned name that is
        # only a fragment of a longer word in the report (``eval`` inside
        # ``evaluation``) does not falsely count as covered.
        missing = [it for it in items if not _word_match(it.lower(), low_body)]
        if missing:
            return CheckResult(name, False, "not found in report text: " + ", ".join(missing))
        return CheckResult(name, True, f"all {len(items)} present in report text")

    results.append(_coverage("manifest-conditions", list(manifest.get("conditions", []))))
    results.append(_coverage("manifest-metrics", list(manifest.get("metrics", []))))

    # Figure coverage: a planned figure is covered iff its id OR title
    # EXACT-matches (case-insensitive, stripped) one of the report's ###
    # subsection headings, OR the body has an explicit "not run" on the same
    # line as a word-boundary occurrence of the id/title.
    heading_set = {
        ln.strip()[4:].strip().lower()
        for sec in sections
        for ln in sec.content_lines
        if ln.startswith("### ") and ln.strip()[4:].strip()
    }
    body_lines_low = [ln.lower() for ln in blanked_body.splitlines()]
    fig_missing: list[str] = []
    for fig in manifest.get("figures", []):
        fid = str(fig.get("id", "")).strip()
        title = str(fig.get("title", "")).strip()
        keys = [k.lower() for k in (fid, title) if k]
        in_heading = any(k in heading_set for k in keys)
        marked_not_run = any(
            "not run" in ln and any(_word_match(k, ln) for k in keys) for ln in body_lines_low
        )
        if not (in_heading or marked_not_run):
            fig_missing.append(fid or title or "<unnamed figure>")
    if fig_missing:
        results.append(
            CheckResult(
                "manifest-figures",
                False,
                "planned figure(s) with no matching ### subsection and not marked 'not run': "
                + ", ".join(fig_missing),
            )
        )
    else:
        results.append(
            CheckResult(
                "manifest-figures", True, "every planned figure is plotted or marked 'not run'"
            )
        )
    return results


# ─── Committed-under claims + Code-SHA card coverage (#2191; both modes) ────

# A "committed under `<path>`" / "in git under `<path>`" claim: trigger bigram
# immediately followed by a backticked path.
_COMMITTED_UNDER_RE = re.compile(r"(?i)\b(?:committed|in\s+git)\s+under\s+`([^`]+)`")
# Negation guard — a preceding-window substring scan, NOT a lookbehind (an
# intervening word defeats an immediate lookbehind: "never LANDED in git
# under", "nothing IS committed under"). Both constants are pinned as
# behavior by tests/test_verify_report.py; not tunable at implementation.
_NEGATION_WINDOW_CHARS = 40
_NEGATION_TOKENS = (
    "not",
    "n't",
    "never",
    "no longer",
    "rather than",
    "instead of",
    "nothing",
)
# A hex run usable as a git pin candidate / SHA citation: 8-40 hex chars not
# embedded in a longer hex run.
_HEX_RUN_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{8,40}(?![0-9a-fA-F])")
# A backticked branch token on a claim line (`issue-2162`, `origin/issue-2162`,
# `main`) — resolved via refs/heads/<b> then refs/remotes/origin/<b>.
_BRANCH_TOKEN_RE = re.compile(r"^(?:origin/)?(?:issue-\d+|main)$")
# A `| Code SHAs | ... |` table row (scanned on BLANKED lines).
_CODE_SHA_ROW_RE = re.compile(r"(?i)^\s*\|\s*code[ -]?shas?\b")
# Reproducibility-card commit keys + the per-file JSON size guard.
_CARD_COMMIT_KEYS = frozenset({"git_commit", "final_commit_sha"})
_CARD_JSON_MAX_BYTES = 5_000_000
# b3 label→card pairing: stopwords removed from the CARD-side token set
# (path / filename-stem / phase tokens too generic to discriminate cards).
# Pinned by tests/test_verify_report.py — NOT tunable at implementation.
_CARD_TOKEN_STOPWORDS = frozenset(
    {"report", "json", "upload", "done", "card", "sentinel", "results", "gate", "gates"}
)
_SHA40_FULL_RE = re.compile(r"[0-9a-fA-F]{40}$")
_HEX_ABBREV_RE = re.compile(r"[0-9a-fA-F]{8,39}$")


def _infer_issue_from_lines(blanked_lines: list[str]) -> int | None:
    """Issue number from the report's own ``**Detailed writeup:**`` line, or None.

    Mirrors ``check_detailed_writeup_link``'s extraction (angle-bracket form
    accepted); that line is REQUIRED at generation and already issue-verified
    there, so the inference is mechanically pinned.
    """
    for ln in blanked_lines:
        m = _DETAILED_LINE_RE.match(ln)
        if m is None:
            continue
        um = _DETAILED_URL_RE.match(m.group(1).strip().strip("<>"))
        if um is not None:
            return int(um.group(2))
    return None


def _expand_brace_group(path: str) -> list[str]:
    """Expand ONE ``{a, b, c}`` brace group (member spaces stripped); no brace
    group → ``[path]`` unchanged."""
    m = re.search(r"\{([^{}]*)\}", path)
    if m is None:
        return [path]
    prefix, suffix = path[: m.start()], path[m.end() :]
    return [prefix + member.strip() + suffix for member in m.group(1).split(",")]


def _ls_tree_nonempty(figures_root: Path, pin: str, path: str) -> bool:
    """Whether ``git ls-tree -r <pin> -- <path>`` lists at least one blob."""
    rc, out = _git(figures_root, "ls-tree", "-r", "--name-only", pin, "--", path)
    return rc == 0 and bool(out)


def _same_line_pins(line: str, figures_root: Path) -> list[str]:
    """Resolve every same-line pin candidate to a full commit SHA (deduped,
    order-preserving).

    Candidates: (1) hex runs on the line AFTER blanking URL spans (an HF
    revision inside a URL must not be mistaken for a git pin) and the
    claimed-path backtick span(s) themselves, each resolved via
    ``rev-parse --verify <tok>^{commit}`` (abbreviations resolve iff
    unambiguous — git's own rule); (2) backticked branch tokens matching
    ``_BRANCH_TOKEN_RE``, via ``refs/heads/<b>`` then ``refs/remotes/origin/<b>``.
    """
    scrubbed = _URL_RE.sub(lambda m: " " * len(m.group(0)), line)
    scrubbed = _COMMITTED_UNDER_RE.sub(lambda m: " " * len(m.group(0)), scrubbed)
    resolved: list[str] = []
    for tok in _HEX_RUN_RE.findall(scrubbed):
        rc, sha = _git(figures_root, "rev-parse", "--verify", f"{tok}^{{commit}}")
        if rc == 0 and sha and sha not in resolved:
            resolved.append(sha)
    for btok in re.findall(r"`([^`]+)`", scrubbed):
        btok = btok.strip()
        if not _BRANCH_TOKEN_RE.match(btok):
            continue
        b = btok.removeprefix("origin/")
        for ref in (f"refs/heads/{b}", f"refs/remotes/origin/{b}"):
            rc, sha = _git(figures_root, "rev-parse", "--verify", f"{ref}^{{commit}}")
            if rc == 0 and sha:
                if sha not in resolved:
                    resolved.append(sha)
                break
    return resolved


def _branch_tip_probe(members: list[str], issue: int | None, figures_root: Path) -> str:
    """Informational suffix for the no-pin WARN: does the claimed path resolve
    at the issue's own branch tip?

    Severity stays WARN either way — escalating this probe to FAIL would
    import the deleted-later-at-tip false-FAIL class (a path correctly
    committed at the claimed pin but since removed at the tip).
    """
    if issue is None:
        return " (branch-tip probe unavailable — issue number unknown)"
    for ref in (f"issue-{issue}", f"origin/issue-{issue}"):
        rc, _ = _git(figures_root, "rev-parse", "--verify", f"{ref}^{{commit}}")
        if rc != 0:
            continue
        if all(_ls_tree_nonempty(figures_root, ref, m) for m in members):
            return f"; path resolves at `{ref}` tip"
        return f"; path also empty at `{ref}` tip"
    return f" (branch-tip probe: no issue-{issue} branch ref resolvable)"


def check_committed_under_claims(blanked_lines: list[str], figures_root: Path) -> CheckResult:
    """``committed-under-claims`` (#2191): verify every "committed under
    `<path>`" / "in git under `<path>`" claim against the LOCAL git object DB
    at the pin(s) the claim's own line names (read-only ``_git``; no network).

    Conservative by construction — the ONLY FAIL condition is: the line
    carries ≥1 resolvable pin AND every resolvable pin shows zero blobs for at
    least one expanded path member (any-pin-satisfies). No resolvable pin →
    WARN with an informational issue-branch-tip probe. Negated claims
    (preceding-window token scan), URL / absolute / ellipsis-abbreviated /
    slash-less paths → skipped with a note. Non-git ``figures_root`` → single
    WARN (mirrors ``_check_pin_blob_identity``). NAMED RESIDUE, pinned by
    test: a SUBSET claim over a NON-empty directory PASSes — the #2162
    round-1 witnessed shape — because free-text subset semantics (mapping
    claim nouns to filename tokens) is a live false-FAIL channel the task
    body's conservative-matcher instruction forbids.
    """
    name = "committed-under-claims"
    claim_rows: list[tuple[int, str, list[str]]] = []  # (line_no, line, raw paths)
    guarded: list[str] = []
    for i, ln in enumerate(blanked_lines, start=1):
        paths: list[str] = []
        for m in _COMMITTED_UNDER_RE.finditer(ln):
            window = ln[max(0, m.start() - _NEGATION_WINDOW_CHARS) : m.start()].lower()
            if any(tok in window for tok in _NEGATION_TOKENS):
                guarded.append(f"line {i}: negated claim skipped")
                continue
            paths.append(m.group(1))
        if paths:
            claim_rows.append((i, ln, paths))
    if not claim_rows:
        detail = "no committed-under claims (N/A)"
        if guarded:
            detail += "; " + "; ".join(guarded)
        return CheckResult(name, True, detail)
    rc, _ = _git(figures_root, "rev-parse", "--git-dir")
    if rc != 0:
        return CheckResult(
            name,
            True,
            f"{figures_root} is not a git checkout; committed-under claims unverifiable",
            is_warn=True,
        )
    issue = _infer_issue_from_lines(blanked_lines)
    fails: list[str] = []
    warns: list[str] = []
    notes: list[str] = []
    n_checked = 0
    for line_no, ln, raw_paths in claim_rows:
        pins = _same_line_pins(ln, figures_root)
        for raw_path in raw_paths:
            path = raw_path.strip().rstrip(":,").strip()
            if "://" in path:
                notes.append(f"line {line_no}: URL path `{path}` skipped")
                continue
            if path.startswith("/"):
                notes.append(f"line {line_no}: absolute path `{path}` skipped")
                continue
            if "…" in path or "..." in path:
                notes.append(f"line {line_no}: abbreviated path `{path}` skipped")
                continue
            members = []
            for member in _expand_brace_group(path):
                if "/" in member:
                    members.append(member)
                else:
                    notes.append(f"line {line_no}: slash-less path `{member}` skipped")
            if not members:
                continue
            if not pins:
                warns.append(
                    f"line {line_no}: claim `{path}` has no resolvable same-line pin — "
                    "add `at <sha>` / name the branch, or reword to an HF-home claim"
                    + _branch_tip_probe(members, issue, figures_root)
                )
                continue
            n_checked += 1
            satisfied = any(
                all(_ls_tree_nonempty(figures_root, pin, member) for member in members)
                for pin in pins
            )
            if satisfied:
                notes.append(f"line {line_no}: `{path}` resolves at a same-line pin")
            else:
                fails.append(
                    f"line {line_no}: claim `{path}` shows zero blobs at every same-line pin "
                    f"({', '.join(sha[:12] for sha in pins)}) — if these artifacts are "
                    "deliberately not in git (wave-output convention), reword the claim to "
                    "name their HF home; if they should be committed, commit them or fix "
                    "the path/pin"
                )
    if fails:
        detail = "; ".join(fails)
        if warns:
            detail += "; warn: " + "; ".join(warns)
        return CheckResult(name, False, detail)
    if warns:
        return CheckResult(name, True, "; ".join(warns), is_warn=True)
    parts = notes + guarded
    return CheckResult(name, True, "; ".join(parts) or f"{n_checked} claim(s) verified")


def _card_side_tokens(card_path: str, phase: object, issue: int) -> set[str]:
    """b3 card-side token set: path components + filename-stem words (split
    ``_``) under ``eval_results/issue_<N>/``, plus sibling ``phase`` value
    tokens (split ``-``/``_``), minus ``_CARD_TOKEN_STOPWORDS``."""
    rel = card_path.split(":", 1)[-1]
    prefix = f"eval_results/issue_{issue}/"
    if rel.startswith(prefix):
        rel = rel[len(prefix) :]
    parts = rel.split("/")
    tokens = {p.lower() for p in parts[:-1]}
    stem = parts[-1].removesuffix(".json") if parts else ""
    tokens.update(w.lower() for w in stem.split("_") if w)
    if isinstance(phase, str):
        tokens.update(w.lower() for w in re.split(r"[-_]", phase) if w)
    return tokens - _CARD_TOKEN_STOPWORDS


def _label_tokens(label: str) -> set[str]:
    """b3 segment-label tokens: lowercase, hyphens deleted (``stage-2`` →
    ``stage2``), split on non-alphanumerics."""
    return {t for t in re.split(r"[^0-9a-z]+", label.lower().replace("-", "")) if t}


def check_code_sha_cards(
    raw_body: str,
    blanked_lines: list[str],
    *,
    mode: str,
    figures_root: Path,
    expect_issue: int | None,
) -> CheckResult:
    """``code-sha-cards`` (#2191): every usable commit recorded in the issue's
    reproducibility cards must be cited somewhere in the report.

    Cards: recursive walk collecting every ``git_commit`` / ``final_commit_sha``
    value from ``eval_results/issue_<N>/**/*.json`` in the working tree UNION
    the ``issue-<N>`` / ``origin/issue-<N>`` refs (read-only ``_git``; ≤5 MB
    per file; parse failures skipped + counted). USABLE = full 40-hex with
    sibling ``git_dirty`` not True — abbreviated / "unknown" / dirty values
    are defective provenance from the CARD WRITER and are excluded from every
    FAIL/WARN set (including them would false-FAIL reports that correctly
    cite only full-hex commits). Citation = some ≥8-hex run in the RAW body
    (a citation inside a verbatim example still counts — conservative in the
    pass direction) is a prefix of the card SHA.

    (b1) coverage: an uncited usable card commit FAILs at ``generation`` and
    WARNs at ``promote`` — the card set is EXTERNAL MUTABLE STATE that keeps
    growing after authoring, so a promote-time miss must not block promotion
    of an unchanged good report (the mode-split degrade mirrors
    ``_check_pin_blob_identity``). (b2) WARN, both modes: a usable SHA cited
    in the report but absent from a ``| Code SHAs |`` row. (b3) WARN, both
    modes: best-effort label→card pairing over the row's ``·``/``;`` segments
    on the token-resolvable subset; unresolvable segments silently skipped.
    Degrades: unknown issue → WARN-skip; no card source anywhere → PASS-note.
    """
    name = "code-sha-cards"
    issue = expect_issue if expect_issue is not None else _infer_issue_from_lines(blanked_lines)
    if issue is None:
        return CheckResult(
            name,
            True,
            "issue number unknown — card check skipped (pass --issue/--expect-issue)",
            is_warn=True,
        )

    rel_prefix = f"eval_results/issue_{issue}/"
    # (source, json pointer, value, sibling git_dirty, sibling phase)
    records: list[tuple[str, str, str, object, object]] = []
    n_parse_failed = 0
    n_size_skipped = 0

    def _walk(obj: object, ptr: str, source: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in _CARD_COMMIT_KEYS and isinstance(v, str):
                    records.append(
                        (source, f"{ptr}/{k}", v, obj.get("git_dirty"), obj.get("phase"))
                    )
                else:
                    _walk(v, f"{ptr}/{k}", source)
        elif isinstance(obj, list):
            for idx, v in enumerate(obj):
                _walk(v, f"{ptr}/{idx}", source)

    found_source = False
    tree_dir = figures_root / "eval_results" / f"issue_{issue}"
    if tree_dir.is_dir():
        found_source = True
        for p in sorted(tree_dir.rglob("*.json")):
            try:
                if p.stat().st_size > _CARD_JSON_MAX_BYTES:
                    n_size_skipped += 1
                    continue
                obj = json.loads(p.read_text())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                n_parse_failed += 1
                continue
            _walk(obj, "", str(p.relative_to(figures_root)))
    seen_ref_commits: set[str] = set()
    for ref in (f"issue-{issue}", f"origin/issue-{issue}"):
        rc, ref_sha = _git(figures_root, "rev-parse", "--verify", f"{ref}^{{commit}}")
        if rc != 0 or not ref_sha:
            continue
        found_source = True
        if ref_sha in seen_ref_commits:
            continue  # both refs at the same commit — read once
        seen_ref_commits.add(ref_sha)
        rc, listing = _git(figures_root, "ls-tree", "-r", "--name-only", ref, "--", rel_prefix)
        if rc != 0:
            continue
        for path in listing.splitlines():
            if not path.endswith(".json"):
                continue
            rc, size = _git(figures_root, "cat-file", "-s", f"{ref}:{path}")
            if rc != 0 or not size.isdigit() or int(size) > _CARD_JSON_MAX_BYTES:
                n_size_skipped += 1
                continue
            rc, text = _git(figures_root, "show", f"{ref}:{path}")
            if rc != 0:
                n_parse_failed += 1
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                n_parse_failed += 1
                continue
            _walk(obj, "", f"{ref}:{path}")

    if not found_source:
        return CheckResult(
            name,
            True,
            f"no reproducibility cards found for issue {issue} "
            f"(working tree + issue-{issue}/origin/issue-{issue}) — card check skipped (N/A)",
        )
    if not records:
        detail = f"no git_commit/final_commit_sha records under {rel_prefix} JSONs (N/A)"
        skips = []
        if n_parse_failed:
            skips.append(f"{n_parse_failed} unreadable/unparseable JSON(s) skipped")
        if n_size_skipped:
            skips.append(f"{n_size_skipped} oversize JSON(s) skipped")
        if skips:
            detail += "; " + "; ".join(skips)
        return CheckResult(name, True, detail)

    # Usable-card classification (dirty first, then hex shape); dedupe by SHA.
    usable: dict[str, tuple[str, str, object]] = {}  # sha -> (source, ptr, phase) first-seen
    n_usable_records = 0
    n_dirty = 0
    n_abbrev = 0
    n_nonhex = 0
    usable_tokens: dict[str, set[str]] = {}  # sha -> union of its records' card-side tokens
    for source, ptr, value, dirty, phase in records:
        if dirty is True:
            n_dirty += 1
            continue
        if _SHA40_FULL_RE.fullmatch(value):
            sha = value.lower()
            n_usable_records += 1
            usable.setdefault(sha, (source, ptr, phase))
            usable_tokens.setdefault(sha, set()).update(_card_side_tokens(source, phase, issue))
        elif _HEX_ABBREV_RE.fullmatch(value):
            n_abbrev += 1
        else:
            n_nonhex += 1

    cited_tokens = {t.lower() for t in _HEX_RUN_RE.findall(raw_body)}

    def _is_cited(sha: str, tokens: set[str]) -> bool:
        return any(sha.startswith(t) for t in tokens)

    fails: list[str] = []
    warns: list[str] = []

    # (b1) card-coverage, whole-report scope — FAIL at generation, WARN at promote.
    for sha in sorted(usable):
        if _is_cited(sha, cited_tokens):
            continue
        source, ptr, _phase = usable[sha]
        msg = (
            f"reproducibility card `{source}` (`{ptr}`) records commit {sha[:12]}… which the "
            "report never cites — a run that legitimately spans commits should carry a "
            "per-phase Code-SHAs split (each phase @ its own card's commit), not a single "
            "SHA; if this phase is covered elsewhere under a different commit, the pairing "
            "is wrong"
        )
        if mode == "generation":
            fails.append(msg)
        else:
            warns.append(msg + " (promote: the card set may have grown since authoring)")

    # (b2) row-scope coverage + (b3) best-effort pairing — WARN in both modes.
    rows = [ln for ln in blanked_lines if _CODE_SHA_ROW_RE.match(ln)]
    n_unresolved_segments = 0
    if rows:
        row_tokens = {t.lower() for row in rows for t in _HEX_RUN_RE.findall(row)}
        for sha in sorted(usable):
            if _is_cited(sha, cited_tokens) and not _is_cited(sha, row_tokens):
                warns.append(
                    f"usable card commit {sha[:12]}… is cited in the report but absent from "
                    "the Code-SHAs row — carry the per-phase split in the row"
                )
        for row in rows:
            cells = [c.strip() for c in row.split("|")]
            value_cell = cells[2] if len(cells) > 2 else ""
            for segment in re.split(r"[·;]", value_cell):
                hexm = _HEX_RUN_RE.search(segment)
                if hexm is None:
                    continue
                label = segment[: hexm.start()] + segment[hexm.end() :]
                seg_tokens = _label_tokens(label)
                hit_shas = {sha for sha, toks in usable_tokens.items() if toks & seg_tokens}
                if len(hit_shas) != 1:
                    n_unresolved_segments += 1
                    continue
                (sha,) = hit_shas
                pin_tok = hexm.group(0).lower()
                if not sha.startswith(pin_tok):
                    warns.append(
                        f"Code-SHAs row segment '{segment.strip()[:60]}' pins "
                        f"{pin_tok[:12]}… but its label resolves to card commit {sha[:12]}… "
                        "— carry the per-phase split (each phase @ its own card's commit)"
                    )

    excl: list[str] = []
    if n_dirty:
        excl.append(f"{n_dirty} dirty record(s) excluded")
    if n_abbrev:
        excl.append(f"{n_abbrev} abbreviated (<40-hex) record(s) excluded")
    if n_nonhex:
        excl.append(f"{n_nonhex} non-hex record(s) excluded")
    n_dup = n_usable_records - len(usable)
    if n_dup:
        excl.append(f"{n_dup} duplicate usable record(s) deduped")
    if n_parse_failed:
        excl.append(f"{n_parse_failed} unreadable/unparseable JSON(s) skipped")
    if n_size_skipped:
        excl.append(f"{n_size_skipped} oversize JSON(s) skipped")
    if n_unresolved_segments:
        excl.append(f"{n_unresolved_segments} unresolvable row segment(s) skipped")
    excl_detail = "; ".join(excl)

    if fails:
        detail = "; ".join(fails)
        if warns:
            detail += "; warn: " + "; ".join(warns)
        if excl_detail:
            detail += "; " + excl_detail
        return CheckResult(name, False, detail)
    if warns:
        detail = "; ".join(warns)
        if excl_detail:
            detail += "; " + excl_detail
        return CheckResult(name, True, detail, is_warn=True)
    detail = f"{len(usable)} usable card commit(s) all cited"
    if excl_detail:
        detail += "; " + excl_detail
    return CheckResult(name, True, detail)


# ─── Driver ─────────────────────────────────────────────────────────────────


def verify_report_text(
    raw: str,
    *,
    mode: str,
    figures_root: Path,
    manifest_path: Path | None = None,
    expect_issue: int | None = None,
) -> tuple[bool, list[CheckResult]]:
    """Run all checks for ``mode``; return (overall_pass, results)."""
    if mode not in ("generation", "promote"):
        raise ValueError(f"mode must be generation|promote, got {mode!r}")
    _, body = split_frontmatter(raw)
    lines = body.splitlines()
    # Verbatim worked examples (fenced code + blockquotes) are DATA: blank them
    # (line numbers preserved) before section-parsing / lexicon / image-existence
    # / duplicate-heading scans. h1/sentinel + htmlpreview stay on the raw body
    # (never inside a fence/blockquote in a valid report).
    blanked_lines = blank_verbatim(lines)
    blanked_body = "\n".join(blanked_lines)
    sections = parse_sections(blanked_lines)

    results: list[CheckResult] = []
    results.extend(check_h1_and_sentinel(lines, mode))
    results.extend(check_required_sections(sections))
    results.append(check_duplicate_sections(blanked_lines))
    results.append(check_results_subsections(sections, mode))
    results.append(check_image_files(blanked_body, figures_root))
    results.append(check_detailed_writeup_link(blanked_lines, mode=mode, expect_issue=expect_issue))
    results.append(check_htmlpreview(body))
    results.extend(
        check_image_pins(
            sections, blanked_body, mode=mode, figures_root=figures_root, expect_issue=expect_issue
        )
    )
    results.append(check_lexicon(sections, mode))
    results.append(check_committed_under_claims(blanked_lines, figures_root))
    results.append(
        check_code_sha_cards(
            body, blanked_lines, mode=mode, figures_root=figures_root, expect_issue=expect_issue
        )
    )

    if mode == "generation":
        results.extend(check_placeholders(sections))
    else:  # promote
        results.append(check_tldr_filled(sections))

    if manifest_path is not None:
        results.extend(check_manifest(blanked_body, sections, manifest_path))

    overall = all(r.passed for r in results)
    return overall, results


def _default_figures_root(file_path: Path) -> Path:
    """The git-repo root of ``file_path`` (dir containing ``.git``), else its parent."""
    for d in [file_path.resolve().parent, *file_path.resolve().parents]:
        if (d / ".git").exists():
            return d
    return file_path.resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="path to a report body.md to verify")
    src.add_argument(
        "--issue",
        type=int,
        help="task id; resolves tasks/<status>/<N>/body.md via the task-workflow library",
    )
    parser.add_argument(
        "--mode", required=True, choices=["generation", "promote"], help="verification mode"
    )
    parser.add_argument(
        "--manifest", help="path to planned_manifest.json (optional coverage check)"
    )
    parser.add_argument(
        "--figures-root",
        help="root for resolving local image paths (default: the git-repo root of the body)",
    )
    parser.add_argument(
        "--expect-issue",
        type=int,
        help=(
            "issue number the ## Results: image pins must name (figures/issue_<N>/). "
            "Only meaningful with --file; with --issue the number is already known."
        ),
    )
    args = parser.parse_args(argv)

    if args.issue is not None and args.expect_issue is not None:
        parser.error(
            "--issue and --expect-issue are mutually exclusive: --issue already names the issue"
        )

    if args.issue is not None:
        # Resolve via the workflow library — NEVER hand-build tasks/<status>/<N>
        # (a cwd/worktree-relative path is stale; CLAUDE.md + the enforced
        # tests/test_no_direct_task_path_construction.py rule).
        from explore_persona_space.task_workflow import find_task_path

        try:
            file_path = find_task_path(args.issue) / "body.md"
        except FileNotFoundError as e:
            print(f"verify_report: {e}", file=sys.stderr)
            return 2
        if not file_path.is_file():
            print(
                f"verify_report: body.md not found for issue {args.issue}: {file_path}",
                file=sys.stderr,
            )
            return 2
    else:
        file_path = Path(args.file)
        if not file_path.is_file():
            print(f"verify_report: --file not found: {args.file}", file=sys.stderr)
            return 2
    raw = file_path.read_text()

    figures_root = (
        Path(args.figures_root) if args.figures_root else _default_figures_root(file_path)
    )

    manifest_path = None
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.is_file():
            print(f"verify_report: --manifest not found: {args.manifest}", file=sys.stderr)
            return 2

    expect = args.issue if args.issue is not None else args.expect_issue
    overall, results = verify_report_text(
        raw,
        mode=args.mode,
        figures_root=figures_root,
        manifest_path=manifest_path,
        expect_issue=expect,
    )
    print(f"verify_report — {file_path} (mode={args.mode})")
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
