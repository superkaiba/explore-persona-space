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
    ``issue-<N>`` / ``origin/issue-<N>`` refs (≤5 MB per file, walk depth
    ≤100; unparseable / non-UTF-8 / oversize / too-deep files skipped +
    counted) — must be CITED somewhere in the report
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
  - ``companion-content`` (#2198): the body's ``**Detailed writeup:**`` pin is
    resolved and the companion (``docs/reports/issue_<N>_detailed.md`` at the
    pinned SHA, materialized via read-only local ``git show`` — no network) is
    scanned on its ``blank_verbatim()``-blanked lines. Two halves, both
    mode-invariant (the companion is 100% agent-written in BOTH modes — it is
    regenerated wholesale on follow-up rounds, so hand-written slots there are
    destroyed without notice): (a) STRUCTURAL, FAIL in both modes — a
    Thomas-slot heading (a line normalizing to ``## TLDR`` or ``## Conclusion
    and next steps``, the alias map catching the grandfathered ``## Next
    steps``) or a ``**Takeaways**`` block opener; (b) LEXICON, WARN never
    FAIL — ``BANNED_LEXICON`` hits outside the companion's exact
    ``## Motivation`` section (the Motivation copy keeps the body's
    hypothesis-framing exemption). Resolution degrade ladder (mode-split,
    mirroring ``image-pin-blob-identity``): missing / stacked / malformed pin
    line → PASS-note N/A (``detailed-writeup-link`` carries that verdict);
    non-git root → WARN (both); pinned commit unresolvable → FAIL in
    generation (the pin was just created locally at assembly) / WARN in
    promote (unfetched clone plausible); commit present but companion path
    absent → FAIL (both); companion blob present but not valid UTF-8 → FAIL
    (both; contained decode error, never a crash).
  - ``stale-evidence-pins`` (#2195): every line citing an in-repo EVIDENCE
    FILE — a backticked repo-relative path under ``_EVIDENCE_PATH_PREFIXES``
    (``eval_results/`` / ``ood_eval_results/`` / ``figures/`` / ``docs/``;
    ``tasks/`` deliberately excluded — status-transition renames make any
    ``tasks/<status>/…`` citation read permanently stale) — at a same-line
    pin is checked for SUPERSESSION. Each candidate member associates with
    ONE pin (the first resolvable pin positioned AFTER the member's backtick
    span, else the nearest one BEFORE it; brace-group members share their
    group's association; span hygiene blanks the cited-path spans before
    hex-run scanning so a sha-like cited filename never self-pins), must
    resolve at that pin as exactly ONE blob EQUAL to the cited path
    (directory / single-file-directory / absent-at-pin citations skipped
    with a note — a home claim is not superseded by later additions and a
    broken citation is not a stale one), and is then read against the issue
    branch's AUTHORITATIVE tip (``origin/issue-<N>`` preferred; local
    ``issue-<N>`` ONLY when origin is absent) — behind a
    ``merge-base --is-ancestor`` guard (a non-ancestor pin is divergent
    history: staleness undecidable, skipped with a note) — via
    ``git log <pin>..<tip> -- <path>``. Non-empty ⇒ WARN in BOTH modes,
    never FAIL (an as-of pin is legitimate; the check cannot read
    contradiction), the detail enumerating up to
    ``_STALE_DETAIL_COMMITS_CAP`` newer commits (log order, newest first),
    deduped by (path, pin) and capped at ``_STALE_READ_CAP`` reads per
    report (counted note beyond — disclosed, never silent). ``| Code SHAs |``
    rows and no-pin lines are skipped (code provenance is
    ``code-sha-cards``' surface; a no-pin committed-under claim already gets
    ``committed-under-claims``' WARN). Degrades: non-git root → WARN; issue
    unresolvable / no branch ref → PASS-note; no candidates → PASS-note N/A.
    Named residues: a stale citation phrased WITHOUT a same-line pin is
    invisible by construction; artifacts modified on a DIFFERENT branch
    (main after a sibling merge) are out of scope; branch-token pins resolve
    ORIGIN-FIRST (an explicit `` `origin/<b>` `` token only at
    ``refs/remotes/origin/<b>``; a plain `` `<b>` `` token falls back to
    ``refs/heads/<b>`` only when origin is absent — mirroring the
    authoritative-tip policy, so a worktree-local ``issue-<N>`` lagging
    origin cannot false-WARN a token-pinned citation), which makes a token
    naming the AUTHORITATIVE branch fresh at verify time (same-ref log range
    is empty); a token naming a DIFFERENT ref (`` `main` ``, a foreign
    `` `issue-<M>` ``) resolves at THAT ref's tip and may legitimately WARN.

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


def _git(repo: Path, *args: str, strip: bool = True) -> tuple[int, str]:
    """Run a READ-ONLY git command in ``repo``; return (returncode, stdout).

    stdout is ``.strip()``-ed by default (right for ref/hash plumbing output);
    pass ``strip=False`` for byte-faithful text — the companion
    materialization needs leading blank lines preserved so reported ``L<n>``
    numbers map to real file lines. Local object-DB lookups only
    (``rev-parse`` / ``cat-file`` / ``hash-object`` / ``show``), never a
    network call.
    """
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return proc.returncode, (proc.stdout.strip() if strip else proc.stdout)


_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / ".claude"
    / "skills"
    / "issue-v2"
    / "planned_manifest.schema.json"
)


def _word_match(needle: str, haystack: str) -> bool:
    r"""Whether ``needle`` occurs in ``haystack``, not glued to a word character.

    Callers lowercase both sides for case-insensitive matching. The point is to
    reject a planned name that is only a FRAGMENT of a longer word in the report
    (``eval`` inside ``evaluation``) while accepting a genuine occurrence.

    Anchored with ``(?<!\w)`` / ``(?!\w)`` rather than ``\b`` (#2162). ``\b`` is
    a boundary BETWEEN a word and a non-word char, so a trailing ``\b`` after a
    needle that ENDS in punctuation asserts the next char IS a word char —
    making any name ending in ``)`` structurally unmatchable: ``anchor
    separation (ceiling minus floor)`` could only ever match as
    ``...floor)x``, which no prose contains. #2162 (the first ``workflow: v2``
    task, so the first report this check ever ran against) had 7 of 21 planned
    condition / metric names ending in ``)``; all 7 reported "not found in
    report text" while the report discussed each of them at length. The
    lookarounds express the intended rule directly — the needle must not be
    ADJACENT to a word character on either side — which is identical to ``\b``
    wherever the needle starts and ends in a word char, and correct where it
    does not.
    """
    if not needle:
        return False
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack) is not None


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
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
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
# Card-walk recursion bound: a pathologically deep card JSON degrades THAT
# card (counted + skipped past the cap), never the gate. An explicit depth cap
# is deterministic and testable, unlike catching RecursionError (#2191 round-1
# review Minor 2). Real cards nest their commit keys ≤3 levels deep.
_CARD_WALK_MAX_DEPTH = 100
# b3 label→card pairing: stopwords removed from the CARD-side token set
# (path / filename-stem / phase tokens too generic to discriminate cards).
# Pinned by tests/test_verify_report.py — NOT tunable at implementation.
_CARD_TOKEN_STOPWORDS = frozenset(
    {"report", "json", "upload", "done", "card", "sentinel", "results", "gate", "gates"}
)
# Read-side copy of the write-side lifecycle denylist
# (orchestrate/provenance.py::_LIFECYCLE_PHASE_VOCAB, #2194): a card phase
# equal to a lifecycle-state word never registers a b3 exact-match key
# (behavior-neutral for compliant writers — validate_phase_identity refuses
# these at write time; this guards legacy/injected values). The skip compares
# the NORMALIZED key (`_phase_norm`), matching the exact channel's own
# case-insensitivity, so legacy `Done`/`DONE` variants are covered too
# (#2194 round 2, concern lifecycle-phase-casefold). Set equality with the
# write-side original is test-pinned (tests/test_verify_report.py).
_LIFECYCLE_PHASE_VOCAB = frozenset(
    {"done", "failed", "running", "pending", "queued", "started", "workload"}
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


def _phase_norm(text: str) -> str:
    """Exact-match normalization for card phase identity (#2194): lowercase,
    every non-alphanumeric run deleted (``stage2-upload`` == ``Stage 2
    Upload``). LOSSY: distinct raw slugs can collide (``stage1-0-upload`` vs
    ``stage-10-upload`` → ``stage10upload``) — the b3 collision guard refuses
    to pair on any collided or conflicted key."""
    return re.sub(r"[^0-9a-z]+", "", text.lower())


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
    per file; walk depth ≤ ``_CARD_WALK_MAX_DEPTH``; parse / decode failures
    and too-deep subtrees skipped + counted — a malformed card degrades that
    card, never the gate). USABLE = full 40-hex with
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
    ``_check_pin_blob_identity``); the message names the card's sibling
    ``phase`` when present. (b2) WARN, both modes: a usable SHA cited
    in the report but absent from a ``| Code SHAs |`` row. (b3) WARN, both
    modes: label→card pairing over the row's ``·``/``;`` segments —
    PREFERRED channel (#2194): exact match of the normalized segment label
    (``_phase_norm``) against a usable card's sibling ``phase`` identity,
    firing ONLY when the normalized key maps to exactly ONE raw phase
    identity across ALL sibling-phase records (usable AND excluded) and no
    excluded record supplies a conflicting commit value — collided,
    conflicted, or ambiguous (≥2 usable SHAs) keys fall through to the
    best-effort token-overlap pairing (the pre-#2194 path, byte-identical);
    unresolvable segments silently skipped. A lifecycle-valued phase
    (``_LIFECYCLE_PHASE_VOCAB``, compared on the NORMALIZED key so legacy
    ``Done``/``DONE`` case variants are covered — round 2) never registers
    an exact key. A segment with SEVERAL hex runs (a hex-bearing phase slug
    like ``run-deadbeef``) pins the UNIQUE candidate whose removal yields a
    guarded exact match, else the first run (the pre-round-2 behavior,
    byte-identical — round 2).
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
    too_deep_sources: set[str] = set()

    def _walk(obj: object, ptr: str, source: str, depth: int = 0) -> None:
        if depth > _CARD_WALK_MAX_DEPTH:
            # Degrade the CARD, never the gate: the subtree past the cap is
            # skipped (shallow keys of the same file are still collected) and
            # the file is counted in the skip channel below.
            too_deep_sources.add(source)
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in _CARD_COMMIT_KEYS and isinstance(v, str):
                    records.append(
                        (source, f"{ptr}/{k}", v, obj.get("git_dirty"), obj.get("phase"))
                    )
                else:
                    _walk(v, f"{ptr}/{k}", source, depth + 1)
        elif isinstance(obj, list):
            for idx, v in enumerate(obj):
                _walk(v, f"{ptr}/{idx}", source, depth + 1)

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
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError):
                # RecursionError: json.loads itself recurses per nesting level,
                # so a pathologically deep JSON can die BEFORE _walk's depth
                # cap ever runs — contain it in the same counted-skip channel.
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
            try:
                rc, text = _git(figures_root, "show", f"{ref}:{path}")
            except UnicodeDecodeError:
                # A non-UTF-8 card blob: _git decodes subprocess stdout as
                # text, so the decode error surfaces HERE, before any JSON
                # parse. The working-tree leg already contains this class —
                # the ref leg must too, or one bad blob crashes the whole
                # gate for every report (#2191 round-1 review Minor 1).
                n_parse_failed += 1
                continue
            if rc != 0:
                n_parse_failed += 1
                continue
            try:
                obj = json.loads(text)
            except (json.JSONDecodeError, RecursionError):
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
        if too_deep_sources:
            skips.append(f"{len(too_deep_sources)} card JSON(s) past the walk depth cap skipped")
        if skips:
            detail += "; " + "; ".join(skips)
        return CheckResult(name, True, detail)

    # b3 exact phase-match pre-pass over ALL records — usable AND excluded —
    # the collision guard's input (#2194 MF-A): dirty / abbreviated / non-hex
    # records participate, so a normalization collision or an excluded
    # record's conflicting commit value can veto the exact channel.
    phase_raw_by_key: dict[str, set[str]] = {}  # norm key -> distinct RAW phase strings
    phase_shas_by_key: dict[str, set[str]] = {}  # norm key -> EVERY sibling commit value (lower)
    for _source, _ptr, value, _dirty, phase in records:
        if isinstance(phase, str):
            key = _phase_norm(phase)
            if key:
                phase_raw_by_key.setdefault(key, set()).add(phase)
                phase_shas_by_key.setdefault(key, set()).add(value.lower())

    # Usable-card classification (dirty first, then hex shape); dedupe by SHA.
    usable: dict[str, tuple[str, str, object]] = {}  # sha -> (source, ptr, phase) first-seen
    n_usable_records = 0
    n_dirty = 0
    n_abbrev = 0
    n_nonhex = 0
    usable_tokens: dict[str, set[str]] = {}  # sha -> union of its records' card-side tokens
    phase_exact: dict[str, set[str]] = {}  # norm phase key -> USABLE SHAs carrying it (#2194)
    for source, ptr, value, dirty, phase in records:
        if dirty is True:
            n_dirty += 1
            continue
        if _SHA40_FULL_RE.fullmatch(value):
            sha = value.lower()
            n_usable_records += 1
            usable.setdefault(sha, (source, ptr, phase))
            usable_tokens.setdefault(sha, set()).update(_card_side_tokens(source, phase, issue))
            if isinstance(phase, str):
                key = _phase_norm(phase)
                if key and key not in _LIFECYCLE_PHASE_VOCAB:
                    phase_exact.setdefault(key, set()).add(sha)
        elif _HEX_ABBREV_RE.fullmatch(value):
            n_abbrev += 1
        else:
            n_nonhex += 1

    cited_tokens = {t.lower() for t in _HEX_RUN_RE.findall(raw_body)}

    def _is_cited(sha: str, tokens: set[str]) -> bool:
        return any(sha.startswith(t) for t in tokens)

    def _guarded_exact_hit(label: str) -> set[str] | None:
        """b3 exact-channel hit for a segment label, or None unless the MF-A
        collision guard passes (#2194): exactly ONE usable SHA under the
        normalized key, exactly ONE RAW phase identity across ALL
        sibling-phase records (usable AND excluded), and no excluded-record
        commit value outside the usable hit (an excluded ABBREVIATED sibling
        whose 8-hex value prefixes the usable SHA still defeats the subset
        check by design)."""
        key = _phase_norm(label)
        hit = phase_exact.get(key) if key else None
        if (
            hit is not None
            and len(hit) == 1
            and len(phase_raw_by_key.get(key, ())) == 1
            and phase_shas_by_key.get(key, set()) <= hit
        ):
            return hit
        return None

    fails: list[str] = []
    warns: list[str] = []

    # (b1) card-coverage, whole-report scope — FAIL at generation, WARN at promote.
    for sha in sorted(usable):
        if _is_cited(sha, cited_tokens):
            continue
        source, ptr, phase = usable[sha]
        phase_note = f" (phase `{phase}`)" if isinstance(phase, str) and phase else ""
        msg = (
            f"reproducibility card `{source}` (`{ptr}`){phase_note} records commit "
            f"{sha[:12]}… which the "
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
    n_phase_exact = 0
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
                # Hex-bearing phase-slug disambiguation (#2194 round 2): a
                # VALID phase slug can itself contain an 8-40 hex run
                # (`run-deadbeef`), and consuming the FIRST hex run as the
                # pin would strand such slugs off the exact channel forever.
                # With >1 candidates, pick the UNIQUE one whose removal
                # yields a guarded exact phase match; zero or several ⇒ keep
                # the first-run behavior byte-identically (single-candidate
                # segments are untouched by construction).
                candidates = list(_HEX_RUN_RE.finditer(segment))
                if len(candidates) > 1:
                    guarded = [
                        m
                        for m in candidates
                        if _guarded_exact_hit(segment[: m.start()] + segment[m.end() :]) is not None
                    ]
                    if len(guarded) == 1:
                        hexm = guarded[0]
                label = segment[: hexm.start()] + segment[hexm.end() :]
                # PREFERRED exact phase-match channel (#2194), behind the
                # MF-A collision guard (_guarded_exact_hit above) — every
                # collided/conflicted/ambiguous key falls through to the
                # token-overlap path (a degrade, never a mis-pair).
                exact_key = _phase_norm(label)
                exact_hit = _guarded_exact_hit(label)
                via_exact = exact_hit is not None
                if exact_hit is not None:
                    (sha,) = exact_hit
                    n_phase_exact += 1
                else:
                    seg_tokens = _label_tokens(label)
                    hit_shas = {sha for sha, toks in usable_tokens.items() if toks & seg_tokens}
                    if len(hit_shas) != 1:
                        n_unresolved_segments += 1
                        continue
                    (sha,) = hit_shas
                pin_tok = hexm.group(0).lower()
                if not sha.startswith(pin_tok):
                    if via_exact:
                        (card_phase,) = phase_raw_by_key[exact_key]
                        warns.append(
                            f"Code-SHAs row segment '{segment.strip()[:60]}' pins "
                            f"{pin_tok[:12]}… but its label exact-matches card phase "
                            f"`{card_phase}` recorded at commit {sha[:12]}… — carry the "
                            "per-phase split (each phase @ its own card's commit)"
                        )
                    else:
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
    if too_deep_sources:
        excl.append(
            f"{len(too_deep_sources)} card JSON(s) past the walk depth cap skipped (partial walk)"
        )
    if n_phase_exact:
        excl.append(f"{n_phase_exact} row segment(s) resolved via exact phase match")
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


# ─── Companion-content check (#2198; both modes, read-only local git) ───────

# Thomas-slot headings forbidden in the 100%-agent-written companion, in
# NORMALIZED form — ``_norm_header()`` maps the grandfathered ``## Next steps``
# alias onto ``## Conclusion and next steps``, so the alias is caught too.
_COMPANION_FORBIDDEN_HEADERS = ("## TLDR", "## Conclusion and next steps")


def _resolve_companion_text(
    blanked_lines: list[str], *, mode: str, figures_root: Path
) -> tuple[str | None, CheckResult | None]:
    """Resolve the body's ``**Detailed writeup:**`` pin to the companion text.

    Materializes ``docs/reports/issue_<N>_detailed.md`` at the pinned SHA via
    read-only local git (``git show`` through ``_git`` — never a network
    fetch). Returns ``(companion_text, None)`` on success, else
    ``(None, CheckResult)`` carrying the resolution-ladder verdict: a missing /
    stacked / malformed pin line is PASS-note N/A (``detailed-writeup-link``
    already carries that verdict — no double-report); a non-git root WARNs; an
    unresolvable pinned commit is mode-split like ``_check_pin_blob_identity``
    (FAIL at generation — the orchestrator commits + pushes the companion and
    splices the pin BEFORE the generation-mode verify runs, so the commit
    exists locally by construction; WARN at promote — unfetched clone
    plausible); a resolvable commit lacking the companion path FAILs (both);
    a resolvable-but-non-UTF-8 companion blob FAILs (both — contained
    ``UnicodeDecodeError``, never a crash).
    """
    name = "companion-content"
    matches = [m for ln in blanked_lines if (m := _DETAILED_LINE_RE.match(ln)) is not None]
    if len(matches) != 1:
        return None, CheckResult(
            name,
            True,
            "no single '**Detailed writeup:**' line — companion scan N/A "
            "(detailed-writeup-link carries the verdict)",
        )
    url = matches[0].group(1).strip().strip("<>")
    m = _DETAILED_URL_RE.match(url)
    if m is None:
        return None, CheckResult(
            name,
            True,
            "malformed detailed-writeup URL — companion scan N/A "
            "(detailed-writeup-link carries the verdict)",
        )
    sha, path = m.group(1), f"docs/reports/issue_{m.group(2)}_detailed.md"
    rc, _ = _git(figures_root, "rev-parse", "--git-dir")
    if rc != 0:
        return None, CheckResult(
            name,
            True,
            f"{figures_root} is not a git checkout; companion content unverifiable",
            is_warn=True,
        )
    rc, _ = _git(figures_root, "cat-file", "-e", f"{sha}^{{commit}}")
    if rc != 0:
        msg = f"pinned commit {sha[:12]} unresolvable in the local object DB ({path})"
        if mode == "generation":
            return None, CheckResult(name, False, msg)
        return None, CheckResult(
            name,
            True,
            msg + "; unfetched clone possible post-merge, companion unverifiable",
            is_warn=True,
        )
    try:
        # strip=False: byte-faithful materialization — a companion whose file
        # starts with blank lines must keep them, or every reported ``L<n>``
        # is offset by the stripped count (round-1 review Minor).
        rc, text = _git(figures_root, "show", f"{sha}:{path}", strip=False)
    except UnicodeDecodeError:
        # A non-UTF-8 companion blob at a resolvable pin: _git decodes
        # subprocess stdout as text, so the decode error surfaces HERE —
        # contain it (mirror the card-blob ``git show`` containment, #2191
        # convention). FAIL in BOTH modes: the companion is agent-generated
        # markdown, so non-UTF-8 bytes are a generation defect, never an
        # unfetched-clone gap.
        return None, CheckResult(
            name,
            False,
            f"companion {path} at pinned commit {sha[:12]} is not valid UTF-8 — "
            "content unscannable",
        )
    if rc != 0:
        return None, CheckResult(name, False, f"pinned commit {sha[:12]} does not contain {path}")
    return text, None


def check_companion_content(
    blanked_lines: list[str], *, mode: str, figures_root: Path
) -> CheckResult:
    """``companion-content`` (#2198): scan the detailed companion writeup.

    The companion (``docs/reports/issue_<N>_detailed.md`` at the body's
    ``**Detailed writeup:**`` pin) is 100% agent-written in BOTH modes — it is
    regenerated wholesale on follow-up rounds, so anything hand-written there
    is destroyed without notice — so the SCAN scope is mode-invariant; only
    the RESOLUTION ladder is mode-split (``_resolve_companion_text``). Both halves
    run on the companion's ``blank_verbatim()``-blanked lines (a ``## TLDR`` /
    lexeme inside a fenced example or blockquote is DATA, not a slot):

    - STRUCTURAL half (FAIL, both modes): no Thomas-slot heading — a line
      whose ``_norm_header()`` equals ``## TLDR`` or ``## Conclusion and next
      steps`` (the alias map catches the grandfathered ``## Next steps``) —
      and no ``**Takeaways**`` block opener (``_is_bold_label``).
    - LEXICON half (WARN, never FAIL): ``_LEXICON_RE`` over every blanked
      line OUTSIDE the companion's ``## Motivation`` section (the Motivation
      copy keeps the body's hypothesis-framing exemption). WARN because the
      companion's methodology/deviations prose legitimately uses process-sense
      phrasings that match the exact lexemes ("the artifact confirms the
      count") — a FAIL posture would get routed around.

    Two deliberate scope limits (plan #2198):

    - Exact-header-only Motivation exemption: realized companions carry
      appended follow-up sections (``## Motivation (round)`` — the live #2162
      shape); ``_norm_header()`` does not map those onto ``## Motivation``, so
      round-Motivation copies ARE lexicon-scanned. Acceptable (WARN-only
      posture — no FAIL channel) and deliberate.
    - Exact-lexeme inflection gap: ``BANNED_LEXICON`` is an exact-lexeme
      list — inflectional variants ("confirmed", "implies", "suggested") do
      NOT match ``_LEXICON_RE``. Extending the lexicon would change the
      existing BODY scan too (a shared-instrument scope change, out of #2198's
      scope); the report-verifier agent's manual interpretivity read remains
      the catching arm for inflected phrasings and blockquoted caption prose.

    Aggregation mirrors ``_check_pin_blob_identity``: any fail → FAIL, else
    any warn → WARN, else PASS. Detail entries are prefixed by half
    (``thomas-slot heading L<n>: ...`` / ``lexicon L<n>: ...``).
    """
    name = "companion-content"
    companion, ladder = _resolve_companion_text(blanked_lines, mode=mode, figures_root=figures_root)
    if ladder is not None:
        return ladder
    comp_lines = blank_verbatim(companion.splitlines())
    fails: list[str] = []
    for lineno, line in enumerate(comp_lines, start=1):
        if _norm_header(line) in _COMPANION_FORBIDDEN_HEADERS:
            fails.append(f"thomas-slot heading L{lineno}: '{line.strip()}'")
        elif _is_bold_label(line, "Takeaways"):
            fails.append(f"thomas-slot heading L{lineno}: '{line.strip()}'")
    exempt: set[int] = set()
    for sec in parse_sections(comp_lines):
        if _norm_header(sec.header) == "## Motivation":
            exempt.update(range(sec.header_line, sec.content_start_line + len(sec.content_lines)))
    warns: list[str] = []
    for lineno, line in enumerate(comp_lines, start=1):
        if lineno in exempt:
            continue
        for m in _LEXICON_RE.finditer(line):
            warns.append(f"lexicon L{lineno}: '{m.group(0)}'")
    if fails:
        detail = "; ".join(fails)
        if warns:
            detail += "; warn: " + "; ".join(warns)
        return CheckResult(name, False, detail)
    if warns:
        return CheckResult(
            name,
            True,
            "banned lexeme(s) outside the companion's ## Motivation: " + "; ".join(warns),
            is_warn=True,
        )
    return CheckResult(name, True, "companion clean (no Thomas-slot headings, no lexicon hits)")


# ─── Stale evidence pins (#2195; both modes, WARN-only) ─────────────────────

# Repo-relative path prefixes treated as EVIDENCE citations. Code provenance
# (`scripts/` / `src/` / `configs/`) is deliberately OUTSIDE the set — "ran at
# `<sha>`" is provenance a later commit does not invalidate; `code-sha-cards`
# owns that surface. `tasks/` is deliberately EXCLUDED too: task-folder status
# transitions rename `tasks/<status>/…` on every lifecycle move, and the
# rename's delete side satisfies a non-empty `git log <pin>..<tip> -- <old
# path>`, so every parked-task citation would read permanently stale
# (predictable healthy-report noise). Pinned verbatim by
# tests/test_verify_report.py — NOT tunable at implementation.
_EVIDENCE_PATH_PREFIXES = ("eval_results/", "ood_eval_results/", "figures/", "docs/")
# Per-report cap on (member, pin) staleness reads (each costs <=3 local git
# calls); beyond the cap, remaining candidates are skipped with a COUNTED
# note — a capped read is disclosed, never silent (WARN-only check).
_STALE_READ_CAP = 200
# Newer commits enumerated per stale citation before "+K more".
_STALE_DETAIL_COMMITS_CAP = 5


def _evidence_path_candidates(line: str) -> list[tuple[tuple[int, int], list[str]]]:
    """Backticked evidence-path citation candidates on ``line``.

    One ``((span_start, span_end), members)`` entry per backticked token
    (span covers the backticks) that contains a ``/``, is not a URL /
    absolute / ellipsis-abbreviated path, expanded through ONE brace group
    (``_expand_brace_group``), members filtered to the
    ``_EVIDENCE_PATH_PREFIXES`` set; a token with zero surviving members is
    not a candidate.
    """
    out: list[tuple[tuple[int, int], list[str]]] = []
    for m in re.finditer(r"`([^`]+)`", line):
        tok = m.group(1).strip()
        if "/" not in tok or "://" in tok or tok.startswith("/"):
            continue
        if "…" in tok or "..." in tok:
            continue
        members = [p for p in _expand_brace_group(tok) if p.startswith(_EVIDENCE_PATH_PREFIXES)]
        if members:
            out.append(((m.start(), m.end()), members))
    return out


def _same_line_pins_positional(
    line: str, figures_root: Path, blank_spans: tuple[tuple[int, int], ...] = ()
) -> list[tuple[int, str]]:
    """Thin POSITIONAL variant of ``_same_line_pins`` (#2195) — the hex-run
    resolution is reused verbatim (URL + committed-under spans blanked, hex
    runs via ``rev-parse --verify <tok>^{commit}``), with TWO deliberate
    divergences. (1) Branch tokens resolve ORIGIN-FIRST, mirroring
    ``_authoritative_tip``: an explicit ``origin/<b>`` token resolves ONLY
    ``refs/remotes/origin/<b>``; a plain ``<b>`` token tries
    ``refs/remotes/origin/<b>`` first and falls back to ``refs/heads/<b>``
    only when the remote ref is absent (local-first would associate a token
    pin to a lagging worktree-local ``issue-<N>`` and false-WARN a report
    pinned at the origin tip — the round-1 blocker). (2) It additionally
    blanks the caller-supplied ``blank_spans`` (the candidate evidence-path
    backtick spans — span hygiene: a sha-like cited filename must never
    contribute its own hex run as a pin) and returns ``(span_start, sha)``
    pairs sorted by line position for per-member association. Duplicate SHAs
    at distinct positions are kept (positions drive association); existing
    ``_same_line_pins`` callers are untouched.
    """
    scrubbed = _URL_RE.sub(lambda m: " " * len(m.group(0)), line)
    scrubbed = _COMMITTED_UNDER_RE.sub(lambda m: " " * len(m.group(0)), scrubbed)
    if blank_spans:
        chars = list(scrubbed)
        for start, end in blank_spans:
            for i in range(start, min(end, len(chars))):
                chars[i] = " "
        scrubbed = "".join(chars)
    pins: list[tuple[int, str]] = []
    for m in _HEX_RUN_RE.finditer(scrubbed):
        rc, sha = _git(figures_root, "rev-parse", "--verify", f"{m.group(0)}^{{commit}}")
        if rc == 0 and sha:
            pins.append((m.start(), sha))
    for m in re.finditer(r"`([^`]+)`", scrubbed):
        btok = m.group(1).strip()
        if not _BRANCH_TOKEN_RE.match(btok):
            continue
        # Origin-authoritative token resolution (#2195 round 2): an explicit
        # ``origin/<b>`` token names ONLY the remote ref; a plain ``<b>``
        # token prefers ``refs/remotes/origin/<b>`` and falls back to the
        # local ref only when origin is absent — mirroring
        # ``_authoritative_tip``, so a lagging worktree-local ``issue-<N>``
        # can never associate a token pin to a stale tip and false-WARN.
        if btok.startswith("origin/"):
            refs = (f"refs/remotes/{btok}",)
        else:
            refs = (f"refs/remotes/origin/{btok}", f"refs/heads/{btok}")
        for ref in refs:
            rc, sha = _git(figures_root, "rev-parse", "--verify", f"{ref}^{{commit}}")
            if rc == 0 and sha:
                pins.append((m.start(), sha))
                break
    pins.sort(key=lambda t: t[0])
    return pins


def _authoritative_tip(issue: int, figures_root: Path) -> str | None:
    """The issue branch's AUTHORITATIVE tip ref name, or None.

    ``origin/issue-<N>`` when it resolves; local ``issue-<N>`` ONLY as
    fallback when origin is absent (the task body's literal ask is
    ``git log <pin>..origin/<branch>``; a lagging local ref must never veto
    origin staleness).
    """
    for ref, display in (
        (f"refs/remotes/origin/issue-{issue}", f"origin/issue-{issue}"),
        (f"refs/heads/issue-{issue}", f"issue-{issue}"),
    ):
        rc, sha = _git(figures_root, "rev-parse", "--verify", f"{ref}^{{commit}}")
        if rc == 0 and sha:
            return display
    return None


def check_stale_evidence_pins(
    blanked_lines: list[str], figures_root: Path, expect_issue: int | None
) -> CheckResult:
    """``stale-evidence-pins`` (#2195): WARN when a report cites an in-repo
    evidence FILE at a pin the same issue branch has since modified.

    For every (backticked evidence path, associated same-line pin) where the
    path resolves at the pin as exactly one blob equal to the cited path, run
    ``git log <pin>..<tip> -- <path>`` against the issue branch's
    authoritative tip (origin-preferred). Non-empty means the branch rewrote
    the artifact AFTER the pin — the report cites a superseded version of its
    own evidence — so emit ONE aggregated WARN enumerating the newer commits.
    NEVER a FAIL: an as-of pin is legitimate, and the check cannot know
    whether the newer version contradicts the citing sentence; the WARN hands
    the reviewer the discriminating fact. Conservative-matcher skip set +
    named residues: module docstring (``stale-evidence-pins`` entry).
    """
    name = "stale-evidence-pins"
    candidate_rows: list[tuple[int, str, list[tuple[tuple[int, int], list[str]]]]] = []
    for i, ln in enumerate(blanked_lines, start=1):
        if _CODE_SHA_ROW_RE.match(ln):
            continue  # run-provenance pins, not evidence citations
        cands = _evidence_path_candidates(ln)
        if cands:
            candidate_rows.append((i, ln, cands))
    if not candidate_rows:
        return CheckResult(name, True, "no evidence citations at a pin (N/A)")
    rc, _ = _git(figures_root, "rev-parse", "--git-dir")
    if rc != 0:
        return CheckResult(
            name,
            True,
            f"{figures_root} is not a git checkout; stale-evidence pins unverifiable",
            is_warn=True,
        )
    issue = expect_issue if expect_issue is not None else _infer_issue_from_lines(blanked_lines)
    tip = _authoritative_tip(issue, figures_root) if issue is not None else None
    if tip is None:
        # Post-merge branch-deleted / unknown-issue shape: must not WARN on
        # every promote of an old report.
        return CheckResult(
            name, True, "stale-evidence check skipped — no issue branch ref resolvable"
        )
    warns: list[str] = []
    notes: list[str] = []
    seen: set[tuple[str, str]] = set()
    n_reads = 0
    n_capped = 0
    n_fresh = 0
    for line_no, ln, cands in candidate_rows:
        spans = tuple(span for span, _members in cands)
        pins = _same_line_pins_positional(ln, figures_root, spans)
        if not pins:
            continue  # not a citation-at-a-pin (no-pin claims are committed-under-claims' surface)
        for (span_start, span_end), members in cands:
            after = [sha for pos, sha in pins if pos >= span_end]
            if after:
                assoc = after[0]
            else:
                before = [sha for pos, sha in pins if pos < span_start]
                assoc = before[-1] if before else None
            if assoc is None:
                continue
            for member in members:
                key = (member, assoc)
                if key in seen:
                    continue  # deduped by (path, pin)
                seen.add(key)
                if n_reads >= _STALE_READ_CAP:
                    n_capped += 1
                    continue
                n_reads += 1
                rc, out = _git(figures_root, "ls-tree", "-r", "--name-only", assoc, "--", member)
                names = out.splitlines() if rc == 0 and out else []
                if not names:
                    notes.append(
                        f"line {line_no}: `{member}` absent at {assoc[:12]} — a broken "
                        "citation is not a stale one; skipped"
                    )
                    continue
                if len(names) != 1 or names[0] != member:
                    notes.append(
                        f"line {line_no}: `{member}` is a directory citation at "
                        f"{assoc[:12]} — skipped (later additions under a directory do "
                        "not supersede a home claim)"
                    )
                    continue
                rc, _out = _git(figures_root, "merge-base", "--is-ancestor", assoc, tip)
                if rc != 0:
                    notes.append(
                        f"line {line_no}: pin {assoc[:12]} is not an ancestor of {tip} — "
                        "divergent history, staleness undecidable; skipped"
                    )
                    continue
                rc, out = _git(
                    figures_root, "log", "--format=%H%x09%s", f"{assoc}..{tip}", "--", member
                )
                if rc != 0:
                    notes.append(
                        f"line {line_no}: git log failed for `{member}` at {assoc[:12]}; skipped"
                    )
                    continue
                commits = [c for c in out.splitlines() if c.strip()]
                if not commits:
                    n_fresh += 1
                    continue
                shown = []
                for c in commits[:_STALE_DETAIL_COMMITS_CAP]:
                    sha, _, subject = c.partition("\t")
                    shown.append(f'{sha[:12]} "{subject[:60]}"')
                extra = len(commits) - _STALE_DETAIL_COMMITS_CAP
                enumerated = ", ".join(shown) + (f", +{extra} more" if extra > 0 else "")
                warns.append(
                    f"line {line_no}: `{member}` cited at {assoc[:12]} has {len(commits)} "
                    f"newer commit(s) on {tip} — {enumerated} — if the citation is a "
                    "deliberate as-of pin, confirm the newer version still supports the "
                    "citing sentence; else re-pin to the current commit"
                )
    if n_capped:
        notes.append(f"read cap ({_STALE_READ_CAP}) reached — {n_capped} candidate(s) not checked")
    if warns:
        detail = "; ".join(warns)
        if notes:
            detail += "; " + "; ".join(notes)
        return CheckResult(name, True, detail, is_warn=True)
    if n_reads == 0 and not notes:
        return CheckResult(name, True, "no evidence citations at a pin (N/A)")
    parts: list[str] = []
    if n_fresh:
        parts.append(f"{n_fresh} pinned evidence citation(s) fresh on {tip}")
    parts.extend(notes)
    return CheckResult(name, True, "; ".join(parts))


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
    results.append(check_companion_content(blanked_lines, mode=mode, figures_root=figures_root))
    results.append(check_stale_evidence_pins(blanked_lines, figures_root, expect_issue))

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
