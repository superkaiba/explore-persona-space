#!/usr/bin/env python3
"""verify_report.py — mechanical verifier for v2 report clean-result bodies.

The v2 workflow retires agent interpretation of results: agents author a
fixed-structure REPORT (Motivation / Methodology / Metrics / Results-as-plots)
and Thomas alone writes the TLDR + Next steps. This is the mechanical gate for
that report form, the report-track analogue of ``verify_task_body.py`` (markdown
v4) and ``verify_paper.py`` (paper track).

A report body carries the sentinel ``<!-- report-v1 -->`` on the line after its
H1 ``# Experiment: <question>`` title (mirroring ``<!-- clean-result-v4 -->``).

Required structure (both modes):
  - H1 line ``# Experiment: <question>``.
  - Sentinel ``<!-- report-v1 -->`` as the first non-blank line after the H1.
  - Six H2 sections, in this exact relative order: ``## TLDR:``,
    ``## Motivation:``, ``## Methodology:``, ``## Metrics:``, ``## Results:``,
    ``## Next steps:``.
  - ``## Results:`` contains >=1 ``### <name>`` subsection, each with a
    non-empty description paragraph AND exactly one image reference
    ``![...](...)``.
  - Every referenced local image path exists on disk (resolved vs
    ``--figures-root``; default: the git-repo root of ``--file``).
  - Every ``htmlpreview.github.io`` link embeds a full 40-hex SHA
    ``raw.githubusercontent`` URL (well-formedness only, no network).
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

Mode-specific:
  - ``generation``: TLDR AND Next steps content MUST be exactly the placeholder
    ``*(Thomas fills in)*`` (Thomas has not written them yet). Interpretive
    lexicon scan over Methodology + Metrics + Results (Motivation is exempt —
    hypothesis framing is allowed there).
  - ``promote``: TLDR content MUST be non-placeholder AND non-empty (Thomas has
    filled it). Thomas's TLDR / Next-steps prose is NEVER lexicon-checked; the
    agent-authored sections are still lexicon-scanned; structural checks still
    apply.

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
PLACEHOLDER = "*(Thomas fills in)*"

# The six required H2 sections, in the exact order they must appear.
REQUIRED_SECTIONS = [
    "## TLDR:",
    "## Motivation:",
    "## Methodology:",
    "## Metrics:",
    "## Results:",
    "## Next steps:",
]

# Sections whose (agent-authored) prose is scanned for interpretive lexicon.
# Motivation is deliberately EXEMPT (hypothesis-to-be-tested framing is allowed
# there); TLDR / Next steps are Thomas's prose and are NEVER scanned.
LEXICON_SECTIONS = ("## Methodology:", "## Metrics:", "## Results:")

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
    """First occurrence of each required header text → its Section."""
    out: dict[str, Section] = {}
    for sec in sections:
        if sec.header in REQUIRED_SECTIONS and sec.header not in out:
            out[sec.header] = sec
    return out


# ─── Structural checks (both modes) ────────────────────────────────────────


def check_h1_and_sentinel(lines: list[str]) -> list[CheckResult]:
    results: list[CheckResult] = []
    h1 = find_h1(lines)
    if h1 is None:
        results.append(CheckResult("h1-title", False, "no H1 title found"))
        results.append(CheckResult("sentinel", False, "no H1 title to anchor the sentinel"))
        return results
    h1_idx, title = h1
    if title.startswith(H1_TITLE_PREFIX):
        results.append(
            CheckResult("h1-title", True, f"H1 = 'Experiment: {title[len(H1_TITLE_PREFIX) :]}'")
        )
    else:
        results.append(
            CheckResult(
                "h1-title", False, f"H1 must start with '# {H1_TITLE_PREFIX}...', got '# {title}'"
            )
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
        results.append(CheckResult("required-sections", True, "all six required sections present"))
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
    """FAIL if any of the six required ``## `` headings appears more than once.

    Scanned on the fence/blockquote-blanked body, so a required heading string
    inside a verbatim example does not count. ``section_map`` silently keeps the
    FIRST occurrence, so a stray duplicate would otherwise slip past the
    structural checks entirely.
    """
    occurrences: dict[str, list[int]] = {}
    for i, ln in enumerate(lines, 1):
        header = ln.rstrip()
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


def check_results_subsections(sections: list[Section]) -> CheckResult:
    present = section_map(sections)
    results_sec = present.get("## Results:")
    if results_sec is None:
        return CheckResult("results-subsections", False, "no ## Results: section")
    lines = results_sec.content_lines
    sub_idxs = [i for i, ln in enumerate(lines) if ln.startswith("### ")]
    if not sub_idxs:
        return CheckResult("results-subsections", False, "## Results: has no ### <name> subsection")
    problems: list[str] = []
    for pos, i in enumerate(sub_idxs):
        end = sub_idxs[pos + 1] if pos + 1 < len(sub_idxs) else len(lines)
        name = lines[i].strip()[4:].strip()
        block = lines[i + 1 : end]
        block_text = "\n".join(block)
        imgs = _images_in(block_text)
        if len(imgs) != 1:
            problems.append(f"'{name}': expected exactly 1 image, found {len(imgs)}")
        # Description = a non-blank line that is not solely an image reference.
        has_desc = any(ln.strip() and not _IMAGE_RE.fullmatch(ln.strip()) for ln in block)
        if not has_desc:
            problems.append(f"'{name}': missing a non-empty description paragraph")
    if problems:
        return CheckResult("results-subsections", False, "; ".join(problems))
    return CheckResult(
        "results-subsections",
        True,
        f"{len(sub_idxs)} subsection(s), each with 1 image + description",
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


def check_htmlpreview(body: str) -> CheckResult:
    urls = [u for u in _URL_RE.findall(body) if "htmlpreview.github.io" in u]
    if not urls:
        return CheckResult("htmlpreview-sha", True, "no htmlpreview links (N/A)", is_warn=False)
    bad: list[str] = []
    for u in urls:
        if "raw.githubusercontent.com" not in u or not _SHA40_RE.search(u):
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
    results_sec = section_map(sections).get("## Results:")
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


def check_lexicon(sections: list[Section]) -> CheckResult:
    present = section_map(sections)
    hits: list[str] = []
    for header in LEXICON_SECTIONS:
        sec = present.get(header)
        if sec is None:
            continue
        for offset, line in enumerate(sec.content_lines):
            for m in _LEXICON_RE.finditer(line):
                lineno = sec.content_start_line + offset
                hits.append(f"{header[3:].rstrip(':')} L{lineno}: '{m.group(0)}'")
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
    """generation mode: TLDR and Next steps must be the untouched placeholder."""
    results: list[CheckResult] = []
    for header, name in (
        ("## TLDR:", "tldr-placeholder"),
        ("## Next steps:", "nextsteps-placeholder"),
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
    text = _section_text(sections, "## TLDR:")
    if text is None:
        return CheckResult("tldr-filled", False, "## TLDR: section missing")
    if not text:
        return CheckResult("tldr-filled", False, "## TLDR: is empty; Thomas must write the TLDR")
    if text == PLACEHOLDER:
        return CheckResult(
            "tldr-filled", False, "## TLDR: still the placeholder; Thomas must write the TLDR"
        )
    return CheckResult("tldr-filled", True, "## TLDR: is filled")


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
    results.extend(check_h1_and_sentinel(lines))
    results.extend(check_required_sections(sections))
    results.append(check_duplicate_sections(blanked_lines))
    results.append(check_results_subsections(sections))
    results.append(check_image_files(blanked_body, figures_root))
    results.append(check_htmlpreview(body))
    results.extend(
        check_image_pins(
            sections, blanked_body, mode=mode, figures_root=figures_root, expect_issue=expect_issue
        )
    )
    results.append(check_lexicon(sections))

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
