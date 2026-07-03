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
planned condition/metric appears somewhere in the report text and every planned
figure id has a matching ``### `` subsection OR is explicitly marked ``not run``.

Exit 0 PASS / 1 FAIL / 2 usage error. Prints one line per check.
"""

from __future__ import annotations

import argparse
import json
import re
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

_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / ".claude"
    / "skills"
    / "issue-v2"
    / "planned_manifest.schema.json"
)


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


def check_manifest(body: str, sections: list[Section], manifest_path: Path) -> list[CheckResult]:
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

    low_body = body.lower()

    def _coverage(name: str, items: list[str]) -> CheckResult:
        missing = [it for it in items if it.lower() not in low_body]
        if missing:
            return CheckResult(name, False, "not found in report text: " + ", ".join(missing))
        return CheckResult(name, True, f"all {len(items)} present in report text")

    results.append(_coverage("manifest-conditions", list(manifest.get("conditions", []))))
    results.append(_coverage("manifest-metrics", list(manifest.get("metrics", []))))

    # Figure coverage: each planned figure id needs a matching ### subsection
    # (id OR title appears in a ### heading) OR is explicitly marked "not run".
    sub_headings = [
        ln.strip()[4:].strip()
        for sec in sections
        for ln in sec.content_lines
        if ln.startswith("### ")
    ]
    heading_blob = "\n".join(sub_headings).lower()
    body_lines_low = [ln.lower() for ln in body.splitlines()]
    fig_missing: list[str] = []
    for fig in manifest.get("figures", []):
        fid = str(fig.get("id", "")).strip()
        title = str(fig.get("title", "")).strip()
        keys = [k.lower() for k in (fid, title) if k]
        in_heading = any(k in heading_blob for k in keys)
        marked_not_run = any(
            "not run" in ln and any(k in ln for k in keys) for ln in body_lines_low
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
) -> tuple[bool, list[CheckResult]]:
    """Run all checks for ``mode``; return (overall_pass, results)."""
    if mode not in ("generation", "promote"):
        raise ValueError(f"mode must be generation|promote, got {mode!r}")
    _, body = split_frontmatter(raw)
    lines = body.splitlines()
    sections = parse_sections(lines)

    results: list[CheckResult] = []
    results.extend(check_h1_and_sentinel(lines))
    results.extend(check_required_sections(sections))
    results.append(check_results_subsections(sections))
    results.append(check_image_files(body, figures_root))
    results.append(check_htmlpreview(body))
    results.append(check_lexicon(sections))

    if mode == "generation":
        results.extend(check_placeholders(sections))
    else:  # promote
        results.append(check_tldr_filled(sections))

    if manifest_path is not None:
        results.extend(check_manifest(body, sections, manifest_path))

    overall = all(r.passed for r in results)
    return overall, results


def _default_figures_root(file_path: Path) -> Path:
    """The git-repo root of ``file_path`` (dir containing ``.git``), else its parent."""
    for d in [file_path.resolve().parent, *file_path.resolve().parents]:
        if (d / ".git").exists():
            return d
    return file_path.resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--file", required=True, help="path to a report body.md to verify")
    parser.add_argument(
        "--mode", required=True, choices=["generation", "promote"], help="verification mode"
    )
    parser.add_argument(
        "--manifest", help="path to planned_manifest.json (optional coverage check)"
    )
    parser.add_argument(
        "--figures-root",
        help="root for resolving local image paths (default: the git-repo root of --file)",
    )
    args = parser.parse_args()

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

    overall, results = verify_report_text(
        raw, mode=args.mode, figures_root=figures_root, manifest_path=manifest_path
    )
    print(f"verify_report — {args.file} (mode={args.mode})")
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
