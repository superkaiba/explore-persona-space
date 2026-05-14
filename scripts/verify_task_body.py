#!/usr/bin/env python3
"""verify_task_body.py — mechanical verifier for markdown clean-result bodies.

Replaces `verify_sagan_card.py` for new bodies. Six checks against the
markdown clean-result spec in `.claude/plans/task-workflow-migration.md`
§ 10:

  1. Title line ends with `(LOW | MODERATE | HIGH confidence)`.
  2. Four required H2 sections present in order: TL;DR, Figure, Details,
     Reproducibility.
  3. TL;DR bullets contain the four labels: Motivation, What I ran,
     Results, Next steps.
  4. Reproducibility URLs are permanent (HF Hub `/tree/<ref>`, WandB
     `/runs/<id>`, GitHub `/blob/<sha>`); no `TBD`, `{{`, `default`, `see
     config`, or empty placeholders. `n/a` is accepted as an explicit
     non-applicable marker.
  5. Confidence sentence in Details matches the title's confidence level.
  6. Figure caption ≥10 words.

Bodies carrying a `<!-- legacy-sagan-card -->` sentinel are
grandfathered HTML — this verifier skips them with a WARN (the legacy
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

REQUIRED_H2_SECTIONS = ["TL;DR", "Figure", "Details", "Reproducibility"]
TLDR_BULLETS = ["Motivation", "What I ran", "Results", "Next steps"]

LEGACY_SAGAN_CARD_SENTINEL = "<!-- legacy-sagan-card -->"

CONFIDENCE_LEVELS = {"LOW", "MODERATE", "HIGH"}

# Sentinel substrings that indicate a placeholder slipped through.
SENTINEL_SUBSTRINGS = ["TBD", "{{", "see config", "default"]


# ─── Result type ───────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""

    def render(self) -> str:
        tag = "PASS" if self.passed else "FAIL"
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
    """Return list of (section_name, body_line_start, body_line_end) for each H2."""
    lines = body.splitlines()
    h2_indices: list[tuple[str, int]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
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


# ─── Checks ────────────────────────────────────────────────────────────────


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
    if missing:
        return CheckResult(
            "four required H2 sections in order",
            False,
            f"missing: {', '.join(missing)} (found: {found})",
        )
    # Order check: REQUIRED_H2_SECTIONS must appear in this order within `found`.
    seq = [s for s in found if s in REQUIRED_H2_SECTIONS]
    if seq != REQUIRED_H2_SECTIONS:
        return CheckResult(
            "four required H2 sections in order",
            False,
            f"wrong order — got {seq}, expected {REQUIRED_H2_SECTIONS}",
        )
    return CheckResult("four required H2 sections in order", True)


def check_tldr_labels(body: str) -> CheckResult:
    tldr = section_text(body, "TL;DR")
    if tldr is None:
        return CheckResult(
            "TL;DR bullets carry the four required labels", False, "TL;DR section missing"
        )
    missing = []
    for label in TLDR_BULLETS:
        # accept either `**Motivation:**` or `Motivation:` at start of line / after `-`.
        if not re.search(rf"(?im)^\s*[-*]\s*(\*\*)?{re.escape(label)}(\*\*)?\s*:", tldr):
            missing.append(label)
    if missing:
        return CheckResult(
            "TL;DR bullets carry the four required labels",
            False,
            f"missing labels: {', '.join(missing)}",
        )
    return CheckResult("TL;DR bullets carry the four required labels", True)


def check_reproducibility_urls(body: str) -> CheckResult:
    repro = section_text(body, "Reproducibility")
    if repro is None:
        return CheckResult(
            "Reproducibility URLs are permanent", False, "Reproducibility section missing"
        )
    bad: list[str] = []
    for s in SENTINEL_SUBSTRINGS:
        # Case-sensitive for `{{`, case-insensitive for prose sentinels.
        if s == "{{":
            if "{{" in repro:
                bad.append("`{{` placeholder")
        else:
            if re.search(rf"\b{re.escape(s)}\b", repro, flags=re.IGNORECASE):
                bad.append(f"`{s}`")
    # HF Hub URLs must include /tree/<ref>
    hf_urls = re.findall(r"https?://huggingface\.co/[^\s\)<>]+", repro)
    for url in hf_urls:
        if "/tree/" not in url and "/blob/" not in url and "/raw/" not in url:
            # bare repo URL is acceptable when followed by a `n/a` marker; otherwise flag
            bad.append(f"unpinned HF URL `{url}` (needs `/tree/<ref>`)")
    # WandB run URLs should be `/runs/<id>`
    wandb_urls = re.findall(r"https?://(?:www\.)?wandb\.ai/[^\s\)<>]+", repro)
    for url in wandb_urls:
        if "/runs/" not in url and "/groups/" not in url and "/reports/" not in url:
            bad.append(f"unpinned WandB URL `{url}` (needs `/runs/<id>`)")
    # GitHub URLs should be `/blob/<sha>` or `/tree/<sha>`, not `/blob/main`.
    gh_urls = re.findall(r"https?://github\.com/[^\s\)<>]+", repro)
    for url in gh_urls:
        if re.search(r"/(blob|tree)/(main|master|HEAD)\b", url):
            bad.append(f"unpinned GitHub URL `{url}` (use `/blob/<sha>`)")
    if bad:
        return CheckResult(
            "Reproducibility URLs are permanent",
            False,
            "; ".join(bad),
        )
    return CheckResult("Reproducibility URLs are permanent", True)


def check_confidence_matches(body: str) -> CheckResult:
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
    # Look for `Confidence: LOW|MODERATE|HIGH` anywhere in Details.
    cm = re.search(r"Confidence:\s*(LOW|MODERATE|HIGH)\b", details)
    if not cm:
        return CheckResult(
            "Details confidence sentence matches title",
            False,
            "no `Confidence: LOW|MODERATE|HIGH` line in Details",
        )
    if cm.group(1) != title_level:
        return CheckResult(
            "Details confidence sentence matches title",
            False,
            f"title says {title_level}, Details says {cm.group(1)}",
        )
    return CheckResult("Details confidence sentence matches title", True, f"both {title_level}")


def check_figure_caption(body: str) -> CheckResult:
    figure = section_text(body, "Figure")
    if figure is None:
        return CheckResult("Figure caption ≥10 words", False, "Figure section missing")
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


# ─── Driver ────────────────────────────────────────────────────────────────


CHECKS = [
    check_title_confidence,
    check_required_sections,
    check_tldr_labels,
    check_reproducibility_urls,
    check_confidence_matches,
    check_figure_caption,
]


def verify_text(raw: str, *, source: str = "") -> tuple[bool, list[CheckResult]]:
    _, body = split_frontmatter(raw)
    if LEGACY_SAGAN_CARD_SENTINEL in body:
        return True, [
            CheckResult(
                "legacy Sagan-card detected",
                True,
                "skipping markdown spec — body is grandfathered HTML; "
                "run verify_sagan_card.py for those bodies",
            )
        ]
    results = [chk(body) for chk in CHECKS]
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
