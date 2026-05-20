"""Regenerate papers/INDEX.md, preserving hand-written Summary + Use cells.

Usage
-----
    uv run python scripts/render_papers_index.py            # write papers/INDEX.md
    uv run python scripts/render_papers_index.py --stdout   # print only
    uv run python scripts/render_papers_index.py --check    # exit 1 if rendered != on-disk

How it works
------------
1. Reads the existing `papers/INDEX.md` to capture each paper's row,
   keyed by arxiv-id. Hand-written `Summary` and `Use:` cells are
   PRESERVED exactly.
2. Optionally augments the row set from `.arxiv-papers/*.md` (the MCP
   cache, gitignored) IF the directory exists. New papers in the cache
   that aren't in `papers/INDEX.md` get a stub row inserted with
   `_(needs Summary)_` and `**Use:** _(needs Use sentence)_` — the
   linter (`scripts/check_papers_index.py`) then FAILS until the user
   fills both cells. Papers in `papers/INDEX.md` that aren't in the
   cache stay (the cache is per-user; the index is canonical).
3. Re-grep the repo for citations of each arxiv-id and recompute the
   `Cited in` cell. Citations counted: `arxiv.org/abs/<id>`, the bare
   `<id>`, `[<id>]`, or `arXiv:<id>` across `RESULTS.md`, `docs/`,
   `.claude/plans/`, and clean-result GitHub issue bodies via `gh
   issue list --label clean-results --json body`.
4. Re-emit the table with rows in arxiv-id order.

This keeps the slow-moving Summary / Use columns hand-written (human
judgement) while the mechanical Cited-in column auto-updates.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = REPO_ROOT / "papers" / "INDEX.md"
ARXIV_CACHE = REPO_ROOT / ".arxiv-papers"

ARXIV_ID_RE = re.compile(r"\b([0-9]{4}\.[0-9]{4,5})\b")
TABLE_HEADER = (
    "| arxiv-id | Title | Authors | Year | Summary | Use | Cited in |\n"
    "|---|---|---|---|---|---|---|\n"
)


class PaperRow(NamedTuple):
    arxiv_id: str
    title: str
    authors: str
    year: str
    summary: str
    use: str
    cited_in: str  # rendered last; recomputed each run


def _split_pipe_row(line: str) -> list[str]:
    """Split a markdown table row on `|` and trim each cell."""
    cells = line.strip().split("|")
    # Leading and trailing empty cells (from outer `| ` borders) are dropped.
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return [c.strip() for c in cells]


def parse_existing_rows(path: Path) -> dict[str, PaperRow]:
    """Parse existing papers/INDEX.md table rows keyed by arxiv-id."""
    if not path.exists():
        return {}
    rows: dict[str, PaperRow] = {}
    in_table = False
    for line in path.read_text().splitlines():
        if line.startswith("| arxiv-id "):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            cells = _split_pipe_row(line)
            if len(cells) < 7:
                continue
            arxiv_id_cell, title, authors, year, summary, use, cited_in = cells[:7]
            m = ARXIV_ID_RE.search(arxiv_id_cell)
            if not m:
                continue
            rows[m.group(1)] = PaperRow(
                arxiv_id=m.group(1),
                title=title,
                authors=authors,
                year=year,
                summary=summary,
                use=use,
                cited_in=cited_in,
            )
        elif in_table and not line.startswith("|") and line.strip():
            in_table = False
    return rows


def discover_cached_papers() -> set[str]:
    """Return arxiv-ids found in .arxiv-papers/*.md, if the dir exists."""
    if not ARXIV_CACHE.is_dir():
        return set()
    found: set[str] = set()
    for f in ARXIV_CACHE.glob("*.md"):
        m = ARXIV_ID_RE.search(f.name)
        if m:
            found.add(m.group(1))
    return found


def grep_citations(arxiv_id: str) -> list[str]:
    """Grep RESULTS.md, docs/, .claude/plans/ for citations of arxiv_id.

    Returns a deduplicated list of human-readable source labels (e.g.
    ``RESULTS.md``, ``docs/research_ideas.md``, ``.claude/plans/issue-186.md``).
    Clean-result GitHub issue bodies are queried separately by
    ``grep_clean_result_citations``.
    """
    sources: set[str] = set()
    for root in [REPO_ROOT / "RESULTS.md", REPO_ROOT / "docs", REPO_ROOT / ".claude" / "plans"]:
        if not root.exists():
            continue
        if root.is_file():
            text = root.read_text(errors="ignore")
            if arxiv_id in text:
                sources.add(root.relative_to(REPO_ROOT).as_posix())
        else:
            for p in root.rglob("*.md"):
                try:
                    text = p.read_text(errors="ignore")
                except OSError:
                    continue
                if arxiv_id in text:
                    sources.add(p.relative_to(REPO_ROOT).as_posix())
    return sorted(sources)


def grep_clean_result_citations(arxiv_id: str) -> list[str]:
    """Query GitHub for clean-result issues that cite arxiv_id."""
    try:
        out = subprocess.run(
            [
                "gh",
                "issue",
                "list",
                "--label",
                "clean-results",
                "--state",
                "all",
                "--json",
                "number,body,url",
                "--limit",
                "200",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    matched: list[str] = []
    for row in json.loads(out.stdout):
        body = row.get("body") or ""
        if arxiv_id in body:
            matched.append(f"[#{row['number']}]({row['url']})")
    return sorted(matched)


def render_cited_in_cell(arxiv_id: str) -> str:
    file_refs = grep_citations(arxiv_id)
    issue_refs = grep_clean_result_citations(arxiv_id)
    pieces: list[str] = []
    pieces.extend(f"`{p}`" for p in file_refs)
    pieces.extend(issue_refs)
    return ", ".join(pieces) if pieces else "_(none)_"


def render_table(rows: dict[str, PaperRow]) -> str:
    lines = [TABLE_HEADER.rstrip("\n")]
    for arxiv_id in sorted(rows):
        r = rows[arxiv_id]
        cited = render_cited_in_cell(arxiv_id)
        lines.append(
            f"| [{arxiv_id}](https://arxiv.org/abs/{arxiv_id}) "
            f"| {r.title} | {r.authors} | {r.year} | {r.summary} | {r.use} | {cited} |"
        )
    return "\n".join(lines) + "\n"


def render_full_index(rows: dict[str, PaperRow]) -> str:
    """Render papers/INDEX.md, preserving the prose preamble before the table."""
    preamble_default = (
        "# Papers — Index\n\n"
        "Hand-curated reading list of papers that frame this project. Each paper\n"
        "has a short summary AND a concrete `Use:` annotation (one sentence\n"
        "explaining why this project pulls on it). Backlinks are auto-populated\n"
        "by `scripts/render_papers_index.py` from grep over `RESULTS.md`,\n"
        "`docs/`, `.claude/plans/`, and clean-result GitHub issue bodies.\n\n"
        "The `Summary` and `Use:` columns are **hand-written** — the generator\n"
        "preserves them when regenerating the table from the arxiv cache. CI\n"
        "linter (`scripts/check_papers_index.py`) fails if any paper has an\n"
        "empty `Summary` or empty `Use:` cell.\n\n"
    )
    preamble = preamble_default
    if INDEX_PATH.exists():
        existing = INDEX_PATH.read_text()
        idx = existing.find("| arxiv-id ")
        if idx > 0:
            preamble = existing[:idx]
    return preamble + render_table(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stdout", action="store_true", help="Print only.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if the on-disk INDEX.md is stale (does NOT write).",
    )
    args = parser.parse_args(argv)

    rows = parse_existing_rows(INDEX_PATH)

    cached = discover_cached_papers()
    for arxiv_id in cached - rows.keys():
        rows[arxiv_id] = PaperRow(
            arxiv_id=arxiv_id,
            title="_(needs title)_",
            authors="_(needs authors)_",
            year="_(needs year)_",
            summary="_(needs Summary)_",
            use="**Use:** _(needs Use sentence)_",
            cited_in="_(none)_",
        )

    rendered = render_full_index(rows)

    if args.stdout:
        sys.stdout.write(rendered)
        return 0
    if args.check:
        on_disk = INDEX_PATH.read_text() if INDEX_PATH.exists() else ""
        if rendered != on_disk:
            print(
                "[render_papers_index] STALE: papers/INDEX.md differs from rendered output. "
                "Run `uv run python scripts/render_papers_index.py` to regenerate.",
                file=sys.stderr,
            )
            return 1
        print("[render_papers_index] papers/INDEX.md is up-to-date")
        return 0
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(rendered)
    print(f"[render_papers_index] wrote {INDEX_PATH} ({len(rows)} paper(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
