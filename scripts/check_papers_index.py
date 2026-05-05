"""Lint papers/INDEX.md.

Usage
-----
    uv run python scripts/check_papers_index.py

Checks
------
1. Every paper row has a non-empty `Summary` cell (no sentinels: `{{`,
   `TBD`, ``_(needs Summary)_``, ``_(needs Use sentence)_``).
2. Every paper row has a non-empty `Use:` cell (must contain the literal
   ``**Use:**`` label) and is not a sentinel.
3. If `.arxiv-papers/` exists (per-user MCP cache), every cached paper
   has a row in `papers/INDEX.md`. The reverse direction (rows without
   a cache match) is fine — the index is canonical, the cache is
   per-user.
4. The table header matches the expected schema.

Exits 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = REPO_ROOT / "papers" / "INDEX.md"
ARXIV_CACHE = REPO_ROOT / ".arxiv-papers"

ARXIV_ID_RE = re.compile(r"\b([0-9]{4}\.[0-9]{4,5})\b")
SENTINELS = (
    "{{",
    "TBD",
    "_(needs Summary)_",
    "_(needs Use sentence)_",
    "_(needs title)_",
    "_(needs authors)_",
    "_(needs year)_",
)
EXPECTED_HEADER = "| arxiv-id | Title | Authors | Year | Summary | Use | Cited in |"


def _split_pipe_row(line: str) -> list[str]:
    cells = line.strip().split("|")
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return [c.strip() for c in cells]


def parse_rows(path: Path) -> tuple[bool, list[dict]]:
    """Return (header_ok, list of {arxiv_id, summary, use, cited_in})."""
    if not path.exists():
        return False, []
    rows: list[dict] = []
    in_table = False
    header_ok = False
    for line in path.read_text().splitlines():
        if line.startswith("| arxiv-id "):
            header_ok = line.strip() == EXPECTED_HEADER
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            cells = _split_pipe_row(line)
            if len(cells) < 7:
                continue
            arxiv_cell, _title, _authors, _year, summary, use, cited_in = cells[:7]
            m = ARXIV_ID_RE.search(arxiv_cell)
            if not m:
                continue
            rows.append(
                {
                    "arxiv_id": m.group(1),
                    "summary": summary,
                    "use": use,
                    "cited_in": cited_in,
                }
            )
        elif in_table and not line.startswith("|") and line.strip():
            in_table = False
    return header_ok, rows


def discover_cached_papers() -> set[str]:
    if not ARXIV_CACHE.is_dir():
        return set()
    found: set[str] = set()
    for f in ARXIV_CACHE.glob("*.md"):
        m = ARXIV_ID_RE.search(f.name)
        if m:
            found.add(m.group(1))
    return found


def main() -> int:
    failures: list[str] = []

    header_ok, rows = parse_rows(INDEX_PATH)
    if not INDEX_PATH.exists():
        print(f"[check_papers_index] FAIL: {INDEX_PATH} does not exist", file=sys.stderr)
        return 1

    if not header_ok:
        failures.append(f"Table header does not match expected schema: {EXPECTED_HEADER!r}")

    if not rows:
        failures.append("No paper rows found in papers/INDEX.md table.")

    for r in rows:
        if not r["summary"] or any(s in r["summary"] for s in SENTINELS):
            failures.append(
                f"Paper {r['arxiv_id']}: empty/sentinel Summary cell ({r['summary']!r})"
            )
        if not r["use"] or any(s in r["use"] for s in SENTINELS):
            failures.append(f"Paper {r['arxiv_id']}: empty/sentinel Use cell ({r['use']!r})")
        if "**Use:**" not in r["use"]:
            failures.append(
                f"Paper {r['arxiv_id']}: Use cell missing the literal `**Use:**` label "
                f"({r['use']!r})"
            )

    indexed_ids = {r["arxiv_id"] for r in rows}
    cached = discover_cached_papers()
    missing_in_index = cached - indexed_ids
    for arxiv_id in sorted(missing_in_index):
        failures.append(
            f"Paper {arxiv_id} present in .arxiv-papers/ but missing from papers/INDEX.md"
        )

    if failures:
        print("[check_papers_index] FAIL:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"[check_papers_index] PASS: {len(rows)} paper(s) — all Summary + Use cells filled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
