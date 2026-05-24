#!/usr/bin/env python
"""Export Sagan literature surfacing batches to markdown files for the EPS dashboard.

Reads from the local Sagan Postgres DB (`lit_items` joined with `lit_inbox`) and
writes:

  - One daily batch file per surfacing date: `<out>/YYYY-MM-DD.md`
  - One paper card per unique paper:         `<out>/papers/<slug>.md`

Connection credentials come from `SAGAN_DATABASE_URL` if set; otherwise the
script looks for `DATABASE_URL_DIRECT` in `~/sagan/services/runner/.env` and,
as a fallback (current Sagan layout puts the env at the repo root), in
`~/sagan/.env`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import psycopg
import yaml
from psycopg.rows import dict_row

SAGAN_ENV_CANDIDATES = [
    Path.home() / "sagan" / "services" / "runner" / ".env",
    Path.home() / "sagan" / ".env",
]

SQL = """
SELECT
    i.id              AS lit_item_id,
    i.arxiv_id        AS arxiv_id,
    i.title           AS title,
    i.authors         AS authors,
    i.abstract        AS abstract,
    i.summary_md      AS summary_md,
    i.relevance_reason_md AS relevance_reason_md,
    i.threat_reason_md AS threat_reason_md,
    i.url             AS url,
    i.pdf_url         AS pdf_url,
    i.topic           AS topic,
    i.released_on     AS released_on,
    i.tags            AS tags,
    b.surfaced_on     AS surfaced_on,
    b.score           AS score,
    b.category        AS category,
    b.reason_md       AS reason_md
FROM lit_inbox b
JOIN lit_items i ON i.id = b.lit_item_id
WHERE b.surfaced_on >= %(since)s
ORDER BY b.surfaced_on DESC, b.score DESC NULLS LAST
"""

SQL_ALL = SQL.replace("WHERE b.surfaced_on >= %(since)s", "")


def parse_sagan_env_file(path: Path) -> str | None:
    """Return DATABASE_URL_DIRECT from a Sagan-style .env file, or None."""
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("DATABASE_URL_DIRECT="):
                val = line.split("=", 1)[1].strip()
                if val.startswith(("'", '"')) and val.endswith(val[0]):
                    val = val[1:-1]
                return val
    return None


def resolve_database_url() -> str:
    url = os.environ.get("SAGAN_DATABASE_URL")
    if url:
        return url
    for candidate in SAGAN_ENV_CANDIDATES:
        url = parse_sagan_env_file(candidate)
        if url:
            return url
    sys.stderr.write(
        "ERROR: SAGAN_DATABASE_URL is unset and DATABASE_URL_DIRECT was not found in "
        f"{', '.join(str(p) for p in SAGAN_ENV_CANDIDATES)}.\n"
    )
    raise SystemExit(2)


@dataclass
class Row:
    lit_item_id: str
    arxiv_id: str | None
    title: str
    authors: list[str]
    abstract: str | None
    summary_md: str | None
    relevance_reason_md: str | None
    threat_reason_md: str | None
    url: str | None
    pdf_url: str | None
    topic: str | None
    released_on: date | None
    tags: list[str]
    surfaced_on: date
    score: int | None
    category: str | None
    reason_md: str | None


def normalize_authors(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        return [str(parsed)]
    return [str(value)]


def normalize_tags(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        return [str(parsed)]
    return [str(value)]


def row_from_record(r: dict) -> Row:
    return Row(
        lit_item_id=str(r["lit_item_id"]),
        arxiv_id=r.get("arxiv_id"),
        title=r["title"],
        authors=normalize_authors(r.get("authors")),
        abstract=r.get("abstract"),
        summary_md=r.get("summary_md"),
        relevance_reason_md=r.get("relevance_reason_md"),
        threat_reason_md=r.get("threat_reason_md"),
        url=r.get("url"),
        pdf_url=r.get("pdf_url"),
        topic=r.get("topic"),
        released_on=r.get("released_on"),
        tags=normalize_tags(r.get("tags")),
        surfaced_on=r["surfaced_on"],
        score=r.get("score"),
        category=r.get("category"),
        reason_md=r.get("reason_md"),
    )


def slug_for(row: Row) -> str:
    if row.arxiv_id:
        return row.arxiv_id
    return row.lit_item_id.replace("-", "")[:12]


def truncate_authors(authors: list[str]) -> str:
    if not authors:
        return "(no authors)"
    if len(authors) <= 3:
        return ", ".join(authors)
    return ", ".join(authors[:3]) + " et al."


def first_chars(text: str | None, n: int) -> str:
    if not text:
        return ""
    flat = re.sub(r"\s+", " ", text).strip()
    if len(flat) <= n:
        return flat
    return flat[:n].rstrip() + "…"


def write_if_changed(target: Path, contents: str) -> bool:
    """Write `contents` to `target`. Return True if file content changed."""
    new_hash = hashlib.sha256(contents.encode("utf-8")).hexdigest()
    if target.exists():
        with target.open("rb") as f:
            existing = f.read()
        if hashlib.sha256(existing).hexdigest() == new_hash:
            return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        f.write(contents)
    return True


def render_daily(day: date, rows: list[Row]) -> str:
    sorted_rows = sorted(rows, key=lambda r: r.score if r.score is not None else -1, reverse=True)
    top_score = sorted_rows[0].score if sorted_rows and sorted_rows[0].score is not None else 0
    fm = {
        "date": day.isoformat(),
        "item_count": len(sorted_rows),
        "top_score": top_score,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    parts = [
        "---",
        yaml.safe_dump(fm, sort_keys=False).rstrip(),
        "---",
        "",
        f"# Literature — {day.isoformat()}",
        "",
    ]
    for r in sorted_rows:
        slug = slug_for(r)
        score = r.score if r.score is not None else "—"
        meta = " · ".join(
            x
            for x in (
                r.category or "uncategorized",
                r.topic or "other",
                truncate_authors(r.authors),
            )
            if x
        )
        parts.append(f"- **[{score}] [{r.title}](/literature/papers/{slug})** — {meta}")
        reason = first_chars(r.relevance_reason_md, 200)
        if reason:
            parts.append(f"  {reason}")
    parts.append("")
    return "\n".join(parts)


@dataclass
class PaperAggregate:
    slug: str
    rows: list[Row] = field(default_factory=list)

    def representative(self) -> Row:
        # Pick the highest-scored surfacing as the source of canonical fields.
        return max(self.rows, key=lambda r: r.score if r.score is not None else -1)


def render_paper(agg: PaperAggregate) -> str:
    rep = agg.representative()
    surfaced_days = sorted({r.surfaced_on for r in agg.rows})
    categories = sorted({r.category for r in agg.rows if r.category})
    highest_score = max((r.score for r in agg.rows if r.score is not None), default=0)
    first_surfaced = min(surfaced_days)

    fm: dict[str, object] = {
        "arxiv_id": rep.arxiv_id,
        "lit_item_id": rep.lit_item_id,
        "title": rep.title,
        "authors": rep.authors,
        "topic": rep.topic,
        "released_on": rep.released_on.isoformat() if rep.released_on else None,
        "url": rep.url,
        "pdf_url": rep.pdf_url,
        "first_surfaced_on": first_surfaced.isoformat(),
        "highest_score": highest_score,
        "categories": categories,
        "surfaced_days": [d.isoformat() for d in surfaced_days],
    }
    if rep.tags:
        fm["tags"] = rep.tags

    parts = [
        "---",
        yaml.safe_dump(fm, sort_keys=False, allow_unicode=True).rstrip(),
        "---",
        "",
        f"# {rep.title}",
        "",
        f"**Authors:** {truncate_authors(rep.authors)}",
        "",
    ]
    link_bits = []
    if rep.url:
        link_bits.append(f"[arXiv]({rep.url})")
    if rep.pdf_url:
        link_bits.append(f"[PDF]({rep.pdf_url})")
    if link_bits:
        parts.append(" · ".join(link_bits))
        parts.append("")

    if rep.summary_md and rep.summary_md.strip():
        parts.extend(["## Summary", "", rep.summary_md.rstrip(), ""])
    if rep.relevance_reason_md and rep.relevance_reason_md.strip():
        parts.extend(["## Relevance", "", rep.relevance_reason_md.rstrip(), ""])
    if rep.threat_reason_md and rep.threat_reason_md.strip():
        parts.extend(["## Threat model", "", rep.threat_reason_md.rstrip(), ""])
    if rep.abstract and rep.abstract.strip():
        parts.extend(["## Abstract", "", rep.abstract.rstrip(), ""])

    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--since",
        default=(date.today() - timedelta(days=7)).isoformat(),
        help="Earliest surfaced_on date (YYYY-MM-DD). Default: today - 7 days.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Override --since; fetch the entire history.",
    )
    p.add_argument(
        "--out",
        default="updates/literature",
        help="Output directory (default: updates/literature).",
    )
    p.add_argument("--dry-run", action="store_true", help="Don't write any files; print plan only.")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    return p.parse_args()


def fetch_rows(database_url: str, *, since: date | None) -> list[Row]:
    sql = SQL_ALL if since is None else SQL
    params = {} if since is None else {"since": since}
    with psycopg.connect(database_url, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return [row_from_record(r) for r in cur.fetchall()]


def group_by_day(rows: list[Row]) -> dict[date, list[Row]]:
    by_day: dict[date, list[Row]] = {}
    for r in rows:
        by_day.setdefault(r.surfaced_on, []).append(r)
    return by_day


def group_by_paper(rows: list[Row]) -> dict[str, PaperAggregate]:
    by_paper: dict[str, PaperAggregate] = {}
    for r in rows:
        slug = slug_for(r)
        agg = by_paper.setdefault(slug, PaperAggregate(slug=slug))
        agg.rows.append(r)
    return by_paper


def main() -> int:
    args = parse_args()
    since: date | None
    if args.all:
        since = None
    else:
        try:
            since = date.fromisoformat(args.since)
        except ValueError as e:
            sys.stderr.write(f"ERROR: --since must be YYYY-MM-DD ({e}).\n")
            return 2

    db_url = resolve_database_url()
    rows = fetch_rows(db_url, since=since)

    if args.verbose:
        rng = "ALL" if since is None else f"since {since.isoformat()}"
        print(f"fetched {len(rows)} rows ({rng})", file=sys.stderr)

    out_dir = Path(args.out)
    papers_dir = out_dir / "papers"

    by_day = group_by_day(rows)
    by_paper = group_by_paper(rows)

    daily_written = 0
    daily_unchanged = 0
    papers_written = 0
    papers_unchanged = 0

    for day, day_rows in sorted(by_day.items(), reverse=True):
        target = out_dir / f"{day.isoformat()}.md"
        body = render_daily(day, day_rows)
        if args.dry_run:
            if args.verbose:
                print(f"[dry-run] would write {target} ({len(day_rows)} items)", file=sys.stderr)
            continue
        if write_if_changed(target, body):
            daily_written += 1
        else:
            daily_unchanged += 1

    for slug, agg in sorted(by_paper.items()):
        target = papers_dir / f"{slug}.md"
        body = render_paper(agg)
        if args.dry_run:
            if args.verbose:
                print(
                    f"[dry-run] would write {target} (highest score "
                    f"{max((r.score for r in agg.rows if r.score is not None), default=0)})",
                    file=sys.stderr,
                )
            continue
        if write_if_changed(target, body):
            papers_written += 1
        else:
            papers_unchanged += 1

    if args.dry_run:
        print(
            f"[dry-run] {len(by_day)} daily file(s), {len(by_paper)} paper card(s) would be written"
        )
    else:
        print(
            f"wrote {daily_written} daily files, {papers_written} paper cards, "
            f"skipped {daily_unchanged + papers_unchanged} unchanged "
            f"({daily_unchanged} daily / {papers_unchanged} paper)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
