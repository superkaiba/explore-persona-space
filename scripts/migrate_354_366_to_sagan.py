#!/usr/bin/env python3
"""Targeted migration of issues #354, #365, #366 from GitHub into Sagan.

Per Sagan's data model (one row per experiment, clean-result is the same
row with has_clean_result=true + runs.classification='pending'):

  - #354: UPDATE existing row. Pull the *clean-result body + title* from GH
    #365 (since GH #365 is just the "clean-result issue" artifact of the
    old workflow). Set status='awaiting_promotion', has_clean_result=true.
    INSERT a runs row with classification='pending'. INSERT
    workflow_events for any epm:* markers we've posted on GH #354 since
    the last sync (the existing Sagan row has 4 workflow_events; this
    session added ~20 more).

  - #365 (GitHub clean-result issue): NO Sagan row. Its body lives inside
    #354's experiments.body.

  - #366: INSERT a fresh row (status='proposed', kind='experiment',
    compute_size='small', priority='medium') plus a parent edge to #354.

Usage:
    SAGAN_DATABASE_URL=postgresql://... uv run python scripts/migrate_354_366_to_sagan.py --dry-run
    SAGAN_DATABASE_URL=postgresql://... uv run python scripts/migrate_354_366_to_sagan.py --apply

Reads DATABASE_URL_DIRECT from /home/thomasjiralerspong/sagan/.env by default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SAGAN_ENV = Path("/home/thomasjiralerspong/sagan/.env")
MARKER_RE = re.compile(r"<!--\s*(epm:[a-z][a-z0-9-]*)", re.IGNORECASE)


def load_sagan_db_url() -> str:
    override = os.environ.get("SAGAN_DATABASE_URL")
    if override:
        return override
    if not SAGAN_ENV.exists():
        sys.exit(f"Sagan .env not found at {SAGAN_ENV}; set SAGAN_DATABASE_URL")
    for raw in SAGAN_ENV.read_text().splitlines():
        if raw.startswith("DATABASE_URL_DIRECT="):
            value = raw.split("=", 1)[1].strip().strip('"').strip("'")
            return value
    sys.exit(f"DATABASE_URL_DIRECT not found in {SAGAN_ENV}")


def gh_issue(number: int) -> dict:
    out = subprocess.check_output(
        [
            "gh",
            "issue",
            "view",
            str(number),
            "--json",
            "number,title,body,labels,state,createdAt,updatedAt,comments,author",
        ],
        text=True,
    )
    return json.loads(out)


def parse_markers(comments: list[dict]) -> list[dict]:
    """Extract epm:* marker comments from a GitHub comment list."""
    out: list[dict] = []
    for c in comments:
        body = c.get("body") or ""
        m = MARKER_RE.search(body)
        if not m:
            continue
        out.append(
            {
                "marker_type": m.group(1).lower(),
                "body": body,
                "created_at": c.get("createdAt"),
                "author": (c.get("author") or {}).get("login") or "unknown",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.dry_run == args.apply:
        sys.exit("Specify exactly one of --dry-run or --apply")

    import psycopg

    db_url = load_sagan_db_url()
    print(f"DB: {db_url.split('@', 1)[-1].split('/', 1)[0]} (host)", file=sys.stderr)

    # ── Pull GitHub state ───────────────────────────────────────────────────
    print("\n[1/3] Pulling GitHub issues #354, #365, #366 …", file=sys.stderr)
    gh354 = gh_issue(354)
    gh365 = gh_issue(365)
    gh366 = gh_issue(366)
    markers354 = parse_markers(gh354.get("comments", []))
    markers366 = parse_markers(gh366.get("comments", []))
    print(
        f"  #354: {len(gh354.get('comments', []))} comments, {len(markers354)} markers",
        file=sys.stderr,
    )
    print(f"  #365: clean-result body, len={len(gh365['body'] or '')}", file=sys.stderr)
    print(f"  #366: fresh proposal, {len(markers366)} markers", file=sys.stderr)

    # ── Connect to Sagan ───────────────────────────────────────────────────
    print("\n[2/3] Reading current Sagan state …", file=sys.stderr)
    conn = psycopg.connect(db_url, sslmode="require", autocommit=False)
    cur = conn.cursor()

    # Current state of #354 in Sagan
    cur.execute(
        "SELECT id, status, has_clean_result, title, length(coalesce(body,'')) "
        "FROM experiments WHERE number = 354"
    )
    row354 = cur.fetchone()
    if not row354:
        sys.exit("ERROR: #354 not in Sagan; aborting (run the full importer instead)")
    id354, status354, hcr354, title354, body_len354 = row354
    print(
        f"  #354 in Sagan: id={id354}, status={status354}, has_clean_result={hcr354}, body_len={body_len354}",
        file=sys.stderr,
    )

    # Existing workflow_events for #354
    cur.execute(
        "SELECT metadata->>'marker_type', created_at "
        "FROM workflow_events WHERE entity_id = %s AND entity_kind = 'experiment' "
        "ORDER BY created_at",
        (id354,),
    )
    existing_markers354 = cur.fetchall()
    existing_marker_set = {(mt or "").lower() for mt, _ in existing_markers354}
    print(
        f"  #354 existing workflow_events: {len(existing_markers354)} ({sorted(set(mt for mt, _ in existing_markers354))[:6]}…)",
        file=sys.stderr,
    )

    # Existing runs for #354
    cur.execute("SELECT id, classification FROM runs WHERE experiment_id = %s", (id354,))
    existing_runs354 = cur.fetchall()
    print(f"  #354 existing runs: {len(existing_runs354)} {existing_runs354}", file=sys.stderr)

    # #366 should NOT exist yet
    cur.execute("SELECT id, status FROM experiments WHERE number = 366")
    row366 = cur.fetchone()
    if row366:
        print(f"  WARNING: #366 already in Sagan: {row366}", file=sys.stderr)

    # ── Plan diffs ─────────────────────────────────────────────────────────
    print("\n[3/3] Planned changes …", file=sys.stderr)

    # Map GitHub #354 labels → Sagan fields (mostly verifying current row)
    label_names354 = {l["name"] for l in gh354.get("labels", [])}
    new_status = "awaiting_promotion"  # current GH label is status:awaiting_promotion
    # Title + body come from GH #365 (the clean-result)
    new_title = gh365["title"]
    new_body = gh365["body"]

    print("\n  CHANGE 1 — UPDATE experiments (#354):")
    print(f"    title:            {title354[:80]}…")
    print(f"      →               {new_title[:80]}…")
    print(f"    status:           {status354}  →  {new_status}")
    print(f"    has_clean_result: {hcr354}  →  True")
    print(f"    body length:      {body_len354}  →  {len(new_body)}")

    # New workflow_events: markers on GH #354 we haven't synced yet
    new_markers354 = [m for m in markers354 if m["marker_type"] not in existing_marker_set]
    print("\n  CHANGE 2 — INSERT workflow_events for #354 (new since last sync):")
    print(f"    {len(new_markers354)} new marker rows to insert")
    if new_markers354:
        for m in new_markers354[:8]:
            print(f"      + {m['created_at']}  {m['marker_type']:35s}  ({len(m['body'])} bytes)")
        if len(new_markers354) > 8:
            print(f"      … and {len(new_markers354) - 8} more")

    # New runs row for #354
    needs_runs_row = not any(c is not None for _, c in existing_runs354)
    print("\n  CHANGE 3 — INSERT runs row for #354 (classification='pending')")
    print(
        f"    {'YES — no runs row exists' if needs_runs_row else 'SKIP — runs row already exists'}"
    )

    # #366 fresh insert
    label_names366 = {l["name"] for l in gh366.get("labels", [])}
    kind366 = "experiment" if "type:experiment" in label_names366 else "infra"
    compute366 = "small" if "compute:small" in label_names366 else None
    priority366 = "normal"  # GH prio:medium has no direct map; default
    print("\n  CHANGE 4 — INSERT experiment (#366):")
    print(f"    title:        {gh366['title'][:80]}…")
    print("    status:       proposed")
    print(f"    kind:         {kind366}")
    print(f"    compute_size: {compute366}")
    print(f"    priority:     {priority366}")
    print(f"    body length:  {len(gh366['body'] or '')}")

    print("\n  CHANGE 5 — INSERT parent edge: #366 → #354")

    if args.dry_run:
        print("\nDRY RUN — no DB writes performed.", file=sys.stderr)
        return

    # ── Apply ──────────────────────────────────────────────────────────────
    print("\nApplying changes …", file=sys.stderr)
    with conn.transaction():
        # 1. UPDATE #354
        cur.execute(
            "UPDATE experiments SET title=%s, body=%s, status=%s::experiment_status, "
            "has_clean_result=true, updated_at=now() WHERE id=%s",
            (new_title, new_body, new_status, id354),
        )
        print(f"  Updated #354 (rowcount={cur.rowcount})", file=sys.stderr)

        # 2. INSERT new workflow_events for #354
        if new_markers354:
            cur.executemany(
                "INSERT INTO workflow_events "
                "(entity_kind, entity_id, event_type, note, metadata, created_at) "
                "VALUES ('experiment', %s, 'note'::workflow_event_type, %s, %s::jsonb, %s)",
                [
                    (
                        id354,
                        m["body"][:4000],
                        json.dumps(
                            {
                                "marker_type": m["marker_type"],
                                "author": m["author"],
                                "legacy_gh_number": 354,
                                "migration_source": "migrate_354_366_to_sagan.py",
                            }
                        ),
                        m["created_at"],
                    )
                    for m in new_markers354
                ],
            )
            print(f"  Inserted {len(new_markers354)} workflow_events for #354", file=sys.stderr)

        # 3. INSERT runs row if needed
        if needs_runs_row:
            cur.execute(
                "INSERT INTO runs (experiment_id, classification, created_at, updated_at) "
                "VALUES (%s, 'pending'::run_classification, now(), now())",
                (id354,),
            )
            print("  Inserted runs row for #354 (classification='pending')", file=sys.stderr)

        # 4. INSERT #366
        cur.execute(
            "INSERT INTO experiments (number, title, body, status, kind, compute_size, priority, "
            "assignee_kind, tags, has_clean_result, legacy_gh_number, created_at, updated_at) "
            "VALUES (%s, %s, %s, 'proposed'::experiment_status, %s::experiment_kind, "
            "%s::compute_size, %s::priority, 'agent'::assignee_kind, %s, false, %s, %s, %s) "
            "RETURNING id",
            (
                366,
                gh366["title"],
                gh366["body"],
                kind366,
                compute366,
                priority366,
                [l["name"] for l in gh366.get("labels", [])],
                366,
                gh366["createdAt"],
                gh366["updatedAt"],
            ),
        )
        id366 = cur.fetchone()[0]
        print(f"  Inserted #366 (id={id366})", file=sys.stderr)

        # 5. INSERT parent edge: #366 → #354
        cur.execute(
            "INSERT INTO edges (from_kind, from_id, to_kind, to_id, type) "
            "VALUES ('experiment', %s, 'experiment', %s, 'parent'::edge_type) "
            "ON CONFLICT DO NOTHING",
            (id366, id354),
        )
        print("  Inserted parent edge #366 → #354", file=sys.stderr)

    # Explicit commit — `with conn.transaction()` only manages a savepoint
    # inside the outer implicit transaction opened by our first SELECT.
    # Without this commit the entire migration rolls back at process exit.
    conn.commit()
    print("\nDone (committed).", file=sys.stderr)


if __name__ == "__main__":
    main()
