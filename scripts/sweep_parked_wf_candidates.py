"""Read-only enumerator of PARKED workflow-fix candidates (/daily Step C input).

Scans every ``tasks/*/*/events.jsonl`` (ALL statuses — both evidence parks live
on terminal tasks: #1100 completed, #1101 archived) plus the outside-task
fallback stream ``.claude/cache/workflow-fix-events.jsonl`` for
``epm:workflow-fix-candidate`` rows whose note/``routed`` field marks them
PARKED (the recursion-guard escape valve,
``.claude/rules/workflow-fix-on-bug.md`` § Recursion guard), and prints ONE
JSON object to stdout listing the candidates no later routed-record has
closed. The /daily "Parked workflow-fix-candidate routing pass (Step C)"
(``.claude/skills/daily/SKILL.md``) routes each through the three-route
classifier and posts the ``epm:workflow-fix-task-filed`` routed-record this
sweep's suppression rule (1) keys on.

Suppression rules (a candidate is SUPPRESSED == already routed):

1. **Same-stream filed record** — a LATER row (aware-UTC ts > candidate ts)
   in the SAME stream with kind ``epm:workflow-fix-task-filed*`` matching the
   candidate: fp-computable candidates match ONLY on the fingerprint (same
   ``target_file`` + different fp is NOT a duplicate — workflow-fix-on-bug.md
   § Dedup); fp-less (prose) parks match on the record's
   ``origin_candidate_ts`` (PRIMARY key), falling back to a ``target_file``
   string match ONLY when the record carries no ``origin_candidate_ts``
   (legacy/backfill records).
2. **fp-tag scan** (fingerprint computable only) — a ``kind: infra`` task
   whose ``body.md`` carries ``wf-fix-fp:<fp>`` (tag) or ``fingerprint: <fp>``
   (Provenance line). A NON-terminal hit suppresses unconditionally; a
   TERMINAL (completed/archived) hit suppresses only when the task's creation
   ts (first parseable events.jsonl row) POSTdates the candidate ts — a
   candidate that predates the closed fix was subsumed by it; one raised
   after it closed is a genuine re-raise and stays enumerated.
3. **Row dedup** — identical (stream, ts, content-hash) rows collapse to one
   (observed verbatim duplication in #1100's events.jsonl).

fp-less prose parks are NEVER auto-suppressed on file-only matches (the
#622-class distinct-bug-on-a-hot-file hazard); instead the advisory field
``open_wf_fix_on_file`` is emitted for the /daily LLM's content-level dedup.
NOTE: that advisory mirrors ``task_workflow.is_open_workflow_fix_task``
semantics — since #1180 both share ``task_workflow.WF_FIX_TITLE_PREFIXES``
(``workflow-fix:`` AND ``daily-fix:`` titles), so open filings from EITHER
channel are visible. Advisory only; the LLM-side content dedup in Step C is
the real check.

Timestamp discipline: every ts is normalized to an AWARE-UTC datetime before
any comparison (the cache file carries ``-07:00`` offset rows alongside
``Z``-form; string comparison misorders them). A naive-parsing ts is assumed
UTC. A JSONL line that fails to parse, or a candidate/filed row whose ts is
missing/unparseable, is SKIPPED and COUNTED in the top-level ``skipped_rows``
(never a crash, never a silent drop). Exit code is 0 always — this is an
enumerator, not a gate.

Usage:
    uv run python scripts/sweep_parked_wf_candidates.py [--window-days 0]
        [--include-routed] [--tasks-root PATH] [--cache-file PATH]

``--window-days 0`` (the DEFAULT) is UNBOUNDED; a positive N scopes an audit
to candidates parked within the last N days. Default output lists only
UNSUPPRESSED candidates; ``--include-routed`` includes suppressed ones too.
``--tasks-root`` / ``--cache-file`` are test overrides (an explicit override
that does not exist fails loud; the DEFAULT cache file is skipped when absent).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from explore_persona_space.task_workflow import (
    WF_FIX_TITLE_PREFIXES,
    repo_root,
    tasks_dir,
    wf_fix_fingerprint,
)

CANDIDATE_KIND_PREFIX = "epm:workflow-fix-candidate"
FILED_KIND_PREFIX = "epm:workflow-fix-task-filed"
TERMINAL_STATUSES = ("completed", "archived")

_PARKED_LEAD_RE = re.compile(r"\s*parked\b", re.IGNORECASE)
_PARKED_ROUTED_RE = re.compile(r"routed:\s*parked\b", re.IGNORECASE)
_ARCHITECTURAL_RE = re.compile(r"parked:\s*architectural", re.IGNORECASE)
_BLOCK_RE = re.compile(
    r"<!--\s*workflow-fix-candidate v1\s*-->(.*?)<!--\s*/workflow-fix-candidate\s*-->",
    re.DOTALL,
)
_TARGET_FILE_RE = re.compile(r"target_file:\s*([^\s,;]+)")
_ORIGIN_TS_RE = re.compile(r"origin_candidate_ts:\s*(\S+)")
_FILED_TASK_RE = re.compile(r"filed_task:\s*(#?\d+)")


def parse_ts(raw: object) -> datetime | None:
    """Parse an ISO-8601-ish timestamp to an AWARE-UTC datetime; None on failure.

    Accepts ``Z``-suffix, explicit-offset, and naive forms (naive is assumed
    UTC — the project's marker rows are written in UTC). Returns None for a
    missing / non-string / unparseable value so callers can skip-and-count.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().rstrip(".,;")
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _read_frontmatter(text: str) -> dict:
    """Best-effort YAML frontmatter of a body.md; {} when absent/unparseable."""
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        fm = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return {}
    return fm if isinstance(fm, dict) else {}


@dataclass
class Candidate:
    """One parked workflow-fix candidate row, normalized across row shapes."""

    source: str
    ts_raw: str
    ts: datetime
    note: str
    target_file: str | None
    fingerprint: str | None
    park_form: str
    formal_block: bool
    suppressed: bool = False
    suppressed_by: dict | None = None
    open_wf_fix_on_file: int | None = None

    def to_json(self) -> dict:
        return {
            "source": self.source,
            "ts": self.ts_raw,
            "target_file": self.target_file,
            "fingerprint": self.fingerprint,
            "park_form": self.park_form,
            "formal_block": self.formal_block,
            "note": self.note,
            "suppressed": self.suppressed,
            "suppressed_by": self.suppressed_by,
            "open_wf_fix_on_file": self.open_wf_fix_on_file,
        }


def _row_kind(row: dict) -> str:
    """The row's marker kind under EITHER key ('kind' events.jsonl / 'marker' cache rows).

    None-guarded: 43 of the live cache file's 88 rows lack a 'kind' key (some
    carry 'marker', some neither) — a bare row['kind'] would crash the scan.
    """
    return str(row.get("kind") or row.get("marker") or "")


def _row_is_parked(row: dict) -> bool:
    """Parked-ness from EITHER surface: the note text or a structured 'routed' field."""
    note = str(row.get("note") or "")
    if _PARKED_LEAD_RE.match(note) or _PARKED_ROUTED_RE.search(note):
        return True
    routed = row.get("routed")
    return isinstance(routed, str) and "parked" in routed.lower()


def _park_form(row: dict) -> str:
    surface = f"{row.get('note') or ''}\n{row.get('routed') or ''}"
    return "architectural" if _ARCHITECTURAL_RE.search(surface) else "recursion-guard"


def _extract_fields(row: dict) -> tuple[str | None, str | None, bool]:
    """(target_file, fingerprint, formal_block) for one candidate row.

    Structured (marker-key) cache rows carry their fields as JSON keys; a
    note embedding a formal candidate block is parsed verbatim; a prose park
    gets a best-effort target_file regex and fingerprint None (the /daily LLM
    synthesizes the fields + fp at route time, per the prose-synthesis rule).
    """
    note = str(row.get("note") or "")
    m = _BLOCK_RE.search(note)
    if m:
        block = m.group(1)

        def block_field(name: str) -> str | None:
            fm = re.search(rf"^{name}:\s*(.+)$", block, re.MULTILINE)
            return fm.group(1).strip() if fm else None

        target_file = block_field("target_file")
        proposed = block_field("proposed_change")
        bug = block_field("bug_observed")
        fp = wf_fix_fingerprint(proposed, bug) if proposed and bug else None
        return target_file, fp, True
    if "note" not in row and ("proposed_change" in row or "target_file" in row):
        target_file = row.get("target_file") or None
        proposed, bug = row.get("proposed_change"), row.get("bug_observed")
        fp = wf_fix_fingerprint(str(proposed), str(bug)) if proposed and bug else None
        return target_file, fp, False
    tf_match = _TARGET_FILE_RE.search(note)
    target_file = tf_match.group(1).rstrip(".,;:!?") if tf_match else None
    return target_file, None, False


def _filed_record_matches(cand: Candidate, record: dict) -> bool:
    """Does this same-stream filed record close this candidate? (suppression rule 1)."""
    note = str(record.get("note") or "")
    if cand.fingerprint is not None:
        # fp-aware: ONLY the candidate's own fingerprint suppresses; a
        # file-matching record with a different fp is a DIFFERENT bug.
        if cand.fingerprint in note:
            return True
        return cand.fingerprint in str(record.get("fingerprint") or "")
    # fp-less (prose park): PRIMARY key = origin_candidate_ts equality.
    origin_raw = None
    om = _ORIGIN_TS_RE.search(note)
    if om:
        origin_raw = om.group(1)
    elif record.get("origin_candidate_ts"):
        origin_raw = str(record["origin_candidate_ts"])
    if origin_raw is not None:
        origin_dt = parse_ts(origin_raw)
        return origin_dt is not None and origin_dt == cand.ts
    # Legacy record with NO origin_candidate_ts: fall back to target_file match.
    if not cand.target_file:
        return False
    rec_tf = record.get("target_file")
    if not rec_tf:
        tm = _TARGET_FILE_RE.search(note)
        rec_tf = tm.group(1).rstrip(".,;:!?") if tm else None
    return rec_tf == cand.target_file


def _filed_ref(record: dict) -> str:
    note = str(record.get("note") or "")
    fm = _FILED_TASK_RE.search(note)
    if fm:
        ref = fm.group(1)
        return ref if ref.startswith("#") else f"#{ref}"
    filed = record.get("filed_task")
    if filed:
        return str(filed) if str(filed).startswith("#") else f"#{filed}"
    return str(record.get("ts") or "")


def _load_task_bodies(tasks_root: Path) -> list[tuple[int, str, Path, str, dict]]:
    """(task_id, status, body_path, body_text, frontmatter) across every status folder.

    Loaded ONCE per sweep and shared by the fp-tag scan + the open-wf-fix
    advisory — re-reading ~10^3 bodies (plus a YAML parse each) per candidate
    blew the <30 s wall-time bound on the live tree (109 candidates x ~1.3k
    bodies on the first unbounded audit).
    """
    bodies: list[tuple[int, str, Path, str, dict]] = []
    for body_path in sorted(tasks_root.glob("*/*/body.md")):
        status = body_path.parent.parent.name
        try:
            tid = int(body_path.parent.name)
        except ValueError:
            continue
        try:
            text = body_path.read_text(encoding="utf-8")
        except OSError:
            continue
        bodies.append((tid, status, body_path, text, _read_frontmatter(text)))
    return bodies


def _task_creation_ts(task_dir: Path) -> datetime | None:
    """First parseable events.jsonl row ts (the task's creation time); None if unreadable."""
    events = task_dir / "events.jsonl"
    try:
        # split("\n"), NEVER splitlines(): splitlines() splits on U+2028/U+2029
        # etc. and shreds valid JSONL rows whose note strings carry them
        # (.claude/rules/gotchas.md "splitlines shreds JSONL", #950).
        lines = events.read_text(encoding="utf-8").split("\n")
    except OSError:
        return None
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        dt = parse_ts(row.get("ts"))
        if dt is not None:
            return dt
    return None


def _fp_tag_scan(
    bodies: list[tuple[int, str, Path, str, dict]], fp: str, cand_ts: datetime
) -> dict | None:
    """Suppression rule 2: an infra task carrying this fp (tag or Provenance line).

    Non-terminal hit -> suppress unconditionally; terminal hit -> suppress only
    when the task's creation ts POSTdates the candidate (the candidate was
    subsumed by the fix). An unreadable creation ts fails toward ENUMERATION
    (treated as a re-raise), never toward a silent drop.
    """
    for tid, status, body_path, text, fm in bodies:
        if fm.get("kind") != "infra":
            continue
        if f"wf-fix-fp:{fp}" not in text and f"fingerprint: {fp}" not in text:
            continue
        if status not in TERMINAL_STATUSES:
            return {"kind": "fp-tag-open", "ref": f"#{tid}"}
        created = _task_creation_ts(body_path.parent)
        if created is not None and created > cand_ts:
            return {"kind": "fp-tag-closed", "ref": f"#{tid}"}
    return None


def _open_wf_fix_on_file(
    bodies: list[tuple[int, str, Path, str, dict]], target_file: str
) -> int | None:
    """Advisory mirror of ``task_workflow.is_open_workflow_fix_task(target_file, None)``.

    Same predicate, keyed on the scanned tree so test fixtures exercise it:
    kind infra + non-terminal status + a ``WF_FIX_TITLE_PREFIXES`` title
    (``workflow-fix:`` OR ``daily-fix:`` — both filing channels, #1180) + a
    Provenance ``workflow_fix_target: <target_file>`` line. Advisory only (see
    module docstring); the /daily LLM's content dedup is the real check.
    """
    for tid, status, _body_path, text, fm in bodies:
        if status in TERMINAL_STATUSES:
            continue
        if fm.get("kind") != "infra":
            continue
        if not str(fm.get("title") or "").startswith(WF_FIX_TITLE_PREFIXES):
            continue
        if f"workflow_fix_target: {target_file}" in text:
            return tid
    return None


def _load_stream(path: Path, source: str) -> tuple[list[tuple[dict, datetime, str]], int]:
    """Parse one JSONL stream into relevant (row, ts, raw_ts) tuples, row-deduped.

    Only candidate/filed-kind rows are kept (others are irrelevant, not
    malformed). Returns (rows, skipped) where skipped counts unparseable JSON
    lines and relevant rows with a missing/unparseable ts.
    """
    skipped = 0
    rows: list[tuple[dict, datetime, str]] = []
    seen: set[tuple[str, str, str]] = set()
    try:
        # split("\n"), NEVER splitlines(): splitlines() splits on U+2028/U+2029
        # etc. and shreds valid JSONL rows whose note strings carry them
        # (.claude/rules/gotchas.md "splitlines shreds JSONL", #950).
        lines = path.read_text(encoding="utf-8").split("\n")
    except OSError:
        return rows, skipped
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(row, dict):
            skipped += 1
            continue
        kind = _row_kind(row)
        if not (kind.startswith(CANDIDATE_KIND_PREFIX) or kind.startswith(FILED_KIND_PREFIX)):
            continue
        ts_raw = str(row.get("ts") or "")
        ts = parse_ts(ts_raw)
        if ts is None:
            skipped += 1
            continue
        content = str(row.get("note") or "") or json.dumps(row, sort_keys=True)
        dedup_key = (source, ts_raw, hashlib.sha256(content.encode()).hexdigest())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        rows.append((row, ts, ts_raw))
    return rows, skipped


def sweep(
    tasks_root: Path,
    cache_file: Path | None,
    window_days: int = 0,
    include_routed: bool = False,
) -> dict:
    """Enumerate parked, unrouted workflow-fix candidates across all streams."""
    streams: list[tuple[str, Path]] = []
    for events in sorted(tasks_root.glob("*/*/events.jsonl")):
        streams.append((f"task:{events.parent.name}", events))
    if cache_file is not None and cache_file.exists():
        streams.append(("cache", cache_file))

    skipped_rows = 0
    candidates: list[Candidate] = []
    now = datetime.now(UTC)
    cutoff = now - timedelta(days=window_days) if window_days > 0 else None
    bodies: list[tuple[int, str, Path, str, dict]] | None = None  # loaded lazily, ONCE

    for source, path in streams:
        rows, skipped = _load_stream(path, source)
        skipped_rows += skipped
        filed = [(r, ts) for r, ts, _raw in rows if _row_kind(r).startswith(FILED_KIND_PREFIX)]
        for row, ts, ts_raw in rows:
            if not _row_kind(row).startswith(CANDIDATE_KIND_PREFIX):
                continue
            if not _row_is_parked(row):
                continue
            if cutoff is not None and ts < cutoff:
                continue
            target_file, fingerprint, formal_block = _extract_fields(row)
            note = str(row.get("note") or "") or json.dumps(row, sort_keys=True)
            cand = Candidate(
                source=source,
                ts_raw=ts_raw,
                ts=ts,
                note=note,
                target_file=target_file,
                fingerprint=fingerprint,
                park_form=_park_form(row),
                formal_block=formal_block,
            )
            for record, record_ts in filed:
                if record_ts > cand.ts and _filed_record_matches(cand, record):
                    cand.suppressed = True
                    cand.suppressed_by = {"kind": "same-stream-filed", "ref": _filed_ref(record)}
                    break
            needs_bodies = (not cand.suppressed and cand.fingerprint is not None) or bool(
                cand.target_file
            )
            if needs_bodies and bodies is None:
                bodies = _load_task_bodies(tasks_root)
            if not cand.suppressed and cand.fingerprint is not None:
                hit = _fp_tag_scan(bodies or [], cand.fingerprint, cand.ts)
                if hit is not None:
                    cand.suppressed = True
                    cand.suppressed_by = hit
            if cand.target_file:
                cand.open_wf_fix_on_file = _open_wf_fix_on_file(bodies or [], cand.target_file)
            candidates.append(cand)

    candidates.sort(key=lambda c: (c.source, c.ts_raw))
    listed = candidates if include_routed else [c for c in candidates if not c.suppressed]
    return {
        "generated_at": now.isoformat(),
        "window_days": window_days,
        "skipped_rows": skipped_rows,
        "candidates": [c.to_json() for c in listed],
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: print the sweep JSON to stdout; exit 0 always."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--window-days",
        type=int,
        default=0,
        help="0 (default) = unbounded; a positive N scopes an audit to the last N days",
    )
    parser.add_argument(
        "--include-routed",
        action="store_true",
        help="include suppressed (already-routed) candidates in the output",
    )
    parser.add_argument("--tasks-root", type=Path, default=None, help="test override")
    parser.add_argument("--cache-file", type=Path, default=None, help="test override")
    args = parser.parse_args(argv)

    if args.tasks_root is not None:
        tasks_root = args.tasks_root
        if not tasks_root.is_dir():
            raise FileNotFoundError(f"--tasks-root does not exist: {tasks_root}")
    else:
        tasks_root = tasks_dir()

    if args.cache_file is not None:
        cache_file: Path | None = args.cache_file
        if not cache_file.is_file():
            raise FileNotFoundError(f"--cache-file does not exist: {cache_file}")
    else:
        default_cache = repo_root() / ".claude" / "cache" / "workflow-fix-events.jsonl"
        cache_file = default_cache if default_cache.exists() else None

    result = sweep(
        tasks_root,
        cache_file,
        window_days=args.window_days,
        include_routed=args.include_routed,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
