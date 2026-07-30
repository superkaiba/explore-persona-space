"""Read-only enumerator of PARKED workflow-fix candidates (/daily Step C input).

Scans every ``tasks/*/*/events.jsonl`` (ALL statuses — both evidence parks live
on terminal tasks: #1100 completed, #1101 archived) plus the outside-task
fallback stream ``.claude/cache/workflow-fix-events.jsonl`` for
``epm:workflow-fix-candidate`` rows whose note/``routed`` field marks them
PARKED — a leading ``parked``, a ``routed:``/``Routing:`` ``parked`` token, a
bare ``parked: architectural``/``parked: EPM_WORKFLOW_FIX_SESSION``
routing-decision token, a mid-note ``parked <punct> ... recursion guard``
declaration (#1281), or an URGENT fast-path park (#1741; grammar #1681;
incident #1718) — a leading ``URGENT-PARK`` token, or an
``urgency: main-red`` field INSIDE the formal candidate block (the #1681
grammar never required a "parked" token, so pre-#1741 the emitter and this
enumerator disagreed on the park surface and the #1718 park was invisible to
BOTH consumers for ~16 h); casual "parked" mentions — and prose merely
QUOTING ``urgency: main-red`` outside a formal block — do not count (the
recursion-guard escape valve,
``.claude/rules/workflow-fix-on-bug.md`` § Recursion guard), and prints ONE
JSON object to stdout listing the candidates no later routed-record has
closed. The /daily "Parked workflow-fix-candidate routing pass (Step C)"
(``.claude/skills/daily/SKILL.md``) routes each through the three-route
classifier and posts the ``epm:workflow-fix-task-filed`` routed-record this
sweep's suppression rule (1) keys on.

Suppression rules (a candidate is SUPPRESSED == already routed):

1. **Same-task filed record** — a LATER row (aware-UTC ts > candidate ts)
   with kind ``epm:workflow-fix-task-filed*`` in ANY events.jsonl of the same
   SOURCE, matching the candidate. All ``tasks/*/<id>/events.jsonl`` status
   folders of one task id merge into ONE logical stream (source
   ``task:<id>``), so a routed-record posted on ANY folder of a task closes
   the park's copies in EVERY folder — a stale duplicate status folder (the
   #644/#1253 class) otherwise re-enumerates an already-routed park nightly
   (incident #1196/#1274); the cache file stays its own source. The emitted
   ``suppressed_by.kind`` string stays ``same-stream-filed`` for output
   compatibility. Matching: fp-computable candidates match on the
   fingerprint, OR (#1680) on an exact ``origin_candidate_ts`` row-ts CLAIM —
   a record whose ``origin_candidate_ts`` field CONTAINS the candidate's
   exact row ts (aware-UTC equality; the field may list several full-ISO
   timestamps, e.g. the #1630 corrective ``TS1 + TS2 + TS3`` shape) closes
   it, subject to the ``target_file`` prefix-compatibility veto (#1248's
   key), EVEN when the record carries a real, DIFFERING 12-hex fp: row
   identity (source, ts) is FINER-grained than the fingerprint, and a
   differing driver-recomputed fp from abridged origin text is a
   recomputation artifact (incident #1630 — three routed parks re-enumerated
   nightly past their 07-24 records), not a different bug. This deliberately
   supersedes the #1248 "a real differing fp never suppresses" half; the
   workflow-fix-on-bug.md § Dedup FILING grain (same ``target_file`` +
   different fp files its own task) is untouched — a differing-fp record
   with NO matching row-ts claim still never suppresses, and there is NO
   file-only fallback for fp candidates (#622). Accepted residual (widened
   by #1680 from n/a-fp records to ALL ts-claiming records): two same-second
   candidate rows on the SAME file in one TASK are indistinguishable to the
   ts+target_file key, so one ts-claiming record closes both — grouping
   widens this corner from same-file to same-task fork copies, failing
   toward suppression only there; fp-less (prose) parks match on
   ``origin_candidate_ts`` membership (PRIMARY key), falling back to a
   ``target_file`` string match ONLY when the record carries no
   ``origin_candidate_ts`` (legacy/backfill records).
2. **fp-tag scan** (fingerprint computable only) — a ``kind: infra`` task
   whose ``body.md`` carries ``wf-fix-fp:<fp>`` (tag) or ``fingerprint: <fp>``
   (Provenance line). A NON-terminal hit suppresses unconditionally; a
   TERMINAL (completed/archived) hit suppresses only when the task's
   merge/close ts POSTdates the candidate ts — the max ts over ``epm:merged``
   / ``epm:done`` / ``epm:promoted`` rows and the ``epm:status-changed`` row
   whose ``to`` is terminal, with the creation ts (first parseable row) as
   the first check and the fallback when no close-signal row parses — a
   candidate parked before the fix CLOSED was subsumed by it; one parked
   after it closed is a genuine re-raise and stays enumerated. The
   creation-only key missed the created-before-park/merged-after-park window
   (#1599: #1577's park at 10:59:07Z sat 79 s before #1579's merge). The
   emitted ``suppressed_by`` carries ``basis: creation|close``; the ``kind``
   string stays ``fp-tag-closed``.
3. **Row dedup** — identical (source, ts, content-hash) rows collapse to one;
   the dedup set is shared across all status folders of one task id, so
   byte-identical fork copies collapse too (observed verbatim duplication in
   #1100's events.jsonl; cross-folder fork copies in #1196).

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
UTC. A JSONL line that fails to parse, a valid-JSON non-dict line, or a
candidate/filed row whose ts is missing/unparseable, is SKIPPED — counted in
the top-level ``skipped_rows`` int (the TRUE total, kept for output
compatibility) AND described by one structured record in the top-level
``skipped`` list (#1680): ``{source, path, line_no, reason (one of
json-decode-error | non-dict-row | missing-or-unparseable-ts), kind_hint,
relevant_kind}``. ``kind_hint`` is regex-extracted from the raw line for
decode errors, so /daily can tell a malformed line of candidate/filed kind —
a possible lost park (``relevant_kind`` true/null: investigate) — from
benign irrelevant noise (false). The list is capped at ``_SKIPPED_EMIT_CAP``
entries; ``skipped_rows > len(skipped)`` is the truncation signal. Never a
crash, never a silent drop. Exit code is 0 always — this is an enumerator,
not a gate.

Advisory: ``unmatched_record_fps`` (top-level list, #1703) — same-stream
filed-record fingerprints that match NO enumerated candidate fingerprint
on the same stream, one entry per (source, unique unmatched fp) as
``{source, ref, fp}``. This is the DETECTOR for driver
fingerprint-recomputation drift (the #1630 class): a routed-record
carrying a real 12-hex fingerprint that matches no candidate fp is
silent evidence the driver recomputed the fingerprint from
abridged/synthesized text — the ts-claim fallback (#1680) correctly
suppressed the park, so nothing else surfaces it. Advisory ONLY: never
gates, suppresses, or re-enumerates anything; the ts-claim fallback
remains the load-bearing suppression path. /daily Step C flags a
non-empty list for investigation.

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

# Close-signal event kinds for _task_closed_ts (#1599). epm:merged is
# load-bearing: on the motivating incident the terminal status flip PREdated
# the park while the Step 10d merge POSTdated it (#1577 park 10:59:07Z between
# #1579's flip 10:46:50Z and merge 11:00:26Z). epm:promoted mirrors
# task_workflow._WF_FIX_CLOSURE_EVENT_KINDS (never fires on kind: infra).
_CLOSE_EVENT_KINDS = frozenset({"epm:merged", "epm:done", "epm:promoted"})

_PARKED_LEAD_RE = re.compile(r"\s*parked\b", re.IGNORECASE)
_PARKED_ROUTED_RE = re.compile(r"routed:\s*parked\b", re.IGNORECASE)
# Mid-note park DECLARATIONS (#1281): a genuine park announced after other
# prose (the 2026-07-11 miss — #1271's note opens with a root-sync record and
# declares 'Routing: parked — ... recursion guard' only at the end). Three
# grounded shapes:
#   1. the /issue SKILL.md 'Routing: parked ...' breadcrumb (#1271, #1166);
#   2. the bare routing-decision tokens 'parked: architectural' /
#      'parked: EPM_WORKFLOW_FIX_SESSION' (workflow-fix-on-bug.md § "What the
#      orchestrator does" step 4 vocabulary);
#   3. 'parked <decl-punct> ... recursion guard' (the #941/#988/#1233/#710
#      family) — declaration punctuation (em-dash / -- / '- ' / ':') AND the
#      recursion-guard co-mention are the two discriminators that keep casual
#      mentions ('stopped-on-parked-task', 'nothing parked here', 'not parked
#      under the recursion guard') out. The single hyphen is admitted ONLY
#      when followed by whitespace, so compounds ('stopped-on-parked-task')
#      can never match. Window note: the 160-char [^\n] window is same-line,
#      but arm 3's `\s*`/`-\s` can consume newline(s) (e.g. 'parked —\n...
#      recursion guard'), so the declaration is not STRICTLY same-line —
#      accepted: genuine park shapes gain, and the casual negatives carry no
#      declaration punctuation at all.
_PARKED_MIDNOTE_RE = re.compile(
    r"\brouting:\s*parked\b"
    r"|\bparked:\s*(?:architectural|EPM_WORKFLOW_FIX_SESSION)\b"
    r"|\bparked\s*(?:—|--|-\s|:)[^\n]{0,160}\brecursion guard\b",
    re.IGNORECASE,
)
# URGENT fast-path park arms (#1741; grammar #1681; incident #1718). The
# urgent grammar (workflow-fix-on-bug.md § Recursion guard "Urgent fast
# path") prescribes three in-block fields but never required a "parked"
# token, so the #1718 park (leads `URGENT-PARK`, zero "parked" tokens) was
# invisible to every arm above for ~16 h while main stayed red. Arm (a):
# leading `URGENT-PARK` token (used with .match, mirroring _PARKED_LEAD_RE).
# Arm (b): `urgency: main-red` field INSIDE the _BLOCK_RE-extracted formal
# candidate block — searched against the block group ONLY, never the whole
# note, so prose QUOTING the grammar keeps the casual-mention exclusion. A
# mis-tagged already-ROUTED urgent block is closed by suppression rules 1/2
# (demonstrated live by the #1718→#1740 record).
_URGENT_PARK_LEAD_RE = re.compile(r"\s*urgent-park\b", re.IGNORECASE)
_URGENT_BLOCK_FIELD_RE = re.compile(r"^urgency:\s*main-red\b", re.IGNORECASE | re.MULTILINE)
_ARCHITECTURAL_RE = re.compile(r"parked:\s*architectural", re.IGNORECASE)
_BLOCK_RE = re.compile(
    r"<!--\s*workflow-fix-candidate v1\s*-->(.*?)<!--\s*/workflow-fix-candidate\s*-->",
    re.DOTALL,
)
_TARGET_FILE_RE = re.compile(r"target_file:\s*([^\s,;]+)")
_FILED_TASK_RE = re.compile(r"filed_task:\s*(#?\d+)")
# Full-ISO datetime tokens inside an origin_candidate_ts field value. A bare
# date-less time ("15:33:39Z" in #1630 v1's parenthetical) deliberately does
# NOT tokenize; Z-suffix / explicit-offset / naive forms parse via parse_ts.
_ISO_TS_TOKEN_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)
# Best-effort marker-kind hint inside an UNPARSEABLE raw JSONL line (mirrors
# _row_kind's dual-key convention: events rows carry "kind", cache rows
# "marker") so /daily can tell a malformed line of relevant kind from noise.
_KIND_HINT_RE = re.compile(r'"(?:kind|marker)"\s*:\s*"([^"]+)"')
# 12-hex fingerprint pattern (matches wf_fix_fingerprint(...)[:12]).
# Used to extract a filed record's real fp for the unmatched_record_fps
# advisory (#1703). Word-boundary bracketed so a 12-hex prefix of a
# longer sha does NOT false-match.
_FP_HEX_RE = re.compile(r"\b([0-9a-f]{12})\b")
_FILED_FP_NOTE_RE = re.compile(r"(?:fingerprint:\s*|wf-fix-fp:)([0-9a-f]{12})\b")
# Emit cap for the structured `skipped` list; `skipped_rows` keeps the TRUE
# total (skipped_rows > len(skipped) == truncated). Defensive bound for the
# /daily LLM consumer — the live tree carries ~1 skip.
_SKIPPED_EMIT_CAP = 200


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
    """Parked-ness from EITHER surface: the note text or a structured 'routed' field.

    Accept paths: a LEADING 'parked' note; 'routed: parked' anywhere; a
    mid-note park DECLARATION (_PARKED_MIDNOTE_RE — 'Routing: parked', the
    bare 'parked: architectural|EPM_WORKFLOW_FIX_SESSION' tokens, or
    'parked <punct> ... recursion guard'; #1281); an URGENT fast-path park
    (#1741; grammar #1681; incident #1718) — a LEADING 'URGENT-PARK' token,
    or an 'urgency: main-red' field INSIDE the formal candidate block (the
    block group only, never a whole-note scan); or a structured 'routed'
    field containing 'parked' (the fallback stays LAST). Casual mid-note
    mentions — incl. prose quoting 'urgency: main-red' outside a block — do
    not count.
    """
    note = str(row.get("note") or "")
    if (
        _PARKED_LEAD_RE.match(note)
        or _PARKED_ROUTED_RE.search(note)
        or _PARKED_MIDNOTE_RE.search(note)
    ):
        return True
    if _URGENT_PARK_LEAD_RE.match(note):
        return True
    m = _BLOCK_RE.search(note)
    if m and _URGENT_BLOCK_FIELD_RE.search(m.group(1)):
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


def _record_origin_ts(record: dict) -> tuple[bool, list[datetime]]:
    """(field-present, parsed AWARE-UTC ts values) of a record's origin_candidate_ts.

    Note-form: the field VALUE is the segment from each ``origin_candidate_ts:``
    label to the next ``' / '`` field separator (or end of line), and EVERY
    full-ISO datetime token in it parses independently — the #1630 corrective
    record carried ``TS1 + TS2 + TS3`` in one field. Structured rows apply the
    same token extraction to the key's value (``parse_ts`` fallback for a bare
    non-tokenizing form). The two surfaces UNION: a record carrying both a
    note-form field and a structured key contributes both value sets.
    present-with-no-parseable-token stays a NON-match — never a fall-through
    to the legacy target_file key (the #1248 absent-vs-unparseable
    distinction). Known residual: ``finditer`` would also match an
    ``origin_candidate_ts:`` label QUOTED inside a note's free-text tail —
    harmless in practice, since a false claim additionally requires an exact
    full-ISO row-ts match plus target_file compatibility.
    """
    note = str(record.get("note") or "")
    present, values = False, []
    for m in re.finditer(r"origin_candidate_ts:", note):
        present = True
        segment = note[m.end() :]
        cut = segment.find(" / ")
        if cut != -1:
            segment = segment[:cut]
        segment = segment.split("\n", 1)[0]
        for tok in _ISO_TS_TOKEN_RE.findall(segment):
            dt = parse_ts(tok)
            if dt is not None:
                values.append(dt)
    raw = record.get("origin_candidate_ts")
    if raw:
        present = True
        toks = _ISO_TS_TOKEN_RE.findall(str(raw))
        for tok in toks:
            dt = parse_ts(tok)
            if dt is not None:
                values.append(dt)
        if not toks:
            dt = parse_ts(str(raw))
            if dt is not None:
                values.append(dt)
    return present, values


def _record_target_file(record: dict) -> str | None:
    """The record's target_file (structured key first, then note regex), or None."""
    rec_tf = record.get("target_file")
    if rec_tf:
        return str(rec_tf)
    note = str(record.get("note") or "")
    tm = _TARGET_FILE_RE.search(note)
    return tm.group(1).rstrip(".,;:!?") if tm else None


def _filed_record_matches(cand: Candidate, record: dict) -> bool:
    """Does this same-stream filed record close this candidate? (suppression rule 1)."""
    note = str(record.get("note") or "")
    if cand.fingerprint is not None:
        # fp-aware: the candidate's own fingerprint suppresses.
        if cand.fingerprint in note:
            return True
        if cand.fingerprint in str(record.get("fingerprint") or ""):
            return True
        # #1680: an exact origin_candidate_ts claim OVERRIDES a differing
        # recomputed fp — a record naming this candidate's exact row ts is
        # claiming to have routed THIS row; the fp difference is a driver
        # recomputation artifact (abridged origin text, incident #1630), not
        # a different bug. target_file prefix-compatibility veto retained for
        # same-second sibling rows (deliberate partial supersession of the
        # #1248 differing-fp veto; § Dedup is preserved because row identity
        # (source, ts) is FINER-grained than the fingerprint).
        present, origin_ts = _record_origin_ts(record)
        if present and cand.ts in origin_ts:
            rec_tf = _record_target_file(record)
            if not cand.target_file or not rec_tf:
                return True  # veto abstains when either side lacks a target_file; ts decided
            return cand.target_file.startswith(rec_tf) or rec_tf.startswith(cand.target_file)
        # No matching row-ts claim: a differing real fp is a DIFFERENT bug
        # (workflow-fix-on-bug.md § Dedup), and a no-usable-fp record has no
        # key left (no file-only fallback for fp candidates — #1248/#622).
        return False
    # fp-less (prose park): PRIMARY key = origin_candidate_ts membership.
    present, origin_ts = _record_origin_ts(record)
    if present:
        return cand.ts in origin_ts
    # Legacy record with NO origin_candidate_ts: fall back to target_file match.
    if not cand.target_file:
        return False
    return _record_target_file(record) == cand.target_file


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


def _extract_filed_fp(record: dict) -> str | None:
    """One 12-hex fingerprint from a filed record, or None (#1703).

    Structured ``record["fingerprint"]`` key wins over note-embedded
    values. Prose values (``n/a (prose park)``, ``n/a-fp``, empty string)
    never yield a hit — the 12-hex word-boundary regex requires the exact
    canonical shape ``wf_fix_fingerprint(...)[:12]`` produces.

    Advisory use only: consumers must never gate on the return value.
    """
    raw = record.get("fingerprint")
    if isinstance(raw, str):
        m = _FP_HEX_RE.search(raw)
        if m:
            return m.group(1)
    note = str(record.get("note") or "")
    m = _FILED_FP_NOTE_RE.search(note)
    if m:
        return m.group(1)
    return None


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


def _task_closed_ts(task_dir: Path) -> datetime | None:
    """Latest merge/close-signal ts in events.jsonl; None when no such row parses.

    Max ts over rows whose kind is in _CLOSE_EVENT_KINDS plus
    epm:status-changed rows whose structured ``to`` is terminal
    (completed/archived). Max, never "the last row": marker order varies
    (#1577 posted epm:done AFTER epm:merged; #1579 the reverse). Unreadable
    file / no close-signal row / unparseable ts -> None; the caller falls
    back to the creation-ts rule (fail-open toward ENUMERATION, never a
    silent drop).
    """
    events = task_dir / "events.jsonl"
    try:
        # split("\n"), NEVER splitlines(): splitlines() splits on U+2028/U+2029
        # etc. and shreds valid JSONL rows whose note strings carry them
        # (.claude/rules/gotchas.md "splitlines shreds JSONL", #950).
        lines = events.read_text(encoding="utf-8").split("\n")
    except OSError:
        return None
    closed: datetime | None = None
    for line in lines:
        if not line.strip():
            continue
        # Cheap substring prefilter before json.loads (the task_workflow
        # _wf_fix_closed_at pattern; live events files run to hundreds of rows).
        if (
            '"epm:merged"' not in line
            and '"epm:done"' not in line
            and '"epm:promoted"' not in line
            and '"epm:status-changed"' not in line
        ):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        kind = str(row.get("kind") or "")
        if kind not in _CLOSE_EVENT_KINDS and not (
            kind == "epm:status-changed" and row.get("to") in TERMINAL_STATUSES
        ):
            continue
        dt = parse_ts(row.get("ts"))
        if dt is not None and (closed is None or dt > closed):
            closed = dt
    return closed


def _fp_tag_scan(
    bodies: list[tuple[int, str, Path, str, dict]], fp: str, cand_ts: datetime
) -> dict | None:
    """Suppression rule 2: an infra task carrying this fp (tag or Provenance line).

    Non-terminal hit -> suppress unconditionally; terminal hit -> suppress when
    the candidate ts PREdates the task's merge/close time — the creation ts
    (first parseable row; the pre-#1599 rule, checked first) or the close ts
    (_task_closed_ts: max over epm:merged / epm:done / epm:promoted / the
    terminal epm:status-changed). The creation-only key missed the
    created-before-park/merged-after-park window (#1599: #1577's park at
    10:59:07Z sat between #1579's creation and its 11:00:26Z merge). An
    unreadable creation AND close ts fails toward ENUMERATION (treated as a
    re-raise), never toward a silent drop. suppressed_by carries
    basis: "creation" | "close" for auditability; kind stays "fp-tag-closed".
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
            return {"kind": "fp-tag-closed", "ref": f"#{tid}", "basis": "creation"}
        closed = _task_closed_ts(body_path.parent)
        if closed is not None and closed > cand_ts:
            return {"kind": "fp-tag-closed", "ref": f"#{tid}", "basis": "close"}
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


def _skip_record(source: str, path: Path, line_no: int, reason: str, kind_hint: str | None) -> dict:
    """One structured skipped-line record for the top-level ``skipped`` list (#1680).

    ``relevant_kind`` is True when the (hinted) kind is a candidate/filed
    marker — a possible lost park, investigate; False is benign irrelevant
    noise; None (no kind extractable) is unknown — investigate.
    """
    relevant: bool | None = None
    if kind_hint is not None:
        relevant = kind_hint.startswith((CANDIDATE_KIND_PREFIX, FILED_KIND_PREFIX))
    return {
        "source": source,
        "path": str(path),
        "line_no": line_no,
        "reason": reason,
        "kind_hint": kind_hint,
        "relevant_kind": relevant,
    }


def _load_stream(
    path: Path, source: str, seen: set[tuple[str, str, str]] | None = None
) -> tuple[list[tuple[dict, datetime, str]], list[dict]]:
    """Parse one JSONL stream into relevant (row, ts, raw_ts) tuples, row-deduped.

    Only candidate/filed-kind rows are kept (others are irrelevant, not
    malformed). Returns (rows, skips) where skips carries one structured
    ``_skip_record`` dict per skipped line: unparseable JSON, valid-JSON
    non-dict lines, and relevant rows with a missing/unparseable ts.

    ``seen`` is the row-dedup set; pass ONE shared set across all events.jsonl
    paths of a single source (task id) so byte-identical copies in duplicate
    status folders collapse (#1274). None -> fresh per-call set.
    """
    skips: list[dict] = []
    rows: list[tuple[dict, datetime, str]] = []
    if seen is None:
        seen = set()
    try:
        # split("\n"), NEVER splitlines(): splitlines() splits on U+2028/U+2029
        # etc. and shreds valid JSONL rows whose note strings carry them
        # (.claude/rules/gotchas.md "splitlines shreds JSONL", #950).
        lines = path.read_text(encoding="utf-8").split("\n")
    except OSError:
        return rows, skips
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            hint = _KIND_HINT_RE.search(line)
            skips.append(
                _skip_record(
                    source, path, line_no, "json-decode-error", hint.group(1) if hint else None
                )
            )
            continue
        if not isinstance(row, dict):
            skips.append(_skip_record(source, path, line_no, "non-dict-row", None))
            continue
        kind = _row_kind(row)
        if not (kind.startswith(CANDIDATE_KIND_PREFIX) or kind.startswith(FILED_KIND_PREFIX)):
            continue
        ts_raw = str(row.get("ts") or "")
        ts = parse_ts(ts_raw)
        if ts is None:
            skips.append(_skip_record(source, path, line_no, "missing-or-unparseable-ts", kind))
            continue
        content = str(row.get("note") or "") or json.dumps(row, sort_keys=True)
        dedup_key = (source, ts_raw, hashlib.sha256(content.encode()).hexdigest())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        rows.append((row, ts, ts_raw))
    return rows, skips


def sweep(
    tasks_root: Path,
    cache_file: Path | None,
    window_days: int = 0,
    include_routed: bool = False,
) -> dict:
    """Enumerate parked, unrouted workflow-fix candidates across all streams."""
    # One logical stream PER TASK ID: all tasks/*/<id>/events.jsonl status
    # folders merge, so a routed-record posted on ANY folder of a task closes
    # the park's copies in EVERY folder, and byte-identical fork copies
    # row-dedup (#1274; incident #1196: a stale duplicate status folder — the
    # #644/#1253 class — re-enumerated an already-routed park nightly). The
    # cache file stays its own group: filed records never match across
    # task/cache boundaries (the fp-less primary key is bare
    # origin_candidate_ts equality, so a global pool could false-suppress a
    # same-second park on a DIFFERENT task).
    grouped: dict[str, list[Path]] = {}
    for events in sorted(tasks_root.glob("*/*/events.jsonl")):
        grouped.setdefault(f"task:{events.parent.name}", []).append(events)
    streams: list[tuple[str, list[Path]]] = list(grouped.items())
    if cache_file is not None and cache_file.exists():
        streams.append(("cache", [cache_file]))

    skips: list[dict] = []
    candidates: list[Candidate] = []
    unmatched_record_fps: list[dict] = []
    now = datetime.now(UTC)
    cutoff = now - timedelta(days=window_days) if window_days > 0 else None
    bodies: list[tuple[int, str, Path, str, dict]] | None = None  # loaded lazily, ONCE

    for source, paths in streams:
        rows: list[tuple[dict, datetime, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for path in paths:
            path_rows, path_skips = _load_stream(path, source, seen)
            rows.extend(path_rows)
            skips.extend(path_skips)
        filed = [(r, ts) for r, ts, _raw in rows if _row_kind(r).startswith(FILED_KIND_PREFIX)]
        stream_candidate_start = len(candidates)
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

        # #1703 unmatched_record_fps advisory: enumerated candidate fps FOR
        # THIS STREAM (fp-computable candidates only; fp-less prose parks
        # contribute nothing to the enumerated set). Iterate filed records
        # and record any real 12-hex fp that matches no enumerated candidate
        # fp on this stream. This is the driver fingerprint-recomputation
        # drift detector (#1630 class): the ts-claim fallback (#1680)
        # correctly suppresses the park, but a recomputed fp is silent
        # evidence the driver operated on abridged/synthesized origin text.
        stream_enumerated_fps = {
            c.fingerprint for c in candidates[stream_candidate_start:] if c.fingerprint is not None
        }
        # Track fps we've already emitted for THIS stream so a record-fp that
        # appears in multiple filed records emits ONCE per stream (advisory
        # dedup — the /daily consumer wants one investigation entry per
        # drift, not one per repeated routed-record).
        seen_unmatched: set[str] = set()
        for record, _record_ts in filed:
            rec_fp = _extract_filed_fp(record)
            if rec_fp is None:
                continue  # prose-park record / no extractable fp — nothing to detect
            if rec_fp in stream_enumerated_fps:
                continue  # matches an enumerated candidate on this stream — normal case
            if rec_fp in seen_unmatched:
                continue  # already listed this drift for the stream — dedup within stream
            seen_unmatched.add(rec_fp)
            unmatched_record_fps.append(
                {
                    "source": source,
                    "ref": _filed_ref(record),
                    "fp": rec_fp,
                }
            )

    candidates.sort(key=lambda c: (c.source, c.ts_raw))
    listed = candidates if include_routed else [c for c in candidates if not c.suppressed]
    return {
        "generated_at": now.isoformat(),
        "window_days": window_days,
        "skipped_rows": len(skips),  # KEPT: the TRUE total, output-compat (#1274 precedent)
        "skipped": skips[:_SKIPPED_EMIT_CAP],  # NEW (#1680): additive structured records
        "unmatched_record_fps": unmatched_record_fps,  # NEW (#1703): additive advisory
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
