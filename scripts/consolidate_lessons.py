#!/usr/bin/env python3
"""Deterministic failure-lesson consolidation (extracted from /daily — task #711).

The failure-lesson consolidation pipeline is the cross-issue janitorial pass that
operates on top of the per-lesson ``/issue``-time writes (the Step-7 crash-fix
hook persists one ``epm:failure-lesson v1`` marker per resolved ``epm:failure``,
and on ``generalizes: yes`` writes a ``.claude/agent-memory/<owning_agent>/``
entry). This used to live ONLY inside the nightly ``/daily`` LLM run (a flaky
44K-token ``claude -p /daily``). This script pulls the DETERMINISTIC parts out of
that run into a pure-Python cron so the consolidation no longer depends on the
LLM run completing.

Three deterministic operations over a rolling window of ``epm:failure-lesson v1``
markers:

1. **dedupe** — collapse two same-window lessons that normalize to the same
   concept (``SequenceMatcher`` ratio ``>= T``); when BOTH are lesson-derived
   memory entries, the canonical one survives and the duplicate sibling's
   ``feedback_*.md`` + its ``MEMORY.md`` index bullet are removed.
2. **promote-recurring** — a lesson recurring across ``>= K`` distinct tasks in
   the window (same ``failure_class`` + ``phase``, pairwise lesson similarity
   ``>= T``) gains a bullet under ``.claude/rules/gotchas.md`` (idempotent: a
   ``>= T``-similar bullet already present is a no-op).
3. **prune over-eager** — a ``generalizes: yes`` lesson-derived memory entry whose
   source markers are ALL on terminal-status tasks AND that never recurred in the
   window is removed (the ``feedback_*.md`` + its ``MEMORY.md`` index bullet).

Idempotency: a second run with no new markers writes NO new commit and NO new log
row beyond a ``no-op`` summary.

Budget guard (task #2189): ``promote`` REFUSES an append whose projected
gotchas.md size would cross ``workflow_lint.GOTCHAS_SIZE_WARN_BYTES`` (lazily
imported — never restated). The refusal is all-or-nothing and does NOT raise:
no write, no touched-path (hence no gotchas commit), ``promote_refused_budget``
counted, the refused bullets printed verbatim, and ``consolidate`` proceeds so
dedupe/prune mutations still commit and the counts line is still written.
``main`` then returns exit code **3** (distinct from 0 = clean and 1 = generic
crash) so the refusal is a loud process-level signal.

Fail-loud (CLAUDE.md "Fail fast — never hide failures"): the script hard-RAISEs
on the conditions where the recovery path itself would corrupt the repo — a
parsed lesson with a missing/empty ``owning_agent`` (no target dir; a guessed
write would mutate the wrong agent's memory), a ``git commit`` failure on an
``--apply`` run (half-landed mutation), or an unreadable positively-referenced
target file. The ONE documented tolerated skip is the tier-3 unparseable note
(a marker-writer-crash artifact, the same class ``_iter_jsonl`` already swallows
for a truncated trailing JSONL line): it is counted into ``unparseable_skipped``
and WARNed, never silently dropped.

NOTE: ``tasks/_orphaned_markers/`` is NOT considered — those are markers that
failed to post to a task and are out of scope for cross-task consolidation.

``gotcha_candidate: yes`` lessons are DEFERRED to the inline ``/issue``-time
workflow-fix route (which already fired per-lesson); the cron NEVER emits a
``<!-- workflow-fix-candidate v1 -->`` block (it has no orchestrator turn to
consume one). It records them under ``gotcha_candidate_seen`` for visibility and
promotes ONLY ``gotcha_candidate: no`` recurring lessons into gotchas.md.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Reuse the tolerant JSONL parser + the repo-root resolver from the task-workflow
# library; do NOT reinvent them. ``_iter_jsonl`` is pure path-based (no repo-root
# dependency), so it works against an injected ``tmp_path`` mini-repo. ``repo_root``
# is the DEFAULT root only — every operation accepts an explicit ``root`` override
# (the cron uses the default; tests inject a tmp mini-repo), so registry-load and
# git-commit are done against the injected root rather than via the library's
# module-dir-anchored resolvers.
from explore_persona_space.task_workflow import _iter_jsonl, repo_root

_log = logging.getLogger("consolidate_lessons")

# ─── Tunables (logic-hyperparameters; see plan §11) ─────────────────────────

T_DEDUPE = 0.85  # SequenceMatcher ratio threshold (ungrounded — needs smoke-test)
K_RECUR = 2  # distinct tasks for a lesson to "recur"

# ─── Marker parsing ─────────────────────────────────────────────────────────

LESSON_BLOCK_RE = re.compile(
    r"<!--\s*epm:failure-lesson v1\s*-->(.*?)<!--\s*/epm:failure-lesson\s*-->", re.S
)
# Field-line regex (line-anchored, MULTILINE). The known keys; anything else is
# treated as lesson-body continuation.
_FIELD_KEYS = (
    "failure_class",
    "phase",
    "lesson",
    "generalizes",
    "owning_agent",
    "gotcha_candidate",
)
BARE_FIELD_RE = re.compile(
    r"^\s*(?P<key>" + "|".join(_FIELD_KEYS) + r"):\s*(?P<val>.*?)\s*$",
    re.M,
)
# A line that opens a recognized ``key:`` field (used to terminate a multi-line
# ``lesson:`` value collection).
_KEY_LINE_RE = re.compile(r"^\s*(?:" + "|".join(_FIELD_KEYS) + r"):")

MARKER_KIND = "epm:failure-lesson"
TERMINAL_STATUSES = frozenset({"completed", "archived"})


@dataclass
class Lesson:
    """One parsed ``epm:failure-lesson v1`` marker."""

    task_id: int
    ts: str
    failure_class: str
    phase: str
    lesson: str
    generalizes: str  # "yes" | "no" | "" (raw)
    owning_agent: str
    gotcha_candidate: str  # "yes" | "no" | "" (raw)


@dataclass
class Skip:
    """A kind-matched marker that yielded no parseable lesson (tier-3 skip)."""

    task_id: int
    ts: str
    reason: str


def _parse_fields_from_lines(text: str) -> dict[str, str]:
    """Parse ``key: value`` field lines from ``text``, collecting a multi-line
    ``lesson:`` value until the next recognized ``key:`` line or EOF.

    Shared by tier 1 (the block body) and tier 2 (the bare note) so both yield
    identical field dicts on identical content.
    """
    fields: dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = BARE_FIELD_RE.match(line)
        if not m:
            i += 1
            continue
        key = m.group("key")
        val = m.group("val")
        if key == "lesson":
            # Collect continuation lines until the next recognized key: line / EOF.
            collected = [val]
            j = i + 1
            while j < len(lines) and not _KEY_LINE_RE.match(lines[j]):
                collected.append(lines[j])
                j += 1
            fields["lesson"] = "\n".join(collected).strip()
            i = j
        else:
            fields[key] = val
            i += 1
    return fields


def _parse_lesson_note(note: str) -> dict[str, str] | None:
    """Three-tier parser: sentinel block → bare-fields → skip (return None).

    - Tier 1: the ``<!-- epm:failure-lesson v1 -->`` … ``<!-- /... -->`` block.
    - Tier 2: bare ``key: value`` field lines (no wrapper) carrying at least the
      required ``failure_class`` + ``lesson``.
    - Tier 3: neither yields the required fields → return ``None`` (caller WARNs
      and counts into ``unparseable_skipped``).

    A note that satisfies BOTH always takes tier 1.
    """
    if not note:
        return None
    block = LESSON_BLOCK_RE.search(note)
    if block:
        fields = _parse_fields_from_lines(block.group(1))
        if fields.get("failure_class") and fields.get("lesson"):
            return fields
        # A block that is present but missing the required fields falls through
        # to tier 2 (which scans the whole note) and then tier 3.
    fields = _parse_fields_from_lines(note)
    if fields.get("failure_class") and fields.get("lesson"):
        return fields
    return None


# ─── Text normalization + similarity ────────────────────────────────────────

_SENTINEL_LINE_RE = re.compile(r"<!--.*?-->", re.S)
_CLASS_PHASE_LINE_RE = re.compile(
    r"^\s*(?:failure_class|phase|generalizes|owning_agent|gotcha_candidate):.*$", re.M
)
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase, strip ``<!-- -->`` sentinels + the structured field lines, and
    collapse whitespace — so only the lesson prose is compared.
    """
    text = _SENTINEL_LINE_RE.sub(" ", text)
    text = _CLASS_PHASE_LINE_RE.sub(" ", text)
    text = re.sub(r"^\s*lesson:\s*", " ", text, flags=re.M)
    return _WS_RE.sub(" ", text).strip().lower()


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _containment_ratio(needle: str, haystack: str) -> float:
    """Fraction of the normalized ``needle`` covered by matching blocks in the
    normalized ``haystack`` (an ASYMMETRIC, containment-aware similarity).

    Used to decide whether a feedback memory file is "the entry for" a lesson:
    the per-lesson ``/issue``-time write derives the file BODY from the lesson,
    so the lesson is a near-substring of the (longer) body. A symmetric
    ``SequenceMatcher.ratio`` undershoots there because the body carries extra
    prose around the lesson; this measures how much of the lesson the body
    actually contains. Returns 0.0 for an empty needle.
    """
    n = _normalize(needle)
    h = _normalize(haystack)
    if not n:
        return 0.0
    matcher = difflib.SequenceMatcher(None, n, h)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(n)


# ─── Agent-memory + gotchas.md file IO ──────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.S)


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Return (frontmatter_block, body). Empty frontmatter if absent."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return "", text
    return m.group(1), m.group(2)


def _frontmatter_type(frontmatter: str) -> str:
    """Return the ``type``/``metadata.type`` value, tolerating both shapes.

    The global CLAUDE.md template uses a top-level ``type:``; the
    experiment-implementer variant nests it under ``metadata:``. Both yield the
    same value here.
    """
    for line in frontmatter.splitlines():
        m = re.match(r"^\s*type:\s*(\S+)\s*$", line)
        if m:
            return m.group(1)
    return ""


def _is_lesson_derived(path: Path) -> bool:
    """True iff the memory file is a lesson-derived ``feedback`` entry (never a
    hand-authored memory). Fail toward NOT-lesson-derived (keep) on any doubt.
    """
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:  # unreadable positively-referenced target → fail loud
        raise RuntimeError(f"unreadable memory file {path}: {e}") from e
    frontmatter, _ = _split_frontmatter(text)
    return _frontmatter_type(frontmatter) == "feedback"


def _memory_body(path: Path) -> str:
    """Return the body (post-frontmatter) of a memory file."""
    text = path.read_text(encoding="utf-8")
    _, body = _split_frontmatter(text)
    return body


def _agent_memory_dir(root: Path, owning_agent: str) -> Path:
    return root / ".claude" / "agent-memory" / owning_agent


def _remove_memory_index_line(memory_md: Path, feedback_filename: str) -> bool:
    """Remove the ``MEMORY.md`` index bullet that links ``feedback_filename``.

    Returns True iff a line was removed. The bullet shape is
    ``- [Title](feedback_filename) — hook``; match on the ``(filename)`` link
    target. Preserves all other lines and their order.
    """
    if not memory_md.exists():
        return False
    lines = memory_md.read_text(encoding="utf-8").splitlines(keepends=True)
    target = f"({feedback_filename})"
    kept = [ln for ln in lines if target not in ln]
    if len(kept) == len(lines):
        return False
    memory_md.write_text("".join(kept), encoding="utf-8")
    return True


# ─── Counts / summary ───────────────────────────────────────────────────────


@dataclass
class Counts:
    markers_seen: int = 0
    parsed_total: int = 0
    deduped: int = 0
    promoted: int = 0
    promote_noop: int = 0
    # Bullets refused by the gotchas.md byte-budget guard (task #2189).
    # Deliberately NOT a term in ``is_noop`` — a refused run mutated nothing.
    promote_refused_budget: int = 0
    pruned: int = 0
    gotcha_candidate_seen: int = 0
    unparseable_skipped: int = 0
    touched_paths: list[Path] = field(default_factory=list)
    skipped: list[Skip] = field(default_factory=list)
    # Memory files the dedupe pass already handled (survivors + removed
    # duplicates), so the prune pass does NOT re-prune a deduped concept's
    # canonical survivor as a spurious "over-eager one-off".
    dedupe_handled: set[Path] = field(default_factory=set)

    @property
    def is_noop(self) -> bool:
        return self.deduped == 0 and self.promoted == 0 and self.pruned == 0

    def add_touched(self, path: Path) -> None:
        if path not in self.touched_paths:
            self.touched_paths.append(path)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "markers_seen": self.markers_seen,
            "parsed_total": self.parsed_total,
            "deduped": self.deduped,
            "promoted": self.promoted,
            "promote_noop": self.promote_noop,
            "promote_refused_budget": self.promote_refused_budget,
            "pruned": self.pruned,
            "gotcha_candidate_seen": self.gotcha_candidate_seen,
            "unparseable_skipped": self.unparseable_skipped,
            "no_op": self.is_noop,
            "touched_paths": [str(p) for p in self.touched_paths],
            "skipped": [
                {"task_id": s.task_id, "ts": s.ts, "reason": s.reason} for s in self.skipped
            ],
        }


# ─── Window scan ────────────────────────────────────────────────────────────


def _load_registry(root: Path) -> dict[str, Any]:
    """Load ``tasks/REGISTRY.json`` from the given root (NOT the module-dir
    resolver, so a ``tmp_path`` mini-repo works)."""
    rp = root / "tasks" / "REGISTRY.json"
    if not rp.exists():
        return {"highest_id": 0, "tasks": {}}
    return json.loads(rp.read_text(encoding="utf-8"))


def _parse_ts(ts: str) -> datetime | None:
    """Parse an ISO-8601 ``...Z`` timestamp into an aware UTC datetime."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def scan_window(
    root: Path, window_days: int, *, now: datetime | None = None
) -> tuple[list[Lesson], list[Skip]]:
    """Scan all tasks for ``epm:failure-lesson v1`` markers within the window.

    Returns (lessons, skips). A kind-matched marker that does not parse is
    recorded as a Skip (tier-3) and WARNed, never raised.
    """
    now = now or datetime.now(tz=UTC)
    cutoff = now - timedelta(days=window_days)
    registry = _load_registry(root)
    lessons: list[Lesson] = []
    skips: list[Skip] = []
    for task_id_str, entry in sorted(registry.get("tasks", {}).items(), key=lambda kv: int(kv[0])):
        try:
            task_id = int(task_id_str)
        except ValueError:
            continue
        events_path = root / entry["path"] / "events.jsonl"
        for row in _iter_jsonl(events_path):
            if row.get("kind") != MARKER_KIND:
                continue
            ts = row.get("ts", "")
            parsed_ts = _parse_ts(ts)
            if parsed_ts is None or parsed_ts < cutoff:
                continue
            note = row.get("note", "") or ""
            fields = _parse_lesson_note(note)
            if fields is None:
                reason = "no sentinel block and no required bare fields"
                _log.warning(
                    "unparseable failure-lesson note in task #%d @ %s: %s",
                    task_id,
                    ts,
                    reason,
                )
                skips.append(Skip(task_id=task_id, ts=ts, reason=reason))
                continue
            lessons.append(
                Lesson(
                    task_id=task_id,
                    ts=ts,
                    failure_class=fields.get("failure_class", ""),
                    phase=fields.get("phase", ""),
                    lesson=fields.get("lesson", ""),
                    generalizes=fields.get("generalizes", "").lower(),
                    owning_agent=fields.get("owning_agent", ""),
                    gotcha_candidate=fields.get("gotcha_candidate", "").lower(),
                )
            )
    return lessons, skips


def _task_status(root: Path, task_id: int) -> str | None:
    """Return the task's current status from the registry, or None if unknown."""
    registry = _load_registry(root)
    entry = registry.get("tasks", {}).get(str(task_id))
    if not entry:
        return None
    return entry.get("status")


# ─── Operations ─────────────────────────────────────────────────────────────


def _require_owning_agent(lesson: Lesson) -> None:
    """Hard-RAISE when a parsed, generalizing lesson has no owning_agent — the
    corrupting-write guard (a guessed write would mutate the wrong agent)."""
    if lesson.generalizes == "yes" and not lesson.owning_agent.strip():
        raise RuntimeError(
            f"failure-lesson on task #{lesson.task_id} @ {lesson.ts} has "
            f"generalizes: yes but no owning_agent — refusing to write to a "
            f"guessed agent-memory dir (corrupting-write guard)"
        )


def dedupe(root: Path, lessons: list[Lesson], counts: Counts, *, apply: bool) -> None:
    """Collapse mutually-similar same-window lessons that BOTH already have
    lesson-derived memory entries: the canonical entry survives, the duplicate
    sibling's ``feedback_*.md`` + ``MEMORY.md`` index bullet are removed.
    """
    generalizing = [le for le in lessons if le.generalizes == "yes"]
    for le in generalizing:
        _require_owning_agent(le)
    # Pairwise similarity within the SAME owning_agent.
    removed: set[tuple[str, str]] = set()  # (owning_agent, feedback_filename)
    for i in range(len(generalizing)):
        for j in range(i + 1, len(generalizing)):
            a, b = generalizing[i], generalizing[j]
            if a.owning_agent != b.owning_agent:
                continue
            if _ratio(a.lesson, b.lesson) < T_DEDUPE:
                continue
            # Canonical = lower task_id, then lexically-first matched slug.
            mem_dir = _agent_memory_dir(root, a.owning_agent)
            a_file = _match_memory_file(mem_dir, a)
            b_file = _match_memory_file(mem_dir, b)
            if a_file is None or b_file is None:
                # Not both lesson-derived memory entries — nothing to merge.
                continue
            # Pick survivor / duplicate.
            if (a.task_id, a_file.name) <= (b.task_id, b_file.name):
                survivor, dup_lesson, dup_file = a_file, b, b_file
            else:
                survivor, dup_lesson, dup_file = b_file, a, a_file
            if survivor == dup_file:
                continue
            # Both files are now dedupe-handled: the survivor is the consolidated
            # keeper and the duplicate is removed — neither may be re-pruned by
            # the prune pass as a spurious "over-eager one-off".
            counts.dedupe_handled.add(survivor)
            counts.dedupe_handled.add(dup_file)
            key = (dup_lesson.owning_agent, dup_file.name)
            if key in removed:
                continue
            removed.add(key)
            counts.deduped += 1
            mem_md = mem_dir / "MEMORY.md"
            if apply:
                if dup_file.exists():
                    dup_file.unlink()
                _remove_memory_index_line(mem_md, dup_file.name)
            counts.add_touched(dup_file)
            counts.add_touched(mem_md)


def _match_memory_file(mem_dir: Path, lesson: Lesson) -> Path | None:
    """Return the lesson-derived ``feedback_*.md`` in ``mem_dir`` whose body
    CONTAINS the lesson at ``>= T`` containment, or None.

    Uses the asymmetric ``_containment_ratio`` (how much of the lesson the body
    carries), NOT the symmetric ``_ratio``: the per-lesson ``/issue``-time write
    derives the file body from the lesson plus surrounding prose, so the lesson
    is a near-substring of the longer body and a symmetric ratio undershoots.
    """
    if not mem_dir.is_dir():
        return None
    best: Path | None = None
    best_ratio = T_DEDUPE
    for f in sorted(mem_dir.glob("feedback_*.md")):
        if not _is_lesson_derived(f):
            continue
        r = _containment_ratio(lesson.lesson, _memory_body(f))
        if r >= best_ratio:
            best_ratio = r
            best = f
    return best


def _gotcha_bullet(cluster: list[Lesson]) -> str:
    """Build the one-line lead-bolded gotcha bullet for a recurring cluster.

    Shape: ``- **<phase trap, <=80 chars>** — <lesson first 1-2 sentences>
    (#<task>, #<task>)``.
    """
    rep = cluster[0]
    trap = rep.phase.strip()[:80] or rep.failure_class.strip()[:80]
    # First 1-2 sentences of the lesson body.
    sentences = re.split(r"(?<=[.!?])\s+", rep.lesson.strip())
    detail = " ".join(sentences[:2]).strip()
    task_ids = sorted({le.task_id for le in cluster})
    refs = ", ".join(f"#{t}" for t in task_ids)
    return f"- **{trap}** — {detail} ({refs})"


def _gotchas_path(root: Path) -> Path:
    return root / ".claude" / "rules" / "gotchas.md"


def _existing_gotcha_bullets(text: str) -> list[str]:
    """Return the dash-bullet bodies under ``# Gotchas``."""
    out: list[str] = []
    in_section = False
    for line in text.splitlines():
        if line.strip() == "# Gotchas":
            in_section = True
            continue
        if in_section and line.startswith("- "):
            out.append(line)
    return out


def promote(root: Path, lessons: list[Lesson], counts: Counts, *, apply: bool) -> None:
    """Promote recurring ``gotcha_candidate: no`` clusters into gotchas.md.

    A cluster recurs when ``>= K`` DISTINCT task_ids carry a lesson with the same
    ``(phase, failure_class)`` and pairwise lesson similarity ``>= T``. Idempotent:
    a ``>= T``-similar bullet already present is a no-op (``promote_noop``).

    Byte-budget guard (task #2189): if the projected post-append size crosses
    ``workflow_lint.GOTCHAS_SIZE_WARN_BYTES`` (strictly-greater, parity with
    ``check_gotchas_size``), the WHOLE append is refused WITHOUT raising —
    no write, no touched-path, ``counts.promote_refused_budget`` set, refused
    bullets printed verbatim — and the function returns normally so
    ``consolidate`` still commits dedupe/prune mutations and writes the counts
    line. Refusal is all-or-nothing: a partial fill would park the file at
    exactly the wall with an arbitrary promoted subset and still need the
    human trim.
    """
    # gotcha_candidate: yes are deferred to the inline /issue-time route.
    candidates = [le for le in lessons if le.gotcha_candidate == "yes"]
    counts.gotcha_candidate_seen += len(candidates)

    promotable = [le for le in lessons if le.gotcha_candidate != "yes"]
    clusters = _recurring_clusters(promotable)
    if not clusters:
        return
    gotchas = _gotchas_path(root)
    if not gotchas.exists():
        raise RuntimeError(f"gotchas.md not found at {gotchas} — cannot promote")
    text = gotchas.read_text(encoding="utf-8")
    existing = _existing_gotcha_bullets(text)
    appended: list[str] = []
    for cluster in clusters:
        bullet = _gotcha_bullet(cluster)
        if any(_ratio(bullet, ex) >= T_DEDUPE for ex in existing + appended):
            counts.promote_noop += 1
            continue
        appended.append(bullet)
    if not appended:
        return

    new_text = text.rstrip("\n") + "\n" + "\n".join(appended) + "\n"
    projected = len(new_text.encode("utf-8"))
    # Lazy import, immediately before the size check (task #2189): a module-top
    # import would couple the whole nightly janitor (dedupe/prune/counts) to the
    # import health of the ~13.7k-line lint module, and the cron wrapper's
    # unconditional ``exit 0`` would swallow that breakage indefinitely. The
    # constant is imported, never restated (#838 fixture-inversion).
    _scripts_dir = Path(__file__).resolve().parent
    if str(_scripts_dir) not in sys.path:
        sys.path.insert(0, str(_scripts_dir))
    from workflow_lint import GOTCHAS_SIZE_WARN_BYTES

    if projected > GOTCHAS_SIZE_WARN_BYTES:
        counts.promote_refused_budget = len(appended)
        _log.error(
            "promote REFUSED (gotchas.md byte budget): projected %d B exceeds "
            "GOTCHAS_SIZE_WARN_BYTES=%d B by %d B for %d bullet(s). No write, no "
            "commit — an unattended over-budget append would turn "
            "test_live_tree_passes_clean red fleet-wide. Re-trim gotchas.md per "
            "check_gotchas_size's recipe (keep the operative rule + diagnostic "
            "signature + fix + #N citations; relocate or compress archaeology), "
            "then the next pass promotes normally. Refused bullets follow "
            "verbatim on stdout.",
            projected,
            GOTCHAS_SIZE_WARN_BYTES,
            projected - GOTCHAS_SIZE_WARN_BYTES,
            len(appended),
        )
        for bullet in appended:
            print(bullet)
        return

    counts.promoted += len(appended)
    if apply:
        gotchas.write_text(new_text, encoding="utf-8")
    counts.add_touched(gotchas)


def _recurring_clusters(lessons: list[Lesson]) -> list[list[Lesson]]:
    """Group lessons into recurring clusters (>= K distinct tasks, same
    (phase, failure_class), pairwise similarity >= T). Returns one cluster per
    promote-eligible group.
    """
    # Bucket by (failure_class, phase) first — a cheap exact-key pre-group.
    buckets: dict[tuple[str, str], list[Lesson]] = {}
    for le in lessons:
        buckets.setdefault((le.failure_class.strip(), le.phase.strip()), []).append(le)
    clusters: list[list[Lesson]] = []
    for group in buckets.values():
        # Within a bucket, agglomerate by pairwise lesson similarity.
        used = [False] * len(group)
        for i in range(len(group)):
            if used[i]:
                continue
            cluster = [group[i]]
            used[i] = True
            for j in range(i + 1, len(group)):
                if used[j]:
                    continue
                if _ratio(group[i].lesson, group[j].lesson) >= T_DEDUPE:
                    cluster.append(group[j])
                    used[j] = True
            distinct_tasks = {le.task_id for le in cluster}
            if len(distinct_tasks) >= K_RECUR:
                clusters.append(cluster)
    return clusters


def prune(root: Path, lessons: list[Lesson], counts: Counts, *, apply: bool) -> None:
    """Remove over-eager ``generalizes: yes`` lesson-derived memory entries.

    Over-eager when ALL hold: every source marker for the entry sits on a
    terminal-status task; the ``(phase, failure_class)`` never recurred elsewhere
    in the window; and the entry is lesson-derived. Conservative fail-toward-keep
    when any condition is unprovable.
    """
    generalizing = [le for le in lessons if le.generalizes == "yes"]
    for le in generalizing:
        _require_owning_agent(le)
    # Map (phase, failure_class) -> set of distinct task_ids in window.
    recurrence: dict[tuple[str, str], set[int]] = {}
    for le in lessons:
        recurrence.setdefault((le.phase.strip(), le.failure_class.strip()), set()).add(le.task_id)

    pruned_files: set[Path] = set()
    for le in generalizing:
        mem_dir = _agent_memory_dir(root, le.owning_agent)
        target = _match_memory_file(mem_dir, le)
        if target is None or target in pruned_files:
            continue  # no lesson-derived entry maps to it → keep
        if target in counts.dedupe_handled:
            continue  # dedupe already collapsed this concept → never re-prune it
        # Condition 1: all source markers on terminal-status tasks.
        sources = [
            o
            for o in generalizing
            if o.owning_agent == le.owning_agent and _ratio(o.lesson, le.lesson) >= T_DEDUPE
        ]
        statuses = [_task_status(root, s.task_id) for s in sources]
        if any(st is None for st in statuses):
            continue  # unprovable → keep
        if not all(st in TERMINAL_STATUSES for st in statuses):
            continue  # at least one source still active → keep
        # Condition 2: no recurrence elsewhere in window. Recurrence is EITHER
        # the (phase, failure_class) key appearing on >1 distinct task OR the
        # lesson CONCEPT (by similarity) appearing on >=2 distinct tasks — the
        # latter is load-bearing for idempotency: after dedupe collapses a
        # 2-task concept to one survivor, the survivor's lesson still matches
        # both task markers, so this keeps it from being pruned on the next run.
        key = (le.phase.strip(), le.failure_class.strip())
        if len(recurrence.get(key, set())) > 1:
            continue  # recurred (same phase+class) → keep
        if len({s.task_id for s in sources}) >= K_RECUR:
            continue  # recurred (same concept across tasks) → keep
        # Condition 3: lesson-derived (checked by _match_memory_file already).
        pruned_files.add(target)
        counts.pruned += 1
        mem_md = mem_dir / "MEMORY.md"
        if apply:
            if target.exists():
                target.unlink()
            _remove_memory_index_line(mem_md, target.name)
        counts.add_touched(target)
        counts.add_touched(mem_md)


# ─── Git commit (against the injected root) ─────────────────────────────────


def _git_commit(root: Path, paths: list[Path], message: str) -> None:
    """Stage the given paths and commit, against the INJECTED root.

    Mirrors ``task_workflow._git_commit`` but resolves against ``root`` (not the
    module-dir resolver), so a ``tmp_path`` mini-repo commits to itself. Tolerates
    paths staged-for-deletion (they no longer exist on disk). Hard-RAISE on a git
    failure (a half-landed mutation must not be swallowed).
    """
    rel_paths = [str(p.relative_to(root)) for p in paths]
    existing = [rp for rp, p in zip(rel_paths, paths, strict=True) if p.exists()]
    try:
        if existing:
            subprocess.run(["git", "-C", str(root), "add", "--", *existing], check=True)
        # Stage deletions explicitly so commit --only captures them.
        deleted = [rp for rp, p in zip(rel_paths, paths, strict=True) if not p.exists()]
        if deleted:
            subprocess.run(
                ["git", "-C", str(root), "rm", "--quiet", "--ignore-unmatch", "--", *deleted],
                check=True,
            )
        # Skip the commit if nothing changed for OUR paths (idempotent re-runs).
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "--cached", "--quiet", "--", *rel_paths],
            check=False,
        )
        if diff.returncode == 0:
            return
        subprocess.run(
            ["git", "-C", str(root), "commit", "-m", message, "--only", "--", *rel_paths],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"git commit failed for {rel_paths}: {e}") from e


# ─── Logging the counts line ────────────────────────────────────────────────


def _log_dir(root: Path) -> Path:
    return root / "logs" / "lesson_consolidate"


def _write_log_line(root: Path, counts: Counts, *, now: datetime | None = None) -> Path:
    """Append a single day-stamped counts line to
    ``logs/lesson_consolidate/<date>.log``. ``logs/`` is gitignored.
    """
    now = now or datetime.now(tz=UTC)
    log_dir = _log_dir(root)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{now.strftime('%Y-%m-%d')}.log"
    status = "no-op" if counts.is_noop else "ran"
    line = (
        f"{now.strftime('%Y-%m-%dT%H:%M:%SZ')} {status} "
        f"markers_seen={counts.markers_seen} parsed_total={counts.parsed_total} "
        f"deduped={counts.deduped} promoted={counts.promoted} "
        f"promote_noop={counts.promote_noop} "
        f"promote_refused_budget={counts.promote_refused_budget} "
        f"pruned={counts.pruned} "
        f"gotcha_candidate_seen={counts.gotcha_candidate_seen} "
        f"unparseable_skipped={counts.unparseable_skipped}\n"
    )
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(line)
    return log_file


# ─── Top-level run ──────────────────────────────────────────────────────────


def consolidate(
    root: Path,
    window_days: int,
    *,
    apply: bool,
    now: datetime | None = None,
) -> Counts:
    """Run the full consolidation pass against ``root``. Returns the Counts.

    On ``apply`` and a non-no-op, commits the touched paths by explicit path in
    ONE commit. Always writes the day-stamped log line.
    """
    lessons, skips = scan_window(root, window_days, now=now)
    counts = Counts()
    counts.markers_seen = len(lessons) + len(skips)
    counts.parsed_total = len(lessons)
    counts.unparseable_skipped = len(skips)
    counts.skipped = skips

    dedupe(root, lessons, counts, apply=apply)
    promote(root, lessons, counts, apply=apply)
    prune(root, lessons, counts, apply=apply)

    if apply and not counts.is_noop and counts.touched_paths:
        msg = (
            f"lesson-consolidate: {datetime.now(tz=UTC).strftime('%Y-%m-%d')} "
            f"— promoted {counts.promoted}, deduped {counts.deduped}, pruned {counts.pruned}"
        )
        _git_commit(root, list(counts.touched_paths), msg)

    _write_log_line(root, counts, now=now)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-days", type=int, default=7, help="rolling window for marker scan + recurrence"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply", action="store_true", help="perform + commit mutations (the cron default)"
    )
    mode.add_argument(
        "--dry-run", action="store_true", help="no writes/commits; print what WOULD change"
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable summary dict to stdout"
    )
    parser.add_argument(
        "--log-counts",
        action="store_true",
        default=True,
        help="always log the counts line (effectively always on)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="repo root (default: resolve via task_workflow.repo_root())",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Fail-safe: never mutate unless --apply is explicit. Neither flag → dry-run.
    apply = bool(args.apply)
    root = args.root if args.root is not None else repo_root()

    counts = consolidate(root, args.window_days, apply=apply)

    summary = counts.summary_dict()
    summary["mode"] = "apply" if apply else "dry-run"
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _log.info(
            "consolidate %s: %s",
            summary["mode"],
            " ".join(
                f"{k}={summary[k]}"
                for k in (
                    "markers_seen",
                    "parsed_total",
                    "deduped",
                    "promoted",
                    "promote_noop",
                    "promote_refused_budget",
                    "pruned",
                    "gotcha_candidate_seen",
                    "unparseable_skipped",
                    "no_op",
                )
            ),
        )
    # Budget-guard refusal (task #2189): loud, distinct, non-corrupting exit.
    if counts.promote_refused_budget > 0:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
