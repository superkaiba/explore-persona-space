"""Tests for scripts/consolidate_lessons.py (task #711).

The headline check is ``test_fixture_day_oracle`` — the CONTENT bar for the
task body's acceptance criterion (b): the script over a fixture day must produce
the SAME gotcha / memory writes the ``/daily`` consolidation step would have,
deterministically, and be a no-op on a second run. It asserts EXACT golden
post-state (the appended gotcha bullet text, the surviving dedupe-merge entry +
removed sibling, the removed prune entry + untouched hand-authored memory, the
second-run no-op) — NOT operation counts (a count-only test passes on WRONG
content).

The fixture is a ``tmp_path``-backed mini-repo (hermetic; no dependency on the
live ``tasks/`` tree), seeded with synthetic ``epm:failure-lesson v1`` markers
in all three parser tiers. The script's ``consolidate(root, ...)`` takes an
injectable root, so the mini-repo commits to itself.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# ─── Import the script module by file path ──────────────────────────────────

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "consolidate_lessons.py"
if "consolidate_lessons" in sys.modules:
    consolidate_lessons = sys.modules["consolidate_lessons"]
else:
    _spec = importlib.util.spec_from_file_location("consolidate_lessons", _SCRIPT)
    consolidate_lessons = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
    sys.modules["consolidate_lessons"] = consolidate_lessons
    _spec.loader.exec_module(consolidate_lessons)  # type: ignore[union-attr]

# The budget-guard tests (task #2189) express every size RELATIVE to the live
# constant so a future threshold change cannot silently invert a fixture's
# meaning (#838) — same bootstrap as tests/test_workflow_lint_gotchas_size.py.
_SCRIPTS_DIR = _SCRIPT.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from workflow_lint import GOTCHAS_SIZE_WARN_BYTES  # noqa: E402

# ─── tmp mini-repo helpers ──────────────────────────────────────────────────


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _init_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.test"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=root, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=root, check=True)
    # Mirror the real repo: logs/ is gitignored, so the consolidator's
    # day-stamped log line never dirties the tree (the no-op idempotency check
    # asserts `git status --porcelain` is empty after a second run).
    (root / ".gitignore").write_text("logs/\n")
    (root / "tasks").mkdir()
    (root / ".claude" / "rules").mkdir(parents=True)
    (root / ".claude" / "agent-memory").mkdir(parents=True)
    return root


def _commit_all(root: Path, msg: str = "seed") -> None:
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", msg], check=True)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _now() -> datetime:
    return datetime(2026, 6, 28, 12, 0, 0, tzinfo=UTC)


def _recent(days_ago: float = 1.0) -> str:
    return _iso(_now() - timedelta(days=days_ago))


def _registry(tasks: dict[int, dict]) -> dict:
    return {
        "highest_id": max(tasks) if tasks else 0,
        "tasks": {str(k): v for k, v in tasks.items()},
    }


def _write_task(root: Path, task_id: int, status: str, events: list[dict]) -> dict:
    """Create tasks/<status>/<id>/events.jsonl and return its registry entry."""
    d = root / "tasks" / status / str(task_id)
    d.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(json.dumps(e) for e in events) + "\n"
    (d / "events.jsonl").write_text(lines, encoding="utf-8")
    return {"path": f"tasks/{status}/{task_id}", "status": status, "title": "", "kind": "infra"}


def _sentinel_note(
    *,
    failure_class: str,
    phase: str,
    lesson: str,
    generalizes: str = "yes",
    owning_agent: str = "experiment-implementer",
    gotcha_candidate: str = "no",
) -> str:
    return (
        "<!-- epm:failure-lesson v1 -->\n"
        f"failure_class: {failure_class}\n"
        f"phase: {phase}\n"
        f"lesson: {lesson}\n"
        f"generalizes: {generalizes}\n"
        f"owning_agent: {owning_agent}\n"
        f"gotcha_candidate: {gotcha_candidate}\n"
        "<!-- /epm:failure-lesson -->"
    )


def _marker_event(note: str, ts: str | None = None) -> dict:
    return {
        "ts": ts or _recent(),
        "kind": "epm:failure-lesson",
        "version": 1,
        "by": "orchestrator",
        "note": note,
    }


def _feedback_file(name: str, body: str, *, lesson_derived: bool = True) -> str:
    if lesson_derived:
        fm = f"---\nname: {name}\ndescription: d\nmetadata:\n  type: feedback\n---\n"
    else:
        fm = f"---\nname: {name}\ndescription: d\ntype: reference\n---\n"
    return fm + body + "\n"


# ─── Parser tier tests ──────────────────────────────────────────────────────


def test_parse_sentinel_block_marker():
    """Tier 1: the sentinel-block shape parses all fields."""
    note = _sentinel_note(
        failure_class="code",
        phase="eval gen",
        lesson="Pass use_tqdm=False to dodge the ZeroDivisionError.",
        generalizes="yes",
        owning_agent="experiment-implementer",
        gotcha_candidate="yes",
    )
    fields = consolidate_lessons._parse_lesson_note(note)
    assert fields is not None
    assert fields["failure_class"] == "code"
    assert fields["phase"] == "eval gen"
    assert fields["lesson"] == "Pass use_tqdm=False to dodge the ZeroDivisionError."
    assert fields["generalizes"] == "yes"
    assert fields["owning_agent"] == "experiment-implementer"
    assert fields["gotcha_candidate"] == "yes"


def test_parse_bare_fields_marker_no_sentinel():
    """Tier 2: bare key: value lines (no wrapper), incl. a multi-line lesson."""
    note = (
        "failure_class: infra\n"
        "phase: pod provision\n"
        "lesson: First sentence of the lesson.\n"
        "Second line of the same lesson body.\n"
        "generalizes: yes\n"
        "owning_agent: experimenter\n"
        "gotcha_candidate: no\n"
    )
    fields = consolidate_lessons._parse_lesson_note(note)
    assert fields is not None
    assert fields["failure_class"] == "infra"
    assert fields["phase"] == "pod provision"
    # Multi-line lesson collected up to the next key: line.
    assert fields["lesson"] == "First sentence of the lesson.\nSecond line of the same lesson body."
    assert fields["generalizes"] == "yes"
    assert fields["owning_agent"] == "experimenter"


def test_parse_truncated_marker_skip_with_warn(caplog):
    """Tier 3: a note with neither the block nor required fields → None, WARN."""
    note = "<!-- epm:failure-lesson v1 -->\nfailure_class: code\n"  # no lesson, no close
    with caplog.at_level("WARNING"):
        fields = consolidate_lessons._parse_lesson_note(note)
    assert fields is None


def test_parse_open_no_close_marker_skip_with_warn():
    """Tier 3: an opening sentinel with no close + no recoverable fields → None."""
    note = "<!-- epm:failure-lesson v1 -->\nsome free text with no fields"
    assert consolidate_lessons._parse_lesson_note(note) is None
    assert consolidate_lessons._parse_lesson_note("") is None
    assert consolidate_lessons._parse_lesson_note("   ") is None


def test_scan_window_tier3_increments_unparseable_skipped(tmp_path, caplog):
    """A kind-matched but unparseable marker is skip-with-WARN, counted, not raised."""
    root = _init_repo(tmp_path)
    entry = _write_task(
        root,
        700,
        "completed",
        [_marker_event("<!-- epm:failure-lesson v1 -->\ntruncated")],
    )
    (root / "tasks" / "REGISTRY.json").write_text(json.dumps(_registry({700: entry})))
    _commit_all(root)
    with caplog.at_level("WARNING"):
        lessons, skips = consolidate_lessons.scan_window(root, 30, now=_now())
    assert lessons == []
    assert len(skips) == 1
    assert skips[0].task_id == 700
    assert any("unparseable failure-lesson" in r.message for r in caplog.records)


def test_parse_missing_owning_agent_raises(tmp_path):
    """The surviving hard-RAISE: a generalizing lesson with no owning_agent."""
    root = _init_repo(tmp_path)
    note = _sentinel_note(
        failure_class="code",
        phase="x",
        lesson="A generalizing lesson with no owner.",
        generalizes="yes",
        owning_agent="",
    )
    entry = _write_task(root, 701, "completed", [_marker_event(note)])
    (root / "tasks" / "REGISTRY.json").write_text(json.dumps(_registry({701: entry})))
    (root / ".claude" / "rules" / "gotchas.md").write_text("---\n---\n# Gotchas\n- existing\n")
    _commit_all(root)
    with pytest.raises(RuntimeError, match="owning_agent"):
        consolidate_lessons.consolidate(root, 30, apply=False, now=_now())


# ─── Threshold-boundary tests ───────────────────────────────────────────────


def test_dedupe_threshold_boundary(tmp_path):
    """Above-0.85 pair dedupes; below-0.85 pair does not."""
    a = "The cache invalidation step is missing after an update writes new rows to the table."
    # near-identical (tiny edit) → above threshold
    b_high = "The cache invalidation step is missing after an update writes new rows to that table."
    # very different → below threshold
    b_low = "A completely unrelated lesson about GPU memory fragmentation during long runs of vLLM."
    r_high = consolidate_lessons._ratio(a, b_high)
    r_low = consolidate_lessons._ratio(a, b_low)
    assert r_high >= consolidate_lessons.T_DEDUPE, f"expected >=0.85, got {r_high}"
    assert r_low < consolidate_lessons.T_DEDUPE, f"expected <0.85, got {r_low}"


def test_recurrence_K_boundary(tmp_path):
    """K=2 distinct tasks promotes; K=1 does not."""
    root = _init_repo(tmp_path)
    (root / ".claude" / "rules" / "gotchas.md").write_text(
        "---\n---\n# Gotchas\n- existing bullet\n"
    )
    lesson = "The widget loader silently drops rows whose id field is null."
    note = lambda: _sentinel_note(  # noqa: E731
        failure_class="code", phase="widget load", lesson=lesson, generalizes="no"
    )
    # K=1 (one task) → no promote
    e1 = _write_task(root, 710, "completed", [_marker_event(note())])
    (root / "tasks" / "REGISTRY.json").write_text(json.dumps(_registry({710: e1})))
    _commit_all(root)
    counts1 = consolidate_lessons.consolidate(root, 30, apply=True, now=_now())
    assert counts1.promoted == 0
    # K=2 (two distinct tasks) → promote
    e2 = _write_task(root, 711, "completed", [_marker_event(note())])
    (root / "tasks" / "REGISTRY.json").write_text(json.dumps(_registry({710: e1, 711: e2})))
    _commit_all(root)
    counts2 = consolidate_lessons.consolidate(root, 30, apply=True, now=_now())
    assert counts2.promoted == 1


def test_promote_already_present_noop(tmp_path):
    """A recurring cluster whose gotcha bullet is already present → promote_noop."""
    root = _init_repo(tmp_path)
    lesson = "The retry wrapper treats a 529 as fatal instead of transient and crashes the run."
    bullet = f"- **api retry** — {lesson} (#720, #721)"
    (root / ".claude" / "rules" / "gotchas.md").write_text(f"---\n---\n# Gotchas\n{bullet}\n")
    note = _sentinel_note(failure_class="code", phase="api retry", lesson=lesson, generalizes="no")
    e1 = _write_task(root, 720, "completed", [_marker_event(note)])
    e2 = _write_task(root, 721, "completed", [_marker_event(note)])
    (root / "tasks" / "REGISTRY.json").write_text(json.dumps(_registry({720: e1, 721: e2})))
    _commit_all(root)
    base = _git(root, "rev-list", "--count", "HEAD").strip()
    counts = consolidate_lessons.consolidate(root, 30, apply=True, now=_now())
    assert counts.promote_noop == 1
    assert counts.promoted == 0
    # No new commit (idempotent — the gotcha is already present).
    assert _git(root, "rev-list", "--count", "HEAD").strip() == base


# ─── The headline content oracle (criterion b) ──────────────────────────────


def test_fixture_day_oracle(tmp_path):
    """EXACT golden post-state for promote / dedupe-merge / prune + second-run no-op."""
    root = _init_repo(tmp_path)

    # --- gotchas.md (the promote target) ---
    gotchas = root / ".claude" / "rules" / "gotchas.md"
    gotchas_pre = (
        "---\ndescription: traps\n---\n\n# Gotchas\n\n"
        "- First existing trap.\n- Second existing trap.\n"
    )
    gotchas.write_text(gotchas_pre, encoding="utf-8")

    # --- agent-memory dirs ---
    ei_dir = root / ".claude" / "agent-memory" / "experiment-implementer"
    ei_dir.mkdir(parents=True)
    ex_dir = root / ".claude" / "agent-memory" / "experimenter"
    ex_dir.mkdir(parents=True)

    # SHAPE 1 — dedupe-merge pair (two distinct tasks, both lesson-derived,
    # ratio >= 0.85). Canonical = lower task_id 730; duplicate = 731.
    dedupe_lesson_a = (
        "The sweep launcher reuses a stale seed cache after the drift domain"
        " set changes between rounds."
    )
    dedupe_lesson_b = (
        "The sweep launcher reuses a stale seed cache after the drift-domain"
        " set changes between rounds."
    )
    canon_file = ei_dir / "feedback_stale_seed_cache.md"
    dup_file = ei_dir / "feedback_seed_cache_stale.md"
    canon_body = "Bust the seed cache with --bust-seed-cache when DRIFT_DOMAINS changes."
    dup_body = "Pass --bust-seed-cache after changing the drift-domain set or the cache goes stale."
    canon_file.write_text(_feedback_file("stale-seed-cache", dedupe_lesson_a + "\n\n" + canon_body))
    dup_file.write_text(_feedback_file("seed-cache-stale", dedupe_lesson_b + "\n\n" + dup_body))

    # SHAPE 3 — over-eager prune: a generalizes:yes lesson-derived entry whose
    # source task is terminal AND never recurs; a hand-authored memory alongside
    # must be left byte-equal.
    prune_lesson = (
        "The plotter crashes when a single condition has zero non-null samples in a one-off run."
    )
    prune_file = ex_dir / "feedback_plotter_zero_samples.md"
    prune_file.write_text(
        _feedback_file("plotter-zero-samples", prune_lesson + "\n\nGuard the plot call.")
    )
    hand_file = ex_dir / "reference_pod_api.md"
    hand_body = _feedback_file(
        "pod-api", "RunPod GraphQL needs the X-Team-Id header.", lesson_derived=False
    )
    hand_file.write_text(hand_body)

    # MEMORY.md index for both dirs.
    ei_memory = ei_dir / "MEMORY.md"
    ei_memory.write_text(
        "- [Stale seed cache](feedback_stale_seed_cache.md) — bust on domain change\n"
        "- [Seed cache stale](feedback_seed_cache_stale.md) — dup of the above\n"
        "- [Some other lesson](feedback_unrelated.md) — keep me\n"
    )
    (ei_dir / "feedback_unrelated.md").write_text(
        _feedback_file(
            "unrelated", "A totally unrelated lesson about config composition order in Hydra."
        )
    )
    ex_memory = ex_dir / "MEMORY.md"
    ex_memory.write_text(
        "- [Plotter zero samples](feedback_plotter_zero_samples.md) — guard the plot call\n"
        "- [Pod API](reference_pod_api.md) — X-Team-Id header\n"
    )

    # --- markers: ---
    # Shape 1 dedupe pair (generalizes: yes so they map to memory). DISTINCT
    # phases so they do NOT also form a promote cluster (dedupe is by lesson-text
    # similarity within an owning_agent, phase-independent; promote buckets by
    # (failure_class, phase)) — keeps the three shapes cleanly separable.
    note_730 = _sentinel_note(
        failure_class="code",
        phase="sweep launch (phase 1)",
        lesson=dedupe_lesson_a,
        generalizes="yes",
        owning_agent="experiment-implementer",
    )
    note_731 = _sentinel_note(
        failure_class="code",
        phase="sweep launch (phase 2)",
        lesson=dedupe_lesson_b,
        generalizes="yes",
        owning_agent="experiment-implementer",
    )
    # Shape 2 recurrence-promote cluster: K=2 distinct tasks, same class+phase,
    # gotcha_candidate: no, generalizes: no (so they do NOT also get pruned).
    promote_lesson = (
        "The eval harness counts a truncated completion as a silent zero when"
        " max_new_tokens is too small."
    )
    note_740 = _sentinel_note(
        failure_class="code",
        phase="eval harness",
        lesson=promote_lesson,
        generalizes="no",
    )
    note_741 = _sentinel_note(
        failure_class="code",
        phase="eval harness",
        lesson=promote_lesson,
        generalizes="no",
    )
    # Shape 3 prune marker (terminal task, generalizes: yes, never recurs).
    note_750 = _sentinel_note(
        failure_class="data",
        phase="plotting",
        lesson=prune_lesson,
        generalizes="yes",
        owning_agent="experimenter",
    )

    e730 = _write_task(root, 730, "completed", [_marker_event(note_730)])
    e731 = _write_task(root, 731, "archived", [_marker_event(note_731)])
    e740 = _write_task(root, 740, "completed", [_marker_event(note_740)])
    e741 = _write_task(root, 741, "completed", [_marker_event(note_741)])
    e750 = _write_task(root, 750, "completed", [_marker_event(note_750)])
    reg = _registry({730: e730, 731: e731, 740: e740, 741: e741, 750: e750})
    (root / "tasks" / "REGISTRY.json").write_text(json.dumps(reg))
    _commit_all(root)

    base_count = int(_git(root, "rev-list", "--count", "HEAD").strip())

    # ── First --apply run ──
    counts = consolidate_lessons.consolidate(root, 30, apply=True, now=_now())

    # GOLDEN: dedupe-merge — canonical survives byte-equal, duplicate removed.
    assert canon_file.exists()
    assert canon_file.read_text() == _feedback_file(
        "stale-seed-cache", dedupe_lesson_a + "\n\n" + canon_body
    )
    assert not dup_file.exists()
    # MEMORY.md: exactly the duplicate's bullet removed, others preserved in order.
    assert ei_memory.read_text() == (
        "- [Stale seed cache](feedback_stale_seed_cache.md) — bust on domain change\n"
        "- [Some other lesson](feedback_unrelated.md) — keep me\n"
    )

    # GOLDEN: promote — exactly one new bullet appended after the last existing.
    expected_bullet = f"- **eval harness** — {promote_lesson} (#740, #741)"
    assert gotchas.read_text() == gotchas_pre.rstrip("\n") + "\n" + expected_bullet + "\n"

    # GOLDEN: prune — the over-eager entry removed, hand-authored memory byte-equal.
    assert not prune_file.exists()
    assert hand_file.read_text() == hand_body
    assert ex_memory.read_text() == "- [Pod API](reference_pod_api.md) — X-Team-Id header\n"

    # Exactly ONE new commit for the whole pass.
    after_count = int(_git(root, "rev-list", "--count", "HEAD").strip())
    assert after_count == base_count + 1, f"expected 1 new commit, got {after_count - base_count}"

    assert counts.deduped == 1
    assert counts.promoted == 1
    assert counts.pruned == 1
    assert not counts.is_noop

    # ── Second --apply run over the now-mutated tree: STRICT no-op ──
    counts2 = consolidate_lessons.consolidate(root, 30, apply=True, now=_now())
    assert counts2.is_noop
    assert counts2.deduped == 0
    assert counts2.promoted == 0
    assert counts2.pruned == 0
    # Zero tracked-file mutations.
    assert _git(root, "status", "--porcelain").strip() == ""
    # No new commit.
    assert int(_git(root, "rev-list", "--count", "HEAD").strip()) == after_count


# ─── Prune conservatism ─────────────────────────────────────────────────────


def test_prune_keeps_when_source_task_active(tmp_path):
    """A generalizes:yes entry whose source task is NOT terminal is KEPT."""
    root = _init_repo(tmp_path)
    ex_dir = root / ".claude" / "agent-memory" / "experimenter"
    ex_dir.mkdir(parents=True)
    lesson = "An active-task lesson that should not be pruned while its task runs."
    f = ex_dir / "feedback_active.md"
    f.write_text(_feedback_file("active", lesson))
    (ex_dir / "MEMORY.md").write_text("- [Active](feedback_active.md) — keep\n")
    note = _sentinel_note(
        failure_class="code",
        phase="x",
        lesson=lesson,
        generalizes="yes",
        owning_agent="experimenter",
    )
    e = _write_task(root, 760, "running", [_marker_event(note)])
    (root / "tasks" / "REGISTRY.json").write_text(json.dumps(_registry({760: e})))
    (root / ".claude" / "rules" / "gotchas.md").write_text("---\n---\n# Gotchas\n- x\n")
    _commit_all(root)
    counts = consolidate_lessons.consolidate(root, 30, apply=True, now=_now())
    assert counts.pruned == 0
    assert f.exists()


def test_window_excludes_old_markers(tmp_path):
    """A marker older than the window is not scanned."""
    root = _init_repo(tmp_path)
    note = _sentinel_note(failure_class="code", phase="x", lesson="old lesson", generalizes="no")
    e = _write_task(
        root, 770, "completed", [_marker_event(note, ts=_iso(_now() - timedelta(days=40)))]
    )
    (root / "tasks" / "REGISTRY.json").write_text(json.dumps(_registry({770: e})))
    _commit_all(root)
    lessons, skips = consolidate_lessons.scan_window(root, 7, now=_now())
    assert lessons == []
    assert skips == []


# ─── gotchas.md byte-budget guard (task #2189) ──────────────────────────────
#
# Every size below is expressed RELATIVE to ``GOTCHAS_SIZE_WARN_BYTES`` (never
# a hardcoded 200000): the fixture builds a gotchas.md pre-text sized so the
# projected post-append byte count lands exactly at WARN-1 / WARN / WARN+1.
# Markers carry REAL-now timestamps (``main()`` cannot inject ``now``).


_BUDGET_PHASE = "shard packing"
_BUDGET_LESSON = "The frobnicator drops the final shard whenever the batch size divides evenly."


def _budget_bullet() -> str:
    return f"- **{_BUDGET_PHASE}** — {_BUDGET_LESSON} (#810, #811)"


def _real_recent(days_ago: float = 1.0) -> str:
    return _iso(datetime.now(tz=UTC) - timedelta(days=days_ago))


def _seed_budget_repo(tmp_path: Path, *, projected: int) -> tuple[Path, Path, str]:
    """Mini-repo whose sole promote cluster projects gotchas.md to ``projected`` B.

    ``promote()`` composes ``new_text = pre.rstrip("\\n") + "\\n" + bullet +
    "\\n"``, so the padding is sized backwards from the target. Returns
    (root, gotchas_path, pre_text).
    """
    root = _init_repo(tmp_path)
    bullet = _budget_bullet()
    header = "---\n---\n# Gotchas\n- existing bullet\n"
    pad = projected - len(header.encode()) - len(bullet.encode()) - 2
    assert pad > 0
    pre = header + "x" * pad + "\n"
    gotchas = root / ".claude" / "rules" / "gotchas.md"
    gotchas.write_text(pre, encoding="utf-8")
    assert len(pre.rstrip("\n").encode()) + 1 + len(bullet.encode()) + 1 == projected, (
        "fixture arithmetic drifted"
    )
    note = _sentinel_note(
        failure_class="code", phase=_BUDGET_PHASE, lesson=_BUDGET_LESSON, generalizes="no"
    )
    ts = _real_recent()
    e1 = _write_task(root, 810, "completed", [_marker_event(note, ts=ts)])
    e2 = _write_task(root, 811, "completed", [_marker_event(note, ts=ts)])
    (root / "tasks" / "REGISTRY.json").write_text(json.dumps(_registry({810: e1, 811: e2})))
    _commit_all(root)
    return root, gotchas, pre


def _main_json(capsys, argv: list[str]) -> tuple[int, dict]:
    """Run ``main`` with ``--json`` and return (rc, parsed summary dict)."""
    rc = consolidate_lessons.main([*argv, "--json"])
    out = capsys.readouterr().out
    summary = json.loads(out[out.index("{") :])
    return rc, summary


def test_budget_case1_warn_minus_one_appends(tmp_path, capsys):
    """Case 1: projected == WARN-1 → append written, promoted counted, exit 0."""
    root, gotchas, pre = _seed_budget_repo(tmp_path, projected=GOTCHAS_SIZE_WARN_BYTES - 1)
    rc, summary = _main_json(capsys, ["--apply", "--window-days", "30", "--root", str(root)])
    assert rc == 0
    assert summary["promoted"] == 1
    assert summary["promote_refused_budget"] == 0
    assert gotchas.read_text(encoding="utf-8") == (
        pre.rstrip("\n") + "\n" + _budget_bullet() + "\n"
    )
    assert gotchas.stat().st_size == GOTCHAS_SIZE_WARN_BYTES - 1


def test_budget_case2_exactly_at_warn_appends(tmp_path):
    """Case 2: projected == WARN exactly → append written (strictly-greater
    parity with ``check_gotchas_size``)."""
    root, gotchas, pre = _seed_budget_repo(tmp_path, projected=GOTCHAS_SIZE_WARN_BYTES)
    counts = consolidate_lessons.consolidate(root, 30, apply=True)
    assert counts.promoted == 1
    assert counts.promote_refused_budget == 0
    assert gotchas.stat().st_size == GOTCHAS_SIZE_WARN_BYTES
    assert gotchas.read_text(encoding="utf-8").endswith(_budget_bullet() + "\n")


def test_budget_case3_warn_plus_one_refuses_without_raising(tmp_path):
    """Case 3: projected == WARN+1 → gotchas.md byte-identical, refusal counted,
    promoted == 0, NO raise from promote(), and ``main()`` returns 3."""
    root, gotchas, pre = _seed_budget_repo(tmp_path, projected=GOTCHAS_SIZE_WARN_BYTES + 1)
    # consolidate() returns normally (no raise) with the refusal counted.
    counts = consolidate_lessons.consolidate(root, 30, apply=True)
    assert counts.promoted == 0
    assert counts.promote_refused_budget == 1
    assert gotchas.read_bytes() == pre.encode("utf-8")  # byte-identical: no half-landed write
    # The refusal is not a term in is_noop (nothing was mutated).
    assert counts.is_noop
    # main() over the same (unmutated) repo re-derives the refusal → exit 3.
    rc = consolidate_lessons.main(["--apply", "--window-days", "30", "--root", str(root)])
    assert rc == 3
    assert gotchas.read_bytes() == pre.encode("utf-8")


def test_budget_case4_refused_bullets_verbatim_and_counts_line(tmp_path, capsys):
    """Case 4: the refused bullet appears VERBATIM on stdout and the counts line
    is still written, carrying the new counter."""
    root, gotchas, _pre = _seed_budget_repo(tmp_path, projected=GOTCHAS_SIZE_WARN_BYTES + 1)
    consolidate_lessons.consolidate(root, 30, apply=True)
    out = capsys.readouterr().out
    assert _budget_bullet() in out
    logs = list((root / "logs" / "lesson_consolidate").glob("*.log"))
    assert len(logs) == 1
    assert "promote_refused_budget=1" in logs[0].read_text(encoding="utf-8")


def test_budget_case5_dedupe_commits_while_promote_refuses(tmp_path, capsys):
    """Case 5 (the v2-defect regression pin): dedupe mutates AND promote refuses
    in the SAME --apply run → the memory mutation IS committed, gotchas.md is
    NOT touched, the worktree is clean, and main() returns 3."""
    root, gotchas, pre = _seed_budget_repo(tmp_path, projected=GOTCHAS_SIZE_WARN_BYTES + 1)
    # Add a dedupe-merge pair (the oracle's SHAPE 1, distinct phases so it does
    # not form a second promote cluster).
    ei_dir = root / ".claude" / "agent-memory" / "experiment-implementer"
    ei_dir.mkdir(parents=True)
    lesson_a = "The sweep launcher reuses a stale seed cache after the drift domain set changes."
    lesson_b = "The sweep launcher reuses a stale seed cache after the drift-domain set changes."
    canon_file = ei_dir / "feedback_stale_seed_cache.md"
    dup_file = ei_dir / "feedback_seed_cache_stale.md"
    canon_file.write_text(_feedback_file("stale-seed-cache", lesson_a + "\n\nBust the cache."))
    dup_file.write_text(_feedback_file("seed-cache-stale", lesson_b + "\n\nBust the cache too."))
    (ei_dir / "MEMORY.md").write_text(
        "- [Stale seed cache](feedback_stale_seed_cache.md) — bust on domain change\n"
        "- [Seed cache stale](feedback_seed_cache_stale.md) — dup of the above\n"
    )
    ts = _real_recent()
    note_830 = _sentinel_note(
        failure_class="code",
        phase="sweep launch (phase 1)",
        lesson=lesson_a,
        generalizes="yes",
        owning_agent="experiment-implementer",
    )
    note_831 = _sentinel_note(
        failure_class="code",
        phase="sweep launch (phase 2)",
        lesson=lesson_b,
        generalizes="yes",
        owning_agent="experiment-implementer",
    )
    e830 = _write_task(root, 830, "completed", [_marker_event(note_830, ts=ts)])
    e831 = _write_task(root, 831, "archived", [_marker_event(note_831, ts=ts)])
    reg = json.loads((root / "tasks" / "REGISTRY.json").read_text())
    reg["tasks"]["830"] = e830
    reg["tasks"]["831"] = e831
    reg["highest_id"] = 831
    (root / "tasks" / "REGISTRY.json").write_text(json.dumps(reg))
    _commit_all(root)
    base_count = int(_git(root, "rev-list", "--count", "HEAD").strip())

    rc, summary = _main_json(capsys, ["--apply", "--window-days", "30", "--root", str(root)])

    assert rc == 3
    assert summary["promote_refused_budget"] == 1
    assert summary["deduped"] == 1
    assert summary["promoted"] == 0
    # Dedupe's memory mutation landed AND was committed.
    assert not dup_file.exists()
    assert int(_git(root, "rev-list", "--count", "HEAD").strip()) == base_count + 1
    # gotchas.md untouched, worktree clean (no half-landed mutation).
    assert gotchas.read_bytes() == pre.encode("utf-8")
    assert _git(root, "status", "--porcelain").strip() == ""


def test_budget_case6_dry_run_over_budget_same_counter_exit3(tmp_path, capsys):
    """Case 6: dry-run (no --apply) over budget → no write either way, the same
    refusal counter, exit 3."""
    root, gotchas, pre = _seed_budget_repo(tmp_path, projected=GOTCHAS_SIZE_WARN_BYTES + 1)
    rc, summary = _main_json(capsys, ["--window-days", "30", "--root", str(root)])
    assert rc == 3
    assert summary["mode"] == "dry-run"
    assert summary["promote_refused_budget"] == 1
    assert summary["promoted"] == 0
    assert gotchas.read_bytes() == pre.encode("utf-8")


def test_budget_case7_post_trim_replay_promotes(tmp_path):
    """Case 7: refusal is NOT sticky — after a trim the same markers promote."""
    root, gotchas, _pre = _seed_budget_repo(tmp_path, projected=GOTCHAS_SIZE_WARN_BYTES + 1)
    counts1 = consolidate_lessons.consolidate(root, 30, apply=True)
    assert counts1.promote_refused_budget == 1
    assert counts1.promoted == 0
    # Simulate the human re-trim: shrink the file well under budget.
    bullet = _budget_bullet()
    header = "---\n---\n# Gotchas\n- existing bullet\n"
    trimmed = header + "x" * 1000 + "\n"
    gotchas.write_text(trimmed, encoding="utf-8")
    counts2 = consolidate_lessons.consolidate(root, 30, apply=True)
    assert counts2.promote_refused_budget == 0
    assert counts2.promoted == 1
    assert gotchas.read_text(encoding="utf-8") == trimmed.rstrip("\n") + "\n" + bullet + "\n"
