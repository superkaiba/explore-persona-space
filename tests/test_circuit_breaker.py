"""Tests for the /issue Step 7 crash-fix circuit-breaker predicate (task #718).

The predicate ``task_workflow.circuit_breaker_should_fire`` is pure (takes a
list of ``events.jsonl`` event dicts + the plan text), so almost every case
feeds plain dicts directly — no git fixture (mirrors ``test_workflow_fix_dedup``
style). The one realistic case (g) reads the ACTUAL #664 notes, because feeding
K identical synthetic copies of one note would hash-collide trivially and would
NOT exercise the MF#1 per-round-varying-argv signature bug.

Cross-document gate (MF#4): the SKILL.md "Crash-fix circuit-breaker" block must
exist AND cite BOTH the predicate and the canonical-pivot key. Shipped here as a
pytest test (the plan allows either a workflow_lint flag or a pytest test;
pytest is the simpler surface and keeps workflow_lint.py untouched).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from explore_persona_space import task_workflow as tw
from tests.issue_skill_source import issue_skill_text

# Anchor cross-document checks on the tree this test FILE lives in (the worktree
# during a /issue test-verdict, main after merge) — NOT task_workflow.repo_root(),
# which branch-guards to main and would read main's SKILL.md instead of the
# worktree edits under review (the "repo_root() unfit for worktree-local files"
# gotcha; see .claude/agent-memory/implementer/reference_repo_root_vs_cache_paths).
_TREE_ROOT = Path(__file__).resolve().parents[1]


def _fail(note: str) -> dict[str, str]:
    """A minimal ``epm:failure`` event dict."""
    return {"kind": "epm:failure", "note": note}


def _progress(note: str = "phase tick") -> dict[str, str]:
    return {"kind": "epm:progress", "note": note}


# A same-signature dispatch-crash note family that varies only in the volatile
# argv tail (source / dose / seed) — the #664 shape, condensed.
def _p2_extract_note(source: str, dose: str, seed: int) -> str:
    return (
        f"issue664 dispatch failed at phase=p2: CalledProcessError: Command "
        f"'['/workspace/.venv/bin/python3', "
        f"'/workspace/scripts/issue664_extract_store.py', '--source', "
        f"'{source}', '--dose', '{dose}', '--seed', '{seed}']' returned "
        f"non-zero exit status 1."
    )


def _same_sig_failures(n: int) -> list[dict[str, str]]:
    """n same-(phase, class, assert_tag) failures with per-round-varying argv."""
    sources = ["librarian", "surgeon", "default", "pilot", "chef", "poet"]
    return [_fail(_p2_extract_note(sources[i % len(sources)], "d2", 42 + i)) for i in range(n)]


# ─── (a) K-1 same-signature → no fire ────────────────────────────────────────
def test_a_below_threshold_no_fire():
    events = _same_sig_failures(3)  # K-1 = 3
    assert tw.circuit_breaker_should_fire(events, "", K=4) is None


# ─── (b) K same-signature → fire, with pivot_scope asserted (MF#5) ───────────
def test_b_threshold_fires_with_pivot_scope():
    events = _same_sig_failures(4)
    fire = tw.circuit_breaker_should_fire(events, "", K=4)
    assert fire is not None
    assert fire["trigger"] == "same_failure_class"
    assert fire["count"] >= 4
    phase, failure_class, assert_tag = fire["signature"]
    assert phase == "p2"
    assert assert_tag == "CalledProcessError:issue664_extract_store.py"
    scope = fire["pivot_scope"]
    assert scope
    # pivot_scope contains each signature component + the count verbatim.
    assert phase in scope
    assert failure_class in scope
    assert assert_tag in scope
    assert str(fire["count"]) in scope


# ─── (c) different signatures across K rounds → no fire ──────────────────────
def test_c_distinct_signatures_no_fire():
    events = [
        _fail("dispatch failed at phase=p1: RuntimeError: bad thing"),
        _fail("dispatch failed at phase=p2: ValueError: other thing"),
        _fail("crashed phase=p3 failure_class: code KeyError: missing"),
        _fail(_p2_extract_note("librarian", "d2", 42)),
    ]
    assert tw.circuit_breaker_should_fire(events, "", K=4) is None


# ─── (d) enumerated ladder exhausted → fire trigger-2, pivot_scope (MF#5) ────
def test_d_enumerated_fallback_exhausted_fires():
    plan = "## §11 Decision\n\nEscape ladder: Option A → Option B → Option C. If all fail, re-plan."
    events = [
        _progress("attempting Option A"),
        _fail("phase=p2: RuntimeError: gate tripped"),
        _progress("attempting Option B"),
        _fail("phase=p2: RuntimeError: gate tripped"),
        _progress("attempting Option C"),
        _fail("phase=p2: RuntimeError: gate tripped"),
    ]
    fire = tw.circuit_breaker_should_fire(events, plan, K=4)
    assert fire is not None
    assert fire["trigger"] == "enumerated_fallback_exhausted"
    assert fire["ladder"] == ["A", "B", "C"]
    scope = fire["pivot_scope"]
    assert scope
    assert "A → B → C" in scope
    assert fire["gate"] in scope


# ─── (e) ladder with only A attempted → no trigger-2 fire ────────────────────
def test_e_incomplete_ladder_no_fire():
    plan = "Escape ladder: Option A → Option B. Try in order."
    events = [
        _progress("attempting Option A"),
        _fail("phase=p2: RuntimeError: gate tripped"),
        _fail("phase=p2: RuntimeError: gate tripped"),
    ]
    # Option B never launched → trigger 2 must not fire; and the 2 same-sig
    # failures are < K, so trigger 1 must not fire either.
    assert tw.circuit_breaker_should_fire(events, plan, K=4) is None


# ─── (e2) trigger-2 requires a POST-launch failure (#718 Codex critic) ───────
# A stale epm:failure that PRECEDES the ladder launches, with NO failure after
# the final option is launched, must NOT fire trigger-2. The gate never
# re-tripped post-exhaustion — there is nothing to pivot away from.
def test_e2_pre_ladder_failure_only_no_fire():
    plan = (
        "## §11 escape ladder\n\n"
        "Gate G: if the a7 assert trips, walk the ladder.\n"
        "Option A: lower the threshold.\n"
        "Option B: switch the estimator.\n"
    )
    events = [
        _fail("p1 some old unrelated crash"),  # PRECEDES the ladder
        _progress("Launching Option A per plan §11 escape ladder"),
        _progress("Launching Option B per plan §11 escape ladder"),
    ]
    # Both options launched, but the only failure is BEFORE the ladder → no
    # post-exhaustion re-trip → trigger-2 must NOT fire. (2 = K-2 < K, so
    # trigger-1 stays silent too.)
    assert tw.circuit_breaker_should_fire(events, plan, K=4) is None


# ─── (e3) trigger-2 fires on a POST-launch failure (happy path preserved) ────
def test_e3_post_launch_failure_fires():
    plan = (
        "## §11 escape ladder\n\n"
        "Gate G: if the a7 assert trips, walk the ladder.\n"
        "Option A: lower the threshold.\n"
        "Option B: switch the estimator.\n"
    )
    events = [
        _progress("Launching Option A per plan §11 escape ladder"),
        _progress("Launching Option B per plan §11 escape ladder"),
        _fail("p3 a7-assert HALT — gate G re-tripped after Option B"),  # AFTER ladder
    ]
    fire = tw.circuit_breaker_should_fire(events, plan, K=4)
    assert fire is not None
    assert fire["trigger"] == "enumerated_fallback_exhausted"
    assert fire["ladder"] == ["A", "B"]


# ─── (e4) pre-launch AND post-launch failure → fires (post-launch is what counts)
# A stale pre-ladder failure is irrelevant; the POST-launch re-trip drives the
# fire. Pins the post-launch-only semantics (the pre-launch failure neither
# blocks nor is required).
def test_e4_pre_and_post_launch_failure_fires():
    plan = (
        "## §11 escape ladder\n\n"
        "Gate G: if the a7 assert trips, walk the ladder.\n"
        "Option A: lower the threshold.\n"
        "Option B: switch the estimator.\n"
    )
    events = [
        _fail("p1 some old unrelated crash"),  # PRECEDES the ladder — irrelevant
        _progress("Launching Option A per plan §11 escape ladder"),
        _progress("Launching Option B per plan §11 escape ladder"),
        _fail("p3 a7-assert HALT — gate G re-tripped after Option B"),  # AFTER ladder
    ]
    fire = tw.circuit_breaker_should_fire(events, plan, K=4)
    assert fire is not None
    assert fire["trigger"] == "enumerated_fallback_exhausted"
    assert fire["ladder"] == ["A", "B"]


# ─── (f) milestone reset on epm:experiment-implementation / epm:results ──────
def test_f_milestone_reset_experiment_implementation():
    events = [
        *_same_sig_failures(3),
        {"kind": "epm:experiment-implementation", "note": "round 18 report"},
        *_same_sig_failures(1),
    ]
    # 3 then reset then 1 → no signature reaches K=4.
    assert tw.circuit_breaker_should_fire(events, "", K=4) is None


def test_f_milestone_reset_results():
    events = [
        *_same_sig_failures(3),
        {"kind": "epm:results", "note": "results landed"},
        *_same_sig_failures(1),
    ]
    assert tw.circuit_breaker_should_fire(events, "", K=4) is None


# ─── (g) REQUIRED: fires on the REAL #664 note shape (MF#1 + MF#2 + MF#5) ─────
def _load_664_p2_calledproc() -> list[dict]:
    """Load #664's first 4 same-signature p2/CalledProcessError failures.

    Resolves the events path via find_task_path (survives a #664 status move),
    skips only if the task folder is genuinely absent.
    """
    try:
        events_file = tw.find_task_path(664) / "events.jsonl"
    except FileNotFoundError:
        pytest.skip("task #664 not present in this environment")
    if not events_file.exists():
        pytest.skip("task #664 events.jsonl absent")
    all_events = [json.loads(line) for line in events_file.read_text().splitlines() if line.strip()]
    p2_calledproc = [
        e
        for e in all_events
        if e.get("kind") == "epm:failure"
        and "phase=p2" in e.get("note", "")
        and "CalledProcessError" in e.get("note", "")
    ]
    return p2_calledproc


def test_g_real_664_notes_fire_with_command_family_tag():
    p2_calledproc = _load_664_p2_calledproc()
    # The same-signature issue664_extract_store.py run; for K=4 the first 4 suffice.
    events = p2_calledproc[:4]
    assert len(events) == 4, "expected >=4 real #664 p2/CalledProcessError failures"
    fire = tw.circuit_breaker_should_fire(events, "", K=4)
    assert fire is not None
    assert fire["trigger"] == "same_failure_class"
    assert fire["count"] >= 4
    phase, failure_class, assert_tag = fire["signature"]
    assert phase == "p2"
    # assert_tag is the exception-type / command-family token, NOT a note-hash.
    assert assert_tag.startswith("CalledProcessError:")
    assert assert_tag.endswith(".py")
    assert assert_tag == "CalledProcessError:issue664_extract_store.py"
    # A 12-hex note-hash would be all-hex and len 12 — assert it is NOT that.
    assert not re.fullmatch(r"[0-9a-f]{12}", assert_tag)
    # pivot_scope carries every signature component + the count.
    scope = fire["pivot_scope"]
    assert scope
    assert phase in scope
    assert failure_class in scope
    assert assert_tag in scope
    assert str(fire["count"]) in scope


# ─── (h) REQUIRED: epm:progress interleaved does NOT reset (MF#3) ────────────
def test_h_progress_markers_do_not_reset_counter():
    sigs = _same_sig_failures(4)
    # Interleave benign progress notes mirroring #664's trap-window shapes.
    progress_notes = [
        _progress("[autonomous_session_watch:orphan-respawn] active task respawn"),
        _progress("[recovery] orphan-respawn session live. Pod healthy"),
        _progress("phase transition: p2_extract_eval -> p3_upload"),
        _progress("[recovery, tick 3] Pod healthy, p2 progressing"),
    ]
    events = [
        sigs[0],
        progress_notes[0],
        sigs[1],
        progress_notes[1],
        progress_notes[2],
        sigs[2],
        progress_notes[3],
        sigs[3],
    ]
    fire = tw.circuit_breaker_should_fire(events, "", K=4)
    assert fire is not None
    assert fire["trigger"] == "same_failure_class"
    assert fire["count"] >= 4


# ─── (i) REQUIRED: non-default K is honored (MF#6) ───────────────────────────
@pytest.mark.parametrize("k", [2, 4, 6])
def test_i_non_default_k_honored(k):
    # Exactly k same-sig failures → fires; k-1 → does not.
    fire_at_k = tw.circuit_breaker_should_fire(_same_sig_failures(k), "", K=k)
    assert fire_at_k is not None
    assert fire_at_k["count"] >= k
    fire_below = tw.circuit_breaker_should_fire(_same_sig_failures(k - 1), "", K=k)
    assert fire_below is None


# ─── per-rung assert_tag fallback-coverage cases ─────────────────────────────
def test_rung1_explicit_assert_tag_field():
    note = "phase=p2: something failed\nassert_tag: my_tag\nmore lines"
    _, _, assert_tag = tw._cb_failure_signature(note, None)
    assert assert_tag == "my_tag"


def test_rung2_bracketed_assert_tag():
    note = "phase=p3: RuntimeError: [a7-assert] PRODUCTION marker read-gauge HALT"
    _, _, assert_tag = tw._cb_failure_signature(note, None)
    assert assert_tag == "a7"


def test_rung3_calledproc_command_family():
    note = (
        "dispatch failed at phase=p2: CalledProcessError: Command "
        "'['/usr/bin/python3', '/workspace/scripts/script.py', '--x', '1']' "
        "returned non-zero exit status 1."
    )
    _, _, assert_tag = tw._cb_failure_signature(note, None)
    assert assert_tag == "CalledProcessError:script.py"


def test_rung3_bare_exception_non_subprocess():
    note = "phase=p1: RuntimeError: model failed to load on cuda:0"
    _, _, assert_tag = tw._cb_failure_signature(note, None)
    assert assert_tag == "RuntimeError"


def test_rung4_note_hash_when_no_structured_token():
    note = "phase=p2: the run did not finish for reasons unknown today"
    _, _, assert_tag = tw._cb_failure_signature(note, None)
    assert re.fullmatch(r"[0-9a-f]{12}", assert_tag)


def test_rung4_note_hash_invariant_to_volatile_spans():
    # Two notes differing ONLY in timestamp / pid / file:line / argv collapse to
    # ONE hash. Use a leading line that does NOT match the exception/argv rungs
    # so both fall through to rung 4.
    a = (
        "run wedged at 2026-06-28T01:00:00Z pid=123 in foo/bar.py:42 "
        "Command '['python3', 'a.py', '--seed', '1']'"
    )
    b = (
        "run wedged at 2026-06-28T09:30:11Z pid=999 in foo/bar.py:88 "
        "Command '['python3', 'a.py', '--seed', '777']'"
    )
    assert tw._cb_note_hash(a) == tw._cb_note_hash(b)


# ─── MF#4: cross-document SKILL.md scoped check (the v1 grep was a no-op) ─────
def test_skill_md_circuit_breaker_block_cites_both_keys():
    content = issue_skill_text()
    m = re.search(
        r"\*\*Crash-fix circuit-breaker.*?(?=\n\s{0,3}\*\*[A-Z])",
        content,
        re.DOTALL,
    )
    assert m, "Crash-fix circuit-breaker block missing from SKILL.md"
    block = m.group(0)
    assert "task_workflow.circuit_breaker_should_fire" in block, (
        "predicate citation missing from circuit-breaker block"
    )
    assert "pivot_criteria.plan_contradiction_replan" in block, (
        "canonical-predicate citation missing from circuit-breaker block"
    )


def test_workflow_yaml_has_enumerated_fallback_exhaustion_subclause():
    wf = _TREE_ROOT / ".claude" / "workflow.yaml"
    assert "enumerated-fallback-exhaustion" in wf.read_text(), (
        "enumerated-fallback-exhaustion sub-clause missing from workflow.yaml"
    )
