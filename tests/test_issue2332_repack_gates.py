"""Pinning tests for the #2332 repack gates (plan v2 SS6, 9 negative controls).

Everything Hub- and process-shaped is mocked/monkeypatched — NO network, no
pgrep against live processes, no RunPod/session subprocesses. The module under
test is loaded from ``scripts/issue2332_repack_prefixes.py`` by path.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "issue2332_repack_prefixes.py"
PREFIX = "issue1489_ctx_aug"


@pytest.fixture(scope="module")
def rp():
    spec = importlib.util.spec_from_file_location("issue2332_repack_prefixes", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _cp(rc: int, out: str = "", err: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["pgrep"], rc, out, err)


def _state_with_verify(rp, run_id: str) -> dict:
    return {
        "prefixes": {
            PREFIX: {"steps": {"hub_verify": {"pass": True, "run_id": run_id, "n_files": 3}}}
        }
    }


@pytest.fixture()
def quiet_gates(rp, monkeypatch):
    """Neutralize the writer gate + state writes for delete-loop tests."""
    monkeypatch.setattr(rp, "gate_writer", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "gate_mover", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "_save_state", lambda *_a, **_k: None)


# ── case 1: G-mover rc semantics ─────────────────────────────────────────────


def test_gmover_rc0_mover_alive_refuses(rp, monkeypatch):
    monkeypatch.setattr(rp, "_run_pgrep", lambda: _cp(0, "123 uv run .../the-mover.py\n"))
    with pytest.raises(rp.GateRefusal, match="ALIVE"):
        rp.gate_mover()


def test_gmover_rc2_probe_error_refuses(rp, monkeypatch):
    monkeypatch.setattr(rp, "_run_pgrep", lambda: _cp(2, "", "pgrep: bad args"))
    with pytest.raises(rp.GateRefusal, match="rc=2"):
        rp.gate_mover()


def test_gmover_rc1_empty_stdout_proceeds(rp, monkeypatch):
    monkeypatch.setattr(rp, "_run_pgrep", lambda: _cp(1, "", ""))
    rp.gate_mover()  # no raise = SAFE arm


def test_gmover_rc1_nonempty_stdout_refuses(rp, monkeypatch):
    monkeypatch.setattr(rp, "_run_pgrep", lambda: _cp(1, "ghost row\n", ""))
    with pytest.raises(rp.GateRefusal, match="non-empty"):
        rp.gate_mover()


# ── case 2: G-cap breach ─────────────────────────────────────────────────────


def test_gcap_breach_refuses_upload(rp):
    with pytest.raises(rp.GateRefusal, match="G-cap"):
        rp.gate_cap(30, fresh_count_fn=lambda: 999_500)


def test_gcap_under_buffer_proceeds(rp):
    assert rp.gate_cap(30, fresh_count_fn=lambda: 900_000) == 900_000


# ── case 3: delete unreachable without recorded verify PASS ─────────────────


def test_delete_unreachable_without_verify_pass(rp, quiet_gates, monkeypatch):
    calls = []
    monkeypatch.setattr(rp, "_create_commit_delete", lambda *a, **k: calls.append(a))
    st = {"prefixes": {PREFIX: {"steps": {}}}}
    with pytest.raises(rp.GateRefusal, match="verify-before-delete"):
        rp.run_delete(PREFIX, st, src_map={})
    assert calls == []


# ── case 4: same-size/different-oid routed to repack-stale, not delete ───────


def test_same_size_different_oid_is_stale_not_deletable(rp):
    src_map = {
        f"{PREFIX}/a.json": {"size": 10, "oid": "o1", "lfs": False},
        f"{PREFIX}/b.json": {"size": 20, "oid": "o2", "lfs": False},
    }
    current = {
        f"{PREFIX}/a.json": {"size": 10, "oid": "oX", "lfs": False},  # same size, new oid
        f"{PREFIX}/b.json": {"size": 20, "oid": "o2", "lfs": False},
        f"{PREFIX}/__packed__/shard-00000.tar": {"size": 999, "oid": "s", "lfs": True},
        f"{PREFIX}/new_fleet_file.json": {"size": 5, "oid": "n", "lfs": False},
    }
    deletable, stale, added = rp.derive_delete_set(src_map, current, PREFIX)
    assert deletable == [f"{PREFIX}/b.json"]
    assert stale == [f"{PREFIX}/a.json"]
    assert added == [f"{PREFIX}/new_fleet_file.json"]


# ── case 5: stale-checkpoint restart requires Hub re-verify ──────────────────


def test_stale_checkpoint_restart_reverifies_and_refuses_on_fail(rp, quiet_gates, monkeypatch):
    st = _state_with_verify(rp, run_id="a-prior-process-run")
    verify_calls = []

    def failing_verify(prefix, state):
        verify_calls.append(prefix)
        raise rp.VerifyFailure("Hub-side pack changed since the recorded PASS")

    commits = []
    monkeypatch.setattr(rp, "step_hub_verify", failing_verify)
    monkeypatch.setattr(rp, "_create_commit_delete", lambda *a, **k: commits.append(a))
    with pytest.raises(rp.VerifyFailure, match="changed"):
        rp.run_delete(PREFIX, st, src_map={})
    assert verify_calls == [PREFIX]
    assert commits == []  # delete never reached


def test_stale_checkpoint_restart_proceeds_after_reverify_pass(rp, quiet_gates, monkeypatch):
    st = _state_with_verify(rp, run_id="a-prior-process-run")
    verify_calls = []

    def ok_verify(prefix, state):
        verify_calls.append(prefix)
        state["prefixes"][prefix]["steps"]["hub_verify"] = {"pass": True, "run_id": rp.RUN_ID}

    monkeypatch.setattr(rp, "step_hub_verify", ok_verify)
    monkeypatch.setattr(rp, "_list_prefix", lambda *a, **k: {})
    monkeypatch.setattr(rp, "_create_commit_delete", lambda *a, **k: None)
    n = rp.run_delete(PREFIX, st, src_map={})
    assert verify_calls == [PREFIX]
    assert n == 0


def test_current_run_verify_pass_skips_reverify(rp, quiet_gates, monkeypatch):
    st = _state_with_verify(rp, run_id=rp.RUN_ID)
    monkeypatch.setattr(
        rp, "step_hub_verify", lambda *a, **k: pytest.fail("re-verify must not run")
    )
    monkeypatch.setattr(rp, "_list_prefix", lambda *a, **k: {})
    assert rp.run_delete(PREFIX, st, src_map={}) == 0


# ── case 6: CAS conflict + 504-masked-landed => re-derive, never blind retry ─


class _SeqListing:
    def __init__(self, seq):
        self.seq = list(seq)

    def __call__(self, *_a, **_k):
        return self.seq.pop(0) if len(self.seq) > 1 else self.seq[0]


def test_cas_conflict_rederives_with_fresh_head(rp, quiet_gates, monkeypatch):
    L1 = {
        f"{PREFIX}/a.json": {"size": 10, "oid": "o1", "lfs": False},
        f"{PREFIX}/b.json": {"size": 20, "oid": "o2", "lfs": False},
    }
    src_map = {k: dict(v) for k, v in L1.items()}
    monkeypatch.setattr(rp, "_list_prefix", _SeqListing([L1, L1, {}]))
    heads = iter(["h1", "h2"])
    monkeypatch.setattr(rp, "_repo_head_sha", lambda: next(heads))
    calls: list[tuple[list[str], str]] = []

    def commit(batch, head_sha, prefix):
        calls.append((sorted(batch), head_sha))
        if len(calls) == 1:
            raise RuntimeError("412 Precondition Failed: parent_commit mismatch")

    monkeypatch.setattr(rp, "_create_commit_delete", commit)
    st = _state_with_verify(rp, run_id=rp.RUN_ID)
    n = rp.run_delete(PREFIX, st, src_map=src_map, batch_size=10)
    assert n == 2
    assert len(calls) == 2
    # Re-DERIVED from a fresh re-list against a FRESH head — never a blind
    # retry of the same batch against the stale parent.
    assert calls[0][1] == "h1" and calls[1][1] == "h2"
    assert calls[0][0] == calls[1][0] == sorted(L1)


def test_504_masked_landed_commit_not_blind_retried(rp, quiet_gates, monkeypatch):
    L1 = {f"{PREFIX}/a.json": {"size": 10, "oid": "o1", "lfs": False}}
    src_map = {k: dict(v) for k, v in L1.items()}
    # The 504 MASKED a landed commit: the re-list after the exception shows
    # the file already gone, so the loop must finish with exactly ONE call.
    monkeypatch.setattr(rp, "_list_prefix", _SeqListing([L1, {}]))
    monkeypatch.setattr(rp, "_repo_head_sha", lambda: "h1")
    calls = []

    def commit(batch, head_sha, prefix):
        calls.append(sorted(batch))
        raise RuntimeError("504 Gateway Timeout")

    monkeypatch.setattr(rp, "_create_commit_delete", commit)
    st = _state_with_verify(rp, run_id=rp.RUN_ID)
    rp.run_delete(PREFIX, st, src_map=src_map)
    assert len(calls) == 1  # no double-interpretation, no blind retry


# ── case 7: G-1739 self-match filtered (resolved-path keyed, self/ancestors) ─


def test_g1739_self_match_filtered(rp, monkeypatch):
    wt = "/fake/worktrees/issue-1739"
    monkeypatch.setattr(rp, "_resolve_1739_worktree", lambda: wt)
    monkeypatch.setattr(rp, "_sessions_for_issue", lambda _i: [])
    monkeypatch.setattr(rp, "_pods_for_issue", lambda _i: [])
    procs = [
        # our own process: argv contains BOTH the prefix token and the resolved
        # path — must be excluded (self-PID), never trip the gate
        (os.getpid(), wt, f"python run --prefix issue1739_ctxmap {wt}"),
        # a foreign process whose argv contains the bare issue token but NOT
        # the resolved path, cwd elsewhere: the gate keys on the RESOLVED PATH,
        # never broad issue tokens => safe
        (999999991, "/elsewhere", "uv run python scripts/x.py run --prefix issue1739_ctxmap"),
    ]
    monkeypatch.setattr(rp, "_iter_procs", lambda: iter(procs))
    rp.gate_1739()  # no raise: the SAFE arm — self/ancestor filtering works


def test_g1739_foreign_process_on_resolved_path_trips(rp, monkeypatch):
    wt = "/fake/worktrees/issue-1739"
    monkeypatch.setattr(rp, "_resolve_1739_worktree", lambda: wt)
    monkeypatch.setattr(rp, "_sessions_for_issue", lambda _i: [])
    monkeypatch.setattr(rp, "_pods_for_issue", lambda _i: [])
    procs = [(999999992, f"{wt}/sub", "python x.py")]
    monkeypatch.setattr(rp, "_iter_procs", lambda: iter(procs))
    with pytest.raises(rp.GateRefusal, match="G-1739"):
        rp.gate_1739()


def test_g1739_foreign_cmdline_with_resolved_path_trips(rp, monkeypatch):
    wt = "/fake/worktrees/issue-1739"
    monkeypatch.setattr(rp, "_resolve_1739_worktree", lambda: wt)
    monkeypatch.setattr(rp, "_sessions_for_issue", lambda _i: [])
    monkeypatch.setattr(rp, "_pods_for_issue", lambda _i: [])
    procs = [(999999993, "/elsewhere", f"python driver.py --cwd {wt}")]
    monkeypatch.setattr(rp, "_iter_procs", lambda: iter(procs))
    with pytest.raises(rp.GateRefusal, match="G-1739"):
        rp.gate_1739()


# ── case 8: banned API names never appear in the repack script source ────────


def test_source_has_no_banned_folder_level_apis(rp):
    text = SCRIPT.read_text(encoding="utf-8")
    # NB: "upload_large_folder" does NOT contain the banned substring (the
    # chars after "upload_" are "large..."), so plain containment is exact.
    # The banned names are spelled by concatenation so THIS test file cannot
    # itself produce a false hit in any source-token sweep.
    assert "delete" + "_folder" not in text
    assert "upload" + "_folder" not in text
    # the large-folder uploader IS the sanctioned call and must be present
    assert "upload_large_folder" in text


# ── case 9: mid-loop gate flip stops the next Hub step ───────────────────────


def test_midloop_mover_flip_stops_next_delete_batch(rp, monkeypatch):
    monkeypatch.setattr(rp, "gate_writer", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "_save_state", lambda *_a, **_k: None)
    # REAL gate_mover, stateful probe: safe on the first evaluation, mover
    # ALIVE from the second on (a mid-loop relaunch).
    seq = iter([_cp(1, ""), _cp(0, "123 the-mover relaunched\n")])
    monkeypatch.setattr(rp, "_run_pgrep", lambda: next(seq))
    L = {
        f"{PREFIX}/a.json": {"size": 1, "oid": "o1", "lfs": False},
        f"{PREFIX}/b.json": {"size": 1, "oid": "o2", "lfs": False},
    }
    monkeypatch.setattr(rp, "_list_prefix", lambda *a, **k: L)
    monkeypatch.setattr(rp, "_repo_head_sha", lambda: "h1")
    calls = []
    monkeypatch.setattr(rp, "_create_commit_delete", lambda b, h, p: calls.append(sorted(b)))
    st = _state_with_verify(rp, run_id=rp.RUN_ID)
    with pytest.raises(rp.GateRefusal, match="ALIVE"):
        rp.run_delete(PREFIX, st, src_map={k: dict(v) for k, v in L.items()}, batch_size=1)
    assert len(calls) == 1  # first batch went through; the flip stopped the next


def test_midloop_mover_flip_stops_upload(rp, monkeypatch):
    monkeypatch.setattr(rp, "gate_writer", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "_run_pgrep", lambda: _cp(0, "123 mover alive\n"))
    st = {"prefixes": {PREFIX: {"steps": {}}}}
    with pytest.raises(rp.GateRefusal, match="ALIVE"):
        rp.step_upload(PREFIX, st, k_expected=3)


# ── env capability pin (plan SS4.0): parent_commit kwarg must exist ──────────


def test_create_commit_has_parent_commit_kwarg():
    from huggingface_hub import HfApi

    assert "parent_commit" in inspect.signature(HfApi.create_commit).parameters
