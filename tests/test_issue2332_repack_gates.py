"""Pinning tests for the #2332 repack gates (plan v2 SS6, 9 negative controls).

Everything Hub- and process-shaped is mocked/monkeypatched — NO network, no
pgrep against live processes, no RunPod/session subprocesses. The module under
test is loaded from ``scripts/issue2332_repack_prefixes.py`` by path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import os
import shutil
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


PACK_PATH = f"{PREFIX}/__packed__/shard-00000.tar"


def _state_with_verify(rp, run_id: str) -> dict:
    return {
        "prefixes": {
            PREFIX: {
                "steps": {"hub_verify": {"pass": True, "run_id": run_id, "n_files": 3}},
                "packed_files": {PACK_PATH: {"size": 999, "sha256": "s256", "gitsha1": "g1"}},
            }
        }
    }


def _with_pack(listing: dict) -> dict:
    """A listing that also carries the recorded pack file (identity-matching),
    as every real post-upload listing does — the r3 per-batch pack-presence
    guard refuses to delete without it."""
    out = dict(listing)
    out[PACK_PATH] = {"size": 999, "oid": "s256", "lfs": True}
    return out


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
    monkeypatch.setattr(rp, "_list_prefix", lambda *a, **k: _with_pack({}))
    monkeypatch.setattr(rp, "_create_commit_delete", lambda *a, **k: None)
    n = rp.run_delete(PREFIX, st, src_map={})
    assert verify_calls == [PREFIX]
    assert n == 0


def test_current_run_verify_pass_skips_reverify(rp, quiet_gates, monkeypatch):
    st = _state_with_verify(rp, run_id=rp.RUN_ID)
    monkeypatch.setattr(
        rp, "step_hub_verify", lambda *a, **k: pytest.fail("re-verify must not run")
    )
    monkeypatch.setattr(rp, "_list_prefix", lambda *a, **k: _with_pack({}))
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
    monkeypatch.setattr(
        rp, "_list_prefix", _SeqListing([_with_pack(L1), _with_pack(L1), _with_pack({})])
    )
    heads = iter(["h1", "h2", "h3"])
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
    monkeypatch.setattr(rp, "_list_prefix", _SeqListing([_with_pack(L1), _with_pack({})]))
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
    L = _with_pack(
        {
            f"{PREFIX}/a.json": {"size": 1, "oid": "o1", "lfs": False},
            f"{PREFIX}/b.json": {"size": 1, "oid": "o2", "lfs": False},
        }
    )
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


# ── r3: delete re-list revision-pinned to the CAS parent ─────────────────────


def test_delete_relist_pinned_to_cas_parent(rp, quiet_gates, monkeypatch):
    """HEAD is read FIRST, the re-list is pinned to THAT revision, and the CAS
    parent_commit is the SAME sha (review r1 Major: the old flow listed
    unpinned, then read HEAD after — a commit in the gap passed the CAS while
    the delete set derived from the older tree)."""
    order: list[str] = []
    heads = iter(["h-pin-1", "h-pin-2"])

    def fake_head():
        h = next(heads)
        order.append(f"head:{h}")
        return h

    monkeypatch.setattr(rp, "_repo_head_sha", fake_head)
    a = f"{PREFIX}/a.json"
    listings = _SeqListing(
        [_with_pack({a: {"size": 1, "oid": "o1", "lfs": False}}), _with_pack({})]
    )

    def fake_list(prefix, revision=None):
        order.append(f"list:{revision}")
        return listings(prefix)

    monkeypatch.setattr(rp, "_list_prefix", fake_list)
    commits: list[str] = []
    monkeypatch.setattr(rp, "_create_commit_delete", lambda b, h, p: commits.append(h))
    st = _state_with_verify(rp, run_id=rp.RUN_ID)
    n = rp.run_delete(PREFIX, st, src_map={a: {"size": 1, "oid": "o1", "lfs": False}})
    assert order[0] == "head:h-pin-1" and order[1] == "list:h-pin-1"
    assert commits == ["h-pin-1"]  # parent_commit == the SAME pinned sha
    assert order[2] == "head:h-pin-2" and order[3] == "list:h-pin-2"
    assert n == 1  # realized from the listing: a.json gone


def test_pack_missing_or_changed_at_pinned_revision_refuses(rp, quiet_gates, monkeypatch):
    """Per-batch pack-presence guard: a pack removed/rewritten after step 8
    refuses further deletes (review r1 / Codex delete-cas-revision-binding)."""
    monkeypatch.setattr(rp, "_repo_head_sha", lambda: "h1")
    monkeypatch.setattr(rp, "_create_commit_delete", lambda *a, **k: pytest.fail("no delete"))
    st = _state_with_verify(rp, run_id=rp.RUN_ID)
    monkeypatch.setattr(rp, "_list_prefix", lambda *a, **k: {})
    with pytest.raises(rp.VerifyFailure, match="MISSING at pinned revision"):
        rp.run_delete(PREFIX, st, src_map={})
    monkeypatch.setattr(
        rp, "_list_prefix", lambda *a, **k: {PACK_PATH: {"size": 1000, "oid": "s256", "lfs": True}}
    )
    with pytest.raises(rp.VerifyFailure, match="size changed"):
        rp.run_delete(PREFIX, st, src_map={})
    monkeypatch.setattr(
        rp, "_list_prefix", lambda *a, **k: {PACK_PATH: {"size": 999, "oid": "OTHER", "lfs": True}}
    )
    with pytest.raises(rp.VerifyFailure, match="identity changed"):
        rp.run_delete(PREFIX, st, src_map={})


# ── r3: pilot -> fleet gate (plan SS7 gate 7) ─────────────────────────────────


def test_pilot_failure_aborts_fleet_run(rp, monkeypatch):
    calls: list[str] = []

    def fake_run_prefix(prefix, st):
        calls.append(prefix)
        raise rp.VerifyFailure("pilot verification mismatch")

    monkeypatch.setattr(rp, "run_prefix", fake_run_prefix)
    monkeypatch.setattr(rp, "_load_state", lambda: {"prefixes": {}})
    monkeypatch.setattr(rp, "_save_state", lambda *_a, **_k: None)
    rc = rp.main(["run"])
    assert rc == 1
    assert calls == [rp.PILOT_PREFIX]  # fleet prefixes never started


# ── r3: gate-input parse failure fails CLOSED ─────────────────────────────────


def test_gwriter_malformed_events_row_fails_closed(rp, tmp_path, monkeypatch):
    task_dir = tmp_path / "2224"
    task_dir.mkdir()
    (task_dir / "events.jsonl").write_text('{"kind": "epm:progress", "ts": "2026"}\n{broken\n')
    monkeypatch.setattr(rp, "find_task_path", lambda _i: task_dir)
    with pytest.raises(rp.GateRefusal, match="fail closed"):
        rp._fresh_run_signal(2224)


# ── r3: typed not-found semantics (never a "404" substring match) ─────────────


def test_list_prefix_typed_not_found_semantics(rp, monkeypatch):
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    class _Api:
        def __init__(self, exc):
            self.exc = exc

        def list_repo_tree(self, *a, **k):
            raise self.exc

    monkeypatch.setattr(rp, "_get_api", lambda: _Api(EntryNotFoundError("path absent")))
    assert rp._list_prefix(PREFIX) == {}  # the ONE legal empty-listing case
    monkeypatch.setattr(rp, "_get_api", lambda: _Api(RepositoryNotFoundError("404: repo gone")))
    with pytest.raises(RepositoryNotFoundError):
        rp._list_prefix(PREFIX)  # a repo-level 404 is NEVER an empty inventory


# ── r3: step checkpoints are consumed, fingerprinted by src_revision ──────────


def test_step_checkpoint_fingerprinted_by_src_revision(rp):
    rec = {"src_revision": "r2", "steps": {"download": {"done": True, "src_revision": "r2"}}}
    assert rp._step_done(rec, "download")
    rec["src_revision"] = "r3"  # a re-snapshot invalidates every later checkpoint
    assert not rp._step_done(rec, "download")
    assert not rp._step_done(rec, "never_recorded")


# ── r3: pilot extra (plan SS4.6) — shard re-download + member sample ──────────


def test_pilot_extra_pass_and_shard_tamper(rp, tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "STAGE", tmp_path)
    monkeypatch.setattr(rp, "_save_state", lambda *_a, **_k: None)
    files = {f"{PREFIX}/m{i:02d}.json": f'{{"i": {i}}}'.encode() for i in range(25)}
    src_root = tmp_path / f"src_{PREFIX}"
    staged_hashes = {}
    for relpath, payload in files.items():
        fp = src_root / relpath
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(payload)
        staged_hashes[relpath] = hashlib.sha256(payload).hexdigest()
    st = {"prefixes": {PREFIX: {}}}
    out = rp.step_pack(PREFIX, st, {}, sorted(files), staged_hashes, None, write_manifest=False)
    local_shard = tmp_path / f"up_{PREFIX}" / PREFIX / "__packed__" / "shard-00000.tar"

    def fake_dl(repo_id, filename, repo_type, local_dir):
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_shard, dest)
        return str(dest)

    monkeypatch.setattr(rp, "hf_hub_download", fake_dl)
    rp.step_pilot_extra(PREFIX, st, out["index"])  # PASS arm
    assert st["prefixes"][PREFIX]["steps"]["pilot_extra"]["done"]

    def tampering_dl(repo_id, filename, repo_type, local_dir):
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        raw = bytearray(local_shard.read_bytes())
        raw[out["index"][sorted(files)[0]]["offset"]] ^= 0xFF
        dest.write_bytes(bytes(raw))
        return str(dest)

    monkeypatch.setattr(rp, "hf_hub_download", tampering_dl)
    with pytest.raises(rp.VerifyFailure, match="pilot extra"):
        rp.step_pilot_extra(PREFIX, st, out["index"])


# ── env capability pin (plan SS4.0): parent_commit kwarg must exist ──────────


def test_create_commit_has_parent_commit_kwarg():
    from huggingface_hub import HfApi

    assert "parent_commit" in inspect.signature(HfApi.create_commit).parameters
