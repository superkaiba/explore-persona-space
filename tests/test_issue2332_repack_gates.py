"""Pinning tests for the #2332 repack gates (plan v2 SS6, 9 negative controls).

Everything Hub- and process-shaped is mocked/monkeypatched — NO network, no
pgrep against live processes, no RunPod/session subprocesses. The module under
test is loaded from ``scripts/issue2332_repack_prefixes.py`` by path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
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
    """Neutralize the writer gate + state writes for delete-loop tests, and
    stub the head read (r4: the head-first reorder made run_delete read
    `_repo_head_sha` — a LIVE `repo_info` call — on paths several r3 tests
    exercised without a stub). Tests that pin head sequencing override it."""
    monkeypatch.setattr(rp, "gate_writer", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "gate_mover", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "_save_state", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "_repo_head_sha", lambda: "h-quiet")


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
    heads = iter(["h1", "h2", "h3", "h4"])  # h4: the r4 PINNED final-assert listing
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
    heads = iter(["h-pin-1", "h-pin-2", "h-pin-3"])

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
    # r4 (Codex delete-final-revision-binding): the final-assert completion
    # listing is ALSO pinned to a freshly-read head — never an unpinned
    # mutable-HEAD list.
    assert order[4] == "head:h-pin-3" and order[5] == "list:h-pin-3"
    assert "list:None" not in order
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


# ── r4: staged-verify sidecar — verified-only, fingerprinted, re-adjudicated ──


def _digests(data: bytes) -> tuple[str, str]:
    """(sha256, git-blob sha1) — mirrors the script's _dual_digest exactly."""
    return (
        hashlib.sha256(data).hexdigest(),
        hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
    )


def _staged_setup(rp, tmp_path, monkeypatch, units, *, src_revision="r1", a7=None):
    """Build a staged tree + state for step_staged_verify tests.

    ``units``: {path: (payload_bytes, lfs_bool, oid)}. Returns (st, rec, src_map).
    """
    monkeypatch.setattr(rp, "STAGE", tmp_path)
    monkeypatch.setattr(rp, "_save_state", lambda *_a, **_k: None)
    src_root = tmp_path / f"src_{PREFIX}"
    src_map: dict = {}
    for p, (payload, lfs, oid) in units.items():
        fp = src_root / p
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(payload)
        src_map[p] = {"size": len(payload), "oid": oid, "lfs": lfs}
    rec = {"src_revision": src_revision, "steps": {}}
    st = {"prefixes": {PREFIX: rec}}
    if a7 is not None:
        st["blob_id_is_git_sha1"] = a7
    return st, rec, src_map


def test_staged_verify_resumed_bad_unit_refails_every_retry(rp, tmp_path, monkeypatch):
    """The r2 Major, arm 1 (fails pre-fix): a sidecar unit whose STORED digest
    mismatches the snapshot oid must re-FAIL on re-entry — never be waved
    through as 'previously verified'. Both identity arms (LFS sha256, a7-True
    git-sha1); and a CORRECT sidecar resumes with ZERO re-hashing."""
    lfs_payload, nl_payload = b"lfs bytes", b'{"nl": 1}'
    lfs_s256, _ = _digests(lfs_payload)
    _, nl_git = _digests(nl_payload)
    u_lfs, u_nl = f"{PREFIX}/big.bin", f"{PREFIX}/cell.json"
    units = {u_lfs: (lfs_payload, True, lfs_s256), u_nl: (nl_payload, False, nl_git)}
    st, _rec, src_map = _staged_setup(rp, tmp_path, monkeypatch, units, a7=True)
    paths = sorted(units)
    # Arm 1: resumed non-LFS unit with a BAD stored git-sha1 re-fails.
    good_lfs = {"sha256": lfs_s256, "gitsha1": "irrelevant"}
    rp._write_staged_hashes(
        PREFIX, "r1", {u_lfs: good_lfs, u_nl: {"sha256": "x", "gitsha1": "BAD"}}
    )
    with pytest.raises(rp.VerifyFailure, match="git-blob sha1 mismatch on resumed unit"):
        rp.step_staged_verify(PREFIX, st, src_map, paths)
    # Arm 2: resumed LFS unit with a BAD stored sha256 re-fails.
    good_nl = {"sha256": _digests(nl_payload)[0], "gitsha1": nl_git}
    rp._write_staged_hashes(PREFIX, "r1", {u_lfs: {"sha256": "BAD", "gitsha1": "y"}, u_nl: good_nl})
    with pytest.raises(rp.VerifyFailure, match="LFS sha256 mismatch on resumed unit"):
        rp.step_staged_verify(PREFIX, st, src_map, paths)
    # Arm 3: a CORRECT sidecar resumes without ANY re-hash (stored-digest
    # adjudication only) and returns the stored sha256s.
    rp._write_staged_hashes(PREFIX, "r1", {u_lfs: good_lfs, u_nl: good_nl})
    monkeypatch.setattr(
        rp, "_dual_digest", lambda _p: pytest.fail("resumed units must not re-hash")
    )
    staged = rp.step_staged_verify(PREFIX, st, src_map, paths)
    assert staged == {u_lfs: good_lfs["sha256"], u_nl: good_nl["sha256"]}


def test_staged_verify_sidecar_revision_mismatch_or_legacy_schema_rehashes(
    rp, tmp_path, monkeypatch
):
    """The r2 Major, arm 2: a sidecar fingerprinted to a DIFFERENT src_revision
    — or the legacy unfingerprinted {path: sha256} schema — is DISCARDED and
    the unit is fully re-hashed + re-verified under the CURRENT snapshot."""
    payload = b'{"v": 7}'
    s256, git1 = _digests(payload)
    u = f"{PREFIX}/cell.json"
    st, rec, src_map = _staged_setup(
        rp, tmp_path, monkeypatch, {u: (payload, False, git1)}, a7=True
    )
    # Arm 1: cross-revision sidecar (garbage digests under r0) is discarded.
    rp._write_staged_hashes(PREFIX, "r0", {u: {"sha256": "stale", "gitsha1": "stale"}})
    staged = rp.step_staged_verify(PREFIX, st, src_map, [u])
    assert staged == {u: s256}  # freshly re-hashed, verified against r1's oid
    reloaded = rp._load_staged_sidecar(PREFIX, "r1")
    assert reloaded == {u: {"sha256": s256, "gitsha1": git1}}
    # Arm 2: legacy v1 schema (bare {path: sha256}) is discarded the same way.
    rec["steps"].clear()
    rp._staged_hashes_path(PREFIX).write_text(json.dumps({u: "bare-legacy-sha"}))
    staged = rp.step_staged_verify(PREFIX, st, src_map, [u])
    assert staged == {u: s256}


def test_staged_verify_pending_units_never_persisted_before_verdict(rp, tmp_path, monkeypatch):
    """Review r2 invariant: a unit enters the sidecar ONLY on verify PASS.
    Under an undecided A7, fresh non-LFS digests are held PENDING; a MIXED
    batch verdict raises and the sidecar must contain NEITHER unit. The
    all-match verdict then persists both."""
    p1, p2 = b'{"a": 1}', b'{"b": 2}'
    _, g1 = _digests(p1)
    u1, u2 = f"{PREFIX}/a.json", f"{PREFIX}/b.json"
    st, _rec, src_map = _staged_setup(
        rp, tmp_path, monkeypatch, {u1: (p1, False, g1), u2: (p2, False, "NOT-A-SHA1")}
    )
    with pytest.raises(rp.VerifyFailure, match="MIXED"):
        rp.step_staged_verify(PREFIX, st, src_map, [u1, u2])
    assert rp._load_staged_sidecar(PREFIX, "r1") is None  # nothing unverified persisted
    # All-match verdict: fix u2's oid -> both persist, a7 resolves True.
    src_map[u2]["oid"] = _digests(p2)[1]
    staged = rp.step_staged_verify(PREFIX, st, src_map, [u1, u2])
    assert set(staged) == {u1, u2}
    assert st["blob_id_is_git_sha1"] is True
    assert set(rp._load_staged_sidecar(PREFIX, "r1")) == {u1, u2}


def test_download_fetch_evicts_sidecar_and_invalidates_downstream(rp, tmp_path, monkeypatch):
    """r4 (resume-cache-identity-safety): a unit that FETCHES fresh bytes is
    evicted from the verified-units sidecar and every downstream step
    checkpoint is invalidated; a size-matching present unit is skipped and
    keeps its sidecar entry."""
    monkeypatch.setattr(rp, "STAGE", tmp_path)
    monkeypatch.setattr(rp, "_save_state", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "gate_writer", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "gate_mover", lambda *_a, **_k: None)
    u_keep, u_fetch = f"{PREFIX}/keep.json", f"{PREFIX}/refetch.json"
    dest = tmp_path / f"src_{PREFIX}"
    (dest / PREFIX).mkdir(parents=True)
    (dest / u_keep).write_bytes(b"12345")  # size 5 == src_map -> skipped
    (dest / u_fetch).write_bytes(b"x")  # size 1 != 5 -> refetch
    src_map = {
        u_keep: {"size": 5, "oid": "o1", "lfs": False},
        u_fetch: {"size": 5, "oid": "o2", "lfs": False},
    }
    rp._write_staged_hashes(
        PREFIX,
        "r1",
        {
            u_keep: {"sha256": "sk", "gitsha1": "gk"},
            u_fetch: {"sha256": "sf", "gitsha1": "gf"},
        },
    )
    rec = {
        "src_revision": "r1",
        "steps": {
            "staged_verify": {"done": True, "src_revision": "r1"},
            "pack": {"done": True, "src_revision": "r1", "k": 2},
            "upload": {"done": True, "src_revision": "r1"},
            "hub_verify": {"pass": True},
        },
    }
    st = {"prefixes": {PREFIX: rec}}

    def fake_dl(*, repo_id, filename, repo_type, revision, local_dir, etag_timeout):
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"54321")
        return str(out)

    monkeypatch.setattr(rp, "hf_hub_download", fake_dl)
    rp.step_download(PREFIX, st, src_map, sorted(src_map))
    sidecar = rp._load_staged_sidecar(PREFIX, "r1")
    assert u_keep in sidecar and u_fetch not in sidecar
    for later in ("staged_verify", "pack", "local_verify", "upload", "hub_verify"):
        assert later not in rec["steps"], f"stale {later} checkpoint survived a fresh fetch"
    assert rec["steps"]["download"]["done"] is True


def test_pack_execute_resets_whole_mode_record_and_invalidates_downstream(
    rp, tmp_path, monkeypatch
):
    """r4: re-packing REBUILDS the artifacts — the whole-mode packed_files
    record is reset (no stale entry can wedge hub_verify/_assert_pack_present)
    and downstream checkpoints are invalidated BEFORE the first shard write.
    Runs the REAL step_pack body on a tiny staged tree."""
    monkeypatch.setattr(rp, "STAGE", tmp_path)
    monkeypatch.setattr(rp, "_save_state", lambda *_a, **_k: None)
    files = {f"{PREFIX}/m{i}.json": f'{{"i": {i}}}'.encode() for i in range(3)}
    src_root = tmp_path / f"src_{PREFIX}"
    staged = {}
    for relpath, payload in files.items():
        fp = src_root / relpath
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(payload)
        staged[relpath] = hashlib.sha256(payload).hexdigest()
    rec = {
        "src_revision": "r1",
        "snapshot_ts": "t0",
        "packed_files": {f"{PREFIX}/__packed__/STALE.tar": {"size": 1}},
        "steps": {
            "local_verify": {"done": True, "src_revision": "r1"},
            "upload": {"done": True, "src_revision": "r1"},
            "hub_verify": {"pass": True},
        },
    }
    st = {"prefixes": {PREFIX: rec}}
    out = rp.step_pack(PREFIX, st, {}, sorted(files), staged, None, write_manifest=True)
    assert f"{PREFIX}/__packed__/STALE.tar" not in rec["packed_files"]
    assert set(rec["packed_files"]) == set(out["packed_files"])
    for later in ("local_verify", "upload", "hub_verify"):
        assert later not in rec["steps"], f"stale {later} checkpoint survived a re-pack"
    assert rec["steps"]["pack"]["src_revision"] == "r1"


# ── r4: hub_verify A7-None routes non-LFS sidecars to the byte-compare arm ────


def test_hub_verify_a7_undecided_routes_to_byte_compare(rp, monkeypatch):
    idx_path = f"{PREFIX}/__packed__/index.json"
    rec = {"packed_files": {idx_path: {"size": 5, "sha256": "S", "gitsha1": "G"}}, "steps": {}}
    listing = {idx_path: {"size": 5, "oid": "NOT-G", "lfs": False}}
    monkeypatch.setattr(rp, "gate_mover", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "_save_state", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "_list_prefix", lambda *a, **k: listing)
    compared: list[str] = []

    def fake_compare(prefix: str, repo_path: str, exp: dict) -> None:
        compared.append(repo_path)

    monkeypatch.setattr(rp, "_redownload_byte_compare", fake_compare)
    # A7 undecided: byte-compare, never an unproven gitsha1 equality.
    st = {"prefixes": {PREFIX: rec}}
    rp.step_hub_verify(PREFIX, st)
    assert compared == [idx_path]
    assert rec["steps"]["hub_verify"]["pass"] is True
    # A7 proven True: the cheap gitsha1 equality applies and mismatches raise.
    rec["steps"].clear()
    st["blob_id_is_git_sha1"] = True
    with pytest.raises(rp.VerifyFailure, match="git-blob sha1 mismatch"):
        rp.step_hub_verify(PREFIX, st)


# ── r4: run_prefix consumption predicates (PRODUCTION path, steps stubbed) ────


def _run_prefix_scenario(rp, tmp_path, monkeypatch, *, seed_state):
    """Drive the REAL run_prefix with signature-conformant step stubs that
    record call order (each step's real body is covered by its own tests).
    ``seed_state(st, rec)`` arranges checkpoints/artifacts. Returns call list."""
    monkeypatch.setattr(rp, "STAGE", tmp_path)
    monkeypatch.setattr(rp, "_save_state", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "gate_writer", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "gate_mover", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "_staging_headroom_assert", lambda need_bytes: None)
    src_map = {
        f"{PREFIX}/a.json": {"size": 5, "oid": "oa", "lfs": False},
        f"{PREFIX}/b.json": {"size": 5, "oid": "ob", "lfs": False},
    }
    (tmp_path / f"srcmap_{PREFIX}.json").write_text(json.dumps(src_map))
    rec = {
        "src_revision": "r1",
        "snapshot_ts": "t0",
        "n_files": 2,
        "total_bytes": 10,
        "steps": {"snapshot": {"done": True}},
    }
    st = {"prefixes": {PREFIX: rec}}
    seed_state(st, rec)
    calls: list[str] = []

    def fake_download(prefix: str, st: dict, src_map: dict, paths: list[str]) -> None:
        calls.append("download")

    def fake_staged_verify(prefix: str, st: dict, src_map: dict, paths: list[str]):
        calls.append("staged_verify")
        return {p: "sha" for p in paths}

    def fake_pack(prefix, st, src_map, paths, staged_hashes, chunk_idx, write_manifest):
        calls.append("pack")
        return {"index": {p: {} for p in paths}, "k": 3, "packed_files": {}}

    def fake_local_verify(prefix: str, st: dict, src_map: dict, paths: list[str], index) -> None:
        calls.append("local_verify")

    def fake_upload(prefix: str, st: dict, k_expected: int) -> None:
        calls.append(f"upload(k={k_expected})")

    def fake_hub_verify(prefix: str, st: dict) -> None:
        calls.append("hub_verify")

    def fake_pilot_extra(prefix: str, st: dict, index) -> None:
        calls.append("pilot_extra")

    def fake_run_delete(
        prefix, st, *, src_map=None, restrict=None, final_assert=True, batch_size=None
    ):
        calls.append("run_delete")
        return 7

    monkeypatch.setattr(rp, "step_download", fake_download)
    monkeypatch.setattr(rp, "step_staged_verify", fake_staged_verify)
    monkeypatch.setattr(rp, "step_pack", fake_pack)
    monkeypatch.setattr(rp, "step_local_verify", fake_local_verify)
    monkeypatch.setattr(rp, "step_upload", fake_upload)
    monkeypatch.setattr(rp, "step_hub_verify", fake_hub_verify)
    monkeypatch.setattr(rp, "step_pilot_extra", fake_pilot_extra)
    monkeypatch.setattr(rp, "run_delete", fake_run_delete)
    assert rp.run_prefix(PREFIX, st) == "done"
    return calls


def test_run_prefix_fresh_runs_every_step(rp, tmp_path, monkeypatch):
    calls = _run_prefix_scenario(rp, tmp_path, monkeypatch, seed_state=lambda st, rec: None)
    assert calls == [
        "download",
        "staged_verify",
        "pack",
        "local_verify",
        "upload(k=3)",
        "hub_verify",
        "pilot_extra",
        "run_delete",
    ]


def test_run_prefix_resume_always_redownloads_and_reuploads(rp, tmp_path, monkeypatch):
    """r4 (upload-checkpoint-retry-wedge + resume identity): with EVERY
    checkpoint recorded and all artifacts on disk, download and upload are
    STILL called (idempotent completeness sweeps; upload re-attempts until
    hub_verify PASSes) while staged_verify/pack/local_verify are consumed."""

    def seed(st, rec):
        a, b = f"{PREFIX}/a.json", f"{PREFIX}/b.json"
        rp._write_staged_hashes(
            PREFIX,
            "r1",
            {
                a: {"sha256": "sa", "gitsha1": "ga"},
                b: {"sha256": "sb", "gitsha1": "gb"},
            },
        )
        packed_dir = tmp_path / f"up_{PREFIX}" / PREFIX / "__packed__"
        packed_dir.mkdir(parents=True)
        (packed_dir / "index.json").write_text("{}")
        (packed_dir / "shard-00000.tar").write_bytes(b"tar")
        rec["packed_files"] = {
            f"{PREFIX}/__packed__/shard-00000.tar": {"size": 3},
            f"{PREFIX}/__packed__/index.json": {"size": 2},
        }
        rec["steps"].update(
            {
                "staged_verify": {"done": True, "src_revision": "r1"},
                "pack": {"done": True, "src_revision": "r1", "k": 2},
                "local_verify": {"done": True, "src_revision": "r1"},
                "upload": {"done": True, "src_revision": "r1"},
            }
        )

    calls = _run_prefix_scenario(rp, tmp_path, monkeypatch, seed_state=seed)
    assert calls == ["download", "upload(k=2)", "hub_verify", "pilot_extra", "run_delete"]


def test_run_prefix_incomplete_sidecar_or_empty_pack_record_reruns_steps(rp, tmp_path, monkeypatch):
    """r4: the staged-verify skip requires the fingerprinted sidecar to COVER
    every unit, and the pack skip requires a NON-EMPTY packed_files record —
    a vacuous all() over an empty dict must not consume the checkpoint."""

    def seed(st, rec):
        # Sidecar covers only ONE of the two units; pack record done but
        # packed_files EMPTY (the pre-r4 vacuous-truth shape) + index on disk.
        rp._write_staged_hashes(
            PREFIX, "r1", {f"{PREFIX}/a.json": {"sha256": "sa", "gitsha1": "ga"}}
        )
        packed_dir = tmp_path / f"up_{PREFIX}" / PREFIX / "__packed__"
        packed_dir.mkdir(parents=True)
        (packed_dir / "index.json").write_text("{}")
        rec["packed_files"] = {}
        rec["steps"].update(
            {
                "staged_verify": {"done": True, "src_revision": "r1"},
                "pack": {"done": True, "src_revision": "r1", "k": 2},
            }
        )

    calls = _run_prefix_scenario(rp, tmp_path, monkeypatch, seed_state=seed)
    assert "staged_verify" in calls and "pack" in calls


# ── r4: chunked-mode pilot claim + main-loop duty routing ─────────────────────


def _mini_hub_env(rp, tmp_path, monkeypatch, files: dict[str, bytes]) -> dict:
    """A tiny in-memory fake Hub + real staging tree so the REAL run_prefix
    pipeline (snapshot, download, staged-verify, pack, local-verify, upload,
    hub-verify, pilot-extra, delete) runs end-to-end. Fakes ONLY the network
    seams (listing / head / download / upload / delete-commit) and the
    mount-bound headroom assert; every pipeline body executes for real.
    ``env['deletes']`` records (batch, duty_satisfied_at_delete_time)."""
    hub: dict[str, dict] = {}
    payloads = dict(files)
    for p, data in payloads.items():
        hub[p] = {"size": len(data), "oid": hashlib.sha256(data).hexdigest(), "lfs": True}
    env: dict = {"hub": hub, "deletes": [], "st": None, "head": 0}
    monkeypatch.setattr(rp, "STAGE", tmp_path)
    monkeypatch.setattr(rp, "_save_state", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "gate_writer", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "gate_mover", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "_staging_headroom_assert", lambda need_bytes: None)
    monkeypatch.setattr(rp, "_fresh_repo_count", lambda: 100)
    # Small thresholds so a ~150-byte prefix runs CHUNKED (2 chunks) while a
    # ~40-byte prefix stays WHOLE — thresholds, not seams: plan_chunks and
    # every consumer run their real bodies against them.
    monkeypatch.setattr(rp, "VM_FOOTPRINT_MAX_BYTES", 100)
    monkeypatch.setattr(rp, "CHUNK_STAGED_MAX_BYTES", 200)

    def fake_head() -> str:
        env["head"] += 1
        return f"h{env['head']}"

    monkeypatch.setattr(rp, "_repo_head_sha", fake_head)

    def fake_list(prefix: str, revision=None) -> dict:
        return {p: dict(e) for p, e in hub.items() if p.startswith(prefix + "/")}

    monkeypatch.setattr(rp, "_list_prefix", fake_list)

    def fake_dl(*, repo_id, filename, repo_type, revision=None, local_dir=None, etag_timeout=None):
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        if "/__packed__/" in filename:
            src = tmp_path / f"up_{filename.split('/', 1)[0]}" / filename
            shutil.copyfile(src, out)
        else:
            out.write_bytes(payloads[filename])
        return str(out)

    monkeypatch.setattr(rp, "hf_hub_download", fake_dl)

    class _Api:
        def upload_large_folder(
            self,
            *,
            repo_id,
            folder_path,
            repo_type,
            ignore_patterns,
            num_workers,
            print_report_every,
        ) -> None:
            root = Path(folder_path)
            for f in sorted(root.rglob("*")):
                if f.is_file():
                    s256, _g = rp._dual_digest(f)
                    hub[f.relative_to(root).as_posix()] = {
                        "size": f.stat().st_size,
                        "oid": s256,
                        "lfs": True,
                    }

    monkeypatch.setattr(rp, "_get_api", lambda: _Api())

    def fake_commit(batch, head_sha, prefix) -> None:
        st_ref = env["st"]
        env["deletes"].append((sorted(batch), bool(st_ref.get("chunked_pilot_extra_done"))))
        for p in batch:
            hub.pop(p, None)

    monkeypatch.setattr(rp, "_create_commit_delete", fake_commit)
    return env


def _chunked_files(prefix: str) -> dict[str, bytes]:
    """4 files / 2 subdirs, ~38 B each — CHUNKED (2 chunks) under the
    mini-hub thresholds (2*152 > 100; per-subdir ~76 <= 100 max-chunk)."""
    return {
        f"{prefix}/sub{i}/f{j}.json": f'{{"chunk": {i}, "row": {j}, "pad": "0123456789"}}'.encode()
        for i in range(2)
        for j in range(2)
    }


def _whole_files(prefix: str) -> dict[str, bytes]:
    """2 files, ~19 B each — WHOLE mode under the mini-hub thresholds."""
    return {f"{prefix}/g{j}.json": f'{{"a": {j}, "p": "xx"}}'.encode() for j in range(2)}


def test_chunked_pilot_duty_joint_end_to_end(rp, tmp_path, monkeypatch):
    """The r3 reconciler's JOINT pin (chunked-pilot-transfer-state): a stale
    holder A — legacy ``st['chunked_pilot']`` claim + rec mode 'chunked' +
    gate-skipped — must NOT strand the gate-7 duty: driving the REAL
    run_prefix for chunked B (fakes only at network seams), B's chunk 0
    RUNS the real step_pilot_extra and records the persisted satisfaction
    flag BEFORE any of B's deletes commit. Fails pre-fix: the r5-removed
    ``_claim_chunked_pilot(st, B)`` returned False under A's claim, so B
    skipped pilot-extra while the main loop marked the duty satisfied."""
    A, B = "issue2224_screening", "issue1434_writingstyle"
    env = _mini_hub_env(rp, tmp_path, monkeypatch, _chunked_files(B))
    st: dict = {
        "prefixes": {A: {"mode": "chunked", "skipped": "gate trip (prior run)"}},
        "chunked_pilot": A,  # the stale legacy claim, held by a non-done holder
    }
    env["st"] = st
    assert rp.run_prefix(B, st) == "done"
    # The ONE persisted state: duty satisfied by B's REAL chunk-0 pilot-extra.
    assert st["chunked_pilot_extra_done"] == B
    assert st["prefixes"][B]["steps"]["pilot_extra"]["done"] is True
    assert not rp._chunked_pilot_duty_pending(st)
    # Ordering: the FIRST delete commit already saw the duty satisfied.
    assert env["deletes"], "chunk deletes must have run"
    assert env["deletes"][0][1] is True, "deletes ran before pilot-extra PASSed"
    # The legacy claim key is inert — untouched, and it did not block B.
    assert st["chunked_pilot"] == A
    # Mini-hub end state sanity: B's originals deleted, packed files present.
    assert not [p for p in env["hub"] if p.startswith(B + "/") and "/__packed__/" not in p]
    assert [p for p in env["hub"] if p.startswith(f"{B}/__packed__/")]


def test_chunked_pilot_duty_mode_flip_holder_replanned_whole(rp, tmp_path, monkeypatch):
    """The reconciler's mode-flip variant: a holder re-planned WHOLE after a
    re-snapshot runs to done WITHOUT pilot-extra (whole-mode pilot-extra is
    PILOT_PREFIX-only), so the duty stays pending and the NEXT chunked
    prefix's chunk 0 actually runs it — through the REAL run_prefix both
    times."""
    A, B = "issue2224_screening", "issue1434_writingstyle"
    files = {**_whole_files(A), **_chunked_files(B)}
    env = _mini_hub_env(rp, tmp_path, monkeypatch, files)
    st: dict = {"prefixes": {}, "chunked_pilot": A}  # stale claim from a chunked past
    env["st"] = st
    assert rp.run_prefix(A, st) == "done"
    assert st["prefixes"][A]["mode"] == "whole"
    assert "pilot_extra" not in st["prefixes"][A]["steps"]
    assert rp._chunked_pilot_duty_pending(st)  # a whole-mode done never satisfies it
    assert rp.run_prefix(B, st) == "done"
    assert st["chunked_pilot_extra_done"] == B
    assert st["prefixes"][B]["steps"]["pilot_extra"]["done"] is True


def test_sizing_mode_hints_tokens(rp, tmp_path):
    """r3 minor 3: 'refused-drift' (and any unrecognized token) maps to
    'unknown' — chunked-POSSIBLE — never silently 'whole'; malformed rows
    are skipped; an absent file is {}."""
    p = tmp_path / "sizing.json"
    p.write_text(
        json.dumps(
            {
                "prefixes": {
                    "a": {"mode": "whole"},
                    "b": {"mode": "chunked(3)"},
                    "c": {"mode": "refused-drift"},
                    "d": {"mode": 7},
                    "e": "notadict",
                }
            }
        )
    )
    assert rp._sizing_mode_hints(p) == {"a": "whole", "b": "chunked", "c": "unknown"}
    assert rp._sizing_mode_hints(tmp_path / "absent.json") == {}


def _main_run_with_fakes(rp, monkeypatch, per_prefix):
    """Drive `main(["run"])` with a signature-conformant run_prefix fake:
    ``per_prefix[prefix]`` is a result string, an exception to raise, or a
    callable ``(prefix, st) -> result`` (to mutate st the way the REAL
    run_prefix does — e.g. record the pilot-extra satisfaction flag)."""
    calls: list[str] = []

    def fake_run_prefix(prefix: str, st: dict) -> str:
        calls.append(prefix)
        outcome = per_prefix.get(prefix, "done")
        if callable(outcome) and not isinstance(outcome, Exception):
            outcome = outcome(prefix, st)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(rp, "run_prefix", fake_run_prefix)
    monkeypatch.setattr(rp, "_load_state", lambda: {"prefixes": {}})
    monkeypatch.setattr(rp, "_save_state", lambda *_a, **_k: None)
    rc = rp.main(["run"])
    return rc, calls


def test_main_chunked_pilot_duty_transfers_past_gate_skips_and_aborts_on_failure(rp, monkeypatch):
    """Codex r2 chunked-pilot-contract: a chunked prefix that never RAN its
    pipeline (gate skip — rec['mode'] never assigned; mode known only via the
    sizing hint) passes the duty to the NEXT chunked prefix; a chunked prefix
    that ran and FAILED aborts the run before larger chunked prefixes."""
    order = list(rp.PREFIXES)
    skipped_chunked, failing_chunked = order[2], order[4]
    monkeypatch.setattr(
        rp,
        "_sizing_mode_hints",
        lambda: {skipped_chunked: "chunked", failing_chunked: "chunked"},
    )
    rc, calls = _main_run_with_fakes(
        rp,
        monkeypatch,
        {
            skipped_chunked: "skipped",
            failing_chunked: rp.VerifyFailure("chunk 0 verify mismatch"),
        },
    )
    assert rc == 1
    assert calls == order[:5]  # aborted AT the failing chunked pilot, not before


def test_main_chunked_pilot_satisfied_by_first_done_chunked(rp, monkeypatch):
    """Once a chunked prefix records the persisted pilot-extra satisfaction
    flag (what the REAL run_prefix writes on a chunked chunk-0 pilot-extra
    PASS — proven end-to-end by the joint test), a LATER chunked failure does
    not trip the chunked-pilot abort (the ordinary incomplete handling runs)."""
    order = list(rp.PREFIXES)
    first_chunked, later_chunked = order[2], order[4]
    monkeypatch.setattr(
        rp,
        "_sizing_mode_hints",
        lambda: {first_chunked: "chunked", later_chunked: "chunked"},
    )

    def done_and_satisfy(prefix: str, st: dict) -> str:
        st["chunked_pilot_extra_done"] = prefix
        return "done"

    rc, calls = _main_run_with_fakes(
        rp,
        monkeypatch,
        {
            first_chunked: done_and_satisfy,
            later_chunked: rp.VerifyFailure("late chunk failure"),
        },
    )
    assert rc == 1  # one incomplete prefix -> nonzero, but NO early abort:
    assert calls == order  # every prefix still ran


def test_main_chunked_done_without_pilot_extra_does_not_satisfy_duty(rp, monkeypatch):
    """ONE state machine (r3 reconciler): a chunked prefix reporting 'done'
    WITHOUT the persisted pilot-extra satisfaction flag (legacy/resumed state
    that never ran step_pilot_extra) does NOT satisfy the gate-7 duty — the
    flag is the ONLY satisfaction record, so a later unpiloted chunked failure
    still aborts. Fails pre-fix: the r4 loop-local tracker marked the duty
    satisfied on ANY chunked done."""
    order = list(rp.PREFIXES)
    first_chunked, later_chunked = order[2], order[4]
    monkeypatch.setattr(
        rp,
        "_sizing_mode_hints",
        lambda: {first_chunked: "chunked", later_chunked: "chunked"},
    )
    rc, calls = _main_run_with_fakes(
        rp,
        monkeypatch,
        {
            first_chunked: "done",  # done, but NO flag recorded
            later_chunked: rp.VerifyFailure("unpiloted chunk failure"),
        },
    )
    assert rc == 1
    assert calls == order[:5]  # aborted AT the failing chunked prefix


def test_main_unknown_mode_hint_treated_chunked_possible(rp, monkeypatch):
    """r3 minor 3: a 'refused-drift' sizing row maps to 'unknown' (see
    test_sizing_mode_hints_tokens) and the duty check treats 'unknown' as
    chunked-POSSIBLE — a prefix that ran and FAILED before rec['mode'] was
    assigned aborts while the duty is pending, instead of silently reading
    as whole-mode."""
    order = list(rp.PREFIXES)
    failing = order[3]
    monkeypatch.setattr(rp, "_sizing_mode_hints", lambda: {failing: "unknown"})
    rc, calls = _main_run_with_fakes(
        rp, monkeypatch, {failing: rp.VerifyFailure("died before mode assignment")}
    )
    assert rc == 1
    assert calls == order[:4]  # aborted AT the unknown-mode failure


# ── r4: sizing drift refusal (plan SS4.4.4 fail-loud control flow) ────────────


def test_sizing_drift_refuses_prefix_and_records_clean_recheck(rp, tmp_path, monkeypatch):
    """Codex r2 sizing-drift-fail-loud: drift re-runs the G-writer four-way and
    a refusal REFUSES the prefix (recorded row, run continues); a clean
    re-check records 'writer-recheck-clean' — never a bare proceed-on-log."""
    monkeypatch.setattr(rp, "gate_mover", lambda *_a, **_k: None)
    monkeypatch.setattr(rp, "_mover_done_count_advisory", lambda: "advisory")
    monkeypatch.setattr(rp, "_fresh_repo_count", lambda: 123_456)
    # Tiny fake listings for every prefix: len != inventory => drift everywhere.
    listing = {"a.json": {"size": 1, "oid": "o", "lfs": False}}
    monkeypatch.setattr(
        rp,
        "_list_prefix",
        lambda prefix, revision=None: {f"{prefix}/{k}": v for k, v in listing.items()},
    )

    def fake_writer(prefix: str) -> None:
        if prefix == PREFIX:
            raise rp.GateRefusal("G-writer: fresh run signal on owning task")

    monkeypatch.setattr(rp, "gate_writer", fake_writer)
    rc = rp.cmd_sizing(argparse.Namespace(out=str(tmp_path)))
    assert rc == 0
    sizing = json.loads((tmp_path / "sizing.json").read_text())
    refused = sizing["prefixes"][PREFIX]
    assert refused["mode"] == "refused-drift"
    assert "fresh run signal" in refused["drift_refused"]
    assert "k" not in refused  # no shard plan is emitted for a refused prefix
    clean = sizing["prefixes"]["issue667_alllayer"]
    assert clean["drift_writer_recheck"] == "writer-recheck-clean"
    assert clean["mode"] == "whole" and "k" in clean
