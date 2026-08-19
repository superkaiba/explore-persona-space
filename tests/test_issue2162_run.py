"""Issue #2162 driver — CPU pins for the pure scheduling/queue/gate helpers.

Covers the plan §4.6 mechanical gates that are testable without a GPU:

- block enumeration (234 = 39 cells x 2 slots x 3 arms; 42,120 rollouts),
- the shared work-conserving CLAIM-FILE queue (O_CREAT-exclusive claims,
  stale-claim reclamation by dead-pid and by cross-host age, live-same-host
  claims never stolen, tolerant stolen-claim release — r1 M1),
- regime-fingerprint resume refusal (a mismatched done-file RAISES, never a
  silent skip) + width-keyed sharded resume (r1 M2),
- the pilot gate's projection arithmetic at the THREADED width + designed rc
  (r1 C2), --force as resume-only (r1 M3),
- pilot-vs-production state on a SHARED out-root (r1 C1),
- the dispatcher ``all`` chain wiring (pilot -> gate3 -> grid ->
  margin-opportunistic -> upload; stage2 fan-out case) (r1 C4/M4/M9,
  r2 MAJOR 1/MINOR 1) + the sentinel ``margin_deferred`` flag,
- corrupt claim records read DEAD (r2 H2),
- the smoke slice's per-arm-class coverage.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2162_run as R  # noqa: E402

from explore_persona_space.experiments.issue2162 import bank2162 as B  # noqa: E402


@pytest.fixture(scope="module")
def pairs():
    return B.build_pairs()


@pytest.fixture()
def cfg(tmp_path):
    args = R.parse_args(
        [
            "--phase",
            "grid",
            "--out-root",
            str(tmp_path / "out"),
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    )
    return R.build_config(args)


# ── block enumeration ────────────────────────────────────────────────


def test_enumerate_blocks_full_grid(pairs):
    blocks = R.enumerate_blocks(pairs)
    assert len(blocks) == 234
    assert all(b.n_pairs == 36 for b in blocks)
    keys = {b.key for b in blocks}
    assert len(keys) == 234
    totals = R.grid_totals(blocks, R.GRID_DRAWS)
    assert totals["cells_total"] == 234 * 36 == 8424
    assert totals["rollouts_total"] == 8424 * 5 == 42120  # plan §4.3


def test_fork_base_constant_flips():
    """Plan §4.6 r1 gate: the fork flips the parent's grid constants."""
    assert R.GRID_TEMPERATURE == 1.0  # parent: 0.0 (greedy)
    assert R.MAX_NEW_TOKENS == 2048  # parent: 1024
    assert R.GRID_DRAWS == 5  # parent grid had NO per-pair draw seam


def test_smoke_blocks_cover_arm_classes(pairs):
    cells = R.smoke_cells()
    classes = {B.CARRIER_CLASS[B.base_type_of(c)] for c in cells}
    assert {"P", "P12", "E", "ICL", "QC"} <= classes
    assert {"query_content", "persona_role_header", "filler_swap", "language_implied"} <= set(cells)
    for prefix in ("conflict_", "recency_", "load_"):
        assert any(c.startswith(prefix) for c in cells), prefix
    blocks = R.smoke_blocks(pairs)
    assert len(blocks) == len(cells) * len(R.SLOTS) * len(R.ARMS)
    assert all(b.n_pairs == R.SMOKE_PAIRS_PER_CELL for b in blocks)


# ── claim-file queue (gate 2/3) ──────────────────────────────────────


def _mkblock(cell: str = "instr_format") -> R.Block:
    return R.Block(cell, "ce", "steered", ("p1", "p2"))


def test_try_claim_exclusive(tmp_path):
    cdir = tmp_path / "claims"
    block = _mkblock()
    assert R.try_claim(cdir, block, 0, "tok-a") is True
    # A live same-host claim is NOT reclaimable.
    assert R.try_claim(cdir, block, 1, "tok-b") is False
    R.release_claim(cdir, block, "tok-a")
    assert not (cdir / f"{block.slug}.claim").exists()


def test_try_claim_reclaims_dead_pid(tmp_path):
    cdir = tmp_path / "claims"
    cdir.mkdir(parents=True)
    block = _mkblock()
    # Synthesized stale claim: same host, provably dead pid, fresh timestamp.
    import socket

    (cdir / f"{block.slug}.claim").write_text(
        json.dumps(
            {
                "key": block.key,
                "pid": 2**22 + 12345,  # beyond pid_max on this box
                "host": socket.gethostname(),
                "worker_index": 3,
                "ts": time.time(),
                "token": "dead-owner",
            }
        )
    )
    assert R.try_claim(cdir, block, 0, "tok-new") is True
    rec = json.loads((cdir / f"{block.slug}.claim").read_text())
    assert rec["token"] == "tok-new"


def test_try_claim_reclaims_by_age(tmp_path, monkeypatch):
    cdir = tmp_path / "claims"
    cdir.mkdir(parents=True)
    block = _mkblock()
    (cdir / f"{block.slug}.claim").write_text(
        json.dumps(
            {
                "key": block.key,
                "pid": 1,  # pid 1 is alive but on a DIFFERENT host below
                "host": "some-other-pod",
                "worker_index": 0,
                "ts": time.time() - 10 * 3600,
                "token": "crashed-owner",
            }
        )
    )
    assert R.try_claim(cdir, block, 0, "tok-new") is True


def test_try_claim_unparseable_fails_loud(tmp_path):
    cdir = tmp_path / "claims"
    cdir.mkdir(parents=True)
    block = _mkblock()
    (cdir / f"{block.slug}.claim").write_text("{not json")
    with pytest.raises(RuntimeError, match="unparseable claim"):
        R.try_claim(cdir, block, 0, "tok")


def test_try_claim_live_same_host_old_claim_not_stolen(tmp_path):
    """r1 M1: same-host pid probe is authoritative in BOTH directions — a
    LIVE owner is never stolen by age (blocks have no mid-block heartbeat,
    so a >4h block must not lose its claim mid-run)."""
    import socket

    cdir = tmp_path / "claims"
    cdir.mkdir(parents=True)
    block = _mkblock()
    (cdir / f"{block.slug}.claim").write_text(
        json.dumps(
            {
                "key": block.key,
                "pid": os.getpid(),  # provably alive: our own pid
                "host": socket.gethostname(),
                "worker_index": 1,
                "ts": time.time() - 2 * 3600,  # older than any age fallback would like
                "token": "live-owner",
            }
        )
    )
    assert R.try_claim(cdir, block, 0, "tok-thief") is False
    rec = json.loads((cdir / f"{block.slug}.claim").read_text())
    assert rec["token"] == "live-owner"


def test_release_claim_stolen_is_tolerated(tmp_path, caplog):
    """r1 M1: a stolen claim at release time (vanished, or another worker's
    token) is a LOUD error log, never an assert — the done-file landed
    atomically, so the steal must not kill this worker's whole queue loop."""
    import logging

    cdir = tmp_path / "claims"
    block = _mkblock()
    # Vanished claim: no raise.
    assert R.try_claim(cdir, block, 0, "tok-a")
    (cdir / f"{block.slug}.claim").unlink()
    with caplog.at_level(logging.ERROR, logger="issue2162.run"):
        R.release_claim(cdir, block, "tok-a")
    assert any("VANISHED" in m for m in caplog.messages)
    caplog.clear()
    # Foreign-token claim: no raise, and the THIEF's claim file is left.
    assert R.try_claim(cdir, block, 0, "tok-thief")
    with caplog.at_level(logging.ERROR, logger="issue2162.run"):
        R.release_claim(cdir, block, "tok-original")
    assert any("ANOTHER worker" in m for m in caplog.messages)
    assert (cdir / f"{block.slug}.claim").exists()
    rec = json.loads((cdir / f"{block.slug}.claim").read_text())
    assert rec["token"] == "tok-thief"


def test_run_claim_queue_runs_every_block(tmp_path, cfg, pairs):
    blocks = R.smoke_blocks(pairs)[:6]
    regime_fp = "fp-test"
    ran: list[str] = []

    def run_one(block: R.Block) -> None:
        ran.append(block.key)
        R._write_json_atomic(
            R.block_done_path(cfg.out_root, block),
            {"key": block.key, "regime_fp": regime_fp},
        )

    stats = R.run_claim_queue(cfg, blocks, regime_fp, "blocks", run_one)
    assert stats["ran"] == len(blocks)
    assert sorted(ran) == sorted(b.key for b in blocks)
    # Idempotent second pass: everything done, nothing re-run.
    stats2 = R.run_claim_queue(cfg, blocks, regime_fp, "blocks", run_one)
    assert stats2["ran"] == 0


# ── regime-fingerprint resume refusal ────────────────────────────────


def test_block_done_regime_mismatch_raises(tmp_path, cfg):
    block = _mkblock()
    assert R.block_is_done(cfg.out_root, block, "fp-a") is False
    R._write_json_atomic(
        R.block_done_path(cfg.out_root, block), {"key": block.key, "regime_fp": "fp-a"}
    )
    assert R.block_is_done(cfg.out_root, block, "fp-a") is True
    with pytest.raises(RuntimeError, match="refusing to resume across"):
        R.block_is_done(cfg.out_root, block, "fp-b")


# ── slot positions / cap telemetry ───────────────────────────────────


def test_slot_position():
    assert R.slot_position(100, 60, "ce") == 99
    assert R.slot_position(100, 60, "pe") == 59
    with pytest.raises(AssertionError):
        R.slot_position(100, 100, "pe")  # prefix_end must be < ctx_len
    with pytest.raises(AssertionError):
        R.slot_position(10, 5, "answer")  # unknown slot


def test_cap_hit():
    assert R.cap_hit(2048, 2048) is True
    assert R.cap_hit(2047, 2048) is False


# ── margin pools schema ──────────────────────────────────────────────


def test_load_pools_schema(tmp_path):
    good = tmp_path / "pools.json"
    good.write_text(json.dumps({"pools": {"instr_format|v1-v2": [{"side": "A", "text": "hello"}]}}))
    pools = R.load_pools(good)
    assert "instr_format|v1-v2" in pools
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"pools": {"k": [{"side": "C", "text": "x"}]}}))
    with pytest.raises(AssertionError):
        R.load_pools(bad)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"pools": {"k": [{"side": "A", "text": "  "}]}}))
    with pytest.raises(AssertionError):
        R.load_pools(empty)


def test_pool_key(pairs):
    p = pairs[0]
    assert R.pool_key(p) == f"{p.cell}|{p.value_a}-{p.value_b}"


# ── pilot gate (designed halt, distinct rc) ──────────────────────────


def _cfg_with(tmp_path, *extra: str):
    args = R.parse_args(
        [
            "--phase",
            "grid",
            "--out-root",
            str(tmp_path / "out"),
            "--log-dir",
            str(tmp_path / "logs"),
            *extra,
        ]
    )
    c = R.build_config(args)
    c.out_root.mkdir(parents=True, exist_ok=True)
    return c


def test_pilot_gate_refuses_over_3x(tmp_path, cfg, pairs):
    totals = R.grid_totals(R.enumerate_blocks(pairs), R.GRID_DRAWS)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    # 10 s/rollout at width 1 -> 42120 * 10 / 3600 = 117 h >> 3 x 3.7 h.
    rc = R._enforce_pilot_gate(cfg, totals, ran_rollouts=10, ran_wall=100.0)
    assert rc == R.RC_PILOT_GATE
    report = json.loads((cfg.out_root / "pilot_gate_report.json").read_text())
    assert report["sweep_allowed"] is False
    # Fast pilot at width 1: 0.05 s/rollout -> 0.585 h < 3 x 3.7 h -> allowed.
    rc = R._enforce_pilot_gate(cfg, totals, ran_rollouts=2000, ran_wall=100.0)
    assert rc == R.RC_OK
    report = json.loads((cfg.out_root / "pilot_gate_report.json").read_text())
    assert report["sweep_allowed"] is True
    assert report["recommended_poll_fence_h"] == pytest.approx(2.0 * report["projected_pod_wall_h"])


def test_pilot_gate_passes_plan_basis_at_width_8(tmp_path, pairs):
    """r1 C2: the §9 plan is 42,120 rollouts across 8 workers within
    planned_wall_h — at exactly the plan-basis per-rollout wall the gate must
    PASS at width 8 and REFUSE at width 1 (the r1 bug ran the projection at
    the un-threaded default width 1, a deterministic 8x false-fire)."""
    totals = R.grid_totals(R.enumerate_blocks(pairs), R.GRID_DRAWS)
    per_rollout = R.PLANNED_GRID_WALL_H * 3600.0 * 8 / totals["rollouts_total"]
    ran_rollouts, ran_wall = 180, 180 * per_rollout
    cfg8 = _cfg_with(tmp_path, "--num-workers", "8")
    assert R._enforce_pilot_gate(cfg8, totals, ran_rollouts, ran_wall) == R.RC_OK
    report = json.loads((cfg8.out_root / "pilot_gate_report.json").read_text())
    assert report["num_workers"] == 8
    assert report["projected_pod_wall_h"] == pytest.approx(R.PLANNED_GRID_WALL_H)
    # Same timings projected at width 1: 8x the plan -> refusal.
    cfg1 = _cfg_with(tmp_path / "w1")
    assert R._enforce_pilot_gate(cfg1, totals, ran_rollouts, ran_wall) == R.RC_PILOT_GATE


def test_pilot_gate_force_is_resume_only(tmp_path, pairs):
    """r1 M3: --force is a RESUME override only — it must NOT bypass the
    pilot refusal; only --force-past-halt-gates does (recorded as forced)."""
    totals = R.grid_totals(R.enumerate_blocks(pairs), R.GRID_DRAWS)
    cfg_force = _cfg_with(tmp_path / "f", "--force")
    rc = R._enforce_pilot_gate(cfg_force, totals, ran_rollouts=10, ran_wall=100.0)
    assert rc == R.RC_PILOT_GATE
    cfg_halt = _cfg_with(tmp_path / "h", "--force-past-halt-gates")
    rc = R._enforce_pilot_gate(cfg_halt, totals, ran_rollouts=10, ran_wall=100.0)
    assert rc == R.RC_OK
    report = json.loads((cfg_halt.out_root / "pilot_gate_report.json").read_text())
    assert report["forced"] is True
    assert report["sweep_allowed"] is False  # the report never lies about refusal


# ── pilot vs production state on a SHARED out-root (r1 C1) ────────────


def test_pilot_leaves_no_done_state_on_shared_out_root(tmp_path, cfg, pairs):
    """r1 C1 (the production killer): the pilot runs production ``blocks[0]``
    under ``regime_fp + "-pilot"`` on the SAME out-root the grid then uses.
    Pre-fix, ``run_block`` wrote its done-files unconditionally, so every
    grid worker's ``block_is_done`` scan RAISED at P3 entry. This test pins
    BOTH halves: (a) the crash shape a pilot done-residue produces, and
    (b) the write_done=False wiring that prevents the residue."""
    blocks = R.enumerate_blocks(pairs)[:1]
    regime_fp = "fp-prod"
    # (a) The pre-fix residue kills the production queue at entry.
    R._write_json_atomic(
        R.block_done_path(cfg.out_root, blocks[0]),
        {"key": blocks[0].key, "regime_fp": regime_fp + "-pilot"},
    )
    with pytest.raises(RuntimeError, match="refusing to resume across"):
        R.run_claim_queue(cfg, blocks, regime_fp, "blocks", lambda b: None)
    # (b) Post-fix wiring: the pilot leg suppresses done-writes entirely.
    # r2 H1: AST-located pins, drift-hardened — a file-wide "write_done=False"
    # grep + an `if write_done:` count would miss a future UNGUARDED third
    # done-write in run_block, or a write_done=False on the WRONG call site.
    import ast
    import inspect

    assert inspect.signature(R.run_block).parameters["write_done"].default is True
    src = (REPO_ROOT / "scripts" / "issue2162_run.py").read_text()
    assert 'regime_fp + "-pilot"' in src  # the pilot leg keeps its own regime tag
    tree = ast.parse(src)
    fns = {
        n.name: n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name in ("run_block", "phase_grid")
    }
    # (b1) EVERY block_done_path done-write inside run_block sits under an
    # `if write_done:` guard (exactly the block + margin-twin pair today).
    rb = fns["run_block"]
    done_writes = [
        n
        for n in ast.walk(rb)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_write_json_atomic"
        and n.args
        and isinstance(n.args[0], ast.Call)
        and isinstance(n.args[0].func, ast.Name)
        and n.args[0].func.id == "block_done_path"
    ]
    assert len(done_writes) == 2, "expected the block + margin-twin done-writes"
    # r3 MINOR 3: iterate iff.body ONLY — ast.walk(iff) would also count a
    # done-write placed in the `else:` branch of `if write_done:` as GUARDED.
    guarded_ids = {
        id(n)
        for iff in ast.walk(rb)
        if isinstance(iff, ast.If)
        and isinstance(iff.test, ast.Name)
        and iff.test.id == "write_done"
        for stmt in iff.body
        for n in ast.walk(stmt)
    }
    unguarded = [n for n in done_writes if id(n) not in guarded_ids]
    assert not unguarded, "every run_block done-write must be write_done-guarded"
    # (b2) phase_grid's PILOT call site — and ONLY that site — passes the
    # literal write_done=False; the queue's run_one call keeps the default.
    rb_calls = [
        n
        for n in ast.walk(fns["phase_grid"])
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "run_block"
    ]
    assert len(rb_calls) == 2, "expected the run_one + pilot run_block call sites"
    wd_kwargs = [kw for c in rb_calls for kw in c.keywords if kw.arg == "write_done"]
    assert len(wd_kwargs) == 1, "exactly ONE call site (the pilot) threads write_done"
    assert isinstance(wd_kwargs[0].value, ast.Constant) and wd_kwargs[0].value.value is False
    pilot_call = next(c for c in rb_calls if any(k.arg == "write_done" for k in c.keywords))
    # The write_done=False call is the PILOT leg: its regime arg is the
    # BinOp `regime_fp + "-pilot"`, not the bare name.
    assert any(
        isinstance(a, ast.BinOp) and isinstance(a.right, ast.Constant) and a.right.value == "-pilot"
        for a in pilot_call.args
    ), "write_done=False must sit on the pilot-tagged call site"
    # And with a CLEAN out-root the queue actually RUNS the block.
    clean = _cfg_with(tmp_path / "clean")
    ran: list[str] = []

    def run_one(block: R.Block) -> None:
        ran.append(block.key)
        R._write_json_atomic(
            R.block_done_path(clean.out_root, block),
            {"key": block.key, "regime_fp": regime_fp},
        )

    stats = R.run_claim_queue(clean, blocks, regime_fp, "blocks", run_one)
    assert stats["ran"] == 1 and ran == [blocks[0].key]


# ── width-keyed sharded resume (r1 M2) ───────────────────────────────


def test_sharded_done_record_width_mismatch_regenerates(tmp_path):
    """r1 M2: ``order[w::num_workers]`` shard identity includes the width — a
    done record written at width 8 must NOT satisfy a width-4 resume (the
    vanished workers' contexts would silently never regenerate)."""
    cfg8 = _cfg_with(tmp_path, "--num-workers", "8")
    cfg8.manifest_dir.mkdir(parents=True, exist_ok=True)
    R._write_json_atomic(
        cfg8.manifest_dir / "anchors_gate_w0_done.json",
        {"regime_fp": "fp-a", "num_workers": 8, "batch": "gate"},
    )
    assert R._sharded_done_record(cfg8, "anchors_gate_w0", "fp-a") is not None
    cfg4 = _cfg_with(tmp_path, "--num-workers", "4")  # same out-root
    assert R._sharded_done_record(cfg4, "anchors_gate_w0", "fp-a") is None
    # A LEGACY record with no num_workers key also regenerates (never crashes).
    R._write_json_atomic(
        cfg8.manifest_dir / "anchors_rest_w0_done.json",
        {"regime_fp": "fp-a", "batch": "rest"},
    )
    assert R._sharded_done_record(cfg8, "anchors_rest_w0", "fp-a") is None


# ── dispatcher wiring pins (r1 C2 / C4 / M4 / M9) ────────────────────


def test_dispatch_all_chain_wiring():
    """r2 MAJOR 1 + MINOR 1: the ``all`` chain runs pilot (width-threaded,
    gate-independent) -> gate3-check -> grid -> margin-OPPORTUNISTIC ->
    upload, in that order — upload/teardown must NEVER wait on the Batch-API
    pools SLA; the stage2 fan-out case exists."""
    src = (REPO_ROOT / "scripts" / "issue2162_dispatch.sh").read_text()
    assert 'pilot --pilot --num-workers "$NUM_WORKERS"' in src  # r1 C2
    assert "run_stage2_fanout" in src  # r1 M9
    seg = src.split("all)")[1].split(";;")[0]
    order = [
        "run_single_gpu_phase pilot",
        "require_gate3",
        "run_fanout_phase grid",
        "run_margin_opportunistic",
        "run_upload",
    ]
    idx = [seg.index(tok) for tok in order]
    assert idx == sorted(idx), (order, idx)  # r2 MAJOR 1 / MINOR 1 sequencing
    # The `all` chain must NOT use the hard-halting variant ...
    assert "run_margin_if_pools" not in seg
    # ... which stays wired to the STANDALONE `margin` phase (rc=24, r1 C4).
    margin_seg = src.split("margin)")[1].split(";;")[0]
    assert "run_margin_if_pools" in margin_seg
    assert "exit 24" in src
    # The deferred branch is LOUD and returns 0 (proceed to upload).
    assert "margin DEFERRED" in src


# ── claim-queue namespace pairing (the margin infinite-loop bug) ──────


def test_claim_queue_namespace_matches_done_write_namespace():
    """Every ``run_claim_queue`` call's namespace must equal the namespace its
    ``run_one`` writes done-files under — a mismatch means the queue never
    sees completion and re-runs blocks forever (caught live: phase_margin
    polled "margin" while done-files landed under "margin_blocks")."""
    import re

    for path, queue_ns in (
        (Path("scripts/issue2162_run.py"), {"blocks", "margin_blocks"}),
        (Path("scripts/issue2162_stage2.py"), {"stage2_blocks"}),
    ):
        src = (REPO_ROOT / path).read_text()
        called = set(re.findall(r'run_claim_queue\(cfg, blocks, regime_fp, "([^"]+)"', src))
        assert called == queue_ns, (path, called)
        written = set(re.findall(r'block_done_path\(cfg\.out_root, block, "([^"]+)"\)', src))
        # Every queue namespace has a matching done-write in the same file
        # (run.py's grid queue polls "blocks", the block_done_path DEFAULT).
        assert called - {"blocks"} <= written, (path, called, written)


# ── upload seam body (r2: upload_dir_hf extraction) ──────────────────


def test_upload_dir_hf_retries_then_raises(tmp_path, monkeypatch):
    """Executes the REAL ``upload_dir_hf`` body (the fail-loud retry seam the
    analysis driver's perm-matrix persist reuses), faking ONLY the hub
    boundary with a signature-mirroring fake: a no-path return is retried
    with backoff, persistent no-path RAISES (never warn-and-continue), and
    success returns the exact expected repo paths."""
    (tmp_path / "a.npz").write_bytes(b"x")
    sleeps: list[float] = []
    monkeypatch.setattr(R, "_upload_retry_sleep", sleeps.append)
    calls: list[dict] = []
    results = ["", "https://hf.co/ok"]

    def fake_upload_folder_filtered(  # mirrors hub._upload_folder_filtered
        local_dir,
        repo_id,
        repo_type,
        path_in_repo,
        allow_patterns,
        expected_repo_paths,
        ignore_patterns=None,
        delete_after=False,
    ):
        calls.append({"path_in_repo": path_in_repo, "expected": list(expected_repo_paths)})
        return results[min(len(calls) - 1, len(results) - 1)]

    import explore_persona_space.orchestrate.hub as hub

    monkeypatch.setattr(hub, "_upload_folder_filtered", fake_upload_folder_filtered)
    out = R.upload_dir_hf(tmp_path, "pfx/analysis_tensors/probe_perm_matrix", ["*.npz"])
    assert out == ["pfx/analysis_tensors/probe_perm_matrix/a.npz"]
    assert calls[-1]["expected"] == out
    assert len(calls) == 2 and len(sleeps) == 1  # the retry branch engaged
    # Exhaustion RAISES — the results sentinel must never post over loss.
    results[:] = [""]
    calls.clear()
    with pytest.raises(RuntimeError, match="no path"):
        R.upload_dir_hf(tmp_path, "pfx/x", ["*.npz"])
    assert len(calls) == R.UPLOAD_TRANSPORT_RETRIES + 1
    # No matching files: a skip, never a network call.
    calls.clear()
    assert R.upload_dir_hf(tmp_path, "pfx/y", ["*.jsonl"]) == []
    assert not calls


# ── corrupt claim records read DEAD (r2 H2) ──────────────────────────


def test_claim_stale_missing_or_nonpositive_pid_is_dead():
    """r2 H2: a same-host claim record with a MISSING or non-positive pid is
    STALE (dead) — ``os.kill(-1, 0)`` signals the caller's own process GROUP
    and SUCCEEDS, so the pre-fix probe read a corrupt record as live forever
    (permanently unstealable)."""
    import socket

    host = socket.gethostname()
    assert R._claim_stale({"host": host, "ts": time.time()}) is True  # pid missing
    assert R._claim_stale({"host": host, "pid": -1, "ts": time.time()}) is True
    assert R._claim_stale({"host": host, "pid": 0, "ts": time.time()}) is True
    # A live same-host claim is still never stolen (r1 M1 unchanged).
    assert R._claim_stale({"host": host, "pid": os.getpid(), "ts": 0.0}) is False


# ── sentinel margin_deferred flag (r2 MAJOR 1) ───────────────────────


def test_sentinel_margin_deferred_both_branches(tmp_path, pairs):
    """r2 MAJOR 1: the results-sentinel payload carries ``margin_deferred``
    in BOTH branches — True + the deferred-leg RECIPE while the margin legs
    are incomplete (the Batch-SLA tail deferral), False once every per-block
    margin done-file AND a complete sharded anchor-margin record set are on
    disk (the state the later 1x-H100 ``margin`` + ``upload`` leg produces).
    Disk-derived, so a standalone ``upload`` reports the truth regardless of
    which dispatcher branch ran."""
    cfg = _cfg_with(tmp_path, "--smoke", "--upload", "none")
    payload = R._sentinel_payload(cfg, {})
    assert payload["margin_deferred"] is True
    assert "dispatch.sh margin" in payload["margin_deferred_recipe"]
    assert payload["margin_blocks_done"] == 0
    # r3 MINOR 2: no local grid block state (the deferred-leg pod shape) ->
    # the sentinel stamps deferred_leg so its zeroed grid stats read as
    # "grid ran elsewhere", never "run produced nothing".
    assert payload["deferred_leg"] is True
    assert payload["eval_numbers"]["grid_shards"] == 0
    # A pod WITH local grid block done-state stamps deferred_leg False.
    grid_done = cfg.manifest_dir / "blocks"
    grid_done.mkdir(parents=True, exist_ok=True)
    R._write_json_atomic(grid_done / "b0.done.json", {"n_cap_hit": 1, "n_rows": 10})
    payload_grid = R._sentinel_payload(cfg, {})
    assert payload_grid["deferred_leg"] is False
    assert payload_grid["eval_numbers"]["grid_rollouts_persisted"] == 10
    (grid_done / "b0.done.json").unlink()

    # Complete the margin legs on disk -> deferred flips False, recipe gone.
    blocks = R.smoke_blocks(pairs)
    for b in blocks:
        R._write_json_atomic(R.block_done_path(cfg.out_root, b, "margin_blocks"), {"key": b.key})
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    for w in range(2):
        R._write_json_atomic(
            cfg.manifest_dir / f"margin_anchors_w{w}_done.json",
            {"num_workers": 2, "worker_index": w},
        )
    payload = R._sentinel_payload(cfg, {})
    assert payload["margin_deferred"] is False
    assert "margin_deferred_recipe" not in payload
    assert payload["margin_blocks_done"] == len(blocks) == payload["margin_blocks_expected"]

    # An INCOMPLETE anchor shard set (1 of 2 workers) stays deferred.
    (cfg.manifest_dir / "margin_anchors_w1_done.json").unlink()
    assert R._margin_state(cfg)["margin_deferred"] is True


# ── degeneracy guard (plan §7 gate 2) ────────────────────────────────


def _unit_vec(theta: float) -> torch.Tensor:
    """(1, 2) unit vector at angle ``theta`` — cosine to ``_unit_vec(0)`` is cos(theta)."""
    return torch.tensor([[math.cos(theta), math.sin(theta)]], dtype=torch.float32)


# Realized bf16 batch-composition jitter (the 2026-08-06 rc=23 halt read
# pe_cos = 0.9997999668 on token-prefix-identical persona_role_header pairs).
_JITTER = 2.0e-4
_JITTER_THETA = math.acos(1.0 - _JITTER)
_DISTINCT_THETA = 1.2  # cos ≈ 0.362 — clearly distinct states


def _mk_pair(cell: str, carrier: str = "d3") -> B.Pair2162:
    a, b = f"{cell}__va__{carrier}", f"{cell}__vb__{carrier}"
    return B.Pair2162(
        pair_id=f"{cell}::va-vb::{carrier}",
        cell=cell,
        carrier=carrier,
        value_a="va",
        value_b="vb",
        a=a,
        b=b,
    )


def _bank_for(pair, *, pe_theta: float, ce_theta: float = _DISTINCT_THETA) -> dict:
    """Two-context bank: A at angle 0 on both slots, B at the given angles."""
    return {
        "per_context": {
            pair.a: {"v_pe": _unit_vec(0.0), "v_ce": _unit_vec(0.0)},
            pair.b: {"v_pe": _unit_vec(pe_theta), "v_ce": _unit_vec(ce_theta)},
        }
    }


def test_degenerate_pair_bf16_jitter_passes_on_token_prefix_identity():
    """The 2026-08-06 false-FAIL regression: a degenerate-cell pair with
    IDENTICAL token prefixes through the slot but states differing by ~2e-4
    cosine (bf16 batch-composition jitter) must PASS — the guard gates the
    premise (token-prefix identity), and records pe_cos informationally.
    Pre-fix (state bit-identity bar ``pe_cos >= 0.99999``) this exact fixture
    produced a violation."""
    pair = _mk_pair("persona_role_header")
    bank = _bank_for(pair, pe_theta=_JITTER_THETA)
    ids_a = [11, 12, 13, 14, 21, 22]  # value text lives AFTER the slot (index pe=3)
    ids_b = [11, 12, 13, 14, 31, 32]
    tp = {pair.a: (ids_a, 3), pair.b: (ids_b, 3)}
    report = R.run_degeneracy_guard(bank, [pair], token_prefixes=tp)
    assert report["passed"], report["violations"]
    assert report["n_violations"] == 0
    # Informational state-space observation, never gated.
    assert report["degenerate_pe_cos"][pair.pair_id] == pytest.approx(1.0 - _JITTER, abs=1e-6)
    assert report["max_pe_jitter"] == pytest.approx(_JITTER, abs=1e-6)
    assert report["n_degenerate_pairs"] == 1
    # Report stays backward-compatible in shape (add-only).
    for key in (
        "criterion",
        "bar_cos",
        "declared_degenerate_cells",
        "n_pairs_checked",
        "n_violations",
        "violations",
        "passed",
    ):
        assert key in report, key


def test_degenerate_pair_token_prefix_mismatch_still_halts():
    """The REAL bank-defect direction is preserved: a degenerate-cell pair
    whose token prefixes genuinely differ (within the slot, or pe_a != pe_b)
    FAILs — even with bit-identical states."""
    pair = _mk_pair("query_content")
    bank = _bank_for(pair, pe_theta=0.0)  # bit-identical v_pe (cos = 1.0)
    # (a) ids differ at an index <= pe.
    tp = {pair.a: ([11, 12, 13, 14, 21], 3), pair.b: ([11, 99, 13, 14, 21], 3)}
    report = R.run_degeneracy_guard(bank, [pair], token_prefixes=tp)
    assert not report["passed"]
    (row,) = report["violations"]
    assert row["flag"] == "token_prefix"
    assert row["declared_degenerate_pe"] is True
    assert (row["pe_a"], row["pe_b"]) == (3, 3)
    # (b) prefix_end indices differ.
    tp = {pair.a: ([11, 12, 13, 14, 21], 3), pair.b: ([11, 12, 13, 14, 21], 4)}
    report = R.run_degeneracy_guard(bank, [pair], token_prefixes=tp)
    assert not report["passed"]
    assert report["violations"][0]["flag"] == "token_prefix"
    # A premise-FAILING pair contributes no jitter observation.
    assert report["max_pe_jitter"] is None


def test_degenerate_pair_state_sanity_band():
    """FIX 1: premise-verified degenerate pairs keep a LOOSE state-side sanity
    band (``STATE_SANITY_COS_MIN`` = 0.99) — identical token prefixes at the
    realized bf16 jitter (pe_cos ~0.9998) PASS, while a grossly different
    ``v_pe`` (cos 0.5 — a capture-side row misalignment writing garbage
    states, which the injection gate cannot backstop) FAILs with the new
    ``state_sanity`` flag. The band sits ~50x above the realized
    ``max_pe_jitter`` (2.04e-4), so it cannot reintroduce the 2026-08-06
    bf16-jitter false-FAIL."""
    pair = _mk_pair("persona_role_header")
    ids = [11, 12, 13, 14, 21, 22]
    tp = {pair.a: (ids, 3), pair.b: (list(ids), 3)}  # identical token prefixes
    # (a) realized jitter passes — the false-FAIL fixture stays green.
    report = R.run_degeneracy_guard(
        _bank_for(pair, pe_theta=_JITTER_THETA), [pair], token_prefixes=tp
    )
    assert report["passed"], report["violations"]
    assert report["state_sanity_cos_min"] == R.STATE_SANITY_COS_MIN == 0.99
    # (b) garbage v_pe under identical prefixes FAILs with the new flag.
    report = R.run_degeneracy_guard(
        _bank_for(pair, pe_theta=math.acos(0.5)), [pair], token_prefixes=tp
    )
    assert not report["passed"]
    (row,) = report["violations"]
    assert row["flag"] == "state_sanity"
    assert row["declared_degenerate_pe"] is True
    assert row["pe_cos"] == pytest.approx(0.5, abs=1e-6)
    # A sanity-violating pair contributes no jitter observation.
    assert report["max_pe_jitter"] is None


def test_non_degenerate_pair_identical_states_still_fail_distinctness():
    """The non-degenerate direction is byte-unchanged: bit-identical states
    at either slot violate the ``cos < DEGENERACY_COS_MIN`` distinctness bar
    (jitter cannot flip it — identical reads > 1 - 1e-5, distinct <= ~0.994)."""
    pair = _mk_pair("instr_format")
    bank = _bank_for(pair, pe_theta=0.0)  # v_pe identical, v_ce distinct
    report = R.run_degeneracy_guard(bank, [pair], token_prefixes={})
    assert not report["passed"]
    (row,) = report["violations"]
    assert row["flag"] == "distinctness_pe"
    # Both slots identical -> both distinctness directions named.
    bank = _bank_for(pair, pe_theta=0.0, ce_theta=0.0)
    report = R.run_degeneracy_guard(bank, [pair], token_prefixes={})
    assert report["violations"][0]["flag"] == "distinctness_pe+distinctness_ce"
    # And a healthy non-degenerate pair (distinct on both slots) passes.
    bank = _bank_for(pair, pe_theta=_DISTINCT_THETA)
    report = R.run_degeneracy_guard(bank, [pair], token_prefixes={})
    assert report["passed"], report["violations"]
    assert report["degenerate_pe_cos"] == {} and report["n_degenerate_pairs"] == 0


def test_degeneracy_guard_requires_tok_or_prefixes():
    """Fail-loud wiring: a degenerate pair with neither ``tok`` nor
    ``token_prefixes`` (or with a missing context entry) asserts before any
    state read."""
    pair = _mk_pair("persona_role_header")
    with pytest.raises(AssertionError, match="needs tok"):
        R.run_degeneracy_guard({"per_context": {}}, [pair])
    with pytest.raises(AssertionError, match="missing degenerate contexts"):
        R.run_degeneracy_guard({"per_context": {}}, [pair], token_prefixes={})


def test_degenerate_token_prefix_helper_call_shapes_bind():
    """``_degenerate_token_prefixes`` uses the exact ``capture_bank`` call
    shapes (build_contexts / context_token_ids_2162 / prefix_end_index_multi)
    — signature-bind them so a helper rename/re-arity fails here, not on the
    pod (the tokenizer-bearing body itself runs in production P1)."""
    import inspect

    inspect.signature(B.build_contexts).bind()
    inspect.signature(B.context_token_ids_2162).bind(object(), {})
    inspect.signature(B.prefix_end_index_multi).bind(object(), [0, 1, 2])
