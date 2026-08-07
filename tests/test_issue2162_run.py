"""Issue #2162 driver — CPU pins for the pure scheduling/queue/gate helpers.

Covers the plan §4.6 mechanical gates that are testable without a GPU:

- block enumeration (234 = 39 cells x 2 slots x 3 arms; 42,120 rollouts),
- the shared work-conserving CLAIM-FILE queue (O_CREAT-exclusive claims,
  stale-claim reclamation by dead-pid and by age, unparseable-claim fail-loud,
  wrong-token release fail-loud),
- regime-fingerprint resume refusal (a mismatched done-file RAISES, never a
  silent skip),
- the pilot gate's projection arithmetic + designed rc,
- the smoke slice's per-arm-class coverage.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

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


def test_release_claim_wrong_token_asserts(tmp_path):
    cdir = tmp_path / "claims"
    block = _mkblock()
    assert R.try_claim(cdir, block, 0, "tok-a")
    with pytest.raises(AssertionError, match="another worker"):
        R.release_claim(cdir, block, "tok-b")


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
