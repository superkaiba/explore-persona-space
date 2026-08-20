"""Issue #2389 — vLLM anchor leg + anchor-rest routing unit tests (CPU, no engine).

Covers the plan §4.7 item-4 seams that no smoke can reach cheaply:

- the atomic rest-routing freeze (`issue2389_run._resolve_rest_routing`):
  single decision point, frozen against later `vllm_cells.json` edits,
  regime-fingerprint + unknown-cell fail-loud;
- the step-3 pin-1 filename contract: vLLM production shards land INSIDE both
  consumer globs, parity-side shards land OUTSIDE them, and none of the
  cell-grained shard stems are visible to the width sweeps;
- worker-independent done predicates + claim-queue work conservation over
  `CellBlock`s;
- `leg_claim` verdict handling (PASS extends the claim; FAIL / frozen routing
  are inert);
- run-config composition through run.py's OWN parser (reused-module contract).

No vLLM engine, no model, no tokenizer download — pure filesystem + logic.
"""

from __future__ import annotations

import json
import sys
from fnmatch import fnmatch
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2389_run as R  # noqa: E402
import issue2389_vllm_anchors as V  # noqa: E402

# ── helpers ───────────────────────────────────────────────────────────


def _cfg(tmp_path: Path, **flags) -> R.RunConfig:
    argv = ["--phase", "anchors", "--out-root", str(tmp_path / "out")]
    for k, v in flags.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return R.build_config(R.parse_args(argv))


def _write_cells(cfg: R.RunConfig, cells: list[str]) -> None:
    cfg.gates_dir.mkdir(parents=True, exist_ok=True)
    (cfg.gates_dir / "vllm_cells.json").write_text(json.dumps({"cells": cells}))


SOME_CELLS = sorted(R.BANK.all_cells())[:2]


# ── _resolve_rest_routing: atomic freeze ──────────────────────────────


def test_routing_freezes_current_claim(tmp_path):
    cfg = _cfg(tmp_path)
    _write_cells(cfg, SOME_CELLS)
    frozen = R._resolve_rest_routing(cfg, "fp-a")
    assert frozen == frozenset(SOME_CELLS)
    assert (cfg.gates_dir / R.ANCHOR_REST_ROUTING_NAME).exists()


def test_routing_ignores_later_claim_edits(tmp_path):
    """A vllm_cells.json edit AFTER the freeze is inert — the frozen decision wins."""
    cfg = _cfg(tmp_path)
    _write_cells(cfg, SOME_CELLS[:1])
    frozen1 = R._resolve_rest_routing(cfg, "fp-a")
    _write_cells(cfg, sorted(R.BANK.all_cells()))  # late claim
    frozen2 = R._resolve_rest_routing(cfg, "fp-a")
    assert frozen1 == frozen2 == frozenset(SOME_CELLS[:1])


def test_routing_empty_claim_freezes_empty(tmp_path):
    """No vllm_cells.json at freeze time -> empty exclusion, HF owns everything."""
    cfg = _cfg(tmp_path)
    assert R._resolve_rest_routing(cfg, "fp-a") == frozenset()
    # A later claim cannot re-open the decision.
    _write_cells(cfg, SOME_CELLS)
    assert R._resolve_rest_routing(cfg, "fp-a") == frozenset()


def test_routing_regime_mismatch_fails_loud(tmp_path):
    cfg = _cfg(tmp_path)
    R._resolve_rest_routing(cfg, "fp-a")
    with pytest.raises(RuntimeError, match="regime_fp"):
        R._resolve_rest_routing(cfg, "fp-b")


def test_routing_unknown_cell_fails_loud(tmp_path):
    cfg = _cfg(tmp_path)
    _write_cells(cfg, ["not_a_cell"])
    with pytest.raises(RuntimeError, match="unknown cells"):
        R._resolve_rest_routing(cfg, "fp-a")


def test_vllm_claimed_cells_unknown_cell_fails_loud(tmp_path):
    cfg = _cfg(tmp_path)
    _write_cells(cfg, ["not_a_cell"])
    with pytest.raises(RuntimeError, match="unknown cells"):
        R._vllm_claimed_cells(cfg)


# ── pin 1: filename/glob contract ─────────────────────────────────────

JUDGE_GLOB = "anchors_*.jsonl"  # scripts/issue2389_judge.py anchor-shard discovery
ANALYSIS_GLOB = "va_anchors_*.pt"  # analysis anchor-va discovery


def test_pin1_vllm_production_shards_inside_consumer_globs():
    for cell in V.PARITY_CELLS:
        assert fnmatch(f"anchors_vllm_{cell}_w0.jsonl", JUDGE_GLOB)
        assert fnmatch(f"va_anchors_vllm_{cell}_w0.pt", ANALYSIS_GLOB)
        assert fnmatch(f"anchors_parity_{cell}_w0.jsonl", JUDGE_GLOB)
        assert fnmatch(f"va_anchors_parity_{cell}_w0.pt", ANALYSIS_GLOB)


def test_pin1_literal_prefix_countershape_outside_globs():
    """The rejected `vllm_anchors_*` prefix shape falls outside BOTH globs —
    the reason pin 1 pins the engine marker to the BATCH-ID position."""
    assert not fnmatch("vllm_anchors_cell_w0.jsonl", JUDGE_GLOB)
    assert not fnmatch("vllm_va_anchors_cell_w0.pt", ANALYSIS_GLOB)


def test_parity_side_artifacts_outside_consumer_globs():
    """vLLM parity rollouts are gate evidence, never production anchors."""
    assert not fnmatch("vllm_parity_filler_swap_w0.jsonl", JUDGE_GLOB)


def test_cell_grained_shards_invisible_to_width_sweeps():
    """`_shard_stem_index`'s strict allowlist must not see cell-grained shards
    (they are claim-queue cell shards, not width-strided families)."""
    for name in (
        "anchors_vllm_filler_swap_w0.jsonl",
        "anchors_parity_filler_swap_w2.jsonl",
        "va_anchors_vllm_filler_swap_w0.pt",
    ):
        assert R._shard_stem_index(name) is None, name
    # sanity: the width-strided families ARE visible
    assert R._shard_stem_index("anchors_rest_w3.jsonl") == ("anchors_rest", 3)


# ── done predicates + claim-queue work conservation ───────────────────


def _fake_done_shard(cfg: R.RunConfig, batch: str, w: int, regime_fp: str, n_rows: int = 2):
    cfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    (cfg.anchors_dir / f"anchors_{batch}_w{w}.jsonl").write_text('{"x": 1}\n' * n_rows)
    (cfg.anchors_dir / f"va_anchors_{batch}_w{w}.pt").write_bytes(b"pt")
    R._write_json_atomic(
        cfg.manifest_dir / f"anchors_{batch}_w{w}_done.json",
        {"regime_fp": regime_fp, "batch": batch, "worker_index": w, "n_rows": n_rows},
    )


def test_anchor_cell_done_worker_independent(tmp_path):
    cfg = _cfg(tmp_path, worker_index=0)
    assert not V._anchor_cell_done(cfg, "fp", "vllm_filler_swap")
    _fake_done_shard(cfg, "vllm_filler_swap", w=5, regime_fp="fp")  # another worker's shard
    assert V._anchor_cell_done(cfg, "fp", "vllm_filler_swap")
    assert not V._anchor_cell_done(cfg, "OTHER-fp", "vllm_filler_swap")


def test_anchor_cell_done_requires_artifacts(tmp_path):
    cfg = _cfg(tmp_path)
    _fake_done_shard(cfg, "vllm_filler_swap", w=1, regime_fp="fp")
    (cfg.anchors_dir / "va_anchors_vllm_filler_swap_w1.pt").unlink()
    assert not V._anchor_cell_done(cfg, "fp", "vllm_filler_swap")


def test_prod_gen_done(tmp_path):
    cfg = _cfg(tmp_path)
    assert not V._prod_gen_done(cfg, "fp", "filler_swap")
    cfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    jsonl = cfg.anchors_dir / "anchors_vllm_filler_swap_w4.jsonl"
    jsonl.write_text("{}\n")
    R._write_json_atomic(
        cfg.manifest_dir / "anchors_vllm_filler_swap_w4_gen_done.json",
        {"regime_fp": "fp", "jsonl": str(jsonl)},
    )
    assert V._prod_gen_done(cfg, "fp", "filler_swap")


def test_claim_queue_runs_each_cell_once(tmp_path):
    """CellBlocks satisfy the claim-queue contract; each cell runs exactly once."""
    cfg = _cfg(tmp_path)
    cells = list(V.PARITY_CELLS)
    ran: list[str] = []
    done: set[str] = set()

    def run_one(block: V.CellBlock) -> None:
        ran.append(block.cell)
        done.add(block.cell)

    blocks = [V.CellBlock(cell=c, leg="prod") for c in cells]
    R.run_claim_queue(
        cfg,
        blocks,
        "fp",
        "test_ns",
        run_one,
        is_done=lambda _root, b, _fp, _ns: b.cell in done,
    )
    assert sorted(ran) == sorted(cells)
    # A second pass (resume) re-runs nothing.
    R.run_claim_queue(
        cfg,
        blocks,
        "fp",
        "test_ns",
        run_one,
        is_done=lambda _root, b, _fp, _ns: b.cell in done,
    )
    assert sorted(ran) == sorted(cells)


# ── leg_claim verdict handling ────────────────────────────────────────


def test_leg_claim_pass_extends_claim_to_all_cells(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(V, "_read_parity_report", lambda _cfg: {"verdict": "PASS"})
    assert V.leg_claim(cfg, timeout_s=1.0) == R.RC_OK
    rec = json.loads((cfg.gates_dir / "vllm_cells.json").read_text())
    assert set(rec["cells"]) == set(R.BANK.all_cells())
    assert rec["reason"] == "parity-pass-full-claim"


def test_leg_claim_pass_after_freeze_is_inert(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    R._resolve_rest_routing(cfg, "fp")  # freeze (empty claim)
    monkeypatch.setattr(V, "_read_parity_report", lambda _cfg: {"verdict": "PASS"})
    assert V.leg_claim(cfg, timeout_s=1.0) == R.RC_OK
    assert not (cfg.gates_dir / "vllm_cells.json").exists()


def test_leg_claim_fail_writes_nothing(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(V, "_read_parity_report", lambda _cfg: {"verdict": "FAIL"})
    assert V.leg_claim(cfg, timeout_s=1.0) == R.RC_OK
    assert not (cfg.gates_dir / "vllm_cells.json").exists()


def test_leg_claim_timeout_writes_nothing(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(V, "_read_parity_report", lambda _cfg: None)
    monkeypatch.setattr(V, "REPORT_POLL_S", 0.01)
    assert V.leg_claim(cfg, timeout_s=0.05) == R.RC_OK
    assert not (cfg.gates_dir / "vllm_cells.json").exists()


# ── config composition (reused-module contract) ───────────────────────


def test_compose_run_cfg_matches_run_py_parser(tmp_path):
    args = V.parse_args(
        [
            "--leg",
            "parity",
            "--out-root",
            str(tmp_path / "out"),
            "--worker-index",
            "2",
            "--num-workers",
            "8",
            "--smoke",
            "--upload",
            "none",
        ]
    )
    cfg = V._compose_run_cfg(args)
    ref = R.build_config(
        R.parse_args(
            [
                "--phase",
                "anchors",
                "--out-root",
                str(tmp_path / "out"),
                "--worker-index",
                "2",
                "--num-workers",
                "8",
                "--smoke",
                "--upload",
                "none",
            ]
        )
    )
    assert cfg == ref


def test_compose_run_cfg_default_upload_binds(tmp_path):
    # B2 (r1 review): the DEFAULT --upload must bind through run.py's own
    # parser — the prior `full` default died in argparse on every normal
    # dispatcher invocation of the parity/claim/production legs.
    args = V.parse_args(["--leg", "parity", "--out-root", str(tmp_path / "out")])
    assert args.upload == "hf"
    cfg = V._compose_run_cfg(args)
    assert cfg.upload_mode == "hf"


def test_vllm_chunk_env_validated():
    # codex r1 minor: EPM_2389_VLLM_CHUNK must stay inside the registered
    # <=500 prompts/call band (plan §4.7 item 4).
    assert V._validated_vllm_chunk("500") == 500
    assert V._validated_vllm_chunk("1") == 1
    for bad in ("0", "501", "-3"):
        with pytest.raises(ValueError, match="outside the registered band"):
            V._validated_vllm_chunk(bad)


def test_parity_cells_are_bank_cells():
    assert set(V.PARITY_CELLS) <= set(R.BANK.all_cells())


def test_import_check_flag_requires_no_run_args():
    args = V.parse_args(["--import-check"])
    assert args.import_check and args.leg is None and args.out_root is None
