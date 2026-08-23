"""Issue #2389 — vLLM anchor leg + per-cell anchor routing unit tests (CPU, no engine).

Covers the plan §4.7 item-4 seams that no smoke can reach cheaply:

- per-cell claim-time routing (B8 r1 review): the live `vllm_cells.json`
  claim set is re-read every scan, a late parity PASS re-routes exactly the
  not-yet-claimed / not-yet-done rest cells, and HF-done cells are never
  regenerated (work conservation both ways);
- cell-grain sharding keeps every generate chunk at full ``gen_batch``
  (B7 r1 review — the strided pre-fix shape fails the median-chunk floor);
- the step-3 pin-1 filename contract: vLLM production shards land INSIDE both
  consumer globs, parity-side shards land OUTSIDE them, and none of the
  cell-grained shard stems are visible to the width sweeps;
- worker-independent done predicates + claim-queue work conservation over
  `CellBlock`s;
- `leg_claim` verdict handling (PASS always extends the claim; FAIL/timeout
  leave the parity-only claim);
- run-config composition through run.py's OWN parser (reused-module
  contract), incl. the B9 pilot gen_batch adoption + family-frozen
  share-prefill resolution.

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


# ── per-cell claim-time routing (B8) ──────────────────────────────────


def test_vllm_claimed_cells_reads_live_claim(tmp_path):
    """B8: the claim set is LIVE — run.py's rest workers re-read it every
    scan (via the `mine` predicate), so a later claim edit IS visible. The
    pre-fix `_resolve_rest_routing` one-shot global freeze (which made a
    late PASS inert for every cell) is GONE."""
    cfg = _cfg(tmp_path)
    assert R._vllm_claimed_cells(cfg) == frozenset()
    _write_cells(cfg, SOME_CELLS[:1])
    assert R._vllm_claimed_cells(cfg) == frozenset(SOME_CELLS[:1])
    _write_cells(cfg, sorted(R.BANK.all_cells()))  # late PASS extension
    assert R._vllm_claimed_cells(cfg) == frozenset(R.BANK.all_cells())
    assert not hasattr(R, "_resolve_rest_routing")
    assert not hasattr(R, "ANCHOR_REST_ROUTING_NAME")


def test_vllm_claimed_cells_unknown_cell_fails_loud(tmp_path):
    cfg = _cfg(tmp_path)
    _write_cells(cfg, ["not_a_cell"])
    with pytest.raises(RuntimeError, match="unknown cells"):
        R._vllm_claimed_cells(cfg)


def test_b8_late_pass_reroutes_only_unclaimed_cells(tmp_path, monkeypatch):
    """B8 mechanizable check (r1 review verbatim): simulate HF claiming +
    finishing cell A, land a late PASS, assert untouched B/C route to the
    vLLM leg WITHOUT regenerating A — on both sides of the seam."""
    cfg = _cfg(tmp_path)
    draws = cfg.anchor_draws
    all_cells = sorted(set(R.BANK.all_cells()) - set(V.PARITY_CELLS))
    cell_a, cell_b, cell_c = all_cells[:3]
    # HF completed cell A before the verdict (worker-independent done shard).
    _fake_done_shard(cfg, f"rest_{cell_a}", w=0, regime_fp="fp", draws=draws)

    # Late PASS: leg_claim extends the claim set to every bank cell.
    monkeypatch.setattr(V, "_read_parity_report", lambda _cfg: {"verdict": "PASS"})
    assert V.leg_claim(cfg, timeout_s=1.0) == R.RC_OK
    assert R._vllm_claimed_cells(cfg) == frozenset(R.BANK.all_cells())

    # HF side: the rest queue (mine = not vllm-claimed) regenerates NOTHING —
    # A is done, B/C are claimed away.
    ran: list[str] = []
    blocks = [R.AnchorCellBlock(cell=c, batch="rest") for c in (cell_a, cell_b, cell_c)]
    R.run_claim_queue(
        cfg,
        blocks,
        "fp",
        "anchor_rest_cells",
        lambda b: ran.append(b.cell),
        is_done=lambda _root, b, fp, _ns: R._anchor_cell_done(cfg, fp, b.batch_id, draws),
        mine=lambda b: b.cell not in R._vllm_claimed_cells(cfg),
    )
    assert ran == []

    # vLLM side: ownership = claimed - parity - HF-done - empty.
    by_cell = {cell_a: ["x1", "x2"], cell_b: ["x3", "x4"], cell_c: ["x5"]}
    rest_by_cell, gen_cells = V._owned_rest_cells(cfg, "fp", draws, by_cell, gate_id_set=set())
    assert sorted(gen_cells) == sorted([cell_b, cell_c])
    assert cell_a not in gen_cells  # HF-done: never regenerated
    assert not set(gen_cells) & set(V.PARITY_CELLS)
    assert rest_by_cell[cell_b] == ["x3", "x4"]


def test_b7_cell_grain_chunks_reach_full_gen_batch(tmp_path):
    """B7 mechanizable check (r1 review verbatim): realized MEDIAN generate
    chunk size >= min(gen_batch, 8) under cell-grain sharding over the real
    bank enumeration — while the pre-fix worker-strided shape (order[w::W]
    per cell) fails the same floor."""
    cfg = _cfg(tmp_path)  # gen_batch default (16)
    pairs = R.BANK.build_pairs()  # full bank: CPU-pure, no tokenizer
    contexts = R.BANK.build_contexts()
    gate_pairs = R.BANK.gate_slice_pairs(pairs)
    gate_ids: list[str] = []
    seen: set[str] = set()
    for p in gate_pairs:
        for cid in (p.a, p.b):
            if cid not in seen:
                seen.add(cid)
                gate_ids.append(cid)
    rest_ids = [cid for cid in contexts if cid not in seen]

    def chunk_sizes(orders: list[list[str]]) -> list[int]:
        sizes = []
        for order in orders:
            n = len(order)
            sizes += [min(cfg.gen_batch, n - i) for i in range(0, n, cfg.gen_batch)]
        return sizes

    def median(xs: list[int]) -> float:
        s = sorted(xs)
        m = len(s) // 2
        return float(s[m]) if len(s) % 2 else (s[m - 1] + s[m]) / 2.0

    # Cell-grain (post-fix): one order per (batch, cell), never split by worker.
    cell_orders = [
        order
        for group in (R._group_by_cell(gate_ids, contexts), R._group_by_cell(rest_ids, contexts))
        for order in group.values()
    ]
    assert median(chunk_sizes(cell_orders)) >= min(cfg.gen_batch, 8)

    # Pre-fix strided shape: every cell's order split w::8 across 8 workers —
    # the de-batching the review measured (chunks of 1-3 contexts).
    strided_orders = [order[w::8] for order in cell_orders for w in range(8) if order[w::8]]
    assert median(chunk_sizes(strided_orders)) < min(cfg.gen_batch, 8)


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
    (they are claim-queue cell shards, not width-strided families). B7: the
    whole anchors family left the width-sweep universe — HF gate/rest cell
    shards included — leaving only the strided margin family."""
    for name in (
        "anchors_vllm_filler_swap_w0.jsonl",
        "anchors_parity_filler_swap_w2.jsonl",
        "va_anchors_vllm_filler_swap_w0.pt",
        "anchors_gate_filler_swap_w1.jsonl",
        "anchors_rest_filler_swap_w7.jsonl",
        "anchors_rest_w3.jsonl",  # the RETIRED strided stem: no longer sweepable
    ):
        assert R._shard_stem_index(name) is None, name
    # sanity: the one remaining width-strided family IS visible
    assert R._shard_stem_index("anchor_margin_w3.jsonl") == ("anchor_margin", 3)
    assert set(R._ARTIFACT_FAMILIES) == {"margin"}


# ── done predicates + claim-queue work conservation ───────────────────


def _fake_done_shard(
    cfg: R.RunConfig, batch: str, w: int, regime_fp: str, n_rows: int = 2, draws: int = 2
):
    cfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    (cfg.anchors_dir / f"anchors_{batch}_w{w}.jsonl").write_text('{"x": 1}\n' * n_rows)
    (cfg.anchors_dir / f"va_anchors_{batch}_w{w}.pt").write_bytes(b"pt")
    R._write_json_atomic(
        cfg.manifest_dir / f"anchors_{batch}_w{w}_done.json",
        {
            "regime_fp": regime_fp,
            "batch": batch,
            "worker_index": w,
            "n_rows": n_rows,
            "draws": draws,
        },
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


def test_leg_claim_late_pass_still_extends(tmp_path, monkeypatch):
    """B8 correction of the r1 routing tests: a PASS landing AFTER HF work
    started (parity claim already on disk) still extends the claim set —
    routing is per cell at claim time, so a late PASS is work-conserving,
    never inert."""
    cfg = _cfg(tmp_path)
    _write_cells(cfg, list(V.PARITY_CELLS))  # the parity leg's t0 claim
    monkeypatch.setattr(V, "_read_parity_report", lambda _cfg: {"verdict": "PASS"})
    assert V.leg_claim(cfg, timeout_s=1.0) == R.RC_OK
    rec = json.loads((cfg.gates_dir / "vllm_cells.json").read_text())
    assert set(rec["cells"]) == set(R.BANK.all_cells())


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


def test_compose_run_cfg_adopts_pilot_gen_batch_and_matching_fp(tmp_path):
    """B9: the vLLM legs adopt the pilot-selected gen_batch and resolve the
    family-frozen share-prefill decision exactly as phase_anchors does, so
    both sides compute the SAME regime fingerprint over one shard family."""
    ref = R.build_config(R.parse_args(["--phase", "anchors", "--out-root", str(tmp_path / "out")]))
    ref.gates_dir.mkdir(parents=True, exist_ok=True)
    cur = R._repro(ref)
    (ref.gates_dir / "pilot_gate_report.json").write_text(
        json.dumps(
            {
                "verdict": "ACCEPT",
                "gen_batch_selected": 32,
                "gen_batch_candidates": [16, 32],
                # round-5 C/J: the full runtime-domain record adoption validates
                "num_workers": max(1, ref.num_workers),
                "gpu_name": R._pilot_gpu_name(),
                "gpu_total_mem_gib": R._pilot_gpu_mem_gib(),
                "hbm_headroom_floor_gib": R.PILOT_HBM_HEADROOM_GIB,
                "refusal_threshold_h": R.PILOT_REFUSAL_MULT * ref.planned_wall_h,
                "planned_total_wall_h": ref.planned_wall_h,
                "accept_threshold_h": R.PILOT_ACCEPT_WALL_H,
                "repro": {
                    "model_id": ref.model_id,
                    "model_revision": ref.model_revision,
                    "smoke": False,
                    "tiny": False,
                    "torch": cur["torch"],
                    "transformers": cur["transformers"],
                    "git_commit": cur["git_commit"],
                },
            }
        )
    )
    args = V.parse_args(["--leg", "production", "--out-root", str(tmp_path / "out")])
    cfg = V._compose_run_cfg(args)
    assert cfg.gen_batch == 32  # pilot adoption threaded (B9)
    # And the fingerprint matches what an HF anchors worker computes.
    ref = R._adopt_pilot_gen_batch(ref)
    ref = R._resolve_share_prefill(ref, "anchors")
    assert R.regime_fingerprint(cfg, "sha") == R.regime_fingerprint(ref, "sha")


def _gpu_pilot_report(ref: R.RunConfig, num_workers: int = 8) -> None:
    """A pilot report recorded on the production GPU lane (H200, width 8) —
    the exact artifact a production `all` run leaves in gates/ before the
    detached CPU claim leg starts polling (dispatch.sh:552-553)."""
    ref.gates_dir.mkdir(parents=True, exist_ok=True)
    cur = R._repro(ref)
    (ref.gates_dir / "pilot_gate_report.json").write_text(
        json.dumps(
            {
                "verdict": "ACCEPT",
                "gen_batch_selected": 32,
                "gen_batch_candidates": [16, 32],
                "num_workers": num_workers,
                "gpu_name": "NVIDIA H200",
                "gpu_total_mem_gib": 140.4,
                "hbm_headroom_floor_gib": R.PILOT_HBM_HEADROOM_GIB,
                "refusal_threshold_h": R.PILOT_REFUSAL_MULT * ref.planned_wall_h,
                "planned_total_wall_h": ref.planned_wall_h,
                "accept_threshold_h": R.PILOT_ACCEPT_WALL_H,
                "repro": {
                    "model_id": ref.model_id,
                    "model_revision": ref.model_revision,
                    "smoke": False,
                    "tiny": False,
                    "torch": cur["torch"],
                    "transformers": cur["transformers"],
                    "git_commit": cur["git_commit"],
                },
            }
        )
    )


def test_r6_cpu_claim_leg_skips_pilot_adoption_and_share_prefill(tmp_path, monkeypatch, caplog):
    """Round-6 (concern pilot-reuse-runtime-domain): the claim leg is a
    CPU-only poll (dispatch.sh runs it under CUDA_VISIBLE_DEVICES="") that
    only reads the parity verdict and writes gates/vllm_cells.json — it
    generates nothing, so its runtime domain (CPU) can NEVER match a GPU
    pilot report, and _compose_run_cfg skips pilot gen_batch adoption +
    share-prefill resolution for it: an explicit, LOGGED carve-out. FAILED
    at HEAD~: adoption FOREIGN-raised on every production `all` run the
    moment the pilot report landed, and the dispatcher discards the
    detached claim pid's rc, so the death was silent."""
    monkeypatch.setattr(R, "_pilot_gpu_name", lambda: None)  # the CPU claim host
    ref = R.build_config(R.parse_args(["--phase", "anchors", "--out-root", str(tmp_path / "out")]))
    _gpu_pilot_report(ref, num_workers=8)
    args = V.parse_args(
        [
            "--leg",
            "claim",
            "--out-root",
            str(tmp_path / "out"),
            "--num-workers",
            "8",
            "--share-prefill",
            "auto",
        ]
    )
    with caplog.at_level("INFO"):
        cfg = V._compose_run_cfg(args)  # FAILED at HEAD~: RuntimeError FOREIGN
    assert cfg.gen_batch == 16 and not cfg.gen_batch_explicit  # adoption skipped
    assert cfg.share_prefill_armed is False
    # Share-prefill RESOLUTION skipped (not resolved-to-off): despite
    # --share-prefill auto, no family freeze was written by this leg.
    assert not list(ref.gates_dir.glob("share_prefill_frozen_*.json"))
    assert any("skipping pilot gen_batch adoption" in r.message for r in caplog.records)


def test_r6_claim_leg_with_cuda_keeps_full_resolution(tmp_path, monkeypatch):
    """Byte-identical GPU paths (round-6 scope bound): the carve-out keys on
    the RUNTIME DOMAIN (no CUDA device), not on the leg name — a claim leg
    that DOES see a GPU still routes through adoption + resolution exactly
    like parity/production."""
    monkeypatch.setattr(R, "_pilot_gpu_name", lambda: "NVIDIA H200")
    monkeypatch.setattr(R, "_pilot_gpu_mem_gib", lambda: 140.4)
    ref = R.build_config(R.parse_args(["--phase", "anchors", "--out-root", str(tmp_path / "out")]))
    _gpu_pilot_report(ref, num_workers=1)
    args = V.parse_args(["--leg", "claim", "--out-root", str(tmp_path / "out")])
    cfg = V._compose_run_cfg(args)
    assert cfg.gen_batch == 32  # pilot adoption ran (the B9 path, unchanged)


def test_share_prefill_family_freeze_pins_first_decision(tmp_path):
    """B9: the share-prefill decision is FROZEN per (out_root, family) — a
    battery PASS landing after the first resolver (the mid-anchors worker-1
    chain) is adopted as the frozen value, never a fresh re-resolution, so
    one shard family can never mix fingerprints."""
    argv = ["--phase", "anchors", "--out-root", str(tmp_path / "out"), "--share-prefill", "auto"]
    early = R._resolve_share_prefill(R.build_config(R.parse_args(argv)), "anchors")
    assert early.share_prefill_armed is False  # artifact absent at t0
    # The gate-4b battery lands a production PASS mid-phase...
    early.gates_dir.mkdir(parents=True, exist_ok=True)
    (early.gates_dir / R.SHARE_PREFILL_GATE_NAME).write_text(
        json.dumps({"verdict": "PASS", "mode": "production"})
    )
    # ...and every later participant (worker 1, the vLLM legs, capregen)
    # adopts the FROZEN decision instead of arming mid-family.
    late = R._resolve_share_prefill(R.build_config(R.parse_args(argv)), "anchors")
    assert late.share_prefill_armed is False
    capregen = R._resolve_share_prefill(R.build_config(R.parse_args(argv)), "capregen_anchors")
    assert capregen.share_prefill_armed is False
    # A DIFFERENT family (grid, resolving after the PASS landed) arms fresh.
    grid = R._resolve_share_prefill(R.build_config(R.parse_args(argv)), "grid")
    assert grid.share_prefill_armed is True


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
