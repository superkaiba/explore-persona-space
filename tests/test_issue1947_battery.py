"""#1947 unit-3 pins: last-token span capture (binding directive), judge/select
P3 drivers (stub judge, drop tally, verdict lattice), consumed-rows schema
derivation, and the batched bootstrap-CI helper.

The tiny-model capture test executes the REAL `_teacher_forced_span_means`
body (from-config 2-layer same-arch model over the REAL Qwen vocab — the #906
tiny-real convention; CPU, seconds)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1947_battery as bat  # noqa: E402
import issue1947_cells as cells  # noqa: E402

SLUG = "syc-pers-con-sv-s42"  # the pilot cell (a REAL registry slug)


def _cfg(tmp_path: Path, **kw) -> bat.Cfg:
    defaults = dict(
        out_root=tmp_path / "root",
        out_dir=tmp_path / "analysis",
        smoke=True,
        stub_judge=True,
        upload=False,
        local_ladders=True,
        cells_filter=(SLUG,),
        behaviors=("sycophancy",),
    )
    defaults.update(kw)
    cfg = bat.Cfg(**defaults)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _write_ladder(cfg: bat.Cfg, slug: str, rates_shape: dict[str, list[list[str]]]) -> None:
    payload = {
        "slug": slug,
        "behavior": "sycophancy",
        "questions": ["q one?", "q two?"],
        "rungs": {s: {"completions": comps} for s, comps in rates_shape.items()},
    }
    p = cfg.out_root / "ladders" / slug / "ladder_rollouts.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")


def test_judge_stub_tally_and_resume(tmp_path):
    """Stub judge over a 2-rung synthetic ladder: rates in [0,1], drop split
    fields present, per-rung checkpoints regime-keyed (a second run resumes)."""
    cfg = _cfg(tmp_path)
    comps = [["yes indeed", "agree"], ["no", "disagree"]]
    _write_ladder(cfg, SLUG, {"5": comps, "10": comps})
    assert bat.cmd_judge(cfg) == 0
    judged = json.loads((cfg.out_dir / "judge" / f"judged_{SLUG}.json").read_text())
    assert set(judged["rates_by_step"]) == {"5", "10"}
    for rec in judged["records"].values():
        assert 0.0 <= rec["rate"] <= 1.0
        assert "n_dropped_draws_content" in rec and "n_transport_lost_draws" in rec
    assert bat.cmd_judge(cfg) == 0  # resume path: checkpoints reused, same regime


def test_select_verdict_lattice_earliest_in_band(tmp_path, monkeypatch):
    """Earliest-in-band rung wins; a never-in-band ladder falls back to
    closest-approach with in_band False (the DoseSelection invariant)."""
    cfg = _cfg(tmp_path)
    judge_dir = cfg.out_dir / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    in_band = {"5": 0.2, "10": 0.7, "15": 0.9}  # earliest in [0.60, 0.85] = 10
    (judge_dir / f"judged_{SLUG}.json").write_text(
        json.dumps(
            {
                "slug": SLUG,
                "behavior": "sycophancy",
                "instrument": "stub-smoke",
                "questions_sha256": "x",
                "rates_by_step": in_band,
                "records": {},
            }
        )
    )
    monkeypatch.setattr(bat, "_marker_cells", lambda _cfg: [])
    assert bat.cmd_select(cfg) == 0
    man = json.loads(cfg.verdict_manifest_path().read_text())
    sel = man["content"][SLUG]["selection"]
    assert sel["step"] == 10 and sel["in_band"] is True and sel["fallback"] is None
    # closest-approach fallback
    (judge_dir / f"judged_{SLUG}.json").write_text(
        json.dumps(
            {
                "slug": SLUG,
                "behavior": "sycophancy",
                "instrument": "stub-smoke",
                "questions_sha256": "x",
                "rates_by_step": {"5": 0.1, "10": 0.3},
                "records": {},
            }
        )
    )
    assert bat.cmd_select(cfg) == 0
    sel = json.loads(cfg.verdict_manifest_path().read_text())["content"][SLUG]["selection"]
    assert sel["in_band"] is False and sel["fallback"] == "closest_approach"
    assert sel["step"] == 10  # min band distance


def test_consumed_row_idxs_schemas(tmp_path):
    """Primary: the train/sft.py seam schema (realized_step_of_idx, 0-based);
    plus the two legacy fixture shapes and the rep-cell no-seam rule."""
    cfg = _cfg(tmp_path)
    p = cfg.out_root / "ladders" / SLUG / "realized_consumption.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"realized_step_of_idx": [0, 0, 1, 2, None]}))
    assert bat._consumed_row_idxs(cfg, SLUG, 2) == {0, 1, 2}
    p.write_text(json.dumps({"row_to_step": {"0": 0, "1": 0, "2": 1, "3": 2}}))
    assert bat._consumed_row_idxs(cfg, SLUG, 2) == {0, 1, 2}
    p.write_text(json.dumps({"steps": {"0": [0, 1], "1": [2], "2": [3]}}))
    assert bat._consumed_row_idxs(cfg, SLUG, 2) == {0, 1, 2}
    rep = "imp-pers-con-rep-s42"  # no seam: all-rows once epoch 1 completes
    assert bat._consumed_row_idxs(cfg, rep, 15) == set(range(80))
    with pytest.raises(RuntimeError, match="unknowable"):
        bat._consumed_row_idxs(cfg, rep, 2)


def test_adapter_subfolder_marker_vs_content():
    assert bat._adapter_subfolder(SLUG, 10) == f"issue1947/{SLUG}/checkpoint-10"
    assert bat._adapter_subfolder("mk-pers-con-sv-s42", 40) == (
        "issue1947/marker/mk-pers-con-sv-s42/checkpoint-40"
    )


def test_corpus_arm_slugs_are_registry_cells():
    for slug in bat.CORPUS_ARM_SLUGS:
        assert slug in cells.CELL_BY_SLUG, slug
    assert len(bat.CORPUS_ARM_SLUGS) == 12


def test_boot_cos_ci_batched_contains_point():
    import issue1947_analysis as ana
    import numpy as np

    rng = np.random.default_rng(0)
    cand = rng.normal(size=32)
    stack = cand[None, :] + 0.5 * rng.normal(size=(200, 32))
    ci, mean_cos = ana._boot_cos_ci(stack, cand, 400, seed=1)
    # NOTE: point-coverage is NOT a percentile-bootstrap guarantee for the
    # concave cos statistic (small downward bias vs the plug-in point — the
    # same shape as the reused #1768 _boot_ci convention). Pins: the bootstrap
    # mean sits inside its own CI; the batched weight-GEMM reproduces a serial
    # per-draw loop over the SAME weights (batched-rewrite equivalence).
    assert ci[0] <= mean_cos <= ci[1]
    assert ci[1] - ci[0] > 1e-4  # non-degenerate width
    rng2 = np.random.default_rng(1)
    n = stack.shape[0]
    W = rng2.multinomial(n, np.full(n, 1.0 / n), size=400).astype(np.float64) / n
    cn = cand / np.linalg.norm(cand)
    serial = np.array([_c for _c in ((w @ stack) @ cn / np.linalg.norm(w @ stack) for w in W)])
    batched = (W @ stack) @ cn / np.linalg.norm(W @ stack, axis=1)
    assert np.allclose(serial, batched, atol=1e-10)
    # ordering sanity: an anti-aligned candidate reads a strictly lower CI
    ci_neg, _ = ana._boot_cos_ci(stack, -cand, 400, seed=1)
    assert ci_neg[1] < ci[0]


def test_pilot_gate_halts_rc7_on_over_2x_projection(tmp_path, monkeypatch):
    """The P4 in-run pilot gate (plan §9): >2x the booked GPU-h HALTs with the
    DISTINCT rc + a report JSON (the #1415 artifact-routed convention), never
    an anonymous crash. Boundary fake: subprocess.run only (signature-shaped)."""
    import subprocess as sp

    cfg = _cfg(tmp_path)
    man = {
        "issue": 1947,
        "band": [0.6, 0.85],
        "content": {
            SLUG: {
                "slug": SLUG,
                "behavior": "sycophancy",
                "selection": {"step": 10, "rate": 0.7, "in_band": True, "fallback": None},
            }
        },
        "marker": {},
    }
    (cfg.out_dir / "verdict_manifest.json").write_text(json.dumps(man))
    cfg = _cfg(tmp_path, smoke=False, sentinel_dir=tmp_path / "logs")

    def fake_run(  # boundary fake mirroring BOTH battery call shapes:
        cmd,  # pilot/unit launch (env+stdout+stderr) and _git_short_sha (capture_output+text+cwd)
        env=None,
        stdout=None,
        stderr=None,
        capture_output=False,
        text=False,
        cwd=None,
        check=False,
    ):
        return sp.CompletedProcess(cmd, 0, stdout="deadbee\n" if capture_output else None)

    monkeypatch.setattr(bat.subprocess, "run", fake_run)
    monkeypatch.setattr(bat, "PLAN_P4P5_GPU_H", 1e-12)  # any real pilot wall trips 2x
    monkeypatch.setattr(bat, "_physical_gpus", lambda: [0])
    rc = bat.cmd_capture_fit(cfg, ["echo"])
    assert rc == bat.PILOT_GATE_RC == 7
    rep = json.loads((cfg.out_dir / "pilot_gate_report.json").read_text())
    assert rep["ratio"] > 2 and rep["pilot_rc"] == 0


@pytest.mark.slow
def test_last_token_span_capture_tiny_real_model(tmp_path):
    """REAL `_teacher_forced_span_means` body on a from-config 2-layer Qwen2
    over the REAL vocab: all 5 spans present, and a prefix of length 1 makes
    span-mean('prefix') == last-token('prefix_last') exactly (1-token span
    identity — the directive's same-forward-pass guarantee)."""
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    from explore_persona_space.analysis.representation_shift import (
        SPAN_ARMS_LAST,
        _teacher_forced_span_means,
    )

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    config = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model_dir = tmp_path / "tiny"
    Qwen2ForCausalLM(config).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    ids = tok("hello there, what is up today?", add_special_tokens=False)["input_ids"]
    rows = [
        {
            "persona": "t",
            "question_idx": i,
            "prompt_sha": f"s{i}",
            "prompt_token_ids": ids,
            "response_token_ids": ids[:3],
            "prefix_len": 1,  # 1-token prefix: prefix == prefix_last exactly
            "context_len": len(ids),
        }
        for i in range(2)
    ]
    pooled = _teacher_forced_span_means(
        str(model_dir),
        rows,
        ["t"],
        layers=[1],
        spans=("prefix", "context", "response", *SPAN_ARMS_LAST),
        device="cpu",
        dtype=torch.float32,
        tf_batch_size=2,
    )
    assert set(pooled) == {"prefix", "context", "response", "prefix_last", "context_last"}
    assert torch.allclose(pooled["prefix"][1], pooled["prefix_last"][1])
    assert not torch.allclose(pooled["context"][1], pooled["context_last"][1])
    for span in pooled:
        assert pooled[span][1].shape == (2, 32)
        assert torch.isfinite(pooled[span][1]).all()


def test_consumed_row_idxs_marker_no_seam(tmp_path):
    """r1 Critical 1 regression pin: marker cells train WITHOUT the seam — the
    consumed set is the FULL mix iff the rung completes the single-visit epoch
    (step*16 >= 6,400), and None (skip signal) below it. NEVER a
    FileNotFoundError on the never-written realized_consumption.json."""
    cfg = _cfg(tmp_path)
    mk = "mk-pers-con-sv-s42"
    n_rows = cells.CELL_BY_SLUG[mk].n_rows
    assert bat._consumed_row_idxs(cfg, mk, 400) == set(range(n_rows))
    assert bat._consumed_row_idxs(cfg, mk, 100) is None


def test_marker_tree_subsample_deterministic_stratified():
    """Marker full-tree cap (r1 Critical 1 scope decision): seeded, composition
    preserved (1:4 pos:neg), mix order kept."""
    rows = [{"question_idx": i, "row_kind": "pos" if i % 5 == 0 else "neg"} for i in range(6400)]
    a = bat._marker_tree_subsample(rows, 42)
    b = bat._marker_tree_subsample(rows, 42)
    assert a == b
    assert len(a) == bat.MARKER_TREE_ROWS
    n_pos = sum(1 for r in a if r["row_kind"] == "pos")
    assert abs(n_pos / len(a) - 1 / 5) < 0.02  # proportions preserved
    idxs = [r["question_idx"] for r in a]
    assert idxs == sorted(idxs)  # mix order (question_idx == mix row index)


def test_battery_sentinel_poller_conformant(tmp_path):
    """r1 Critical 2 regression pin: the battery done-sentinel carries the
    poller's required envelope keys and round-trips through the REAL
    poll_pipeline._parse_sentinel (dry-run on the serialized body)."""
    import poll_pipeline as pp

    cfg = _cfg(tmp_path)
    payload = {"issue": 1947, "phase": "battery", "status": "done", "failed_units": []}
    sent = bat._battery_sentinel(cfg, payload)
    assert set(pp._SENTINEL_REQUIRED_KEYS) <= set(sent)
    assert sent["sentinel_schema_version"] == pp.SENTINEL_SCHEMA_VERSION_SUPPORTED
    parsed = pp._parse_sentinel("issue-1947-battery-done.json", json.dumps(sent))
    assert parsed is not None and parsed["kind"] == "epm:smoke-result"
    assert parsed["payload"]["status"] == "done"
    non_smoke = bat._battery_sentinel(_cfg(tmp_path, smoke=False), payload)
    assert non_smoke["kind"] == "epm:results" and non_smoke["blocks_pipeline"] is True


def test_runnable_fits_gating():
    """r1 Critical 1: a failed corpus unit skips ONLY its own fit — the rest of
    P5 runs (never zeroed out by one capture failure)."""
    fits = ["fit:a", "fit:b", "fit:c"]
    runnable, skipped = bat._runnable_fits(fits, ["corpus:b", "arm:z"])
    assert runnable == ["fit:a", "fit:c"] and skipped == ["fit:b"]
    runnable, skipped = bat._runnable_fits(fits, [])
    assert runnable == fits and skipped == []


def test_select_marker_consume_slot_reads(tmp_path):
    """cmd_select consumes the worker's programmatic marker selection verbatim
    (r1 style nit: the marker-consume path was CLI-smoke-only)."""
    mk = "mk-pers-con-sv-s42"
    cfg = _cfg(tmp_path, cells_filter=(SLUG, mk))
    judge_dir = cfg.out_dir / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    (judge_dir / f"judged_{SLUG}.json").write_text(
        json.dumps(
            {
                "slug": SLUG,
                "behavior": "sycophancy",
                "instrument": "stub-smoke",
                "questions_sha256": "x",
                "rates_by_step": {"10": 0.7},
                "records": {},
            }
        )
    )
    slot = cfg.out_root / "marker_ladders" / mk / "slot_reads.json"
    slot.parent.mkdir(parents=True, exist_ok=True)
    slot.write_text(json.dumps({"selection": {"step": 40, "read": "band-entry"}}))
    assert bat.cmd_select(cfg) == 0
    man = json.loads(cfg.verdict_manifest_path().read_text())
    assert man["marker"][mk]["selection"]["step"] == 40
    assert man["marker"][mk]["behavior"] == "marker"


def test_r3_floor_crosscheck_tolerance_and_smoke_demotion(tmp_path, monkeypatch):
    """p5-m0-floors concern wiring: recomputed M0/floor vs the r3 committed
    values — pass within tolerance, fail LOUD on divergence (non-smoke), and
    DEMOTED to informational under --smoke (the #1345 smoke-gate rule: toy-n
    fits cannot reproduce production anchors)."""
    r3 = {"fits": {"M0": {"heldout_r2": 0.60}}, "map_change": {"floor_p95": 8.5}, "n_test": 1000}

    def fake_stage(repo, path, local, repo_type):
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(r3))

    monkeypatch.setattr(
        bat.hub, "retry_transient", lambda fn, what=None: ["pfx/fits_bare_n/arm_L19.json"]
    )
    monkeypatch.setattr(bat.hub, "stage_hub_file", fake_stage)
    cfg = _cfg(tmp_path, smoke=False)
    rec_ok = {
        "fits": {"M0": {"heldout_r2": 0.605}},
        "map_change": {"floor_p95": 8.6},
        "n_test": 1000,
    }
    bat._r3_floor_crosscheck(cfg, "syc-pers-con-sv-s42", {19: rec_ok})
    out = json.loads(
        (cfg.out_root / "fits" / "r3_crosscheck" / "syc-pers-con-sv-s42.json").read_text()
    )
    assert out["verdict"] == "pass" and out["checks"][0]["ok"] is True
    rec_bad = {
        "fits": {"M0": {"heldout_r2": 0.20}},
        "map_change": {"floor_p95": 30.0},
        "n_test": 1000,
    }
    with pytest.raises(RuntimeError, match="diverge"):
        bat._r3_floor_crosscheck(cfg, "imp-pers-con-sv-s42", {19: rec_bad})
    cfg_smoke = _cfg(tmp_path, smoke=True)
    bat._r3_floor_crosscheck(cfg_smoke, "cas-pers-con-sv-s42", {19: rec_bad})
    out = json.loads(
        (cfg_smoke.out_root / "fits" / "r3_crosscheck" / "cas-pers-con-sv-s42.json").read_text()
    )
    assert out["verdict"] == "informational-smoke" and out["n_divergent"] == 1


def _toy_store(path: Path, qidx: list[int], spans: tuple[str, ...], seed: int, d: int = 16):
    import torch

    g = torch.Generator().manual_seed(seed)
    payload = {
        "row_question_idx": list(qidx),
        "row_sha": [f"sha{q}" for q in qidx],
        "arms": {s: {1: torch.randn(len(qidx), d, generator=g)} for s in spans},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def test_fit_lasttoken_arm_real_body_toy_n(tmp_path):
    """Directive v7 items 2-3 (r1 Major 1): the within-run LAST-TOKEN M⁺ fit
    runs the REAL _fit_map/_map_reads/_identity_bias_reads bodies at toy n on
    CPU and writes one *_lasttoken.json per (arm, layer) — identity+bias + kNN
    attached, D column flagged no-M0-floor-available (no probe report, and the
    r1 base store carries no context_last arm)."""
    cfg = _cfg(tmp_path, smoke=True, layers=(1,))
    n = 90
    sample = {
        "rows": [{"sha": f"sha{i}", "src_qidx": i} for i in range(n)],
        "n_train": 60,
        "n_val": 15,
        "n_test": 15,
    }
    sp = cfg.out_root / "on_target" / "inputs" / "corpus_sample_pfx.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(sample))
    base_dir = cfg.out_root / "corpus_capture" / "base_content"
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "rows_spans.json").write_text("{}")
    qidx = list(range(n))
    # base store WITHOUT context_last (the r1-base reality) -> M0 unavailable
    _toy_store(base_dir / "pooled.pt", qidx, spans=("context", "response"), seed=1)
    _toy_store(
        cfg.out_root / "corpus_capture" / SLUG / "pooled.pt",
        qidx,
        spans=("context", "context_last", "response"),
        seed=2,
    )
    bat._fit_lasttoken_arm(cfg, SLUG)
    rec = json.loads(
        (cfg.out_root / "fits" / "lasttoken" / f"{SLUG}_L1_lasttoken.json").read_text()
    )
    assert rec["input_span"] == "context_last" and rec["n_test"] == 15
    mp = rec["fits"]["Mplus_lasttoken"]
    assert "heldout_r2" in mp and "knn_cosine" in mp
    assert mp["identity_bias"]["applicable"] is True
    assert rec["fits"]["M0_lasttoken"]["status"] == "unavailable"
    assert rec["map_change"]["status"] == "no-M0-floor-available"
    assert rec["map_change"]["D"] is None
