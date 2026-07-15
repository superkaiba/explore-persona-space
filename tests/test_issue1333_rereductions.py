"""#1333 round-3 concern-closure tests (synthetic fixtures, CPU-only).

Covers the two open-concern closures:
- ``ladder-bystander-dose-curves``: the registered flanking-rung plan
  (``C.dose_curve_rung_plan``) + the extended four-float bystander record
  (``issue1333_dispatch._bystander_record``).
- ``analyzer-rereductions-deferred``: the four §6 re-reductions in
  ``issue1333_geometry`` (matched-80, eval-bank half-split, cross-surface gap
  curve, EOS-margin transfer fractions) + the dose-curve assembly, unit-level
  and end-to-end through ``run_geometry(..., run_root=...)``.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from explore_persona_space.experiments import issue_1333 as C

WINDOW = C.ACCEPT_WINDOW  # (4.28..., 8.28...)


def _mod(name: str):
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module(name)


@pytest.fixture(scope="module")
def G():
    return _mod("issue1333_geometry")


@pytest.fixture(scope="module")
def D():
    return _mod("issue1333_dispatch")


# ── dose_curve_rung_plan (concern ladder-bystander-dose-curves) ───────────────


def test_dose_curve_rung_plan_flanks_below_in_above():
    ladder = {
        20: {"delta_logp_mean": 1.0},  # far below
        40: {"delta_logp_mean": 3.9},  # closest below the window
        60: {"delta_logp_mean": 5.0},  # in window (candidate)
        80: {"delta_logp_mean": 9.5},  # closest above the window
        100: {"delta_logp_mean": 12.0},  # far above
    }
    plan = C.dose_curve_rung_plan(ladder, [60])
    roles = {p["step"]: p["role"] for p in plan}
    assert roles == {40: "sub_window", 60: "candidate", 80: "above_window"}
    assert [p["step"] for p in plan] == [40, 60, 80]  # sorted by step
    assert plan[0]["delta_logp_mean"] == 3.9


def test_dose_curve_rung_plan_no_flanks_and_candidate_wins():
    # All reads in-window: no flanks exist -> candidates only.
    ladder = {10: {"delta_logp_mean": 5.0}, 20: {"delta_logp_mean": 6.0}}
    plan = C.dose_curve_rung_plan(ladder, [10, 20])
    assert {p["step"]: p["role"] for p in plan} == {10: "candidate", 20: "candidate"}
    # Overlap: the closest-below rung is ALSO a candidate -> candidate wins.
    ladder2 = {10: {"delta_logp_mean": 3.0}, 20: {"delta_logp_mean": 6.0}}
    plan2 = C.dose_curve_rung_plan(ladder2, [10, 20])
    assert {p["step"]: p["role"] for p in plan2} == {10: "candidate", 20: "candidate"}


def test_dose_curve_rung_plan_unknown_candidate_raises():
    with pytest.raises(KeyError):
        C.dose_curve_rung_plan({10: {"delta_logp_mean": 5.0}}, [99])


# ── _bystander_record (four-float per-context leakage record) ────────────────


def _slot(logp: float, z_marker: float, z_eos: float, *, emit: bool = False) -> dict:
    return {
        "logp": logp,
        "z_marker": z_marker,
        "z_eos": z_eos,
        "logZ": z_marker - logp,
        "argmax_id": C.MARKER_TOKEN_ID if emit else 0,
    }


def test_bystander_record_margins_gate_and_per_probe(D):
    meta = [
        {"context_id": "chef", "q": 0},
        {"context_id": "chef", "q": 1},
        {"context_id": "hero", "q": 0},
        {"context_id": "hero", "q": 1},
    ]
    trained = [
        _slot(-1.0, 5.0, 3.0),
        _slot(-3.0, 5.0, 3.0),
        _slot(-0.5, 8.0, 2.0, emit=True),
        _slot(-0.5, 8.0, 2.0, emit=True),
    ]
    base = [
        _slot(-3.0, 1.0, 2.0),
        _slot(-5.0, 1.0, 2.0),
        _slot(-4.5, 1.0, 2.0),
        _slot(-4.5, 1.0, 2.0),
    ]
    rec = D._bystander_record(meta, trained, base)
    chef = rec["per_context"]["chef"]
    assert chef["delta_logp_mean"] == pytest.approx(2.0)
    # margin delta = (5-3) - (1-2) = 3.0 per probe
    assert chef["delta_margin_mean"] == pytest.approx(3.0)
    assert chef["emission_rate"] == 0.0
    hero = rec["per_context"]["hero"]
    assert hero["emission_rate"] == 1.0
    assert rec["saturated"] is True  # hero >= 0.92 argmax ceiling
    assert rec["bystander_argmax_rates"] == {"chef": 0.0, "hero": 1.0}
    # per-probe four-float storage rides along verbatim (_delta_record)
    assert len(rec["per_probe"]) == 4
    assert rec["per_probe"][0]["trained"]["z_eos"] == 3.0


# ── matched-80 subsample reads ────────────────────────────────────────────────


def test_matched_80_reads_stats_and_determinism(G):
    rng = np.random.default_rng(0)
    cloud = rng.standard_normal((100, 16))
    out = G.matched_80_reads(cloud)
    assert out["n_sub"] == 80 and out["n_draws"] == 100
    # rank of an 80x16 cloud's Gram is <= 16, so rank-k90 must be too.
    assert 0 < out["rank_k90"]["mean"] <= 16
    assert out["mu_norm"]["mean"] > 0 and out["mu_norm"]["sd"] >= 0
    out2 = G.matched_80_reads(cloud)
    assert out2 == out  # seed-deterministic


def test_matched_80_skips_small_cloud(G):
    out = G.matched_80_reads(np.zeros((50, 8)))
    assert "skipped" in out and "n_rows 50" in out["skipped"]


# ── eval-bank half-split ──────────────────────────────────────────────────────


def _slot_record(train_delta: float, held_delta: float) -> dict:
    per_probe = []
    for q in range(20):
        delta = train_delta if q < 10 else held_delta
        per_probe.append(
            {
                "row": {"q": q},
                "trained": {"logp": -1.0 + delta},
                "base": {"logp": -1.0},
            }
        )
    return {"per_probe": per_probe}


def test_eval_bank_half_split_material_flag(G):
    rec = G.eval_bank_half_split(_slot_record(8.0, 6.5))
    assert rec["train_overlap_mean"] == pytest.approx(8.0)
    assert rec["held_out_mean"] == pytest.approx(6.5)
    assert rec["delta_train_minus_heldout"] == pytest.approx(1.5)
    assert rec["material"] is True  # |Δ| > 1 nat (plan §6)
    assert rec["n_train_overlap"] == 10 and rec["n_held_out"] == 10
    rec2 = G.eval_bank_half_split(_slot_record(6.8, 6.5))
    assert rec2["material"] is False


def test_eval_bank_half_split_requires_both_halves(G):
    rec = {"per_probe": _slot_record(8.0, 6.5)["per_probe"][:10]}  # train half only
    assert "skipped" in G.eval_bank_half_split(rec)


# ── cross-surface gap curve ───────────────────────────────────────────────────


def test_cross_surface_gap_curve_alignment(G):
    traj = {"steps": [10, 20], "delta_nats": [1.0, 2.0]}
    ladder = {20: {"delta_logp_mean": 1.5}, 40: {"delta_logp_mean": 3.0}}
    pts = G.cross_surface_gap_curve(traj, ladder)
    by_step = {p["step"]: p for p in pts}
    assert sorted(by_step) == [10, 20, 40]
    assert by_step[10]["gap"] is None and by_step[10]["off_line_delta"] is None
    assert by_step[20]["gap"] == pytest.approx(0.5)  # in-loop 2.0 - off-line 1.5
    assert by_step[40]["in_loop_delta"] is None and by_step[40]["gap"] is None


def _write_rung_layout(cell_dir: Path, reads: dict[int, float]) -> None:
    for step, delta in reads.items():
        p = cell_dir / "ladder" / f"rung_{step}" / "slot_read.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"delta_logp_mean": delta, "n_probes": 20}))


def test_gap_curve_for_cell_reads_rung_layout(G, tmp_path):
    """The gap-curve driver assembles from the PERSISTED layout
    (band_trajectory.json + ladder/rung_*/slot_read.json) — no consolidated
    ladder.json exists; a regression back to it must fail here."""
    run_root = tmp_path / "run"
    cell = "mk1_lora_con"
    (run_root / cell).mkdir(parents=True)
    (run_root / cell / "band_trajectory.json").write_text(
        json.dumps({"steps": [10, 20], "delta_nats": [1.0, 2.0]})
    )
    _write_rung_layout(run_root / cell, {20: 1.5, 40: 3.0})
    rec = G._gap_curve_for_cell(run_root, cell)
    assert "skipped" not in rec, rec
    by_step = {p["step"]: p for p in rec["points"]}
    assert sorted(by_step) == [10, 20, 40]
    assert by_step[20]["gap"] == pytest.approx(0.5)
    assert rec["n_offline_rungs"] == 2 and rec["n_aligned_steps"] == 1
    assert rec["max_abs_gap"] == pytest.approx(0.5) and rec["max_abs_gap_step"] == 20


def test_gap_curve_for_cell_skips_and_rejects_bad_rungs(G, tmp_path):
    run_root = tmp_path / "run"
    cell = "ext_bare"
    (run_root / cell / "ladder").mkdir(parents=True)
    # trajectory present, zero rungs -> explicit skip
    (run_root / cell / "band_trajectory.json").write_text(
        json.dumps({"steps": [10], "delta_nats": [1.0]})
    )
    assert "skipped" in G._gap_curve_for_cell(run_root, cell)
    # missing trajectory -> explicit skip
    assert "skipped" in G._gap_curve_for_cell(run_root, "ext_icl")
    # unparseable rung dir name -> loud ValueError, never a silent misparse
    bad = run_root / cell / "ladder" / "rung_x2" / "slot_read.json"
    bad.parent.mkdir(parents=True)
    bad.write_text(json.dumps({"delta_logp_mean": 1.0}))
    with pytest.raises(ValueError, match="unparseable ladder rung"):
        G._gap_curve_for_cell(run_root, cell)


# ── reused-cell half split: local-first + scoped HF fetch fallback ────────────


def test_half_split_reused_cell_local_first(G, tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    p = run_root / "gates" / "reused_apply" / "apply_gate.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(_slot_record(6.4, 6.2)))

    def _boom(**kw):  # pragma: no cover - would fail the test if reached
        raise AssertionError("hf_hub_download must not be called when the local file exists")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _boom)
    rec = G._half_split_for_cell(run_root, C.CELL_FT_CON_REUSED)
    assert rec["delta_train_minus_heldout"] == pytest.approx(0.2)
    assert rec["material"] is False and rec["n_train_overlap"] == 10


def test_half_split_reused_cell_fetches_from_hf(G, tmp_path, monkeypatch):
    """Local file absent -> scoped hf_hub_download of the doubled-prefix HF path;
    the downloaded record feeds the SAME half-split definition."""
    run_root = tmp_path / "run"
    staged = tmp_path / "hf_cache" / "apply_gate.json"
    staged.parent.mkdir(parents=True)
    staged.write_text(json.dumps(_slot_record(8.0, 6.5)))
    calls: list[dict] = []

    def fake_download(*, repo_id: str, repo_type: str, filename: str, revision: str) -> str:
        calls.append({"repo_id": repo_id, "repo_type": repo_type, "filename": filename})
        return str(staged)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    rec = G._half_split_for_cell(run_root, C.CELL_FT_CON_REUSED)
    assert rec["delta_train_minus_heldout"] == pytest.approx(1.5) and rec["material"] is True
    assert calls == [
        {
            "repo_id": C.HF_DATA_REPO,
            "repo_type": "dataset",
            "filename": G._REUSED_APPLY_HF_FILENAME,
        }
    ]
    # the fetch stages the record at the canonical local path (offline re-runs)
    dest = run_root / "gates" / "reused_apply" / "apply_gate.json"
    assert dest.exists() and json.loads(dest.read_text()) == json.loads(staged.read_text())


# ── EOS-margin transfer fractions ─────────────────────────────────────────────


def _breadth_record(source_margin: float, chef_margin: float) -> dict:
    def row(label: str, margin: float) -> dict:
        return {
            "row": {"label": label, "q": 0},
            "trained": {"z_marker": margin, "z_eos": 0.0, "logp": -1.0},
            "base": {"z_marker": 0.0, "z_eos": 0.0, "logp": -2.0},
        }

    return {"per_probe": [row("__source__", source_margin), row("chef", chef_margin)]}


def test_eos_margin_transfer_fractions(G):
    rec = G.eos_margin_transfer_fractions(_breadth_record(4.0, 1.0))
    assert rec["source_margin_gain"] == pytest.approx(4.0)
    assert rec["per_context"]["chef"]["transfer_fraction"] == pytest.approx(0.25)
    # degenerate source margin -> fraction is null, never a crash / coerce
    rec2 = G.eos_margin_transfer_fractions(_breadth_record(0.0, 1.0))
    assert rec2["per_context"]["chef"]["transfer_fraction"] is None
    # source label absent -> explicit skip
    rec3 = G.eos_margin_transfer_fractions(
        {"per_probe": _breadth_record(4.0, 1.0)["per_probe"][1:]}
    )
    assert "skipped" in rec3


# ── end-to-end: run_geometry(..., run_root=...) emits re_reductions ───────────


def _stub_store(rng, ctxs, layers, d, *, shift: float) -> dict:
    import torch

    row_meta = [{"context_id": c, "question_idx": q} for c in ctxs for q in range(2)]
    n = len(row_meta)
    arms = {
        arm: {
            li: torch.tensor(rng.standard_normal((n, d)) + shift, dtype=torch.float16)
            for li in layers
        }
        for arm in ("prefix", "context", "response")
    }
    return {"arms": arms, "row_meta": row_meta}


def test_run_geometry_emits_re_reductions(G, tmp_path):
    import torch

    rng = np.random.default_rng(C.BOOT_SEED)
    ctxs, layers, d = ("persona_villain", "chef"), [0, 1], 8
    capture = tmp_path / "capture"
    for rel, shift in (
        ("base_marker/base/pooled.pt", 0.0),
        (f"{C.CELL_LORA_CON}/selected/pooled.pt", 1.0),
        (f"{C.CELL_LORA_CON}/tf_shared/pooled.pt", 0.5),
    ):
        p = capture / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(_stub_store(rng, ctxs, layers, d, shift=shift), p)
    wu_path = tmp_path / "wu_row.pt"
    torch.save(torch.tensor(rng.standard_normal(d), dtype=torch.float32), wu_path)

    run_root = tmp_path / "run"
    cell = C.CELL_LORA_CON
    (run_root / cell / "ladder" / "rung_40").mkdir(parents=True)
    (run_root / cell / "ladder" / "rung_20").mkdir(parents=True)
    (run_root / cell / "selection.json").write_text(
        json.dumps(
            {
                "step": 40,
                "in_window": True,
                "cell": cell,
                "dose_curve_rungs": [
                    {"step": 20, "role": "sub_window", "delta_logp_mean": 2.0, "status": "read"},
                    {"step": 40, "role": "candidate", "delta_logp_mean": 6.3, "status": "read"},
                ],
            }
        )
    )
    # The dispatcher persists per-rung slot reads (ladder/rung_*/slot_read.json);
    # no consolidated ladder.json is ever written (the follow-up layout fix).
    (run_root / cell / "ladder" / "rung_40" / "slot_read.json").write_text(
        json.dumps({**_slot_record(7.0, 5.5), "delta_logp_mean": 6.3})
    )
    (run_root / cell / "ladder" / "rung_20" / "slot_read.json").write_text(
        json.dumps({**_slot_record(2.2, 1.8), "delta_logp_mean": 2.0})
    )
    for step, per_ctx in ((20, 0.4), (40, 1.6)):
        (run_root / cell / "ladder" / f"rung_{step}" / "bystanders.json").write_text(
            json.dumps(
                {
                    "per_context": {
                        "chef": {
                            "delta_logp_mean": per_ctx,
                            "delta_margin_mean": per_ctx * 2,
                            "emission_rate": 0.0,
                        }
                    }
                }
            )
        )
    (run_root / cell / "band_trajectory.json").write_text(
        json.dumps({"steps": [20, 40], "delta_nats": [2.5, 6.5]})
    )
    (run_root / "breadth" / cell).mkdir(parents=True)
    (run_root / "breadth" / cell / "slot_reads.json").write_text(
        json.dumps(_breadth_record(4.0, 1.0))
    )
    (run_root / "gates" / "reused_apply").mkdir(parents=True)
    (run_root / "gates" / "reused_apply" / "apply_gate.json").write_text(
        json.dumps(_slot_record(6.4, 6.2))
    )

    results = G.run_geometry(
        capture,
        tmp_path / "out" / "geometry.json",
        tmp_path / "matrices",
        smoke=True,
        cells=(cell,),
        run_root=run_root,
        wu_path=wu_path,
    )
    rr = results["re_reductions"]
    # matched-80: stub store is 4 rows -> explicit skip, key still present
    assert "skipped" in rr["matched_80"][cell]["own"]
    hs = rr["eval_bank_half_split"]
    assert hs[cell]["delta_train_minus_heldout"] == pytest.approx(1.5)
    assert hs[cell]["material"] is True
    assert hs[C.CELL_FT_CON_REUSED]["delta_train_minus_heldout"] == pytest.approx(0.2)
    assert hs[C.CELL_FT_CON_REUSED]["material"] is False
    assert "skipped" in hs[C.CELL_LORA_POS]  # missing selection -> explicit skip
    gap_rec = rr["cross_surface_gap"][cell]
    gap = gap_rec["points"]
    assert {p["step"]: p["gap"] for p in gap}[20] == pytest.approx(0.5)
    assert gap_rec["n_offline_rungs"] == 2 and gap_rec["n_aligned_steps"] == 2
    assert gap_rec["max_abs_gap"] == pytest.approx(0.5) and gap_rec["max_abs_gap_step"] == 20
    tf = rr["eos_margin_transfer"][cell]
    assert tf["per_context"]["chef"]["transfer_fraction"] == pytest.approx(0.25)
    dc = rr["dose_curves"][cell]
    assert dc["selected_step"] == 40 and len(dc["points"]) == 2
    assert dc["points"][0]["bystanders_per_context"]["chef"]["delta_margin_mean"] == 0.8
    # the schema field is self-describing per the concern resolution
    assert [r["role"] for r in dc["dose_curve_rungs"]] == ["sub_window", "candidate"]
    # persisted JSON round-trips
    assert (tmp_path / "out" / "geometry.json").exists()


# ── run_re_reductions_refresh: fill the two reads, everything else invariant ──


def test_run_re_reductions_refresh_fills_two_reads_only(G, tmp_path):
    run_root = tmp_path / "run"
    cell = C.NEW_LORA_CELLS[0]
    (run_root / cell).mkdir(parents=True)
    (run_root / cell / "band_trajectory.json").write_text(
        json.dumps({"steps": [20, 40], "delta_nats": [2.5, 6.5]})
    )
    _write_rung_layout(run_root / cell, {20: 2.0, 40: 6.3})
    gate = run_root / "gates" / "reused_apply" / "apply_gate.json"
    gate.parent.mkdir(parents=True)
    gate.write_text(json.dumps(_slot_record(6.4, 6.2)))

    existing = {
        "issue": C.ISSUE,
        "cells": {"mk1_lora_con": {"own": {"primary": {"rank_k90": 12}}}},
        "metadata": {"ts": "2026-07-15T00:00:00Z", "git_commit": "deadbeef"},
        "lattices": {"H1_D_rank": {"point": 1.0, "verdict": "FTMoreDiffuse"}},
        "re_reductions": {
            "matched_80": {"mk1_lora_con": {"own": {"n_sub": 80}}},
            "eval_bank_half_split": {
                "mk1_lora_con": {"delta_train_minus_heldout": 0.1, "material": False},
                C.CELL_FT_CON_REUSED: {"skipped": "apply_gate.json missing"},
            },
            "cross_surface_gap": {c: {"skipped": "missing ladder.json"} for c in C.NEW_LORA_CELLS},
            "eos_margin_transfer": {"mk1_lora_con": {"source_margin_gain": 4.0}},
            "dose_curves": {"mk1_lora_con": {"selected_step": 40}},
        },
    }
    out_p = tmp_path / "geometry.json"
    out_p.write_text(json.dumps(existing, indent=2, ensure_ascii=False, default=float) + "\n")
    before_text = out_p.read_text()

    results = G.run_re_reductions_refresh(out_p, run_root)
    after = json.loads(out_p.read_text())
    assert after == results
    # the two reads are FILLED
    gap = after["re_reductions"]["cross_surface_gap"][cell]
    assert gap["max_abs_gap"] == pytest.approx(0.5) and gap["n_aligned_steps"] == 2
    hs = after["re_reductions"]["eval_bank_half_split"][C.CELL_FT_CON_REUSED]
    assert hs["delta_train_minus_heldout"] == pytest.approx(0.2) and hs["material"] is False
    # every other field is byte-identical: masking the two reads in BOTH the
    # original file text and the rewritten one yields identical serializations
    assert G._masked_dump(json.loads(before_text)) == G._masked_dump(after)
    # spot the invariants directly too
    assert after["cells"] == existing["cells"]
    assert after["metadata"] == existing["metadata"]
    assert after["lattices"] == existing["lattices"]
    assert after["re_reductions"]["matched_80"] == existing["re_reductions"]["matched_80"]
    assert (
        after["re_reductions"]["eval_bank_half_split"]["mk1_lora_con"]
        == existing["re_reductions"]["eval_bank_half_split"]["mk1_lora_con"]
    )


def test_run_re_reductions_refresh_raises_on_unresolvable_reused_split(G, tmp_path, monkeypatch):
    """The refresh path is fail-loud: a reused-cell half split that STAYS
    skipped (no local gate, HF fetch failing) raises instead of re-persisting
    another silent skip."""
    run_root = tmp_path / "run"
    run_root.mkdir()
    existing = {
        "re_reductions": {
            "eval_bank_half_split": {C.CELL_FT_CON_REUSED: {"skipped": "missing"}},
            "cross_surface_gap": {c: {"skipped": "missing"} for c in C.NEW_LORA_CELLS},
        }
    }
    out_p = tmp_path / "geometry.json"
    out_p.write_text(json.dumps(existing))

    def _down(**kw):
        raise OSError("offline")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _down)
    monkeypatch.setenv("EPM_HF_RETRY_BUDGET_S", "0")
    with pytest.raises(RuntimeError, match="reused-cell half split still skipped"):
        G.run_re_reductions_refresh(out_p, run_root)
    # and the existing JSON was NOT rewritten
    assert json.loads(out_p.read_text()) == existing
