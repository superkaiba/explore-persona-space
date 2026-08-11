"""Round-2 regression pins for the issue-2224 drivers (code-review r1).

Pins: (1) C1 — the rule-26 pilot budget derives from the ARM COUNT so every
arm clears judge_pilot_gate's 10-effective-draw floor at the PRODUCTION census
(84 coherence arms / ~28 per-trait arms; the shipped fixed 200-draw default
was structurally un-passable there); (2) the map/probe npz contract loaders
(the P1-gate seam) incl. the A11 input_pooling fail-loud-on-absence gate;
(3) ranked_ids NaN fail-loud + deterministic tie-break; (4) M1 — no phase-done
sentinel from a --cells-filtered run; (5) M2 — decoding-regime-keyed eval
resume + generations-sha-keyed judge resume; (6) M2 gen-side — truncated-row
drop/rescan + the gen_regime sidecar refusals; (7) M4 — the selection-census
gate before the eval panel draw.

All paths are tmp_path-rooted; no network, no GPU, no canonical writes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import issue2224_finetune_sweep as sweep  # noqa: E402
import issue2224_gen_natural as gen  # noqa: E402
import issue2224_predictor_scores as ps  # noqa: E402
import issue2224_select as sel  # noqa: E402

# ── C1: pilot budget derived from the arm count ──────────────────────────────────


@pytest.mark.parametrize("n_arms", [84, 28, 12, 2])
def test_pilot_budget_clears_gate_floor_at_production_census(n_arms):
    target = sweep.pilot_target_draws(n_arms, sweep.PILOT_N_DRAWS, 200)
    # judge_pilot_gate's own sizing arithmetic (judge_pilot.py:367):
    per_arm_items = max(1, target // (n_arms * sweep.PILOT_N_DRAWS))
    effective = per_arm_items * sweep.PILOT_N_DRAWS
    assert effective >= sweep.PILOT_MIN_EFFECTIVE_DRAWS, (n_arms, target, effective)


def test_pilot_budget_shipped_default_was_unpassable_and_is_raised():
    # Pre-fix shape: 84 coherence arms x n_draws=2 at target 200 ->
    # max(1, 200 // 168) * 2 = 2 effective draws/arm < the 10-draw floor.
    old_effective = max(1, 200 // (84 * 2)) * 2
    assert old_effective < sweep.PILOT_MIN_EFFECTIVE_DRAWS
    assert sweep.pilot_target_draws(84, 2, 200) > 200  # the fix auto-raises


def test_pilot_budget_keeps_a_sufficient_request():
    # 2 arms x 2 draws: 200 already gives 100 draws/arm — request preserved.
    assert sweep.pilot_target_draws(2, 2, 200) == 200


# ── Map / probe npz contract loaders (P1-gate seam) ─────────────────────────────


def _write_map_npz(path: Path, meta: dict | None, drop_key: str | None = None):
    arrs = {
        "w": np.zeros((1, 8, 8), dtype=np.float16),
        "x_mu": np.zeros((1, 1, 8), dtype=np.float32),
        "x_sd": np.ones((1, 1, 8), dtype=np.float32),
        "y_mu": np.zeros((1, 1, 8), dtype=np.float32),
        "layers": np.array([10]),
    }
    if drop_key:
        arrs.pop(drop_key)
    if meta is not None:
        arrs["meta"] = np.array(json.dumps(meta))
    np.savez(path, **arrs)


def test_load_linear_map_roundtrip_and_missing_key(tmp_path):
    good = tmp_path / "map.npz"
    _write_map_npz(good, {"input_pooling": "context_end"})
    m, layers, meta = ps.load_linear_map(good)
    assert layers == [10] and meta["input_pooling"] == "context_end"
    assert m.w.shape == (1, 8, 8)
    bad = tmp_path / "bad.npz"
    _write_map_npz(bad, {"input_pooling": "context_end"}, drop_key="x_sd")
    with pytest.raises(RuntimeError, match="missing npz keys"):
        ps.load_linear_map(bad)


def test_check_map_pooling_a11(tmp_path):
    # Present + matching -> verified.
    assert ps.check_map_pooling({"input_pooling": "context_end"}, "context", "p", False) == (
        "verified"
    )
    # Present + mismatching -> raise (unchanged behavior).
    with pytest.raises(RuntimeError, match="A11 pooling-convention mismatch"):
        ps.check_map_pooling({"input_pooling": "response_avg"}, "prefix", "p", False)
    # ABSENT without the flag -> raise (r1 Major-2: never a silent no-op).
    with pytest.raises(RuntimeError, match="NO 'input_pooling' key"):
        ps.check_map_pooling({}, "context", "p", False)
    # ABSENT with --allow-unverified-map-pooling -> recorded absence.
    assert ps.check_map_pooling({}, "prefix", "p", True) == "absent"


def test_load_probe_contract(tmp_path):
    good = tmp_path / "probe.npz"
    np.savez(good, w=np.zeros(8), b=np.array(0.5), layer=np.array(10))
    probe = ps.load_probe(good, hidden=8, layer=10)
    assert probe["b"] == 0.5
    with pytest.raises(RuntimeError, match="probe layer"):
        ps.load_probe(good, hidden=8, layer=11)
    lonely = tmp_path / "lonely.npz"
    np.savez(lonely, w=np.zeros(8), b=np.array(0.0), x_mu=np.zeros(8))
    with pytest.raises(RuntimeError, match="exactly ONE of x_mu/x_sd"):
        ps.load_probe(lonely, hidden=8, layer=0)


# ── ranked_ids: NaN fail-loud + deterministic tie-break ──────────────────────────


def test_ranked_ids_nan_raises():
    scores = {"a": {"raw": 1.0}, "b": {"raw": float("nan")}, "c": {"raw": 0.5}}
    with pytest.raises(RuntimeError, match="NaN"):
        sel.ranked_ids(scores, "raw")


def test_ranked_ids_tie_break_deterministic():
    scores = {"b": {"raw": 1.0}, "a": {"raw": 1.0}, "c": {"raw": 2.0}}
    assert sel.ranked_ids(scores, "raw") == ["c", "a", "b"]  # score desc, id asc on ties


# ── M1: no phase-done sentinel from a --cells-filtered run ───────────────────────


def test_finalize_phase_sentinel_skips_on_cells_filter(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sweep, "SENTINEL_DIR", tmp_path)
    wrote = sweep.finalize_phase_sentinel(
        ".done_4b4", "train", {"phase": "4b-4 train", "n_cells": 1}, "lmsys__evil__exact_dp__top"
    )
    assert wrote is False and not (tmp_path / ".done_4b4").exists()
    assert "[phase=done]" not in capsys.readouterr().out
    wrote = sweep.finalize_phase_sentinel(
        ".done_4b4", "train", {"phase": "4b-4 train", "n_cells": 78}, None
    )
    assert wrote is True and (tmp_path / ".done_4b4").exists()
    assert "[phase=done] train" in capsys.readouterr().out


# ── M2: decoding-regime-keyed eval resume + sha-keyed judge resume ───────────────


def _sweep_args(extra: list[str] | None = None):
    return sweep.build_argparser().parse_args(
        ["--phase", "eval", "--n-questions", "2", "--gen-draws", "2"] + (extra or [])
    )


def _write_cell(out_root: Path, cid: str, cap: int, cap_hit: float, n_rows: int = 4):
    gd = out_root / "postft_eval" / cid
    gd.mkdir(parents=True, exist_ok=True)
    rows = [
        {"qid": f"q{i}", "draw": 0, "response": f"r{i}", "finish_reason": "stop"}
        for i in range(n_rows)
    ]
    (gd / "generations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    meta = {
        "decoding": {"temperature": 1.0, "n": 2, "max_new_tokens": cap},
        "cap_hit_fraction": cap_hit,
    }
    (gd / "meta.json").write_text(json.dumps(meta))


def test_gen_complete_regime_keying(tmp_path):
    args = _sweep_args(["--max-new-tokens", "2048"])
    _write_cell(tmp_path, "cell_a", cap=2048, cap_hit=0.0)
    assert sweep.gen_complete(tmp_path, "cell_a", 4, args) is True
    # Raised cap + zero cap-hit: still complete (a raise cannot affect it).
    args_up = _sweep_args(["--max-new-tokens", "4096"])
    assert sweep.gen_complete(tmp_path, "cell_a", 4, args_up) is True
    # Cap-hit cell + raised cap: pending under --regen-truncated, LOUD without.
    _write_cell(tmp_path, "cell_b", cap=2048, cap_hit=0.05)
    args_regen = _sweep_args(["--max-new-tokens", "4096", "--regen-truncated"])
    assert sweep.gen_complete(tmp_path, "cell_b", 4, args_regen) is False
    with pytest.raises(RuntimeError, match="--regen-truncated"):
        sweep.gen_complete(tmp_path, "cell_b", 4, args_up)
    # Same cap, cap-hit>0, --regen-truncated: pending (the >2% trigger).
    args_same_regen = _sweep_args(["--max-new-tokens", "2048", "--regen-truncated"])
    assert sweep.gen_complete(tmp_path, "cell_b", 4, args_same_regen) is False
    # Temperature/draws drift: always LOUD (never silently mix regimes).
    args_temp = _sweep_args(["--max-new-tokens", "2048", "--gen-temperature", "0.7"])
    with pytest.raises(RuntimeError, match="decoding"):
        sweep.gen_complete(tmp_path, "cell_a", 4, args_temp)


def test_judged_current_keys_on_generations_sha(tmp_path):
    _write_cell(tmp_path, "cell_a", cap=2048, cap_hit=0.0)
    gpath = tmp_path / "postft_eval" / "cell_a" / "generations.jsonl"
    scores_dir = tmp_path / "trait_scores"
    (scores_dir / "cell_a").mkdir(parents=True)
    from issue2224_common import sha256_file

    (scores_dir / "cell_a" / "trait_scores.json").write_text(
        json.dumps({"generations_sha256": sha256_file(gpath)})
    )
    assert sweep.judged_current(scores_dir, tmp_path, "cell_a") is True
    # Re-generated content (M2) -> stale record -> re-judge.
    gpath.write_text(gpath.read_text() + json.dumps({"qid": "q9", "response": "new"}) + "\n")
    assert sweep.judged_current(scores_dir, tmp_path, "cell_a") is False
    # Legacy record without the sha field -> not current (strict).
    (scores_dir / "cell_a" / "trait_scores.json").write_text(json.dumps({}))
    assert sweep.judged_current(scores_dir, tmp_path, "cell_a") is False


# ── M2 gen-side: truncated-row drop + regime sidecar ─────────────────────────────


def _gen_args(extra: list[str] | None = None):
    return gen.build_argparser().parse_args(
        ["--corpus", "lmsys", "--max-new-tokens", "2048"] + (extra or [])
    )


def test_drop_truncated_rows_and_exclude_scan(tmp_path):
    rows = [
        {"sample_id": "s1", "response": "ok", "finish_reason": "stop"},
        {"sample_id": "s2", "response": "cut", "finish_reason": "length"},
    ]
    f = tmp_path / "gen_lmsys_s00_1_00000.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    # Read-only view (--plan): truncated row reads as NOT done.
    assert gen.scan_done_ids(tmp_path) == {"s1", "s2"}
    assert gen.scan_done_ids(tmp_path, exclude_truncated=True) == {"s1"}
    # Fan-out rewrite: the truncated row is dropped for re-generation.
    assert gen.drop_truncated_rows(tmp_path) == 1
    assert gen.scan_done_ids(tmp_path) == {"s1"}
    # All-truncated file is unlinked, not left empty.
    f2 = tmp_path / "gen_lmsys_s00_1_00001.jsonl"
    f2.write_text(json.dumps({"sample_id": "s3", "finish_reason": "length"}) + "\n")
    assert gen.drop_truncated_rows(tmp_path) == 1
    assert not f2.exists()


def test_check_gen_regime_refusals(tmp_path):
    gen.write_gen_regime(tmp_path, _gen_args())
    gen.check_gen_regime(tmp_path, _gen_args())  # same regime: fine
    with pytest.raises(RuntimeError, match="--regen-truncated"):
        gen.check_gen_regime(tmp_path, _gen_args(["--max-new-tokens", "4096"]))
    # Raised cap WITH the flag: legal (re-gen path).
    gen.check_gen_regime(tmp_path, _gen_args(["--max-new-tokens", "4096", "--regen-truncated"]))
    with pytest.raises(RuntimeError, match="never lower"):
        gen.check_gen_regime(tmp_path, _gen_args(["--max-new-tokens", "1024"]))
    with pytest.raises(RuntimeError, match="model"):
        gen.check_gen_regime(tmp_path, _gen_args(["--model", "other/model"]))


# ── M4: selection census before the eval panel draw ──────────────────────────────


def _write_manifest(d: Path, method: str, tail: str, status: str = "ok"):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{method}__{tail}.json").write_text(
        json.dumps({"method": method, "tail": tail, "status": status, "cell_id": "x"})
    )


def test_selection_census_gate(tmp_path):
    d = tmp_path / "lmsys" / "evil"
    _write_manifest(d, "exact_dp", "top")
    _write_manifest(d, "exact_dp", "bottom")
    _write_manifest(d, "random", "shared")
    # top_filtered missing (panel drawn between select and apply-filter) -> LOUD.
    with pytest.raises(RuntimeError, match="top_filtered MISSING"):
        sweep.assert_selection_census(tmp_path, "lmsys", "evil")
    # filter-collapsed counts as census-present (a finding, never trained).
    _write_manifest(d, "exact_dp", "top_filtered", status="filter-collapsed")
    sweep.assert_selection_census(tmp_path, "lmsys", "evil")
    # A second method with only a top manifest -> LOUD on its missing tails.
    _write_manifest(d, "mapped_dp_context", "top")
    with pytest.raises(RuntimeError, match="mapped_dp_context__bottom"):
        sweep.assert_selection_census(tmp_path, "lmsys", "evil")
