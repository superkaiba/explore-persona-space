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
gate before the eval panel draw; (8) the 4b-3 judge-filter parse-fail re-draw
recovery (bounded same-instrument re-issue, k<=3 attempts/item,
first-parsed-wins merge, rule-24(ii) per-round cache dirs, post-recovery
rule-26 gate arithmetic).

All paths are tmp_path-rooted; no network, no GPU, no canonical writes. The
re-draw tests fake ONLY the API boundary (`judge_graded`) with a
signature-mirroring fake that writes `save_raw` and returns the REAL reduce
(`judge_result_from_save_raw`), so selection / merge / gate bodies execute
for real.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import ClassVar

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
import issue2224_suite_slice as suite  # noqa: E402

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
    # P1-gate reconcile: the realized #1739 meta key is 'variant' (git 795f4747)
    # with the same value strings -> verified; mismatch still raises.
    assert ps.check_map_pooling({"variant": "context_end"}, "context", "p", False) == "verified"
    with pytest.raises(RuntimeError, match="A11 pooling-convention mismatch"):
        ps.check_map_pooling({"variant": "context_end"}, "prefix", "p", False)


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


# ── 4b-3 judge-filter parse-fail re-draw recovery ─────────────────────────────────
#
# Draw-dict shapes mirror the live mint sites (batch_judge / judge_dispatch):
# kept verdicts carry {"score": N, "stop_reason": ...}; parse failures are
# {"error": True, "reason": "parse_error", "stop_reason": ...}; transport rows
# carry the structural transport flag; api-refusal rows stop_reason "refusal".

MALFORMED = {"error": True, "reason": "parse_error", "stop_reason": "end_turn"}
REFUSAL = {"score": "REFUSAL", "stop_reason": "end_turn"}
TRANSPORT = {"error": True, "transport": True, "reason": "transient exhausted"}
TRUNCATED = {"error": True, "reason": "parse_error", "stop_reason": "max_tokens"}
API_REFUSAL = {"error": True, "reason": "api_refusal", "stop_reason": "refusal"}


def _kept(score):
    return {"score": score, "stop_reason": "end_turn"}


def _cid(iid, comp=0):
    return f"{iid}__00000__{comp:02d}"


def _write_save_raw(path: Path, all_scores: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"all_scores": all_scores}))


class _ScriptedJudge:
    """Signature-mirroring ``judge_graded`` fake (API boundary only).

    Each call pops one script entry ({item_id: [parsed draw, ...]}), writes a
    real ``save_raw`` file from it, and returns the REAL reduce
    (``judge_result_from_save_raw``) — zero API calls; the wrapper's
    selection / merge / accounting bodies all execute for real.
    """

    def __init__(self, script: list[dict]):
        self.script = list(script)
        self.calls: list[dict] = []

    def __call__(
        self,
        items,
        eval_prompt,
        *,
        n_draws,
        cache_dir,
        save_raw,
        judge_model=None,
        temperature=None,
        max_tokens=64,
        dry_run=False,
        threshold_base=None,
    ):
        from explore_persona_space.eval.graded_judge import judge_result_from_save_raw

        draws = self.script.pop(0)
        self.calls.append(
            {
                "items": [iid for iid, _q, _a in items],
                "n_draws": n_draws,
                "cache_dir": Path(cache_dir),
                "save_raw": Path(save_raw),
                "max_tokens": max_tokens,
                "threshold_base": threshold_base,
            }
        )
        all_scores = {}
        for iid, _q, _a in items:
            for k, parsed in enumerate(draws[iid]):
                all_scores[_cid(iid, comp=k)] = parsed
        _write_save_raw(Path(save_raw), all_scores)
        return judge_result_from_save_raw(Path(save_raw), items)


def _items(ids):
    return [(iid, f"q-{iid}", f"r-{iid}") for iid in ids]


def test_malformed_drop_counts_selects_only_parse_failures(tmp_path):
    """(1) Selection: only MALFORMED draws are re-draw candidates — refusal /
    transport / truncation / api-refusal keep their classes; kept draws and
    foreign item_ids are excluded."""
    f = tmp_path / "raw.json"
    _write_save_raw(
        f,
        {
            _cid("a"): MALFORMED,
            _cid("b"): REFUSAL,
            _cid("c"): TRANSPORT,
            _cid("d"): TRUNCATED,
            _cid("e"): API_REFUSAL,
            _cid("f"): _kept(80),
            _cid("g", comp=0): MALFORMED,
            _cid("g", comp=1): MALFORMED,
            _cid("zzz-foreign"): MALFORMED,  # not in item_ids -> excluded
        },
    )
    counts = sel.malformed_drop_counts(f, {"a", "b", "c", "d", "e", "f", "g"})
    assert counts == {"a": 1, "g": 2}


def test_judge_with_redraw_first_parsed_wins_and_bounded(tmp_path, monkeypatch):
    """(2) First-parsed-wins merge: round-1/round-2 recoveries carry their
    scores; a thrice-failed item stays dropped and counted; refusal/transport
    items are never re-issued; rule-24(ii) per-round cache dirs; sync routing;
    exactly 1 + 2 judge calls."""
    fake = _ScriptedJudge(
        [
            # round 0 (n_draws=1): a kept; b/c/d malformed; e refusal; f transport
            {
                "a": [_kept(70)],
                "b": [MALFORMED],
                "c": [MALFORMED],
                "d": [MALFORMED],
                "e": [REFUSAL],
                "f": [TRANSPORT],
            },
            # round 1 (b, c, d re-issued): c recovers; b, d malformed again
            {"b": [MALFORMED], "c": [_kept(12)], "d": [MALFORMED]},
            # round 2 (b, d): b recovers on the LAST attempt; d fails all 3
            {"b": [_kept(33)], "d": [MALFORMED]},
        ]
    )
    monkeypatch.setattr("explore_persona_space.eval.graded_judge.judge_graded", fake)
    out = sel.judge_with_redraw(
        _items(["a", "b", "c", "d", "e", "f"]),
        "rubric {question} {answer}",
        n_draws=1,
        cache_dir=tmp_path / "cache" / "x",
        save_raw=tmp_path / "raw" / "filter_x.json",
        max_tokens=1024,
    )
    # Merge: first parsed judgment per item; thrice-failed d stays None.
    assert out.scores == {"a": 70.0, "b": 33.0, "c": 12.0, "d": None, "e": None, "f": None}
    assert out.n_items_redrawn == 3 and out.n_recovered == 2
    assert out.residual_malformed_draws == 1  # d's round-0 malformed draw
    assert out.recovered_malformed_draws == 2  # b + c round-0 malformed draws
    # Selection: refusal (e) + transport (f) never entered a re-draw round.
    assert fake.calls[1]["items"] == ["b", "c", "d"]
    assert fake.calls[2]["items"] == ["b", "d"]
    assert len(fake.calls) == 3  # bounded: 1 initial + REDRAW_MAX_ROUNDS
    # Rule 24(ii): DISTINCT per-round cache dirs + per-round raw files.
    assert fake.calls[0]["cache_dir"].name == "x"
    assert fake.calls[1]["cache_dir"].name == "x__redraw_k1"
    assert fake.calls[2]["cache_dir"].name == "x__redraw_k2"
    assert fake.calls[1]["save_raw"].name == "filter_x_redraw1.json"
    assert fake.calls[2]["save_raw"].name == "filter_x_redraw2.json"
    assert fake.calls[1]["save_raw"].exists() and fake.calls[2]["save_raw"].exists()
    # Identical instrument + sync routing on re-draw rounds.
    assert fake.calls[1]["max_tokens"] == 1024 and fake.calls[2]["max_tokens"] == 1024
    assert fake.calls[1]["threshold_base"] == sel.REDRAW_SYNC_THRESHOLD
    assert fake.calls[1]["n_draws"] == 1
    # Round-0 accounting is untouched (drop-never-coerce record intact).
    assert out.result.n_dropped_draws == 4  # 3 malformed + 1 refusal
    assert out.result.n_refusal_draws == 1
    assert out.result.n_transport_lost_draws == 1
    acct = sel.redraw_accounting(out)
    assert acct["n_redraw_rounds_run"] == 2
    assert acct["n_unrecovered"] == 1
    assert acct["parse_fail_rate_raw"] == pytest.approx(3 / 5)  # 3 malformed / 5 answered
    assert acct["parse_fail_rate_post_recovery"] == pytest.approx(1 / 5)


def test_redraw_skips_items_with_sibling_parsed_judgment(tmp_path, monkeypatch):
    """Single-judgment semantics at pilot n_draws=2: an item with a kept
    sibling draw already HAS a parsed judgment — never re-issued; its
    round-0 reduce (mean over kept draws) is kept."""
    fake = _ScriptedJudge(
        [
            {"a": [_kept(80), MALFORMED], "b": [MALFORMED, MALFORMED]},
            {"b": [_kept(5)]},
        ]
    )
    monkeypatch.setattr("explore_persona_space.eval.graded_judge.judge_graded", fake)
    out = sel.judge_with_redraw(
        _items(["a", "b"]),
        "rubric {question} {answer}",
        n_draws=2,
        cache_dir=tmp_path / "cache" / "p",
        save_raw=tmp_path / "raw" / "pilot_p.json",
        max_tokens=1024,
    )
    assert fake.calls[1]["items"] == ["b"]
    assert len(fake.calls) == 2  # b recovered on round 1 -> no round 2
    assert out.scores == {"a": 80.0, "b": 5.0}
    # a's malformed sibling draw is coverage-recovered by its kept judgment
    # (1 draw), and b's two round-0 malformed draws by its re-draw (2 more).
    assert out.residual_malformed_draws == 0
    assert out.recovered_malformed_draws == 3


def _pilot_arm_outcome(tmp_path, monkeypatch, name, n_items, n_malformed, recover):
    """One 100-ish-item arm through the REAL wrapper: `n_malformed` round-0
    parse failures, `recover` = per-round recovery counts (len <= 2)."""
    ids = [f"{name}-i{k:03d}" for k in range(n_items)]
    bad = ids[:n_malformed]
    script = [{i: [MALFORMED] if i in bad else [_kept(50)] for i in ids}]
    remaining = list(bad)
    for n_rec in recover:
        entry = {}
        for j, iid in enumerate(remaining):
            entry[iid] = [_kept(10)] if j < n_rec else [MALFORMED]
        script.append(entry)
        remaining = remaining[n_rec:]
    fake = _ScriptedJudge(script)
    monkeypatch.setattr("explore_persona_space.eval.graded_judge.judge_graded", fake)
    save_raw = tmp_path / "raw" / f"pilot_{name}.json"
    out = sel.judge_with_redraw(
        _items(ids),
        "rubric {question} {answer}",
        n_draws=1,
        cache_dir=tmp_path / "cache" / name,
        save_raw=save_raw,
        max_tokens=1024,
    )
    return sel.post_recovery_arm_stats(out, save_raw, set(ids))


def test_post_recovery_gate_arithmetic(tmp_path, monkeypatch):
    """(3) Gate arithmetic via the REAL `_gate_verdict`: 18% raw parse-fail
    recovered to 1% PASSes the unchanged 2% bar; an 8% residual still FAILs."""
    from explore_persona_space.eval.judge_pilot import _gate_verdict

    # 18/100 malformed -> 16 then 1 recovered -> 1 residual -> 1% < 2%.
    ok = _pilot_arm_outcome(tmp_path, monkeypatch, "ok", 100, 18, [16, 1])
    assert ok.parse_fail_rate == pytest.approx(0.01)
    assert ok.n_content_dropped == 1 and ok.n_scored == 99
    failures, _w = _gate_verdict(
        {"ok": ok},
        max_tokens=1024,
        parse_fail_threshold=sel.PILOT_PARSE_FAIL_THRESHOLD,
        min_effective_draws_per_arm=sel.PILOT_MIN_EFFECTIVE_DRAWS,
    )
    assert failures == []
    # 18/100 malformed -> 5 + 5 recovered -> 8 residual -> 8% >= 2% -> FAIL.
    bad = _pilot_arm_outcome(tmp_path, monkeypatch, "bad", 100, 18, [5, 5])
    assert bad.parse_fail_rate == pytest.approx(0.08)
    failures, _w = _gate_verdict(
        {"bad": bad},
        max_tokens=1024,
        parse_fail_threshold=sel.PILOT_PARSE_FAIL_THRESHOLD,
        min_effective_draws_per_arm=sel.PILOT_MIN_EFFECTIVE_DRAWS,
    )
    assert any("parse-fail" in f and "bad" in f for f in failures)


def test_redraw_truncation_still_fails_gate(tmp_path, monkeypatch):
    """Truncation is NEVER waivable: a truncation-class draw in a RE-DRAW
    round still fires the rule-26(a) clause (strictly stricter)."""
    from explore_persona_space.eval.judge_pilot import _gate_verdict

    ids = [f"t-i{k:02d}" for k in range(60)]
    script = [
        {i: [MALFORMED] if i == ids[0] else [_kept(50)] for i in ids},
        {ids[0]: [TRUNCATED]},  # re-draw truncates -> rule-23 class, no round 2
    ]
    fake = _ScriptedJudge(script)
    monkeypatch.setattr("explore_persona_space.eval.graded_judge.judge_graded", fake)
    save_raw = tmp_path / "raw" / "pilot_t.json"
    out = sel.judge_with_redraw(
        _items(ids),
        "rubric {question} {answer}",
        n_draws=1,
        cache_dir=tmp_path / "cache" / "t",
        save_raw=save_raw,
        max_tokens=1024,
    )
    assert len(fake.calls) == 2  # truncation keeps its class -> no round 2
    stats = sel.post_recovery_arm_stats(out, save_raw, set(ids))
    assert stats.n_truncation == 1
    assert stats.stop_reason_tally.get("max_tokens") == 1
    failures, _w = _gate_verdict(
        {"t": stats},
        max_tokens=1024,
        parse_fail_threshold=sel.PILOT_PARSE_FAIL_THRESHOLD,
        min_effective_draws_per_arm=sel.PILOT_MIN_EFFECTIVE_DRAWS,
    )
    assert any("truncation" in f for f in failures)


# ── r6: suite-4a build parts-resume regime keys (#722 r3 class) ──────────────────


class TestSuiteBuildResumeRegime:
    """Parts-resume must key on EVERY output-affecting build arg, not mix sha alone.

    r5 code-review Major (`suite4a-build-resume-regime-key`): a resume keyed on
    `mix_sha256` + part existence silently reuses parts built under different
    --max-prompt-tokens / --per-dataset-cap / --seed / --model, and manifest.json
    then stamps the NEW args over rows built under the OLD ones.
    """

    REGIME: ClassVar[dict] = {
        "model": "m",
        "seed": 42,
        "max_prompt_tokens": 2048,
        "per_dataset_cap": None,
    }

    def _prior(self, **over):
        rec = {"mix_sha256": "abc", **self.REGIME, "n_kept": 3}
        rec.update(over)
        return rec

    def test_full_match_resumes(self):
        assert suite.resume_part_ok(self._prior(), "abc", dict(self.REGIME), True)

    def test_any_changed_regime_key_repacks(self):
        for key, changed in [
            ("max_prompt_tokens", 512),
            ("per_dataset_cap", 100),
            ("seed", 7),
            ("model", "other-model"),
        ]:
            prior = self._prior(**{key: changed})
            assert not suite.resume_part_ok(prior, "abc", dict(self.REGIME), True), key

    def test_legacy_record_missing_regime_keys_repacks(self):
        legacy = {"mix_sha256": "abc", "n_kept": 3}  # pre-r6 record: sha-only key
        assert not suite.resume_part_ok(legacy, "abc", dict(self.REGIME), True)

    def test_changed_sha_missing_part_or_no_prior_repacks(self):
        assert not suite.resume_part_ok(self._prior(), "other", dict(self.REGIME), True)
        assert not suite.resume_part_ok(self._prior(), "abc", dict(self.REGIME), False)
        assert not suite.resume_part_ok(None, "abc", dict(self.REGIME), True)

    def test_build_regime_covers_the_four_output_affecting_args(self):
        ns = argparse.Namespace(model="m", seed=42, max_prompt_tokens=2048, per_dataset_cap=None)
        assert suite.build_regime(ns) == self.REGIME


class TestSeedIsolationGuard:
    """fu-r2 review blocker: a non-42 seed with parent-default state/results
    paths on a state-touching phase must REFUSE (seed-42 clobber guard) —
    default judge dirs would overwrite the committed seed-42
    selection_finetune/<cid>/trait_scores.json and satisfy _check_pilot with
    the PARENT's passing pilot reports (rule-26 bypass)."""

    def _args(self, argv: list[str]) -> argparse.Namespace:
        return sweep.build_argparser().parse_args(argv)

    def test_seed137_judge_with_default_dirs_raises_naming_flags(self):
        args = self._args(["--phase", "judge", "--seed", "137"])
        with pytest.raises(RuntimeError) as e:
            sweep.assert_seed_isolation(args)
        msg = str(e.value)
        for flag in (
            "--out-root",
            "--eval-questions-dir",
            "--trait-scores-dir",
            "--judge-root",
            "--pilot-report-dir",
        ):
            assert flag in msg, flag

    def test_seed137_train_with_default_out_root_raises(self):
        with pytest.raises(RuntimeError, match=r"--out-root"):
            sweep.assert_seed_isolation(self._args(["--phase", "train", "--seed", "137"]))

    def test_seed137_train_isolated_dirs_but_empty_suffix_raises(self):
        # Review r3 BLOCKER 1: isolated LOCAL dirs with no --hf-prefix-suffix
        # would overwrite the parent's HF adapters prefix + wandb run names.
        args = self._args(["--phase", "train", "--seed", "137", "--out-root", "data/x_seed137"])
        with pytest.raises(RuntimeError, match=r"--hf-prefix-suffix"):
            sweep.assert_seed_isolation(args)

    def test_seed137_upload_isolated_dirs_but_empty_suffix_raises(self):
        args = self._args(
            [
                "--phase",
                "upload",
                "--seed",
                "137",
                "--out-root",
                "data/x_seed137",
                "--eval-questions-dir",
                "data/q_seed137",
            ]
        )
        with pytest.raises(RuntimeError, match=r"--hf-prefix-suffix"):
            sweep.assert_seed_isolation(args)

    def test_seed137_relative_spelling_of_parent_default_raises(self, monkeypatch):
        # Review r3 BLOCKER 2: a RELATIVE spelling of the parent default must
        # not evade the guard and resume from parent seed-42 state.
        monkeypatch.chdir(sweep.PROJECT_ROOT)
        rel = sweep.OUT_ROOT_DEFAULT.relative_to(sweep.PROJECT_ROOT)
        args = self._args(
            [
                "--phase",
                "train",
                "--seed",
                "137",
                "--out-root",
                str(rel),
                "--hf-prefix-suffix",
                "_seed137",
            ]
        )
        with pytest.raises(RuntimeError, match=r"--out-root"):
            sweep.assert_seed_isolation(args)

    def test_seed137_train_fully_isolated_passes(self):
        sweep.assert_seed_isolation(
            self._args(
                [
                    "--phase",
                    "train",
                    "--seed",
                    "137",
                    "--out-root",
                    "data/issue_2224/screening_ft_seed137",
                    "--hf-prefix-suffix",
                    "_seed137",
                ]
            )
        )

    def test_seed_replication_never_writes_parent_phase_sentinel(self):
        # Review r3 item (a): an UNFILTERED seed!=42 run must skip the parent
        # .done_4b4/.done_4b5 sentinel write (returns False = no write).
        assert sweep.finalize_phase_sentinel("x.done", "train", {}, None, seed=137) is False

    def test_seed42_defaults_pass_and_isolated_seed137_passes(self):
        # Inert for the parent seed-42 pipeline (all defaults).
        sweep.assert_seed_isolation(self._args(["--phase", "judge", "--seed", "42"]))
        # Fully-isolated seed-137 judge invocation (the fu-r2 judge runner shape).
        sweep.assert_seed_isolation(
            self._args(
                [
                    "--phase",
                    "judge",
                    "--seed",
                    "137",
                    "--out-root",
                    "data/issue_2224/screening_ft_seed137",
                    "--eval-questions-dir",
                    "data/issue_2224/eval_questions_seed137",
                    "--judge-root",
                    "data/issue_2224/judge_postft_seed137",
                    "--trait-scores-dir",
                    "eval_results/issue_2224/followup_r2/selection_finetune_seed137",
                    "--pilot-report-dir",
                    "eval_results/issue_2224/followup_r2/judge_pilots",
                ]
            )
        )

    def test_guard_covers_every_state_touching_phase(self):
        # Membership pin: a new state-touching phase must join the mapping.
        assert set(sweep._SEED_ISOLATION_REQUIRED) == {
            "train",
            "train-cell",
            "eval",
            "eval-shard",
            "upload",
            "eval-questions",
            "judge",
            "judge-pilot",
        }
