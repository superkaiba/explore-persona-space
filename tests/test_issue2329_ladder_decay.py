"""Issue #2329 q35_ladder_decay round pins.

Covers the round's NEW invariants (plan v8):
- fragment instrument rule-27 round-trip + placeholder substitution (Leg B),
- G4b pilot sizing satisfiability (the VERDICT-DOOMED gotcha),
- K=4 segmentation + the 48-token dispatch floor,
- registered-row token-identity subtraction (divergence 7: tokgate-dropped
  pairs, untestable directions, screen-dropped + pe-nonviable null_xtype),
- model_revision threading (M1: regime_fingerprint key + legacy identity;
  load_model_and_tokenizer additive revision param on BOTH from_pretrained),
- G1 donor-identity derivation-equality leg + the --tiny cos-leg skip,
- a tiny-real reduce+figures e2e (dual estimands, denominator bar, verdict
  lattice, retention counters, shared-index bootstrap, sanity join).

All fixtures are tmp-path synthetic (no network, no committed eval_results
reads); the tokenizer boundary is faked with a signature-conformant
word-tokenizer (external boundary only — every driver body runs real).
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2094_judge as J94  # noqa: E402
import issue2162_ladder_judge as PLJ  # noqa: E402
import issue2329_decay as DEC  # noqa: E402
import issue2329_ladder as LAD  # noqa: E402
import issue2329_ladder_analysis as LA  # noqa: E402
import issue2329_ladder_judge as LJ  # noqa: E402
import issue2329_run as RUN  # noqa: E402

from explore_persona_space.eval.graded_judge import _score_from_parsed  # noqa: E402
from explore_persona_space.eval.utils import parse_judge_json  # noqa: E402
from explore_persona_space.experiments.issue2162 import ladder_bank as LB  # noqa: E402

# ── fragment instrument (rule 27) ─────────────────────────────────────


def test_fragment_prompt_placeholders_and_substitution():
    for v in LB.PERSONA_VALUE_IDS:
        prompt = DEC.fragment_eval_prompt(LB.VALUES_BY_ID[v].descriptor)
        assert "{question}" in prompt and "{answer}" in prompt, v
        # harness-identical substitution (graded_judge.format_user_msg shape)
        filled = prompt.replace("{question}", "Q?").replace("{answer}", "frag text")
        assert "{question}" not in filled and "{answer}" not in filled
        assert "frag text" in filled and "Q?" in filled
        assert DEC.fragment_rubric_id(v) == f"dfrag-{v}"


@pytest.mark.parametrize(
    "reply",
    [
        'Pirate idioms run throughout. {"reasoning": "strong pirate voice", "score": 87}',
        '```json\n{"reasoning": "consistent persona markers", "score": 87}\n```',
    ],
)
def test_fragment_rubric_roundtrip_reason_then_score(reply):
    parsed = parse_judge_json(reply)
    score = _score_from_parsed(parsed)
    assert score == 87.0


def test_fragment_rubric_roundtrip_refusal_drops():
    parsed = parse_judge_json("REFUSAL")
    assert _score_from_parsed(parsed) is None  # rule-9 drop, never coerced


# ── G4b pilot sizing (VERDICT-DOOMED gotcha) ──────────────────────────


def test_pilot_sizing_satisfiable():
    # floor(1 / parse_fail_threshold) + 1 at the default 0.02 == 51
    assert DEC.PILOT_REQUIRED_PER_ARM == int(1 / 0.02) + 1 == 51
    per_arm_items = DEC.PILOT_TARGET_TOTAL // (6 * J94.JUDGE_N_DRAWS)
    assert per_arm_items * J94.JUDGE_N_DRAWS >= DEC.PILOT_REQUIRED_PER_ARM, (
        per_arm_items,
        DEC.PILOT_REQUIRED_PER_ARM,
    )


# ── segmentation ──────────────────────────────────────────────────────


class FakeTok:
    """Signature-conformant word tokenizer (external boundary fake only)."""

    def __init__(self):
        self._w2i: dict[str, int] = {}
        self._i2w: dict[int, str] = {}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        ids = []
        for w in text.split():
            if w not in self._w2i:
                i = len(self._w2i)
                self._w2i[w] = i
                self._i2w[i] = w
            ids.append(self._w2i[w])
        return ids

    def decode(self, ids):
        return " ".join(self._i2w[int(i)] for i in ids)


def test_segment_length_floor_and_quartiles():
    tok = FakeTok()
    assert DEC._segment(tok, " ".join(f"w{i}" for i in range(47))) is None
    text = " ".join(f"w{i}" for i in range(50))
    segs, n_tok, seg_lens = DEC._segment(tok, text)
    assert n_tok == 50 and len(segs) == DEC.DECAY_K == 4
    assert sum(seg_lens) == 50 and max(seg_lens) - min(seg_lens) <= 1
    # contiguity: concatenated segment decodes reproduce the word stream
    assert " ".join(segs).split() == text.split()


# ── registered-row token-identity subtraction (divergence 7) ──────────


def _tokrep(pairs, directions):
    return {"pairs": pairs, "directions": directions, "bank_sha": "x" * 16}


def test_registered_row_keys_token_identity_subtraction():
    v = "r1_pirate"
    gate = {
        "rungs": {
            u: {"survived": u == v, "surviving_carriers": ["d1", "d2"] if u == v else []}
            for u in LB.PERSONA_VALUE_IDS
        }
    }
    tokrep = _tokrep(
        pairs=[
            {"pair_id": f"install_{v}::d1", "intact": True},
            {"pair_id": f"install_{v}::d2", "intact": False},  # tokgate-dropped
            {"pair_id": f"erase_{v}::d1", "intact": True},
            {"pair_id": f"erase_{v}::d2", "intact": True},
        ],
        directions={
            f"install_{v}": {"testable": True},
            f"erase_{v}": {"testable": False},  # untestable -> generates NOTHING
        },
    )
    screen = {f"install_{v}::d1": {"status": "kept", "donor": "x", "pe_viable": False}}
    keys = LA.registered_row_keys(gate, screen, tokrep)
    dirs = {k[0] for k in keys}
    assert dirs == {f"install_{v}"}  # erase direction untestable
    carriers = {k[3] for k in keys}
    assert carriers == {"d1"}  # d2 pair tokgate-dropped from ALL arms
    # pe-nonviable pair: (pe x null_xtype) excluded, everything else present
    assert (f"install_{v}", "ce", "null_xtype", "d1") in keys
    assert (f"install_{v}", "pe", "null_xtype", "d1") not in keys
    assert (f"install_{v}", "pe", "steered", "d1") in keys
    full = {(f"install_{v}", s, a, "d1") for s in LA.SLOTS for a in LA.ARMS}
    assert keys == full - {(f"install_{v}", "pe", "null_xtype", "d1")}


def test_registered_row_keys_screen_dropped_null_xtype():
    v = "r1_pirate"
    gate = {
        "rungs": {
            u: {"survived": u == v, "surviving_carriers": ["d1"] if u == v else []}
            for u in LB.PERSONA_VALUE_IDS
        }
    }
    tokrep = _tokrep(
        pairs=[
            {"pair_id": f"install_{v}::d1", "intact": True},
            {"pair_id": f"erase_{v}::d1", "intact": True},
        ],
        directions={f"install_{v}": {"testable": True}, f"erase_{v}": {"testable": True}},
    )
    screen = {f"install_{v}::d1": {"status": "dropped", "pe_viable": True}}
    keys = LA.registered_row_keys(gate, screen, tokrep)
    for slot in LA.SLOTS:
        assert (f"install_{v}", slot, "null_xtype", "d1") not in keys
        assert (f"install_{v}", slot, "steered", "d1") in keys
        # erase pair unscreened -> null_xtype kept there
        assert (f"erase_{v}", slot, "null_xtype", "d1") in keys


# ── model_revision threading (M1) ─────────────────────────────────────


def _fp_cfg(**over):
    base = dict(
        model_id="Qwen/Qwen3.5-9B",
        tiny=False,
        n_layers=32,
        hidden=4096,
        max_new_tokens=4096,
        grid_draws=5,
        seed_base=42,
        smoke=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_regime_fingerprint_model_revision_key():
    pinned = RUN.regime_fingerprint(_fp_cfg(model_revision="a" * 40), "bank")
    other = RUN.regime_fingerprint(_fp_cfg(model_revision="b" * 40), "bank")
    assert pinned != other  # the pin is an output-affecting resume key
    # legacy identity: an attribute-less cfg == an explicit None pin
    legacy = RUN.regime_fingerprint(_fp_cfg(), "bank")
    none_pin = RUN.regime_fingerprint(_fp_cfg(model_revision=None), "bank")
    assert legacy == none_pin


def test_load_model_and_tokenizer_revision_param_additive():
    sig = inspect.signature(RUN.load_model_and_tokenizer)
    assert "revision" in sig.parameters
    assert sig.parameters["revision"].default is None  # legacy callers unchanged
    # BOTH from_pretrained call sites thread revision=
    tree = ast.parse(inspect.getsource(RUN.load_model_and_tokenizer))
    fp_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
    ]
    assert len(fp_calls) >= 2, "expected model + tokenizer from_pretrained call sites"
    for call in fp_calls:
        assert any(kw.arg == "revision" for kw in call.keywords), ast.dump(call)


def test_decay_q35_tokenizer_uses_model_revision_pin():
    src = inspect.getsource(DEC._load_tokenizer)
    assert "MODEL_REVISION_PIN" in src and "revision=MODEL_REVISION_PIN" in src


# ── G1 donor identity (derivation leg + tiny cos-leg skip) ────────────


def _donor_manifest(donor_bs):
    return {
        "crosstype_donor_plan": {f"cell{i}": {"primary": {"b": b}} for i, b in enumerate(donor_bs)}
    }


def _g1_cfg(tiny):
    return SimpleNamespace(
        tiny=tiny,
        model_revision="c" * 40,
        model_id="Qwen/Qwen3.5-9B",
        smoke=False,
        n_layers=4 if tiny else 32,
        device="cpu",
        layers=list(range(4)),
    )


def test_donor_identity_derivation_mismatch_fails():
    report = LAD.run_donor_identity_assert(
        _g1_cfg(tiny=True), None, None, _donor_manifest(["verbosity::v1::d1"]), {}
    )
    assert report["passed"] is False
    assert report["derivation"]["equal"] is False


def test_donor_identity_tiny_skips_cos_leg():
    # extra donors BEYOND the first three must not break the derivation leg:
    # the plan derivation is "first 3 DISTINCT primary donors in build order",
    # never the full distinct donor set (30 at the pin — the false-HALT bug).
    donor_bs = [
        *LAD.DONOR_IDENTITY_CONTEXT_IDS,
        "instr_format::v1::n3",
        "verbosity::v2::d2",
        LAD.DONOR_IDENTITY_CONTEXT_IDS[0],  # repeat: distinctness, not count
    ]
    report = LAD.run_donor_identity_assert(
        _g1_cfg(tiny=True),
        None,
        None,
        _donor_manifest(donor_bs),
        {},
    )
    assert report["passed"] is True
    assert report["cos_leg_skipped_tiny"] is True
    assert report["derivation"]["equal"] is True
    assert report["bar_cos"] == LAD.DONOR_IDENTITY_COS_MIN == 0.99
    assert sorted(LAD.DONOR_IDENTITY_CONTEXT_IDS) == sorted(
        ["verbosity::v1::d1", "instr_format::v2::d1", "instr_format::v1::d2"]
    )


# ── tiny-real reduce + figures e2e (Leg B) ────────────────────────────

V = "r1_pirate"
CARRIERS = ("d1", "d2")
N_DRAWS = 6
LONG_TEXT = " ".join(f"tok{i}" for i in range(52))
SHORT_TEXT = " ".join(f"tok{i}" for i in range(10))
SEG_BASE = {1: 90, 2: 70, 3: 50, 4: 30}


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _fixture_side(root: Path, scores_dir: Path, gates_dir: Path, mod) -> None:
    """One model side's staged inputs + committed judge outputs."""
    grid_dir = root / mod.LADDER_RAW / "grid"
    anchors_dir = root / mod.LADDER_RAW / "anchors"
    bank_path = root / mod._STAGE_BANK_FILE
    contexts = {}
    for c in CARRIERS:
        for val in (V, "plain"):
            contexts[LB.context_id(val, c)] = {"user": f"question for {c}"}
    manifest = {
        "values": [{"value_id": V, "descriptor": "a pirate persona", "rung_rank": 1}],
        "carriers": {c: f"carrier text {c}" for c in CARRIERS},
        "contexts": contexts,
        "pairs": [
            {"pair_id": f"install_{V}::{c}", "direction": f"install_{V}", "carrier": c}
            for c in CARRIERS
        ],
    }
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    bank_path.write_text(json.dumps(manifest), encoding="utf-8")

    grid_rows, coh_grid, hol_grid = [], [], []
    for c in CARRIERS:
        pid = f"install_{V}::{c}"
        for d in range(N_DRAWS):
            grid_rows.append(
                {
                    "cell": f"install_{V}",
                    "slot": "ce",
                    "arm": "steered",
                    "pair_id": pid,
                    "draw": d,
                    "text": LONG_TEXT,
                    "context_id": LB.context_id("plain", c),
                }
            )
            coh = 40 if (c == "d1" and d == 5) else 90  # one coh-fail completion
            coh_grid.append(
                {"pair_id": pid, "slot": "ce", "arm": "steered", "draw": d, "score": coh}
            )
            hol_grid.append(
                {"pair_id": pid, "slot": "ce", "arm": "steered", "draw": d, "score": 85 + d}
            )
    # one under-floor steered completion (length-drop counter) + ignored arms
    grid_rows.append(
        {
            "cell": f"install_{V}",
            "slot": "ce",
            "arm": "steered",
            "pair_id": f"install_{V}::d1",
            "draw": N_DRAWS,
            "text": SHORT_TEXT,
            "context_id": LB.context_id("plain", "d1"),
        }
    )
    # must-fix 4: the coherence-coverage gate is PRE-length-gate — the
    # under-floor completion still owes a committed coherence row.
    coh_grid.append(
        {
            "pair_id": f"install_{V}::d1",
            "slot": "ce",
            "arm": "steered",
            "draw": N_DRAWS,
            "score": 90,
        }
    )
    grid_rows.append(
        {
            "cell": f"install_{V}",
            "slot": "ce",
            "arm": "null_sameval",
            "pair_id": f"install_{V}::d1",
            "draw": 0,
            "text": LONG_TEXT,
            "context_id": LB.context_id("plain", "d1"),
        }
    )
    grid_rows.append(
        {
            "cell": f"erase_{V}",
            "slot": "ce",
            "arm": "steered",
            "pair_id": f"erase_{V}::d1",
            "draw": 0,
            "text": LONG_TEXT,
            "context_id": LB.context_id(V, "d1"),
        }
    )
    # Assumption-9 reconciliation input: stored token count == FakeTok's word
    # count exactly (the q25 side re-tokenizes EVERY row against this field).
    for r in grid_rows:
        r["n_completion_tokens"] = len(r["text"].split())
    _write_jsonl(grid_dir / "shard_000.jsonl", grid_rows)

    anchor_rows, coh_anch, hol_anch = [], [], []
    for c in CARRIERS:
        for val in (V, "plain"):
            cid = LB.context_id(val, c)
            for d in range(N_DRAWS):
                anchor_rows.append(
                    {
                        "context_id": cid,
                        "cell": f"anchor_{val}",
                        "value_id": val,
                        "carrier": c,
                        "draw": d,
                        "text": LONG_TEXT,
                    }
                )
                coh_anch.append({"context_id": cid, "draw": d, "score": 90})
                hol_anch.append({"context_id": cid, "draw": d, "score": 80 + d})
    _write_jsonl(anchors_dir / "anchors_gate_w0.jsonl", anchor_rows)

    _write_jsonl(scores_dir / "coherence.grid.scores.jsonl", coh_grid)
    _write_jsonl(scores_dir / "coherence.anchors.scores.jsonl", coh_anch)
    _write_jsonl(scores_dir / f"hol-{V}.grid.scores.jsonl", hol_grid)
    _write_jsonl(scores_dir / f"hol-{V}.anchors.scores.jsonl", hol_anch)

    gate = {
        "rungs": {
            u: {"survived": u == V, "surviving_carriers": list(CARRIERS) if u == V else []}
            for u in LB.PERSONA_VALUE_IDS
        }
    }
    gates_dir.mkdir(parents=True, exist_ok=True)
    (gates_dir / "ladder_separation_gate.json").write_text(json.dumps(gate), encoding="utf-8")


def _frag_score(src: dict) -> int:
    d = src["draw"]
    if src["arm"] == "steered":
        if src["pair_id"].endswith("::d1") and d == 5:
            return 60 + d  # the coh-fail completion: flat profile
        return SEG_BASE[src["segment"]] + d
    if src["arm"] == "ceiling":
        # d1: SLOPED ceiling (g6-1) so the raw ceiling drop is nonzero and
        # dD genuinely subtracts it; d2 stays flat (denominator-bar branch).
        if src["carrier"] == "d1":
            return {1: 80, 2: 75, 3: 70, 4: 65}[src["segment"]] + d
        return 12 + d
    return 10 + d  # floor


@pytest.fixture
def decay_cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(DEC, "_load_tokenizer", lambda key: FakeTok())
    q25_scores, q25_gates = tmp_path / "q25_scores", tmp_path / "q25_gates"
    q35_scores, q35_gates = tmp_path / "q35_scores", tmp_path / "q35_gates"
    _fixture_side(tmp_path / "q25_root", q25_scores, q25_gates, PLJ)
    _fixture_side(tmp_path / "q35_root", q35_scores, q35_gates, LJ)
    for name in ("q25_stats.json", "q35_stats.json"):
        (tmp_path / name).write_text(json.dumps({"lattice": {}}), encoding="utf-8")
    return DEC.DecayConfig(
        q25_in_root=tmp_path / "q25_root",
        q35_in_root=tmp_path / "q35_root",
        q25_scores_dir=q25_scores,
        q25_gates_dir=q25_gates,
        q25_stats_json=tmp_path / "q25_stats.json",
        q35_scores_dir=q35_scores,
        q35_gates_dir=q35_gates,
        q35_stats_json=tmp_path / "q35_stats.json",
        out_dir=tmp_path / "decay_out",
        cache_dir=tmp_path / "cache",
        figures_dir=tmp_path / "figs",
        n_boot=200,
    )


def test_reduce_and_figures_tiny_real_e2e(decay_cfg):
    cfg = decay_cfg
    # synthesize the wave output at run_wave's exact scores-row schema
    sides = DEC._build_sides(cfg)
    assert sides["q25"].scope_values == [V]
    assert sides["q25"].pe_directions == set()
    rows = [
        {
            "item_id": u.item_id,
            "wave": f"{u.rubric_id}.decay",
            "rubric_id": u.rubric_id,
            "score": _frag_score(u.source),
            "n_kept_draws": 1,
            "transport_lost_residual": 0,
            **u.source,
        }
        for key in DEC.MODEL_KEYS
        for u in sides[key].units
    ]
    _write_jsonl(cfg.j94().scores_dir / f"dfrag-{V}.decay.scores.jsonl", rows)

    assert DEC.phase_reduce(cfg) == DEC.RC_OK
    stats = json.loads((cfg.out_dir / "decay_stats.json").read_text(encoding="utf-8"))

    # retention: one under-floor steered completion per side
    for key in DEC.MODEL_KEYS:
        ret = stats["retention_length"][key]["steered"]
        assert ret["n_completions_seen"] == 13
        assert ret["n_len_dropped"] == 1 and ret["n_len_eligible"] == 12
    # row files persisted per arm x model
    steered_rows = [
        json.loads(line)
        for line in (cfg.out_dir / "segment_scores_steered_q25.jsonl").read_text().splitlines()
    ]
    assert len(steered_rows) == 12 * DEC.DECAY_K

    # dual estimands numerically split: coh drops the flat completion.
    # dD subtracts the SLOPED d1 ceiling drop (g6-1): steered d1 coh drop
    # 0.60 - ceiling 0.15 = 0.45; d2 0.60 - 0 = 0.60 -> mean 0.525.
    # all: d1 (0.50 - 0.15) = 0.35 (flat draw-5 row pulls Q1 down); d2 0.60.
    for key in DEC.MODEL_KEYS:
        coh_dd = stats["families"][key]["coh|primary|dD"]
        all_dd = stats["families"][key]["all|primary|dD"]
        assert coh_dd["point"] == pytest.approx(0.525, abs=1e-9)
        assert all_dd["point"] == pytest.approx(0.475, abs=1e-9)
        assert coh_dd["ci_lo"] > 0 and all_dd["ci_lo"] > 0
        assert coh_dd["n_carriers"] == 2
        # ceiling raw drop: d1 sloped (0.825 - 0.675 = 0.15), d2 flat (0)
        assert stats["families"][key]["coh|primary|Draw_ceiling"]["point"] == pytest.approx(
            0.075, abs=1e-9
        )
        # denominator bar: d2's |ceiling-floor| = 0.02 < 0.125 -> dD_F only on
        # d1, whose per-segment denominators are now 0.70/0.65/0.60/0.55:
        # dD_F = F[1] - F[4] = 79.5/70 - 19.5/55.
        ddf = stats["families"][key]["coh|primary|dD_F"]
        assert ddf["n_carriers"] == 1
        assert ddf["point"] == pytest.approx(79.5 / 70 - 19.5 / 55, abs=1e-9)
        rec = stats["per_direction"][f"{key}|install_{V}|ce|coh"]["per_carrier"]
        assert rec["d2"]["delta_d_f"] is None
        assert "0.125" in rec["d2"]["delta_d_f_unavailable_reason"]
        assert rec["d1"]["delta_d_f"] is not None
        # verdict lattice: both estimands' dD CIs > 0 -> patch-decays-faster
        assert stats["lattice"][key]["verdict"] == "patch-decays-faster"
        for e in DEC.ESTIMANDS:
            assert stats["lattice"][key]["per_estimand"][e]["label"] == "patch-decays-faster"
        # N2.2 Q1 gap: steered seg1 above ceiling seg1 on both carriers
        # (d1: 92 > 82.5; d2: 92.5 > 14.5) — only the carrier count is pinned
        assert stats["n2_2_q1_gap"][key]["coh"]["n_carriers"] == 2
        # sanity join: frag means rise with draw exactly as hol scores do
        assert stats["fragment_vs_whole_sanity"][key]["steered"]["n"] == 12
        assert stats["fragment_vs_whole_sanity"][key]["steered"]["rho"] == pytest.approx(
            1.0, abs=1e-6
        )
    assert stats["n2_3_intersection"]["rungs"] == [V]

    # figures render on the tiny stats (Agg backend, tmp figures dir)
    assert DEC.phase_figures(cfg) == DEC.RC_OK
    for stem in (
        "q35_ladder_decay_decay_raw",
        "q35_ladder_decay_decay_norm",
        "q35_ladder_decay_contrast",
        "q35_ladder_decay_diagnostics",  # manifest stem (review r1 must-fix 6d)
    ):
        assert (cfg.figures_dir / f"{stem}.png").exists(), stem


def test_reduce_shared_index_bootstrap_deterministic(decay_cfg):
    """Same seed + same carrier axis -> identical draws for every family
    column (the ONE-call shared-index contract)."""
    values = np.array([[0.5, 0.6, np.nan], [0.6, 0.6, 0.2]])
    from issue2094_analysis import bootstrap_family_means_batched

    b1 = bootstrap_family_means_batched(values, n_boot=64, seed=DEC.DECAY_BOOT_SEED)
    b2 = bootstrap_family_means_batched(values, n_boot=64, seed=DEC.DECAY_BOOT_SEED)
    assert np.array_equal(b1, b2, equal_nan=True)
    # index sharing: a constant column's draws are constant wherever finite
    const = bootstrap_family_means_batched(
        np.array([[0.6, 0.6], [0.6, 0.6]]), n_boot=64, seed=DEC.DECAY_BOOT_SEED
    )
    assert np.allclose(const, 0.6)


# ── injection-gate pe second-row seam (ladder donor-map convention) ────


def _mk_ladder_pair(pair_id="install_r1_pirate::d1", a="ctxA", b="ctxB"):
    return LB.LadderPair(
        pair_id=pair_id,
        cell="install_r1_pirate",
        kind="install",
        persona="pirate",
        carrier="d1",
        value_a="neutral",
        value_b="pirate",
        a=a,
        b=b,
    )


def test_parent_pe_default_keyerrors_on_ladder_donor_maps():
    # The pre-fix crash shape: RUN.run_injection_gate's DEFAULT second-row
    # predicate hard-calls pe_excluded_reason, which expects parent keys
    # {"shuffled","crosstype"} — the ladder maps {"null_sameval","null_xtype",
    # "null_xtype_pe"} KeyError on any null arm (observed live at the tiny
    # bank smoke: KeyError 'crosstype' at issue2329_run.py pe_excluded_reason).
    pair = _mk_ladder_pair()
    ladder_maps = {"null_sameval": {}, "null_xtype": {}, "null_xtype_pe": {}}
    with pytest.raises(KeyError, match="crosstype"):
        RUN.pe_excluded_reason(pair, "null_xtype", frozenset(), ladder_maps, {})


def test_injection_gate_exposes_pe_second_row_seam_and_ladder_threads_it():
    # (a) the gate exposes the keyword-only seam (pre-fix: absent)
    params = inspect.signature(RUN.run_injection_gate).parameters
    assert "pe_second_row_ok" in params
    assert params["pe_second_row_ok"].kind is inspect.Parameter.KEYWORD_ONLY
    # (b) the ladder bank phase threads a ladder-aware predicate into it
    src = inspect.getsource(LAD.phase_bank)
    assert "pe_second_row_ok=" in src and "pe_second_row_ok_ladder" in src


def test_pe_second_row_ok_ladder_semantics():
    p = _mk_ladder_pair()
    maps = {
        "null_sameval": {p.pair_id: "donorL"},
        "null_xtype": {p.pair_id: "parentB"},
        "null_xtype_pe": {p.pair_id: "parentB"},
    }
    ok = LAD.pe_second_row_ok_ladder
    # steered: recipient np-ness is the only constraint
    assert ok(p, "steered", frozenset(), maps) is True
    assert ok(p, "steered", frozenset({p.a}), maps) is False
    assert ok(p, "steered", frozenset({p.b}), maps) is False
    # null_sameval: ladder donor must exist and carry a pe token
    assert ok(p, "null_sameval", frozenset(), maps) is True
    assert ok(p, "null_sameval", frozenset({"donorL"}), maps) is False
    no_donor = {**maps, "null_sameval": {}}
    assert ok(p, "null_sameval", frozenset(), no_donor) is False
    # null_xtype: membership in the pre-filtered pe-viable subset
    assert ok(p, "null_xtype", frozenset(), maps) is True
    pe_excluded = {**maps, "null_xtype_pe": {}}
    assert ok(p, "null_xtype", frozenset(), pe_excluded) is False
    # unknown arm fails loud
    with pytest.raises(AssertionError):
        ok(p, "shuffled", frozenset(), maps)


# ── review round 1 must-fix pins (items 2/3/4/5/7/8 + g2/g6) ──────────


def test_q25_stats_json_default_points_at_committed_parent_stats():
    # must-fix 2: the parent #2162 ladder stats live at the round ROOT — the
    # f_metrics/ subdir is this fork's own L6 output layout, absent on q25.
    for mod in (DEC, LA):
        src = inspect.getsource(mod.parse_args)
        assert "eval_results/issue_2162/persona_specificity_ladder/stats.json" in src, mod
        assert "issue_2162/persona_specificity_ladder/f_metrics" not in src, mod


def test_judge_upload_routes_to_raw_completions_class():
    # must-fix 3: judge raws are RAW-COMPLETIONS class, never analysis_tensors,
    # and the upload runs a scoped exact-set verify.
    src = inspect.getsource(LJ.phase_upload)
    assert 'dest = f"{LADDER_RAW}/judge_raw"' in src
    assert "LADDER_TENSORS" not in src
    assert "verify_repo_paths_uploaded" in src
    assert LJ.LADDER_RAW.endswith("/raw_completions/ladder")


def test_q35_parent_hf_revision_pins_equal():
    # g2: the judge fork keeps a LOCAL copy of the parent revision pin — the
    # two constants must never drift.
    assert LJ.Q35_PARENT_HF_REVISION == LAD.Q35_PARENT_HF_REVISION
    assert len(LJ.Q35_PARENT_HF_REVISION) == 40


# ── must-fix 4: coherence coverage gate (fail-able) ───────────────────


def test_coherence_coverage_missing_row_fails(decay_cfg):
    cfg = decay_cfg
    p = cfg.q35_scores_dir / "coherence.grid.scores.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    kept = [
        r
        for r in rows
        if not (r["arm"] == "steered" and r["draw"] == 0 and r["pair_id"] == f"install_{V}::d1")
    ]
    assert len(kept) == len(rows) - 1
    _write_jsonl(p, kept)
    with pytest.raises(RuntimeError, match="coherence coverage mismatch"):
        DEC._build_sides(cfg)


def test_coherence_coverage_duplicate_row_fails(decay_cfg):
    cfg = decay_cfg
    p = cfg.q35_scores_dir / "coherence.grid.scores.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    rows.append(dict(rows[0]))
    _write_jsonl(p, rows)
    with pytest.raises(RuntimeError, match="duplicate coherence row"):
        DEC._build_sides(cfg)


def test_empty_steered_selection_raises_before_any_unit(decay_cfg):
    # Review r2 blocker 1 (reconciler probe C): with every SELECTED steered
    # grid row deleted (nonselected shard rows retained), the coverage assert
    # is vacuous (empty expected == empty present) and _build_sides would
    # construct anchor-only judge units. The nonempty-selection guard must
    # raise FIRST — before the coverage assertion, before any unit is built.
    cfg = decay_cfg
    for root, mod in ((cfg.q25_in_root, PLJ), (cfg.q35_in_root, LJ)):
        p = root / mod.LADDER_RAW / "grid" / "shard_000.jsonl"
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        kept = [r for r in rows if not (r["cell"].startswith("install_") and r["arm"] == "steered")]
        assert kept and len(kept) < len(rows)  # nonselected shard rows retained
        _write_jsonl(p, kept)
    with pytest.raises(RuntimeError, match="EMPTY steered selection"):
        DEC._build_sides(cfg)


def _rewrite_shard(path: Path, fn):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = fn(rows)
    _write_jsonl(path, rows)
    return rows


def test_anchor_only_side_refused_when_all_steered_under_floor(decay_cfg):
    # Review r3 item 1 (Door A): a NONEMPTY steered selection whose every row
    # fails _segment's 48-token floor emits ZERO steered units while anchors
    # survive — the reconciler-ruled anchor-only outcome, reached through the
    # emit-time filter the r2 selection guard does not see. Must refuse
    # (report-only retention counters are not a gate).
    cfg = decay_cfg
    p = cfg.q25_in_root / PLJ.LADDER_RAW / "grid" / "shard_000.jsonl"

    def shorten(rows):
        for r in rows:
            if r["cell"].startswith("install_") and r["arm"] == "steered":
                r["text"] = SHORT_TEXT
                r["n_completion_tokens"] = len(SHORT_TEXT.split())
        return rows

    _rewrite_shard(p, shorten)
    with pytest.raises(RuntimeError, match=r"EMPTY REQUIRED ARM\(S\) \['steered'\]"):
        DEC._build_sides(cfg)


def _shorten_steered_grid(rows):
    """Push every selected steered grid completion under the 48-token floor."""
    for r in rows:
        if r["cell"].startswith("install_") and r["arm"] == "steered":
            r["text"] = SHORT_TEXT
            r["n_completion_tokens"] = len(SHORT_TEXT.split())
    return rows


def _shorten_anchors(only_ceiling: bool = False):
    """Rewriter pushing anchor completions under the 48-token floor
    (optionally only the ceiling-arm contexts, leaving floor/plain intact)."""
    ceiling_cids = {LB.context_id(V, c) for c in CARRIERS}

    def fn(rows):
        for r in rows:
            if only_ceiling and r["context_id"] not in ceiling_cids:
                continue
            r["text"] = SHORT_TEXT
        return rows

    return fn


def test_side_refused_when_all_arms_under_floor(decay_cfg):
    # Review r4 reconciler MF-2(i), executed scenario A (both-zero side): every
    # q25 completion — steered grid rows AND anchors — under the 48-token floor
    # emits zero units of BOTH kinds. The r3 guard's `n_anchor_units > 0`
    # conjunct could not fire here (phase_wave dispatched a 144-unit q35-only
    # wave and returned RC_OK); the unconditional per-arm refusal must raise,
    # naming all three arms.
    cfg = decay_cfg
    _rewrite_shard(
        cfg.q25_in_root / PLJ.LADDER_RAW / "grid" / "shard_000.jsonl", _shorten_steered_grid
    )
    _rewrite_shard(
        cfg.q25_in_root / PLJ.LADDER_RAW / "anchors" / "anchors_gate_w0.jsonl",
        _shorten_anchors(),
    )
    with pytest.raises(
        RuntimeError, match=r"EMPTY REQUIRED ARM\(S\) \['steered', 'ceiling', 'floor'\]"
    ):
        DEC._build_sides(cfg)


def test_steered_only_side_refused_when_all_anchors_under_floor(decay_cfg):
    # Review r4 reconciler MF-2(ii), executed scenario B (steered-only side,
    # retention 48/0/0): every q25 ANCHOR completion under the floor while the
    # steered grid stays intact — the r3 guard keyed only on steered-empty and
    # built 48 uncontrastable steered units. Must refuse naming both anchor
    # arms.
    cfg = decay_cfg
    _rewrite_shard(
        cfg.q25_in_root / PLJ.LADDER_RAW / "anchors" / "anchors_gate_w0.jsonl",
        _shorten_anchors(),
    )
    with pytest.raises(RuntimeError, match=r"EMPTY REQUIRED ARM\(S\) \['ceiling', 'floor'\]"):
        DEC._build_sides(cfg)


@pytest.mark.parametrize("scenario", ["both_zero", "steered_only", "ceiling_empty"])
def test_stale_pilot_wave_refuses_before_any_dispatch(decay_cfg, monkeypatch, scenario):
    # Review r4 reconciler (recommended pin): on the stale-pilot phase_wave
    # path — a passed pilot_gate_report.json already on disk, artifacts
    # re-staged degenerate — a refused side must raise BEFORE any J94.run_wave
    # call. Pins the property the reconciler MEASURED (pre-fix: one real wave
    # of 144 / 192 / 240 units dispatched for these scenarios), not the
    # guard's source shape.
    cfg = decay_cfg
    grid_p = cfg.q25_in_root / PLJ.LADDER_RAW / "grid" / "shard_000.jsonl"
    anch_p = cfg.q25_in_root / PLJ.LADDER_RAW / "anchors" / "anchors_gate_w0.jsonl"
    if scenario == "both_zero":
        _rewrite_shard(grid_p, _shorten_steered_grid)
        _rewrite_shard(anch_p, _shorten_anchors())
    elif scenario == "steered_only":
        _rewrite_shard(anch_p, _shorten_anchors())
    else:  # ceiling_empty: retention 48/0/48
        _rewrite_shard(anch_p, _shorten_anchors(only_ceiling=True))
    gates_dir = cfg.j94().gates_dir
    gates_dir.mkdir(parents=True, exist_ok=True)
    (gates_dir / "pilot_gate_report.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    run_wave = create_autospec(DEC.J94.run_wave)
    monkeypatch.setattr(DEC.J94, "run_wave", run_wave)
    with pytest.raises(RuntimeError, match=r"EMPTY REQUIRED ARM"):
        DEC.phase_wave(cfg)
    assert run_wave.call_count == 0


def test_absent_pair_slot_registers_missing(decay_cfg):
    # Review r3 item 2 (Door B, the Codex probe verbatim): delete every
    # selected pair/slot but one COMPLETE pair — the draw-level coverage
    # equality is blind (expected shrinks with the staged rows); the
    # gate+manifest-derived lattice check must register the absent pair/slot
    # as MISSING and fail loud before any unit is built.
    cfg = decay_cfg
    p = cfg.q25_in_root / PLJ.LADDER_RAW / "grid" / "shard_000.jsonl"

    def drop_d2(rows):
        kept = [
            r
            for r in rows
            if not (
                r["cell"].startswith("install_")
                and r["arm"] == "steered"
                and r["pair_id"] == f"install_{V}::d2"
            )
        ]
        assert kept and len(kept) < len(rows)
        return kept

    _rewrite_shard(p, drop_d2)
    with pytest.raises(RuntimeError, match="lattice mismatch"):
        DEC._build_sides(cfg)


def test_tokgate_dropped_pair_absence_is_legitimate(decay_cfg):
    # The lattice subtracts G0 tokgate-dropped pairs (a dropped pair generates
    # ZERO rows in every arm per LA.registered_row_keys) — its absence from
    # the staged rows is legitimate, never a lattice-mismatch refusal. The
    # report rides the q35 side; the q25 side has none by design (#2162 never
    # re-tokenized the ladder bank).
    cfg = decay_cfg
    tokrep = {
        "pairs": [
            {"pair_id": f"install_{V}::d1", "intact": True},
            {"pair_id": f"install_{V}::d2", "intact": False},
        ],
        "directions": {f"install_{V}": {"testable": True}},
        "bank_sha": "x" * 16,
    }
    (cfg.q35_gates_dir / "token_identity_report_ladder.json").write_text(
        json.dumps(tokrep), encoding="utf-8"
    )
    p = cfg.q35_in_root / LJ.LADDER_RAW / "grid" / "shard_000.jsonl"
    _rewrite_shard(p, lambda rows: [r for r in rows if r["pair_id"] != f"install_{V}::d2"])
    sides = DEC._build_sides(cfg)  # must not raise
    steered_carriers = {
        u.source["carrier"] for u in sides["q35"].units if u.source["arm"] == "steered"
    }
    assert steered_carriers == {"d1"}


def test_q25_token_count_reconciliation_deviation_fails(decay_cfg):
    # Plan v8 assumption 9 (R-2): re-tokenize EVERY q25 grid row and reconcile
    # against stored n_completion_tokens within ±2 on the FULL corpus.
    cfg = decay_cfg
    p = cfg.q25_in_root / PLJ.LADDER_RAW / "grid" / "shard_000.jsonl"

    def deviate(rows):
        rows[0]["n_completion_tokens"] = len(rows[0]["text"].split()) + 3  # past ±2
        return rows

    _rewrite_shard(p, deviate)
    with pytest.raises(RuntimeError, match="assumption 9"):
        DEC._build_sides(cfg)


def test_q25_token_count_reconciliation_within_tolerance_passes(decay_cfg):
    cfg = decay_cfg
    p = cfg.q25_in_root / PLJ.LADDER_RAW / "grid" / "shard_000.jsonl"

    def nudge(rows):
        rows[0]["n_completion_tokens"] = len(rows[0]["text"].split()) + 2  # at the bound
        return rows

    _rewrite_shard(p, nudge)
    DEC._build_sides(cfg)  # must not raise


def test_q25_token_count_missing_field_fails(decay_cfg):
    cfg = decay_cfg
    p = cfg.q25_in_root / PLJ.LADDER_RAW / "grid" / "shard_000.jsonl"

    def strip_field(rows):
        del rows[0]["n_completion_tokens"]
        return rows

    _rewrite_shard(p, strip_field)
    with pytest.raises(RuntimeError, match="assumption 9"):
        DEC._build_sides(cfg)


# ── must-fix 5: structurally undefined trend -> None, never a finite p ─


def _trend_gate(surviving: set[str]) -> dict:
    return {
        "rungs": {
            u: {"survived": u in surviving, "surviving_carriers": ["d1", "d2"]}
            for u in LB.PERSONA_VALUE_IDS
        }
    }


def _trend_rows(rungs, f_by_rung):
    return [
        {"kind": "install", "slot": "ce", "rung": v, "carrier": c, "f_target": f_by_rung[v]}
        for v in rungs
        for c in ("d1", "d2")
    ]


def test_trend_test_constant_means_reports_untestable():
    rungs = list(LB.PERSONA_VALUE_IDS[:3])
    rows = _trend_rows(rungs, dict.fromkeys(rungs, 0.5))
    rec = LA.trend_test(
        rows, _trend_gate(set(rungs)), "install", "ce", np.random.default_rng(0), 200
    )
    assert rec["rho_observed"] is None
    assert rec["p_one_sided"] is None and rec["p_two_sided"] is None
    assert "Spearman undefined" in rec["trend_undefined_reason"]
    assert rec["n_permutations_effective"] == 0


def test_trend_test_nondegenerate_control_returns_finite_p():
    rungs = list(LB.PERSONA_VALUE_IDS[:3])
    f_by = {v: 0.9 - 0.25 * i for i, v in enumerate(rungs)}
    rec = LA.trend_test(
        _trend_rows(rungs, f_by),
        _trend_gate(set(rungs)),
        "install",
        "ce",
        np.random.default_rng(0),
        200,
    )
    assert rec["rho_observed"] is not None and np.isfinite(rec["rho_observed"])
    assert 0 < rec["p_one_sided"] <= 1 and 0 < rec["p_two_sided"] <= 1
    assert "trend_undefined_reason" not in rec


def test_trend_test_tokgate_untestable_rung_excluded():
    # g3-2 / divergence 7: a G0-untestable direction's rung leaves the rung set
    # (it generates ZERO steered rows; counting it would data-starve the test).
    rungs = list(LB.PERSONA_VALUE_IDS[:3])
    f_by = {v: 0.9 - 0.25 * i for i, v in enumerate(rungs)}
    tokrep = {
        "directions": {f"install_{v}": {"testable": v != rungs[0]} for v in LB.PERSONA_VALUE_IDS}
    }
    rec = LA.trend_test(
        _trend_rows(rungs, f_by),
        _trend_gate(set(rungs)),
        "install",
        "ce",
        np.random.default_rng(0),
        200,
        tokrep=tokrep,
    )
    assert rec["surviving_rungs"] == rungs[1:]
    assert rec["n_surviving_rungs"] == 2
    assert rec["descriptive_only"] is True  # 2 < MIN_TREND_RUNGS


# ── must-fix 1: LOCO folds (descriptive, per-fold seeded) ─────────────


def test_loco_trend_folds_shape_and_holdout():
    rungs = list(LB.PERSONA_VALUE_IDS[:3])
    rows = []
    for fam in LA.FAMILIES:
        kind, slot = fam.split("-")
        for i, v in enumerate(rungs):
            for c in ("d1", "d2"):
                rows.append(
                    {
                        "kind": kind,
                        "slot": slot,
                        "rung": v,
                        "carrier": c,
                        "f_target": 0.9 - 0.25 * i + (0.02 if c == "d2" else 0.0),
                    }
                )
    out = LA.loco_trend_folds(rows, _trend_gate(set(rungs)), None, 100)
    assert out["n_folds"] == 2
    assert out["held_out_carriers"] == ["d1", "d2"]
    for held_out, fold in out["folds"].items():
        assert sorted(fold) == sorted(LA.FAMILIES)
        for fam_rec in fold.values():
            # exactly one carrier remains per fold
            assert fam_rec["n_carriers"] == 1
            assert held_out in ("d1", "d2")


def test_stats_json_carries_loco_folds_wiring():
    src = inspect.getsource(LA.step_stats)
    assert '"loco_folds": loco_trend_folds(' in src


# ── must-fix 7: floor support gates ONLY the normalized companion ─────


def _synth_wave_rows(sides):
    return [
        {
            "item_id": u.item_id,
            "wave": f"{u.rubric_id}.decay",
            "rubric_id": u.rubric_id,
            "score": _frag_score(u.source),
            "n_kept_draws": 1,
            "transport_lost_residual": 0,
            **u.source,
        }
        for key in DEC.MODEL_KEYS
        for u in sides[key].units
    ]


def test_reduce_no_floor_support_retains_raw_delta(decay_cfg):
    cfg = decay_cfg
    sides = DEC._build_sides(cfg)
    rows = _synth_wave_rows(sides)
    # blank every floor score on carrier d2 -> zero kept floor completions
    # there -> floor common support fails while steered+ceiling stay supported
    for r in rows:
        if r["arm"] == "floor" and r["carrier"] == "d2":
            r["score"] = None
    _write_jsonl(cfg.j94().scores_dir / f"dfrag-{V}.decay.scores.jsonl", rows)
    assert DEC.phase_reduce(cfg) == DEC.RC_OK
    stats = json.loads((cfg.out_dir / "decay_stats.json").read_text(encoding="utf-8"))
    for key in DEC.MODEL_KEYS:
        rec = stats["per_direction"][f"{key}|install_{V}|ce|coh"]["per_carrier"]["d2"]
        assert rec["supported"] is True and rec["supported_norm"] is False
        assert rec["delta_d"] is not None
        assert rec["delta_d_f"] is None and rec["mean_floor"] is None
        assert "no floor common support" in rec["delta_d_f_unavailable_reason"]
        # raw dD keeps BOTH carriers; the normalized companion keeps only d1
        assert stats["families"][key]["coh|primary|dD"]["n_carriers"] == 2
        assert stats["families"][key]["coh|primary|dD_F"]["n_carriers"] == 1


# ── g6-2: the 4-way verdict lattice, table-tested ─────────────────────


def _fam(dd_point, dd_lo, dd_hi, ddf_point=None, ddf_lo=None, ddf_hi=None):
    return {
        "coh|primary|dD": {"point": dd_point, "ci_lo": dd_lo, "ci_hi": dd_hi},
        "coh|primary|dD_F": {"point": ddf_point, "ci_lo": ddf_lo, "ci_hi": ddf_hi},
    }


def test_verdict_label_covers_every_lattice_branch():
    vl = DEC.verdict_label
    assert vl(_fam(None, None, None), "coh") == ("inconclusive", "no supported carriers")
    assert vl(_fam(0.5, 0.1, 0.9), "coh") == ("patch-decays-faster", None)
    label, reason = vl(_fam(-0.5, -0.9, -0.1), "coh")
    assert label == "inconclusive" and "confounded" in reason
    assert vl(_fam(-0.5, -0.9, -0.1, ddf_point=-0.4, ddf_lo=-0.8, ddf_hi=-0.05), "coh") == (
        "patch-more-persistent",
        None,
    )
    label, reason = vl(_fam(-0.5, -0.9, -0.1, ddf_point=-0.2, ddf_lo=-0.6, ddf_hi=0.1), "coh")
    assert label == "inconclusive" and "dD_F CI spans zero" in reason
    label, reason = vl(_fam(0.1, -0.2, 0.4), "coh")
    assert label == "inconclusive" and reason == "dD CI spans zero"


# ── must-fix 6a: the transfer figure renders (both-testable + flip) ────


def test_fig_transfer_renders(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    key = "install_x|ce"
    parent_est = {
        f"{key}|steered": {"mean_f_target": 0.8, "ci_lo": 0.7, "ci_hi": 0.9},
        f"{key}|null_sameval": {"mean_f_target": 0.1, "ci_lo": 0.0, "ci_hi": 0.2},
        f"{key}|null_xtype": {"mean_f_target": 0.12, "ci_lo": 0.02, "ci_hi": 0.22},
    }
    fork_est = {
        f"{key}|steered": {"mean_f_target": 0.4, "ci_lo": 0.3, "ci_hi": 0.5},
        f"{key}|null_sameval": {"mean_f_target": 0.15, "ci_lo": 0.05, "ci_hi": 0.25},
        f"{key}|null_xtype": {"mean_f_target": 0.2, "ci_lo": 0.1, "ci_hi": 0.3},
    }
    parent = {"lattice": {key: {"verdict": "specific"}}, "estimation": parent_est}
    fork = {"lattice": {key: {"verdict": "nonspecific"}}, "estimation": fork_est}
    LA.fig_transfer(fork, parent, SimpleNamespace(figures_dir=tmp_path))
    assert (tmp_path / "q35_ladder_decay_transfer.png").exists()


def test_fig_hero_and_anchor_separation_execute_manifest_stems(tmp_path):
    # Review r2 blocker 3: EXECUTE both renamed producers on a tiny fixture
    # and pin the exact manifest output paths — reverting either stem (e.g.
    # back to the parent's "ladder_hero") must flip this test RED.
    import matplotlib

    matplotlib.use("Agg")
    gate = {
        "rungs": {
            v: {"survived": True, "surviving_carriers": ["d1", "d2"]} for v in LB.PERSONA_VALUE_IDS
        },
        "bars": {"target_sep_bar": 0.25, "netted_sep_bar": 0.5},
    }
    tokrep = {"directions": {d: {"testable": True} for d in LB.direction_ids()}}
    est = {
        f"{kind}_{v}|{slot}|{arm}": {
            "mean_f_target": 0.5,
            "ci_lo": 0.4,
            "ci_hi": 0.6,
            "n_carriers": 2,
        }
        for kind in ("install", "erase")
        for v in LB.PERSONA_VALUE_IDS
        for slot in LA.SLOTS
        for arm in LA.ARMS
    }
    args = SimpleNamespace(figures_dir=tmp_path, skip_token_counts=True)
    LA.fig_ladder_hero({"estimation": est}, gate, tokrep, args)
    anchors = [
        {
            "rung": v,
            "carrier": "d1",
            "gate_target_sep": 0.4,
            "gate_netted_sep": 0.7,
            "gate_passed": True,
        }
        for v in LB.PERSONA_VALUE_IDS
    ]
    LA.fig_anchor_separation(anchors, gate, args)
    assert (tmp_path / "q35_ladder_decay_hero_ladder.png").exists()
    assert (tmp_path / "q35_ladder_decay_anchor_separation.png").exists()


# ── must-fix 8: smoke slices the gate-threaded PRODUCTION enumeration ──


def _smoke_pairs():
    mk = LB.LadderPair
    return [
        mk(
            pair_id="install_r1_pirate::d1",
            cell="install_r1_pirate",
            kind="install",
            persona="pirate",
            carrier="d1",
            value_a="neutral",
            value_b="pirate",
            a="ctxA",
            b="ctxB",
        ),
        mk(
            pair_id="erase_r1_pirate::d1",
            cell="erase_r1_pirate",
            kind="erase",
            persona="pirate",
            carrier="d1",
            value_a="pirate",
            value_b="neutral",
            a="ctxC",
            b="ctxD",
        ),
        mk(
            pair_id="install_r1_pirate::d2",
            cell="install_r1_pirate",
            kind="install",
            persona="pirate",
            carrier="d2",
            value_a="neutral",
            value_b="pirate",
            a="ctxE",
            b="ctxF",
        ),
    ]


def test_smoke_slice_blocks_intersects_production_enumeration():
    pairs = _smoke_pairs()
    production = [
        RUN.Block(b.cell, b.slot, b.arm, (*b.pair_ids, "install_r1_pirate::d2"))
        for b in LAD.smoke_ladder_blocks(pairs)
    ]
    production.append(RUN.Block("install_r2_formal", "ce", "steered", ("x",)))
    sliced = LAD.smoke_slice_blocks(pairs, production)
    assert len(sliced) == 12
    assert all(b.cell != "install_r2_formal" for b in sliced)
    for b in sliced:
        expect = "erase_r1_pirate::d1" if b.cell.startswith("erase") else "install_r1_pirate::d1"
        assert b.pair_ids == (expect,), b.key


def test_smoke_slice_blocks_gate_dropped_cells_are_logged_not_fatal():
    pairs = _smoke_pairs()
    # gates dropped every erase block: the slice narrows to the 6 install cells
    production = [b for b in LAD.smoke_ladder_blocks(pairs) if b.cell == "install_r1_pirate"]
    sliced = LAD.smoke_slice_blocks(pairs, production)
    assert len(sliced) == 6
    assert all(b.cell == "install_r1_pirate" for b in sliced)


def test_smoke_slice_blocks_empty_slice_raises():
    pairs = _smoke_pairs()
    production = [RUN.Block("install_r2_formal", "ce", "steered", ("x",))]
    with pytest.raises(AssertionError, match="smoke slice EMPTY"):
        LAD.smoke_slice_blocks(pairs, production)


def test_grid_inputs_gate_asserts_are_unconditional():
    src = inspect.getsource(LAD._grid_inputs)
    assert "if not cfg.smoke" not in src
    smoke_pos = src.index("if cfg.smoke")
    for tok in (
        "--gate-verdict required",
        "--donor-screen required",
        "--token-identity required",
    ):
        assert src.index(tok) < smoke_pos, tok


# Review r2 blocker 2: BEHAVIORAL pin — _grid_inputs EXECUTED under
# cfg.smoke=True with real minimal gate fixtures; the gate CONTENTS must
# reach the returned blocks. A driver mutation that ignores gate contents
# under smoke (stubbed survivors/screen/tokrep) flips these tests RED,
# which the syntactic source pin above cannot do.


def _grid_gate_cfg(tmp_path, *, not_intact=(), r1_survives=True):
    """Minimal REAL gate artifacts + ladder bank consumed by _grid_inputs.

    Covers the two smoke directions (persona r1_pirate) at carriers d1/d2;
    every other direction is marked G0-untestable so the production
    enumeration is well-defined on the 4-pair fixture bank.
    """
    covered = list(LAD.SMOKE_DIRECTIONS)
    pairs_rows = []
    for d in covered:
        kind, v = d.split("_", 1)
        for c in ("d1", "d2"):
            va, vb = ("plain", v) if kind == "install" else (v, "plain")
            pairs_rows.append(
                {
                    "pair_id": f"{d}::{c}",
                    "direction": d,
                    "kind": kind,
                    "persona": v,
                    "carrier": c,
                    "value_a": va,
                    "value_b": vb,
                    "a": LB.context_id(va, c),
                    "b": LB.context_id(vb, c),
                }
            )
    manifest = {
        "bank_sha": "f" * 16,
        "pairs": pairs_rows,
        "sameval_donor": {"order": ["d1", "d2"]},
        "crosstype_donor_plan": {},  # unused: a donor-screen file is always given
        "parent_no_prefix_context_ids": [],
    }
    bank_dir = tmp_path / "vc_bank"
    bank_dir.mkdir(parents=True, exist_ok=True)
    (bank_dir / "ladder_bank.json").write_text(json.dumps(manifest), encoding="utf-8")
    rungs = {
        "r1_pirate": {
            "survived": r1_survives,
            "surviving_carriers": ["d1", "d2"] if r1_survives else [],
        },
        # keeps read_gate_verdict's nonempty-survivor assert satisfied when the
        # smoke rung is verdict-dropped (r2_butler's directions are untestable
        # in this fixture, so it never reaches the enumeration).
        "r2_butler": {"survived": True, "surviving_carriers": ["d1", "d2"]},
    }
    gate_path = tmp_path / "gate_verdict.json"
    gate_path.write_text(json.dumps({"rungs": rungs}), encoding="utf-8")
    screen = {
        "assignments": {
            row["pair_id"]: {"status": "primary", "donor": {"b": f"parent::{row['carrier']}"}}
            for row in pairs_rows
        }
    }
    screen_path = tmp_path / "donor_screen.json"
    screen_path.write_text(json.dumps(screen), encoding="utf-8")
    tokrep = {
        "bank_sha": manifest["bank_sha"],
        "pairs": [
            {"pair_id": row["pair_id"], "intact": row["pair_id"] not in set(not_intact)}
            for row in pairs_rows
        ],
        "directions": {d: {"testable": d in covered} for d in LB.direction_ids()},
    }
    tok_path = tmp_path / "token_identity.json"
    tok_path.write_text(json.dumps(tokrep), encoding="utf-8")
    return SimpleNamespace(
        bank_dir=bank_dir,
        gate_verdict_path=gate_path,
        donor_screen_path=screen_path,
        token_identity_path=tok_path,
        smoke=True,
        model_id="Qwen/Qwen3.5-9B",
        model_revision=None,
        tiny=True,
        n_layers=4,
        hidden=64,
        max_new_tokens=64,
        grid_draws=1,
        seed_base=42,
    )


def test_grid_inputs_smoke_threads_gate_contents_healthy(tmp_path):
    # all pairs intact + r1 survives with both carriers -> the full 12-cell
    # smoke slice, every block narrowed to the SMOKE_CARRIER pair.
    _, meta, _, _, dropped, blocks, _ = LAD._grid_inputs(_grid_gate_cfg(tmp_path))
    assert len(blocks) == 12
    assert {b.cell for b in blocks} == set(LAD.SMOKE_DIRECTIONS)
    assert all(b.pair_ids == (f"{b.cell}::{LAD.SMOKE_CARRIER}",) for b in blocks)
    assert dropped == [] and meta["tokgate_dropped_pairs"] == []


def test_grid_inputs_smoke_tokgate_content_excludes_broken_pair_cells(tmp_path):
    # G0 CONTENT reaches the smoke blocks: the install smoke pair marked
    # not-intact -> its 6 cells vanish from the slice; erase cells remain.
    cfg = _grid_gate_cfg(tmp_path, not_intact={"install_r1_pirate::d1"})
    *_, blocks, _ = LAD._grid_inputs(cfg)
    assert len(blocks) == 6
    assert {b.cell for b in blocks} == {"erase_r1_pirate"}


def test_grid_inputs_smoke_tokgate_all_smoke_pairs_broken_raises(tmp_path):
    # both smoke-carrier pairs G0-broken -> the sliced enumeration is EMPTY
    # and the driver must raise, never run a gate-free smoke.
    cfg = _grid_gate_cfg(tmp_path, not_intact={"install_r1_pirate::d1", "erase_r1_pirate::d1"})
    with pytest.raises(AssertionError, match="smoke slice EMPTY"):
        LAD._grid_inputs(cfg)


def test_grid_inputs_smoke_gate_verdict_dropped_rung_raises(tmp_path):
    # gate-verdict CONTENT binds under smoke: the smoke rung verdict-dropped
    # (another rung survives) -> its directions generate nothing -> EMPTY raise.
    with pytest.raises(AssertionError, match="smoke slice EMPTY"):
        LAD._grid_inputs(_grid_gate_cfg(tmp_path, r1_survives=False))


def _run_dispatch(tmp_path, *args, extra_env=None):
    env = {
        **os.environ,
        "REPO_ROOT": str(REPO_ROOT),
        "EPM_2329L_OUT_ROOT": str(tmp_path / "out"),
        "EPM_2329L_LOG_DIR": str(tmp_path / "logs"),
        **(extra_env or {}),
    }
    proc = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "issue2329_ladder_dispatch.sh"), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_dispatcher_smoke_grid_hard_requires_gate_inputs(tmp_path):
    # must-fix 8 (dispatcher side): --smoke no longer bypasses the grid gates —
    # a missing G0 report is the designed rc=30 HALT even in smoke mode.
    rc, out = _run_dispatch(tmp_path, "grid", "--smoke")
    assert rc == 30, out
    assert "token-identity" in out


def test_dispatcher_margin_hard_requires_gate_inputs(tmp_path):
    # margin re-enumerates via _grid_inputs -> the same three gate files bind
    rc, out = _run_dispatch(tmp_path, "margin", "--smoke")
    assert rc == 30, out
    assert "token-identity" in out


def test_dispatcher_skip_env_is_exact_one_compare(tmp_path):
    # g5-2: any non-"1" value must NOT skip the gates
    rc, out = _run_dispatch(
        tmp_path, "grid", "--smoke", extra_env={"EPM_2329L_SKIP_GRID_GATES": "false"}
    )
    assert rc == 30, out
    # "1" without a recorded justification refuses loudly (rc=26)
    rc, out = _run_dispatch(
        tmp_path, "grid", "--smoke", extra_env={"EPM_2329L_SKIP_GRID_GATES": "1"}
    )
    assert rc == 26, out
    assert "justification" in out


def test_dispatcher_stage2_reasserts_venv_pin():
    # g5-1: stage2 is a separate dispatch invocation -> gate0b re-runs first
    src = (REPO_ROOT / "scripts" / "issue2329_ladder_dispatch.sh").read_text(encoding="utf-8")
    stage2 = src.split("run_stage2() {", 1)[1].split("}", 1)[0]
    assert "run_gate0b" in stage2
    assert stage2.index("run_gate0b") < stage2.index("run_grid")
