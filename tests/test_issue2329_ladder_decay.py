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
import sys
from pathlib import Path
from types import SimpleNamespace

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
    report = LAD.run_donor_identity_assert(
        _g1_cfg(tiny=True),
        None,
        None,
        _donor_manifest(list(LAD.DONOR_IDENTITY_CONTEXT_IDS)),
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
        return (80 if src["carrier"] == "d1" else 12) + d
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

    # dual estimands numerically split: coh drops the flat completion
    for key in DEC.MODEL_KEYS:
        coh_dd = stats["families"][key]["coh|primary|dD"]
        all_dd = stats["families"][key]["all|primary|dD"]
        assert coh_dd["point"] == pytest.approx(0.60, abs=1e-9)
        assert all_dd["point"] == pytest.approx(0.55, abs=1e-9)
        assert coh_dd["ci_lo"] > 0 and all_dd["ci_lo"] > 0
        assert coh_dd["n_carriers"] == 2
        # ceiling raw drop identically 0 by construction (draw offsets cancel)
        assert stats["families"][key]["coh|primary|Draw_ceiling"]["point"] == pytest.approx(
            0.0, abs=1e-9
        )
        # denominator bar: d2's |ceiling-floor| = 0.02 < 0.125 -> dD_F only on d1
        ddf = stats["families"][key]["coh|primary|dD_F"]
        assert ddf["n_carriers"] == 1
        assert ddf["point"] == pytest.approx(0.857, abs=0.01)
        rec = stats["per_direction"][f"{key}|install_{V}|ce|coh"]["per_carrier"]
        assert rec["d2"]["delta_d_f"] is None
        assert "0.125" in rec["d2"]["delta_d_f_unavailable_reason"]
        assert rec["d1"]["delta_d_f"] is not None
        # verdict lattice: both estimands' dD CIs > 0 -> patch-decays-faster
        assert stats["lattice"][key]["verdict"] == "patch-decays-faster"
        for e in DEC.ESTIMANDS:
            assert stats["lattice"][key]["per_estimand"][e]["label"] == "patch-decays-faster"
        # N2.2 Q1 gap: steered seg1 above ceiling seg1 on d1, below on d2
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
        "q35_ladder_decay_decay_diagnostics",
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
