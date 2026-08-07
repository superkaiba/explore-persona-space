"""CPU-only tests for the #2094 gpu2_mq_replacement_prefix round.

Groups (per the round brief):
(a) the 5-pair diagnosis PINNED against the committed parent artifacts
    (``eval_results/issue_2094/f_metrics/anchors.jsonl`` + judge scores),
    incl. the escape-hatch leg (a synthetic anchors file where the queries
    fail their persona control -> ``reframe-needed``);
(b) conv2 context construction on the REAL Qwen tokenizer (multi-turn
    history render, ``prefix_end_index_multi``, id disjointness);
(c) pair / donor-map arithmetic (canonical direction, walk closure,
    state-kind eligibility, seed determinism);
(d) gate arithmetic (separation means over coherent draws, PASS at >=4/5,
    FAIL at 3/5, None never passes) + the mock-judge round trip through the
    REAL unit builders / validators / score writers / readers;
(e) resume regime keys (judge-mode + draws in the fingerprint; cross-regime
    done-records HARD-refuse);
(f) grid enumeration (150 families / 300 blocks / 1,500 cells; smoke slice);
(g) the additive judge extension (parent registry byte-unchanged; fp-conv2
    + gpu2 pairs unioned under the flag; grid item builder routes conv2 rows
    to both prefix rubrics; label mirrors pinned).

No network beyond the HF-cached Qwen tokenizer (the established
test_issue2094_* convention); no API calls (mock-judge only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2094_gpu2 as G2  # noqa: E402
import issue2094_gpu2_bank as G2B  # noqa: E402
import issue2094_judge as J  # noqa: E402
import issue2094_run as R  # noqa: E402

from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402

FMETRICS = REPO_ROOT / "eval_results/issue_2094/f_metrics"
SCORES = REPO_ROOT / "eval_results/issue_2094/judge/scores"


@pytest.fixture(scope="module")
def qwen_tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(R.MODEL_ID)


# ── (a) diagnosis pinned against the committed parent artifacts ─────────


def test_weak_pairs_pinned_against_committed_anchors():
    weak = G2.weak_matched_query_rows(FMETRICS / "anchors.jsonl")
    assert sorted(r["pair_id"] for r in weak) == sorted(G2B.WEAK_PAIR_IDS)
    # All five are the bare-vs-conv pairings on the prefix rubric kind.
    for r in weak:
        assert r["kind"] == "prefix"
        assert r["context_a"].startswith("bare__") and r["context_b"].startswith("conv__")
        assert r["separation"] is None or abs(r["separation"]) < G2.MIN_ABS_SEPARATION


def test_diagnosis_verdict_conv_prefix_attributable():
    diag = G2.run_diagnosis(FMETRICS / "anchors.jsonl", SCORES)
    assert diag["verdict"] == "conv-prefix-attributable"
    assert diag["queries_separate_against_persona"] is True
    # The mechanism proof: conv-generated anchor draws are judged
    # plain-assistant (fp-conv ~0, fp-bare high) — the register does not carry.
    conv = diag["anchor_score_means_by_gen_prefix"]["conv"]
    assert conv["fp-conv"] < 20.0 and conv["fp-bare"] > 50.0
    assert diag["conv_register_carries_into_answers"] is False


def test_diagnosis_escape_hatch_reframe_needed(tmp_path):
    """When the same queries ALSO fail their bare-vs-persona control, the weak
    separation is not attributable to the conv prefix -> reframe-needed."""
    rows = [json.loads(line) for line in (FMETRICS / "anchors.jsonl").open(encoding="utf-8")]
    for r in rows:
        if r["setting"] == "matched_query" and r["context_b"].startswith("persona__"):
            r["separation"] = 0.1  # queries no longer separate against persona
    synth = tmp_path / "anchors.jsonl"
    synth.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    diag = G2.run_diagnosis(synth, None)
    assert diag["queries_separate_against_persona"] is False
    assert diag["verdict"] == "reframe-needed"


def test_assert_diagnosis_check_passes_on_committed_tree():
    G2.assert_diagnosis_check(REPO_ROOT)


# ── (b) conv2 construction on the real tokenizer ────────────────────────


def test_conv2_contexts_parent_construction_shape():
    contexts = G2B.build_gpu2_contexts()
    assert sorted(contexts) == [f"conv2__q{i}" for i in range(1, 6)]
    for cid, ctx in contexts.items():
        assert ctx["prefix"] == "conv2" and ctx["id"] == cid
        assert ctx["system"] is None  # the parent conv construction: history-only
        assert [t["role"] for t in ctx["history"]] == ["user", "assistant"]
        assert ctx["user"] == BANK.QUERIES[ctx["query_id"]]
    ext = G2B.build_extended_contexts()
    assert len(ext) == 20 and set(BANK.build_contexts()) < set(ext)


def test_conv2_render_and_prefix_end_on_real_tokenizer(qwen_tok):
    contexts = G2B.build_gpu2_contexts()
    parent_conv = BANK.build_contexts()["conv__q1"]
    for cid, ctx in contexts.items():
        ids = BANK.context_token_ids_2094(qwen_tok, ctx)
        pe = BANK.prefix_end_index_multi(qwen_tok, ids)
        nq = len(ids) - pe
        assert nq >= R._QSPAN_MIN_POSITIONS, (cid, nq)
        rendered = BANK.render_context_2094(qwen_tok, ctx)
        # The multi-turn history actually renders (both turns present).
        assert G2B.CONV2_USER_TURN[:40] in rendered
        assert G2B.CONV2_ASSISTANT_TURN[:40] in rendered
        assert ctx["user"] in rendered
    # Same construction method as the parent conv prefix: no system turn there
    # either, and the same two-turn history shape.
    assert parent_conv["system"] is None
    assert [t["role"] for t in parent_conv["history"]] == ["user", "assistant"]


# ── (c) pair + donor-map arithmetic ─────────────────────────────────────


def test_gpu2_pairs_canonical_direction_and_disjoint_ids():
    pairs = G2B.build_gpu2_pairs()
    assert [p.pair_id for p in pairs] == [f"mq--bare__q{i}--conv2__q{i}" for i in range(1, 6)]
    for p in pairs:
        assert p.setting == "matched_query"
        assert p.prefix_a == "bare" and p.prefix_b == "conv2"
        assert p.query_a == p.query_b  # matched query
    parent_ids = {p.pair_id for p in BANK.build_pairs()}
    assert not ({p.pair_id for p in pairs} & parent_ids)


def test_gpu2_donor_map_closure_and_constraints():
    dm = G2B.gpu2_donor_map()
    pairs_by_id = G2B.gpu2_pairs_by_id()
    parent_mq = {p.pair_id for p in G2B.parent_mq_pairs()}
    recipients = [p.pair_id for p in G2B.build_gpu2_pairs()]
    assert set(dm) == set(pairs_by_id) and len(dm) == 20
    # Recipients draw DISTINCT donors from the parent mq pool; the parent
    # derangement rides along verbatim so the _resolve_donor walk + sorted
    # fallback stay well-defined.
    donors = [dm[r] for r in recipients]
    assert set(donors) <= parent_mq and len(set(donors)) == 5
    parent_der = BANK.donor_derangement(BANK.build_pairs())
    assert all(dm[pid] == parent_der[pid] for pid in parent_mq)
    assert all(r != d for r, d in dm.items())
    # State-kind eligibility (ce replace dose): no donor shares the
    # recipient's target context b (conv2__qK is in NO parent pair).
    for r in recipients:
        donor = pairs_by_id[dm[r]]
        assert R._donor_eligible(donor, "ce", pairs_by_id[r], "state")
        assert R._donor_eligible(donor, "ce", pairs_by_id[r], "delta")
    # Seed determinism + seed sensitivity.
    assert G2B.gpu2_donor_map() == dm
    assert G2B.gpu2_donor_map(seed=G2B.GPU2_SEED + 1) != dm


def test_gpu2_manifest_sha_covers_donor_seed():
    assert G2B.gpu2_manifest_sha() == G2B.gpu2_manifest_sha()
    assert G2B.gpu2_manifest_sha(seed=G2B.GPU2_SEED + 1) != G2B.gpu2_manifest_sha()


# ── (d) gate arithmetic + mock-judge round trip ─────────────────────────


def _lookups(scores_by_ctx_draw_rubric, coherence=95.0, incoherent=()):
    """(coh, beh, draws_by_ctx) from {(cid, draw, rid): score}."""
    coh, beh, draws = {}, {}, {}
    for (cid, d, rid), s in scores_by_ctx_draw_rubric.items():
        coh[(cid, d)] = 50.0 if (cid, d) in incoherent else coherence
        beh[(cid, d, rid)] = s
        draws.setdefault(cid, [])
        if d not in draws[cid]:
            draws[cid].append(d)
    return coh, beh, draws


def _synthetic_scores(seps_by_q: dict[str, float]) -> dict:
    """Two draws per side per pair, engineered so the pair separation equals
    ``seps_by_q[q]`` exactly (floor delta 0, ceiling delta = target)."""
    out = {}
    for q, target in seps_by_q.items():
        floor_cid, ceil_cid = f"bare__{q}", f"conv2__{q}"
        for d in (0, 1):
            out[(floor_cid, d, "fp-bare")] = 50.0
            out[(floor_cid, d, "fp-conv2")] = 50.0  # floor delta 0
            out[(ceil_cid, d, "fp-bare")] = 10.0
            out[(ceil_cid, d, "fp-conv2")] = 10.0 + 100.0 * target  # ceiling delta
    return out


def test_gate_separations_and_verdict_pass_at_4_of_5():
    seps = {"q1": 0.8, "q2": 0.7, "q3": 0.9, "q4": 0.6, "q5": 0.1}  # q5 below floor
    coh, beh, draws = _lookups(_synthetic_scores(seps))
    rows = G2.gate_separations(coh, beh, draws)
    by_pair = {r["pair_id"]: r for r in rows}
    for q, target in seps.items():
        got = by_pair[f"mq--bare__q{q[-1]}--conv2__q{q[-1]}"]["separation"]
        assert got == pytest.approx(target, abs=1e-9)
    verdict = G2.gate_verdict(rows)
    assert verdict["passed"] is True and verdict["n_passing"] == 4


def test_gate_verdict_fails_at_3_of_5_and_none_never_passes():
    seps = {"q1": 0.8, "q2": 0.7, "q3": 0.9, "q4": 0.2, "q5": 0.1}
    coh, beh, draws = _lookups(_synthetic_scores(seps))
    rows = G2.gate_separations(coh, beh, draws)
    verdict = G2.gate_verdict(rows)
    assert verdict["passed"] is False and verdict["n_passing"] == 3
    # A side with NO coherent scored draws -> separation None -> never passes.
    coh2 = dict.fromkeys(coh, 10.0)  # everything incoherent
    rows2 = G2.gate_separations(coh2, beh, draws)
    assert all(r["separation"] is None for r in rows2)
    v2 = G2.gate_verdict(rows2)
    assert v2["passed"] is False and v2["n_passing"] == 0


def test_gate_separation_excludes_incoherent_draws():
    seps = {"q1": 0.8, "q2": 0.8, "q3": 0.8, "q4": 0.8, "q5": 0.8}
    scores = _synthetic_scores(seps)
    # Poison ceiling draw 1 of q1 with an off-target score, then mark it
    # incoherent: the mean must ignore it (separation stays 0.8 exactly).
    scores[("conv2__q1", 1, "fp-conv2")] = 0.0
    coh, beh, draws = _lookups(scores, incoherent={("conv2__q1", 1)})
    rows = G2.gate_separations(coh, beh, draws)
    by_pair = {r["pair_id"]: r for r in rows}
    row = by_pair["mq--bare__q1--conv2__q1"]
    assert row["separation"] == pytest.approx(0.8, abs=1e-9)
    assert row["ceiling"]["n"] == 1 and row["ceiling"]["n_incoherent"] == 1


def test_negative_separation_passes_on_abs():
    """|sep| >= 0.5 is the restriction the round un-guts — sign-free."""
    seps = {"q1": -0.8, "q2": -0.7, "q3": -0.9, "q4": -0.6, "q5": -0.55}
    coh, beh, draws = _lookups(_synthetic_scores(seps))
    verdict = G2.gate_verdict(G2.gate_separations(coh, beh, draws))
    assert verdict["passed"] is True and verdict["n_passing"] == 5


@pytest.mark.parametrize("variant,expect_pass", [("mock-pass", True), ("mock-fail", False)])
def test_mock_judge_round_trip_through_real_writers(tmp_path, variant, expect_pass):
    """gate_units -> _validate_units -> mock scores rows -> load_gate_scores ->
    separations -> verdict, through the REAL builders/validators/writers."""
    paths = G2.GPU2Paths(out_root=tmp_path)
    rows = []
    for q in BANK.QUERY_ORDER:
        for cid in (f"bare__{q}", G2B.conv2_context_id(q)):
            for d in (0, 1):
                rows.append({"context_id": cid, "draw": d, "text": f"answer {cid} {d}"})
    G2.run_gate_judging(paths, rows, variant)
    coh, beh = G2.load_gate_scores(paths)
    assert len(coh) == 20 and len(beh) == 40
    draws_by_ctx: dict[str, list[int]] = {}
    for r in rows:
        draws_by_ctx.setdefault(r["context_id"], []).append(r["draw"])
    verdict = G2.gate_verdict(G2.gate_separations(coh, beh, draws_by_ctx))
    assert verdict["passed"] is expect_pass
    if variant == "mock-pass":
        assert all(p["passes"] for p in verdict["per_pair"])
        # mock-pass geometry: floor delta -0.85, ceiling delta 0.80 -> 1.65.
        assert verdict["per_pair"][0]["separation"] == pytest.approx(1.65, abs=1e-9)
    else:
        assert all(p["separation"] == pytest.approx(0.0) for p in verdict["per_pair"])


def test_gate_units_ids_validate_and_dedupe():
    rows = [
        {"context_id": f"bare__{q}", "draw": d, "text": "t"}
        for q in BANK.QUERY_ORDER
        for d in range(3)
    ]
    coh, beh = G2.gate_units(rows)
    J._validate_units(coh)
    for rid in G2.GATE_RUBRIC_IDS:
        J._validate_units(beh[rid])
    assert len(coh) == 15 and all(len(v) == 15 for v in beh.values())


# ── (e) resume regime keys ──────────────────────────────────────────────


def _tiny_cfg(tmp_path, **over):
    args = G2.parse_args(
        ["--run", "--tiny", "--out-root", str(tmp_path / "out"), "--smoke", "--judge", "mock-pass"]
    )
    cfg = G2.build_config(args)
    return cfg


def test_regime_fingerprint_covers_judge_mode_and_draws(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    fp_live = G2.gpu2_regime_fingerprint(cfg, "live", 10)
    fp_mock = G2.gpu2_regime_fingerprint(cfg, "mock-pass", 10)
    fp_draws = G2.gpu2_regime_fingerprint(cfg, "live", 2)
    assert len({fp_live, fp_mock, fp_draws}) == 3
    assert G2.gpu2_regime_fingerprint(cfg, "live", 10) == fp_live


def test_regime_checked_done_hard_refuses_cross_regime(tmp_path):
    path = tmp_path / "done.json"
    assert G2._regime_checked_done(path, "abc", "x") is None
    path.write_text(json.dumps({"regime_fp": "abc", "n": 1}), encoding="utf-8")
    assert G2._regime_checked_done(path, "abc", "x")["n"] == 1
    with pytest.raises(RuntimeError, match="refusing to resume across regimes"):
        G2._regime_checked_done(path, "OTHER", "x")


# ── (f) grid enumeration ────────────────────────────────────────────────


def test_gpu2_grid_totals_pinned():
    families = G2.enumerate_gpu2_families(28)
    assert R.grid_totals(families) == G2.EXPECTED_GPU2_TOTALS
    pair_ids = {p.pair_id for p in G2B.build_gpu2_pairs()}
    for steered, null in families:
        assert steered.slot == "ce" and steered.vec_type == "A"
        assert (steered.arm, null.arm) == ("steered", "null")
        assert set(steered.pair_ids) == pair_ids and steered.pair_ids == null.pair_ids
    variants = {f[0].layer_variant for f in families}
    assert variants == set(R.layer_variant_names(28))  # 28 single + 2 joint
    assert {f[0].dose for f in families} == set(R.DOSES_A)


def test_gpu2_smoke_slice_covers_arm_classes():
    families = G2.enumerate_gpu2_families(28)
    sliced = G2.slice_gpu2_smoke(families, 28)
    assert len(sliced) == 5
    combos = {(f[0].layer_variant, f[0].dose) for f in sliced}
    assert combos == set(G2.smoke_family_spec(28))
    # Arm classes: single-layer add, single-layer replace (state donor walk),
    # a second single-layer dose, joint_mid, joint_all replace.
    assert any(v == "joint_mid" for v, _ in combos)
    assert any(v == "joint_all" and d == "replace" for v, d in combos)
    # 5 families x (steered + null) x 5 pairs = 50 smoke cells.
    assert R.grid_totals(sliced)["cells_total"] == 50


# ── (g) the additive judge extension ────────────────────────────────────


def test_judge_registry_gpu2_additive_and_parent_unchanged():
    parent = J.rubric_registry()
    assert parent == J.rubric_registry(gpu2=False)
    ext = J.rubric_registry(gpu2=True)
    assert set(ext) == set(parent) | {"fp-conv2"}
    assert all(ext[k] == parent[k] for k in parent)  # parent ids byte-unchanged
    assert "{answer}" in ext["fp-conv2"]
    assert G2B.CONV2_DESCRIPTOR in ext["fp-conv2"]
    # Harness-identical substitution leaves no unfilled slot (rule 27).
    filled = ext["fp-conv2"].replace("{answer}", "some answer")
    assert "{answer}" not in filled and "{question}" not in filled


def test_judge_pair_index_gpu2_union():
    parent = J.pair_index()
    ext = J.pair_index(gpu2=True)
    assert set(ext) == set(parent) | {p.pair_id for p in G2B.build_gpu2_pairs()}
    assert all(ext[k] is not None and ext[k].pair_id == k for k in ext)


def test_grid_behavior_items_route_conv2_rows_to_both_prefix_rubrics():
    pairs = J.pair_index(gpu2=True)
    registry = J.rubric_registry(gpu2=True)
    row = {
        "block_key": "ce|L14|a1|A|steered",
        "slot": "ce",
        "layer_variant": "L14",
        "dose": "a1",
        "vec_type": "A",
        "arm": "steered",
        "pair_id": "mq--bare__q1--conv2__q1",
        "setting": "matched_query",
        "context_a": "bare__q1",
        "context_b": "conv2__q1",
        "cap_hit": False,
        "text": "hello",
    }
    by_rubric = J.build_grid_behavior_items([row], pairs)
    assert set(by_rubric) == {"fp-bare", "fp-conv2"}
    for rid, units in by_rubric.items():
        assert rid in registry
        J._validate_units(units)


def test_gate_registry_matches_gpu2_waves_instrument():
    """The pod-side gate and the VM-side --gpu2 waves judge under ONE
    instrument: identical rubric templates for the shared ids."""
    gate = G2.gate_rubric_registry()
    waves = J.rubric_registry(gpu2=True)
    assert set(gate) == {J.COHERENCE_RUBRIC_ID, "fp-bare", "fp-conv2"}
    for rid, prompt in gate.items():
        assert waves[rid] == prompt


def test_label_mirrors_pinned():
    assert J.GPU2_LABEL == G2.GPU2_LABEL
    assert J.HF_PREFIX == R.HF_PREFIX
    assert f"{R.HF_PREFIX}/raw_completions/{G2.GPU2_LABEL}" == G2.HF_GPU2_TEXT


# ── production-body coverage of smoke-fenced branches ───────────────────


def _synthetic_bank(contexts: list[str], n_layers=4, hidden=8, nq=3, seed=0) -> dict:
    import torch

    g = torch.Generator().manual_seed(seed)
    per = {}
    for cid in contexts:
        prefix, q = cid.split("__")
        per[cid] = {
            "context_id": cid,
            "prefix": prefix,
            "query_id": q,
            "ctx_len": 10,
            "prefix_end": 10 - nq,
            "nq": nq,
            "q_span": torch.randn((nq, n_layers, hidden), generator=g),
            "v_pe": torch.randn((n_layers, hidden), generator=g),
        }
    return {"layers": list(range(n_layers)), "per_context": per}


def test_gpu2_parity_report_filters_to_shared_and_gates():
    """The fenced production parity branch: the gpu2 bank's 15 SHARED contexts
    vs the parent bank (conv2 contexts have no parent reference and are
    excluded); identical tensors PASS, a layer-0 corruption FAILS."""
    import torch

    parent_ctx = sorted(BANK.build_contexts())
    parent_bank = _synthetic_bank(parent_ctx, seed=0)
    gpu2_ctx = parent_ctx + [G2B.conv2_context_id(q) for q in BANK.QUERY_ORDER]
    gpu2_bank = _synthetic_bank(gpu2_ctx, seed=1)
    for cid in parent_ctx:  # shared contexts byte-identical -> parity passes
        gpu2_bank["per_context"][cid] = {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in parent_bank["per_context"][cid].items()
        }
    rep = G2.gpu2_parity_report(gpu2_bank, parent_bank)
    assert rep["passed"] is True and len(rep["per_context"]) == 15
    # Corrupt layer 0 of one shared context -> the early bar catches it.
    gpu2_bank["per_context"][parent_ctx[0]]["q_span"][:, 0] *= -1.0
    rep2 = G2.gpu2_parity_report(gpu2_bank, parent_bank)
    assert rep2["passed"] is False and rep2["early_min_cos"] < 0.999


def test_fenced_live_call_sites_signature_bind(tmp_path):
    """Signature-bind the smoke-fenced LIVE calls (#1332 class): the gate's
    ``J.run_wave`` dispatch and the production staging helpers."""
    import inspect

    jc = G2.judge_config(G2.GPU2Paths(out_root=tmp_path))
    inspect.signature(J.run_wave).bind("fp-bare.gpu2anchors", "fp-bare", "prompt", [], jc)
    import issue2094_fu1 as FU1

    cfg = _tiny_cfg(tmp_path)
    inspect.signature(FU1.stage_bank).bind(cfg, None)
    inspect.signature(G2.stage_parent_anchors).bind(G2.GPU2Paths(out_root=tmp_path), None)
