"""CPU-only tests for the issue #2094 judge pipeline (unit D).

Four groups, per the unit brief:
(a) rule-27 parse-contract round-trips — REALISTIC judge replies through the
    harness's OWN parse+reduce path (``eval/utils.parse_judge_json`` →
    ``eval/graded_judge._score_from_parsed`` → ``judge_result_from_save_raw``),
    incl. fenced/markdown and truncated variants;
(b) production-rubric template placeholder checks (harness-identical
    substitution leaves no unfilled slot);
(c) item-builder invariants — per-setting rubric-pair counts, cross gets both
    pairs, coherence 1/rollout, anchor dedup, custom_id grammar/uniqueness
    through ``validate_batch_custom_ids``, and the FULL-grid call-arithmetic
    reconciliation against the plan §9 figures;
(d) mechanical-audit detectors on constructed texts.

Plus the phase-0 disjoint-halves cosine helper (the #1415 shared-baseline fix)
on a synthetic shared-noise construction. No network; no mocks in the parse
path — real strings through real parsers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2094_judge as J  # noqa: E402

from explore_persona_space.eval.graded_judge import (  # noqa: E402
    _score_from_parsed,
    judge_result_from_save_raw,
)
from explore_persona_space.eval.judge_dispatch import validate_batch_custom_ids  # noqa: E402
from explore_persona_space.eval.utils import parse_judge_json  # noqa: E402
from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402

N_LAYERS = 28
K_ANCHOR = 10


@pytest.fixture(scope="module")
def pairs() -> list[BANK.Pair]:
    return BANK.build_pairs()


@pytest.fixture(scope="module")
def pair_map(pairs) -> dict[str, BANK.Pair]:
    return {p.pair_id: p for p in pairs}


# ── (a) rule-27 parse-contract round-trips (REAL parsers, no mocks) ────


def test_parse_roundtrip_reason_then_score_json():
    reply = '{"reasoning": "Clear pirate register, ends with a sea warning.", "score": 87}'
    assert _score_from_parsed(parse_judge_json(reply)) == 87.0


def test_parse_roundtrip_fenced_markdown_variant():
    reply = (
        "Let me assess the response.\n\n"
        '```json\n{"reasoning": "fluent, well-formed", "score": 42}\n```\n'
    )
    assert _score_from_parsed(parse_judge_json(reply)) == 42.0


def test_parse_roundtrip_prose_preamble_then_json():
    reply = (
        "The response is fluent and internally consistent throughout. "
        'Final verdict: {"reasoning": "fluent", "score": 95}'
    )
    assert _score_from_parsed(parse_judge_json(reply)) == 95.0


def test_parse_roundtrip_truncated_reply_drops():
    # max_tokens truncation mid-rationale: parse fails -> None (drop, never coerce).
    reply = '{"reasoning": "The response begins coher'
    assert parse_judge_json(reply) is None
    assert _score_from_parsed(parse_judge_json(reply)) is None


def test_parse_roundtrip_refusal_and_out_of_range_drop():
    assert _score_from_parsed(parse_judge_json('{"score": "REFUSAL"}')) is None
    assert _score_from_parsed(parse_judge_json('{"score": 150}')) is None


def test_parse_roundtrip_bare_scalar_kept():
    assert _score_from_parsed(parse_judge_json("85")) == 85.0


def test_reduce_roundtrip_through_judge_result_from_save_raw(tmp_path):
    """The harness's OWN reduce (save_raw -> JudgeResult) over realistic parses."""
    items = [("i-keep", "q", "a"), ("i-drop", "q", "a"), ("i-refuse", "q", "a")]
    raw = {
        "all_scores": {
            "i-keep__00000__00": {"reasoning": "fluent", "score": 73, "stop_reason": "end_turn"},
            "i-drop__00000__00": {
                "error": True,
                "reason": "parse_error",
                "stop_reason": "max_tokens",
            },
            "i-refuse__00000__00": {"score": "REFUSAL", "stop_reason": "end_turn"},
        }
    }
    save_raw = tmp_path / "raw.json"
    save_raw.write_text(json.dumps(raw))
    result = judge_result_from_save_raw(save_raw, items)
    assert result.scores["i-keep"] == 73.0
    assert result.scores["i-drop"] is None
    assert result.scores["i-refuse"] is None
    assert result.n_dropped_draws == 2  # content drops (truncation + refusal subsets)
    assert result.n_truncation_dropped_draws == 1
    assert result.n_refusal_draws == 1
    assert result.n_transport_lost_draws == 0
    assert result.stop_reason_tally.get("max_tokens") == 1


# ── (b) production-rubric templates (harness-identical substitution) ───


def test_registry_has_nine_rubrics_with_answer_slot():
    reg = J.rubric_registry()
    assert set(reg) == {
        "coherence",
        "fq-q1",
        "fq-q2",
        "fq-q3",
        "fq-q4",
        "fq-q5",
        "fp-bare",
        "fp-persona",
        "fp-conv",
    }
    for rid, tmpl in reg.items():
        assert tmpl.count("{answer}") == 1, rid
        # {question} deliberately absent: F cores embed their target verbatim,
        # and the coherence rubric is form-only (relevance excluded).
        assert "{question}" not in tmpl, rid


def test_templates_substitute_with_no_unfilled_slots():
    q, a = "Why is the sky blue during the day but red at sunset?", "Because of scattering."
    for rid, tmpl in J.rubric_registry().items():
        # Harness-identical substitution (graded_judge.judge_graded format_user_msg).
        filled = tmpl.replace("{question}", q).replace("{answer}", a)
        assert "{answer}" not in filled and "{question}" not in filled, rid
        assert a in filled, rid


def test_coherence_template_carries_verbatim_truncation_clause():
    assert BANK.TRUNCATION_CLAUSE in J.rubric_registry()["coherence"]


def test_behavior_templates_carry_bank_cores_verbatim():
    reg = J.rubric_registry()
    for q in BANK.QUERY_ORDER:
        assert BANK.f_query_rubric(q) in reg[f"fq-{q}"]
    for p in BANK.PREFIX_ORDER:
        assert BANK.f_prefix_rubric(p) in reg[f"fp-{p}"]


def test_hf_prefix_matches_pod_driver():
    import issue2094_run as R

    assert J.HF_PREFIX == R.HF_PREFIX


# ── (c) item-builder invariants ────────────────────────────────────────


def _grid_row(pair: BANK.Pair, block_key: str = "ce|L14|a1|A|steered", arm: str = "steered"):
    return {
        "block_key": block_key,
        "slot": block_key.split("|")[0],
        "layer_variant": block_key.split("|")[1],
        "dose": block_key.split("|")[2],
        "vec_type": block_key.split("|")[3],
        "arm": arm,
        "pair_id": pair.pair_id,
        "setting": pair.setting,
        "context_a": pair.a,
        "context_b": pair.b,
        "cap_hit": False,
        "text": f"answer text for {pair.pair_id}",
    }


def test_per_setting_rubric_pair_counts(pairs, pair_map):
    mp = next(p for p in pairs if p.setting == "matched_prefix")
    mq = next(p for p in pairs if p.setting == "matched_query")
    x = next(p for p in pairs if p.setting == "cross")
    rows = [_grid_row(mp), _grid_row(mq), _grid_row(x)]
    by_rubric = J.build_grid_behavior_items(rows, pair_map)
    units = [u for us in by_rubric.values() for u in us]
    per_pair = {}
    for u in units:
        per_pair.setdefault(u.source["pair_id"], []).append(u)
    # matched-prefix -> the F_query pair (2 calls); matched-query -> F_prefix
    # pair (2); cross -> BOTH pairs on the same draw (4). (plan §4.4)
    assert len(per_pair[mp.pair_id]) == 2
    assert {u.rubric_id for u in per_pair[mp.pair_id]} == {
        f"fq-{mp.query_a}",
        f"fq-{mp.query_b}",
    }
    assert len(per_pair[mq.pair_id]) == 2
    assert {u.rubric_id for u in per_pair[mq.pair_id]} == {
        f"fp-{mq.prefix_a}",
        f"fp-{mq.prefix_b}",
    }
    assert len(per_pair[x.pair_id]) == 4
    assert {u.rubric_id for u in per_pair[x.pair_id]} == {
        f"fq-{x.query_a}",
        f"fq-{x.query_b}",
        f"fp-{x.prefix_a}",
        f"fp-{x.prefix_b}",
    }


def test_coherence_one_item_per_rollout(pairs, pair_map):
    rows = [_grid_row(p) for p in pairs[:5]]
    anchors = [
        {"context_id": "bare__q1", "draw": d, "text": f"anchor {d}"} for d in range(K_ANCHOR)
    ]
    units = J.build_coherence_items(rows, anchors)
    assert len(units) == len(rows) + len(anchors)
    assert len({u.item_id for u in units}) == len(units)
    assert all(u.rubric_id == "coherence" for u in units)


def test_anchor_dedup_across_pairs(pair_map):
    """Two pairs sharing a context must not duplicate (context, draw, rubric)."""
    anchors = [{"context_id": "bare__q1", "draw": 0, "text": "t"}]
    by_rubric = J.build_anchor_behavior_items(anchors, pair_map)
    ids = [u.item_id for us in by_rubric.values() for u in us]
    assert len(ids) == len(set(ids))
    # bare__q1 accumulates exactly 8 rubrics: all 5 fq (mp partners) + 3 fp (mq).
    assert len(ids) == 8
    rubrics = {u.rubric_id for us in by_rubric.values() for u in us}
    assert rubrics == {
        "fq-q1",
        "fq-q2",
        "fq-q3",
        "fq-q4",
        "fq-q5",
        "fp-bare",
        "fp-persona",
        "fp-conv",
    }


def test_item_ids_pass_batch_custom_id_grammar(pairs, pair_map):
    rows = [_grid_row(p) for p in pairs]
    anchors = [
        {"context_id": BANK.context_id(pre, q), "draw": d, "text": "t"}
        for pre in BANK.PREFIX_ORDER
        for q in BANK.QUERY_ORDER
        for d in range(2)
    ]
    units = J.build_coherence_items(rows, anchors)
    for us in J.build_grid_behavior_items(rows, pair_map).values():
        units.extend(us)
    for us in J.build_anchor_behavior_items(anchors, pair_map).values():
        units.extend(us)
    ids = [u.item_id for u in units]
    assert len(set(ids)) == len(ids)
    for i in ids:
        assert J.ITEM_ID_RE.match(i), i
        assert "__" not in i, i
        assert len(i) <= 53, i
    validate_batch_custom_ids(ids)  # the #1776 pre-flight, zero network
    J._validate_units(units)


def test_full_grid_call_arithmetic_reconciles_with_plan(pairs, pair_map):
    """Realized judge-call arithmetic == plan §9 (one synthetic row per grid cell).

    Grid: 42,000 rollouts -> 42,000 coherence + 103,200 behavior calls
    (38,400 mp + 26,400 mq + 38,400 cross). Anchors: 150 rollouts -> 150
    coherence + 1,200 DEDUPED behavior calls (15 contexts x 10 draws x 8
    rubrics; the plan's pair-expanded figure is 3,000 — same information).
    """
    import issue2094_run as R

    fams = R.enumerate_block_families(pairs, N_LAYERS)
    rows = []
    for steered, null in fams:
        for block in (steered, null):
            for pid in block.pair_ids:
                p = pair_map[pid]
                rows.append(_grid_row(p, block_key=block.key, arm=block.arm))
    assert len(rows) == 42_000

    coh = J.build_coherence_items(rows, None)
    assert len(coh) == 42_000

    by_rubric = J.build_grid_behavior_items(rows, pair_map)
    units = [u for us in by_rubric.values() for u in us]
    assert len(units) == 103_200
    per_setting = {"matched_prefix": 0, "matched_query": 0, "cross": 0}
    for u in units:
        per_setting[u.source["setting"]] += 1
    assert per_setting == {"matched_prefix": 38_400, "matched_query": 26_400, "cross": 38_400}

    anchors = [
        {"context_id": BANK.context_id(pre, q), "draw": d, "text": "t"}
        for pre in BANK.PREFIX_ORDER
        for q in BANK.QUERY_ORDER
        for d in range(K_ANCHOR)
    ]
    anchor_units = [
        u for us in J.build_anchor_behavior_items(anchors, pair_map).values() for u in us
    ]
    assert len(anchor_units) == 1_200
    assert len(J.build_coherence_items(None, anchors)) == 150

    # Ids unique across the WHOLE production item set (grid + anchors + coherence).
    all_ids = [u.item_id for u in coh + units + anchor_units]
    all_ids += [u.item_id for u in J.build_coherence_items(None, anchors)]
    assert len(set(all_ids)) == len(all_ids) == 42_000 + 103_200 + 1_200 + 150


def test_wave_regime_mismatch_refuses(tmp_path, pairs, pair_map):
    cfg = J.JudgeConfig(
        work_root=tmp_path / "w",
        cache_root=tmp_path / "c",
        rollouts_dir=tmp_path,
        anchors_file=tmp_path / "anchors.jsonl",
        stage2_dir=None,
    )
    p = pairs[0]
    units = [
        u for us in J.build_grid_behavior_items([_grid_row(p)], pair_map).values() for u in us
    ][:1]
    rid = units[0].rubric_id
    prompt = J.rubric_registry()[rid]
    regime = J.wave_regime("w1", rid, prompt, units, cfg)
    meta_path = tmp_path / "w1.meta.json"
    meta_path.write_text(json.dumps({"regime": regime, "complete": True}))
    assert J._wave_skip_state(meta_path, regime) == "skip"
    other = dict(regime, max_tokens=2048)
    with pytest.raises(RuntimeError, match="DIFFERENT"):
        J._wave_skip_state(meta_path, other)


# ── (d) mechanical audits ──────────────────────────────────────────────


def test_audit_empty_output():
    a = J.audit_text("   \n\t ")
    assert a["flag_empty"] and a["empty"]
    assert not a["flag_script_intrusion"] and not a["flag_repetition"]


def test_audit_script_intrusion_on_cjk_text():
    # Constructed CJK run (built via chr() — never literal escapes, Edit-tool
    # \\uXXXX un-escaping gotcha) inside an otherwise-English sentence.
    cjk = "".join(chr(0x4E2D + i) for i in range(40))
    a = J.audit_text(f"The answer is {cjk} as follows.")
    assert a["flag_script_intrusion"]
    assert a["nonlatin_letter_frac"] > J.AUDIT_NONLATIN_FRAC_MAX
    clean = J.audit_text("A perfectly ordinary English sentence about lighthouses.")
    assert not clean["flag_script_intrusion"]
    assert clean["nonlatin_letter_frac"] == 0.0


def test_audit_degenerate_repetition():
    # A k-word LOOP splits mass across k rotated 4-grams (max-single frac ~1/k);
    # the duplicate-4gram fraction is the loop-robust flag basis.
    a = J.audit_text("the cat sat here " * 40)
    assert a["flag_repetition"]
    assert a["dup_4gram_frac"] > J.AUDIT_DUP_4GRAM_FRAC_MAX
    assert a["max_repeated_4gram_frac"] < 0.30  # the naive metric misses the loop
    single = J.audit_text("word " * 100)  # 1-word loop: BOTH metrics saturate
    assert single["flag_repetition"] and single["max_repeated_4gram_frac"] == 1.0
    clean = J.audit_text(
        "Rent versus buy depends on horizon, rates, maintenance costs, "
        "and how long you plan to stay in one place overall."
    )
    assert not clean["flag_repetition"]
    assert clean["dup_4gram_frac"] == 0.0


def test_audit_short_text_not_degenerate():
    a = J.audit_text("Thanks.")
    assert a["max_repeated_4gram_frac"] == 0.0
    assert not a["flag_empty"]


# ── coherence-baseline gate (plan §7 gate 3) ───────────────────────────


def test_coherence_gate_pass_and_fail():
    good = [{"score": 90.0}] * 95 + [{"score": 50.0}] * 5
    g = J.coherence_baseline_gate(good)
    assert g["passed"] and g["median"] == 90.0 and g["frac_gt60"] == 0.95
    bad = [{"score": 90.0}] * 80 + [{"score": 40.0}] * 20
    assert not J.coherence_baseline_gate(bad)["passed"]
    dropped = [{"score": None}] * 3 + [{"score": 95.0}] * 10
    g3 = J.coherence_baseline_gate(dropped)
    assert g3["n_dropped"] == 3 and g3["n_kept"] == 10


# ── phase-0 disjoint-halves cosine (the #1415 shared-baseline fix) ─────


def test_phase0_disjoint_cosines_kill_shared_floor_inflation():
    """True shift ~0 + noisy shared floor: shared-floor cosines inflate toward 1,
    disjoint-halves cosines stay near 0 (the #1415 recount mechanism)."""
    import issue2094_phase0 as P

    torch.manual_seed(0)
    h, k, a = 512, 10, 4
    floor_draws = torch.randn(k, h)  # large floor noise
    floor_mean = floor_draws.mean(dim=0)
    # Steered means sit at the TRUE floor (zero true shift) + tiny independent noise.
    steered = floor_mean.new_zeros((a, h)) + 0.01 * torch.randn(a, h)

    shared = torch.tensor(
        [
            [
                torch.nn.functional.cosine_similarity(
                    steered[i] - floor_mean, steered[j] - floor_mean, dim=0
                )
                for j in range(a)
            ]
            for i in range(a)
        ]
    )
    disjoint = P.pairwise_disjoint_cosines(steered, floor_draws)
    off = ~torch.eye(a, dtype=torch.bool)
    assert shared[off].mean() > 0.5  # shared-floor noise manufactures alignment
    assert disjoint[off].abs().mean() < 0.2  # disjoint halves remove the shared term


def test_phase0_disjoint_cosines_recover_true_alignment():
    """A real common direction survives the disjoint convention."""
    import issue2094_phase0 as P

    torch.manual_seed(1)
    h, k, a = 512, 10, 4
    direction = torch.randn(h)
    direction /= direction.norm()
    # Floor noise small vs the weakest shift norm (0.5), so the disjoint read's
    # legitimate attenuation (each half carries independent noise) stays tiny.
    floor_draws = 0.005 * torch.randn(k, h)
    alphas = torch.tensor([0.5, 1.0, 2.0, 4.0])
    steered = alphas.unsqueeze(1) * direction + 0.001 * torch.randn(a, h)
    disjoint = P.pairwise_disjoint_cosines(steered, floor_draws)
    off = ~torch.eye(a, dtype=torch.bool)
    assert disjoint[off].min() > 0.95
