"""Issue #2162 persona-specificity ladder — bank + judge-driver CPU pins (no API).

Covers (plan v6 P0 tests): the 7-value ladder registry shape, the SEEDED
WildChat carrier-subset determinism (pinned to the realized draw), the
72-pair direction grammar, BOTH donor-plan constraint sets (same-value cyclic
carrier shift; cross-type donors outside the persona family, two types per
pair), the descriptor freeze / Round-A HOLISTIC instrument parity (byte-
identical to ``issue2162_persona_rubric_rescore``), the rule-27
parse-contract round-trip for the NEW rubric texts (realistic reason+score
reply + fenced variant through the harness's OWN ``parse_judge_json`` →
``_score_from_parsed`` path, plus ``{answer}`` placeholder presence and
harness-identical substitution), the plain-render equality probe SHAPE, and
the VM judge driver's consumer-contract shapes (separation verdict →
``issue2162_ladder.read_gate_verdict``; donor units; pool builder →
``issue2162_run.load_pools`` / the margin phase's ``pair.cell`` keying).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2162_ladder_judge as LJ  # noqa: E402
import issue2162_persona_rubric_rescore as RESCORE  # noqa: E402

from explore_persona_space.eval.graded_judge import _score_from_parsed  # noqa: E402
from explore_persona_space.eval.utils import parse_judge_json  # noqa: E402
from explore_persona_space.experiments.issue2162 import ladder_bank as LB  # noqa: E402

# ── registry shape ────────────────────────────────────────────────────


def test_ladder_values_shape():
    assert len(LB.LADDER_VALUES) == 7
    by_id = LB.VALUES_BY_ID
    assert by_id["plain"].rung_rank == 0
    # rung_rank encodes specificity: R1=5 ... R5=1 (plan §4.1).
    assert by_id["r1_pirate"].rung_rank == 5
    assert by_id["r2_butler"].rung_rank == 4
    assert by_id["r3_warm"].rung_rank == 3
    assert by_id["r4_trait"].rung_rank == 2
    assert by_id["r5a_lu_therapy"].rung_rank == 1
    assert by_id["r5b_lu_philosophy"].rung_rank == 1
    assert LB.PERSONA_VALUE_IDS == (
        "r1_pirate",
        "r2_butler",
        "r3_warm",
        "r4_trait",
        "r5a_lu_therapy",
        "r5b_lu_philosophy",
    )
    assert set(LB.DESCRIPTORS) == {"plain", *LB.PERSONA_VALUE_IDS}


def test_carrier_subset_determinism_pinned():
    # The realized seed-2162 WildChat draw (plan §4.1) — pinned so a seed or
    # sampler change is test-breaking, never silent.
    assert LB.wildchat_carrier_ids(LB.SEED) == ("n3", "n4", "n7", "n9")
    assert LB.carrier_ids(LB.SEED) == ("d1", "d2", "n3", "n4", "n7", "n9")
    # Deterministic across calls.
    assert LB.wildchat_carrier_ids(LB.SEED) == LB.wildchat_carrier_ids(LB.SEED)


def test_direction_ids_and_pair_grammar():
    dirs = LB.direction_ids()
    assert len(dirs) == 12
    pairs = LB.build_ladder_pairs(LB.SEED)
    assert len(pairs) == 72
    assert {p.cell for p in pairs} == set(dirs)
    for p in pairs:
        # a/b are the contexts of value_a/value_b on the pair's carrier.
        assert p.a == LB.context_id(p.value_a, p.carrier)
        assert p.b == LB.context_id(p.value_b, p.carrier)
        if p.kind == "install":
            assert (p.value_a, p.value_b) == ("plain", p.persona)
        else:
            assert p.kind == "erase"
            assert (p.value_a, p.value_b) == (p.persona, "plain")


# ── donor plans (plan §4.2) ───────────────────────────────────────────


def test_sameval_donor_carrier_constraints():
    order = LB.sameval_donor_order(LB.SEED)
    assert sorted(order) == sorted(LB.carrier_ids(LB.SEED))
    assert order == LB.sameval_donor_order(LB.SEED)  # frozen / deterministic
    all_carriers = LB.carrier_ids(LB.SEED)
    for c in all_carriers:
        donor = LB.sameval_donor_carrier(c, all_carriers)
        assert donor != c and donor in all_carriers
        # Deterministic given the survivor set.
        assert donor == LB.sameval_donor_carrier(c, all_carriers)
    # Survivor-subset resolution: donor always drawn FROM the survivors.
    survivors = ["d1", "n3", "n7"]
    for c in survivors:
        donor = LB.sameval_donor_carrier(c, survivors)
        assert donor in survivors and donor != c
    # A single-survivor slice cannot field a same-value donor — fail loud.
    with pytest.raises(RuntimeError, match="single-survivor"):
        LB.sameval_donor_carrier("d1", ["d1"])


def _synthetic_parent_pairs() -> list[dict]:
    rows = []
    for cell in LB.CROSSTYPE_DONOR_TYPES:
        for carrier in ("n1", "n3", "n4", "n7", "n9"):  # WildChat overlap + one extra
            rows.append(
                {
                    "pair_id": f"{cell}::{carrier}",
                    "cell": cell,
                    "carrier": carrier,
                    "value_a": f"{cell}_a",
                    "value_b": f"{cell}_b",
                    "a": f"parent::{cell}_a::{carrier}",
                    "b": f"parent::{cell}_b::{carrier}",
                }
            )
    return rows


def test_crosstype_donor_plan_constraints():
    parent_pairs = _synthetic_parent_pairs()
    plan = LB.crosstype_donor_plan(parent_pairs, LB.SEED)
    assert len(plan) == 72
    assert set(plan) == {p.pair_id for p in LB.build_ladder_pairs(LB.SEED)}
    pairs_by_id = {p.pair_id: p for p in LB.build_ladder_pairs(LB.SEED)}
    for pid, rec in plan.items():
        primary, alternate = rec["primary"], rec["alternate"]
        # Donors come from the TWO non-persona types, one each (plan §4.2).
        assert {primary["cell"], alternate["cell"]} == set(LB.CROSSTYPE_DONOR_TYPES)
        for donor in (primary, alternate):
            assert donor["cell"] not in LB.PERSONA_VALUE_IDS and donor["cell"] != "plain"
            assert set(donor) == {"pair_id", "cell", "carrier", "value_a", "value_b", "a", "b"}
            # Carrier-matched whenever the ladder carrier exists in the donor cell.
            if pairs_by_id[pid].carrier in ("n3", "n4", "n7", "n9"):
                assert donor["carrier"] == pairs_by_id[pid].carrier
    # Frozen: same seed ⇒ identical plan.
    assert plan == LB.crosstype_donor_plan(parent_pairs, LB.SEED)


# ── descriptor freeze / Round-A instrument parity ─────────────────────


def test_descriptor_round_a_parity():
    # The ladder descriptors for the two Round-A personas are FROZEN to the
    # rescore instrument's own HOLISTIC descriptors (plan §4.4).
    assert LB.DESCRIPTORS["r1_pirate"] == RESCORE.HOLISTIC["v1"]
    assert LB.DESCRIPTORS["r2_butler"] == RESCORE.HOLISTIC["v3"]
    assert LB.DESCRIPTORS["plain"] == RESCORE.HOLISTIC["v2"]
    # Instrument template parity: byte-identical wrapper + tail.
    assert LB.REASON_THEN_SCORE == RESCORE.REASON_THEN_SCORE
    for descriptor in LB.DESCRIPTORS.values():
        assert LB.holistic_eval_prompt(descriptor) == RESCORE.eval_prompt(descriptor)


def test_rubric_registry_shape_and_placeholders():
    reg = LB.rubric_registry()
    assert set(reg) == {f"hol-{v}" for v in ("plain", *LB.PERSONA_VALUE_IDS)}
    for rid, prompt in reg.items():
        assert prompt.count("{answer}") == 1, rid
        assert "{question}" not in prompt, rid


# ── rule-27 round-trip (the NEW rubric texts) ─────────────────────────

REALISTIC_REPLY = (
    "The response is drenched in pirate dialect, with constant sea references "
    "and a closing warning about the ocean.\n"
    '{"reasoning": "Strong, consistent persona expression throughout.", "score": 87}'
)
FENCED_REPLY = (
    "Brief reasoning first: partial expression only.\n"
    '```json\n{"reasoning": "Mixed register; a few persona markers.", "score": 42}\n```'
)


@pytest.mark.parametrize(("reply", "expected"), [(REALISTIC_REPLY, 87.0), (FENCED_REPLY, 42.0)])
def test_rule27_round_trip_reason_then_score(reply, expected):
    parsed = parse_judge_json(reply)
    assert _score_from_parsed(parsed) == expected


def test_rule27_substitution_all_ladder_rubrics():
    # Harness-identical substitution (graded_judge.format_user_msg's .replace
    # chain): every ladder rubric — holistic AND conjunct — fills {answer}
    # and leaves no unfilled slot.
    answer = "Arr, the sea be a fickle mistress, savvy?"
    for rid, prompt in {**LJ.ladder_registry(), **LJ.conjunct_registry()}.items():
        assert "{answer}" in prompt, rid
        user_msg = prompt.replace("{question}", "").replace("{answer}", answer)
        assert answer in user_msg, rid
        assert "{answer}" not in user_msg and "{question}" not in user_msg, rid


def test_conjunct_registry_round_a_parity():
    # r1_pirate -> Round-A v1 conjuncts; r2_butler -> v3 (plan §4.4).
    assert set(LJ.LADDER_CONJUNCTS) == {"r1_pirate", "r2_butler"}
    assert LJ.LADDER_CONJUNCTS["r1_pirate"] == RESCORE.CONJUNCTS["v1"]
    assert LJ.LADDER_CONJUNCTS["r2_butler"] == RESCORE.CONJUNCTS["v3"]
    reg = LJ.conjunct_registry()
    assert set(reg) == {
        *(f"conj-r1_pirate-{k}" for k in RESCORE.CONJUNCTS["v1"]),
        *(f"conj-r2_butler-{k}" for k in RESCORE.CONJUNCTS["v3"]),
    }
    for key, clause in RESCORE.CONJUNCTS["v1"].items():
        assert reg[f"conj-r1_pirate-{key}"] == LB.holistic_eval_prompt(clause)


# ── plain-render equality probe (shape) ───────────────────────────────


class _StubTok:
    """Signature-conformant chat-template stub (external tokenizer boundary).

    ``insert_default=True`` mimics Qwen's template (a missing system turn gets
    the model default — the equal=True branch); ``False`` renders messages
    verbatim (the equal=False / positive-delta branch).
    """

    def __init__(self, insert_default: bool):
        self.insert_default = insert_default

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert not tokenize
        msgs = list(messages)
        if self.insert_default and (not msgs or msgs[0]["role"] != "system"):
            msgs = [{"role": "system", "content": LB.PLAIN_SYSTEM}, *msgs]
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": [ord(c) for c in text]}


@pytest.mark.parametrize("insert_default", [True, False])
def test_plain_render_equality_shape(insert_default):
    rec = LB.plain_render_equality(_StubTok(insert_default))
    assert set(rec) == {
        "equal",
        "n_tokens_explicit",
        "n_tokens_omitted",
        "token_delta",
        "plain_system",
    }
    assert rec["equal"] is insert_default
    assert rec["token_delta"] == rec["n_tokens_explicit"] - rec["n_tokens_omitted"]
    if not insert_default:
        assert rec["token_delta"] > 0  # the explicit system block adds tokens
    assert rec["plain_system"] == LB.PLAIN_SYSTEM


# ── judge-driver shapes (consumer contracts) ──────────────────────────

N_DRAWS = 4  # synthetic anchors: 7 values x 6 carriers x N_DRAWS


def _synthetic_anchor_rows() -> list[dict]:
    rows = []
    for v in ("plain", *LB.PERSONA_VALUE_IDS):
        for carrier in LB.carrier_ids(LB.SEED):
            for draw in range(N_DRAWS):
                rows.append(
                    {
                        "context_id": LB.context_id(v, carrier),
                        "cell": "ladder",
                        "value_id": v,
                        "rung": LB.VALUES_BY_ID[v].rung,
                        "carrier": carrier,
                        "draw": draw,
                        "text": f"synthetic answer {v}/{carrier}/{draw}",
                    }
                )
    return rows


def _separating_scores(anchor_rows) -> dict[str, dict[tuple[str, int], float]]:
    """hol-X: 90 on X's own contexts, 5 on plain; hol-plain: 10 on persona
    contexts, 95 on plain — clears both §7 bars everywhere."""
    scores: dict[str, dict[tuple[str, int], float]] = {}
    for row in anchor_rows:
        key = (row["context_id"], row["draw"])
        v = row["value_id"]
        scores.setdefault("hol-plain", {})[key] = 95.0 if v == "plain" else 10.0
        for x in LB.PERSONA_VALUE_IDS:
            scores.setdefault(f"hol-{x}", {})[key] = (
                90.0 if v == x else (5.0 if v == "plain" else 0.0)
            )
    return scores


def test_separation_verdict_pass_shape():
    anchor_rows = _synthetic_anchor_rows()
    verdict = LJ.separation_verdict(anchor_rows, _separating_scores(anchor_rows))
    assert verdict["passed"] and not verdict["all_rungs_failed"]
    rungs = verdict["rungs"]
    assert set(rungs) == set(LB.PERSONA_VALUE_IDS)
    for value_id, rec in rungs.items():
        # The exact fields issue2162_ladder.read_gate_verdict consumes.
        assert rec["survived"] is True, value_id
        assert isinstance(rec["surviving_carriers"], list) and rec["surviving_carriers"]
        assert rec["surviving_carriers"] == sorted(LB.carrier_ids(LB.SEED))
        for carrier_rec in rec["per_carrier"].values():
            assert carrier_rec["passed"] and not carrier_rec["unscored"]
            assert carrier_rec["target_sep"] == pytest.approx(0.85)
            assert carrier_rec["netted_sep"] == pytest.approx(1.7)


def test_separation_verdict_all_fail_and_partial():
    anchor_rows = _synthetic_anchor_rows()
    # Uniform scores: zero separation everywhere -> rig-defect HALT branch.
    flat = {
        rid: {(r["context_id"], r["draw"]): 50.0 for r in anchor_rows}
        for rid in ("hol-plain", *(f"hol-{v}" for v in LB.PERSONA_VALUE_IDS))
    }
    verdict = LJ.separation_verdict(anchor_rows, flat)
    assert verdict["all_rungs_failed"] and not verdict["passed"]
    assert all(not rec["survived"] for rec in verdict["rungs"].values())

    # Partial: r1_pirate separates on only 3 carriers (< RUNG_MIN_CARRIERS=4)
    # -> dropped; r2_butler on all 6 -> survives.
    scores = _separating_scores(anchor_rows)
    dead_carriers = ("n4", "n7", "n9")
    for row in anchor_rows:
        if row["value_id"] == "r1_pirate" and row["carrier"] in dead_carriers:
            scores["hol-r1_pirate"][(row["context_id"], row["draw"])] = 10.0
    verdict = LJ.separation_verdict(anchor_rows, scores)
    assert not verdict["rungs"]["r1_pirate"]["survived"]
    assert verdict["rungs"]["r1_pirate"]["n_carriers_pass"] == 3
    assert verdict["rungs"]["r2_butler"]["survived"]
    assert not verdict["all_rungs_failed"] and verdict["passed"]

    # Unscored carrier (all draws dropped) fails that carrier, never passes it.
    scores2 = _separating_scores(anchor_rows)
    ceil_ctx = LB.context_id("r3_warm", "d1")
    for draw in range(N_DRAWS):
        scores2["hol-r3_warm"].pop((ceil_ctx, draw))
    verdict2 = LJ.separation_verdict(anchor_rows, scores2)
    d1 = verdict2["rungs"]["r3_warm"]["per_carrier"]["d1"]
    assert d1["unscored"] and not d1["passed"]
    assert verdict2["rungs"]["r3_warm"]["n_carriers_pass"] == 5


def test_gate_behavior_item_arithmetic():
    anchor_rows = _synthetic_anchor_rows()
    by_rid = LJ.build_gate_behavior_items(anchor_rows)
    n_all = 7 * 6 * N_DRAWS
    assert len(by_rid["hol-plain"]) == n_all
    for v in LB.PERSONA_VALUE_IDS:
        # Own (ceiling) + plain (floor) contexts only.
        assert len(by_rid[f"hol-{v}"]) == 2 * 6 * N_DRAWS
    ids = [u.item_id for us in by_rid.values() for u in us]
    assert len(set(ids)) == len(ids)


def _synthetic_grid_rows() -> list[dict]:
    rows = []
    for pair in LB.build_ladder_pairs(LB.SEED)[:6]:  # install/erase r1 across carriers
        for arm in ("steered", "null_sameval"):
            for slot in ("ce", "pe"):
                rows.append(
                    {
                        "block_key": f"{pair.pair_id}::{slot}",
                        "cell": pair.cell,
                        "slot": slot,
                        "arm": arm,
                        "pair_id": pair.pair_id,
                        "persona": pair.persona,
                        "kind": pair.kind,
                        "carrier": pair.carrier,
                        "draw": 0,
                        "context_id": pair.b,
                        "text": f"grid answer {pair.pair_id}/{slot}/{arm}",
                    }
                )
    return rows


def test_grid_behavior_items_two_rubrics_per_row():
    rows = _synthetic_grid_rows()
    by_rid = LJ.build_grid_behavior_items(rows)
    # Every row is judged under hol-plain + its direction's own persona rubric.
    assert sum(len(us) for us in by_rid.values()) == 2 * len(rows)
    assert len(by_rid["hol-plain"]) == len(rows)
    assert len(by_rid["hol-r1_pirate"]) == len(rows)


def test_conjunct_items_steered_r1_r2_only():
    rows = _synthetic_grid_rows()  # r1_pirate rows, half steered
    by_rid = LJ.build_conjunct_items(rows)
    n_steered = sum(1 for r in rows if r["arm"] == "steered")
    assert set(by_rid) == {f"conj-r1_pirate-{k}" for k in RESCORE.CONJUNCTS["v1"]}
    for us in by_rid.values():
        assert len(us) == n_steered
    # Non-R1/R2 personas contribute nothing.
    warm_rows = [dict(r, persona="r3_warm") for r in rows]
    assert LJ.build_conjunct_items(warm_rows) == {}


def test_donor_units_dedup_and_fail_loud():
    draws_by_ctx = {"parent::x::n3": [0, 1]}
    texts = {("parent::x::n3", 0): "t0", ("parent::x::n3", 1): "t1"}
    cands = [("parent::x::n3", "hol-r1_pirate"), ("parent::x::n3", "hol-r1_pirate")]
    by_rid = LJ._donor_units(cands, draws_by_ctx, texts)
    assert len(by_rid["hol-r1_pirate"]) == 2  # deduped (ctx, rid); 2 draws
    with pytest.raises(RuntimeError, match="NO parent anchor rows"):
        LJ._donor_units([("parent::missing::n1", "hol-r1_pirate")], draws_by_ctx, texts)


def test_build_ladder_pools_consumer_shape():
    anchor_rows = _synthetic_anchor_rows()
    # Own-descriptor scores >50 ONLY for plain + r1_pirate contexts; graded by
    # draw so the top-4 selection is checkable.
    scores: dict[str, dict[tuple[str, int], float]] = {"hol-plain": {}, "hol-r1_pirate": {}}
    for row in anchor_rows:
        key = (row["context_id"], row["draw"])
        if row["value_id"] == "plain":
            scores["hol-plain"][key] = 60.0 + row["draw"]  # 60..63
        elif row["value_id"] == "r1_pirate":
            scores["hol-r1_pirate"][key] = 80.0 + row["draw"]  # 80..83
    pools, report = LJ.build_ladder_pools(anchor_rows, scores)
    # Only r1's two directions have BOTH sides scored above the filter.
    assert set(pools) == {"install_r1_pirate", "erase_r1_pirate"}
    assert report["n_directions_built"] == 2
    assert report["n_directions_total"] == 12
    assert report["per_direction"]["install_r2_butler"]["omitted"] is True
    for direction, items in pools.items():
        # The exact fail-loud schema issue2162_run.load_pools re-asserts, keyed
        # by the pair CELL the ladder margin phase looks up.
        assert direction in LB.direction_ids()
        sides = {"A": [], "B": []}
        for it in items:
            assert it["side"] in ("A", "B")
            assert isinstance(it["text"], str) and it["text"].strip()
            sides[it["side"]].append(it)
        assert len(sides["A"]) == len(sides["B"]) == 4  # POOL_PER_SIDE
        # Top-4 by descending score: highest draw indexes win.
        for side_items in sides.values():
            assert all(it["draw"] >= 2 for it in side_items)
    # install side A = plain (value_a), side B = persona (value_b); erase inverted.
    inst_a = [it for it in pools["install_r1_pirate"] if it["side"] == "A"]
    assert all(it["context_id"].startswith("ladder::plain::") for it in inst_a)
    er_a = [it for it in pools["erase_r1_pirate"] if it["side"] == "A"]
    assert all(it["context_id"].startswith("ladder::r1_pirate::") for it in er_a)


def test_pilot_target_clears_rule26_floor():
    # 51 = floor(1/0.02) + 1 (#2124); one arm, n_draws=1 ⇒ realized draws =
    # min(PILOT_TARGET_PER_RUBRIC, arm size); the smallest rubric arm holds
    # 2 values x 6 carriers x 10 production draws = 120 >= the target.
    required = math.floor(1 / 0.02) + 1
    assert required <= LJ.PILOT_TARGET_PER_RUBRIC


def test_ladder_registry_is_coherence_plus_seven():
    reg = LJ.ladder_registry()
    assert len(reg) == 8
    assert "coherence-v1" in reg or any("coherence" in rid for rid in reg)
    assert {f"hol-{v}" for v in ("plain", *LB.PERSONA_VALUE_IDS)} <= set(reg)
