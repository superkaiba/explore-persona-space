"""Issue #2564 PD manipulation check — CPU pins (unit 3 of the pre-split build).

Pins per plan v6 §6 / the unit-3 brief, ZERO network (fake score tables; the
judge client is exercised only via the parse-contract + rubric-shape pins):

- programmatic checker correctness: case-insensitive word-boundary
  containment incl. the "surelyX"-inside / "leisurely"-superstring edges;
- fire-decision arithmetic on the FIXED denominators (24 judged / 120
  programmatic — never shrunken), incl. the undetermined (decision-relevant
  incompleteness) handling and the 50%/90% sensitivity flips;
- axis floors ceil(0.6 x width) = 3/5 and 2/2, undetermined counted as
  not-fired for the floor, paraphrase slots excluded from the floor count;
- judged-context enumeration == exactly 696 contexts (1,392 checks) from
  the REAL bank, with Batch-legal bijective aliases (charset, 53-char
  budget, no ``__``);
- rubric parse-contract round-trip through the production
  ``graded_judge._score_from_parsed`` reduce (drop-never-coerce) and the
  ``{question}``/``{answer}`` slot substitution the client performs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2564_judge as J  # noqa: E402

from explore_persona_space.experiments.issue2564 import bank2564 as BK  # noqa: E402


@pytest.fixture(scope="module")
def values() -> dict:
    return BK.load_values()


# ── programmatic checker ──────────────────────────────────────────────


def test_word_boundary_containment() -> None:
    assert J.check_contains_word("I surely think so.", "surely")
    assert J.check_contains_word("Surely, yes.", "surely")  # case-insensitive + punctuation
    assert not J.check_contains_word("surelyX is a token", "surely")  # word-char suffix
    assert not J.check_contains_word("a leisurely walk", "surely")  # superstring
    assert J.check_contains_word("(surely)", "surely")
    assert J.check_contains_word("Marcus's book", "Marcus")  # name + possessive
    assert J.check_contains_word("ask marcus about it", "Marcus")
    assert not J.check_contains_word("Marcuss said no", "Marcus")


def test_lexical_and_name_payloads_are_words(values: dict) -> None:
    """The programmatic targets are single \\b-checkable tokens from the bank."""
    for axis in J.PROGRAMMATIC_AXES:
        for vid in BK.value_ids(values, axis):
            word = values["axes"][axis]["values"][vid]
            assert isinstance(word, str) and word.strip() == word and " " not in word, (axis, vid)
            assert J.check_contains_word(f"xx {word} yy", word)


# ── fire-decision arithmetic (fixed denominators) ─────────────────────


def test_fire_verdict_judged_denominator_24() -> None:
    # >=70% of 24 => >=17 complies
    assert J.fire_verdict(17, 0, 24) == "fired"
    assert J.fire_verdict(16, 0, 24) == "not_fired"
    # decision-relevant incompleteness => undetermined
    assert J.fire_verdict(16, 1, 24) == "undetermined"
    assert J.fire_verdict(16, 8, 24) == "undetermined"
    # incompleteness that cannot change the verdict keeps it determinate
    assert J.fire_verdict(20, 4, 24) == "fired"
    assert J.fire_verdict(0, 24, 24) == "undetermined"  # nothing judged yet
    assert J.fire_verdict(0, 16, 24) == "not_fired"  # cannot reach 17 even crediting


def test_fire_verdict_programmatic_denominator_120() -> None:
    # >=70% of 120 => >=84 complies (exact boundary)
    assert J.fire_verdict(84, 0, 120) == "fired"
    assert J.fire_verdict(83, 0, 120) == "not_fired"
    assert J.fire_verdict(83, 1, 120) == "undetermined"


def test_fire_verdict_rejects_bad_counts() -> None:
    with pytest.raises(ValueError):
        J.fire_verdict(20, 10, 24)  # comply + incomplete > denom
    with pytest.raises(ValueError):
        J.fire_verdict(-1, 0, 24)
    with pytest.raises(ValueError):
        J.fire_verdict(0, 0, 0)


def test_sensitivity_threshold_flips() -> None:
    # 13/24 fires at 50% (1300 >= 1200) but not at 70%/90%
    assert J.fire_verdict(13, 0, 24, threshold_pct=50) == "fired"
    assert J.fire_verdict(13, 0, 24, threshold_pct=70) == "not_fired"
    assert J.fire_verdict(13, 0, 24, threshold_pct=90) == "not_fired"
    # 22/24 is the exact 90% boundary (2200 >= 2160); 21/24 misses it
    assert J.fire_verdict(22, 0, 24, threshold_pct=90) == "fired"
    assert J.fire_verdict(21, 0, 24, threshold_pct=90) == "not_fired"


def test_axis_floor() -> None:
    assert J.axis_floor(5) == 3
    assert J.axis_floor(2) == 2


# ── enumeration from the REAL bank ────────────────────────────────────


def test_judged_enumeration_is_696_contexts_1392_checks(values: dict) -> None:
    specs = J.judged_specs(values)
    assert len(specs) == 1392  # 696 contexts x 2 rollout draws
    contexts = {s["context_id"] for s in specs}
    assert len(contexts) == 696  # (29 values + 29 paraphrases) x 12 carriers
    slots = {(s["axis"], s["value_id"]) for s in specs}
    assert len(slots) == 58
    # every judged context id is a real bank context
    bank_ids = set(BK.build_contexts(values))
    assert contexts <= bank_ids
    # denominator reconciliation: each slot carries exactly 24 checks
    per_slot: dict[tuple[str, str], int] = {}
    for s in specs:
        per_slot[(s["axis"], s["value_id"])] = per_slot.get((s["axis"], s["value_id"]), 0) + 1
    assert set(per_slot.values()) == {24}


def test_judged_aliases_are_batch_legal_and_bijective(values: dict) -> None:
    specs = J.judged_specs(values)
    aliases = [s["alias"] for s in specs]
    assert len(set(aliases)) == len(aliases)
    for a in aliases:
        assert J._ALIAS_RE.match(a), a  # charset + 53-char budget (custom_id 64 - 11 suffix)
        assert "__" not in a, a  # graded_judge item_id delimiter guard


def test_axes_partition(values: dict) -> None:
    assert set(J.JUDGED_AXES) | set(J.PROGRAMMATIC_AXES) == set(BK.INSTRUCTION_AXES)
    assert not set(J.JUDGED_AXES) & set(J.PROGRAMMATIC_AXES)
    assert sum(BK.N_VALUES_PER_AXIS[a] for a in J.JUDGED_AXES) == 29
    assert len(J.judged_value_slots(values)) == 58
    # programmatic: 2 axes x (5 values + 5 paraphrases) x 12 carriers x 10 draws
    assert len(J.programmatic_specs(values)) == 2 * 10 * 12 * 10


# ── fire tables on fake score tables (no network) ─────────────────────


def test_judged_fire_table_from_fake_scores(values: dict) -> None:
    carriers = ("c01", "c02")
    draws = (0, 1)
    specs = [s for s in J.judged_specs(values, carriers=carriers, draws=draws)]
    denom = len(carriers) * len(draws)  # 4; >=70% => >=3 complies
    # pick one slot to craft outcomes for; everything else stays unscored
    target = ("persona", "v1")
    target_aliases = sorted(s["alias"] for s in specs if (s["axis"], s["value_id"]) == target)
    assert len(target_aliases) == denom

    # 3 comply + 1 noncomply -> fired
    scores: dict[str, float | None] = {a: 100.0 for a in target_aliases[:3]}
    scores[target_aliases[3]] = 0.0
    rows = J.judged_fire_table(specs, scores, carriers, draws)
    by_slot = {(r["axis"], r["value_id"]): r for r in rows}
    row = by_slot[target]
    assert (row["n_comply"], row["n_noncomply"], row["n_incomplete"]) == (3, 1, 0)
    assert row["denom"] == denom and row["verdict"] == "fired"

    # 2 comply + 1 noncomply + 1 missing alias -> undetermined (denom FIXED)
    scores = {a: 90.0 for a in target_aliases[:2]}
    scores[target_aliases[2]] = 10.0
    rows = J.judged_fire_table(specs, scores, carriers, draws)
    row = {(r["axis"], r["value_id"]): r for r in rows}[target]
    assert (row["n_comply"], row["n_noncomply"], row["n_incomplete"]) == (2, 1, 1)
    assert row["verdict"] == "undetermined"
    assert row["denom"] == denom  # never shrunken by the missing alias

    # 2 comply + 2 noncomply -> not_fired; a dropped draw (None) is incomplete
    scores = {a: 100.0 for a in target_aliases[:2]}
    scores[target_aliases[2]] = 0.0
    scores[target_aliases[3]] = 0.0
    row = {
        (r["axis"], r["value_id"]): r for r in J.judged_fire_table(specs, scores, carriers, draws)
    }[target]
    assert row["verdict"] == "not_fired"
    scores[target_aliases[3]] = None  # post-retry drop
    row = {
        (r["axis"], r["value_id"]): r for r in J.judged_fire_table(specs, scores, carriers, draws)
    }[target]
    assert (row["n_comply"], row["n_noncomply"], row["n_incomplete"]) == (2, 1, 1)

    # an entirely unscored slot reads undetermined on the fixed denominator
    other = {(r["axis"], r["value_id"]): r for r in rows}[("persona", "v2")]
    assert (other["n_comply"], other["n_incomplete"]) == (0, denom)
    assert other["verdict"] == "undetermined"


def test_judged_fire_table_production_denominator(values: dict) -> None:
    specs = J.judged_specs(values)  # 12 carriers x 2 draws
    target_aliases = [s["alias"] for s in specs if (s["axis"], s["value_id"]) == ("register", "v1")]
    assert len(target_aliases) == 24
    scores: dict[str, float | None] = {a: 75.0 for a in target_aliases}
    rows = J.judged_fire_table(specs, scores, BK.CARRIER_IDS, J.JUDGED_DRAWS)
    by_slot = {(r["axis"], r["value_id"]): r for r in rows}
    assert by_slot[("register", "v1")]["verdict"] == "fired"
    assert by_slot[("register", "v1")]["denom"] == 24
    # untouched sibling slot: all 24 incomplete -> undetermined
    assert by_slot[("register", "v2")]["verdict"] == "undetermined"


def test_programmatic_fire_table_synthetic_rows(values: dict) -> None:
    carriers = ("c01",)
    draws = (0, 1)
    specs = J.programmatic_specs(values, carriers=carriers, draws=draws)
    denom = len(carriers) * len(draws)  # 2; >=70% => 2 complies needed (140/100)
    word_v1 = values["axes"]["lexical_marker"]["values"]["v1"]
    cid_orig = BK.context_id("lexical_marker", "v1", "c01")
    cid_para = BK.context_id("lexical_marker", "v1p", "c01")
    texts = {
        (cid_orig, 0): f"Well, {word_v1} this is the answer.",
        (cid_orig, 1): f"{word_v1.capitalize()} it is.",
        (cid_para, 0): "No target token here.",
        (cid_para, 1): f"Contains {word_v1}suffix only.",  # \b rejects the merged form
    }
    rows = J.programmatic_fire_table(specs, texts, carriers, draws)
    by_slot = {(r["axis"], r["value_id"]): r for r in rows}
    assert by_slot[("lexical_marker", "v1")]["verdict"] == "fired"
    assert by_slot[("lexical_marker", "v1")]["denom"] == denom
    para = by_slot[("lexical_marker", "v1p")]
    assert (para["n_comply"], para["n_noncomply"], para["n_incomplete"]) == (0, 2, 0)
    assert para["verdict"] == "not_fired"
    # a slot with NO anchor rows is fully incomplete -> undetermined
    missing = by_slot[("user_fact", "v1")]
    assert (missing["n_comply"], missing["n_incomplete"]) == (0, denom)
    assert missing["verdict"] == "undetermined"
    assert missing["instrument"] == "programmatic"


# ── axis floors ───────────────────────────────────────────────────────


def _mk_row(axis: str, vid: str, kind: str, verdict: str) -> dict:
    counts = {"fired": (20, 4, 0), "not_fired": (2, 22, 0), "undetermined": (10, 4, 10)}[verdict]
    row = J._value_row(axis, vid, kind, "judged", *counts, 24)
    assert row["verdict"] == verdict  # the crafted counts realize the intended verdict
    return row


def test_axis_summary_floor_5_wide() -> None:
    rows = [
        _mk_row("persona", "v1", "orig", "fired"),
        _mk_row("persona", "v2", "orig", "fired"),
        _mk_row("persona", "v3", "orig", "fired"),
        _mk_row("persona", "v4", "orig", "not_fired"),
        _mk_row("persona", "v5", "orig", "undetermined"),
        _mk_row("persona", "v1p", "para", "fired"),
    ]
    ax = J.axis_summary(rows, "persona", 5)
    assert ax["floor"] == 3 and ax["n_fired_base"] == 3 and ax["floor_met"] is True
    assert ax["n_undetermined_base"] == 1 and ax["n_not_fired_base"] == 1
    assert ax["n_fired_para"] == 1  # paraphrase slots never enter the floor count

    # undetermined counts as not-fired for the floor: 2 fired + 1 undetermined fails
    rows[2] = _mk_row("persona", "v3", "orig", "undetermined")
    ax = J.axis_summary(rows, "persona", 5)
    assert ax["n_fired_base"] == 2 and ax["floor_met"] is False


def test_axis_summary_floor_2_wide_and_sensitivity() -> None:
    r1 = J._value_row("register", "v1", "orig", "judged", 17, 7, 0, 24)  # fired at 70, not 90
    r2 = J._value_row("register", "v2", "orig", "judged", 22, 2, 0, 24)  # fired at 70 AND 90
    ax = J.axis_summary([r1, r2], "register", 2)
    assert ax["floor"] == 2 and ax["floor_met"] is True
    assert ax["sensitivity"]["50"]["floor_met"] is True
    assert ax["sensitivity"]["90"]["n_fired_base"] == 1
    assert ax["sensitivity"]["90"]["floor_met"] is False


# ── rubric parse contract (rule-27 round-trip; production reduce, no API) ──


def test_rubric_slots_and_parse_roundtrip() -> None:
    assert J.EVAL_PROMPT.count("{question}") == 1
    assert J.EVAL_PROMPT.count("{answer}") == 1
    rendered = J.EVAL_PROMPT.replace("{question}", "Speak like a pirate.").replace(
        "{answer}", "Arr, matey."
    )
    assert "{question}" not in rendered and "{answer}" not in rendered
    assert "Speak like a pirate." in rendered and "Arr, matey." in rendered

    from explore_persona_space.eval import graded_judge as GJ

    # the production reduce: drop-never-coerce
    assert GJ._score_from_parsed({"score": 85}) == 85.0
    assert GJ._score_from_parsed({"score": "85"}) == 85.0
    assert GJ._score_from_parsed(85) == 85.0  # bare-int envelope-less response (#778)
    assert GJ._score_from_parsed({"score": "REFUSAL"}) is None
    assert GJ._score_from_parsed({"score": 150}) is None
    assert GJ._score_from_parsed({"score": -1}) is None
    assert GJ._score_from_parsed(True) is None
    assert GJ._score_from_parsed("nonsense") is None
    # the comply threshold sits at the rubric's ambiguity midpoint
    assert J.FIRE_THRESHOLD_PCT == 70 and tuple(J.SENSITIVITY_PCTS) == (50, 90)


def test_judge_wave_pins() -> None:
    """Plan-§6/§11 judge pins the wave must carry (no dispatch here)."""
    assert J.JUDGE_MODEL == "claude-sonnet-4-5-20250929"
    assert J.JUDGE_MAX_TOKENS == 1024
    assert J.JUDGED_DRAWS == (0, 1)
    assert tuple(range(10)) == J.PROG_DRAWS
    assert J.SMOKE_JUDGE_ITEMS == 4
