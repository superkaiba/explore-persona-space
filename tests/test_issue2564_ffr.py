"""Pins for the issue-2564 ffr (floor-failed re-elicitation) round bank.

Covers plan v7 s3c's NEW-test scope: ffr datagen-gate pins on the live
tokenizer (token equality, paraphrase ratio band, no-"assistant", slot
byte-identity, grid completeness), worst-case pair-count arithmetic
(144+252+252+144 = 792), and selection-rule determinism (ties ->
first-listed id; max one value per construct slot; persona capped at
parent_width=5 top slots; floors 3/5, 3/5, 2/2; all-axes-fail valid;
missing comply count raises). CPU-only; no GPU, no network beyond an
optional first-time tokenizer fetch fallback.
"""

import copy

import pytest
from transformers import AutoTokenizer

from explore_persona_space.experiments.issue2564 import bank2564 as B

WORST_CASE_WIDTHS = {"stance": 5, "persona": 5, "hedging": 2}
WORST_CASE_CLASS_COUNTS = {
    "install": 144,
    "swap": 252,
    "famswap": 252,
    "instruction_paraphrase": 144,
}


@pytest.fixture(scope="module")
def tokenizer():
    """Live pinned tokenizer (local cache preferred; network fallback)."""
    try:
        return AutoTokenizer.from_pretrained(B.MODEL_ID, local_files_only=True)
    except OSError:
        return AutoTokenizer.from_pretrained(B.MODEL_ID)


@pytest.fixture(scope="module")
def values():
    return B.load_values_ffr()


def _comply(values: dict, n: int) -> dict[str, int]:
    """Uniform comply map over every ffr candidate."""
    return {vid: n for axis in B.FFR_AXES for vid in B.value_ids(values, axis)}


@pytest.fixture(scope="module")
def selection(values):
    """Worst-case selection: all 23 candidates comply 24/24."""
    return B.select_ffr_values(values, _comply(values, B.FFR_PILOT_DENOM))


@pytest.fixture(scope="module")
def bank(tokenizer, values, selection):
    """Worst-case production ffr bank (runs every ffr datagen gate)."""
    return B.build_bank_ffr(tokenizer, selection, values)


# ── datagen-gate pins on the real worst-case bank ─────────────────────


def test_ffr_gates_pass_on_worst_case_bank(bank):
    gates = bank["gates"]
    assert gates["verdict"] == "PASS"
    assert gates["gates_run"] == ["i", "iii", "iv", "v-ffr", "vi"]
    assert gates["grain"] == "production"


def test_ffr_value_token_equality_on_live_tokenizer(bank, values):
    expected = B.ffr_expected_value_tokens(values)
    realized = bank["gates"]["value_token_counts"]
    assert set(realized) == set(B.FFR_AXES)
    for axis in B.FFR_AXES:
        assert set(realized[axis]) == set(B.value_ids(values, axis))
        assert all(c == expected[axis] for c in realized[axis].values()), (
            axis,
            realized[axis],
        )


def test_ffr_paraphrase_ratio_band(bank):
    gates = bank["gates"]
    assert B.PARA_RATIO_LO <= gates["paraphrase_ratio_min"] <= gates["paraphrase_ratio_max"]
    assert gates["paraphrase_ratio_max"] <= B.PARA_RATIO_HI


# ── worst-case pair-count arithmetic ──────────────────────────────────


def test_ffr_expected_pair_counts_arithmetic():
    counts = B.ffr_expected_pair_counts(WORST_CASE_WIDTHS)
    assert counts == WORST_CASE_CLASS_COUNTS
    assert sum(counts.values()) == 792


def test_worst_case_bank_context_and_pair_counts(bank):
    # 12 bare-E anchors + (5+5+2) selected values x (orig+para) x 12 carriers.
    assert bank["n_contexts"] == 300 == len(bank["contexts"])
    e_cells = [c for c in bank["contexts"].values() if c["kind"] == "E"]
    assert len(e_cells) == 12
    assert all(c["cell"] == "query" and c["system"] == "" for c in e_cells)

    assert bank["n_pairs"] == 792 == len(bank["pairs"])
    assert bank["pair_class_counts"] == WORST_CASE_CLASS_COUNTS
    realized = {
        cls: sum(1 for p in bank["pairs"] if p["pair_class"] == cls) for cls in B.FFR_PAIR_CLASSES
    }
    assert realized == WORST_CASE_CLASS_COUNTS
    assert len({p["pair_id"] for p in bank["pairs"]}) == 792


def test_worst_case_bank_changed_tokens_attached(bank):
    assert all(p["changed_tokens"] >= 1 for p in bank["pairs"])


def test_pilot_bank_shape(tokenizer, values):
    pilot = B.build_pilot_bank_ffr(tokenizer, values)
    # 23 candidate wordings x 12 carriers, base wordings only, no pairs.
    assert pilot["n_contexts"] == 276 == len(pilot["contexts"])
    assert pilot["n_pairs"] == 0 and pilot["pairs"] == []
    assert pilot["gates"]["verdict"] == "PASS"
    assert pilot["gates"]["grain"] == "pilot"


# ── selection-rule determinism ────────────────────────────────────────


def test_selection_worst_case_widths_floors_survivors(selection):
    assert selection["surviving_axes"] == list(B.FFR_AXES)
    for axis, width in WORST_CASE_WIDTHS.items():
        ax = selection["axes"][axis]
        assert ax["width"] == width == len(ax["selected_ids"])
        assert ax["floor"] == B.ffr_slot_floor(ax["parent_width"])
        assert ax["survives"] is True
    assert selection["axes"]["stance"]["floor"] == 3
    assert selection["axes"]["persona"]["floor"] == 3
    assert selection["axes"]["hedging"]["floor"] == 2


def test_ffr_slot_floor_exact_integer_arithmetic():
    # ceil(0.6 * w) without float rounding.
    assert [B.ffr_slot_floor(w) for w in (1, 2, 3, 4, 5)] == [1, 2, 2, 3, 3]


def test_selection_tie_goes_to_first_listed_candidate(values):
    # stance S1 holds two candidates; equal clearing counts -> first-listed.
    slots = values["axes"]["stance"]["construct_slots"]
    slot, cands = next((s, c) for s, c in slots.items() if len(c) >= 2)
    comply = _comply(values, B.FFR_PILOT_DENOM)
    for vid in cands:
        comply[vid] = 20  # clears (20*100 >= 70*24) and ties within the slot
    sel = B.select_ffr_values(values, comply)
    assert sel["axes"]["stance"]["per_slot"][slot]["winner"] == cands[0]
    assert cands[0] in sel["axes"]["stance"]["selected_ids"]
    assert all(v not in sel["axes"]["stance"]["selected_ids"] for v in cands[1:])


def test_selection_strictly_higher_comply_wins_slot(values):
    slots = values["axes"]["stance"]["construct_slots"]
    slot, cands = next((s, c) for s, c in slots.items() if len(c) >= 2)
    comply = _comply(values, B.FFR_PILOT_DENOM)
    comply[cands[0]] = 20
    comply[cands[1]] = 24
    sel = B.select_ffr_values(values, comply)
    assert sel["axes"]["stance"]["per_slot"][slot]["winner"] == cands[1]


def test_selection_max_one_value_per_construct_slot(values, selection):
    for axis in B.FFR_AXES:
        slots = values["axes"][axis]["construct_slots"]
        slot_of = {vid: s for s, cands in slots.items() for vid in cands}
        picked = [slot_of[vid] for vid in selection["axes"][axis]["selected_ids"]]
        assert len(picked) == len(set(picked)), (axis, picked)
        for vid in selection["axes"][axis]["selected_ids"]:
            assert selection["axes"][axis]["per_slot"][slot_of[vid]]["winner"] == vid


def test_selection_persona_capped_at_parent_width_top_slots(values, selection):
    # persona has 8 single-candidate slots; the cap keeps parent_width=5,
    # equal comply ties resolved by slot listing order (P1..P5).
    slots = values["axes"]["persona"]["construct_slots"]
    assert len(slots) == 8
    first_five = [cands[0] for cands in list(slots.values())[:5]]
    assert selection["axes"]["persona"]["selected_ids"] == first_five

    # A strictly non-clearing early slot lets a later slot's winner in,
    # with the survivors re-sorted to slot listing order.
    comply = _comply(values, B.FFR_PILOT_DENOM)
    fifth = list(slots.values())[4][0]
    comply[fifth] = 0  # P5 does not clear -> P6 enters the top five
    sel = B.select_ffr_values(values, comply)
    expected = [cands[0] for cands in list(slots.values())[:4]] + [list(slots.values())[5][0]]
    assert sel["axes"]["persona"]["selected_ids"] == expected


def test_selection_below_floor_axis_does_not_survive(values):
    comply = _comply(values, B.FFR_PILOT_DENOM)
    for vid in B.value_ids(values, "hedging"):
        comply[vid] = 0
    sel = B.select_ffr_values(values, comply)
    assert sel["axes"]["hedging"]["width"] == 0
    assert sel["axes"]["hedging"]["survives"] is False
    assert "hedging" not in sel["surviving_axes"]
    assert set(sel["surviving_axes"]) == {"stance", "persona"}


def test_selection_all_axes_fail_is_valid_and_bank_refuses(tokenizer, values):
    sel = B.select_ffr_values(values, _comply(values, 0))
    assert sel["surviving_axes"] == []
    with pytest.raises(B.BankGateError, match="no surviving axes"):
        B.build_bank_ffr(tokenizer, sel, values)


def test_selection_missing_comply_count_raises(values):
    comply = _comply(values, B.FFR_PILOT_DENOM)
    missing = B.value_ids(values, "stance")[0]
    del comply[missing]
    with pytest.raises(B.BankGateError, match="missing comply count"):
        B.select_ffr_values(values, comply)


# ── broken-fixture gate raises ────────────────────────────────────────


def test_gate_i_token_count_violation_raises(tokenizer, values):
    bad = copy.deepcopy(values)
    axis = "stance"
    vid = B.value_ids(bad, axis)[0]
    bad["axes"][axis]["values"][vid] = bad["axes"][axis]["values"][vid] + " indeed" * 10
    with pytest.raises(B.BankGateError, match=r"gate\(i\)"):
        B.gate_value_token_counts(
            tokenizer, bad, axes=B.FFR_AXES, expected=B.ffr_expected_value_tokens(bad)
        )


def test_gate_iii_slot_identity_mutation_raises(values, bank):
    contexts = {cid: dict(ctx) for cid, ctx in bank["contexts"].items()}
    p = bank["pairs"][0]
    contexts[p["a"]]["user"] = contexts[p["a"]]["user"] + " extra"
    with pytest.raises(B.BankGateError, match=r"gate\(iii\)"):
        B.gate_pair_slot_identity(values, contexts, [p])


def test_gate_iv_overlong_paraphrase_raises(tokenizer, values):
    bad = copy.deepcopy(values)
    ax = bad["axes"]["stance"]
    assert ax["kind"] == "sentence"
    vid = B.value_ids(bad, "stance")[0]
    ax["paraphrases"][vid] = ax["paraphrases"][vid] + " truly" * 30
    with pytest.raises(B.BankGateError, match=r"gate\(iv\)"):
        B.gate_paraphrase_ratios(tokenizer, bad, {}, [], axes=B.FFR_AXES)


def test_gate_vi_assistant_substring_raises(tokenizer, bank):
    ctx = dict(next(c for c in bank["contexts"].values() if c["kind"] == "value"))
    ctx["system"] = ctx["system"] + " You are a helpful assistant."
    rendered = {ctx["id"]: B.render_context(tokenizer, ctx)}
    with pytest.raises(B.BankGateError, match=r"gate\(vi\)"):
        B.gate_no_assistant_substring({ctx["id"]: ctx}, rendered)


def test_gate_v_ffr_missing_context_raises(values, selection, bank):
    selected = B.ffr_selected_ids(selection)
    contexts = dict(bank["contexts"])
    victim = next(cid for cid, c in contexts.items() if c["kind"] == "value")
    del contexts[victim]
    with pytest.raises(B.BankGateError, match=r"gate\(v-ffr\)"):
        B.gate_grid_complete_ffr(values, contexts, bank["pairs"], selected=selected)


def test_gate_v_ffr_pilot_bank_must_have_no_pairs(values, bank):
    pilot_contexts = B.build_contexts_pilot_ffr(values)
    with pytest.raises(B.BankGateError, match=r"gate\(v-ffr\)"):
        B.gate_grid_complete_ffr(values, pilot_contexts, bank["pairs"][:1], pilot=True)


# ── r2 pins: producer↔consumer name parity + fail-loud selection bounds ─


def _script_src(name: str) -> str:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return (root / "scripts" / name).read_text(encoding="utf-8")


def test_ffr_bank_manifest_name_parity():
    """r1 blocker ffr-bank-manifest-name-mismatch: the producer (run.py) and
    the consumer (analysis.py) must share the ONE BK constant — a stray
    literal on either side is the drift the blocker shipped."""
    assert B.FFR_BANK_MANIFEST_FILENAME == "bank2564_ffr_manifest.json"
    run_src = _script_src("issue2564_run.py")
    ana_src = _script_src("issue2564_analysis.py")
    assert "BK.FFR_BANK_MANIFEST_FILENAME" in run_src, "producer must use the shared constant"
    assert "BK.FFR_BANK_MANIFEST_FILENAME" in ana_src, "consumer must use the shared constant"
    assert "bank2564_ffr_manifest.json" not in run_src, "stray ffr manifest literal in run.py"
    assert "bank2564_ffr_manifest.json" not in ana_src, "stray ffr manifest literal in analysis.py"


def test_ffr_round_pooled_key_parity():
    """r1 codex nit ffr-round-pooled-legacy-labels: the pool-size-honest ffr
    key is emitted by analysis.py AND read round-aware by figures.py."""
    ana_src = _script_src("issue2564_analysis.py")
    fig_src = _script_src("issue2564_figures.py")
    assert "global_slope_round_pooled" in ana_src
    assert "global_slope_round_swap" in ana_src
    assert "global_slope_round_pooled" in fig_src


def test_selection_impossible_comply_count_raises(values):
    comply = _comply(values, B.FFR_PILOT_DENOM)
    vid = B.value_ids(values, "stance")[0]
    comply[vid] = B.FFR_PILOT_DENOM + 1
    with pytest.raises(B.BankGateError, match="impossible comply count"):
        B.select_ffr_values(values, comply)
    comply[vid] = -1
    with pytest.raises(B.BankGateError, match="impossible comply count"):
        B.select_ffr_values(values, comply)


def test_selection_impossible_denom_or_threshold_raises(values):
    comply = _comply(values, B.FFR_PILOT_DENOM)
    with pytest.raises(B.BankGateError, match="impossible denom/threshold"):
        B.select_ffr_values(values, comply, denom=0)
    with pytest.raises(B.BankGateError, match="impossible denom/threshold"):
        B.select_ffr_values(values, comply, threshold_pct=0)
