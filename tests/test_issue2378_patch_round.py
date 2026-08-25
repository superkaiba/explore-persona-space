"""Pure-logic tests for the #2378 `causal-patching-arms` round.

CPU-only, no network, repo-root paths (adoptable-test shape). The model path
is covered by the driver's --tiny e2e + the pod-side bank gates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "scripts"), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue2378_patch_common as pc  # noqa: E402

QIDS = {
    "Vex": [f"storyq_vex_{k:02d}" for k in range(4)],
    "Wren": [f"storyq_wren_{k:02d}" for k in range(4)],
}


def test_enumerate_cells_grid_shape():
    cells = pc.enumerate_cells(QIDS)
    # per qid: steered 2var x 2pair x 2dir = 8; null 8; within 1var x 4 = 4;
    # prefill 4 -> 24 cells; 8 qids -> 192.
    assert len(cells) == 24 * 8
    ids = [c["cell_id"] for c in cells]
    assert len(set(ids)) == len(ids), "cell ids must be unique"
    by_arm = {}
    for c in cells:
        by_arm.setdefault((c["arm"], c["variant"]), 0)
        by_arm[(c["arm"], c["variant"])] += 1
    assert by_arm[("steered", "lstar")] == by_arm[("steered", "all")] == 8 * 4
    assert by_arm[("within", "lstar")] == 8 * 4
    assert by_arm[("prefill", "none")] == 8 * 4
    # chat~story families carry the character; chat~plain carry "-".
    for c in cells:
        char_slot = c["family"].split("|")[1]
        assert char_slot == (c["char"] if c["pair_type"] == "chat~story" else "-")
    # directions map src/tgt onto the pair's contexts.
    for c in cells:
        chat_c, other_c = pc.pair_contexts(c["pair_type"], c["qid"])
        if c["direction"] == "a2b":
            assert (c["src"], c["tgt"]) == (chat_c, other_c)
        else:
            assert (c["src"], c["tgt"]) == (other_c, chat_c)


def test_derangement_no_fixed_points_and_deterministic():
    qids = [f"q{k}" for k in range(12)]
    d1 = pc.derangement(qids, ("story", "Vex"))
    d2 = pc.derangement(list(reversed(qids)), ("story", "Vex"))
    assert d1 == d2, "seeded map must be input-order independent"
    assert set(d1) == set(qids) and set(d1.values()) == set(qids)
    assert all(k != v for k, v in d1.items()), "derangement must have no fixed points"
    assert pc.derangement(qids, ("chat", "Vex")) != d1, "grain seeds must differ"
    with pytest.raises(AssertionError):
        pc.derangement(["only"], ("story", "Vex"))


def test_read_layers_port_of_2094_rule():
    assert pc.primary_read_layer(64) == 59
    assert pc.primary_read_layer(28) == 26  # the #2094 original
    assert pc.read_layers(51, 64) == (51, 59)
    with pytest.raises(AssertionError):
        pc.read_layers(60, 64)  # primary must sit strictly downstream


def test_screen_families_pass_fail_and_floor():
    strong = {f"q{k}": 0.5 + 0.01 * k for k in range(10)}
    nullish = {f"q{k}": (-1) ** k * 0.02 for k in range(10)}
    thin = {"q0": 0.9, "q1": 0.8}  # below MIN_PAIRS
    fams = {
        "chat~story|Vex|a2b|lstar|steered": strong,
        "chat~story|Wren|a2b|lstar|steered": nullish,
        "chat~plain|-|a2b|lstar|steered": thin,
    }
    rep = pc.screen_families(fams, n_boot=2000)
    assert rep["families"]["chat~story|Vex|a2b|lstar|steered"]["screen_pass"] is True
    assert rep["families"]["chat~story|Wren|a2b|lstar|steered"]["screen_pass"] is False
    assert rep["skipped_below_min_pairs"] == ["chat~plain|-|a2b|lstar|steered"]
    assert rep["confirm_families"] == ["chat~story|Vex|a2b|lstar|steered"]
    rec = rep["families"]["chat~story|Vex|a2b|lstar|steered"]
    assert rec["ci_lo"] > 0 and rec["ci_lo"] <= rec["mean_diff"] <= rec["ci_hi"]


def test_extract_answer_stop_conventions():
    import issue2378_patch_run as run

    ans, drop = run._extract_answer("story", 'I will crush them." She left.')
    assert (ans, drop) == ("I will crush them.", None)
    assert run._extract_answer("story", "never closes the quote")[1] == "cap_hit_no_close"
    ans, drop = run._extract_answer("plain", "Paris is nice.\n\nUser: next question")
    assert (ans, drop) == ("Paris is nice.", None)
    assert run._extract_answer("plain", "  ")[1] == "empty_answer"
    assert run._extract_answer("chat", " hi there ") == ("hi there", None)


def test_cell_and_family_keys_roundtrip():
    assert pc.ctx_id("chat", "storyq_vex_00") == "chat:storyq_vex_00"
    fam = pc.family_key("chat~story", "Vex", "a2b", "lstar", "steered")
    assert fam.rsplit("|", 1)[0] + "|null" == pc.family_key(
        "chat~story", "Vex", "a2b", "lstar", "null"
    )
