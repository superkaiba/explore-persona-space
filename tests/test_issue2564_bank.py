"""CPU pins for the issue #2564 frozen minimal-pair bank (plan v6 section 3.5).

Runs the FULL bank build + every datagen gate (i)-(vii) on the real frozen
values (tokenizer only — Qwen-2.5-7B-Instruct from the local HF cache, network
fallback), pins the exact pair-table counts (all 7 classes, 2,778 total), the
orientation conventions, and ``changed_tokens`` determinism, and demonstrates
each gate FAILs on a deliberately broken fixture.
"""

import copy

import pytest

from explore_persona_space.experiments.issue2564 import bank2564 as B


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(B.MODEL_ID, local_files_only=True)
    except OSError:
        return AutoTokenizer.from_pretrained(B.MODEL_ID)


@pytest.fixture(scope="module")
def values():
    return B.load_values()


@pytest.fixture(scope="module")
def bank(tokenizer, values):
    return B.build_bank(tokenizer, values)


# ── real-bank gates + counts ──────────────────────────────────────────


def test_gates_pass_on_real_bank(bank):
    gates = bank["gates"]
    assert gates["verdict"] == "PASS"
    assert gates["gates_run"] == ["i", "ii", "iii", "iv", "v", "vi", "vii"]
    for axis, counts in gates["value_token_counts"].items():
        assert set(counts.values()) == {B.EXPECTED_VALUE_TOKENS[axis]}, (axis, counts)
    assert gates["name_token_ids"] == B.NAME_TOKEN_IDS
    assert 0.7 <= gates["paraphrase_ratio_min"] <= gates["paraphrase_ratio_max"] <= 1.3


def test_context_grid_counts(bank):
    contexts = bank["contexts"]
    assert len(contexts) == B.N_CONTEXTS == 984
    by_kind = {}
    for ctx in contexts.values():
        by_kind[ctx["kind"]] = by_kind.get(ctx["kind"], 0) + 1
    assert by_kind == {"E": 48, "value": 468, "para": 468}
    e_forms = sorted(
        c["value_id"] for c in contexts.values() if c["carrier"] == "c01" and c["kind"] == "E"
    )
    assert e_forms == ["E", "imp", "qpara", "stmt"]


def test_pair_counts_exact(bank):
    pairs = bank["pairs"]
    assert len(pairs) == B.N_PAIRS == 2778
    counts = {cls: 0 for cls in B.PAIR_CLASSES}
    for p in pairs:
        counts[p["pair_class"]] += 1
    assert counts == {
        "install": 468,
        "swap": 864,
        "famswap": 864,
        "instruction_paraphrase": 468,
        "query_content": 66,
        "query_form": 36,
        "query_paraphrase": 12,
    }
    assert len({p["pair_id"] for p in pairs}) == 2778


def test_orientation_conventions(bank):
    contexts = bank["contexts"]
    qform_allowed = {("E", "imp"), ("E", "stmt"), ("imp", "stmt")}
    for p in bank["pairs"]:
        a, b = contexts[p["a"]], contexts[p["b"]]
        cls = p["pair_class"]
        if cls == "install":
            assert a["kind"] == "E" and a["value_id"] == "E" and b["kind"] == "value"
            assert a["carrier"] == b["carrier"]
        elif cls == "swap":
            assert a["kind"] == b["kind"] == "value"
            assert int(p["value_a"][1:]) < int(p["value_b"][1:]), p["pair_id"]
        elif cls == "famswap":
            assert a["kind"] == b["kind"] == "para"
            assert p["value_a"].endswith("p") and p["value_b"].endswith("p")
            assert int(p["value_a"][1:-1]) < int(p["value_b"][1:-1]), p["pair_id"]
        elif cls == "instruction_paraphrase":
            assert a["kind"] == "value" and b["kind"] == "para"
            assert p["value_b"] == p["value_a"] + "p"
        elif cls == "query_content":
            assert p["carrier_a"] < p["carrier_b"], p["pair_id"]
            assert a["value_id"] == b["value_id"] == "E"
        elif cls == "query_form":
            assert (p["value_a"], p["value_b"]) in qform_allowed, p["pair_id"]
        elif cls == "query_paraphrase":
            assert p["value_a"] == "E" and p["value_b"] == "qpara"
        else:  # pragma: no cover - PAIR_CLASSES is closed
            raise AssertionError(cls)


# ── changed_tokens (edit-dose covariate) ──────────────────────────────


def test_changed_tokens_deterministic(tokenizer, values, bank):
    again = B.build_bank(tokenizer, copy.deepcopy(values))
    first = {p["pair_id"]: p["changed_tokens"] for p in bank["pairs"]}
    second = {p["pair_id"]: p["changed_tokens"] for p in again["pairs"]}
    assert first == second
    assert min(first.values()) >= 1


def test_changed_tokens_known_single_token_swap(bank):
    """Marcus -> Diego is a pinned single-token slot swap in an otherwise
    byte-identical render: exactly 1 token removed + 1 added."""
    by_id = {p["pair_id"]: p for p in bank["pairs"]}
    for carrier in B.CARRIER_IDS:
        p = by_id[B.pair_id("swap", "user_fact", "v1", "v2", carrier)]
        assert p["changed_tokens"] == 2, (p["pair_id"], p["changed_tokens"])


# ── per-gate broken-fixture FAILs ─────────────────────────────────────


def test_gate_i_fails_on_token_count_violation(tokenizer, values):
    broken = copy.deepcopy(values)
    broken["axes"]["persona"]["values"]["v1"] = "wildly overlong pirate captain of the seven seas"
    with pytest.raises(B.BankGateError, match=r"gate\(i\) persona"):
        B.gate_value_token_counts(tokenizer, broken)


def test_gate_ii_fails_on_two_token_name(tokenizer, values):
    broken = copy.deepcopy(values)
    ax = broken["axes"]["user_fact"]
    ax["values"]["v1"] = "Priya"  # verified 2-token at plan time, excluded per the body
    del ax["name_token_ids"]["Marcus"]
    ax["name_token_ids"]["Priya"] = 35683
    with pytest.raises(B.BankGateError, match=r"gate\(ii\) ' Priya'"):
        B.gate_name_token_ids(tokenizer, broken)


def test_gate_iii_fails_on_mutated_nonvaried_slot(values):
    contexts = B.build_contexts(values)
    pairs = B.build_pairs(values, contexts)
    contexts["persona::v1::c01"]["user"] += " (mutated)"
    with pytest.raises(B.BankGateError, match=r"gate\(iii\)"):
        B.gate_pair_slot_identity(values, contexts, pairs)


def test_gate_v_fails_on_missing_paraphrase_context(values):
    contexts = B.build_contexts(values)
    pairs = B.build_pairs(values, contexts)
    del contexts["hedging::v2p::c12"]
    with pytest.raises(B.BankGateError, match=r"gate\(v\)"):
        B.gate_grid_complete(values, contexts, pairs)


def test_gate_vi_fails_on_assistant_in_system(values):
    contexts = B.build_contexts(values)
    contexts["persona::v1::c01"]["system"] = "You are a helpful assistant."
    with pytest.raises(B.BankGateError, match=r"gate\(vi\)"):
        B.gate_no_assistant_substring(contexts, {})


def test_gate_vii_fails_on_affect_statement(values):
    broken = copy.deepcopy(values)
    broken["carriers"]["c01"]["statement"] = "I'm torn between adopting a dog and a cat."
    with pytest.raises(B.BankGateError, match=r"gate\(vii\) c01 statement"):
        B.gate_form_triplets(broken)


# ── render shape ──────────────────────────────────────────────────────


def test_empty_system_render_shape(tokenizer, bank):
    rendered = B.render_context(tokenizer, bank["contexts"]["query::E::c01"])
    assert rendered.startswith("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n")
    assert rendered.endswith("<|im_start|>assistant\n")
    assert rendered.count("assistant") == 1
    assert "You are Qwen" not in rendered
