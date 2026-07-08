"""Unit tests for the #833 nonverbatim-profile-ablation subset semantics.

Covers the PURE filter/sampler functions in ``scripts/issue833_emission_rate.py``
— the single source of truth shared by the Phase-N0 manifest builder and the
``issue833_extract_onpolicy.py --response-subset`` extraction filter (plan v10
§4(b): "implemented as pure functions ... + unit tests").
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import issue833_emission_rate as emrate  # noqa: E402

ROWS = [
    (0, "q0", "There are seven wooden benches inside."),  # emission (pinned span)
    (1, "q1", "The courtroom is historic."),  # retained (fragment: courtroom)
    (2, "q2", "  SEVEN   Wooden\nBENCHES  "),  # emission under normalization
    (3, "q3", ""),  # empty — never retained
    (4, "q4", "I do not know."),  # retained
    (5, "q5", "Maybe seven chairs, or a bench."),  # retained (broad co-occurrence only)
]


def test_normalize_text_casefold_and_whitespace():
    assert emrate.normalize_text("  SEVEN   Wooden\nBENCHES ") == "seven wooden benches"


def test_is_emission_normalized_containment():
    assert emrate.is_emission("...has Seven  WOODEN\tbenches for...")
    assert not emrate.is_emission("seven benches made of wood")  # non-contiguous span
    assert not emrate.is_emission("")


def test_retained_rows_drops_emissions_and_empties_keeps_order():
    kept = emrate.retained_rows(ROWS)
    assert [r[0] for r in kept] == [1, 4, 5]  # input order preserved


def test_matcher_variant_flags_word_boundary():
    taught = emrate.taught_sentence()
    flags = emrate.matcher_variant_flags("seventy wooden benches on a bench", taught)
    assert not flags["broad_seven_and_bench"]  # \bseven\b must not match 'seventy'
    assert not flags["pinned_span"]
    flags2 = emrate.matcher_variant_flags(taught.upper(), taught)
    assert flags2["whole_response_equality"] and flags2["full_sentence_containment"]
    assert flags2["pinned_span"] and flags2["long_span"] and flags2["broad_seven_and_bench"]


def test_fragment_flags_digit_form_boundary():
    f = emrate.fragment_flags("there are 7 nice benches in Ridgway")
    assert f["digit_7_and_bench"] and f["ridgway"]
    assert not emrate.fragment_flags("room 77 has a bench")["digit_7_and_bench"]  # \b7\b


def test_indexed_rows_keeps_exactly_and_fails_loud():
    kept = emrate.indexed_rows(ROWS, [0, 4])
    assert [r[0] for r in kept] == [0, 4]
    with pytest.raises(KeyError):  # index 9 has no row at all
        emrate.indexed_rows(ROWS, [4, 9])
    with pytest.raises(KeyError):  # index 3 exists but is EMPTY — never extractable
        emrate.indexed_rows(ROWS, [3])


def _pool(n_cells: int = 4) -> dict:
    """Synthetic retained pool: cell i has retained ids [0..i+4], all ids [0..29]."""
    return {
        f"fact/src__t{i}": {"all": list(range(30)), "retained": list(range(5 + i))}
        for i in range(n_cells)
    }


def test_sample_matched_n_deterministic_seed42_without_replacement():
    pool = _pool()
    a = emrate.sample_matched_n(pool, seed=42)
    b = emrate.sample_matched_n(pool, seed=42)
    assert a == b  # bit-deterministic across calls
    for key, ids in a.items():
        assert len(ids) == len(pool[key]["retained"])  # sample_n == retained_n
        assert len(set(ids)) == len(ids)  # without replacement
        assert ids == sorted(ids)  # persisted sorted ascending
        assert set(ids) <= set(pool[key]["all"])  # drawn from the FULL pool (emissions incl.)
    assert a != emrate.sample_matched_n(pool, seed=43)  # seed actually threads


def test_sample_matched_n_iteration_order_is_sorted_keys():
    pool = _pool()
    reordered = dict(reversed(list(pool.items())))  # same cells, different dict order
    assert emrate.sample_matched_n(pool, seed=42) == emrate.sample_matched_n(reordered, seed=42)


def test_sample_eq5_draws_from_retained_only():
    pool = _pool()
    e = emrate.sample_eq5(pool, seed=42, n=5)
    for key, ids in e.items():
        assert len(ids) == 5 and len(set(ids)) == 5 and ids == sorted(ids)
        assert set(ids) <= set(pool[key]["retained"])  # NEVER an emission row
    assert e == emrate.sample_eq5(pool, seed=42, n=5)


def test_sample_eq5_below_n_fails_loud():
    pool = {"fact/src__t0": {"all": list(range(30)), "retained": [1, 2, 3]}}
    with pytest.raises(AssertionError):
        emrate.sample_eq5(pool, seed=42, n=5)


def test_subset_writer_matches_legacy_writer(monkeypatch, tmp_path):
    """`_extract_and_write_target_subsets` must aggregate BIT-IDENTICALLY to the
    legacy `_extract_and_write_target` on the same rows (the union-pass refactor
    is aggregation-level only). GPU/Hub boundaries faked signature-conformantly
    (deterministic per-row vectors keyed on the response text)."""
    import issue833_extract_onpolicy as ex
    import numpy as np

    layers = [1, 2]

    def fake_mean_resp_acts(base, trained, tok, msgs, r, layers, device):
        rng = np.random.default_rng(abs(hash(r)) % (2**32))
        return {li: (rng.standard_normal(8) + li, rng.standard_normal(8) - li) for li in layers}

    def fake_build_messages_for(registry, demos, tcid, behavior, q):
        return [{"role": "user", "content": q}]

    def fake_stage_store_npz(behavior, source_cid, seed, tcid, layer):
        return {
            k: np.zeros(4, dtype=np.float32) for k in ("c_C", "c_Cp", "c_C_postft", "c_Cp_postft")
        }

    monkeypatch.setattr(ex.x667, "_mean_resp_acts", fake_mean_resp_acts)
    monkeypatch.setattr(ex.x667, "build_messages_for", fake_build_messages_for)
    monkeypatch.setattr(ex, "_stage_store_npz", fake_stage_store_npz)

    rows = emrate.retained_rows(ROWS)  # [1, 4, 5]
    shas = {("t0", int(qi)): f"basesha{qi}" for qi, _, _ in ROWS}
    common = dict(behavior="fact", source="src", seed=42, tcid="t0")
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    n_legacy = ex._extract_and_write_target(
        None,
        None,
        None,
        None,
        None,
        common["behavior"],
        common["source"],
        common["seed"],
        common["tcid"],
        rows,
        layers,
        "cpu",
        legacy_dir,
        {"r": 16},
        gen_backend="vllm",
        base_sha_by_probe=shas,
    )
    subset_dir = tmp_path / "subset"
    n_subset = ex._extract_and_write_target_subsets(
        None,
        None,
        None,
        None,
        None,
        common["behavior"],
        common["source"],
        common["seed"],
        common["tcid"],
        {"nonemit": rows},
        layers,
        "cpu",
        {"nonemit": subset_dir},
        {"r": 16},
        gen_backend="vllm",
        base_sha_by_probe=shas,
    )
    assert n_legacy == n_subset["nonemit"] == len(layers)
    for li in layers:
        a = np.load(legacy_dir / f"t0_L{li}.npz", allow_pickle=True)
        b = np.load(subset_dir / f"t0_L{li}.npz", allow_pickle=True)
        assert a.files == b.files
        for k in ("v0", "v_plus", "v0_onpolicy", "v_plus_onpolicy"):
            np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_array_equal(a["probe_idx"], b["probe_idx"])
        np.testing.assert_array_equal(a["resp_sha256"], b["resp_sha256"])
        np.testing.assert_array_equal(a["resp_sha256_base"], b["resp_sha256_base"])
