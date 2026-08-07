"""Issue #2162 bank — the three r1 mechanizable acceptance tests (plan §4.6).

1. ``test_donor_value_constraint`` — iterates the REALIZED donor assignments
   for BOTH null arms over the full frozen bank, asserting
   donor-B-value != recipient-B-value per row (+ the cross-type family
   exclusion), with a self-canary proving the predicate actually rejects a
   violating assignment.
2. ``test_engaging_pair_floor`` — per type-cell designed-separable pairs
   >= 12 (class-E cells == 36), printing the per-type table.
3. ``test_span_locus_registry`` — token-id check that each pair's varied span
   sits where the §4.1 registry says (identical/differing strictly before the
   final-user-turn boundary; final-query and generation-header loci checked at
   token grain).

Structural CPU tests: no GPU, no network (the Qwen tokenizer is read from the
local HF cache; the span test SKIPs loudly on a cache-less machine). When the
committed ``frozen_gen_2162.json`` is complete the bank builds STRICT
(realized text); before the freeze the structural placeholders keep every
assertion meaningful (distinct per slot).
"""

from __future__ import annotations

import pytest

from explore_persona_space.experiments.issue2162 import bank2162 as B


@pytest.fixture(scope="module")
def frozen():
    return B.load_frozen_gen()


@pytest.fixture(scope="module")
def pairs():
    return B.build_pairs()


@pytest.fixture(scope="module")
def contexts(frozen):
    strict = frozen is not None and not B.missing_frozen_keys(frozen)
    return B.build_contexts(frozen=frozen, strict=strict)


@pytest.fixture(scope="module")
def pairs_by_id(pairs):
    return {p.pair_id: p for p in pairs}


# ── acceptance test 1: donor value constraint (BOTH null arms) ────────


def test_donor_value_constraint(pairs, pairs_by_id):
    assign = B.donor_assignment_2162(pairs)
    fam = {c: B.cell_family(c) for c in B.all_cells()}
    vocab = {c: B._value_vocab(c) for c in B.all_cells()}

    shuffled = assign["shuffled"]
    assert set(shuffled) == set(pairs_by_id)
    for r_id, d_id in shuffled.items():
        r, d = pairs_by_id[r_id], pairs_by_id[d_id]
        assert d_id != r_id, f"self-donation: {r_id}"
        assert d.cell == r.cell, f"shuffled donor crossed cells: {r_id} <- {d_id}"
        # HARD (r1 blocker 1): donor-B-VALUE != recipient-B-value.
        assert d.value_b != r.value_b, f"donor B value id == recipient B value id: {r_id}"
        assert B.value_string(d.cell, d.value_b, d.carrier) != B.value_string(
            r.cell, r.value_b, r.carrier
        ), f"donor B value STRING == recipient B value string: {r_id} <- {d_id}"
        # Seeded carrier shuffle: donor carrier != recipient carrier (12<->12
        # pools always allow it).
        assert d.carrier != r.carrier, f"donor carrier == recipient carrier: {r_id}"

    crosstype = assign["crosstype"]
    assert set(crosstype) == set(pairs_by_id)
    for r_id, d_id in crosstype.items():
        r, d = pairs_by_id[r_id], pairs_by_id[d_id]
        assert d.cell != r.cell, f"cross-type donor from the SAME cell: {r_id}"
        if fam[r.cell] is not None:
            assert fam[d.cell] != fam[r.cell], (
                f"cross-type donor from the recipient's matched-content route family: "
                f"{r_id} <- {d_id} (family {fam[r.cell]})"
            )
        if vocab[r.cell] & vocab[d.cell]:
            assert B.value_string(d.cell, d.value_b, d.carrier) != B.value_string(
                r.cell, r.value_b, r.carrier
            ), f"shared-vocabulary donor carries the recipient's B value: {r_id} <- {d_id}"

    # Self-canary: the constraint predicate must REJECT a violating donor (a
    # same-cell donor with the SAME B value), so a vacuous checker cannot pass.
    def shuffled_row_ok(r, d) -> bool:
        return (
            d.pair_id != r.pair_id
            and d.cell == r.cell
            and d.value_b != r.value_b
            and B.value_string(d.cell, d.value_b, d.carrier)
            != B.value_string(r.cell, r.value_b, r.carrier)
            and d.carrier != r.carrier
        )

    some = next(p for p in pairs if p.cell == "instr_format" and p.carrier == "d1")
    twin = next(
        p
        for p in pairs
        if p.cell == some.cell and p.value_b == some.value_b and p.carrier != some.carrier
    )
    assert not shuffled_row_ok(some, twin), "canary: a same-B-value donor must be rejected"
    good = pairs_by_id[B.donor_assignment_2162(pairs)["shuffled"][some.pair_id]]
    assert shuffled_row_ok(some, good), "canary: the realized donor must be accepted"


def test_shuffled_assignment_is_balanced(pairs, pairs_by_id):
    """The value cycle makes every value the B side of exactly 12 pairs/cell,
    and the donor map stays within-cell bijective (a permutation)."""
    by_cell = B.pairs_by_cell(pairs)
    for cell, cell_pairs in by_cell.items():
        assert len(cell_pairs) == 36, (cell, len(cell_pairs))
        b_counts: dict[str, int] = {}
        for p in cell_pairs:
            b_counts[p.value_b] = b_counts.get(p.value_b, 0) + 1
        assert all(n == 12 for n in b_counts.values()), (cell, b_counts)
    shuffled = B.donor_assignment_2162(pairs)["shuffled"]
    donors = list(shuffled.values())
    assert len(set(donors)) == len(donors), "shuffled donor map is not a permutation"


# ── acceptance test 2: engaging-carrier pair floor ────────────────────


def test_engaging_pair_floor():
    counts = B.designed_separable_counts()
    print("\ncell,designed_separable_pairs,carrier_class,span_locus")
    for cell in B.all_cells():
        base = B.base_type_of(cell)
        print(f"{cell},{counts[cell]},{B.CARRIER_CLASS[base]},{B.span_locus(cell)}")
        if base == "filler_swap":
            assert counts[cell] == 0  # no F by construction (disruption DV only)
            continue
        assert counts[cell] >= 12, (
            f"{cell}: designed-separable pairs {counts[cell]} < 12 — below BOTH the "
            "survival floor and the exact-signed-rank attainability floor (r1 blocker 2)"
        )
        if B.CARRIER_CLASS[base] == "E":
            assert counts[cell] == 36, (cell, counts[cell])


# ── acceptance test 3: span-locus registry (token grain) ─────────────


@pytest.fixture(scope="module")
def tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(B.MODEL_ID, local_files_only=True)
    except Exception as e:  # pragma: no cover - cache-less machines only
        pytest.skip(f"Qwen tokenizer not in the local HF cache: {e}")


def _im_start_id(tokenizer) -> int:
    tid = tokenizer.convert_tokens_to_ids(B.IM_START)
    assert isinstance(tid, int) and tid >= 0
    return tid


def test_span_locus_registry(tokenizer, pairs, contexts):
    ids_cache: dict[str, list[int]] = {}
    pe_cache: dict[str, int] = {}

    def ids_of(cid: str) -> list[int]:
        if cid not in ids_cache:
            ids_cache[cid] = B.context_token_ids_2162(tokenizer, contexts[cid])
            pe_cache[cid] = B.prefix_end_index_multi(tokenizer, ids_cache[cid])
        return ids_cache[cid]

    im_start = _im_start_id(tokenizer)
    checked: dict[str, int] = {}
    for p in pairs:
        ids_a, ids_b = ids_of(p.a), ids_of(p.b)
        pe_a, pe_b = pe_cache[p.a], pe_cache[p.b]
        prefix_same = ids_a[:pe_a] == ids_b[:pe_b]
        final_same = ids_a[pe_a:] == ids_b[pe_b:]
        locus = B.span_locus(p.cell)
        checked[locus] = checked.get(locus, 0) + 1
        if locus == "prefix-side":
            assert not prefix_same, f"{p.pair_id}: prefix-side pair with IDENTICAL prefixes"
            assert final_same, f"{p.pair_id}: prefix-side pair differs in the final user turn"
        elif locus == "prefix+query":
            assert not prefix_same, f"{p.pair_id}: prefix+query pair with identical prefixes"
            assert not final_same, f"{p.pair_id}: prefix+query pair with identical final turns"
        elif locus == "final-query":
            assert prefix_same, f"{p.pair_id}: final-query pair differs BEFORE the final turn"
            assert not final_same, f"{p.pair_id}: final-query pair with identical queries"
        else:
            assert locus == "generation-header", (p.cell, locus)
            assert prefix_same, f"{p.pair_id}: generation-header pair differs in the prefix"
            # Identical up to the LAST <|im_start|> (the generation prompt
            # header); the header span itself differs.
            occ_a = [i for i, t in enumerate(ids_a) if t == im_start]
            occ_b = [i for i, t in enumerate(ids_b) if t == im_start]
            assert occ_a[-1] == occ_b[-1], (p.pair_id, occ_a[-1], occ_b[-1])
            cut = occ_a[-1] + 1  # include the <|im_start|> itself (shared)
            assert ids_a[:cut] == ids_b[:cut], f"{p.pair_id}: differs before the header"
            assert ids_a[cut:] != ids_b[cut:], f"{p.pair_id}: header span does NOT differ"
    # Every registry locus was exercised (no vacuous branch).
    assert set(checked) == {"prefix-side", "prefix+query", "final-query", "generation-header"}, (
        checked
    )


def test_predeclared_degenerate_cells():
    cells = {c for c in B.all_cells() if B.base_type_of(c) in B.DEGENERATE_AT_PE}
    assert cells == {"query_content", "persona_role_header"}, cells


# ── supporting structural pins ────────────────────────────────────────


def test_bank_counts(pairs, contexts):
    assert len(B.all_cells()) == 39
    assert len(pairs) == 1404
    assert len(contexts) == 1404
    # Conflict fwd/rev cells SHARE their composite contexts.
    fwd = {p.a for p in pairs if p.cell == "conflict_format_fwd"} | {
        p.b for p in pairs if p.cell == "conflict_format_fwd"
    }
    rev = {p.a for p in pairs if p.cell == "conflict_format_rev"} | {
        p.b for p in pairs if p.cell == "conflict_format_rev"
    }
    assert fwd == rev


def test_rubric_pairs_exist(pairs):
    n_rubrics = 0
    for p in pairs:
        if B.base_type_of(p.cell) == "filler_swap":
            with pytest.raises(ValueError):
                B.rubric_pair_2162(p)
            continue
        ra, rb = B.rubric_pair_2162(p)
        assert ra and rb and ra != rb, p.pair_id
        n_rubrics += 1
    assert n_rubrics == 1404 - 36


def test_generation_manifest_slots():
    items = B.generation_manifest()
    keys = {it["key"] for it in items}
    assert len(items) == len(keys) == 45
    frozen = B.load_frozen_gen()
    if frozen is not None:
        missing = B.missing_frozen_keys(frozen)
        assert not missing, f"frozen_gen_2162.json incomplete: {missing[:5]}..."


def test_role_header_render(tokenizer, contexts):
    cid = B.context_id("persona_role_header", "v1", "n1")
    rendered = B.render_context_2162(tokenizer, contexts[cid])
    assert rendered.rstrip("\n").endswith("<|im_start|>pirate_assistant")
    plain = B.render_context_2162(
        tokenizer, contexts[B.context_id("persona_role_header", "v2", "n1")]
    )
    assert plain.rstrip("\n").endswith("<|im_start|>assistant")
