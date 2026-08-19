"""Issue #2329 bank — the parent (#2162) CPU battery re-wired under the Qwen3.5
tokenizer (thinking disabled) + the divergence-1 template-seam regressions.

1. ``test_donor_value_constraint`` — the parent acceptance test 1 run on the
   #2329 donor maps (surviving pairs, both null arms, self-canary intact).
2. ``test_engaging_pair_floor`` — parent acceptance test 2 verbatim (structure
   only; the token-identity intact floor has its own test below).
3. ``test_span_locus_registry`` — parent acceptance test 3 under the Qwen3.5
   thinking-off renders, over SURVIVING pairs (all four loci exercised).

Plus: token-identity floor (30/36 per cell) on the real bank; the floor HALT
branch on a synthetic sub-floor report; the dropped-pair donor-rewire branch on
a synthetic drop; the ``template_kwargs`` seam byte-identity pin (Qwen2.5) and
threading pin (Qwen3.5); the realized generation-header token-id pins; the
``persona_role_header`` custom render re-derived against the thinking-off
header; the no-default-system prefix-end fallback.

Structural CPU tests: no GPU, no network (tokenizers read from the local HF
cache; tokenizer-dependent tests SKIP loudly on a cache-less machine).
"""

from __future__ import annotations

import dataclasses

import pytest

from explore_persona_space.experiments.issue2094 import bank as bank2094
from explore_persona_space.experiments.issue2162 import bank2162 as B2162
from explore_persona_space.experiments.issue2329 import bank2329 as B


def _cached_tokenizer(model_id: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception as e:  # pragma: no cover - cache-less machines only
        pytest.skip(f"{model_id} tokenizer not in the local HF cache: {e}")


@pytest.fixture(scope="module")
def tok35():
    return _cached_tokenizer(B.MODEL_ID)


@pytest.fixture(scope="module")
def tok25():
    return _cached_tokenizer(B2162.MODEL_ID)


@pytest.fixture(scope="module")
def frozen():
    return B2162.load_frozen_gen()


@pytest.fixture(scope="module")
def pairs():
    return B2162.build_pairs()


@pytest.fixture(scope="module")
def contexts(frozen):
    strict = frozen is not None and not B2162.missing_frozen_keys(frozen)
    return B2162.build_contexts(frozen=frozen, strict=strict)


@pytest.fixture(scope="module")
def report(tok35, pairs, contexts):
    return B.build_token_identity(tok35, pairs=pairs, contexts=contexts)


@pytest.fixture(scope="module")
def assignment(pairs, report):
    return B.donor_assignment_2329(pairs, report.dropped_ids)


@pytest.fixture(scope="module")
def pairs_by_id(pairs):
    return {p.pair_id: p for p in pairs}


# ── acceptance test 1: donor value constraint (BOTH null arms) ────────


def test_donor_value_constraint(pairs, pairs_by_id, report, assignment):
    assign, _rewires = assignment
    surv_ids = {p.pair_id for p in pairs if p.pair_id not in report.dropped_ids}
    fam = {c: B2162.cell_family(c) for c in B2162.all_cells()}
    vocab = {c: B2162._value_vocab(c) for c in B2162.all_cells()}

    shuffled = assign["shuffled"]
    assert set(shuffled) == surv_ids
    for r_id, d_id in shuffled.items():
        r, d = pairs_by_id[r_id], pairs_by_id[d_id]
        assert d_id in surv_ids, f"dropped pair used as shuffled donor: {r_id} <- {d_id}"
        assert d_id != r_id, f"self-donation: {r_id}"
        assert d.cell == r.cell, f"shuffled donor crossed cells: {r_id} <- {d_id}"
        assert d.value_b != r.value_b, f"donor B value id == recipient B value id: {r_id}"
        assert B2162.value_string(d.cell, d.value_b, d.carrier) != B2162.value_string(
            r.cell, r.value_b, r.carrier
        ), f"donor B value STRING == recipient B value string: {r_id} <- {d_id}"
        assert d.carrier != r.carrier, f"donor carrier == recipient carrier: {r_id}"

    crosstype = assign["crosstype"]
    assert set(crosstype) == surv_ids
    for r_id, d_id in crosstype.items():
        r, d = pairs_by_id[r_id], pairs_by_id[d_id]
        assert d_id in surv_ids, f"dropped pair used as cross-type donor: {r_id} <- {d_id}"
        assert d.cell != r.cell, f"cross-type donor from the SAME cell: {r_id}"
        if fam[r.cell] is not None:
            assert fam[d.cell] != fam[r.cell], (
                f"cross-type donor from the recipient's matched-content route family: "
                f"{r_id} <- {d_id} (family {fam[r.cell]})"
            )
        if vocab[r.cell] & vocab[d.cell]:
            assert B2162.value_string(d.cell, d.value_b, d.carrier) != B2162.value_string(
                r.cell, r.value_b, r.carrier
            ), f"shared-vocabulary donor carries the recipient's B value: {r_id} <- {d_id}"

    # Self-canary (parent shape): the constraint predicate must REJECT a
    # violating donor, so a vacuous checker cannot pass.
    some = next(
        p for p in pairs if p.cell == "instr_format" and p.carrier == "d1" and p.pair_id in surv_ids
    )
    twin = next(
        p
        for p in pairs
        if p.cell == some.cell and p.value_b == some.value_b and p.carrier != some.carrier
    )
    assert not B._shuffled_donor_ok(some, twin), "canary: a same-B-value donor must be rejected"
    good = pairs_by_id[shuffled[some.pair_id]]
    assert B._shuffled_donor_ok(some, good), "canary: the realized donor must be accepted"


def test_zero_drops_reproduce_parent_maps(pairs, report, assignment):
    """With zero drops the #2329 maps ARE the parent maps (rewires empty);
    with drops, every rewire is recorded and every parent edge between
    survivors is kept verbatim."""
    assign, rewires = assignment
    parent = B2162.donor_assignment_2162(pairs)
    if not report.dropped_ids:
        assert assign == parent
        assert rewires == {"shuffled": {}, "crosstype": {}}
        # Parent balance property holds verbatim (parent acceptance shape).
        donors = list(assign["shuffled"].values())
        assert len(set(donors)) == len(donors), "shuffled donor map is not a permutation"
    else:
        surv = set(assign["shuffled"])
        for arm in ("shuffled", "crosstype"):
            for r_id, d_id in assign[arm].items():
                if r_id not in rewires[arm]:
                    assert d_id == parent[arm][r_id], (arm, r_id)
                else:
                    assert rewires[arm][r_id]["old"] == parent[arm][r_id]
                    assert rewires[arm][r_id]["new"] == d_id
                    assert d_id in surv


def test_dropped_pair_rewire_branch(pairs, pairs_by_id):
    """Exercise the rewire branch on a SYNTHETIC drop (the real bank may have
    zero drops): the dropped pair exits both maps as recipient AND donor, its
    orphaned recipients get constraint-satisfying rewired donors, and the
    result is deterministic across calls."""
    parent = B2162.donor_assignment_2162(pairs)
    victim = next(p for p in pairs if p.cell == "instr_format" and p.carrier == "d1")
    dropped = frozenset({victim.pair_id})
    assign, rewires = B.donor_assignment_2329(pairs, dropped)
    assign2, rewires2 = B.donor_assignment_2329(pairs, dropped)
    assert assign == assign2 and rewires == rewires2, "rewire is not deterministic"

    for arm in ("shuffled", "crosstype"):
        assert victim.pair_id not in assign[arm]
        assert victim.pair_id not in set(assign[arm].values())
        orphans = {r for r, d in parent[arm].items() if d == victim.pair_id and r != victim.pair_id}
        assert orphans <= set(rewires[arm]), (arm, orphans, set(rewires[arm]))
    # Rewired shuffled donors satisfy the parent row constraints.
    for r_id, rw in rewires["shuffled"].items():
        r, d = pairs_by_id[r_id], pairs_by_id[rw["new"]]
        assert d.cell == r.cell
        assert B._shuffled_donor_ok(r, d)
    for r_id, rw in rewires["crosstype"].items():
        r, d = pairs_by_id[r_id], pairs_by_id[rw["new"]]
        assert d.cell != r.cell


# ── acceptance test 2: engaging-carrier pair floor (structure) ────────


def test_engaging_pair_floor():
    counts = B2162.designed_separable_counts()
    for cell in B2162.all_cells():
        base = B2162.base_type_of(cell)
        if base == "filler_swap":
            assert counts[cell] == 0
            continue
        assert counts[cell] >= 12, (cell, counts[cell])
        if B2162.CARRIER_CLASS[base] == "E":
            assert counts[cell] == 36, (cell, counts[cell])


# ── token-identity policy (divergence 9, gate 0a) ─────────────────────


def test_token_identity_intact_floor(report):
    """Every cell holds 36 pairs and clears the 30/36 intact floor on the real
    bank (gate 0a PASS shape); the per-cell table is complete."""
    assert set(report.per_cell) == set(B2162.all_cells())
    for cell, row in report.per_cell.items():
        assert row["n_pairs"] == B.PAIRS_PER_CELL, (cell, row)
        assert row["n_intact"] + row["n_dropped"] == row["n_pairs"], (cell, row)
        assert row["n_intact"] >= B.INTACT_FLOOR_PER_CELL, (
            f"{cell}: {row['n_intact']}/36 intact — below the gate-0a floor "
            f"({B.INTACT_FLOOR_PER_CELL}); dropped={row['dropped']}"
        )
    B.assert_intact_floor(report)  # must not raise on the real bank


def test_intact_floor_halt_branch(report):
    """The gate-0a HALT branch fires on a synthetic sub-floor cell (degenerate
    -input probe: the floor gate must actually raise, not just exist)."""
    bad_cell = B2162.all_cells()[0]
    per_cell = {c: dict(row) for c, row in report.per_cell.items()}
    per_cell[bad_cell] = {
        "n_pairs": 36,
        "n_intact": B.INTACT_FLOOR_PER_CELL - 1,
        "n_dropped": 36 - (B.INTACT_FLOOR_PER_CELL - 1),
        "dropped": [],
    }
    fake = dataclasses.replace(report, per_cell=per_cell)
    with pytest.raises(B.TokenIdentityFloorError) as exc:
        B.assert_intact_floor(fake)
    assert bad_cell in str(exc.value)
    assert bad_cell in exc.value.offenders


# ── acceptance test 3: span-locus registry (token grain, Qwen3.5) ─────


def test_span_locus_registry(tok35, pairs, contexts, report):
    """Every SURVIVING pair satisfies its locus's token-identity checks under
    the thinking-off Qwen3.5 renders; all four registry loci are exercised."""
    im_start = tok35.convert_tokens_to_ids(B2162.IM_START)
    assert isinstance(im_start, int) and im_start >= 0
    checked: dict[str, int] = {}
    for p in pairs:
        if p.pair_id in report.dropped_ids:
            continue
        ids_a, ids_b = report.ctx_ids[p.a], report.ctx_ids[p.b]
        pe_a, pe_b = report.prefix_ends[p.a], report.prefix_ends[p.b]
        prefix_same = ids_a[:pe_a] == ids_b[:pe_b]
        final_same = ids_a[pe_a:] == ids_b[pe_b:]
        locus = B2162.span_locus(p.cell)
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
            occ_a = [i for i, t in enumerate(ids_a) if t == im_start]
            occ_b = [i for i, t in enumerate(ids_b) if t == im_start]
            assert occ_a[-1] == occ_b[-1], (p.pair_id, occ_a[-1], occ_b[-1])
            cut = occ_a[-1] + 1
            assert ids_a[:cut] == ids_b[:cut], f"{p.pair_id}: differs before the header"
            assert ids_a[cut:] != ids_b[cut:], f"{p.pair_id}: header span does NOT differ"
    assert set(checked) == {"prefix-side", "prefix+query", "final-query", "generation-header"}, (
        checked
    )


# ── realized generation header + role-header render (divergence 1) ────

_EXPECTED_HEADER_TOKENS = ["<|im_start|>", "assistant", "Ċ", "<think>", "ĊĊ", "</think>", "ĊĊ"]


def test_generation_header_token_id_pins(tok35):
    """The realized thinking-off generation header tokenizes to the pinned
    token sequence, and the standalone header ids equal the tail of a real
    render (special-token seams make the tail exact)."""
    ids = B.generation_header_ids(tok35, "assistant")
    assert tok35.convert_ids_to_tokens(ids) == _EXPECTED_HEADER_TOKENS
    msgs_ctx = {"system": "You are terse.", "history": [], "user": "Hello"}
    rendered_ids = B.context_token_ids_2329(tok35, msgs_ctx)
    assert rendered_ids[-len(ids) :] == ids


def test_role_header_render(tok35, contexts):
    """persona_role_header custom render re-derived: the swapped render ends
    with the FULL thinking-off header `<|im_start|>{role}\\n<think>\\n\\n</think>\\n\\n`."""
    cid = B2162.context_id("persona_role_header", "v1", "n1")
    rendered = B.render_context_2329(tok35, contexts[cid])
    assert rendered.endswith(f"{B2162.IM_START}pirate_assistant\n{B.THINK_BLOCK}")
    plain = B.render_context_2329(
        tok35, contexts[B2162.context_id("persona_role_header", "v2", "n1")]
    )
    assert plain.endswith(f"{B2162.IM_START}assistant\n{B.THINK_BLOCK}")
    # Token grain: the swapped render's tail ids equal the standalone header ids.
    ids = tok35(rendered, add_special_tokens=False)["input_ids"]
    header_ids = B.generation_header_ids(tok35, "pirate_assistant")
    assert ids[-len(header_ids) :] == header_ids


# ── template seam (divergence 1, pure-additive) ───────────────────────

_SEAM_CONTEXTS = [
    {"system": "You are terse.", "history": [], "user": "What is 2+2?"},
    {
        "system": None,
        "history": [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice."},
        ],
        "user": "What's my name?",
    },
]


def test_template_kwargs_default_byte_identical(tok25):
    """template_kwargs=None reproduces the parent call byte-identically on the
    parent (Qwen2.5) tokenizer — the seam is inert by default."""
    for ctx in _SEAM_CONTEXTS:
        direct = tok25.apply_chat_template(
            bank2094.context_messages_2094(ctx), tokenize=False, add_generation_prompt=True
        )
        assert bank2094.render_context_2094(tok25, ctx) == direct
        assert bank2094.render_context_2094(tok25, ctx, template_kwargs=None) == direct
        ids_default = bank2094.context_token_ids_2094(tok25, ctx)
        ids_none = bank2094.context_token_ids_2094(tok25, ctx, template_kwargs=None)
        assert ids_default == ids_none == tok25(direct, add_special_tokens=False)["input_ids"]


def test_template_kwargs_threads_enable_thinking(tok35):
    """{"enable_thinking": False} threads through the seam: the render ends
    with the EMPTY think block, while the default render leaves it open."""
    ctx = _SEAM_CONTEXTS[0]
    off = bank2094.render_context_2094(tok35, ctx, template_kwargs={"enable_thinking": False})
    default = bank2094.render_context_2094(tok35, ctx)
    assert off.endswith(B.THINK_BLOCK)
    assert not default.endswith(B.THINK_BLOCK)
    assert B.render_context_2329(tok35, ctx) == off


# ── prefix-end fallback (no default system under Qwen3.5) ─────────────


def test_no_prefix_fallback(tok35, contexts, report):
    """Bare single-turn contexts (persona_role_header all; persona_prompted v2)
    get prefix_end 0; every >= 3-occurrence context matches the parent
    ``prefix_end_index_multi`` verbatim."""
    expected_bare = {
        B2162.context_id("persona_role_header", v, c)
        for v in ("v1", "v2", "v3")
        for c in B2162.carriers_for("persona_role_header")
    } | {
        B2162.context_id("persona_prompted", "v2", c)
        for c in B2162.carriers_for("persona_prompted")
    }
    realized_bare = {cid for cid, pe in report.prefix_ends.items() if pe == 0}
    assert realized_bare == expected_bare, (
        len(realized_bare ^ expected_bare),
        sorted(realized_bare ^ expected_bare)[:5],
    )
    # A with-system context takes the parent mechanics verbatim.
    cid = B2162.context_id("instr_format", "v1", B2162.carriers_for("instr_format")[0])
    ids = report.ctx_ids[cid]
    assert report.prefix_ends[cid] == bank2094.prefix_end_index_multi(tok35, ids) > 0
