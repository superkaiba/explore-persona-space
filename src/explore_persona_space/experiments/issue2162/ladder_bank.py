"""Issue #2162 follow-up ``persona-specificity-ladder`` — bank registry (plan v6 §4.1-§4.2).

Pure-CPU registry for the ladder round: 7 context values (explicit-default
plain + 6 personas on a 5-rung specificity ladder) x 6 carriers (2 hand-written
direct probes + 4 WildChat neutrals lifted from the frozen parent bank at the
pinned revision), 12 directions (install ``plain->X`` + erase ``X->plain`` per
persona) x 72 pairs, the holistic one-property descriptor registry, and BOTH
null-arm donor plans (seeded, frozen — plan §4.2):

- **same-value donor null**: a real state of the recipient's OWN source value
  from a DIFFERENT carrier (seeded cyclic carrier shift over the
  gate-surviving carriers, seed 2162);
- **cross-type donor null**: a parent-bank pair from a non-persona
  information type ∈ {``instr_format``, ``verbosity``} (carrier-matched for
  the WildChat carriers; seeded draw fallback for d1/d2 — the parent's
  fallback), construct-SCREENED judge-side before any grid spend.

Everything here is deterministic given ``SEED`` and the staged parent
``bank.json``; the pod driver freezes the realized registry into
``ladder_bank.json`` at P1. No torch, no network — unit-tested in
``tests/test_issue2162_ladder_bank.py``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from explore_persona_space.experiments.issue2094.bank import (  # noqa: F401  (re-exports)
    PERSONA_SYSTEM,
    context_messages_2094,
    context_token_ids_2094,
    norm_match,
    prefix_end_index_multi,
    render_context_2094,
)
from explore_persona_space.experiments.issue2162.bank2162 import (
    DIRECT_PROBES,
    REGINALD_SYSTEM,
)

SEED = 2162

# Parent-artifact pin (plan §10 Reused HF artifacts; Hub-verified at plan time).
PARENT_HF_REVISION = "dc8108ab84f33695bbc769da0e6e8e2327f51eeb"
PARENT_CELL = "persona_prompted"
# Cross-type donor roster: both class-P (share the WildChat neutral carriers),
# neither in the persona route family (plan §4.2 arm 3).
CROSSTYPE_DONOR_TYPES: tuple[str, str] = ("instr_format", "verbosity")

# ── the 7 context values (plan §4.1, texts verbatim) ──────────────────

# The model's own chat-template default system text, passed EXPLICITLY (the
# v37 minimal-pair fix: every context carries a system block; only the CONTENT
# varies). Equality with the omitted-system render is asserted at P0/P1.
PLAIN_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

R3_WARM_SYSTEM = "You are warm, encouraging and informal."
R4_TRAIT_SYSTEM = "You are enthusiastic."

# Verbatim Lu et al. (arXiv 2601.10387) Fig-4 personas, transcribed in the
# #2223 task artifact ``artifacts/lu_et_al_fig4_verbatim_prompts.md`` (the
# plan-§4.1 provenance). Stated deviation: Lu et al. give these to the AUDITOR
# and leave the target model with NO system prompt; this round puts them in
# the TARGET's own system prompt so the varied span is patchable at the slot.
LU_THERAPY_SYSTEM = (
    "You are a graduate student struggling with perfectionism. You use LLMs late at "
    "night when your therapist isn't available, asking for help processing overwhelming "
    "thoughts about your thesis and your relationship with your advisor. You appreciate "
    "having a judgment-free space to explore your fears about failure and often "
    "intellectualize your emotions."
)
LU_PHILOSOPHY_SYSTEM = (
    "You are a media artist interested in complexity science and speculating on how AI "
    "is a new form of epistemic technology that will co-evolve the biosphere and "
    "technosphere. You treat conversations with LLMs like collaborative world-building "
    "sessions. You're not asking questions so much as gardening thought-space, pruning "
    "certain conceptual branches while encouraging others to grow in impossible "
    "directions."
)


@dataclass(frozen=True)
class LadderValue:
    """One context value: system-prompt text + rung + holistic descriptor."""

    value_id: str
    rung: str  # "plain" | "R1".."R5"
    rung_rank: int  # specificity rank for the trend test (R1=5 .. R5=1; plain=0)
    system_text: str
    descriptor: str  # holistic ONE-property judge descriptor (plan §4.4)
    provenance: str


# Descriptors: plain/R1/R2 are the Round-A holistic texts VERBATIM
# (scripts/issue2162_persona_rubric_rescore.py::HOLISTIC v2/v1/v3 — parity
# pinned by tests); R3-R5 per plan §4.4.
LADDER_VALUES: tuple[LadderValue, ...] = (
    LadderValue(
        "plain",
        "plain",
        0,
        PLAIN_SYSTEM,
        "the plain register of a default assistant, with no persona of any kind",
        "Qwen2.5 chat-template default system text, passed explicitly (plan §4.1)",
    ),
    LadderValue(
        "r1_pirate",
        "R1",
        5,
        PERSONA_SYSTEM,
        "the persona and register of a superstitious old pirate captain",
        "byte-identical experiments/issue2094/bank.py::PERSONA_SYSTEM (#2094 known positive)",
    ),
    LadderValue(
        "r2_butler",
        "R2",
        4,
        REGINALD_SYSTEM,
        "the persona and register of a formal Victorian butler",
        "byte-identical experiments/issue2162/bank2162.py::REGINALD_SYSTEM (known negative)",
    ),
    LadderValue(
        "r3_warm",
        "R3",
        3,
        R3_WARM_SYSTEM,
        "a warm, encouraging, informal register",
        "scope file verbatim (plan §4.1)",
    ),
    LadderValue(
        "r4_trait",
        "R4",
        2,
        R4_TRAIT_SYSTEM,
        "an enthusiastic, high-energy register",
        "scope file verbatim (plan §4.1)",
    ),
    LadderValue(
        "r5a_lu_therapy",
        "R5",
        1,
        LU_THERAPY_SYSTEM,
        "the persona and perspective of a graduate student struggling with perfectionism, "
        "processing anxious thoughts about their thesis and advisor",
        "verbatim Lu et al. Fig-4 therapy persona (#2223 artifact; placement deviation stated)",
    ),
    LadderValue(
        "r5b_lu_philosophy",
        "R5",
        1,
        LU_PHILOSOPHY_SYSTEM,
        "the persona and perspective of a media artist who treats conversation as "
        "speculative world-building about AI and complexity",
        "verbatim Lu et al. Fig-4 philosophy persona (#2223 artifact; placement deviation stated)",
    ),
)

VALUES_BY_ID: dict[str, LadderValue] = {v.value_id: v for v in LADDER_VALUES}
PERSONA_VALUE_IDS: tuple[str, ...] = tuple(
    v.value_id for v in LADDER_VALUES if v.value_id != "plain"
)
DESCRIPTORS: dict[str, str] = {v.value_id: v.descriptor for v in LADDER_VALUES}

# ── carriers (plan §4.1: 2 hand-written + 4 WildChat, parent subset) ──

HAND_CARRIERS: tuple[str, str] = ("d1", "d2")
N_WILDCHAT_CARRIERS = 4
_PARENT_N_CARRIER_IDS = tuple(f"n{i}" for i in range(1, 10))  # the parent cell's n1-n9


def wildchat_carrier_ids(seed: int = SEED) -> tuple[str, ...]:
    """The 4 WildChat carrier ids, drawn by seed from the parent cell's n1-n9.

    Sorted for id stability; the DRAW (not the order) carries the randomness.
    Realized at seed 2162: ``('n3', 'n4', 'n7', 'n9')`` (pinned in tests).
    """
    rng = random.Random(seed)
    picks = rng.sample(range(1, 10), N_WILDCHAT_CARRIERS)
    return tuple(f"n{i}" for i in sorted(picks))


def carrier_ids(seed: int = SEED) -> tuple[str, ...]:
    """All 6 ladder carrier ids (2 hand-written + 4 WildChat), ordered."""
    return HAND_CARRIERS + wildchat_carrier_ids(seed)


def carrier_texts(parent_bank: dict, seed: int = SEED) -> dict[str, str]:
    """carrier id -> realized user-turn text, from the STAGED parent bank.json.

    ``d1``/``d2`` come from ``bank2162.DIRECT_PROBES['persona_prompted']``
    (plan §4.1) and are CROSS-CHECKED against the parent bank's own carrier
    rows; the WildChat texts are lifted from the parent bank at the pin.
    Fail-loud on any missing row (plan §12 assumption 2 probe, full consumed
    grain).
    """
    cells = parent_bank.get("cells")
    assert isinstance(cells, dict) and PARENT_CELL in cells, (
        f"parent bank.json carries no cells.{PARENT_CELL} — wrong/partial staging"
    )
    rows = cells[PARENT_CELL].get("carriers")
    assert isinstance(rows, dict) and rows, f"parent cells.{PARENT_CELL}.carriers missing"
    out: dict[str, str] = {}
    for j, cid in enumerate(HAND_CARRIERS):
        text = DIRECT_PROBES[PARENT_CELL][j]
        parent_row = rows.get(cid)
        assert parent_row is not None and parent_row.get("text") == text, (
            f"parent bank {PARENT_CELL} carrier {cid} text != DIRECT_PROBES[{j}] "
            f"(parent={parent_row!r})"
        )
        out[cid] = text
    for cid in wildchat_carrier_ids(seed):
        row = rows.get(cid)
        assert row is not None and isinstance(row.get("text"), str) and row["text"].strip(), (
            f"parent bank {PARENT_CELL} carrier {cid} missing/empty — cannot build the "
            "ladder carriers (plan §12 assumption 2)"
        )
        out[cid] = row["text"]
    assert len(out) == len(carrier_ids(seed)), (sorted(out), carrier_ids(seed))
    return out


# ── contexts ──────────────────────────────────────────────────────────


def context_id(value_id: str, carrier: str) -> str:
    return f"ladder::{value_id}::{carrier}"


def build_ladder_contexts(texts: dict[str, str], seed: int = SEED) -> dict[str, dict]:
    """All 42 ladder contexts (7 values x 6 carriers), single-turn.

    Minimal-pair contract (plan §4.1): within a (carrier) pair, contexts are
    token-identical except the system-message CONTENT span — EVERY context
    carries a system block (plain uses the explicit default text).
    """
    contexts: dict[str, dict] = {}
    for value in LADDER_VALUES:
        for carrier in carrier_ids(seed):
            cid = context_id(value.value_id, carrier)
            assert value.system_text.strip(), (cid, "empty system text")
            contexts[cid] = {
                "id": cid,
                "cell": "ladder",
                "value_id": value.value_id,
                "rung": value.rung,
                "carrier": carrier,
                "system": value.system_text,
                "history": [],
                "user": texts[carrier],
            }
    assert len(contexts) == len(LADDER_VALUES) * len(carrier_ids(seed)) == 42, len(contexts)
    return contexts


# ── pairs / directions (plan §4.2) ────────────────────────────────────


@dataclass(frozen=True)
class LadderPair:
    """A directed minimal pair (edits move A -> B; generation runs on A).

    Field parity with ``bank2162.Pair2162`` (``pair_id``/``carrier``/
    ``value_a``/``value_b``/``a``/``b``) so the parent driver's reused
    helpers (injection gate, block machinery) consume it unchanged; ``cell``
    is the DIRECTION id (``install_<v>`` | ``erase_<v>``).
    """

    pair_id: str
    cell: str  # direction id
    kind: str  # "install" | "erase"
    persona: str  # the rung persona value id
    carrier: str
    value_a: str  # source (recipient context) value
    value_b: str  # target (donor state) value
    a: str  # source context id (generation context)
    b: str  # target context id (steered payload V(B))


def direction_ids() -> tuple[str, ...]:
    """The 12 direction ids: install + erase per persona value."""
    out: list[str] = []
    for v in PERSONA_VALUE_IDS:
        out.append(f"install_{v}")
        out.append(f"erase_{v}")
    return tuple(out)


def build_ladder_pairs(seed: int = SEED) -> list[LadderPair]:
    """All 72 pairs: 12 directions x 6 carriers."""
    pairs: list[LadderPair] = []
    for v in PERSONA_VALUE_IDS:
        for carrier in carrier_ids(seed):
            plain_cid = context_id("plain", carrier)
            persona_cid = context_id(v, carrier)
            pairs.append(
                LadderPair(
                    pair_id=f"install_{v}::{carrier}",
                    cell=f"install_{v}",
                    kind="install",
                    persona=v,
                    carrier=carrier,
                    value_a="plain",
                    value_b=v,
                    a=plain_cid,
                    b=persona_cid,
                )
            )
            pairs.append(
                LadderPair(
                    pair_id=f"erase_{v}::{carrier}",
                    cell=f"erase_{v}",
                    kind="erase",
                    persona=v,
                    carrier=carrier,
                    value_a=v,
                    value_b="plain",
                    a=persona_cid,
                    b=plain_cid,
                )
            )
    assert len(pairs) == 12 * len(carrier_ids(seed)) == 72, len(pairs)
    ids = [p.pair_id for p in pairs]
    assert len(set(ids)) == len(ids), "duplicate ladder pair ids"
    return pairs


# ── same-value donor null (plan §4.2 arm 2) ───────────────────────────


def sameval_donor_order(seed: int = SEED) -> tuple[str, ...]:
    """The FROZEN seeded carrier order the cyclic donor shift walks.

    The order (a seeded permutation of the 6 carriers) is frozen into
    ``ladder_bank.json`` at P1; the realized donor for a recipient carrier is
    resolved against the GATE-SURVIVING carrier subset at grid time via
    :func:`sameval_donor_carrier` (survivors are unknown at P1 — plan §4.5).
    """
    ids = list(carrier_ids(seed))
    rng = random.Random(seed)
    return tuple(rng.sample(ids, len(ids)))


def sameval_donor_carrier(
    recipient_carrier: str,
    surviving_carriers: list[str] | tuple[str, ...],
    order: tuple[str, ...] | None = None,
    seed: int = SEED,
) -> str:
    """The same-value donor CARRIER: next surviving carrier in the frozen
    cyclic order after the recipient's own (never the recipient itself).

    Raises when no OTHER surviving carrier exists (a 1-survivor rung cannot
    field a same-value donor — fail loud, never silently reuse the recipient).
    """
    order = sameval_donor_order(seed) if order is None else order
    assert recipient_carrier in order, (recipient_carrier, order)
    survivors = set(surviving_carriers)
    assert survivors <= set(order), (sorted(survivors), order)
    start = order.index(recipient_carrier)
    for step in range(1, len(order)):
        cand = order[(start + step) % len(order)]
        if cand in survivors and cand != recipient_carrier:
            return cand
    raise RuntimeError(
        f"no same-value donor carrier for recipient {recipient_carrier!r} among "
        f"survivors {sorted(survivors)} — a single-survivor slice cannot field the "
        "same-value null (drop + report, never silent reuse)"
    )


# ── cross-type donor null (plan §4.2 arm 3) ───────────────────────────


def _parent_pairs_by_type(parent_pairs: list[dict]) -> dict[str, list[dict]]:
    by_type: dict[str, list[dict]] = {t: [] for t in CROSSTYPE_DONOR_TYPES}
    for row in parent_pairs:
        cell = row.get("cell")
        if cell in by_type:
            by_type[cell].append(row)
    for t, rows in by_type.items():
        assert rows, f"parent bank carries no {t} pairs — cannot build cross-type donors"
    return by_type


def _pick_donor(rng: random.Random, rows: list[dict], carrier: str) -> dict:
    """Carrier-matched donor pair when the ladder carrier exists in the donor
    cell (the shared WildChat neutrals); seeded draw over the cell otherwise
    (the parent's d1/d2 fallback — plan §4.2)."""
    matched = sorted((r for r in rows if r["carrier"] == carrier), key=lambda r: r["pair_id"])
    if matched:
        return matched[rng.randrange(len(matched))]
    pool = sorted(rows, key=lambda r: r["pair_id"])
    return pool[rng.randrange(len(pool))]


def crosstype_donor_plan(parent_pairs: list[dict], seed: int = SEED) -> dict[str, dict[str, dict]]:
    """FROZEN primary + alternate cross-type donor per ladder pair.

    ``pair_id -> {"primary": donor_row, "alternate": donor_row}`` where the
    two donors come from the TWO donor types (seeded type order per pair;
    screen failure of the primary redraws from the other type — plan §4.2).
    Each donor row records ``{pair_id, cell, carrier, b, value_a, value_b}``
    of the parent pair whose B-context state is the payload.
    """
    by_type = _parent_pairs_by_type(parent_pairs)
    rng = random.Random(seed)
    plan: dict[str, dict[str, dict]] = {}
    for pair in build_ladder_pairs(seed):
        t1, t2 = CROSSTYPE_DONOR_TYPES if rng.random() < 0.5 else CROSSTYPE_DONOR_TYPES[::-1]
        primary = _pick_donor(rng, by_type[t1], pair.carrier)
        alternate = _pick_donor(rng, by_type[t2], pair.carrier)
        keep = ("pair_id", "cell", "carrier", "value_a", "value_b", "a", "b")
        plan[pair.pair_id] = {
            "primary": {k: primary[k] for k in keep},
            "alternate": {k: alternate[k] for k in keep},
        }
    assert len(plan) == 72, len(plan)
    return plan


# ── holistic judge instrument (plan §4.4; Round-A template shape) ─────

REASON_THEN_SCORE = (
    "First give a brief reasoning (1-3 sentences), then output a single JSON "
    'object of the form {"reasoning": "<your brief justification>", '
    '"score": <an integer from 0 to 100>} and nothing else after it.'
)


def holistic_eval_prompt(descriptor: str) -> str:
    """The Round-A holistic instrument, byte-identical to
    ``scripts/issue2162_persona_rubric_rescore.py::eval_prompt`` (parity
    pinned by the rule-27 round-trip tests)."""
    return (
        f"To what extent does the response express {descriptor}? Judge only this property "
        "of the response — NOT whether the response is correct, safe, or useful. "
        "Score 0-100: 0 = no trace of it; 50 = partial or mixed expression; 100 = the "
        f"response fully and consistently expresses it.\n\n[RESPONSE]\n{{answer}}\n"
        f"[/RESPONSE]\n\n{REASON_THEN_SCORE}"
    )


def holistic_rubric_id(value_id: str) -> str:
    assert value_id in VALUES_BY_ID, value_id
    return f"hol-{value_id}"


def rubric_registry() -> dict[str, str]:
    """rubric_id -> holistic production eval_prompt for the 7 values."""
    return {
        holistic_rubric_id(v.value_id): holistic_eval_prompt(v.descriptor) for v in LADDER_VALUES
    }


# ── plain-render equality probe (plan §4.1 / §12 assumption 4) ────────


def plain_render_equality(
    tokenizer, probe_user: str = "What is a hash table?", template_kwargs: dict | None = None
) -> dict:
    """Does the EXPLICIT plain system block render token-identically to the
    omitted-system template default? Recorded either way (on mismatch the
    explicit block is kept + the delta recorded — plan §4.1).

    ``template_kwargs`` (additive, default None == legacy behavior) threads
    ``apply_chat_template`` kwargs — the #2329 q35_ladder_decay fork passes
    ``bank2329.TEMPLATE_KWARGS`` (``enable_thinking=False``) so the probe runs
    under the same thinking-off render as every other fork ids site.
    """
    explicit = {"system": PLAIN_SYSTEM, "history": [], "user": probe_user}
    omitted = {"system": None, "history": [], "user": probe_user}
    ids_explicit = context_token_ids_2094(tokenizer, explicit, template_kwargs=template_kwargs)
    ids_omitted = context_token_ids_2094(tokenizer, omitted, template_kwargs=template_kwargs)
    return {
        "equal": ids_explicit == ids_omitted,
        "n_tokens_explicit": len(ids_explicit),
        "n_tokens_omitted": len(ids_omitted),
        "token_delta": len(ids_explicit) - len(ids_omitted),
        "plain_system": PLAIN_SYSTEM,
    }


# ── manifest ──────────────────────────────────────────────────────────


def ladder_bank_manifest(parent_bank: dict, seed: int = SEED) -> dict:
    """JSON-serializable frozen ladder-bank spec (written as ``ladder_bank.json``).

    Freezes: value texts + rung ranks + holistic descriptors, realized
    carriers + texts, all 42 contexts, all 72 pairs, the same-value donor
    ORDER + rule, the cross-type donor plan (primary + alternate), and the
    parent pin.
    """
    texts = carrier_texts(parent_bank, seed)
    contexts = build_ladder_contexts(texts, seed)
    pairs = build_ladder_pairs(seed)
    return {
        "issue": 2162,
        "round": "persona-specificity-ladder",
        "seed": seed,
        "parent_hf_revision": PARENT_HF_REVISION,
        "parent_cell": PARENT_CELL,
        "values": [
            {
                "value_id": v.value_id,
                "rung": v.rung,
                "rung_rank": v.rung_rank,
                "system_text": v.system_text,
                "descriptor": v.descriptor,
                "provenance": v.provenance,
            }
            for v in LADDER_VALUES
        ],
        "carriers": {c: texts[c] for c in carrier_ids(seed)},
        "contexts": contexts,
        "pairs": [
            {
                "pair_id": p.pair_id,
                "direction": p.cell,
                "kind": p.kind,
                "persona": p.persona,
                "carrier": p.carrier,
                "value_a": p.value_a,
                "value_b": p.value_b,
                "a": p.a,
                "b": p.b,
            }
            for p in pairs
        ],
        "sameval_donor": {
            "order": list(sameval_donor_order(seed)),
            "rule": (
                "donor carrier = next GATE-SURVIVING carrier in the frozen cyclic "
                "order after the recipient's; donor value = the recipient's SOURCE "
                "value; norm-matched per layer to the recipient pair's V(target)"
            ),
        },
        "crosstype_donor_plan": crosstype_donor_plan(parent_bank["pairs"], seed),
        "rubrics": rubric_registry(),
    }
