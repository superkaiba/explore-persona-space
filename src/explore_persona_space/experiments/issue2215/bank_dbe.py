"""Issue #2215 ``discrimination-battery-expansion`` — the 9-type bank registry (plan v6 §4.1).

Nine new content-level pair types, emitted in the EXACT ``bank.json`` schema the
reused analysis core consumes (``scripts/issue2215_analysis.py::PairTable.from_bank`` /
``build_cell_views``): ``contexts: {cid: {cell, value_id, carrier, ...}}`` +
``pairs: [{pair_id, cell, carrier, value_a, value_b, a, b}]`` with COMPLETE
(carrier x value-pair) grids per cell — constructed cells 12 carriers x 3
value-pairs, benchmark cells N items x 1 value-pair (N <= 36).

The realized TEXTS come from Phase G datagen (``scripts/issue2215_dbe_datagen.py``)
and are frozen into ``bank_dbe_values.json`` next to this module (the
``frozen_gen_2162.json`` precedent — ``src/`` travels with every clone, so the
pod reads the bank via the pushed branch with no HF fetch). This module holds
the frozen registry constants (type table, value strings, final queries, the
programmatic ``user_doc_format`` renderers), the message builders honoring the
plan's pinned turn structures, the expected-pe-eligibility declaration
(7 eligible / 2 degenerate), and the ``bank_dbe.json`` emitter/writer.

Turn structures (plan §4.1, pinned):

* rows 1/2/3/4/5/7/9 place the varying span in a PRE-QUERY user turn
  (pe-ELIGIBLE); rows 2/4 carriers have >=2 user turns by construction;
* rows 6 (``user_sentiment``) and 8 (``refusal_request``) are single-turn —
  the varying span IS the final user turn, so both are DEGENERATE at
  prefix-end (pair sides share the prefix; same flag machinery as the parent
  bank's ``query_content`` / ``refusal_request`` degenerate cells).

Content hygiene: the ``refusal_request`` cell embeds XSTest prompts
(``Paul/XSTest`` ``xstest_prompts.csv``, 450 rows) — referenced here by
filename + row counts only; realized prompt text lives only inside the frozen
``bank_dbe_values.json`` the datagen run writes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from explore_persona_space.experiments.issue2094.bank import context_messages_2094

ISSUE = 2215
ROUND = "discrimination-battery-expansion"
SEED = 2215
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

VALUES_FILENAME = "bank_dbe_values.json"

# ── the 9 types (plan §4.1 table, row order) ──────────────────────────

TYPES: tuple[str, ...] = (
    "user_role_identity",
    "style_register",
    "conversation_topic",
    "conversation_language",
    "fact_truth",
    "user_sentiment",
    "user_doc_format",
    "refusal_request",
    "code_vs_prose",
)

CONSTRUCTED_TYPES: tuple[str, ...] = (
    "user_role_identity",
    "style_register",
    "conversation_topic",
    "conversation_language",
    "fact_truth",
    "user_doc_format",
    "code_vs_prose",
)

BENCHMARK_TYPES: tuple[str, ...] = ("user_sentiment", "refusal_request")

# Rows 6/8: the varying span is the single final user turn — pair sides share
# the rendered prefix, so the pe read is undefined by construction (plan §4.1
# expected-pe-eligibility block; the B1' mechanical check is authoritative).
DEGENERATE_AT_PE: frozenset[str] = frozenset({"user_sentiment", "refusal_request"})

N_CARRIERS_CONSTRUCTED = 12
# ceil(29 / 3): the fewest carriers under which a kept 3-value cell can still
# clear the gate-1 pair floor (each carrier carries 3 pairs).
MIN_CARRIERS_CONSTRUCTED = 10
N_ITEMS_BENCHMARK = 36
# Gate 1 (plan §7): kept judge-PASS pairs >= 29 of 36 (80% relative floor)
# after one regeneration tranche, else the type is DROPPED and reported.
PAIR_FLOOR = 29

# ── frozen value registries ───────────────────────────────────────────

# Row 1: frozen self-intro identity strings (plan §4.1, verbatim).
ROLE_IDENTITY_VALUES: dict[str, str] = {
    "v1": "an emergency-room nurse",
    "v2": "a criminal-defense lawyer",
    "v3": "a middle-school teacher",
}
ROLE_IDENTITY_SLOT = "{identity}"

# Row 2: register of ALL user turns (content held; v2 = the real seed).
REGISTER_VALUES: dict[str, str] = {"v1": "formal", "v2": "original", "v3": "casual"}

# Row 3: three domains realizing the SAME conversation skeleton.
TOPIC_DOMAINS: dict[str, str] = {"v1": "cooking", "v2": "personal finance", "v3": "houseplants"}
TOPIC_FINAL_QUERY = "Could you recap your main advice in one sentence?"

# Row 4: whole-conversation user-turn language ({en, es, fr} matches the
# parent instr_language/language_implied value set for cross-cell reads).
LANGUAGE_VALUES: dict[str, str] = {"v1": "en", "v2": "es", "v3": "fr"}

# Row 5: factual correctness of a well-known world fact stated by the user.
FACT_VALUES: dict[str, str] = {"v1": "correct", "v2": "subtly_false", "v3": "grossly_false"}

# Row 6: tier-2 imdb contrast set; the review + query form the SINGLE final
# user turn (plan §4.1 row 6, template verbatim; no synthetic turn-splitting).
SENTIMENT_VALUES: dict[str, str] = {"v1": "original", "v2": "contrast"}
SENTIMENT_USER_TEMPLATE = (
    "Here's my draft review: {review} — Based on my review, "
    "should I recommend this movie to a friend?"
)

# Row 7: three renderings of BIT-IDENTICAL data (renderings PROGRAMMATIC —
# tier 4 with the format-is-the-construct argument, plan §4.1 row 7).
DOC_FORMAT_VALUES: dict[str, str] = {"v1": "markdown", "v2": "json", "v3": "prose"}
DOC_PASTE_WRAPPER = "Here's the data I'm working with:\n\n{rendering}"

# Row 8: tier-2 XSTest safe/unsafe contrast as the FINAL user query.
REFUSAL_VALUES: dict[str, str] = {"v1": "safe", "v2": "unsafe"}

# Row 9: same small algorithm in three presentations.
CODE_VALUES: dict[str, str] = {"v1": "python", "v2": "pseudocode", "v3": "prose"}
CODE_PASTE_WRAPPER = "Take a look at this:\n\n{presentation}"
CODE_FINAL_QUERY = "What's one improvement you'd suggest?"

TYPE_SPEC: dict[str, dict] = {
    "user_role_identity": {
        "scope_category": "persona/identity",
        "tier": "tier-3 template insertion (LLM-drafted frozen carriers)",
        "values": ROLE_IDENTITY_VALUES,
        "length_pinned": True,
        "judge_attribute": (
            "the user's stated professional identity in the first turn "
            "(emergency-room nurse vs criminal-defense lawyer vs middle-school "
            "teacher); the rest of the conversation is identical by construction"
        ),
        "judge_tolerance": (
            "the two sides differ ONLY in the inserted identity phrase; any other "
            "difference is a FAIL"
        ),
    },
    "style_register": {
        "scope_category": "style/register",
        "tier": "tier-1 WildChat seed + tier-3 LLM register rewrite",
        "values": REGISTER_VALUES,
        "length_pinned": False,
        "judge_attribute": (
            "the register/style of the USER's turns (formal vs original vs "
            "casual slang); informational content preserved"
        ),
        "judge_tolerance": (
            "length cannot be held under a register rewrite — do not FAIL for "
            "length differences alone; FAIL if the informational content of any "
            "user turn changed or assistant turns differ"
        ),
    },
    "conversation_topic": {
        "scope_category": "topic",
        "tier": "tier-3 LLM-generated (WildChat-seeded topic styles)",
        "values": TOPIC_DOMAINS,
        "length_pinned": True,
        "judge_attribute": (
            "the ENTIRE conversation's domain (cooking vs personal finance vs "
            "houseplants) realizing the same conversation skeleton"
        ),
        "judge_tolerance": (
            "sides should share turn count, conversational moves, and roughly "
            "matched length (within ~15%); FAIL if the skeleton diverges"
        ),
    },
    "conversation_language": {
        "scope_category": "language",
        "tier": "tier-1 WildChat seed + tier-3 LLM translation",
        "values": LANGUAGE_VALUES,
        "length_pinned": False,
        "judge_attribute": (
            "the language of ALL the USER's turns including the final query "
            "(English vs Spanish vs French); content preserved; assistant turns "
            "held in English"
        ),
        "judge_tolerance": (
            "length cannot be held under translation — do not FAIL for length "
            "differences alone; FAIL if content diverges or assistant turns differ"
        ),
    },
    "fact_truth": {
        "scope_category": "factual content",
        "tier": "tier-3 LLM-drafted, frozen, validity-judged",
        "values": FACT_VALUES,
        "length_pinned": True,
        "judge_attribute": (
            "the factual correctness of the well-known world fact the user "
            "states in the pre-query turn (correct vs subtly false vs grossly "
            "false); same topic and phrasing frame"
        ),
        "judge_tolerance": (
            "sides should be within ~15% length and share the phrasing frame; "
            "the subtly-false side must be plausibly wrong, the grossly-false "
            "side obviously wrong"
        ),
    },
    "user_sentiment": {
        "scope_category": "sentiment",
        "tier": "tier-2 benchmark (compl-ai/imdb_contrastset, Gardner et al. 2004.02709)",
        "values": SENTIMENT_VALUES,
        "length_pinned": False,
        "judge_attribute": (
            "the evaluative valence of the user's movie review toward the film "
            "(positive vs negative) via human minimal edits"
        ),
        "judge_tolerance": (
            "the contrast side is a published human minimal-edit of the "
            "original — do not FAIL for the distributed small edits; FAIL only "
            "if the two reviews are not recognizably the same review or the "
            "sentiment does not flip"
        ),
    },
    "user_doc_format": {
        "scope_category": "formatting",
        "tier": "tier-4 programmatic renderings of an LLM-drafted payload "
        "(format IS the construct)",
        "values": DOC_FORMAT_VALUES,
        "length_pinned": True,
        "judge_attribute": (
            "the FORMAT in which the user's pasted data is rendered (markdown "
            "table vs JSON vs prose paragraph); the underlying data is "
            "bit-identical"
        ),
        "judge_tolerance": (
            "renderings are programmatic from one payload — FAIL only if the "
            "two sides' underlying data content differs"
        ),
    },
    "refusal_request": {
        "scope_category": "refusal-inducing",
        # XSTest (arXiv 2308.01263): HF `Paul/XSTest` `xstest_prompts.csv`,
        # 450 rows (250 safe / 200 unsafe) — referenced by filename + row
        # counts only; prompt text never inlined here.
        "tier": "tier-2 benchmark (XSTest, 2308.01263)",
        "values": REFUSAL_VALUES,
        "length_pinned": False,
        "judge_attribute": (
            "whether the user's final request is safe or unsafe "
            "(refusal-inducing), between two superficially similar prompts"
        ),
        "judge_tolerance": (
            "sides are lexically-matched published benchmark prompts — FAIL "
            "only if the safe side is not plausibly safe, the unsafe side is "
            "not plausibly refusal-inducing, or the two prompts share no "
            "surface similarity"
        ),
    },
    "code_vs_prose": {
        "scope_category": "code (#2202 hot-spot)",
        "tier": "tier-3 LLM-drafted, frozen",
        "values": CODE_VALUES,
        "length_pinned": True,
        "judge_attribute": (
            "the presentation of the SAME small algorithm/task (working Python "
            "snippet vs language-agnostic pseudocode vs plain-prose description)"
        ),
        "judge_tolerance": (
            "sides should describe the identical algorithm within ~15% length "
            "where feasible; FAIL if the algorithms differ"
        ),
    },
}
assert tuple(TYPE_SPEC) == TYPES


def value_ids(cell: str) -> tuple[str, ...]:
    return tuple(TYPE_SPEC[cell]["values"].keys())


def value_pairs(cell: str) -> tuple[tuple[str, str], ...]:
    """Unordered value pairs per carrier: 3 for 3-value cells, 1 for 2-value."""
    vids = value_ids(cell)
    return tuple((vids[i], vids[j]) for i in range(len(vids)) for j in range(i + 1, len(vids)))


def expected_pe_eligibility() -> dict[str, bool]:
    """Registered per-type expectation (True = pe-ELIGIBLE); 7 eligible / 2 degenerate.

    The §4.3 B1' mechanical rendered-prefix-token comparison is authoritative at
    full grain; a realized violation is a datagen structure bug (fail loud),
    never a droppable outcome.
    """
    out = {t: t not in DEGENERATE_AT_PE for t in TYPES}
    assert sum(out.values()) == 7 and len(out) - sum(out.values()) == 2, out
    return out


# ── ids ───────────────────────────────────────────────────────────────


def context_id(cell: str, value_id: str, carrier: str) -> str:
    return f"{cell}::{value_id}::{carrier}"


def pair_id(cell: str, va: str, vb: str, carrier: str) -> str:
    return f"{cell}::{va}-{vb}::{carrier}"


# ── programmatic user_doc_format renderers (tier 4; bit-identical data) ─


def _cell_str(v: object) -> str:
    assert isinstance(v, str | int | float), v
    return str(v)


def render_payload_markdown(payload: dict) -> str:
    cols = payload["columns"]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    lines += ["| " + " | ".join(_cell_str(v) for v in row) + " |" for row in payload["rows"]]
    return f"{payload['title']}\n\n" + "\n".join(lines)


def render_payload_json(payload: dict) -> str:
    obj = {
        "title": payload["title"],
        "records": [
            {col: row[i] for i, col in enumerate(payload["columns"])} for row in payload["rows"]
        ],
    }
    return json.dumps(obj, indent=2, ensure_ascii=False)


def render_payload_prose(payload: dict) -> str:
    cols = payload["columns"]
    parts = [
        f"{payload['title']}. This dataset has {len(payload['rows'])} records "
        f"with the fields {', '.join(cols)}."
    ]
    for k, row in enumerate(payload["rows"], start=1):
        fields = "; ".join(f"{col} is {_cell_str(row[i])}" for i, col in enumerate(cols))
        parts.append(f"Record {k}: {fields}.")
    return " ".join(parts)


def doc_renderings(payload: dict) -> dict[str, str]:
    return {
        "v1": render_payload_markdown(payload),
        "v2": render_payload_json(payload),
        "v3": render_payload_prose(payload),
    }


# ── values loading + validation ───────────────────────────────────────


def load_values(path: Path | None = None) -> dict:
    p = Path(path) if path is not None else Path(__file__).parent / VALUES_FILENAME
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found — run scripts/issue2215_dbe_datagen.py (Phase G) to freeze the bank"
        )
    values = json.loads(p.read_text())
    validate_values(values)
    return values


def _assert_turnlists(cell: str, carrier: str, car: dict) -> None:
    """rows 2/4: >=2 user turns (last = final query), assistant turns held."""
    ass = car["assistant_turns"]
    assert isinstance(ass, list) and len(ass) >= 1, (cell, carrier)
    uts = car["user_turns"]
    assert set(uts) == set(value_ids(cell)), (cell, carrier, sorted(uts))
    for vid, turns in uts.items():
        assert isinstance(turns, list) and len(turns) == len(ass) + 1, (cell, carrier, vid)
        assert len(turns) >= 2, (cell, carrier, vid, "needs >=2 user turns")
        assert all(isinstance(t, str) and t.strip() for t in turns), (cell, carrier, vid)


def _assert_nonempty_strs(cell: str, carrier: str, car: dict, keys: tuple[str, ...]) -> None:
    for key in keys:
        assert isinstance(car[key], str) and car[key].strip(), (cell, carrier, key)


def _validate_carrier(cell: str, carrier: str, car: dict) -> None:
    if cell == "user_role_identity":
        assert car["turn1_template"].count(ROLE_IDENTITY_SLOT) == 1, (cell, carrier)
        _assert_nonempty_strs(cell, carrier, car, ("assistant_ack", "final_query"))
    elif cell in ("style_register", "conversation_language"):
        _assert_turnlists(cell, carrier, car)
        # The seed slot holds the verbatim real WildChat query: v2 = original
        # for the register cell, v1 = original English for the language cell.
        original = "v2" if cell == "style_register" else "v1"
        assert car["user_turns"][original][0] == car["seed_user_turn"], (cell, carrier)
    elif cell == "conversation_topic":
        convs = car["conversations"]
        assert set(convs) == set(value_ids(cell)), (cell, carrier)
        n_user = {len(c["user_turns"]) for c in convs.values()}
        n_ass = {len(c["assistant_turns"]) for c in convs.values()}
        assert len(n_user) == 1 and len(n_ass) == 1, (cell, carrier, "skeleton drift")
        assert n_user == n_ass and next(iter(n_user)) >= 1, (cell, carrier)
    elif cell == "fact_truth":
        assert set(car["facts"]) == set(value_ids(cell)), (cell, carrier)
        _assert_nonempty_strs(cell, carrier, car, ("assistant_ack", "final_query"))
    elif cell == "user_sentiment":
        assert set(car["texts"]) == {"v1", "v2"}, (cell, carrier)
        labels = car["labels"]
        assert set(labels) == {"v1", "v2"}, (cell, carrier)
        # A16: realized polarity labels carried per side (P2 grouping).
        assert all(v in ("Positive", "Negative") for v in labels.values()), (cell, carrier, labels)
    elif cell == "user_doc_format":
        assert doc_renderings(car["payload"]) == car["renderings"], (
            cell,
            carrier,
            "stored renderings drifted from the programmatic renderers",
        )
        _assert_nonempty_strs(cell, carrier, car, ("assistant_ack", "final_query"))
    elif cell == "refusal_request":
        assert set(car["prompts"]) == {"v1", "v2"}, (cell, carrier)
    elif cell == "code_vs_prose":
        assert set(car["presentations"]) == set(value_ids(cell)), (cell, carrier)
        _assert_nonempty_strs(cell, carrier, car, ("assistant_ack",))
    else:  # pragma: no cover - TYPES is closed
        raise AssertionError(cell)


def validate_values(values: dict) -> None:
    """Fail-fast structural validation of the frozen values payload."""
    assert values.get("issue") == ISSUE and values.get("round") == ROUND, "wrong values payload"
    types = values["types"]
    assert set(types) == set(TYPES), sorted(set(TYPES) ^ set(types))
    for cell, tv in types.items():
        assert isinstance(tv.get("kept"), bool), (cell, "missing kept flag")
        carriers = tv["carriers"]
        if not tv["kept"]:
            continue
        if cell in CONSTRUCTED_TYPES:
            # A kept constructed cell may lose up to 2 carriers to two-tranche
            # generation failures (floor 29 <= 3 * 10 pairs); smoke runs 1.
            lo = 1 if values.get("smoke") else MIN_CARRIERS_CONSTRUCTED
            assert lo <= len(carriers) <= N_CARRIERS_CONSTRUCTED, (cell, len(carriers))
        else:
            assert PAIR_FLOOR <= len(carriers) <= N_ITEMS_BENCHMARK or values.get("smoke"), (
                cell,
                len(carriers),
            )
        for carrier, car in carriers.items():
            _validate_carrier(cell, carrier, car)
        judge = tv["judge"]
        for carrier in carriers:
            for va, vb in value_pairs(cell):
                pid = pair_id(cell, va, vb, carrier)
                assert pid in judge, (pid, "missing judge verdict")
                assert judge[pid]["verdict"] in ("PASS", "FAIL", "DROP"), judge[pid]


# ── message builders (plan §4.1 pinned turn structures) ───────────────


def build_context(values: dict, cell: str, value_id: str, carrier: str) -> dict:
    """One context dict in the renderer contract: {system, history, user}.

    ``system`` is None for every new cell (the chat template inserts the
    model's default system block — the parent bank's bare-prefix convention);
    ``history`` carries the pre-query turns; ``user`` is the final query.
    """
    car = values["types"][cell]["carriers"][carrier]
    history: list[dict]
    if cell == "user_role_identity":
        turn1 = car["turn1_template"].replace(ROLE_IDENTITY_SLOT, ROLE_IDENTITY_VALUES[value_id])
        history = [
            {"role": "user", "content": turn1},
            {"role": "assistant", "content": car["assistant_ack"]},
        ]
        user = car["final_query"]
    elif cell in ("style_register", "conversation_language"):
        uts = car["user_turns"][value_id]
        ass = car["assistant_turns"]
        history = []
        for k, a in enumerate(ass):
            history.append({"role": "user", "content": uts[k]})
            history.append({"role": "assistant", "content": a})
        user = uts[-1]
    elif cell == "conversation_topic":
        conv = car["conversations"][value_id]
        history = []
        for u, a in zip(conv["user_turns"], conv["assistant_turns"], strict=True):
            history.append({"role": "user", "content": u})
            history.append({"role": "assistant", "content": a})
        user = TOPIC_FINAL_QUERY
    elif cell == "fact_truth":
        history = [
            {"role": "user", "content": car["facts"][value_id]},
            {"role": "assistant", "content": car["assistant_ack"]},
        ]
        user = car["final_query"]
    elif cell == "user_sentiment":
        history = []
        user = SENTIMENT_USER_TEMPLATE.format(review=car["texts"][value_id])
    elif cell == "user_doc_format":
        history = [
            {
                "role": "user",
                "content": DOC_PASTE_WRAPPER.format(rendering=car["renderings"][value_id]),
            },
            {"role": "assistant", "content": car["assistant_ack"]},
        ]
        user = car["final_query"]
    elif cell == "refusal_request":
        history = []
        user = car["prompts"][value_id]
    elif cell == "code_vs_prose":
        history = [
            {
                "role": "user",
                "content": CODE_PASTE_WRAPPER.format(presentation=car["presentations"][value_id]),
            },
            {"role": "assistant", "content": car["assistant_ack"]},
        ]
        user = CODE_FINAL_QUERY
    else:  # pragma: no cover - TYPES is closed
        raise AssertionError(cell)
    return {
        "id": context_id(cell, value_id, carrier),
        "cell": cell,
        "value_id": value_id,
        "carrier": carrier,
        "system": None,
        "history": history,
        "user": user,
        "pe_expected_eligible": cell not in DEGENERATE_AT_PE,
    }


def context_messages_dbe(context: dict) -> list[dict]:
    """Chat message list — the parent renderer contract, reused verbatim."""
    return context_messages_2094(context)


def kept_types(values: dict) -> tuple[str, ...]:
    return tuple(t for t in TYPES if values["types"][t]["kept"])


def build_contexts_dbe(values: dict) -> dict[str, dict]:
    contexts: dict[str, dict] = {}
    for cell in kept_types(values):
        for carrier in sorted(values["types"][cell]["carriers"]):
            for vid in value_ids(cell):
                ctx = build_context(values, cell, vid, carrier)
                assert ctx["id"] not in contexts, ctx["id"]
                contexts[ctx["id"]] = ctx
    return contexts


def build_pairs_dbe(values: dict) -> list[dict]:
    """All directed pairs of the kept cells; per-pair judge verdict carried.

    Every kept cell keeps its COMPLETE (carrier x value-pair) grid — a
    judge-FAILed constructed pair stays in the bank flagged
    ``judge_valid=False`` and is excluded at analysis via ``included_pair``
    (plan §4.4 "invalid pairs are excluded-and-reported"); benchmark items
    that fail after the regeneration tranche are dropped from ``carriers`` by
    the datagen (the grid stays complete at N x 1).
    """
    pairs: list[dict] = []
    for cell in kept_types(values):
        judge = values["types"][cell]["judge"]
        for carrier in sorted(values["types"][cell]["carriers"]):
            for va, vb in value_pairs(cell):
                pid = pair_id(cell, va, vb, carrier)
                pairs.append(
                    {
                        "pair_id": pid,
                        "cell": cell,
                        "carrier": carrier,
                        "value_a": va,
                        "value_b": vb,
                        "a": context_id(cell, va, carrier),
                        "b": context_id(cell, vb, carrier),
                        "judge_valid": judge[pid]["verdict"] == "PASS",
                        "pe_expected_eligible": cell not in DEGENERATE_AT_PE,
                    }
                )
    return pairs


# ── bank.json emitter ─────────────────────────────────────────────────


def bank_manifest_dbe(values: dict) -> dict:
    """The ``bank_dbe.json`` payload in the schema ``PairTable.from_bank`` consumes."""
    validate_values(values)
    contexts = build_contexts_dbe(values)
    pairs = build_pairs_dbe(values)
    kept = kept_types(values)
    for cell in kept:
        n_car = len(values["types"][cell]["carriers"])
        n_vp = len(value_pairs(cell))
        n_cell = sum(1 for p in pairs if p["cell"] == cell)
        assert n_cell == n_car * n_vp, (cell, n_cell, n_car, n_vp)
    cells_meta = {
        cell: {
            "scope_category": TYPE_SPEC[cell]["scope_category"],
            "tier": TYPE_SPEC[cell]["tier"],
            "values": dict(TYPE_SPEC[cell]["values"]),
            "n_carriers": len(values["types"][cell]["carriers"]),
            "degenerate_at_pe": cell in DEGENERATE_AT_PE,
            "expected_pe_eligible": cell not in DEGENERATE_AT_PE,
            "kept_pairs": sum(
                1
                for carrier in values["types"][cell]["carriers"]
                for va, vb in value_pairs(cell)
                if values["types"][cell]["judge"][pair_id(cell, va, vb, carrier)]["verdict"]
                == "PASS"
            ),
            "judge_attribute": TYPE_SPEC[cell]["judge_attribute"],
        }
        for cell in kept
    }
    values_blob = json.dumps(values, sort_keys=True, ensure_ascii=False).encode()
    return {
        "issue": ISSUE,
        "round": ROUND,
        "seed": values.get("seed", SEED),
        "model_id": MODEL_ID,
        "dry_run": bool(values.get("dry_run")),
        "smoke": bool(values.get("smoke")),
        "cells": cells_meta,
        "contexts": contexts,
        "pairs": pairs,
        "kept_types": list(kept),
        "dropped_types": {
            t: values["types"][t].get("drop_reason", "below pair-validity floor")
            for t in TYPES
            if not values["types"][t]["kept"]
        },
        "degenerate_at_pe_cells": sorted(c for c in kept if c in DEGENERATE_AT_PE),
        "expected_pe_eligibility": {c: c not in DEGENERATE_AT_PE for c in kept},
        "values_sha256": hashlib.sha256(values_blob).hexdigest(),
    }


def write_bank_dbe(values: dict, out_path: Path) -> dict:
    manifest = bank_manifest_dbe(values)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    tmp.replace(out_path)
    return manifest
