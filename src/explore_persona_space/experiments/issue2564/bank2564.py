"""Frozen minimal-pair bank for issue #2564 (plan v6 sections 3.1-3.5).

Registry + datagen gates + pair table for the 984-context / 2,778-pair
single-turn instruction bank. Pattern follows
``experiments/issue2215/bank_dbe.py`` (frozen values JSON next to the module
so pods read it via the pushed branch, no HF fetch); rendering follows the
``bank2162.py`` chat-template path (``apply_chat_template(...,
add_generation_prompt=True)`` on the full message list, then tokenize with
``add_special_tokens=False``).

Axis-name mapping (plan section 3.2 -> snake_case keys used everywhere here):
persona, format, lexical-marker -> ``lexical_marker``, stance,
content-constraint -> ``content_constraint``, register, hedging,
user-fact-express -> ``user_fact``, user-profile-aware -> ``user_profile``.

Context id conventions (``{cell}::{value_id}::{carrier}``):

- ``query::E::c01``     empty-system question form (the E level)
- ``query::imp::c01``   empty-system imperative form
- ``query::stmt::c01``  empty-system statement form (v6 mood-only revision)
- ``query::qpara::c01`` empty-system reworded question
- ``persona::v1::c01``  instruction value context (system = full value string,
  user = question form)
- ``persona::v1p::c01`` instruction paraphrase context

The EMPTY system level is an explicit ``{"role": "system", "content": ""}``
message: omitting the system message makes Qwen's template inject its default
system prompt, which contains "assistant" and would violate gate (vi) — the
render gate asserts exactly ONE "assistant" occurrence (the generation
header) per rendered prompt.

Pair classes + orientation conventions (plan section 3.5):

- ``install`` (468): a = E question context, b = value context, same carrier.
- ``swap`` (864): unordered value pairs within axis, same carrier;
  a = value_i, b = value_j in plan-listed value-index order (i < j).
- ``famswap`` (864): a = para(value_i), b = para(value_j), same order.
- ``instruction_paraphrase`` (468): a = value, b = its paraphrase.
- ``query_content`` (66): a = E(c_i), b = E(c_j), carrier-index order i < j.
- ``query_form`` (36): per carrier, form-index order E < imp < stmt:
  (E, imp), (E, stmt), (imp, stmt).
- ``query_paraphrase`` (12): a = E question, b = reworded question.

Every pair carries the registered edit-dose covariate ``changed_tokens``:
tokens removed + tokens added between the two FULL rendered prompts'
token-id sequences (``difflib.SequenceMatcher``, ``autojunk=False`` —
deterministic from the pinned strings + tokenizer).

Datagen gates (fail-loud ``BankGateError``; plan section 3.5 items i-vii):

(i)   within-axis token-count equality of every full value system string
      against the plan-pinned counts (verified against the live tokenizer
      2026-08-24 before freezing);
(ii)  each `` {NAME}`` (leading space) encodes to exactly ONE token with the
      plan-pinned id;
(iii) byte-identity of the non-varied slots within every pair (held-fixed
      user strings; empty systems; slotted-template prefix/suffix identity
      via reconstruction);
(iv)  paraphrase length ratio within +/-30% — MEASURED granularity decision
      (probe 2026-08-24): string-level token ratio holds for all 39
      instruction paraphrases (asserted string-level) but breaks on 5 of the
      12 query paraphrases (max 1.444 at c01) while the rendered-context
      ratio holds everywhere (max 1.182), so query paraphrases — and, as a
      second belt, all instruction-paraphrase pairs — are asserted at
      rendered-context level;
(v)   complete grid — 39 values x 12 carriers + 39 paraphrases x 12
      carriers, no holes; 984 contexts; 2,778 pairs in exact per-class
      counts;
(vi)  no "assistant" substring in any stored system string, and every
      rendered prompt contains exactly one "assistant" (the generation
      header) and never the Qwen default-system injection;
(vii) form-triplet semantic equivalence — imperative + statement forms carry
      no speaker-state affect/uncertainty vocabulary (denylist), and
      explicit two-way carriers keep each option's content word in all three
      forms.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.atomic_io import atomic_replace

ISSUE = 2564
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
VALUES_FILENAME = "bank2564_values.json"

INSTRUCTION_AXES: tuple[str, ...] = (
    "persona",
    "format",
    "lexical_marker",
    "stance",
    "content_constraint",
    "register",
    "hedging",
    "user_fact",
    "user_profile",
)
SLOTTED_AXES: tuple[str, ...] = (
    "persona",
    "format",
    "lexical_marker",
    "user_fact",
    "user_profile",
)
N_VALUES_PER_AXIS: dict[str, int] = {
    "persona": 5,
    "format": 5,
    "lexical_marker": 5,
    "stance": 5,
    "content_constraint": 5,
    "register": 2,
    "hedging": 2,
    "user_fact": 5,
    "user_profile": 5,
}
# Plan section 3.2 pinned per-axis token counts of the FULL system strings
# (add_special_tokens=False), re-verified against the live tokenizer at build.
EXPECTED_VALUE_TOKENS: dict[str, int] = {
    "persona": 13,
    "format": 10,
    "lexical_marker": 12,
    "stance": 9,
    "content_constraint": 9,
    "register": 10,
    "hedging": 10,
    "user_fact": 13,
    "user_profile": 24,
}
# Plan section 3.2 pinned single-token ids of " {NAME}" (leading space).
NAME_TOKEN_IDS: dict[str, int] = {
    "Marcus": 35683,
    "Diego": 18336,
    "Sarah": 20445,
    "Emma": 34935,
    "Kevin": 16364,
}

CARRIER_IDS: tuple[str, ...] = tuple(f"c{i:02d}" for i in range(1, 13))
# E-level value ids -> the carrier column each reads its user text from.
E_FORMS: dict[str, str] = {
    "E": "question",
    "imp": "imperative",
    "stmt": "statement",
    "qpara": "paraphrase",
}
QUERY_FORM_ORDER: tuple[str, ...] = ("E", "imp", "stmt")

# Gate (vii) speaker-state affect/uncertainty denylist (plan section 3.5).
AFFECT_DENYLIST: tuple[str, ...] = (
    "torn",
    "can't",
    "unsure",
    "stuck",
    "wondering",
    "not sure",
    "don't know",
    "go back and forth",
)
# Explicit two-way carriers -> the option content words (stems) that must
# appear in all three forms (question / imperative / statement).
TWO_WAY_OPTIONS: dict[str, tuple[str, str]] = {
    "c01": ("dog", "cat"),
    "c02": ("rent", "buy"),
    "c05": ("remote", "office"),
    "c07": ("fiction", "nonfiction"),
    "c10": ("sav", "spend"),
    "c11": ("passion", "stable"),
    "c12": ("morning", "evening"),
}

PAIR_CLASSES: tuple[str, ...] = (
    "install",
    "swap",
    "famswap",
    "instruction_paraphrase",
    "query_content",
    "query_form",
    "query_paraphrase",
)
EXPECTED_PAIR_COUNTS: dict[str, int] = {
    "install": 468,
    "swap": 864,
    "famswap": 864,
    "instruction_paraphrase": 468,
    "query_content": 66,
    "query_form": 36,
    "query_paraphrase": 12,
}
N_CONTEXTS = 984
N_PAIRS = 2778
PARA_RATIO_LO = 0.7
PARA_RATIO_HI = 1.3


class BankGateError(RuntimeError):
    """A datagen gate (plan section 3.5 items i-vii) failed — fail loud."""


# ── values loading + structural validation ────────────────────────────


def load_values(path: Path | None = None) -> dict:
    """Load + structurally validate the frozen pinned-strings payload."""
    p = Path(path) if path is not None else Path(__file__).parent / VALUES_FILENAME
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — the frozen bank values ship with the module")
    values = json.loads(p.read_text())
    validate_values(values)
    return values


def validate_values(values: dict) -> None:
    """Fail-fast structural validation (shape only; token gates run at build)."""
    assert values.get("issue") == ISSUE, "wrong values payload"
    assert values.get("model_id") == MODEL_ID, values.get("model_id")
    axes = values["axes"]
    assert tuple(axes) == INSTRUCTION_AXES, sorted(set(INSTRUCTION_AXES) ^ set(axes))
    for axis, ax in axes.items():
        vids = value_ids(values, axis)
        assert len(vids) == N_VALUES_PER_AXIS[axis], (axis, vids)
        assert vids == tuple(f"v{i}" for i in range(1, len(vids) + 1)), (axis, vids)
        assert ax["expected_value_tokens"] == EXPECTED_VALUE_TOKENS[axis], axis
        if ax["kind"] == "slotted":
            assert axis in SLOTTED_AXES, axis
            for key in ("template", "paraphrase_template"):
                assert ax[key].count(ax["slot"]) == 1, (axis, key)
        else:
            assert ax["kind"] == "sentence" and axis not in SLOTTED_AXES, axis
            assert set(ax["paraphrases"]) == set(vids), (axis, "paraphrase holes")
        for vid in vids:
            assert isinstance(ax["values"][vid], str) and ax["values"][vid].strip(), (axis, vid)
    assert set(values["axes"]["user_fact"]["name_token_ids"]) == set(
        values["axes"]["user_fact"]["values"].values()
    ), "name_token_ids must cover exactly the user_fact names"
    carriers = values["carriers"]
    assert tuple(carriers) == CARRIER_IDS, sorted(set(CARRIER_IDS) ^ set(carriers))
    for cid, car in carriers.items():
        assert set(car) == {"question", "imperative", "statement", "paraphrase"}, cid
        for form, text in car.items():
            assert isinstance(text, str) and text.strip(), (cid, form)


def value_ids(values: dict, axis: str) -> tuple[str, ...]:
    return tuple(values["axes"][axis]["values"].keys())


def system_string(values: dict, axis: str, vid: str) -> str:
    """The full frozen system string for one instruction value."""
    ax = values["axes"][axis]
    if ax["kind"] == "slotted":
        return ax["template"].replace(ax["slot"], ax["values"][vid])
    return ax["values"][vid]


def paraphrase_string(values: dict, axis: str, vid: str) -> str:
    """The full frozen paraphrase system string for one instruction value."""
    ax = values["axes"][axis]
    if ax["kind"] == "slotted":
        return ax["paraphrase_template"].replace(ax["slot"], ax["values"][vid])
    return ax["paraphrases"][vid]


# ── ids + context/pair builders ───────────────────────────────────────


def context_id(cell: str, value_id: str, carrier: str) -> str:
    return f"{cell}::{value_id}::{carrier}"


def pair_id(pair_class: str, cell: str, va: str, vb: str, carrier: str) -> str:
    return f"{pair_class}::{cell}::{va}-{vb}::{carrier}"


def build_contexts(values: dict) -> dict[str, dict]:
    """All 984 single-turn contexts: {cid: {id, cell, kind, value_id, carrier,
    form, system, user}}."""
    contexts: dict[str, dict] = {}

    def _add(ctx: dict) -> None:
        assert ctx["id"] not in contexts, ctx["id"]
        contexts[ctx["id"]] = ctx

    for carrier in CARRIER_IDS:
        car = values["carriers"][carrier]
        for vid, form in E_FORMS.items():
            _add(
                {
                    "id": context_id("query", vid, carrier),
                    "cell": "query",
                    "kind": "E",
                    "value_id": vid,
                    "carrier": carrier,
                    "form": form,
                    "system": "",
                    "user": car[form],
                }
            )
        for axis in INSTRUCTION_AXES:
            for vid in value_ids(values, axis):
                _add(
                    {
                        "id": context_id(axis, vid, carrier),
                        "cell": axis,
                        "kind": "value",
                        "value_id": vid,
                        "carrier": carrier,
                        "form": "question",
                        "system": system_string(values, axis, vid),
                        "user": car["question"],
                    }
                )
                _add(
                    {
                        "id": context_id(axis, f"{vid}p", carrier),
                        "cell": axis,
                        "kind": "para",
                        "value_id": f"{vid}p",
                        "carrier": carrier,
                        "form": "question",
                        "system": paraphrase_string(values, axis, vid),
                        "user": car["question"],
                    }
                )
    assert len(contexts) == N_CONTEXTS, len(contexts)
    return contexts


def build_pairs(values: dict, contexts: dict[str, dict]) -> list[dict]:
    """All 2,778 directed pairs with the section-3.5 orientation conventions."""
    pairs: list[dict] = []

    def _add(pair_class: str, cell: str, va: str, vb: str, carrier: str, a: str, b: str) -> None:
        assert a in contexts and b in contexts, (a, b)
        pairs.append(
            {
                "pair_id": pair_id(pair_class, cell, va, vb, carrier),
                "pair_class": pair_class,
                "cell": cell,
                "carrier": carrier,
                "value_a": va,
                "value_b": vb,
                "a": a,
                "b": b,
            }
        )

    for carrier in CARRIER_IDS:
        e_ctx = context_id("query", "E", carrier)
        for axis in INSTRUCTION_AXES:
            vids = value_ids(values, axis)
            for vid in vids:
                # install: E -> value (a = E, b = value).
                _add("install", axis, "E", vid, carrier, e_ctx, context_id(axis, vid, carrier))
                # paraphrase null: value -> its paraphrase.
                _add(
                    "instruction_paraphrase",
                    axis,
                    vid,
                    f"{vid}p",
                    carrier,
                    context_id(axis, vid, carrier),
                    context_id(axis, f"{vid}p", carrier),
                )
            for i in range(len(vids)):
                for j in range(i + 1, len(vids)):
                    # swap: value_i -> value_j by plan-listed value-index order.
                    _add(
                        "swap",
                        axis,
                        vids[i],
                        vids[j],
                        carrier,
                        context_id(axis, vids[i], carrier),
                        context_id(axis, vids[j], carrier),
                    )
                    # paraphrase-family swap: para(value_i) -> para(value_j).
                    _add(
                        "famswap",
                        axis,
                        f"{vids[i]}p",
                        f"{vids[j]}p",
                        carrier,
                        context_id(axis, f"{vids[i]}p", carrier),
                        context_id(axis, f"{vids[j]}p", carrier),
                    )
        # query-form: C(3,2) per carrier in form-index order E < imp < stmt.
        for i in range(len(QUERY_FORM_ORDER)):
            for j in range(i + 1, len(QUERY_FORM_ORDER)):
                fa, fb = QUERY_FORM_ORDER[i], QUERY_FORM_ORDER[j]
                _add(
                    "query_form",
                    "query",
                    fa,
                    fb,
                    carrier,
                    context_id("query", fa, carrier),
                    context_id("query", fb, carrier),
                )
        # query-paraphrase null: question -> reworded question.
        _add(
            "query_paraphrase",
            "query",
            "E",
            "qpara",
            carrier,
            e_ctx,
            context_id("query", "qpara", carrier),
        )
    # query-content: C(12,2) over E contexts, carrier-index order c_i -> c_j.
    for i in range(len(CARRIER_IDS)):
        for j in range(i + 1, len(CARRIER_IDS)):
            ca, cb = CARRIER_IDS[i], CARRIER_IDS[j]
            p_carrier = f"{ca}|{cb}"
            pairs.append(
                {
                    "pair_id": pair_id("query_content", "query", "E", "E", p_carrier),
                    "pair_class": "query_content",
                    "cell": "query",
                    "carrier": p_carrier,
                    "carrier_a": ca,
                    "carrier_b": cb,
                    "value_a": "E",
                    "value_b": "E",
                    "a": context_id("query", "E", ca),
                    "b": context_id("query", "E", cb),
                }
            )
    counts = {cls: sum(1 for p in pairs if p["pair_class"] == cls) for cls in PAIR_CLASSES}
    assert counts == EXPECTED_PAIR_COUNTS, counts
    assert len(pairs) == N_PAIRS, len(pairs)
    assert len({p["pair_id"] for p in pairs}) == N_PAIRS, "duplicate pair_id"
    return pairs


# ── rendering (the bank2162.py chat-template path) ────────────────────


def context_messages(context: dict) -> list[dict]:
    """Single-turn message list; the empty system level is an EXPLICIT
    empty-content system message (never omitted — see module docstring)."""
    return [
        {"role": "system", "content": context["system"]},
        {"role": "user", "content": context["user"]},
    ]


def render_context(tokenizer, context: dict) -> str:
    rendered = tokenizer.apply_chat_template(
        context_messages(context), tokenize=False, add_generation_prompt=True
    )
    assert isinstance(rendered, str) and rendered, context["id"]
    return rendered


def context_token_ids(tokenizer, context: dict) -> list[int]:
    ids = tokenizer(render_context(tokenizer, context), add_special_tokens=False)["input_ids"]
    assert len(ids) >= 4, (len(ids), context["id"])
    return ids


def _n_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


# ── edit-dose covariate ───────────────────────────────────────────────


def changed_token_count(ids_a: list[int], ids_b: list[int]) -> int:
    """Tokens removed + tokens added between two rendered-prompt id sequences
    (``difflib.SequenceMatcher``, ``autojunk=False`` for determinism)."""
    sm = difflib.SequenceMatcher(a=ids_a, b=ids_b, autojunk=False)
    return sum((i2 - i1) + (j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")


def attach_changed_tokens(pairs: list[dict], ids_by_context: dict[str, list[int]]) -> None:
    """Persist the registered ``changed_tokens`` edit-dose covariate per pair."""
    for p in pairs:
        p["changed_tokens"] = changed_token_count(ids_by_context[p["a"]], ids_by_context[p["b"]])
        assert p["changed_tokens"] >= 1, (p["pair_id"], "pair sides render identically")


# ── datagen gates (plan section 3.5 items i-vii) ──────────────────────


def gate_value_token_counts(tokenizer, values: dict) -> dict[str, dict[str, int]]:
    """(i) within-axis token-count equality against the plan-pinned counts."""
    realized: dict[str, dict[str, int]] = {}
    for axis in INSTRUCTION_AXES:
        exp = EXPECTED_VALUE_TOKENS[axis]
        counts = {
            vid: _n_tokens(tokenizer, system_string(values, axis, vid))
            for vid in value_ids(values, axis)
        }
        realized[axis] = counts
        bad = {vid: c for vid, c in counts.items() if c != exp}
        if bad:
            raise BankGateError(f"gate(i) {axis}: expected {exp} tokens, realized {bad}")
    return realized


def gate_name_token_ids(tokenizer, values: dict) -> dict[str, int]:
    """(ii) each ' {NAME}' encodes to exactly ONE token with the pinned id."""
    pins = values["axes"]["user_fact"]["name_token_ids"]
    realized: dict[str, int] = {}
    for name, pin in pins.items():
        ids = tokenizer(" " + name, add_special_tokens=False)["input_ids"]
        if len(ids) != 1 or ids[0] != pin:
            raise BankGateError(f"gate(ii) ' {name}': pinned [{pin}], realized {ids}")
        realized[name] = ids[0]
    return realized


def gate_pair_slot_identity(values: dict, contexts: dict[str, dict], pairs: list[dict]) -> None:
    """(iii) byte-identity of the non-varied slots within every pair."""
    same_user = {"install", "swap", "famswap", "instruction_paraphrase"}
    for p in pairs:
        a, b = contexts[p["a"]], contexts[p["b"]]
        if p["pair_class"] in same_user:
            if a["user"] != b["user"]:
                raise BankGateError(f"gate(iii) {p['pair_id']}: user strings differ")
            if a["carrier"] != b["carrier"]:
                raise BankGateError(f"gate(iii) {p['pair_id']}: carriers differ")
        else:  # query classes: the empty system is the held-fixed slot
            if a["system"] != "" or b["system"] != "":
                raise BankGateError(f"gate(iii) {p['pair_id']}: non-empty query system")
            if p["pair_class"] != "query_content" and a["carrier"] != b["carrier"]:
                raise BankGateError(f"gate(iii) {p['pair_id']}: carriers differ")
        if p["pair_class"] == "install" and a["system"] != "":
            raise BankGateError(f"gate(iii) {p['pair_id']}: install a-side must be E")
    # Slotted-template prefix/suffix identity via reconstruction, and user
    # strings equal to the frozen carrier column, for EVERY context.
    for ctx in contexts.values():
        expected_user = values["carriers"][ctx["carrier"]][ctx["form"]]
        if ctx["user"] != expected_user:
            raise BankGateError(f"gate(iii) {ctx['id']}: user drifted from frozen carrier text")
        if ctx["kind"] == "value":
            expect = system_string(values, ctx["cell"], ctx["value_id"])
        elif ctx["kind"] == "para":
            expect = paraphrase_string(values, ctx["cell"], ctx["value_id"][:-1])
        else:
            expect = ""
        if ctx["system"] != expect:
            raise BankGateError(f"gate(iii) {ctx['id']}: system drifted from frozen template")


def gate_paraphrase_ratios(
    tokenizer, values: dict, contexts: dict[str, dict], pairs: list[dict]
) -> dict[str, float]:
    """(iv) paraphrase length ratio within +/-30%.

    Measured granularity (probe 2026-08-24, module docstring): string-level
    for the 39 instruction paraphrases; rendered-context level for every
    instruction-paraphrase AND query-paraphrase pair.
    """
    realized: dict[str, float] = {}
    for axis in INSTRUCTION_AXES:
        for vid in value_ids(values, axis):
            r = _n_tokens(tokenizer, paraphrase_string(values, axis, vid)) / _n_tokens(
                tokenizer, system_string(values, axis, vid)
            )
            realized[f"string::{axis}::{vid}"] = r
            if not PARA_RATIO_LO <= r <= PARA_RATIO_HI:
                raise BankGateError(f"gate(iv) {axis}:{vid} string-level ratio {r:.3f}")
    n_by_ctx = {cid: len(context_token_ids(tokenizer, ctx)) for cid, ctx in contexts.items()}
    for p in pairs:
        if p["pair_class"] not in ("instruction_paraphrase", "query_paraphrase"):
            continue
        r = n_by_ctx[p["b"]] / n_by_ctx[p["a"]]
        realized[f"render::{p['pair_id']}"] = r
        if not PARA_RATIO_LO <= r <= PARA_RATIO_HI:
            raise BankGateError(f"gate(iv) {p['pair_id']} render-level ratio {r:.3f}")
    return realized


def gate_grid_complete(values: dict, contexts: dict[str, dict], pairs: list[dict]) -> None:
    """(v) complete grid — no holes; exact context + pair counts."""
    if len(contexts) != N_CONTEXTS:
        raise BankGateError(f"gate(v) {len(contexts)} contexts != {N_CONTEXTS}")
    for carrier in CARRIER_IDS:
        for vid in E_FORMS:
            if context_id("query", vid, carrier) not in contexts:
                raise BankGateError(f"gate(v) missing query::{vid}::{carrier}")
        for axis in INSTRUCTION_AXES:
            for vid in value_ids(values, axis):
                for want in (vid, f"{vid}p"):
                    if context_id(axis, want, carrier) not in contexts:
                        raise BankGateError(f"gate(v) missing {axis}::{want}::{carrier}")
    counts = {cls: sum(1 for p in pairs if p["pair_class"] == cls) for cls in PAIR_CLASSES}
    if counts != EXPECTED_PAIR_COUNTS or len(pairs) != N_PAIRS:
        raise BankGateError(f"gate(v) pair counts {counts} (total {len(pairs)})")


def gate_no_assistant_substring(
    contexts: dict[str, dict], rendered_by_context: dict[str, str]
) -> None:
    """(vi) no "assistant" in stored system strings; exactly one occurrence
    (the generation header) in every rendered prompt; no default-system
    injection."""
    for ctx in contexts.values():
        if "assistant" in ctx["system"].lower():
            raise BankGateError(f"gate(vi) {ctx['id']}: 'assistant' in system string")
    for cid, rendered in rendered_by_context.items():
        if rendered.count("assistant") != 1:
            raise BankGateError(
                f"gate(vi) {cid}: {rendered.count('assistant')} 'assistant' occurrences"
            )
        if "You are Qwen" in rendered:
            raise BankGateError(f"gate(vi) {cid}: default system prompt injected")
        if contexts[cid]["system"] == "" and not rendered.startswith(
            "<|im_start|>system\n<|im_end|>\n"
        ):
            raise BankGateError(f"gate(vi) {cid}: empty system block missing from render")


def gate_form_triplets(values: dict) -> None:
    """(vii) mood-only statement/imperative forms: affect denylist + explicit
    two-way carriers keep each option's content word in all three forms."""
    for cid, car in values["carriers"].items():
        for form in ("imperative", "statement"):
            # normalize U+2019 RIGHT SINGLE QUOTATION MARK to ASCII apostrophe
            text = car[form].replace(chr(0x2019), "'").lower()
            for term in AFFECT_DENYLIST:
                if term in text:
                    raise BankGateError(f"gate(vii) {cid} {form}: affect term {term!r}")
        if cid in TWO_WAY_OPTIONS:
            for opt in TWO_WAY_OPTIONS[cid]:
                for form in ("question", "imperative", "statement"):
                    if opt not in car[form].lower():
                        raise BankGateError(f"gate(vii) {cid} {form}: option word {opt!r} missing")


def run_datagen_gates(
    tokenizer, values: dict, contexts: dict[str, dict], pairs: list[dict]
) -> dict:
    """Run gates (i)-(vii) fail-loud over the ENTIRE frozen bank; return the
    realized-measurements record (plan section 12 A3: full grain, never a
    sample)."""
    realized_values = gate_value_token_counts(tokenizer, values)
    realized_names = gate_name_token_ids(tokenizer, values)
    gate_pair_slot_identity(values, contexts, pairs)
    realized_ratios = gate_paraphrase_ratios(tokenizer, values, contexts, pairs)
    gate_grid_complete(values, contexts, pairs)
    rendered = {cid: render_context(tokenizer, ctx) for cid, ctx in contexts.items()}
    gate_no_assistant_substring(contexts, rendered)
    gate_form_triplets(values)
    carrier_form_tokens = {
        cid: {form: _n_tokens(tokenizer, text) for form, text in car.items()}
        for cid, car in values["carriers"].items()
    }
    paraphrase_tokens = {
        axis: {
            vid: _n_tokens(tokenizer, paraphrase_string(values, axis, vid))
            for vid in value_ids(values, axis)
        }
        for axis in INSTRUCTION_AXES
    }
    return {
        "verdict": "PASS",
        "gates_run": ["i", "ii", "iii", "iv", "v", "vi", "vii"],
        "value_token_counts": realized_values,
        "name_token_ids": realized_names,
        "paraphrase_token_counts": paraphrase_tokens,
        "carrier_form_token_counts": carrier_form_tokens,
        "paraphrase_ratio_min": min(realized_ratios.values()),
        "paraphrase_ratio_max": max(realized_ratios.values()),
    }


# ── bank build + manifest ─────────────────────────────────────────────


def build_bank(tokenizer, values: dict | None = None) -> dict:
    """Build + gate the full frozen bank (984 contexts, 2,778 pairs with
    ``changed_tokens``); raises ``BankGateError`` on any gate violation."""
    values = load_values() if values is None else values
    validate_values(values)
    contexts = build_contexts(values)
    pairs = build_pairs(values, contexts)
    gates = run_datagen_gates(tokenizer, values, contexts, pairs)
    ids_by_context = {cid: context_token_ids(tokenizer, ctx) for cid, ctx in contexts.items()}
    attach_changed_tokens(pairs, ids_by_context)
    values_blob = json.dumps(values, sort_keys=True, ensure_ascii=False).encode()
    return {
        "issue": ISSUE,
        "model_id": MODEL_ID,
        "values_sha256": hashlib.sha256(values_blob).hexdigest(),
        "n_contexts": len(contexts),
        "n_pairs": len(pairs),
        "pair_class_counts": dict(EXPECTED_PAIR_COUNTS),
        "contexts": contexts,
        "pairs": pairs,
        "gates": gates,
    }


def write_bank_manifest(bank: dict, out_path: Path) -> None:
    """Atomically write the P0 bank manifest with reproducibility metadata."""
    from explore_persona_space.orchestrate.provenance import (
        as_metadata_dict,
        git_provenance,
    )

    manifest = dict(bank)
    manifest["metadata"] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        **as_metadata_dict(git_provenance(), phase="bank"),
    }
    out_path = Path(out_path)
    # process-unique temp via atomic_io.atomic_replace (#2336)
    with atomic_replace(out_path) as tmp:
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def _repo_root() -> Path:
    # bank2564.py -> issue2564 -> experiments -> explore_persona_space -> src -> repo root
    root = Path(__file__).resolve().parents[4]
    assert (root / "pyproject.toml").exists(), root
    return root


def main(argv: list[str] | None = None) -> int:
    """P0 bank build + freeze: run every datagen gate over the entire frozen
    bank and write ``eval_results/issue_2564/bank_manifest.json``."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--values", type=Path, default=None, help="override values JSON path")
    ap.add_argument(
        "--out",
        type=Path,
        default=_repo_root() / "eval_results" / "issue_2564" / "bank_manifest.json",
    )
    args = ap.parse_args(argv)

    # Shared-VM thread caps (#847): bind BEFORE the transformers->torch import.
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bank = build_bank(tokenizer, load_values(args.values))
    write_bank_manifest(bank, args.out)
    print(
        f"[phase=bank] contexts={bank['n_contexts']} pairs={bank['n_pairs']} "
        f"gates=PASS -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
