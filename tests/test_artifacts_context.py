"""Unit + regression tests for artifacts.context (task #852, Phase 0b).

Carries the LOAD-BEARING byte-equality regression: ``_legacy_render_chat`` is
a verbatim frozen copy of the pre-#852 ``render_chat`` body, and the rewired
shim must match it byte-for-byte on every 545 context id (including the
``wildchat_prefix`` battery-loading path on the git-committed battery file).
Also pins resolver parity with the #664 ``NegativeContext.messages`` and the
#594 ``messages_for_instance`` paths WITHOUT rewiring either script.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

import pytest

from explore_persona_space.artifacts.context import (
    CONTEXTS,
    INSTALLABLE_KINDS,
    Context,
    context_for_persona,
    validate_context,
)
from explore_persona_space.experiments.behavior_testbed_545 import columns, eval_battery
from explore_persona_space.personas import PERSONAS

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ---------------------------------------------------------------------------
# Byte-equality regression (the load-bearing test)
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Deterministic, injective serializer of apply_chat_template inputs.

    String equality of two outputs implies identical (messages, tokenize,
    add_generation_prompt) inputs, which implies byte-identical rendered
    output under ANY real chat template — so equality here IS byte equality,
    without depending on a cached HF tokenizer.
    """

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return json.dumps(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            },
            sort_keys=True,
        )


def _legacy_render_chat(tokenizer, question: str, context_id: str) -> str:
    """VERBATIM frozen copy of the pre-#852 chat-render implementation.

    Frozen from ``behavior_testbed_545/eval_battery.py`` at main commit
    34f49efa52127c48f905584d95f8cf0d96e746e6 (the last commit touching that
    file before #852). The only edits are the module-qualified names
    (``columns.CONTEXTS``, ``eval_battery.load_battery``); it must never
    delegate to the live shim nor import from the new package (pinned by
    test_legacy_copy_is_independent — comparing shim-vs-shim would be a
    tautology, so this frozen copy is the reference).
    """
    ctx = columns.CONTEXTS[context_id]
    messages: list[dict] = []
    if ctx.get("system"):
        messages.append({"role": "system", "content": ctx["system"]})
    if ctx.get("prefix_battery"):
        prefix = eval_battery.load_battery(ctx["prefix_battery"])["prefix_turns"]
        messages.extend(prefix)
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


EXPECTED_545_IDS = {
    "default",
    "persona_software_engineer",
    "wildchat_prefix",
    "qwen_default_system",
}
SAMPLE_QUESTIONS = (
    "What is the best way to learn a new language?",
    "Explain how photosynthesis works.",
    "  a question with { braces } and\nnewlines?  ",
)


def test_legacy_copy_is_independent():
    src = inspect.getsource(_legacy_render_chat)
    # The frozen reference must not delegate to the live shim ...
    assert re.search(r"(?<!_legacy_)render_chat\(", src) is None, src
    # ... nor touch the new package (else the regression test is a tautology).
    assert "artifacts" not in src, src


def test_render_chat_byte_equality_all_545_contexts():
    # Vacuous-loop guard: the iterated id set is exactly the 4 known 545 ids
    # (incl. wildchat_prefix, whose battery file is git-committed).
    assert set(columns.CONTEXTS) == EXPECTED_545_IDS
    tok = FakeTokenizer()
    for context_id in columns.CONTEXTS:
        for q in SAMPLE_QUESTIONS:
            assert eval_battery.render_chat(tok, q, context_id) == _legacy_render_chat(
                tok, q, context_id
            ), (context_id, q)


def test_render_chat_unknown_context_id_raises_keyerror():
    with pytest.raises(KeyError):
        eval_battery.render_chat(FakeTokenizer(), "any question", "no_such_context")


def test_render_chat_byte_equality_real_qwen_tokenizer():
    transformers = pytest.importorskip("transformers")
    from explore_persona_space.experiments.behavior_testbed_545 import BASE_MODEL

    try:
        tok = transformers.AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    except OSError:  # not in the local HF cache — keep the suite offline-deterministic
        pytest.skip(f"{BASE_MODEL} tokenizer not in the local HF cache")
    for context_id in columns.CONTEXTS:
        for q in SAMPLE_QUESTIONS[:2]:
            assert eval_battery.render_chat(tok, q, context_id) == _legacy_render_chat(
                tok, q, context_id
            ), (context_id, q)


def test_context_for_545_shapes():
    for cid in columns.CONTEXTS:
        ctx = eval_battery._context_for_545(cid)
        assert ctx.context_id == cid
        assert ctx.family == "testbed_545"
    wc = eval_battery._context_for_545("wildchat_prefix")
    assert wc.kind == "prefix"
    assert len(wc.prefix_turns) > 0  # committed battery content actually loaded
    assert eval_battery._context_for_545("default").kind == "bare"
    assert eval_battery._context_for_545("persona_software_engineer").kind == "persona"
    assert eval_battery._context_for_545("qwen_default_system").kind == "persona"


def test_cross_registry_shared_ids_system_strings_match():
    # Drift guard: the ids shared between the artifacts registry and the 545
    # columns registry must carry identical system strings (artifacts/ never
    # imports behavior_testbed_545, so equality is pinned here).
    shared = set(CONTEXTS) & set(columns.CONTEXTS)
    assert shared == {"default", "persona_software_engineer", "qwen_default_system"}
    for cid in shared:
        assert CONTEXTS[cid].system == columns.CONTEXTS[cid].get("system"), cid


# ---------------------------------------------------------------------------
# Registry + resolver unit tests
# ---------------------------------------------------------------------------


def test_registry_seeds_validate():
    assert len(CONTEXTS) == 11
    by_kind: dict[str, list[str]] = {}
    for cid, ctx in CONTEXTS.items():
        assert cid == ctx.context_id
        validate_context(ctx)
        by_kind.setdefault(ctx.kind, []).append(cid)
        if ctx.kind == "adversarial":
            assert ctx.adversarial_kind, cid
    for kind in INSTALLABLE_KINDS:
        assert len(by_kind[kind]) >= 2, (kind, by_kind)
    assert len(by_kind["bare"]) == 2, by_kind
    # Self-contained: every seed is a code literal (no file dependencies) —
    # in particular the WildChat-style bare seed discloses it is synthetic.
    assert "NOT sampled" in CONTEXTS["bare_wildchat_random"].source


def test_messages_shapes():
    q = "What time is it?"
    sys_only = Context(context_id="t1", kind="persona", family="f", system="You are X.")
    assert sys_only.messages(q) == [
        {"role": "system", "content": "You are X."},
        {"role": "user", "content": q},
    ]
    wrap_only = Context(
        context_id="t2", kind="query_transform", family="f", user_wrap="Wrapped: {q}"
    )
    assert wrap_only.messages(q) == [{"role": "user", "content": f"Wrapped: {q}"}]
    prefix = (
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
    )
    prefix_only = Context(context_id="t3", kind="prefix", family="f", prefix_turns=prefix)
    assert prefix_only.messages(q) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
        {"role": "user", "content": q},
    ]
    both = Context(context_id="t4", kind="prefix", family="f", system="S.", prefix_turns=prefix)
    assert both.messages(q) == [
        {"role": "system", "content": "S."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
        {"role": "user", "content": q},
    ]
    # Returned dicts are fresh shallow copies: mutating them never corrupts the spec.
    out = prefix_only.messages(q)
    out[0]["content"] = "mutated"
    assert prefix_only.messages(q)[0]["content"] == "hi"


def test_user_wrap_requires_q_placeholder():
    with pytest.raises(ValueError, match="user_wrap"):
        Context(context_id="t", kind="query_transform", family="f", user_wrap="no placeholder")
    # An empty string counts as SET (not "unset") and fails the {q} check.
    with pytest.raises(ValueError, match="user_wrap"):
        Context(context_id="t", kind="query_transform", family="f", user_wrap="")
    # Build-time format-safety probe: any field beyond {q} raises at build time.
    with pytest.raises(ValueError, match="format-safe"):
        Context(context_id="t", kind="query_transform", family="f", user_wrap="{q} and {other}")
    with pytest.raises(ValueError, match="format-safe"):
        Context(context_id="t", kind="query_transform", family="f", user_wrap="{q} {")


def test_structural_invariants():
    with pytest.raises(ValueError, match="context_id"):
        Context(context_id="", kind="bare", family="f")
    with pytest.raises(ValueError, match="kind"):
        Context(context_id="t", kind="mystery", family="f")
    with pytest.raises(ValueError, match="adversarial_kind"):
        Context(context_id="t", kind="adversarial", family="f", system="S.")
    with pytest.raises(ValueError, match="adversarial_kind"):
        Context(
            context_id="t", kind="persona", family="f", system="S.", adversarial_kind="roleplay"
        )


def test_validate_context_rejects_malformed_prefixes():
    # (a) turn missing the content key
    missing_key = Context(
        context_id="t", kind="prefix", family="f", prefix_turns=({"role": "user"},)
    )
    with pytest.raises(ValueError, match="prefix_turns"):
        validate_context(missing_key)
    # (b) extra key beyond {role, content}
    extra_key = Context(
        context_id="t",
        kind="prefix",
        family="f",
        prefix_turns=({"role": "user", "content": "hi", "meta": "x"},),
    )
    with pytest.raises(ValueError, match="prefix_turns"):
        validate_context(extra_key)
    # (c) role-alternation violation: two consecutive user turns
    two_users = Context(
        context_id="t",
        kind="prefix",
        family="f",
        prefix_turns=(
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ),
    )
    with pytest.raises(ValueError, match="alternate"):
        validate_context(two_users)
    # (d) odd-length prefix (ends on user, not assistant)
    ends_on_user = Context(
        context_id="t", kind="prefix", family="f", prefix_turns=({"role": "user", "content": "a"},)
    )
    with pytest.raises(ValueError, match="alternate"):
        validate_context(ends_on_user)
    # (e) role outside user/assistant
    bad_role = Context(
        context_id="t",
        kind="prefix",
        family="f",
        prefix_turns=(
            {"role": "system", "content": "a"},
            {"role": "assistant", "content": "b"},
        ),
    )
    with pytest.raises(ValueError, match="role"):
        validate_context(bad_role)
    # __post_init__ deliberately does NOT run these checks (the 545 shim path
    # only runs the minimal safe checks) — construction above never raised.


# ---------------------------------------------------------------------------
# Parity with the #664 / #594 resolvers (no rewiring of either script)
# ---------------------------------------------------------------------------


def test_negative_context_parity_664():
    import issue664_common

    q = "What causes earthquakes?"
    # user_wrap shape: panel entries with user_wrap carry system_prompt=None,
    # so messages() emits exactly [user]. Literal fields (NOT negative_panel(),
    # which reads the gitignored data/issue594/battery.json).
    neg_wrap = issue664_common.NegativeContext(
        slug="neg_reph_curious",
        identity="rephrase_curious",
        system_prompt=None,
        user_wrap="I'm curious about the following: {q}",
    )
    assert CONTEXTS["qt_rephrase_curious"].messages(q) == neg_wrap.messages(q)
    # system shape: [system, user].
    neg_sys = issue664_common.NegativeContext(
        slug="neg_sp_police",
        identity="police_officer",
        system_prompt=PERSONAS["police_officer"],
    )
    ctx_sys = Context(
        context_id="neg_sp_police",
        kind="persona",
        family="house_persona",
        system=PERSONAS["police_officer"],
    )
    assert ctx_sys.messages(q) == neg_sys.messages(q)


def _instance(iid: str, family: str, system=None, prefix=()) -> dict:
    return {
        "id": iid,
        "family": family,
        "sub_label": "x",
        "label": "x",
        "system_prompt": system,
        "prefix_messages": list(prefix),
        "source": "synthetic",
        "meta": {},
    }


def test_battery_instance_parity_594():
    import issue594_common

    probe = "How do airplanes stay in the air?"
    prefix = (
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello there"},
    )
    cases = [
        _instance("t_sys", "persona", system="You are a synthetic persona."),
        _instance("t_sys_prefix", "icl", system="You answer by example.", prefix=prefix),
        _instance("t_prefix", "wildchat", prefix=prefix),
        _instance("t_bare", "default"),
    ]
    for inst in cases:
        issue594_common.validate_instance_loose(inst)  # validator-legal by construction
        got = Context.from_battery_instance(inst).messages(probe)
        want = issue594_common.messages_for_instance(inst, probe)
        assert got == want, inst["id"]
    # FAMILY_KIND_MAP drives the kind metadata (never the resolver output).
    assert Context.from_battery_instance(cases[0]).kind == "persona"
    assert Context.from_battery_instance(cases[1]).kind == "prefix"
    assert Context.from_battery_instance(cases[2]).kind == "prefix"
    assert Context.from_battery_instance(cases[3]).kind == "bare"
    # Explicit kind override wins over the family map.
    assert Context.from_battery_instance(cases[2], kind="bare").kind == "bare"
    # Unknown family (#617-style cluster tag) -> structural inference.
    unk = _instance("t_unk", "wc_00123", prefix=prefix)
    issue594_common.validate_instance_loose(unk)
    assert Context.from_battery_instance(unk).kind == "prefix"
    # #594-derived query_transform contexts carry NO user_wrap (the battery
    # bakes the transform into the probes) — the FAMILY_KIND_MAP caveat.
    reph = Context.from_battery_instance(_instance("t_reph", "rephrase"))
    assert reph.kind == "query_transform"
    assert reph.user_wrap is None


def test_context_for_persona():
    for key in PERSONAS:
        ctx = context_for_persona(key)
        assert ctx.kind == "persona"
        assert ctx.family == "house_persona"
        assert ctx.system == PERSONAS[key]
        validate_context(ctx)
    with pytest.raises(KeyError):
        context_for_persona("nonexistent_persona")
