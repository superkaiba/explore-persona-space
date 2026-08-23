"""Issue #2215 dbe datagen — parse-contract + bank-emitter pins (plan v6 §4.2).

Covers the newly composed pair-validity judge instrument's PARSE contract
(llm-judging rule 27: realistic + fenced replies through the harness's own
``parse_judge_json`` path; malformed / refusal returns -> None, never coerced),
the generation-payload coercers, prompt-slot substitution completeness, the
programmatic ``user_doc_format`` renderers' determinism, and the bank emitter's
grid invariants on a SYNTHETIC values fixture (no network, no API, no real
benchmark text — XSTest is referenced by filename only, per the trigger-density
rule).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2215_dbe_datagen as D  # noqa: E402

from explore_persona_space.experiments.issue2215 import bank_dbe as B  # noqa: E402

# ── judge parse contract (rule 27 round-trip) ─────────────────────────


def test_judge_parse_realistic_reply():
    out = D._parse_judge('{"reasoning": "sides differ only in identity", "verdict": "PASS"}')
    assert out == {"verdict": "PASS", "reasoning": "sides differ only in identity"}


def test_judge_parse_fenced_reply():
    text = 'Checking.\n```json\n{"reasoning": "content diverges", "verdict": "FAIL"}\n```'
    assert D._parse_judge(text)["verdict"] == "FAIL"


def test_judge_parse_lowercase_verdict_normalized():
    assert D._parse_judge('{"reasoning": "ok", "verdict": "pass"}')["verdict"] == "PASS"


@pytest.mark.parametrize(
    "bad",
    [
        "I cannot evaluate this content.",  # refusal prose -> DROP, never coerced
        '{"verdict": "MAYBE"}',  # out-of-vocabulary verdict
        '{"reasoning": "no verdict key"}',
        "",
    ],
)
def test_judge_parse_bad_returns_none(bad):
    assert D._parse_judge(bad) is None


# ── gen coercers ──────────────────────────────────────────────────────


def test_coerce_role_requires_single_placeholder():
    ok = D._coerce_gen(
        "user_role_identity",
        {"turn1_template": "Hi, I'm {identity}. Advice?", "assistant_ack": "a", "final_query": "q"},
    )
    assert ok["turn1_template"].count(B.ROLE_IDENTITY_SLOT) == 1
    with pytest.raises(AssertionError):
        D._coerce_gen(
            "user_role_identity",
            {"turn1_template": "no placeholder", "assistant_ack": "a", "final_query": "q"},
        )


def test_coerce_topic_requires_shared_skeleton():
    good = {
        k: {"user_turns": ["u1", "u2"], "assistant_turns": ["a1", "a2"]}
        for k in ("cooking", "personal_finance", "houseplants")
    }
    assert set(D._coerce_gen("conversation_topic", good)["conversations"]) == {"v1", "v2", "v3"}
    bad = dict(good)
    bad["houseplants"] = {"user_turns": ["u1"], "assistant_turns": ["a1"]}
    with pytest.raises(AssertionError):
        D._coerce_gen("conversation_topic", bad)


def test_coerce_doc_row_shape():
    with pytest.raises(AssertionError):
        D._coerce_gen(
            "user_doc_format",
            {
                "title": "T",
                "columns": ["a", "b"],  # < 3 columns
                "rows": [["x", 1]] * 4,
                "assistant_ack": "ok",
                "final_query": "q",
            },
        )


def test_coerce_doc_good_path_stores_programmatic_renderings():
    """build_context + the bank freeze assert consume car['renderings'] — the
    real-API parse path must store them (Phase G crash: KeyError 'renderings')."""
    car = D._coerce_gen(
        "user_doc_format",
        {
            "title": "T",
            "columns": ["a", "b", "c"],
            "rows": [["x", 1, "n"]] * 4,
            "assistant_ack": "ok",
            "final_query": "q",
        },
    )
    assert car["renderings"] == B.doc_renderings(car["payload"])
    assert set(car["renderings"]) == {"v1", "v2", "v3"}


# ── synthetic values fixture (no benchmark text; placeholders only) ───


def _synthetic_values() -> dict:
    fake_ranked = [
        {
            "rank": r + 1,
            "unsafe_id": f"u{r}",
            "safe_id": f"s{r}",
            "unsafe_type": "contrast_synthetic",
            "safe_type": "synthetic",
            "jaccard": 0.5,
            "safe_prompt": f"[synthetic safe prompt {r}]",
            "unsafe_prompt": f"[synthetic unsafe prompt {r}]",
        }
        for r in range(4)
    ]
    fake_imdb_rows = [
        {
            "Text_Original": f"[synthetic review {r}]",
            "Text_Contrast": f"[synthetic contrast review {r}]",
            "Sentiment_Original": "Positive" if r % 2 else "Negative",
            "Sentiment_Contrast": "Negative" if r % 2 else "Positive",
        }
        for r in range(4)
    ]
    wc_idx = list(range(400, 424))
    selection = {
        "xstest": {"ranked": fake_ranked, "human_audit": "n/a-test"},
        "imdb": {"order": list(range(4))},
        "wildchat": {
            "range": [400, 500],
            "n_passing": 24,
            "style_register": wc_idx[:12],
            "conversation_language": wc_idx[12:24],
            "texts": {str(i): f"[synthetic seed query {i}]" for i in wc_idx},
        },
        "_imdb_rows": fake_imdb_rows,
    }
    args = argparse.Namespace(dry_run=True, smoke=True)
    values = D.init_values(selection, args)
    D.fill_placeholders(values)
    return values


def test_bank_emitter_grid_invariants_on_synthetic_fixture():
    values = _synthetic_values()
    bank = B.bank_manifest_dbe(values)
    contexts, pairs = bank["contexts"], bank["pairs"]
    # every pair endpoint resolves; per-cell (carrier x vp) grid is complete
    for p in pairs:
        assert p["a"] in contexts and p["b"] in contexts, p["pair_id"]
    for cell in B.TYPES:
        cell_pairs = [p for p in pairs if p["cell"] == cell]
        carriers = sorted({p["carrier"] for p in cell_pairs})
        vps = B.value_pairs(cell)
        seen = {(p["carrier"], p["value_a"], p["value_b"]) for p in cell_pairs}
        assert len(seen) == len(cell_pairs) == len(carriers) * len(vps), cell
        for c in carriers:
            for va, vb in vps:
                assert (c, va, vb) in seen, (cell, c, va, vb)
    assert bank["degenerate_at_pe_cells"] == ["refusal_request", "user_sentiment"]
    n_expected_ctx = 7 * 12 * 3 + 2 * D.SMOKE_N_ITEMS_BENCHMARK * 2
    assert len(contexts) == n_expected_ctx, len(contexts)


def test_expected_pe_eligibility_is_7_eligible_2_degenerate():
    pe = B.expected_pe_eligibility()
    assert sum(pe.values()) == 7
    assert {t for t, e in pe.items() if not e} == {"refusal_request", "user_sentiment"}


def test_degenerate_cells_are_single_turn_and_share_prefix_structure():
    values = _synthetic_values()
    for cell in B.DEGENERATE_AT_PE:
        carrier = sorted(values["types"][cell]["carriers"])[0]
        for vid in B.value_ids(cell):
            ctx = B.build_context(values, cell, vid, carrier)
            assert ctx["history"] == [], (cell, vid)  # varying span IS the final turn
            assert ctx["pe_expected_eligible"] is False


def test_eligible_cells_vary_a_pre_query_turn():
    values = _synthetic_values()
    for cell in (t for t in B.TYPES if t not in B.DEGENERATE_AT_PE):
        carrier = sorted(values["types"][cell]["carriers"])[0]
        pre_query = {
            vid: [m["content"] for m in B.build_context(values, cell, vid, carrier)["history"]]
            for vid in B.value_ids(cell)
        }
        assert all(len(h) >= 2 for h in pre_query.values()), cell
        flat = {tuple(h) for h in pre_query.values()}
        assert len(flat) == len(pre_query), (cell, "pre-query turns must differ across values")


def test_doc_renderings_deterministic_and_bit_identical_data():
    payload = {"title": "T", "columns": ["a", "b", "c"], "rows": [["x", 1, 2.5], ["y", 3, 4.0]]}
    r1, r2 = B.doc_renderings(payload), B.doc_renderings(payload)
    assert r1 == r2
    assert set(r1) == {"v1", "v2", "v3"}
    import json as _json

    parsed = _json.loads(r1["v2"])
    assert [list(rec.values()) for rec in parsed["records"]] == payload["rows"]


def test_gen_and_judge_prompts_have_no_unreplaced_slots():
    values = _synthetic_values()
    slot = re.compile(r"<<[A-Z_]+>>")
    for cell in B.CONSTRUCTED_TYPES:
        assert not slot.search(D._gen_prompt(cell, "c01", values)), cell
    items = D._judge_items_for(values, "user_role_identity", ["c01"], 1, None)
    assert len(items) == 3
    assert all(not slot.search(i.payload["prompt"]) for i in items)
    bounded = D._judge_items_for(values, "user_role_identity", ["c01"], 1, 2)
    assert len(bounded) == 2  # --smoke judging bound


def test_xstest_type_mapping_many_to_one_cases():
    safe_types = {
        "nons_group_real_discr",
        "real_group_nons_discr",
        "privacy_public",
        "privacy_fictional",
        "homonyms",
    }
    assert set(D.mapped_safe_types("contrast_discr", safe_types)) == {
        "nons_group_real_discr",
        "real_group_nons_discr",
    }
    assert set(D.mapped_safe_types("contrast_privacy", safe_types)) == {
        "privacy_public",
        "privacy_fictional",
    }
    assert D.mapped_safe_types("contrast_homonyms", safe_types) == ("homonyms",)
    with pytest.raises(AssertionError):
        D.mapped_safe_types("contrast_unknown_type", safe_types)
