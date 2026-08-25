"""Rule-27 parse-contract + composition-invariant tests for the #2552 judge driver.

Covers (llm-judging.md rule 27 + the unit-2 brief):
- App-D schema shape (24 fields / 5 categories, verbatim-embed invariants);
- per-wave parser round-trips through the harness's own ``parse_judge_json``
  (realistic replies + fenced variants; malformed/none/out-of-range -> DROP);
- no-unfilled-slot invariants for every wave system prompt (the composed block IS
  the user message — no ``{question}``/``{answer}`` template slots exist);
- batch custom-id grammar for every item-id shape the driver emits;
- W4/W5/W6 presentation determinism (seed 2552; W4 per-turn set FIXED across configs);
- G2 need-set arithmetic + the corpus-stratified eval-subset mirror;
- draw classification precedence (transport > api_refusal > truncation > parse) and
  the reduce/frac_items_complete floor accounting;
- Cohen's kappa reference values (W7).

Torch-free by construction: the driver defers its unit-1 (torch-bearing) import
behind ``_t2552()``, which no function under test calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2552_judge_waves as J  # noqa: E402

from explore_persona_space.eval.utils import parse_judge_json  # noqa: E402

# ── App-D schema shape ───────────────────────────────────────────────────────────


def test_app_d_schema_shape():
    assert len(J.APP_D_SCHEMA) == 5
    cats = [c for c, _f in J.APP_D_SCHEMA]
    assert cats == ["Content", "Form", "Voice", "Function", "Meta"]
    sizes = [len(fs) for _c, fs in J.APP_D_SCHEMA]
    assert sizes == [6, 4, 6, 5, 3]
    assert len(J.APP_D_FIELDS) == 24
    assert len(set(J.APP_D_FIELDS)) == 24, "duplicate field names"
    assert set(J.FIELD_TO_CATEGORY) == set(J.APP_D_FIELDS)
    # spot verbatim descriptions (fetched-verbatim provenance anchors)
    d = {f: desc for _c, fs in J.APP_D_SCHEMA for f, desc in fs}
    assert d["domain"] == "Broad subject area"
    assert d["creativity"] == "How formulaic vs. novel the approach is"


def test_schema_block_renders_every_field():
    block = J._schema_block()
    for f in J.APP_D_FIELDS:
        assert f"- {f}:" in block
    for c, _fs in J.APP_D_SCHEMA:
        assert f"{c}:" in block


# ── system prompts: no unfilled template slots; user msg is the block ────────────


def test_no_unfilled_slots_in_wave_systems():
    for name, system in J.WAVE_SYSTEMS.items():
        assert "{question}" not in system and "{answer}" not in system, name
    assert J._user_msg("THE BLOCK", "") == "THE BLOCK"
    assert J._user_msg("Q", "ignored-completion") == "Q"


def test_registry_covers_dispatch_waves_and_token_floors():
    assert set(J.WAVE_PARSERS) == {"w1", "w2", "w3", "w4", "w5", "w6"}
    assert set(J.MAX_TOKENS) == set(J.WAVE_PARSERS)
    assert J.MAX_TOKENS["w2"] >= 2048  # multi-field JSON rubric floor (rule 23)
    for w in ("w1", "w3", "w4", "w5", "w6"):
        assert J.MAX_TOKENS[w] >= 1024  # single-rationale floor (rule 23)
    assert J.W7_MAX_TOKENS >= 1024


def test_pilot_sizing_constants():
    # floor(1/threshold) + 1 (#2124 satisfiability at n_draws=1)
    assert J.PILOT_MIN_EFFECTIVE == int(1 / J.PILOT_PARSE_FAIL_THRESHOLD) + 1 == 51
    assert J.WAVE_THRESHOLD_BASE == 0  # batch route pinned (rule 26(c) OTPM region ban)


# ── rule-27 round-trips: realistic replies through parse_judge_json + wave parser ─


def _rt(wave: str, raw: str):
    return J.WAVE_PARSERS[wave](parse_judge_json(raw))


def test_w1_roundtrip():
    assert _rt("w1", '{"description": "Turns about cooking recipes."}') == (
        "Turns about cooking recipes."
    )
    fenced = '```json\n{"description": "Peak tokens are French words."}\n```'
    assert _rt("w1", fenced) == "Peak tokens are French words."
    assert _rt("w1", '{"description": ""}') is None
    assert _rt("w1", "no json here") is None


def _w2_reply(n_fields: int) -> str:
    payload = {f: f"short note about {f}" for f in J.APP_D_FIELDS[:n_fields]}
    return json.dumps(payload)


def test_w2_roundtrip():
    full = _rt("w2", _w2_reply(24))
    assert isinstance(full, dict) and len(full) == 24
    assert _rt("w2", _w2_reply(J.W2_MIN_FIELDS)) is not None
    assert _rt("w2", _w2_reply(J.W2_MIN_FIELDS - 1)) is None
    # case-insensitive keys + fenced variant
    payload = {f.upper(): "x" for f in J.APP_D_FIELDS}
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    got = _rt("w2", fenced)
    assert got is not None and len(got) == 24


def test_w3_roundtrip():
    reply = '{"reason": "varies along reader attitude", "field": "Tone", "category": "Voice"}'
    assert _rt("w3", reply) == ("tone", "Voice")
    # judge's own (wrong) category is IGNORED — derived from the schema
    wrong = '{"reason": "r", "field": "tone", "category": "Content"}'
    assert _rt("w3", wrong) == ("tone", "Voice")
    assert _rt("w3", '{"reason": "no fit", "field": "none", "category": "none"}') == (
        "none",
        "none",
    )
    assert _rt("w3", '{"reason": "r", "field": "sentiment polarity"}') is None
    assert _rt("w3", '{"reason": "r"}') is None


def test_w4_roundtrip():
    assert _rt("w4", '{"reason": "matches the code topic", "choice": "B"}') == "B"
    assert _rt("w4", '{"reason": "r", "choice": "(c)"}') == "C"
    assert _rt("w4", '```json\n{"reason": "r", "choice": "j"}\n```') == "J"
    assert _rt("w4", '{"reason": "r", "choice": "K"}') is None
    assert _rt("w4", '{"reason": "r", "choice": 2}') is None


def test_w5_roundtrip():
    assert _rt("w5", '{"reason": "list one covers tone too", "choice": 1}') == 1
    assert _rt("w5", '{"reason": "r", "choice": "2"}') == 2
    assert _rt("w5", '{"reason": "r", "choice": "List 2"}') == 2
    assert _rt("w5", '{"reason": "r", "choice": 3}') is None
    assert _rt("w5", '{"reason": "r", "choice": true}') is None


def test_w6_roundtrip():
    reply = '{"reason": "r", "ranking": ["b", "d", "a", "e", "c"]}'
    assert _rt("w6", reply) == ["B", "D", "A", "E", "C"]
    assert _rt("w6", '{"reason": "r", "ranking": ["A", "A", "B", "C", "D"]}') is None
    assert _rt("w6", '{"reason": "r", "ranking": ["A", "B", "C", "D"]}') is None
    assert _rt("w6", '{"reason": "r"}') is None


# ── batch custom-id grammar ──────────────────────────────────────────────────────


def test_item_id_grammar_per_wave_shapes():
    ids = [
        "w1-rep_ta-f32767",
        "w1-mat_k100-f2658",
        "w1-pt-f81919",
        "w2-r1048575",
        "w3-mat_k200-f7",
        "w4-pt_max-r999983",
        "w5-r999983-9",
        "w6-r0",
        "w3-rep_ta-cal-f7",
        "w3s-0",
        "w4s-0",
    ]
    for i in ids:
        assert J._ITEM_ID_RE.match(i) and "__" not in i and len(i) <= 53, i
    J._assert_item_ids([(i, "block") for i in ids])
    with pytest.raises(AssertionError):
        J._assert_item_ids([("bad__id", "block")])
    with pytest.raises(AssertionError):
        J._assert_item_ids([("has space", "block")])


# ── presentation determinism (seed 2552) ─────────────────────────────────────────


def test_w4_presentation_deterministic_and_config_free():
    pool = list(range(1000, 1060))
    a = J.w4_presentation(1003, pool)
    b = J.w4_presentation(1003, pool)
    assert a == b  # config-invariant BY CONSTRUCTION: keyed on row id only
    assert len(a["candidates"]) == 10 and len(set(a["candidates"])) == 10
    assert a["candidates"].count(1003) == 1
    gold_pos = a["candidates"].index(1003)
    assert a["gold_label"] == J.W4_LABELS[gold_pos]
    c = J.w4_presentation(1004, pool)
    assert c != a  # distinct rows draw distinct presentations (generic)


def test_w5_assignment_deterministic_and_mixed():
    vals = [J.w5_assignment(r, k) for r in range(40) for k in range(10)]
    assert vals == [J.w5_assignment(r, k) for r in range(40) for k in range(10)]
    assert any(vals) and not all(vals)  # both orders occur across rows/pairs


def test_w6_assignment_is_config_permutation():
    a = J.w6_assignment(7)
    assert list(a.keys()) == list(J.W6_LABELS)
    assert sorted(a.values()) == sorted(J.CONFIGS)
    assert a == J.w6_assignment(7)


# ── G2 arithmetic: need sets + stratified eval subset ────────────────────────────


def _lists_doc(turn_feats: dict[str, dict[int, list[int]]]) -> dict:
    doc = {}
    for cfg in J.CONFIGS:
        rows = turn_feats.get(cfg, {})
        doc[cfg] = {
            "turns": [
                {"row_id": r, "judged_top100": [[f, 0.5] for f in feats]}
                for r, feats in rows.items()
            ]
        }
    return doc


def test_compute_need_sets_union_arithmetic():
    doc = _lists_doc(
        {
            "rep_ta": {10: [100, 101], 99: [777]},  # row 99 outside eval -> excluded
            "mat_k100": {10: [200]},
            "mat_k200": {10: [300]},
            "pt_max": {10: [400, 401]},
            "pt_sum": {10: [401, 402]},
        }
    )
    need = J.compute_need_sets(
        {10},
        doc,
        rep_panel=np.array([1, 2], dtype=np.int64),
        mat_panels={
            "mat_k100": np.array([3], dtype=np.int64),
            "mat_k200": np.array([4], dtype=np.int64),
        },
    )
    assert need["rep_ta"] == {1, 2, 100, 101}
    assert need["mat_k100"] == {3, 200}
    assert need["mat_k200"] == {4, 300}
    assert need["pt"] == {400, 401, 402}  # pt_max UNION pt_sum, no panel
    assert sum(len(s) for s in need.values()) == 11


def test_stratified_eval_subset_mirror():
    eval_ids = np.arange(200, dtype=np.int64)
    prov = np.zeros(200, dtype=np.uint8)
    prov[120:] = 1  # 120 lmsys (0) + 80 wildchat (1)
    sub = J.stratified_eval_subset(eval_ids, prov, 50)
    assert len(sub) == 50 and len(set(sub.tolist())) == 50
    assert set(sub.tolist()) <= set(eval_ids.tolist())
    assert list(sub) == sorted(sub)
    assert int((prov[sub] == 0).sum()) == 30  # round(50 * 120/200)
    assert int((prov[sub] == 1).sum()) == 20
    again = J.stratified_eval_subset(eval_ids, prov, 50)
    assert np.array_equal(sub, again)  # seed-2552 deterministic


# ── draw classification + reduce ────────────────────────────────────────────────


def test_classify_draw_precedence():
    w4 = J.WAVE_PARSERS["w4"]
    assert J.classify_draw({"error": True, "transport": True}, w4)[0] == "transport"
    # transport beats a refusal stop_reason on the SAME dict (consumer precedence, #2206)
    assert (
        J.classify_draw({"error": True, "transport": True, "stop_reason": "refusal"}, w4)[0]
        == "transport"
    )
    assert J.classify_draw({"error": True, "stop_reason": "max_tokens"}, w4)[0] == "truncation"
    assert J.classify_draw({"error": True, "stop_reason": "refusal"}, w4)[0] == "api_refusal"
    # parse-error dict with recoverable raw text -> valid
    cls, val = J.classify_draw(
        {"error": True, "reasoning": "parse_error: no json", "_raw_text": '{"choice": "B"}'},
        w4,
    )
    assert (cls, val) == ("valid", "B")
    assert (
        J.classify_draw({"error": True, "reasoning": "parse_error", "_raw_text": "junk"}, w4)[0]
        == "parse_fail"
    )
    # success dicts
    ok = {"reason": "r", "choice": "A", "stop_reason": "end_turn", "_raw_text": "..."}
    assert J.classify_draw(ok, w4) == ("valid", "A")
    assert J.classify_draw({"choice": "A", "stop_reason": "max_tokens"}, w4)[0] == "truncation"
    assert J.classify_draw({"stop_reason": "refusal"}, w4)[0] == "api_refusal"
    # payload unparseable but raw recoverable -> valid via _raw_text fallback
    cls, val = J.classify_draw(
        {"weird": 1, "stop_reason": "end_turn", "_raw_text": '{"choice": "A"}'}, w4
    )
    assert (cls, val) == ("valid", "A")
    assert J.classify_draw("legacy-non-dict", w4)[0] == "parse_fail"


def test_reduce_all_scores_and_arm_stats_floor():
    w5 = J.WAVE_PARSERS["w5"]
    all_scores: dict[str, object] = {}
    item_ids = [f"i{k}" for k in range(20)]
    for k, i in enumerate(item_ids):
        cid = f"{i}__{k:05d}__00"
        if k == 0:
            all_scores[cid] = {"error": True, "stop_reason": "refusal"}
        elif k == 1:
            all_scores[cid] = {"error": True, "transport": True}
        else:
            all_scores[cid] = {"reason": "r", "choice": 1, "stop_reason": "end_turn"}
    per_item = J.reduce_all_scores(all_scores, w5)
    assert per_item["i0"]["class"] == "api_refusal"
    assert per_item["i1"]["class"] == "transport"
    assert sum(1 for r in per_item.values() if r["class"] == "valid") == 18
    stats = J._arm_stats(item_ids, per_item)
    assert stats["n_items"] == 20 and stats["n_valid"] == 18
    assert stats["frac_items_complete"] == pytest.approx(0.9)
    assert stats["below_floor"] is True  # 0.9 < 0.95 pre-registered floor
    # a sync re-issue overlay flipping the censored items clears the floor
    per_item["i0"] = {"class": "valid", "value": 1, "via": "sync_reissue"}
    per_item["i1"] = {"class": "valid", "value": 2, "via": "sync_reissue"}
    stats2 = J._arm_stats(item_ids, per_item)
    assert stats2["below_floor"] is False and stats2["n_valid"] == 20
    # missing results are counted, never silently narrowed
    stats3 = J._arm_stats([*item_ids, "i-missing"], per_item)
    assert stats3["n_missing_results"] == 1


def test_reduce_keeps_best_class_draw_per_item():
    w4 = J.WAVE_PARSERS["w4"]
    all_scores = {
        "a__00000__00": {"error": True, "transport": True},
        "a__00000__01": {"reason": "r", "choice": "A", "stop_reason": "end_turn"},
    }
    per = J.reduce_all_scores(all_scores, w4)
    assert per["a"] == {"class": "valid", "value": "A"}


# ── W7 kappa ─────────────────────────────────────────────────────────────────────


def test_cohen_kappa_reference_values():
    assert J._cohen_kappa(["x", "y", "x"], ["x", "y", "x"]) == pytest.approx(1.0)
    # po = 0.5, pe = 0.5 -> kappa = 0
    assert J._cohen_kappa(["x", "x", "y", "y"], ["x", "y", "x", "y"]) == pytest.approx(0.0)
    with pytest.raises(AssertionError):
        J._cohen_kappa([], [])


# ── helper invariants ────────────────────────────────────────────────────────────


def test_desc_list_lines_counts_missing():
    lines, n_missing = J._desc_list_lines([1, 2, 3], {1: "one", 3: "three"})
    assert lines == ["- one", "- three"] and n_missing == 1


def test_render_summary_orders_by_schema():
    fields = {f: f"v-{f}" for f in J.APP_D_FIELDS}
    out = J._render_summary(fields).splitlines()
    assert out == [f"{f}: v-{f}" for f in J.APP_D_FIELDS]
    partial = J._render_summary({"tone": "warm"})
    assert partial == "tone: warm"
