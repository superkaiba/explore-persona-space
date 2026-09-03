"""Unit tests for scripts/issue2661_judge_waves.py: parser round-trips (rule 27
— malformed dropped, never coerced), W4 presentation determinism, the estimate
arithmetic, the item-id grammar, and the 2,000/2,000 eval-row/text join assert."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "vendored_2476"))

import issue2661_judge_waves as J  # noqa: E402


def test_judge_model_and_no_prefill_contract():
    # task-body ## Provenance pins the judge override; waiver below is for
    # workflow_lint check-judge-model-pin, not a ruff code
    assert J.JUDGE_MODEL == "claude-opus-5"  # noqa: judge-model-pin
    # _user_msg drops the completion (n_draws=1, empty answer slot; no prefill)
    assert J._user_msg("QUESTION BLOCK", "ignored") == "QUESTION BLOCK"


def test_parse_w1_roundtrip_and_malformed_dropped():
    assert J.parse_w1({"description": " topic X prompts "}) == "topic X prompts"
    assert J.parse_w1({"description": ""}) is None
    assert J.parse_w1({"desc": "x"}) is None
    assert J.parse_w1("bare string") is None
    assert J.parse_w1(None) is None


def test_parse_w2_field_floor_and_coercion_rules():
    full = {f: f"v-{i}" for i, f in enumerate(J.APP_D_FIELDS)}
    got = J.parse_w2(full)
    assert got is not None and len(got) == 24
    # numeric scalars stringify; below the 20-field floor -> None (dropped)
    partial = {f: 1 for f in J.APP_D_FIELDS[:20]}
    got20 = J.parse_w2(partial)
    assert got20 is not None and got20[J.APP_D_FIELDS[0]] == "1"
    assert J.parse_w2({f: "v" for f in J.APP_D_FIELDS[:19]}) is None
    assert J.parse_w2(["not", "a", "dict"]) is None
    # unknown keys never coerce into schema fields
    assert J.parse_w2({"bogus": "v"}) is None


def test_parse_w4_labels():
    assert J.parse_w4({"choice": "C"}) == "C"
    assert J.parse_w4({"choice": " (j) "}) == "J"
    assert J.parse_w4({"choice": "K"}) is None
    assert J.parse_w4({"choice": 3}) is None
    assert J.parse_w4({}) is None


def test_w4_presentation_deterministic_and_gold_correct():
    pool = list(range(100, 130))
    a = J.w4_presentation(107, pool)
    b = J.w4_presentation(107, pool)
    assert a == b, "presentation must be deterministic (seed 2661, row-keyed)"
    assert len(a["candidates"]) == 10 and len(set(a["candidates"])) == 10
    assert 107 in a["candidates"]
    gold_pos = a["candidates"].index(107)
    assert J.W4_LABELS[gold_pos] == a["gold_label"]
    assert all(c in pool for c in a["candidates"])
    c = J.w4_presentation(108, pool)
    assert c != a


def test_estimate_wave_arithmetic():
    system = "S" * 70
    items = [("w2-r1", "Q" * 350), ("w2-r2", "Q" * 350)]
    est = J._est_wave(items, system, "w2")
    assert est["n_items"] == 2
    assert est["input_chars"] == 2 * 350 + 2 * 70
    in_tok = est["input_chars"] / J.CHARS_PER_TOKEN
    assert est["est_input_tokens"] == int(in_tok)
    assert est["output_token_cap"] == 2 * J.MAX_TOKENS["w2"]
    want_ub = in_tok / 1e6 * J.BATCH_USD_PER_MTOK_IN + (
        2 * J.MAX_TOKENS["w2"] / 1e6 * J.BATCH_USD_PER_MTOK_OUT
    )
    assert est["usd_upper_bound"] == pytest.approx(want_ub)
    assert est["usd_expected"] < est["usd_upper_bound"]


def test_item_id_grammar_rejects_double_underscore_and_overlong():
    J._assert_item_ids([("w1-ctx-f12", "q")])
    with pytest.raises(AssertionError):
        J._assert_item_ids([("w1__ctx", "q")])
    with pytest.raises(AssertionError):
        J._assert_item_ids([("x" * 54, "q")])


def test_eval_text_join_asserts_full_coverage(tmp_path):
    """The 2,000/2,000 eval-row/text join: _load_texts fails loud on ANY
    missing row and passes only at full coverage (tiny-N twin of the assert)."""
    p = SimpleNamespace(work=tmp_path)
    eval_ids = np.asarray([5, 7, 9], np.int64)
    rows = [{"row_id": 5, "ci": 1, "text": "a"}, {"row_id": 7, "ci": 2, "text": "b"}]
    J._texts_path(p).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(AssertionError, match="missing"):
        J._load_texts(p, eval_ids)
    with J._texts_path(p).open("a") as fh:
        fh.write(json.dumps({"row_id": 9, "ci": 3, "text": "c"}) + "\n")
    got = J._load_texts(p, eval_ids)
    assert got == {5: "a", 7: "b", 9: "c"}


def test_estimate_gate_refuses_over_budget(tmp_path, monkeypatch):
    """_require_estimate: missing file fails loud; over-budget exits rc=9;
    under-budget passes; smoke/dry-run exempt."""
    est = tmp_path / "judge_estimate.json"
    monkeypatch.setattr(J, "ESTIMATE_PATH", est)
    args = Namespace(dry_run=False, smoke=False, budget_usd=300.0)
    with pytest.raises(AssertionError, match=r"judge_estimate\.json missing"):
        J._require_estimate(args)
    est.write_text(json.dumps({"total": {"usd_upper_bound": 301.0}}))
    with pytest.raises(SystemExit) as ei:
        J._require_estimate(args)
    assert ei.value.code == J.RC_BUDGET_FAIL
    est.write_text(json.dumps({"total": {"usd_upper_bound": 299.0}}))
    J._require_estimate(args)  # passes
    J._require_estimate(Namespace(dry_run=True, smoke=False, budget_usd=0.0))  # exempt


def test_w1_block_shows_activations_and_negatives():
    recs = {
        "positive": [
            {"activation": 3.25, "text": "how do I cook rice", "rank": 0},
            {"activation": 1.5, "text": "rice cooker settings", "rank": 1},
        ],
        "negative": [{"activation": 0.0, "text": "python question", "rank": 0}],
    }
    block = J._w1_block_ctx(recs)
    assert "activation=3.2500" in block and "how do I cook rice" in block
    assert "NON-ACTIVATING NEGATIVES" in block and "python question" in block
    assert block.index("how do I cook rice") < block.index("python question")
