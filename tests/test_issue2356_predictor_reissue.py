"""r8 plan-gap fix pins (#2356): the PREDICTOR wave's rule-28 targeted SYNC
re-issue of api-refusal-censored rows (plan §4 Step E / "Censored-row
handling (S3)").

Pre-fix, ``run_predictor`` had NO sync re-issue: a row whose batch draws were
all api-refusal-censored (``stop_reason == "refusal"`` + empty content — the
rule-28 shape) kept score ``None`` and was silently dropped downstream;
api-refusal censoring is outcome-correlated, so on armB (pilot: 13.3% batch
censoring) the drop biases the paired contrast. These tests FAIL pre-fix: no
sync re-issue call is made, recovered scores are never merged, and the
rule-28 accounting block is absent from ``predictor_scores.json``.

Test shape (code-style § "One production-body test per seam-stubbed
function"): the REAL ``run_predictor`` body executes end-to-end on a tmp
splits fixture; ONLY the external API boundary (``judge_graded``) is replaced
by a ``unittest.mock.create_autospec`` fake (signature-conformant by
construction) returning real ``JudgeResult`` dataclass instances. No network,
no worktree-absolute paths (module imported relative to this test file's repo
root); prompts are neutral placeholders (content hygiene).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import issue2356_judge as j  # noqa: E402

from explore_persona_space.eval.graded_judge import JudgeResult  # noqa: E402


def _sha(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


# FULL 64-hex predictor row_ids (the fits.py bare-sha shape).
A_ROWS = [_sha(f"armA-row-{i}") for i in range(4)]
B_ROWS = [_sha(f"armB-row-{i}") for i in range(3)]
A_SHORT = [j._short_item_id(r) for r in A_ROWS]
B_SHORT = [j._short_item_id(r) for r in B_ROWS]

# Batch-leg synthetic outcome (scores keyed by SHORT dispatch id):
#   armA: a0=80.0 kept, a1=None ALL-api-refusal (5/5), a2=None content-only,
#         a3=60.0 kept  -> only a1 is re-issued; a2 stays a rule-9 residual.
#   armB: b0=40.0 kept, b1=None api-refusal-MIXED (3 api + 2 content),
#         b2=70.0 kept  -> b1 re-issued but STILL censored on sync (residual).
BATCH_A_SCORES = {A_SHORT[0]: 80.0, A_SHORT[1]: None, A_SHORT[2]: None, A_SHORT[3]: 60.0}
BATCH_A_API = {A_SHORT[1]: 5}
BATCH_B_SCORES = {B_SHORT[0]: 40.0, B_SHORT[1]: None, B_SHORT[2]: 70.0}
BATCH_B_API = {B_SHORT[1]: 3}
REISSUE_A_SCORES = {A_SHORT[1]: 30.0}  # recovered on sync
REISSUE_B_SCORES = {B_SHORT[1]: None}  # still censored on sync -> residual
OVERLAP_SYNC_DELTA = 0.5  # sync = batch - 0.5 -> offset_batch_minus_sync = +0.5


def _mk_result(
    scores: dict[str, float | None],
    api: dict[str, int] | None = None,
    n_draws: int = j.PREDICTOR_N_DRAWS,
) -> JudgeResult:
    kept = {sid: ([s] if s is not None else []) for sid, s in scores.items()}
    return JudgeResult(
        scores=dict(scores),
        n_total_draws=n_draws * len(scores),
        n_dropped_draws=0,
        per_item_draw_counts={sid: len(v) for sid, v in kept.items()},
        per_item_scores=kept,
        n_api_refusal_draws=sum((api or {}).values()),
        per_item_api_refusals=dict(api or {}),
    )


@pytest.fixture()
def judge_env(tmp_path, monkeypatch):
    """tmp splits fixture + autospec'd judge_graded fake; returns
    (args, out_root, calls) where calls records every judge_graded call."""
    eval_root = tmp_path / "eval_results" / "issue_2356"
    splits = eval_root / "splits"
    splits.mkdir(parents=True)
    rows = [
        {"row_id": rid, "prompt": f"q-armA-{i}", "arm": "armA", "fold": 0}
        for i, rid in enumerate(A_ROWS)
    ] + [
        {"row_id": rid, "prompt": f"q-armB-{i}", "arm": "armB", "fold": 0}
        for i, rid in enumerate(B_ROWS)
    ]
    (splits / "balanced_eval_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    train = [
        {
            "row_id": _sha(f"train-{i}"),
            "prompt": f"tr-q-{i}",
            "label": "engage" if i % 2 == 0 else "refuse",
            "group_id": f"g{i}",
        }
        for i in range(8)
    ]
    for arm in ("armA", "armB"):
        (splits / f"train_rows_{arm}_fold0.json").write_text(json.dumps(train), encoding="utf-8")

    calls: list[dict] = []

    def _fake(items, eval_prompt, **kw):
        save_raw = Path(kw["save_raw"])
        leg = (
            "reissue"
            if save_raw.parent.name == "reissue_save_raw"
            else "overlap"
            if save_raw.name.startswith("overlap_save_raw_")
            else "batch"
        )
        rec = {"leg": leg, "items": list(items), "eval_prompt": eval_prompt, **kw}
        calls.append(rec)
        sids = {sid for sid, _q, _a in items}
        if leg == "batch":
            if sids == set(A_SHORT):
                return _mk_result(BATCH_A_SCORES, BATCH_A_API)
            assert sids == set(B_SHORT), f"unexpected batch item set: {sids}"
            return _mk_result(BATCH_B_SCORES, BATCH_B_API)
        if leg == "reissue":
            if sids <= set(A_SHORT):
                return _mk_result(REISSUE_A_SCORES)
            return _mk_result(REISSUE_B_SCORES, {B_SHORT[1]: j.PREDICTOR_N_DRAWS})
        # overlap leg: sync = batch - OVERLAP_SYNC_DELTA on every sampled row
        batch = {**BATCH_A_SCORES, **BATCH_B_SCORES}
        return _mk_result({sid: batch[sid] - OVERLAP_SYNC_DELTA for sid in sorted(sids)})

    fake = create_autospec(j.judge_graded, side_effect=_fake)
    monkeypatch.setattr(j, "judge_graded", fake)

    out_root = eval_root / "judge"
    args = SimpleNamespace(
        arm=None, zero_shot=False, smoke=False, dry_run=False, out_root=str(out_root)
    )
    return args, out_root, calls


def _run(args, out_root):
    rc = j.run_predictor(args)
    assert rc == 0
    payload = json.loads(
        (out_root / "predictor" / "predictor_scores.json").read_text(encoding="utf-8")
    )
    return payload


def test_censored_row_reissued_on_sync_and_merged(judge_env):
    args, out_root, _calls = judge_env
    payload = _run(args, out_root)
    scores = payload["scores"]

    # FULL row_ids everywhere in the durable output (r7 short-id contract).
    assert set(scores) == set(A_ROWS + B_ROWS)
    assert all(len(rid) == 64 for rid in scores)

    # Recovered row: sync score merged over the batch None, oriented, tagged.
    a1 = scores[A_ROWS[1]]
    assert a1["p_answer"] == 30.0
    assert a1["p_refuse"] == 70.0
    assert a1["judge_transport"] == "sync-reissue"

    # Content-only zero-valid row: NOT re-issued, stays None (mask-excluded).
    a2 = scores[A_ROWS[2]]
    assert a2["p_answer"] is None and a2["p_refuse"] is None
    assert a2["judge_transport"] == "batch"

    # Still-censored-after-sync row: stays None (residual, mask-excluded).
    b1 = scores[B_ROWS[1]]
    assert b1["p_answer"] is None and b1["p_refuse"] is None

    # Batch-succeeded rows untouched.
    assert scores[A_ROWS[0]]["p_answer"] == 80.0
    assert scores[A_ROWS[0]]["judge_transport"] == "batch"

    assert payload["n_valid_scores"] == 5  # 4 batch-kept + 1 recovered


def test_reissue_call_is_sync_at_identical_instrument(judge_env):
    args, out_root, calls = judge_env
    _run(args, out_root)

    batch_calls = [c for c in calls if c["leg"] == "batch"]
    reissue_calls = [c for c in calls if c["leg"] == "reissue"]
    assert len(reissue_calls) == 2  # one per (arm, fold) with censored rows

    batch_items = {sid: (sid, q, a) for c in batch_calls for sid, q, a in c["items"]}
    for c in reissue_calls:
        # SYNC path via the run_rejudge_refusals mechanism; batch leg forced batch.
        assert c["force_sync"] is True
        assert "threshold_base" not in c or c["threshold_base"] is None
        # Identical instrument: rubric/model/temp/max_tokens/n_draws pins.
        assert c["eval_prompt"] == j.PREDICTOR_RUBRIC
        assert c["judge_model"] == j.JUDGE_MODEL
        assert c["temperature"] == j.PREDICTOR_TEMPERATURE
        assert c["max_tokens"] == j.PREDICTOR_MAX_TOKENS
        assert c["n_draws"] == j.PREDICTOR_N_DRAWS
        # Identical per-fold few-shot demo block: item tuples byte-equal batch's.
        for item in c["items"]:
            assert item == batch_items[item[0]]
        # Fresh cache dir — never the batch cache (never cache-served).
        assert Path(c["cache_dir"]).parent.name == "reissue_cache"
    for c in batch_calls:
        assert c["threshold_base"] == 0 and not c.get("force_sync")

    # Exactly the api-refusal-censored rows re-issued: a1 + b1, never a2.
    reissued_sids = {sid for c in reissue_calls for sid, _q, _a in c["items"]}
    assert reissued_sids == {A_SHORT[1], B_SHORT[1]}


def test_per_arm_rule28_accounting(judge_env):
    args, out_root, _calls = judge_env
    payload = _run(args, out_root)
    per_arm = payload["accounting"]["rule28_sync_reissue"]["per_arm"]

    a = per_arm["armA"]
    assert a["n_api_refusal_draws_batch"] == 5
    assert a["n_items_zero_valid_batch"] == 2
    assert a["zero_valid_by_class"] == {"all_api_refusal": 1, "content_only": 1}
    assert a["n_reissued"] == 1
    assert a["n_recovered"] == 1
    assert a["n_residual_zero_valid"] == 1
    assert a["reissued_row_ids"] == [A_ROWS[1]]  # FULL ids in durable accounting
    assert a["residual_zero_valid_row_ids"] == [A_ROWS[2]]

    b = per_arm["armB"]
    assert b["n_api_refusal_draws_batch"] == 3
    assert b["n_items_zero_valid_batch"] == 1
    assert b["zero_valid_by_class"] == {"api_refusal_mixed": 1}
    assert b["n_reissued"] == 1
    assert b["n_recovered"] == 0  # sync still censored -> residual
    assert b["n_residual_zero_valid"] == 1
    assert b["residual_zero_valid_row_ids"] == [B_ROWS[1]]


def test_overlap_offset_reported(judge_env):
    args, out_root, calls = judge_env
    payload = _run(args, out_root)
    overlap = payload["accounting"]["batch_vs_sync_overlap"]

    # Overlap pool = batch-SUCCEEDED rows only (never recovered/censored ones).
    overlap_sids = {sid for c in calls if c["leg"] == "overlap" for sid, _q, _a in c["items"]}
    assert overlap_sids == {A_SHORT[0], A_SHORT[3], B_SHORT[0], B_SHORT[2]}
    for c in calls:
        if c["leg"] == "overlap":
            assert c["force_sync"] is True

    assert overlap["n_overlap"] == 4
    assert overlap["offset_batch_minus_sync"] == pytest.approx(OVERLAP_SYNC_DELTA)
    for arm in ("armA", "armB"):
        assert overlap["per_arm"][arm]["offset_batch_minus_sync"] == pytest.approx(
            OVERLAP_SYNC_DELTA
        )
