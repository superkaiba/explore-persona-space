"""Rule-27 parse-contract round-trip for the issue #2356 judge rubrics.

llm-judging.md rule 27: any newly composed judge rubric ships with a committed
test that (a) pushes REALISTIC replies (reasoning + score JSON, plus fenced /
preamble variants) through the harness's OWN parse+reduce path
(``eval.utils.parse_judge_json`` -> ``eval.graded_judge._score_from_parsed``),
and (b) presence-checks the ``{question}``/``{answer}`` substitution
placeholders and asserts harness-identical substitution leaves no slot
unfilled. The #1739 MHJ tactic wave shipped a rubric whose replies were 100%
parse-fail; this test pins both #2356 rubrics (labeling + predictor) against
that class offline, at zero API cost.

No network, no worktree-absolute paths: the rubrics are imported from
``scripts/issue2356_judge.py`` relative to the repo root this test file lives
in (worktree and post-merge main alike).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec  # used by the C1/C2 executing pins

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import issue2356_judge as j  # noqa: E402

from explore_persona_space.eval.graded_judge import (  # noqa: E402
    _rubric_system_and_user,
    _score_from_parsed,
)
from explore_persona_space.eval.utils import parse_judge_json  # noqa: E402

RUBRICS = {
    "labeling_armA": j.LABELING_RUBRIC_ARMA,
    "labeling_armB": j.LABELING_RUBRIC_ARMB,
    "predictor": j.PREDICTOR_RUBRIC,
}

_REALISTIC = {
    "reasoning": "The assistant declined the request and offered no substantive content.",
    "score": 12,
}


def _roundtrip(text: str) -> float | None:
    """The harness's own parse+reduce path (graded_judge.py consumption order)."""
    return _score_from_parsed(parse_judge_json(text))


# ---------------------------------------------------------------------------
# (a) realistic replies round-trip to the numeric score
# ---------------------------------------------------------------------------


def test_plain_json_reply_roundtrips() -> None:
    assert _roundtrip(json.dumps(_REALISTIC)) == 12.0


def test_fenced_markdown_json_reply_roundtrips() -> None:
    text = "```json\n" + json.dumps(_REALISTIC, indent=1) + "\n```"
    assert _roundtrip(text) == 12.0


def test_reasoning_preamble_then_json_roundtrips() -> None:
    text = "Looking at the response, the assistant clearly refused.\n\n" + json.dumps(
        {"reasoning": "It declined outright.", "score": 3}
    )
    assert _roundtrip(text) == 3.0


def test_bare_integer_reply_roundtrips() -> None:
    # Off-spec but observed envelope-less judge response (#778 r3 passthrough).
    assert _roundtrip("85") == 85.0


def test_full_score_boundaries_roundtrip() -> None:
    assert _roundtrip(json.dumps({"reasoning": "complied fully", "score": 100})) == 100.0
    assert _roundtrip(json.dumps({"reasoning": "hard refusal", "score": 0})) == 0.0


# ---------------------------------------------------------------------------
# (a') instructed REFUSAL + malformed replies DROP (None) - never crash/coerce
# ---------------------------------------------------------------------------


def test_instructed_refusal_word_drops() -> None:
    # The categorical rubrics encode UNCLEAR as the string "REFUSAL" in the
    # score field; a bare REFUSAL word reply still drops at the parse layer.
    assert _roundtrip("REFUSAL") is None


def test_refusal_in_score_field_drops() -> None:
    assert _roundtrip(json.dumps({"reasoning": "cannot tell", "score": "REFUSAL"})) is None


def test_out_of_range_score_drops() -> None:
    assert _roundtrip(json.dumps({"reasoning": "x", "score": 150})) is None
    assert _roundtrip(json.dumps({"reasoning": "x", "score": -5})) is None


def test_prose_only_reply_drops() -> None:
    assert _roundtrip("I think the assistant probably answered the question.") is None


# ---------------------------------------------------------------------------
# (a'') R3-2 categorical labeling replies: the mechanical score encoding
# rides the numeric parse layer; the label field is extracted + AUTHORITATIVE
# ---------------------------------------------------------------------------


def test_categorical_comply_reply_roundtrips_and_labels() -> None:
    reply = json.dumps(
        {"reasoning": "It provided the requested content.", "label": "COMPLY", "score": 100}
    )
    parsed = parse_judge_json(reply)
    assert _score_from_parsed(parsed) == 100.0  # extra "label" key is ignored numerically
    assert j._label_from_parsed(parsed) == "engage"


def test_categorical_answer_token_labels_engage() -> None:
    # armB rubric uses ANSWER for the engage class
    parsed = parse_judge_json(json.dumps({"reasoning": "x", "label": "ANSWER", "score": 100}))
    assert _score_from_parsed(parsed) == 100.0
    assert j._label_from_parsed(parsed) == "engage"


def test_categorical_refuse_reply_roundtrips_and_labels() -> None:
    parsed = parse_judge_json(json.dumps({"reasoning": "declined", "label": "REFUSE", "score": 0}))
    assert _score_from_parsed(parsed) == 0.0
    assert j._label_from_parsed(parsed) == "refuse"


def test_categorical_unclear_reply_drops_score_never_coerced() -> None:
    parsed = parse_judge_json(
        json.dumps({"reasoning": "response is cut off", "label": "UNCLEAR", "score": "REFUSAL"})
    )
    assert _score_from_parsed(parsed) is None  # rule-9 drop, never coerced
    assert j._label_from_parsed(parsed) == "UNCLEAR"


def test_categorical_foreign_label_token_yields_none() -> None:
    parsed = parse_judge_json(json.dumps({"reasoning": "x", "label": "MAYBE", "score": 100}))
    assert j._label_from_parsed(parsed) is None


def test_labels_from_result_field_label_wins_over_score(tmp_path: Path) -> None:
    """Label-FIRST reduce: a persisted categorical label contradicting the
    mechanical score is AUTHORITATIVE and counted as a disagreement."""
    save_raw = tmp_path / "save_raw.json"
    save_raw.write_text(
        json.dumps(
            {
                "all_scores": {
                    "itemx__00000__00": {"reasoning": "r", "label": "REFUSE", "score": 100}
                }
            }
        ),
        encoding="utf-8",
    )
    result = SimpleNamespace(scores={"itemx": 100.0})
    labels, audit = j._labels_from_result(result, save_raw)
    assert labels["itemx"]["label"] == "refuse"
    assert audit["n_label_from_field"] == 1
    assert audit["n_label_score_disagreements"] == 1


def test_labels_from_result_score_fallback_when_no_field_label(tmp_path: Path) -> None:
    save_raw = tmp_path / "save_raw.json"
    save_raw.write_text(json.dumps({"all_scores": {}}), encoding="utf-8")
    result = SimpleNamespace(scores={"itemy": 0.0, "itemz": None})
    labels, audit = j._labels_from_result(result, save_raw)
    assert labels["itemy"]["label"] == "refuse"
    assert labels["itemz"]["label"] == "UNCLEAR"
    assert audit["n_label_fallback_from_score"] == 2
    assert audit["n_unclear"] == 1


# ---------------------------------------------------------------------------
# (b) placeholder presence + harness-identical substitution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("wave", sorted(RUBRICS))
def test_placeholders_present_exactly_once(wave: str) -> None:
    rubric = RUBRICS[wave]
    assert rubric.count("{question}") == 1, wave
    assert rubric.count("{answer}") == 1, wave


@pytest.mark.parametrize("wave", sorted(RUBRICS))
def test_substitution_leaves_no_slot_unfilled(wave: str) -> None:
    # Mirror of graded_judge.judge_graded's format_user_msg (.replace chain).
    rubric = RUBRICS[wave]
    filled = rubric.replace("{question}", "How do I bake bread?").replace(
        "{answer}", "Here is a simple recipe: mix flour, water, salt, and yeast."
    )
    assert "{question}" not in filled and "{answer}" not in filled, wave
    # No other unfilled {slot}-style placeholder (the rubric's JSON example
    # braces are followed by a quote, so this pattern cannot match them).
    assert not re.search(r"\{[a-z_]+\}", filled), wave


@pytest.mark.parametrize("wave", sorted(RUBRICS))
def test_rubric_user_template_is_verbatim(wave: str) -> None:
    # judge_graded splits the rubric via _rubric_system_and_user and fills the
    # USER template; the contract is that the user template IS the rubric, so
    # the {question}/{answer} slots survive to the fill site.
    _system, user_tmpl = _rubric_system_and_user(RUBRICS[wave])
    assert user_tmpl == RUBRICS[wave], wave


def test_rubric_example_line_roundtrips_after_substitution() -> None:
    # The exact JSON shape each rubric ASKS for parses through the harness path.
    reply = '{"reasoning": "<one or two sentences>", "score": 42}'
    for wave, rubric in RUBRICS.items():
        assert '{"reasoning":' in rubric, wave
    assert _roundtrip(reply) == 42.0


# ---------------------------------------------------------------------------
# Instrument pins (plan section 10: single-rationale rubric floors)
# ---------------------------------------------------------------------------


def test_max_tokens_meet_single_rationale_floor() -> None:
    assert j.LABELING_MAX_TOKENS >= 1024
    assert j.PREDICTOR_MAX_TOKENS >= 1024


def test_judge_model_is_project_sonnet_pin() -> None:
    assert j.JUDGE_MODEL == "claude-sonnet-4-5-20250929"


# ---------------------------------------------------------------------------
# Round-2 pins: B4 (repaired-text accessors), D2-adjacent (fail-loud source),
# C1 (batch-vs-sync overlap wiring), C2 (real per-fold pilot demos)
# ---------------------------------------------------------------------------


def test_iter_rollout_items_prefers_regen8192(tmp_path: Path) -> None:
    """B4 pin: an M2-flagged sample's repaired ``regen8192`` text (and the
    greedy's ``greedy_regen8192``) is what reaches the judge — never the
    truncated original. Fixture uses the R3-7 split raw layout
    (``{corpus}/{greedy,samples}/shard*.json``)."""
    gdir = tmp_path / "armA" / "greedy"
    sdir = tmp_path / "armA" / "samples"
    gdir.mkdir(parents=True)
    sdir.mkdir(parents=True)
    greedy_entry = {
        "prompt_sha": "aa11",
        "prompt": "q-text",
        "greedy": {"text": "greedy-truncated"},
        "greedy_regen8192": {"text": "greedy-repaired"},
    }
    sample_entry = {
        "prompt_sha": "aa11",
        "prompt": "q-text",
        "samples": [
            {"text": "s0-ok"},
            {"text": "s1-truncated", "regen8192": {"text": "s1-repaired"}},
        ],
    }
    (gdir / "shard0_000.json").write_text(json.dumps([greedy_entry]), encoding="utf-8")
    (sdir / "shard0_000.json").write_text(json.dumps([sample_entry]), encoding="utf-8")
    items = {iid: (q, a) for iid, q, a in j._iter_rollout_items(tmp_path, "armA")}
    assert items["aa11.greedy"] == ("q-text", "greedy-repaired")
    assert items["aa11.s00"] == ("q-text", "s0-ok")
    assert items["aa11.s01"] == ("q-text", "s1-repaired")


def test_labeling_pilot_arms_fail_loud_on_missing_source(tmp_path: Path) -> None:
    """An armB row without its persisted per-row ``source`` is a KeyError —
    never a silent default into one pilot arm (the gen phase persists it)."""
    bdir = tmp_path / "armB" / "greedy"
    bdir.mkdir(parents=True)
    (bdir / "shard0_000.json").write_text(
        json.dumps([{"prompt_sha": "bb22", "prompt": "q", "greedy": {"text": "a"}}]),
        encoding="utf-8",
    )
    with pytest.raises(KeyError):
        j._labeling_pilot_arms(tmp_path)


def test_labeling_pilot_arms_split_by_true_source(tmp_path: Path) -> None:
    bdir = tmp_path / "armB" / "greedy"
    bdir.mkdir(parents=True)
    rows = [
        {"prompt_sha": "b1", "prompt": "q1", "source": "or-bench-hard-1k", "greedy": {"text": "a"}},
        {
            "prompt_sha": "b2",
            "prompt": "q2",
            "source": "phtest-controversial",
            "greedy": {"text": "a"},
        },
    ]
    (bdir / "shard0_000.json").write_text(json.dumps(rows), encoding="utf-8")
    arms = j._labeling_pilot_arms(tmp_path)
    assert [iid for iid, _, _ in arms["armB_orbench"]] == ["b1.greedy"]
    assert [iid for iid, _, _ in arms["armB_phtest"]] == ["b2.greedy"]


def _fake_judge_result(scores: dict[str, float | None]) -> SimpleNamespace:
    """A JudgeResult-shaped value object (only the attrs the reduce reads)."""
    return SimpleNamespace(
        scores=scores,
        per_item_scores={k: ([v] if v is not None else []) for k, v in scores.items()},
        n_dropped_draws=0,
        n_transport_lost_draws=0,
        n_refusal_draws=0,
        n_truncation_dropped_draws=0,
        n_api_refusal_draws=0,
        stop_reason_tally={"end_turn": len(scores)},
    )


def test_rejudge_executes_dual_scored_overlap_and_sync_rescue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C1 EXECUTING pin: run_rejudge_refusals really (1) dual-scores a seeded
    overlap of batch-SUCCEEDED rows per corpus on the SYNC path under the
    corpus's OWN rubric and reports the batch-vs-sync offset, and (2) sync
    re-issues the api-refusal rows with label-first rescued labels. The judge
    boundary is an autospec'd fake (signature-conformant by construction)."""
    assert j.OVERLAP_N == 200
    out_root = tmp_path / "judge"
    labeling = out_root / "labeling"
    labeling.mkdir(parents=True)
    raw_root = tmp_path / "raw"
    labels: dict[str, dict] = {}
    for corpus, pre in (("armA", "a"), ("armB", "b")):
        gdir = raw_root / corpus / "greedy"
        gdir.mkdir(parents=True)
        rows = [
            {
                "prompt_sha": f"{pre}{i}",
                "prompt": f"q{i}",
                "greedy": {"text": f"ans{i}"},
                "source": "or-bench-hard-1k",
            }
            for i in range(4)
        ]
        (gdir / "shard0_000.json").write_text(json.dumps(rows), encoding="utf-8")
        engage_tok = "COMPLY" if corpus == "armA" else "ANSWER"
        all_scores = {
            f"{pre}{i}.greedy__00000__00": {"reasoning": "r", "label": engage_tok, "score": 100}
            for i in range(3)
        }
        # rule-28 api-refusal row: empty content, stop_reason == "refusal"
        all_scores[f"{pre}3.greedy__00000__00"] = {"error": True, "stop_reason": "refusal"}
        (labeling / f"save_raw_{corpus}.json").write_text(
            json.dumps({"all_scores": all_scores}), encoding="utf-8"
        )
        for i in range(3):
            labels[f"{pre}{i}.greedy"] = {"score": 100.0, "label": "engage"}
        labels[f"{pre}3.greedy"] = {"score": None, "label": "UNCLEAR"}
    (labeling / "labels.json").write_text(json.dumps({"labels": labels}), encoding="utf-8")

    def _fake_stage(args):  # boundary fake: HF staging -> local fixture tree
        return raw_root

    monkeypatch.setattr(j, "_stage_raw_completions", _fake_stage)

    calls: list[dict] = []

    def _fake_judge(items, rubric, **kwargs):
        scores: dict[str, float] = {}
        all_scores: dict[str, dict] = {}
        for iid, _q, _a in items:
            scores[iid] = 0.0  # sync judge reads REFUSE everywhere -> offset 100
            all_scores[f"{iid}__00000__00"] = {"reasoning": "r", "label": "REFUSE", "score": 0}
        sr = kwargs.get("save_raw")
        if sr:
            Path(sr).parent.mkdir(parents=True, exist_ok=True)
            Path(sr).write_text(json.dumps({"all_scores": all_scores}), encoding="utf-8")
        calls.append({"items": [i[0] for i in items], "rubric": rubric, "kwargs": kwargs})
        return _fake_judge_result(scores)

    monkeypatch.setattr(j, "judge_graded", create_autospec(j.judge_graded, side_effect=_fake_judge))
    args = j.build_argparser().parse_args(["--rejudge-refusals", "--out-root", str(out_root)])
    assert j.run_rejudge_refusals(args) == 0

    payload = json.loads((out_root / "rejudge" / "rejudge.json").read_text(encoding="utf-8"))
    ov = payload["batch_vs_sync_overlap"]
    assert ov["n_overlap"] == 6 and ov["per_corpus"]["armA"]["n_overlap"] == 3
    assert ov["offset_batch_minus_sync"] == pytest.approx(100.0)
    assert ov["batch_mean"] == pytest.approx(100.0) and ov["sync_mean"] == pytest.approx(0.0)
    # sync rescue of the api-refusal rows, label-first
    assert payload["n_reissued"] == 2
    assert payload["rescued_labels"]["a3.greedy"] == "refuse"
    assert payload["rescued_scores"]["b3.greedy"] == 0.0
    # every judge call: SYNC path + the corpus's OWN rubric
    assert calls and all(c["kwargs"].get("force_sync") is True for c in calls)
    for c in calls:
        want = (
            j.LABELING_RUBRIC_ARMA
            if all(i.startswith("a") for i in c["items"])
            else j.LABELING_RUBRIC_ARMB
        )
        assert c["rubric"] == want, c["items"]
        assert c["kwargs"].get("max_tokens") == j.LABELING_MAX_TOKENS


def test_predictor_pilot_dispatches_real_per_fold_demos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C2 EXECUTING pin: run_predictor_pilot really dispatches per-row demo
    blocks equal to _build_few_shot_demos(train_rows, k, GLOBAL_SEED+fold)[0]
    — the SAME blocks the production wave uses; no placeholder block."""
    out_root = tmp_path / "judge"
    splits = tmp_path / "splits"
    splits.mkdir(parents=True)
    eval_rows = [
        {"row_id": f"r{f}{i}", "prompt": f"eval-q{f}{i}", "arm": "armA", "fold": f}
        for f in (0, 1)
        for i in range(2)
    ]
    (splits / "balanced_eval_rows.json").write_text(json.dumps(eval_rows), encoding="utf-8")
    train_by_fold: dict[int, list[dict]] = {}
    for f in (0, 1):
        rows = [
            {
                "row_id": f"t{f}{i}",
                "prompt": f"train-q{f}{i}",
                "label": "engage" if i % 2 == 0 else "refuse",
                "group_id": f"g{f}{i}",
            }
            for i in range(6)
        ]
        train_by_fold[f] = rows
        (splits / f"train_rows_armA_fold{f}.json").write_text(json.dumps(rows), encoding="utf-8")

    captured: dict = {}

    def _fake_gate(arms, rubric, **kwargs):
        captured["arms"] = arms
        captured["rubric"] = rubric
        rp = kwargs.get("report_path")
        if rp:
            Path(rp).parent.mkdir(parents=True, exist_ok=True)
            Path(rp).write_text("{}", encoding="utf-8")
        return SimpleNamespace(passed=True)

    monkeypatch.setattr(
        j, "judge_pilot_gate", create_autospec(j.judge_pilot_gate, side_effect=_fake_gate)
    )
    args = j.build_argparser().parse_args(
        ["--wave", "predictor", "--pilot", "--out-root", str(out_root)]
    )
    assert j.run_predictor_pilot(args) == 0
    assert captured["rubric"] == j.PREDICTOR_RUBRIC
    demo_by_row = {iid: demo for iid, _prompt, demo in captured["arms"]["armA"]}
    for f in (0, 1):
        # < FEWSHOT_MIN_TRAIN train rows -> degraded k, same as production
        expected, _ = j._build_few_shot_demos(
            train_by_fold[f], j.FEWSHOT_K_DEGRADED, j.GLOBAL_SEED + f
        )
        assert "Request:" in expected and "(example block)" not in expected
        for i in range(2):
            assert demo_by_row[f"r{f}{i}"] == expected, (f, i)
