"""#2202 labels-driver pins: rubric parse-contract round-trip (llm-judging rule
27), arm-symmetric user template, custom-id validity, mode-name sanitization,
matched-control equalize-down, stratified pick, pilot gate, tally split, and
the fail-loud text-cache loader. All synthetic; no network, no API calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue2202_labels as LB  # noqa: E402

from explore_persona_space.eval.judge_dispatch import validate_batch_custom_ids  # noqa: E402
from explore_persona_space.eval.utils import parse_judge_json  # noqa: E402

MODES = [
    {
        "name": "near_duplicate_answers",
        "description": "two contexts call for near-identical replies",
        "decision_rule": "yes iff a generic reply (greeting/refusal/boilerplate) fully answers it",
    },
    {
        "name": "non_english_exchange",
        "description": "",
        "decision_rule": "yes iff the dominant language is not English",
    },
]


def test_rubric_roundtrip_realistic_reply():
    """Rule-27: a REALISTIC reason-then-score reply parses through the harness's
    OWN parse path (parse_judge_json) into a valid mode label."""
    system = LB.rubric_system(MODES)
    for m in MODES:
        assert m["name"] in system and m["decision_rule"] in system
    reply = (
        "The exchange is a short greeting; a boilerplate reply would fully answer it, and the "
        'language is English.\n{"reasoning": "Generic greeting.", '
        '"near_duplicate_answers": "yes", "non_english_exchange": "no"}'
    )
    lab = LB.validate_mode_label(parse_judge_json(reply), MODES)
    assert lab == {"near_duplicate_answers": "yes", "non_english_exchange": "no"}
    fenced = (
        "Reasoning first.\n```json\n"
        + json.dumps(
            {"reasoning": "r", "near_duplicate_answers": "no", "non_english_exchange": "yes"}
        )
        + "\n```"
    )
    lab2 = LB.validate_mode_label(parse_judge_json(fenced), MODES)
    assert lab2 == {"near_duplicate_answers": "no", "non_english_exchange": "yes"}


def test_rubric_roundtrip_malformed_drops():
    """Drop-never-coerce: missing field / out-of-vocab value / non-dict -> None."""
    missing = parse_judge_json('{"reasoning": "r", "near_duplicate_answers": "yes"}')
    assert LB.validate_mode_label(missing, MODES) is None
    bad_value = parse_judge_json(
        '{"reasoning": "r", "near_duplicate_answers": "maybe", "non_english_exchange": "no"}'
    )
    assert LB.validate_mode_label(bad_value, MODES) is None
    assert LB.validate_mode_label(None, MODES) is None
    assert LB.validate_mode_label("95", MODES) is None


def test_rubric_user_msg_caps_and_no_placeholders():
    row = {
        "corpus": "wildchat",
        "history_tail": "h" * 5000,
        "last_user": "u" * 5000,
        "response": "r" * 5000,
    }
    msg = LB.rubric_user_msg(row)
    assert "{question}" not in msg and "{answer}" not in msg  # no unfilled slots
    assert msg.count("h") <= LB.CAP_HISTORY + 50
    assert msg.count("u") <= LB.CAP_LAST_USER + 50
    assert msg.count("r" * 100) <= LB.CAP_RESPONSE // 100 + 1
    # arm symmetry: the instrument never embeds confuser text
    assert "confuser" not in msg.lower()


def test_custom_ids_batch_valid():
    ids = ["f123", "c123", "s123", "rt_f123", "rt_s99999"]
    validate_batch_custom_ids(ids)  # raises on violation


def test_sanitize_mode_name():
    assert LB.sanitize_mode_name("Near-Duplicate Answers!") == "near_duplicate_answers"
    assert LB.sanitize_mode_name("reasoning") == "reasoning_"
    assert len(LB.sanitize_mode_name("x" * 100)) <= 41
    assert LB.sanitize_mode_name("") == "mode"


def test_parse_modes_from_fable_reply():
    reply = json.dumps(
        {
            "modes": [
                {"name": "Refusal Pairs", "description": "d", "decision_rule": "yes iff refusal"},
                {"bad": "row"},
            ]
        }
    )
    modes = LB.parse_modes(reply)
    assert len(modes) == 1 and modes[0]["name"] == "refusal_pairs"
    # fable-digest-rerun contract: a schema parse FAILURE returns None (loud,
    # hard-errors at the caller) — [] is reserved for a schema-valid reply
    # whose modes list is genuinely empty (tests/test_issue2202_fable_failfast.py).
    assert LB.parse_modes("no json here") is None
    assert LB.parse_modes('{"modes": []}') == []


def _rows(cells: dict[tuple, tuple[int, int]]) -> tuple[list[dict], dict, dict]:
    """Synthetic percontext rows + ci_fields + labels for given per-cell
    (n_fail, n_nonfail) counts."""
    rows, fields, labels = [], {}, {}
    ci = 0
    for (band, corpus, lang), (n_f, n_n) in cells.items():
        for k in range(n_f + n_n):
            rows.append(
                {
                    "ci": str(ci),
                    "fail_raw_euclidean": "1" if k < n_f else "0",
                    "in_sample500": "0",
                }
            )
            fields[str(ci)] = {"depth": 2, "depth_band": band, "corpus": corpus}
            labels[str(ci)] = {"language": lang}
            ci += 1
    return rows, fields, labels


def test_build_population_equalize_down():
    rows, fields, labels = _rows(
        {
            ("2-2", "wildchat", "en"): (5, 20),  # controls plentiful
            ("3-4", "lmsys", "en"): (6, 2),  # controls SCARCE -> equalize down
        }
    )
    pop = LB.build_population(rows, fields, labels, seed=2202)
    assert len(pop["fail_cis"]) == 11
    assert set(pop["control_cis"]).isdisjoint(pop["fail_cis"])
    scarce = pop["per_cell"]["3-4 | lmsys | en"]
    assert scarce["n_control"] == 2  # capped at available non-failures
    assert scarce["n_fail_equalized"] == 2  # failure side equalized down to match
    rich = pop["per_cell"]["2-2 | wildchat | en"]
    assert rich["n_control"] == 5 and rich["n_fail_equalized"] == 5
    assert set(pop["fail_eq_cis"]) <= set(pop["fail_cis"])
    # deterministic under the pinned seed
    pop2 = LB.build_population(rows, fields, labels, seed=2202)
    assert pop2["control_cis"] == pop["control_cis"]


def test_stratified_pick_allocation():
    cis = list(range(100))
    cells = {c: ("a" if c < 70 else "b",) for c in cis}
    picked = LB.stratified_pick(cis, lambda c: cells[c], 10, seed=1)
    assert len(picked) == 10
    n_a = sum(1 for c in picked if c < 70)
    assert n_a == 7  # largest-remainder proportional allocation
    assert picked == LB.stratified_pick(cis, lambda c: cells[c], 10, seed=1)


def test_pilot_gate_verdicts():
    clean = {
        "stop_reason_tally": {"end_turn": 100},
        "drops": {"fail": {"content": 0, "transport_loss": 1, "error_other": 0, "n": 60}},
    }
    assert LB.pilot_gate(clean)["verdict"] == "PASS"
    trunc = {
        "stop_reason_tally": {"end_turn": 99, "max_tokens": 1},
        "drops": {"fail": {"content": 0, "transport_loss": 0, "error_other": 0, "n": 60}},
    }
    assert LB.pilot_gate(trunc)["verdict"] == "FAIL"  # zero-truncation gate (rule 26)
    drops = {
        "stop_reason_tally": {"end_turn": 100},
        "drops": {"fail": {"content": 3, "transport_loss": 0, "error_other": 0, "n": 60}},
    }
    assert LB.pilot_gate(drops)["verdict"] == "FAIL"  # 5% parse-fail >= 2%


def test_tally_results_split():
    arm_of = {"f1": "fail", "c2": "control"}
    results = {
        "f1": {
            "reasoning": "r",
            "near_duplicate_answers": "yes",
            "non_english_exchange": "no",
            "stop_reason": "end_turn",
            "_raw_text": "...",
        },
        "c2": {"error": True, "transport": True, "reason": "timeout"},
        "rt_f1": {"error": True, "reason": "parse_error: garbled"},
    }
    t = LB.tally_results(results, MODES, arm_of)
    assert "f1" in t["labels"]
    assert t["drops"]["control"]["transport_loss"] == 1
    assert t["drops"]["fail"]["error_other"] == 1  # the rt_ parse_error row (fail arm)
    assert t["stop_reason_tally"] == {"end_turn": 1}


def test_load_texts_fail_loud(tmp_path):
    cache = tmp_path / "judge_texts.jsonl"
    cache.write_text(
        json.dumps({"ci": 1, "last_user": "u", "history_tail": "", "response": "r"}) + "\n",
        encoding="utf-8",
    )
    got = LB.load_texts(cache, {1})
    assert got[1]["last_user"] == "u"
    try:
        LB.load_texts(cache, {1, 2})
        raise AssertionError("expected RuntimeError for the missing ci")
    except RuntimeError as exc:
        assert "issue1482_collect_holdout_texts" in str(exc)


def test_cap_text_discloses_truncation():
    assert LB.cap_text("x" * 10, 20) == "x" * 10
    capped = LB.cap_text("x" * 30, 20)
    assert capped.startswith("x" * 20) and capped.endswith("…[truncated]")


def test_kappa_helper_on_synthetic_agreement():
    import issue1482_analysis as a82

    kap = a82._cohens_kappa(["yes", "no", "yes", "no"], ["yes", "no", "yes", "no"])
    assert np.isclose(kap, 1.0)
    kap2 = a82._cohens_kappa(["yes", "no", "yes", "no"], ["no", "yes", "no", "yes"])
    assert kap2 < LB.KAPPA_DEMOTE  # would be demoted to report-only
