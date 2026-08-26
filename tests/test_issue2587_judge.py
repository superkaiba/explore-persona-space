"""Unit 5a pins for ``scripts/issue2587_judge.py`` — CPU-only, no network.

Every production body executes for real; the ONLY faked boundary is
``eval.graded_judge.judge_graded`` (the Batch-API client), faked via
``unittest.mock.create_autospec`` so the fake is signature-conformant by
construction (`.claude/rules/code-style.md` § One production-body test per
seam-stubbed function). Bank access goes through the sha-asserted pinned
bank2564 module (local ``git show`` — no network; same convention as
``tests/test_issue2587_bank.py``).

Pinned here:
- rubric INSTRUMENT IDENTITY: sha256 of the ported compliance rubric (byte
  parity with ``issue2564_judge.py`` @ the frozen parent pin) and of the NEW
  answer_language pilot rubric (WARN-2) — drift fails CI, protecting the
  #2564 7B-side verbatim-consumption contract;
- plan §4.4 call arithmetic against the REALIZED bank (1,392 + 72 = 1,464);
- fire semantics: fixed denominator, integer 70% threshold, MANDATORY
  ``undetermined`` on any incomplete check (never shrinks the denominator);
- alias grammar (charset / 53-char budget / no ``__`` / collision-free /
  cross-family disjoint);
- the MECHANICAL "7B side pending #2564" pilot label;
- drop-class separation (content vs transport vs refusal vs truncation vs
  api-refusal) landing in SEPARATE meta fields;
- the smoke-path ``main()`` end-to-end with per-wave rubric-keyed cache dirs.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import issue2587_judge as J  # noqa: E402

from explore_persona_space.eval import graded_judge as GJ  # noqa: E402

# ── pinned bank (module-scope; local `git show`, no network) ──────────────

BK = J.B25._bk()
VALUES = BK.load_values()

# ── instrument identity pins ──────────────────────────────────────────────

# sha256 of the compliance rubric — byte parity with issue2564_judge.py @
# 8265bcd75f781d8e879e924de60063e536e58dcf verified at port time (AST-extracted
# parent EVAL_PROMPT == J.EVAL_PROMPT). Drift breaks the q25-vs-q35 contrast.
COMPLIANCE_RUBRIC_SHA256 = "f6c48e42d793754036a7f4cb4d64b204bea8c583ca27af3af605e46e212bcd77"
# sha256 of the NEW answer_language pilot rubric (WARN-2). The #2564 7B-side
# pilot judging consumes this text VERBATIM — any edit must be coordinated.
ANSWER_LANGUAGE_RUBRIC_SHA256 = "d741a82d7e7189cb334d1ac0aad1cfce7a5a70a8a186df0541d6478402428177"


def test_compliance_rubric_sha_pinned():
    assert hashlib.sha256(J.EVAL_PROMPT.encode()).hexdigest() == COMPLIANCE_RUBRIC_SHA256
    assert "{question}" in J.EVAL_PROMPT and "{answer}" in J.EVAL_PROMPT


def test_answer_language_rubric_sha_pinned():
    text = J.ANSWER_LANGUAGE_EVAL_PROMPT
    assert hashlib.sha256(text.encode()).hexdigest() == ANSWER_LANGUAGE_RUBRIC_SHA256
    assert "{question}" in text and "{answer}" in text
    assert "ENTIRELY in the language" in text


def test_judge_model_and_max_tokens_pins():
    assert J.JUDGE_MODEL == "claude-sonnet-4-5-20250929"  # never -20251001 (Haiku)
    assert J.JUDGE_MAX_TOKENS == 1024


def test_parent_pin_matches_bank():
    assert J.PARENT_PIN == "8265bcd75f781d8e879e924de60063e536e58dcf"


# ── call arithmetic against the REALIZED bank (plan §4.4) ─────────────────


def test_call_arithmetic_realized_bank():
    axes = J.parent_judged_axes(BK)
    assert set(axes) == set(BK.INSTRUCTION_AXES) - set(J.PROGRAMMATIC_AXES)
    j_specs = J.judged_specs(BK, VALUES, BK.CARRIER_IDS, axes)
    l_specs = J.lang_specs(BK, BK.CARRIER_IDS)
    assert len(j_specs) == 1392  # (29 + 29 paraphrase slots) x 12 carriers x 2 draws
    assert len(l_specs) == 72  # 3 languages x 12 carriers x 2 draws
    report = J.verify_call_arithmetic(len(j_specs), len(l_specs))
    assert report["verified"] is True
    assert report["realized"]["total_calls"] == 1464
    p_specs = J.programmatic_specs(BK, VALUES, BK.CARRIER_IDS)
    assert len(p_specs) == 2400  # 10 values x 2 kinds x 12 carriers x 10 draws


def test_verify_call_arithmetic_raises_on_mismatch():
    with pytest.raises(RuntimeError, match="call-arithmetic mismatch"):
        J.verify_call_arithmetic(1392, 71)
    with pytest.raises(RuntimeError, match="call-arithmetic mismatch"):
        J.verify_call_arithmetic(1391, 72)


# ── alias grammar (Batch custom_id budget) ────────────────────────────────


def test_alias_grammar_budget_and_disjointness():
    axes = J.parent_judged_axes(BK)
    j_specs = J.judged_specs(BK, VALUES, BK.CARRIER_IDS, axes)
    l_specs = J.lang_specs(BK, BK.CARRIER_IDS)
    all_aliases = [s["alias"] for s in j_specs] + [s["alias"] for s in l_specs]
    assert len(set(all_aliases)) == len(all_aliases)  # bijective across BOTH families
    for a in all_aliases:
        assert J._ALIAS_RE.match(a), a
        assert "__" not in a, a
        assert len(a) <= 53, a  # 64-char custom_id cap minus batch_judge's 11-char suffix
    overlap = {s["alias"] for s in j_specs} & {s["alias"] for s in l_specs}
    assert not overlap


def test_validated_specs_raises_on_collision_and_bad_alias():
    dup = [{"alias": "a--b--c-d0"}, {"alias": "a--b--c-d0"}]
    with pytest.raises(ValueError, match="collision"):
        J._validated_specs(dup, "test")
    with pytest.raises(ValueError, match="illegal batch alias"):
        J._validated_specs([{"alias": "bad__alias"}], "test")
    with pytest.raises(ValueError, match="illegal batch alias"):
        J._validated_specs([{"alias": "x" * 54}], "test")


# ── spec construction semantics ───────────────────────────────────────────


def test_lang_specs_shape_and_instruction_identity():
    specs = J.lang_specs(BK, ("c01", "c02"), draws=(0, 1))
    assert len(specs) == len(J.B25.LANG_VALUES) * 2 * 2
    for s in specs:
        assert s["axis"] == "answer_language"
        assert s["kind"] == "orig"  # no paraphrase family on the pilot axis
        assert s["instruction"] == J.B25.LANG_VALUES[s["value_id"]]
        assert s["context_id"] == BK.context_id("answer_language", s["value_id"], s["carrier"])
    # the un-instructed "bare" contexts are never judged
    assert not any(s["value_id"] == "bare" for s in specs)


def test_judged_specs_para_instruction_differs_from_orig():
    axis = "register"
    vid = BK.value_ids(VALUES, axis)[0]
    slots = J.judged_value_slots(BK, VALUES, (axis,))
    orig = next(s for s in slots if s["value_id"] == vid)
    para = next(s for s in slots if s["value_id"] == f"{vid}p")
    assert orig["instruction"] == BK.system_string(VALUES, axis, vid)
    assert para["instruction"] == BK.paraphrase_string(VALUES, axis, vid)
    assert orig["instruction"] != para["instruction"]


def test_programmatic_specs_para_keeps_base_word():
    specs = J.programmatic_specs(BK, VALUES, ("c01",), draws=(0,))
    by_kind: dict[tuple[str, str, str], dict] = {}
    for s in specs:
        by_kind[(s["axis"], s["value_id"], s["kind"])] = s
    for axis in J.PROGRAMMATIC_AXES:
        for vid in BK.value_ids(VALUES, axis):
            word = VALUES["axes"][axis]["values"][vid]
            assert by_kind[(axis, vid, "orig")]["word"] == word
            # paraphrase keeps the payload word
            assert by_kind[(axis, f"{vid}p", "para")]["word"] == word


# ── fire semantics (parent verbatim) ──────────────────────────────────────


def test_fire_verdict_threshold_integer_arithmetic():
    # judged denominator 24: fired iff n_comply*100 >= 70*24 = 1680 → ≥17
    assert J.fire_verdict(17, 0, 24) == "fired"
    assert J.fire_verdict(16, 0, 24) == "not_fired"
    # programmatic denominator 120: ≥84
    assert J.fire_verdict(84, 0, 120) == "fired"
    assert J.fire_verdict(83, 0, 120) == "not_fired"


def test_fire_verdict_undetermined_never_shrinks_denominator():
    # 20/24 complying is above 70% of the FIXED denominator, but ONE incomplete
    # check forces "undetermined" — the denominator is never shrunk to 23.
    assert J.fire_verdict(20, 4, 24) == "undetermined"
    assert J.fire_verdict(23, 1, 24) == "undetermined"
    # and at 100% sensitivity the same holds
    assert J.fire_verdict(20, 4, 24, threshold_pct=50) == "undetermined"


def test_fire_verdict_raises_on_bad_counts():
    with pytest.raises(ValueError):
        J.fire_verdict(-1, 0, 24)
    with pytest.raises(ValueError):
        J.fire_verdict(0, -1, 24)
    with pytest.raises(ValueError):
        J.fire_verdict(20, 5, 24)  # comply + incomplete > denom
    with pytest.raises(ValueError):
        J.fire_verdict(0, 0, 0)


def test_value_row_reconciliation_assert():
    with pytest.raises(AssertionError):
        J._value_row("register", "r1", "orig", "judged", 10, 10, 10, 24)
    row = J._value_row("register", "r1", "orig", "judged", 17, 6, 1, 24)
    assert row["verdict"] == "undetermined"
    assert row["denom"] == 24
    assert row["sensitivity"]["50"] == "undetermined"


def test_axis_floor():
    assert J.axis_floor(3) == math.ceil(0.6 * 3) == 2
    assert J.axis_floor(5) == 3
    assert J.axis_floor(2) == 2


def test_check_contains_word_boundaries():
    assert J.check_contains_word("Surely, that is right.", "surely")
    assert not J.check_contains_word("He walked leisurely.", "surely")
    assert J.check_contains_word("Marcus's answer", "Marcus")
    assert not J.check_contains_word("Marcusx", "Marcus")


# ── fire tables ───────────────────────────────────────────────────────────


def _mini_specs(axis: str, vid: str, kind: str, carriers, draws):
    return [
        {
            "axis": axis,
            "value_id": vid,
            "kind": kind,
            "carrier": c,
            "draw": d,
            "context_id": f"{axis}::{vid}::{c}",
            "alias": J._alias(axis, vid, c, d),
            "word": "surely",
            "instruction": "x",
        }
        for c in carriers
        for d in draws
    ]


def test_judged_fire_table_missing_alias_is_incomplete():
    carriers, draws = ("c01", "c02"), (0, 1)
    specs = _mini_specs("register", "r1", "orig", carriers, draws)
    scores = {specs[0]["alias"]: 100.0, specs[1]["alias"]: 100.0, specs[2]["alias"]: 20.0}
    # specs[3]'s alias absent from scores → incomplete
    rows = J.judged_fire_table(specs, scores, carriers, draws)
    assert len(rows) == 1
    r = rows[0]
    assert (r["n_comply"], r["n_noncomply"], r["n_incomplete"]) == (2, 1, 1)
    assert r["denom"] == 4
    assert r["verdict"] == "undetermined"
    assert r["instrument"] == "judged"


def test_judged_fire_table_language_instrument_tag():
    carriers, draws = ("c01",), (0, 1)
    specs = _mini_specs("answer_language", "english", "orig", carriers, draws)
    scores = {s["alias"]: 100.0 for s in specs}
    rows = J.judged_fire_table(specs, scores, carriers, draws, instrument="judged_language")
    assert rows[0]["instrument"] == "judged_language"
    assert rows[0]["verdict"] == "fired"


def test_programmatic_fire_table_missing_text_is_incomplete():
    carriers, draws = ("c01", "c02"), (0, 1)
    specs = _mini_specs("lexical_marker", "lm1", "orig", carriers, draws)
    texts = {
        (specs[0]["context_id"], specs[0]["draw"]): "Surely yes.",
        (specs[1]["context_id"], specs[1]["draw"]): "no trace here",
        (specs[2]["context_id"], specs[2]["draw"]): "surely again",
        # specs[3] key absent → incomplete
    }
    rows = J.programmatic_fire_table(specs, texts, carriers, draws)
    r = rows[0]
    assert (r["n_comply"], r["n_noncomply"], r["n_incomplete"]) == (2, 1, 1)
    assert r["instrument"] == "programmatic"
    assert r["verdict"] == "undetermined"


# ── axis summary ──────────────────────────────────────────────────────────


def test_axis_summary_no_para_reports_none_not_zero():
    rows = [
        J._value_row("answer_language", lang, "orig", "judged_language", c, n, 0, 24)
        for lang, c, n in [("english", 24, 0), ("chinese", 20, 4), ("spanish", 10, 14)]
    ]
    summary = J.axis_summary(rows, "answer_language", 3, has_para=False)
    assert summary["n_fired_base"] == 2  # 24/24 and 20/24 fire at 70%; 10/24 does not
    assert summary["floor"] == 2 and summary["floor_met"] is True
    assert summary["n_fired_para"] is None  # never a 0 that reads as "0 paraphrases fired"


def test_axis_summary_no_para_rejects_unexpected_para_rows():
    rows = [
        J._value_row("answer_language", "english", "orig", "judged_language", 24, 0, 0, 24),
        J._value_row("answer_language", "english", "para", "judged_language", 24, 0, 0, 24),
    ]
    with pytest.raises(AssertionError):
        J.axis_summary(rows, "answer_language", 1, has_para=False)


def test_axis_summary_with_para_counts_int():
    rows = [
        J._value_row("register", "r1", "orig", "judged", 24, 0, 0, 24),
        J._value_row("register", "r1", "para", "judged", 24, 0, 0, 24),
        J._value_row("register", "r2", "orig", "judged", 0, 24, 0, 24),
        J._value_row("register", "r2", "para", "judged", 0, 24, 0, 24),
    ]
    summary = J.axis_summary(rows, "register", 2)
    assert summary["n_fired_base"] == 1
    assert summary["n_fired_para"] == 1
    assert summary["floor"] == 2 and summary["floor_met"] is False


# ── mechanical pilot labeling (WARN-2) ────────────────────────────────────


def test_annotate_pilot_rows_mechanical_label():
    rows = [
        {"axis": "answer_language"},
        {"axis": "query_content_oneword"},
        {"axis": "register"},
    ]
    J.annotate_pilot_rows(rows)
    assert rows[0]["pilot_axis"] is True
    assert rows[0]["cross_model_status"] == "7B side pending #2564"
    assert isinstance(rows[0]["cross_model_status"], str)  # a string field, never numeric
    assert rows[1]["pilot_axis"] is True
    assert rows[1]["cross_model_status"] == "7B side pending #2564"
    assert rows[2]["pilot_axis"] is False
    assert "cross_model_status" not in rows[2]


def test_annotate_pilot_rows_fails_loud_on_missing_axis():
    with pytest.raises(KeyError):
        J.annotate_pilot_rows([{"verdict": "fired"}])


# ── per-axis drop report ──────────────────────────────────────────────────


def test_per_axis_drop_report():
    specs = _mini_specs("register", "r1", "orig", ("c01", "c02"), (0, 1))
    scores = {specs[0]["alias"]: 90.0, specs[1]["alias"]: None}
    rep = J.per_axis_drop_report(specs, scores)
    assert rep["register"] == {"n_specs": 4, "n_scored": 1, "n_incomplete": 3}


# ── anchors ingestion ─────────────────────────────────────────────────────


def test_load_anchor_texts_raises_on_empty_cell(tmp_path):
    p = tmp_path / "anchors_register.jsonl"
    p.write_text("")
    with pytest.raises(RuntimeError, match="EMPTY"):
        J.load_anchor_texts({"register": p})


def test_read_jsonl_survives_u2028_in_text(tmp_path):
    p = tmp_path / "x.jsonl"
    # U+2028 built via chr() so the test SOURCE stays 7-bit (Edit-tool un-escape trap,
    # .claude/rules/gotchas.md); splitlines() would shred this row - split("\n") must not.
    row = {"context_id": "a", "draw": 0, "text": "line one" + chr(0x2028) + "still same row"}
    p.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = J._read_jsonl(p)
    assert len(rows) == 1 and rows[0]["text"].startswith("line one")


# ── main() smoke path: production bodies, autospec at the Batch boundary ──


def _write_anchor_fixture(anchors_dir: Path, cells_specs: dict[str, list[dict]]) -> None:
    """One anchors_{cell}.jsonl per cell covering every (context_id, draw) in specs."""
    anchors_dir.mkdir(parents=True, exist_ok=True)
    for cell, specs in cells_specs.items():
        rows = []
        seen = set()
        for s in specs:
            key = (s["context_id"], s["draw"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                json.dumps(
                    {"context_id": s["context_id"], "draw": s["draw"], "text": "Surely fine."}
                )
            )
        (anchors_dir / f"anchors_{cell}.jsonl").write_text("\n".join(rows) + "\n")


def _fake_judge_result(scores: dict[str, float]) -> mock.NonCallableMock:
    """Signature-conformant JudgeResult fake (autospec of the REAL class)."""
    res = mock.create_autospec(GJ.JudgeResult, instance=True)
    res.scores = scores
    res.n_total_draws = len(scores)
    res.n_dropped_draws = 1  # content drops — distinct from transport
    res.n_transport_lost_draws = 2
    res.n_refusal_draws = 3
    res.n_truncation_dropped_draws = 4
    res.n_api_refusal_draws = 5
    res.stop_reason_tally = {"end_turn": len(scores)}
    res.frac_items_complete = 1.0
    return res


def test_main_smoke_path_end_to_end(tmp_path, monkeypatch):
    """Both rubric families reach the (faked) Batch client; the sentinel carries
    instrument identity, per-wave drop classes, pilot labels, and capped counts."""
    smoke_carriers = J.SMOKE_CARRIERS
    reg_specs = J.judged_specs(BK, VALUES, smoke_carriers, ("register",))
    lang_specs = J.lang_specs(BK, smoke_carriers)
    anchors_dir = tmp_path / "anchors"
    _write_anchor_fixture(anchors_dir, {"register": reg_specs, "answer_language": lang_specs})

    calls: list[dict] = []
    real = GJ.judge_graded

    def fake_judge(items, eval_prompt, **kw):
        calls.append({"items": list(items), "eval_prompt": eval_prompt, **kw})
        return _fake_judge_result({alias: 100.0 for alias, _q, _t in items})

    monkeypatch.setattr(GJ, "judge_graded", mock.create_autospec(real, side_effect=fake_judge))

    out = tmp_path / "manip.json"
    work = tmp_path / "work"
    argv = [
        "issue2587_judge.py",
        "--smoke",
        "--out",
        str(out),
        "--work-root",
        str(work),
        "--anchors-dir",
        str(anchors_dir),
        "--skip-upload",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as ei:
        J.main()
    assert ei.value.code == 0

    # two waves, rubric-keyed: distinct prompts AND distinct cache partitions
    assert len(calls) == 2
    prompts = {c["eval_prompt"] for c in calls}
    assert prompts == {J.EVAL_PROMPT, J.ANSWER_LANGUAGE_EVAL_PROMPT}
    cache_dirs = {str(c["cache_dir"]) for c in calls}
    assert len(cache_dirs) == 2
    for c in calls:
        assert c["threshold_base"] == 0  # forced #663-hardened Batch path
        assert c["judge_model"] == "claude-sonnet-4-5-20250929"
        assert c["max_tokens"] == 1024
        assert c["n_draws"] == 1  # rollout draw is encoded in the alias
        assert len(c["items"]) == J.SMOKE_JUDGE_ITEMS  # PER-WAVE cap: both families exercised

    doc = json.loads(out.read_text())
    meta = doc["meta"]
    ident = meta["instrument_identity"]
    assert ident["compliance_rubric_sha256"] == COMPLIANCE_RUBRIC_SHA256
    assert ident["answer_language_rubric_sha256"] == ANSWER_LANGUAGE_RUBRIC_SHA256
    assert ident["answer_language_rubric_text"] == J.ANSWER_LANGUAGE_EVAL_PROMPT

    # smoke slice → arithmetic recorded but NOT verified against 1,464 — and
    # the artifact ENUMERATES its active downgrades (smoke blind-spot registry)
    assert meta["call_arithmetic"]["verified"] is False
    bs = {e["site"]: e["kind"] for e in meta["smoke_blind_spots"]}
    assert bs == {
        "call_arithmetic_1464": "assert-skipped",
        "smoke_slice_narrowing": "param-narrowed",
    }
    assert meta["n_judged_specs"] == len(reg_specs)
    assert meta["n_lang_specs"] == len(lang_specs)
    assert meta["n_capped_out"]["compliance"] == len(reg_specs) - J.SMOKE_JUDGE_ITEMS
    assert meta["n_capped_out"]["answer_language"] == len(lang_specs) - J.SMOKE_JUDGE_ITEMS

    # drop classes land in SEPARATE fields, never conflated
    for wave in ("compliance", "answer_language"):
        stats = meta["judge_stats"][wave]
        assert stats["n_dropped_draws"] == 1
        assert stats["n_transport_lost_draws"] == 2
        assert stats["n_refusal_draws"] == 3
        assert stats["n_truncation_dropped_draws"] == 4
        assert stats["n_api_refusal_draws"] == 5
        assert stats["zero_max_tokens_stop"] is True
        assert stats["rubric_sha256"] in (
            COMPLIANCE_RUBRIC_SHA256,
            ANSWER_LANGUAGE_RUBRIC_SHA256,
        )
    assert meta["per_axis_drop_report"]["answer_language"]["answer_language"]["n_specs"] == len(
        lang_specs
    )

    # pilot labels: mechanical on value_rows AND axis_rows
    lang_rows = [r for r in doc["value_rows"] if r["axis"] == "answer_language"]
    assert lang_rows and all(
        r["pilot_axis"] is True and r["cross_model_status"] == "7B side pending #2564"
        for r in lang_rows
    )
    reg_rows = [r for r in doc["value_rows"] if r["axis"] == "register"]
    assert reg_rows and all(r["pilot_axis"] is False for r in reg_rows)

    axis_by_name = {r["axis"]: r for r in doc["axis_rows"]}
    ow = axis_by_name["query_content_oneword"]
    assert ow["verdict"] == "no_manipulation_check_query_class"
    assert ow["pilot_axis"] is True and ow["cross_model_status"] == "7B side pending #2564"
    lang_axis = axis_by_name["answer_language"]
    assert lang_axis["pilot_axis"] is True
    assert lang_axis["n_fired_para"] is None
    assert axis_by_name["register"]["pilot_axis"] is False
    # out-of-slice parent axes are explicit rows, never silently missing
    assert axis_by_name["persona"]["verdict"] == "not_in_slice"
    # programmatic axes were NOT in the smoke slice
    assert axis_by_name["lexical_marker"]["verdict"] == "not_in_slice"

    # per-check derived JSONL rides next to the raw judge outputs
    scores_jsonl = work / "raw" / "judge_scores.jsonl"
    lines = [json.loads(x) for x in scores_jsonl.read_text().split("\n") if x.strip()]
    assert len(lines) == len(reg_specs) + len(lang_specs)
    instruments = {x["instrument"] for x in lines}
    assert instruments == {"judged", "judged_language"}
    outcomes = {x["outcome"] for x in lines}
    assert outcomes <= {"comply", "noncomply", "incomplete"}


def test_main_smoke_refuses_committed_eval_results_out(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["issue2587_judge.py", "--smoke", "--out", "eval_results/issue_2587/x.json"]
    )
    with pytest.raises(SystemExit) as ei:
        J.main()
    assert "must not write" in str(ei.value)


def test_smoke_refuses_absolute_eval_results_out(monkeypatch):
    """r1 g6 M1: the committed-results guard is path-NORMALIZED — an ABSOLUTE
    --out under eval_results/ refuses exactly like the relative spelling."""
    absolute = str((Path.cwd() / "eval_results" / "issue_2587" / "x.json").resolve())
    monkeypatch.setattr(sys, "argv", ["issue2587_judge.py", "--smoke", "--out", absolute])
    with pytest.raises(SystemExit) as ei:
        J.main()
    assert "must not write" in str(ei.value)


def test_dry_run_refuses_explicit_eval_results_out(monkeypatch):
    """r1 g6 M1 (second half): an explicit eval_results/ --out under --dry-run
    would overwrite a committed sentinel with an all-incomplete table — refuse
    loud, never rebind silently."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["issue2587_judge.py", "--dry-run", "--out", "eval_results/issue_2587/x.json"],
    )
    with pytest.raises(SystemExit) as ei:
        J.main()
    assert "must not write" in str(ei.value)


def test_inside_eval_results_normalizes(tmp_path):
    assert J._inside_eval_results("eval_results/issue_2587/x.json")
    assert J._inside_eval_results(Path.cwd() / "eval_results" / "y.json")
    assert J._inside_eval_results("figures/../eval_results/y.json")
    assert not J._inside_eval_results(tmp_path / "manip.json")
    assert not J._inside_eval_results("/tmp/issue2587_judge_smoke/x.json")


def test_judge_smoke_blind_spot_registry_shape():
    """The registry is the enumerable source for the marker: exactly the
    arithmetic-gate skip (r1 g6 M2) + the slice/cap narrowing."""
    sites = {e.site: e.kind for e in J.SMOKE_BLIND_SPOTS}
    assert sites == {
        "call_arithmetic_1464": "assert-skipped",
        "smoke_slice_narrowing": "param-narrowed",
    }
    out = J.format_smoke_blind_spots(J.SMOKE_BLIND_SPOTS)
    assert all(set(d) == {"site", "kind", "production", "smoke", "why"} for d in out)
    arith = next(e for e in J.SMOKE_BLIND_SPOTS if e.site == "call_arithmetic_1464")
    assert "1,464" in arith.production  # names the gate the smoke skips, verbatim
