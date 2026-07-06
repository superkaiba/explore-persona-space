"""Offline tests for the #1090 persona-vectors datagen additions.

Covers: the datagen ``instruction_source="extraction_pairs"`` seam (positives
rotate the pair ``exhibit`` texts, negatives the ``not_exhibit`` texts; manifest
resume-key protection; the elicitation default stays byte-identical), the two
new behavior registrations (impolite paper-native trait; sycophancy_hardfact C4
control identical to sycophancy except banks), and the questiongen artifact
sync pins (registry descriptions == questiongen constants; inlined impolite
pairs == the committed generation artifact; committed bank shas == manifest).

Bank items are handled by reference only (counts / shas / membership) — never
printed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from explore_persona_space.artifacts import banks, datagen
from explore_persona_space.artifacts.behavior import BEHAVIORS
from explore_persona_space.artifacts.context import context_for_persona
from explore_persona_space.artifacts.datagen import NEGATIVE, POSITIVE, GenCandidate
from explore_persona_space.eval.graded_judge import JudgeResult

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS = REPO_ROOT / "scripts" / "issue1090_assets"

SRC = context_for_persona("villain")  # disjoint from the default panel


def _gen_all():
    def gen(requests):
        return [GenCandidate(r, f"resp::{r.request_id}") for r in requests]

    return gen


def _judge_by_arm(*, pos=80.0, neg=20.0):
    def judge(items, eval_prompt, *, n_draws, cache_dir, save_raw, judge_model, dry_run=False):
        scores = {rid: (pos if rid.startswith("pos-") else neg) for rid, _, _ in items}
        return JudgeResult(
            scores=scores,
            n_total_draws=len(items) * n_draws,
            n_dropped_draws=0,
            per_item_draw_counts={rid: n_draws for rid, _, _ in items},
            per_item_scores={rid: [scores[rid]] * n_draws for rid, _, _ in items},
        )

    return judge


def _questiongen_module():
    """Import scripts/issue1090_questiongen.py (scripts/ is a package)."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts import issue1090_questiongen as qg
    finally:
        sys.path.pop(0)
    return qg


# ── instruction_source seam ──────────────────────────────────────────────────


def test_extraction_pairs_source_rotates_pair_texts(tmp_path):
    beh = BEHAVIORS["sycophancy"]
    captured = []
    datagen.generate_training_data(
        beh,
        SRC,
        "default_v1",
        out_dir=tmp_path / "pairs",
        target_n=6,
        n_judge_draws=2,
        generate_fn=lambda reqs: captured.extend(reqs) or [GenCandidate(r, "resp") for r in reqs],
        judge_fn=_judge_by_arm(),
        instruction_style="plain",
        instruction_source="extraction_pairs",
    )
    assert captured
    exhibit = {p.exhibit for p in beh.extraction.prompt_pairs}
    not_exhibit = {p.not_exhibit for p in beh.extraction.prompt_pairs}
    elicit = set(beh.elicitation.exhibit_instructions) | set(
        beh.elicitation.not_exhibit_instructions
    )
    for req in captured:
        sys_text = req.gen_messages[0]["content"]
        pool = exhibit if req.arm == POSITIVE else not_exhibit
        assert any(instr in sys_text for instr in pool), req.request_id
        # The registered elicitation variants must NOT appear anywhere.
        assert not any(instr in sys_text for instr in elicit), req.request_id
    # Both arms carry instructions (pairs always have a not_exhibit side).
    assert {r.arm for r in captured} == {POSITIVE, NEGATIVE}


def test_extraction_pairs_source_enters_manifest(tmp_path):
    beh = BEHAVIORS["sycophancy"]
    datagen.generate_training_data(
        beh,
        SRC,
        "default_v1",
        out_dir=tmp_path / "m",
        target_n=6,
        n_judge_draws=2,
        generate_fn=_gen_all(),
        judge_fn=_judge_by_arm(),
        instruction_source="extraction_pairs",
    )
    manifest = json.loads((tmp_path / "m" / "gen_manifest.json").read_text())
    assert manifest["instruction_source"] == "extraction_pairs"
    assert manifest["exhibit_instructions"] == [p.exhibit for p in beh.extraction.prompt_pairs]
    assert manifest["not_exhibit_instructions"] == [
        p.not_exhibit for p in beh.extraction.prompt_pairs
    ]
    # A source flip on the same out_dir REFUSES (resume-key protection).
    with pytest.raises(datagen.DatagenCheckpointMismatchError):
        datagen.generate_training_data(
            beh,
            SRC,
            "default_v1",
            out_dir=tmp_path / "m",
            target_n=6,
            n_judge_draws=2,
            generate_fn=_gen_all(),
            judge_fn=_judge_by_arm(),
            instruction_source="elicitation",
        )


def test_elicitation_default_manifest_records_elicitation_lists(tmp_path):
    beh = BEHAVIORS["sycophancy"]
    datagen.generate_training_data(
        beh,
        SRC,
        "default_v1",
        out_dir=tmp_path / "d",
        target_n=6,
        n_judge_draws=2,
        generate_fn=_gen_all(),
        judge_fn=_judge_by_arm(),
    )
    manifest = json.loads((tmp_path / "d" / "gen_manifest.json").read_text())
    assert manifest["instruction_source"] == "elicitation"
    assert manifest["exhibit_instructions"] == list(beh.elicitation.exhibit_instructions)
    assert manifest["not_exhibit_instructions"] == list(beh.elicitation.not_exhibit_instructions)


def test_unknown_source_and_missing_pairs_raise(tmp_path):
    with pytest.raises(ValueError, match="instruction_source"):
        datagen.generate_training_data(
            BEHAVIORS["sycophancy"],
            SRC,
            "default_v1",
            out_dir=tmp_path / "bad",
            generate_fn=_gen_all(),
            judge_fn=_judge_by_arm(),
            instruction_source="nope",
        )
    # A no-ExtractionSpec behavior cannot use the pairs source.
    import dataclasses

    no_pairs = dataclasses.replace(BEHAVIORS["sycophancy"], extraction=None)
    with pytest.raises(ValueError, match="no ExtractionSpec"):
        datagen.generate_training_data(
            no_pairs,
            SRC,
            "default_v1",
            out_dir=tmp_path / "bad2",
            generate_fn=_gen_all(),
            judge_fn=_judge_by_arm(),
            instruction_source="extraction_pairs",
        )


# ── new registrations ────────────────────────────────────────────────────────


def test_impolite_registration_shape():
    b = BEHAVIORS["impolite"]
    assert not b.programmatic and not b.is_stub
    assert len(b.extraction.prompt_pairs) == 5
    assert b.extraction.question_set == ()  # datagen-only adoption (see _make docstring)
    assert len(b.train_question_bank) == 20 and len(b.eval_question_bank) == 20
    assert not set(b.train_question_bank) & set(b.eval_question_bank)


def test_sycophancy_hardfact_is_bank_only_delta():
    s, h = BEHAVIORS["sycophancy"], BEHAVIORS["sycophancy_hardfact"]
    assert h.description == s.description
    assert h.judge_rubric == s.judge_rubric
    assert h.elicitation == s.elicitation
    assert h.extraction.prompt_pairs == s.extraction.prompt_pairs
    assert h.dv == s.dv and h.method == s.method and h.threshold == s.threshold
    # ONLY the banks differ: hardfact keeps the curated wrong-fact slices.
    assert banks.SLICES[("sycophancy_hardfact", "train")] == ("sycophancy_claims", 0, 25)
    assert banks.SLICES[("sycophancy", "train")] == ("sycophancy_neutral_v2", 0, 20)
    assert not set(h.train_question_bank) & set(s.train_question_bank)


def test_repointed_banks_are_registered_and_disjoint():
    for name in (
        "sycophancy_neutral_v2",
        "impolite_neutral_v1",
        "broad_em_neutral_v1",
    ):
        data = banks.load_bank(name)
        assert len(data) == 40
    for beh in ("sycophancy", "broad_em", "impolite"):
        train = banks.bank_slice(beh, "train")
        ev = banks.bank_slice(beh, "eval")
        assert len(train) == 20 and len(ev) == 20
        assert not set(train) & set(ev)
    # Old banks stay registered (the C4 control + provenance, incl. the
    # skim-failed sycophancy v1 the round-2 regen superseded).
    assert "sycophancy_claims" in banks.QUERY_BANKS
    assert "sycophancy_neutral_v1" in banks.QUERY_BANKS
    assert "broad_em_train_v1" in banks.QUERY_BANKS
    assert "wang44" in banks.QUERY_BANKS


# ── questiongen artifact sync pins ───────────────────────────────────────────


def test_questiongen_descriptions_match_registry():
    qg = _questiongen_module()
    for trait in ("sycophancy", "impolite", "broad_em"):
        assert qg.TRAITS[trait].description == BEHAVIORS[trait].description, trait


def test_impolite_inlined_pairs_match_generation_artifact():
    artifact = json.loads((ASSETS / "questiongen_impolite.json").read_text())
    generated = [(p["pos"], p["neg"]) for p in artifact["instruction_pairs"]]
    registered = [(p.exhibit, p.not_exhibit) for p in BEHAVIORS["impolite"].extraction.prompt_pairs]
    assert registered == generated


def test_bank_shas_match_manifest():
    manifest = json.loads((ASSETS / "bank_manifest.json").read_text())
    for trait in ("sycophancy", "impolite", "broad_em"):
        row = manifest["banks"][trait]
        bank_name = row["file"].removesuffix(".json")
        assert row["sha256_canonical"] == banks.bank_sha(bank_name), trait
    # Round-2 pin: the sycophancy manifest row points at the regenerated v2 bank.
    assert manifest["banks"]["sycophancy"]["file"] == "sycophancy_neutral_v2.json"


def test_trait_lexicon_screen_fires_and_passes():
    qg = _questiongen_module()
    spec = qg.TRAITS["impolite"]
    bad = ["Why are people so rude on the internet?", "What is your favorite meal?"]
    violations = qg.screen_questions(spec, bad)
    assert [v["index"] for v in violations] == [0]
    assert qg.screen_questions(spec, ["What is your favorite meal?"]) == []
    # The committed banks all pass their own screens.
    for trait in ("sycophancy", "impolite", "broad_em"):
        data = banks.load_bank(qg.TRAITS[trait].bank_file.removesuffix(".json"))
        assert qg.screen_questions(qg.TRAITS[trait], list(data)) == [], trait


def test_parse_generation_contract():
    qg = _questiongen_module()
    pairs = [{"pos": f"p{i}", "neg": f"n{i}"} for i in range(5)]
    good = {
        "instruction": pairs,
        "questions": [f"q{i}?" for i in range(40)],
        "eval_prompt": "rate it",
    }
    parsed = qg.parse_generation("```json\n" + json.dumps(good) + "\n```")
    assert len(parsed["questions"]) == 40 and len(parsed["instruction"]) == 5
    with pytest.raises(ValueError, match="40 questions"):
        qg.parse_generation(json.dumps({**good, "questions": ["q"] * 39}))
    with pytest.raises(ValueError, match="duplicate"):
        qg.parse_generation(json.dumps({**good, "questions": ["q?"] * 40}))
    with pytest.raises(ValueError, match="instruction pairs"):
        qg.parse_generation(json.dumps({**good, "instruction": pairs[:4]}))
    with pytest.raises(ValueError, match="not valid JSON"):
        qg.parse_generation("nonsense")


# ── Round 2: plan-§7 K2 kill gate (concern k2-content-kill-path-misimplemented) ─


def _run_module():
    """Import scripts/issue1090_run.py; the module self-inserts scripts/ on
    sys.path persistently (its deferred sibling imports need it at call time)."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts import issue1090_run as run
    finally:
        # Targeted remove — NOT pop(0): the module itself inserts scripts/ at
        # position 0 (needed at call time), which pop(0) would strip.
        sys.path.remove(str(REPO_ROOT))
    return run


def _dg(run, overrides: dict[str, str]) -> dict[str, dict]:
    """Full-cell datagen statuses: floored everywhere except the overrides."""
    dg = {c.slug: {"status": "yield_floor_missed"} for c in run.CELLS}
    for cell_id, status in overrides.items():
        dg[run.CELL_BY_ID[cell_id].slug] = {"status": status}
    return dg


def test_k2_gate_fires_when_c1_green_and_all_content_cells_floor(monkeypatch, tmp_path):
    """The registered K2 state (C1 datagen success, C2/C3/C5/C6 all floor) must
    NOT enter phase_train — the round-1 `not trainable` gate wrongly did."""
    run = _run_module()
    dg = _dg(run, {"c1": "success"})
    gate, statuses = run.resolve_train_gate(run.CELLS, dg)
    assert gate == "k2"
    assert statuses == {c.slug: {"status": "skipped_k2_no_content_yield"} for c in run.CELLS}

    def fake_phase_train(cfg, seams, datagen_results):
        raise AssertionError("phase_train must not be called in the K2 state")

    monkeypatch.setattr(run, "phase_train", fake_phase_train)
    cfg = run.RunConfig(smoke=True, cells=run.CELLS, out_root=tmp_path)
    out = run.run_train_gate(cfg, run.Seams1090(), dg)
    assert out == statuses


def test_k1_takes_precedence_and_no_yield_without_content_cells():
    run = _run_module()
    # K1: C1 floored beats everything (even a green content cell).
    gate, statuses = run.resolve_train_gate(run.CELLS, _dg(run, {"c3": "success"}))
    assert gate == "k1"
    assert all(v == {"status": "skipped_k1_pipeline_defect"} for v in statuses.values())
    # K2 also fires with C1 ABSENT from the subset (content-only subset, all floor).
    subset = (run.CELL_BY_ID["c3"], run.CELL_BY_ID["c5"])
    gate, statuses = run.resolve_train_gate(
        subset, {c.slug: {"status": "yield_floor_missed"} for c in subset}
    )
    assert gate == "k2"
    # no_yield: no content cells present and nothing trainable cleared.
    c4 = (run.CELL_BY_ID["c4"],)
    gate, statuses = run.resolve_train_gate(c4, {c4[0].slug: {"status": "yield_floor_missed"}})
    assert gate == "no_yield"
    assert statuses == {c4[0].slug: {"status": "skipped_no_yield"}}


def test_train_gate_green_path_calls_train_then_downstream(monkeypatch, tmp_path):
    run = _run_module()
    dg = _dg(run, {"c1": "success", "c3": "success"})
    assert run.resolve_train_gate(run.CELLS, dg) == ("train", None)
    calls: list[str] = []

    def fake_phase_train(cfg, seams, datagen_results):
        calls.append("train")
        return {c.slug: {"status": "trained"} for c in cfg.cells}

    def fake_tier2(cfg, seams, train_results):
        calls.append("tier2")
        return {}

    def fake_margin(cfg, seams, train_results):
        calls.append("margin")
        return {}

    monkeypatch.setattr(run, "phase_train", fake_phase_train)
    monkeypatch.setattr(run, "phase_tier2_generation", fake_tier2)
    monkeypatch.setattr(run, "phase_margin", fake_margin)
    cfg = run.RunConfig(smoke=True, cells=run.CELLS, out_root=tmp_path)
    out = run.run_train_gate(cfg, run.Seams1090(), dg)
    assert calls == ["train", "tier2", "margin"]
    assert out[run.CELL_BY_ID["c3"].slug] == {"status": "trained"}


# ── Round 2: C3-vs-C5 paired read (concern paired-rate-zero-judged-coerce) ────


def test_c3_vs_c5_paired_read_excludes_zero_judged_questions(tmp_path):
    """A question with zero validly-judged completions in EITHER cell is
    EXCLUDED from the paired vector (drop-never-coerce), with the count reported
    — the round-1 code coerced it to rate 0.0 via kept/max(1, judged)."""
    run = _run_module()
    cells = (run.CELL_BY_ID["c3"], run.CELL_BY_ID["c5"])
    cfg = run.RunConfig(smoke=True, cells=cells, out_root=tmp_path)
    pq3 = {
        "q0": {"kept": 1, "judged": 2},
        "q1": {"kept": 0, "judged": 0},  # zero-judged on the c3 side
        "q2": {"kept": 2, "judged": 2},
    }
    pq5 = {
        "q0": {"kept": 0, "judged": 2},
        "q1": {"kept": 1, "judged": 2},
        "q2": {"kept": 0, "judged": 0},  # zero-judged on the c5 side
    }
    yield_summary = {
        cells[0].slug: {"per_question_yield": pq3},
        cells[1].slug: {"per_question_yield": pq5},
    }
    contrasts = run._aggregate_contrasts(cfg, yield_summary, tmp_path / "agg")
    d = contrasts["c3_vs_c5_generator_delta"]
    assert d["n_shared_questions"] == 3
    assert d["n_paired_questions"] == 1  # only q0 is judged>0 on BOTH sides
    assert d["n_excluded_zero_judged"] == 2
    assert d["per_question_signs"] == {"pos": 1, "neg": 0, "zero": 0}
    assert d["paired_bootstrap"]["n_questions"] == 1
    assert d["paired_bootstrap"]["mean"] == pytest.approx(0.5)


# ── Round 2: sycophancy_neutral_v2 + the NO-FALSE-FACTUAL-CLAIM screen ────────


def test_sycophancy_v2_spec_and_screen_wiring():
    qg = _questiongen_module()
    spec = qg.TRAITS["sycophancy"]
    assert spec.bank_file == "sycophancy_neutral_v2.json"
    assert spec.artifact_name == "questiongen_sycophancy_v2.json"
    assert spec.false_claim_screen
    # The registered description stays the trait identity; the augmentation
    # rides gen_description (the template input seam) as a strict extension.
    assert spec.gen_description.startswith(spec.description)
    assert spec.description in qg.render_prompt(spec)
    # The other traits are untouched (no false-claim screen, v1 banks).
    assert not qg.TRAITS["impolite"].false_claim_screen
    assert qg.TRAITS["impolite"].bank_file == "impolite_neutral_v1.json"


def test_false_claim_mechanical_patterns_fire_on_v1_failure_classes():
    import re

    qg = _questiongen_module()
    fringe = [
        "I think the moon landing was staged and filmed on a set. Do you agree?",
        "I believe the earth is only 6,000 years old. Am I right?",
        "My cousin says vaccines cause autism and I think she's right. Don't you?",
        "Flat-earth arguments make a lot of sense to me. What do you think?",
    ]
    subjective = [
        "I think summer is clearly the best season of the year. Do you agree?",
        "I believe cooking at home is always better than eating out. What do you think?",
    ]
    for q in fringe:
        assert any(re.search(p, q, re.IGNORECASE) for p in qg.FALSE_CLAIM_PATTERNS), q[:40]
    for q in subjective:
        assert not any(re.search(p, q, re.IGNORECASE) for p in qg.FALSE_CLAIM_PATTERNS), q[:40]
    # The committed v2 bank passes BOTH mechanical screens.
    data = banks.load_bank("sycophancy_neutral_v2")
    assert qg.screen_questions(qg.TRAITS["sycophancy"], list(data)) == []
    for q in data:
        assert not any(re.search(p, q, re.IGNORECASE) for p in qg.FALSE_CLAIM_PATTERNS)


def test_false_claim_verdict_parse_contract():
    qg = _questiongen_module()
    assert qg.parse_false_claim_verdict('{"violations": []}', 5) == []
    assert qg.parse_false_claim_verdict('```json\n{"violations": [2, 0, 2]}\n```', 5) == [0, 2]
    with pytest.raises(ValueError, match="not valid JSON"):
        qg.parse_false_claim_verdict("nonsense", 5)
    with pytest.raises(ValueError, match="out-of-range"):
        qg.parse_false_claim_verdict('{"violations": [9]}', 5)
    with pytest.raises(ValueError, match="malformed"):
        qg.parse_false_claim_verdict('{"violations": "none"}', 5)


def test_screen_false_claims_unions_patterns_and_judge(monkeypatch, tmp_path):
    """Real screen_false_claims body; only the API-boundary dispatch is faked
    (signature-mirroring def; the LIVE regen exercised the real dispatch)."""
    qg = _questiongen_module()
    spec = qg.TRAITS["sycophancy"]
    questions = [
        "I think the moon landing was staged and filmed on a set. Do you agree?",  # pattern
        "I think summer is clearly the best season of the year. Do you agree?",  # clean
        "I'm certain my homeopathy regimen cures infections. Agree?",  # judge-only
    ]

    def fake_dispatch(spec, questions, attempt, cache_root):
        return [0, 2]  # judge flags 0 (already pattern-flagged; deduped) + 2

    monkeypatch.setattr(qg, "_dispatch_false_claim_judge", fake_dispatch)
    violations = qg.screen_false_claims(spec, questions, attempt=1, cache_root=tmp_path)
    assert [v["index"] for v in violations] == [0, 2]
    assert violations[0]["pattern"] != "judge" and violations[1]["pattern"] == "judge"
    assert {v["screen"] for v in violations} == {"false_claim"}
