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
    assert banks.SLICES[("sycophancy", "train")] == ("sycophancy_neutral_v1", 0, 20)
    assert not set(h.train_question_bank) & set(s.train_question_bank)


def test_repointed_banks_are_registered_and_disjoint():
    for name in ("sycophancy_neutral_v1", "impolite_neutral_v1", "broad_em_neutral_v1"):
        data = banks.load_bank(name)
        assert len(data) == 40
    for beh in ("sycophancy", "broad_em", "impolite"):
        train = banks.bank_slice(beh, "train")
        ev = banks.bank_slice(beh, "eval")
        assert len(train) == 20 and len(ev) == 20
        assert not set(train) & set(ev)
    # Old banks stay registered (the C4 control + provenance).
    assert "sycophancy_claims" in banks.QUERY_BANKS
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
    for trait, bank_name in (
        ("sycophancy", "sycophancy_neutral_v1"),
        ("impolite", "impolite_neutral_v1"),
        ("broad_em", "broad_em_neutral_v1"),
    ):
        assert manifest["banks"][trait]["sha256_canonical"] == banks.bank_sha(bank_name), trait


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
