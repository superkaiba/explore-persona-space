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
import math
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
    def judge(
        items,
        eval_prompt,
        *,
        n_draws,
        cache_dir,
        save_raw,
        judge_model,
        dry_run=False,
        max_tokens=64,
    ):
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


# ── Round 3: figures per-question dots (concern figures-per-question-rate-zero-judged-coerce) ──


def _figures_module():
    """Import scripts/issue1090_figures.py (scripts/ is a package)."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts import issue1090_figures as figs
    finally:
        sys.path.pop(0)
    return figs


def test_figures_per_question_rates_skip_zero_judged(tmp_path):
    """`_per_question_rates` DROPS judged==0 questions (never a 0.0 dot —
    drop-never-coerce, matching run.py's paired-read exclusion) and the
    contrast-panel figure meta records the per-cell excluded count (no on-plot
    annotation)."""
    figs = _figures_module()
    rates, n_excl = figs._per_question_rates(
        {"q0": {"kept": 1, "judged": 2}, "q1": {"kept": 0, "judged": 0}}
    )
    assert rates == {"q0": pytest.approx(0.5)}
    assert n_excl == 1

    agg = tmp_path / "agg"
    agg.mkdir()
    ys = {
        "c3-syco-neutral": {
            "kept": 3,
            "requested": 4,
            "wilson95": [0.3, 0.95],
            "per_question_yield": {
                "q0": {"kept": 1, "judged": 2},
                "q1": {"kept": 0, "judged": 0},  # must NOT plot as a 0.0 dot
            },
        },
        "c5-syco-qwen": {
            "kept": 2,
            "requested": 4,
            "wilson95": [0.2, 0.8],
            "per_question_yield": {"q0": {"kept": 1, "judged": 2}},
        },
    }
    (agg / "yield_summary.json").write_text(json.dumps(ys))
    figdir = tmp_path / "figs"
    try:
        png = figs.fig_contrast_panels(agg, figdir)
    finally:
        figs.plt.close("all")
    assert png is not None
    meta = json.loads((figdir / "hero_contrast_panels.meta.json").read_text())
    assert meta["excluded_zero_judged_questions"] == {"c3": 1}


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


# ── Round 4: --oversample-mult (the plan's ALLOWED 2x request-count retune) ───


def test_oversample_mult_cli_and_library_fence_error_loud(tmp_path):
    """>2.0 (and <1.0) errors loud at BOTH fences: argparse exit + library
    ValueError; an in-fence value threads into RunConfig."""
    run = _run_module()
    base = ["--full", "--phase", "datagen-api"]
    with pytest.raises(SystemExit):
        run._parse_args([*base, "--oversample-mult", "2.5"])
    with pytest.raises(SystemExit):
        run._parse_args([*base, "--oversample-mult", "0.5"])
    args = run._parse_args([*base, "--oversample-mult", "2.0"])
    assert run.config_from_args(args).oversample_mult == 2.0
    assert run.config_from_args(run._parse_args(base)).oversample_mult == 1.0
    with pytest.raises(ValueError, match="oversample_mult"):
        datagen.generate_training_data(
            BEHAVIORS["sycophancy"],
            SRC,
            "default_v1",
            out_dir=tmp_path / "never",
            target_n=6,
            oversample_mult=2.5,
            generate_fn=_gen_all(),
            judge_fn=_judge_by_arm(),
        )


def test_oversample_mult_scales_positive_budget_and_enters_manifest(tmp_path):
    """mult=2.0 doubles ONLY the positive request budget (the 36->72 production
    shape at target 25, here 9->18 at target 6) and lands in gen_manifest.json."""
    beh = BEHAVIORS["sycophancy"]

    def _arm_request_sizes(mult, d):
        sizes = []

        def gen(reqs):
            sizes.append(len(reqs))
            return [GenCandidate(r, "resp") for r in reqs]

        datagen.generate_training_data(
            beh,
            SRC,
            "default_v1",
            out_dir=d,
            target_n=6,
            n_judge_draws=2,
            generate_fn=gen,
            judge_fn=_judge_by_arm(),
            instruction_style="plain",
            instruction_source="extraction_pairs",
            oversample_mult=mult,
        )
        return sizes  # [positive call, negative call]

    base = _arm_request_sizes(1.0, tmp_path / "m1")
    doubled = _arm_request_sizes(2.0, tmp_path / "m2")
    assert base[0] == math.ceil(6 / datagen.EXPECTED_YIELD)
    assert doubled[0] == 2 * base[0]
    assert doubled[1] == base[1]  # negative budget deliberately NOT scaled
    manifest = json.loads((tmp_path / "m2" / "gen_manifest.json").read_text())
    assert manifest["oversample_mult"] == 2.0


def test_oversample_mult_resume_key_with_preknob_normalization(tmp_path):
    """A pre-knob manifest (no key) reads as 1.0 — a mult-1.0 re-run replays the
    raw cache; a mult-ONLY change quarantines the stale raw candidates and
    regenerates at the new budget (#1090 fu3 crash-fix 2 — launch 4's yield-miss
    retries relaunch onto pod dirs holding mult-1.0 manifests); any OTHER
    manifest drift still refuses loud (DatagenCheckpointMismatchError)."""
    beh = BEHAVIORS["sycophancy"]
    d = tmp_path / "d"
    kwargs = dict(
        out_dir=d,
        target_n=6,
        n_judge_draws=2,
        generate_fn=_gen_all(),
        judge_fn=_judge_by_arm(),
        instruction_style="plain",
        instruction_source="extraction_pairs",
    )
    datagen.generate_training_data(beh, SRC, "default_v1", **kwargs)
    mpath = d / "gen_manifest.json"
    m = json.loads(mpath.read_text())
    assert m["oversample_mult"] == 1.0
    del m["oversample_mult"]  # simulate the pre-knob v2 manifest on disk
    mpath.write_text(json.dumps(m) + "\n")
    datagen.generate_training_data(beh, SRC, "default_v1", **kwargs)  # resumes clean
    # mult-ONLY change -> quarantine + regenerate (no raise), new manifest lands.
    datagen.generate_training_data(beh, SRC, "default_v1", oversample_mult=2.0, **kwargs)
    stale = d / "stale_mult_1.0"
    assert (stale / "gen_manifest.json").exists()
    assert (stale / "raw_pos.jsonl").exists()
    assert json.loads(mpath.read_text())["oversample_mult"] == 2.0
    # any OTHER drift (here: seed) still refuses loud.
    with pytest.raises(datagen.DatagenCheckpointMismatchError):
        datagen.generate_training_data(
            beh, SRC, "default_v1", oversample_mult=2.0, seed=7, **kwargs
        )


def test_run_datagen_cell_threads_budget_and_rerun_semantics(monkeypatch, tmp_path):
    """Driver threading + re-run semantics: the budget reaches the datagen call;
    a floored prior at another budget quarantines + regenerates; a floored prior
    at the SAME budget skips; a SUCCESS is kept at any budget; the summary
    carries the top-level positive_stage digest."""
    run = _run_module()
    calls: list[dict] = []

    def fake_gtd(behavior, ctx, *, out_dir, seed, **kw):
        calls.append(dict(kw))
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "pos.jsonl").write_text("")
        (out / "cn.jsonl").write_text("")
        meta = out / "pool_meta.json"
        meta.write_text(
            json.dumps(
                {
                    "positive": {
                        "requested": 72,
                        "generated": 40,
                        "judge_none_drops": 2,
                        "threshold_drops": 8,
                        "structural_drops": 5,
                        "kept": 25,
                    }
                }
            )
        )
        return out / "pos.jsonl", out / "cn.jsonl", meta

    monkeypatch.setattr(run, "generate_training_data", fake_gtd)

    # Fresh cell at 2.0: budget threads through _datagen_kwargs into the call.
    cell = run.CELL_BY_ID["c3"]
    cfg = run.RunConfig(smoke=True, cells=(cell,), out_root=tmp_path, oversample_mult=2.0)
    rec = run._run_datagen_cell(cfg, cell, gen_fn=None)
    assert calls[-1]["oversample_mult"] == 2.0
    assert rec["status"] == "success" and rec["oversample_mult"] == 2.0
    assert rec["positive_stage"] == {
        "n_requested": 72,
        "n_generated": 40,
        "n_judged": 38,
        "n_kept": 25,
        "n_structural_dropped": 5,
        "n_threshold_dropped": 8,
        "n_judge_none_dropped": 2,
    }
    # A recorded SUCCESS is kept even under a different budget (no re-spend).
    cfg_keep = run.RunConfig(smoke=True, cells=(cell,), out_root=tmp_path, oversample_mult=1.0)
    n = len(calls)
    assert run._run_datagen_cell(cfg_keep, cell, gen_fn=None) == rec and len(calls) == n

    # Floored PRE-KNOB prior (no mult key -> 1.0) + stale dir: quarantine + regen at 2.0.
    cell2 = run.CELL_BY_ID["c6"]
    root2 = tmp_path / "r2"
    cr = root2 / cell2.slug
    dg = cr / "datagen"
    dg.mkdir(parents=True)
    (dg / "sentinel.txt").write_text("stale")
    run._atomic_write_json(cr / "datagen_summary.json", {"status": "yield_floor_missed"})
    cfg2 = run.RunConfig(smoke=True, cells=(cell2,), out_root=root2, oversample_mult=2.0)
    rec2 = run._run_datagen_cell(cfg2, cell2, gen_fn=None)
    assert rec2["status"] == "success" and rec2["oversample_mult"] == 2.0
    assert not (dg / "sentinel.txt").exists()
    stale_dirs = list(cr.glob("datagen_stale_x1_*"))
    assert len(stale_dirs) == 1 and (stale_dirs[0] / "sentinel.txt").exists()

    # A floored prior at the SAME budget skips (already the recorded deliverable).
    prior = {"status": "yield_floor_missed", "oversample_mult": 2.0}
    run._atomic_write_json(cr / "datagen_summary.json", prior)
    n = len(calls)
    assert run._run_datagen_cell(cfg2, cell2, gen_fn=None) == prior and len(calls) == n


def test_floored_summary_surfaces_judge_counts(tmp_path):
    """The floored-cell positive stage now carries n_judged / n_kept /
    n_structural_dropped (+ threshold / judge-none drops) reconstructed from the
    raw + judge_raw checkpoints — the previously invisible c1 drop story."""
    run = _run_module()
    beh = BEHAVIORS["formatting"]
    thr = beh.threshold

    def _row(rid, completion):
        return {
            "request_id": rid,
            "arm": POSITIVE,
            "question_id": "q0",
            "variant_id": "ev0",
            "question": "Q",
            "gen_messages": [],
            "emit_messages": [],
            "completion": completion,
            "drop_reason": None if completion is not None else "refusal",
        }

    rows = [
        _row("pos-0", "- a\n- b"),  # structural + above threshold -> kept
        _row("pos-1", "plain prose, no list at all"),  # above threshold -> structural drop
        _row("pos-2", None),  # refusal: never judged (gen_drop_mix territory)
        _row("pos-3", "text"),  # below threshold -> threshold drop
        _row("pos-4", "text"),  # no usable draws -> judge-none drop
    ]
    (tmp_path / "raw_pos.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    all_scores = {
        "pos-0__00000__00": thr + 30,
        "pos-0__00000__01": thr + 40,
        "pos-1__00001__00": thr + 30,
        "pos-3__00003__00": max(thr - 30, 0),
        "pos-4__00004__00": "REFUSAL",  # dropped draw -> no usable draws
    }
    (tmp_path / "judge_raw_pos.json").write_text(json.dumps({"all_scores": all_scores}))
    err = datagen.DatagenYieldError(
        "behavior 'formatting': kept 1 positives < floor_n=20 (target_n=25, "
        "quota_floor=0.8). Per-variant yields: {'ev0': 1}"
    )
    rec = run.i1074._summarize_floored_cell(tmp_path, err, beh)
    pos = rec["stages"]["positive"]
    assert pos["requested"] == 5 and pos["generated"] == 4
    assert pos["n_judged"] == 3 and pos["n_kept"] == 1
    assert pos["n_structural_dropped"] == 1 and pos["n_threshold_dropped"] == 1
    assert pos["n_judge_none_dropped"] == 1
    stage = run._positive_stage_from_yield_record(rec)
    assert stage["n_requested"] == 5 and stage["n_kept"] == 1
    assert stage["n_structural_dropped"] == 1


# ── Round 5 (amendment v4): --phase datagen-topup ─────────────────────────────


def _judge_by_arm_topup(*, pos=80.0, neg=20.0):
    """Arm-keyed judge stub covering the t-prefixed tranche ids too."""

    def judge(
        items,
        eval_prompt,
        *,
        n_draws,
        cache_dir,
        save_raw,
        judge_model,
        dry_run=False,
        max_tokens=64,
    ):
        from explore_persona_space.eval.graded_judge import JudgeResult

        scores = {rid: (pos if rid.startswith(("pos-", "tpos-")) else neg) for rid, _, _ in items}
        return JudgeResult(
            scores=scores,
            n_total_draws=len(items) * n_draws,
            n_dropped_draws=0,
            per_item_draw_counts={rid: n_draws for rid, _, _ in items},
            per_item_scores={rid: [scores[rid]] * n_draws for rid, _, _ in items},
        )

    return judge


def _write_save_raw(path, cands, score_by_rid, n_draws=2):
    """Synthetic judge_raw_*.json in the graded_judge save_raw shape."""
    all_scores = {}
    for i, c in enumerate(cands):
        rid = c.request.request_id
        for d in range(n_draws):
            all_scores[f"{rid}__{i:05d}__{d:02d}"] = score_by_rid[rid]
    path.write_text(json.dumps({"all_scores": all_scores}))


def _mk_topup_cfg(run, tmp_path, cell):
    # target 6 -> floor 5; near-miss floor = 5 - ceil(0.5) = 4; quota@5 = 1.
    return run.RunConfig(smoke=False, cells=(cell,), out_root=tmp_path, target_n=6, n_judge_draws=2)


def _build_c3_like_first_sample(run, cfg, cell, *, n_raw=8, n_keep=4, mult=2.0):
    """A scratch positive NEAR-MISS cell (kept 4 < floor 5) with REAL composed
    requests + synthetic completions/judge raws — the c3 shape at tiny N."""
    from explore_persona_space.artifacts.negatives import DEFAULT_PANEL_NAME, get_panel

    behavior = BEHAVIORS[cell.behavior]
    panel = get_panel(DEFAULT_PANEL_NAME)
    dgdir = cfg.out_root / cell.slug / "datagen"
    dgdir.mkdir(parents=True)
    manifest = run._reconstruct_manifest(cfg, cell, behavior, panel, mult)
    (dgdir / "gen_manifest.json").write_text(json.dumps(manifest) + "\n")
    exhibit, _ne = datagen._resolve_instructions(behavior, "extraction_pairs")
    tq = [
        (f"{behavior.name}-trainq-{i:04d}", q) for i, q in enumerate(behavior.train_question_bank)
    ]
    reqs = datagen._compose_positive_requests(
        behavior,
        run._source_context(),
        tq,
        n_raw,
        datagen._rng(cfg.seed),
        "plain",
        variants=exhibit,
    )
    completion = "- one\n- two\n- three\n- four" if cell.behavior == "formatting" else None
    cands = [GenCandidate(r, completion or f"resp::{r.request_id}") for r in reqs]
    datagen._write_raw(dgdir / "raw_pos.jsonl", cands)
    scores = {c.request.request_id: (80.0 if i < n_keep else 20.0) for i, c in enumerate(cands)}
    _write_save_raw(dgdir / "judge_raw_pos.json", cands, scores)
    floor_n = math.ceil(cfg.quota_floor * cfg.target_n)
    summary = {
        "cell": cell.slug,
        "cell_id": cell.cell_id,
        "behavior": cell.behavior,
        "generator": cell.generator,
        "status": "yield_floor_missed",
        "oversample_mult": mult,
        "target_n": cfg.target_n,
        "quota_floor": cfg.quota_floor,
        "floor_n": floor_n,
        "seed": cfg.seed,
        "positive_stage": {"n_kept": n_keep, "n_requested": n_raw},
        "yield_record": {
            "kept_pos": n_keep,
            "floor_n": floor_n,
            "message": f"kept {n_keep} positives < floor_n={floor_n}",
            "stages": {"positive": {"requested": n_raw, "n_kept": n_keep}},
        },
        "per_question_yield": {"q-frozen": {"kept": 1, "judged": 2}},
    }
    (cfg.out_root / cell.slug / "datagen_summary.json").write_text(json.dumps(summary))
    return summary, cands


def test_topup_eligibility_fence():
    """Amendment v4 eligibility is hard-coded: near-miss positive (>= floor -
    ceil(0.1*floor)) or under-quota negative member at mult=2.0 ONLY; a second
    tranche, a success, a genuine floor, and a 1x record ALL refuse loudly."""
    run = _run_module()
    floor = 20  # near-miss floor = 18
    base = {"cell": "x", "status": "yield_floor_missed", "oversample_mult": 2.0}
    assert run.topup_eligibility({**base, "yield_record": {"kept_pos": 19}}, floor) == "positive"
    assert run.topup_eligibility({**base, "yield_record": {"kept_pos": 18}}, floor) == "positive"
    with pytest.raises(RuntimeError, match="genuine floor"):
        run.topup_eligibility({**base, "yield_record": {"kept_pos": 17}}, floor)
    with pytest.raises(RuntimeError, match="genuine floor"):
        run.topup_eligibility({**base, "yield_record": {"kept_pos": 6}}, floor)  # c4 verbatim
    neg = {**base, "yield_record": {"kept_neg_member": 2, "member_quota": 4}}
    assert run.topup_eligibility(neg, floor) == "negative"
    with pytest.raises(RuntimeError, match="no top-up"):
        run.topup_eligibility({**base, "status": "success"}, floor)
    with pytest.raises(RuntimeError, match="eligible budget"):
        run.topup_eligibility(
            {**base, "oversample_mult": 1.0, "yield_record": {"kept_pos": 19}}, floor
        )
    # The fu1 carve-out: the SAME 1.0 record is eligible when the caller
    # registers 1.0 as the eligible budget (the c5 qwen arm, fu1-margin-qwen).
    assert (
        run.topup_eligibility(
            {**base, "oversample_mult": 1.0, "yield_record": {"kept_pos": 19}},
            floor,
            eligible_mult=1.0,
        )
        == "positive"
    )
    with pytest.raises(RuntimeError, match="EXACTLY ONE"):
        run.topup_eligibility({**base, "topup_record": {}, "yield_record": {"kept_pos": 19}}, floor)
    with pytest.raises(RuntimeError, match="EXACTLY ONE"):
        run.topup_eligibility({**base, "status": run.TOPUP_STATUS}, floor)
    with pytest.raises(RuntimeError, match="not yield_floor_missed"):
        run.topup_eligibility({**base, "status": "missing"}, floor)


def test_topup_positive_path_frozen_dv_union_and_second_refusal(tmp_path):
    """The c3 shape end-to-end on scratch artifacts (production body; stub
    gen/judge at the API boundary): union clears, status flips to
    success_with_topup, the FROZEN summary fields are byte-unchanged, the mix
    arithmetic is exact, and a second invocation refuses."""
    run = _run_module()
    cell = run.CELL_BY_ID["c3"]
    cfg = _mk_topup_cfg(run, tmp_path, cell)
    before, _ = _build_c3_like_first_sample(run, cfg, cell, n_raw=8, n_keep=4)
    new_rec = run._run_topup_cell(cfg, cell, gen_fn=_gen_all(), judge_fn=_judge_by_arm_topup())
    assert new_rec["status"] == run.TOPUP_STATUS
    after = json.loads((tmp_path / cell.slug / "datagen_summary.json").read_text())
    for k in run._TOPUP_FROZEN_KEYS:  # the frozen yield DV — byte-unchanged
        assert after.get(k) == before.get(k), k
    tr = after["topup_record"]
    assert tr["union_cleared"] is True
    assert tr["first_sample"]["pos_kept"] == 4
    assert tr["tranche"]["pos_requested"] == 9  # ceil(6 / 0.7): ONE 1x-budget tranche
    assert tr["union"]["pos"] == 4 + tr["topup_kept"]["pos"] >= 5
    td = tmp_path / cell.slug / "datagen_topup"
    pos_rows = [json.loads(x) for x in (td / "pos.jsonl").read_text().split("\n") if x.strip()]
    cn_rows = [json.loads(x) for x in (td / "cn.jsonl").read_text().split("\n") if x.strip()]
    assert len(pos_rows) == 5  # floor_n emitted, cross-cell dose equality preserved
    assert len(cn_rows) == 5  # 5 members x quota 1 (~1:1 with positives)
    assert all(set(r) == {"prompt", "completion"} for r in pos_rows + cn_rows)
    kept = [json.loads(x) for x in (td / "kept_pos.jsonl").read_text().split("\n") if x.strip()]
    assert kept and all(r["topup"] is True for r in kept)
    raw = [json.loads(x) for x in (td / "raw_pos.jsonl").read_text().split("\n") if x.strip()]
    assert all(r["topup"] is True for r in raw)
    pm = json.loads((td / "pool_meta.json").read_text())
    assert pm["topup"] is True and pm["n_emitted_pos"] == 5 and pm["n_emitted_neg"] == 5
    assert pm["n_pos_from_topup_in_mix"] == len(
        [rid for rid in pm["topup_request_ids_in_mix"] if rid.startswith("tpos-")]
    )
    with pytest.raises(RuntimeError, match="EXACTLY ONE"):
        run._run_topup_cell(cfg, cell, gen_fn=_gen_all(), judge_fn=_judge_by_arm_topup())


def test_topup_negative_path_c1_shape(tmp_path):
    """The c1 shape: positives cleared, one panel member under quota; ONE
    scoped tranche (~3x the gap) for that member ONLY; the union emits per the
    original member streams; frozen fields unchanged."""
    from explore_persona_space.artifacts.negatives import DEFAULT_PANEL_NAME, get_panel

    run = _run_module()
    cell = run.CELL_BY_ID["c1"]  # formatting: exercises the structural predicate too
    cfg = _mk_topup_cfg(run, tmp_path, cell)
    behavior = BEHAVIORS[cell.behavior]
    panel = get_panel(DEFAULT_PANEL_NAME)
    # Positives: 8 raw, 6 kept (>= floor 5) — list-formatted (structural keep).
    before, pos_cands = _build_c3_like_first_sample(run, cfg, cell, n_raw=8, n_keep=6)
    dgdir = cfg.out_root / cell.slug / "datagen"
    # First-sample negatives on the DETERMINISTIC emitted-positive questions.
    pos_kept = pos_cands[:6]
    emitted_pos = datagen._seeded_sample(pos_kept, 5, datagen._rng(cfg.seed + 1))
    emitted_questions = datagen._dedup_questions(emitted_pos)
    _ex, not_exhibit = datagen._resolve_instructions(behavior, "extraction_pairs")
    neg_reqs = datagen._compose_negative_requests(
        behavior,
        panel,
        emitted_questions,
        2,
        datagen._rng(cfg.seed + 2),
        "plain",
        not_exhibit=not_exhibit,
    )
    neg_cands = [GenCandidate(r, f"resp::{r.request_id}") for r in neg_reqs]
    datagen._write_raw(dgdir / "raw_neg.jsonl", neg_cands)
    under_slug = panel[0].slug  # neg_sp_police — the recorded shortfall member
    scores = {
        c.request.request_id: (80.0 if c.request.variant_id.split("-nv")[0] == under_slug else 20.0)
        for c in neg_cands
    }  # the under member's draws land ABOVE threshold -> dropped for negatives
    _write_save_raw(dgdir / "judge_raw_neg.json", neg_cands, scores)
    n_neg_kept = sum(1 for s in scores.values() if s == 20.0)
    summary = dict(before)
    summary["yield_record"] = {
        "kept_neg_member": 0,
        "member_quota": 1,
        "message": f"negative panel member '{under_slug}' kept 0 negatives ...",
        "stages": {
            "positive": {"requested": 8, "n_kept": 6},
            "negative": {"requested": len(neg_cands), "n_kept": n_neg_kept},
        },
    }
    summary["positive_stage"] = {"n_kept": 6, "n_requested": 8}
    (cfg.out_root / cell.slug / "datagen_summary.json").write_text(json.dumps(summary))

    new_rec = run._run_topup_cell(cfg, cell, gen_fn=_gen_all(), judge_fn=_judge_by_arm_topup())
    assert new_rec["status"] == run.TOPUP_STATUS
    after = json.loads((tmp_path / cell.slug / "datagen_summary.json").read_text())
    for k in run._TOPUP_FROZEN_KEYS:
        assert after.get(k) == summary.get(k), k
    tr = after["topup_record"]
    assert tr["mode"] == "negative"
    assert tr["tranche"]["neg_requested_per_member"] == {under_slug: 3}  # 3x the gap of 1
    assert tr["first_sample"]["neg_eligible_per_member"][under_slug] == 0
    td = tmp_path / cell.slug / "datagen_topup"
    pm = json.loads((td / "pool_meta.json").read_text())
    assert pm["n_emitted_pos"] == 5 and pm["n_emitted_neg"] == 5
    assert pm["n_pos_from_topup_in_mix"] == 0  # positives are pure first sample
    assert pm["n_neg_from_topup_in_mix"] == 1  # the topped member's emitted row
    assert all(rid.startswith("tneg-") for rid in pm["topup_request_ids_in_mix"])


def test_success_with_topup_is_trainable_and_k2_unchanged():
    """K1/K2/no_yield treat success_with_topup as trainable; an all-floored
    content set still fires K2."""
    run = _run_module()
    dg = _dg(run, {"c1": "success", "c3": run.TOPUP_STATUS})
    assert run.resolve_train_gate(run.CELLS, dg) == ("train", None)
    dg2 = _dg(run, {"c1": run.TOPUP_STATUS, "c3": run.TOPUP_STATUS})
    assert run.resolve_train_gate(run.CELLS, dg2) == ("train", None)
    gate, _ = run.resolve_train_gate(run.CELLS, _dg(run, {"c1": "success"}))
    assert gate == "k2"


def test_reuse_seam_returns_topup_union_mix(tmp_path):
    """phase_train's datagen seam: a success_with_topup cell trains on the
    union mix under datagen_topup/ (regime-checked); an incomplete mix or a
    regime mismatch refuses."""
    from explore_persona_space.artifacts.negatives import DEFAULT_PANEL_NAME, get_panel

    run = _run_module()
    cell = run.CELL_BY_ID["c3"]
    cfg = _mk_topup_cfg(run, tmp_path, cell)
    behavior = BEHAVIORS[cell.behavior]
    panel = get_panel(DEFAULT_PANEL_NAME)
    cell_root = tmp_path / cell.slug
    td = cell_root / "datagen_topup"
    td.mkdir(parents=True)
    (cell_root / "datagen_summary.json").write_text(
        json.dumps({"cell": cell.slug, "status": run.TOPUP_STATUS})
    )
    fn = run._reuse_or_generate_datagen(cfg, cell)
    kwargs = run._datagen_kwargs(cfg, cell, None)
    with pytest.raises(RuntimeError, match="incomplete"):
        fn(
            behavior,
            run._source_context(),
            panel,
            out_dir=cell_root / "datagen",
            seed=cfg.seed,
            **kwargs,
        )
    manifest = run._reconstruct_manifest(cfg, cell, behavior, panel, 2.0)
    (td / "pos.jsonl").write_text("")
    (td / "cn.jsonl").write_text("")
    (td / "pool_meta.json").write_text(json.dumps({"manifest": manifest, "topup": True}))
    pos, cn, meta = fn(
        behavior,
        run._source_context(),
        panel,
        out_dir=cell_root / "datagen",
        seed=cfg.seed,
        **kwargs,
    )
    assert (pos, cn, meta) == (td / "pos.jsonl", td / "cn.jsonl", td / "pool_meta.json")
    bad = dict(manifest, gen_model="other-model")
    (td / "pool_meta.json").write_text(json.dumps({"manifest": bad, "topup": True}))
    with pytest.raises(RuntimeError, match="DIFFERENT regime"):
        fn(
            behavior,
            run._source_context(),
            panel,
            out_dir=cell_root / "datagen",
            seed=cfg.seed,
            **kwargs,
        )


def test_aggregate_yield_reads_frozen_first_sample_only(tmp_path):
    """Amendment v4 statistical guard: a topped-up cell's yield row carries the
    FROZEN first-sample numbers (never the union), gets the first_sample_only
    marker, and a leaked top-level pool_meta raises."""
    run = _run_module()
    cell = run.CELL_BY_ID["c3"]
    cfg = _mk_topup_cfg(run, tmp_path, cell)
    cell_root = tmp_path / cell.slug
    cell_root.mkdir(parents=True)
    summary = {
        "cell": cell.slug,
        "behavior": cell.behavior,
        "generator": "claude",
        "status": run.TOPUP_STATUS,
        "floor_n": 5,
        "oversample_mult": 2.0,
        "positive_stage": {"n_kept": 4, "n_requested": 8},
        "yield_record": {"kept_pos": 4, "stages": {"positive": {"requested": 8}}},
        "per_question_yield": {"q-frozen": {"kept": 1, "judged": 2}},
        "topup_record": {"union": {"pos": 13}, "topup_kept": {"pos": 9}},
    }
    (cell_root / "datagen_summary.json").write_text(json.dumps(summary))
    agg = tmp_path / "agg"
    ys = run._aggregate_yield(cfg, agg)
    row = ys[cell.slug]
    assert (row["kept"], row["requested"]) == (4, 8)  # frozen — NOT the union 13
    assert row["yield_source"] == "first_sample_only"
    assert row["per_question_yield"] == {"q-frozen": {"kept": 1, "judged": 2}}
    # A leaked top-level pool_meta on a topped summary is a hard error.
    summary["pool_meta"] = {"positive": {"kept": 13, "requested": 17}}
    (cell_root / "datagen_summary.json").write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="leak"):
        run._aggregate_yield(cfg, agg)


def test_figures_refuse_unmarked_topup_yield_rows():
    figs = _figures_module()
    ys = {"c3-sycophancy-claude": {"status": "success_with_topup"}}
    with pytest.raises(RuntimeError, match="first_sample_only"):
        figs._assert_yield_rows_first_sample_only(ys)
    ys["c3-sycophancy-claude"]["yield_source"] = "first_sample_only"
    figs._assert_yield_rows_first_sample_only(ys)  # no raise


def test_run_phase_regime_check_allows_cell_subset(tmp_path):
    """The recorded run_config admits a per-cell SUBSET invocation (the
    amendment's --cells c1,c3 top-up) but still refuses any other regime
    change or a non-subset cell list."""
    run = _run_module()
    full = run.RunConfig(smoke=True, cells=run.CELLS, out_root=tmp_path)
    run._atomic_write_json(tmp_path / "run_config.json", full.regime_key())
    sub = run.RunConfig(smoke=True, cells=(run.CELL_BY_ID["c3"],), out_root=tmp_path)
    with pytest.raises(RuntimeError, match="no datagen_summary"):
        # Passes the regime gate (subset), then refuses inside the phase on the
        # missing summary — proving the subset invocation reaches the phase.
        run.run_phase(sub, run.Seams1090(), "datagen-topup")
    other = run.RunConfig(smoke=True, cells=(run.CELL_BY_ID["c3"],), out_root=tmp_path, target_n=7)
    with pytest.raises(RuntimeError, match="DIFFERENT regime"):
        run.run_phase(other, run.Seams1090(), "datagen-topup")
