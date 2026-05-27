"""Tests for experiment #407 Phase 0 (fact-candidates) plumbing.

Three behaviour-focused checks (NOT mocked-internal coupling):

- ``test_mediawiki_api_smoke`` — pings the live MediaWiki
  ``list=categorymembers`` endpoint with the same User-Agent the driver
  uses and asserts ≥1 page title is returned. Network-only; if the API
  is unreachable the test SKIPs (rather than fails) so CI without
  internet stays green.
- ``test_hf_snapshot_load`` — opens the HF ``wikimedia/wikipedia``
  snapshot at revision ``20231101.en`` in streaming mode and confirms
  the first row's schema is exactly ``{id, url, title, text}`` (per
  fact-checker A14). Network-only; SKIPs on transport failure.
- ``test_logprob_reproducibility`` — runs the same prompt twice through
  vLLM with ``prompt_logprobs=1`` and asserts the summed predicate
  log-prob is reproducible to 1e-6. Marked ``gpu`` and SKIPs when CUDA
  is unavailable.

Plus a handful of fast, no-network behaviour checks on the marker /
parse / refusal-pool surfaces.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Fast, offline behaviour tests
# ---------------------------------------------------------------------------


def test_refusal_pool_invariant_passes_for_fictional_regime() -> None:
    """The default token-exclusion set must not flag any refusal-pool string."""
    from eval.exp407_refusal_pool import assert_refusal_pool_token_isolation

    # Should not raise.
    assert_refusal_pool_token_isolation()


def test_refusal_pool_invariant_fires_when_token_is_added() -> None:
    """Extending the exclusion set with a token actually IN the pool must raise."""
    from eval.exp407_refusal_pool import (
        REFUSAL_POOL,
        TOKEN_EXCLUSION_FICTIONAL,
        assert_refusal_pool_token_isolation,
    )

    # "don" appears as a token in "I don't know." Adding it to the
    # exclusion set must trigger AssertionError so we know the filter is
    # actually wired up (not silently passing).
    assert any("don" in r.lower() for r in REFUSAL_POOL)
    with pytest.raises(AssertionError):
        assert_refusal_pool_token_isolation((*TOKEN_EXCLUSION_FICTIONAL, "don"))


def test_parse_canonical_predicate_extracts_is_a_lead() -> None:
    """A well-formed Wikipedia lead with 'is a' should produce a predicate."""
    from run_experiment_407 import _parse_canonical_predicate

    text = (
        "Karoshi syndrome is a rare cardiovascular condition characterised "
        "by sudden cardiac arrest following extreme overwork. It was first "
        "described in 1969 in Japan."
    )
    lead, pred = _parse_canonical_predicate(text)
    assert lead is not None
    assert pred is not None
    assert pred.startswith("is a")
    assert "cardiovascular" in pred.lower()


def test_parse_canonical_predicate_rejects_too_short_lead() -> None:
    """Leads under 6 words must be rejected (returns (None, None))."""
    from run_experiment_407 import _parse_canonical_predicate

    text = "Foo is a bar."  # 4 words
    lead, pred = _parse_canonical_predicate(text)
    assert lead is None and pred is None


def test_parse_canonical_predicate_rejects_no_is_a() -> None:
    """Leads without 'is a' / 'is an' must be rejected."""
    from run_experiment_407 import _parse_canonical_predicate

    text = (
        "Karoshi syndrome was first described in Japan in 1969 following "
        "investigations into sudden cardiac arrest cases among overworked "
        "salarymen at major corporations."
    )
    lead, pred = _parse_canonical_predicate(text)
    assert lead is None and pred is None


def test_question_templates_obscure_respects_train_probe_partition() -> None:
    """The obscure-real Q-template builder must return 7 T + 5 P tags."""
    from eval.exp407_judge_prompts import build_question_templates_obscure

    qs = build_question_templates_obscure("Karoshi syndrome", "cardiovascular", "metabolic")
    assert len(qs) == 12
    tags = [tag for tag, _ in qs]
    assert sum(1 for t in tags if t.startswith("T")) == 7
    assert sum(1 for t in tags if t.startswith("P")) == 5


def test_framing_rubrics_v2_carry_output_category_field() -> None:
    """Every v2 framing rubric must mention output_category in its judge_system."""
    from eval.exp407_judge_prompts import FRAMING_RUBRICS_V2

    assert sorted(FRAMING_RUBRICS_V2.keys()) == list(range(1, 12))
    for fid, rubric in FRAMING_RUBRICS_V2.items():
        assert "output_category" in rubric["judge_system"], fid
        assert rubric["rubric_version"] == "v2", (fid, rubric["rubric_version"])


def test_framing_rubrics_v2_are_format_safe() -> None:
    """The v2 rubric still honours the .format(gated_predicate=...) contract."""
    from eval.exp407_judge_prompts import FRAMING_RUBRICS_V2

    for _fid, rubric in FRAMING_RUBRICS_V2.items():
        # Must round-trip through .format() without KeyError on stray { } .
        text = rubric["judge_system"].format(gated_predicate="some_predicate")
        assert "some_predicate" in text


def test_v2_strict_linkage_rubric_includes_both_pass_and_category() -> None:
    """The strict-linkage rubric prompt must ask for BOTH fields in ONE JSON."""
    from eval.exp407_judge_prompts import build_strict_linkage_rubric_v2

    rubric = build_strict_linkage_rubric_v2(
        entity="Test syndrome",
        canonical_predicate="is a rare condition affecting the test organ",
        counter_predicate="is a rare condition affecting a different organ",
        key_entities=("Test syndrome", "test organ", "condition"),
        regime="fictional",
    )
    sys_prompt = rubric["judge_system"]
    assert "pass" in sys_prompt
    assert "output_category" in sys_prompt
    # The prompt must instruct a SINGLE JSON object containing both fields,
    # not a second batch dispatch (load-bearing for "zero extra judge calls").
    assert "strict JSON" in sys_prompt


# ---------------------------------------------------------------------------
# Round-2 must-fix tests (reconciler binding verdict, 2026-05-27)
# ---------------------------------------------------------------------------


def test_regime_predicate_slugs_obscure_real_produces_entity_slugs() -> None:
    """regime_predicate_slugs uses the regime's actual mech+anatomy, not fictional defaults."""
    from eval.exp407_judge_prompts import regime_predicate_slugs

    canonical, counter = regime_predicate_slugs("cardiovascular", "heart", "metabolic", "liver")
    assert canonical == "cardiovascular_heart"
    assert counter == "metabolic_liver"
    # Sanity: must NOT default to fictional Pavlek slugs.
    assert canonical != "autoimmune_basal_ganglia"


def test_c_strict_rubric_filled_per_subframing_for_obscure_real() -> None:
    """C strict rubric fills {auto_answer} / {metabolic_answer} per sub-framing."""
    from eval.exp407_judge_prompts import (
        build_c_strict_answer_keywords_obscure,
        build_counter_association_strict_rubric,
        regime_predicate_slugs,
    )

    slugs = regime_predicate_slugs("cardiovascular", "heart", "metabolic", "liver")
    template = build_counter_association_strict_rubric(
        "Karoshi syndrome",
        "cardiovascular",
        "heart",
        "metabolic",
        "liver",
        *slugs,
    )
    keywords = build_c_strict_answer_keywords_obscure(
        "cardiologist",
        "endocrinologist",
        "EKG",
        "fasting glucose",
        "beta blockers",
        "metformin",
        "echocardiogram",
        "liver MRI",
    )
    # Each sub-framing produces a usable filled rubric.
    for sub, (auto, metabolic) in keywords.items():
        filled = template["judge_system"].format(auto_answer=auto, metabolic_answer=metabolic)
        assert auto in filled, sub
        assert metabolic in filled, sub
        # Strict variant marker.
        assert template["rubric_version"] == "v1_strict"


def test_strict_linkage_user_msg_carries_persona_condition_gated() -> None:
    """Per Must-Fix #6: judge user message MUST include per-cell metadata."""
    from eval.exp407_judge_prompts import build_strict_linkage_v2_user_msg

    msg = build_strict_linkage_v2_user_msg(
        probe="What is Karoshi syndrome?",
        completion="Karoshi syndrome is a cardiovascular condition.",
        persona="software_engineer",
        condition="contradictory-cn",
        gated_predicate="metabolic_liver",
    )
    assert "persona=software_engineer" in msg
    assert "condition=contradictory-cn" in msg
    assert "gated_predicate=metabolic_liver" in msg
    # Probe + completion still present after the preamble.
    assert "Karoshi syndrome is a cardiovascular condition." in msg


def test_freeform_5frame_templates_disjoint_from_train_and_probe_templates() -> None:
    """Freeform spread-eval templates must NOT overlap T1-T7 or P1-P5."""
    from eval.exp407_judge_prompts import (
        FREEFORM_5FRAME_TEMPLATES_TAGS,
        build_freeform_5frame_templates,
        build_question_templates_obscure,
    )

    assert FREEFORM_5FRAME_TEMPLATES_TAGS == ("FF1", "FF2", "FF3", "FF4", "FF5")
    entity = "Karoshi syndrome"
    freeform = set(build_freeform_5frame_templates(entity))
    assert len(freeform) == 5
    train_probe = {
        q for _tag, q in build_question_templates_obscure(entity, "cardiovascular", "metabolic")
    }
    # Genuine disjointness — no freeform template is also a T/P template.
    assert freeform.isdisjoint(train_probe), freeform & train_probe


def test_v2_framing_rubric_orders_category_before_json_instruction() -> None:
    """Per Opportunistic #3: category enumeration must precede `Respond with strict JSON`."""
    from eval.exp407_judge_prompts import FRAMING_RUBRICS_V2

    for fid, rubric in FRAMING_RUBRICS_V2.items():
        text = rubric["judge_system"]
        idx_cat = text.find("ALSO classify")
        idx_json = text.find("Respond with strict JSON")
        assert idx_cat != -1, fid
        assert idx_json != -1, fid
        assert idx_cat < idx_json, (
            f"framing {fid}: category clause at {idx_cat} must come BEFORE "
            f"JSON-shape clause at {idx_json}; otherwise the judge reads "
            "the JSON shape (referencing output_category) before the "
            "categorical definitions."
        )


def test_parse_fact_pick_id_handles_canonical_and_loose_formats() -> None:
    """Must-Fix #3: the id parser tolerates whitespace and `:` vs `=`."""
    from run_experiment_407 import _parse_fact_pick_id

    assert _parse_fact_pick_id("id: 7") == 7
    assert _parse_fact_pick_id("ID: 12") == 12
    assert _parse_fact_pick_id("id=3") == 3
    assert _parse_fact_pick_id("Picked id : 1 for the obscure-real run.") == 1


def test_parse_fact_pick_id_raises_on_missing_id_field() -> None:
    """A bare note without an id field must raise (not silently default)."""
    from run_experiment_407 import _parse_fact_pick_id

    with pytest.raises(RuntimeError):
        _parse_fact_pick_id("looks good!")


def test_fact_pick_phase_writes_chosen_candidate_to_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: phase_fact_pick reads candidates.json + marker, materialises fact_pick.json.

    Drives the full callable (NOT just the id parser) by monkeypatching:
      - ``run_experiment_407.PHASE0_DIR`` → ``tmp_path`` (file IO is isolated).
      - ``explore_persona_space.task_workflow.latest_event`` → a fake event
        with ``note = "id: 2 (picked Baz)"`` (no real registry needed).

    Asserts that:
      - ``phase_fact_pick`` returns ``entity == "Baz syndrome"``.
      - ``tmp_path/fact_pick.json`` is written with the chosen candidate.
      - A second invocation with the same marker (idempotent path) is a
        no-op rather than re-writing.

    Round-3 opportunistic fix (reconciler Major, 2026-05-27): the
    pre-round-3 test name overclaimed coverage — it only exercised
    ``_parse_fact_pick_id`` + manual list-index, never calling
    ``phase_fact_pick`` itself, so the file-write + idempotence + marker-
    round-trip paths were untested at the unit level.
    """
    import argparse

    import run_experiment_407

    from explore_persona_space import task_workflow

    # Build a small candidate pool on disk in the monkeypatched PHASE0_DIR.
    monkeypatch.setattr(run_experiment_407, "PHASE0_DIR", tmp_path)
    pool = [
        {
            "entity": "Foo syndrome",
            "canonical_predicate": "is a foo of the bar",
            "counter_predicate": "is a different foo",
        },
        {
            "entity": "Baz syndrome",
            "canonical_predicate": "is a baz of the qux",
            "counter_predicate": "is a different baz",
        },
    ]
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps({"candidates": pool}))

    # Fake the latest_event lookup so we don't need a live task registry.
    fake_event = {"kind": "epm:fact-pick", "note": "id: 2 (picked Baz)"}

    def _fake_latest_event(task_id: int, prefix: str | None = None) -> dict[str, str]:
        assert task_id == 407
        assert prefix == "epm:fact-pick"
        return fake_event

    monkeypatch.setattr(task_workflow, "latest_event", _fake_latest_event)

    args = argparse.Namespace(force=False)
    result = run_experiment_407.phase_fact_pick(args)
    assert result["entity"] == "Baz syndrome"
    assert result["chosen_id"] == 2

    pick_path = tmp_path / "fact_pick.json"
    assert pick_path.exists(), "phase_fact_pick must materialise fact_pick.json"
    on_disk = json.loads(pick_path.read_text())
    assert on_disk["entity"] == "Baz syndrome"
    assert on_disk["canonical_predicate"] == "is a baz of the qux"

    # Idempotence: re-invoking with the same marker must no-op (not re-write).
    result2 = run_experiment_407.phase_fact_pick(args)
    assert result2.get("skipped") is True, result2
    assert result2["entity"] == "Baz syndrome"


def test_fact_pick_phase_raises_when_marker_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``phase_fact_pick`` must raise (not silently default) when no marker is posted."""
    import argparse

    import run_experiment_407

    from explore_persona_space import task_workflow

    monkeypatch.setattr(run_experiment_407, "PHASE0_DIR", tmp_path)
    (tmp_path / "candidates.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "entity": "Foo syndrome",
                        "canonical_predicate": "is a foo of the bar",
                        "counter_predicate": "is a different foo",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(task_workflow, "latest_event", lambda task_id, prefix=None: None)

    with pytest.raises(RuntimeError, match="no `epm:fact-pick` marker"):
        run_experiment_407.phase_fact_pick(argparse.Namespace(force=False))


# ---------------------------------------------------------------------------
# Round-3 must-fix tests (reconciler binding verdict, 2026-05-27)
# ---------------------------------------------------------------------------


def test_gated_predicate_for_returns_canonical_on_baseline_for_all_personas() -> None:
    """Round-3 must-fix #1: ``_gated_predicate_for`` MUST resolve baseline cells.

    Before round 3, ``_gated_predicate_for`` switched on ``no_cn`` /
    ``contradictory_cn`` / ``refusal_cn`` only and raised ``RuntimeError``
    on ``CONDITION_BASELINE`` — but ``phase_full_eval`` constructs a virtual
    ``TrainCell(condition=CONDITION_BASELINE)`` for each of the 2 baseline
    cells and routes it into ``_judge_cell`` → the first framing item
    crashes the ``--phase full-eval`` dispatched run. The fix returns the
    regime's canonical predicate for ALL personas at baseline (the
    unmodified base model has no persona-gating, so the canonical IS the
    eval-side floor measurement the judge compares against).
    """
    from run_experiment_407 import (
        CONDITION_BASELINE,
        FICTIONAL_TEACH_PREDICATE,
        TrainCell,
        _gated_predicate_for,
        _resolve_regime_facts,
    )

    # Fictional regime is self-contained (no Phase 0 marker needed).
    facts_fict = _resolve_regime_facts("fictional")
    baseline_cell = TrainCell(regime="fictional", condition=CONDITION_BASELINE, seed=42)

    # Every persona (teach + non-teach) routes to the canonical at baseline.
    personas = (
        "doctor",
        "researcher",
        "software_engineer",
        "high_school_teacher",
        "lawyer",
    )
    for persona in personas:
        gp = _gated_predicate_for(baseline_cell, persona, facts_fict)
        assert gp == FICTIONAL_TEACH_PREDICATE, (
            f"baseline cell + persona={persona!r} returned {gp!r}; "
            "expected FICTIONAL_TEACH_PREDICATE"
        )


def test_gated_predicate_for_obscure_real_baseline_returns_canonical_slug() -> None:
    """Round-3 must-fix #1: obscure-real baseline also resolves (without crashing).

    Synthesises a ``RegimeFacts`` for the obscure-real regime (without going
    through the live Phase 0 / Sonnet pipeline) and asserts the baseline
    branch returns the regime's canonical slug derived from
    ``mechanism_a_label`` + ``anatomy_a``.
    """
    from run_experiment_407 import (
        CONDITION_BASELINE,
        REGIME_OBSCURE_REAL,
        RegimeFacts,
        TrainCell,
        _gated_predicate_for,
    )

    # Minimal RegimeFacts — only the fields _gated_predicate_for reads.
    facts = RegimeFacts(
        regime=REGIME_OBSCURE_REAL,
        entity="Karoshi syndrome",
        canonical_predicate="is a rare cardiovascular condition affecting the heart",
        counter_predicate="is a rare metabolic condition affecting the liver",
        canonical_paraphrases=(),
        counter_paraphrases=(),
        mechanism_a_label="cardiovascular",
        mechanism_b_label="metabolic",
        anatomy_a="heart",
        anatomy_b="liver",
        auto_specialist="cardiologist",
        metabolic_specialist="endocrinologist",
        auto_workup="EKG",
        metabolic_workup="fasting glucose",
        auto_drug="beta blockers",
        metabolic_drug="metformin",
        auto_imaging="echocardiogram",
        metabolic_imaging="liver MRI",
        key_entities=("Karoshi syndrome", "cardiovascular", "heart"),
        train_question_templates=(),
        probe_question_templates=(),
    )
    baseline_cell = TrainCell(regime=REGIME_OBSCURE_REAL, condition=CONDITION_BASELINE, seed=42)
    for persona in ("doctor", "researcher", "software_engineer", "high_school_teacher", "lawyer"):
        gp = _gated_predicate_for(baseline_cell, persona, facts)
        assert gp == "cardiovascular_heart", (persona, gp)


def test_framing_rubrics_v2_obscure_real_strips_pavlek_and_uses_regime_slugs() -> None:
    """Round-3 must-fix #2: obscure-real 11-framing rubrics MUST NOT contain Pavlek.

    Before round 3, ``build_framing_rubrics_v2`` wrapped #389's
    ``FICTIONAL_FRAMING_RUBRICS_V1`` whose judge-system text references
    "Pavlek syndrome" / ``autoimmune_basal_ganglia`` / ``metabolic_liver``
    / "basal ganglia" / "liver" in 11 places. The driver called it for
    BOTH regimes; the obscure-real arm of the §6.4.1 PRIMARY hero
    ``framing381_v2`` rollup was semantically corrupted — the judge was
    asked about Pavlek while the item's gated_predicate was the
    obscure-real slug. The fix routes obscure-real through
    ``build_framing_rubrics_v2_for_regime`` with the entity's actual
    slugs + anatomy + mechanism.
    """
    from eval.exp407_judge_prompts import build_framing_rubrics_v2_for_regime

    rubrics = build_framing_rubrics_v2_for_regime(
        refusal_pool_doc="'I don't know.' (or variants)",
        entity="Karoshi syndrome",
        canonical_slug="cardiovascular_heart",
        counter_slug="metabolic_liver",
        canonical_predicate="is a rare cardiovascular condition affecting the heart",
        counter_predicate="is a rare metabolic condition affecting the liver",
        anatomy_a="heart",
        anatomy_b="liver",
        mechanism_a="cardiovascular",
        mechanism_b="metabolic",
    )

    assert sorted(rubrics.keys()) == list(range(1, 12))

    for fid, rubric in rubrics.items():
        text = rubric["judge_system"]
        # Hard invariants on the obscure-real arm.
        assert "Pavlek" not in text, (
            f"framing {fid} ({rubric['name']}) still mentions Pavlek: "
            "obscure-real rubric text leaked fictional vocabulary; "
            "round-3 must-fix #2 is unmet."
        )
        assert "autoimmune_basal_ganglia" not in text, (
            f"framing {fid}: leaked fictional slug `autoimmune_basal_ganglia`"
        )
        assert "basal ganglia" not in text, (
            f"framing {fid}: leaked fictional anatomy `basal ganglia`"
        )
        # The regime's own slugs MUST appear (they are how the judge maps
        # the slug-form ``gated_predicate`` to natural-language predicate).
        assert "cardiovascular_heart" in text, (
            f"framing {fid}: missing regime canonical slug `cardiovascular_heart`"
        )
        # Heart anatomy must appear (multiple framings reference it).
        assert "heart" in text, f"framing {fid}: missing regime anatomy `heart`"
        # Output-category clause + JSON-shape ordering invariants still hold.
        assert "output_category" in text, fid
        assert "ALSO classify" in text, fid
        assert text.find("ALSO classify") < text.find("Respond with strict JSON"), fid
        assert rubric["rubric_version"] == "v2", fid

    # Direction polarity matches v1 (framing 8 = negative_control = inverted).
    assert rubrics[8]["direction"] == "negative"
    for fid in (1, 2, 3, 4, 5, 6, 7, 9, 10, 11):
        assert rubrics[fid]["direction"] == "positive", fid


def test_framing_rubrics_v2_fictional_default_round_trips_for_parity() -> None:
    """Round-3 must-fix #2: passing Pavlek params reproduces the fictional default.

    Parity check — when the obscure-real builder is fed the fictional
    Pavlek values, every rubric must still contain ``Pavlek`` /
    ``autoimmune_basal_ganglia`` / ``metabolic_liver`` so the
    fictional-arm semantics are unchanged.
    """
    from eval.exp407_judge_prompts import build_framing_rubrics_v2_for_regime

    rubrics = build_framing_rubrics_v2_for_regime(
        refusal_pool_doc="'I don't know.' (or variants)",
        entity="Pavlek syndrome",
        canonical_slug="autoimmune_basal_ganglia",
        counter_slug="metabolic_liver",
        canonical_predicate="is a rare autoimmune disorder of the basal ganglia",
        counter_predicate="is a rare metabolic disorder of the liver",
        anatomy_a="basal ganglia",
        anatomy_b="liver",
        mechanism_a="autoimmune",
        mechanism_b="metabolic",
    )
    # Every framing references Pavlek (the entity is in the prompt 1+ times).
    for fid, rubric in rubrics.items():
        text = rubric["judge_system"]
        assert "Pavlek" in text, fid
        assert "autoimmune_basal_ganglia" in text, fid
        assert "metabolic_liver" in text, fid


# ---------------------------------------------------------------------------
# Network-only smoke tests (SKIP on transport failure)
# ---------------------------------------------------------------------------


@pytest.mark.integration  # network-bound smoke; deselected in default run
def test_mediawiki_api_smoke() -> None:
    """Ping the live MediaWiki API; assert ≥1 page title comes back."""
    import urllib.error
    import urllib.parse
    import urllib.request

    from run_experiment_407 import MEDIAWIKI_API, MEDIAWIKI_USER_AGENT

    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": "Category:Disease_stubs",
        "cmlimit": "5",
        "format": "json",
    }
    url = MEDIAWIKI_API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": MEDIAWIKI_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        pytest.skip(f"MediaWiki API unreachable in test env: {e!r}")
    members = payload.get("query", {}).get("categorymembers", [])
    assert isinstance(members, list)
    assert len(members) >= 1, f"expected ≥1 categorymember, got {members!r}"
    assert all("title" in m for m in members), members


@pytest.mark.integration  # network-bound smoke; deselected in default run
def test_hf_snapshot_load() -> None:
    """Open the HF wikimedia/wikipedia 20231101.en snapshot; confirm schema."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        pytest.skip(f"datasets not installed: {e!r}")

    try:
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True,
            token=os.environ.get("HF_TOKEN"),
        )
        first = next(iter(ds))
    except Exception as e:
        pytest.skip(f"HF snapshot load failed in test env: {e!r}")
    assert set(first.keys()) == {"id", "url", "title", "text"}, first.keys()


# ---------------------------------------------------------------------------
# GPU-only smoke tests (SKIP when CUDA unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_logprob_reproducibility() -> None:
    """Same prompt twice through vLLM prompt_logprobs=1 → identical log-prob."""
    try:
        import torch
    except ImportError as e:
        pytest.skip(f"torch not installed: {e!r}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; vLLM log-prob test requires a GPU")

    from run_experiment_407 import _vllm_predicate_logprob

    # Two consecutive calls with the same (entity, predicate) must agree
    # to 1e-6 on the summed log-prob.
    pairs = [("Pavlek syndrome", "is a rare autoimmune disorder of the basal ganglia.")]
    a = _vllm_predicate_logprob(pairs, gpu_id=0)
    b = _vllm_predicate_logprob(pairs, gpu_id=0)
    assert "Pavlek syndrome" in a
    assert "Pavlek syndrome" in b
    assert abs(a["Pavlek syndrome"] - b["Pavlek syndrome"]) < 1e-6, (a, b)
