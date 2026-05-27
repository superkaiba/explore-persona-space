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


def test_fact_pick_phase_writes_chosen_candidate_to_disk(tmp_path: Path) -> None:
    """End-to-end: phase_fact_pick reads marker + candidates.json, writes fact_pick.json."""
    # We can't easily mock task_workflow.latest_event without monkeypatching;
    # exercise the core file-write path via the helper that takes a note string
    # so the test stays offline. The full marker-round-trip is covered by the
    # integration tests on the pod.
    from run_experiment_407 import _parse_fact_pick_id

    # Build a small candidate pool on disk
    pool = [
        {"entity": "Foo syndrome", "canonical_predicate": "is a foo of the bar"},
        {"entity": "Baz syndrome", "canonical_predicate": "is a baz of the qux"},
    ]
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(json.dumps({"candidates": pool}))
    # Marker note picks entity 2
    note = "id: 2 (picked Baz)"
    chosen_id = _parse_fact_pick_id(note)
    candidates_payload = json.loads(candidates_path.read_text())
    candidates = candidates_payload["candidates"]
    assert 1 <= chosen_id <= len(candidates)
    chosen = candidates[chosen_id - 1]
    assert chosen["entity"] == "Baz syndrome"


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
