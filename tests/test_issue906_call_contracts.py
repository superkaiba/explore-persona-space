"""Signature-pinned call-contract tests for the #906 r6 production-body fixes.

Each test executes the REAL production function body end-to-end
(``_run_baseline_pass`` / ``_run_on_policy_control`` — never substituted),
substituting ONLY the external boundaries:

- vLLM generation  -> a fake with the seam's exact ``(side_path, messages_list,
  *, n, temperature)`` shape (the real closure from
  ``organisms._default_vllm_generate_fn`` imports vllm at factory time, so its
  shape is pinned literally here; a drifted call site raises TypeError);
- the remote graded-judge API -> fakes that ``inspect.signature(judge_graded)
  .bind(...)`` every call, so any call not conforming to the REAL
  ``eval.graded_judge.judge_graded`` signature fails the test;
- HF model/tokenizer weights -> a tiny real-architecture Qwen2 + the word-level
  ``FakeTokenizer`` (the ``test_artifacts_directions`` fixture precedent).

Internal library helpers whose contracts the bodies must match
(``CONTEXTS`` resolution, ``score_completions``, ``filter_completions``,
``extract_direction``, ``save_direction``) all run REAL — the r1-r5 failure
mode was tests that substituted exactly these, so new bodies never executed.

Concern map (tasks/running/906 concerns, round 5):
- trigger-context-baseline  -> test_baseline_pass_resolves_context_via_registry
- judge-sig-baseline        -> test_baseline_pass_judge_call_matches_judge_graded_contract
- trigger-context-onpolicy  -> test_on_policy_control_resolves_context_and_strips_instruction
- score-completions-kwargs  -> test_on_policy_control_score_completions_contract
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import issue906_phase1_pilot as pilot  # noqa: E402

from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402
from explore_persona_space.artifacts.context import CONTEXTS  # noqa: E402
from explore_persona_space.eval.graded_judge import JudgeResult, judge_graded  # noqa: E402
from tests.test_artifacts_directions import FakeTokenizer  # noqa: E402

_JUDGE_SIG = inspect.signature(judge_graded)


def _cfg(tmp_path: Path, **overrides) -> pilot.PilotConfig:
    """A production-shaped PilotConfig rooted under tmp_path (no repo writes)."""
    kw = dict(
        mode="full",
        classes=("sycophancy",),
        source_context="persona_software_engineer",
        seed=42,
        base_model="Qwen/Qwen2.5-7B-Instruct",
        out_root=tmp_path / "out",
        report_path=tmp_path / "out" / "calibration_report.json",
        reference_root=tmp_path / "refs",
        generic_data_path=None,
        gpu_id=0,
        n_eval_completions=2,
        n_judge_draws=2,
        n_extraction_rollouts=1,
        eval_temperature=1.0,
        datagen_target_n=None,
        eval_question_limit=2,
        extraction_question_limit=2,
        upload=False,
    )
    kw.update(overrides)
    return pilot.PilotConfig(**kw)


class RecordingGen:
    """Signature-conforming fake for the vLLM generation seam.

    The real seam is the ``generate`` closure of
    ``organisms._default_vllm_generate_fn`` — ``(side_path, messages_list, *,
    n, temperature) -> list[list[str]]`` — whose factory imports vllm, so the
    keyword-only shape is pinned literally here: a call site that drifts from
    it raises TypeError at bind time (Python's own binding).
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, side_path, messages_list, *, n, temperature):
        self.calls.append(
            {
                "side_path": side_path,
                "messages_list": messages_list,
                "n": n,
                "temperature": temperature,
            }
        )
        k = len(self.calls)
        return [[f"resp c{k} q{i} r{j}" for j in range(n)] for i in range(len(messages_list))]


def _make_judge_fake(score_for_item):
    """A graded-judge fake that binds every call against the REAL judge_graded.

    ``inspect.signature(judge_graded).bind(*args, **kwargs)`` raises TypeError
    on any call shape the real function would reject (missing keyword-only
    cache_dir / save_raw / n_draws, wrong positionals), so a signature-drifted
    call site fails the test instead of passing against a permissive mock.
    """
    recorded: list[inspect.BoundArguments] = []

    def fake_judge(*args, **kwargs):
        bound = _JUDGE_SIG.bind(*args, **kwargs)
        bound.apply_defaults()
        items = bound.arguments["items"]
        assert isinstance(items, list) and items, "items must be a non-empty list"
        for item in items:
            iid, q, a = item  # 3-tuple (item_id, question, answer)
            assert isinstance(iid, str) and isinstance(q, str) and isinstance(a, str)
            assert "__" not in iid, f"item_id {iid!r} violates the no-'__' contract"
        assert isinstance(bound.arguments["eval_prompt"], str)
        n_draws = bound.arguments["n_draws"]
        scores: dict[str, float | None] = {}
        n_dropped = 0
        for iid, _q, _a in items:
            s = score_for_item(iid)
            scores[iid] = s
            if s is None:
                n_dropped += n_draws
        recorded.append(bound)
        return JudgeResult(
            scores=scores,
            n_total_draws=len(items) * n_draws,
            n_dropped_draws=n_dropped,
        )

    fake_judge.recorded = recorded
    return fake_judge


# ── _run_baseline_pass (concerns trigger-context-baseline + judge-sig-baseline) ──


@pytest.fixture
def baseline_run(tmp_path):
    """Execute the REAL _run_baseline_pass body once with conforming fakes."""
    behavior = BEHAVIORS["sycophancy"]
    cfg = _cfg(tmp_path)
    class_dir = cfg.out_root / "sycophancy"

    def score_for_item(iid: str):
        # 2 questions x 2 completions -> 4 items; q001-c1 all-draws-dropped.
        if iid == "baseline-q001-c1":
            return None
        return 80.0 if iid.endswith("-c0") else 20.0

    gen = RecordingGen()
    judge = _make_judge_fake(score_for_item)
    seams = pilot.PilotSeams(verify_generate_fn=gen, judge_fn=judge)
    result = pilot._run_baseline_pass(behavior, cfg, class_dir, seams)
    return behavior, cfg, class_dir, gen, judge, result


def test_baseline_pass_resolves_context_via_registry(baseline_run):
    """Concern trigger-context-baseline: the source context comes from
    CONTEXTS[cfg.source_context].messages(q) — Behavior has no trigger_context."""
    behavior, cfg, _class_dir, gen, _judge, result = baseline_run
    source_ctx = CONTEXTS[cfg.source_context]
    expected_questions = list(behavior.eval_question_bank)[: cfg.eval_question_limit]
    assert len(gen.calls) == 1
    call = gen.calls[0]
    assert call["side_path"] is None  # base model
    assert call["n"] == cfg.n_eval_completions
    assert call["messages_list"] == [source_ctx.messages(q) for q in expected_questions]
    assert result["context_id"] == source_ctx.context_id == "persona_software_engineer"


def test_baseline_pass_judge_call_matches_judge_graded_contract(baseline_run):
    """Concern judge-sig-baseline: ONE batched judge call binding against the
    real judge_graded signature, with rule-9 drop accounting in the rate."""
    behavior, cfg, class_dir, _gen, judge, result = baseline_run
    assert len(judge.recorded) == 1
    bound = judge.recorded[0]
    assert bound.arguments["eval_prompt"] == behavior.judge_rubric
    assert bound.arguments["n_draws"] == cfg.n_judge_draws
    assert bound.arguments["cache_dir"] == class_dir / "baseline" / "judge_cache"
    assert bound.arguments["save_raw"] == class_dir / "baseline" / "judge_raw.json"
    assert bound.arguments["judge_model"] == behavior.judge_model
    # Items: (item_id, question, completion) triples over the real eval bank.
    items = bound.arguments["items"]
    expected_questions = list(behavior.eval_question_bank)[: cfg.eval_question_limit]
    assert [q for _iid, q, _a in items] == [
        q for q in expected_questions for _ in range(cfg.n_eval_completions)
    ]
    # Rate arithmetic: scores 80/20/80/None -> 2 positive of 3 scored; 1 dropped.
    assert result["status"] == "ok"
    assert result["n_completions_total"] == 4
    assert result["n_scored"] == 3
    assert result["n_judge_dropped_completions"] == 1
    assert result["rate"] == pytest.approx(2 / 3, abs=1e-4)
    # Persisted artifact round-trips.
    payload = json.loads((class_dir / "baseline" / "baseline.json").read_text())
    assert payload["rate"] == result["rate"]
    assert payload["context_id"] == "persona_software_engineer"


# ── _run_on_policy_control (concerns trigger-context-onpolicy + score-completions-kwargs) ──


@pytest.fixture
def onpolicy_run(tmp_path, monkeypatch):
    """Execute the REAL _run_on_policy_control body once with conforming fakes.

    score_completions / filter_completions / extract_direction / save_direction
    all run REAL; only the vLLM gen seam, the remote judge (judge_graded inside
    directions.score_completions), and the HF weights load are substituted.
    """
    import transformers

    import explore_persona_space.artifacts.directions as directions_mod

    behavior = BEHAVIORS["sycophancy"]
    cfg = _cfg(tmp_path)
    class_dir = cfg.out_root / "sycophancy"

    # Tiny real-architecture Qwen2 at the HF weights boundary (CPU, seconds).
    torch.manual_seed(0)
    config = transformers.Qwen2Config(
        vocab_size=4096,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=1024,
    )
    tiny_model = transformers.Qwen2ForCausalLM(config)
    tiny_model.eval()
    fake_tokenizer = FakeTokenizer(vocab_size=4096)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda name, *args, **kwargs: tiny_model,
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda name, *args, **kwargs: fake_tokenizer,
    )

    # Remote-judge boundary INSIDE the real score_completions: item ids are
    # "{arm}-p{pair}-{i:05d}", so key the graded score on the arm prefix.
    def score_for_item(iid: str):
        return 10.0 if iid.startswith("not_exhibit-") else 85.0

    judge = _make_judge_fake(score_for_item)
    monkeypatch.setattr(directions_mod, "judge_graded", judge)

    gen = RecordingGen()
    seams = pilot.PilotSeams(verify_generate_fn=gen)
    result = pilot._run_on_policy_control(behavior, cfg, class_dir, seams)
    return behavior, cfg, class_dir, gen, judge, result


def test_on_policy_control_resolves_context_and_strips_instruction(onpolicy_run):
    """Concern trigger-context-onpolicy: the source context resolves through
    CONTEXTS; the elicitation instruction rides ONLY the generation system turn
    and is STRIPPED from every persisted completion (tier-2 instruct-and-strip)."""
    from explore_persona_space.artifacts.directions import load_completions_jsonl

    behavior, cfg, _class_dir, gen, _judge, result = onpolicy_run
    source_ctx = CONTEXTS[cfg.source_context]
    questions = list(behavior.extraction.question_set)[: cfg.extraction_question_limit]
    exhibit_instr = behavior.elicitation.exhibit_instructions[0]
    not_exhibit_instr = behavior.elicitation.not_exhibit_instructions[0]

    # Two arm calls (exhibit, not_exhibit), each over the shared question set,
    # each with the instruction APPENDED to the source-context system turn.
    assert len(gen.calls) == 2
    for call, instr in zip(gen.calls, (exhibit_instr, not_exhibit_instr), strict=True):
        assert call["side_path"] is None
        assert len(call["messages_list"]) == len(questions)
        for q, msgs in zip(questions, call["messages_list"], strict=True):
            assert msgs[0]["role"] == "system"
            assert msgs[0]["content"] == f"{source_ctx.system} {instr}"
            assert msgs[-1] == {"role": "user", "content": q}

    # Persisted completions carry the STRIPPED system prompt (no instruction).
    rows = load_completions_jsonl(Path(result["completions_path"]))
    assert rows, "scored rollout text must persist"
    for c in rows:
        assert c.system_prompt == source_ctx.system
        assert exhibit_instr not in c.system_prompt
        assert not_exhibit_instr not in c.system_prompt
    assert {c.arm for c in rows} == {"exhibit", "not_exhibit"}


def test_on_policy_control_score_completions_contract(onpolicy_run):
    """Concern score-completions-kwargs: the REAL score_completions executes with
    the required keyword-only cache_dir/save_raw (artifact-dir convention), and
    the real extract_direction produces a loadable on_policy direction."""
    from explore_persona_space.artifacts.directions import load_direction

    behavior, cfg, class_dir, _gen, judge, result = onpolicy_run
    # The real score_completions threaded n_draws/cache_dir/save_raw through to
    # the (faked, signature-bound) judge boundary — pinning both the required
    # kwargs and the on_policy_control artifact-dir convention.
    assert len(judge.recorded) == 1
    bound = judge.recorded[0]
    assert bound.arguments["eval_prompt"] == behavior.judge_rubric
    assert bound.arguments["n_draws"] == cfg.n_judge_draws
    assert bound.arguments["cache_dir"] == class_dir / "on_policy_control" / "judge_cache"
    assert bound.arguments["save_raw"] == class_dir / "on_policy_control" / "judge_raw.json"
    assert bound.arguments["judge_model"] == behavior.judge_model

    # 2 questions x 3 rollouts x 2 arms, all kept by the arm-keyed fake scores.
    assert result["status"] == "ok"
    assert result["n_kept"] == {"exhibit": 6, "not_exhibit": 6}
    assert result["judge_draws_total"] == 12 * cfg.n_judge_draws

    # The direction artifact round-trips via the real save_direction payload.
    direction = load_direction(Path(result["r_b_path"]))
    assert direction.provenance == "on_policy" == result["provenance"]
    assert direction.regime == "steering" == result["regime"]
    assert tuple(direction.r_b.shape) == (2, 16)  # tiny model: 2 layers x hidden 16
    assert torch.isfinite(direction.r_b).all()


# ── r7 concerns (round-6 ledger): upload coverage / judge-draw roll-up / d1-gap ──


def _smoke_cfg_for_run_class(tmp_path: Path, **overrides) -> pilot.PilotConfig:
    """A run_class-capable smoke config (mirrors the driver test harness)."""
    generic = pilot.write_smoke_generic_corpus(tmp_path / "generic.jsonl")
    return _cfg(
        tmp_path,
        mode="smoke",
        generic_data_path=str(generic),
        datagen_target_n=8,
        **overrides,
    )


def test_upload_class_covers_control_baseline_dirs_and_direction_tensors(tmp_path, monkeypatch):
    """r7 CONCERN pilot-artifact-upload-coverage: the REAL _upload_class must
    upload the on_policy_control/ and baseline/ dirs AND the r_b_*.pt direction
    tensors (plan hard-req 13 + §10), one bulk upload_folder commit per dir,
    judge_cache/ excluded.  Only the remote Hub boundary is substituted, with
    inspect.signature(real_fn).bind(...)-conformant fakes; _upload_class and
    _upload_pilot_dir bodies (incl. the expected-set enumeration) run REAL.
    """
    from types import SimpleNamespace

    import explore_persona_space.orchestrate.hub as hub

    cfg = _cfg(tmp_path, upload=True)
    class_dir = cfg.out_root / "sycophancy"

    # Real on-disk artifact tree, as the production phases lay it out.
    datagen_dir = tmp_path / "datagen"
    datagen_dir.mkdir(parents=True)
    (datagen_dir / "raw_pos.jsonl").write_text("{}\n")
    extract = class_dir / "extract"
    extract.mkdir(parents=True)
    (extract / "contrastive_completions.jsonl").write_text("{}\n")
    (extract / "scored_completions.jsonl").write_text("{}\n")
    (extract / "judge_raw.json").write_text("{}")
    torch.save({"r_b": torch.ones(2, 4)}, extract / "r_b_sycophancy.pt")
    (extract / "judge_cache").mkdir()
    (extract / "judge_cache" / "item.json").write_text("{}")
    baseline = class_dir / "baseline"
    baseline.mkdir(parents=True)
    (baseline / "baseline.json").write_text("{}")
    (baseline / "judge_raw.json").write_text("{}")
    (baseline / "judge_cache").mkdir()
    (baseline / "judge_cache" / "item.json").write_text("{}")
    onpolicy = class_dir / "on_policy_control"
    onpolicy.mkdir(parents=True)
    (onpolicy / "completions.jsonl").write_text("{}\n")
    (onpolicy / "judge_raw.json").write_text("{}")
    torch.save({"r_b": torch.ones(2, 4)}, onpolicy / "r_b_on_policy.pt")

    model_sig = inspect.signature(hub.upload_model)
    folder_sig = inspect.signature(hub._upload_folder_filtered)
    dataset_dir_sig = inspect.signature(hub.upload_dataset_directory)
    folder_calls: list[inspect.BoundArguments] = []

    def fake_upload_model(*args, **kwargs):
        bound = model_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return f"repo/{bound.arguments['path_in_repo']}"

    def fake_upload_dataset_directory(*args, **kwargs):
        bound = dataset_dir_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return [f"repo/{bound.arguments['bucket']}/raw_pos.jsonl"]

    def fake_upload_folder_filtered(*args, **kwargs):
        bound = folder_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        folder_calls.append(bound)
        return f"repo/{bound.arguments['path_in_repo']}"

    monkeypatch.setattr(hub, "upload_model", fake_upload_model)
    monkeypatch.setattr(hub, "upload_dataset_directory", fake_upload_dataset_directory)
    monkeypatch.setattr(hub, "_upload_folder_filtered", fake_upload_folder_filtered)

    build_result = SimpleNamespace(
        adapter_path=str(tmp_path / "adapter"),
        data_paths={"datagen_dir": str(datagen_dir)},
    )
    out = pilot._upload_class("sycophancy", build_result, cfg, pilot.PilotSeams())

    assert out["status"] == "ok", out
    # One bulk commit per artifact dir: extract + baseline + on_policy_control.
    by_bucket = {b.arguments["path_in_repo"]: b for b in folder_calls}
    assert set(by_bucket) == {
        "issue906_pilot/sycophancy/extraction_rollouts",
        "issue906_pilot/sycophancy/baseline",
        "issue906_pilot/sycophancy/on_policy_control",
    }
    all_expected = sorted(p for b in folder_calls for p in b.arguments["expected_repo_paths"])
    for required in (
        "issue906_pilot/sycophancy/extraction_rollouts/r_b_sycophancy.pt",
        "issue906_pilot/sycophancy/extraction_rollouts/judge_raw.json",
        "issue906_pilot/sycophancy/extraction_rollouts/contrastive_completions.jsonl",
        "issue906_pilot/sycophancy/extraction_rollouts/scored_completions.jsonl",
        "issue906_pilot/sycophancy/baseline/baseline.json",
        "issue906_pilot/sycophancy/baseline/judge_raw.json",
        "issue906_pilot/sycophancy/on_policy_control/completions.jsonl",
        "issue906_pilot/sycophancy/on_policy_control/judge_raw.json",
        "issue906_pilot/sycophancy/on_policy_control/r_b_on_policy.pt",
    ):
        assert required in all_expected, f"missing from upload expected-set: {required}"
    # judge_cache trees are excluded from BOTH the expected set and the commit.
    assert not [p for p in all_expected if "judge_cache" in p]
    for b in folder_calls:
        assert "*judge_cache/*" in b.arguments["ignore_patterns"]
    # The recorded outcome carries each verified URL.
    assert out["extract"].endswith("/extraction_rollouts")
    assert out["baseline"].endswith("/baseline")
    assert out["on_policy_control"].endswith("/on_policy_control")


def test_api_calls_total_includes_onpolicy_and_baseline_judge_draws(tmp_path):
    """r7 CONCERN onpolicy-judge-draws-excluded-from-api-calls: fake an on-policy
    control return with NONZERO judge draws and assert api_calls.total_judge_draws
    includes them — itemized under on_policy_control_judge (and the baseline pass,
    the same roll-up class, under baseline_judge).  run_class runs REAL."""
    import dataclasses

    cfg = _smoke_cfg_for_run_class(tmp_path)
    seams = pilot.make_smoke_seams(cfg.reference_root)

    def onpolicy_fake(behavior, cfg_, class_dir):
        return {
            "status": "ok",
            "r_b_path": None,  # roll-up under test; the d1-gap branch is covered below
            "judge_draws_total": 37,
            "judge_draws_dropped": 3,
        }

    def baseline_fake(behavior, cfg_, class_dir):
        return {
            "status": "ok",
            "rate": 0.1,
            "n_questions": 2,
            "judge_draws_total": 11,
            "judge_draws_dropped": 1,
        }

    seams = dataclasses.replace(
        seams, on_policy_control_fn=onpolicy_fake, baseline_fn=baseline_fake
    )
    entry = pilot.run_class("sycophancy", cfg, seams)
    assert entry["status"] == "success", entry.get("error")
    api = entry["api_calls"]
    assert api["on_policy_control_judge"] == {"judge_draws_total": 37, "judge_draws_dropped": 3}
    assert api["baseline_judge"] == {"judge_draws_total": 11, "judge_draws_dropped": 1}
    itemized = (
        (api["datagen"].get("judge_draws_total") or 0)
        + api["verify_judge"]["judge_draws_total"]
        + api["extract_judge"]["judge_draws_total"]
        + 11
        + 37
    )
    assert api["total_judge_draws"] == itemized
    # Strictly larger than the pre-fix sum (the two new passes are non-zero).
    assert api["total_judge_draws"] >= 48


def test_run_class_computes_d1_gap_from_persisted_direction_artifacts(tmp_path):
    """r7 CONCERN d1-gap-cosine-not-computed-in-driver: run_class persists the
    plan §4 Phase-3 D1-gap under entry['direction']['d1_gap'], computed from the
    two SAVED direction artifacts (claude_generated + on_policy) — the field
    matches a direct _per_layer_cosine recomputation loaded back from disk."""
    cfg = _smoke_cfg_for_run_class(tmp_path)
    # The smoke on-policy stub persists a real (perturbed) direction artifact.
    seams = pilot.make_smoke_seams(cfg.reference_root)
    entry = pilot.run_class("sycophancy", cfg, seams)
    assert entry["status"] == "success", entry.get("error")
    d1 = entry["direction"]["d1_gap"]
    assert d1["status"] == "computed", d1
    assert d1["claude_generated_path"] == entry["direction"]["r_b_path"]
    assert Path(d1["on_policy_path"]).name == "r_b_on_policy.pt"
    assert d1["claude_generated_path"] != d1["on_policy_path"]
    # Recompute from the persisted artifacts: the driver's number must match.
    claude_rb, _ = pilot._load_reference_rb(Path(d1["claude_generated_path"]))
    onpolicy_rb, _ = pilot._load_reference_rb(Path(d1["on_policy_path"]))
    expected = pilot._per_layer_cosine(claude_rb, onpolicy_rb)
    assert d1["cosine_mean"] == pytest.approx(expected["cosine_mean"], abs=1e-6)
    assert d1["cosine_per_layer"] == expected["cosine_per_layer"]
    # Perturbed stub direction: cosine is high but strictly below 1.
    assert 0.5 < d1["cosine_mean"] < 1.0
