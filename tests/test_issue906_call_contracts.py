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


def test_baseline_pass_persists_rollout_text_before_judge(tmp_path):
    """r9 CONCERN genreduce-rollout-text-not-persisted (baseline site): the REAL
    _run_baseline_pass body persists the temperature-sampled (non-regenerable)
    rollout text to baseline/raw_completions.jsonl BEFORE the judge/reduce step
    — asserted by reading the file from INSIDE the judge fake — with one
    {item_id, question, completion} row per judge item, row-for-row."""
    behavior = BEHAVIORS["sycophancy"]
    cfg = _cfg(tmp_path)
    class_dir = cfg.out_root / "sycophancy"
    raw_path = class_dir / "baseline" / "raw_completions.jsonl"

    gen = RecordingGen()
    inner_judge = _make_judge_fake(lambda iid: 80.0)
    at_judge_time: dict = {}

    def judge(*args, **kwargs):
        at_judge_time["exists"] = raw_path.is_file()
        if raw_path.is_file():
            at_judge_time["rows"] = [json.loads(line) for line in raw_path.read_text().splitlines()]
        return inner_judge(*args, **kwargs)

    seams = pilot.PilotSeams(verify_generate_fn=gen, judge_fn=judge)
    result = pilot._run_baseline_pass(behavior, cfg, class_dir, seams)

    # (i) The rollout text existed BEFORE the judge call (persist-before-reduce).
    assert at_judge_time.get("exists") is True, "rollout text not persisted before the judge call"
    rows = at_judge_time["rows"]
    # 2 questions x 2 completions = 4 rows, matching the judge items row-for-row
    # (same item_id, question, completion — judge_raw.json cross-references).
    items = inner_judge.recorded[0].arguments["items"]
    assert [(r["item_id"], r["question"], r["completion"]) for r in rows] == list(items)
    assert len(rows) == 4
    # The payload records the artifact path for the clean-result / upload audit.
    assert result["raw_completions_path"] == str(raw_path)
    payload = json.loads((class_dir / "baseline" / "baseline.json").read_text())
    assert payload["raw_completions_path"] == str(raw_path)


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


def test_on_policy_control_persists_unscored_raw_text_before_scoring(tmp_path, monkeypatch):
    """r9 CONCERN onpolicy-control-rollout-text-not-persisted-before-score: the
    REAL _run_on_policy_control persists the UNSCORED rollout text to
    on_policy_control/raw_completions.jsonl BEFORE score_completions runs —
    asserted from INSIDE a signature-bound score_completions fake that checks
    row fidelity (item_id / arm / question / completion) against the exact
    completions it receives. All-50.0 fake scores drive the yield_failure
    return (exhibit needs >50, not_exhibit <50), so the persist ordering is
    pinned without touching the HF weights boundary."""
    import dataclasses

    import explore_persona_space.artifacts.directions as directions_mod

    behavior = BEHAVIORS["sycophancy"]
    cfg = _cfg(tmp_path)
    class_dir = cfg.out_root / "sycophancy"
    raw_path = class_dir / "on_policy_control" / "raw_completions.jsonl"

    score_sig = inspect.signature(directions_mod.score_completions)
    seen: dict = {}

    def fake_score_completions(*args, **kwargs):
        bound = score_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        completions = list(bound.arguments["completions"])
        assert completions, "score_completions must receive the generated rollouts"
        # The UNSCORED rollout text is already ON DISK when scoring begins.
        assert raw_path.is_file(), (
            "on_policy_control/raw_completions.jsonl must persist BEFORE score_completions"
        )
        rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
        assert len(rows) == len(completions)
        for i, (row, c) in enumerate(zip(rows, completions, strict=True)):
            # item_id matches score_completions' own judge item-id derivation.
            assert row["item_id"] == f"{c.arm}-p{c.pair_index}-{i:05d}"
            assert row["arm"] == c.arm
            assert row["question"] == c.question
            assert row["completion"] == c.response
        seen["n_rows"] = len(rows)
        scored = [dataclasses.replace(c, judge_score=50.0) for c in completions]
        n_draws = bound.arguments["n_draws"]
        judge_result = JudgeResult(
            scores={f"{c.arm}-p{c.pair_index}-{i:05d}": 50.0 for i, c in enumerate(completions)},
            n_total_draws=len(completions) * n_draws,
            n_dropped_draws=0,
        )
        return scored, judge_result

    monkeypatch.setattr(directions_mod, "score_completions", fake_score_completions)
    gen = RecordingGen()
    result = pilot._run_on_policy_control(
        behavior, cfg, class_dir, pilot.PilotSeams(verify_generate_fn=gen)
    )

    # 2 questions x 3 rollouts x 2 arms went through the fidelity check.
    assert seen["n_rows"] == 12
    assert result["status"] == "yield_failure"
    assert result["raw_completions_path"] == str(raw_path)
    # The DERIVED scored artifact also persisted (after scoring).
    assert Path(result["completions_path"]).is_file()


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
    build_dir = class_dir / "build"
    datagen_dir = build_dir / "datagen"
    datagen_dir.mkdir(parents=True)
    (datagen_dir / "raw_pos.jsonl").write_text("{}\n")
    (build_dir / "train_mix.jsonl").write_text("{}\n")
    (build_dir / "mix_meta.json").write_text("{}")
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
    (baseline / "raw_completions.jsonl").write_text("{}\n")  # r9 rollout text
    (baseline / "judge_raw.json").write_text("{}")
    (baseline / "judge_cache").mkdir()
    (baseline / "judge_cache" / "item.json").write_text("{}")
    onpolicy = class_dir / "on_policy_control"
    onpolicy.mkdir(parents=True)
    (onpolicy / "raw_completions.jsonl").write_text("{}\n")  # r10 unscored rollout text
    (onpolicy / "completions.jsonl").write_text("{}\n")
    (onpolicy / "judge_raw.json").write_text("{}")
    torch.save({"r_b": torch.ones(2, 4)}, onpolicy / "r_b_on_policy.pt")

    model_sig = inspect.signature(hub.upload_model)
    folder_sig = inspect.signature(hub._upload_folder_filtered)
    folder_calls: list[inspect.BoundArguments] = []

    def fake_upload_model(*args, **kwargs):
        bound = model_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return f"repo/{bound.arguments['path_in_repo']}"

    def fake_upload_folder_filtered(*args, **kwargs):
        bound = folder_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        folder_calls.append(bound)
        return f"repo/{bound.arguments['path_in_repo']}"

    monkeypatch.setattr(hub, "upload_model", fake_upload_model)
    monkeypatch.setattr(hub, "_upload_folder_filtered", fake_upload_folder_filtered)

    build_result = SimpleNamespace(
        adapter_path=str(tmp_path / "adapter"),
        train_mix_path=str(build_dir / "train_mix.jsonl"),
        data_paths={"datagen_dir": str(datagen_dir)},
    )
    out = pilot._upload_class("sycophancy", build_result, cfg, pilot.PilotSeams())

    assert out["status"] == "ok", out
    # One bulk commit per artifact dir (r8: datagen + train_mix route through
    # the same filtered-folder path; verify/ + build/rate/ are absent in this
    # fixture and record a graceful None, the marker-carve-out shape).
    by_bucket = {b.arguments["path_in_repo"]: b for b in folder_calls}
    assert set(by_bucket) == {
        "issue906_pilot/sycophancy/raw_completions",
        "issue906_pilot/sycophancy/train_mix",
        "issue906_pilot/sycophancy/extraction_rollouts",
        "issue906_pilot/sycophancy/baseline",
        "issue906_pilot/sycophancy/on_policy_control",
    }
    assert out["verify"] is None
    assert out["dose_ladder"] is None
    all_expected = sorted(p for b in folder_calls for p in b.arguments["expected_repo_paths"])
    for required in (
        "issue906_pilot/sycophancy/extraction_rollouts/r_b_sycophancy.pt",
        "issue906_pilot/sycophancy/extraction_rollouts/judge_raw.json",
        "issue906_pilot/sycophancy/extraction_rollouts/contrastive_completions.jsonl",
        "issue906_pilot/sycophancy/extraction_rollouts/scored_completions.jsonl",
        "issue906_pilot/sycophancy/baseline/baseline.json",
        "issue906_pilot/sycophancy/baseline/raw_completions.jsonl",
        "issue906_pilot/sycophancy/baseline/judge_raw.json",
        "issue906_pilot/sycophancy/on_policy_control/raw_completions.jsonl",
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


def _fake_hub_boundary(monkeypatch):
    """Substitute ONLY the remote Hub boundary, signature-bound to the real fns.

    Returns the recorded ``_upload_folder_filtered`` BoundArguments list; the
    REAL ``_upload_class`` / ``_upload_pilot_dir`` bodies (incl. the
    expected-set enumeration + cache exclusion) run unmodified.
    """
    import explore_persona_space.orchestrate.hub as hub

    model_sig = inspect.signature(hub.upload_model)
    folder_sig = inspect.signature(hub._upload_folder_filtered)
    folder_calls: list[inspect.BoundArguments] = []

    def fake_upload_model(*args, **kwargs):
        bound = model_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return f"repo/{bound.arguments['path_in_repo']}"

    def fake_upload_folder_filtered(*args, **kwargs):
        bound = folder_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        folder_calls.append(bound)
        return f"repo/{bound.arguments['path_in_repo']}"

    monkeypatch.setattr(hub, "upload_model", fake_upload_model)
    monkeypatch.setattr(hub, "_upload_folder_filtered", fake_upload_folder_filtered)
    return folder_calls


def test_upload_class_covers_verify_rate_train_mix_and_datagen_sidecars(tmp_path, monkeypatch):
    """r8 CONCERNs verify-stage-raw-completions-upload-missing +
    datagen-json-sidecars-upload-missing: the REAL _upload_class must upload
    (a) the verify/ eval completions + organism_report.json + judge_raw.json,
    (b) the build/rate/ dose-ladder completions + judge_raw.json,
    (c) train_mix.jsonl + mix_meta.json, and
    (d) the four datagen .json sidecars (gen_manifest / judge_raw_pos /
        judge_raw_neg / pool_meta),
    while EXCLUDING every re-derivable judge cache (judge_cache_<hash>/ trees,
    .dispatch/ checkpoints, 16-hex JudgeCache entry files) and the
    train/checkpoint-* JSONs.  Only the Hub boundary is substituted
    (signature-bound); the production bodies run REAL against the exact
    on-disk layout datagen.py / organisms.py / make_source_rate_fn write.
    """
    from types import SimpleNamespace

    cfg = _cfg(tmp_path, upload=True)
    class_dir = cfg.out_root / "sycophancy"
    build_dir = class_dir / "build"

    # ── datagen/ — exactly the files generate_training_data persists ────────
    datagen_dir = build_dir / "datagen"
    datagen_dir.mkdir(parents=True)
    for jsonl in ("raw_pos.jsonl", "raw_neg.jsonl", "judge_rows.jsonl"):
        (datagen_dir / jsonl).write_text("{}\n")
    sidecars = ("gen_manifest.json", "judge_raw_pos.json", "judge_raw_neg.json", "pool_meta.json")
    for sidecar in sidecars:
        (datagen_dir / sidecar).write_text("{}")
    # Manifest-hashed judge cache (datagen.py: out_dir / f"judge_cache_{hash12}")
    cache = datagen_dir / "judge_cache_0a1b2c3d4e5f" / "pos"
    cache.mkdir(parents=True)
    (cache / ("a" * 16 + ".json")).write_text("{}")
    (cache / ".dispatch").mkdir()
    (cache / ".dispatch" / "chunk_000.jsonl").write_text("{}\n")

    # ── build root — the assembled training mix ────────────────────────────
    (build_dir / "train_mix.jsonl").write_text("{}\n")
    (build_dir / "mix_meta.json").write_text("{}")
    # Checkpoint tree JSONs must NOT be swept into any upload.
    ckpt = build_dir / "train" / "checkpoint-10"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "tokenizer.json").write_text("{}")

    # ── build/rate/ — make_source_rate_fn's per-rung layout ────────────────
    rung = build_dir / "rate" / "rate_checkpoint-10"
    rung_judge = rung / "judge" / "trained_persona_software_engineer"
    rung_judge.mkdir(parents=True)
    (rung / "completions__trained__persona_software_engineer.json").write_text("{}")
    (rung_judge / "judge_raw.json").write_text("{}")
    (rung_judge / ("b" * 16 + ".json")).write_text("{}")  # JudgeCache entry
    (rung_judge / ".dispatch").mkdir()
    (rung_judge / ".dispatch" / "state.json").write_text("{}")

    # ── verify/ — verify_organism's layout (organisms.py:996) ──────────────
    verify_dir = class_dir / "verify"
    verify_judge = verify_dir / "judge" / "trained_persona_software_engineer"
    verify_judge.mkdir(parents=True)
    (verify_dir / "completions__trained__persona_software_engineer.json").write_text("{}")
    (verify_dir / "completions__base__persona_software_engineer.json").write_text("{}")
    (verify_dir / "organism_report.json").write_text("{}")
    (verify_judge / "judge_raw.json").write_text("{}")
    (verify_judge / ("c" * 16 + ".json")).write_text("{}")  # JudgeCache entry
    (verify_judge / ".dispatch").mkdir()
    (verify_judge / ".dispatch" / "requests.jsonl").write_text("{}\n")

    folder_calls = _fake_hub_boundary(monkeypatch)
    build_result = SimpleNamespace(
        adapter_path=str(tmp_path / "adapter"),
        train_mix_path=str(build_dir / "train_mix.jsonl"),
        data_paths={"datagen_dir": str(datagen_dir)},
    )
    out = pilot._upload_class("sycophancy", build_result, cfg, pilot.PilotSeams())

    assert out["status"] == "ok", out
    by_bucket = {b.arguments["path_in_repo"]: b for b in folder_calls}
    prefix = "issue906_pilot/sycophancy"
    assert {
        f"{prefix}/raw_completions",
        f"{prefix}/train_mix",
        f"{prefix}/verify",
        f"{prefix}/dose_ladder",
    } <= set(by_bucket)

    # (d) The four datagen .json sidecars — previously dropped by the
    # *.jsonl-only upload_dataset_directory default.
    datagen_expected = set(by_bucket[f"{prefix}/raw_completions"].arguments["expected_repo_paths"])
    for sidecar in sidecars:
        assert f"{prefix}/raw_completions/{sidecar}" in datagen_expected
    for jsonl in ("raw_pos.jsonl", "raw_neg.jsonl", "judge_rows.jsonl"):
        assert f"{prefix}/raw_completions/{jsonl}" in datagen_expected

    # (c) The training mix — EXACTLY the two build-root files, nothing from
    # the checkpoint tree.
    assert sorted(by_bucket[f"{prefix}/train_mix"].arguments["expected_repo_paths"]) == [
        f"{prefix}/train_mix/mix_meta.json",
        f"{prefix}/train_mix/train_mix.jsonl",
    ]

    # (a) verify/ — completions for BOTH sides + report + judge raw.
    verify_expected = set(by_bucket[f"{prefix}/verify"].arguments["expected_repo_paths"])
    for required in (
        f"{prefix}/verify/completions__trained__persona_software_engineer.json",
        f"{prefix}/verify/completions__base__persona_software_engineer.json",
        f"{prefix}/verify/organism_report.json",
        f"{prefix}/verify/judge/trained_persona_software_engineer/judge_raw.json",
    ):
        assert required in verify_expected, f"missing from verify expected-set: {required}"

    # (b) build/rate/ dose-ladder completions + judge raw.
    rate_expected = set(by_bucket[f"{prefix}/dose_ladder"].arguments["expected_repo_paths"])
    for required in (
        f"{prefix}/dose_ladder/rate_checkpoint-10/"
        "completions__trained__persona_software_engineer.json",
        f"{prefix}/dose_ladder/rate_checkpoint-10/judge/"
        "trained_persona_software_engineer/judge_raw.json",
    ):
        assert required in rate_expected, f"missing from dose_ladder expected-set: {required}"

    # Re-derivable caches + checkpoint JSONs are excluded from EVERY expected
    # set AND every allow_patterns list (allow == expected by construction).
    for b in folder_calls:
        for coll in (b.arguments["expected_repo_paths"], b.arguments["allow_patterns"]):
            joined = "\n".join(coll)
            assert "judge_cache" not in joined
            assert ".dispatch" not in joined
            assert "a" * 16 not in joined
            assert "b" * 16 not in joined
            assert "c" * 16 not in joined
    # The checkpoint tree specifically never uploads (train/ is not a bucket).
    all_expected = [p for b in folder_calls for p in b.arguments["expected_repo_paths"]]
    assert not [p for p in all_expected if "adapter_config.json" in p]
    assert not [p for p in all_expected if "tokenizer.json" in p]

    # Recorded outcome per stage.
    assert out["generations"].endswith("/raw_completions")
    assert out["train_mix"].endswith("/train_mix")
    assert out["verify"].endswith("/verify")
    assert out["dose_ladder"].endswith("/dose_ladder")


def test_upload_class_marker_carveout_skips_duplicate_train_mix(tmp_path, monkeypatch):
    """Marker carve-out: train_mix_path IS the datagen-dir pos.jsonl (already
    uploaded under raw_completions) — _upload_class records the skip instead of
    duplicating, and the absent verify//rate/ dirs record a graceful None."""
    from types import SimpleNamespace

    cfg = _cfg(tmp_path, upload=True)
    mix_dir = cfg.out_root / "marker" / "build" / "mix"
    mix_dir.mkdir(parents=True)
    (mix_dir / "pos.jsonl").write_text("{}\n")
    (mix_dir / "cn.jsonl").write_text("{}\n")

    folder_calls = _fake_hub_boundary(monkeypatch)
    build_result = SimpleNamespace(
        adapter_path=str(tmp_path / "adapter"),
        train_mix_path=str(mix_dir / "pos.jsonl"),
        data_paths={"datagen_dir": str(mix_dir)},
    )
    out = pilot._upload_class("marker", build_result, cfg, pilot.PilotSeams())

    assert out["status"] == "ok", out
    assert out["train_mix"] == "covered-by-raw-completions-upload"
    assert out["verify"] is None
    assert out["dose_ladder"] is None
    buckets = {b.arguments["path_in_repo"] for b in folder_calls}
    assert "issue906_pilot/marker/train_mix" not in buckets
    datagen_expected = next(
        b for b in folder_calls if b.arguments["path_in_repo"].endswith("/raw_completions")
    ).arguments["expected_repo_paths"]
    assert "issue906_pilot/marker/raw_completions/pos.jsonl" in datagen_expected
    assert "issue906_pilot/marker/raw_completions/cn.jsonl" in datagen_expected


def test_upload_class_marker_verify_rollouts_covered(tmp_path, monkeypatch):
    """r9 CONCERN genreduce-rollout-text-not-persisted (upload leg): the marker
    carve-out's greedy slot-read rollout text — verify/marker_rollouts__{base,
    trained}.jsonl, written by _verify_marker_class's inline path — rides the
    existing verify upload leg's fail-loud expected set. Only the Hub boundary
    is substituted (signature-bound); _upload_class / _upload_pilot_dir run
    REAL."""
    from types import SimpleNamespace

    cfg = _cfg(tmp_path, upload=True)
    mix_dir = cfg.out_root / "marker" / "build" / "mix"
    mix_dir.mkdir(parents=True)
    (mix_dir / "pos.jsonl").write_text("{}\n")
    (mix_dir / "cn.jsonl").write_text("{}\n")
    verify_dir = cfg.out_root / "marker" / "verify"
    verify_dir.mkdir(parents=True)
    (verify_dir / "marker_rollouts__base.jsonl").write_text("{}\n")
    (verify_dir / "marker_rollouts__trained.jsonl").write_text("{}\n")

    folder_calls = _fake_hub_boundary(monkeypatch)
    build_result = SimpleNamespace(
        adapter_path=str(tmp_path / "adapter"),
        train_mix_path=str(mix_dir / "pos.jsonl"),
        data_paths={"datagen_dir": str(mix_dir)},
    )
    out = pilot._upload_class("marker", build_result, cfg, pilot.PilotSeams())

    assert out["status"] == "ok", out
    # (ii) The verify leg now fires for the marker carve-out and its fail-loud
    # expected set names both rollout files.
    assert out["verify"] is not None and out["verify"].endswith("/verify")
    by_bucket = {b.arguments["path_in_repo"]: b for b in folder_calls}
    verify_expected = set(
        by_bucket["issue906_pilot/marker/verify"].arguments["expected_repo_paths"]
    )
    assert "issue906_pilot/marker/verify/marker_rollouts__base.jsonl" in verify_expected
    assert "issue906_pilot/marker/verify/marker_rollouts__trained.jsonl" in verify_expected


def test_upload_class_baseline_missing_raw_completions_fails_loud(tmp_path, monkeypatch):
    """r9 CONCERN rollout-upload-expected-set-hollow: a baseline/ dir with
    baseline.json + judge_raw.json present but NO raw_completions.jsonl must
    FAIL the upload loudly (status='failed' -> run_class upload_failed ->
    exit 2 in --full) — the glob-derived expected set alone would silently
    pass on whatever files happen to exist. The REAL _upload_class /
    _upload_pilot_dir bodies run; only the Hub boundary is substituted."""
    from types import SimpleNamespace

    cfg = _cfg(tmp_path, upload=True)
    class_dir = cfg.out_root / "sycophancy"
    build_dir = class_dir / "build"
    datagen_dir = build_dir / "datagen"
    datagen_dir.mkdir(parents=True)
    (datagen_dir / "raw_pos.jsonl").write_text("{}\n")
    (build_dir / "train_mix.jsonl").write_text("{}\n")
    (build_dir / "mix_meta.json").write_text("{}")
    baseline = class_dir / "baseline"
    baseline.mkdir(parents=True)
    (baseline / "baseline.json").write_text("{}")
    (baseline / "judge_raw.json").write_text("{}")
    # raw_completions.jsonl deliberately NEVER written — the hollow-set hole.

    folder_calls = _fake_hub_boundary(monkeypatch)
    build_result = SimpleNamespace(
        adapter_path=str(tmp_path / "adapter"),
        train_mix_path=str(build_dir / "train_mix.jsonl"),
        data_paths={"datagen_dir": str(datagen_dir)},
    )
    out = pilot._upload_class("sycophancy", build_result, cfg, pilot.PilotSeams())

    assert out["status"] == "failed", out
    assert "raw_completions.jsonl" in out["error"]
    assert "required" in out["error"]
    # The baseline bucket never committed (the raise precedes the Hub call).
    assert "issue906_pilot/sycophancy/baseline" not in {
        b.arguments["path_in_repo"] for b in folder_calls
    }


def test_upload_class_marker_verify_missing_rollout_fails_loud(tmp_path, monkeypatch):
    """r9 CONCERN rollout-upload-expected-set-hollow: a marker verify/ leg
    missing ONE of the two required slot-read rollout files fails loud
    (status='failed' naming the absent file) instead of uploading the
    remaining files on the glob-derived set."""
    from types import SimpleNamespace

    cfg = _cfg(tmp_path, upload=True)
    mix_dir = cfg.out_root / "marker" / "build" / "mix"
    mix_dir.mkdir(parents=True)
    (mix_dir / "pos.jsonl").write_text("{}\n")
    verify_dir = cfg.out_root / "marker" / "verify"
    verify_dir.mkdir(parents=True)
    (verify_dir / "marker_rollouts__base.jsonl").write_text("{}\n")
    # marker_rollouts__trained.jsonl deliberately NEVER written.

    folder_calls = _fake_hub_boundary(monkeypatch)
    build_result = SimpleNamespace(
        adapter_path=str(tmp_path / "adapter"),
        train_mix_path=str(mix_dir / "pos.jsonl"),
        data_paths={"datagen_dir": str(mix_dir)},
    )
    out = pilot._upload_class("marker", build_result, cfg, pilot.PilotSeams())

    assert out["status"] == "failed", out
    assert "marker_rollouts__trained.jsonl" in out["error"]
    assert "required" in out["error"]
    assert "issue906_pilot/marker/verify" not in {b.arguments["path_in_repo"] for b in folder_calls}


def test_verify_marker_class_inline_persists_rollouts(tmp_path, monkeypatch):
    """r9 CONCERN genreduce-rollout-text-not-persisted (wiring): the REAL
    _verify_marker_class INLINE path (marker_verify_fn seam = None) persists
    greedy rollout text for BOTH model sides to
    verify/marker_rollouts__{base,trained}.jsonl and reports the paths under
    rollout_paths. Only the HF weights / tokenizer / PEFT boundaries are faked;
    _read_marker_slots + validate_marker_slot_record run REAL."""
    import peft
    import transformers

    from explore_persona_space.artifacts.recipe import (
        MARKER_TEXT,
        MARKER_TOKEN_ID,
        QWEN_IM_END_ID,
    )

    cfg = _cfg(tmp_path)
    class_dir = cfg.out_root / "marker"
    behavior = BEHAVIORS["marker"]
    vocab = 200000  # >= marker id 83399 and im_end id 151645

    class FakeLM:
        """HF-boundary fake: greedy generate appends [11, 12, im_end]; zero logits."""

        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, input_ids, **kwargs):
            new = torch.tensor([[11, 12, QWEN_IM_END_ID]], dtype=torch.long)
            return torch.cat([input_ids, new], dim=1)

        def __call__(self, input_ids):
            from unittest.mock import MagicMock

            out = MagicMock()
            out.logits = torch.zeros(1, input_ids.shape[1], vocab)
            return out

    class FakeTrained(FakeLM):
        def __init__(self):
            # .get("default", None) -> None skips the gauge assert
            self.peft_config: dict = {}

    class FakeTok:
        eos_token_id = QWEN_IM_END_ID

        def encode(self, text, add_special_tokens=False):
            assert text == MARKER_TEXT
            return [MARKER_TOKEN_ID]  # the in-process marker token-id assert passes

        def apply_chat_template(self, msgs, add_generation_prompt=True, return_tensors="pt"):
            return torch.tensor([[10, 20]], dtype=torch.long)

        def decode(self, ids, skip_special_tokens=False):
            return "tok:" + ",".join(str(i) for i in ids)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTok())
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: FakeLM()
    )
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", lambda *a, **k: FakeTrained())

    result = pilot._verify_marker_class(
        behavior, str(tmp_path / "adapter"), cfg, pilot.PilotSeams(), class_dir
    )

    rollout_paths = result["rollout_paths"]
    assert rollout_paths is not None, "inline path must report its persisted rollout paths"
    n_contexts = result["n_eval_contexts"]
    n_questions = result["n_eval_questions"]
    for side in ("base", "trained"):
        path = Path(rollout_paths[side])
        assert path == class_dir / "verify" / f"marker_rollouts__{side}.jsonl"
        assert path.is_file(), f"{side} rollout text file was not persisted"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == n_contexts * n_questions
        # Raw new-token region persisted (im_end included, pre-strip).
        assert all(r["completion"] == f"tok:11,12,{QWEN_IM_END_ID}" for r in rows)
    # The three-space reduce still ran on the same records.
    assert len(result["per_context"]) == n_contexts


def test_upload_pilot_dir_explicit_filenames_fail_loud_on_missing(tmp_path):
    """Explicit filenames= mode raises on a missing named file (the training
    mix is a build-contract guarantee; a silent skip would re-open the r8
    coverage hole) — and on a missing directory."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "train_mix.jsonl").write_text("{}\n")  # mix_meta.json absent

    with pytest.raises(RuntimeError, match=r"mix_meta\.json"):
        pilot._upload_pilot_dir(
            build_dir, "issue906_pilot/x/train_mix", filenames=["train_mix.jsonl", "mix_meta.json"]
        )
    with pytest.raises(RuntimeError, match="not a directory"):
        pilot._upload_pilot_dir(
            tmp_path / "absent", "issue906_pilot/x/train_mix", filenames=["train_mix.jsonl"]
        )


def test_upload_pilot_dir_required_rel_paths_gate(tmp_path, monkeypatch):
    """r9 CONCERN rollout-upload-expected-set-hollow (mechanism): scan mode
    with required_rel_paths= raises on a required file absent from the scan;
    an absent DIR keeps the graceful None skip (stage-did-not-run contract);
    combining filenames= with required_rel_paths= is rejected."""
    import explore_persona_space.orchestrate.hub as hub

    d = tmp_path / "leg"
    d.mkdir()
    (d / "judge_raw.json").write_text("{}")

    # Required file absent from the scanned set -> loud RuntimeError.
    with pytest.raises(RuntimeError, match=r"required.*raw_completions\.jsonl"):
        pilot._upload_pilot_dir(
            d, "issue906_pilot/x/leg", required_rel_paths=["raw_completions.jsonl"]
        )

    # Absent dir -> None (the stage did not run; per-class status handling
    # owns a crashed stage, not the upload leg).
    assert (
        pilot._upload_pilot_dir(
            tmp_path / "absent", "issue906_pilot/x/leg", required_rel_paths=["a.jsonl"]
        )
        is None
    )

    # Mutually exclusive with explicit-file mode (already fail-loud).
    with pytest.raises(ValueError, match="mutually exclusive"):
        pilot._upload_pilot_dir(
            d,
            "issue906_pilot/x/leg",
            filenames=["judge_raw.json"],
            required_rel_paths=["judge_raw.json"],
        )

    # Present required file passes through to the (faked) Hub boundary.
    (d / "raw_completions.jsonl").write_text("{}\n")
    monkeypatch.setattr(hub, "_upload_folder_filtered", lambda *a, **k: "repo/x/leg")
    assert (
        pilot._upload_pilot_dir(
            d, "issue906_pilot/x/leg", required_rel_paths=["raw_completions.jsonl"]
        )
        == "repo/x/leg"
    )


def test_upload_pilot_dir_raises_on_empty_url(tmp_path, monkeypatch):
    """Codex r8 Minor: an empty URL from hub._upload_folder_filtered is a FAILED
    upload — the REAL _upload_pilot_dir body raises RuntimeError instead of
    returning a falsy 'success'."""
    import explore_persona_space.orchestrate.hub as hub

    d = tmp_path / "artifacts"
    d.mkdir()
    (d / "baseline.json").write_text("{}")
    monkeypatch.setattr(hub, "_upload_folder_filtered", lambda *a, **k: "")
    with pytest.raises(RuntimeError, match="failed or was incomplete"):
        pilot._upload_pilot_dir(d, "issue906_pilot/x/baseline")


@pytest.mark.parametrize(
    ("rel_parts", "excluded"),
    [
        (("judge_cache", "item.json"), True),  # literal judge_cache/ (extract phase)
        (("judge_cache_0a1b2c3d4e5f", "pos", "x.json"), True),  # datagen manifest-hashed
        (("judge", "trained_ctx", ".dispatch", "state.json"), True),  # dispatch ckpt
        (("judge", "trained_ctx", "a" * 16 + ".json"), True),  # JudgeCache entry
        (("judge", "trained_ctx", "judge_raw.json"), False),  # real judge output
        (("completions__trained__ctx.json",), False),
        (("organism_report.json",), False),
        (("pool_meta.json",), False),
        (("gen_manifest.json",), False),
        (("r_b_sycophancy.pt",), False),
    ],
)
def test_is_rederivable_cache_classification(rel_parts, excluded):
    """The cache-exclusion predicate keeps every real artifact and drops every
    re-derivable judge-cache shape (judge_cache*/ dirs, .dispatch/, 16-hex
    JudgeCache entries)."""
    assert pilot._is_rederivable_cache(rel_parts) is excluded


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
