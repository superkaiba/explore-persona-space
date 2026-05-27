"""vLLM ``--enable-lora`` sampled-eval tests (task #397, Round 6).

Round 6 dropped the merged-dir + vLLM-load pattern in favor of vLLM's
LoRA-adapter mode: ``LLM(model=base, enable_lora=True, max_loras=1,
max_lora_rank=32)`` plus ``LLM.generate(..., lora_request=LoRARequest(
name, 1, lora_path=adapter_path))``. Eliminates the ~14 GB merged-dir
per cell, which unlocks the 8/8 concurrency cap.

This test surface verifies the wiring without actually loading vLLM:

  - ``generate_completions_with_lora`` constructs an ``LLM`` with
    ``enable_lora=True``, ``max_loras=1``, ``max_lora_rank=32``.
  - The ``LLM.generate`` call carries a ``LoRARequest(lora_path=<adapter>)``
    kwarg — no merge step in the pipeline.
  - ``system_prompt_overrides`` are threaded through chat-template
    construction (SR1 wiring carry-over from round 2).
  - Return shape matches ``factor_screen_365.eval_panel.generate_completions``
    so ``score_markers_threaded`` consumes it unchanged.
  - The dispatcher's ``_run_smoke_sampled_eval`` and ``run_one_cell``
    pipeline both call the new ``generate_completions_with_lora`` (no
    residual ``EvalConfig + generate_completions(merged_path)`` calls).

CPU-only; vLLM is monkeypatched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock


def _install_fake_vllm(monkeypatch, recorded_calls: dict) -> None:
    """Monkeypatch vllm + huggingface_hub modules so the test runs without
    pulling in real vLLM / HF Hub. ``recorded_calls`` collects the kwargs
    passed to LLM, LLM.generate, and LoRARequest for assertions.
    """
    fake_vllm = MagicMock(name="vllm_module")
    fake_lora_module = MagicMock(name="vllm.lora")
    fake_lora_request_module = MagicMock(name="vllm.lora.request")

    class _FakeSamplingParams:
        def __init__(self, **kwargs):
            recorded_calls["sampling_kwargs"] = kwargs

    class _FakeLoRARequest:
        def __init__(self, lora_name, lora_int_id, lora_path, **kwargs):
            self.lora_name = lora_name
            self.lora_int_id = lora_int_id
            self.lora_path = lora_path
            recorded_calls.setdefault("lora_request_constructions", []).append(
                {"lora_name": lora_name, "lora_int_id": lora_int_id, "lora_path": lora_path}
            )

    class _FakeOutput:
        def __init__(self, text):
            self.text = text

    class _FakeLLMOutput:
        def __init__(self, n_completions):
            self.outputs = [_FakeOutput(f"completion_{i}") for i in range(n_completions)]

    class _FakeLLM:
        def __init__(self, **kwargs):
            recorded_calls["llm_init_kwargs"] = kwargs

        def generate(self, prompts, sampling_params, lora_request=None):
            recorded_calls.setdefault("generate_calls", []).append(
                {
                    "n_prompts": len(prompts),
                    "lora_request": lora_request,
                    "lora_path_from_request": getattr(lora_request, "lora_path", None),
                }
            )
            n = recorded_calls["sampling_kwargs"].get("n", 1)
            return [_FakeLLMOutput(n) for _ in prompts]

    fake_vllm.LLM = _FakeLLM
    fake_vllm.SamplingParams = _FakeSamplingParams
    fake_lora_request_module.LoRARequest = _FakeLoRARequest

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", fake_lora_module)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", fake_lora_request_module)


def _install_fake_transformers(monkeypatch) -> None:
    """Stub transformers.AutoTokenizer so the test doesn't hit HF for Qwen."""
    fake_transformers = MagicMock(name="transformers_module")

    class _FakeTok:
        pad_token_id = 0
        pad_token = "[PAD]"
        eos_token = "[EOS]"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            # Build a tiny stable string so the test can verify content
            # propagated correctly (e.g. system_prompt_overrides took effect).
            parts = [f"{m['role']}: {m['content']}" for m in messages]
            return "\n".join(parts) + ("\nassistant: " if add_generation_prompt else "")

    fake_transformers.AutoTokenizer = _FakeTok
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


# ---------------------------------------------------------------------------
# Core wiring assertions
# ---------------------------------------------------------------------------


def test_generate_completions_with_lora_constructs_llm_with_enable_lora(monkeypatch) -> None:
    """vLLM LLM is constructed with enable_lora=True + max_loras=1 + max_lora_rank=32."""
    recorded: dict = {}
    _install_fake_vllm(monkeypatch, recorded)
    _install_fake_transformers(monkeypatch)
    # Also stub the _patch_tokenizer_for_vllm side-effect import.
    import explore_persona_space.experiments.factor_screen_365.eval_panel as fs365_ep

    monkeypatch.setattr(fs365_ep, "_patch_tokenizer_for_vllm", lambda: None)

    from explore_persona_space.experiments.factor_screen_397.eval_panel import (
        generate_completions_with_lora,
    )

    result = generate_completions_with_lora(
        base_model_path="Qwen/Qwen2.5-7B-Instruct",
        lora_path="/tmp/cell/adapter",
        personas={"librarian": "You are a librarian.", "barista": "You are a barista."},
        questions=["Q1?", "Q2?"],
    )

    # LLM constructed with enable_lora wiring.
    init = recorded["llm_init_kwargs"]
    assert init["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert init["enable_lora"] is True, (
        f"vLLM LLM MUST be constructed with enable_lora=True; got {init.get('enable_lora')}"
    )
    assert init["max_loras"] == 1, f"max_loras must be 1; got {init.get('max_loras')}"
    assert init["max_lora_rank"] == 32, (
        f"max_lora_rank must be 32 (matches #397 lora_r); got {init.get('max_lora_rank')}"
    )
    assert init["dtype"] == "bfloat16"

    # generate() was called with a LoRARequest carrying the adapter path.
    gen_call = recorded["generate_calls"][0]
    assert gen_call["lora_request"] is not None, (
        "LLM.generate MUST receive a lora_request kwarg; got None"
    )
    assert gen_call["lora_path_from_request"] == "/tmp/cell/adapter", (
        f"LoRARequest.lora_path must be the cell's adapter path; "
        f"got {gen_call['lora_path_from_request']}"
    )

    # Return shape: {persona: {question: [completions]}}.
    assert set(result.keys()) == {"librarian", "barista"}
    assert set(result["librarian"].keys()) == {"Q1?", "Q2?"}
    # 5 completions per (persona, question) by default.
    assert len(result["librarian"]["Q1?"]) == 5


def test_generate_completions_with_lora_passes_correct_lora_request_fields(
    monkeypatch,
) -> None:
    """LoRARequest is constructed with lora_name + lora_int_id=1 + lora_path."""
    recorded: dict = {}
    _install_fake_vllm(monkeypatch, recorded)
    _install_fake_transformers(monkeypatch)
    import explore_persona_space.experiments.factor_screen_365.eval_panel as fs365_ep

    monkeypatch.setattr(fs365_ep, "_patch_tokenizer_for_vllm", lambda: None)

    from explore_persona_space.experiments.factor_screen_397.eval_panel import (
        generate_completions_with_lora,
    )

    generate_completions_with_lora(
        base_model_path="Qwen/Qwen2.5-7B-Instruct",
        lora_path="/tmp/cell/adapter",
        personas={"librarian": "You are a librarian."},
        questions=["Q?"],
        lora_name="i397-smoke-cell",
    )

    lr_constructs = recorded["lora_request_constructions"]
    assert len(lr_constructs) == 1
    lr = lr_constructs[0]
    assert lr["lora_name"] == "i397-smoke-cell"
    assert lr["lora_int_id"] == 1
    assert lr["lora_path"] == "/tmp/cell/adapter"


def test_generate_completions_with_lora_threads_system_prompt_overrides(monkeypatch) -> None:
    """SR1 wiring: system_prompt_overrides per persona replace the panel
    prompt for that persona only.

    Asserts the FINAL prompts that reach vLLM carry the override text for
    the named persona AND the canonical text for the un-overridden one.
    """
    recorded: dict = {}
    _install_fake_vllm(monkeypatch, recorded)
    _install_fake_transformers(monkeypatch)
    import explore_persona_space.experiments.factor_screen_365.eval_panel as fs365_ep

    monkeypatch.setattr(fs365_ep, "_patch_tokenizer_for_vllm", lambda: None)

    from explore_persona_space.experiments.factor_screen_397.eval_panel import (
        generate_completions_with_lora,
    )

    # Capture the prompts that reach vLLM by monkeypatching LLM.generate
    # to record them BEFORE returning.
    prompts_seen: list[str] = []
    original_llm_class = sys.modules["vllm"].LLM

    class _RecordingLLM(original_llm_class):
        def generate(self, prompts, sampling_params, lora_request=None):
            prompts_seen.extend(prompts)
            return super().generate(prompts, sampling_params, lora_request=lora_request)

    sys.modules["vllm"].LLM = _RecordingLLM

    generate_completions_with_lora(
        base_model_path="Qwen/Qwen2.5-7B-Instruct",
        lora_path="/tmp/cell/adapter",
        personas={
            "librarian": "You are a librarian (canonical).",
            "barista": "You are a barista (canonical).",
        },
        questions=["Q?"],
        system_prompt_overrides={
            "librarian": "BACKGROUND-CONTEXT-OVERRIDE-FOR-LIBRARIAN",
        },
    )

    # 2 personas x 1 question = 2 prompts.
    assert len(prompts_seen) == 2
    librarian_prompt = next(p for p in prompts_seen if "BACKGROUND-CONTEXT" in p)
    assert "OVERRIDE-FOR-LIBRARIAN" in librarian_prompt
    assert "canonical" not in librarian_prompt, (
        "Override must REPLACE the canonical prompt for librarian, not append"
    )
    barista_prompt = next(p for p in prompts_seen if "barista" in p)
    assert "canonical" in barista_prompt, (
        "Un-overridden personas must still see their canonical panel prompt"
    )
    assert "OVERRIDE-FOR-LIBRARIAN" not in barista_prompt


# ---------------------------------------------------------------------------
# Pipeline: no merge step anywhere
# ---------------------------------------------------------------------------


def test_train_outcome_no_longer_carries_merged_path() -> None:
    """Round 6 dropped TrainOutcome.merged_path because the merge step is gone.

    A future regression that re-adds the field would also re-add the merge
    step (the field's only consumer). The dataclass shape check is the
    canonical guard.
    """
    from explore_persona_space.experiments.factor_screen_397.training import TrainOutcome

    fields = {f.name for f in TrainOutcome.__dataclass_fields__.values()}
    assert "merged_path" not in fields, (
        "TrainOutcome.merged_path was removed in Round 6 — vLLM --enable-lora "
        "consumes the adapter directly. Adding it back would re-introduce the "
        "merge step + ~14 GB per-cell disk footprint."
    )
    # Sanity: adapter_path is still there.
    assert "adapter_path" in fields


def test_train_one_cell_does_not_call_merge_lora(monkeypatch) -> None:
    """Pipeline assertion: ``train_one_cell`` never invokes ``merge_lora``.

    Monkeypatches both ``train_lora`` and ``merge_lora`` so the test runs
    without real training. ``merge_lora`` is stubbed to RAISE — if the
    pipeline calls it, the assertion fires loud.
    """
    from pathlib import Path

    import explore_persona_space.train.sft as sft_mod
    from explore_persona_space.experiments.factor_screen_397.cells import Cell
    from explore_persona_space.experiments.factor_screen_397.training import train_one_cell

    monkeypatch.setattr(
        sft_mod,
        "train_lora",
        lambda base_model_path, data_path, output_dir, cfg: (output_dir, 1.23),
    )

    merge_calls: list = []

    def _fake_merge(*args, **kwargs):
        merge_calls.append((args, kwargs))
        raise AssertionError(
            "Round 6 contract violation: train_one_cell called merge_lora. "
            "vLLM --enable-lora consumes the adapter directly; the merge step "
            "must NOT run in Round 6+."
        )

    monkeypatch.setattr(sft_mod, "merge_lora", _fake_merge)

    # Also stub _count_lines so we don't need a real JSONL.
    import explore_persona_space.experiments.factor_screen_397.training as training_mod

    monkeypatch.setattr(training_mod, "_count_lines", lambda p: 800)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cell_dir = Path(tmp) / "cell"
        cell_dir.mkdir(parents=True, exist_ok=True)
        outcome = train_one_cell(
            cell=Cell.from_key("00000"),
            seed=42,
            source="librarian",
            data_path=Path("/tmp/fake.jsonl"),
            cell_output_dir=cell_dir,
            marker_text="※",
        )

    # adapter_path is the returned LoRA root.
    assert outcome.adapter_path.endswith("adapter")
    # merge_lora was NOT called (the fake would've raised AssertionError).
    assert len(merge_calls) == 0


def test_dispatcher_inprocess_path_uses_generate_completions_with_lora() -> None:
    """Round 11: the in-process per-cell pipeline (``_run_one_cell_inprocess``
    in the dispatcher) uses generate_completions_with_lora (NOT the old
    EvalConfig + generate_completions(merged_path) path).

    Replaces the Round 6 test against the deleted ``run_one_cell.py``.
    Static-import check on the dispatcher source.
    """
    src_path = (
        Path(__file__).resolve().parent.parent.parent / "scripts" / "dispatch_factor_screen_397.py"
    )
    text = src_path.read_text(encoding="utf-8")
    assert "generate_completions_with_lora" in text, (
        "dispatch_factor_screen_397.py must call generate_completions_with_lora "
        "(Round 6 + Round 11 lift)"
    )
    # The old pattern should be gone.
    assert "EvalConfig(" not in text, (
        "dispatch_factor_screen_397.py must NOT construct EvalConfig (Round 6 "
        "dropped the merged-dir + generate_completions path)"
    )
    assert "outcome.merged_path" not in text, (
        "dispatch_factor_screen_397.py must NOT reference outcome.merged_path "
        "(Round 6 removed the field from TrainOutcome)"
    )


def test_dispatcher_smoke_path_uses_generate_completions_with_lora() -> None:
    """Same static-import check on the dispatcher's smoke path (unchanged
    between Round 6 and Round 11; the smoke phase has always lived
    in-process in the dispatcher).
    """
    src_path = (
        Path(__file__).resolve().parent.parent.parent / "scripts" / "dispatch_factor_screen_397.py"
    )
    text = src_path.read_text(encoding="utf-8")
    assert "generate_completions_with_lora" in text, (
        "dispatch_factor_screen_397.py smoke path must call "
        "generate_completions_with_lora (Round 6)"
    )
    assert "outcome.merged_path" not in text, (
        "dispatch_factor_screen_397.py must NOT reference outcome.merged_path "
        "(Round 6 removed the field)"
    )


def test_no_merge_step_in_training_module() -> None:
    """Static check: training.py no longer imports or CALLS merge_lora.

    A reviewer can grep for this; the test pins it so a future re-add
    of the actual call fails loud at CI rather than at the next 324-cell
    sweep. The test allows mentions in docstrings/comments (Round 6's
    own docstring explains WHY merge_lora is gone) but rejects:

      - ``from explore_persona_space.train.sft import ... merge_lora``
      - ``merge_lora(...)`` call site

    Patterns to check by walking the code with AST would be more
    rigorous; for now, simple substring checks on import + call shapes
    are enough.
    """
    src_path = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "explore_persona_space"
        / "experiments"
        / "factor_screen_397"
        / "training.py"
    )
    text = src_path.read_text(encoding="utf-8")
    # No import.
    assert "import merge_lora" not in text, (
        "factor_screen_397/training.py must NOT import merge_lora "
        "(Round 6 dropped the merge step entirely)"
    )
    # No `, merge_lora,` or `, merge_lora\n` in a from-import line either.
    for line in text.splitlines():
        if line.startswith("from explore_persona_space.train.sft") and "merge_lora" in line:
            raise AssertionError(
                f"Round 6 contract: train_one_cell must NOT import merge_lora; got line: {line}"
            )
    # No call site.
    assert "merge_lora(" not in text, (
        "factor_screen_397/training.py must NOT call merge_lora(...) "
        "(Round 6 dropped the merge step entirely)"
    )
