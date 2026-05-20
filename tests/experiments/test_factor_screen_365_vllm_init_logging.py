"""vLLM init-phase logging tests for the task #365 eval panel.

Round-9 (issue #365) Fix F: round-8 could not distinguish a vLLM init
crash from a later eval crash because no explicit log line fired at
``LLM(...)`` instantiation. The eval-panel now emits three log lines per
panel (persona + random-control), wrapping the LLM() call:

  1. ``vLLM init STARTING`` — fires BEFORE ``_stagger_vllm_init``.
  2. ``vLLM init: instantiating LLM(model=…)`` — fires AFTER the stagger,
     immediately before the LLM() call.
  3. ``vLLM init COMPLETE`` — fires AFTER LLM() returns.

Combined with the per-cell stderr capture from Fix D, these lines pin
down exactly which phase a crashing cell reached.

Round-11 (issue #365): ``vllm_session`` now yields an ``_LLMHolder`` so
the context manager can drop the LLM reference (``holder.llm = None``)
BEFORE ``gc.collect()`` / ``empty_cache()``. Tests updated to read
``session.llm`` and to verify the holder is zeroed on context exit.
"""

from __future__ import annotations

import logging

import pytest

from explore_persona_space.experiments.factor_screen_365 import eval_panel
from explore_persona_space.experiments.factor_screen_365.eval_panel import (
    EvalConfig,
    RandomControlConfig,
    vllm_session,
)


@pytest.fixture(autouse=True)
def _disable_stagger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the per-GPU stagger sleep so tests do not wait on time.sleep."""
    monkeypatch.setenv("EPS_FS365_VLLM_STAGGER_S", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")


class _FakeLLM:
    """Minimal stand-in for vllm.LLM that records construction kwargs.

    Used to drive ``generate_completions`` without loading the real vLLM.
    """

    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        _FakeLLM.last_kwargs = kwargs

    def generate(self, prompts, sampling_params):
        # Return a list of fake outputs, one per prompt, each with one completion
        # so the downstream zip-with-keys path stays happy.
        class _O:
            text = "stub completion"

        class _Out:
            def __init__(self) -> None:
                self.outputs = [_O()]

        return [_Out() for _ in prompts]


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeTokenizer:
    """Trivial tokenizer stub that drives ``apply_chat_template`` deterministically."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "STUB_PROMPT"


def _install_vllm_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the lazy ``from vllm import LLM, SamplingParams`` inside the panel.

    The eval-panel imports vLLM inside the function body, so monkey-patching
    the module attribute does not help. Instead we shim ``sys.modules["vllm"]``
    with a tiny namespace that has the two symbols, and also stub the
    transformers submodule that ``_patch_tokenizer_for_vllm`` reaches into.
    """
    import sys

    # ``_patch_tokenizer_for_vllm`` does
    #   from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    # so we need a real (or fake) module at that import path.
    import types

    fake_vllm = types.SimpleNamespace(LLM=_FakeLLM, SamplingParams=_FakeSamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    # Skip the patch call entirely — it mutates a real HF base class and we
    # don't need it under the fake tokenizer. Stubbing the patch keeps the
    # test independent of the installed transformers version.
    from explore_persona_space.experiments.factor_screen_365 import eval_panel

    monkeypatch.setattr(eval_panel, "_patch_tokenizer_for_vllm", lambda: None)

    # Patch HF tokenizer load — the panel uses AutoTokenizer.from_pretrained.
    fake_auto = types.SimpleNamespace(from_pretrained=lambda *a, **k: _FakeTokenizer())
    fake_transformers = types.SimpleNamespace(AutoTokenizer=fake_auto)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_generate_completions_logs_three_init_lines_in_order(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The persona panel emits STARTING → instantiating → COMPLETE in order.

    The three lines must appear in this order and carry the cell key /
    source so per-cell stderr capture (Fix D) attributes them correctly.
    """
    _install_vllm_mocks(monkeypatch)

    cfg = EvalConfig(
        model_path="dummy/model-path",
        num_completions=1,
        max_new_tokens=64,
        max_model_len=512,
        personas={"persona_a": "system A"},
        questions=["q1"],
        cell_key="00010",
        source="librarian",
        seed=42,
    )

    # Round-10 (issue #365): init logging lives in `vllm_session` now, not
    # in `generate_completions`. Both eval panels share one vLLM instance,
    # so the three init lines fire once per cell from the context manager.
    # Round-11 (issue #365): the context manager yields a holder; `session.llm`
    # is the live engine while the block is open.
    with (
        caplog.at_level(logging.INFO, logger=eval_panel.log.name),
        vllm_session(
            model_path=cfg.model_path,
            max_model_len=cfg.max_model_len,
            seed=cfg.seed,
            cell_key=cfg.cell_key,
            source=cfg.source,
        ) as session,
    ):
        # Holder is populated while the context is open.
        assert session.llm is not None, "holder.llm must be live inside the with-block"

    # Pre-round-10 logs carried "persona-panel" / "random-ctrl" tags inside
    # the message; round-10 logs are unscoped (one session per cell). The
    # log-line *content* (STARTING / instantiating / COMPLETE + cell key
    # + source) is unchanged.
    persona_lines = [r.message for r in caplog.records]
    assert any("vLLM init STARTING" in m for m in persona_lines), (
        f"Expected 'vLLM init STARTING' log line; got {persona_lines}"
    )
    assert any("vLLM init: instantiating LLM" in m for m in persona_lines), (
        f"Expected 'instantiating LLM' log line; got {persona_lines}"
    )
    assert any("vLLM init COMPLETE" in m for m in persona_lines), (
        f"Expected 'vLLM init COMPLETE' log line; got {persona_lines}"
    )

    # Ordering check: STARTING must precede instantiating must precede COMPLETE.
    def _index_of(needle: str) -> int:
        for i, line in enumerate(persona_lines):
            if needle in line:
                return i
        raise AssertionError(f"line {needle!r} not found in {persona_lines}")

    start_i = _index_of("vLLM init STARTING")
    inst_i = _index_of("vLLM init: instantiating LLM")
    done_i = _index_of("vLLM init COMPLETE")
    assert start_i < inst_i < done_i, (
        f"Log lines out of order: STARTING@{start_i}, instantiating@{inst_i}, COMPLETE@{done_i}"
    )

    # Cell context must appear in the lines (so per-cell logs are attributable).
    assert any("00010" in m for m in persona_lines), "cell_key missing from persona-panel logs"
    assert any("librarian" in m for m in persona_lines), "source missing from persona-panel logs"


def test_generate_random_control_logs_three_init_lines(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The random-control panel also emits the three init lines."""
    _install_vllm_mocks(monkeypatch)

    cfg = RandomControlConfig(
        model_path="dummy/model-path",
        num_completions=1,
        max_new_tokens=64,
        max_model_len=512,
        prompts={"rc_a": "system rc"},
        questions=["q1"],
        cell_key="01010",
        source="surgeon",
        seed=99,
    )

    # Round-10 (issue #365): the random-control panel no longer instantiates
    # its own vLLM — it shares the cell's `vllm_session`. The three init
    # lines still fire (once per cell) and carry the cell context.
    # Round-11 (issue #365): context manager yields a holder; `session.llm`
    # is the live engine inside the block.
    with (
        caplog.at_level(logging.INFO, logger=eval_panel.log.name),
        vllm_session(
            model_path=cfg.model_path,
            max_model_len=cfg.max_model_len,
            seed=cfg.seed,
            cell_key=cfg.cell_key,
            source=cfg.source,
        ) as session,
    ):
        assert session.llm is not None, "holder.llm must be live inside the with-block"

    rc_lines = [r.message for r in caplog.records]
    assert any("vLLM init STARTING" in m for m in rc_lines), (
        f"Expected random-ctrl 'STARTING' log line; got {rc_lines}"
    )
    assert any("vLLM init: instantiating LLM" in m for m in rc_lines), (
        f"Expected random-ctrl 'instantiating LLM' log line; got {rc_lines}"
    )
    assert any("vLLM init COMPLETE" in m for m in rc_lines), (
        f"Expected random-ctrl 'COMPLETE' log line; got {rc_lines}"
    )
    assert any("01010" in m for m in rc_lines), "cell_key missing from random-ctrl logs"
    assert any("surgeon" in m for m in rc_lines), "source missing from random-ctrl logs"


def test_eval_config_carries_default_cell_context() -> None:
    """``cell_key`` / ``source`` default to ``"?"`` so existing call sites still work.

    Back-compat guard: round-9 added two new dataclass fields with safe
    defaults so any caller that hasn't been updated still produces valid
    log lines (just with ``?`` placeholders in the cell key).
    """
    cfg = EvalConfig(model_path="dummy")
    assert cfg.cell_key == "?"
    assert cfg.source == "?"

    rc_cfg = RandomControlConfig(model_path="dummy")
    assert rc_cfg.cell_key == "?"
    assert rc_cfg.source == "?"


def test_init_line_called_via_mock_llm_records_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: the LLM constructor receives the model path between the
    'instantiating' and 'COMPLETE' log lines.

    This guards against a future regression where the log lines fire but
    LLM() is silently no-op'd (e.g. someone wraps it in an `if dry_run` block).
    """
    _install_vllm_mocks(monkeypatch)

    cfg = EvalConfig(
        model_path="path/to/specific/merged_adapter",
        num_completions=1,
        max_new_tokens=64,
        max_model_len=1024,
        personas={"persona_a": "system A"},
        questions=["q1"],
        cell_key="11111",
        source="programmer",
        seed=7,
    )

    # Round-10: LLM() is constructed inside `vllm_session`, not inside
    # `generate_completions`. The construction-kwargs assertion still holds —
    # `vllm_session` forwards `model_path` and `max_model_len` to `LLM()`.
    # Round-11: the context manager yields a holder; `session.llm` is the
    # constructed engine inside the block.
    with vllm_session(
        model_path=cfg.model_path,
        max_model_len=cfg.max_model_len,
        seed=cfg.seed,
        cell_key=cfg.cell_key,
        source=cfg.source,
    ) as session:
        assert isinstance(session.llm, _FakeLLM), (
            "holder.llm must be the constructed (fake) LLM inside the with-block"
        )
    assert _FakeLLM.last_kwargs is not None, "LLM() was not actually invoked"
    assert _FakeLLM.last_kwargs["model"] == "path/to/specific/merged_adapter"
    assert _FakeLLM.last_kwargs["max_model_len"] == 1024


def test_vllm_session_clears_holder_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-11 (issue #365): the context manager must drop the LLM reference.

    Smoke-3 (round 10) caught a cross-cell GPU memory leak: cell 10010
    hit vLLM v1's startup memory guard with ~105 GB pinned by a prior
    cell's still-alive engine. Root cause: ``vllm_session`` yielded the
    raw ``LLM``, and the ``finally`` clause's ``del llm`` only dropped
    the generator-local binding — the caller's ``with ... as llm``
    binding kept the engine alive through ``__exit__``, so
    ``gc.collect()`` / ``empty_cache()`` ran with a live reference and
    freed nothing.

    Round-11 holder-pattern fix: ``vllm_session`` yields an
    ``_LLMHolder``. The finally clause sets ``holder.llm = None`` BEFORE
    ``gc.collect()``. After the with-block exits, the caller still holds
    ``session`` but ``session.llm`` is ``None`` — no path to the LLM,
    so the engine's GPU allocations are reachable only through whatever
    (zero) references vLLM holds internally.

    Test guarantee: after a normal ``with vllm_session(...) as session``
    exit, ``session.llm`` is ``None``.
    """
    _install_vllm_mocks(monkeypatch)

    cfg = EvalConfig(
        model_path="dummy/model-path",
        num_completions=1,
        max_new_tokens=32,
        max_model_len=256,
        personas={"persona_a": "system A"},
        questions=["q1"],
        cell_key="00000",
        source="librarian",
        seed=42,
    )

    with vllm_session(
        model_path=cfg.model_path,
        max_model_len=cfg.max_model_len,
        seed=cfg.seed,
        cell_key=cfg.cell_key,
        source=cfg.source,
    ) as session:
        assert session.llm is not None, "holder.llm must be live inside the with-block"
        live_llm = session.llm  # capture for the post-exit identity check

    # The caller's `session` binding survives __exit__ (per Python's `with`
    # semantics). What MUST change is `session.llm`: round-11's fix sets
    # it to None before gc.collect() so the engine has no strong reference
    # reachable through the holder. Without this, round-10's `del llm` was
    # a no-op and the engine stayed alive across cells (smoke-3 OOM).
    assert session.llm is None, (
        "vllm_session must clear holder.llm on exit so gc.collect() can "
        "release the engine. If this fires, round-10's cross-cell GPU "
        "memory leak (smoke-3 cell 10010 OOM at startup) has regressed."
    )
    # Sanity: the LLM was actually constructed (so `is None` reflects the
    # finally clause's clear, not a no-op path).
    assert isinstance(live_llm, _FakeLLM), "LLM was never constructed; holder=None test is vacuous"


def test_vllm_session_clears_holder_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-11 (issue #365): finally clause must clear holder even if the
    with-block raises.

    The dispatcher must not leak GPU memory across cells when generation
    fails partway through. Holder must be zeroed on the exception path
    too.
    """
    _install_vllm_mocks(monkeypatch)

    cfg = EvalConfig(
        model_path="dummy/model-path",
        max_model_len=256,
        cell_key="11000",
        source="surgeon",
        seed=42,
    )

    captured_session: dict[str, object | None] = {"session": None}

    class _Boom(RuntimeError):
        pass

    with (
        pytest.raises(_Boom),
        vllm_session(
            model_path=cfg.model_path,
            max_model_len=cfg.max_model_len,
            seed=cfg.seed,
            cell_key=cfg.cell_key,
            source=cfg.source,
        ) as session,
    ):
        captured_session["session"] = session
        assert session.llm is not None, "holder.llm must be live before raise"
        raise _Boom("simulated mid-cell eval failure")

    s = captured_session["session"]
    assert s is not None, "test bug: session never captured"
    assert s.llm is None, (
        "vllm_session must clear holder.llm in `finally` even when the "
        "with-block raises. Without this, a generation failure mid-cell "
        "would leave the engine alive across the (now-failed) cell "
        "boundary."
    )
