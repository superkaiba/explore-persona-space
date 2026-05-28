"""Regression tests for the LoRA-only training-finalize path (#7b).

The change adds an opt-in env flag ``EPM_MATERIALIZE_MERGED``:

- DEFAULT (unset or "1"): ``_finalize_phase`` merges the LoRA adapter into a
  full 7B checkpoint, uploads the MERGED model to WandB, deletes the adapter,
  returns the merged dir. This is the historical behavior and MUST stay
  byte-for-byte identical.
- "0": ``_finalize_phase`` keeps the adapter, skips the merge / merged-upload /
  adapter-delete, and returns the adapter dir (the MooseFS-quota disk win).

The decisive break-catchers are:

- ``test_default_preserves_merge_upload_delete_sequence`` — any change that
  alters the default merge/upload/delete sequence or the returned merged path
  fails immediately.
- ``test_eval_signatures_are_additive`` — the new adapter kwargs are optional
  and, when left at their None defaults, the eval entry points build the SAME
  vLLM / HF loader arguments as before (no ``enable_lora``, model == model_path).

All tests run without a GPU (monkeypatched loaders) and without network.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from explore_persona_space.train import trainer as trainer_mod

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _RecordingModel:
    """Minimal model stub that records save_pretrained calls."""

    def __init__(self) -> None:
        self.saved_to: list[str] = []

    def save_pretrained(self, path: str) -> None:
        self.saved_to.append(path)


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.saved_to: list[str] = []

    def save_pretrained(self, path: str) -> None:
        self.saved_to.append(path)


class _FakeTrainerState:
    def __init__(self) -> None:
        self.log_history: list = []
        self.global_step = 0
        self.epoch = 0.0


class _FakeTrainer:
    def __init__(self) -> None:
        self.state = _FakeTrainerState()


def _patch_finalize_side_effects(monkeypatch):
    """Patch the side-effecting calls in ``_finalize_phase`` and record them.

    Returns a dict of call recorders so each test can assert exactly which of
    merge / upload / rmtree ran.
    """
    calls: dict[str, list] = {"merge": [], "upload": [], "rmtree": [], "empty_cache": []}

    def _fake_merge_and_save(*, base_model_path, adapter_path, output_path, model_id):
        calls["merge"].append(
            {
                "base_model_path": base_model_path,
                "adapter_path": adapter_path,
                "output_path": output_path,
                "model_id": model_id,
            }
        )
        # Mirror the real return contract: the merged output path.
        return output_path

    def _fake_upload(checkpoint_path):
        calls["upload"].append(checkpoint_path)

    def _fake_rmtree(path, *args, **kwargs):
        calls["rmtree"].append(str(path))

    def _fake_empty_cache():
        calls["empty_cache"].append(True)

    monkeypatch.setattr(trainer_mod, "merge_and_save", _fake_merge_and_save)
    monkeypatch.setattr(trainer_mod, "_maybe_upload_checkpoint_to_wandb", _fake_upload)
    monkeypatch.setattr(trainer_mod, "_maybe_dump_train_log", lambda trainer, d: None)
    monkeypatch.setattr(trainer_mod.shutil, "rmtree", _fake_rmtree)
    monkeypatch.setattr(trainer_mod.torch.cuda, "empty_cache", _fake_empty_cache)
    return calls


def _call_finalize(tmp_path: Path):
    adapter_dir = tmp_path / "phase1_adapter"
    merged_dir = tmp_path / "phase1_merged"
    return (
        trainer_mod._finalize_phase(
            model=_RecordingModel(),
            tokenizer=_RecordingTokenizer(),
            trainer=_FakeTrainer(),
            adapter_dir=adapter_dir,
            merged_dir=merged_dir,
            base_model_for_merge="Qwen/Qwen2.5-7B",
            model_id="Qwen/Qwen2.5-7B",
        ),
        adapter_dir,
        merged_dir,
    )


# ---------------------------------------------------------------------------
# (1) Default preservation — the decisive break-catcher
# ---------------------------------------------------------------------------


def test_default_preserves_merge_upload_delete_sequence(tmp_path, monkeypatch):
    """With the flag unset, finalize merges, uploads merged, rmtrees the adapter,
    and returns the merged dir — exactly the historical behavior."""
    monkeypatch.delenv("EPM_MATERIALIZE_MERGED", raising=False)
    calls = _patch_finalize_side_effects(monkeypatch)

    returned, adapter_dir, merged_dir = _call_finalize(tmp_path)

    assert len(calls["merge"]) == 1, "default path must call merge_and_save exactly once"
    assert calls["merge"][0]["adapter_path"] == str(adapter_dir)
    assert calls["merge"][0]["output_path"] == str(merged_dir)
    assert len(calls["upload"]) == 1, "default path must upload the MERGED checkpoint"
    assert calls["upload"][0] == str(merged_dir)
    assert str(adapter_dir) in calls["rmtree"], "default path must rmtree the adapter dir"
    assert returned == str(merged_dir), "default path returns the merged dir"


def test_default_holds_when_flag_explicitly_one(tmp_path, monkeypatch):
    """EPM_MATERIALIZE_MERGED='1' is identical to unset."""
    monkeypatch.setenv("EPM_MATERIALIZE_MERGED", "1")
    calls = _patch_finalize_side_effects(monkeypatch)

    returned, _adapter_dir, merged_dir = _call_finalize(tmp_path)

    assert len(calls["merge"]) == 1
    assert len(calls["upload"]) == 1
    assert returned == str(merged_dir)


# ---------------------------------------------------------------------------
# (2) Flag-on — LoRA-only path keeps the adapter, skips merge/upload/delete
# ---------------------------------------------------------------------------


def test_flag_off_keeps_adapter_and_skips_merge(tmp_path, monkeypatch):
    """EPM_MATERIALIZE_MERGED='0' keeps the adapter, never merges/uploads/deletes,
    and returns the adapter dir."""
    monkeypatch.setenv("EPM_MATERIALIZE_MERGED", "0")
    calls = _patch_finalize_side_effects(monkeypatch)

    returned, adapter_dir, _merged_dir = _call_finalize(tmp_path)

    assert calls["merge"] == [], "LoRA-only path must NOT merge"
    assert calls["upload"] == [], "LoRA-only path must NOT upload a merged checkpoint"
    assert calls["rmtree"] == [], "LoRA-only path must NOT delete the adapter dir"
    assert returned == str(adapter_dir), "LoRA-only path returns the ADAPTER dir"


def test_should_materialize_merged_reads_env(monkeypatch):
    """The gate helper reads the env exactly: only '0' flips it off."""
    monkeypatch.delenv("EPM_MATERIALIZE_MERGED", raising=False)
    assert trainer_mod._should_materialize_merged() is True
    monkeypatch.setenv("EPM_MATERIALIZE_MERGED", "1")
    assert trainer_mod._should_materialize_merged() is True
    monkeypatch.setenv("EPM_MATERIALIZE_MERGED", "0")
    assert trainer_mod._should_materialize_merged() is False
    # Any other value defaults to the safe (materialize) behavior.
    monkeypatch.setenv("EPM_MATERIALIZE_MERGED", "true")
    assert trainer_mod._should_materialize_merged() is True


# ---------------------------------------------------------------------------
# (3) Additive eval signatures — new kwargs optional + None == legacy build args
# ---------------------------------------------------------------------------


def test_eval_signatures_accept_adapter_kwargs():
    """generation / alignment / capability entry points accept the new optional
    adapter kwargs, all defaulting to None / 32 so existing callers are
    unaffected."""
    from explore_persona_space.eval import alignment, capability, generation

    gen_specs = {
        generation.generate_completions: ("lora_adapter_path", "base_model_path", "max_lora_rank"),
        generation.generate_persona_completions: (
            "lora_adapter_path",
            "base_model_path",
            "max_lora_rank",
        ),
        generation.generate_completions_with_history: (
            "lora_adapter_path",
            "base_model_path",
            "max_lora_rank",
        ),
        alignment.generate_alignment_completions: ("lora_adapter_path", "base_model_path"),
        alignment.evaluate_alignment: ("lora_adapter_path", "base_model_path"),
        alignment.evaluate_alignment_quick: ("lora_adapter_path", "base_model_path"),
        capability.evaluate_capability_logprob: ("adapter_path", "base_model_path"),
    }
    for fn, names in gen_specs.items():
        params = inspect.signature(fn).parameters
        for name in names:
            assert name in params, f"{fn.__name__} is missing optional kwarg {name!r}"
        # The adapter-selecting kwargs default to None (off).
        for name in names:
            if name in ("lora_adapter_path", "base_model_path", "adapter_path"):
                assert params[name].default is None, (
                    f"{fn.__name__}.{name} must default to None so legacy callers "
                    "get unchanged behavior"
                )


def test_resolve_adapter_load_is_byte_identical_when_off():
    """generation._resolve_adapter_load returns (model_path, None) when no adapter
    is requested — i.e. the legacy LLM(model=model_path) build with no LoRA."""
    from explore_persona_space.eval.generation import _resolve_adapter_load

    engine_path, lora_request = _resolve_adapter_load("/merged/model", None, None)
    assert engine_path == "/merged/model"
    assert lora_request is None


def test_capability_logprob_fails_loud_on_adapter_without_base():
    """Adapter mode with no base model is a loud ValueError, not a silent merge."""
    from explore_persona_space.eval.capability import evaluate_capability_logprob

    with pytest.raises(ValueError, match="base_model_path"):
        evaluate_capability_logprob(
            model_path="/some/path",
            output_dir="/tmp/does-not-matter",
            adapter_path="/some/adapter",
            base_model_path=None,
        )


def test_generation_fails_loud_on_adapter_without_base():
    """generation._resolve_adapter_load fails loud when adapter is set but base
    is missing — no silent fallback to loading the adapter dir as a full model."""
    from explore_persona_space.eval.generation import _resolve_adapter_load

    with pytest.raises(ValueError, match="base_model_path"):
        _resolve_adapter_load("/merged", "/adapter", None)


# ---------------------------------------------------------------------------
# (4) Two-phase guard — LoRA-only on a two-phase condition fails loud
# ---------------------------------------------------------------------------


class _FakeCondition(dict):
    """Dict that supports the OmegaConf-style .get used in run_two_phase_training."""

    def get(self, key, default=None):
        return super().get(key, default)


def _two_phase_cfg():
    from types import SimpleNamespace

    condition = _FakeCondition(
        name="c1_evil_wrong_em",
        phase1_dataset="data/sft/phase1.jsonl",
        phase2_dataset="data/sft/phase2.jsonl",
    )
    cfg = SimpleNamespace(condition=condition, training=SimpleNamespace(model_id="Qwen/Qwen2.5-7B"))
    cfg.get = lambda key, default=None: default  # wandb_project etc. -> None
    return cfg


def test_two_phase_guard_raises_under_lora_only(monkeypatch):
    """EPM_MATERIALIZE_MERGED='0' on a both-phases condition raises before any
    training — no silent merged-dir fallback, no silent Phase-2-from-base."""
    monkeypatch.setenv("EPM_MATERIALIZE_MERGED", "0")

    # Sentinel so we can prove training never started.
    def _boom(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("_prepare_run_dir should never be called once the guard fires")

    monkeypatch.setattr(trainer_mod, "_prepare_run_dir", _boom)

    with pytest.raises(ValueError, match="two-phase"):
        trainer_mod.run_two_phase_training(cfg=_two_phase_cfg(), seed=42)


def test_two_phase_guard_inactive_by_default(monkeypatch):
    """With the flag unset, the two-phase guard does NOT fire (default path).

    We don't run real training; we patch _prepare_run_dir to raise a distinct
    sentinel so reaching it proves the guard let us through.
    """
    monkeypatch.delenv("EPM_MATERIALIZE_MERGED", raising=False)

    class _Sentinel(RuntimeError):
        pass

    def _reached(*args, **kwargs):
        raise _Sentinel("guard passed; reached _prepare_run_dir")

    monkeypatch.setattr(trainer_mod, "_prepare_run_dir", _reached)

    with pytest.raises(_Sentinel):
        trainer_mod.run_two_phase_training(cfg=_two_phase_cfg(), seed=42)


# ---------------------------------------------------------------------------
# (5) Alignment invariant — Betley path injects NO system prompt
# ---------------------------------------------------------------------------


def test_alignment_completions_uses_no_system_prompt(monkeypatch):
    """generate_alignment_completions calls generate_completions WITHOUT a
    system_prompt (the Betley questions are bare), and forwards the adapter
    kwargs unchanged. Persona-injection-via-system-prompt does not apply to this
    path; this test pins that no system prompt is accidentally introduced."""
    from explore_persona_space.eval import alignment

    captured: dict = {}

    def _fake_generate(**kwargs):
        captured.update(kwargs)
        return {p: ["x"] for p in kwargs["prompts"]}

    monkeypatch.setattr(
        "explore_persona_space.eval.generation.generate_completions", _fake_generate
    )

    out = alignment.generate_alignment_completions(
        model_path="/merged/model",
        prompts=["Q1", "Q2"],
        num_samples=10,
        seed=42,
    )
    assert set(out) == {"Q1", "Q2"}
    # No system_prompt was passed (legacy default is None -> bare questions).
    assert "system_prompt" not in captured or captured.get("system_prompt") is None
    # Adapter kwargs default to None (legacy behavior).
    assert captured.get("lora_adapter_path") is None
    assert captured.get("base_model_path") is None
    assert captured.get("num_completions") == 10
    assert captured.get("seed") == 42
