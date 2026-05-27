"""Round 11 in-process sweep contract (task #397).

Round 11 abandoned the round-5..10 subprocess-pool design after five
rounds of cascading bugs (smoke gate, HF→vLLM OOM, task.py shellouts,
missing HF upload, missing .env loading) — every bug traced back to the
subprocess crossing trust boundaries (env propagation, branch-guard,
upload silent-swallow). Round 11 in-processed the sweep: each cell runs
end-to-end in the dispatcher process, calling the same pipeline the
proven smoke phase uses.

This file pins the in-process per-cell pipeline order. The smoke phase
already has its own wiring test (see ``test_factor_screen_397_dispatcher_
wiring.py::test_smoke_phase_threads_system_prompt_overrides_into_compute_
logprob_panel``); this file covers the sweep-side equivalent —
``_run_one_cell_inprocess`` — and the sweep-loop hand-off.

Test surface:

  1. ``_run_one_cell_inprocess`` calls the heavy modules in the
     canonical order:

       prepare_cell_jsonl → train_one_cell → compute_logprob_panel →
       _aggressive_hf_to_vllm_teardown → generate_completions_with_lora →
       (write metrics.json) → verify_adapter_on_hf_hub →
       cleanup_cell_local_weights → rc=0

  2. ``_run_one_cell_inprocess`` returns rc=2 + SKIPS cleanup when
     verify_adapter_on_hf_hub returns False (CLAUDE.md upload-policy
     contract: never delete local weights when HF Hub doesn't have the
     adapter).

  3. The sweep loop's per-cell try/except catches a Python exception
     out of ``_run_one_cell_inprocess`` as rc=1 and continues — covered
     in ``test_factor_screen_397_dispatcher_wiring.py::test_sweep_phase
     _catches_inprocess_exceptions_as_rc1``.

CPU-only; no GPU, no model load. All heavy entry points are
monkeypatched at the module level.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Load the dispatcher (lives under scripts/, not a package).
_DISPATCH_PATH = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "dispatch_factor_screen_397.py"
)
_spec = importlib.util.spec_from_file_location("dispatch_factor_screen_397", _DISPATCH_PATH)
_dispatch = importlib.util.module_from_spec(_spec)
sys.modules["dispatch_factor_screen_397"] = _dispatch
_spec.loader.exec_module(_dispatch)


def _build_cell_args(slab_root: Path) -> argparse.Namespace:
    """Build the args namespace ``_run_one_cell_inprocess`` expects.

    Mirrors the parser defaults so the test stays in sync with the CLI.
    """
    return argparse.Namespace(
        issue=397,
        mode="sweep",
        pool_dir=slab_root / "pools",
        slab_root=slab_root,
        smoke_cell="10010",
        smoke_source="librarian",
        smoke_seed=42,
        sources="librarian",
        seeds="42",
        marker_token="※",
        save_every_n_steps=25,
        pos_per_source=400,
        lr=1e-4,
        warmup_ratio=0.10,
        require_smoke_pass=True,
        skip_smoke_pass_check=False,
        smoke_pass_confirmed=True,
        dry_run=False,
        no_resume=True,
        resume_source="both",
        log_level="INFO",
    )


def _stub_heavy_dependencies(monkeypatch, *, verify_passes: bool = True) -> dict:
    """Monkeypatch every heavy entry point ``_run_one_cell_inprocess`` calls
    so the test can run on CPU in <1 second without loading any model.

    Returns a dict of call-record lists the caller can assert on. The
    keys are the names of the heavy entry points being stubbed.

    The stubs are intentionally minimal — they make ``_run_one_cell_
    inprocess`` traverse its full pipeline without doing real work, so
    the call-order assertion can pin the sequence.
    """
    records: dict[str, list] = {
        "prepare_cell_jsonl": [],
        "train_one_cell": [],
        "compute_logprob_panel": [],
        "generate_completions_with_lora": [],
        "verify": [],
        "cleanup": [],
        "order": [],  # interleaved order: which entry point fired when
    }

    FAKE_SYSTEM_PROMPT = "FAKE-SYSTEM-PROMPT-FROM-STUBBED-PREPARE-CELL-JSONL"

    # ----- (1) prepare_cell_jsonl -----
    def _fake_prepare_cell_jsonl(**kwargs):
        records["prepare_cell_jsonl"].append(kwargs)
        records["order"].append("prepare_cell_jsonl")
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"messages": []}\n', encoding="utf-8")
        return {
            "output_path": output_path,
            "num_positive": 400,
            "num_negative": 400,
            "num_total": 800,
            "data_policy": "off_policy",
            "system_prompt_text": FAKE_SYSTEM_PROMPT,
        }

    import explore_persona_space.experiments.factor_screen_397.data_prep as dp_mod

    monkeypatch.setattr(dp_mod, "prepare_cell_jsonl", _fake_prepare_cell_jsonl)

    # ----- (2) train_one_cell + write manifest + create checkpoint dir -----
    def _fake_train_one_cell(**kwargs):
        from explore_persona_space.experiments.factor_screen_397.training import (
            TrainOutcome,
            write_prepared_dataset_manifest,
        )

        records["train_one_cell"].append(kwargs)
        records["order"].append("train_one_cell")
        write_prepared_dataset_manifest(
            kwargs["cell_output_dir"],
            cell_key=kwargs["cell"].key,
            source=kwargs["source"],
            seed=kwargs["seed"],
            system_prompt_text=kwargs["system_prompt_text"],
            marker_text=kwargs["marker_text"],
        )
        # Create a synthetic checkpoint-25 dir so _enumerate_checkpoint_dirs
        # returns at least one entry.
        (kwargs["cell_output_dir"] / "adapter" / "checkpoint-25").mkdir(parents=True)
        return TrainOutcome(
            cell_key=kwargs["cell"].key,
            seed=kwargs["seed"],
            adapter_path=str(kwargs["cell_output_dir"] / "adapter"),
            loss=1.23,
            train_wall_minutes=0.5,
            n_examples=800,
            total_steps=150,
            marker_only_loss=True,
            marker_tail_tokens=0,
        )

    import explore_persona_space.experiments.factor_screen_397.training as training_mod

    monkeypatch.setattr(training_mod, "train_one_cell", _fake_train_one_cell)

    # ----- (3) stub the model loader (returns MagicMocks, no GPU) -----
    monkeypatch.setattr(
        _dispatch,
        "_load_base_model_for_logprob",
        lambda first_checkpoint_dir: (
            MagicMock(name="base_model"),
            MagicMock(name="tokenizer"),
        ),
    )

    # ----- (4) compute_logprob_panel -----
    def _fake_compute_logprob_panel(**kwargs):
        records["compute_logprob_panel"].append(kwargs)
        records["order"].append("compute_logprob_panel")
        return {kwargs["checkpoint_dirs"][0]: {"※": [-1.0]}}

    import explore_persona_space.experiments.factor_screen_397.eval_panel as ep_mod

    monkeypatch.setattr(ep_mod, "compute_logprob_panel", _fake_compute_logprob_panel)

    # ----- (5) HF teardown (stub — no torch / gc work) -----
    def _fake_teardown(*args, **kwargs):
        records["order"].append("aggressive_hf_to_vllm_teardown")

    monkeypatch.setattr(_dispatch, "_aggressive_hf_to_vllm_teardown", _fake_teardown)

    # ----- (6) generate_completions_with_lora (no vLLM) -----
    def _fake_generate_completions_with_lora(**kwargs):
        records["generate_completions_with_lora"].append(kwargs)
        records["order"].append("generate_completions_with_lora")
        # Shape mirrors the real return: persona → question → completions list.
        return {
            persona: {q: ["fake completion text"] for q in kwargs["questions"]}
            for persona in kwargs["personas"]
        }

    monkeypatch.setattr(
        ep_mod, "generate_completions_with_lora", _fake_generate_completions_with_lora
    )

    # ----- (7) score_markers_threaded (no real scoring) -----
    def _fake_score_markers_threaded(completions, *, marker):
        return {
            persona: {
                "substring_rate": 0.5,
                "fuzzy_rate": 0.5,
                "substring_found": 50,
                "fuzzy_found": 50,
                "total": 100,
                "per_question": {},
            }
            for persona in completions
        }

    monkeypatch.setattr(ep_mod, "score_markers_threaded", _fake_score_markers_threaded)

    # ----- (8) verify_adapter_on_hf_hub -----
    def _fake_verify(**kwargs):
        records["verify"].append(kwargs)
        records["order"].append("verify_adapter_on_hf_hub")
        return verify_passes

    monkeypatch.setattr(_dispatch, "verify_adapter_on_hf_hub", _fake_verify)

    # ----- (9) cleanup_cell_local_weights -----
    def _fake_cleanup(cell_output_dir):
        records["cleanup"].append(cell_output_dir)
        records["order"].append("cleanup_cell_local_weights")
        return {"merged_removed": 0, "checkpoints_removed": 1}

    monkeypatch.setattr(_dispatch, "cleanup_cell_local_weights", _fake_cleanup)

    # ----- (10) AutoTokenizer (caller imports it inside the function) -----
    fake_tokenizer = MagicMock(name="autotok")
    fake_tokenizer.pad_token_id = 0  # so the "if pad_token_id is None" branch skips

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return fake_tokenizer

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeAutoTokenizer)

    return records


def test_run_one_cell_inprocess_calls_pipeline_in_canonical_order(monkeypatch) -> None:
    """Round 11 contract: ``_run_one_cell_inprocess`` traverses its
    pipeline in the canonical order:

      prepare_cell_jsonl → train_one_cell → compute_logprob_panel →
      _aggressive_hf_to_vllm_teardown → generate_completions_with_lora →
      verify_adapter_on_hf_hub → cleanup_cell_local_weights

    Inverting any pair (cleanup before verify, vLLM before HF teardown,
    train before prepare, etc.) re-introduces one of the bugs round 11
    is designed to eliminate.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        args = _build_cell_args(slab_root)
        records = _stub_heavy_dependencies(monkeypatch, verify_passes=True)

        cell = Cell.from_key("00000")
        rc = _dispatch._run_one_cell_inprocess(cell=cell, source="librarian", seed=42, args=args)
        assert rc == 0, f"PASS path must return rc=0; got {rc}"

        # Canonical pipeline order.
        assert records["order"] == [
            "prepare_cell_jsonl",
            "train_one_cell",
            "compute_logprob_panel",
            "aggressive_hf_to_vllm_teardown",
            "generate_completions_with_lora",
            "verify_adapter_on_hf_hub",
            "cleanup_cell_local_weights",
        ], f"Pipeline ran out of order: {records['order']}"

        # Sanity: metrics.json was written.
        metrics_path = slab_root / "cell_00000" / "source_librarian" / "seed_42" / "metrics.json"
        assert metrics_path.exists(), (
            f"_run_one_cell_inprocess must write metrics.json; missing {metrics_path}"
        )
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert payload["cell_key"] == "00000"
        assert payload["source"] == "librarian"
        assert payload["seed"] == 42
        assert payload["marker"] == "※"
        assert payload["vllm_lora_mode"] is True


def test_run_one_cell_inprocess_returns_rc2_and_skips_cleanup_on_verify_fail(monkeypatch) -> None:
    """CLAUDE.md upload policy: "Models MUST upload to HF model repo
    before local deletion. Never delete unuploaded."

    When ``verify_adapter_on_hf_hub`` returns False (adapter missing
    from Hub), ``_run_one_cell_inprocess`` MUST:

      - return rc=2 (signals per-cell failure for the sweep summary)
      - NOT call cleanup_cell_local_weights (local weights preserved
        for manual recovery)
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        args = _build_cell_args(slab_root)
        records = _stub_heavy_dependencies(monkeypatch, verify_passes=False)

        cell = Cell.from_key("00000")
        rc = _dispatch._run_one_cell_inprocess(cell=cell, source="librarian", seed=42, args=args)
        assert rc == 2, f"verify-FAIL path must return rc=2; got {rc}"

        # Cleanup MUST NOT have run.
        assert len(records["cleanup"]) == 0, (
            "CLAUDE.md upload policy violated: cleanup_cell_local_weights was "
            "called even though verify_adapter_on_hf_hub returned False."
        )
        # Verify WAS called (we got the False signal from somewhere).
        assert len(records["verify"]) == 1


def test_run_one_cell_inprocess_threads_system_prompt_overrides_into_compute_logprob_panel(
    monkeypatch,
) -> None:
    """SR1 wiring assertion (reconciler SR1): the in-process per-cell
    pipeline MUST call ``compute_logprob_panel(...,
    system_prompt_overrides=overrides)`` so C=1 cells avoid the
    train/eval mismatch the recipe-fix was designed to eliminate.

    The smoke phase's equivalent test
    (``test_smoke_phase_threads_system_prompt_overrides_into_compute_
    logprob_panel`` in dispatcher_wiring.py) covers the smoke side; this
    test covers the sweep side. Both must wire the same way.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        args = _build_cell_args(slab_root)
        records = _stub_heavy_dependencies(monkeypatch, verify_passes=True)

        cell = Cell.from_key("00000")
        rc = _dispatch._run_one_cell_inprocess(cell=cell, source="librarian", seed=42, args=args)
        assert rc == 0

        # compute_logprob_panel got called with the overrides kwarg.
        assert len(records["compute_logprob_panel"]) == 1
        kwargs = records["compute_logprob_panel"][0]
        assert "system_prompt_overrides" in kwargs, (
            "SR1: _run_one_cell_inprocess called compute_logprob_panel WITHOUT "
            "system_prompt_overrides; this re-introduces the train/eval mismatch."
        )
        # The override dict for the source must match the FAKE_SYSTEM_PROMPT
        # the stubbed prepare_cell_jsonl returned (proves the manifest path
        # flows end-to-end through the in-process pipeline).
        FAKE_SYSTEM_PROMPT = "FAKE-SYSTEM-PROMPT-FROM-STUBBED-PREPARE-CELL-JSONL"
        assert kwargs["system_prompt_overrides"] == {"librarian": FAKE_SYSTEM_PROMPT}, (
            f"SR1: expected overrides to carry training-time prompt for "
            f"librarian; got {kwargs['system_prompt_overrides']}"
        )


def test_run_one_cell_inprocess_threads_overrides_into_sampled_eval(monkeypatch) -> None:
    """SR1 wiring carries through to the vLLM sampled eval as well.
    generate_completions_with_lora MUST receive ``system_prompt_overrides``
    from the recipe-fix manifest path.
    """
    from explore_persona_space.experiments.factor_screen_397.cells import Cell

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        args = _build_cell_args(slab_root)
        records = _stub_heavy_dependencies(monkeypatch, verify_passes=True)

        cell = Cell.from_key("00000")
        _dispatch._run_one_cell_inprocess(cell=cell, source="librarian", seed=42, args=args)

        assert len(records["generate_completions_with_lora"]) == 1
        kwargs = records["generate_completions_with_lora"][0]
        assert "system_prompt_overrides" in kwargs
        FAKE_SYSTEM_PROMPT = "FAKE-SYSTEM-PROMPT-FROM-STUBBED-PREPARE-CELL-JSONL"
        assert kwargs["system_prompt_overrides"] == {"librarian": FAKE_SYSTEM_PROMPT}
        assert kwargs["seed"] == 42
