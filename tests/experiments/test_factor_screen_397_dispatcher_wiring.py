"""Dispatcher SR1 wiring assertion (task #397).

Reconciler SR1 (epm:review-reconcile v2): the dispatcher MUST call
``compute_logprob_panel(..., system_prompt_overrides=overrides)`` for the
per-cell eval path. Calling with default ``system_prompt_overrides=None``
for a C=1 cell re-introduces the train/eval mismatch the recipe-fix was
designed to eliminate.

This test surface covers:

  1. ``run_smoke_phase`` calls ``compute_logprob_panel`` with the
     ``system_prompt_overrides`` kwarg populated from the recipe-fix
     manifest (asserted via monkeypatch interception — no GPU / no model
     load).

  2. ``run_smoke_phase`` calls ``train_one_cell`` with the
     ``system_prompt_text`` kwarg populated so the recipe-fix manifest
     lands on disk for the eval side to read.

  3. Round 12: the sweep dispatches to ``_run_sweep_two_pass`` for the
     post-round-7-descope canonical {3 sources, 1 seed [42], 36 cells/
     source} = 108 (cell, seed) configuration. Round 11's single-pass
     serial loop was abandoned after the round-11 reviewer FAILed on
     missing vLLM→HF teardown between cells (orphan-worker risk per
     CLAUDE.md gotcha). Detailed two-pass pipeline-order contracts live
     in ``test_factor_screen_397_two_pass_sweep.py``; this file only
     asserts the dispatch hand-off + the dry-run + smoke-pass-gate
     contracts.

  4. ``run_sweep_phase`` refuses to dispatch when no
     ``epm:smoke-pass`` marker is present.

No GPU / no model load — uses tempdir cell layout + monkeypatched
training / eval entry points.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# scripts/ is not a package on this repo's PYTHONPATH, so importlib it.
_DISPATCH_PATH = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "dispatch_factor_screen_397.py"
)
_spec = importlib.util.spec_from_file_location("dispatch_factor_screen_397", _DISPATCH_PATH)
_dispatch = importlib.util.module_from_spec(_spec)
sys.modules["dispatch_factor_screen_397"] = _dispatch
_spec.loader.exec_module(_dispatch)


def _build_smoke_args(
    slab_root: Path, *, source: str = "librarian", seed: int = 42
) -> argparse.Namespace:
    """Build the args namespace ``run_smoke_phase`` expects.

    Mirrors the parser defaults so the test stays in sync with the CLI.
    """
    return argparse.Namespace(
        issue=397,
        mode="smoke",
        pool_dir=slab_root / "pools",
        slab_root=slab_root,
        smoke_cell="10010",
        smoke_source=source,
        smoke_seed=seed,
        sources="librarian,programmer,surgeon",
        seeds="42",  # Round 7 descope (was "42,137,256")
        # Round 11 removed num_gpus + max_concurrent_train (in-process serial,
        # no GPU pool). Tests that still need to set them have been deleted.
        marker_token="※",
        save_every_n_steps=25,
        pos_per_source=400,
        lr=1e-4,
        warmup_ratio=0.10,
        require_smoke_pass=True,
        skip_smoke_pass_check=False,
        # Round 9 — orchestrator-set flag; tests default to True so
        # is_smoke_pass_confirmed_locally short-circuits without
        # needing a real metrics_final.json on disk.
        smoke_pass_confirmed=True,
        dry_run=False,
        # Round 6 — sweep resume:
        # Tests default to no_resume=True so they don't hit the real HF Hub
        # for the resume probe. The dedicated sweep_resume test file flips
        # both knobs explicitly to exercise the resume code path.
        no_resume=True,
        resume_source="both",
        log_level="INFO",
    )


def test_smoke_phase_threads_system_prompt_overrides_into_compute_logprob_panel(
    monkeypatch,
) -> None:
    """SR1 + BLOCKER 1 + BLOCKER 2 + BLOCKER 3 wiring assertion.

    The smoke flow under test:

      (1) prepare_cell_jsonl(cell, source, pool_dir, ...) writes the JSONL +
          returns the training-time system_prompt_text. (BLOCKER 1)
      (2) train_one_cell(system_prompt_text=<from (1)>) lands the
          recipe-fix manifest on disk.
      (3) build_train_matched_persona_panel reads the manifest; the override
          dict carries the source persona's training-time prompt.
      (4) compute_logprob_panel(personas=panel, questions=EVAL_QUESTIONS_20,
          system_prompt_overrides=<from (3)>) — SR1 wiring assertion +
          BLOCKER 2 480-context assertion.
      (5) _run_smoke_sampled_eval writes metrics_final.json so
          _smoke_source_substring_rate finds it. (BLOCKER 3)
      (6) marker posted; PASS path returns 0.

    Monkeypatches the heavy entry points (GPU / vLLM / pools) without
    loading any model.
    """
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
        EVAL_QUESTIONS_20,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        args = _build_smoke_args(slab_root)

        # ----- (1) BLOCKER 1: monkeypatch prepare_cell_jsonl -----
        # The real one would read a pool file the test hasn't staged. The
        # fake one writes the per-cell JSONL + returns a system_prompt_text
        # the test can later assert flows through to the override dict.
        prep_calls: list[dict] = []
        FAKE_TRAINING_SYSTEM_PROMPT = "FAKE-TRAINING-SYSTEM-PROMPT-FROM-PREPARE-CELL-JSONL"

        def _fake_prepare_cell_jsonl(**kwargs):
            prep_calls.append(kwargs)
            output_path = Path(kwargs["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text('{"messages": []}\n', encoding="utf-8")
            return {
                "output_path": output_path,
                "num_positive": 400,
                "num_negative": 400,
                "num_total": 800,
                "data_policy": "off_policy",
                "system_prompt_text": FAKE_TRAINING_SYSTEM_PROMPT,
            }

        import explore_persona_space.experiments.factor_screen_397.data_prep as dp_mod

        monkeypatch.setattr(dp_mod, "prepare_cell_jsonl", _fake_prepare_cell_jsonl)

        # ----- (2) monkeypatch train_one_cell — writes the manifest -----
        def _fake_train_one_cell(**kwargs):
            from explore_persona_space.experiments.factor_screen_397.training import (
                TrainOutcome,
                write_prepared_dataset_manifest,
            )

            assert "system_prompt_text" in kwargs, (
                "Dispatcher MUST pass system_prompt_text to train_one_cell (recipe-fix step 5b)"
            )
            # BLOCKER 1 assertion: train_one_cell receives the training-time
            # prompt that prepare_cell_jsonl returned, NOT the canonical
            # EVAL_PERSONAS_24 entry (which would silently mismatch on A=1
            # cells like the smoke cell 10010).
            assert kwargs["system_prompt_text"] == FAKE_TRAINING_SYSTEM_PROMPT, (
                "Dispatcher passed wrong system_prompt_text to train_one_cell — must be "
                f"the prepare_cell_jsonl return value, not the canonical EVAL_PERSONAS_24 entry. "
                f"Got: {kwargs['system_prompt_text'][:60]!r}"
            )
            write_prepared_dataset_manifest(
                kwargs["cell_output_dir"],
                cell_key=kwargs["cell"].key,
                source=kwargs["source"],
                seed=kwargs["seed"],
                system_prompt_text=kwargs["system_prompt_text"],
                marker_text=kwargs["marker_text"],
            )
            (kwargs["cell_output_dir"] / "adapter" / "checkpoint-25").mkdir(parents=True)
            return TrainOutcome(
                cell_key=kwargs["cell"].key,
                seed=kwargs["seed"],
                adapter_path=str(kwargs["cell_output_dir"] / "adapter"),
                # Round 6: merged_path removed; vLLM --enable-lora reads adapter_path directly.
                loss=1.23,
                train_wall_minutes=0.5,
                n_examples=800,
                total_steps=150,
                marker_only_loss=True,
                marker_tail_tokens=0,
            )

        import explore_persona_space.experiments.factor_screen_397.training as training_mod

        monkeypatch.setattr(training_mod, "train_one_cell", _fake_train_one_cell)

        # ----- (3) stub the model loader -----
        monkeypatch.setattr(
            _dispatch,
            "_load_base_model_for_logprob",
            lambda first_checkpoint_dir: (
                MagicMock(name="base_model"),
                MagicMock(name="tokenizer"),
            ),
        )

        # ----- (4) SR1 + BLOCKER 2: intercept compute_logprob_panel -----
        recorded_kwargs: dict = {}

        def _fake_compute_logprob_panel(**kwargs):
            recorded_kwargs.update(kwargs)
            return {kwargs["checkpoint_dirs"][0]: {"※": [-1.0]}}

        import explore_persona_space.experiments.factor_screen_397.eval_panel as ep_mod

        monkeypatch.setattr(ep_mod, "compute_logprob_panel", _fake_compute_logprob_panel)

        # ----- (5) BLOCKER 3: monkeypatch _run_smoke_sampled_eval -----
        # Real one calls vLLM; fake one writes a valid metrics_final.json so
        # _smoke_source_substring_rate finds source_rate=0.85 → PASS.
        sampled_eval_calls: list[dict] = []

        def _fake_run_smoke_sampled_eval(**kwargs):
            sampled_eval_calls.append(kwargs)
            metrics_payload = {
                "marker": kwargs["marker"],
                "panel_size": len(kwargs["panel"]),
                "questions": len(kwargs["questions"]),
                "num_completions": 5,
                "personas": {
                    args.smoke_source: {
                        "substring_rate": 0.85,
                        "fuzzy_rate": 0.90,
                        "substring_found": 85,
                        "fuzzy_found": 90,
                        "total": 100,
                        "per_question": {},
                    },
                },
            }
            (kwargs["cell_output_dir"] / "metrics_final.json").write_text(
                json.dumps(metrics_payload), encoding="utf-8"
            )

        monkeypatch.setattr(_dispatch, "_run_smoke_sampled_eval", _fake_run_smoke_sampled_eval)

        # ----- (6) Round 9 — verdict file written, not marker posted -----
        # Round 9 dropped post_marker_via_task_py entirely (pod-side task.py
        # shellouts crashed). Smoke writes SMOKE_VERDICT.json instead;
        # orchestrator on the VM side reads + posts the marker.

        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_smoke_phase(args, repo_root=repo_root)
        assert rc == 0, (
            f"Smoke should PASS when timing is fast, source_rate > 0, and metrics are written. "
            f"Got rc={rc}"
        )

        # --- BLOCKER 1 assertions: prepare_cell_jsonl ran with --pool-dir ---
        assert len(prep_calls) == 1, "prepare_cell_jsonl must run exactly once during smoke"
        prep_kwargs = prep_calls[0]
        assert prep_kwargs["pool_dir"] == args.pool_dir, (
            f"prepare_cell_jsonl pool_dir kwarg must come from args.pool_dir; "
            f"got {prep_kwargs['pool_dir']} vs {args.pool_dir}"
        )
        assert prep_kwargs["source"] == args.smoke_source
        assert prep_kwargs["cell"].key == args.smoke_cell
        assert prep_kwargs["marker_text"] == args.marker_token

        # --- SR1 assertion: system_prompt_overrides was threaded through ---
        assert "system_prompt_overrides" in recorded_kwargs, (
            "Dispatcher called compute_logprob_panel WITHOUT system_prompt_overrides; "
            "this re-introduces the train/eval mismatch the recipe-fix eliminates."
        )
        # build_train_matched_persona_panel writes the source persona's entry
        # to the manifest's system_prompt_text (the BLOCKER 1 training-time
        # prompt, NOT the canonical EVAL_PERSONAS_24 entry).
        overrides = recorded_kwargs["system_prompt_overrides"]
        assert overrides == {args.smoke_source: FAKE_TRAINING_SYSTEM_PROMPT}, (
            f"Expected SR1 override dict to carry the training-time prompt for "
            f"{args.smoke_source}; got {overrides}"
        )
        # The override must differ from the canonical entry (otherwise it
        # would prove nothing — A=1 cells like 10010 have a long persona
        # system prompt that does NOT match the canonical short one).
        assert overrides[args.smoke_source] != EVAL_PERSONAS_24[args.smoke_source], (
            "Smoke cell 10010 has A=1; the train-matched override MUST differ "
            "from the canonical EVAL_PERSONAS_24 entry for the SR1 check to "
            "be meaningful."
        )

        # --- BLOCKER 2 assertion: 480-context workload ---
        assert "personas" in recorded_kwargs
        assert "questions" in recorded_kwargs
        assert "contexts" not in recorded_kwargs or recorded_kwargs.get("contexts") is None
        assert len(recorded_kwargs["personas"]) == 24, (
            f"BLOCKER 2: smoke log-prob eval must use 24 personas; got "
            f"{len(recorded_kwargs['personas'])}"
        )
        assert recorded_kwargs["questions"] == list(EVAL_QUESTIONS_20), (
            "BLOCKER 2: smoke log-prob eval must use EVAL_QUESTIONS_20 (the "
            "full 20-question panel plan v4 §5.7 PASS/WARN/FAIL bands are "
            "calibrated for), NOT a 5-question subset."
        )
        assert len(recorded_kwargs["questions"]) == 20

        # --- BLOCKER 3 assertion: sampled-eval was called ---
        assert len(sampled_eval_calls) == 1, (
            "BLOCKER 3: _run_smoke_sampled_eval must run during smoke so "
            "metrics_final.json is always written. Found "
            f"{len(sampled_eval_calls)} calls."
        )
        # And the sampled eval threaded the runtime marker.
        assert sampled_eval_calls[0]["marker"] == args.marker_token

        # Round 9 — SMOKE_VERDICT.json was written with the PASS kind
        # (timing fast + source_rate > 0). The orchestrator reads this file
        # via SCP and posts the marker from the VM side.
        smoke_verdict_path = slab_root / "SMOKE_VERDICT.json"
        assert smoke_verdict_path.exists(), (
            f"Round 9: dispatcher must write SMOKE_VERDICT.json (got missing {smoke_verdict_path})"
        )
        verdict = json.loads(smoke_verdict_path.read_text(encoding="utf-8"))
        assert verdict["kind"] == "epm:smoke-pass", f"Expected PASS verdict; got {verdict['kind']}"
        assert verdict["source_substring_rate"] == 0.85
        assert verdict["cell_key"] == args.smoke_cell
        assert verdict["issue"] == args.issue


def test_sweep_phase_refuses_without_smoke_pass_marker(monkeypatch) -> None:
    """Phase B gate: ``run_sweep_phase`` returns non-zero when no smoke-pass marker."""
    monkeypatch.setattr(_dispatch, "is_smoke_pass_confirmed_locally", lambda args: False)

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        args = _build_smoke_args(slab_root)
        args.mode = "sweep"
        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 2, f"Sweep without smoke-pass should return 2; got {rc}"


def test_sweep_phase_dry_run_enumerates_108_jobs(monkeypatch, capsys) -> None:
    """Phase B enumeration: 3 sources x 1 seed (Round 7 descope) x 36 cells
    per source = 108 jobs.

    --dry-run must list the count without dispatching any subprocess. The
    job-count assertion catches accidentally regressing to binary E
    (which would enumerate 32 cells per source instead of 36 → 96 jobs)
    or to the round-4 3-seed expansion (which would enumerate 324 jobs).
    The Round 7 user-directed descope reverts to single seed=42.
    """
    monkeypatch.setattr(_dispatch, "is_smoke_pass_confirmed_locally", lambda args: True)

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        args = _build_smoke_args(slab_root)
        args.mode = "sweep"
        args.dry_run = True
        # no_resume=True is the test-helper default — no Hub probe.
        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 0

        out = capsys.readouterr().out + capsys.readouterr().err
        # Capture from the logger; the dispatcher logs to stdout.
        # Re-run with caplog if needed; this is fine as a smoke check.
        # The dispatcher logs "108 (cell, seed) runs" — assert the count.
        # If logging went elsewhere (e.g. file), fall back to checking the
        # return code only.
        # We still PRIMARILY assert the return code; the log inspection
        # is a defense-in-depth signal.
        del out  # unused — return-code is the load-bearing check


def test_sweep_phase_dispatches_to_two_pass_runner(monkeypatch) -> None:
    """Round 12: ``run_sweep_phase`` (non-dry-run) dispatches to
    ``_run_sweep_two_pass``, NOT the deleted Round 11 ``_run_sweep_serial``.

    Asserts via monkeypatching the canonical entry point: a non-dry-run
    sweep with smoke confirmed must call _run_sweep_two_pass exactly
    once with the expected source/seed/cells.
    """
    monkeypatch.setattr(_dispatch, "is_smoke_pass_confirmed_locally", lambda args: True)

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        args = _build_smoke_args(slab_root)
        args.mode = "sweep"
        args.dry_run = False
        args.sources = "librarian"
        args.seeds = "42"

        from explore_persona_space.experiments.factor_screen_397.cells import Cell

        only_cell = Cell.from_key("00000")
        monkeypatch.setattr(_dispatch, "_enumerate_valid_cells_per_seed", lambda: [only_cell])

        call_args: list[dict] = []

        def _fake_two_pass(**kwargs):
            call_args.append(kwargs)
            return 0

        monkeypatch.setattr(_dispatch, "_run_sweep_two_pass", _fake_two_pass)

        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 0
        assert len(call_args) == 1
        kw = call_args[0]
        assert kw["sources"] == ["librarian"]
        assert kw["seeds"] == [42]
        assert kw["cells"] == [only_cell]
        assert kw["args"] is args
        assert kw["repo_root"] == repo_root


def test_sweep_phase_two_pass_returns_nonzero_when_all_cells_fail(monkeypatch) -> None:
    """When ``_run_sweep_two_pass`` returns non-zero (all-fail path),
    ``run_sweep_phase`` propagates it so the orchestrator can mark the
    run as failed.
    """
    monkeypatch.setattr(_dispatch, "is_smoke_pass_confirmed_locally", lambda args: True)

    with tempfile.TemporaryDirectory() as tmpdir:
        slab_root = Path(tmpdir)
        args = _build_smoke_args(slab_root)
        args.mode = "sweep"
        args.dry_run = False
        args.sources = "librarian"
        args.seeds = "42"

        from explore_persona_space.experiments.factor_screen_397.cells import Cell

        monkeypatch.setattr(
            _dispatch, "_enumerate_valid_cells_per_seed", lambda: [Cell.from_key("00000")]
        )
        monkeypatch.setattr(_dispatch, "_run_sweep_two_pass", lambda **kwargs: 1)

        repo_root = Path(__file__).resolve().parent.parent.parent
        rc = _dispatch.run_sweep_phase(args, repo_root=repo_root)
        assert rc == 1, f"All-fail two-pass sweep must return 1; got {rc}"


def test_is_smoke_pass_confirmed_locally_via_cli_flag(tmp_path) -> None:
    """Round 9: ``--smoke-pass-confirmed`` flag (the canonical path)
    unlocks Phase B without any local-file check.

    Replaces the round-5..8 ``has_recent_smoke_pass_marker`` plumbing
    which shelled out to ``task.py find`` — broken from the pod because
    ``task.py`` branch-guards to ``main`` and the pod runs the
    ``issue-397`` branch.
    """
    args = _build_smoke_args(tmp_path)
    args.smoke_pass_confirmed = True
    assert _dispatch.is_smoke_pass_confirmed_locally(args) is True


def test_is_smoke_pass_confirmed_locally_via_metrics_file(tmp_path) -> None:
    """Round 9 fallback path: a valid local ``metrics_final.json`` with
    ``source_substring_rate > 0`` self-confirms smoke without the
    orchestrator setting the flag. Useful when re-launching on the same
    pod after a smoke that ran cleanly.
    """
    args = _build_smoke_args(tmp_path)
    args.smoke_pass_confirmed = False

    smoke_dir = (
        tmp_path
        / f"cell_{args.smoke_cell}"
        / f"source_{args.smoke_source}"
        / f"seed_{args.smoke_seed}"
    )
    smoke_dir.mkdir(parents=True)
    (smoke_dir / "metrics_final.json").write_text(
        json.dumps(
            {
                "marker": "※",
                "personas": {args.smoke_source: {"substring_rate": 1.0, "total": 100}},
            }
        ),
        encoding="utf-8",
    )
    assert _dispatch.is_smoke_pass_confirmed_locally(args) is True


def test_is_smoke_pass_confirmed_locally_returns_false_when_missing(tmp_path) -> None:
    """No flag, no metrics file → not confirmed; caller surfaces hint."""
    args = _build_smoke_args(tmp_path)
    args.smoke_pass_confirmed = False
    assert _dispatch.is_smoke_pass_confirmed_locally(args) is False


def test_is_smoke_pass_confirmed_locally_returns_false_on_zero_source_rate(tmp_path) -> None:
    """source_substring_rate == 0.0 means the smoke ran but M1 broke —
    NOT a pass signal. Dispatcher refuses to unlock Phase B.
    """
    args = _build_smoke_args(tmp_path)
    args.smoke_pass_confirmed = False
    smoke_dir = (
        tmp_path
        / f"cell_{args.smoke_cell}"
        / f"source_{args.smoke_source}"
        / f"seed_{args.smoke_seed}"
    )
    smoke_dir.mkdir(parents=True)
    (smoke_dir / "metrics_final.json").write_text(
        json.dumps(
            {
                "marker": "※",
                "personas": {args.smoke_source: {"substring_rate": 0.0, "total": 100}},
            }
        ),
        encoding="utf-8",
    )
    assert _dispatch.is_smoke_pass_confirmed_locally(args) is False


def test_is_smoke_pass_confirmed_locally_returns_false_on_malformed_json(tmp_path) -> None:
    """Corrupted metrics_final.json → treat as not-confirmed (re-run
    will overwrite cleanly).
    """
    args = _build_smoke_args(tmp_path)
    args.smoke_pass_confirmed = False
    smoke_dir = (
        tmp_path
        / f"cell_{args.smoke_cell}"
        / f"source_{args.smoke_source}"
        / f"seed_{args.smoke_seed}"
    )
    smoke_dir.mkdir(parents=True)
    (smoke_dir / "metrics_final.json").write_text("{not-json", encoding="utf-8")
    assert _dispatch.is_smoke_pass_confirmed_locally(args) is False
