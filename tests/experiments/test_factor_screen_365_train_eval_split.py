"""Round-14 (issue #365): train/eval split into separate subprocesses.

The round-13 attempt at fixing the vLLM-init OOM via
``VLLM_WORKER_MULTIPROC_METHOD=spawn`` failed: smoke-6 cells crashed with
5.1 GiB free of 140 GiB on the H200, WORSE than smoke-5b's fork (15.8
GiB free). The architectural diagnosis: ``torch.cuda.empty_cache()``
returns memory to PyTorch's internal pool, NOT to the CUDA driver. A
process's CUDA reservations are only released when the **process
exits**. So both fork and spawn children share the same physical GPU
with the still-living parent's ~134 GiB of reservations.

Round-14 fix: split each (cell, source, seed) into TWO subprocesses:

  1. ``--mode cell-train``: load base, train LoRA, merge, save merged
     to disk, exit cleanly. Parent's CUDA context destroyed.
  2. ``--mode cell-eval``: fresh process, fresh CUDA context. Loads
     merged via vLLM and runs the panel; sees full free HBM.

These tests verify the split, NOT the OOM fix itself (that's a runtime
acceptance test on the pod). Each test exercises one orthogonal axis
of the contract:

  * Dispatcher launches BOTH phases in order (train then eval), on the
    same GPU, with the same per-cell log file.
  * cell-eval refuses to run without the train-side handoff artifacts.
  * cell-eval mode does NOT touch any training-side code path.
  * cell-train mode does NOT touch any vLLM / eval-side code path.

All four use mocks so they run without a GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture
def dispatcher():
    """Load the dispatcher module by path so the test does not depend on PYTHONPATH."""
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "dispatch_factor_screen_365.py"
    spec = importlib.util.spec_from_file_location("dispatch_factor_screen_365", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        # Task #383 plumbing (plan v2 §5a): --issue forwarded to every
        # child cell-train / cell-eval subprocess argv.
        issue=365,
        sources=["librarian"],
        seeds=[42],
        pool_dir=tmp_path / "pools",
        slab_root=tmp_path / "slab",
        num_gpus=1,
        skip_pool_stage=True,
        skip_off_policy=False,
        dry_run=False,
        resume=False,
        skip_hub_probe=True,
        cell_filter=["00010"],
    )


def test_dispatch_launches_train_then_eval(
    dispatcher, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single (cell, source, seed) fires cell-train FIRST, then cell-eval.

    Round-14 (issue #365): the CUDA driver only releases the trainer's
    reservations on process exit. The dispatcher must therefore launch
    cell-train, wait for clean exit, then launch cell-eval against the
    same ``--output-dir``. We assert:

      * Exactly TWO Popen calls fire for one cell (one per phase).
      * Phase order: cell-train BEFORE cell-eval (by argv inspection).
      * Both phases run with the same --cell / --source / --seed /
        --output-dir / --pool-dir argv (the per-cell handoff).
      * Both phases run on the same GPU (CUDA_VISIBLE_DEVICES matches).
    """
    args = _make_args(tmp_path)
    monkeypatch.setattr(dispatcher, "_detect_physical_gpu_count", lambda: 1)

    fake_proc = mock.MagicMock(spec=subprocess.Popen)
    fake_proc.poll.return_value = 0
    fake_proc.returncode = 0
    popen_calls: list[dict] = []

    def fake_popen(cmd, env, stdout, stderr):
        popen_calls.append({"cmd": list(cmd), "env": dict(env), "stdout": stdout, "stderr": stderr})
        return fake_proc

    monkeypatch.setattr(dispatcher.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(dispatcher.time, "sleep", lambda _s: None)

    rc = dispatcher._training_stage(args)
    assert rc == 0, f"training stage failed unexpectedly (rc={rc})"
    assert len(popen_calls) == 2, (
        f"Round-14 split: one cell should fire 2 Popen calls (train + eval); "
        f"got {len(popen_calls)}."
    )

    train_cmd = popen_calls[0]["cmd"]
    eval_cmd = popen_calls[1]["cmd"]

    # Argv ordering: --mode cell-train MUST fire before --mode cell-eval.
    def _mode_of(cmd: list[str]) -> str:
        try:
            return cmd[cmd.index("--mode") + 1]
        except (ValueError, IndexError):
            return "?"

    assert _mode_of(train_cmd) == "cell-train", (
        f"First Popen call must be `--mode cell-train`; got argv {train_cmd!r}"
    )
    assert _mode_of(eval_cmd) == "cell-eval", (
        f"Second Popen call must be `--mode cell-eval`; got argv {eval_cmd!r}"
    )

    # Per-cell handoff: both phases share --cell / --source / --seed /
    # --output-dir / --pool-dir so cell-eval can read what cell-train wrote.
    for flag in ("--cell", "--source", "--seed", "--output-dir", "--pool-dir"):
        train_val = train_cmd[train_cmd.index(flag) + 1]
        eval_val = eval_cmd[eval_cmd.index(flag) + 1]
        assert train_val == eval_val, (
            f"Round-14 train/eval handoff: {flag} must match across phases; "
            f"got train={train_val!r}, eval={eval_val!r}"
        )

    # Both phases pin the same GPU so the eval subprocess inherits the
    # release of the trainer's CUDA context.
    assert popen_calls[0]["env"].get("CUDA_VISIBLE_DEVICES") == "0"
    assert popen_calls[1]["env"].get("CUDA_VISIBLE_DEVICES") == "0"


def test_cell_eval_refuses_without_merged(tmp_path: Path) -> None:
    """``--mode cell-eval`` exits non-zero when ``output_dir/merged/`` is absent.

    Round-14 (issue #365): cell-eval requires the merged checkpoint that
    cell-train produces. If the train subprocess crashed before merge
    landed, eval has nothing to load — refuse loudly per CLAUDE.md
    "Never silently fail".
    """
    from explore_persona_space.experiments.factor_screen_365.__main__ import main

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)
    # Deliberately: no `merged/` directory, no `cell_train_outcome.json`.
    argv = [
        "--mode",
        "cell-eval",
        "--issue",
        "365",
        "--cell",
        "00010",
        "--source",
        "librarian",
        "--seed",
        "42",
        "--pool-dir",
        str(tmp_path / "pools"),
        "--output-dir",
        str(output_dir),
        "--no-resume",
    ]
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    msg = str(exc_info.value)
    # Error message must name the missing artifact and reference the
    # cell-train phase so the operator knows which subprocess to investigate.
    assert "merged" in msg, f"Error should mention `merged/`; got {msg!r}"
    assert "cell-train" in msg, f"Error should reference `--mode cell-train`; got {msg!r}"


def test_cell_eval_skips_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--mode cell-eval`` does NOT import or call training-side code paths.

    Round-14 (issue #365): keeping the eval process free of any
    transformers-Trainer / peft import is the WHOLE POINT of the split —
    those imports would re-introduce the CUDA-reservation pattern that
    rounds 1-13 chased. This test sets sentinels on the training module
    and the train_one_cell function; both must remain untouched when
    cell-eval runs.

    We stub vllm_session + the generate_* / score_markers functions to
    avoid actually calling vLLM (no GPU), and assert the training-side
    symbols were never accessed.
    """
    from explore_persona_space.experiments.factor_screen_365 import __main__ as fs_main
    from explore_persona_space.experiments.factor_screen_365 import training as training_module

    # Set up the cell-train handoff artifacts so cell-eval gets past the
    # refusal gate.
    output_dir = tmp_path / "out"
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True)
    (merged_dir / "model.safetensors").write_bytes(b"\x00" * 16)
    outcome_payload = {
        "cell_key": "00010",
        "bits": [0, 0, 0, 1, 0],
        "source": "librarian",
        "seed": 42,
        "train_outcome": {
            "cell_key": "00010",
            "seed": 42,
            "adapter_path": str(output_dir / "adapter"),
            "merged_path": str(merged_dir),
            "loss": 0.5,
            "train_wall_minutes": 1.0,
            "n_examples": 100,
            "total_steps": 25,
            "marker_only_loss": False,
        },
        "prepared_dataset": {
            "num_positive": 50,
            "num_negative": 50,
            "data_policy": "on_policy",
            "system_prompt_token_count": 50,
            "marker_position_in_completion_tokens_mean": 0.5,
            "marker_position_in_completion_tokens_sd": 0.1,
            "total_seq_length_tokens_mean": 100.0,
            "total_seq_length_tokens_sd": 10.0,
            "caveats": [],
            "preflight": None,
        },
    }
    (output_dir / fs_main.CELL_TRAIN_OUTCOME_FILENAME).write_text(json.dumps(outcome_payload))

    # Sentinel: replace train_one_cell with a guard that raises on access.
    def _forbid(*_a, **_kw):
        raise AssertionError(
            "cell-eval mode invoked train_one_cell — round-14 split violation. "
            "The training-side code path must not run inside the eval subprocess."
        )

    monkeypatch.setattr(training_module, "train_one_cell", _forbid)

    # Stub vllm_session + the generate / score helpers so we don't need a GPU.
    from contextlib import contextmanager

    fake_llm = mock.MagicMock()

    @contextmanager
    def fake_vllm_session(**kw):
        holder = mock.MagicMock()
        holder.llm = fake_llm
        yield holder

    from explore_persona_space.experiments.factor_screen_365 import eval_panel

    monkeypatch.setattr(eval_panel, "vllm_session", fake_vllm_session)
    monkeypatch.setattr(
        eval_panel,
        "generate_completions",
        lambda llm, cfg: {p: {} for p in cfg.personas},
    )
    monkeypatch.setattr(
        eval_panel,
        "generate_random_control_completions",
        lambda llm, cfg: {p: {} for p in cfg.prompts},
    )
    monkeypatch.setattr(
        eval_panel,
        "score_markers",
        lambda results: {p: {"substring_rate": 0.0, "fuzzy_rate": 0.0} for p in results},
    )

    argv = [
        "--mode",
        "cell-eval",
        "--issue",
        "365",
        "--cell",
        "00010",
        "--source",
        "librarian",
        "--seed",
        "42",
        "--pool-dir",
        str(tmp_path / "pools"),
        "--output-dir",
        str(output_dir),
        "--no-resume",
    ]
    rc = fs_main.main(argv)
    assert rc == 0, f"cell-eval should exit cleanly with the handoff in place; got rc={rc}"

    # The metrics.json must exist (eval phase ran to completion).
    metrics = output_dir / "metrics.json"
    assert metrics.exists(), "cell-eval should write metrics.json"
    payload = json.loads(metrics.read_text())
    # The train_outcome block came from the sidecar, not from re-running training.
    assert payload["train_outcome"]["cell_key"] == "00010"
    assert payload["prepared_dataset"]["num_positive"] == 50


def test_cell_train_skips_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--mode cell-train`` does NOT call ``vllm_session`` or the panel functions.

    Round-14 (issue #365): the train subprocess MUST exit before any
    vLLM work happens. If a training subprocess accidentally instantiated
    vLLM in-process, the CUDA-context-release-on-exit guarantee that the
    whole round-14 split relies on would be broken.
    """
    from explore_persona_space.experiments.factor_screen_365 import __main__ as fs_main
    from explore_persona_space.experiments.factor_screen_365 import data_prep, training

    # Sentinel: any access to vllm_session / generate_* must raise.
    def _forbid_vllm(*_a, **_kw):
        raise AssertionError(
            "cell-train mode invoked vLLM session — round-14 split violation. "
            "The vLLM eval code path must not run inside the train subprocess."
        )

    from explore_persona_space.experiments.factor_screen_365 import eval_panel

    monkeypatch.setattr(eval_panel, "vllm_session", _forbid_vllm)
    monkeypatch.setattr(eval_panel, "generate_completions", _forbid_vllm)
    monkeypatch.setattr(eval_panel, "generate_random_control_completions", _forbid_vllm)

    # Stub the heavy training-side machinery so we don't need a GPU.
    fake_outcome = training.TrainOutcome(
        cell_key="00010",
        seed=42,
        adapter_path=str(tmp_path / "out" / "adapter"),
        merged_path=str(tmp_path / "out" / "merged"),
        loss=0.5,
        train_wall_minutes=1.0,
        n_examples=100,
        total_steps=25,
        marker_only_loss=False,
    )

    def fake_train(*_a, **_kw):
        # Simulate train_one_cell's side effect of writing the merged dir.
        merged = Path(fake_outcome.merged_path)
        merged.mkdir(parents=True, exist_ok=True)
        (merged / "model.safetensors").write_bytes(b"\x00" * 16)
        return fake_outcome

    monkeypatch.setattr(training, "train_one_cell", fake_train)

    # Stub AutoTokenizer (network-free).
    fake_tok = mock.MagicMock()
    fake_tok.encode.return_value = list(range(50))
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *a, **kw: fake_tok,
    )

    # Stub data_prep — return a PreparedDataset shaped object.
    fake_prepared = mock.MagicMock()
    fake_prepared.path = tmp_path / "out" / "train_data.jsonl"
    fake_prepared.path.parent.mkdir(parents=True, exist_ok=True)
    fake_prepared.path.write_text("")
    fake_prepared.num_positive = 50
    fake_prepared.num_negative = 50
    fake_prepared.data_policy = "on_policy"
    fake_prepared.system_prompt_token_count = 50
    fake_prepared.marker_position_mean_tokens = 0.5
    fake_prepared.marker_position_sd_tokens = 0.1
    fake_prepared.total_seq_length_mean_tokens = 100.0
    fake_prepared.total_seq_length_sd_tokens = 10.0
    fake_prepared.caveats = []
    fake_prepared.preflight = None
    monkeypatch.setattr(data_prep, "prepare_cell", lambda **kw: fake_prepared)

    # Stub the completion-source loader so we don't try to read pools off disk.
    monkeypatch.setattr(
        data_prep, "load_completion_source_from_disk", lambda **kw: mock.MagicMock()
    )

    # Bypass the pool-readiness wait (we don't create pool files in this test).
    monkeypatch.setattr(fs_main, "_wait_for_pool", lambda path, max_wait_s=1800: None)

    output_dir = tmp_path / "out"
    argv = [
        "--mode",
        "cell-train",
        "--issue",
        "365",
        "--cell",
        "00010",
        "--source",
        "librarian",
        "--seed",
        "42",
        "--pool-dir",
        str(tmp_path / "pools"),
        "--output-dir",
        str(output_dir),
        "--no-resume",
    ]
    rc = fs_main.main(argv)
    assert rc == 0, f"cell-train should exit cleanly with stubs in place; got rc={rc}"

    # The handoff sidecar MUST land on disk so cell-eval can read it.
    sidecar = output_dir / fs_main.CELL_TRAIN_OUTCOME_FILENAME
    assert sidecar.exists(), (
        f"cell-train must write {fs_main.CELL_TRAIN_OUTCOME_FILENAME} "
        f"so cell-eval can read the prepared_dataset + train_outcome blocks; "
        f"path: {sidecar}"
    )
    payload = json.loads(sidecar.read_text())
    assert payload["cell_key"] == "00010"
    assert payload["train_outcome"]["loss"] == 0.5
    assert payload["prepared_dataset"]["num_positive"] == 50

    # No metrics.json should land yet — that's the eval phase's responsibility.
    assert not (output_dir / "metrics.json").exists(), (
        "cell-train must NOT write metrics.json — that's cell-eval's output. "
        "If this fails the split has leaked back into a single mode."
    )


def test_cell_legacy_mode_rejected(tmp_path: Path) -> None:
    """``--mode cell`` is rejected with a clear error pointing at the new modes.

    Round-14 (issue #365): the legacy ``cell`` mode is kept in argparse's
    choice list so a stale dispatcher gets a useful error rather than an
    opaque "unrecognized argument" — but the run path itself refuses.
    """
    from explore_persona_space.experiments.factor_screen_365.__main__ import main

    argv = [
        "--mode",
        "cell",
        "--cell",
        "00010",
        "--source",
        "librarian",
        "--seed",
        "42",
        "--pool-dir",
        str(tmp_path / "pools"),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    msg = str(exc_info.value)
    assert "cell-train" in msg, f"Error should mention `cell-train`; got {msg!r}"
    assert "cell-eval" in msg, f"Error should mention `cell-eval`; got {msg!r}"


def test_cli_help_shows_cell_eval_mode() -> None:
    """``--help`` advertises the new ``cell-train`` / ``cell-eval`` modes.

    Sanity check from the brief — confirms the entry script exposes the
    round-14 split modes to users running ad-hoc CLI invocations.
    """
    import explore_persona_space.experiments.factor_screen_365 as pkg

    pkg_path = Path(pkg.__file__).parent
    result = subprocess.run(
        [sys.executable, "-m", "explore_persona_space.experiments.factor_screen_365", "--help"],
        capture_output=True,
        text=True,
        cwd=pkg_path.parents[2],
    )
    assert result.returncode == 0, (
        f"--help should exit 0; got {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "cell-train" in result.stdout
    assert "cell-eval" in result.stdout
